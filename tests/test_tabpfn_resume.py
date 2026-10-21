"""Checkpoint/device regressions for VSC job 11688276.

The meta-device case exposes delayed placement without needing CUDA locally.
The real-weight round trip also runs on CUDA in the mandatory VSC suite.
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.train.tabpfn_trainer import TabPFNTrainer
from src.utils.config import expand_with_seeds, load

ROOT = Path(__file__).resolve().parents[1]


def resume_config(task):
    run = copy.deepcopy(next(r for r in expand_with_seeds(load(ROOT / "config" / f"Exp0_{task.upper()}.yaml"))
                             if r["architecture"] == "tabpfn3"))
    run["prior"].update(n_rows_range=[96, 96], n_features_range=[3, 5], max_features=8,
                        n_nodes_range=[2, 4], max_filter_attempts=3)
    run["prior"]["filter"]["mode"] = "off"
    run["train"].update(max_steps=2, batch_size=2, num_workers=0, log_every=1,
                        save_temp_every=0, save_perm_every=2, warmup_proportion=0.)
    run["logging"] = {"level": "WARNING", "console": False, "to_file": False,
                      "log_prior_every": 0, "log_hardware_every": 0,
                      "log_grad_every": 0, "log_weights_every": 0}
    run["progress"]["every_datasets"] = 0
    return run


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_restored_optimizer_follows_training_device_before_first_forward(tmp_path, monkeypatch, device):
    from src.train import tabpfn_trainer as module
    from src.utils.preflight import training_fingerprint

    # An unfitted upstream estimator initially owns CPU parameters. The meta
    # target also replaces Parameter objects, catching an optimizer constructed
    # before placement even on this CPU-only laptop.
    monkeypatch.setattr(module, "build_estimator", lambda *a, **kw:
                        SimpleNamespace(models_=[torch.nn.Linear(2, 1)]))
    monkeypatch.setattr(module, "resolve_tabpfn_weights", lambda *a: tmp_path / "unused.ckpt")
    run = resume_config("pd")
    run["train"]["amp"] = False
    source = TabPFNTrainer(run, tmp_path / "source", device="cpu")
    source.model(torch.ones(2, 2)).square().mean().backward()
    source.optimizer.step()
    source.scheduler.step()
    checkpoint = tmp_path / "restored" / "checkpoints" / "step-1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"state_dict": source.model.state_dict(), "optimizer": source.optimizer.state_dict(),
                "scheduler": source.scheduler.state_dict(), "step": 1,
                "crediticl_config": run, "training_fingerprint": training_fingerprint(),
                "extra": {"datasets_seen": 2}}, checkpoint)
    source.close()

    restored = TabPFNTrainer(run, tmp_path / "restored", device=device)
    try:
        restored.maybe_resume()
        live = list(restored.model.parameters())
        optimized = [p for group in restored.optimizer.param_groups for p in group["params"]]
        assert {id(p) for p in live} == {id(p) for p in optimized}
        assert restored.step == 1 and restored.datasets_seen == 2
        for parameter in live:
            assert parameter.device.type == device
            state = restored.optimizer.state[parameter]
            assert state["exp_avg"].device == parameter.device
            assert state["exp_avg_sq"].device == parameter.device
            # Adam's non-capturable/non-fused step counter belongs on CPU.
            assert state["step"].device.type == "cpu" and int(state["step"]) == 1
    finally:
        restored.close()


def assert_real_checkpoint_resume(directory, task, device, *, force_full_precision=False):
    """Stop after one real update, recreate the trainer, restore, and update again."""
    run = resume_config(task)
    run["train"]["amp"] = device.startswith("cuda")
    trainer = TabPFNTrainer(run, directory / "run", device=device)
    # This probe injects a stop; the longer Exp0 child-process tests deliver the
    # real signal. Do not leave this short-lived trainer in global signal handlers.
    trainer._install_stop_signals = lambda: None
    trainer.maybe_resume()
    trainer._stop_requested = "SIGUSR1"
    if force_full_precision:
        # A real batched TabPFN executor must disable its own autocast context,
        # not just the trainer's outer context, before gradients can recover.
        for parameter in trainer.model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: torch.full_like(grad, float('nan'))
                                        if trainer.est.use_autocast_ else grad)
    try:
        stopped = trainer.train()
        assert not stopped["completed"] and stopped["steps"] == 1
        assert trainer.optimizer.state, "a checkpoint without Adam moments cannot test this bug"
        name, parameter = next((name, p) for name, p in trainer.model.named_parameters()
                               if p in trainer.optimizer.state)
        expected_moment = trainer.optimizer.state[parameter]["exp_avg"].detach().cpu().clone()
        expected_weight = parameter.detach().cpu().clone()
        expected_schedule = copy.deepcopy(trainer.scheduler.state_dict())
        expected_scaler = copy.deepcopy(trainer.scaler.state_dict()) if trainer.scaler else None
        expected_retries = trainer.amp_retries
        expected_full_precision_retries = trainer.full_precision_retries
        if force_full_precision:
            assert expected_full_precision_retries == 1
            assert trainer.amp and trainer.est.use_autocast_
    finally:
        trainer.close()
    del trainer, parameter
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    restored = TabPFNTrainer(run, directory / "run", device=device)
    restored._install_stop_signals = lambda: None
    try:
        restored.maybe_resume()
        assert restored.step == 1 and restored.datasets_seen == 2
        assert restored.scheduler.state_dict() == expected_schedule
        assert restored.amp_retries == expected_retries
        assert restored.full_precision_retries == expected_full_precision_retries
        if expected_scaler is not None:
            assert restored.scaler.state_dict() == expected_scaler
        parameter = dict(restored.model.named_parameters())[name]
        torch.testing.assert_close(parameter.detach().cpu(), expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(restored.optimizer.state[parameter]["exp_avg"].cpu(),
                                   expected_moment, rtol=0, atol=0)
        for p, state in restored.optimizer.state.items():
            assert p.device.type == device.split(":")[0]
            assert state["exp_avg"].device == p.device
            assert state["exp_avg_sq"].device == p.device
            assert state["step"].device.type == "cpu"
        completed = restored.train()
        assert completed["completed"] and completed["steps"] == 2
        assert completed["resumed_at"] == 1 and completed["datasets_seen"] == 4
        assert completed["skipped_tasks"] == 0
        assert restored.scheduler.last_epoch == 2
        steps = {int(state["step"]) for state in restored.optimizer.state.values()}
        assert steps <= {1, 2} and 2 in steps
        assert all(torch.isfinite(p).all() for p in restored.model.parameters())
        assert not torch.equal(parameter.detach().cpu(), expected_weight), "resumed weights must change"
    finally:
        restored.close()


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_real_checkpoint_resume_on_cpu(tmp_path, task):
    assert_real_checkpoint_resume(tmp_path, task, "cpu")
