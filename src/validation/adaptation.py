"""Released-weight train/stop/resume/native-inference checks for Experiment 3.

Diagnostic tables are deliberately small; separate Experiment 0 capacity stages
exercise the production long tables. No pilot or research result is produced.
"""
from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
import tempfile

import numpy as np
import torch


def diagnostic_config(track: str, backbone: str, method: str) -> dict:
    from src.utils.config import expand_with_seeds, load
    from src.experiments.design import apply_design

    cfg = copy.deepcopy(next(r for r in expand_with_seeds(load(f"config/Exp2_{track.upper()}.yaml"))
                             if r["arm"] == backbone and r["seed"] == 0
                             and r["lr_index"] == 0 and r["prior"]["credit_fraction"] == .5))
    cfg.update(design="retention_pilot", method=method, candidate=1)
    apply_design(cfg, {"choices": {f"{track}/{backbone}": {"full_lr": cfg["train"]["lr"]}}})
    cfg["experiment"] = "exp0_adaptation"
    cfg["_run_name"] = f"exp0_adaptation_{track}_{backbone}_{method}"
    cfg["prior"].update(n_rows_range=[96, 96], n_features_range=[3, 5], max_features=8,
                        n_nodes_range=[2, 4], max_filter_attempts=3)
    cfg["prior"]["filter"]["mode"] = "off"
    cfg["train"].update(max_steps=2, batch_size=2, micro_batch_size=1, num_workers=0,
                        max_workers=0, warmup_proportion=0., schedule_total_steps=2,
                        save_temp_every=1, save_perm_every=2, keep_steps=[1, 2], log_every=1)
    cfg["progress"] = {"every_datasets": 0}
    cfg["logging"] = {"level": "WARNING", "console": True, "to_file": False,
                      "log_prior_every": 0, "log_hardware_every": 0,
                      "log_grad_every": 0, "log_weights_every": 0}
    return cfg


def run_probe(track: str, backbone: str, method: str, directory: Path, device="cuda") -> dict:
    from src.train.loop import Trainer
    from src.train.tabpfn_trainer import TabPFNTrainer
    from src.eval.baselines import TabPFNBaseline
    from src.eval.crediticl_baseline import CreditICLBaseline

    cfg = diagnostic_config(track, backbone, method)
    cls = Trainer if backbone == "tabicl" else TabPFNTrainer
    trainer = cls(cfg, directory / "run", device=device)
    trainer._install_stop_signals = lambda: None
    trainer.maybe_resume()
    frozen = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if not p.requires_grad}
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    trainer._stop_requested = "SIGUSR1"
    try:
        summary = trainer.train()
        if summary["completed"] or summary["steps"] != 1:
            raise ValueError("Diagnostic stop did not retain exactly one update")
        first = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    finally:
        trainer.close()
    del trainer
    gc.collect()
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    resumed = cls(cfg, directory / "run", device=device)
    resumed._install_stop_signals = lambda: None
    resumed.maybe_resume()
    try:
        for name, p in resumed.model.named_parameters():
            if name in first:
                torch.testing.assert_close(p.detach().cpu(), first[name], atol=0, rtol=0)
        summary = resumed.train()
        if not summary["completed"] or summary["steps"] != 2 or summary["resumed_at"] != 1:
            raise ValueError("Resume did not finish the diagnostic budget")
        changed = 0
        for name, p in resumed.model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p.detach().cpu(), frozen[name], atol=0, rtol=0)
            elif name in first:
                changed += int(not torch.equal(p.detach().cpu(), first[name]))
        if not changed:
            raise ValueError("No trainable parameter moved after resume")
        checkpoint = resumed.ckpt_dir / "step-2.ckpt"
    finally:
        resumed.close()
    del resumed, first, frozen
    gc.collect()
    rng = np.random.default_rng(92)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = np.arange(40) % 2 if track == "pd" else rng.uniform(size=40)
    wrapper = (CreditICLBaseline(task=track, checkpoint=checkpoint, n_estimators=1) if backbone == "tabicl"
               else TabPFNBaseline(task=track, model_path=str(checkpoint), n_estimators=1))
    wrapper.fit(x[:30], y[:30])
    prediction = wrapper.predict(x[30:])
    if not np.isfinite(prediction).all() or len(prediction) != 10:
        raise ValueError("Native checkpoint prediction failed")
    return dict(track=track, backbone=backbone, method=method, steps=2,
                trainable_parameters=trainable, updated_tensors=changed,
                frozen_unchanged=True, checkpoint_prediction_finite=True,
                scope="small-table training/resume/prediction; capacity is validated separately")


def main(argv=None):
    from src.validation.checks import Report, PASS, FAIL, write
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=("pd", "lgd"))
    parser.add_argument("--backbone", required=True, choices=("tabicl", "tabpfn3"))
    parser.add_argument("--method", required=True, choices=("no_decay", "l2sp", "diagscale", "lora"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = Report()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This validation stage requires the allocated CUDA GPU")
        with tempfile.TemporaryDirectory(prefix="crediticl-adaptation-") as directory:
            result = run_probe(args.track, args.backbone, args.method, Path(directory))
        report.add("adaptation", f"{args.track}/{args.backbone}/{args.method}", PASS, result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report.add("adaptation", "released-weight round trip", FAIL, f"{type(exc).__name__}: {exc}")
    write(report, args.out)
    return int(report.failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
