"""Mandatory Experiment 0 checks that must fail, never skip, on the VSC node.

Full-model training/save/load tests and selected full-table benchmark probes run as
separate child processes in src.utils.validation, avoiding shared CUDA failures.
"""
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.vsc


@pytest.fixture(autouse=True)
def require_compute_allocation():
    assert sys.platform == "linux", "VSC validation requires Linux"
    assert os.environ.get("SLURM_JOB_ID"), "Submit exp0.slurm"
    assert torch.cuda.is_available(), "Allocated GPU is unavailable"


def test_cuda_stack_executes_and_backpropagates():
    from src.utils.preflight import dependency_checks

    failures = [c for c in dependency_checks() if not c.ok]
    assert not failures, failures
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_cuda_bfloat16_regression_retains_the_999_quantile_grid():
    from src.train.loop import pinball_loss
    pred = torch.ones(32, 999, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.zeros(32, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = pinball_loss(pred, target, 999)
    loss.backward()
    assert loss.dtype == torch.float32 and abs(loss.item() - .5) < 1e-6
    assert torch.isfinite(pred.grad).all()


def test_regression_cpu_preparer_cuda_loss():
    from test_tabpfn_safety import assert_regression_loss_devices

    assert_regression_loss_devices("cuda")


def test_cuda_scaler_retries_without_losing_training_tasks():
    from test_tabpfn_safety import tiny_trainer

    trainer = tiny_trainer(device="cuda", scale=8.)
    trainer.model.weight.register_hook(lambda g: torch.full_like(g, float("inf")) if g.abs().max() > 2 else g)
    result = trainer.train_step([{"item": 0}, {"item": 1}])
    assert result["amp_retries"] == 2
    assert trainer.datasets_seen == 2 and trainer.scheduler.last_epoch == 1
    assert trainer.model.weight.item() < 1.


def test_every_real_input_and_released_weight_is_present():
    from src.utils.preflight import run_checks
    from src.utils.validation import input_inventory

    failures = [c for c in run_checks() if not c.ok]
    assert not failures, failures
    inventory = input_inventory()
    assert len(inventory) == 8
    for request in inventory.values():
        assert request["credit_inputs"] and len(request["ood_inputs"]) >= 25


def test_storage_tiers_handle_spaces_and_atomic_writes():
    import json
    import tempfile
    from pathlib import Path

    from src.utils import paths
    from src.utils.validation import write_json

    for root in (paths.outputs_dir(), paths.big_outputs_dir()):
        assert "output CreditICL" in str(root)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="exp0 probe ", dir=root) as directory:
            target = Path(directory) / "atomic write.json"
            write_json(target, {"probe": 1})
            write_json(target, {"probe": 2})
            assert json.loads(target.read_text()) == {"probe": 2}
