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


def test_cluster_working_storage_publishes_complete_checkpoint():
    from src.utils import paths
    from src.utils.working_storage import checkpoint_work_dir, local_input, publish_checkpoint, work_root
    import uuid

    permanent = paths.experiment_dir(0, big=True) / "storage_probe" / uuid.uuid4().hex
    permanent.mkdir(parents=True)
    source = permanent / "input.bin"
    source.write_bytes(b"CreditICL input integrity")
    assert local_input(source).read_bytes() == source.read_bytes()
    work = checkpoint_work_dir(permanent)
    if os.environ.get("SLURM_CLUSTER_NAME", "").lower() == "mindwell":
        assert work_root() in work.parents and work != permanent
    (work / "step-1.ckpt").write_bytes(b"complete checkpoint probe")
    publish_checkpoint(work, 1, permanent_every=10, final=False, stopped=True)
    assert (permanent / "step-1.ckpt").read_bytes() == b"complete checkpoint probe"


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


@pytest.mark.parametrize("model_name", ["tabiclv2", "tabpfn3"])
@pytest.mark.parametrize("task,width", [("pd", 2986), ("lgd", 601)])
def test_released_models_retain_wide_inputs(model_name, task, width):
    from test_full_features import assert_released_full_width
    assert_released_full_width(model_name, task, width)


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
