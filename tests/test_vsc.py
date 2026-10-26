"""Mandatory Experiment 0 checks that must fail, never skip, on the VSC node.

Full-model training/save/load tests and selected full-table benchmark probes run as
separate child processes in src.validation.campaign, avoiding shared CUDA failures.
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


@pytest.mark.parametrize("method", ["lora", "diagscale"])
def test_adapter_initialisation_preserves_cpu_and_cuda_random_streams(method):
    from test_adapters import tiny_model
    from src.models.adapters import inject
    model = tiny_model("tabicl", "pd").cuda()
    cpu = torch.get_rng_state().clone()
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()]
    inject(model, {"adaptation": method}, seed=777)
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))


@pytest.mark.parametrize("amp", [False, True])
def test_tabpfn_nested_training_precision(monkeypatch, amp):
    from test_exp2_precision import assert_task_precision
    assert_task_precision(monkeypatch, "cuda", amp)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prior_sampling_preserves_the_training_cuda_rng(device):
    from src.prior.upstream import seeded

    before = torch.cuda.get_rng_state().clone()
    draws = []
    for _ in range(2):
        with seeded(123, device):
            draws.append(torch.rand(8, device=device))
        torch.testing.assert_close(torch.cuda.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_tabpfn_cuda_checkpoint_resume_before_long_training(tmp_path, task):
    """Catch misplaced restored Adam moments before the 600-step validation arms."""
    import tempfile
    from pathlib import Path

    from src.utils.working_storage import work_root
    from test_tabpfn_resume import assert_real_checkpoint_resume

    root = work_root() or tmp_path
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"resume-{task}-", dir=root) as directory:
        assert_real_checkpoint_resume(Path(directory), task, "cuda")


def test_tabpfn_cuda_full_precision_replay_survives_real_checkpoint_resume(tmp_path, monkeypatch):
    """Prove the nested executor/scaler/checkpoint contract before expensive stages."""
    import tempfile
    from pathlib import Path

    from src.train import tabpfn_trainer
    from src.utils.working_storage import work_root
    from test_tabpfn_resume import assert_real_checkpoint_resume

    # One forced AMP failure is sufficient to exercise the real executor here;
    # the portable and CUDA magnitude tests verify all 16 production backoffs.
    monkeypatch.setattr(tabpfn_trainer, 'MAX_AMP_RETRIES', 1)
    root = work_root() or tmp_path
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='precision-resume-lgd-', dir=root) as directory:
        assert_real_checkpoint_resume(Path(directory), 'lgd', 'cuda', force_full_precision=True)


def test_cuda_extreme_regression_targets_keep_the_whole_training_batch():
    from test_tabpfn_safety import assert_extreme_regression_replay

    assert_extreme_regression_replay('cuda')


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
    from src.validation.campaign import input_inventory

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


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_tabicl_disk_offload_preserves_full_ensemble_predictions(task):
    from test_tabicl_memory import assert_released_memory_equivalence

    assert_released_memory_equivalence(task, device="cuda", force_disk=True)


@pytest.mark.parametrize("task,width", [("pd", 2986), ("lgd", 601)])
def test_tabicl_streaming_preserves_full_width_ensemble_predictions(task, width):
    from test_tabicl_memory import assert_released_memory_equivalence

    assert_released_memory_equivalence(task, device="cuda", force_stream=True, width=width)


def test_storage_tiers_handle_spaces_and_atomic_writes():
    import json
    import tempfile
    from pathlib import Path

    from src.utils import paths
    from src.validation.campaign import write_json

    for root in (paths.outputs_dir(), paths.big_outputs_dir()):
        assert "output CreditICL" in str(root)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="exp0 probe ", dir=root) as directory:
            target = Path(directory) / "atomic write.json"
            write_json(target, {"probe": 1})
            write_json(target, {"probe": 2})
            assert json.loads(target.read_text()) == {"probe": 2}


@pytest.mark.parametrize("model_name", ["tabiclv2", "tabpfn3"])
@pytest.mark.parametrize("classes", [3, 11])
def test_native_multiclass_on_released_gpu_wrappers(model_name, classes):
    from test_multiclass import assert_multiclass_wrapper
    assert_multiclass_wrapper(model_name, classes)


@pytest.mark.parametrize("classes", [3, 11])
def test_streaming_multiclass_on_gpu(classes):
    from test_tabicl_memory import assert_released_memory_equivalence
    assert_released_memory_equivalence("pd", device="cuda", force_stream=True, classes=classes)


def test_l2sp_references_follow_the_cuda_model(lgd_cfg, tmp_path, monkeypatch):
    from test_l2sp import assert_l2sp_device_and_gradient
    assert_l2sp_device_and_gradient(lgd_cfg, tmp_path, monkeypatch, device="cuda")
