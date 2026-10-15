"""Full-table inference regressions from Exp0 GPU job 11689448 (SIGKILL)."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tabicl._model.inference import DiskTensor, InferenceManager, OffloadMode
from tabicl._model.inference_config import InferenceConfig

from src.eval.tabicl_memory import BoundedInferenceManager, install, prediction_storage


def manager(folder=None):
    mgr = BoundedInferenceManager("tf_col", 128)
    mgr.configure(device="cpu", use_amp=False, use_async=False,
                  disk_offload_dir=str(folder) if folder else None)
    return mgr


def test_node_memory_cannot_authorize_a_large_host_allocation(tmp_path):
    mgr = manager(tmp_path)
    # 240 GiB output, 180 GiB job allocation, 1 TiB visible host RAM.
    # Stock 2.1.1 chooses CPU; ours must use disk without allocating either here.
    args = (240 * 1024, 160 * 1024, 1024 * 1024, 1000 * 1024)
    assert InferenceManager._resolve_offload_mode(mgr, *args)[0] == OffloadMode.CPU
    assert mgr._resolve_offload_mode(*args)[0] == OffloadMode.DISK


@pytest.mark.parametrize("disk,free", [(False, 1e6), (True, 0), (True, 1024)])
def test_no_disk_or_full_disk_fails_before_host_allocation(tmp_path, disk, free):
    mgr = manager(tmp_path if disk else None)
    with pytest.raises(MemoryError, match="refusing an unbounded CPU"):
        mgr._resolve_offload_mode(240 * 1024, 160 * 1024, 1024 * 1024, free)


def test_small_outputs_still_use_gpu(tmp_path):
    assert manager(tmp_path)._resolve_offload_mode(100, 1000, 0, 0)[0] == OffloadMode.GPU


def test_gpu_allocation_failure_never_falls_back_to_cpu(tmp_path, monkeypatch):
    mgr = manager(tmp_path)
    mgr.out_dim = 2
    mgr.exe_device = torch.device("cuda")
    mgr.disk_flush_mb = .000001
    monkeypatch.setattr(mgr, "estimate_safe_batch_size", lambda *a, **k: (1000, 1))
    monkeypatch.setattr(mgr, "_prepare_inputs", lambda inputs: dict(inputs))
    monkeypatch.setattr(mgr, "_to_exe_device", lambda tensor: tensor)
    flushes = []
    original_flush = DiskTensor.flush

    def flushed(output):
        flushes.append(output.nbytes)
        original_flush(output)

    monkeypatch.setattr(DiskTensor, "flush", flushed)

    def allocation(*args, **kwargs):
        assert torch.device(kwargs["device"]).type == "cuda", "full CPU allocation attempted"
        raise torch.OutOfMemoryError("simulated fragmented VRAM")

    monkeypatch.setattr(torch, "empty", allocation)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    result = mgr(lambda features: features * 2, OrderedDict(features=x))
    torch.testing.assert_close(result, x * 2)
    assert result.dtype == torch.float32
    assert len(flushes) == 4  # every write plus final flush, not only after the whole tensor
    assert mgr.offload_mode == OffloadMode.AUTO
    del result


def test_disk_capacity_is_rechecked_before_allocation(tmp_path, monkeypatch):
    mgr = manager(tmp_path)
    monkeypatch.setattr(mgr, "get_available_disk_space", lambda _: 0)
    with pytest.raises(MemoryError, match="disk allocation refused"):
        mgr._allocate_output_buffer(OffloadMode.DISK, (2, 3), torch.float32)
    assert not list(tmp_path.iterdir())


def test_disk_quota_refusal_is_an_exception_before_mmap_writes(tmp_path, monkeypatch):
    import src.eval.tabicl_memory as memory

    def full(*args):
        raise OSError("quota exceeded")

    monkeypatch.setattr(memory.os, "posix_fallocate", full, raising=False)
    with pytest.raises(OSError, match="Cannot reserve TabICL scratch storage"):
        manager(tmp_path)._allocate_output_buffer(OffloadMode.DISK, (2, 3), torch.float32)


def test_column_transfer_bound_and_upstream_compute_bound_both_apply(monkeypatch):
    mgr = manager()
    monkeypatch.setattr(InferenceManager, "estimate_safe_batch_size", lambda *a, **k: (100000, 500))
    free, count = mgr.estimate_safe_batch_size(100_000, in_dim=1)
    assert free == 100000 and 1 <= count < 500
    assert count * 100_000 * 128 * 4 <= mgr.transfer_mb * 2**20
    monkeypatch.setattr(InferenceManager, "estimate_safe_batch_size", lambda *a, **k: (100000, 1))
    assert mgr.estimate_safe_batch_size(100_000, in_dim=1)[1] == 1


def test_real_disk_tensor_survives_multibatch_forward_on_cpu(tmp_path, monkeypatch):
    """Exercise actual memmap writes; only accelerator calls are simulated locally."""
    mgr = manager(tmp_path)
    mgr.out_dim = 2
    mgr.exe_device = torch.device("cuda")
    monkeypatch.setattr(mgr, "estimate_safe_batch_size", lambda *a, **k: (0, 1))
    monkeypatch.setattr(mgr, "_prepare_inputs", lambda inputs: dict(inputs))
    monkeypatch.setattr(mgr, "_to_exe_device", lambda tensor: tensor)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    result = mgr(lambda features: features * 2, OrderedDict(features=x))
    torch.testing.assert_close(result, x * 2)
    assert result.dtype == x.dtype
    del result


@pytest.mark.parametrize("failed", [False, True])
def test_prediction_scratch_is_unique_scoped_and_restored(tmp_path, monkeypatch, failed):
    monkeypatch.setenv("CREDITICL_WORK_ROOT", str(tmp_path))
    sentinel = tmp_path / "keep.bin"
    sentinel.write_bytes(b"other job files")
    original = InferenceConfig()
    est = SimpleNamespace(device_=torch.device("cuda"), inference_config_=original)
    try:
        with prediction_storage(est):
            cfg = est.inference_config_.COL_CONFIG
            folder = Path(cfg["disk_offload_dir"])
            assert folder.parent == tmp_path and folder.is_dir()
            assert cfg["use_async"] is False and cfg["disk_dtype"] == original.COL_CONFIG["disk_dtype"]
            (folder / "temporary.bin").write_bytes(b"temporary")
            if failed:
                raise ValueError("probe")
    except ValueError:
        assert failed
    assert est.inference_config_ is original
    assert not folder.exists() and sentinel.read_bytes() == b"other job files"


def test_accelerator_without_job_scratch_uses_big_output_not_system_tmp(tmp_path, monkeypatch):
    from src.eval import tabicl_memory
    from src.utils import paths

    monkeypatch.setattr(tabicl_memory, "work_root", lambda: None)
    monkeypatch.setattr(paths, "big_outputs_dir", lambda: tmp_path)
    est = SimpleNamespace(device_=torch.device("cuda"), inference_config_=InferenceConfig())
    with prediction_storage(est):
        folder = Path(est.inference_config_.COL_CONFIG["disk_offload_dir"])
        assert folder.parent == tmp_path / "inference_tmp"
    assert not folder.exists()


def assert_released_memory_equivalence(task, *, device="cpu", force_disk=False):
    """Real released weights + every ensemble member, reused by mandatory GPU tests."""
    from tabicl import TabICLClassifier, TabICLRegressor
    from src.eval.baselines import TABICL_RELEASED
    from src.utils.paths import find_pretrained
    from src.utils.working_storage import local_input

    cls = TabICLClassifier if task == "pd" else TabICLRegressor
    kwargs = dict(device=device, random_state=11, allow_auto_download=False,
                  model_path=str(local_input(find_pretrained(f"checkpoints/{TABICL_RELEASED[task]}"))))
    rng = np.random.default_rng(51)
    X = rng.normal(size=(40, 7)).astype(np.float32)
    y = np.arange(32) % 2 if task == "pd" else rng.random(32)
    stock = cls(**kwargs).fit(X[:32], y)
    bounded = cls(**kwargs, batch_size=1).fit(X[:32], y)
    install(bounded)
    install(bounded)  # no duplicate hooks/managers on a repeated installation
    assert bounded.n_estimators == stock.n_estimators == 8
    for key, value in stock.model_.state_dict().items():
        torch.testing.assert_close(bounded.model_.state_dict()[key], value, rtol=0, atol=0)
    predict = (lambda m: m.predict_proba(X[32:])) if task == "pd" else (
        lambda m: m.predict(X[32:], output_type=["mean", "quantiles"], alphas=[.1, .5, .9]))
    expected = predict(stock)
    with prediction_storage(bounded):
        if force_disk:
            bounded.inference_config_.COL_CONFIG.update({"offload": "disk"})
            bounded.model_.col_embedder.inference_mgr.transfer_mb = .01  # multiple actual disk writes
        actual = predict(bounded)
    if task == "pd":
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
    else:
        for output in ("mean", "quantiles"):
            np.testing.assert_allclose(actual[output], expected[output], rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_full_ensemble_predictions_match_upstream(task):
    assert_released_memory_equivalence(task)


def test_single_member_shuffle_does_not_copy_the_full_embedding(monkeypatch):
    """Fail if the classifier re-enters the repeat path, even repeat(1)."""
    from src.eval.tabicl_memory import _single_member_embeddings
    from tabicl._model.embedding import ColEmbedding

    col = ColEmbedding(embed_dim=8, num_blocks=1, nhead=2, num_inds=4, dim_feedforward=16,
                       feature_group=False, target_aware=False, reserve_cls_tokens=0).eval()
    col.register_forward_pre_hook(_single_member_embeddings, with_kwargs=True)
    # A storage view returned by the manager must reach the row encoder as a view.
    storage = torch.arange(3 * 5 * 8, dtype=torch.float32).reshape(1, 3, 5, 8)
    monkeypatch.setattr(col, "inference_mgr", lambda *a, **kw: storage)
    # ColEmbedding calls configure before invoking the manager.
    col.inference_mgr.configure = lambda **kw: None
    result = col(torch.ones(1, 5, 3), y_train=torch.zeros(1, 4), feature_shuffles=[[2, 0, 1]])
    assert result.data_ptr() == storage.data_ptr()
    torch.testing.assert_close(result, storage.transpose(1, 2))


def test_wrong_ensemble_batch_fails_explicitly():
    with pytest.raises(ValueError, match="batch_size=1"):
        install(SimpleNamespace(batch_size=8))
