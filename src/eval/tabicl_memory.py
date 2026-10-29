"""Bound TabICL inference storage without changing its ensemble or model math.

The pinned 2.1.1 manager measures host RAM, not a Slurm allocation, and can fall
back to an arbitrarily large CPU tensor. Use GPU/disk for column embeddings,
one ensemble member at a time. Upstream still does all preprocessing, network
operations and aggregation. Released and trained weights share this adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
import time
from pathlib import Path
import tempfile

import torch
from tabicl._model.inference import InferenceManager, OffloadMode

from src.utils.working_storage import work_root
from src.utils.logging_setup import get_logger

LOG = get_logger()


class _RetryOnDisk(RuntimeError):
    """A GPU output allocation failed before any forward computation."""


class BoundedInferenceManager(InferenceManager):
    """Never fall back to a full host tensor or rely on swap on a compute node.

    CPU execution (small local checks) retains upstream's CPU path. On CUDA,
    transfers are synchronous and column batches write at most about 1 GiB
    each, unless a single column itself exceeds that. The full disk tensor is
    paged by the OS; there is no full-sized anonymous host allocation.
    """

    transfer_mb = 1024

    def __call__(self, *args, **kwargs):
        started = time.monotonic()
        try:
            try:
                return super().__call__(*args, **kwargs)
            except _RetryOnDisk:
                previous = self.offload_mode
                self.offload_mode = OffloadMode.DISK
                try:
                    return super().__call__(*args, **kwargs)
                finally:
                    self.offload_mode = previous
        finally:
            LOG.info("TabICL %s completed/returned after %.2fs", self.enc_name, time.monotonic() - started)

    def estimate_safe_batch_size(self, seq_len, include_inputs=True, in_dim=None, max_bs=50000):
        free, batch = super().estimate_safe_batch_size(seq_len, include_inputs=include_inputs,
                                                       in_dim=in_dim, max_bs=max_bs)
        if self.enc_name == "tf_col":
            # Inputs/storage are float32 even when upstream computes under AMP.
            per_column_mb = seq_len * self.out_dim * 4 / 2**20
            batch = min(batch, max(1, int(self.transfer_mb / max(per_column_mb, 1e-9))))
        return free, batch

    def _resolve_offload_mode(self, output_mb, gpu_free_mb, cpu_free_mb, disk_free_mb):
        mode, reason = super()._resolve_offload_mode(output_mb, gpu_free_mb, 0, disk_free_mb)
        if mode == OffloadMode.CPU:
            raise MemoryError(
                f"TabICL {self.enc_name}: {output_mb / 1024:.2f} GiB output does not fit "
                "GPU or configured scratch disk; refusing an unbounded CPU/swap fallback. "
                "Check scratch space/quota. No rows or features were dropped."
            )
        if self.enc_name == "tf_col":
            LOG.info("TabICL embedding storage=%s, estimated=%.2f GiB, ensemble batch=1",
                     mode.name, output_mb / 1024)
        return mode, reason

    def _allocate_output_buffer(self, mode, shape, dtype):
        # Upstream's allocator catches a failed GPU allocation and immediately
        # tries CPU, bypassing _resolve_offload_mode. Do not enter that branch.
        if mode == OffloadMode.GPU:
            try:
                out = torch.empty(shape, dtype=dtype, device=self.exe_device)
                return out, {"mode": mode.name, "shape": shape, "dtype": str(dtype)}
            except torch.OutOfMemoryError as exc:
                raise _RetryOnDisk from exc
        if mode != OffloadMode.DISK:
            raise MemoryError("TabICL full-output CPU allocation is disabled")
        size = self._estimate_tensor_mb(shape, self.disk_dtype or dtype)
        free = self.get_available_disk_space(self.disk_offload_dir)
        safe = max(0, free - self.disk_min_free_mb) * self.disk_safety_factor
        if self.disk_offload_dir is None or size > safe:
            raise MemoryError(f"TabICL needs {size / 1024:.2f} GiB scratch space; "
                              "disk allocation refused before writing")
        output, info = super()._allocate_output_buffer(OffloadMode.DISK, shape, dtype)
        if hasattr(os, "posix_fallocate"):
            # truncate() creates a sparse file. Reserve blocks before touching a
            # huge mmap, so exhausted quota produces an OSError, not a SIGBUS.
            try:
                with open(output.path, "r+b") as stream:
                    os.posix_fallocate(stream.fileno(), 0, output.nbytes)
            except OSError as exc:
                del output
                raise OSError(f"Cannot reserve TabICL scratch storage; check quota: {exc}") from exc
        return output, info


def _single_member_embeddings(module, args, kwargs):
    """Avoid upstream's full-tensor repeat, including repeat(1), after offload.

    The ensemble generator has ALREADY shuffled X's columns. With one member,
    the optional shuffle shortcut just copies identical embeddings. Taking the
    ordinary upstream path preserves the values and the disk-backed storage.
    """
    if not module.training and kwargs.get("feature_shuffles") is not None:
        X = args[0] if args else kwargs["X"]
        if X.shape[0] != 1:
            raise ValueError("Bounded TabICL inference requires ensemble batch_size=1")
        kwargs = {**kwargs, "feature_shuffles": None}
    return args, kwargs


def install(estimator) -> None:
    """Install on this fitted estimator only; never patch package/global state."""
    if estimator.batch_size != 1:
        raise ValueError("Bounded TabICL inference requires ensemble batch_size=1")
    model = estimator.model_
    if isinstance(model.col_embedder.inference_mgr, BoundedInferenceManager):
        return
    for module in model.modules():
        manager = getattr(module, "inference_mgr", None)
        if isinstance(manager, InferenceManager):
            module.inference_mgr = BoundedInferenceManager(manager.enc_name, manager.out_dim, manager.out_no_seq)
    model.col_embedder.register_forward_pre_hook(_single_member_embeddings, with_kwargs=True)
    from src.eval.tabicl_streaming import install_streaming

    install_streaming(model)


@contextmanager
def prediction_storage(estimator):
    """Give each prediction temporary disk storage in the configured job workspace.

    Preserve upstream's storage dtype and all inference settings affecting
    predictions. Disable pooled asynchronous copies so multiple large pinned
    buffers cannot accumulate outside the job's memory budget.
    """
    if estimator.device_.type == "cpu":
        yield
        return
    root = work_root()
    if root is None:
        from src.utils.paths import big_outputs_dir

        # A generic /tmp may be a RAM filesystem; use the project's big tier.
        root = big_outputs_dir() / "inference_tmp"
    root.mkdir(parents=True, exist_ok=True)
    original = estimator.inference_config_
    current = deepcopy(original)
    # TemporaryDirectory removes only its own unique directory, never the job root.
    with tempfile.TemporaryDirectory(prefix="tabicl-offload-", dir=root) as folder:
        try:
            for cfg in (current.COL_CONFIG, current.ROW_CONFIG, current.ICL_CONFIG):
                cfg.update(dict(disk_offload_dir=str(Path(folder)), use_async=False,
                                max_pinned_memory_mb=0.0, disk_flush_mb=256.0))
            estimator.inference_config_ = current
            LOG.info("TabICL temporary inference storage: %s", folder)
            yield
        finally:
            estimator.inference_config_ = original
