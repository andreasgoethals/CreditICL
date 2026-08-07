"""Multi-GPU data-parallel training, for the full-scale confirmation runs.

WHY THIS EXISTS

A full TabICLv2-budget checkpoint costs **24.5 GPU-days** (the paper's own figure:
20 + 2.5 + 2 across the three stages). VSC caps a job at **72 hours**. So a
single-GPU full-budget run cannot fit in one job — it needs 9 chained jobs, or more
GPUs. On Mindwell's 8x B200 nodes it becomes ~3 days, i.e. **one job**.

The parallelism is over the **batch**. One training step consumes a batch of
independent synthetic datasets, and nothing couples them, so splitting the batch
across N GPUs and averaging the gradients is exactly equivalent to one big batch —
no approximation. That is what makes this worth doing rather than merely possible.

HOW IT IS LAUNCHED

`torchrun`, not `srun`. VSC's docs note that `srun` inside a job can conflict with
the scheduler's task management; `torchrun` brings its own rendezvous and sidesteps
it:

    torchrun --standalone --nproc_per_node=4 -m scripts.pretrain --config ...

Every process runs the same script. `torchrun` sets `RANK`, `LOCAL_RANK` and
`WORLD_SIZE`; everything here reads those and does nothing at all when they are
absent, so single-GPU and CPU runs are unaffected.

WHAT CHANGES, AND WHAT MUST NOT

* **The batch is split.** With `batch_size: 64` and 4 GPUs each process handles 16.
  The *effective* batch, and therefore the number of datasets seen per step, is
  unchanged — which is what keeps a multi-GPU run comparable to a single-GPU one.
  Getting this wrong in the other direction (each GPU taking the full batch) would
  quadruple the compute budget and invalidate "matched compute".
* **Only rank 0 writes.** Logs, metrics and checkpoints come from one process.
  Without this, four processes interleave into the same log file and race on the
  same checkpoint path.
* **The prior seed differs per rank.** Otherwise all four GPUs would generate the
  *same* datasets and the effective batch would be 16 unique tables presented four
  times — a silent 4x reduction in data diversity that would look like a successful
  run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DistInfo:
    """How this process fits into the job. All-defaults means single-process."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Only the main process writes logs, metrics and checkpoints."""
        return self.rank == 0

    def describe(self) -> dict[str, Any]:
        return {
            "distributed": self.enabled,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
        }


def detect() -> DistInfo:
    """Read torchrun's environment. Returns a single-process DistInfo if unset."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return DistInfo()
    return DistInfo(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=world,
    )


def setup(info: DistInfo) -> str:
    """Initialise the process group and return the device this rank should use.

    NCCL on GPU, gloo on CPU — NCCL is GPU-only, so a CPU smoke test of the
    distributed path would hang forever on an NCCL rendezvous rather than fail.
    """
    if not info.enabled:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
        backend, device = "nccl", f"cuda:{info.local_rank}"
    else:
        backend, device = "gloo", "cpu"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    return device


def teardown(info: DistInfo) -> None:
    """Tear the group down, so a chained job does not inherit a stale one."""
    if info.enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def wrap_model(model: torch.nn.Module, info: DistInfo) -> torch.nn.Module:
    """Wrap in DistributedDataParallel when there is more than one process."""
    if not info.enabled:
        return model
    device_ids = [info.local_rank] if torch.cuda.is_available() else None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying model, for saving.

    A DDP checkpoint has every key prefixed with `module.`, so saving the wrapper
    produces weights that will not load into a plain model at evaluation time. Always
    save what this returns.
    """
    return model.module if hasattr(model, "module") else model


def local_batch_size(batch_size: int, info: DistInfo) -> int:
    """Per-process batch size, so the EFFECTIVE batch matches the config.

    Raises rather than silently rounding: if 4 GPUs were handed a batch of 10, the
    quiet options are an effective batch of 8 or of 12, and both break the matched
    -compute claim that every arm sees the same number of datasets.
    """
    if not info.enabled:
        return batch_size
    if batch_size % info.world_size != 0:
        raise ValueError(
            f"train.batch_size={batch_size} is not divisible by world_size="
            f"{info.world_size}. Pick a batch size that divides evenly, or the "
            f"effective batch — and so the compute budget — would differ between runs."
        )
    return batch_size // info.world_size


def rank_seed(seed: int, info: DistInfo) -> int:
    """A distinct prior seed per rank.

    Without this every rank generates identical datasets, so an N-GPU run would see
    the same batch N times and the real data diversity would be 1/N of what the
    config says — while every log line still looked correct.
    """
    return seed + 100_003 * info.rank


def reduce_mean(value: float, info: DistInfo, device: str) -> float:
    """Average a scalar across ranks, for logging.

    Rank 0 does the logging, and its own loss is only 1/N of the batch. Reporting it
    unreduced would make the loss curve noisier than the training actually is.
    """
    if not info.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return float(t.item() / info.world_size)
