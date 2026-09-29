"""Streaming batches of synthetic episodes for pretraining.

The prior is infinite, so this is an `IterableDataset` that yields whole batches. A batch is
built exactly as upstream's `GraphPrior.get_batch` builds one (pinned dump,
`src/tabicl/prior/_dataset.py`), and yielded in upstream's format:

    (X, y, d, seq_lens, train_sizes)
    X            (B, T, max_features)  features, each table zero-padded to max_features columns
    y            (B, T)                targets
    d            (B,)                  real (non-constant) columns per table
    seq_lens     (B,)                  rows per table
    train_sizes  (B,)                  context rows per table (the rest are queries)

The trainer splits it into micro-batches of one group each, checks that the group shares its
split, and trims the micro-batch to its widest real table and its row count — upstream's
`validate_micro_batch` and `align_micro_batch`. A table therefore never reaches the model
padded to 100 columns when it has 12, which is how Experiment 1's batches were built until
28-09-2026: one shape and one split for all 64 tables, always 100 columns wide.

WHICH TASK LANDS WHERE is decided by `src/prior/stream.py`: batch `b` depends on `b` alone, so
every arm with the same seed sees the same base tasks, the stream does not depend on how many
workers generate it, and a resumed run continues it exactly. Worker `w` of `W` generates
batches `start + w, start + w + W, ...`, and the DataLoader returns them in order.

Prior generation is CPU-bound; `num_workers` matters more here than in a typical job.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .generator import TaskGenerator
from .rng import PriorRNG
from .stream import plan_batch

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class PriorBatchDataset(IterableDataset):
    """Yields upstream-format batches forever, starting at batch `start_batch`."""

    def __init__(self, prior_cfg: dict[str, Any], task: str, batch_size: int, seed: int,
                 start_batch: int = 0, transform: Any = None):
        super().__init__()
        # Applied to each batch INSIDE the worker that generated it, `transform(batch, b)`: the
        # TabPFN arms preprocess their tasks there (`TabPFNTaskPreparer`), off the training loop.
        self.transform = transform
        self.prior_cfg = prior_cfg
        self.task = task
        self.batch_size = batch_size
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        self.regression = task == "lgd"
        self.credit_fraction = float(prior_cfg.get("credit_fraction", 0.0))
        self.pool_cfg = prior_cfg.get("pool", {}) or {}
        self.source = str(self.pool_cfg.get("source", "generate"))
        if self.source not in ("generate", "pool"):
            raise ValueError(f"prior.pool.source must be 'generate' or 'pool', got {self.source!r}")

    def _worker(self) -> tuple[int, int]:
        info = get_worker_info()
        return (0, 1) if info is None else (info.id, info.num_workers)

    def __iter__(self) -> Iterator[Batch]:
        worker_id, n_workers = self._worker()
        if self.source == "pool":
            sampler = self._make_pool_sampler(worker_id)
            while True:
                yield self._sample_batch_from_pool(sampler)
        gen = TaskGenerator(self.prior_cfg, self.task, PriorRNG(self.seed, worker_id=worker_id),
                            stream_seed=self.seed)
        b = self.start_batch + worker_id
        while True:
            batch = self.batch(gen, b)
            yield batch if self.transform is None else self.transform(batch, b)
            b += n_workers

    def batch(self, gen: TaskGenerator, b: int) -> Batch:
        """Batch number `b` of the stream."""
        plan = plan_batch(self.seed, b, self.prior_cfg, self.batch_size,
                          regression=self.regression, credit_fraction=self.credit_fraction)
        tasks = [gen.sample_slot(slot) for slot in plan.slots]
        T = max(plan.seq_lens)
        F = max(int(t.X.shape[1]) for t in tasks)
        X = torch.zeros(len(tasks), T, F)
        y = torch.zeros(len(tasks), T)
        for i, t in enumerate(tasks):
            n = t.X.shape[0]
            X[i, :n, : t.X.shape[1]] = t.X
            y[i, :n] = t.y
        d = torch.tensor([int(t.meta["d"]) for t in tasks], dtype=torch.long)
        return (X, y, d, torch.tensor(plan.seq_lens, dtype=torch.long),
                torch.tensor(plan.train_sizes, dtype=torch.long))

    # -- pre-generated pools (legacy) ---------------------------------------------
    def _make_pool_sampler(self, worker_id: int):
        from src.prior.pool import MixedPoolSampler

        return MixedPoolSampler(
            self.task, self.credit_fraction, PriorRNG(self.seed, worker_id=worker_id),
            original_variant=self.pool_cfg.get("original_variant", "original"),
            credit_variant=self.pool_cfg.get("credit_variant", "credit_v1"),
        )

    def _sample_batch_from_pool(self, sampler) -> Batch:
        """A batch from pooled episodes, trimmed to the smallest row count and width drawn so
        every value is real, in the same five-tensor format (one split for the whole batch)."""
        drawn = [sampler.sample() for _ in range(self.batch_size)]
        n_rows = min(int(x.shape[0]) for x, _, _ in drawn)
        width = min(int(x.shape[1]) for x, _, _ in drawn)
        lo, hi = self.prior_cfg.get("train_frac_range", [0.3, 0.9])
        train_size = max(8, min(n_rows - 4, int(round(n_rows * sampler.rng.uniform(lo, hi)))))
        X = torch.zeros(self.batch_size, n_rows, width)
        y = torch.zeros(self.batch_size, n_rows)
        for i, (xi, yi, _) in enumerate(drawn):
            X[i] = xi[:n_rows, :width]
            y[i] = yi[:n_rows]
        full = torch.full((self.batch_size,), 1, dtype=torch.long)
        return X, y, full * width, full * n_rows, full * train_size


def collate_identity(batch: list[Batch]) -> Batch:
    """The dataset already yields batches, so the loader must not re-batch."""
    return batch[0]


def build_loader(
    prior_cfg: dict[str, Any],
    task: str,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    start_batch: int = 0,
    transform: Any = None,
) -> DataLoader:
    dataset = PriorBatchDataset(prior_cfg, task, batch_size, seed, start_batch=start_batch,
                                transform=transform)
    kwargs: dict[str, Any] = {
        "batch_size": 1,
        "collate_fn": collate_identity,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
