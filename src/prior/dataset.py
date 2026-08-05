"""Streaming batches of synthetic episodes for pretraining.

The prior is infinite, so this is an `IterableDataset` that yields whole batches
rather than a `Dataset` of fixed length. Batch construction mirrors TabICL:
every dataset in a batch shares (n_rows, n_features, train_size), because the
model consumes a dense (batch, rows, cols) tensor. TabICL achieves the same via
`batch_size_per_gp`.

Prior generation is CPU-bound and dominates wall-clock — TabICLv2's reference
scripts run it with `--prior_device cpu --n_jobs 16`. That is why `num_workers`
matters more here than in a typical training job, and why `docs/vsc.md` picks
partitions by cores-per-GPU rather than by GPU speed alone.

Each worker gets its own `PriorRNG`, seeded from (base_seed, worker_id), so the
streams are distinct but reproducible. Worker RNG state is *not* recoverable
across a restart (workers are re-spawned); see `train/checkpoint.py` for what
resume does and does not guarantee.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .generator import TaskGenerator
from .rng import PriorRNG


class PriorBatchDataset(IterableDataset):
    """Yields (X, y, train_size) batches forever."""

    def __init__(self, prior_cfg: dict[str, Any], task: str, batch_size: int, seed: int):
        super().__init__()
        self.prior_cfg = prior_cfg
        self.task = task
        self.batch_size = batch_size
        self.seed = seed
        self.train_frac_range = tuple(prior_cfg.get("train_frac_range", [0.3, 0.9]))

    def _make_generator(self) -> TaskGenerator:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        rng = PriorRNG(self.seed, worker_id=worker_id)
        return TaskGenerator(self.prior_cfg, self.task, rng)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        gen = self._make_generator()
        while True:
            yield self._sample_batch(gen)

    def _sample_batch(self, gen: TaskGenerator) -> tuple[torch.Tensor, torch.Tensor, int]:
        n_rows, n_features = gen.sample_shape()
        lo, hi = self.train_frac_range
        train_size = int(round(n_rows * gen.rng.uniform(lo, hi)))
        train_size = max(8, min(n_rows - 4, train_size))  # leave room for both splits

        xs, ys = [], []
        width = n_features
        for _ in range(self.batch_size):
            task = gen.sample(shape=(n_rows, n_features))
            xs.append(task.X)
            ys.append(task.y)
            width = max(width, task.X.shape[1])  # missingness indicators can widen X

        # Pad to a common width with zeros. Zero is the post-standardisation mean,
        # so a padded column is uninformative rather than misleading; the model
        # treats all (padded) columns uniformly, as TabICLv2 does via `ignore_d`.
        X = torch.zeros(self.batch_size, n_rows, width)
        y = torch.zeros(self.batch_size, n_rows)
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            rows = min(xi.shape[0], n_rows)
            X[i, :rows, : xi.shape[1]] = xi[:rows]
            y[i, :rows] = yi[:rows]
            if rows < n_rows:
                # A task can come back short — PD's underwriting selection drops
                # rows and rounding can leave it one or two under. Repeat existing
                # rows to fill rather than zero-padding: for classification a
                # zero-padded y would invent extra class-0 labels and quietly
                # shift the base rate we are trying to control.
                fill = gen.rng.torch_randint(rows, (n_rows - rows,))
                X[i, rows:, : xi.shape[1]] = xi[fill]
                y[i, rows:] = yi[fill]
        return X, y, train_size


def collate_identity(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    """The dataset already yields batches, so the loader must not re-batch."""
    return batch[0]


def build_loader(
    prior_cfg: dict[str, Any],
    task: str,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader:
    dataset = PriorBatchDataset(prior_cfg, task, batch_size, seed)
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
