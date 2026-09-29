"""The task stream: which synthetic dataset lands in which slot of which batch.

WHY THIS MODULE EXISTS — COMMON RANDOM NUMBERS. Experiment 1 compares arms that differ only in
`credit_fraction`. If every arm draws its own tasks, a difference between two arms mixes the
effect of the prior with the luck of the draw. So the stream is decided by the SLOT, not by
the process that generates it:

* batch ``b`` holds ``B`` slots; slot ``j`` of batch ``b`` has its own seeds, derived from
  ``(stream_seed, b, j, kind, attempt)`` with numpy's ``SeedSequence``;
* a slot is a CREDIT slot when its coin ``u < credit_fraction``. The coin is the same in every
  arm, so the 25 % arm's credit slots are a subset of the 50 % arm's, and every slot that is a
  base slot in two arms holds the IDENTICAL upstream task in both;
* group-level draws (row count, context/query split, feature count) come from the group's own
  seed, so they are shared too.

Two arms with the same seed therefore see the same TabICL tasks in the same order and start
from the same weights (`Trainer` seeds torch before building the model); they differ only where
one of them swaps a base task for a credit task. The stream is also independent of the number
of DataLoader workers and survives a resume exactly, since batch ``b`` depends on ``b`` alone.

THE BATCH STRUCTURE IS UPSTREAM'S (`GraphPrior.get_batch` in the pinned dump,
`src/tabicl/prior/_dataset.py`):

* groups of ``group_size`` datasets share the row count and the context/query split
  (``seq_len``, ``train_size = int(seq_len * U(min_train_size, max_train_size))``);
* subgroups of ``subgroup_size`` (upstream's default: the whole group) share the number of
  features, ``round(U(min_features, max_features))``, after `adjust_max_features`;
* every dataset draws its own number of classes, ``randint(2, max_classes + 1)``.

A micro-batch is one group (upstream's ``micro_batch_size == batch_size_per_gp``), so each
forward pass sees one split point, and the trainer trims it to the widest real table in it.

Python's per-process string hashing also enters: upstream's `RandomDataset.sample` orders its
feature groups with ``list(set(...))``. `scripts/pretrain.py` fixes ``PYTHONHASHSEED`` so the
same slot seed yields the same table in every arm's process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Distinct stream "kinds", so the coin, the class count and the task itself never share a seed.
KINDS = {"group": 1, "subgroup": 2, "coin": 3, "classes": 4, "base": 5, "credit": 6}


def slot_seed(stream_seed: int, batch: int, slot: int, kind: str, attempt: int = 0) -> int:
    """A 31-bit seed for one draw, decided by where it sits in the stream and nothing else."""
    words = [int(stream_seed) & 0xFFFFFFFF, int(batch) & 0xFFFFFFFF, int(slot) & 0xFFFFFFFF,
             KINDS[kind], int(attempt) & 0xFFFFFFFF]
    return int(np.random.SeedSequence(words).generate_state(1, dtype=np.uint32)[0] >> 1)


def _rng(stream_seed: int, batch: int, slot: int, kind: str) -> np.random.Generator:
    return np.random.default_rng(slot_seed(stream_seed, batch, slot, kind))


def adjust_max_features(seq_len: int, max_features: int) -> int:
    """Upstream's `Prior.adjust_max_features`, verbatim: fewer features for longer tables."""
    if seq_len <= 10240:
        return max_features
    if seq_len <= 20000:
        return min(80, max_features)
    if seq_len <= 30000:
        return min(60, max_features)
    if seq_len <= 40000:
        return min(40, max_features)
    if seq_len <= 50000:
        return min(30, max_features)
    if seq_len <= 60000:
        return min(20, max_features)
    if seq_len <= 65000:
        return min(15, max_features)
    return 10


@dataclass
class SlotPlan:
    """Everything decided about one slot before its table is generated."""

    batch: int
    slot: int
    group: int
    seq_len: int
    train_size: int
    num_features: int
    num_classes: int | None  # None for regression
    use_credit: bool


@dataclass
class BatchPlan:
    batch: int
    slots: list[SlotPlan] = field(default_factory=list)

    @property
    def seq_lens(self) -> list[int]:
        return [s.seq_len for s in self.slots]

    @property
    def train_sizes(self) -> list[int]:
        return [s.train_size for s in self.slots]


def _sample_seq_len(rng: np.random.Generator, lo: int, hi: int) -> int:
    # Upstream: `max_seq_len` when there is no range, else `np.random.randint(min, max)`.
    return int(hi) if int(lo) >= int(hi) else int(rng.integers(int(lo), int(hi)))


def plan_batch(stream_seed: int, batch: int, cfg: dict[str, Any], batch_size: int,
               *, regression: bool, credit_fraction: float) -> BatchPlan:
    """The slots of batch `batch`, in upstream's group / subgroup / dataset hierarchy."""
    gcfg = cfg.get("grouping", {}) or {}
    group_size = max(1, int(gcfg.get("group_size", 4)))
    subgroup_size = max(1, min(int(gcfg.get("subgroup_size", group_size)), group_size))
    n_lo, n_hi = (int(v) for v in cfg.get("n_rows_range", [1024, 1024]))
    f_lo, f_hi = (int(v) for v in cfg.get("n_features_range", [1, 100]))
    max_features = int(cfg.get("max_features", 100))
    t_lo, t_hi = (float(v) for v in cfg.get("train_frac_range", [0.3, 0.9]))
    max_classes = max_classes_of(cfg)

    plan = BatchPlan(batch=batch)
    n_groups = -(-batch_size // group_size)
    for g in range(n_groups):
        grng = _rng(stream_seed, batch, g, "group")
        seq_len = _sample_seq_len(grng, n_lo, n_hi)
        # Upstream: `int(seq_len * np.random.uniform(min_train_size, max_train_size))`. Clamped
        # so both sides of the split keep rows even at the extremes of a custom range.
        train_size = int(seq_len * grng.uniform(t_lo, t_hi))
        train_size = max(2, min(seq_len - 2, train_size))
        gp_max_features = min(adjust_max_features(seq_len, max_features), max_features)
        size = min(group_size, batch_size - g * group_size)
        for sg_start in range(0, size, subgroup_size):
            srng = _rng(stream_seed, batch, g * group_size + sg_start, "subgroup")
            hi = max(f_lo, min(f_hi, gp_max_features))
            num_features = int(round(srng.uniform(f_lo, hi)))
            num_features = max(1, min(num_features, gp_max_features))
            for k in range(sg_start, min(sg_start + subgroup_size, size)):
                j = g * group_size + k
                coin = float(_rng(stream_seed, batch, j, "coin").random())
                use_credit = coin < credit_fraction
                if regression:
                    n_cls = None
                elif use_credit:
                    n_cls = 2  # a credit PD task is binary: default or not
                else:
                    n_cls = int(_rng(stream_seed, batch, j, "classes").integers(2, max_classes + 1))
                plan.slots.append(SlotPlan(batch=batch, slot=j, group=g, seq_len=seq_len,
                                           train_size=train_size, num_features=num_features,
                                           num_classes=n_cls, use_credit=use_credit))
    return plan


def max_classes_of(cfg: dict[str, Any]) -> int:
    """`prior.max_classes` (upstream's name); `prior.n_classes` from older configs is read as
    the same ceiling. Upstream's stage scripts pass `--max_classes 10`."""
    if "max_classes" in cfg:
        return int(cfg["max_classes"])
    return int(cfg.get("n_classes", 10))
