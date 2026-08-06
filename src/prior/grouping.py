"""Correlated hyperparameter sampling — TabICL's groups and subgroups.

WHAT IT IS. TabICL does not draw a dataset's hyperparameters afresh every time.
Its `SCMPrior` samples in a hierarchy:

    group    (4 datasets)  one draw of META-distributions
      subgroup             concrete values drawn from those
        dataset            its own causal graph

So the datasets inside a batch are *relatives*, not strangers. Some batches come
out uniformly easy, some uniformly hard, some all wide. This is inherited from
TabPFN's design and the intent is that the prior covers a *hierarchy* of variation
rather than flat noise.

WHY IT IS ON BY DEFAULT HERE. NanoTabICL — which our prior is built from — removed
it ("no correlated sampling of scalar variables") and reports similar performance,
which is why the first version of this project did not have it. That was the wrong
default, for a reason worth stating plainly: **`credit_fraction=0.0` is the control
arm, and the control is supposed to be TabICL.** Grouping is a TabICL feature, so
having it makes the control *more* faithful, not less. Turning it off is the
ablation, not the baseline.

There is also a reason specific to this project. A group that shares
hyperparameters *is* a small domain — a set of related tasks drawn from one
regime. That is the same object as "a domain-targeted prior", which is the whole
research question. So `group_size` is not just fidelity plumbing; it is a lever
worth sweeping, and "does coherent grouping matter more or less than
credit-specific content?" is a real result either way.

Set `group_size: 1` to recover the old flat behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rng import PriorRNG


@dataclass
class GroupState:
    """The hyperparameters shared by the datasets currently in one group."""

    group_index: int = 0
    subgroup_index: int = 0
    remaining_in_group: int = 0
    remaining_in_subgroup: int = 0
    # Drawn once per group, then narrowed per subgroup.
    meta: dict[str, Any] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {"group": self.group_index, "subgroup": self.subgroup_index, **self.shared}


class GroupedSampler:
    """Hands out correlated hyperparameters, one dataset at a time.

    Call `next_dataset()` before generating each dataset; it returns the shared
    values that dataset should use, advancing the group and subgroup counters as
    needed. Which keys are shared is deliberately narrow — the *shape* of the task
    and how hard it is, not the causal graph itself, which stays independent per
    dataset exactly as upstream does.
    """

    def __init__(self, cfg: dict[str, Any], rng: PriorRNG):
        self.rng = rng
        gcfg = cfg.get("grouping", {}) or {}
        self.group_size = max(1, int(gcfg.get("group_size", 4)))
        self.subgroup_size = max(1, int(gcfg.get("subgroup_size", self.group_size)))
        self.enabled = self.group_size > 1

        # Ranges the group draws its meta-distribution from. Narrower than the
        # global ranges on purpose: the point is that a group is a *region* of the
        # space, not the whole thing.
        self.n_rows_range = tuple(cfg.get("n_rows_range", [512, 1024]))
        self.n_features_range = tuple(cfg.get("n_features_range", [3, 50]))
        self.max_features = int(cfg.get("max_features", 64))
        self.state = GroupState()

    # -- one group -----------------------------------------------------------
    def _new_group(self) -> None:
        s = self.state
        s.group_index += 1
        s.remaining_in_group = self.group_size

        # The META level: pick a REGION of each range, not a value. A group whose
        # region sits at the small end produces several small tables in a row,
        # which is the coherence the hierarchy is for.
        lo, hi = self.n_rows_range
        span = max(1, (hi - lo) // 2)
        rows_lo = self.rng.randint(lo, max(lo + 1, hi - span + 1))
        flo, fhi = self.n_features_range
        fhi = min(fhi, self.max_features)
        fspan = max(1, (fhi - flo) // 2)
        feat_lo = self.rng.randint(flo, max(flo + 1, fhi - fspan + 1))

        s.meta = {
            "n_rows_region": (rows_lo, min(hi, rows_lo + span)),
            "n_features_region": (feat_lo, min(fhi, feat_lo + fspan)),
            # Difficulty region, so a group is uniformly easy or uniformly hard.
            "signal_region": (self.rng.uniform(0.3, 0.8), 1.0),
        }
        s.remaining_in_subgroup = 0

    def _new_subgroup(self) -> None:
        s = self.state
        s.subgroup_index += 1
        s.remaining_in_subgroup = min(self.subgroup_size, s.remaining_in_group)

        # The CONCRETE level: fixed values, drawn from the group's regions, shared
        # by every dataset in this subgroup.
        rlo, rhi = s.meta["n_rows_region"]
        flo, fhi = s.meta["n_features_region"]
        slo, shi = s.meta["signal_region"]
        s.shared = {
            "n_rows": self.rng.randint(rlo, rhi + 1),
            "n_features": self.rng.randint(flo, fhi + 1),
            "signal_strength": self.rng.uniform(slo, shi),
        }

    def next_dataset(self) -> dict[str, Any]:
        """Shared hyperparameters for the next dataset. Empty when disabled."""
        if not self.enabled:
            return {}
        s = self.state
        if s.remaining_in_group <= 0:
            self._new_group()
        if s.remaining_in_subgroup <= 0:
            self._new_subgroup()
        s.remaining_in_group -= 1
        s.remaining_in_subgroup -= 1
        return dict(s.shared)

    def describe(self) -> dict[str, Any]:
        """For the log, so a run records how its batches were built."""
        return {
            "grouping_enabled": self.enabled,
            "group_size": self.group_size,
            "subgroup_size": self.subgroup_size,
            "groups_drawn": self.state.group_index,
            "subgroups_drawn": self.state.subgroup_index,
        }
