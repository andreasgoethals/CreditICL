"""How to measure "mass on the boundary" without fooling yourself.

This module exists because the obvious version is wrong, and wrong in a direction
that flatters the project.

The naive metric is `(y <= 0).mean()` and `(y >= 1).mean()`. That is only correct
when the target actually lives on [0,1]. Applied to a **standard-scaled** target —
which is exactly what the original TabICL prior produces — `y <= 0` means
"below the mean", so it returns about 0.5. Measured that way, the unmodified prior
appears to put 54% of its mass "at zero", which would make it look like it already
solves the LGD problem. It does not; the metric was lying.

The scale-invariant fix is to ask how much mass sits on the target's **extreme
values**: `(y == y.min()).mean()` and `(y == y.max()).mean()`. For a continuous
target those are about 1/n each. For a genuine atom they are large. This works
whether the target has been rescaled or not, which is the whole point, since
`target_scaling` is one of the levers we sweep.

`unit_mass_at_0` / `unit_mass_at_1` are still reported, but only when the target
really is inside [0,1]; otherwise they come back as None rather than as a
misleading number.
"""

from __future__ import annotations

from typing import Any

import torch

# A continuous target on n rows puts about 1/n on its own minimum. Anything above
# this many times 1/n is a real atom rather than a tie by chance.
ATOM_MULTIPLE = 5.0


def target_stats(y: torch.Tensor) -> dict[str, Any]:
    """Describe a target's shape in a way that survives rescaling."""
    y = y.reshape(-1).float()
    n = int(y.numel())
    if n == 0:
        return {"n": 0}

    y_min, y_max = float(y.min()), float(y.max())
    frac_at_min = float((y == y_min).float().mean())
    frac_at_max = float((y == y_max).float().mean())
    n_distinct = int(len(torch.unique(y)))

    in_unit = y_min >= -1e-6 and y_max <= 1.0 + 1e-6
    threshold = ATOM_MULTIPLE / max(n, 1)

    return {
        "n": n,
        "min": y_min,
        "max": y_max,
        # The scale-invariant measures. Use these to compare priors.
        "frac_at_min": frac_at_min,
        "frac_at_max": frac_at_max,
        "boundary_mass": frac_at_min + frac_at_max,
        "has_atom_at_min": frac_at_min > threshold,
        "has_atom_at_max": frac_at_max > threshold,
        "n_distinct": n_distinct,
        "distinct_fraction": n_distinct / n,
        # Only meaningful on a [0,1] target; None otherwise, deliberately, so a
        # standard-scaled target cannot be reported as if it were bounded.
        "in_unit_interval": in_unit,
        "unit_mass_at_0": float((y <= 1e-9).float().mean()) if in_unit else None,
        "unit_mass_at_1": float((y >= 1.0 - 1e-9).float().mean()) if in_unit else None,
    }


def summarise(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-task stats across many sampled tasks."""
    import numpy as np

    if not stats_list:
        return {}

    def col(key: str) -> np.ndarray:
        return np.asarray([s[key] for s in stats_list if s.get(key) is not None], dtype=float)

    at_min, at_max = col("frac_at_min"), col("frac_at_max")
    out: dict[str, Any] = {
        "tasks": len(stats_list),
        "frac_at_min_mean": round(float(at_min.mean()), 4),
        "frac_at_max_mean": round(float(at_max.mean()), 4),
        "frac_at_min_p90": round(float(np.percentile(at_min, 90)), 4),
        "frac_at_max_p90": round(float(np.percentile(at_max, 90)), 4),
        "boundary_mass_mean": round(float((at_min + at_max).mean()), 4),
        "tasks_with_any_atom": round(
            float(np.mean([s["has_atom_at_min"] or s["has_atom_at_max"] for s in stats_list])), 4
        ),
        "tasks_with_both_atoms": round(
            float(np.mean([s["has_atom_at_min"] and s["has_atom_at_max"] for s in stats_list])), 4
        ),
        "tasks_in_unit_interval": round(float(np.mean([s["in_unit_interval"] for s in stats_list])), 4),
        "distinct_fraction_mean": round(float(col("distinct_fraction").mean()), 4),
        "target_min": round(min(s["min"] for s in stats_list), 4),
        "target_max": round(max(s["max"] for s in stats_list), 4),
    }

    unit = [s for s in stats_list if s["in_unit_interval"]]
    if unit:
        out["unit_mass_at_0_mean"] = round(float(np.mean([s["unit_mass_at_0"] for s in unit])), 4)
        out["unit_mass_at_1_mean"] = round(float(np.mean([s["unit_mass_at_1"] for s in unit])), 4)
    return out
