"""Config loading and grid expansion.

The convention this project uses, per the design brief: **any lever in a YAML
config may be a single value or a list**. A single value is used as-is; a list
means "run one experiment per value". All lists are crossed, so five levers
with two values each produce 32 runs.

Each expanded combination is one training run, one SLURM array task, and one
checkpoint directory. Combination order is *deterministic* (sorted lever paths,
list order preserved), so `SLURM_ARRAY_TASK_ID` maps to the same configuration
on every invocation — that is what makes a resumed or re-submitted array task
land on the run it was meant to.

Levers are addressed by dotted path (``prior.credit.target.atom_prob``).

**Literal lists vs sweeps.** Some values genuinely *are* lists — an interval to
sample from, a seed list. The rule is a naming convention, not a hand-maintained
allowlist: any key ending in ``_range`` (or named in ``NO_EXPAND_EXACT``) is
literal data and is never crossed. A curated list was tried first and immediately
leaked two keys (`n_nodes_range`, `rule_quantile_range`) into the grid, silently
turning a sampling interval into a two-point sweep — hence the suffix rule.

To sweep a range, nest it: ``boundary_mass_range: [[0.0, 0.1], [0.1, 0.3]]``.
A list whose elements are themselves lists or dicts is always structural.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

# Any key with this suffix holds a literal [lo, hi] interval to sample from.
# A suffix rule rather than an allowlist, so a new `*_range` lever cannot be
# silently reinterpreted as a two-point sweep.
NO_EXPAND_SUFFIX: str = "_range"

# Literal lists that do not follow the suffix convention. Getting one of these
# wrong is not a crash but something worse: `metrics: [pinball, crps, coverage]`
# read as a sweep would quietly turn one run into three, each computing a single
# metric.
NO_EXPAND_EXACT: frozenset[str] = frozenset(
    {
        "quantile_band",  # [lo, hi] pseudo-R^2 band for the banded filter
        "seeds",  # crossed separately, in expand_with_seeds
        "datasets",  # a dataset registry, not a sweep
        "dev_datasets",  # the frozen development split
        "holdout_datasets",  # the frozen holdout split
        "metrics",  # a list of metrics to compute, all of them, in one run
        "betas",  # optimizer (beta1, beta2)
    }
)


def is_literal_list(key: str, value: list | None = None) -> bool:
    """True when a list under `key` is data rather than a sweep.

    The `value` argument implements the documented escape hatch: a `*_range` key
    whose entries are themselves lists IS a sweep over ranges, e.g.
    ``boundary_mass_range: [[0.0, 0.1], [0.1, 0.3]]``. Without this, a nested
    range would be silently flattened into one literal value.
    """
    if value is not None and any(isinstance(v, (dict, list)) for v in value):
        return False
    return key.endswith(NO_EXPAND_SUFFIX) or key in NO_EXPAND_EXACT


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config must be a mapping, got {type(data).__name__}")
    return data


def _walk(node: Any, prefix: str = "") -> list[tuple[str, list[Any]]]:
    """Collect (dotted_path, values) for every lever that is a sweep list."""
    found: list[tuple[str, list[Any]]] = []
    if isinstance(node, dict):
        for key in sorted(node):
            path = f"{prefix}.{key}" if prefix else key
            value = node[key]
            if isinstance(value, list) and not is_literal_list(key, value):
                if len(value) == 0:
                    raise ValueError(
                        f"{path}: sweep list is empty. Give it at least one value, or — if this "
                        f"is meant to be literal data rather than a sweep — add {key!r} to "
                        "NO_EXPAND_EXACT in src/utils/config.py."
                    )
                if key.endswith(NO_EXPAND_SUFFIX):
                    # A nested range list: sweep over the ranges themselves.
                    found.append((path, value))
                elif any(isinstance(v, (dict, list)) for v in value):
                    # Structural nesting, e.g. a list of blocks. Recurse into it.
                    found.extend(_walk(value, path))
                else:
                    found.append((path, value))
            else:
                found.extend(_walk(value, path))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(_walk(item, f"{prefix}[{i}]"))
    return found


def _set_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    """Assign `value` at a dotted path, which must already exist."""
    parts = path.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def sweep_axes(cfg: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    """Return the sweep axes of a config, in deterministic order."""
    return _walk(cfg)


def expand_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a config into one concrete config per lever combination.

    Every returned config has all sweep lists collapsed to single values, plus a
    ``_grid`` block recording which combination it is. Order is deterministic.
    """
    axes = sweep_axes(cfg)
    if not axes:
        out = copy.deepcopy(cfg)
        out["_grid"] = {"index": 0, "total": 1, "assignments": {}, "tag": "base", "hash": _hash({})}
        return [out]

    paths = [p for p, _ in axes]
    value_lists = [v for _, v in axes]
    combos = list(itertools.product(*value_lists))

    expanded: list[dict[str, Any]] = []
    for index, combo in enumerate(combos):
        out = copy.deepcopy(cfg)
        assignments: dict[str, Any] = {}
        for path, value in zip(paths, combo):
            _set_path(out, path, value)
            assignments[path] = value
        out["_grid"] = {
            "index": index,
            "total": len(combos),
            "assignments": assignments,
            "tag": grid_tag(assignments),
            "hash": _hash(assignments),
        }
        expanded.append(out)
    return expanded


def _short(path: str) -> str:
    """Shorten a dotted path to its last two components for use in a run name."""
    parts = [p for p in path.split(".") if not p.startswith("[")]
    return "-".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value).replace("/", "-").replace(" ", "")


def grid_tag(assignments: dict[str, Any]) -> str:
    """A filesystem-safe, human-readable tag for one grid point."""
    if not assignments:
        return "base"
    return "__".join(f"{_short(p)}={_fmt(v)}" for p, v in sorted(assignments.items()))


def _hash(assignments: dict[str, Any]) -> str:
    blob = json.dumps(assignments, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def run_name(cfg: dict[str, Any]) -> str:
    """Directory-safe name for a run: <experiment>__<grid tag>__s<seed>."""
    experiment = cfg.get("experiment", "run")
    tag = cfg.get("_grid", {}).get("tag", "base")
    seed = cfg.get("seed", 0)
    name = f"{experiment}__{tag}__s{seed}"
    # Keep it under typical filesystem limits; fall back to the hash if long.
    if len(name) > 180:
        name = f"{experiment}__{cfg['_grid']['hash']}__s{seed}"
    return name


def expand_with_seeds(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the lever grid, then cross it with the ``seeds`` list.

    Seeds are crossed *outermost* so that consecutive array indices differ in
    levers rather than in seed. That way a truncated array still covers the
    whole lever grid at one seed instead of one lever at every seed.
    """
    seeds = cfg.get("seeds", [cfg.get("seed", 0)])
    if not isinstance(seeds, list):
        seeds = [seeds]

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for point in expand_grid(cfg):
            point = copy.deepcopy(point)
            point["seed"] = int(seed)
            point.pop("seeds", None)
            runs.append(point)

    for i, run in enumerate(runs):
        run["_grid"]["array_index"] = i
        run["_grid"]["array_total"] = len(runs)
        run["_run_name"] = run_name(run)
    return runs


def select_run(cfg: dict[str, Any], index: int) -> dict[str, Any]:
    """Return the single run at array `index`."""
    runs = expand_with_seeds(cfg)
    if not 0 <= index < len(runs):
        raise IndexError(f"array index {index} out of range: this config expands to {len(runs)} runs")
    return runs[index]
