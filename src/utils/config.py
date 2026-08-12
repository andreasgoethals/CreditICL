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


def resolve_config_path(path: str | Path) -> Path:
    """Find a config file whether the caller's cwd is the repo root or not.

    A relative path like `config/Exp1_LGD.yaml` only resolves when the process was started
    from the repo root. Jupyter starts in `notebooks/`, so every notebook died with
    `FileNotFoundError: config/Exp1_LGD.yaml` even though the file was plainly there and
    `sys.path` had been fixed up. Falling back to the repo root makes the same string
    work from anywhere, which is what callers already assume.
    """
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_absolute():
        from src.utils.paths import REPO_ROOT

        candidate = REPO_ROOT / p
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no config at {path!r}. Tried it as given and relative to the repo root. "
        f"Pass a path that exists, or an absolute one."
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Relative paths resolve against the repo root."""
    path = resolve_config_path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config must be a mapping, got {type(data).__name__}")
    return data


#: What an unfilled Exp2/Exp3 config still contains. Exp1 chooses the winning prior
#: setting, and until it has run there is nothing to put in these slots. A YAML default
#: would be worse than a hole: it would submit, run for hours, and quietly measure the
#: wrong arm.
PLACEHOLDER = "FILL_FROM_EXP1"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` wins, recursing into nested mappings rather than replacing them.

    A plain `dict.update` would let an experiment file that overrides one prior knob
    delete the other forty, which is exactly the accident this guards against.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def find_placeholders(node: Any, prefix: str = "") -> list[str]:
    """Dotted paths still holding `FILL_FROM_EXP1`. Empty means the config is runnable."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += find_placeholders(value, f"{prefix}{key}.")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += find_placeholders(value, f"{prefix}[{i}].")
    elif node == PLACEHOLDER:
        found.append(prefix.rstrip("."))
    return found


def load(path: str | Path, *, allow_placeholders: bool = False) -> dict[str, Any]:
    """THE way to read a config: resolve `prior_file:`, then apply the `sweep:` block.

    `prior_file:` is the one piece of indirection in the layout, and it earns its place.
    All three experiments on a track must sample the SAME prior — that is what makes
    them comparable — so the prior lives in one file and each experiment names it.
    Copying sixty lines of prior into six files would guarantee they drift, and a drift
    of one number silently invalidates the comparison between two experiments.

    The experiment file's own `prior:` block is merged ON TOP, key by key, which is how
    Exp2 and Exp3 pin the single setting Exp1 chose without restating the rest.

    Refuses a config still holding `FILL_FROM_EXP1` unless asked not to. `--list` and the
    tests pass `allow_placeholders=True` so a template can still be inspected; nothing
    that starts a training run does.
    """
    resolved = resolve_config_path(path)
    cfg = load_yaml(resolved)

    prior_file = cfg.pop("prior_file", None)
    if prior_file:
        # Resolve beside the experiment file, so config/ can be moved as a unit.
        cfg = _deep_merge(load_yaml(resolved.parent / prior_file), cfg)

    if not allow_placeholders:
        holes = find_placeholders(cfg)
        if holes:
            raise ValueError(
                f"{resolved.name} is still a template: {len(holes)} unfilled value(s) — "
                f"{', '.join(holes)}. Run Exp1 first, then replace every "
                f"{PLACEHOLDER} with the winning value. Refusing to expand it, because a "
                f"config that runs with a placeholder wastes GPU-hours measuring nothing."
            )
    return apply_sweep_block(cfg)


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


def apply_sweep_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Move a top-level ``sweep:`` block into the config tree it refers to.

    The block maps dotted paths to lists::

        sweep:
          prior.credit_fraction: [0.0, 0.1, 0.2, 0.3]
          seeds: [0, 1, 2]

    Why it exists: the swept knobs are what define the experiment, and they were
    scattered four levels deep across a 200-line file, so reading off "what does this
    run vary?" meant hunting for square brackets. Collecting them at the top makes the
    whole design visible at a glance, and everything below the block is then a single
    value by construction.

    Paths are written into the tree, so the rest of the pipeline sees exactly the nested
    structure it always did — nothing downstream knows this block exists.
    """
    if "sweep" not in cfg:
        return cfg
    out = copy.deepcopy(cfg)
    block = out.pop("sweep") or {}
    if not isinstance(block, dict):
        raise ValueError(f"sweep: must be a mapping of dotted paths to lists, got {type(block).__name__}")

    for path, values in block.items():
        if not isinstance(values, list):
            raise ValueError(
                f"sweep.{path} must be a LIST — the block exists to hold multi-value "
                f"knobs. Single values belong in the config body below it."
            )
        if len(values) < 2:
            raise ValueError(
                f"sweep.{path} has {len(values)} value(s). A one-value entry here is a "
                f"single value pretending to be a sweep; move it into the config body."
            )
        if path == "seeds":
            out["seeds"] = values
        else:
            _set_path(out, path, values)
    return out


def expand_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a config into one concrete config per lever combination.

    Every returned config has all sweep lists collapsed to single values, plus a
    ``_grid`` block recording which combination it is. Order is deterministic.
    """
    cfg = apply_sweep_block(cfg)
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
    cfg = apply_sweep_block(cfg)
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
