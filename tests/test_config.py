"""Config loading and grid expansion.

These tests exist because this code already shipped one real bug: a hand-curated
list of "do not sweep" keys leaked `n_nodes_range` and `rule_quantile_range` into
the grid, turning two sampling intervals into two-point sweeps and inflating the
PD grid to 6,144 runs. The `_range` suffix rule replaced it, and
`test_range_keys_are_never_swept` is the regression test.

Deliberately no torch import here, so these run even on a bare environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import (  # noqa: E402
    apply_sweep_block,
    expand_grid,
    expand_with_seeds,
    is_literal_list,
    load_yaml,
    sweep_axes,
)

CONFIGS = ["config/LGD.yaml", "config/PD.yaml"]


@pytest.mark.parametrize("path", CONFIGS)
def test_config_loads(path):
    cfg = apply_sweep_block(load_yaml(ROOT / path))
    # No "model": the architecture is fixed in NanoTabICLv2's defaults, not configured.
    for key in ("experiment", "task", "seeds", "prior", "train", "init"):
        assert key in cfg, f"{path} is missing the '{key}' block"


@pytest.mark.parametrize("path", CONFIGS)
def test_task_matches_filename(path):
    cfg = apply_sweep_block(load_yaml(ROOT / path))
    expected = "lgd" if "LGD" in path else "pd"
    assert cfg["task"] == expected
    assert cfg["experiment"] == expected


def test_single_value_is_not_swept():
    cfg = {"a": 1, "b": "x", "c": {"d": True}}
    assert sweep_axes(cfg) == []
    assert len(expand_grid(cfg)) == 1


def test_list_becomes_a_sweep():
    cfg = {"a": [1, 2, 3]}
    assert len(expand_grid(cfg)) == 3


def test_lists_are_crossed():
    """Five settings with two values each must give 32 runs, as the docs claim."""
    cfg = {f"k{i}": [0, 1] for i in range(5)}
    assert len(expand_grid(cfg)) == 32


def test_nested_paths_are_addressed_by_dots():
    cfg = {"prior": {"credit": {"x": [1, 2]}}}
    axes = sweep_axes(cfg)
    assert axes == [("prior.credit.x", [1, 2])]
    runs = expand_grid(cfg)
    assert [r["prior"]["credit"]["x"] for r in runs] == [1, 2]


def test_range_keys_are_never_swept():
    """Regression test for the bug that inflated the PD grid to 6,144 runs."""
    cfg = {
        "n_rows_range": [512, 1024],
        "rule_quantile_range": [0.1, 0.9],
        "some_range": [1, 2],
        "quantile_band": [0.0, 0.5],
        "real_sweep": [1, 2],
    }
    axes = dict(sweep_axes(cfg))
    assert "real_sweep" in axes
    for literal in ("n_rows_range", "rule_quantile_range", "some_range", "quantile_band"):
        assert literal not in axes, f"{literal} must be literal data, not a sweep"
    assert len(expand_grid(cfg)) == 2


def test_is_literal_list_uses_a_suffix_rule():
    assert is_literal_list("anything_range")
    assert is_literal_list("quantile_band")
    assert is_literal_list("seeds")
    assert not is_literal_list("credit_fraction")


def test_nested_range_can_be_swept():
    """The documented escape hatch: a list of ranges IS a sweep."""
    cfg = {"boundary_mass_range": [[0.0, 0.1], [0.1, 0.3]]}
    runs = expand_grid(cfg)
    assert len(runs) == 2
    assert runs[0]["boundary_mass_range"] == [0.0, 0.1]


def test_empty_sweep_list_is_an_error():
    with pytest.raises(ValueError):
        expand_grid({"a": []})


@pytest.mark.parametrize("path", CONFIGS)
def test_grid_is_deterministic(path):
    """A resubmitted array task must land on the same config. Non-negotiable."""
    cfg = load_yaml(ROOT / path)
    a = [r["_run_name"] for r in expand_with_seeds(cfg)]
    b = [r["_run_name"] for r in expand_with_seeds(load_yaml(ROOT / path))]
    assert a == b


@pytest.mark.parametrize("path", CONFIGS)
def test_run_names_are_unique(path):
    """Two runs sharing a name would overwrite each other's checkpoints."""
    names = [r["_run_name"] for r in expand_with_seeds(load_yaml(ROOT / path))]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("path", CONFIGS)
def test_seeds_are_crossed_outermost(path):
    """A cut-short array should cover the lever grid at one seed, not one lever
    at every seed. That only holds if seed is the outer loop."""
    runs = expand_with_seeds(load_yaml(ROOT / path))
    n_seeds = len(apply_sweep_block(load_yaml(ROOT / path))["seeds"])
    per_seed = len(runs) // n_seeds
    assert {r["seed"] for r in runs[:per_seed]} == {runs[0]["seed"]}


@pytest.mark.parametrize("path", CONFIGS)
def test_grid_stays_a_sane_size(path):
    """A guard against re-committing an accidental combinatorial explosion."""
    runs = expand_with_seeds(load_yaml(ROOT / path))
    assert len(runs) <= 200, (
        f"{path} expands to {len(runs)} runs. That is almost certainly an accident — "
        "open one lever group at a time."
    )


@pytest.mark.parametrize("path", CONFIGS)
def test_credit_fraction_is_a_probability(path):
    cfg = apply_sweep_block(load_yaml(ROOT / path))
    values = cfg["prior"]["credit_fraction"]
    for v in values if isinstance(values, list) else [values]:
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("path", CONFIGS)
def test_control_arm_is_present(path):
    """credit_fraction = 0 is the baseline everything is measured against. If it
    is missing there is nothing to compare to."""
    values = apply_sweep_block(load_yaml(ROOT / path))["prior"]["credit_fraction"]
    values = values if isinstance(values, list) else [values]
    assert 0.0 in values, "the control arm (credit_fraction 0.0) must be in the sweep"


@pytest.mark.parametrize("path", CONFIGS)
def test_init_strategy_is_known(path):
    cfg = load_yaml(ROOT / path)
    strategies = cfg["init"]["strategy"]
    for s in strategies if isinstance(strategies, list) else [strategies]:
        assert s in ("scratch", "full", "icl_only", "head_only")


@pytest.mark.parametrize("path", CONFIGS)
def test_pretrained_path_required_when_not_scratch(path):
    """Catch the config mistake, not the crash six minutes into a queued job."""
    cfg = load_yaml(ROOT / path)
    strategies = cfg["init"]["strategy"]
    strategies = strategies if isinstance(strategies, list) else [strategies]
    if any(s != "scratch" for s in strategies):
        assert cfg["init"].get("pretrained_path"), (
            "init.strategy includes a non-scratch option but pretrained_path is null"
        )


@pytest.mark.parametrize("path", CONFIGS)
def test_no_model_block_in_the_configs(path):
    """The architecture is TabICLv2's, is identical for LGD and PD, and never varies — so
    it lives in NanoTabICLv2's defaults, not in a config block each file would have to
    repeat and keep in sync. See test_train.py for the check that those defaults match
    the paper's Table A.1."""
    assert "model" not in load_yaml(ROOT / path), (
        f"{path} has a model: block; the architecture is fixed in code"
    )


def test_config_folder_has_no_subfolders():
    """Everything lives in LGD.yaml or PD.yaml, by request."""
    entries = sorted(p.name for p in (ROOT / "config").iterdir())
    assert entries == ["LGD.yaml", "PD.yaml"], f"unexpected entries in config/: {entries}"
