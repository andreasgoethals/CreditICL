"""Short diagnostic routes must not become ETAs for a different table distribution."""

import pytest

from src.experiments.design import MAIN_STEPS
from src.utils.config import expand_with_seeds, load
from src.validation.verify import _print_legacy_training_costs, _print_training_costs


def _summaries():
    return {
        "scratch_tabicl": {"completed": True, "steps": 100, "train_seconds_total": 200.0},
        "finetune_tabicl": {"completed": True, "steps": 100, "train_seconds_total": 300.0},
        "finetune_tabpfn3": {"completed": True, "steps": 100, "train_seconds_total": 500.0},
    }


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_current_exp2_reports_unknown_long_context_cost_and_only_tabpfn_subtotal(track, capsys):
    runs = expand_with_seeds(load(f"config/Exp2_{track.upper()}.yaml", allow_placeholders=True))
    _print_training_costs(2, "", track, runs, _summaries())
    text = capsys.readouterr().out

    assert f"Exp2 {track} TabICL: GPU-hours unknown" in text
    assert "single capacity-test update is not a reliable campaign estimate" in text
    expected_hours = 24 * MAIN_STEPS * 5 / 3600
    assert f"Exp2 {track} TabPFN: 24 runs, about {expected_hours:.1f} GPU-h (short-run extrapolation; subtotal only)" in text
    assert f"Exp2 {track} total: unknown" in text
    assert "48 runs, about" not in text
    assert "TabICL: 24 runs, about" not in text


@pytest.mark.parametrize("exp,variant,count,hours", [
    (1, "", 12, "83.3"),
    (1, "_ablations", 15, "104.2"),
])
def test_compatible_scratch_route_projection_is_unchanged(exp, variant, count, hours, capsys):
    runs = expand_with_seeds(load(f"config/Exp{exp}_PD{variant}.yaml", allow_placeholders=True))
    _print_training_costs(exp, variant, "pd", runs, _summaries())
    text = capsys.readouterr().out

    assert f"Exp{exp}{variant} pd: {count} runs, about {hours} GPU-h (short-run extrapolation)" in text
    assert "unknown" not in text


def test_incomplete_tabpfn_route_cannot_supply_even_a_subtotal(capsys):
    summaries = _summaries()
    summaries["finetune_tabpfn3"]["completed"] = False
    runs = expand_with_seeds(load("config/Exp2_PD.yaml", allow_placeholders=True))
    _print_training_costs(2, "", "pd", runs, summaries)
    text = capsys.readouterr().out

    assert "Exp2 pd total: unknown" in text
    assert "about" not in text


def test_legacy_report_cannot_project_long_context_from_complete_short_routes(capsys):
    speeds = {(track, arm): 0.5 for track in ("pd", "lgd") for arm in _summaries()}
    sizes = {key: 200.0 for key in speeds}
    _print_legacy_training_costs(speeds, sizes)
    text = capsys.readouterr().out

    for track in ("pd", "lgd"):
        assert f"Exp2 {track} finetune_tabicl: GPU-hours unknown" in text
        assert f"Exp2 {track} finetune_tabpfn3" in text
    assert "Exp2 TabPFN subtotal only: ~33 GPU-h" in text
    assert "Exp2 total: unknown" in text
    assert "Exp2 total: ~" not in text
    assert "Exp1 total: ~167 GPU-h" in text
    assert "Exp3 total: unknown" in text


def test_retention_does_not_reuse_from_scratch_timing(capsys):
    runs = expand_with_seeds(load("config/Exp3_PD.yaml", allow_placeholders=True))
    _print_training_costs(3, "", "pd", runs, _summaries())
    text = capsys.readouterr().out
    assert "24 new runs; GPU-hours unknown" in text
    assert "about" not in text
