"""Fixed grids keep all rate outcomes and cannot consume pilot winners."""
import copy

import pytest

from src.experiments.design import apply_design, DESIGN_REVISION, MAIN_STEPS, MILESTONES, RATES
from src.utils.config import expand_with_seeds, load, run_name


@pytest.mark.parametrize("task", ["PD", "LGD"])
def test_both_rates_cross_all_mixtures_and_two_seeds_without_selection(task, isolated_output):
    cfg = load(f"config/Exp2_{task}.yaml")
    runs = expand_with_seeds(cfg)
    assert len(runs) == 32
    assert all(not r["design_unresolved"] and r["design_revision"] == DESIGN_REVISION for r in runs)
    assert {(r["architecture"], r["lr_index"], r["prior"]["credit_fraction"], r["seed"]) for r in runs} == {
        (model, rate, mixture, seed) for model in RATES for rate in (0, 1)
        for mixture in (0., .25, .5, 1.) for seed in (0, 1)}
    assert all(r["train"]["max_steps"] == r["train"]["schedule_total_steps"] == MAIN_STEPS for r in runs)
    assert all(r["train"]["keep_steps"] == list(MILESTONES) for r in runs)
    assert len(set(MILESTONES)) == len(MILESTONES) == 5 and MILESTONES[-1] == MAIN_STEPS
    assert all(r["train"]["l2sp_alpha"] == 0. for r in runs)
    assert "selection_scope" not in cfg["eval"]


@pytest.mark.parametrize("index", [-1, 2, True, .5, None])
def test_invalid_rate_id_cannot_silently_select_a_python_index(index):
    run = expand_with_seeds(load("config/Exp2_PD.yaml"))[0]
    run["lr_index"] = index
    with pytest.raises(ValueError, match="lr_index"):
        apply_design(run)


@pytest.mark.parametrize("design", ["cpt_pilot", "retention_pilot"])
def test_legacy_pilots_fail_even_if_a_local_copy_remains(design, tmp_path):
    run = expand_with_seeds(load("config/Exp2_PD.yaml"))[0]
    run["design"] = design
    with pytest.raises(ValueError, match="retired"):
        apply_design(run)
    path = tmp_path / "old.yaml"
    path.write_text(f"design: {design}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retired"):
        load(path, allow_placeholders=True)


def test_a_historical_winner_cannot_override_the_registered_rate():
    original = expand_with_seeds(load("config/Exp2_PD.yaml"))[0]
    with pytest.raises(ValueError, match="Selection"):
        apply_design(copy.deepcopy(original), {"choices": {"pd/tabicl": {"full_lr": 999.}}})
    with pytest.raises(ValueError, match="revision"):
        apply_design(dict(original, design_revision="old"))
    assert original["train"]["lr"] == RATES[original["architecture"]][original["lr_index"]]


def test_new_revisions_isolate_even_the_early_checkpoint_run_names():
    for experiment in (2, 3):
        runs = expand_with_seeds(load(f"config/Exp{experiment}_PD.yaml"))
        assert len({run_name(r) for r in runs}) == len(runs)
        for run in runs:
            historical = dict(run)
            historical.pop("design_revision")
            assert run_name(historical) != run_name(run)
            assert DESIGN_REVISION in run_name(run)
    for run in expand_with_seeds(load("config/Exp1_PD.yaml")):
        assert "design_revision" not in run and DESIGN_REVISION not in run_name(run)


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_experiment1_keeps_its_existing_names_and_appended_quarter_mixture_indices(task):
    runs = expand_with_seeds(load(f"config/Exp1_{task.upper()}.yaml"))
    expected = [f"exp1_{task}__prior-credit_fraction={mixture}__s{seed}"
                for seed in range(3) for mixture in ("0", "0p5", "1")]
    expected += [f"exp1_{task}__prior-credit_fraction=0p25__s{seed}" for seed in range(3)]
    assert [r["_run_name"] for r in runs] == expected
    assert [r["_grid"]["array_index"] for r in runs] == list(range(12))


def test_checkpoint_plot_draws_measured_milestones_and_step_zero(monkeypatch):
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.visualize import continuation as plots

    frame = pd.DataFrame([
        dict(architecture="tabicl", lr_index=rate, mixture=fraction,
             checkpoint_step=step, training_seed=seed, dataset="held-out",
             roc_auc=.6 + step / 100000 + fraction / 100 + rate / 1000)
        for rate in (0, 1) for fraction in (0, .25, .5, 1) for step in MILESTONES
        for seed in (0, 1)])
    monkeypatch.setattr(plots.results, "benchmark_frames", lambda *a, **k: frame)
    monkeypatch.setattr(plots.results, "released_score", lambda *a: .6)
    fig = plots.checkpoint_curves("pd", "tabicl")
    try:
        assert len(fig.axes) == 4  # One row per fixed rate, credit and OOD columns.
        for index, ax in enumerate(fig.axes):
            assert len(ax.lines) == 5  # Four mixtures plus released reference.
            assert list(ax.lines[0].get_xdata()) == list(MILESTONES)
            rate = index // 2
            assert list(ax.lines[0].get_ydata()) == pytest.approx(
                [.6 + step / 100000 + rate / 1000 for step in MILESTONES])
            assert f"LR {RATES['tabicl'][rate]:g}" in ax.get_title(loc="left")
            assert ax.lines[-1].get_ydata() == [.6, .6]
    finally:
        plt.close(fig)
