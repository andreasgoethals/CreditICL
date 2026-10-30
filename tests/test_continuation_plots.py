"""Nonempty continuation figures use paired seeds and keep rates/backbones separate."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.experiments.design import MAIN_STEPS, MILESTONES
from src.visualize import continuation as c


def curve_cells(experiment=2):
    rows = []
    for backbone in ("tabicl", "tabpfn3"):
        for lr in ((0, 1) if experiment == 2 else (0,)):
            for method in (0., .25, .5, 1.) if experiment == 2 else tuple(c.METHODS):
                for step in MILESTONES:
                    for seed in (0, 1):
                        for dataset in ("a", "b", "c"):
                            for fold in range(5):
                                rows.append(dict(architecture=backbone, lr_index=lr, mixture=method if experiment == 2 else .5,
                                    treatment=method if experiment == 3 else "full_wd", checkpoint_step=step,
                                    training_seed=seed, dataset=dataset, seed=0, fold=fold, model="crediticl", status="ok",
                                    info_checkpoint_step=step, roc_auc=.6+step/100000+seed/1000+lr*.01))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("experiment", [2, 3])
def test_checkpoint_curves_show_fixed_rate_and_shared_released_reference(monkeypatch, experiment):
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: curve_cells(experiment))
    monkeypatch.setattr(c.results, "released_score", lambda *a: .67)
    fig = c.checkpoint_curves("pd", "tabicl", experiment=experiment)
    fig.canvas.draw()
    assert len(fig.axes) == (4 if experiment == 2 else 2)
    for ax in fig.axes:
        assert ax.get_xscale() == "log"
        assert len(ax.get_xticks(minor=True)) == 0
        assert len(ax.lines) == (5 if experiment == 2 else 6)
        assert all(list(line.get_xdata()) == list(MILESTONES) for line in ax.lines[:-1])
    assert "Released weights (step 0)" in [t.get_text() for t in fig.legends[0].texts]
    plt.close(fig)


def test_retention_effects_require_all_methods_and_pair_reused_control(monkeypatch):
    frame = curve_cells(3)
    frame = frame[frame.checkpoint_step.eq(MAIN_STEPS)].copy()
    frame.loc[frame.treatment.eq("l2sp"), "roc_auc"] += .025
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: frame)
    fig = c.retention_effects("pd", "tabicl")
    fig.canvas.draw()
    assert len(fig.axes) == 2
    assert any(float(collection.get_offsets()[0, 0]) == pytest.approx(.025) for collection in fig.axes[0].collections)
    plt.close(fig)

    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: frame[frame.treatment.ne("lora")])
    fig = c.retention_effects("pd", "tabicl")
    assert not fig.axes[0].has_data()
    plt.close(fig)


def test_historical_training_traces_do_not_create_empty_current_panels(monkeypatch):
    monkeypatch.setattr(c.training, "load_loss_logs", lambda *a: {
        "exp2_old_design__s0": pd.DataFrame({"step": [1], "train_loss": [.5]})})
    fig = c.training_curves("pd", "tabicl")
    assert len(fig.axes) == 1 and not fig.axes[0].has_data()
    assert any("current fixed-rate design" in t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)


def test_another_backbones_results_do_not_populate_an_empty_curve(monkeypatch):
    frame = curve_cells()
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: frame[frame.architecture.eq("tabicl")])
    fig = c.checkpoint_curves("pd", "tabpfn3")
    assert len(fig.axes) == 1 and not fig.axes[0].has_data()
    plt.close(fig)


def test_missing_milestones_and_incomplete_seed_pairs_break_the_curve(monkeypatch):
    frame = curve_cells()
    frame = frame[~frame.checkpoint_step.eq(500)]
    frame = frame[~(frame.checkpoint_step.eq(1000) & frame.training_seed.eq(1))]
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: frame)
    monkeypatch.setattr(c.results, "released_score", lambda *a: None)
    fig = c.checkpoint_curves("pd", "tabicl")
    try:
        for ax in fig.axes:
            for line in ax.lines:
                assert list(line.get_xdata()) == list(MILESTONES)
                assert np.isnan(line.get_ydata()[2:4]).all()
                assert np.isfinite(line.get_ydata()[[0, 1, 4]]).all()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("experiment", [2, 3])
def test_native_regression_crps_is_not_averaged_across_incompatible_units(monkeypatch, experiment):
    requested = []
    def frames(*a, domain, **k):
        requested.append(domain)
        return curve_cells(experiment).assign(crps=.12)
    monkeypatch.setattr(c.results, "benchmark_frames", frames)
    monkeypatch.setattr(c.results, "released_score", lambda *a: .13)
    fig = c.checkpoint_curves("lgd", "tabicl", experiment=experiment, metric="crps")
    try:
        assert requested == ["credit"]
        assert len(fig.axes) == (2 if experiment == 2 else 1)
        assert "different target units" in c.trajectory_caption("lgd", "crps", experiment=experiment)
    finally:
        plt.close(fig)


def test_pending_domain_has_a_notice_and_no_unmatched_released_line(monkeypatch):
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, domain, **k:
                        curve_cells() if domain == "credit" else pd.DataFrame())
    monkeypatch.setattr(c.results, "released_score", lambda *a: .67)
    fig = c.checkpoint_curves("pd", "tabicl")
    try:
        for ax in (fig.axes[1], fig.axes[3]):
            assert not ax.lines
            assert any("pending" in t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)


def test_retention_lines_and_legend_repeat_method_identity_with_shapes(monkeypatch):
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: curve_cells(3))
    monkeypatch.setattr(c.results, "released_score", lambda *a: None)
    fig = c.checkpoint_curves("pd", "tabicl", experiment=3)
    try:
        for ax in fig.axes:
            assert len({line.get_marker() for line in ax.lines}) == len(c.METHODS)
            assert len({line.get_linestyle() for line in ax.lines}) > 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_exp2_printed_summary_matches_section_order_and_registered_checkpoint_contract(monkeypatch, track):
    from src.visualize import paper_plots as paper
    monkeypatch.setattr(paper, "final_ready", lambda *a: False)
    monkeypatch.setattr(paper, "final_credit", lambda *a: None)
    monkeypatch.setattr(paper, "ood_ready", lambda *a: False)
    monkeypatch.setattr(paper.analysis, "load_ood_results", lambda *a: None)
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: pd.DataFrame())
    text = c.results_summary(track)
    headings = [line[0] for line in text.splitlines() if len(line) > 3 and line[1:4] == " · "]
    assert headings == list("ABCDE" if track == "lgd" else "ABCD")
    assert all(f"{step:,}" in text for step in MILESTONES)
    assert f"all {len(c.SEEDS)} training seeds" in text
    assert "not completely benchmarked" in text


def test_notebook_descriptions_keep_exp1_and_continuation_seed_counts_distinct():
    import json
    from pathlib import Path

    from src.utils.config import expand_with_seeds, load
    for path in Path("notebooks").rglob("*.ipynb"):
        source = "\n".join("".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"])
        if path.name.startswith(("2.", "3.")):
            assert "three training seeds" not in source and "three complete seeds" not in source
            assert "four checkpoint" not in source.lower() and "96 total" not in source
        if path.name.startswith(("2.1", "2.2")):
            total = 2 * len(expand_with_seeds(load("config/Exp2_PD.yaml")))
            assert f"{total} total" in source
            assert "Five checkpoint evaluations" in source
        if path.name.startswith("1."):
            assert "choosing an Experiment 3 mixture" not in source


def test_rate_panels_use_the_same_vertical_scale_within_each_domain(monkeypatch):
    monkeypatch.setattr(c.results, "benchmark_frames", lambda *a, **k: curve_cells())
    monkeypatch.setattr(c.results, "released_score", lambda *a: .67)
    fig = c.checkpoint_curves("pd", "tabicl")
    try:
        fig.canvas.draw()
        assert fig.axes[0].get_ylim() == fig.axes[2].get_ylim()
        assert fig.axes[1].get_ylim() == fig.axes[3].get_ylim()
    finally:
        plt.close(fig)


def test_recipe_table_includes_each_reused_primary_rate_decay_baseline():
    frame = c.retention_recipe()
    assert frame.groupby("Backbone").size().tolist() == [5, 5]
    rows = frame[frame.Method.eq(c.METHODS["full_wd"])].set_index("Backbone")
    for backbone, label in c.BACKBONES.items():
        assert float(rows.loc[label, "Peak LR"]) == c.RATES[backbone][0]
        assert rows.loc[label, "Weight decay"] == .01
        assert rows.loc[label, "Updates"] == MAIN_STEPS


def test_early_loss_trace_is_reported_before_any_monitoring_csv(monkeypatch):
    from src.visualize import campaign, training_plots
    monkeypatch.setattr(campaign, "provenance", lambda *a: "local fixture")
    monkeypatch.setattr(training_plots, "load_progress", lambda *a: {})
    monkeypatch.setattr(training_plots, "load_loss_logs", lambda *a: {
        "early_run": pd.DataFrame({"step": [100, 125], "train_loss": [.8, .7]})})
    text = campaign.training_summary("pd", "exp2")
    assert "early_run: 125 updates recorded" in text and "loss 0.7" in text
    assert "No downloaded monitoring CSVs" in text
    assert "Training health is unavailable" not in text


def test_plotting_dose_order_is_numeric_without_changing_append_only_run_ids():
    from src.utils.config import expand_with_seeds, load
    from src.visualize import training_plots
    assert training_plots._value_order("credit_fraction", {"cf=1", "cf=0.25", "cf=0.5", "cf=0", "cf=0.75"}) == [
        "cf=0", "cf=0.25", "cf=0.5", "cf=0.75", "cf=1"]
    runs = expand_with_seeds(load("config/Exp1_PD.yaml"))
    assert all(r["prior"]["credit_fraction"] == .25 for r in runs[9:12])
