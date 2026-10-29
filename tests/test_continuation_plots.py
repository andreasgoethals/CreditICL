"""Nonempty continuation figures use paired seeds and keep rates/backbones separate."""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.visualize import continuation as c
from src.experiments.design import MILESTONES, MAIN_STEPS


def curve_cells(experiment=2):
    rows = []
    for backbone in ("tabicl", "tabpfn3"):
        for lr in ((0, 1) if experiment == 2 else (0,)):
            for method in (0., .25, .5, 1.) if experiment == 2 else tuple(c.METHODS):
                for step in MILESTONES:
                    for seed in (0, 1, 2):
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
