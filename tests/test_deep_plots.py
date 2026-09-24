"""The "deeper views" added to the Level-1/2 plot modules — the lever breakdowns, per-dataset
curves, gradient-to-weight ratios, the every-metric grid, the heatmap and the reference gap.

Same discipline as the other plot tests: assert the figure BUILDS and encodes the right quantity,
never what it looks like. These exist because the functions read a real on-disk schema (the arm's
run name lives in `info_run_name`, the reference in `model`) that is easy to get subtly wrong.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualize import results_plots as rp
from src.visualize import training_plots as tp

EXP1_ARM = "exp1_pd__mechanism-rho_range=[0.12,0.3]__prior-credit_fraction=1__filter-mode=banded__s2"
EXP2_ARM = ("exp2_pd__init-strategy=icl_only__prior-credit_fraction=0p5__"
            "train-l2sp_alpha=0p003__train-lr=1em05__s0")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _drew(fig) -> bool:
    return any(ax.patches or ax.images or ax.collections or ax.lines for ax in fig.axes)


# -- the lever parser, which every by-lever figure rests on -------------------


def test_lever_value_reads_exp1_and_exp2_levers():
    assert tp._lever_value(EXP1_ARM, "credit_fraction") == "cf=1"
    assert tp._lever_value(EXP1_ARM, "filter") == "banded"
    assert tp._lever_value(EXP1_ARM, "intensity") == "aggressive"
    assert tp._lever_value(EXP2_ARM, "strategy") == "icl_only"
    assert tp._lever_value(EXP2_ARM, "l2sp") == "L2-SP on"
    assert tp._lever_value(EXP2_ARM, "intensity") is None  # Exp2 has no intensity lever


def test_intensity_reads_mild_from_the_narrow_range():
    mild = EXP1_ARM.replace("[0.12,0.3]", "[0.03,0.12]")
    assert tp._lever_value(mild, "intensity") == "mild"


# -- the new training views build on a partial sweep --------------------------


def _write_progress(man, arm, with_ood=True):
    cols = {"step": [0, 100, 200], "train_loss": [0.7, 0.5, 0.45],
            "real__german__roc_auc": [0.60, 0.68, 0.71], "real__hmeq__roc_auc": [0.70, 0.80, 0.85]}
    if with_ood:
        cols["ood__letter__roc_auc"] = [0.80, 0.74, 0.70]
    pd.DataFrame(cols).to_csv(man / f"{arm}__progress.csv", index=False)


def _write_telemetry(man, arm):
    pd.DataFrame({"step": [0, 100, 200], "gw_ratio_col": [1e-2, 8e-3, 6e-3],
                  "gw_ratio_row": [1e-3, 9e-4, 8e-4], "gw_ratio_icl": [5e-3, 4e-3, 3e-3],
                  "gw_ratio_head": [2e-2, 1e-2, 9e-3], "steps_per_s": [0.6, 0.63, 0.64],
                  "gpu0_utilization_gpu": [88, 90, 91]}).to_csv(man / f"{arm}__telemetry.csv", index=False)


def test_new_training_views_build_on_synthetic_manifests(isolated_output):
    from src.utils import paths

    man = paths.manifests_dir()
    man.mkdir(parents=True, exist_ok=True)
    control = EXP1_ARM.replace("credit_fraction=1", "credit_fraction=0")
    for arm in (EXP1_ARM, control):
        _write_progress(man, arm)
        _write_telemetry(man, arm)
    assert _drew(tp.credit_vs_control_over_training("pd", "exp1"))
    assert _drew(tp.metric_by_lever("pd", "filter", "exp1"))
    assert _drew(tp.per_dataset_curves("pd", 1, "exp1"))
    assert tp.per_dataset_pages("pd", "exp1") >= 1
    assert _drew(tp.weight_gradient_ratios("pd", "exp1"))
    assert _drew(tp.final_metric_by_lever("pd", "exp1"))


def test_story_figures_build_on_synthetic_manifests(isolated_output):
    """The Part A/B figures added for the story: the sweep map, the monitoring map, the cost of each
    arm, the lever-within-fraction view and the seed spread."""
    from src.utils import paths

    man = paths.manifests_dir()
    man.mkdir(parents=True, exist_ok=True)
    control = EXP1_ARM.replace("credit_fraction=1", "credit_fraction=0")
    for arm in (EXP1_ARM, control, EXP1_ARM.replace("__s2", "__s1")):
        _write_progress(man, arm)
        _write_telemetry(man, arm)
    for fn in (tp.sweep_map, tp.monitoring_coverage, tp.throughput, tp.lever_interaction,
               tp.seed_spread, tp.real_vs_ood):
        assert _drew(fn("pd", "exp1")), fn.__name__
    assert "A. THE RUN" in tp.training_summary("pd", "exp1")


# -- what a curve may average over: development data, finished arms --------------


def _arms(**cols):
    return {name: pd.DataFrame({"step": [0, 100, 200], **c}) for name, c in cols.items()}


def test_a_holdout_dataset_is_never_averaged_into_a_development_mean():
    """THE LEAK THIS GUARDS: curves from before the development-only protocol scored "the smallest
    few" datasets, holdout ones among them (PD hmeq/thomas). Averaging those into "which prior is
    best" selects on the test set. `german` is development in config/Exp1_PD.yaml, `hmeq` holdout."""
    runs = _arms(**{EXP1_ARM: {"real__german__roc_auc": [0.6, 0.7, 0.8],
                               "real__hmeq__roc_auc": [0.9, 0.9, 0.9]}})
    kept, excluded = tp.comparable_datasets(runs, "roc_auc")
    assert kept == ["german"] and "hmeq" in excluded
    assert tp._final(tp._mean_metric(runs[EXP1_ARM], "roc_auc", "real", kept)) == 0.8


def test_an_arm_without_a_development_monitor_drops_out_rather_than_shrinking():
    """A monitor that logged only missing values (LGD base_model, 30 of 45 arms) must neither be
    averaged over fewer datasets nor erase the dataset for everyone: that arm leaves the average."""
    other = EXP1_ARM.replace("__s2", "__s0")
    runs = _arms(**{EXP1_ARM: {"real__german__roc_auc": [0.6, 0.7, 0.8]},
                    other: {"real__german__roc_auc": [np.nan] * 3}})
    kept, _ = tp.comparable_datasets(runs, "roc_auc")
    assert kept == ["german"]
    assert set(tp._finals(runs, "roc_auc", kept)) == {EXP1_ARM}


def test_aggregates_use_finished_arms_only(isolated_output):
    """Averaging an unfinished arm up to where it stopped changes the composition of the mean there,
    and the curve jumps. The summary's `completed` flag decides; unfinished arms are left out."""
    import json

    from src.utils import paths

    runs = _arms(**{EXP1_ARM: {"train_loss": [1, 1, 1]},
                    EXP1_ARM.replace("__s2", "__s0"): {"train_loss": [1, 1, 1]}})
    paths.manifests_dir().mkdir(parents=True, exist_ok=True)
    paths.run_summary_path(EXP1_ARM).write_text(json.dumps({"completed": True}), encoding="utf-8")
    paths.run_summary_path(EXP1_ARM.replace("__s2", "__s0")).write_text(
        json.dumps({"completed": False}), encoding="utf-8")
    assert tp.completed_arms(runs) == {EXP1_ARM}
    assert set(tp._aggregate(runs)) == {EXP1_ARM}


def test_metric_labels_and_fraction_colours_are_publication_ready():
    from src.visualize import style

    assert style.metric_label("roc_auc") == "ROC-AUC" and style.metric_label("r2") == "R²"
    assert style.metric_label("some_new_metric") == "some new metric"  # words, never code
    assert style.credit_fraction_colour(0) == style.ORIGINAL
    assert style.credit_fraction_colour(1) == style.CREDIT
    assert style.credit_fraction_colour(0.5) not in (style.CREDIT_MILD, style.CREDIT_STRONG)


def test_new_training_views_degrade_without_data():
    # No isolated_output: manifests_dir has no exp3_ files, so every view is a placeholder.
    for fn in (tp.credit_vs_control_over_training, tp.weight_gradient_ratios,
               tp.final_metric_by_lever):
        fig = fn("pd", "exp3")
        assert fig is not None and len(fig.axes) >= 1


# -- the new results views, and the reference-row identity fix ----------------


def _results_df():
    rows = []
    for cf in ("0", "0p5", "1"):
        for ds in ("german", "hmeq"):
            rows.append({"dataset": ds, "model": "crediticl", "seed": 0,
                         "info_run_name": f"exp2_pd__prior-credit_fraction={cf}__train-lr=1em06__s0",
                         "roc_auc": 0.70 + (0.02 if cf != "0" else 0.0), "brier": 0.1})
    for ds, a in (("german", 0.735), ("hmeq", 0.9)):  # the reference: name in `model`, blank run-name
        rows.append({"dataset": ds, "model": "tabiclv2", "seed": 0, "info_run_name": "",
                     "roc_auc": a, "brier": 0.09})
    return pd.DataFrame(rows)


def test_identity_classifies_the_reference_row_correctly():
    """The bug this guards: the reference's run-name column is blank, so classifying kind off it
    alone made `tabiclv2` read as `credit`. The identity must fall back to the `model` column."""
    df, idc = rp._with_identity(_results_df())
    kinds = {m: rp._kind(m) for m in df[idc].unique()}
    assert kinds["tabiclv2"] == "baseline"
    assert sum(k == "credit" for k in kinds.values()) == 2   # cf=0.5 and cf=1
    assert sum(k == "control" for k in kinds.values()) == 1  # cf=0


def test_new_results_views_build_with_a_reference(monkeypatch):
    df = _results_df()
    monkeypatch.setattr(rp, "load_results", lambda track, exp="exp1": df)
    for fn in (rp.metric_grid, rp.per_dataset_heatmap, rp.beats_reference,
               rp.credit_vs_control):
        assert _drew(fn("pd", "exp2")), fn.__name__
    # This partial fixture cannot be used to choose a prior.
    assert not _drew(rp.overall_ranking("pd", "exp2"))


def test_results_restrict_to_one_side_of_the_split(monkeypatch):
    """EXPERIMENTAL_DESIGN §5: the prior is chosen on development data and REPORTED on the holdout,
    which stays untouched until the end. `german` is development in config/Exp1_PD.yaml and `hmeq`
    holdout, so a holdout view must see only hmeq — and label each dataset with its side."""
    df = _results_df()
    holdout = rp._restrict(df, "pd", "exp1", "holdout")
    assert set(holdout["dataset"]) == {"hmeq"}
    assert set(rp._restrict(df, "pd", "exp1", "development")["dataset"]) == {"german"}
    assert len(rp._restrict(df, "pd", "exp1", None)) == len(df)
    roles = rp._roles("pd", "exp1")
    assert rp._role_tag("0008.german", roles) == "german · dev"
    assert rp._role_tag("hmeq", roles) == "hmeq · holdout"
    monkeypatch.setattr(rp, "load_results", lambda track, exp="exp1": df)
    for fn in (rp.credit_vs_control, rp.beats_reference):
        assert _drew(fn("pd", "exp2", role="holdout")), fn.__name__
    assert _drew(rp.metric_grid("pd", "exp2", role="holdout"))
    assert "B. HOW IT DOES ON THE HOLDOUT" in rp.results_summary("pd", "exp2")


def test_new_results_views_degrade_before_the_benchmark(monkeypatch):
    monkeypatch.setattr(rp, "load_results", lambda track, exp="exp1": None)
    for fn in (rp.metric_grid, rp.per_dataset_heatmap, rp.beats_reference):
        fig = fn("pd", "exp2")
        assert fig is not None and len(fig.axes) >= 1
