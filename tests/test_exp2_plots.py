"""The Exp2 fine-tuning figures — training behaviour and benchmark scores for `2.1`-`2.4`.

Exp2 reuses the Level-1 plot modules with `exp="exp2"`; what is genuinely new is (a) reading an
Exp2 arm's fine-tuning levers out of its name, (b) the credit-vs-out-of-domain retention curve,
and (c) isolating the effect of each fine-tuning lever on the benchmark. These assert the QUANTITY
each encodes, not the drawing — the same discipline as `test_exp1_plots.py`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.visualize import results_plots as rp
from src.visualize import training_plots as tp

# A representative Exp2 arm name, exactly as `run_name` builds it (sorted lever paths).
ARM = ("exp2_pd__init-strategy=icl_only__prior-credit_fraction=0p5__"
       "train-l2sp_alpha=0p003__train-lr=1em05__s0")
CONTROL = ("exp2_pd__init-strategy=full__prior-credit_fraction=0__"
           "train-l2sp_alpha=0__train-lr=1em06__s0")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# -- reading an Exp2 arm's fine-tuning levers ---------------------------------


def test_arm_label_reads_the_finetuning_levers_not_the_prior_levers():
    """An Exp2 label must show strategy / L2-SP / LR, and never leak the `__prior…` that
    follows `strategy=icl_only` in the name (a greedy `[a-z_]+` used to swallow it)."""
    label = tp.arm_label(ARM)
    assert label == "cf0.5·icl·l2·lr1e-05·s0", label
    assert "__prior" not in label


def test_arm_label_marks_l2sp_off_for_the_zero_arm():
    assert "noL2" in tp.arm_label(CONTROL)
    assert "l2" in tp.arm_label(ARM) and "noL2" not in tp.arm_label(ARM)


def test_arm_label_covers_every_freeze_strategy():
    for strat, short in (("full", "full"), ("icl_only", "icl"), ("head_only", "head")):
        name = ARM.replace("init-strategy=icl_only", f"init-strategy={strat}")
        assert short in tp.arm_label(name).split("·")


def test_exp1_labels_are_unchanged_by_the_exp2_branch():
    """The Exp1 notebooks still call `arm_label`; their format must not move."""
    exp1 = ("exp1_pd__mechanism-rho_range=[0.12,0.3]__prior-credit_fraction=1__"
            "filter-mode=banded__s2")
    assert tp.arm_label(exp1) == "cf1·banded·aggr·s2"


# -- the model column and kind, on Exp2 results -------------------------------


def test_model_column_prefers_the_lever_bearing_column_over_constant_model():
    """Every one of our arms has model='crediticl'; grouping on it would collapse 60 arms into
    one bar. The arm's run name lives in `info_run_name`, and that is what must be grouped on."""
    df = pd.DataFrame({
        "model": ["crediticl", "crediticl", "catboost"],
        "info_run_name": [ARM, CONTROL, ""],
        "auc": [0.72, 0.70, 0.71],
    })
    assert rp._model_col(df) == "info_run_name"


def test_kind_separates_control_from_credit_by_the_fraction():
    assert rp._kind(ARM) == "credit"
    assert rp._kind(CONTROL) == "control"
    assert rp._kind("catboost") == "baseline"


# -- the manifests are read per experiment ------------------------------------


def _progress(with_ood: bool) -> pd.DataFrame:
    cols = {"step": [0, 100, 200], "train_loss": [0.7, 0.5, 0.45],
            "real__german__auc": [0.60, 0.68, 0.71]}
    if with_ood:
        cols["ood__letter__auc"] = [0.80, 0.74, 0.70]  # OOD drifting DOWN as credit rises
    return pd.DataFrame(cols)


def test_progress_is_loaded_per_experiment_not_across_them(isolated_output):
    from src.utils import paths

    man = paths.manifests_dir()
    man.mkdir(parents=True, exist_ok=True)
    _progress(True).to_csv(man / "exp1_pd__a__progress.csv", index=False)
    _progress(True).to_csv(man / "exp2_pd__b__progress.csv", index=False)
    assert set(tp.load_progress("pd", "exp1")) == {"exp1_pd__a"}
    assert set(tp.load_progress("pd", "exp2")) == {"exp2_pd__b"}


def test_real_vs_ood_draws_both_curves_when_ood_is_logged(isolated_output):
    from src.utils import paths

    man = paths.manifests_dir()
    man.mkdir(parents=True, exist_ok=True)
    _progress(True).to_csv(man / f"{ARM}__progress.csv", index=False)
    fig = tp.real_vs_ood("pd", "exp2")
    labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert any("real" in str(x) for x in labels), labels
    assert any("out-of-domain" in str(x) for x in labels), labels


def test_real_vs_ood_degrades_to_the_credit_curve_without_ood(isolated_output):
    from src.utils import paths

    man = paths.manifests_dir()
    man.mkdir(parents=True, exist_ok=True)
    _progress(False).to_csv(man / f"{ARM}__progress.csv", index=False)
    fig = tp.real_vs_ood("pd", "exp2")  # must not raise when no ood__ columns exist
    labels = [str(ln.get_label()) for ln in fig.axes[0].get_lines()]
    assert any("real" in x for x in labels)


# -- the lever-effect figure groups by each swept knob ------------------------


def test_lever_effect_groups_the_score_by_each_finetuning_lever(isolated_output):
    from src.utils import paths

    out = paths.results_dir("pd", "eval")
    out.mkdir(parents=True, exist_ok=True)
    # Two learning rates, two strategies — a real benchmark carries the run name in info_run_name.
    rows = []
    for lr in ("1em05", "1em06"):
        for strat in ("full", "icl_only"):
            name = (f"exp2_pd__init-strategy={strat}__prior-credit_fraction=0p5__"
                    f"train-l2sp_alpha=0p003__train-lr={lr}__s0")
            rows.append({"dataset": "german", "model": "crediticl", "seed": 0,
                         "info_run_name": name, "auc": 0.70 if lr == "1em06" else 0.73})
    pd.DataFrame(rows).to_csv(out / "results_exp2bench_pd_a0.csv", index=False)

    fig = rp.lever_effect("pd", "exp2")
    # At least the learning-rate and strategy panels should have bars.
    drawn = [ax for ax in fig.axes if ax.patches]
    assert len(drawn) >= 2, "lever_effect drew no grouped bars"


def test_lever_effect_degrades_before_the_benchmark_runs(isolated_output):
    fig = rp.lever_effect("pd", "exp2")  # no results on disk
    assert fig is not None and len(fig.axes) >= 1
