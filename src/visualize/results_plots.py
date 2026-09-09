"""Level-1 RESULTS visualisation: how the trained arms score on the real datasets.

Reads the benchmark output — `output/results/<task>/eval/results_<stamp>.csv`, one row per
(dataset, model, seed) written by `scripts/evaluate.py` — and turns it into the final-scores
figures for `1.3_pd_results` / `1.4_lgd_results`.

The benchmark is phase 2 and only runs once every arm of a track has trained, so until then
these degrade to a single "not scored yet" panel rather than an error: the notebook is meant
to be safe to run at any point in the sweep. The notebooks call these and hold no logic.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import paths
from src.visualize import style

HEADLINE = {"pd": "auc", "lgd": "r2"}
HIGHER_IS_BETTER = {"auc": True, "ap": True, "r2": True, "rmse": False, "mae": False,
                    "pinball": False, "crps": False, "brier": False, "logloss": False}

# Names that are external baselines rather than one of our trained arms.
BASELINES = ("catboost", "xgboost", "lightgbm", "tabpfn", "tabiclv2", "tabicl", "logreg",
             "linear", "mean", "gbm", "rf", "randomforest")


def load_results(track: str) -> pd.DataFrame | None:
    """The most recent per-(dataset, model, seed) results table, or `None` if none exist."""
    out = paths.results_dir(track, "eval")
    files = sorted(out.glob("results_*.csv")) if out.exists() else []
    if not files:
        return None
    try:
        df = pd.read_csv(files[-1])
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return df if len(df) else None


def _model_col(df: pd.DataFrame) -> str:
    for c in ("model", "arm", "run", "run_name", "tag", "checkpoint"):
        if c in df.columns:
            return c
    return df.columns[0]


def _kind(model: str) -> str:
    """`baseline`, `control` (our prior at credit_fraction 0) or `credit` (our prior)."""
    low = str(model).lower()
    if any(b in low for b in BASELINES) and "crediticl" not in low:
        return "baseline"
    if "credit_fraction=0" in low or "cf0" in low or "cf=0" in low or low.endswith("control"):
        return "control"
    return "credit"


_KIND_COLOUR = {"credit": style.CREDIT, "control": style.ORIGINAL, "baseline": style.REAL}


def available_metrics(df: pd.DataFrame) -> list[str]:
    known = ["auc", "ap", "r2", "rmse", "mae", "pinball", "crps", "brier", "logloss", "ks",
             "calibration_slope", "boundary_mass_abs_err", "coverage_80"]
    return [m for m in known if m in df.columns]


# ---------------------------------------------------------------------------
# 1. Overall ranking
# ---------------------------------------------------------------------------


def overall_ranking(track: str, metric: str | None = None):
    """Mean headline score per model across datasets and seeds, sorted, coloured by kind."""
    df = load_results(track)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no benchmark results in output/results/%s/eval/ yet" % track)
        return fig

    mc = _model_col(df)
    agg = df.groupby(mc)[metric].agg(["mean", "std"]).reset_index()
    agg["kind"] = agg[mc].map(_kind)
    agg = agg.sort_values("mean", ascending=HIGHER_IS_BETTER.get(metric, True))
    fig, ax = plt.subplots(figsize=style.row_figsize(len(agg)))
    y = np.arange(len(agg))
    ax.barh(y, agg["mean"], xerr=agg["std"].fillna(0), color=[_KIND_COLOUR[k] for k in agg["kind"]],
            error_kw={"elinewidth": 0.8, "ecolor": style.MUTED})
    ax.set_yticks(y)
    ax.set_yticklabels([str(m)[:34] for m in agg[mc]], fontsize=7)
    ax.set_xlabel(f"real-data {metric} (mean over datasets and seeds)")
    ax.legend(handles=style.legend_patches({k: _KIND_COLOUR[k] for k in ("credit", "control", "baseline")}),
              loc="lower right")
    style.title(ax, f"Ranking by {metric}")
    fig.suptitle(f"{track.upper()} benchmark ranking")
    return fig


# ---------------------------------------------------------------------------
# 2. Per-dataset breakdown
# ---------------------------------------------------------------------------


def per_dataset(track: str, metric: str | None = None):
    """Headline score per dataset, model-kind best summarised, so no dataset is hidden by a mean."""
    df = load_results(track)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no per-dataset benchmark results yet")
        return fig

    mc = _model_col(df)
    df = df.copy()
    df["kind"] = df[mc].map(_kind)
    # Best score of each kind on each dataset — the fair per-dataset comparison.
    best = df.groupby(["dataset", "kind"])[metric].agg("max" if HIGHER_IS_BETTER.get(metric, True) else "min")
    piv = best.unstack("kind")
    datasets = list(piv.index)
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    x = np.arange(len(datasets))
    kinds = [k for k in ("credit", "control", "baseline") if k in piv.columns]
    w = 0.8 / max(len(kinds), 1)
    for i, k in enumerate(kinds):
        ax.bar(x + i * w, piv[k].values, width=w, color=_KIND_COLOUR[k], label=k)
    ax.set_xticks(x + w * (len(kinds) - 1) / 2)
    ax.set_xticklabels([str(d)[:16] for d in datasets], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"best {metric} per kind")
    ax.legend(loc="lower right")
    style.title(ax, f"{metric} by dataset")
    fig.suptitle(f"{track.upper()} benchmark by dataset")
    return fig


# ---------------------------------------------------------------------------
# 3. Credit prior vs control — the headline comparison
# ---------------------------------------------------------------------------


def credit_vs_control(track: str, metric: str | None = None):
    """The distribution of headline scores for credit-prior arms, control arms and baselines."""
    df = load_results(track)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no benchmark results yet — this is the figure that answers the experiment")
        return fig

    mc = _model_col(df)
    per_model = df.groupby(mc)[metric].mean().reset_index()
    per_model["kind"] = per_model[mc].map(_kind)
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.44))
    kinds = [k for k in ("credit", "control", "baseline") if (per_model["kind"] == k).any()]
    for i, k in enumerate(kinds):
        vals = per_model.loc[per_model["kind"] == k, metric].values
        ax.scatter(np.full(len(vals), i) + style_jitter(len(vals)), vals, s=34, alpha=0.6,
                   color=_KIND_COLOUR[k], edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [vals.mean(), vals.mean()], color=style.INK, lw=2, zorder=4)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_ylabel(f"real-data {metric} (per model)")
    style.title(ax, f"Credit prior vs control vs baseline by {metric}",
                "each point is one model; the bar is the group mean")
    fig.suptitle(f"{track.upper()}: does the credit prior beat control?")
    return fig


def results_summary(track: str) -> str:
    """Text summary of the benchmark, for the notebook's final cell."""
    df = load_results(track)
    if df is None:
        return (f"{track.upper()} RESULTS: no benchmark output in output/results/{track}/eval/ yet.\n"
                f"  The benchmark (phase 2) runs once every arm of the track has trained;\n"
                f"  re-run this notebook after `run_experiment 1 --submit` finishes scoring.")
    metric = HEADLINE[track]
    mc = _model_col(df)
    lines = [f"{track.upper()} RESULTS — {df[mc].nunique()} models on {df['dataset'].nunique() if 'dataset' in df else '?'} datasets",
             f"  metrics: {', '.join(available_metrics(df)) or 'none'}"]
    if metric in df.columns:
        per = df.groupby(mc)[metric].mean()
        per_kind = df.assign(kind=df[mc].map(_kind)).groupby("kind")[metric].mean()
        for k in ("credit", "control", "baseline"):
            if k in per_kind:
                lines.append(f"  {k:<9} mean {metric} = {per_kind[k]:.4f}")
        top = per.sort_values(ascending=not HIGHER_IS_BETTER.get(metric, True)).index[0]
        lines.append(f"  best model: {top}  ({metric}={per[top]:.4f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def style_jitter(n: int) -> np.ndarray:
    return (np.random.default_rng(0).random(n) - 0.5) * 0.22


def _empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=style.MUTED, fontsize=9,
            wrap=True, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
