"""Level-1 RESULTS visualisation: how the trained arms score on the real datasets.

Reads the benchmark output — `output/results/<task>/eval/results_<tag>.csv`, one row per
(dataset, model, seed) written by `scripts/evaluate.py` — and turns it into the final-scores
figures for `1.3_pd_results` / `1.4_lgd_results` (Exp1, `exp="exp1"`) and
`2.3_pd_results` / `2.4_lgd_results` (Exp2, `exp="exp2"`).

Each phase-2 array task writes its own `results_<tag>.csv` (arms tagged `exp{N}bench_<track>_a<i>`,
the shared reference column `reference_<track>`), so an experiment's table is the concatenation of
its arm files plus the reference. `exp` selects which arm files to read; the reference is shared
across experiments and always included.

The benchmark only runs once every arm of a track has trained, so until then these degrade to a
single "not scored yet" panel rather than an error: the notebook is meant to be safe to run at
any point in the sweep. The notebooks call these and hold no logic.
"""

from __future__ import annotations

import re
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

# Sweep levers that identify an Exp2 arm, as they appear in a run name / benchmark tag.
_LEVER_TOKENS = ("credit_fraction=", "strategy=", "l2sp_alpha=", "-lr=", "exp1_", "exp2_")


def load_results(track: str, exp: str = "exp1") -> pd.DataFrame | None:
    """Per-(dataset, model, seed) results for `exp` on `track`, or `None` if none exist.

    Concatenates the experiment's arm files (`results_<exp>bench_<track>_*.csv`) with the shared
    reference (`results_reference_<track>.csv`). Falls back to every `results_*.csv` when the
    tagged files are absent, so an older single-file run still renders.
    """
    out = paths.results_dir(track, "eval")
    if not out.exists():
        return None
    files = sorted(out.glob(f"results_{exp}bench_{track}_*.csv"))
    ref = out / f"results_reference_{track}.csv"
    if ref.is_file():
        files.append(ref)
    if not files:
        files = sorted(out.glob("results_*.csv"))
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df if len(df) else None


def _model_col(df: pd.DataFrame) -> str:
    """The column that best identifies an arm.

    Prefers whichever column actually carries the sweep levers or an `exp{N}_` run name, because
    the `model` column is the constant `"crediticl"` for every one of our arms — grouping on it
    would collapse 60 fine-tuning arms into a single bar. Falls back to the first familiar name.
    """
    best, best_hits = None, 0
    for c in df.columns:
        if df[c].dtype != object:
            continue
        hits = int(df[c].astype(str).str.contains("|".join(map(re.escape, _LEVER_TOKENS))).sum())
        if hits > best_hits:
            best, best_hits = c, hits
    if best is not None:
        return best
    for c in ("tag", "run_name", "run", "arm", "checkpoint", "model"):
        if c in df.columns:
            return c
    return df.columns[0]


def _kind(model: str) -> str:
    """`baseline`, `control` (our prior at credit_fraction 0) or `credit` (our prior)."""
    low = str(model).lower()
    if any(b in low for b in BASELINES) and "crediticl" not in low:
        return "baseline"
    # cf=0 EXACTLY: `credit_fraction=0__` and `cf0·`, not the `credit_fraction=0p5` of a 0.5 arm,
    # which a bare `"credit_fraction=0" in low` substring test wrongly read as a control.
    if (re.search(r"credit_fraction=0(?:\.0|p0)?(?=__|$)", low)
            or re.search(r"(?:^|[^0-9])cf0(?:\.0)?(?=[^0-9.]|$)", low)
            or low.endswith("control")):
        return "control"
    return "credit"


_KIND_COLOUR = {"credit": style.CREDIT, "control": style.ORIGINAL, "baseline": style.REAL}


def _with_identity(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """`(df + an '__id__' column, '__id__')`, where the id identifies a model correctly for BOTH
    our arms and the reference.

    Our 60 fine-tuning arms all share `model = "crediticl"` and are told apart by the run name in
    `info_run_name`; the reference and baselines (released TabICLv2, CatBoost, …) carry their name in
    `model` and leave the run-name column blank. Grouping on either column alone therefore collapses
    or misclassifies one side — so the id is the run name where it is present and `model` otherwise.
    """
    df = df.copy()
    lever_col = _model_col(df)
    ident = df[lever_col].astype(str)
    if "model" in df.columns and lever_col != "model":
        has_lever = ident.str.contains("|".join(map(re.escape, _LEVER_TOKENS)), na=False)
        ident = ident.where(has_lever, df["model"].astype(str))
    df["__id__"] = ident
    return df, "__id__"


def available_metrics(df: pd.DataFrame) -> list[str]:
    known = ["auc", "ap", "r2", "rmse", "mae", "pinball", "crps", "brier", "logloss", "ks",
             "calibration_slope", "boundary_mass_abs_err", "coverage_80"]
    return [m for m in known if m in df.columns]


# ---------------------------------------------------------------------------
# 1. Overall ranking
# ---------------------------------------------------------------------------


def overall_ranking(track: str, exp: str = "exp1", metric: str | None = None):
    """Mean headline score per model across datasets and seeds, sorted, coloured by kind."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no benchmark results in output/results/%s/eval/ yet" % track)
        return fig

    df, mc = _with_identity(df)
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


def per_dataset(track: str, exp: str = "exp1", metric: str | None = None):
    """Headline score per dataset, model-kind best summarised, so no dataset is hidden by a mean."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no per-dataset benchmark results yet")
        return fig

    df, mc = _with_identity(df)
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


def credit_vs_control(track: str, exp: str = "exp1", metric: str | None = None):
    """The distribution of headline scores for credit-prior arms, control arms and baselines."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no benchmark results yet — this is the figure that answers the experiment")
        return fig

    df, mc = _with_identity(df)
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


# ---------------------------------------------------------------------------
# 4. Which fine-tuning lever moved the score (Exp2)
# ---------------------------------------------------------------------------

#: The Exp2 sweep levers, and how each reads in a benchmark tag / run name.
_LEVERS = (("credit_fraction", r"credit_fraction=([0-9p.]+)", "credit fraction"),
           ("strategy", r"strategy=(full|icl_only|head_only|scratch)", "freeze strategy"),
           ("l2sp_alpha", r"l2sp_alpha=([0-9pm.e+-]+)", "L2-SP alpha"),
           ("lr", r"-lr=([0-9pm.e+-]+)", "learning rate"))


def lever_effect(track: str, exp: str = "exp2", metric: str | None = None):
    """One panel per fine-tuning lever: mean headline score grouped by that lever's value.

    Reads the arm's swept levers out of its benchmark tag and averages the headline metric over
    every arm that shares a value, so each panel isolates one knob — credit fraction, freeze
    strategy, L2-SP on/off, learning rate. Degrades to a placeholder before the benchmark runs.
    """
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=style.grid_figsize(2, 2, panel_ratio=0.72))
    axes = np.atleast_1d(axes).ravel()
    if df is None or metric not in df.columns:
        for ax in axes:
            ax.axis("off")
        _empty(axes[0], "no benchmark results yet — this figure isolates each fine-tuning knob")
        return fig

    mc = _model_col(df)
    names = df[mc].astype(str)
    drawn = False
    for ax, (_key, pattern, label) in zip(axes, _LEVERS):
        vals = names.str.extract(pattern, expand=False)
        d = df.assign(_lever=vals).dropna(subset=["_lever"])
        if d["_lever"].nunique() < 2:
            ax.axis("off")
            continue
        grp = d.groupby("_lever")[metric].mean().sort_index()
        x = np.arange(len(grp))
        ax.bar(x, grp.values, color=style.CREDIT, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).replace("p", ".").replace("m", "-") for v in grp.index],
                           fontsize=7, rotation=20, ha="right")
        ax.set_ylabel(metric, fontsize=8)
        style.title(ax, label)
        drawn = True
    for ax in axes:
        if not ax.has_data() and ax.axison:
            ax.axis("off")
    if not drawn:
        _empty(axes[0], "results carry no recognisable sweep levers to group by")
    fig.suptitle(f"{track.upper()} effect of each fine-tuning lever on {metric}")
    return fig


# ---------------------------------------------------------------------------
# 5. Deeper views — every metric, every dataset, and the reference gap
# ---------------------------------------------------------------------------


def metric_grid(track: str, exp: str = "exp1"):
    """One panel per benchmark metric, each a bar per model kind — the whole scoreboard at once."""
    df = load_results(track, exp)
    style.apply()
    metrics = available_metrics(df) if df is not None else []
    ncols = 3
    nrows = max(1, int(np.ceil(len(metrics) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.82))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    if df is None or not metrics:
        _empty(axes[0], "no benchmark results in output/results/ yet")
        return fig
    df, mc = _with_identity(df)
    kinds = [k for k in ("credit", "control", "baseline") if df[mc].map(_kind).eq(k).any()]
    dk = df.assign(_kind=df[mc].map(_kind))
    for ax, metric in zip(axes, metrics):
        ax.axis("on")
        means = dk.groupby("_kind")[metric].mean()
        vals = [means.get(k, np.nan) for k in kinds]
        ax.bar(range(len(kinds)), vals, color=[_KIND_COLOUR[k] for k in kinds], width=0.66)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds, fontsize=7, rotation=15, ha="right")
        arrow = "↑" if HIGHER_IS_BETTER.get(metric, True) else "↓"
        style.title(ax, f"{metric}  ({arrow} better)")
    fig.suptitle(f"{track.upper()} every metric by model kind")
    return fig


def per_dataset_heatmap(track: str, exp: str = "exp1", metric: str | None = None):
    """Best headline score of each model kind on each dataset, as an annotated heatmap."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.35))
        _empty(ax, "no per-dataset benchmark results yet")
        return fig
    df, mc = _with_identity(df)
    d = df.assign(_kind=df[mc].map(_kind))
    agg = "max" if HIGHER_IS_BETTER.get(metric, True) else "min"
    piv = d.groupby(["dataset", "_kind"])[metric].agg(agg).unstack("_kind")
    kinds = [k for k in ("credit", "control", "baseline") if k in piv.columns]
    piv = piv[kinds]
    fig, ax = plt.subplots(figsize=style.row_figsize(len(piv)))
    im = ax.imshow(piv.values, aspect="auto", cmap=style.CMAP_SEQ)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([str(v)[:18] for v in piv.index], fontsize=7)
    ax.grid(False)
    for i in range(len(piv)):
        for j in range(len(kinds)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=ax, shrink=0.55, label=metric)
    style.title(ax, f"Best {metric} per dataset and kind")
    fig.suptitle(f"{track.upper()} {metric} heatmap")
    return fig


def beats_reference(track: str, exp: str = "exp1", metric: str | None = None):
    """Every trained arm's headline score minus the released TabICLv2's — who clears the frontier."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    if df is None or metric not in df.columns:
        _empty(ax, "no benchmark results yet")
        return fig
    df, mc = _with_identity(df)
    per = df.groupby(mc)[metric].mean()
    is_ref = [("tabiclv2" in str(m).lower()) for m in per.index]
    if not any(is_ref):
        _empty(ax, "no released-TabICLv2 reference column in results yet")
        return fig
    ref = float(per[is_ref].mean())
    ours = per[[_kind(m) in ("credit", "control") for m in per.index]]
    if ours.empty:
        _empty(ax, "no trained arms scored yet")
        return fig
    higher = HIGHER_IS_BETTER.get(metric, True)
    order = ours.sort_values(ascending=not higher)
    delta = order.values - ref
    better = delta > 0 if higher else delta < 0
    ax.barh(np.arange(len(order)), delta,
            color=[style.CREDIT if b else style.MUTED for b in better])
    ax.axvline(0, color=style.REFERENCE, lw=1.2)
    ax.set_yticks([])
    ax.set_xlabel(f"{metric} minus released TabICLv2 (right of 0 = beats it)")
    n = int(better.sum())
    style.title(ax, f"{n}/{len(order)} arms beat released TabICLv2")
    fig.suptitle(f"{track.upper()} vs the released reference")
    return fig


def results_summary(track: str, exp: str = "exp1") -> str:
    """Text summary of the benchmark, for the notebook's final cell."""
    df = load_results(track, exp)
    if df is None:
        return (f"{exp.upper()} {track.upper()} RESULTS: no benchmark output in "
                f"output/results/{track}/eval/ yet.\n"
                f"  The benchmark (phase 2) runs once every arm of the track has trained;\n"
                f"  re-run this notebook after the phase-2 array finishes scoring.")
    metric = HEADLINE[track]
    df, mc = _with_identity(df)
    lines = [f"{exp.upper()} {track.upper()} RESULTS — {df[mc].nunique()} models on "
             f"{df['dataset'].nunique() if 'dataset' in df else '?'} datasets",
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
