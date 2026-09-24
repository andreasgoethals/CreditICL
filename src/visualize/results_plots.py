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

HEADLINE = {"pd": "roc_auc", "lgd": "r2"}
HIGHER_IS_BETTER = {"auc": True, "ap": True, "r2": True, "rmse": False, "mae": False,
                    "roc_auc": True, "pr_auc": True,
                    "pinball": False, "crps": False, "brier": False, "logloss": False}

# Names that are external baselines rather than one of our trained arms.
BASELINES = ("catboost", "xgboost", "lightgbm", "tabpfn", "tabpfn3", "tabiclv2", "tabicl", "logreg",
             "linear", "mean", "gbm", "rf", "randomforest")

# Sweep levers that identify an Exp2 arm, as they appear in a run name / benchmark tag.
_LEVER_TOKENS = ("credit_fraction=", "strategy=", "l2sp_alpha=", "-lr=", "exp1_", "exp2_")


def load_results(track: str, exp: str = "exp1") -> pd.DataFrame | None:
    """Per-(dataset, model, seed) results for `exp` on `track`, or `None` if none exist.

    Concatenates the experiment's arm files (`results_<exp>bench_<track>_*.csv`) with the shared
    reference (`results_reference_<track>.csv`). Other experiment and legacy CSVs are
    excluded so old results cannot be mistaken for this experiment's benchmark.
    """
    out = paths.results_dir(track, "eval")
    if not out.exists():
        return None
    from src.eval.selection import ROOT
    from src.utils.config import expand_with_seeds, load
    cfg = load(ROOT / f"config/Exp{int(exp.removeprefix('exp'))}_{track.upper()}.yaml",
               allow_placeholders=True)
    files = [out / f"results_{exp}bench_{track}_a{i}.csv"
             for i in range(len(expand_with_seeds(cfg)))]
    files = [p for p in files if p.is_file()]
    ref = out / f"results_reference_{track}.csv"
    if ref.is_file():
        files.append(ref)
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "status" in df:
        df = df[df["status"].eq("ok")]
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


#: Credit prior blue and control grey as everywhere; external baselines olive — they were drawn in
#: the real-data orange, which made "CatBoost" read as a measurement on real data.
_KIND_COLOUR = {"credit": style.CREDIT, "control": style.ORIGINAL, "baseline": style.BASELINE}
_KIND_LABEL = {"credit": "credit prior (cf > 0)", "control": "control (cf = 0)",
               "baseline": "external baseline"}


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
    known = ["roc_auc", "pr_auc", "auc", "ap", "r2", "rmse", "mae", "pinball", "crps", "brier", "logloss", "ks",
             "calibration_slope", "boundary_mass_abs_err", "coverage_80"]
    return [m for m in known if m in df.columns]


# ---------------------------------------------------------------------------
# The split: development chooses the prior, holdout reports it
# ---------------------------------------------------------------------------

#: How a split reads in a heading — the protocol in EXPERIMENTAL_DESIGN.md §5: development is where
#: the prior is chosen, the holdout is untouched until the end and is what gets reported.
ROLE_LABEL = {"development": "development datasets", "holdout": "holdout datasets", None: "all datasets"}


def _roles(track: str, exp: str) -> dict[str, str]:
    from src.visualize.training_plots import dataset_roles

    return dataset_roles(track, exp)


def _restrict(df: pd.DataFrame, track: str, exp: str, role: str | None) -> pd.DataFrame:
    """The rows of the datasets with `role` ("development" / "holdout"). Every row when `role` is
    None, or when the config carries no split to restrict by."""
    roles = _roles(track, exp)
    if role is None or "dataset" not in df.columns or not roles:
        return df
    keep = df["dataset"].astype(str).map(lambda d: roles.get(d.split(".", 1)[-1]) == role)
    return df[keep]


def _role_tag(dataset: str, roles: dict[str, str]) -> str:
    """`german · dev` / `hmeq · holdout` — a dataset tick that says which side of the split it is on."""
    role = roles.get(str(dataset).split(".", 1)[-1])
    short = {"development": "dev", "holdout": "holdout"}.get(role)
    name = str(dataset).split(".", 1)[-1][:16]
    return f"{name} · {short}" if short else name


# ---------------------------------------------------------------------------
# 1. Overall ranking
# ---------------------------------------------------------------------------


def overall_ranking(track: str, exp: str = "exp1", metric: str | None = None):
    """Development-only configuration ranking, with variation over training seeds."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns:
        return _placeholder(f"no benchmark results in output/results/{track}/eval/ yet")

    from src.eval.selection import development_ranking
    agg = development_ranking(df, track, exp, metric)
    mc = "configuration"
    if agg.empty:
        return _placeholder("no complete development-set configuration scores yet")
    agg["kind"] = agg[mc].map(_kind)
    agg = agg.sort_values("mean", ascending=HIGHER_IS_BETTER.get(metric, True))
    fig, ax = plt.subplots(figsize=style.row_figsize(len(agg), base=1.3))
    y = np.arange(len(agg))
    ax.barh(y, agg["mean"], xerr=agg["std"].fillna(0), color=[_KIND_COLOUR[k] for k in agg["kind"]],
            error_kw={"elinewidth": 0.8, "ecolor": style.MUTED})
    ax.set_yticks(y)
    ax.set_yticklabels([str(m)[:34] for m in agg[mc]], fontsize=7)
    ax.set_xlabel(f"development {style.metric_label(metric)}: mean over datasets and seeds "
                  f"(whisker: seed SD)")
    kinds = [k for k in ("credit", "control", "baseline") if (agg["kind"] == k).any()]
    style.legend_below(ax, style.legend_patches({_KIND_LABEL[k]: _KIND_COLOUR[k] for k in kinds}))
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
        return _placeholder("no per-dataset benchmark results yet")

    df, mc = _with_identity(df)
    df["kind"] = df[mc].map(_kind)
    roles = _roles(track, exp)
    # Best score of each kind on each dataset — the fair per-dataset comparison.
    best = df.groupby(["dataset", "kind"])[metric].agg("max" if HIGHER_IS_BETTER.get(metric, True) else "min")
    piv = best.unstack("kind")
    rank = {"development": 0, "holdout": 1}
    datasets = sorted(piv.index, key=lambda d: (rank.get(roles.get(str(d).split(".", 1)[-1]), 2), str(d)))
    piv = piv.loc[datasets]
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    x = np.arange(len(datasets))
    kinds = [k for k in ("credit", "control", "baseline") if k in piv.columns]
    w = 0.8 / max(len(kinds), 1)
    for i, k in enumerate(kinds):
        ax.bar(x + i * w, piv[k].values, width=w, color=_KIND_COLOUR[k], label=_KIND_LABEL[k])
    ax.set_xticks(x + w * (len(kinds) - 1) / 2)
    ax.set_xticklabels([_role_tag(d, roles) for d in datasets], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"best {style.metric_label(metric)} of each kind")
    style.legend_below(ax, ncol=3)
    return fig


# ---------------------------------------------------------------------------
# 3. Credit prior vs control — the headline comparison
# ---------------------------------------------------------------------------


def credit_vs_control(track: str, exp: str = "exp1", metric: str | None = None,
                      role: str | None = None):
    """The distribution of headline scores for credit-prior arms, control arms and baselines, on
    the datasets of one side of the split (`role="holdout"` is what the experiment reports)."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is not None:
        df = _restrict(df, track, exp, role)
    if df is None or metric not in df.columns or not len(df):
        return _placeholder("no benchmark results yet")

    df, mc = _with_identity(df)
    per_model = df.groupby(mc)[metric].mean().reset_index()
    per_model["kind"] = per_model[mc].map(_kind)
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.42))
    kinds = [k for k in ("credit", "control", "baseline") if (per_model["kind"] == k).any()]
    for i, k in enumerate(kinds):
        vals = per_model.loc[per_model["kind"] == k, metric].values
        ax.scatter(np.full(len(vals), i) + style_jitter(len(vals)), vals, s=34, alpha=0.6,
                   color=_KIND_COLOUR[k], edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [vals.mean(), vals.mean()], color=style.INK, lw=2, zorder=4)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([_KIND_LABEL[k] for k in kinds])
    ax.set_ylabel(f"{style.metric_label(metric)}, mean over {ROLE_LABEL.get(role, role)}")
    style.legend_below(ax, [plt.Line2D([], [], color=style.MUTED, marker="o", ls="none",
                                       label="one model"),
                            plt.Line2D([], [], color=style.INK, lw=2, label="group mean")], ncol=2)
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
    if df is None or metric not in df.columns:
        return _placeholder("no benchmark results yet")
    fig, axes = plt.subplots(2, 2, figsize=style.grid_figsize(2, 2, panel_ratio=0.62), sharey=True)
    axes = np.atleast_1d(axes).ravel()

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
                           fontsize=7)
        ax.set_ylabel(f"mean {style.metric_label(metric)}", fontsize=8)
        ax.set_xlabel(label, fontsize=8)
        drawn = True
    for ax in axes:
        if not ax.has_data() and ax.axison:
            ax.axis("off")
    if not drawn:
        _empty(axes[0], "results carry no recognisable sweep levers to group by")
    return fig


# ---------------------------------------------------------------------------
# 5. Deeper views — every metric, every dataset, and the reference gap
# ---------------------------------------------------------------------------


def metric_grid(track: str, exp: str = "exp1", role: str | None = None):
    """One panel per benchmark metric, each a bar per model kind — the whole scoreboard at once,
    on the datasets of one side of the split."""
    df = load_results(track, exp)
    style.apply()
    if df is not None:
        df = _restrict(df, track, exp, role)
        df = df if len(df) else None
    metrics = available_metrics(df) if df is not None else []
    if df is None or not metrics:
        return _placeholder("no benchmark results in output/results/ yet")
    ncols = 3
    nrows = max(1, int(np.ceil(len(metrics) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    df, mc = _with_identity(df)
    kinds = [k for k in ("credit", "control", "baseline") if df[mc].map(_kind).eq(k).any()]
    dk = df.assign(_kind=df[mc].map(_kind))
    for ax, metric in zip(axes, metrics):
        ax.axis("on")
        means = dk.groupby("_kind")[metric].mean()
        vals = [means.get(k, np.nan) for k in kinds]
        ax.bar(range(len(kinds)), vals, color=[_KIND_COLOUR[k] for k in kinds], width=0.66)
        goal = style.metric_goal(metric)
        if isinstance(goal, (int, float)):
            ax.axhline(goal, color=style.INK, lw=0.9, ls="--")
        ax.set_xticks([])
        ax.set_title(f"{style.metric_label(metric)} {style.goal_mark(metric)}".strip(), loc="left",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    style.legend_below(fig, style.legend_patches({_KIND_LABEL[k]: _KIND_COLOUR[k] for k in kinds}),
                       ncol=3)
    return fig


def per_dataset_heatmap(track: str, exp: str = "exp1", metric: str | None = None):
    """Best headline score of each model kind on each dataset, as an annotated heatmap."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        return _placeholder("no per-dataset benchmark results yet")
    df, mc = _with_identity(df)
    d = df.assign(_kind=df[mc].map(_kind))
    roles = _roles(track, exp)
    agg = "max" if HIGHER_IS_BETTER.get(metric, True) else "min"
    piv = d.groupby(["dataset", "_kind"])[metric].agg(agg).unstack("_kind")
    kinds = [k for k in ("credit", "control", "baseline") if k in piv.columns]
    rank = {"development": 0, "holdout": 1}
    order = sorted(piv.index, key=lambda v: (rank.get(roles.get(str(v).split(".", 1)[-1]), 2), str(v)))
    piv = piv.loc[order, kinds]
    fig, ax = plt.subplots(figsize=style.row_figsize(len(piv), base=1.2))
    im = ax.imshow(piv.values, aspect="auto", cmap=style.CMAP_SEQ)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([_KIND_LABEL[k] for k in kinds])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([_role_tag(v, roles) for v in piv.index], fontsize=7)
    ax.grid(False)
    finite = piv.values[np.isfinite(piv.values)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    for i in range(len(piv)):
        for j in range(len(kinds)):
            v = piv.values[i, j]
            if not np.isnan(v):
                # cividis runs dark to light: white text on the dark half, ink on the light half.
                dark = (v - lo) / (hi - lo + 1e-12) < 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if dark else style.INK)
    fig.colorbar(im, ax=ax, shrink=0.55, label=f"best {style.metric_label(metric)} of the kind")
    return fig


def beats_reference(track: str, exp: str = "exp1", metric: str | None = None,
                    role: str | None = None):
    """Every trained arm's headline score minus the released TabICLv2's, on the datasets of one side
    of the split — who clears the frontier."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is not None:
        df = _restrict(df, track, exp, role)
    if df is None or metric not in df.columns or not len(df):
        return _placeholder("no benchmark results yet")
    df, mc = _with_identity(df)
    per = df.groupby(mc)[metric].mean()
    is_ref = [("tabiclv2" in str(m).lower()) for m in per.index]
    if not any(is_ref):
        return _placeholder("no released-TabICLv2 reference column in the results yet")
    ref = float(per[is_ref].mean())
    ours = per[[_kind(m) in ("credit", "control") for m in per.index]]
    if ours.empty:
        return _placeholder("no trained arms scored yet")
    higher = HIGHER_IS_BETTER.get(metric, True)
    order = ours.sort_values(ascending=not higher)
    delta = order.values - ref
    fig, ax = plt.subplots(figsize=style.row_figsize(len(order), per_row=0.12, base=1.3))
    ax.barh(np.arange(len(order)), delta,
            color=[_KIND_COLOUR[_kind(m)] for m in order.index])
    ax.axvline(0, color=style.REFERENCE, lw=1.2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(m)[:30] for m in order.index], fontsize=6)
    ax.set_xlabel(f"{style.metric_label(metric)} minus the released TabICLv2's, mean over the "
                  f"{ROLE_LABEL.get(role, role)}")
    kinds = [k for k in ("credit", "control") if any(_kind(m) == k for m in order.index)]
    style.legend_below(ax, style.legend_patches({_KIND_LABEL[k]: _KIND_COLOUR[k] for k in kinds})
                       + [plt.Line2D([], [], color=style.REFERENCE, lw=1.2,
                                     label="released TabICLv2 (0)")], ncol=3)
    return fig


def results_summary(track: str, exp: str = "exp1") -> str:
    """Text summary of the benchmark, in the notebook's order: what development selects, then how
    it does on the holdout."""
    df = load_results(track, exp)
    title = f"{exp.upper()} {track.upper()} RESULTS — the benchmark"
    if df is None:
        return (f"{title}\n  no benchmark output in output/results/{track}/eval/ yet.\n"
                f"  The benchmark (phase 2) runs once every arm of the track has trained;\n"
                f"  re-run this notebook after the phase-2 array finishes scoring.")
    metric = HEADLINE[track]
    label = style.metric_label(metric)
    df, mc = _with_identity(df)
    roles = _roles(track, exp)
    lines = [title, f"  {df[mc].nunique()} models on "
             f"{df['dataset'].nunique() if 'dataset' in df else '?'} datasets"
             f" | metrics: {', '.join(available_metrics(df)) or 'none'}"]
    if metric not in df.columns:
        return "\n".join(lines)
    lines += ["", f"A. WHICH PRIOR DEVELOPMENT SELECTS ({label})"]
    from src.eval.selection import development_ranking

    ranking = development_ranking(df, track, exp)
    if not ranking.empty:
        for _, row in ranking.head(3).iterrows():
            lines.append(f"  {row['configuration'][:60]:<60} {row['mean']:.4f}")
    else:
        lines.append("  no complete development configuration scores yet")
    for role, heading in (("holdout", "B. HOW IT DOES ON THE HOLDOUT"),):
        part = _restrict(df, track, exp, role) if roles else df
        lines += ["", f"{heading} ({label}, {ROLE_LABEL[role if roles else None]})"]
        if not len(part):
            lines.append("  no holdout rows scored yet")
            continue
        per_kind = part.assign(kind=part[mc].map(_kind)).groupby("kind")[metric].mean()
        for k in ("credit", "control", "baseline"):
            if k in per_kind:
                lines.append(f"  {k:<9} mean {label} = {per_kind[k]:.4f}")
        if "credit" in per_kind and "control" in per_kind:
            lines.append(f"  credit prior minus control: {per_kind['credit'] - per_kind['control']:+.4f}")
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


def _placeholder(message: str):
    """One compact line saying what is missing — the figure itself replaces it once the benchmark
    has run. A half-page of empty axes (or a 2x2 grid with the message wedged in one corner, as the
    lever figure had) said nothing more than this line does."""
    style.apply()
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 0.9))
    _empty(ax, message)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    return fig
