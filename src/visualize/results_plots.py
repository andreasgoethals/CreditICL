"""Level-1 RESULTS visualisation: how the trained arms score on the real datasets.

Reads the benchmark output — `output CreditICL/experiment_<N>/benchmark/<task>/results_<tag>.csv`
and the shared `output CreditICL/reference/benchmark/<task>/`, one row per (dataset, model, seed,
fold) written by `scripts/evaluate.py` — and turns it into the final-scores
figures for `1.3_pd_results` / `1.4_lgd_results` (Exp1, `exp="exp1"`) and
`2.3_pd_results` / `2.4_lgd_results` (Exp2, `exp="exp2"`).

Each phase-2 array task writes its own `results_<tag>.csv` (an arm's FINAL checkpoint is tagged
`exp{N}bench_<track>_a<i>`, its earlier ones `..._a<i>_s<step>`, each reference model
`reference_<track>_<model>`), so an experiment's table is the concatenation of its arm files plus
the reference. `exp` selects which arm files to read; the reference is shared across experiments
and always included. Rows of ONE evaluation protocol are read (`src/eval/protocol.py`) — the newest
present, unless one is asked for: a table mixing a 1,024-row-context score (protocol 2) with a
full-context 5-fold one (protocol 3) would compare two different measurements. For the same reason
the capped-context evaluation (`eval.context_caps`, files tagged `_ctx<cap>`, rows carrying
`context_cap`) is read only when asked for (`context=`), never mixed into the full-context table.

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
from src.eval.metrics import METRIC_DIRECTIONS as HIGHER_IS_BETTER, higher_is_better

HEADLINE = {"pd": "roc_auc", "lgd": "r2"}

# Names that are external baselines rather than one of our trained arms.
BASELINES = ("catboost", "xgboost", "lightgbm", "tabpfn", "tabpfn3", "tabiclv2", "tabicl", "logreg",
             "linear", "mean", "gbm", "rf", "randomforest")

# Sweep levers that identify an Exp2 arm, as they appear in a run name / benchmark tag.
_LEVER_TOKENS = ("credit_fraction=", "arm=", "strategy=", "l2sp_alpha=", "-lr=", "exp1_", "exp2_")


#: A capped-context result file: `results_<tag>_ctx<cap>.csv`.
_CTX_FILE = re.compile(r"_ctx\d+\.csv$")


from src.eval.results import (  # shared by analysis, notebooks and development selection
    _read_frames, _raw_config, context_caps, headline_step, load_results, load_checkpoint_results,
)


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
    # One of OUR arms whenever the name carries our run levers. The baseline test is a substring
    # match, and until 25-09-2026 it ran first: every arm trained with the TabICL filter has
    # `filter-mode=tabicl` in its name, so a third of the arms were filed as external baselines —
    # left out of the credit-vs-control figures and summary, and coloured as references.
    ours = low.startswith(("exp1", "exp2", "exp3")) or "credit_fraction=" in low or "crediticl" in low
    if not ours and any(b in low for b in BASELINES):
        return "baseline"
    # cf=0 EXACTLY: `credit_fraction=0__` and `cf0·`, not the `credit_fraction=0p5` of a 0.5 arm,
    # which a bare `"credit_fraction=0" in low` substring test wrongly read as a control.
    if (re.search(r"credit_fraction=0(?:\.0|p0)?(?=__|$)", low)
            or re.search(r"(?:^|[^0-9])cf0(?:\.0)?(?=[^0-9.]|$)", low)
            or low.endswith("control")):
        return "control"
    return "credit"


# -- the groups the results figures compare -----------------------------------
#
# Our arms by the share of our credit prior in their training mix (0 % is the control), then the
# reference models. The figures used to pool 50 % and 100 % into one "credit prior" group, whose
# mean fell between two clusters that sit far apart — on LGD 0.16 R², between arms at 0.45 and at
# −0.1 — and described neither. Our arms run from control grey to credit blue with the share; the
# references are olive — they were drawn in the real-data orange, which made "CatBoost" read as a
# measurement on real data.

_REFERENCES = "reference models (not trained by us)"


def _share(model: str) -> float | None:
    """Our arm's credit share (0 for the control), or `None` for a reference model."""
    if _kind(model) == "baseline":
        return None
    m = re.search(r"credit_fraction=([0-9p.]+)", str(model))
    return float(m.group(1).replace("p", ".")) if m else 0.0


def _group_key(model: str):
    share = _share(model)
    if share is None:
        return str(model)
    family = re.search(r"(?:^|__)arm=([^=]+?)(?:__|$)", str(model))
    return (share, family.group(1)) if family else share


def _groups(models: Any) -> list:
    """Keep architectures, mixtures and individual references separate."""
    keys = {_group_key(m) for m in models}
    return sorted(keys, key=lambda k: (1, 0, k) if isinstance(k, str) else
                  (0, k[0], k[1]) if isinstance(k, tuple) else (0, k, ""))


def _group_label(group) -> str:
    if isinstance(group, str):
        return style.reference_label(group)
    if isinstance(group, tuple):
        return style.prior_mix_label(group[0], control=True) + " · " + style.CONTINUED_LABEL.get(group[1], group[1])
    return _REFERENCES if group is None else style.prior_mix_label(group, control=True)


def _group_tick(group) -> str:
    return _group_label(group).replace(" (", "\n(")


def _group_colour(group) -> str:
    if group is None or isinstance(group, str):
        return style.BASELINE
    return style.credit_fraction_colour(group[0] if isinstance(group, tuple) else group)


def _group_marker(group) -> str:
    if group is None or isinstance(group, str):
        return "D"
    return style.credit_fraction_marker(group[0] if isinstance(group, tuple) else group)


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
    known = ["roc_auc", "pr_auc", "auc", "ap", "f1_tuned", "mcc_tuned", "r2", "rmse", "mae", "pinball", "crps",
             "brier", "log_loss", "logloss", "ece", "ks", "calibration_slope", "boundary_mass_abs_err",
             "coverage_80"]
    return [m for m in known if m in df.columns]


# ---------------------------------------------------------------------------
# The split: development chooses the prior, holdout reports it
# ---------------------------------------------------------------------------

#: How a split reads in a heading — the protocol in EXPERIMENTAL_DESIGN.md §5: development is where
#: the prior is chosen; holdout scores are reported separately. Historical exposure is disclosed in docs/PRIORS.md.
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
        return _placeholder(f"no benchmark results in output CreditICL/experiment_<N>/benchmark/{track}/ yet")

    from src.eval.selection import development_ranking
    agg = development_ranking(df, track, exp, metric)
    mc = "configuration"
    if agg.empty:
        return _placeholder("no complete development-set configuration scores yet")
    agg = agg.sort_values("mean", ascending=higher_is_better(metric))
    fig, ax = plt.subplots(figsize=style.row_figsize(len(agg), base=1.3))
    y = np.arange(len(agg))
    # Points, not bars. A bar is read by its length from zero, and on PD every configuration sits
    # within a few hundredths of the others: bars from zero drew them all the same length.
    ax.errorbar(agg["mean"], y, xerr=agg["std"].fillna(0), fmt="none", ecolor=style.MUTED,
                elinewidth=0.8, zorder=2)
    keys = [_group_key(m) for m in agg[mc]]
    for g in _groups(agg[mc]):
        rows = [i for i, key in enumerate(keys) if key == g]
        ax.scatter(agg["mean"].values[rows], y[rows], s=30, color=_group_colour(g),
                   marker=_group_marker(g), edgecolor="white", linewidth=0.4, zorder=3,
                   label=_group_label(g))
    ax.set_yticks(y)
    ax.set_yticklabels([style.arm_name(m, track, seed=False) for m in agg[mc]], fontsize=7)
    ax.set_xlabel(f"development {style.metric_label(metric)}: mean over datasets and seeds "
                  f"(whisker: seed SD)")
    style.legend_below(ax, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# 2. Per-dataset breakdown
# ---------------------------------------------------------------------------


def per_dataset(track: str, exp: str = "exp1", metric: str | None = None):
    """Per-dataset descriptive group means; no selection on test-set scores."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        return _placeholder("no per-dataset benchmark results yet")

    piv, groups = _means_per_group(df, metric)
    roles = _roles(track, exp)
    rank = {"development": 0, "holdout": 1}
    datasets = sorted(piv.index, key=lambda d: (rank.get(roles.get(str(d).split(".", 1)[-1]), 2), str(d)))
    piv = piv.loc[datasets]
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    x = np.arange(len(datasets))
    w = 0.8 / max(len(groups), 1)
    for i, g in enumerate(groups):
        ax.bar(x + i * w, piv[_group_label(g)].values, width=w, color=_group_colour(g),
               label=_group_label(g))
    ax.set_xticks(x + w * (len(groups) - 1) / 2)
    ax.set_xticklabels([_role_tag(d, roles) for d in datasets], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"mean {style.metric_label(metric)} in the group")
    # Under the whole figure: the rotated dataset names are taller than the one line of ticks that
    # a legend under the axes makes room for, and it sat on them.
    style.legend_below(fig, ncol=2)
    return fig


def _means_per_group(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[float | None]]:
    """Dataset × configuration descriptive means, averaging training seeds.

    Average folds within each model, then models/training seeds equally. Taking
    the best training seed on each test dataset is optimistic model selection.
    Formal comparisons keep configurations and reference models separate.
    """
    df, mc = _with_identity(df)
    per_model = df.groupby(["dataset", mc])[metric].mean().reset_index()
    groups = _groups(per_model[mc].unique())
    per_model["group"] = per_model[mc].map(lambda m: _group_label(_group_key(m)))
    piv = per_model.groupby(["dataset", "group"])[metric].mean().unstack("group")
    return piv[[_group_label(g) for g in groups]], groups


# ---------------------------------------------------------------------------
# 3. Credit prior vs control — the headline comparison
# ---------------------------------------------------------------------------


def credit_vs_control(track: str, exp: str = "exp1", metric: str | None = None,
                      role: str | None = None, context: int | None = None):
    """The headline score of every model, one column per share of our credit prior (0 % is the
    control) and one for the reference models, on the datasets of one side of the split
    (`role="holdout"` is what the experiment reports). `context` shows the capped-context
    evaluation instead (`eval.context_caps`)."""
    df = load_results(track, exp, context=context)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is not None:
        df = _restrict(df, track, exp, role)
    if df is None or metric not in df.columns or not len(df):
        return _placeholder("no benchmark results yet" if context is None
                            else f"no results with {context:,}-row contexts yet")

    df, mc = _with_identity(df)
    per_model = df.groupby(mc)[metric].mean()
    groups = _groups(per_model.index)
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.42))
    for i, g in enumerate(groups):
        vals = per_model[[(_group_key(m)) == g for m in per_model.index]].values
        ax.scatter(np.full(len(vals), i) + style_jitter(len(vals)), vals, s=34, alpha=0.7,
                   color=_group_colour(g), marker=_group_marker(g), edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.nanmean(vals)] * 2, color=style.INK, lw=2, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_group_tick(g) for g in groups])
    ax.set_ylabel(f"{style.metric_label(metric)}, mean over {ROLE_LABEL.get(role, role)}")
    style.legend_below(ax, [plt.Line2D([], [], color=style.MUTED, marker="o", ls="none",
                                       label="one model"),
                            plt.Line2D([], [], color=style.INK, lw=2, label="group mean")], ncol=2)
    return fig


def _checkpoint_means(track: str, exp: str, metric: str, role: str | None) -> pd.DataFrame | None:
    """`(share, checkpoint_step, mean, low, high)`: each arm's score per saved checkpoint — folds
    within a dataset first, then datasets equally — then the mean and range over training seeds."""
    df = load_checkpoint_results(track, exp)
    if df is not None:
        df = _restrict(df, track, exp, role)
    if df is None or metric not in df.columns or not len(df) or df["checkpoint_step"].nunique() < 2:
        return None
    df, mc = _with_identity(df)
    # Use the same identities at every plotted step. A missing final checkpoint
    # must not make the mean jump by dropping its least successful seed.
    step_sets = [set(part.checkpoint_step) for _, part in df.groupby(mc)]
    common_steps = set.intersection(*step_sets)
    df = df[df.checkpoint_step.isin(common_steps)]
    if len(common_steps) < 2:
        return None
    per_arm = (df.groupby([mc, "checkpoint_step", "dataset"])[metric].mean()
               .groupby(level=[0, 1]).mean().reset_index())
    per_arm["share"] = per_arm[mc].map(_share)
    per_arm["configuration"] = per_arm[mc].str.replace(r"__s\d+$", "", regex=True)
    per_arm = per_arm.dropna(subset=["share"])
    out = per_arm.groupby(["configuration", "share", "checkpoint_step"])[metric].agg(["mean", "min", "max"]).reset_index()
    return out.rename(columns={"min": "low", "max": "high"})


def checkpoint_curve(track: str, exp: str = "exp1", metric: str | None = None,
                     role: str | None = "holdout"):
    """The learning curve on real data: every saved checkpoint's score, one line per share of our
    credit prior (the mean over its training seeds, the band their range), and the reference
    models' scores as dashed horizontal lines."""
    metric = metric or HEADLINE[track]
    style.apply()
    curve = _checkpoint_means(track, exp, metric, role)
    if curve is None:
        return _placeholder("no learning curve yet: the protocol-3 benchmark scores every saved checkpoint")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.42))
    handles = []
    for configuration, part in curve.groupby("configuration"):
        share = float(part["share"].iloc[0])
        colour = _group_colour(share)
        ax.fill_between(part["checkpoint_step"], part["low"], part["high"], color=colour, alpha=0.18, lw=0)
        line, = ax.plot(part["checkpoint_step"], part["mean"], color=colour, marker=_group_marker(share),
                        lw=1.6, ms=4, label=style.arm_name(configuration, track, seed=False),
                        ls="--" if "tabpfn3" in configuration else "-")
        handles.append(line)
    ref = load_results(track, exp)
    if ref is not None:
        ref = _restrict(ref, track, exp, role)
    if ref is not None and len(ref) and metric in ref.columns:
        ref, ref_id = _with_identity(ref)
        ref = ref[ref[ref_id].map(_kind).eq("baseline")]
        per_ref = ref.groupby(["model", "dataset"])[metric].mean().groupby(level=0).mean()
        right = float(curve["checkpoint_step"].max())
        for model, value in per_ref.items():
            ax.axhline(value, color=style.BASELINE, lw=1.0, ls="--", zorder=1)
        if len(per_ref):
            style.place_labels(ax, [right] * len(per_ref), per_ref.values,
                               [style.reference_label(m, track) for m in per_ref.index], color=style.BASELINE)
            handles.append(plt.Line2D([], [], color=style.BASELINE, lw=1.0, ls="--", label=_REFERENCES))
    ax.set_xlabel("training step of the saved checkpoint")
    ax.set_ylabel(f"{style.metric_label(metric)}, mean over {ROLE_LABEL.get(role, role)}")
    style.legend_below(ax, handles, ncol=min(len(handles), 4))
    return fig


# ---------------------------------------------------------------------------
# 4. Which fine-tuning lever moved the score (Exp2)
# ---------------------------------------------------------------------------

#: The Exp2 sweep levers, and how each reads in a benchmark tag / run name.
#: Experiment 2's levers, read off the run name: the released model continued and the share of our
#: prior. Optimizer and rate read too, for run names of the retired search (until 30-09-2026).
_LEVERS = (("credit_fraction", r"credit_fraction=([0-9p.]+)", "prior mix"),
           ("model", r"arm=(tabicl|tabpfn3)", "released model continued"),
           ("optimizer", r"arm=(?:tabicl|tabpfn3)_(adamw|muon)", "optimizer"),
           ("lr", r"arm=(?:tabicl|tabpfn3)_(?:adamw|muon)_([0-9.e-]+)", "learning rate"))


def _lever_tick(key: str, value: str) -> str:
    """A lever value in the notebooks' words: `50 % credit`, `ICL stack + head`, `no L2-SP`."""
    text = str(value).replace("p", ".").replace("m", "-")
    if key == "credit_fraction":
        return style.prior_mix_label(float(text))
    if key == "model":
        return style.CONTINUED_LABEL.get(str(value), str(value))
    if key == "optimizer":
        return {"adamw": "AdamW", "muon": "Muon"}.get(str(value), str(value))
    return str(value)


def lever_effect(track: str, exp: str = "exp2", metric: str | None = None,
                 role: str | None = "development"):
    """One panel per fine-tuning lever: mean headline score grouped by that lever's value.

    Reads the arm's swept levers out of its run name and averages the headline metric over every
    arm that shares a value, so each panel isolates one knob — the credit fraction, the released
    model continued (and the optimizer and rate of the retired search). A lever with one value draws
    no panel. Degrades to a placeholder before the benchmark runs.
    """
    df = load_results(track, exp)
    if df is not None:
        df = _restrict(df, track, exp, role)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or df.empty:
        return _placeholder("no benchmark results yet")
    fig, axes = plt.subplots(2, 2, figsize=style.grid_figsize(2, 2, panel_ratio=0.62), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    mc = _model_col(df)
    names = df[mc].astype(str)
    drawn = False
    for ax, (key, pattern, label) in zip(axes, _LEVERS):
        vals = names.str.extract(pattern, expand=False)
        d = df.assign(_lever=vals).dropna(subset=["_lever"])
        if d["_lever"].nunique() < 2:
            ax.axis("off")
            continue
        grp = d.groupby("_lever")[metric].mean().sort_index()
        x = np.arange(len(grp))
        ax.bar(x, grp.values, color=style.CREDIT, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([_lever_tick(key, v) for v in grp.index], fontsize=7)
        ax.set_ylabel(f"{ROLE_LABEL.get(role, role)} mean {style.metric_label(metric)}", fontsize=8)
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
    """One panel per benchmark metric, each a bar per group (share of our credit prior, then the
    reference models) — the whole scoreboard at once, on the datasets of one side of the split."""
    df = load_results(track, exp)
    style.apply()
    if df is not None:
        df = _restrict(df, track, exp, role)
        df = df if len(df) else None
    metrics = available_metrics(df) if df is not None else []
    if df is None or not metrics:
        return _placeholder("no benchmark results in output CreditICL/ yet")
    ncols = 3
    nrows = max(1, int(np.ceil(len(metrics) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    df, mc = _with_identity(df)
    groups = _groups(df[mc].unique())
    for ax, metric in zip(axes, metrics):
        ax.axis("on")
        # Each model's mean first, then the group's: every model weighs the same.
        per_model = df.groupby(mc)[metric].mean()
        vals = [per_model[[(_group_key(m)) == g for m in per_model.index]].mean() for g in groups]
        ax.bar(range(len(groups)), vals, color=[_group_colour(g) for g in groups], width=0.66)
        goal = style.metric_goal(metric)
        if isinstance(goal, (int, float)):
            ax.axhline(goal, color=style.INK, lw=0.9, ls="--")
        ax.set_xticks([])
        ax.set_title(f"{style.metric_label(metric)} {style.goal_mark(metric)}".strip(), loc="left",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    style.legend_below(fig, style.legend_patches({_group_label(g): _group_colour(g) for g in groups}),
                       ncol=2)
    return fig


def per_dataset_heatmap(track: str, exp: str = "exp1", metric: str | None = None):
    """Mean headline score in each group on each dataset, as an annotated heatmap."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        return _placeholder("no per-dataset benchmark results yet")
    piv, groups = _means_per_group(df, metric)
    roles = _roles(track, exp)
    rank = {"development": 0, "holdout": 1}
    order = sorted(piv.index, key=lambda v: (rank.get(roles.get(str(v).split(".", 1)[-1]), 2), str(v)))
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=style.row_figsize(len(piv), base=1.2))
    im = ax.imshow(piv.values, aspect="auto", cmap=style.CMAP_SEQ)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_group_tick(g) for g in groups], fontsize=7)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([_role_tag(v, roles) for v in piv.index], fontsize=7)
    ax.grid(False)
    finite = piv.values[np.isfinite(piv.values)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    for i in range(len(piv)):
        for j in range(len(groups)):
            v = piv.values[i, j]
            if not np.isnan(v):
                # cividis runs dark to light: white text on the dark half, ink on the light half.
                dark = (v - lo) / (hi - lo + 1e-12) < 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if dark else style.INK)
    fig.colorbar(im, ax=ax, shrink=0.55, label=f"mean {style.metric_label(metric)} in the group")
    return fig


def beats_reference(track: str, exp: str = "exp1", metric: str | None = None,
                    role: str | None = None):
    """Every trained arm's headline score minus its matching released model's, on the datasets of one side
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
    ours = per[[_kind(m) in ("credit", "control") for m in per.index]]
    if ours.empty:
        return _placeholder("no trained arms scored yet")
    references = {name: ("tabpfn3" if "tabpfn3" in str(name) else "tabiclv2") for name in ours.index}
    if any(reference not in per for reference in references.values()):
        return _placeholder("matching released-model references have not been scored yet")
    higher = higher_is_better(metric)
    order = ours.sort_values(ascending=not higher)
    delta = np.array([score - per[references[name]] for name, score in order.items()])
    fig, ax = plt.subplots(figsize=style.row_figsize(len(order), per_row=0.12, base=1.3))
    ax.barh(np.arange(len(order)), delta,
            color=[_group_colour(_share(m)) for m in order.index])
    ax.axvline(0, color=style.REFERENCE, lw=1.2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([style.arm_name(m, track) for m in order.index], fontsize=6)
    ax.set_xlabel(f"{style.metric_label(metric)} minus its matching released model, mean over the "
                  f"{ROLE_LABEL.get(role, role)}")
    groups = _groups(order.index)
    style.legend_below(ax, style.legend_patches({_group_label(g): _group_colour(g) for g in groups})
                       + [plt.Line2D([], [], color=style.REFERENCE, lw=1.2,
                                     label="matching released model (0)")], ncol=2)
    return fig


def results_summary(track: str, exp: str = "exp1") -> str:
    """Notebook sections A–C in order; later literature/OOD/component sections print separately."""
    from src.eval.selection import development_ranking
    from src.eval.comparisons import paired_prior_effects, paired_reference_effects

    df = load_results(track, exp)
    title = f"{exp.upper()} {track.upper()} RESULTS — full-table five-fold benchmark"
    headings = ["A. DEVELOPMENT COMPARISON" if exp == "exp2" else "A. WHICH PRIOR DEVELOPMENT SELECTS",
                "B. HOW IT DOES ON THE HOLDOUT",
                "B1. CREDIT PRIOR AGAINST THE CONTROL", "B2. AGAINST MATCHING RELEASED WEIGHTS",
                "B3. EVERY BENCHMARK METRIC", "B4. EVERY SAVED CHECKPOINT", "C. EVERY DATASET"]
    if df is None:
        return title + "\n" + "\n".join(h + "\n  No complete benchmark output yet." for h in headings)
    if "protocol" in df:
        title += f" (protocol {int(df.protocol.iloc[0])})"
    metric = HEADLINE[track]
    df, mc = _with_identity(df)
    lines = [title, headings[0]]
    ranking = development_ranking(df, track, exp)
    lines.append(ranking.to_string(index=False) if not ranking.empty else "No complete development ranking.")
    holdout = _restrict(df, track, exp, "holdout") if _roles(track, exp) else df
    lines += [headings[1], "Datasets receive equal weight; CV folds are not independent replications."]
    for heading, compare in ((headings[2], paired_prior_effects), (headings[3], paired_reference_effects)):
        lines.append(heading)
        if "protocol" in df and df.protocol.min() >= 6 and not holdout.empty:
            try:
                table = compare(holdout, metric)
                lines.append(table.to_string(index=False) if not table.empty else "No matched contrasts.")
            except ValueError as exc:
                lines.append(f"Comparison withheld: {exc}")
        else:
            lines.append("Legacy/partial output: no current-protocol uncertainty claim.")
    lines += ["Intervals are descriptive paired-dataset bootstraps; small benchmark sets limit precision.",
              headings[4]]
    for measure in available_metrics(holdout):
        lines.append(measure + "\n" + _group_means(holdout, mc, measure, track).to_string())
    lines.append(headings[5])
    curve = _checkpoint_means(track, exp, metric, "holdout")
    if curve is None:
        lines.append("No complete learning curve yet.")
    else:
        for configuration, part in curve.groupby("configuration"):
            steps = ", ".join(f"{int(r.checkpoint_step):,}: {r.mean:.4f}" for r in part.itertuples())
            lines.append(f"{style.arm_name(configuration, track, seed=False)}: {steps}")
    lines.append(headings[6])
    if metric in df:
        table, _ = _means_per_group(df, metric)
        lines.append(table.to_string())
    return "\n".join(lines)


def _group_means(df: pd.DataFrame, mc: str, metric: str, track: str) -> pd.Series:
    """`metric`'s mean per group — each share of our prior, then each reference model — in that
    order: our shares increasing, then the references by name."""
    per_dataset = df.groupby([mc, "dataset"])[metric].mean()
    per_model = per_dataset.groupby(level=0).mean()
    means = per_model.groupby(per_model.index.map(lambda m: _share_group(m, track))).mean()

    def order(g: str) -> tuple:
        share = re.match(r"([0-9.]+) % credit", g)
        return (0, float(share.group(1)), g) if share else (1, 0.0, g)

    return means.reindex(sorted(means.index, key=order))


def _share_group(model: str, track: str) -> str:
    """`0 % credit (TabICL prior only)`, `50 % credit`, … for our arms; `reference: <name>`."""
    if _kind(model) == "baseline":
        return f"reference: {style.reference_label(model, track)}"
    m = re.search(r"credit_fraction=([0-9p.]+)", str(model))
    share = float(m.group(1).replace("p", ".")) if m else 0.0
    family = re.search(r"(?:architecture|arm)=([^_]+)", str(model))
    suffix = f" · {family.group(1)}" if family else ""
    return style.prior_mix_label(share, control=True) + suffix


def _pair_key(run_name: str) -> str:
    """An arm's name with its credit share and prior intensity removed: two arms with the same key
    differ only in how much of our prior they saw — same filter (or fine-tuning levers) and seed."""
    key = re.sub(r"credit_fraction=[0-9p.]+", "", run_name)
    return re.sub(r"(?:rho_range|boundary_mass_range)=\[[0-9.,]+\]", "", key)


def paired_differences(df: pd.DataFrame, track: str, exp: str,
                       role: str) -> dict[float, tuple[float, int, int]]:
    """For each credit share above 0: `(mean of arm − its matched control, pairs where the arm is
    better, number of pairs)` on one side of the split. An arm's score is its mean over the side's
    datasets (equal weight) and evaluation seeds; arms of one share that differ only in prior
    intensity are averaged first, so each pair is one (filter, seed) cell."""
    metric = HEADLINE[track]
    part = _restrict(df, track, exp, role) if _roles(track, exp) else df
    if part is None or not len(part) or "info_run_name" not in part.columns:
        return {}
    ours = part[part["info_run_name"].fillna("").ne("") & part[metric].notna()]
    if ours.empty:
        return {}
    if "protocol" in ours and ours["protocol"].min() >= 6:
        from src.eval.comparisons import require_balanced_cells
        require_balanced_cells(ours)
    arm = ours.groupby(["info_run_name", "dataset"])[metric].mean().groupby(level=0).mean()
    frame = pd.DataFrame({"score": arm})
    frame["key"] = [_pair_key(n) for n in frame.index]
    frame["share"] = [float(re.search(r"credit_fraction=([0-9p.]+)", n).group(1).replace("p", "."))
                      for n in frame.index]
    cell = frame.groupby(["share", "key"])["score"].mean()
    if 0.0 not in cell.index.get_level_values(0):
        return {}
    control = cell.xs(0.0, level="share")
    out: dict[float, tuple[float, int, int]] = {}
    for share in sorted(s for s in cell.index.get_level_values(0).unique() if s > 0):
        diff = (cell.xs(share, level="share") - control).dropna()
        if len(diff):
            better = int((diff > 0).sum()) if higher_is_better(metric) else int((diff < 0).sum())
            out[share] = (float(diff.mean()), better, int(len(diff)))
    return out


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
