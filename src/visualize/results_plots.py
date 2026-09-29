"""Level-1 RESULTS visualisation: how the trained arms score on the real datasets.

Reads the benchmark output — `output_CreditICL/experiment_<N>/benchmark/<task>/results_<tag>.csv`
and the shared `output_CreditICL/reference/benchmark/<task>/`, one row per (dataset, model, seed,
fold) written by `scripts/evaluate.py` — and turns it into the final-scores
figures for `1.3_pd_results` / `1.4_lgd_results` (Exp1, `exp="exp1"`) and
`2.3_pd_results` / `2.4_lgd_results` (Exp2, `exp="exp2"`).

Each phase-2 array task writes its own `results_<tag>.csv` (an arm's FINAL checkpoint is tagged
`exp{N}bench_<track>_a<i>`, its earlier ones `..._a<i>_s<step>`, each reference model
`reference_<track>_<model>`), so an experiment's table is the concatenation of its arm files plus
the reference. `exp` selects which arm files to read; the reference is shared across experiments
and always included. Rows of ONE evaluation protocol are read (`src/eval/protocol.py`) — the newest
present, unless one is asked for: a table mixing a 1,024-row-context score (protocol 2) with a
full-context 5-fold one (protocol 3) would compare two different measurements.

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
                    "roc_auc": True, "pr_auc": True, "f1_tuned": True, "mcc_tuned": True,
                    "pinball": False, "crps": False, "brier": False, "logloss": False,
                    "log_loss": False, "ece": False}

# Names that are external baselines rather than one of our trained arms.
BASELINES = ("catboost", "xgboost", "lightgbm", "tabpfn", "tabpfn3", "tabiclv2", "tabicl", "logreg",
             "linear", "mean", "gbm", "rf", "randomforest")

# Sweep levers that identify an Exp2 arm, as they appear in a run name / benchmark tag.
_LEVER_TOKENS = ("credit_fraction=", "arm=", "strategy=", "l2sp_alpha=", "-lr=", "exp1_", "exp2_")


def _read_frames(files: list[Any], protocol: int | None) -> pd.DataFrame | None:
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
    # Rows written before the protocol column existed are protocol 2 (one split per seed,
    # context capped at 1,024 rows).
    version = pd.to_numeric(df["protocol"], errors="coerce").fillna(2) if "protocol" in df \
        else pd.Series(2, index=df.index)
    want = int(version.max()) if protocol is None and len(df) else protocol
    df = df[version.eq(want)].assign(protocol=want)
    return df if len(df) else None


def load_results(track: str, exp: str = "exp1", protocol: int | None = None) -> pd.DataFrame | None:
    """Per-(dataset, model, seed, fold) results of the FINAL checkpoints for `exp` on `track`, or
    `None` if none exist.

    Concatenates the experiment's arm files (`results_<exp>bench_<track>_a<i>.csv`) with the shared
    reference (`results_reference_<track>_<model>.csv`). Other experiments and earlier checkpoints are
    excluded, and so is every protocol but one: the newest present when `protocol` is None (the
    `protocol` column says which), else the one asked for.
    """
    out = paths.benchmark_dir(paths.experiment_of(f"{exp}_"), track)
    ref = paths.benchmark_dir(paths.REFERENCE, track)
    if not out.exists() and not ref.exists():
        return None
    # Every FINAL-checkpoint arm file (no `_s<step>` suffix), whatever grid wrote it: the protocol
    # filter below keeps one benchmark.
    final = re.compile(rf"results_{re.escape(exp)}bench_{re.escape(track)}_a\d+\.csv$")
    files = sorted(p for p in out.glob(f"results_{exp}bench_{track}_a*.csv") if final.match(p.name))
    files += sorted(ref.glob(f"results_reference_{track}_*.csv"))
    return _read_frames(files, protocol)


def load_checkpoint_results(track: str, exp: str = "exp1", protocol: int | None = None) -> pd.DataFrame | None:
    """Every saved checkpoint of every arm — the final one and the `_s<step>` ones — with the
    step in `checkpoint_step`: the learning curve on real data, one point per saved checkpoint.
    One protocol only, as `load_results`."""
    out = paths.benchmark_dir(paths.experiment_of(f"{exp}_"), track)
    if not out.exists():
        return None
    files = sorted(out.glob(f"results_{exp}bench_{track}_a*.csv"))
    df = _read_frames(files, protocol)
    if df is None or "info_checkpoint_step" not in df:
        return None
    df = df.assign(checkpoint_step=pd.to_numeric(df["info_checkpoint_step"], errors="coerce"))
    return df.dropna(subset=["checkpoint_step"])


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


def _groups(models: Any) -> list[float | None]:
    """The groups present, in reading order: our shares ascending, then the references (`None`)."""
    shares = {_share(m) for m in models}
    return sorted(s for s in shares if s is not None) + ([None] if None in shares else [])


def _group_label(share: float | None) -> str:
    return _REFERENCES if share is None else style.prior_mix_label(share, control=True)


def _group_tick(share: float | None) -> str:
    """The label on two lines, the aside in brackets on the second, so that ticks side by side
    do not run into each other."""
    return _group_label(share).replace(" (", "\n(")


def _group_colour(share: float | None) -> str:
    return style.BASELINE if share is None else style.credit_fraction_colour(share)


def _group_marker(share: float | None) -> str:
    return "D" if share is None else style.credit_fraction_marker(share)


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
        return _placeholder(f"no benchmark results in output_CreditICL/experiment_<N>/benchmark/{track}/ yet")

    from src.eval.selection import development_ranking
    agg = development_ranking(df, track, exp, metric)
    mc = "configuration"
    if agg.empty:
        return _placeholder("no complete development-set configuration scores yet")
    agg = agg.sort_values("mean", ascending=HIGHER_IS_BETTER.get(metric, True))
    fig, ax = plt.subplots(figsize=style.row_figsize(len(agg), base=1.3))
    y = np.arange(len(agg))
    # Points, not bars. A bar is read by its length from zero, and on PD every configuration sits
    # within a few hundredths of the others: bars from zero drew them all the same length.
    ax.errorbar(agg["mean"], y, xerr=agg["std"].fillna(0), fmt="none", ecolor=style.MUTED,
                elinewidth=0.8, zorder=2)
    shares = [_share(m) for m in agg[mc]]
    for g in _groups(agg[mc]):
        rows = [i for i, s in enumerate(shares) if s == g]
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
    """Headline score per dataset, the best model of each group, so no dataset is hidden by a mean."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        return _placeholder("no per-dataset benchmark results yet")

    piv, groups = _best_per_group(df, metric)
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
    ax.set_ylabel(f"best {style.metric_label(metric)} in the group")
    # Under the whole figure: the rotated dataset names are taller than the one line of ticks that
    # a legend under the axes makes room for, and it sat on them.
    style.legend_below(fig, ncol=2)
    return fig


def _best_per_group(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[float | None]]:
    """`(dataset × group table of the best model's score, the groups in order)`.

    A model's score on a dataset is its mean over evaluation seeds, and the best of those is taken.
    The best single (model, seed) row favoured whichever group had the most rows to find a lucky
    seed among.
    """
    df, mc = _with_identity(df)
    per_model = df.groupby(["dataset", mc])[metric].mean().reset_index()
    groups = _groups(per_model[mc].unique())
    per_model["group"] = per_model[mc].map(lambda m: _group_label(_share(m)))
    agg = "max" if HIGHER_IS_BETTER.get(metric, True) else "min"
    piv = per_model.groupby(["dataset", "group"])[metric].agg(agg).unstack("group")
    return piv[[_group_label(g) for g in groups]], groups


# ---------------------------------------------------------------------------
# 3. Credit prior vs control — the headline comparison
# ---------------------------------------------------------------------------


def credit_vs_control(track: str, exp: str = "exp1", metric: str | None = None,
                      role: str | None = None):
    """The headline score of every model, one column per share of our credit prior (0 % is the
    control) and one for the reference models, on the datasets of one side of the split
    (`role="holdout"` is what the experiment reports)."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is not None:
        df = _restrict(df, track, exp, role)
    if df is None or metric not in df.columns or not len(df):
        return _placeholder("no benchmark results yet")

    df, mc = _with_identity(df)
    per_model = df.groupby(mc)[metric].mean()
    groups = _groups(per_model.index)
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.42))
    for i, g in enumerate(groups):
        vals = per_model[[_share(m) == g for m in per_model.index]].values
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
    per_arm = (df.groupby([mc, "checkpoint_step", "dataset"])[metric].mean()
               .groupby(level=[0, 1]).mean().reset_index())
    per_arm["share"] = per_arm[mc].map(_share)
    per_arm = per_arm.dropna(subset=["share"])
    out = per_arm.groupby(["share", "checkpoint_step"])[metric].agg(["mean", "min", "max"]).reset_index()
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
    for share, part in curve.groupby("share"):
        colour = _group_colour(share)
        ax.fill_between(part["checkpoint_step"], part["low"], part["high"], color=colour, alpha=0.18, lw=0)
        line, = ax.plot(part["checkpoint_step"], part["mean"], color=colour, marker=_group_marker(share),
                        lw=1.6, ms=4, label=_group_label(share))
        handles.append(line)
    ref = load_results(track, exp)
    if ref is not None:
        ref = _restrict(ref, track, exp, role)
    if ref is not None and len(ref) and metric in ref.columns:
        ref = ref[ref["model"].map(_kind).eq("baseline")]
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
#: Experiment 2's levers, read off the run name: the released model continued, the share of our
#: prior (stage B), and the optimizer and rate (the stage-A search).
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


def lever_effect(track: str, exp: str = "exp2", metric: str | None = None):
    """One panel per fine-tuning lever: mean headline score grouped by that lever's value.

    Reads the arm's swept levers out of its run name and averages the headline metric over every
    arm that shares a value, so each panel isolates one knob — the credit fraction, the released
    model continued, and (in the search) the optimizer and the rate. A lever with one value draws
    no panel. Degrades to a placeholder before the benchmark runs.
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
    """One panel per benchmark metric, each a bar per group (share of our credit prior, then the
    reference models) — the whole scoreboard at once, on the datasets of one side of the split."""
    df = load_results(track, exp)
    style.apply()
    if df is not None:
        df = _restrict(df, track, exp, role)
        df = df if len(df) else None
    metrics = available_metrics(df) if df is not None else []
    if df is None or not metrics:
        return _placeholder("no benchmark results in output_CreditICL/ yet")
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
        vals = [per_model[[_share(m) == g for m in per_model.index]].mean() for g in groups]
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
    """Best headline score in each group on each dataset, as an annotated heatmap."""
    df = load_results(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    if df is None or metric not in df.columns or "dataset" not in df.columns:
        return _placeholder("no per-dataset benchmark results yet")
    piv, groups = _best_per_group(df, metric)
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
    fig.colorbar(im, ax=ax, shrink=0.55, label=f"best {style.metric_label(metric)} in the group")
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
            color=[_group_colour(_share(m)) for m in order.index])
    ax.axvline(0, color=style.REFERENCE, lw=1.2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([style.arm_name(m, track) for m in order.index], fontsize=6)
    ax.set_xlabel(f"{style.metric_label(metric)} minus the released TabICLv2's, mean over the "
                  f"{ROLE_LABEL.get(role, role)}")
    groups = _groups(order.index)
    style.legend_below(ax, style.legend_patches({_group_label(g): _group_colour(g) for g in groups})
                       + [plt.Line2D([], [], color=style.REFERENCE, lw=1.2,
                                     label="released TabICLv2 (0)")], ncol=2)
    return fig


def results_summary(track: str, exp: str = "exp1") -> str:
    """Text summary of the benchmark, in the notebook's order: what development selects, then how
    it does on the holdout."""
    df = load_results(track, exp)
    title = f"{exp.upper()} {track.upper()} RESULTS — the benchmark"
    if df is not None and "protocol" in df:
        title += (f" (evaluation protocol {int(df['protocol'].iloc[0])}: "
                  + ("5-fold CV, whole training pool as context, validation-tuned F1)"
                     if int(df["protocol"].iloc[0]) >= 3 else "one split per seed, context capped at 1,024 rows)"))
    if df is None:
        return (f"{title}\n  no benchmark output in output_CreditICL/experiment_<N>/benchmark/{track}/ yet.\n"
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
    lines += ["", f"A. WHICH PRIOR DEVELOPMENT SELECTS ({label}, development datasets, "
              f"mean over datasets and seeds)"]
    from src.eval.selection import development_ranking

    ranking = development_ranking(df, track, exp)
    if not ranking.empty:
        for _, row in ranking.head(6).iterrows():
            name = style.arm_name(row["configuration"], track, seed=False)
            lines.append(f"  {row['mean']:.4f}  {name}")
    else:
        lines.append("  no complete development configuration scores yet")
    for role, heading in (("holdout", "B. HOW IT DOES ON THE HOLDOUT"),):
        part = _restrict(df, track, exp, role) if roles else df
        lines += ["", f"{heading} ({label}, {ROLE_LABEL[role if roles else None]}, "
                  f"mean over every arm of each share)"]
        if not len(part):
            lines.append("  no holdout rows scored yet")
            continue
        groups = part[mc].map(lambda m: _share_group(m, track))
        means = part.groupby(groups)[metric].mean()

        def order(g: str) -> tuple:  # our shares in increasing order, then the references
            share = re.match(r"([0-9.]+) % credit", g)
            return (0, float(share.group(1)), g) if share else (1, 0.0, g)

        for g in sorted(means.index, key=order):
            lines.append(f"  {means[g]:.4f}  {g}")
    curve = _checkpoint_means(track, exp, metric, "holdout" if roles else None)
    lines += ["", f"B4. EVERY SAVED CHECKPOINT ({label}, holdout datasets, mean over training seeds)"]
    if curve is None:
        lines.append("  no learning curve yet: only final checkpoints have been scored")
    else:
        for share, part in curve.groupby("share"):
            steps = ", ".join(f"{int(r.checkpoint_step):,}: {r.mean:.4f}" for r in part.itertuples())
            lines.append(f"  {style.prior_mix_label(share)} — {steps}")
    lines += ["", "C. OUR CREDIT PRIOR AGAINST THE CONTROL — matched arms that differ only in "
              "the credit share", "  (same filter and training seed; a positive difference "
              "means the credit prior helped)"]
    for role in ("development", "holdout"):
        for share, (mean, better, n) in paired_differences(df, track, exp, role).items():
            lines.append(f"  {role:<11} {style.prior_mix_label(share)} minus 0 % credit: "
                         f"{mean:+.4f} {label}, better in {better} of {n} pairs")
    return "\n".join(lines)


def _share_group(model: str, track: str) -> str:
    """`0 % credit (TabICL prior only)`, `50 % credit`, … for our arms; `reference: <name>`."""
    if _kind(model) == "baseline":
        return f"reference: {style.reference_label(model, track)}"
    m = re.search(r"credit_fraction=([0-9p.]+)", str(model))
    share = float(m.group(1).replace("p", ".")) if m else 0.0
    return style.prior_mix_label(share, control=True)


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
    ours = part[part["model"].astype(str).eq("crediticl") & part[metric].notna()]
    if ours.empty:
        return {}
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
            better = int((diff > 0).sum()) if HIGHER_IS_BETTER.get(metric, True) else int((diff < 0).sum())
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
