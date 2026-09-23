"""Level-1 TRAINING visualisation: what every arm did while it trained.

Reads the two CSVs each arm writes to `output/manifests/` — `<run>__progress.csv` (train
loss and real/OOD evaluation metrics, sampled every `progress.every_datasets`) and
`<run>__telemetry.csv` (GPU utilisation, throughput, per-block gradient norms) — and turns
them into the training-behaviour figures for `1.1_pd_training` / `1.2_lgd_training` (Exp1,
`exp="exp1"`) and `2.1_pd_finetuning` / `2.2_lgd_finetuning` (Exp2, `exp="exp2"`).

The two experiments differ only in which manifests are read (the `exp1_`/`exp2_` filename
prefix) and how an arm's sweep levers are named (`arm_label` reads either Exp1's prior levers
or Exp2's fine-tuning levers). Every curve, metric and ranking is computed identically, so one
module serves both. Exp2 additionally cares about OUT-OF-DOMAIN retention — whether fine-tuning
on credit degrades the model elsewhere — which `real_vs_ood` draws from the same progress CSVs.

Every arm that has *started* leaves a progress CSV, so these work on a partial sweep: a run
that is 15/45 done still has 15 curves to show. Nothing here needs the cluster — it reads
what the run wrote down. The notebooks call these and hold no logic of their own.
"""

from __future__ import annotations

import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import paths
from src.visualize import style

# Headline evaluation metric per task: the one curve a reader looks at first. Higher is better
# for both (AUC for classification, R2 for regression), so "best/worst" is unambiguous.
HEADLINE = {"pd": "roc_auc", "lgd": "r2"}
HIGHER_IS_BETTER = {"auc": True, "ap": True, "r2": True, "spearman": True, "kendall": True,
                    "roc_auc": True, "pr_auc": True,
                    "brier": False, "logloss": False, "rmse": False, "mae": False,
                    "pinball": False, "crps": False, "ks": True, "calibration_slope": True}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _manifest_files(track: str, kind: str, exp: str = "exp1") -> list:
    return sorted(paths.manifests_dir().glob(f"{exp}_{track}__*__{kind}.csv"))


def load_progress(track: str, exp: str = "exp1") -> dict[str, pd.DataFrame]:
    """`{run_name: progress DataFrame}` for every started arm of `track` in experiment `exp`."""
    out: dict[str, pd.DataFrame] = {}
    for path in _manifest_files(track, "progress", exp):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if "step" in df and len(df):
            out[path.name.replace("__progress.csv", "")] = df.sort_values("step")
    return out


def load_telemetry(track: str, exp: str = "exp1") -> dict[str, pd.DataFrame]:
    """`{run_name: telemetry DataFrame}` for every started arm of `track` in experiment `exp`."""
    out: dict[str, pd.DataFrame] = {}
    for path in _manifest_files(track, "telemetry", exp):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if "step" in df and len(df):
            out[path.name.replace("__telemetry.csv", "")] = df
    return out


# ---------------------------------------------------------------------------
# Naming and grouping the arms
# ---------------------------------------------------------------------------


def arm_label(run_name: str) -> str:
    """A short, readable label from the sweep levers.

    Exp1 arms read `cf1·banded·aggr·s2` (prior levers); Exp2 arms read `cf0.5·icl·l2·lr1e-5`
    (fine-tuning levers), because the two experiments sweep different knobs. The experiment is
    taken from the `exp1_`/`exp2_` prefix the run name always carries.
    """
    if run_name.startswith("exp2"):
        return _arm_label_exp2(run_name)
    cf = re.search(r"credit_fraction=([0-9p.]+)", run_name)
    fm = re.search(r"filter-mode=([a-z]+)", run_name)
    seed = re.search(r"__s(\d+)", run_name)
    intensity = "aggr" if ("0.6" in run_name or "0,6" in run_name or "0.3]" in run_name.split("rho_range=")[-1][:12]) else "mild"
    # boundary/rho ranges are the two intensity axes; the aggressive arm carries the wider upper bound.
    if "boundary_mass_range=[0.15" in run_name or "rho_range=[0.12" in run_name:
        intensity = "aggr"
    elif "boundary_mass_range=[0.02" in run_name or "rho_range=[0.03" in run_name:
        intensity = "mild"
    parts = []
    if cf:
        parts.append("cf" + cf.group(1).replace("p", "."))
    if fm:
        parts.append(fm.group(1))
    if cf and cf.group(1) not in ("0", "0.0"):
        parts.append(intensity)
    if seed:
        parts.append("s" + seed.group(1))
    return "·".join(parts) or run_name[:20]


#: How Exp2's freeze strategies read in a label — short enough for a small-multiple title.
_STRATEGY_SHORT = {"full": "full", "icl_only": "icl", "head_only": "head", "scratch": "scratch"}


def _arm_label_exp2(run_name: str) -> str:
    """Label an Exp2 arm from its fine-tuning levers: credit fraction, freeze strategy,
    L2-SP on/off, learning rate, seed — e.g. `cf0.5·icl·l2·lr1e-5`."""
    cf = re.search(r"credit_fraction=([0-9p.]+)", run_name)
    # A fixed set, matched explicitly: `[a-z_]+` is greedy and swallows the `__prior…` that
    # follows `strategy=icl_only` in the run name.
    strat = re.search(r"strategy=(full|icl_only|head_only|scratch)", run_name)
    l2 = re.search(r"l2sp_alpha=([0-9pm.e+-]+)", run_name)
    lr = re.search(r"-lr=([0-9pm.e+-]+)", run_name)
    seed = re.search(r"__s(\d+)", run_name)
    parts: list[str] = []
    if cf:
        parts.append("cf" + cf.group(1).replace("p", "."))
    if strat:
        parts.append(_STRATEGY_SHORT.get(strat.group(1), strat.group(1)))
    if l2:
        # `_fmt` writes 0.0 as "0" and 0.003 as "0p003"; only the latter is L2-SP on.
        parts.append("l2" if l2.group(1) not in ("0", "0p0") else "noL2")
    if lr:
        parts.append("lr" + lr.group(1).replace("m", "-").replace("p", "."))
    if seed:
        parts.append("s" + seed.group(1))
    return "·".join(parts) or run_name[:20]


def _is_control(run_name: str) -> bool:
    return bool(re.search(r"credit_fraction=0(?:\.0|p0)?(?=__)", run_name))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _metric_cols(df: pd.DataFrame, metric: str, domain: str = "real") -> list[str]:
    """Columns like `real__<dataset>__<metric>` for one metric and domain."""
    cols = [c for c in df.columns if c.startswith(f"{domain}__") and c.endswith(f"__{metric}")]
    alias = {"roc_auc": "auc", "pr_auc": "ap", "auc": "roc_auc", "ap": "pr_auc"}.get(metric)
    if not cols and alias:
        cols = [c for c in df.columns if c.startswith(f"{domain}__") and c.endswith(f"__{alias}")]
    return cols


def _mean_metric(df: pd.DataFrame, metric: str, domain: str = "real") -> pd.Series:
    """A domain mean requires every tracked dataset; missing values remain visible."""
    cols = _metric_cols(df, metric, domain)
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].mean(axis=1, skipna=False)


def available_metrics(runs: dict[str, pd.DataFrame], domain: str = "real") -> list[str]:
    """Every evaluation metric present in the progress CSVs, in a sensible order."""
    seen: set[str] = set()
    for df in runs.values():
        for c in df.columns:
            m = re.match(rf"{domain}__.+__([a-z_0-9]+)$", c)
            if m:
                seen.add(m.group(1))
    # Put the interpretable headline metrics first.
    priority = ["auc", "ap", "r2", "rmse", "mae", "pinball", "crps", "brier", "logloss",
                "ks", "calibration_slope", "spearman", "kendall", "boundary_mass_abs_err",
                "coverage_80", "pit_mean"]
    ordered = [m for m in priority if m in seen] + sorted(seen - set(priority))
    # Drop bookkeeping / non-metric columns.
    return [m for m in ordered if m not in ("n_test", "n_boundary", "n_interior")]


def _final(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def rank_arms(runs: dict[str, pd.DataFrame], track: str) -> list[tuple[str, float]]:
    """Arms sorted best -> worst by the final headline metric (averaged over real datasets)."""
    metric = HEADLINE[track]
    scored = [(name, _final(_mean_metric(df, metric))) for name, df in runs.items()]
    scored = [(n, v) for n, v in scored if not np.isnan(v)]
    return sorted(scored, key=lambda kv: kv[1], reverse=HIGHER_IS_BETTER.get(metric, True))


# ---------------------------------------------------------------------------
# 1. Training loss
# ---------------------------------------------------------------------------


def training_loss(track: str, exp: str = "exp1"):
    """Train loss vs step: every arm faint, the mean bold, best and worst arms highlighted."""
    runs = load_progress(track, exp)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found in output/manifests/")
        return fig

    grid = _common_step_grid(runs, "train_loss")
    stack = []
    for name, df in runs.items():
        d = df.dropna(subset=["train_loss"])
        if len(d) < 2:
            continue
        ax.plot(d["step"], d["train_loss"], color=style.MUTED, alpha=0.28, lw=0.8, zorder=1)
        stack.append(np.interp(grid, d["step"], d["train_loss"], left=np.nan, right=np.nan))
    if stack:
        mean = np.nanmean(np.array(stack), axis=0)
        ax.plot(grid, mean, color=style.INK, lw=2.2, label="mean of all arms", zorder=4)
    ranked = rank_arms(runs, track)
    for name, colour, tag in _best_worst_pair(ranked):
        d = runs[name].dropna(subset=["train_loss"])
        ax.plot(d["step"], d["train_loss"], color=colour, lw=1.8, zorder=5,
                label=f"{tag}: {arm_label(name)}")
    ax.set_xlabel("training step")
    ax.set_ylabel("train loss")
    ax.legend(loc="upper right")
    style.title(ax, f"Training loss, {len(runs)} arms")
    fig.suptitle(f"{track.upper()} training loss")
    return fig


# ---------------------------------------------------------------------------
# 2. Headline metric over training
# ---------------------------------------------------------------------------


def metric_over_training(track: str, exp: str = "exp1", metric: str | None = None):
    """A real-data metric vs step (averaged over the real datasets): every arm + the mean."""
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found")
        return fig

    grid = _common_step_grid(runs, metric_series=lambda df: _mean_metric(df, metric))
    stack = []
    for name, df in runs.items():
        s = _mean_metric(df, metric)
        d = pd.DataFrame({"step": df["step"], "m": s}).dropna()
        if len(d) < 2:
            continue
        colour = style.ORIGINAL if _is_control(name) else style.CREDIT
        ax.plot(d["step"], d["m"], color=colour, alpha=0.30, lw=0.9, zorder=1)
        stack.append(np.interp(grid, d["step"], d["m"], left=np.nan, right=np.nan))
    if stack:
        ax.plot(grid, np.nanmean(np.array(stack), axis=0), color=style.INK, lw=2.2,
                label="mean of all arms", zorder=4)
    ax.plot([], [], color=style.CREDIT, lw=1.5, label="credit prior arms")
    ax.plot([], [], color=style.ORIGINAL, lw=1.5, label="control (cf=0) arms")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"real-data {metric} (mean over datasets)")
    ax.legend(loc="lower right")
    style.title(ax, f"{metric.upper()} on the real datasets during training")
    fig.suptitle(f"{track.upper()} real-data {metric} over training")
    return fig


# ---------------------------------------------------------------------------
# 2b. Real-credit vs out-of-domain retention (the Exp2 question)
# ---------------------------------------------------------------------------


def real_vs_ood(track: str, exp: str = "exp1", metric: str | None = None):
    """Headline metric on the real-credit datasets vs the out-of-domain suites, over training.

    Fine-tuning on a narrow credit prior can lift credit scores while eroding the generality the
    released model came with. Each arm contributes two curves — credit (solid) and OOD (dashed) —
    both averaged over their datasets, so a gap opening up is the cost of specialisation made
    visible. Degrades to the credit curve alone when no OOD columns were logged.
    """
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found in output/manifests/")
        return fig

    grid = _common_step_grid(runs)
    real_stack, ood_stack = [], []
    for df in runs.values():
        for domain, stack in (("real", real_stack), ("ood", ood_stack)):
            s = _mean_metric(df, metric, domain)
            d = pd.DataFrame({"step": df["step"], "m": s}).dropna()
            if len(d) >= 2:
                stack.append(np.interp(grid, d["step"], d["m"], left=np.nan, right=np.nan))
    if real_stack:
        ax.plot(grid, np.nanmean(np.array(real_stack), axis=0), color=style.CREDIT, lw=2.2,
                label=f"real credit (mean {metric})", zorder=4)
    if ood_stack:
        ax.plot(grid, np.nanmean(np.array(ood_stack), axis=0), color=style.WARN, lw=2.2,
                ls="--", label=f"out-of-domain (mean {metric})", zorder=4)
    else:
        ax.text(0.5, 0.08, "no out-of-domain columns logged (progress.n_ood = 0?)",
                ha="center", va="center", color=style.MUTED, fontsize=8, transform=ax.transAxes)
    ax.set_xlabel("training step")
    ax.set_ylabel(f"mean {metric}")
    ax.legend(loc="lower right")
    style.title(ax, "Credit vs out-of-domain during training", "does specialising cost generality?")
    fig.suptitle(f"{track.upper()} credit vs out-of-domain {metric}")
    return fig


# ---------------------------------------------------------------------------
# 3. Every logged evaluation metric
# ---------------------------------------------------------------------------


def metric_pages(track: str, exp: str = "exp1", per_page: int = 9) -> int:
    runs = load_progress(track, exp)
    n = len(available_metrics(runs)) if runs else 0
    return max(1, int(np.ceil(n / per_page)))


def all_eval_metrics(track: str, page: int = 1, exp: str = "exp1", per_page: int = 9):
    """A grid, one panel per logged real-data metric, each the mean over arms and datasets."""
    runs = load_progress(track, exp)
    style.apply()
    metrics = available_metrics(runs) if runs else []
    pages = max(1, int(np.ceil(len(metrics) / per_page)))
    chunk = metrics[(page - 1) * per_page: page * per_page]
    ncols = 3
    nrows = max(1, int(np.ceil(len(chunk) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.80))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    if not chunk:
        _empty(axes[0], "no evaluation metrics logged yet")
    for ax, metric in zip(axes, chunk):
        ax.axis("on")
        grid = _common_step_grid(runs, metric_series=lambda df, m=metric: _mean_metric(df, m))
        stack = []
        for df in runs.values():
            s = _mean_metric(df, metric)
            d = pd.DataFrame({"step": df["step"], "m": s}).dropna()
            if len(d) < 2:
                continue
            stack.append(np.interp(grid, d["step"], d["m"], left=np.nan, right=np.nan))
        if stack:
            ax.plot(grid, np.nanmean(np.array(stack), axis=0), color=style.CREDIT, lw=1.6)
        ax.set_xlabel("step")
        arrow = "↑" if HIGHER_IS_BETTER.get(metric, True) else "↓"
        style.title(ax, f"{metric}  ({arrow} better)")
    fig.suptitle(f"{track.upper()} evaluation metrics over training{style.page_suffix(page, pages)}")
    return fig


# ---------------------------------------------------------------------------
# 4. Per-configuration small multiples
# ---------------------------------------------------------------------------


def config_pages(track: str, exp: str = "exp1", per_page: int = 12) -> int:
    runs = load_progress(track, exp)
    return max(1, int(np.ceil(len(runs) / per_page)))


def per_config(track: str, page: int = 1, exp: str = "exp1", per_page: int = 12):
    """One panel per arm: its train loss (grey) and headline metric (blue, right axis)."""
    runs = load_progress(track, exp)
    metric = HEADLINE[track]
    names = sorted(runs, key=lambda n: (_is_control(n), n))
    pages = max(1, int(np.ceil(len(names) / per_page)))
    chunk = names[(page - 1) * per_page: page * per_page]
    style.apply()
    ncols = 4
    nrows = max(1, int(np.ceil(len(chunk) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.82))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    if not chunk:
        _empty(axes[0], "no arms have started yet")
    for ax, name in zip(axes, chunk):
        ax.axis("on")
        df = runs[name]
        d = df.dropna(subset=["train_loss"])
        ax.plot(d["step"], d["train_loss"], color=style.MUTED, lw=1.0)
        ax.set_yticks([])
        ax2 = ax.twinx()
        s = _mean_metric(df, metric)
        m = pd.DataFrame({"step": df["step"], "m": s}).dropna()
        if len(m) >= 2:
            ax2.plot(m["step"], m["m"], color=style.CREDIT, lw=1.2)
        ax2.set_yticks([])
        ax.set_xticks([])
        style.title(ax, arm_label(name))
    fig.suptitle(f"{track.upper()} per-arm: loss (grey) and {metric} (blue)"
                 f"{style.page_suffix(page, pages)}")
    return fig


# ---------------------------------------------------------------------------
# 5. Best vs worst configuration
# ---------------------------------------------------------------------------


def best_and_worst(track: str, exp: str = "exp1"):
    """The best and worst arm by final headline metric, loss and metric side by side."""
    runs = load_progress(track, exp)
    metric = HEADLINE[track]
    ranked = rank_arms(runs, track)
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.44))
    if len(ranked) < 2:
        _empty(ax1, "need at least two finished arms")
        ax2.axis("off")
        return fig
    picks = [(ranked[0][0], style.CREDIT_STRONG, f"best ({ranked[0][1]:.3f})"),
             (ranked[-1][0], style.WARN, f"worst ({ranked[-1][1]:.3f})")]
    for name, colour, tag in picks:
        d = runs[name].dropna(subset=["train_loss"])
        ax1.plot(d["step"], d["train_loss"], color=colour, lw=1.8, label=f"{arm_label(name)}")
        s = _mean_metric(runs[name], metric)
        m = pd.DataFrame({"step": runs[name]["step"], "m": s}).dropna()
        ax2.plot(m["step"], m["m"], color=colour, lw=1.8, label=f"{tag}: {arm_label(name)}")
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss")
    style.title(ax1, "Train loss")
    ax2.set_xlabel("step"); ax2.set_ylabel(f"real-data {metric}")
    ax2.legend(loc="lower right")
    style.title(ax2, f"{metric.upper()} on real data")
    fig.suptitle(f"{track.upper()} best vs worst arm")
    return fig


# ---------------------------------------------------------------------------
# 6. Hardware telemetry
# ---------------------------------------------------------------------------


def hardware(track: str, exp: str = "exp1"):
    """GPU utilisation, throughput and peak memory over the run, pooled across arms."""
    tel = load_telemetry(track, exp)
    style.apply()
    panels = [("gpu0_utilization_gpu", "GPU utilisation %", (0, 100)),
              ("steps_per_s", "throughput (steps/s)", None),
              ("mem_max_allocated_gb", "peak allocated GB", None)]
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(style.WIDTH_FULL, 0.34))
    if not tel:
        _empty(axes[0], "no telemetry CSVs found")
        for ax in axes[1:]:
            ax.axis("off")
        return fig
    for ax, (col, label, ylim) in zip(axes, panels):
        any_data = False
        for df in tel.values():
            if col in df:
                d = df.dropna(subset=[col])
                if len(d):
                    ax.plot(d["step"], d[col], color=style.CREDIT, alpha=0.35, lw=0.9)
                    any_data = True
        if col == "gpu0_utilization_gpu" and any_data:
            ax.axhline(70, color=style.WARN, ls="--", lw=1.0)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("step")
        style.title(ax, label)
    fig.suptitle(f"{track.upper()} hardware during training")
    return fig


# ---------------------------------------------------------------------------
# 7. Per-block gradient flow
# ---------------------------------------------------------------------------


def gradient_flow(track: str, exp: str = "exp1"):
    """Per-block gradient norm over training — every stack should carry signal, none flat.

    Under an Exp2 freeze strategy (`icl_only`, `head_only`) the frozen stacks legitimately sit on
    the floor: this figure is where a reader confirms the freeze took, so a flat line is a result
    there rather than a warning.
    """
    tel = load_telemetry(track, exp)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.50))
    blocks = [("grad_col", "column encoder"), ("grad_row", "row encoder"),
              ("grad_icl", "ICL blocks"), ("grad_head", "head")]
    have = False
    for (col, label), colour in zip(blocks, style.SERIES):
        stacks = []
        allsteps = sorted({int(s) for df in tel.values() if col in df
                           for s in df.dropna(subset=[col])["step"]})
        if not allsteps:
            continue
        grid = np.array(allsteps)
        for df in tel.values():
            if col not in df:
                continue
            d = df.dropna(subset=[col])
            if len(d) >= 2:
                stacks.append(np.interp(grid, d["step"], d[col], left=np.nan, right=np.nan))
        if stacks:
            ax.plot(grid, np.nanmean(np.array(stacks), axis=0), color=colour, lw=1.6, label=label)
            have = True
    if not have:
        _empty(ax, "no gradient-norm samples logged (grad_every=0?)")
        return fig
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient L2 norm (log)")
    ax.legend(loc="upper right")
    style.title(ax, "Per-block gradient norm", "every stack should stay off the floor")
    fig.suptitle(f"{track.upper()} gradient flow")
    return fig


# ---------------------------------------------------------------------------
# 8. Deeper views — the comparison, the levers, and per-dataset behaviour
# ---------------------------------------------------------------------------


#: How each swept lever reads out of a run name, per experiment. Used to colour and group curves
#: by the one knob a figure is about, rather than by the arm as a whole.
def _lever_value(run_name: str, lever: str) -> str | None:
    if lever == "intensity":  # Exp1's two prior intensities, read off the swept range
        if "[0.12,0.3]" in run_name or "[0.15,0.6]" in run_name:
            return "aggressive"
        if "[0.03,0.12]" in run_name or "[0.02,0.3]" in run_name:
            return "mild"
        return None
    pats = {"credit_fraction": r"credit_fraction=([0-9p.]+)", "filter": r"filter-mode=([a-z]+)",
            "strategy": r"strategy=(full|icl_only|head_only|scratch)",
            "l2sp": r"l2sp_alpha=([0-9pm.e+-]+)", "lr": r"-lr=([0-9pm.e+-]+)"}
    m = re.search(pats.get(lever, r"(?!)"), run_name)
    if not m:
        return None
    v = m.group(1)
    if lever == "credit_fraction":
        return "cf=" + v.replace("p", ".")
    if lever == "l2sp":
        return "L2-SP on" if v not in ("0", "0p0") else "L2-SP off"
    if lever == "lr":
        return "lr " + v.replace("m", "-").replace("p", ".")
    return v


def _real_datasets(runs: dict[str, pd.DataFrame], metric: str) -> list[str]:
    """The real credit datasets that carry `metric`, in first-seen order."""
    seen: list[str] = []
    for df in runs.values():
        for c in df.columns:
            m = re.match(rf"real__(.+)__{re.escape(metric)}$", c)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
    return seen


def credit_vs_control_over_training(track: str, exp: str = "exp1", metric: str | None = None):
    """The headline comparison as a curve: credit-prior arms vs control arms, median and IQR."""
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found in output/manifests/")
        return fig
    grid = _common_step_grid(runs)
    groups: dict[str, list] = {"credit prior": [], "control (cf=0)": []}
    for name, df in runs.items():
        s = _mean_metric(df, metric)
        d = pd.DataFrame({"step": df["step"], "m": s}).dropna()
        if len(d) < 2:
            continue
        arr = np.interp(grid, d["step"], d["m"], left=np.nan, right=np.nan)
        groups["control (cf=0)" if _is_control(name) else "credit prior"].append(arr)
    for label, colour in (("credit prior", style.CREDIT), ("control (cf=0)", style.ORIGINAL)):
        stack = groups[label]
        if not stack:
            continue
        arr = np.array(stack)
        med = np.nanmedian(arr, axis=0)
        lo, hi = np.nanpercentile(arr, [25, 75], axis=0)
        ax.fill_between(grid, lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot(grid, med, color=colour, lw=2.3, label=f"{label} (n={len(stack)})", zorder=4)
    ax.set_xlabel("training step")
    ax.set_ylabel(f"real-data {metric} (median, IQR band)")
    ax.legend(loc="lower right")
    style.title(ax, "Credit prior vs control", "median across arms, shaded IQR")
    fig.suptitle(f"{track.upper()} credit vs control over training")
    return fig


def metric_by_lever(track: str, lever: str, exp: str = "exp1", metric: str | None = None):
    """The headline metric over training, one mean line per value of a single swept lever."""
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found")
        return fig
    grid = _common_step_grid(runs)
    by_val: dict[str, list] = {}
    for name, df in runs.items():
        v = _lever_value(name, lever)
        if v is None:
            continue
        s = _mean_metric(df, metric)
        d = pd.DataFrame({"step": df["step"], "m": s}).dropna()
        if len(d) < 2:
            continue
        by_val.setdefault(v, []).append(np.interp(grid, d["step"], d["m"], left=np.nan, right=np.nan))
    if not by_val:
        _empty(ax, f"no arms carry the '{lever}' lever")
        return fig
    for (v, stack), colour in zip(sorted(by_val.items()), style.SERIES):
        ax.plot(grid, np.nanmean(np.array(stack), axis=0), color=colour, lw=1.9,
                label=f"{v}  (n={len(stack)})")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"real-data {metric} (mean)")
    ax.legend(loc="lower right", title=lever.replace("_", " "))
    style.title(ax, f"{metric.upper()} by {lever.replace('_', ' ')}")
    fig.suptitle(f"{track.upper()} {metric} by {lever}")
    return fig


def per_dataset_pages(track: str, exp: str = "exp1", per_page: int = 6) -> int:
    runs = load_progress(track, exp)
    n = len(_real_datasets(runs, HEADLINE[track])) if runs else 0
    return max(1, int(np.ceil(n / per_page)))


def per_dataset_curves(track: str, page: int = 1, exp: str = "exp1", per_page: int = 6,
                       metric: str | None = None):
    """The headline metric over training, one panel per real dataset — credit (blue) vs control."""
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    datasets = _real_datasets(runs, metric) if runs else []
    pages = max(1, int(np.ceil(len(datasets) / per_page)))
    chunk = datasets[(page - 1) * per_page: page * per_page]
    ncols = 3
    nrows = max(1, int(np.ceil(len(chunk) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.84))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    if not chunk:
        _empty(axes[0], "no per-dataset metrics logged yet")
    grid = _common_step_grid(runs) if runs else np.array([0.0, 1.0])
    for ax, ds in zip(axes, chunk):
        ax.axis("on")
        col = f"real__{ds}__{metric}"
        cstack, ostack = [], []
        for name, df in runs.items():
            if col not in df:
                continue
            d = df[["step", col]].dropna()
            if len(d) < 2:
                continue
            arr = np.interp(grid, d["step"], d[col], left=np.nan, right=np.nan)
            (ostack if _is_control(name) else cstack).append(arr)
        if cstack:
            ax.plot(grid, np.nanmean(np.array(cstack), axis=0), color=style.CREDIT, lw=1.6)
        if ostack:
            ax.plot(grid, np.nanmean(np.array(ostack), axis=0), color=style.ORIGINAL, lw=1.4)
        ax.set_xlabel("step")
        style.title(ax, ds[:22])
    fig.suptitle(f"{track.upper()} {metric} per dataset{style.page_suffix(page, pages)}")
    return fig


def weight_gradient_ratios(track: str, exp: str = "exp1"):
    """Per-block gradient-to-weight ratio over training — the interpretable "is it learning?"."""
    tel = load_telemetry(track, exp)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.50))
    blocks = [("gw_ratio_col", "column encoder"), ("gw_ratio_row", "row encoder"),
              ("gw_ratio_icl", "ICL blocks"), ("gw_ratio_head", "head")]
    have = False
    for (col, label), colour in zip(blocks, style.SERIES):
        allsteps = sorted({int(s) for df in tel.values() if col in df
                           for s in df.dropna(subset=[col])["step"]})
        if not allsteps:
            continue
        grid = np.array(allsteps)
        stacks = []
        for df in tel.values():
            if col not in df:
                continue
            d = df.dropna(subset=[col])
            if len(d) >= 2:
                stacks.append(np.interp(grid, d["step"], d[col], left=np.nan, right=np.nan))
        if stacks:
            ax.plot(grid, np.nanmean(np.array(stacks), axis=0), color=colour, lw=1.6, label=label)
            have = True
    if not have:
        _empty(ax, "no gradient-to-weight ratios logged (grad_every=0?)")
        return fig
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient / weight (log)")
    ax.legend(loc="upper right")
    style.title(ax, "Gradient-to-weight ratio", "a block far below the rest is frozen")
    fig.suptitle(f"{track.upper()} gradient-to-weight ratio")
    return fig


def final_metric_by_lever(track: str, exp: str = "exp1", metric: str | None = None):
    """Final headline metric of every finished arm, grouped by each swept lever in turn."""
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    levers = (("credit_fraction", "credit fraction"), ("filter", "filter mode"),
              ("intensity", "prior intensity")) if exp == "exp1" else \
             (("credit_fraction", "credit fraction"), ("strategy", "freeze strategy"),
              ("l2sp", "L2-SP"), ("lr", "learning rate"))
    n = len(levers)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.72))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    finals = {name: _final(_mean_metric(df, metric)) for name, df in runs.items()} if runs else {}
    finals = {k: v for k, v in finals.items() if not np.isnan(v)}
    if not finals:
        _empty(axes[0], "no finished arms to summarise yet")
        return fig
    jitter = np.random.default_rng(0)
    for ax, (lever, label) in zip(axes, levers):
        by: dict[str, list] = {}
        for name, v in finals.items():
            lv = _lever_value(name, lever)
            if lv is not None:
                by.setdefault(lv, []).append(v)
        if not by:
            continue
        ax.axis("on")
        xs = sorted(by)
        for i, x in enumerate(xs):
            vals = np.array(by[x])
            ax.scatter(np.full(len(vals), i) + (jitter.random(len(vals)) - 0.5) * 0.16, vals,
                       s=20, color=style.CREDIT, alpha=0.6, edgecolor="white", linewidth=0.3)
            ax.plot([i - 0.22, i + 0.22], [vals.mean()] * 2, color=style.INK, lw=1.8, zorder=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(metric, fontsize=8)
        style.title(ax, label)
    fig.suptitle(f"{track.upper()} final {metric} by lever")
    return fig


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def training_summary(track: str, exp: str = "exp1") -> str:
    """A text summary of the training runs, for the notebook's final cell."""
    runs = load_progress(track, exp)
    tel = load_telemetry(track, exp)
    if not runs:
        return f"{exp.upper()} {track.upper()} training: no progress CSVs in output/manifests/ yet."
    metric = HEADLINE[track]
    ranked = rank_arms(runs, track)
    lines = [f"{exp.upper()} {track.upper()} TRAINING — {len(runs)} arms with progress data",
             f"  telemetry CSVs: {len(tel)}",
             f"  metrics logged: {', '.join(available_metrics(runs)) or 'none'}"]
    steps = [int(df['step'].max()) for df in runs.values() if len(df)]
    if steps:
        lines.append(f"  furthest step reached: {max(steps):,}")
    if ranked:
        lines.append(f"  best  {metric}={ranked[0][1]:.4f}  {arm_label(ranked[0][0])}")
        lines.append(f"  worst {metric}={ranked[-1][1]:.4f}  {arm_label(ranked[-1][0])}")
        vals = np.array([v for _, v in ranked])
        lines.append(f"  {metric} across arms: mean {vals.mean():.4f}  sd {vals.std():.4f}")
    ood = [n for n, df in runs.items() if _metric_cols(df, metric, "ood")]
    if ood:
        lines.append(f"  out-of-domain {metric} tracked on {len(ood)}/{len(runs)} arms")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------


def _common_step_grid(runs: dict[str, pd.DataFrame], col: str | None = None,
                      metric_series=None) -> np.ndarray:
    """A shared step axis to average curves of different lengths onto."""
    hi = 0
    for df in runs.values():
        if len(df):
            hi = max(hi, int(df["step"].max()))
    return np.linspace(0, max(hi, 1), 60)


def _best_worst_pair(ranked: list[tuple[str, float]]):
    if len(ranked) >= 2:
        return [(ranked[0][0], style.CREDIT_STRONG, "best"), (ranked[-1][0], style.WARN, "worst")]
    return []


def _empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=style.MUTED,
            fontsize=9, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
