"""Level-1 TRAINING visualisation: what every arm did while it trained.

Reads the two CSVs each arm writes to `output/manifests/` — `<run>__progress.csv` (train
loss and real/OOD evaluation metrics, sampled every `progress.every_datasets`) and
`<run>__telemetry.csv` (GPU utilisation, throughput, per-block gradient norms) — and turns
them into the training-behaviour figures for `1.1_pd_training` / `1.2_lgd_training`.

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
HEADLINE = {"pd": "auc", "lgd": "r2"}
HIGHER_IS_BETTER = {"auc": True, "ap": True, "r2": True, "spearman": True, "kendall": True,
                    "brier": False, "logloss": False, "rmse": False, "mae": False,
                    "pinball": False, "crps": False, "ks": True, "calibration_slope": True}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _manifest_files(track: str, kind: str) -> list:
    return sorted(paths.manifests_dir().glob(f"exp1_{track}__*__{kind}.csv"))


def load_progress(track: str) -> dict[str, pd.DataFrame]:
    """`{run_name: progress DataFrame}` for every started arm of `track`."""
    out: dict[str, pd.DataFrame] = {}
    for path in _manifest_files(track, "progress"):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if "step" in df and len(df):
            out[path.name.replace("__progress.csv", "")] = df.sort_values("step")
    return out


def load_telemetry(track: str) -> dict[str, pd.DataFrame]:
    """`{run_name: telemetry DataFrame}` for every started arm of `track`."""
    out: dict[str, pd.DataFrame] = {}
    for path in _manifest_files(track, "telemetry"):
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
    """A short, readable label from the sweep levers, e.g. `cf1·banded·aggr·s2`."""
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


def _is_control(run_name: str) -> bool:
    return bool(re.search(r"credit_fraction=0(?:\.0|p0)?(?=__)", run_name))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _metric_cols(df: pd.DataFrame, metric: str, domain: str = "real") -> list[str]:
    """Columns like `real__<dataset>__<metric>` for one metric and domain."""
    return [c for c in df.columns if c.startswith(f"{domain}__") and c.endswith(f"__{metric}")]


def _mean_metric(df: pd.DataFrame, metric: str, domain: str = "real") -> pd.Series:
    """The metric averaged over that domain's datasets, per step (NaN datasets ignored)."""
    cols = _metric_cols(df, metric, domain)
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].mean(axis=1, skipna=True)


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


def training_loss(track: str):
    """Train loss vs step: every arm faint, the mean bold, best and worst arms highlighted."""
    runs = load_progress(track)
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


def metric_over_training(track: str, metric: str | None = None):
    """A real-data metric vs step (averaged over the real datasets): every arm + the mean."""
    runs = load_progress(track)
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
# 3. Every logged evaluation metric
# ---------------------------------------------------------------------------


def metric_pages(track: str, per_page: int = 9) -> int:
    runs = load_progress(track)
    n = len(available_metrics(runs)) if runs else 0
    return max(1, int(np.ceil(n / per_page)))


def all_eval_metrics(track: str, page: int = 1, per_page: int = 9):
    """A grid, one panel per logged real-data metric, each the mean over arms and datasets."""
    runs = load_progress(track)
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


def config_pages(track: str, per_page: int = 12) -> int:
    runs = load_progress(track)
    return max(1, int(np.ceil(len(runs) / per_page)))


def per_config(track: str, page: int = 1, per_page: int = 12):
    """One panel per arm: its train loss (grey) and headline metric (blue, right axis)."""
    runs = load_progress(track)
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


def best_and_worst(track: str):
    """The best and worst arm by final headline metric, loss and metric side by side."""
    runs = load_progress(track)
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


def hardware(track: str):
    """GPU utilisation, throughput and peak memory over the run, pooled across arms."""
    tel = load_telemetry(track)
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


def gradient_flow(track: str):
    """Per-block gradient norm over training — every stack should carry signal, none flat."""
    tel = load_telemetry(track)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.50))
    blocks = [("grad_col", "column encoder"), ("grad_row", "row encoder"),
              ("grad_icl", "ICL blocks"), ("grad_head", "head")]
    have = False
    for (col, label), colour in zip(blocks, style.SERIES):
        grids, stacks = None, []
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
# Summary
# ---------------------------------------------------------------------------


def training_summary(track: str) -> str:
    """A text summary of the training runs, for the notebook's final cell."""
    runs = load_progress(track)
    tel = load_telemetry(track)
    if not runs:
        return f"{track.upper()} training: no progress CSVs in output/manifests/ yet."
    metric = HEADLINE[track]
    ranked = rank_arms(runs, track)
    lines = [f"{track.upper()} TRAINING — {len(runs)} arms with progress data",
             f"  telemetry CSVs: {len(tel)}",
             f"  metrics logged: {', '.join(available_metrics(runs)) or 'none'}"]
    steps = [int(df['step'].max()) for df in runs.values() if len(df)]
    if steps:
        lines.append(f"  furthest step reached: {max(steps):,} (of 12,500)")
    if ranked:
        lines.append(f"  best  {metric}={ranked[0][1]:.4f}  {arm_label(ranked[0][0])}")
        lines.append(f"  worst {metric}={ranked[-1][1]:.4f}  {arm_label(ranked[-1][0])}")
        vals = np.array([v for _, v in ranked])
        lines.append(f"  {metric} across arms: mean {vals.mean():.4f}  sd {vals.std():.4f}")
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
