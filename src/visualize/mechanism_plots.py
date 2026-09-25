"""Visualise the credit-specific MECHANISMS the adjusted prior adds.

`prior_plots` and `pool_plots` show generic prior properties — target shapes, table
sizes, correlation spectra. This module shows the parts that make the prior
*credit-shaped*, the ones the Exp1 redesign is actually about:

* boundary atoms and their INTENSITY (LGD: mild vs aggressive);
* controlled imbalance (PD: the base rate held in credit's measured regime);
* correlated defaults (PD: the Vasicek systematic factor, mild vs aggressive `rho`);
* reject inference (PD: the approved book in context, the rejected region in the query);
* distribution shift (both: each task's context mean against its query mean, per kind);
* informative missingness (both: a missing rate that depends on the outcome);
* predictability (both: synthetic tasks and real datasets scored by the filter's own
  `predictability`, with what `tabicl` / `banded` / `off` keep).

Every figure GENERATES LIVE from the current config through `TaskGenerator` — the exact
code path training uses — so it can never show a stale prior the way a pre-generated
pool can. Rows are capped (`n_rows`) purely so a notebook runs in seconds; the
mechanisms are shape-invariant. The notebooks call these and hold no logic of their own.
"""

from __future__ import annotations

import copy
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.prior.filters import predictability
from src.prior.rng import PriorRNG
from src.visualize.draw import draw, parallel_map
from src.prior.targets.pd import apply_informative_missingness
from src.utils.config import expand_with_seeds, load
from src.utils.target_stats import target_stats
from src.visualize import literature, style

# Real base rates measured in data/raw/pd (see docs/PRIORS.md / config comments): the
# regime our controlled imbalance targets, drawn as a reference band.
REAL_PD_BAND = (0.067, 0.221)


# ---------------------------------------------------------------------------
# Live generation with config overrides
# ---------------------------------------------------------------------------


def _prior(config_path: str, *, n_rows: int = 400, grid_index: int = 0) -> tuple[str, dict]:
    """The resolved prior config for one arm, with rows capped for fast interactive draws."""
    cfg = expand_with_seeds(load(config_path))[grid_index]
    prior = copy.deepcopy(cfg["prior"])
    prior["n_rows_range"] = [n_rows, n_rows]
    return cfg["task"], prior


def _set(prior: dict, dotted: str, value: Any) -> None:
    d = prior
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _generate(task: str, prior: dict, n: int, seed: int, overrides: dict | None = None) -> list[Any]:
    """`n` tasks generated exactly as training would, after applying dotted-path overrides —
    in parallel worker processes for a large draw (src/visualize/draw.py)."""
    p = copy.deepcopy(prior)
    for dotted, value in (overrides or {}).items():
        _set(p, dotted, value)
    return draw(task, p, n, seed)[0]


def _base_rate(t: Any) -> float:
    return float(t.y.float().mean())


# ---------------------------------------------------------------------------
# LGD — boundary atoms at two intensities
# ---------------------------------------------------------------------------


def intensity_atoms(config_path: str, n: int = 100, seed: int = 0):
    """LGD: the original prior against our prior at the two swept boundary intensities.

    One panel per prior, each the target pooled over `n` tasks as a share of rows. The panel title
    carries the mean boundary mass — for our prior the share of rows at exactly 0 or 1, for the
    original prior (whose target is standard-scaled, not on [0, 1]) the share tied at its own
    minimum or maximum, which is what the ±4 SD outlier clamp produces.
    """
    task, prior = _prior(config_path)
    if task != "lgd":
        raise ValueError("intensity_atoms is an LGD figure")
    arms = [
        ("original prior", style.ORIGINAL, {"credit_fraction": 0.0}),
        ("credit prior, mild", style.CREDIT_MILD,
         {"credit_fraction": 1.0, "filter.mode": "off", "credit.target.boundary_mass_range": [0.02, 0.30]}),
        ("credit prior, aggressive", style.CREDIT_STRONG,
         {"credit_fraction": 1.0, "filter.mode": "off", "credit.target.boundary_mass_range": [0.15, 0.60]}),
    ]
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(style.WIDTH_FULL, 0.36))
    for ax, (label, color, ov) in zip(axes, arms):
        tasks = _generate(task, prior, n, seed, ov)
        pooled = np.concatenate([t.y.numpy() for t in tasks]).astype(float)
        bounded = pooled.min() >= -1e-6 and pooled.max() <= 1 + 1e-6
        bins = np.linspace(0, 1, 41) if bounded else np.linspace(pooled.min(), pooled.max(), 41)
        ax.hist(pooled, bins=bins, weights=np.full(pooled.size, 1.0 / pooled.size), color=color,
                linewidth=0)
        stats = [target_stats(t.y) for t in tasks]
        bm = float(np.mean([s["frac_at_min"] + s["frac_at_max"] for s in stats]))
        ax.set_xlabel("LGD target" if bounded else "target, standardised scale")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(label, loc="left", fontsize=9)
        # The number where the panel is empty: between the two spikes of a bounded target, in the
        # corner beside the bell of the standardised one.
        where = "at 0 or 1" if bounded else "at its\nmin or max"
        ax.annotate(f"{bm:.0%} of rows\n{where}", (0.5, 0.95) if bounded else (0.03, 0.95),
                    xycoords="axes fraction", ha="center" if bounded else "left", va="top",
                    fontsize=7, color=style.INK)
    axes[0].set_ylabel("share of rows")
    return fig


# ---------------------------------------------------------------------------
# PD — controlled imbalance
# ---------------------------------------------------------------------------


def imbalance_control(config_path: str, n: int = 200, seed: int = 0):
    """PD: our prior holds the base rate in credit's measured 7-22% regime; the original does not."""
    task, prior = _prior(config_path)
    if task != "pd":
        raise ValueError("imbalance_control is a PD figure")
    orig = _generate(task, prior, n, seed, {"credit_fraction": 0.0})
    ours = _generate(task, prior, n, seed, {"credit_fraction": 1.0, "filter.mode": "off"})
    br_orig = np.array([_base_rate(t) for t in orig])
    br_ours = np.array([_base_rate(t) for t in ours])

    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.44))
    bins = np.linspace(0, 1, 41)
    ax.hist(br_orig, bins=bins, color=style.ORIGINAL, alpha=0.75, label="original TabICL prior")
    ax.hist(br_ours, bins=bins, color=style.CREDIT, alpha=0.8, label="our prior")
    ax.axvspan(*REAL_PD_BAND, color=style.REAL, alpha=0.15, zorder=0)
    ax.annotate("real credit\n7–22%", (np.mean(REAL_PD_BAND), ax.get_ylim()[1] * 0.9),
                color=style.REAL, fontsize=8, ha="center", weight="semibold")
    # The base rate below which a default-threshold classifier collapses to the majority class,
    # measured on real credit data — the regime our controlled imbalance deliberately reaches into.
    literature.line(ax, "tanna_paradox", label="Tanna: collapse < 10%")
    ax.set_xlabel("positive (default) rate per task")
    ax.set_ylabel("number of tasks")
    style.legend_below(ax, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# PD — correlated defaults (the Vasicek systematic factor)
# ---------------------------------------------------------------------------


def correlated_defaults(config_path: str, n: int = 200, seed: int = 0):
    """PD: with the target rate fixed, a higher `rho` widens the realised rate — a bad year
    moves the whole book. This is what "correlated defaults" buys, and why `rho` is swept."""
    task, prior = _prior(config_path)
    if task != "pd":
        raise ValueError("correlated_defaults is a PD figure")
    # Fix the TARGET rate so the only thing dispersing the REALISED rate is the correlation.
    fixed = {
        "credit_fraction": 1.0, "filter.mode": "off",
        "credit.target.base_rate_range": [0.15, 0.15],
        "credit.target.mechanism.base_rate_range": [0.15, 0.15],
        "credit.target.mechanism.woe_prob": 0.0,
    }
    mild = _generate(task, prior, n, seed, {**fixed, "credit.target.mechanism.rho_range": [0.03, 0.12]})
    aggr = _generate(task, prior, n, seed, {**fixed, "credit.target.mechanism.rho_range": [0.12, 0.30]})
    r_mild = np.array([_base_rate(t) for t in mild])
    r_aggr = np.array([_base_rate(t) for t in aggr])

    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.42))
    bins = np.linspace(0, 0.4, 33)
    ax1.hist(r_mild, bins=bins, color=style.CREDIT_MILD, alpha=0.8,
             label=f"mild, ρ in [0.03, 0.12] (SD {r_mild.std():.3f})")
    ax1.hist(r_aggr, bins=bins, color=style.CREDIT_STRONG, alpha=0.8,
             label=f"aggressive, ρ in [0.12, 0.30] (SD {r_aggr.std():.3f})")
    ax1.axvline(0.15, color=style.INK, ls="--", lw=1.1, label="target rate 15%")
    ax1.set_xlabel("realised default rate per task")
    ax1.set_ylabel("number of tasks")
    ax1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax1.set_title("Realised rate at a fixed 15% target", loc="left", fontsize=9)

    rhos = [0.03, 0.08, 0.15, 0.22, 0.30]
    sds = []
    for r in rhos:
        ts = _generate(task, prior, max(30, n // 3), seed,
                       {**fixed, "credit.target.mechanism.rho_range": [r, r]})
        sds.append(float(np.std([_base_rate(t) for t in ts])))
    ax2.plot(rhos, sds, "o-", color=style.CREDIT, label="our prior at a fixed ρ")
    # The Basel IRB corporate asset-correlation cap sits on this exact axis; our aggressive arm
    # (ρ up to 0.30) reaches past it. External domain knowledge, so it draws amber and marked.
    literature.line(ax2, "basel_corp", label="Basel IRB corporate cap 0.24", inline=False)
    ax2.set_xlabel("asset correlation ρ")
    ax2.set_ylabel("SD of the realised default rate")
    ax2.set_ylim(bottom=0)
    ax2.set_title("Spread of the realised rate against ρ", loc="left", fontsize=9)
    style.legend_below(fig, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# PD — reject inference (the selection shift)
# ---------------------------------------------------------------------------


def reject_inference(config_path: str, n: int = 120, seed: int = 0):
    """PD: the context is the approved (low-risk) book; the query reaches into the applicants the
    screen turned away, so the query is systematically the riskier book."""
    task, prior = _prior(config_path)
    if task != "pd":
        raise ValueError("reject_inference is a PD figure")
    tasks = _generate(task, prior, n, seed, {
        "credit_fraction": 1.0, "filter.mode": "off",
        "credit.shift.shift_prob": 1.0,
        "credit.shift.kind_weights": {"selection": 1.0},
    })
    sel = [t for t in tasks if t.meta.get("shift") == "selection"]
    if not sel:
        raise RuntimeError("no selection-shift tasks were produced")
    ctx = np.array([t.meta["context_risk_mean"] for t in sel])
    qry = np.array([t.meta["query_risk_mean"] for t in sel])

    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.44))
    ax1.scatter(ctx, qry, s=28, alpha=0.5, color=style.CREDIT, edgecolor="white", lw=0.4, zorder=3,
                label="one task")
    lim = float(max(ctx.max(), qry.max())) * 1.1 + 1e-3
    ax1.plot([0, lim], [0, lim], color=style.MUTED, ls="--", lw=1.0, zorder=1,
             label="no difference (y = x)")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("context (approved book) default rate")
    ax1.set_ylabel("query (through-the-door) default rate")
    ax1.set_title("Default rate, context vs query", loc="left", fontsize=9)

    gap = qry - ctx
    ax2.hist(gap, bins=20, color=style.CREDIT)
    ax2.axvline(0, color=style.INK, lw=1.0)
    ax2.axvline(float(gap.mean()), color=style.INK, ls="--", lw=1.2, label="mean difference")
    ax2.set_xlabel("query minus context default rate")
    ax2.set_ylabel("number of tasks")
    ax2.set_title("Query minus context", loc="left", fontsize=9)
    style.legend_below(fig, ncol=3)
    return fig


# ---------------------------------------------------------------------------
# Both — distribution shift, context vs query per kind
# ---------------------------------------------------------------------------


#: How each shift kind reads as a panel heading.
_SHIFT_TITLE = {"cohort": "cohort", "covariate": "covariate", "prior_prob": "prior probability",
                "selection": "selection (reject inference)"}


def _split_means(task_obj: Any, kind: str, cut: int, feature: int | None = None) -> tuple[float, float]:
    """(context mean, query mean) of what a shift kind moves: one feature, in SD units, for a
    covariate shift; the target (default rate, or mean LGD) for every other kind."""
    if kind == "covariate":
        col = np.asarray(task_obj.X, dtype=float)[:, feature]
        sd = float(col.std()) or 1.0
        z = (col - col.mean()) / sd
        return float(z[:cut].mean()), float(z[cut:].mean())
    y = np.asarray(task_obj.y, dtype=float).ravel()
    return float(y[:cut].mean()), float(y[cut:].mean())


def shift_kinds(config_path: str, n: int = 40, seed: int = 0):
    """What each distribution shift does to ONE task: its context mean against its query mean.

    One panel per shift kind, one point per generated task with that shift switched on. A point on
    the diagonal is a task whose query looks like its context; the distance from the diagonal is
    the shift, and its side is the direction. The grey cloud is the yardstick: the same prior with
    the shift switched OFF, split at the same point, so its spread around the diagonal is what
    sampling alone produces.

    Replaces a figure that POOLED every task's context rows against every task's query rows. Shifts
    whose direction varies from task to task (a prior-probability shift goes either way) cancel in
    such a pool, so the histograms coincided and showed nothing; per task, nothing cancels.
    `selection` (reject inference) is PD-only.
    """
    task, prior = _prior(config_path)
    kinds = ["cohort", "covariate", "prior_prob"] + (["selection"] if task == "pd" else [])
    quantity = "default rate" if task == "pd" else "mean LGD"
    frac = float(np.mean(prior.get("train_frac_range", [0.3, 0.9])))
    base = _generate(task, prior, n, seed + 1, {
        "credit_fraction": 1.0, "filter.mode": "off", "credit.shift.shift_prob": 0.0})
    pick = np.random.default_rng(seed)

    style.apply()
    ncols = 2 if len(kinds) == 4 else len(kinds)
    nrows = int(np.ceil(len(kinds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.82
                                                                      if ncols == 2 else 1.0),
                             squeeze=False)
    for i, (ax, kind) in enumerate(zip(axes.ravel(), kinds)):
        tasks = _generate(task, prior, n, seed, {
            "credit_fraction": 1.0, "filter.mode": "off",
            "credit.shift.shift_prob": 1.0,
            "credit.shift.kind_weights": {kind: 1.0},
        })
        shifted = [_split_means(t, kind, int(t.meta["shift_cut"]), t.meta.get("shift_feature"))
                   for t in tasks if t.meta.get("shift") == kind]
        reference = []
        for t in base:
            cut = max(8, min(t.n_rows - 8, int(round(t.n_rows * frac))))
            feature = None
            if kind == "covariate":
                X = np.asarray(t.X, dtype=float)
                varying = np.flatnonzero(X.std(axis=0) > 1e-12)
                if not varying.size:
                    continue
                feature = int(pick.choice(varying))
            reference.append(_split_means(t, kind, cut, feature))
        for pts, colour, size, label in ((reference, style.MUTED, 12, "same prior, shift switched off"),
                                         (shifted, style.CREDIT, 18, "task with this shift switched on")):
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=size, color=colour, alpha=0.45 if colour == style.MUTED else 0.7,
                           linewidths=0, zorder=2 if colour == style.MUTED else 3,
                           label=label if i == 0 else None)
        every = [v for pt in reference + shifted for v in pt]
        lo, hi = (min(every), max(every)) if every else (0.0, 1.0)
        pad = 0.05 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=style.INK, lw=0.8, ls="--", zorder=1,
                label="query = context" if i == 0 else None)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if kind == "covariate":
            ax.set_xlabel("feature, context mean (SD)")
            ax.set_ylabel("feature, query mean (SD)")
        else:
            ax.set_xlabel(f"{quantity}, context rows")
            ax.set_ylabel(f"{quantity}, query rows")
        ax.set_title(_SHIFT_TITLE[kind], loc="left", fontsize=9)
    style.legend_below(fig, ncol=3)
    return fig


# ---------------------------------------------------------------------------
# Both — informative missingness (MNAR)
# ---------------------------------------------------------------------------


def informative_missingness(task: str, n_rows: int = 4000, seed: int = 0):
    """The missing rate depends on the outcome (MNAR) — a thin file is itself a risk signal.

    Applies the real `apply_informative_missingness` to a controlled (X, y) at two couplings, and
    reads the missing rate back off the was-missing indicator columns it appends.
    """
    torch.manual_seed(seed)
    X = torch.randn(n_rows, 6)
    if task == "pd":
        y = (torch.rand(n_rows) < 0.2).float()
    else:
        y = torch.rand(n_rows).float()

    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.44))
    for beta, color, label in [(0.0, style.MUTED, "completely at random (coupling 0)"),
                               (2.0, style.CREDIT, "tied to the outcome (coupling 2)")]:
        rng = PriorRNG(seed)
        Xn, meta = apply_informative_missingness(
            rng, X.clone(), y,
            {"missing_col_fraction": 1.0, "missing_rate_range": [0.20, 0.20],
             "missing_target_coupling": beta, "missing_indicators": True},
            max_features=100,
        )
        n_ind = int(meta["missing_indicators"])
        if n_ind == 0:
            continue
        miss = Xn[:, 6:6 + n_ind].numpy().mean(axis=1)  # per-row was-missing fraction
        if task == "pd":
            yv = y.numpy()
            rates = [float(miss[yv == 0].mean()), float(miss[yv == 1].mean())]
            ax.plot([0, 1], rates, "o-", color=color, label=label, markersize=6)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["non-default (y = 0)", "default (y = 1)"])
            ax.set_xlim(-0.25, 1.25)
            ax.set_xlabel("outcome of the row")
        else:
            yv = y.numpy()
            edges = np.quantile(yv, np.linspace(0, 1, 7))
            centers = 0.5 * (edges[:-1] + edges[1:])
            rates = [float(miss[(yv >= lo) & (yv <= hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(centers, rates, "o-", color=color, label=label, markersize=5)
            ax.set_xlabel("LGD outcome")
    ax.set_ylabel("share of values missing")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(bottom=0)
    if task != "pd":
        ax.set_xlabel("LGD outcome (sextile midpoints)")
    style.legend_below(ax, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# Both — what the predictability filter keeps
# ---------------------------------------------------------------------------


def filter_modes(config_path: str, n: int = 120, seed: int = 0):
    """Distribution of ExtraTrees pseudo-R² over generated tasks, with each mode's keep-region.

    `off` keeps everything; `tabicl` keeps the predictable tail; `banded` keeps only the shaded
    low-signal band — the regime real credit data occupies.
    """
    task, prior = _prior(config_path, n_rows=512)
    is_classif = task == "pd"
    tasks = _generate(task, prior, n, seed, {"credit_fraction": 1.0, "filter.mode": "off"})
    r2 = []
    for t in tasks:
        if t.X.shape[1] < 1:
            continue
        _, s = predictability(t.X, t.y, is_classif=is_classif)
        r2.append(float(s))
    r2 = np.clip(np.array(r2), -0.05, 1.0)
    lo, hi = (float(v) for v in prior["filter"]["quantile_band"])

    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.40))
    ax.hist(r2, bins=30, color=style.CREDIT, label="generated tasks")
    ax.axvspan(lo, hi, color=style.CREDIT_MILD, alpha=0.30, zorder=0,
               label=f"band kept by 'banded' [{lo:g}, {hi:g}]")
    ax.set_xlabel("ExtraTrees out-of-bag pseudo-R²")
    ax.set_ylabel("number of tasks")
    style.legend_below(ax, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# Both — how predictable the tasks are, on the filter's own yardstick, against the real data
# ---------------------------------------------------------------------------


def predictability_scores(config_path: str, real: dict[str, Any] | None = None, n: int = 120,
                          seed: int = 0, real_repeats: int = 3) -> pd.DataFrame:
    """Every task's and every real dataset's predictability, measured by the filter itself.

    `src.prior.filters.predictability` is what the predictability filter runs during training: a
    25-tree ExtraTrees with out-of-bag predictions, returning a bootstrap p-value (the `tabicl`
    mode keeps a task at p < 0.05) and a pseudo-R² (the `banded` mode keeps it inside
    `filter.quantile_band`). Scoring synthetic tasks AND real datasets with it puts both on the one
    axis the filter acts on.

    Synthetic tasks are generated with the filter OFF — the population the filter chooses from —
    at the config's table size (1,024 rows). A real dataset is scored on `real_repeats` random
    1,024-row samples, averaged, so it is measured at the size the prior's tables have.

    Returns one row per task or dataset: `source` (`original prior` / `credit prior` / `real`),
    `name`, `pseudo_r2`, `pvalue`.
    """
    cfg = expand_with_seeds(load(config_path))[0]
    task = cfg["task"]
    n_rows = int(cfg["prior"].get("n_rows_range", [1024, 1024])[-1])
    _, prior = _prior(config_path, n_rows=n_rows)
    is_classif = task == "pd"
    # Every item is scored independently and deterministically, so they are collected first
    # and scored in parallel (`parallel_map`); the numbers are those of the sequential loop.
    items: list[tuple[str, str, Any, Any]] = []
    for label, cf in (("original prior", 0.0), ("credit prior", 1.0)):
        for k, t in enumerate(_generate(task, prior, n, seed, {"credit_fraction": cf,
                                                                "filter.mode": "off"})):
            if t.X.shape[1] >= 1:
                items.append((label, f"task {k}", t.X, t.y))
    rng = np.random.default_rng(seed)
    for name, ds in (real or {}).items():
        X = np.nan_to_num(np.asarray(getattr(ds, "X", ds), dtype=np.float32), nan=0.0,
                          posinf=0.0, neginf=0.0)
        y = np.asarray(ds.y, dtype=np.float32).ravel()
        for _ in range(real_repeats):
            idx = rng.choice(len(y), size=min(n_rows, len(y)), replace=False)
            yt = torch.as_tensor(y[idx])
            if is_classif:
                yt = (yt >= 0.5).float()
                if float(yt.std()) == 0.0:
                    continue
            items.append(("real", name.split(".", 1)[-1], torch.as_tensor(X[idx]), yt))
    scored = parallel_map(_score, [(X, y, is_classif) for _, _, X, y in items])

    rows: list[dict[str, Any]] = []
    real_scores: dict[str, list[tuple[float, float]]] = {}
    for (label, name, _, _), (p, s) in zip(items, scored):
        if label == "real":
            real_scores.setdefault(name, []).append((p, s))
        else:
            rows.append({"source": label, "name": name, "pseudo_r2": s, "pvalue": p})
    for name, ps in real_scores.items():
        rows.append({"source": "real", "name": name,
                     "pseudo_r2": float(np.mean([s for _, s in ps])),
                     "pvalue": float(np.mean([p for p, _ in ps]))})
    return pd.DataFrame(rows)


def _score(args: tuple) -> tuple[float, float]:
    """`predictability` for one (X, y, is_classif), as floats — the unit `parallel_map` farms out."""
    X, y, is_classif = args
    p, s = predictability(X, y, is_classif=is_classif)
    return float(p), float(s)


def _band(config_path: str) -> tuple[float, float]:
    cfg = expand_with_seeds(load(config_path))[0]
    lo, hi = cfg["prior"]["filter"]["quantile_band"]
    return float(lo), float(hi)


def plot_predictability(scores: pd.DataFrame, config_path: str):
    """The priors' tasks and the real datasets on the filter's own axis, with what each mode keeps.

    One row per population: the original prior's tasks, our prior's tasks, the real datasets. The
    shaded band is what `banded` keeps; a hollow point is a task the `tabicl` filter rejects
    (bootstrap p ≥ 0.05); the black tick is each row's median. `off` keeps every point.
    """
    lo, hi = _band(config_path)
    style.apply()
    order = [s for s in ("original prior", "credit prior", "real") if (scores["source"] == s).any()]
    fig, ax = plt.subplots(figsize=style.row_figsize(len(order), per_row=0.42, base=1.25))
    ax.axvspan(lo, hi, color=style.CREDIT_MILD, alpha=0.22, zorder=0,
               label=f"kept by 'banded': pseudo-R² in [{lo:g}, {hi:g}]")
    jitter = np.random.default_rng(0)
    for i, source in enumerate(order):
        part = scores[scores["source"] == source]
        x = np.clip(part["pseudo_r2"].to_numpy(), -0.1, 1.0)
        yy = i + (jitter.random(len(x)) - 0.5) * 0.36
        if source == "real":
            ax.scatter(x, yy, marker="*", s=70, color=style.STAR, zorder=4,
                       edgecolors="white", linewidths=0.5, label="one real dataset")
        else:
            colour = style.variant_colour(source.split()[0])
            passed = part["pvalue"].to_numpy() < 0.05
            ax.scatter(x[passed], yy[passed], s=13, color=colour, alpha=0.75, linewidths=0,
                       zorder=3, label="task kept by 'tabicl' (p < 0.05)" if i == 0 else None)
            ax.scatter(x[~passed], yy[~passed], s=13, facecolors="none", edgecolors=colour,
                       linewidths=0.8, zorder=3,
                       label="task rejected by 'tabicl'" if i == 0 else None)
        ax.plot([np.median(x)] * 2, [i - 0.3, i + 0.3], color=style.INK, lw=1.8, zorder=5,
                label="median" if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["real datasets" if s == "real" else s for s in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel("ExtraTrees out-of-bag pseudo-R² (the filter's measure; 0 = no signal)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.legend_below(ax, ncol=2)
    return fig


def predictability_summary(scores: pd.DataFrame, config_path: str) -> str:
    """The predictability figure in numbers: per prior, the median pseudo-R², the share `banded`
    keeps and the share `tabicl` rejects; per real dataset, its pseudo-R²."""
    lo, hi = _band(config_path)
    lines = [f"PREDICTABILITY — the filter's pseudo-R² (banded keeps [{lo:g}, {hi:g}])"]
    for source in ("original prior", "credit prior"):
        part = scores[scores["source"] == source]
        if not len(part):
            continue
        r2 = part["pseudo_r2"].to_numpy()
        lines.append(
            f"  {source:<15} {len(part)} unfiltered tasks | median {np.median(r2):.2f} | "
            f"banded keeps {np.mean((r2 >= lo) & (r2 <= hi)):.0%} | "
            f"tabicl rejects {np.mean(part['pvalue'].to_numpy() >= 0.05):.0%}")
    real = scores[scores["source"] == "real"].sort_values("pseudo_r2")
    if len(real):
        lines.append(f"  real datasets   {len(real)} | median {real['pseudo_r2'].median():.2f} | "
                     f"inside the band {np.mean((real['pseudo_r2'] >= lo) & (real['pseudo_r2'] <= hi)):.0%}")
        lines.append("    " + ", ".join(f"{r.name} {r.pseudo_r2:.2f}" for r in real.itertuples()))
    return "\n".join(lines)
