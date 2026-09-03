"""Visualise the credit-specific MECHANISMS the adjusted prior adds.

`prior_plots` and `pool_plots` show generic prior properties — target shapes, table
sizes, correlation spectra. This module shows the parts that make the prior
*credit-shaped*, the ones the Exp1 redesign is actually about:

* boundary atoms and their INTENSITY (LGD: mild vs aggressive);
* controlled imbalance (PD: the base rate held in credit's measured regime);
* correlated defaults (PD: the Vasicek systematic factor, mild vs aggressive `rho`);
* reject inference (PD: the approved book in context, the rejected region in the query);
* distribution shift (both: context vs query per kind);
* informative missingness (both: a missing rate that depends on the outcome);
* the predictability filter (both: what `tabicl` / `banded` / `off` keep).

Every figure GENERATES LIVE from the current config through `TaskGenerator` — the exact
code path training uses — so it can never show a stale prior the way a pre-generated
pool can. Rows are capped (`n_rows`) purely so a notebook runs in seconds; the
mechanisms are shape-invariant. The notebooks call these and hold no logic of their own.
"""

from __future__ import annotations

import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.prior.filters import predictability
from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG
from src.prior.targets.pd import apply_informative_missingness
from src.utils.config import expand_with_seeds, load
from src.utils.target_stats import target_stats
from src.visualize import style

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
    """`n` tasks generated exactly as training would, after applying dotted-path overrides."""
    p = copy.deepcopy(prior)
    for dotted, value in (overrides or {}).items():
        _set(p, dotted, value)
    gen = TaskGenerator(p, task, PriorRNG(seed))
    return [gen.sample() for _ in range(n)]


def _base_rate(t: Any) -> float:
    return float(t.y.float().mean())


# ---------------------------------------------------------------------------
# LGD — boundary atoms at two intensities
# ---------------------------------------------------------------------------


def intensity_atoms(config_path: str, n: int = 100, seed: int = 0):
    """LGD: the original prior against our prior at mild vs aggressive boundary intensity."""
    task, prior = _prior(config_path)
    if task != "lgd":
        raise ValueError("intensity_atoms is an LGD figure")
    arms = [
        ("original TabICL", style.ORIGINAL, {"credit_fraction": 0.0}),
        ("ours — mild", style.CREDIT_MILD,
         {"credit_fraction": 1.0, "filter.mode": "off", "credit.target.boundary_mass_range": [0.02, 0.30]}),
        ("ours — aggressive", style.CREDIT_STRONG,
         {"credit_fraction": 1.0, "filter.mode": "off", "credit.target.boundary_mass_range": [0.15, 0.60]}),
    ]
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(style.WIDTH_FULL, 0.40), sharey=True)
    for ax, (label, color, ov) in zip(axes, arms):
        tasks = _generate(task, prior, n, seed, ov)
        pooled = np.concatenate([t.y.numpy() for t in tasks])
        ax.hist(pooled, bins=60, color=color)
        stats = [target_stats(t.y) for t in tasks]
        bm = float(np.mean([s["frac_at_min"] + s["frac_at_max"] for s in stats]))
        ax.set_xlabel("target (LGD)")
        style.title(ax, label, f"mean boundary mass {bm:.2f}")
    axes[0].set_ylabel("pooled count")
    fig.suptitle("Boundary atoms by intensity")
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
    ax.set_xlabel("positive (default) rate per task")
    ax.set_ylabel("number of tasks")
    ax.legend(loc="upper right")
    style.title(ax, "Controlled imbalance",
                "our prior concentrates the base rate where real books sit")
    fig.suptitle("PD base rate: upstream vs ours")
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
    bins = np.linspace(0, 0.5, 41)
    ax1.hist(r_mild, bins=bins, color=style.CREDIT_MILD, alpha=0.8,
             label=f"mild ρ  (SD {r_mild.std():.3f})")
    ax1.hist(r_aggr, bins=bins, color=style.CREDIT_STRONG, alpha=0.8,
             label=f"aggressive ρ  (SD {r_aggr.std():.3f})")
    ax1.axvline(0.15, color=style.INK, ls="--", lw=1.2)
    ax1.annotate("15% target", (0.15, ax1.get_ylim()[1] * 0.95), color=style.INK, fontsize=8,
                 xytext=(5, 0), textcoords="offset points")
    ax1.set_xlabel("realised default rate")
    ax1.set_ylabel("number of tasks")
    ax1.legend(loc="upper right")
    style.title(ax1, "Realised rate around a fixed target")

    rhos = [0.03, 0.15, 0.30]
    sds = []
    for r in rhos:
        ts = _generate(task, prior, max(30, n // 3), seed,
                       {**fixed, "credit.target.mechanism.rho_range": [r, r]})
        sds.append(float(np.std([_base_rate(t) for t in ts])))
    ax2.plot(rhos, sds, "o-", color=style.CREDIT)
    ax2.set_xlabel("asset correlation ρ")
    ax2.set_ylabel("SD of realised default rate")
    style.title(ax2, "Default clustering grows with ρ")
    fig.suptitle("Correlated defaults by ρ")
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
    ax1.scatter(ctx, qry, s=28, alpha=0.5, color=style.CREDIT, edgecolor="white", lw=0.4, zorder=3)
    lim = float(max(ctx.max(), qry.max())) * 1.1 + 1e-3
    ax1.plot([0, lim], [0, lim], color=style.MUTED, ls="--", lw=1.0, zorder=1)
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("context (approved book) default rate")
    ax1.set_ylabel("query (through-the-door) default rate")
    style.title(ax1, "Query is the riskier book", "every task sits above the diagonal")

    gap = qry - ctx
    ax2.hist(gap, bins=20, color=style.QUERY)
    ax2.axvline(0, color=style.INK, lw=1.0)
    ax2.axvline(float(gap.mean()), color=style.WARN, ls="--", lw=1.4)
    ax2.set_xlabel("query minus context default rate")
    ax2.set_ylabel("number of tasks")
    style.title(ax2, f"Approved book understates risk by {gap.mean():+.2f}")
    fig.suptitle("Reject inference: context vs query")
    return fig


# ---------------------------------------------------------------------------
# Both — distribution shift, context vs query per kind
# ---------------------------------------------------------------------------


def shift_kinds(config_path: str, n: int = 80, seed: int = 0):
    """Context (given) vs query (predicted), one panel per shift kind. `selection` is PD-only."""
    task, prior = _prior(config_path)
    kinds = ["cohort", "covariate", "prior_prob"] + (["selection"] if task == "pd" else [])
    style.apply()
    fig, axes = plt.subplots(1, len(kinds), figsize=style.figsize(style.WIDTH_FULL, 0.36))
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        tasks = _generate(task, prior, n, seed, {
            "credit_fraction": 1.0, "filter.mode": "off",
            "credit.shift.shift_prob": 1.0,
            "credit.shift.kind_weights": {kind: 1.0},
        })
        sel = [t for t in tasks if t.meta.get("shift") == kind]
        if kind == "covariate":
            # Covariate shift moves the FEATURES, not the target; show the shifted feature.
            ctx, qry = [], []
            for t in sel:
                cut, j = t.meta["shift_cut"], t.meta["shift_feature"]
                col = t.X.numpy()[:, j]
                ctx.append(col[:cut]); qry.append(col[cut:])
            ctx = np.concatenate(ctx) if ctx else np.array([0.0])
            qry = np.concatenate(qry) if qry else np.array([0.0])
            lo, hi = np.percentile(np.concatenate([ctx, qry]), [1, 99])
            bins = np.linspace(lo, hi, 30)
            ax.hist(ctx, bins=bins, color=style.CONTEXT, alpha=0.7, density=True)
            ax.hist(qry, bins=bins, color=style.QUERY, alpha=0.7, density=True)
            ax.set_xlabel("shifted feature")
            ax.set_yticks([])
            style.title(ax, "covariate", "feature range moves")
            continue
        ctx = np.concatenate([t.y.numpy()[:t.meta["shift_cut"]] for t in sel]) if sel else np.array([0.0])
        qry = np.concatenate([t.y.numpy()[t.meta["shift_cut"]:] for t in sel]) if sel else np.array([0.0])
        if task == "pd":
            ax.bar([0, 1], [float(ctx.mean()), float(qry.mean())],
                   color=[style.CONTEXT, style.QUERY], width=0.7)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["context", "query"])
            ax.set_ylabel("default rate")
            style.title(ax, kind)
        else:
            bins = np.linspace(0, 1, 30)
            ax.hist(ctx, bins=bins, color=style.CONTEXT, alpha=0.7, density=True)
            ax.hist(qry, bins=bins, color=style.QUERY, alpha=0.7, density=True)
            ax.set_xlabel("target")
            ax.set_yticks([])
            style.title(ax, kind)
    # One shared legend via proxies (bars/hists carry no label).
    axes[-1].legend(handles=style.legend_patches({"context": style.CONTEXT, "query": style.QUERY}),
                    loc="upper right")
    fig.suptitle(f"Distribution shift, {task.upper()}: context vs query")
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
    for beta, color, label in [(0.0, style.MUTED, "MCAR — coupling 0"),
                               (2.0, style.CREDIT, "MNAR — coupling 2")]:
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
            ax.set_xticks([0, 1]); ax.set_xticklabels(["non-default (y=0)", "default (y=1)"])
        else:
            yv = y.numpy()
            edges = np.quantile(yv, np.linspace(0, 1, 7))
            centers = 0.5 * (edges[:-1] + edges[1:])
            rates = [float(miss[(yv >= lo) & (yv <= hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(centers, rates, "o-", color=color, label=label, markersize=5)
            ax.set_xlabel("LGD outcome")
    ax.set_ylabel("missing rate")
    ax.legend(loc="upper left")
    style.title(ax, "Informative missingness (MNAR)",
                "under coupling the missing rate rises with the outcome; under MCAR it is flat")
    fig.suptitle(f"Informative missingness ({task.upper()})")
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
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    ax.hist(r2, bins=30, color=style.CREDIT)
    ax.axvspan(lo, hi, color=style.CREDIT_MILD, alpha=0.30, zorder=0)
    ax.annotate(f"banded keeps\n[{lo:g}, {hi:g}]", (0.5 * (lo + hi), ax.get_ylim()[1] * 0.88),
                color=style.CREDIT_STRONG, fontsize=8, ha="center", weight="semibold")
    ax.set_xlabel("ExtraTrees pseudo-R²  (0 = unpredictable, 1 = trivially easy)")
    ax.set_ylabel("number of tasks")
    frac_band = float(((r2 >= lo) & (r2 <= hi)).mean())
    style.title(ax, "What each filter keeps",
                f"off: all · tabicl: the predictable tail · banded: the {frac_band:.0%} in the band")
    fig.suptitle(f"Predictability filter, {task.upper()}")
    return fig
