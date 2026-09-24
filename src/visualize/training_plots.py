"""Level-1 TRAINING visualisation: what every arm of a sweep did while it trained.

Reads what each arm writes to `output/manifests/` — `<run>__progress.csv` (train loss and the
development-split monitoring metrics, real-credit and out-of-domain, sampled every
`progress.every_datasets`), `<run>__telemetry.csv` (throughput, GPU, per-block gradients) and
`<run>__summary.json` — and turns it into the figures of `1.1_pd_training` / `1.2_lgd_training`
(Exp1, `exp="exp1"`) and `2.1_pd_finetuning` / `2.2_lgd_finetuning` (Exp2, `exp="exp2"`).

TWO RULES DECIDE WHAT A CURVE MAY AVERAGE OVER. Both exist because the real sweep broke the naive
version, and both are stated in each notebook so a reader knows what a line is an average of.

* **Finished arms only.** Averaging every arm at every step changes the composition of the average
  wherever an unfinished arm stops contributing, so the curve JUMPS near the end for no reason but
  bookkeeping — the banded and credit curves both did. Aggregates use the arms whose summary says
  `completed: true` (the flag `sweep_status` resubmits on), and fall back to every arm while none
  has finished. Unfinished arms stay visible: in the sweep map, faint in the curves, and per arm.
* **Development datasets only, the same ones for every arm.** A prior is chosen on the development
  split; the holdout datasets are what the benchmark finally reports. But curves from before the
  development-only monitoring protocol (23-09-2026) scored "the smallest few" datasets whatever
  their role — for Exp1 that put holdout sets (PD `hmeq`, `thomas`; LGD `axa`, `loss2`,
  `base_modelisation`) into every old arm's mean. So a domain mean covers only the config's
  development datasets that EVERY arm carries (PD: `german`, `myhom`), never a holdout one. A
  missing value is still never skipped (`skipna=False`): an arm whose development monitor is
  missing (LGD `base_model`, all-NaN in 30 of 45 arms) drops out of an average and is COUNTED in the
  printed summary, rather than being averaged over fewer datasets. `monitoring_coverage` draws the
  whole situation, arm by arm.

The experiments differ only in which manifests are read (the `exp1_`/`exp2_` prefix) and which
levers name an arm — the prior's for Exp1, fine-tuning's for Exp2 — so one module serves both.
Everything works on a partial sweep. The notebooks call these and hold no logic of their own.
"""

from __future__ import annotations

import json
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import paths
from src.visualize import style

# Headline evaluation metric per task: the one curve a reader looks at first. Higher is better
# for both (ROC-AUC for classification, R² for regression), so "best/worst" is unambiguous.
HEADLINE = {"pd": "roc_auc", "lgd": "r2"}
HIGHER_IS_BETTER = {"auc": True, "ap": True, "r2": True, "spearman": True, "kendall": True,
                    "roc_auc": True, "pr_auc": True,
                    "brier": False, "logloss": False, "rmse": False, "mae": False,
                    "pinball": False, "crps": False, "ks": True, "calibration_slope": True}

#: The levers that name an arm, per experiment, and how each reads in a heading. Credit fraction
#: comes first in both: it is the lever the sweep's rows are grouped by.
LEVERS = {
    "exp1": (("credit_fraction", "credit fraction"), ("filter", "filter mode"),
             ("intensity", "prior intensity")),
    "exp2": (("credit_fraction", "credit fraction"), ("strategy", "freeze strategy"),
             ("l2sp", "L2-SP"), ("lr", "learning rate")),
}


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


def _cf_value(run_name: str) -> float:
    """The arm's credit fraction as a number (0 for an arm that does not name one)."""
    m = re.search(r"credit_fraction=([0-9p.]+)", run_name)
    return float(m.group(1).replace("p", ".")) if m else 0.0


def _config_of(run_name: str) -> str:
    """The run name without its seed: one configuration, whose seeds are its repeats."""
    return re.sub(r"__s\d+$", "", run_name)


def _seed_of(run_name: str) -> int | None:
    m = re.search(r"__s(\d+)$", run_name)
    return int(m.group(1)) if m else None


def _config_label(run_name: str) -> str:
    """`arm_label` without the seed — the label of a configuration."""
    return re.sub(r"·s\d+$", "", arm_label(run_name))


#: The order configurations are listed in, everywhere a figure has one row per configuration:
#: credit fraction first (the lever that dominates, so its effect reads as a block), then the
#: experiment's other levers.
_ORDER = {"intensity": ("mild", "aggressive"), "filter": ("tabicl", "banded", "off"),
          "strategy": ("full", "icl_only", "head_only", "scratch"), "l2sp": ("L2-SP off", "L2-SP on")}


def _config_sort_key(run_name: str, exp: str = "exp1") -> tuple:
    key: list[Any] = [_cf_value(run_name)]
    for lever, _label in LEVERS.get(exp, LEVERS["exp1"])[1:]:
        value = _lever_value(run_name, lever)
        if lever == "intensity" and _is_control(run_name):
            value = None  # the control has no credit prior, so no intensity either
        order = _ORDER.get(lever)
        key.append(order.index(value) if order and value in order else str(value or ""))
    return tuple(key)


# ---------------------------------------------------------------------------
# What a curve may average over: finished arms, and the datasets every arm can score
# ---------------------------------------------------------------------------


def _target_step(runs: dict[str, pd.DataFrame]) -> int:
    """The step the sweep trains to — the furthest any arm got, since all share `max_steps`."""
    return max((int(df["step"].max()) for df in runs.values() if len(df)), default=0)


def completed_arms(runs: dict[str, pd.DataFrame]) -> set[str]:
    """The arms that finished training.

    `completed: true` in the arm's summary — the flag `sweep_status` resubmits on — which a
    requeued arm writes as `false` each time a walltime stops it. Where no arm has a summary at all
    (a local smoke run), the arms that reached the furthest step any arm reached.
    """
    done: set[str] = set()
    any_summary = False
    for name in runs:
        path = paths.run_summary_path(name)
        if not path.is_file():
            continue
        any_summary = True
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("completed"):
                done.add(name)
        except (OSError, ValueError):
            continue
    if any_summary:
        return done
    target = _target_step(runs)
    return {n for n, df in runs.items() if len(df) and int(df["step"].max()) >= target}


def _aggregate(runs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The arms a curve averages over: the finished ones, or every arm while none has finished."""
    done = completed_arms(runs)
    return {n: df for n, df in runs.items() if n in done} or runs


def _metric_cols(df: pd.DataFrame, metric: str, domain: str = "real") -> list[str]:
    """Columns like `real__<dataset>__<metric>` for one metric and domain."""
    cols = [c for c in df.columns if c.startswith(f"{domain}__") and c.endswith(f"__{metric}")]
    alias = {"roc_auc": "auc", "pr_auc": "ap", "auc": "roc_auc", "ap": "pr_auc"}.get(metric)
    if not cols and alias:
        cols = [c for c in df.columns if c.startswith(f"{domain}__") and c.endswith(f"__{alias}")]
    return cols


def _dataset_cols(df: pd.DataFrame, metric: str, domain: str = "real") -> dict[str, str]:
    """`{dataset: column}` for one arm and metric, alias-resolved (older CSVs say `auc`)."""
    return {c[len(domain) + 2: c.rindex("__")]: c for c in _metric_cols(df, metric, domain)}


def _track_exp_of(runs: dict[str, pd.DataFrame]) -> tuple[str, str]:
    """`(track, exp)` read off the run names (`exp1_pd__...`), so no caller has to pass them."""
    m = re.match(r"(exp\d+)_(pd|lgd)__", next(iter(runs), ""))
    return (m.group(2), m.group(1)) if m else ("pd", "exp1")


def dataset_roles(track: str, exp: str = "exp1") -> dict[str, str]:
    """`{dataset: "development" | "holdout"}` from the experiment's config, numeric prefixes
    stripped (`0008.german` -> `german`); `{}` when the config cannot be read.

    The roles are the experiment's selection protocol: a prior is chosen on development data only,
    and the holdout datasets are what the benchmark finally reports on. A figure that averages a
    holdout dataset into "which prior is best" is selecting on the test set.
    """
    try:
        from src.utils.config import load

        cfg = load(paths.REPO_ROOT / f"config/Exp{int(exp.removeprefix('exp'))}_{track.upper()}.yaml",
                   allow_placeholders=True)
    except Exception:  # noqa: BLE001 — without a config, fall back to "every dataset is usable"
        return {}
    ev = cfg.get("eval") or {}
    roles = {str(d).split(".", 1)[-1]: "holdout" for d in ev.get("holdout_datasets") or []}
    roles.update({str(d).split(".", 1)[-1]: "development" for d in ev.get("dev_datasets") or []})
    return roles


def comparable_datasets(runs: dict[str, pd.DataFrame], metric: str,
                        domain: str = "real") -> tuple[list[str], list[str]]:
    """`(kept, excluded)`: the datasets a domain mean is taken over, and the monitored ones it is not.

    For the real domain, `kept` is the experiment's DEVELOPMENT datasets whose column every arm
    carries — a holdout dataset is never averaged, whatever an arm happened to monitor (curves from
    before the development-only protocol scored "the smallest few" datasets, holdout ones among
    them). Out-of-domain suites have no roles: every suite every arm carries. An arm whose value is
    missing for a kept dataset gets a missing mean and drops out of an average — counted by the
    caller, never averaged over fewer datasets.
    """
    if not runs:
        return [], []
    per_arm = [_dataset_cols(df, metric, domain) for df in runs.values()]
    every = sorted(set().union(*per_arm))
    carried = [d for d in every if all(d in dc for dc in per_arm)]
    if domain == "real":
        roles = dataset_roles(*_track_exp_of(runs))
        if roles:
            carried = [d for d in carried if roles.get(d) == "development"]
    kept = [d for d in carried
            if any(df[dc[d]].notna().any() for dc, df in zip(per_arm, runs.values()))]
    return kept, [d for d in every if d not in kept]


def _mean_metric(df: pd.DataFrame, metric: str, domain: str = "real",
                 datasets: list[str] | None = None) -> pd.Series:
    """The domain mean per step. A missing value is never skipped — it stays visible — so pass
    `datasets` (from `comparable_datasets`) to make every arm's mean cover the same set."""
    dc = _dataset_cols(df, metric, domain)
    if datasets is None:
        cols = list(dc.values())
    elif any(d not in dc for d in datasets):
        return pd.Series(np.nan, index=df.index)  # cannot cover the set: missing, not smaller
    else:
        cols = [dc[d] for d in datasets]
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
    priority = ["roc_auc", "auc", "pr_auc", "ap", "r2", "rmse", "mae", "pinball", "crps", "brier",
                "logloss", "ks", "calibration_slope", "spearman", "kendall",
                "boundary_mass_abs_err", "coverage_80", "pit_mean"]
    ordered = [m for m in priority if m in seen] + sorted(seen - set(priority))
    # Drop bookkeeping / non-metric columns.
    return [m for m in ordered if m not in ("n_test", "n_boundary", "n_interior")]


def _final(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _finals(arms: dict[str, pd.DataFrame], metric: str, datasets: list[str],
            domain: str = "real") -> dict[str, float]:
    """Each arm's last logged domain mean over `datasets`; arms with none are left out."""
    out = {n: _final(_mean_metric(df, metric, domain, datasets)) for n, df in arms.items()}
    return {n: v for n, v in out.items() if np.isfinite(v)}


def rank_arms(runs: dict[str, pd.DataFrame], track: str) -> list[tuple[str, float]]:
    """Finished arms sorted best -> worst by the final headline metric, over the datasets every
    arm can score."""
    metric = HEADLINE[track]
    kept, _ = comparable_datasets(runs, metric)
    scored = _finals(_aggregate(runs), metric, kept)
    return sorted(scored.items(), key=lambda kv: kv[1], reverse=HIGHER_IS_BETTER.get(metric, True))


def _curve(df: pd.DataFrame, metric: str, datasets: list[str], grid: np.ndarray,
           domain: str = "real") -> np.ndarray | None:
    """One arm's domain mean, interpolated onto `grid` (NaN outside the steps it logged)."""
    d = pd.DataFrame({"step": df["step"], "m": _mean_metric(df, metric, domain, datasets)}).dropna()
    if len(d) < 2:
        return None
    return np.interp(grid, d["step"], d["m"], left=np.nan, right=np.nan)


def _view(track: str, exp: str, metric: str, domain: str = "real"):
    """`(all arms, the finished arms, datasets kept, datasets dropped)` — what every figure needs."""
    runs = load_progress(track, exp)
    kept, dropped = comparable_datasets(runs, metric, domain)
    return runs, _aggregate(runs), kept, dropped


def _sweep_grid(track: str, exp: str) -> list[str]:
    """Every arm the experiment's config schedules, as run names in array order — `[]` when the
    config cannot be expanded (a template with placeholders), so the arms on disk are used."""
    try:
        from src.utils.config import expand_with_seeds, load

        cfg = load(paths.REPO_ROOT / f"config/Exp{int(exp.removeprefix('exp'))}_{track.upper()}.yaml",
                   allow_placeholders=True)
        return [run["_run_name"] for run in expand_with_seeds(cfg)]
    except Exception:  # noqa: BLE001 — a figure degrades to what is on disk rather than raising
        return []


def _cf_legend(ax: Any, fractions, *, loc: str = "lower right", title: str | None = None) -> None:
    handles = style.legend_patches({f"cf = {f:g}": style.credit_fraction_colour(f)
                                    for f in sorted(set(fractions))})
    ax.legend(handles=handles, loc=loc, fontsize=7, title=title, title_fontsize=7)


# ---------------------------------------------------------------------------
# A. The run — is the experiment sound?
# ---------------------------------------------------------------------------


def sweep_map(track: str, exp: str = "exp1", metric: str | None = None):
    """Every arm of the sweep in one grid: one row per configuration, one column per seed.

    Each cell is that arm's final development score on one shared colour scale. A hatched cell is
    still training (the share says how far it got); a grey one never started. Rows are grouped by
    credit fraction, white rules between the groups, so that lever's effect reads as a block.
    """
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    names = _sweep_grid(track, exp) or sorted(runs)
    if not names:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no arms scheduled or started yet")
        return fig
    configs = sorted({_config_of(n) for n in names}, key=lambda c: _config_sort_key(c, exp))
    seeds = sorted({s for s in (_seed_of(n) for n in names) if s is not None})
    kept, _ = comparable_datasets(runs, metric)
    done = completed_arms(runs)
    target = _target_step(runs)
    values = np.full((len(configs), len(seeds)), np.nan)
    status: dict[tuple[int, int], str] = {}
    share: dict[tuple[int, int], float] = {}
    for i, config in enumerate(configs):
        for j, seed in enumerate(seeds):
            name = f"{config}__s{seed}"
            if name not in runs:
                status[i, j] = "todo"
                continue
            values[i, j] = _final(_mean_metric(runs[name], metric, "real", kept))
            status[i, j] = "done" if name in done else "partial"
            share[i, j] = int(runs[name]["step"].max()) / target if target else 0.0

    fig, ax = plt.subplots(figsize=style.row_figsize(len(configs), per_row=0.22, base=1.25))
    finite = values[np.isfinite(values)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    cmap = plt.get_cmap(style.CMAP_SEQ).with_extremes(bad=style.GRID)
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
    for (i, j), state in status.items():
        v = values[i, j]
        if state == "todo":
            ax.text(j, i, "not started", ha="center", va="center", fontsize=6, color=style.MUTED)
            continue
        dark = np.isfinite(v) and (v - lo) / (hi - lo + 1e-12) < 0.55
        text = f"{v:.3f}" if np.isfinite(v) else "no development monitor"
        if state == "partial":
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                       edgecolor=style.WARN, linewidth=0.0))
            text += f"   ({share[i, j]:.0%} trained)"
        ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                color="white" if dark else style.INK)
    fractions = [_cf_value(c) for c in configs]
    for i in range(1, len(configs)):
        if fractions[i] != fractions[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=2.4)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([_config_label(c) for c in configs], fontsize=7)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.grid(False)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02,
                 label=f"final development {style.metric_label(metric)}")
    n_done = sum(state == "done" for state in status.values())
    style.title(ax, f"{n_done} of {len(status)} arms finished",
                "hatched: still training · grey: not started")
    fig.suptitle(f"{track.upper()} sweep: every arm")
    return fig


#: What a (dataset, arm) cell of the monitoring map contributes, most useful first. Teal is the
#: palette's "grounded" colour: only those cells enter a development mean.
_COVERAGE = (("averaged: development, carried by every arm", style.REFERENCE),
             ("development, but not carried by every arm", style.CREDIT_MILD),
             ("holdout: scored, never averaged", style.MUTED),
             ("monitored, but every value missing", style.WARN),
             ("not monitored", style.GRID))


def monitoring_coverage(track: str, exp: str = "exp1", metric: str | None = None):
    """Which datasets each arm was scored on during training, and what each score is allowed to do.

    One row per dataset (development first, then holdout), one column per arm in sweep-map order.
    Only the teal cells enter a development mean: a holdout dataset an arm happened to monitor is
    drawn but never averaged, and a red row is a monitor that logged nothing but missing values.
    """
    metric = metric or HEADLINE[track]
    runs = load_progress(track, exp)
    style.apply()
    if not runs:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no progress CSVs found in output/manifests/")
        return fig
    roles = dataset_roles(track, exp)
    kept, _ = comparable_datasets(runs, metric)
    names = sorted(runs, key=lambda n: (_config_sort_key(n, exp), _seed_of(n) or 0))
    cols = {n: _dataset_cols(runs[n], metric) for n in names}
    monitored = set().union(*cols.values())
    rank = {"development": 0, "holdout": 1}
    datasets = sorted(monitored | {d for d, r in roles.items() if r == "development"},
                      key=lambda d: (rank.get(roles.get(d, ""), 2), d))
    codes = np.full((len(datasets), len(names)), 4)
    for j, name in enumerate(names):
        for i, ds in enumerate(datasets):
            col = cols[name].get(ds)
            if col is None:
                continue
            if runs[name][col].isna().all():
                codes[i, j] = 3
            elif ds in kept:
                codes[i, j] = 0
            elif roles.get(ds) == "holdout":
                codes[i, j] = 2
            else:
                codes[i, j] = 1
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=style.row_figsize(len(datasets), per_row=0.27, base=1.7))
    ax.imshow(codes, aspect="auto", cmap=ListedColormap([c for _, c in _COVERAGE]),
              vmin=-0.5, vmax=len(_COVERAGE) - 0.5, interpolation="nearest")
    fractions = [_cf_value(n) for n in names]
    starts = [0] + [j for j in range(1, len(names)) if fractions[j] != fractions[j - 1]]
    for j in starts[1:]:
        ax.axvline(j - 0.5, color="white", lw=2.4)
    # The credit-fraction groups label the x axis — below the map, where nothing competes with them.
    ends = starts[1:] + [len(names)]
    ax.set_xticks([(a + b - 1) / 2 for a, b in zip(starts, ends)])
    ax.set_xticklabels([f"cf = {fractions[a]:g}  ({b - a} arms)" for a, b in zip(starts, ends)],
                       fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([f"{d} · {roles.get(d, 'no role')}" for d in datasets], fontsize=7)
    ax.set_xlabel("arms in sweep-map order, grouped by credit fraction")
    ax.grid(False)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.legend(handles=style.legend_patches(dict(_COVERAGE)), loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7, frameon=False)
    style.title(ax, "Datasets each arm was scored on", "only teal cells enter a development mean")
    fig.suptitle(f"{track.upper()} monitoring coverage")
    return fig


def throughput(track: str, exp: str = "exp1"):
    """What each arm cost: its training speed, grouped by the lever that sets it, and the wall-clock
    time that speed implies for the whole run.

    Prior generation runs on the CPU, so a filter that rejects most candidate tasks starves the GPU
    — which shows here and nowhere in the loss. Each point is one arm (its median steps per second
    over the telemetry), coloured by credit fraction; the bar is the group median.
    """
    tel = load_telemetry(track, exp)
    runs = load_progress(track, exp)
    lever, lever_label = ("filter", "filter mode") if exp == "exp1" else ("strategy", "freeze strategy")
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.44))
    speed: dict[str, float] = {}
    for name, df in tel.items():
        if "steps_per_s" not in df:
            continue
        s = pd.to_numeric(df["steps_per_s"], errors="coerce")
        s = s[np.isfinite(s) & (s > 0)]
        if len(s):
            speed[name] = float(s.median())
    groups = sorted({_lever_value(n, lever) for n in speed} - {None})
    if not groups:
        _empty(ax1, "no throughput in the telemetry CSVs yet")
        ax2.axis("off")
        return fig
    target = _target_step(runs) or 12500
    jitter = np.random.default_rng(0)
    for i, group in enumerate(groups):
        members = [n for n in speed if _lever_value(n, lever) == group]
        steps = np.array([speed[n] for n in members])
        hours = target / steps / 3600.0
        colours = [style.credit_fraction_colour(_cf_value(n)) for n in members]
        xs = i + (jitter.random(len(members)) - 0.5) * 0.24
        for ax, vals in ((ax1, steps), (ax2, hours)):
            ax.scatter(xs, vals, s=18, c=colours, edgecolor="white", linewidth=0.3, zorder=3)
            ax.plot([i - 0.26, i + 0.26], [np.median(vals)] * 2, color=style.INK, lw=1.8, zorder=4)
        style.annotate_value(ax2, i, float(np.median(hours)), f"{np.median(hours):.0f} h")
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_xlabel(lever_label)
    ax1.set_ylabel("training steps per second")
    ax2.set_ylabel(f"implied hours for {target:,} steps")
    style.title(ax1, "Training speed per arm")
    style.title(ax2, "Implied time for the full run")
    _cf_legend(ax1, [_cf_value(n) for n in speed], loc="upper right")
    fig.suptitle(f"{track.upper()} cost of each arm")
    return fig


def training_loss(track: str, exp: str = "exp1"):
    """Train loss vs step: every arm faint, the mean of the finished arms bold, best and worst
    highlighted."""
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found in output/manifests/")
        return fig

    grid = _common_step_grid(runs)
    stack = []
    for name, df in runs.items():
        d = df.dropna(subset=["train_loss"])
        if len(d) < 2:
            continue
        ax.plot(d["step"], d["train_loss"], color=style.MUTED, alpha=0.28, lw=0.8, zorder=1)
        if name in agg:
            stack.append(np.interp(grid, d["step"], d["train_loss"], left=np.nan, right=np.nan))
    if stack:
        ax.plot(grid, _nanmean(stack), color=style.INK, lw=2.2,
                label=f"mean of {len(stack)} finished arms", zorder=4)
    for name, colour, tag in _best_worst_pair(rank_arms(runs, track)):
        d = runs[name].dropna(subset=["train_loss"])
        ax.plot(d["step"], d["train_loss"], color=colour, lw=1.8, zorder=5,
                label=f"{tag}: {arm_label(name)}")
    ax.set_xlabel("training step")
    ax.set_ylabel("train loss")
    ax.legend(loc="upper right")
    style.title(ax, f"Training loss, {len(runs)} arms", "best and worst by final development score")
    fig.suptitle(f"{track.upper()} training loss")
    return fig


# ---------------------------------------------------------------------------
# B. The answer — development monitoring
# ---------------------------------------------------------------------------


def metric_over_training(track: str, exp: str = "exp1", metric: str | None = None):
    """The development metric vs step, averaged over the datasets every arm can score: each arm
    faint in its credit-fraction colour (dotted while unfinished), the finished arms' mean bold."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found")
        return fig
    grid = _common_step_grid(runs)
    stack = []
    for name, df in runs.items():
        curve = _curve(df, metric, kept, grid)
        if curve is None:
            continue
        finished = name in agg
        ax.plot(grid, curve, color=style.credit_fraction_colour(_cf_value(name)),
                alpha=0.35 if finished else 0.25, lw=0.9, ls="-" if finished else ":", zorder=1)
        if finished:
            stack.append(curve)
    if stack:
        ax.plot(grid, _nanmean(stack), color=style.INK, lw=2.2,
                label=f"mean of {len(stack)} finished arms", zorder=4)
    for f in sorted({_cf_value(n) for n in runs}):
        ax.plot([], [], color=style.credit_fraction_colour(f), lw=1.6, label=f"cf = {f:g}")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    ax.legend(loc="lower right")
    style.title(ax, f"{style.metric_label(metric)} during training",
                f"mean over {len(kept)} development datasets, per arm")
    fig.suptitle(f"{track.upper()} development score over training")
    return fig


def credit_vs_control_over_training(track: str, exp: str = "exp1", metric: str | None = None):
    """The headline comparison as a curve: credit-prior arms vs control arms, median and IQR over
    the finished arms of each group."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found in output/manifests/")
        return fig
    grid = _common_step_grid(runs)
    groups: dict[str, list] = {"credit prior": [], "control (cf = 0)": []}
    for name, df in agg.items():
        curve = _curve(df, metric, kept, grid)
        if curve is not None:
            groups["control (cf = 0)" if _is_control(name) else "credit prior"].append(curve)
    for label, colour in (("credit prior", style.CREDIT), ("control (cf = 0)", style.ORIGINAL)):
        stack = groups[label]
        if not stack:
            continue
        arr = np.array(stack)
        med = _quietly(np.nanmedian, arr, axis=0)
        lo, hi = _quietly(np.nanpercentile, arr, [25, 75], axis=0)
        ax.fill_between(grid, lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot(grid, med, color=colour, lw=2.3, label=f"{label} (n = {len(stack)})", zorder=4)
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    ax.legend(loc="lower right")
    style.title(ax, "Credit prior vs control", "median over finished arms, interquartile range shaded")
    fig.suptitle(f"{track.upper()} credit prior versus control")
    return fig


def metric_by_lever(track: str, lever: str, exp: str = "exp1", metric: str | None = None):
    """The development metric over training, one mean line per value of a single swept lever
    (finished arms). Credit fraction is drawn in its own grey-to-blue scale."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found")
        return fig
    grid = _common_step_grid(runs)
    by_val: dict[str, list] = {}
    for name, df in agg.items():
        v = _lever_value(name, lever)
        if v is None or (lever == "intensity" and _is_control(name)):
            continue
        curve = _curve(df, metric, kept, grid)
        if curve is not None:
            by_val.setdefault(v, []).append(curve)
    if not by_val:
        _empty(ax, f"no arms carry the '{lever}' lever")
        return fig
    order = _ORDER.get(lever)
    values = sorted(by_val, key=lambda v: (order.index(v) if order and v in order else 99, v))
    for i, v in enumerate(values):
        colour = (style.credit_fraction_colour(float(v.split("=")[1])) if lever == "credit_fraction"
                  else style.SERIES[i % len(style.SERIES)])
        ax.plot(grid, _nanmean(by_val[v]), color=colour, lw=2.0,
                label=f"{v}  (n = {len(by_val[v])})")
    label = dict(LEVERS.get(exp, LEVERS["exp1"])).get(lever, lever.replace("_", " "))
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    ax.legend(loc="lower right", title=label)
    style.title(ax, f"{style.metric_label(metric)} by {label}", "mean over finished arms")
    fig.suptitle(f"{track.upper()} score by {label}")
    return fig


def final_metric_by_lever(track: str, exp: str = "exp1", metric: str | None = None):
    """Final development score of every finished arm, grouped by each swept lever in turn; points
    coloured by credit fraction, so a lever that only moves with the fraction is visible as such."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    levers = LEVERS.get(exp, LEVERS["exp1"])
    ncols = 2
    nrows = int(np.ceil(len(levers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.72))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    finals = _finals(agg, metric, kept)
    if not finals:
        _empty(axes[0], "no finished arms to summarise yet")
        return fig
    jitter = np.random.default_rng(0)
    for ax, (lever, label) in zip(axes, levers):
        by: dict[str, list] = {}
        for name, v in finals.items():
            lv = _lever_value(name, lever)
            if lv is not None and not (lever == "intensity" and _is_control(name)):
                by.setdefault(lv, []).append((name, v))
        if not by:
            continue
        ax.axis("on")
        order = _ORDER.get(lever)
        xs = sorted(by, key=lambda v: (order.index(v) if order and v in order else 99, v))
        for i, x in enumerate(xs):
            vals = np.array([v for _, v in by[x]])
            colours = [style.credit_fraction_colour(_cf_value(n)) for n, _ in by[x]]
            ax.scatter(i + (jitter.random(len(vals)) - 0.5) * 0.18, vals, s=20, c=colours,
                       alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
            ax.plot([i - 0.24, i + 0.24], [vals.mean()] * 2, color=style.INK, lw=1.8, zorder=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(style.metric_label(metric), fontsize=8)
        style.title(ax, label)
    _cf_legend(axes[0], [_cf_value(n) for n in finals])
    fig.suptitle(f"{track.upper()} final score by lever")
    return fig


def lever_interaction(track: str, exp: str = "exp1", metric: str | None = None):
    """Each lever other than credit fraction, drawn WITHIN each credit fraction.

    The sweep is factorial, so a lever pooled across fractions inherits the fraction's (much larger)
    effect as spread. Here each fraction is its own series: points are finished arms, the line joins
    the per-value means, and the whiskers are ±1 SD over seeds. A lever the control does not have
    (prior intensity) shows the control as a grey band instead.
    """
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    levers = LEVERS.get(exp, LEVERS["exp1"])[1:]
    fig, axes = plt.subplots(1, len(levers), figsize=style.figsize(style.WIDTH_FULL, 0.46),
                             sharey=True)
    axes = np.atleast_1d(axes)
    finals = _finals(agg, metric, kept)
    if not finals:
        for ax in axes[1:]:
            ax.axis("off")
        _empty(axes[0], "no finished arms to compare yet")
        return fig
    fractions = sorted({_cf_value(n) for n in finals})
    for ax, (lever, label) in zip(axes, levers):
        control = [v for n, v in finals.items() if _is_control(n)]
        control_has_lever = len({_lever_value(n, lever) for n in finals if _is_control(n)}) > 1
        order = _ORDER.get(lever)
        values = sorted({_lever_value(n, lever) for n in finals} - {None},
                        key=lambda v: (order.index(v) if order and v in order else 99, v))
        if not control_has_lever and control:
            m, s = float(np.mean(control)), float(np.std(control))
            ax.axhspan(m - s, m + s, color=style.ORIGINAL, alpha=0.22, lw=0, zorder=0)
            ax.axhline(m, color=style.ORIGINAL, lw=1.2, zorder=1)
        series = [f for f in fractions if control_has_lever or f > 0]
        for k, f in enumerate(series):
            offset = (k - (len(series) - 1) / 2) * 0.16
            colour = style.credit_fraction_colour(f)
            means = []
            for i, value in enumerate(values):
                vals = np.array([v for n, v in finals.items()
                                 if _cf_value(n) == f and _lever_value(n, lever) == value])
                if not vals.size:
                    means.append(np.nan)
                    continue
                ax.scatter(np.full(vals.size, i + offset), vals, s=12, color=colour, alpha=0.55,
                           edgecolor="none", zorder=2)
                ax.errorbar(i + offset, vals.mean(), yerr=vals.std(), color=colour, lw=1.2,
                            capsize=2, zorder=3)
                means.append(vals.mean())
            ax.plot(np.arange(len(values)) + offset, means, color=colour, lw=1.6, marker="o",
                    markersize=3.5, zorder=4)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values, fontsize=7)
        ax.set_xlabel(label)
        style.title(ax, label)
    axes[0].set_ylabel(f"final development {style.metric_label(metric)}")
    handles = style.legend_patches({f"cf = {f:g}": style.credit_fraction_colour(f) for f in fractions})
    if any(not len({_lever_value(n, lv) for n in finals if _is_control(n)}) > 1 for lv, _ in levers):
        handles += style.legend_patches({"control ±1 SD": style.ORIGINAL})
    axes[-1].legend(handles=handles, loc="lower right", fontsize=7)
    fig.suptitle(f"{track.upper()} levers within each credit fraction")
    return fig


def seed_spread(track: str, exp: str = "exp1", metric: str | None = None):
    """Every configuration's seeds side by side, best on top.

    A difference between two configurations is only worth reading where it clearly exceeds the
    spread among one configuration's own seeds. Dots are seeds (coloured by credit fraction), the
    thin line their range, the tick their mean; the dashed line is the control's mean.
    """
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    finals = _finals(agg, metric, kept)
    by_config: dict[str, list[float]] = {}
    for name, v in finals.items():
        by_config.setdefault(_config_of(name), []).append(v)
    if not by_config:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.30))
        _empty(ax, "no finished arms to compare yet")
        return fig
    higher = HIGHER_IS_BETTER.get(metric, True)
    # Best first; the axis is inverted below, so the best configuration is the top row.
    configs = sorted(by_config, key=lambda c: np.mean(by_config[c]), reverse=higher)
    fig, ax = plt.subplots(figsize=style.row_figsize(len(configs), per_row=0.2, base=1.15))
    for i, config in enumerate(configs):
        vals = np.array(by_config[config])
        colour = style.credit_fraction_colour(_cf_value(config))
        ax.plot([vals.min(), vals.max()], [i, i], color=colour, lw=1.4, alpha=0.7, zorder=2)
        ax.scatter(vals, np.full(vals.size, i), s=16, color=colour, edgecolor="white",
                   linewidth=0.3, zorder=3)
        ax.plot([vals.mean()] * 2, [i - 0.3, i + 0.3], color=style.INK, lw=1.6, zorder=4)
    control = [v for n, v in finals.items() if _is_control(n)]
    if control:
        ax.axvline(float(np.mean(control)), color=style.ORIGINAL, ls="--", lw=1.1, zorder=1)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([_config_label(c) for c in configs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"final development {style.metric_label(metric)}")
    ax.grid(axis="x")
    spread = [np.ptp(v) for v in by_config.values() if len(v) > 1]
    style.title(ax, "Seed spread per configuration",
                f"median range over seeds: {np.median(spread):.3f}" if spread
                else "one scored seed per configuration: no spread to read")
    _cf_legend(ax, [_cf_value(c) for c in configs], loc="lower left" if higher else "lower right")
    fig.suptitle(f"{track.upper()} effect versus seed noise")
    return fig


# ---------------------------------------------------------------------------
# C. Where it holds — every dataset, out of domain, every metric
# ---------------------------------------------------------------------------


def _real_datasets(runs: dict[str, pd.DataFrame], metric: str) -> list[str]:
    """The development datasets that carry `metric` in any arm, in first-seen order."""
    seen: list[str] = []
    for df in runs.values():
        for d in _dataset_cols(df, metric):
            if d not in seen:
                seen.append(d)
    return seen


def per_dataset_pages(track: str, exp: str = "exp1", per_page: int = 6) -> int:
    runs = load_progress(track, exp)
    n = len(_real_datasets(runs, HEADLINE[track])) if runs else 0
    return max(1, int(np.ceil(n / per_page)))


def per_dataset_curves(track: str, page: int = 1, exp: str = "exp1", per_page: int = 6,
                       metric: str | None = None):
    """The monitored metric over training, one panel per dataset — credit (blue) vs control (grey),
    finished arms. Every panel names the dataset's role: a holdout dataset is shown as the evidence
    it is, but it never enters a development mean, and "missing in N arms" marks a monitor that
    logged nothing for those arms."""
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
    metric = metric or HEADLINE[track]
    roles = dataset_roles(track, exp)
    style.apply()
    datasets = _real_datasets(runs, metric) if runs else []
    rank = {"development": 0, "holdout": 1}
    datasets = sorted(datasets, key=lambda d: (rank.get(roles.get(d, ""), 2), d))
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
        cstack, ostack, missing = [], [], 0
        for name, df in agg.items():
            col = _dataset_cols(df, metric).get(ds)
            if col is None or df[col].isna().all():
                missing += 1
                continue
            d = df[["step", col]].dropna()
            if len(d) < 2:
                continue
            arr = np.interp(grid, d["step"], d[col], left=np.nan, right=np.nan)
            (ostack if _is_control(name) else cstack).append(arr)
        if cstack:
            ax.plot(grid, _nanmean(cstack), color=style.CREDIT, lw=1.6)
        if ostack:
            ax.plot(grid, _nanmean(ostack), color=style.ORIGINAL, lw=1.4)
        ax.set_xlabel("step")
        role = roles.get(ds, "no role")
        note = "holdout · never averaged" if role == "holdout" else role
        style.title(ax, ds[:22], note + (f" · missing in {missing} arms" if missing else ""))
    fig.suptitle(f"{track.upper()} {style.metric_label(metric)} per dataset"
                 f"{style.page_suffix(page, pages)}")
    return fig


def real_vs_ood(track: str, exp: str = "exp1", metric: str | None = None):
    """Development metric on the real-credit datasets vs the out-of-domain suites, over training.

    Specialising on a credit prior can lift credit scores while eroding the generality a model
    started with. Credit (solid) and out-of-domain (dashed) are each averaged over the datasets
    every finished arm can score in that domain, so a gap opening up is the cost of specialisation.
    Degrades to the credit curve alone when no out-of-domain columns were logged.
    """
    metric = metric or HEADLINE[track]
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.52))
    if not runs:
        _empty(ax, "no progress CSVs found in output/manifests/")
        return fig
    grid = _common_step_grid(runs)
    for domain, label, colour, ls in (("real", "real credit", style.CREDIT, "-"),
                                      ("ood", "out-of-domain", style.WARN, "--")):
        kept, _ = comparable_datasets(runs, metric, domain)
        if not kept:
            if domain == "ood":
                ax.text(0.5, 0.08, "no out-of-domain columns logged", ha="center", va="center",
                        color=style.MUTED, fontsize=8, transform=ax.transAxes)
            continue
        for cf_group, alpha in ((True, 1.0), (False, 0.55)):
            stack = [c for n, df in agg.items() if _is_control(n) is not cf_group
                     for c in [_curve(df, metric, kept, grid, domain)] if c is not None]
            if stack:
                who = "credit arms" if cf_group else "control arms"
                ax.plot(grid, _nanmean(stack), color=colour, lw=2.0,
                        ls=ls, alpha=alpha, label=f"{label}, {who} ({len(kept)} datasets)")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    ax.legend(loc="lower right", fontsize=7)
    style.title(ax, "Credit vs out-of-domain", "does specialising cost generality?")
    fig.suptitle(f"{track.upper()} credit versus out-of-domain")
    return fig


def metric_pages(track: str, exp: str = "exp1", per_page: int = 9) -> int:
    runs = load_progress(track, exp)
    n = len(available_metrics(runs)) if runs else 0
    return max(1, int(np.ceil(n / per_page)))


def all_eval_metrics(track: str, page: int = 1, exp: str = "exp1", per_page: int = 9):
    """One panel per logged development metric: credit and control means over the finished arms,
    each over the datasets every arm can score for that metric."""
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
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
    grid = _common_step_grid(runs) if runs else np.array([0.0, 1.0])
    for ax, metric in zip(axes, chunk):
        ax.axis("on")
        kept, _ = comparable_datasets(runs, metric)
        for is_control, colour in ((False, style.CREDIT), (True, style.ORIGINAL)):
            stack = [c for n, df in agg.items() if _is_control(n) is is_control
                     for c in [_curve(df, metric, kept, grid)] if c is not None]
            if stack:
                ax.plot(grid, _nanmean(stack), color=colour, lw=1.5)
        ax.set_xlabel("step", fontsize=7)
        arrow = "↑" if HIGHER_IS_BETTER.get(metric, True) else "↓"
        style.title(ax, f"{style.metric_label(metric)} {arrow}")
    fig.suptitle(f"{track.upper()} every development metric{style.page_suffix(page, pages)}")
    return fig


# ---------------------------------------------------------------------------
# D. Every arm
# ---------------------------------------------------------------------------


def config_pages(track: str, exp: str = "exp1", per_page: int = 12) -> int:
    runs = load_progress(track, exp)
    return max(1, int(np.ceil(len(runs) / per_page)))


def per_config(track: str, page: int = 1, exp: str = "exp1", per_page: int = 12):
    """One panel per arm, in sweep-map order: its train loss (grey) and development headline
    (blue, right axis). An unfinished arm's title carries the share it trained."""
    runs = load_progress(track, exp)
    metric = HEADLINE[track]
    kept, _ = comparable_datasets(runs, metric)
    done = completed_arms(runs)
    target = _target_step(runs)
    names = sorted(runs, key=lambda n: (_config_sort_key(n, exp), _seed_of(n) or 0))
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
        m = pd.DataFrame({"step": df["step"], "m": _mean_metric(df, metric, "real", kept)}).dropna()
        if len(m) >= 2:
            ax2.plot(m["step"], m["m"], color=style.CREDIT, lw=1.2)
        ax2.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, max(target, 1))
        note = None if name in done else f"{int(df['step'].max()) / max(target, 1):.0%} trained"
        style.title(ax, arm_label(name), note)
    fig.suptitle(f"{track.upper()} every arm: loss and score{style.page_suffix(page, pages)}")
    return fig


def best_and_worst(track: str, exp: str = "exp1"):
    """The best and worst finished arm by final development score, loss and score side by side."""
    runs = load_progress(track, exp)
    metric = HEADLINE[track]
    kept, _ = comparable_datasets(runs, metric)
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
        m = pd.DataFrame({"step": runs[name]["step"],
                          "m": _mean_metric(runs[name], metric, "real", kept)}).dropna()
        ax2.plot(m["step"], m["m"], color=colour, lw=1.8, label=f"{tag}: {arm_label(name)}")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss")
    style.title(ax1, "Train loss")
    ax2.set_xlabel("step")
    ax2.set_ylabel(f"development {style.metric_label(metric)}")
    ax2.legend(loc="lower right", fontsize=7)
    style.title(ax2, f"Development {style.metric_label(metric)}")
    fig.suptitle(f"{track.upper()} best versus worst arm")
    return fig


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
            ax.plot(grid, _nanmean(stacks), color=colour, lw=1.6, label=label)
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
            ax.plot(grid, _nanmean(stacks), color=colour, lw=1.6, label=label)
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


# ---------------------------------------------------------------------------
# Lever parsing (used by every by-lever figure)
# ---------------------------------------------------------------------------


def _lever_value(run_name: str, lever: str) -> str | None:
    """How one swept lever reads out of a run name. Used to colour and group curves by the one
    knob a figure is about, rather than by the arm as a whole."""
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


# ---------------------------------------------------------------------------
# Summary — printed last, in the notebook's section order
# ---------------------------------------------------------------------------


def _by_lever(finals: dict[str, float], lever: str) -> list[tuple[str, float, float, int]]:
    """`[(value, mean, sd, n)]` of final scores grouped by one lever, in the lever's order."""
    groups: dict[str, list[float]] = {}
    for name, v in finals.items():
        lv = _lever_value(name, lever)
        if lv is not None and not (lever == "intensity" and _is_control(name)):
            groups.setdefault(lv, []).append(v)
    order = _ORDER.get(lever)
    keys = sorted(groups, key=lambda k: (order.index(k) if order and k in order else 99, k))
    return [(k, float(np.mean(groups[k])), float(np.std(groups[k])), len(groups[k])) for k in keys]


def training_summary(track: str, exp: str = "exp1") -> str:
    """A text summary of the sweep, section by section in the notebook's order: the run, the
    answer, where it holds."""
    runs = load_progress(track, exp)
    tel = load_telemetry(track, exp)
    title = f"{exp.upper()} {track.upper()} TRAINING — development-split monitoring"
    if not runs:
        return f"{title}\n  no progress CSVs in output/manifests/ yet."
    metric = HEADLINE[track]
    label = style.metric_label(metric)
    kept, dropped = comparable_datasets(runs, metric)
    done = completed_arms(runs)
    agg = _aggregate(runs)
    target = _target_step(runs)
    scheduled = len(_sweep_grid(track, exp)) or len(runs)
    lines = [title, "", "A. THE RUN",
             f"  arms: {scheduled} scheduled | {len(runs)} started | {len(done)} finished"
             f" | {len(runs) - len(done)} unfinished"]
    for name in sorted(set(runs) - done, key=lambda n: (_config_sort_key(n, exp), _seed_of(n) or 0)):
        lines.append(f"    unfinished: {arm_label(name):<24} {int(runs[name]['step'].max()):>6,}"
                     f" / {target:,} steps")
    roles = dataset_roles(track, exp)
    lines.append(f"  development datasets averaged ({len(kept)}): {', '.join(kept) or 'none'}")
    holdout = [d for d in dropped if roles.get(d) == "holdout"]
    if holdout:
        lines.append(f"  holdout datasets monitored but never averaged: {', '.join(holdout)}")
    partial = [d for d in dropped if roles.get(d) == "development"]
    if partial:
        lines.append(f"  development datasets not carried by every arm: {', '.join(partial)}")
    missing = [n for n in agg if not np.isfinite(_final(_mean_metric(agg[n], metric, "real", kept)))]
    if missing and kept:
        lines.append(f"  finished arms without a development score: {len(missing)} of {len(agg)}"
                     f" (a monitor that logged only missing values)")
    old = sum("progress_protocol" not in df.columns for df in runs.values())
    if old:
        lines.append(f"  arms monitored before the development-only protocol: {old} of {len(runs)}")
    lever = "filter" if exp == "exp1" else "strategy"
    speeds: dict[str, list[float]] = {}
    for name, df in tel.items():
        if "steps_per_s" in df:
            s = pd.to_numeric(df["steps_per_s"], errors="coerce")
            s = s[np.isfinite(s) & (s > 0)]
            if len(s) and _lever_value(name, lever):
                speeds.setdefault(_lever_value(name, lever), []).append(float(s.median()))
    if speeds:
        parts = [f"{k} {np.median(v):.2f} steps/s (~{target / np.median(v) / 3600:.0f} h)"
                 for k, v in sorted(speeds.items())]
        lines.append(f"  median speed by {lever}: " + " | ".join(parts))

    finals = _finals(agg, metric, kept)
    lines += ["", f"B. THE ANSWER — final development {label}, {len(finals)} finished arms"]
    for lv, lv_label in LEVERS.get(exp, LEVERS["exp1"]):
        rows = _by_lever(finals, lv)
        if rows:
            lines.append(f"  by {lv_label}: " + " | ".join(
                f"{k} {m:.4f} ±{s:.4f} (n={n})" for k, m, s, n in rows))
    credit = [v for n, v in finals.items() if not _is_control(n)]
    control = [v for n, v in finals.items() if _is_control(n)]
    if credit and control:
        lines.append(f"  credit prior minus control: {np.mean(credit) - np.mean(control):+.4f}")
    ranked = sorted(finals.items(), key=lambda kv: kv[1], reverse=HIGHER_IS_BETTER.get(metric, True))
    if ranked:
        lines.append(f"  best  {label} = {ranked[0][1]:.4f}  {arm_label(ranked[0][0])}")
        lines.append(f"  worst {label} = {ranked[-1][1]:.4f}  {arm_label(ranked[-1][0])}")
    by_config: dict[str, list[float]] = {}
    for name, v in finals.items():
        by_config.setdefault(_config_of(name), []).append(v)
    spread = [np.ptp(v) for v in by_config.values() if len(v) > 1]
    if spread:
        lines.append(f"  seed noise: median range over a configuration's seeds = {np.median(spread):.4f}")

    lines += ["", "C. WHERE IT HOLDS"]
    ood_kept, _ = comparable_datasets(runs, metric, "ood")
    if ood_kept:
        ood = _finals(agg, metric, ood_kept, "ood")
        both = [n for n in finals if n in ood]
        if both:
            lines.append(f"  out-of-domain {label} over {len(ood_kept)} suites, {len(both)} finished arms: "
                         f"credit datasets {np.mean([finals[n] for n in both]):.4f} | "
                         f"out-of-domain {np.mean([ood[n] for n in both]):.4f}")
    else:
        lines.append("  no out-of-domain monitoring logged")
    lines.append(f"  metrics logged: {', '.join(available_metrics(runs)) or 'none'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------


def _quietly(fn, *args, **kwargs):
    """Run a NaN-aware numpy reduction without its "empty slice" RuntimeWarning, which fires in
    exactly the expected case — a step no arm has reached — and would otherwise litter every
    notebook's output."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def _nanmean(stack) -> np.ndarray:
    """Mean over arms at each step, NaN where no arm has a value."""
    return _quietly(np.nanmean, np.array(stack), axis=0)


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
