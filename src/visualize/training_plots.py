"""Level-1 TRAINING visualisation: what every arm of a sweep did while it trained.

Reads what each arm writes to `output/manifests/` — `<run>__progress.csv` (train loss and the
development-split monitoring metrics, real-credit and out-of-domain, sampled every
`progress.every_datasets`), `<run>__telemetry.csv` (throughput, GPU, per-block gradients) and
`<run>__summary.json` — and turns it into the figures of `1.1_pd_training` / `1.2_lgd_training`
(Exp1, `exp="exp1"`) and `2.1_pd_finetuning` / `2.2_lgd_finetuning` (Exp2, `exp="exp2"`).

THREE RULES DECIDE WHAT A CURVE MAY AVERAGE OVER. Each exists because the real sweep broke the
naive version, and each is stated in the notebooks so a reader knows what a line is an average of.

* **Finished arms only.** Averaging every arm at every step changes the composition of the average
  wherever an unfinished arm stops contributing, so the curve JUMPS near the end for no reason but
  bookkeeping — the banded and credit curves both did. Aggregates use the arms whose summary says
  `completed: true` (the flag `sweep_status` resubmits on), and fall back to every arm while none
  has finished. Unfinished arms stay visible: in the sweep map, dotted in the curves, and per arm.
* **Only where every averaged arm has a value.** A requeued arm's progress log restarts its
  schedule, so finished arms log their last point anywhere between step ~11,900 and 12,500, and
  three PD banded arms' logs only start at steps 3,251-4,001 (restarted under the new protocol). A
  mean taken where some arms are missing changes composition and jumps — the training-loss mean
  did. So a group's curve is drawn only over the steps all its arms share (`_complete`).
* **Development datasets only, the same ones for every arm.** A prior is chosen on the development
  split; the holdout datasets are what the benchmark finally reports. But curves from before the
  development-only monitoring protocol (23-09-2026) scored "the smallest few" datasets whatever
  their role — for Exp1 that put holdout sets (PD `hmeq`, `thomas`; LGD `axa`, `loss2`,
  `base_modelisation`) into every old arm's mean. So a domain mean covers only the config's
  development datasets that EVERY arm carries (PD: `german`, `myhom`), never a holdout one. A
  missing value is still never skipped (`skipna=False`): an arm whose development monitor is
  missing (LGD `base_model` in every seed-0 and seed-2 arm, where one of 381 predictions is
  non-finite) drops out of an average and is COUNTED in the printed summary, rather than being
  averaged over fewer datasets. `monitoring_coverage` draws the whole situation, arm by arm.

A FIGURE CARRIES DATA, AXES AND A LEGEND — NOT PROSE. No figure here has a second, interpretive
heading line; legends sit below the data (`style.legend_below`), never on it; every axis is labelled.
What a figure means is the notebook's markdown and the printed summary.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils import paths
from src.visualize import style

# Headline evaluation metric per task: the one curve a reader looks at first. Higher is better
# for both (ROC-AUC for classification, R² for regression), so "best/worst" is unambiguous.
HEADLINE = {"pd": "roc_auc", "lgd": "r2"}
HIGHER_IS_BETTER = {"auc": True, "ap": True, "r2": True, "spearman": True, "kendall": True,
                    "roc_auc": True, "pr_auc": True,
                    "brier": False, "logloss": False, "rmse": False, "mae": False,
                    "pinball": False, "crps": False, "ks": True}

#: The levers that name an arm, per experiment, and how each reads in a heading. Credit fraction
#: comes first in both: it is the lever the sweep's rows are grouped by.
LEVERS = {
    "exp1": (("credit_fraction", "credit fraction"), ("filter", "filter mode"),
             ("intensity", "prior intensity")),
    "exp2": (("credit_fraction", "credit fraction"), ("strategy", "freeze strategy"),
             ("l2sp", "L2-SP"), ("lr", "learning rate")),
}

#: The development metrics grouped by what they measure, one figure per group, in reading order.
#: Only real metrics: bookkeeping columns (`n_test`), properties of the DATA (`true_mass_at_0`,
#: drawn as a reference line in the predicted-mass panels instead) and prediction diagnostics
#: (`pred_min`, `nan_predictions`, `share outside [0, 1]`) were once each given a panel and an arrow.
METRIC_THEMES = {
    "lgd": (("Accuracy", ("r2", "rmse", "mae", "pinball", "crps", "brier")),
            ("Ranking and calibration", ("spearman", "kendall", "calibration_slope", "bias",
                                         "pit_mean", "pit_uniformity_error")),
            ("Intervals and boundary atoms", ("coverage_50", "coverage_80", "coverage_90",
                                              "pred_mass_at_0", "pred_mass_at_1",
                                              "boundary_mass_abs_err", "mae_boundary",
                                              "mae_interior"))),
    "pd": (("Discrimination and calibration", ("roc_auc", "pr_auc", "ks", "brier", "logloss",
                                                "calibration_slope")),),
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


#: The order a lever's values are listed in, everywhere: the filter from none to strictest, the
#: intensity from mild to aggressive, the freeze strategies from most to least trained. Credit
#: fraction always comes first in a configuration's sort key, so its effect reads as a block.
_ORDER = {"intensity": ("mild", "aggressive"), "filter": ("off", "tabicl", "banded"),
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


def _value_order(lever: str, values) -> list:
    order = _ORDER.get(lever)
    return sorted(values, key=lambda v: (order.index(v) if order and v in order else 99, str(v)))


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


def _cf_handles(fractions, *, markers: bool = True) -> list[Any]:
    """Legend entries for the credit fractions present: colour, and marker where points are drawn."""
    return [Line2D([], [], color=style.credit_fraction_colour(f),
                   marker=style.credit_fraction_marker(f) if markers else None,
                   linestyle="none" if markers else "-", lw=2.0, markersize=5,
                   label=f"cf = {f:g}" + (" (control)" if f == 0 else ""))
            for f in sorted(set(fractions))]


def _cf_legend(ax: Any, fractions, *, loc: str = "lower right", title: str | None = None) -> None:
    """Kept for callers that still want an in-axes legend; the figures here use `_cf_handles`."""
    ax.legend(handles=_cf_handles(fractions), loc=loc, fontsize=7, title=title, title_fontsize=7)


def _placeholder(message: str):
    """A figure that says what is missing, in the smallest honest space: the real figure takes its
    place once the data exists, and a half-page of white said nothing more than this line does."""
    style.apply()
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 0.9))
    _empty(ax, message)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    return fig


def _smooth(values: pd.Series, window: int = 3) -> pd.Series:
    """A centred rolling mean — for the training loss, whose logged value is ONE batch's loss."""
    return values.rolling(window, center=True, min_periods=1).mean()


# ---------------------------------------------------------------------------
# A. The run — is the experiment sound?
# ---------------------------------------------------------------------------


def sweep_map(track: str, exp: str = "exp1", metric: str | None = None):
    """Every arm of the sweep in one grid, coloured by its final development score.

    Rows are configurations grouped by credit fraction (white rules between the groups), columns
    the seeds — or, when the sweep runs one seed (Exp2's 60 arms), the credit fractions, because a
    single column of 60 rows cannot be read. A cell with a number has a development score; a grey
    cell has none (its monitor logged only missing values); a hatched cell is still training and
    says how far it got; a dotted cell never started. Only the scores carry text.
    """
    runs = load_progress(track, exp)
    metric = metric or HEADLINE[track]
    style.apply()
    names = _sweep_grid(track, exp) or sorted(runs)
    if not names:
        return _placeholder("no arms scheduled or started yet")
    seeds = sorted({s for s in (_seed_of(n) for n in names) if s is not None})
    by_cf = len(seeds) <= 1 and len({_cf_value(n) for n in names}) > 1
    if by_cf:
        def row_of(n: str) -> str:
            return re.sub(r"credit_fraction=[0-9p.]+", "credit_fraction=x", _config_of(n))
        col_values = sorted({_cf_value(n) for n in names})
        col_labels = [f"cf = {v:g}" for v in col_values]

        def col_of(n: str) -> int:
            return col_values.index(_cf_value(n))
    else:
        row_of = _config_of
        col_values = seeds or [None]
        col_labels = [f"seed {s}" for s in seeds] or ["arm"]

        def col_of(n: str) -> int:
            return col_values.index(_seed_of(n)) if seeds else 0
    rows = sorted({row_of(n) for n in names}, key=lambda c: _config_sort_key(c, exp))
    kept, _ = comparable_datasets(runs, metric)
    done = completed_arms(runs)
    target = _target_step(runs)
    values = np.full((len(rows), len(col_values)), np.nan)
    status: dict[tuple[int, int], str] = {}
    share: dict[tuple[int, int], float] = {}
    for n in names:
        i, j = rows.index(row_of(n)), col_of(n)
        if n not in runs:
            status[i, j] = "todo"
            continue
        values[i, j] = _final(_mean_metric(runs[n], metric, "real", kept))
        status[i, j] = "done" if n in done else "partial"
        share[i, j] = int(runs[n]["step"].max()) / target if target else 0.0

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.2, base=1.25))
    finite = values[np.isfinite(values)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    cmap = plt.get_cmap(style.CMAP_SEQ).with_extremes(bad=style.GRID)
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
    nothing_yet = all(state == "todo" for state in status.values())
    for (i, j), state in status.items():
        v = values[i, j]
        if state == "todo":
            # Before anything has run the map is the DESIGN: plain cells, one per scheduled arm.
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                       hatch=None if nothing_yet else "...",
                                       edgecolor=style.GRID if nothing_yet else style.MUTED,
                                       linewidth=0.8 if nothing_yet else 0.0))
            continue
        if state == "partial":
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                       edgecolor=style.WARN, linewidth=0.0))
        text = f"{v:.3f}" if np.isfinite(v) else ""
        if state == "partial":
            text = (text + "  " if text else "") + f"({share[i, j]:.0%})"
        if text:
            dark = np.isfinite(v) and (v - lo) / (hi - lo + 1e-12) < 0.55
            # A white backing where the cell is hatched: text laid over the stripes is unreadable.
            box = (dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85)
                   if state == "partial" else None)
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                    color=style.INK if box or not dark else "white", bbox=box)
    if not by_cf:
        fractions = [_cf_value(c) for c in rows]
        for i in range(1, len(rows)):
            if fractions[i] != fractions[i - 1]:
                ax.axhline(i - 0.5, color="white", lw=2.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_config_label(c).replace("cfx·", "") for c in rows],
                       fontsize=6.5 if len(rows) > 20 else 7)
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels(col_labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    if finite.size:
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02,
                     label=f"final development {style.metric_label(metric)}")
    states = set(status.values())
    handles = []
    if any(state != "todo" and not np.isfinite(values[i, j]) for (i, j), state in status.items()):
        handles.append(Patch(facecolor=style.GRID, label="no development score logged"))
    if "partial" in states:
        handles.append(Patch(facecolor="white", hatch="////", edgecolor=style.WARN,
                             label="still training (share of steps done)"))
    if "todo" in states:
        handles.append(Patch(facecolor="white", hatch=None if nothing_yet else "...",
                             edgecolor=style.GRID if nothing_yet else style.MUTED,
                             label="scheduled, not started"))
    if handles:
        style.legend_below(ax, handles, ncol=len(handles))
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

    One row per dataset (development first, then holdout), one column per arm in sweep-map order,
    grouped by credit fraction. Only the teal cells enter a development mean: a holdout dataset an
    arm happened to monitor is drawn but never averaged, and a red cell is a monitor that logged
    nothing but missing values. The legend lists only the categories that occur.
    """
    metric = metric or HEADLINE[track]
    runs = load_progress(track, exp)
    style.apply()
    if not runs:
        return _placeholder("no progress logs in output/manifests/ yet")
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

    fig, ax = plt.subplots(figsize=style.row_figsize(len(datasets), per_row=0.26, base=1.55))
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
    present = sorted(set(codes.ravel().tolist()))
    style.legend_below(ax, style.legend_patches({_COVERAGE[k][0]: _COVERAGE[k][1] for k in present}),
                       ncol=2)
    return fig


def throughput(track: str, exp: str = "exp1"):
    """What each arm cost: its training speed, grouped by the lever that sets it, and the wall-clock
    time that speed implies for the whole run.

    Each point is one arm (its median steps per second over the telemetry), coloured and shaped by
    credit fraction; the black bar is the group median, with its value beside it. Groups are in the
    lever's order — for the filter, from none (`off`) to the strictest (`banded`).
    """
    tel = load_telemetry(track, exp)
    runs = load_progress(track, exp)
    lever, lever_label = ("filter", "filter mode") if exp == "exp1" else ("strategy", "freeze strategy")
    style.apply()
    speed: dict[str, float] = {}
    for name, df in tel.items():
        if "steps_per_s" not in df:
            continue
        s = pd.to_numeric(df["steps_per_s"], errors="coerce")
        s = s[np.isfinite(s) & (s > 0)]
        if len(s):
            speed[name] = float(s.median())
    groups = _value_order(lever, {_lever_value(n, lever) for n in speed} - {None})
    if not groups:
        return _placeholder("no throughput in the telemetry logs yet")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.42))
    target = _target_step(runs) or 12500
    jitter = np.random.default_rng(0)
    for i, group in enumerate(groups):
        members = [n for n in speed if _lever_value(n, lever) == group]
        steps = np.array([speed[n] for n in members])
        hours = target / steps / 3600.0
        xs = i + (jitter.random(len(members)) - 0.5) * 0.24
        for ax, vals, fmt in ((ax1, steps, "{:.2f}"), (ax2, hours, "{:.0f} h")):
            for x, v, n in zip(xs, vals, members):
                f = _cf_value(n)
                ax.scatter([x], [v], s=20, color=style.credit_fraction_colour(f),
                           marker=style.credit_fraction_marker(f), edgecolor="white",
                           linewidth=0.3, zorder=3)
            med = float(np.median(vals))
            ax.plot([i - 0.26, i + 0.26], [med] * 2, color=style.INK, lw=1.8, zorder=4)
            ax.annotate(fmt.format(med), (i + 0.3, med), va="center", ha="left", fontsize=7,
                        color=style.INK)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_xlim(-0.5, len(groups) - 0.1)
        ax.set_xlabel(lever_label)
        ax.set_ylim(bottom=0)
    ax1.set_ylabel("training steps per second")
    ax2.set_ylabel(f"hours for {target:,} steps")
    ax1.set_title("Speed of each arm", loc="left", fontsize=9)
    ax2.set_title("Implied time for the full run", loc="left", fontsize=9)
    style.legend_below(fig, _cf_handles([_cf_value(n) for n in speed])
                       + [Line2D([], [], color=style.INK, lw=1.8, label="group median")], ncol=4)
    return fig


def training_loss(track: str, exp: str = "exp1"):
    """Training loss against step for every arm, coloured by credit fraction, with the median of
    the finished arms of each fraction drawn bold.

    The logged loss is one batch's loss, so each arm is drawn as a rolling mean over three logged
    points. The loss is computed on each arm's OWN synthetic tasks, so its level is a property of
    the prior (a low base rate makes cross-entropy small), not of how well the arm learned: read
    each line's shape — does it descend, does it diverge — never one line's level against another's.
    """
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
    style.apply()
    if not runs:
        return _placeholder("no progress logs in output/manifests/ yet")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    grid = _common_step_grid(agg)
    by_cf: dict[float, list] = {}
    for name, df in runs.items():
        d = df.dropna(subset=["train_loss"])
        if len(d) < 2:
            continue
        f = _cf_value(name)
        smooth = _smooth(d["train_loss"])
        finished = name in agg
        ax.plot(d["step"], smooth, color=style.credit_fraction_colour(f), alpha=0.3, lw=0.7,
                ls="-" if finished else ":", zorder=1)
        if finished:
            by_cf.setdefault(f, []).append(np.interp(grid, d["step"], smooth, left=np.nan,
                                                     right=np.nan))
    for f in sorted(by_cf):
        ax.plot(grid, _complete(by_cf[f], np.median), color=style.credit_fraction_colour(f),
                lw=2.2, zorder=4, marker=style.credit_fraction_marker(f), markevery=10,
                markersize=4, label=f"cf = {f:g}: median of {len(by_cf[f])} finished arms")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss (cross-entropy)" if track == "pd"
                  else "training loss (pinball)")
    ax.set_ylim(bottom=0)
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color=style.MUTED, lw=0.7, label="one arm (dotted: unfinished)")]
    style.legend_below(ax, handles, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# B. The answer — development monitoring
# ---------------------------------------------------------------------------


def metric_over_training(track: str, exp: str = "exp1", metric: str | None = None):
    """The development metric against step for every arm, in its credit-fraction colour (dotted
    while unfinished), with the mean of the finished arms of each fraction drawn bold."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    if not runs:
        return _placeholder("no progress logs in output/manifests/ yet")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.48))
    grid = _common_step_grid(agg)
    by_cf: dict[float, list] = {}
    for name, df in runs.items():
        d = pd.DataFrame({"step": df["step"], "m": _mean_metric(df, metric, "real", kept)}).dropna()
        if len(d) < 2:
            continue
        f = _cf_value(name)
        finished = name in agg
        ax.plot(d["step"], d["m"], color=style.credit_fraction_colour(f), alpha=0.3, lw=0.7,
                ls="-" if finished else ":", zorder=1)
        if finished:
            curve = _curve(df, metric, kept, grid)
            if curve is not None:
                by_cf.setdefault(f, []).append(curve)
    for f in sorted(by_cf):
        ax.plot(grid, _complete(by_cf[f]), color=style.credit_fraction_colour(f), lw=2.2, zorder=4,
                marker=style.credit_fraction_marker(f), markevery=10, markersize=4,
                label=f"cf = {f:g}: mean of {len(by_cf[f])} finished arms")
    if not by_cf:
        _empty(ax, "no arm carries a development score yet")
        return fig
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color=style.MUTED, lw=0.7, label="one arm (dotted: unfinished)")]
    style.legend_below(ax, handles, ncol=2)
    return fig


def credit_vs_control_over_training(track: str, exp: str = "exp1", metric: str | None = None):
    """The headline comparison as a curve: credit-prior arms (any credit fraction above 0) against
    the control arms, median and inter-quartile band over the finished arms of each group."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    if not runs:
        return _placeholder("no progress logs in output/manifests/ yet")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    grid = _common_step_grid(agg)
    groups: dict[str, list] = {"credit prior": [], "control": []}
    for name, df in agg.items():
        curve = _curve(df, metric, kept, grid)
        if curve is not None:
            groups["control" if _is_control(name) else "credit prior"].append(curve)
    labels = {"credit prior": "credit prior (cf > 0)", "control": "control (cf = 0)"}
    for key, colour in (("credit prior", style.CREDIT), ("control", style.ORIGINAL)):
        stack = groups[key]
        if not stack:
            continue
        med = _complete(stack, np.median)
        lo, hi = _complete_band(stack)
        ax.fill_between(grid, lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot(grid, med, color=colour, lw=2.3, zorder=4,
                label=f"{labels[key]}: median of {len(stack)} arms, inter-quartile band")
    if not any(groups.values()):
        _empty(ax, "no finished arm carries a development score yet")
        return fig
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    style.legend_below(ax, ncol=1)
    return fig


def _lever_style(lever: str, value: str, index: int) -> tuple[str, str]:
    """(colour, line style) for one value of a lever: the credit fraction on its grey-to-blue
    scale, prior intensity in our prior's two blues, every other lever in ink shades and dashes."""
    if lever == "credit_fraction":
        return style.credit_fraction_colour(float(value.split("=")[1])), "-"
    if lever == "intensity":
        return (style.CREDIT_MILD if value == "mild" else style.CREDIT_STRONG), "-"
    return style.LEVEL_STYLES[index % len(style.LEVEL_STYLES)]


def _lever_marker(lever: str, value: str) -> dict[str, Any]:
    """Marker kwargs for a credit-fraction curve — the fraction's shape every tenth point, so
    cf 0.5 and cf 1 separate by shape as well as by their two similar blues."""
    if lever != "credit_fraction":
        return {}
    f = float(value.split("=")[1])
    return {"marker": style.credit_fraction_marker(f), "markevery": 10, "markersize": 4}


def _lever_curves(agg: dict[str, pd.DataFrame], lever: str, metric: str, kept: list[str],
                  grid: np.ndarray) -> dict[str, list]:
    by_val: dict[str, list] = {}
    for name, df in agg.items():
        v = _lever_value(name, lever)
        if v is None or (lever == "intensity" and _is_control(name)):
            continue
        curve = _curve(df, metric, kept, grid)
        if curve is not None:
            by_val.setdefault(v, []).append(curve)
    return by_val


def metric_by_lever(track: str, lever: str, exp: str = "exp1", metric: str | None = None):
    """The development metric over training, one mean line per value of a single swept lever,
    averaged over the finished arms that share it."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    if not runs:
        return _placeholder("no progress logs in output/manifests/ yet")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    grid = _common_step_grid(agg)
    by_val = _lever_curves(agg, lever, metric, kept, grid)
    if not by_val:
        _empty(ax, f"no arms carry the '{lever}' lever")
        return fig
    for i, v in enumerate(_value_order(lever, by_val)):
        colour, ls = _lever_style(lever, v, i)
        ax.plot(grid, _complete(by_val[v]), color=colour, ls=ls, lw=2.0,
                label=f"{v}  ({len(by_val[v])} arms)", **_lever_marker(lever, v))
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    style.legend_below(ax, ncol=3)
    return fig


def metric_by_levers(track: str, exp: str = "exp1", metric: str | None = None):
    """Every swept lever side by side: one panel per lever, one mean curve per value, on ONE shared
    y axis — so the size of each lever's effect can be compared at a glance.

    Each curve averages the finished arms that share a value of that lever (and have a development
    score). The control has no credit prior, so it has no intensity and is left out of that panel.
    """
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    levers = LEVERS.get(exp, LEVERS["exp1"])
    if not runs:
        return _placeholder("no progress logs in output/manifests/ yet")
    ncols = len(levers) if len(levers) <= 3 else 2
    nrows = int(np.ceil(len(levers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(
        ncols, nrows, panel_ratio=1.05 if ncols == 3 else 0.62), sharey=True, squeeze=False)
    grid = _common_step_grid(agg)
    drawn = False
    for ax, (lever, label) in zip(axes.ravel(), levers):
        by_val = _lever_curves(agg, lever, metric, kept, grid)
        for i, v in enumerate(_value_order(lever, by_val)):
            colour, ls = _lever_style(lever, v, i)
            ax.plot(grid, _complete(by_val[v]), color=colour, ls=ls, lw=1.8,
                    label=f"{v} ({len(by_val[v])})", **_lever_marker(lever, v))
            drawn = True
        ax.set_title(label, loc="left", fontsize=9)
        ax.set_xlabel("training step")
        ax.tick_params(axis="x", labelsize=7)
        if by_val:
            style.legend_below(ax, ncol=1)
    for ax in axes.ravel()[len(levers):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel(f"development {style.metric_label(metric)}")
    if not drawn:
        _empty(axes[0][0], "no finished arm carries a development score yet")
    return fig


def final_metric_by_lever(track: str, exp: str = "exp1", metric: str | None = None):
    """Final development score of every finished arm, grouped by each swept lever in turn: points
    coloured and shaped by credit fraction, a black bar at each group mean, one shared y axis."""
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    levers = LEVERS.get(exp, LEVERS["exp1"])
    ncols = len(levers) if len(levers) <= 3 else 2
    nrows = int(np.ceil(len(levers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(
        ncols, nrows, panel_ratio=0.95 if ncols == 3 else 0.62), sharey=True, squeeze=False)
    flat = axes.ravel()
    finals = _finals(agg, metric, kept)
    if not finals:
        for ax in flat[1:]:
            ax.axis("off")
        _empty(flat[0], "no finished arms to summarise yet")
        return fig
    jitter = np.random.default_rng(0)
    for ax, (lever, label) in zip(flat, levers):
        by: dict[str, list] = {}
        for name, v in finals.items():
            lv = _lever_value(name, lever)
            if lv is not None and not (lever == "intensity" and _is_control(name)):
                by.setdefault(lv, []).append((name, v))
        if not by:
            ax.axis("off")
            continue
        xs = _value_order(lever, by)
        for i, x in enumerate(xs):
            vals = np.array([v for _, v in by[x]])
            for (name, v), dx in zip(by[x], (jitter.random(len(vals)) - 0.5) * 0.22):
                f = _cf_value(name)
                ax.scatter([i + dx], [v], s=20, color=style.credit_fraction_colour(f),
                           marker=style.credit_fraction_marker(f), alpha=0.9, edgecolor="white",
                           linewidth=0.3, zorder=3)
            ax.plot([i - 0.26, i + 0.26], [vals.mean()] * 2, color=style.INK, lw=1.8, zorder=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([x.replace("cf=", "") for x in xs], fontsize=7)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_xlabel(label)
    for ax in flat[len(levers):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel(f"final development {style.metric_label(metric)}")
    style.legend_below(fig, _cf_handles([_cf_value(n) for n in finals])
                       + [Line2D([], [], color=style.INK, lw=1.8, label="group mean")], ncol=4)
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
    ncols = len(levers) if len(levers) <= 3 else 2
    nrows = int(np.ceil(len(levers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(
        ncols, nrows, panel_ratio=0.8 if ncols <= 2 else 1.0), sharey=True, squeeze=False)
    flat = axes.ravel()
    finals = _finals(agg, metric, kept)
    if not finals:
        for ax in flat[1:]:
            ax.axis("off")
        _empty(flat[0], "no finished arms to compare yet")
        return fig
    fractions = sorted({_cf_value(n) for n in finals})
    band_drawn = False
    for ax, (lever, label) in zip(flat, levers):
        control = [v for n, v in finals.items() if _is_control(n)]
        control_has_lever = len({_lever_value(n, lever) for n in finals if _is_control(n)}) > 1
        values = _value_order(lever, {_lever_value(n, lever) for n in finals} - {None})
        if not control_has_lever and control:
            m, s = float(np.mean(control)), float(np.std(control))
            ax.axhspan(m - s, m + s, color=style.ORIGINAL, alpha=0.22, lw=0, zorder=0)
            ax.axhline(m, color=style.ORIGINAL, lw=1.2, zorder=1)
            band_drawn = True
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
                           marker=style.credit_fraction_marker(f), edgecolor="none", zorder=2)
                ax.errorbar(i + offset, vals.mean(), yerr=vals.std(), color=colour, lw=1.2,
                            capsize=2, zorder=3)
                means.append(vals.mean())
            ax.plot(np.arange(len(values)) + offset, means, color=colour, lw=1.6,
                    marker=style.credit_fraction_marker(f), markersize=3.5, zorder=4)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values, fontsize=7)
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.set_xlabel(label)
    for ax in flat[len(levers):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel(f"final development {style.metric_label(metric)}")
    handles = [Line2D([], [], color=style.credit_fraction_colour(f), lw=1.6,
                      marker=style.credit_fraction_marker(f), markersize=4,
                      label=f"cf = {f:g}: mean ± 1 SD over seeds") for f in fractions]
    if band_drawn:
        handles.append(Patch(facecolor=style.ORIGINAL, alpha=0.4,
                             label="control (cf = 0): mean ± 1 SD"))
    style.legend_below(fig, handles, ncol=2)
    return fig


def seed_spread(track: str, exp: str = "exp1", metric: str | None = None):
    """Every configuration's seeds on one axis, best configuration on top.

    Points are seeds, coloured and shaped by credit fraction; the thin line their range; the black
    tick their mean; the dashed line the control's mean. A difference between two configurations is
    only worth reading where it clearly exceeds the spread among one configuration's own seeds.
    """
    metric = metric or HEADLINE[track]
    runs, agg, kept, _ = _view(track, exp, metric)
    style.apply()
    finals = _finals(agg, metric, kept)
    by_config: dict[str, list[float]] = {}
    for name, v in finals.items():
        by_config.setdefault(_config_of(name), []).append(v)
    if not by_config:
        return _placeholder("no finished arms to compare yet")
    higher = HIGHER_IS_BETTER.get(metric, True)
    # Best first; the axis is inverted below, so the best configuration is the top row.
    configs = sorted(by_config, key=lambda c: np.mean(by_config[c]), reverse=higher)
    fig, ax = plt.subplots(figsize=style.row_figsize(len(configs), per_row=0.19, base=1.35))
    for i, config in enumerate(configs):
        vals = np.array(by_config[config])
        f = _cf_value(config)
        colour = style.credit_fraction_colour(f)
        ax.plot([vals.min(), vals.max()], [i, i], color=colour, lw=1.4, alpha=0.7, zorder=2)
        ax.scatter(vals, np.full(vals.size, i), s=18, color=colour,
                   marker=style.credit_fraction_marker(f), edgecolor="white", linewidth=0.3,
                   zorder=3)
        ax.plot([vals.mean()] * 2, [i - 0.3, i + 0.3], color=style.INK, lw=1.6, zorder=4)
    control = [v for n, v in finals.items() if _is_control(n)]
    handles = _cf_handles([_cf_value(c) for c in configs]) + [
        Line2D([], [], color=style.INK, lw=1.6, label="mean over seeds")]
    if control:
        ax.axvline(float(np.mean(control)), color=style.ORIGINAL, ls="--", lw=1.1, zorder=1)
        handles.append(Line2D([], [], color=style.ORIGINAL, ls="--", lw=1.1,
                              label="control mean"))
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([_config_label(c) for c in configs], fontsize=7)
    ax.set_ylim(len(configs) - 0.5, -0.5)
    ax.set_xlabel(f"final development {style.metric_label(metric)}")
    ax.grid(axis="x")
    style.legend_below(ax, handles, ncol=3)
    return fig


# ---------------------------------------------------------------------------
# C. Where it holds — every dataset, out of domain, every metric
# ---------------------------------------------------------------------------


def _real_datasets(runs: dict[str, pd.DataFrame], metric: str) -> list[str]:
    """The real datasets that carry `metric` in any arm, in first-seen order."""
    seen: list[str] = []
    for df in runs.values():
        for d in _dataset_cols(df, metric):
            if d not in seen:
                seen.append(d)
    return seen


def per_dataset_pages(track: str, exp: str = "exp1", per_page: int = 9) -> int:
    runs = load_progress(track, exp)
    n = len(_real_datasets(runs, HEADLINE[track])) if runs else 0
    return max(1, int(np.ceil(n / per_page)))


def per_dataset_curves(track: str, page: int = 1, exp: str = "exp1", per_page: int = 9,
                       metric: str | None = None):
    """The monitored metric over training, one panel per real dataset, credit-prior arms against
    control arms (means over the finished arms that scored the dataset). Each panel names the
    dataset and its side of the split: a holdout dataset is shown, never averaged."""
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
    metric = metric or HEADLINE[track]
    roles = dataset_roles(track, exp)
    style.apply()
    datasets = _real_datasets(runs, metric) if runs else []
    if not datasets:
        return _placeholder("no per-dataset metrics logged yet")
    rank = {"development": 0, "holdout": 1}
    datasets = sorted(datasets, key=lambda d: (rank.get(roles.get(d, ""), 2), d))
    chunk = datasets[(page - 1) * per_page: page * per_page]
    ncols = 2 if len(chunk) <= 4 else 3
    nrows = max(1, int(np.ceil(len(chunk) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.6),
                             squeeze=False)
    flat = axes.ravel()
    grid = _common_step_grid(agg)
    short = {"development": "dev", "holdout": "holdout"}
    for ax, ds in zip(flat, chunk):
        n_arms = 0
        for is_control, colour in ((False, style.CREDIT), (True, style.ORIGINAL)):
            stack = []
            for name, df in agg.items():
                col = _dataset_cols(df, metric).get(ds)
                if col is None or _is_control(name) is not is_control:
                    continue
                d = df[["step", col]].dropna()
                if len(d) >= 2:
                    stack.append(np.interp(grid, d["step"], d[col], left=np.nan, right=np.nan))
            if stack:
                ax.plot(grid, _complete(stack), color=colour, lw=1.6)
            n_arms += len(stack)
        # Name, side of the split, and how many finished arms scored it — short enough for one line.
        ax.set_title(f"{ds[:18]} · {short.get(roles.get(ds, ''), 'no role')} · {n_arms} arms",
                     loc="left", fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in flat[len(chunk):]:
        ax.axis("off")
    for r in range(nrows):
        axes[r][0].set_ylabel(style.metric_label(metric))
    for c in range(ncols):
        for r in range(nrows - 1, -1, -1):
            if axes[r][c].axison:
                axes[r][c].set_xlabel("training step")
                break
    style.legend_below(fig, [Line2D([], [], color=style.CREDIT, lw=1.6,
                                    label="credit-prior arms (mean of finished arms)"),
                             Line2D([], [], color=style.ORIGINAL, lw=1.6,
                                    label="control arms (mean of finished arms)")], ncol=2)
    return fig


def real_vs_ood(track: str, exp: str = "exp1", metric: str | None = None):
    """Development metric on the real-credit datasets against the out-of-domain suites, over
    training, each averaged separately over credit-prior and control arms.

    Credit arms are blue and control arms grey, as everywhere; the domain is the line style — solid
    for the real credit datasets, dashed for the out-of-domain suites. Each domain mean covers the
    datasets every finished arm scored in it. Degrades to the credit curves alone when no
    out-of-domain columns were logged.
    """
    metric = metric or HEADLINE[track]
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
    style.apply()
    if not runs:
        return _placeholder("no progress logs in output/manifests/ yet")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    grid = _common_step_grid(agg)
    for domain, label, ls in (("real", "real credit", "-"), ("ood", "out-of-domain", "--")):
        kept, _ = comparable_datasets(runs, metric, domain)
        if not kept:
            continue
        for is_control, colour in ((False, style.CREDIT), (True, style.ORIGINAL)):
            stack = [c for n, df in agg.items() if _is_control(n) is is_control
                     for c in [_curve(df, metric, kept, grid, domain)] if c is not None]
            if stack:
                who = "control arms" if is_control else "credit-prior arms"
                n_ds = f"{len(kept)} dataset{'s' if len(kept) != 1 else ''}"
                ax.plot(grid, _complete(stack), color=colour, lw=2.0, ls=ls,
                        label=f"{label} ({n_ds}), {who}")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"development {style.metric_label(metric)}")
    if not ax.get_legend_handles_labels()[0]:
        _empty(ax, "no development score logged yet")
        return fig
    style.legend_below(ax, ncol=2)
    return fig


def _themes(track: str, runs: dict[str, pd.DataFrame]) -> list[tuple[str, list[str]]]:
    """The metric themes of `track` that have at least one logged metric, each with its metrics."""
    have = set(available_metrics(runs)) if runs else set()
    out = []
    for theme, metrics in METRIC_THEMES.get(track, ()):
        present = [m for m in metrics if m in have or (m == "roc_auc" and "auc" in have)]
        if present:
            out.append((theme, present))
    return out


def metric_pages(track: str, exp: str = "exp1", per_page: int = 9) -> int:
    """How many figures `all_eval_metrics` draws: one per metric theme with data (at least one)."""
    runs = load_progress(track, exp)
    return max(1, len(_themes(track, runs)))


def all_eval_metrics(track: str, page: int = 1, exp: str = "exp1", per_page: int = 9):
    """One figure per metric theme (accuracy; ranking and calibration; intervals and boundary
    atoms), one panel per metric: credit-prior and control means over the finished arms.

    Each panel's title carries the metric's goal — `↑`, `↓`, or `→ value` for a metric that is right
    at a target (a calibration slope of 1, an 80% interval covering 80%) — and a target-valued panel
    draws that value as a dashed line. A predicted boundary mass is drawn against the TRUE share in
    the data, which is the value it should reach.
    """
    runs = load_progress(track, exp)
    agg = _aggregate(runs)
    style.apply()
    themes = _themes(track, runs)
    if not themes:
        return _placeholder("no evaluation metrics logged yet")
    theme, metrics = themes[min(max(page, 1), len(themes)) - 1]
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.62),
                             squeeze=False)
    flat = axes.ravel()
    grid = _common_step_grid(agg)
    extra: dict[str, Any] = {}
    for ax, metric in zip(flat, metrics):
        kept, _ = comparable_datasets(runs, metric)
        for is_control, colour in ((False, style.CREDIT), (True, style.ORIGINAL)):
            stack = [c for n, df in agg.items() if _is_control(n) is is_control
                     for c in [_curve(df, metric, kept, grid)] if c is not None]
            if stack:
                ax.plot(grid, _complete(stack), color=colour, lw=1.5)
        goal = style.metric_goal(metric)
        if isinstance(goal, (int, float)):
            ax.axhline(goal, color=style.INK, lw=0.9, ls="--", zorder=1)
            extra["ideal"] = Line2D([], [], color=style.INK, lw=0.9, ls="--", label="ideal value")
        if metric.startswith("pred_mass_at_"):
            truth = metric.replace("pred_", "true_")
            vals = [df[c].dropna().mean() for df in agg.values()
                    for d, c in _dataset_cols(df, truth).items() if d in kept]
            if vals:
                ax.axhline(float(np.nanmean(vals)), color=style.REAL, lw=1.0, ls="--", zorder=1)
                extra["truth"] = Line2D([], [], color=style.REAL, lw=1.0, ls="--",
                                        label="true share in the data")
        ax.set_title(f"{style.metric_label(metric)} {style.goal_mark(metric)}".strip(), loc="left",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in flat[len(metrics):]:
        ax.axis("off")
    for c in range(ncols):
        for r in range(nrows - 1, -1, -1):
            if axes[r][c].axison:
                axes[r][c].set_xlabel("training step", fontsize=7)
                break
    fig.suptitle(theme, fontsize=10)
    handles = [Line2D([], [], color=style.CREDIT, lw=1.5, label="credit-prior arms (mean)"),
               Line2D([], [], color=style.ORIGINAL, lw=1.5, label="control arms (mean)")]
    style.legend_below(fig, handles + list(extra.values()), ncol=4)
    return fig


# ---------------------------------------------------------------------------
# D. Every arm
# ---------------------------------------------------------------------------


def _configs(track: str, exp: str, runs: dict[str, pd.DataFrame]) -> list[str]:
    names = _sweep_grid(track, exp) or sorted(runs)
    return sorted({_config_of(n) for n in names}, key=lambda c: _config_sort_key(c, exp))


def config_pages(track: str, exp: str = "exp1", per_page: int = 15) -> int:
    runs = load_progress(track, exp)
    return max(1, int(np.ceil(len(_configs(track, exp, runs)) / per_page))) if runs else 1


def per_config(track: str, page: int = 1, exp: str = "exp1", per_page: int = 15):
    """One panel per CONFIGURATION, its seeds as separate lines of the development score, all
    panels on shared axes — so an arm that behaved unlike its neighbours, or unlike its own other
    seeds, stands out. Panels follow the sweep map's order; an unfinished arm's line stops where
    its training has got to."""
    runs = load_progress(track, exp)
    metric = HEADLINE[track]
    style.apply()
    if not runs:
        return _placeholder("no arms have started yet")
    kept, _ = comparable_datasets(runs, metric)
    configs = _configs(track, exp, runs)
    chunk = configs[(page - 1) * per_page: page * per_page]
    seeds = sorted({s for s in (_seed_of(n) for n in runs) if s is not None}) or [None]
    ncols = 3
    nrows = max(1, int(np.ceil(len(chunk) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.52),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    target = _target_step(runs)
    dashes = ["-", "--", ":", "-."]
    drawn_seeds: set = set()
    for ax, config in zip(flat, chunk):
        colour = style.credit_fraction_colour(_cf_value(config))
        any_line = False
        for k, seed in enumerate(seeds):
            name = f"{config}__s{seed}" if seed is not None else config
            if name not in runs:
                continue
            d = pd.DataFrame({"step": runs[name]["step"],
                              "m": _mean_metric(runs[name], metric, "real", kept)}).dropna()
            if len(d) >= 2:
                ax.plot(d["step"], d["m"], color=colour, ls=dashes[k % len(dashes)], lw=1.1)
                any_line = True
                drawn_seeds.add(seed)
        if not any_line:
            ax.text(0.5, 0.5, "no development score", transform=ax.transAxes, ha="center",
                    va="center", fontsize=6.5, color=style.MUTED)
        ax.set_title(_config_label(config), loc="left", fontsize=7, pad=2)
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(0, max(target, 1))
    for ax in flat[len(chunk):]:
        ax.axis("off")
    for r in range(nrows):
        axes[r][0].set_ylabel(style.metric_label(metric), fontsize=7)
    for c in range(ncols):
        for r in range(nrows - 1, -1, -1):
            if axes[r][c].axison:
                axes[r][c].set_xlabel("training step", fontsize=7)
                axes[r][c].tick_params(labelbottom=True)
                break
    shown = [(k, s) for k, s in enumerate(seeds) if s in drawn_seeds and s is not None]
    if shown:
        style.legend_below(fig, [Line2D([], [], color=style.INK, ls=dashes[k % len(dashes)], lw=1.1,
                                        label=f"seed {s}") for k, s in shown], ncol=len(shown))
    return fig


def best_and_worst(track: str, exp: str = "exp1"):
    """The best and worst finished arm by final development score: training loss (rolling mean of
    three logged batches) and development score side by side."""
    runs = load_progress(track, exp)
    metric = HEADLINE[track]
    kept, _ = comparable_datasets(runs, metric)
    ranked = rank_arms(runs, track)
    style.apply()
    if len(ranked) < 2:
        return _placeholder("need at least two finished arms with a development score")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.42))
    picks = [(ranked[0][0], style.CREDIT_STRONG, f"best, {ranked[0][1]:.3f}"),
             (ranked[-1][0], style.WARN, f"worst, {ranked[-1][1]:.3f}")]
    for name, colour, tag in picks:
        d = runs[name].dropna(subset=["train_loss"])
        ax1.plot(d["step"], _smooth(d["train_loss"]), color=colour, lw=1.8)
        m = pd.DataFrame({"step": runs[name]["step"],
                          "m": _mean_metric(runs[name], metric, "real", kept)}).dropna()
        ax2.plot(m["step"], m["m"], color=colour, lw=1.8, label=f"{tag}: {arm_label(name)}")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("training loss (own synthetic batch)")
    ax1.set_ylim(bottom=0)
    ax1.set_title("Training loss", loc="left", fontsize=9)
    ax2.set_xlabel("training step")
    ax2.set_ylabel(f"development {style.metric_label(metric)}")
    ax2.set_title(f"Development {style.metric_label(metric)}", loc="left", fontsize=9)
    style.legend_below(fig, ncol=2)
    return fig


def hardware(track: str, exp: str = "exp1"):
    """GPU utilisation, throughput and peak memory over the run, one faint line per arm."""
    tel = load_telemetry(track, exp)
    style.apply()
    if not tel:
        return _placeholder("no telemetry logs found")
    panels = [("gpu0_utilization_gpu", "GPU utilisation (%)", (0, 100)),
              ("steps_per_s", "steps per second", None),
              ("mem_max_allocated_gb", "peak allocated memory (GB)", None)]
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(style.WIDTH_FULL, 0.34))
    for ax, (col, label, ylim) in zip(axes, panels):
        for df in tel.values():
            if col in df:
                d = df.dropna(subset=[col])
                if len(d):
                    ax.plot(d["step"], d[col], color=style.CREDIT, alpha=0.35, lw=0.9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("training step")
        ax.set_ylabel(label, fontsize=7)
    return fig


#: The architecture's blocks as the telemetry names them, with how each reads in a legend.
_BLOCKS = (("col", "column encoder"), ("row", "row encoder"), ("icl", "ICL blocks"),
           ("head", "head"))


def _block_curves(tel: dict[str, pd.DataFrame], prefix: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for key, label in _BLOCKS:
        col = f"{prefix}_{key}"
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
            out.append((label, grid, _nanmean(stacks)))
    return out


def gradient_health(track: str, exp: str = "exp1"):
    """Is every block of the network learning? Left, each block's gradient norm; right, the same
    gradient divided by the block's weight norm — the scale-free version, where a block far below
    the others is effectively frozen. Means over every arm, on logarithmic axes.

    Under an Exp2 freeze strategy (`icl_only`, `head_only`) the frozen blocks legitimately sit on
    the floor: this figure is where a reader confirms the freeze took.
    """
    tel = load_telemetry(track, exp)
    style.apply()
    grads, ratios = _block_curves(tel, "grad"), _block_curves(tel, "gw_ratio")
    if not grads and not ratios:
        return _placeholder("no gradient samples logged (log_grad_every = 0?)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.42))
    styles = {label: style.LEVEL_STYLES[i % len(style.LEVEL_STYLES)]
              for i, (_k, label) in enumerate(_BLOCKS)}
    for ax, curves, ylabel, title in ((ax1, grads, "gradient norm", "Gradient norm"),
                                      (ax2, ratios, "gradient ÷ weight norm",
                                       "Gradient-to-weight ratio")):
        for label, grid, mean in curves:
            colour, ls = styles[label]
            ax.plot(grid, mean, color=colour, ls=ls, lw=1.5, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel(f"{ylabel} (log)")
        ax.set_title(title, loc="left", fontsize=9)
    style.legend_below(fig, ncol=4)
    return fig


def gradient_flow(track: str, exp: str = "exp1"):
    """Per-block gradient norm over training, mean over arms, on a logarithmic axis."""
    tel = load_telemetry(track, exp)
    style.apply()
    curves = _block_curves(tel, "grad")
    if not curves:
        return _placeholder("no gradient-norm samples logged (log_grad_every = 0?)")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    for i, (label, grid, mean) in enumerate(curves):
        colour, ls = style.LEVEL_STYLES[i % len(style.LEVEL_STYLES)]
        ax.plot(grid, mean, color=colour, ls=ls, lw=1.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient L2 norm (log scale)")
    style.legend_below(ax, ncol=4)
    return fig


def weight_gradient_ratios(track: str, exp: str = "exp1"):
    """Per-block gradient-to-weight ratio over training, mean over arms — the scale-free "is it
    learning?"."""
    tel = load_telemetry(track, exp)
    style.apply()
    curves = _block_curves(tel, "gw_ratio")
    if not curves:
        return _placeholder("no gradient-to-weight ratios logged (log_grad_every = 0?)")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.46))
    for i, (label, grid, mean) in enumerate(curves):
        colour, ls = style.LEVEL_STYLES[i % len(style.LEVEL_STYLES)]
        ax.plot(grid, mean, color=colour, ls=ls, lw=1.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm ÷ weight norm (log scale)")
    style.legend_below(ax, ncol=4)
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
    keys = _value_order(lever, groups)
    return [(k, float(np.mean(groups[k])), float(np.std(groups[k])), len(groups[k])) for k in keys]


def training_summary(track: str, exp: str = "exp1") -> str:
    """A text summary of the sweep, section by section in the notebook's order: the run, the
    answer, where it holds."""
    runs = load_progress(track, exp)
    tel = load_telemetry(track, exp)
    title = f"{exp.upper()} {track.upper()} TRAINING — development-split monitoring"
    if not runs:
        return f"{title}\n  no progress logs in output/manifests/ yet."
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
        by_seed: dict[Any, int] = {}
        for n in missing:
            by_seed[_seed_of(n)] = by_seed.get(_seed_of(n), 0) + 1
        seeds = ", ".join(f"seed {s}: {c}" for s, c in sorted(by_seed.items(), key=lambda kv: str(kv[0])))
        lines.append(f"  finished arms without a development score: {len(missing)} of {len(agg)}"
                     f" ({seeds}) — a monitor that logged only missing values")
    old = sum("progress_protocol" not in df.columns for df in runs.values())
    if old:
        lines.append(f"  arms monitored before the development-only protocol: {old} of {len(runs)}")
    ends = sorted(int(df["step"].max()) for n, df in runs.items() if n in agg)
    starts = sorted(int(df["step"].min()) for n, df in runs.items() if n in agg)
    if ends and (ends[0] < ends[-1] or starts[0] < starts[-1]):
        lines.append(f"  group curves are drawn only where all their arms have a value: finished arms'"
                     f" logs start at step {starts[0]:,}-{starts[-1]:,} and end at {ends[0]:,}-{ends[-1]:,}")
    lever = "filter" if exp == "exp1" else "strategy"
    speeds: dict[str, list[float]] = {}
    utils: list[float] = []
    mems: list[float] = []
    for name, df in tel.items():
        if "steps_per_s" in df:
            s = pd.to_numeric(df["steps_per_s"], errors="coerce")
            s = s[np.isfinite(s) & (s > 0)]
            if len(s) and _lever_value(name, lever):
                speeds.setdefault(_lever_value(name, lever), []).append(float(s.median()))
        if "gpu0_utilization_gpu" in df:
            u = pd.to_numeric(df["gpu0_utilization_gpu"], errors="coerce").dropna()
            if len(u):
                utils.append(float(u.median()))
        if "mem_max_allocated_gb" in df:
            m = pd.to_numeric(df["mem_max_allocated_gb"], errors="coerce").dropna()
            if len(m):
                mems.append(float(m.max()))
    if speeds:
        parts = [f"{k} {np.median(speeds[k]):.2f} steps/s (~{target / np.median(speeds[k]) / 3600:.0f} h)"
                 for k in _value_order(lever, speeds)]
        lines.append(f"  median speed by {lever}: " + " | ".join(parts))
    if utils:
        lines.append(f"  GPU utilisation, median per arm: {np.median(utils):.0f}% "
                     f"(range {min(utils):.0f}-{max(utils):.0f}%)")
    if mems:
        lines.append(f"  peak allocated GPU memory: {min(mems):.1f}-{max(mems):.1f} GB per arm")
    losses: dict[float, list[float]] = {}
    for name, df in agg.items():
        d = df.dropna(subset=["train_loss"])
        if len(d):
            losses.setdefault(_cf_value(name), []).append(float(d["train_loss"].tail(3).mean()))
    if losses:
        lines.append("  final training loss by credit fraction (each on its own prior's tasks): "
                     + " | ".join(f"cf {f:g} {np.median(v):.3f}" for f, v in sorted(losses.items())))

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
        # Which seed tops each configuration — a shared-seed effect shows up as one seed winning
        # far more often than its share. The seed sets the initialisation AND the monitor's rows.
        tops: dict[Any, int] = {}
        scored = 0
        for config in by_config:
            seeds = {_seed_of(n): v for n, v in finals.items() if _config_of(n) == config}
            if len(seeds) > 1:
                scored += 1
                best = max(seeds, key=seeds.get) if HIGHER_IS_BETTER.get(metric, True) else min(seeds, key=seeds.get)
                tops[best] = tops.get(best, 0) + 1
        if tops:
            lines.append("  best seed within a configuration: " + ", ".join(
                f"seed {s} in {c} of {scored}" for s, c in sorted(tops.items(), key=lambda kv: -kv[1])))
    elif by_config:
        lines.append("  seed noise: not measurable — one scored seed per configuration")

    lines += ["", "C. WHERE IT HOLDS"]
    ood_kept, _ = comparable_datasets(runs, metric, "ood")
    if ood_kept:
        ood = _finals(agg, metric, ood_kept, "ood")
        both = [n for n in finals if n in ood]
        if both:
            for grp, members in (("credit arms", [n for n in both if not _is_control(n)]),
                                 ("control arms", [n for n in both if _is_control(n)])):
                if members:
                    lines.append(
                        f"  {grp:<12} ({len(members)}): real credit {np.mean([finals[n] for n in members]):.4f}"
                        f" | out-of-domain ({len(ood_kept)} suites) {np.mean([ood[n] for n in members]):.4f}")
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
                      metric_series=None, n: int = 80) -> np.ndarray:
    """A shared step axis spanning every given arm, first log to last. Each arm is interpolated onto
    it and is missing outside its own span; `_complete` then draws a group statistic only where
    EVERY arm of the group has a value."""
    spans = [(float(df["step"].min()), float(df["step"].max())) for df in runs.values() if len(df)]
    if not spans:
        return np.linspace(0.0, 1.0, n)
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    return np.linspace(lo, max(hi, lo + 1.0), n)


def _complete(stack, fn=np.mean) -> np.ndarray:
    """A statistic over arms at each step, drawn ONLY where every arm of the stack has a value.

    Averaging where some arms are missing changes the composition of the average, and the curve
    jumps: finished arms log their last point anywhere between ~11,900 and 12,500 steps (a requeue
    restarts the progress schedule) and the training-loss mean dropped at its last point for
    exactly that reason; three PD banded arms' logs start only at steps 3,251-4,001, because their
    progress files were restarted under the development-only protocol. So a group's curve spans
    the steps its arms share — it starts late or stops early instead of jumping.
    """
    arr = np.array(stack, dtype=float)
    if arr.ndim != 2 or not arr.size:
        return np.array([])
    out = np.full(arr.shape[1], np.nan)
    ok = np.isfinite(arr).all(axis=0)
    if ok.any():
        out[ok] = fn(arr[:, ok], axis=0)
    return out


def _complete_band(stack, q=(25, 75)) -> tuple[np.ndarray, np.ndarray]:
    """The inter-quartile band over arms, on the same all-arms-present rule as `_complete`."""
    arr = np.array(stack, dtype=float)
    lo, hi = np.full(arr.shape[1], np.nan), np.full(arr.shape[1], np.nan)
    ok = np.isfinite(arr).all(axis=0)
    if ok.any():
        lo[ok], hi[ok] = np.percentile(arr[:, ok], q, axis=0)
    return lo, hi


def _best_worst_pair(ranked: list[tuple[str, float]]):
    if len(ranked) >= 2:
        return [(ranked[0][0], style.CREDIT_STRONG, "best"), (ranked[-1][0], style.WARN, "worst")]
    return []


def _empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=style.MUTED,
            fontsize=9, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
