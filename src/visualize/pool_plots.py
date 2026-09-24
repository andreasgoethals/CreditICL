"""Inspect pre-generated prior pools — however many of them exist.

WHY THIS IS ONE MODULE AND NOT ONE NOTEBOOK PER VARIANT

The interesting question is never "what does `credit_v1` look like" on its own. It is
always "how does `credit_v1` differ from `original`, and from `credit_v2`". A notebook
per variant answers the uninteresting question and makes the interesting one manual:
you end up flipping between saved outputs and comparing histograms from memory.

So instead: the pools on disk are **discovered**, and every comparison plot takes a
list of variants. Adding `credit_v2` means generating it — the notebook needs no edit.

Two kinds of plot, and the distinction matters:

* **comparison** plots put all variants on one axis (boundary mass, base rate,
  correlation spectrum, summary table). These are the ones you actually reason with.
* **detail** plots only make sense one variant at a time — you cannot show 100
  histograms for four variants at once. These take a single `focus` variant.

READS POOLS, NOT A FRESH DRAW. `prior_plots.sample_tasks` generates new datasets from
a config, which answers "what would this config produce". This module reads the files
training actually consumed, which answers "what did the model actually see". When you
have pools, that is strictly better evidence.

PARTIAL DOWNLOADS ARE FINE, AND ARE THE POINT. A full pool is ~4 GB (LGD) to ~5.4 GB
(PD) per variant. Every plot here needs a few hundred datasets, so **one shard**
(~200-270 MB) is 20x more than enough. `PoolReader` globs whatever shards are present,
so copying `shard_00000.*` from the cluster just works — and `describe_pools` labels
such a pool a SAMPLE so a partial download can never be mistaken for the whole thing.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.prior.base import SyntheticTask
from src.prior.pool import POOL_VERSION, PoolReader, variant_dir
from src.prior.rng import PriorRNG
from src.utils.paths import prior_cache_dir
from src.utils.target_stats import target_stats
from src.visualize import literature, style

#: Variants are drawn in this order when present, so `original` is always the
#: leftmost/greyest reference rather than landing wherever the filesystem put it.
PREFERRED_ORDER = ["original", "credit_v1", "credit_v2", "credit_v3"]


# ---------------------------------------------------------------------------
# Discovery — what have I actually got on this machine?
# ---------------------------------------------------------------------------


def discover_pools(task: str) -> list[str]:
    """Variant names with at least one readable shard, in a sensible order.

    Globs the pool root rather than taking a hard-coded list, so a new variant shows
    up in the notebook the moment it is generated or downloaded.
    """
    root = prior_cache_dir("x").parent
    if not root.is_dir():
        return []
    found = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(f"{task}__"):
            continue
        if any(d.glob("shard_*.pt")):
            found.append(d.name.split("__", 1)[1])
    ranked = [v for v in PREFERRED_ORDER if v in found]
    return ranked + sorted(v for v in found if v not in ranked)


def describe_pools(task: str, variants: list[str] | None = None) -> pd.DataFrame:
    """What is on disk, and whether it is the whole pool or a sample.

    Deliberately counts the `.pt` files itself instead of trusting the manifests. A
    download that brought the payloads but not the JSON would otherwise report zero
    datasets while the plots worked fine — confusing in exactly the wrong way.
    """
    variants = variants if variants is not None else discover_pools(task)
    rows = []
    for variant in variants:
        d = variant_dir(task, variant)
        shards = sorted(d.glob("shard_*.pt"))
        expected, counted, cfrac, valid = None, 0, None, 0
        for shard in shards:
            manifest = shard.with_suffix(".json")
            if not manifest.is_file():
                continue
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — a torn manifest is just unknown
                continue
            if m.get("pool_version") != POOL_VERSION:
                continue
            valid += 1
            counted += int(m.get("n_datasets", 0))
            expected = int(m.get("n_shards", 0)) or expected
            cfrac = m.get("credit_fraction", cfrac)

        # COMPLETE requires every shard to have a readable, current-version manifest —
        # not merely the right number of `.pt` files. Counting payloads alone let a
        # pool with one stale-layout shard report COMPLETE while silently
        # under-counting its datasets, which is the one thing this table exists to
        # prevent.
        complete = expected is not None and valid == expected == len(shards)
        rows.append(
            {
                "variant": variant,
                "shards": len(shards),
                "shards_expected": expected if expected is not None else "?",
                "datasets": counted if counted else "? (no manifests)",
                "credit_fraction": cfrac,
                "state": "COMPLETE" if complete else "SAMPLE",
                "size_MB": round(sum(s.stat().st_size for s in shards) / 1e6, 1),
                "path": str(d),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_variant(task: str, variant: str, n: int = 100, seed: int = 0) -> list[SyntheticTask]:
    """Draw `n` episodes from a pool as `SyntheticTask` objects.

    Returning the same type the generator returns is what lets every function in
    `prior_plots` work on pooled data with no changes.
    """
    reader = PoolReader(task, variant)
    rng = PriorRNG(seed)
    return [
        SyntheticTask(X=ep["X"], y=ep["y"], source=ep.get("source", "base"))
        for ep in (reader.sample(rng) for _ in range(n))
    ]


def load_all_variants(
    task: str, variants: list[str] | None = None, n: int = 100, seed: int = 0
) -> dict[str, list[SyntheticTask]]:
    """{variant: tasks} for every pool present. Skips unreadable pools with a note."""
    variants = variants if variants is not None else discover_pools(task)
    out: dict[str, list[SyntheticTask]] = {}
    for v in variants:
        try:
            out[v] = load_variant(task, v, n=n, seed=seed)
        except Exception as exc:  # noqa: BLE001 — a notebook should keep going
            print(f"  skipped {task}/{v}: {type(exc).__name__}: {exc}")
    return out


def load_variants_or_generate(
    task: str, n: int = 100, seed: int = 0, config: str | None = None
) -> tuple[dict[str, list[SyntheticTask]], str]:
    """Pools if any exist, otherwise generate the two arms live.

    Returns `(loaded, source)` where source is "pool" or "live". This exists so the
    notebook works on a machine with nothing downloaded yet — the alternative was an
    empty dict flowing into every plot and failing with matplotlib's unhelpful
    "Number of rows must be a positive integer, not 0".

    Live variants are named plainly, `original` and `credit`; where they came from is the
    returned `source`, which the notebook prints once and `summaries.prior_summary` reports.
    They used to be called `original (live)` / `credit (live)`, which put the provenance into
    every legend and tick label, and broke the colour lookup keyed on the variant's name.
    """
    variants = discover_pools(task)
    if variants:
        loaded = load_all_variants(task, variants, n=n, seed=seed)
        if loaded:
            return loaded, "pool"

    from src.visualize.prior_plots import sample_tasks

    # Exp1's config, because all three experiments on a track name the SAME prior_file — so
    # any of them describes the prior these figures visualise, and Exp1 is the one that is
    # never a template. Exp2/Exp3 still hold `FILL_FROM_EXP1` and would refuse to load.
    cfg = config or f"config/Exp1_{task.upper()}.yaml"
    print(f"no pools found for {task} — generating {n} datasets per arm live from {cfg}")
    original, _ = sample_tasks(cfg, n=n, credit_fraction=0.0, seed=seed)
    ours, _ = sample_tasks(cfg, n=n, credit_fraction=1.0, seed=seed)
    return {"original": original, "credit": ours}, "live"


def _require_variants(loaded: dict[str, list[SyntheticTask]]) -> None:
    """Fail with the fix, not with a matplotlib internals error."""
    if not loaded:
        raise ValueError(
            "no variants to plot. Either generate a pool:\n"
            "  python scripts/generate_prior.py --config config/Exp1_LGD.yaml --variant original --all\n"
            "or copy one from the cluster:\n"
            "  bash scripts/transfer/fetch_prior_sample.sh\n"
            "or use load_variants_or_generate(), which falls back to live generation."
        )


def variant_color(variant: str, index: int = 0) -> str:
    """`original` is always the grey reference, our prior always blue; see `style.variant_colour`."""
    return style.variant_colour(variant, index)


def informative_columns(X: Any) -> np.ndarray:
    """Indices of the columns that actually vary.

    Every generated table is ZERO-PADDED to `max_features` (100) columns — a draw with 68 real
    features carries 32 all-zero ones — so `n_features` is 100 for every task and says nothing,
    and a correlation spectrum over the padded matrix is mostly the padding's zero eigenvalues.
    """
    X = np.asarray(X, dtype=float)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(X, axis=0)
    return np.flatnonzero(np.isfinite(sd) & (sd > 1e-12))


# ---------------------------------------------------------------------------
# The summary table — usually the first and last thing you look at
# ---------------------------------------------------------------------------


def variant_summary(loaded: dict[str, list[SyntheticTask]], task: str) -> pd.DataFrame:
    """One row per variant, with the numbers the research question turns on.

    For LGD that is boundary mass and whether the target is genuinely inside [0,1];
    for PD it is the base rate. Reporting the wrong set for the task is how a
    meaningless number ends up in a paper, so the columns switch on `task`.
    """
    _require_variants(loaded)
    rows = []
    for variant, tasks in loaded.items():
        stats = [target_stats(t.y) for t in tasks]
        informative = [len(informative_columns(t.X)) for t in tasks]
        rec: dict[str, Any] = {
            "variant": variant,
            "n_sampled": len(tasks),
            "rows_median": int(np.median([t.n_rows for t in tasks])),
            # Columns that vary, not the padded width: every table is padded to 100 columns.
            "features_median": int(np.median(informative)),
            "features_range": f"{min(informative)}-{max(informative)}",
        }
        if task == "lgd":
            boundary = np.array([s["frac_at_min"] + s["frac_at_max"] for s in stats])
            in_unit = np.array([
                float(t.y.min()) >= -1e-6 and float(t.y.max()) <= 1 + 1e-6 for t in tasks
            ])
            rec.update(
                {
                    "in [0,1]": round(float(in_unit.mean()), 3),
                    "boundary mass mean": round(float(boundary.mean()), 4),
                    "boundary p10": round(float(np.percentile(boundary, 10)), 4),
                    "boundary p90": round(float(np.percentile(boundary, 90)), 4),
                    "any atoms": round(float((boundary > 0.01).mean()), 3),
                }
            )
        else:
            rates = np.array([float((t.y > 0.5).float().mean()) for t in tasks])
            rec.update(
                {
                    "base rate mean": round(float(rates.mean()), 4),
                    "base rate p10": round(float(np.percentile(rates, 10)), 4),
                    "base rate p90": round(float(np.percentile(rates, 90)), 4),
                    "below 5%": round(float((rates < 0.05).mean()), 3),
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison plots — all variants on one axis
# ---------------------------------------------------------------------------


def plot_boundary_mass_by_variant(loaded: dict[str, list[SyntheticTask]], real_reference=None):
    """Boundary-mass distribution per variant, plus the real datasets as a target.

    The single most important figure for LGD. Read it as: does any variant's cloud
    actually cover where the real datasets sit?
    """
    _require_variants(loaded)
    style.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.38))

    for i, (variant, tasks) in enumerate(loaded.items()):
        stats = [target_stats(t.y) for t in tasks]
        boundary = np.array([s["frac_at_min"] + s["frac_at_max"] for s in stats])
        colour = variant_color(variant, i)
        # Step histograms, not filled bars: four filled histograms hide each other.
        ax1.hist(boundary, bins=30, histtype="step", lw=2.2, color=colour,
                 label=f"{style.variant_label(variant)} (mean {boundary.mean():.3f})")
        ax2.scatter(
            [s["frac_at_min"] for s in stats], [s["frac_at_max"] for s in stats],
            s=26, alpha=0.5, color=colour, edgecolor="none", label=style.variant_label(variant),
        )

    if real_reference:
        for name, (m0, m1) in real_reference.items():
            ax2.scatter([m0], [m1], marker="*", s=320, color=style.REAL,
                        edgecolor="white", linewidth=0.8, zorder=6)
            ax2.annotate(name, (m0, m1), fontsize=8, color=style.REAL, weight="semibold",
                         xytext=(7, 5), textcoords="offset points")
            ax1.axvline(m0 + m1, color=style.REAL, lw=1, ls=":", alpha=0.7)
        ax2.scatter([], [], marker="*", s=200, color=style.REAL, label="real datasets")

    ax1.set_xlabel("total boundary mass")
    ax1.set_ylabel("number of tasks")
    ax1.legend()
    style.title(ax1, "Total boundary mass")

    ax2.set_xlabel("mass at the low boundary")
    ax2.set_ylabel("mass at the high boundary")
    ax2.grid(axis="x")
    ax2.legend()
    style.title(ax2, "Mass at 0 vs at 1")
    return fig


def plot_base_rate_by_variant(loaded: dict[str, list[SyntheticTask]], real_reference=None):
    """Base-rate distribution per variant, with the real datasets on their OWN strip.

    TWO PANELS SHARING THE X AXIS, because one axis could not hold both. Every previous
    attempt collided: 14 rotated dataset names smeared into each other; replacing them with a
    thin rug made the real data almost invisible AND put its summary label across the
    histogram bars; and the 50% reference line ran through the legend.
    None of that is fixable by nudging positions — the histogram needs the vertical space and
    the real datasets need somewhere that is not the histogram. So they get a strip of their
    own, and nothing can overlap by construction.
    """
    _require_variants(loaded)
    style.apply()
    has_real = bool(real_reference)
    fig, axes = plt.subplots(
        2 if has_real else 1, 1,
        figsize=style.figsize(style.WIDTH_FULL, 0.52),
        sharex=True,
        # The strip needs only enough height for one row of dots.
        gridspec_kw={"height_ratios": [5, 1]} if has_real else None,
        squeeze=False,
    )
    ax = axes[0][0]

    bins = np.linspace(0.0, 1.0, 41)
    for i, (variant, tasks) in enumerate(loaded.items()):
        rates = np.array([float((t.y > 0.5).float().mean()) for t in tasks])
        ax.hist(rates, bins=bins, histtype="step", lw=1.8, color=variant_color(variant, i),
                label=f"{style.variant_label(variant)} (median {np.median(rates):.0%})")
    ax.axvline(0.5, color=style.MUTED, lw=1.0, ls=":", zorder=1, label="balance (50%)")
    # Below ~10% a default-threshold classifier collapses to the majority class on real credit
    # data (Tanna 2026). In the LEGEND: a rotated label beside the line was written across the bars.
    literature.line(ax, "tanna_paradox", label="Tanna: collapse below 10%", inline=False)
    ax.set_xlim(0, 1)
    ax.set_ylabel("number of tasks")
    # Legend OUTSIDE the axes, above it. Inside, it either sat on the bars or on the 50% line
    # depending on where the data happened to fall — which is a bug that reappears with new
    # data rather than one you can fix once.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=7, frameon=False)

    if has_real:
        strip = axes[1][0]
        rates = np.asarray(list(real_reference.values()), dtype=float)
        strip.scatter(rates, np.zeros_like(rates), marker="|", s=260, linewidths=1.4,
                      color=style.STAR, clip_on=False)
        strip.set_ylim(-0.5, 0.5)
        strip.set_yticks([0])
        # The row is labelled on the AXIS, so no floating text can drift onto anything.
        strip.set_yticklabels(["real"], fontsize=7, color=style.STAR)
        strip.set_xlabel("default rate per dataset")
        strip.grid(visible=False)
        for side in ("left", "right", "top"):
            strip.spines[side].set_visible(False)
        strip.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    else:
        ax.set_xlabel("default rate per dataset")

    # NO heading. The legend already occupies the space above the axes, and a title there
    # collides with it — as it did. The caption names the figure, which is the policy anyway:
    # a heading that repeats the caption is ink for nothing.
    return fig


def shape_pages(n_per: int = 5) -> int:
    """How many figures `plot_target_shapes_by_variant` needs for `n_per` draws per variant.

    Five draws fit one figure; more are split over pages, because the constraint is the panel
    WIDTH — ten panels across the page are thumbnails however tall the figure is.
    """
    return max(1, len(style.paginate(list(range(n_per)))))


def plot_target_shapes_by_variant(
    loaded: dict[str, list[SyntheticTask]], n_per: int = 5, page: int = 1
):
    """The target of individual tasks, one row per variant: what a single draw actually looks like.

    Every panel is one task's target histogram, with its axes: the share of rows on y, the
    target's own values on x. An unbounded target (the original prior standard-scales its target)
    gets its own range and its two end values as ticks; a target on [0, 1] gets 0, 0.5 and 1, so
    the boundary atoms sit visibly on the edges. The rows show the same draw indices, so a column
    compares the two priors on the same random seed.

    Axes were missing before — no ticks, no labels — so a panel could not say where 0 and 1 were
    or how tall an atom was, which is the whole content of the figure.
    """
    from matplotlib.ticker import PercentFormatter

    _require_variants(loaded)
    style.apply()
    pages = style.paginate(list(range(n_per)))
    n_pages = max(1, len(pages))
    if not 1 <= page <= n_pages:
        raise ValueError(f"page {page} out of range; {n_pages} page(s) for n_per={n_per}")
    columns = pages[page - 1]
    n_var, n_cols = len(loaded), len(columns)
    fig, axes = plt.subplots(
        n_var, n_cols,
        figsize=style.grid_figsize(n_cols, n_var, panel_ratio=0.95), squeeze=False,
    )
    small = mpl.rcParams["xtick.labelsize"] * 0.85
    for r, (variant, tasks) in enumerate(loaded.items()):
        colour = variant_color(variant, r)
        bounded_row = True
        for c, draw in enumerate(columns):
            ax = axes[r][c]
            # `draw` indexes the FULL set of draws, not this page — so page 2 shows draws
            # 5..9 rather than repeating 0..4 with different data.
            if draw >= len(tasks):
                ax.axis("off")
                continue
            y = np.asarray(tasks[draw].y, dtype=float).ravel()
            lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
            bounded = lo >= -1e-6 and hi <= 1 + 1e-6
            bounded_row &= bounded
            edges = np.linspace(0.0, 1.0, 26) if bounded else np.linspace(lo, hi + 1e-9, 26)
            ax.hist(y, bins=edges, weights=np.full(y.size, 1.0 / max(y.size, 1)),
                    color=colour, linewidth=0)
            if bounded:
                ax.set_xlim(-0.03, 1.03)
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0", "0.5", "1"])
            else:
                # Two significant figures, not one decimal: a narrow standardised range printed
                # with "%.1f" came out as the useless pair "-0.0" and "0.0".
                ax.set_xticks([lo, hi])
                ax.set_xticklabels([f"{lo:.2g}", f"{hi:.2g}"])
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.tick_params(labelsize=small)
            ax.grid(axis="y")
        axes[r][0].set_title(style.variant_label(variant), loc="left", fontsize=9)
        axes[r][0].set_ylabel("share of rows", fontsize=small)
        axes[r][n_cols // 2].set_xlabel(
            "LGD target (0 = full recovery, 1 = total loss)" if bounded_row
            else "target, on its own standardised scale", fontsize=small)
    if n_pages > 1:
        fig.suptitle(f"Target of single tasks{style.page_suffix(page, n_pages)}")
    return fig


def _spectrum(X: Any, grid: np.ndarray, *, max_cols: int = 100, max_rows: int = 2000,
              seed: int = 0) -> np.ndarray | None:
    """One table's correlation spectrum on `grid`: eigenvalues over the largest, against rank over
    the number of columns — computed on the columns that VARY, so zero padding does not pose as
    independent features. A table wider than `max_cols` is reduced to a random `max_cols` of its
    columns, so real tables are measured at the width the prior generates."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_rows:
        X = X[rng.choice(X.shape[0], size=max_rows, replace=False)]
    cols = informative_columns(X)
    if cols.size > max_cols:
        cols = np.sort(rng.choice(cols, size=max_cols, replace=False))
    if cols.size < 2:
        return None
    X = np.nan_to_num(X[:, cols], nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.corrcoef(X, rowvar=False))
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = np.clip(ev / max(ev[0], 1e-9), 0.0, 1.0)
    return np.interp(grid, np.arange(1, len(ev) + 1) / len(ev), ev)


def plot_spectrum_by_variant(loaded: dict[str, list[SyntheticTask]], n_curves: int = 40,
                             real: dict[str, Any] | None = None):
    """Feature-correlation spectra: each prior against the real datasets, medians drawn bold.

    O'Prior's central measurement. A spectrum that falls steeply means a few directions carry most
    of the variance — strongly dependent features; a flat one means near-independent columns. Two
    priors whose medians coincide teach a similar dependence structure however different their
    targets look, and the real datasets say which of them is the realistic one.

    Measured on the columns that vary: every generated table is zero-padded to 100 columns, and the
    padding's zero eigenvalues used to pull every synthetic spectrum down to zero at a third of the
    rank axis. Real tables wider than 100 columns are measured on a random 100 of them.
    """
    _require_variants(loaded)
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.50))
    grid = np.linspace(0, 1, 60)
    series = [(style.variant_label(v), variant_color(v, i), [t.X for t in tasks])
              for i, (v, tasks) in enumerate(loaded.items())]
    if real:
        series.append(("real datasets", style.REAL, [getattr(d, "X", d) for d in real.values()]))
    for label, colour, tables in series:
        curves = [c for c in (_spectrum(X, grid) for X in tables) if c is not None]
        if not curves:
            continue
        for c in curves[:n_curves]:
            ax.plot(grid, c, color=colour, alpha=0.10 if label != "real datasets" else 0.35,
                    lw=0.7)
        ax.plot(grid, np.median(curves, axis=0), color=colour, lw=2.2,
                label=f"{label} (median of {len(curves)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("eigenvalue rank ÷ number of varying columns")
    ax.set_ylabel("eigenvalue ÷ largest eigenvalue")
    ax.grid(axis="x")
    style.legend_below(ax, ncol=3)
    return fig


#: Recorded fallbacks, measured 2026-08-06 from the processed datasets. Only used when
#: the real data is not on this machine; `real_reference` prefers a live measurement so
#: these can never quietly go stale.
RECORDED_LGD_BOUNDARY = {
    "heloc": (0.211, 0.519),
    "axa": (0.208, 0.134),
    "freddie": (0.106, 0.089),
    "lendingclub": (0.015, 0.003),
}
RECORDED_PD_BASE_RATE = {
    "gmsc": 0.0668,
    "home_credit": 0.0807,
    "hmeq": 0.1995,
    "taiwan": 0.2212,
}


def real_reference(task: str, *, quiet: bool = False) -> dict[str, Any]:
    """The real datasets' values, measured now if the data is here.

    Prefers a live measurement over the recorded constants, so the reference lines in
    the plots cannot drift away from what the datasets actually say.
    """
    try:
        from src.visualize.data_plots import load_all

        datasets = load_all(task, verbose=False)
        if not datasets:
            raise RuntimeError("no datasets loaded")
        out: dict[str, Any] = {}
        for slug, ds in datasets.items():
            name = slug.split(".", 1)[-1]
            y = np.asarray(ds.y, dtype=float)
            if task == "lgd":
                st = target_stats(y)
                out[name] = (st["frac_at_min"], st["frac_at_max"])
            else:
                out[name] = float(y.mean())
        if not quiet:
            print(f"reference: measured from {len(out)} real {task.upper()} datasets")
        return out
    except Exception as exc:  # noqa: BLE001 — the recorded values are a valid fallback
        if not quiet:
            print(f"reference: using recorded values ({type(exc).__name__}: {exc})")
        return dict(RECORDED_LGD_BOUNDARY if task == "lgd" else RECORDED_PD_BASE_RATE)


def plot_target_comparison(loaded: dict[str, list[SyntheticTask]], task: str, reference=None):
    """The key comparison figure, whichever task you are on.

    Dispatches so the notebook makes one call instead of carrying an `if` — LGD's
    question is boundary mass, PD's is the base rate, and they need different plots.
    """
    if reference is None:
        reference = real_reference(task)
    if task == "lgd":
        return plot_boundary_mass_by_variant(loaded, real_reference=reference)
    return plot_base_rate_by_variant(loaded, real_reference=reference)


def plot_shapes_by_variant(loaded: dict[str, list[SyntheticTask]]):
    """Rows and features per variant. A sanity check, mostly.

    All variants should look the SAME here: shape is not what we are changing, so a
    difference would mean an accidental confound rather than a finding.
    """
    _require_variants(loaded)
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.38))
    for i, (variant, tasks) in enumerate(loaded.items()):
        colour = variant_color(variant, i)
        axes[0].hist([t.n_rows for t in tasks], bins=20, histtype="step", lw=2.2,
                     color=colour, label=style.variant_label(variant))
        # Varying columns, not the padded width — every generated table is padded to 100.
        axes[1].hist([len(informative_columns(t.X)) for t in tasks], bins=20, histtype="step",
                     lw=2.2, color=colour, label=style.variant_label(variant))
    axes[0].set_xlabel("rows per task")
    axes[1].set_xlabel("varying features per task")
    axes[0].set_ylabel("number of tasks")
    # ONE legend, on the figure, below both panels. Two per-axes legends each carried
    # "original (live)" and "credit (live)" — long labels that overlapped the histograms and
    # each other, and said the same thing twice.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    style.title(axes[0], "Rows")
    style.title(axes[1], "Features")
    return fig
