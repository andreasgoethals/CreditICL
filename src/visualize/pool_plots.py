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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.prior.base import SyntheticTask
from src.prior.pool import POOL_VERSION, PoolReader, variant_dir
from src.prior.rng import PriorRNG
from src.utils.paths import prior_cache_dir
from src.utils.target_stats import target_stats
from src.visualize import style

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
        expected, counted, cfrac = None, 0, None
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
            counted += int(m.get("n_datasets", 0))
            expected = int(m.get("n_shards", 0)) or expected
            cfrac = m.get("credit_fraction", cfrac)

        complete = expected is not None and len(shards) == expected
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

    Live variants are labelled `... (live)` so a figure can never be mistaken for one
    made from the pools the model actually trained on.
    """
    variants = discover_pools(task)
    if variants:
        loaded = load_all_variants(task, variants, n=n, seed=seed)
        if loaded:
            return loaded, "pool"

    from src.visualize.prior_plots import sample_tasks

    cfg = config or f"config/{task.upper()}.yaml"
    print(f"no pools found for {task} — generating {n} datasets per arm live from {cfg}")
    original, _ = sample_tasks(cfg, n=n, credit_fraction=0.0, seed=seed)
    ours, _ = sample_tasks(cfg, n=n, credit_fraction=1.0, seed=seed)
    return {"original (live)": original, "credit (live)": ours}, "live"


def _require_variants(loaded: dict[str, list[SyntheticTask]]) -> None:
    """Fail with the fix, not with a matplotlib internals error."""
    if not loaded:
        raise ValueError(
            "no variants to plot. Either generate a pool:\n"
            "  python scripts/generate_prior.py --config config/LGD.yaml --variant original --all\n"
            "or copy one from the cluster:\n"
            "  bash scripts/fetch_prior_sample.sh\n"
            "or use load_variants_or_generate(), which falls back to live generation."
        )


def variant_color(variant: str, index: int = 0) -> str:
    """`original` is always the grey reference; ours get distinct colours."""
    if variant == "original":
        return style.ORIGINAL
    if variant == "credit_v1":
        return style.CREDIT
    return style.SERIES[(index + 1) % len(style.SERIES)]


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
        rec: dict[str, Any] = {
            "variant": variant,
            "n_sampled": len(tasks),
            "rows_median": int(np.median([t.n_rows for t in tasks])),
            "features_median": int(np.median([t.n_features for t in tasks])),
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
    style.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for i, (variant, tasks) in enumerate(loaded.items()):
        stats = [target_stats(t.y) for t in tasks]
        boundary = np.array([s["frac_at_min"] + s["frac_at_max"] for s in stats])
        colour = variant_color(variant, i)
        # Step histograms, not filled bars: four filled histograms hide each other.
        ax1.hist(boundary, bins=30, histtype="step", lw=2.2, color=colour,
                 label=f"{variant} (mean {boundary.mean():.3f})")
        ax2.scatter(
            [s["frac_at_min"] for s in stats], [s["frac_at_max"] for s in stats],
            s=26, alpha=0.5, color=colour, edgecolor="none", label=variant,
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
    style.title(ax1, "Boundary mass by variant",
                "Dotted red lines are the real datasets' values")

    ax2.set_xlabel("mass at the low boundary")
    ax2.set_ylabel("mass at the high boundary")
    ax2.grid(axis="x")
    ax2.legend()
    style.title(ax2, "Which corner of the space each variant occupies",
                "Stars are real credit datasets — the cloud should cover them")
    return fig


def plot_base_rate_by_variant(loaded: dict[str, list[SyntheticTask]], real_reference=None):
    """Base-rate distribution per variant. The PD counterpart of boundary mass."""
    _require_variants(loaded)
    style.use_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (variant, tasks) in enumerate(loaded.items()):
        rates = np.array([float((t.y > 0.5).float().mean()) for t in tasks])
        ax.hist(rates, bins=30, histtype="step", lw=2.2, color=variant_color(variant, i),
                label=f"{variant} (mean {rates.mean():.3f})")
    if real_reference:
        for name, rate in real_reference.items():
            ax.axvline(rate, color=style.REAL, lw=1, ls=":", alpha=0.75)
            ax.annotate(name, (rate, ax.get_ylim()[1] * 0.95), rotation=90, fontsize=7.5,
                        color=style.REAL, ha="right", va="top")
    ax.axvline(0.5, color=style.MUTED, lw=1.5, ls="--")
    ax.set_xlabel("positive (default) rate")
    ax.set_ylabel("number of tasks")
    ax.legend()
    style.title(ax, "Base rate by variant",
                "Dashed grey = balance; dotted red = real datasets, all well left of it")
    return fig


def plot_target_shapes_by_variant(loaded: dict[str, list[SyntheticTask]], n_per: int = 10):
    """One row of target histograms per variant — the visual gist, side by side.

    This is the compromise that replaces one-notebook-per-variant: not 100 panels for
    one variant, but 10 for each of them, on the same figure, so differences in shape
    are seen rather than remembered.
    """
    _require_variants(loaded)
    style.use_style()
    n_var = len(loaded)
    fig, axes = plt.subplots(n_var, n_per, figsize=(1.35 * n_per, 1.5 * n_var), squeeze=False)
    for r, (variant, tasks) in enumerate(loaded.items()):
        colour = variant_color(variant, r)
        for c in range(n_per):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
            if c >= len(tasks):
                ax.axis("off")
                continue
            ax.hist(tasks[c].y.numpy(), bins=25, color=colour)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color(style.GRID)
        axes[r][0].set_ylabel(variant, fontsize=9, color=style.INK, rotation=0,
                              ha="right", va="center", labelpad=8)
    fig.suptitle("Target shapes, one row per variant")
    style.figure_note(fig, "Same random draw for each row, so rows are comparable.")
    return fig


def plot_spectrum_by_variant(loaded: dict[str, list[SyntheticTask]], n_curves: int = 40):
    """Correlation spectra, one colour per variant, medians drawn bold.

    O'Prior's central measurement. If two variants' spectra sit on top of each other,
    they teach a similar dependence structure however different the targets look.
    """
    _require_variants(loaded)
    style.use_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    grid = np.linspace(0, 1, 50)
    for i, (variant, tasks) in enumerate(loaded.items()):
        colour = variant_color(variant, i)
        curves = []
        for t in tasks:
            X = t.X.numpy()
            if X.shape[1] < 2:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.nan_to_num(np.corrcoef(X, rowvar=False))
            ev = np.sort(np.linalg.eigvalsh(C))[::-1]
            ev = ev / max(ev[0], 1e-9)
            curves.append(np.interp(grid, np.arange(1, len(ev) + 1) / len(ev), ev))
        if not curves:
            continue
        for c in curves[:n_curves]:
            ax.plot(grid, c, color=colour, alpha=0.12, lw=0.8)
        ax.plot(grid, np.median(curves, axis=0), color=colour, lw=2.6, label=variant)
    ax.set_xlabel("eigenvalue rank (normalised)")
    ax.set_ylabel("eigenvalue / largest")
    ax.grid(axis="x")
    ax.legend()
    style.title(ax, "Correlation spectrum by variant",
                "Bold lines are medians; overlapping spectra = similar dependence structure")
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
    style.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (variant, tasks) in enumerate(loaded.items()):
        colour = variant_color(variant, i)
        axes[0].hist([t.n_rows for t in tasks], bins=20, histtype="step", lw=2.2,
                     color=colour, label=variant)
        axes[1].hist([t.n_features for t in tasks], bins=20, histtype="step", lw=2.2,
                     color=colour, label=variant)
    axes[0].set_xlabel("rows per task")
    axes[1].set_xlabel("features per task")
    for ax in axes:
        ax.set_ylabel("number of tasks")
        ax.legend()
    style.title(axes[0], "Rows", "should MATCH across variants")
    style.title(axes[1], "Features", "a difference here would be a confound, not a finding")
    return fig
