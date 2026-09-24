"""Look at the real credit datasets — the things the prior is supposed to suit.

This is the other half of `prior_plots.py`. That module shows what we generate;
this one shows what we are generating *for*. Read together they answer the
question the project rests on: does our synthetic prior look like credit data in
the ways that matter, and not merely in the ways that are easy to hit?

The measurements here are the ones that drove the design:

* **boundary mass** — the share of LGD targets sitting exactly at 0 or 1. This is
  the single feature the original prior does not produce, and the reason the
  project exists.
* **base rate** — how rare default is in each PD dataset. TabICL's prior spreads its
  tasks' base rates over the whole [0, 1], centred on 50%; real PD data runs 6.7%-40%.
* **shape and type mix** — rows, columns, and how many columns are categorical.
  These set the ranges the prior samples over; if the prior generated 500-column
  tables and every real dataset has 20, the extra capacity is wasted.

Everything reads from the processed parquet cache via `src.data.pipeline`, so the
notebook sees exactly the tables the evaluation sees — not a separate re-read of
the raw CSVs that might disagree.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_registry import datasets_for_task
from src.data.pipeline import ensure_processed, load_processed
from src.utils.target_stats import target_stats
from src.visualize import style


def load_all(task: str, *, verbose: bool = True) -> dict[str, Any]:
    """Load every processed dataset for a task, preprocessing any that are missing.

    Returns {slug: ProcessedDataset}. A dataset that cannot be loaded is skipped
    with a message rather than killing the notebook — one broken raw file should
    not stop you looking at the other twenty.
    """
    out: dict[str, Any] = {}
    for slug in datasets_for_task(task):
        try:
            ensure_processed(task, slug)
            out[slug] = load_processed(task, slug)
        except Exception as exc:  # noqa: BLE001 — a notebook wants to keep going
            if verbose:
                print(f"  skipped {task}/{slug}: {type(exc).__name__}: {exc}")
    return out


def summary_table(datasets: dict[str, Any], task: str) -> pd.DataFrame:
    """One row per dataset: the numbers that set the prior's ranges.

    Sorted by rows, because dataset size is what decides which are usable for
    in-context learning at all — TabICL's context has a practical ceiling.
    """
    rows = []
    for slug, ds in datasets.items():
        y = np.asarray(ds.y, dtype=float)
        st = target_stats(y)
        rec = {
            "dataset": slug,
            "rows": ds.n_rows,
            "features": ds.n_features,
            "categorical": len(ds.cat_indices),
            "% categorical": round(100 * len(ds.cat_indices) / max(1, ds.n_features), 1),
            "missing %": round(100 * float(np.isnan(ds.X).mean()), 2),
        }
        if task == "lgd":
            rec.update(
                {
                    "mass at 0": round(st["frac_at_min"], 4),
                    "mass at 1": round(st["frac_at_max"], 4),
                    "boundary mass": round(st["frac_at_min"] + st["frac_at_max"], 4),
                    "mean": round(float(y.mean()), 4),
                    "in [0,1]": bool(y.min() >= 0 and y.max() <= 1),
                }
            )
        else:
            rec.update(
                {
                    "base rate": round(float(y.mean()), 4),
                    "imbalance 1:n": round((1 - y.mean()) / max(y.mean(), 1e-9), 1),
                    "n positive": int(y.sum()),
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LGD — the boundary mass story
# ---------------------------------------------------------------------------


def plot_lgd_targets(datasets: dict[str, Any], ncols: int = 4):
    """One target histogram per LGD dataset: the share of rows per bin, on labelled axes.

    Each panel's title is the dataset and its share of rows at exactly 0 or 1, on ONE line. An
    earlier version put that share on a second title line, which wrapped, collided with the panel
    above and collapsed the grid ("axes sizes collapsed to zero"); the version after it dropped the
    share and every axis, which left panels that could not say how tall an atom was — the one thing
    the figure is for.
    """
    style.apply()
    items = sorted(datasets.items(), key=lambda kv: -kv[1].n_rows)
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.90),
        squeeze=False, sharex=True, sharey=False,
    )
    flat = axes.ravel()

    small = mpl.rcParams["xtick.labelsize"] * 0.9
    for ax, (slug, ds) in zip(flat, items):
        y = np.asarray(ds.y, dtype=float)
        st = target_stats(y)
        # Share of rows per bin, with its axis: the height of an atom against the interior IS the
        # content of this figure, and a panel without y ticks could not say it.
        ax.hist(y, bins=40, range=(0.0, 1.0), weights=np.full(y.size, 1.0 / max(y.size, 1)),
                color=style.REAL, alpha=0.9, linewidth=0)
        # Mark the boundaries explicitly; in a 40-bin histogram an exact atom at 0 and a
        # cluster near 0.02 look identical, and only one of them is the point.
        for edge, frac in ((y.min(), st["frac_at_min"]), (y.max(), st["frac_at_max"])):
            if frac > 0.01:
                ax.axvline(edge, color=style.WARN, lw=0.9, ls="--", alpha=0.85)
        # The NAME and its boundary share, on ONE line: a wrapped two-line title is what once
        # collapsed this grid, and the share is the number each panel exists to show.
        ax.set_title(f"{slug.split('.', 1)[-1]} · {st['frac_at_min'] + st['frac_at_max']:.0%} at 0 or 1",
                     fontsize=small, loc="left", pad=3)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.tick_params(labelsize=small)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.tick_params(labelbottom=True)
    for ax in flat[len(items):]:
        ax.axis("off")
    for row in range(nrows):
        if axes[row][0].axison:
            axes[row][0].set_ylabel("share of rows", fontsize=small)
    # The x label on the lowest visible panel of every column.
    for col in range(ncols):
        for row in range(nrows - 1, -1, -1):
            if axes[row][col].axison:
                axes[row][col].set_xlabel("LGD", fontsize=small)
                break
    return fig


def plot_boundary_mass_ranking(datasets: dict[str, Any]):
    """Boundary mass per dataset as a ranked bar chart, split by which edge.

    A ranking rather than a scatter, because the practical question is "which
    datasets does this actually matter for?" and the answer is a sorted list.
    """
    style.apply()
    rows = []
    for slug, ds in datasets.items():
        st = target_stats(np.asarray(ds.y, dtype=float))
        rows.append((slug.split(".", 1)[-1], st["frac_at_min"], st["frac_at_max"]))
    rows.sort(key=lambda r: r[1] + r[2])
    names = [r[0] for r in rows]
    at0 = np.array([r[1] for r in rows])
    at1 = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.22, base=1.1))
    ypos = np.arange(len(rows))
    # The palette's two atom colours — full recovery and total loss mean the same in every
    # figure; they were CREDIT blue and REAL orange here, which mean something else entirely.
    ax.barh(ypos, at0, color=style.ATOM_LO, label="at exactly 0 (full recovery)")
    ax.barh(ypos, at1, left=at0, color=style.ATOM_HI, label="at exactly 1 (total loss)")
    for i, (a, b) in enumerate(zip(at0, at1)):
        if a + b > 0.005:
            ax.text(a + b + 0.01, i, f"{a + b:.1%}", va="center", fontsize=8, color=style.MUTED)
    ax.set_yticks(ypos, names)
    ax.set_xlabel("share of rows at a boundary")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(0.05, float((at0 + at1).max()) * 1.18))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.legend_below(ax, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# PD — the imbalance story
# ---------------------------------------------------------------------------


def plot_pd_base_rates(datasets: dict[str, Any]):
    """Default rate per dataset, against the rate below which a classifier collapses.

    LINEAR, not log: the rates run 6.7%-40%, well inside one order of magnitude. The reference is
    Tanna 2026's ~10% (`literature.tanna_paradox`): below it a default-threshold classifier
    predicts "no default" for everyone on real credit data. It replaces a dashed 50% line labelled
    "TabICL prior", which claimed more than it could show — the original prior's base rates spread
    over the whole [0, 1] (0.2 A2), centred on 50%.
    """
    from src.visualize import literature

    style.apply()
    rows = sorted(
        ((slug.split(".", 1)[-1], float(np.asarray(ds.y).mean()), ds.n_rows)
         for slug, ds in datasets.items()),
        key=lambda r: r[1],
    )
    names = [r[0] for r in rows]
    rates = np.array([r[1] for r in rows])

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.20, base=1.15))
    ypos = np.arange(len(rows))
    ax.barh(ypos, rates, color=style.TASK_COLOR["pd"], alpha=0.9, label="default rate")
    literature.line(ax, "tanna_paradox", label="Tanna 2026: classifier collapse below 10%",
                    inline=False)
    # Value labels INSIDE the bar when there is room, outside when there is not — and then to the
    # right of the 10% line, which a label just past a short bar would be written across.
    for i, r in enumerate(rates):
        inside = r > 0.12
        ax.text(r - 0.01 if inside else max(r + 0.01, 0.106), i, f"{r:.1%}",
                va="center", ha="right" if inside else "left", fontsize=7,
                color="white" if inside else style.MUTED)
    ax.set_yticks(ypos, names)
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("default rate (share of rows that defaulted)")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.legend_below(ax, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# Both tasks — shape, types, missingness
# ---------------------------------------------------------------------------


def plot_shapes(datasets_by_task: dict[str, dict[str, Any]]):
    """Rows against columns, per dataset. Sets the prior's shape ranges.

    Both axes are log: dataset sizes here span 1,000 to over a million rows, and on
    a linear axis every small dataset would sit on top of the origin.
    """
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.62))
    xs_all, ys_all, names = [], [], []
    for task, datasets in datasets_by_task.items():
        xs = [ds.n_rows for ds in datasets.values()]
        ys = [ds.n_features for ds in datasets.values()]
        ax.scatter(xs, ys, s=60, alpha=0.85, color=style.TASK_COLOR[task],
                   label=f"{task.upper()} ({len(xs)} datasets)", edgecolor="white", zorder=3)
        xs_all += xs
        ys_all += ys
        names += [slug.split(".", 1)[-1] for slug in datasets]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rows (log scale)")
    ax.set_ylabel("features (log scale)")
    ax.grid(axis="x")
    # The prior's own table: 1,024 rows and up to 100 features (config/Exp1_*.yaml).
    ax.axvline(1024, color=style.MUTED, lw=0.8, ls=":", zorder=1,
               label="the prior's table: 1,024 rows, at most 100 features")
    ax.axhline(100, color=style.MUTED, lw=0.8, ls=":", zorder=1)
    ax.margins(x=0.12, y=0.15)
    style.legend_below(ax, ncol=3)
    # Last: the labels need the final limits. Placed so that no two names collide — at one fixed
    # offset `lgd_lendingclub` sat on `hmeq` and `lgd_freddie` under a marker.
    style.place_labels(ax, xs_all, ys_all, names, fontsize=6.5)
    return fig


def plot_type_mix(datasets_by_task: dict[str, dict[str, Any]]):
    """Share of columns that are categorical, per dataset.

    Relevant because TabICL's prior turns a fraction of its columns categorical, and
    that fraction is a hyperparameter we could be setting from evidence rather than
    from the default.
    """
    style.apply()
    rows = []
    for task, datasets in datasets_by_task.items():
        for slug, ds in datasets.items():
            share = len(ds.cat_indices) / max(1, ds.n_features)
            rows.append((f"{slug.split('.', 1)[-1]}", task, share))
    rows.sort(key=lambda r: r[2])
    names = [r[0] for r in rows]
    colours = [style.TASK_COLOR[r[1]] for r in rows]
    shares = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.16, base=1.2))
    ypos = np.arange(len(rows))
    ax.barh(ypos, shares, color=colours, alpha=0.9)
    ax.set_yticks(ypos, names, fontsize=7)
    ax.set_xlabel("share of columns that are categorical")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.legend_below(ax, style.legend_patches({t.upper(): c for t, c in style.TASK_COLOR.items()}))
    return fig


def plot_missingness(datasets_by_task: dict[str, dict[str, Any]]):
    """Missing-value share per dataset.

    Our prior injects missingness as an explicit mechanism, so it is worth knowing
    whether real credit data has 0.1% missing or 30% — the two call for different
    settings, and several of these datasets have already been imputed upstream,
    which is itself worth seeing.
    """
    style.apply()
    rows = []
    for task, datasets in datasets_by_task.items():
        for slug, ds in datasets.items():
            rows.append((slug.split(".", 1)[-1], task, float(np.isnan(ds.X).mean())))
    # Ascending, so the largest share is the TOP bar — the same reading order as every other
    # ranked bar chart here (a horizontal bar chart draws its first row at the bottom).
    rows.sort(key=lambda r: r[2])
    names = [r[0] for r in rows]
    vals = np.array([r[2] for r in rows])
    colours = [style.TASK_COLOR[r[1]] for r in rows]

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.16, base=1.2))
    ypos = np.arange(len(rows))
    ax.barh(ypos, vals, color=colours, alpha=0.9)
    ax.set_yticks(ypos, names, fontsize=7)
    ax.set_xlabel("share of cells missing, after preprocessing")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.legend_below(ax, style.legend_patches({t.upper(): c for t, c in style.TASK_COLOR.items()}))
    return fig


def correlation_pages(datasets: dict[str, Any], per_page: int = 9) -> int:
    """How many figures `plot_feature_correlations` needs to show EVERY dataset."""
    return max(1, int(np.ceil(len(datasets) / max(per_page, 1))))


def plot_feature_correlations(
    datasets: dict[str, Any], n_show: int | None = None, per_page: int = 9, page: int = 1
):
    """Correlation heatmaps — PAGINATED so every dataset is included.

    O'Prior's argument is that a prior's *feature dependence structure* is what transfers. So
    it matters what real credit data looks like: strong blocks of correlated features (several
    measures of the same balance), not independent columns. If our DAGs produced independent
    features they would be wrong here.

    Previously it took the `n_show` largest and silently dropped the rest — a figure that
    claims to describe "real credit data" while showing 6 of 14 datasets. Six panels per page
    keeps each one big enough to read a block structure in; call `correlation_pages()` for the
    count and loop.

    `n_show` is still accepted so old calls do not break: it caps the total considered.
    """
    style.apply()
    mats = {slug: _correlation_matrix(ds) for slug, ds in datasets.items()}
    # Ordered from the most to the least dependent — the story is "some books come in strong
    # blocks, others are close to independent columns", and an order by size hid it.
    items = sorted(datasets.items(), key=lambda kv: -_mean_abs_offdiag(mats[kv[0]]))
    if n_show is not None:
        items = items[:n_show]
    pages = style.paginate(items, per_page=per_page)
    n_pages = max(1, len(pages))
    if not 1 <= page <= n_pages:
        raise ValueError(f"page {page} out of range; {n_pages} page(s) for {len(items)} datasets")
    items = pages[page - 1]

    ncols = min(3, len(items))
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.95), squeeze=False)
    flat = axes.ravel()

    small = mpl.rcParams["xtick.labelsize"]
    im = None
    for ax, (slug, _ds) in zip(flat, items):
        C = mats[slug]
        im = ax.imshow(C, cmap=style.CMAP_DIV, vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        # ONE line of heading and one of numbers: the three-line bold headings ate a third of each
        # panel. The mean |r| is the panel's summary, so it sits where an axis label would.
        ax.set_title(slug.split(".", 1)[-1], loc="left", fontsize=9, pad=3)
        ax.set_xlabel(f"{len(C)} features · mean |r| {_mean_abs_offdiag(C):.2f}", fontsize=small)
    for ax in flat[len(items):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson correlation between two features")
    return fig


def _correlation_matrix(ds: Any, max_rows: int = 5000, seed: int = 0) -> np.ndarray:
    """Feature-feature Pearson correlations on a fixed RANDOM sample of rows (a file's first rows
    encode its ordering), columns that are entirely missing or constant removed."""
    X = np.asarray(ds.X, dtype=float)
    if X.shape[0] > max_rows:
        X = X[np.sort(np.random.default_rng(seed).choice(X.shape[0], max_rows, replace=False))]
    keep = ~np.all(np.isnan(X), axis=0)
    X = np.nan_to_num(X[:, keep], nan=0.0)
    # Drop constant columns: their correlation is undefined and numpy would return NaN for the
    # whole row, blanking out the heatmap.
    X = X[:, X.std(axis=0) > 0]
    return np.corrcoef(X, rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))


def _mean_abs_offdiag(C: np.ndarray) -> float:
    return float(np.abs(C[~np.eye(len(C), dtype=bool)]).mean()) if len(C) > 1 else 0.0


def leakage_check(datasets: dict[str, Any], task: str, top_k: int = 3) -> pd.DataFrame:
    """Flag features suspiciously predictive of the target on their own.

    Motivated by a real finding: `lgd_lendingclub` gives R^2 in the 0.71-0.76 range,
    far above anything published for LGD, which usually means a column encodes the
    answer. This does not prove leakage — it points at where to look.

    One-feature correlation only, deliberately: a cheap screen that a reader can
    check by hand, not a model whose own capacity muddies the question.
    """
    rows = []
    for slug, ds in datasets.items():
        X = np.asarray(ds.X, dtype=float)
        y = np.asarray(ds.y, dtype=float)
        names = list(ds.feature_names)
        scores = []
        for j in range(X.shape[1]):
            col = X[:, j]
            ok = ~np.isnan(col)
            if ok.sum() < 50 or col[ok].std() == 0:
                continue
            scores.append((abs(float(np.corrcoef(col[ok], y[ok])[0, 1])), names[j]))
        scores.sort(reverse=True)
        for r, name in scores[:top_k]:
            rows.append({"dataset": slug, "feature": name, "|corr with target|": round(r, 3),
                         "suspicious": r > 0.9})
    df = pd.DataFrame(rows)
    return df.sort_values("|corr with target|", ascending=False).reset_index(drop=True)


def plot_leakage_screen(leakage: pd.DataFrame, threshold: float = 0.9):
    """The strongest single feature of every dataset: its absolute correlation with the target.

    One bar per dataset — the top row of `leakage_check` for it — labelled with the feature, the
    dashed line at the `threshold` above which `leakage_check` marks a feature suspicious. The
    screen exists because `lgd_lendingclub` reaches an R² of 0.71-0.76, far above published LGD
    results; a bar near 1 is where to look first, never proof on its own.
    """
    style.apply()
    top = (leakage.sort_values("|corr with target|", ascending=False)
           .drop_duplicates("dataset").sort_values("|corr with target|"))
    fig, ax = plt.subplots(figsize=style.row_figsize(len(top), per_row=0.2, base=1.2))
    if top.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "no dataset to screen", ha="center", va="center", color=style.MUTED)
        return fig
    names = [str(d).split(".", 1)[-1] for d in top["dataset"]]
    values = top["|corr with target|"].to_numpy()
    # Coloured by task when the caller says which (a `task` column), flagged bars outlined in red.
    tasks = top["task"].tolist() if "task" in top.columns else [None] * len(top)
    colours = [style.TASK_COLOR.get(t, style.MUTED) for t in tasks]
    ypos = np.arange(len(top))
    ax.barh(ypos, values, color=colours, alpha=0.9,
            edgecolor=[style.WARN if v > threshold else "none" for v in values], linewidth=1.2)
    for i, (v, feat) in enumerate(zip(values, top["feature"])):
        ax.text(v + 0.01, i, str(feat)[:28], va="center", fontsize=6.5, color=style.MUTED)
    ax.axvline(threshold, color=style.WARN, ls="--", lw=1.0)
    ax.set_yticks(ypos, names, fontsize=7)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("strongest single-feature |correlation| with the target (label: that feature)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    present = [t for t in ("lgd", "pd") if t in tasks]
    handles = style.legend_patches({t.upper(): style.TASK_COLOR[t] for t in present}) + [
        plt.Line2D([], [], color=style.WARN, ls="--", lw=1.0,
                   label=f"flag threshold |r| = {threshold:g}")]
    style.legend_below(ax, handles, ncol=len(handles))
    return fig
