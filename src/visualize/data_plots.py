"""Look at the real credit datasets — the things the prior is supposed to suit.

This is the other half of `prior_plots.py`. That module shows what we generate;
this one shows what we are generating *for*. Read together they answer the
question the project rests on: does our synthetic prior look like credit data in
the ways that matter, and not merely in the ways that are easy to hit?

The measurements here are the ones that drove the design:

* **boundary mass** — the share of LGD targets sitting exactly at 0 or 1. This is
  the single feature the original prior does not produce, and the reason the
  project exists.
* **base rate** — how rare default is in each PD dataset. TabICL's prior is
  roughly balanced; real PD data is not.
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
    """One target histogram per LGD dataset. The figure only — the caption explains it.

    REWRITTEN because it was unreadable. Each panel carried a two-line title
    ("axa" / "34% of rows at a boundary") which wrapped, collided with the panel above and
    with the suptitle, and matplotlib gave up: "axes sizes collapsed to zero". The boundary
    percentages are the subject of `plot_boundary_mass_ranking`, so repeating them here bought
    nothing and cost the figure its legibility.

    Now: dataset name only, shared axes, and nothing else drawn.
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

    for ax, (slug, ds) in zip(flat, items):
        y = np.asarray(ds.y, dtype=float)
        st = target_stats(y)
        ax.hist(y, bins=40, color=style.REAL, alpha=0.9)
        # Mark the boundaries explicitly; in a 40-bin histogram an exact atom at 0 and a
        # cluster near 0.02 look identical, and only one of them is the point.
        for edge, frac in ((y.min(), st["frac_at_min"]), (y.max(), st["frac_at_max"])):
            if frac > 0.01:
                ax.axvline(edge, color=style.WARN, lw=1.0, ls="--", alpha=0.85)
        # The NAME, one line, nothing else. `set_title` directly rather than `style.title`,
        # because a wrapped two-line title is what broke this figure.
        ax.set_title(slug.split(".", 1)[-1], fontsize=mpl.rcParams["xtick.labelsize"],
                     loc="left", pad=3)
        ax.set_yticks([])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.5, 1.0])
    for ax in flat[len(items):]:
        ax.axis("off")
    # Axis label once, on the bottom-left panel only. Repeating "LGD" seven times is seven
    # times the ink for the same information.
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("LGD")
    # The last row may be short (7 datasets in a 4x2 grid), which leaves the panel above an
    # "off" cell with no visible tick labels. Restore them there.
    for col in range(ncols):
        for row in range(nrows - 1, -1, -1):
            if axes[row][col].axison:
                axes[row][col].set_xlabel("LGD")
                axes[row][col].tick_params(labelbottom=True)
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
    ax.barh(ypos, at0, color=style.CREDIT, label="at 0 (full recovery)")
    ax.barh(ypos, at1, left=at0, color=style.REAL, label="at 1 (total loss)")
    for i, (a, b) in enumerate(zip(at0, at1)):
        if a + b > 0.005:
            ax.text(a + b + 0.01, i, f"{a + b:.1%}", va="center", fontsize=9, color=style.MUTED)
    ax.set_yticks(ypos, names)
    ax.set_xlabel("share of rows sitting exactly at a boundary")
    ax.set_xlim(0, max(0.05, float((at0 + at1).max()) * 1.25))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    style.title(ax, "Boundary mass per dataset")
    return fig


# ---------------------------------------------------------------------------
# PD — the imbalance story
# ---------------------------------------------------------------------------


def plot_pd_base_rates(datasets: dict[str, Any]):
    """Default rate per dataset, against the prior's balance point.

    LINEAR, not log. The log axis was justified as showing two orders of magnitude, but the
    rates actually run 6%-40% — well inside one order — so it bought nothing and cost the
    reader tick labels like `6 x 10^-2`.
    """
    style.apply()
    rows = sorted(
        ((slug.split(".", 1)[-1], float(np.asarray(ds.y).mean()), ds.n_rows)
         for slug, ds in datasets.items()),
        key=lambda r: r[1],
    )
    names = [r[0] for r in rows]
    rates = np.array([r[1] for r in rows])

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.20, base=1.0))
    ypos = np.arange(len(rows))
    ax.barh(ypos, rates, color=style.TASK_COLOR["pd"], alpha=0.9)
    ax.axvline(0.5, color=style.ORIGINAL, lw=1.4, ls="--")
    # Two words, rotated along the line it labels. "TabICL's prior sits near here" wrapped to
    # two lines, sat over the top bar's value, and said in six words what two say.
    ax.text(0.5, len(rows) - 0.5, " TabICL prior", fontsize=7, color=style.MUTED,
            va="top", ha="left", rotation=90, rotation_mode="anchor")
    # Value labels INSIDE the bar when there is room, outside when there is not. Placed
    # outside at `r * 1.08` they drifted right as the rate grew and the 40% label landed on
    # top of the 50% reference line.
    for i, r in enumerate(rates):
        inside = r > 0.12
        ax.text(r - 0.01 if inside else r + 0.01, i, f"{r:.1%}",
                va="center", ha="right" if inside else "left", fontsize=7,
                color="white" if inside else style.MUTED)
    ax.set_yticks(ypos, names)
    ax.set_xlim(0, 0.56)
    ax.set_xlabel("default rate")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
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
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.68))
    for task, datasets in datasets_by_task.items():
        xs = [ds.n_rows for ds in datasets.values()]
        ys = [ds.n_features for ds in datasets.values()]
        ax.scatter(xs, ys, s=90, alpha=0.85, color=style.TASK_COLOR[task],
                   label=f"{task.upper()} ({len(xs)})", edgecolor="white", zorder=3)
        for slug, ds in datasets.items():
            ax.annotate(slug.split(".", 1)[-1], (ds.n_rows, ds.n_features),
                        fontsize=7.5, color=style.MUTED,
                        textcoords="offset points", xytext=(6, 3))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rows")
    ax.set_ylabel("features")
    ax.grid(axis="x")
    ax.legend(title="task")
    style.title(ax, "Dataset shapes")
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

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.16, base=1.1))
    ypos = np.arange(len(rows))
    ax.barh(ypos, shares, color=colours, alpha=0.9)
    ax.set_yticks(ypos, names, fontsize=8)
    ax.set_xlabel("share of columns that are categorical")
    ax.set_xlim(0, 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=style.legend_patches({t.upper(): c for t, c in style.TASK_COLOR.items()}))
    style.title(ax, "Categorical share")
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
    rows.sort(key=lambda r: -r[2])
    names = [r[0] for r in rows]
    vals = np.array([r[2] for r in rows])
    colours = [style.TASK_COLOR[r[1]] for r in rows]

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.16, base=1.1))
    ypos = np.arange(len(rows))
    ax.barh(ypos, vals, color=colours, alpha=0.9)
    ax.set_yticks(ypos, names, fontsize=8)
    ax.set_xlabel("share of cells missing")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.title(ax, "Missing values in the processed datasets")
    return fig


def correlation_pages(datasets: dict[str, Any], per_page: int = 6) -> int:
    """How many figures `plot_feature_correlations` needs to show EVERY dataset."""
    return max(1, int(np.ceil(len(datasets) / max(per_page, 1))))


def plot_feature_correlations(
    datasets: dict[str, Any], n_show: int | None = None, per_page: int = 6, page: int = 1
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
    items = sorted(datasets.items(), key=lambda kv: -kv[1].n_rows)
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

    for ax, (slug, ds) in zip(flat, items):
        X = np.asarray(ds.X, dtype=float)[:5000]
        keep = ~np.all(np.isnan(X), axis=0)
        X = np.nan_to_num(X[:, keep], nan=0.0)
        # Drop constant columns: their correlation is undefined and numpy would
        # return NaN for the whole row, blanking out the heatmap.
        keep2 = X.std(axis=0) > 0
        X = X[:, keep2]
        C = np.corrcoef(X, rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        off = C[~np.eye(len(C), dtype=bool)] if len(C) > 1 else np.array([0.0])
        style.title(ax, slug.split(".", 1)[-1],
                    f"{X.shape[1]} cols, mean |corr| {np.abs(off).mean():.2f}")
    for ax in flat[len(items):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7, label="correlation")
    fig.suptitle("Feature correlations")
    return fig


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
