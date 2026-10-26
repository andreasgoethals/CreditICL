"""Descriptive prior diagnostics; no selection or tuning on benchmark target histograms.

Experiment 1 compares four mixtures and five component removals. These figures describe
synthetic mechanisms and their differences from previously inspected credit datasets.
Standardized control regression targets have no loss-fraction scale and cannot be used
in LGD boundary-mass comparisons.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.utils.target_stats import target_stats
from src.visualize import style

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _hist_density(values: np.ndarray, bins: int = 40, lo: float = 0.0, hi: float = 1.0):
    """A normalised histogram on a FIXED support, so two of them can be compared.

    Fixed edges matter: `np.histogram` with `bins=40` picks its own range per array, so two
    distributions would be measured on different grids and any distance between them would be
    meaningless.
    """
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=(lo, hi))
    total = counts.sum()
    return (counts / total if total else counts.astype(float)), edges


def distribution_distance(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    """How far apart two `[0,1]` distributions are: total variation, in [0, 1].

    Total variation (half the L1 distance between histograms) rather than a KS statistic,
    because **it is not fooled by point masses**. LGD's whole story is atoms at 0 and 1; a
    metric built on CDFs treats a spike as a step and understates how different two
    distributions with different atom sizes are. TV compares the mass in each bin directly, so
    an atom is just a very full bin.

    0 = identical, 1 = disjoint. Symmetric, and bounded, which makes it readable on an axis.
    """
    pa, _ = _hist_density(a, bins=bins)
    pb, _ = _hist_density(b, bins=bins)
    return float(0.5 * np.abs(pa - pb).sum())


def _unit_target(y: Any) -> np.ndarray:
    """Loss fractions only: arbitrary standardized targets have no LGD boundary scale."""
    y = np.asarray(y, dtype=float).ravel()
    if not np.isfinite(y).all() or y.min() < -1e-6 or y.max() > 1 + 1e-6:
        raise ValueError("An unbounded target cannot be ranked as a loss-fraction distribution")
    return np.clip(y, 0.0, 1.0)


def _real_targets(task: str, datasets: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """{name: y} for the real datasets, as plain arrays in [0,1]."""
    if not datasets:
        return {}
    out = {}
    for name, ds in datasets.items():
        y = np.asarray(getattr(ds, "y", ds), dtype=float).ravel()
        out[name.split(".", 1)[-1]] = y
    return out


# ---------------------------------------------------------------------------
# 1. Which priors look like real credit data?
# ---------------------------------------------------------------------------


def plot_prior_realism_ranking(
    variants: dict[str, list[Any]],
    real: dict[str, Any] | None = None,
    task: str = "lgd",
):
    """Descriptive target-distribution distances for priors on a comparable [0, 1] scale."""
    style.apply()
    reals = _real_targets(task, real)
    if not reals:
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.3))
        ax.text(0.5, 0.5, "No real datasets available to compare against.",
                ha="center", va="center", color=style.MUTED)
        ax.axis("off")
        return fig

    rows = []
    for name, tasks in variants.items():
        # Pool every synthetic target for this variant into one distribution: the question is
        # what the variant produces ON AVERAGE, not what one draw did. Each task on [0, 1] first.
        if task == "lgd" and any(getattr(t, "source", None) == "base" for t in tasks):
            continue  # upstream standardised targets do not define losses at 0 and 1
        try:
            pooled = np.concatenate([_unit_target(t.y) for t in tasks])
        except ValueError:
            continue
        per_real = {r: distribution_distance(pooled, y) for r, y in reals.items()}
        rows.append((name, float(np.mean(list(per_real.values()))), per_real))
    if not rows:
        from src.visualize.results_plots import _placeholder
        return _placeholder("No comparable targets on the loss-fraction scale")
    rows.sort(key=lambda r: r[1])

    if len(rows) == 1:
        name, _, per_real = rows[0]
        ordered = sorted(per_real.items(), key=lambda item: item[1])
        fig, ax = plt.subplots(figsize=style.row_figsize(len(ordered), per_row=.25, base=1.15))
        values = np.array([value for _, value in ordered])
        ypos = np.arange(len(ordered))
        ax.barh(ypos, values, color=style.CREDIT)
        ax.set_yticks(ypos, [key.split(".", 1)[-1].removeprefix("lgd_") for key, _ in ordered])
        for y, value in zip(ypos, values):
            ax.text(value+.01, y, f"{value:.3f}", va="center", fontsize=style.PANEL_TEXT)
        ax.set(xlabel="Marginal target distance (total variation)",
               xlim=(0, max(.5, float(values.max())*1.2)),
               title="Raw credit prior versus real portfolios")
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        return fig

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.34, base=1.05))
    labels = [style.variant_label(r[0]) for r in rows]
    ypos = np.arange(len(rows))
    for i, (name, mean_d, per_real) in enumerate(rows):
        values = list(per_real.values())
        colour = style.variant_colour(name, i)
        # A guide line per row. Without it the eye cannot carry a dot at x=0.6 back to its
        # label three rows up, which is the one thing this chart is for.
        ax.plot([min(values), max(values)], [i, i], color=colour, alpha=0.25, linewidth=3.0,
                solid_capstyle="round", zorder=1)
        # Every real dataset as a dot, so the SPREAD is visible: a prior that matches one
        # dataset and misses six is not a good prior, and a mean alone would hide that.
        ax.scatter(values, np.full(len(values), i), s=18, color=colour, alpha=0.85,
                   zorder=2, linewidths=0, label="one real dataset" if i == 0 else None)
        ax.scatter([mean_d], [i], s=62, color=style.INK, zorder=3, marker="D",
                   edgecolors="white", linewidths=0.8, label="mean over datasets" if i == 0 else None)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("total-variation distance from a real dataset's target (0 = identical, 1 = disjoint)")
    ax.set_xlim(0, max(0.5, max(max(r[2].values()) for r in rows) * 1.08))
    ax.grid(visible=True, axis="x")
    ax.grid(visible=False, axis="y")
    style.legend_below(ax, ncol=2)
    return fig


# ---------------------------------------------------------------------------
# 2. Do the LGD atoms come from the economics?
# ---------------------------------------------------------------------------


def plot_mechanism_decomposition(tasks: list[Any], bins: int = 40):
    """THE PAPER'S CENTRAL CLAIM AS A PICTURE: the atoms at 0 and 1 are *consequences*.

    The credit prior derives LGD from a loss story — collateral, workout, or a segment
    mixture — and the mass at 0 (full recovery) and 1 (total loss) falls out of that rather
    than being dialled in. Splitting the pooled target by which mechanism produced it is what
    shows this: `collateral` should own most of the atom at 0, because over-collateralised
    loans recover in full by construction.

    Falls back to a single pooled panel when mechanism labels are absent (a `quantile`-mode
    arm has none), rather than drawing an empty grid.
    """
    style.apply()
    groups: dict[str, list[np.ndarray]] = {}
    for t in tasks:
        mech = getattr(t, "mechanism", None) or (getattr(t, "meta", {}) or {}).get("mechanism")
        groups.setdefault(str(mech) if mech else "unlabelled", []).append(
            np.clip(np.asarray(t.y, dtype=float).ravel(), 0.0, 1.0)
        )

    names = [k for k in ("collateral", "workout", "segment_mixture", "unlabelled") if k in groups]
    names += [k for k in groups if k not in names]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=style.grid_figsize(n, 1, panel_ratio=0.95),
                             squeeze=False, sharey=True)
    for i, name in enumerate(names):
        ax = axes[0][i]
        pooled = np.concatenate(groups[name])
        counts, edges = _hist_density(pooled, bins=bins)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color=style.SERIES[i % len(style.SERIES)], linewidth=0)
        stats = target_stats(pooled)
        at0, at1 = stats["frac_at_min"], stats["frac_at_max"]
        # Annotate the two atoms ABOVE the bars, not on them. At the boundary the bar reaches
        # the top of the axis, so text at 92% of the height sat directly on the spike it was
        # labelling. Headroom is added first so the labels have somewhere to live.
        ax.set_ylim(0, counts.max() * 1.28)
        ax.annotate(f"{at0:.0%} at 0", (0.0, counts.max() * 1.06), fontsize=7,
                    color=style.INK, ha="left", va="bottom")
        ax.annotate(f"{at1:.0%} at 1", (1.0, counts.max() * 1.06), fontsize=7,
                    color=style.INK, ha="right", va="bottom")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("LGD")
        # Just the name. "N datasets" is bookkeeping, and giving every panel a second bold line
        # costs a third of the height in a three-panel row.
        style.title(ax, name.replace("_", " "))
    axes[0][0].set_ylabel("share of rows")
    fig.suptitle("Target by loss mechanism")
    return fig


# ---------------------------------------------------------------------------
# 3. Do PD defaults cluster, as real ones do?
# ---------------------------------------------------------------------------


def plot_default_clustering(
    variants: dict[str, list[Any]],
    real: dict[str, Any] | None = None,
    n_cohorts: int = 12,
):
    """Row-block variability; this does not identify economic cohorts.

    Synthetic tasks may be shuffled, and the benchmark inputs do not carry
    verified cohort labels here. Contiguous blocks therefore describe row order,
    not evidence for or against the Vasicek mechanism. The dashed reference is a
    binomial scale comparison, not a test of default independence.
    """
    style.apply()
    fig, (ax_spread, ax_example) = plt.subplots(
        1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.42)
    )

    def cohort_rates(y: np.ndarray) -> np.ndarray:
        """Default rate per equal-sized contiguous block."""
        y = np.asarray(y, dtype=float).ravel()
        blocks = np.array_split(y, min(n_cohorts, max(len(y) // 20, 2)))
        return np.array([float((b >= 0.5).mean()) for b in blocks if b.size])

    rows: list[tuple[str, list[float], list[float], str]] = []
    for i, (name, tasks) in enumerate(variants.items()):
        observed, expected = [], []
        for t in tasks:
            y = np.asarray(t.y, dtype=float).ravel()
            rates = cohort_rates(y)
            if rates.size < 2:
                continue
            base = float((y >= 0.5).mean())
            if not 0.0 < base < 1.0:
                continue
            observed.append(float(np.std(rates)))
            # Binomial standard error at this base rate and cohort size: the spread you would
            # see from chance alone. Dividing by it turns "wide" into "wider than chance".
            per = max(len(y) // max(len(rates), 1), 1)
            expected.append(float(np.sqrt(base * (1 - base) / per)))
        if observed:
            rows.append((name, observed, expected, style.variant_colour(name, i)))

    # LEFT: the clustering ratio per variant.
    for i, (_name, observed, expected, colour) in enumerate(rows):
        ratio = np.array(observed) / np.maximum(np.array(expected), 1e-12)
        parts = ax_spread.violinplot([ratio], positions=[i], widths=0.7, showextrema=False,
                                     showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.55)
            body.set_linewidth(0)
        if "cmedians" in parts:
            parts["cmedians"].set_color(style.INK)
            parts["cmedians"].set_linewidth(1.0)
    if real:
        real_ratios = []
        for y in _real_targets("pd", real).values():
            rates = cohort_rates(y)
            base = float((np.asarray(y) >= 0.5).mean())
            if rates.size >= 2 and 0.0 < base < 1.0:
                per = max(len(y) // max(len(rates), 1), 1)
                real_ratios.append(np.std(rates) / max(np.sqrt(base * (1 - base) / per), 1e-12))
        if real_ratios:
            ax_spread.scatter(
                np.full(len(real_ratios), len(rows)), real_ratios, marker="*", s=90,
                color=style.STAR, zorder=5, label="real datasets",
            )
    ax_spread.axhline(1.0, color=style.MUTED, linestyle="--", linewidth=0.9,
                      label="independent rows (ratio 1)")
    labels = [style.variant_label(r[0]) for r in rows] + (["real datasets"] if real else [])
    ax_spread.set_xticks(range(len(labels)))
    ax_spread.set_xticklabels(labels)
    ax_spread.set_ylabel("between-block SD ÷ binomial SD")
    ax_spread.set_ylim(bottom=0)
    style.title(ax_spread, "Row-block variability (not cohorts)")

    # RIGHT: one concrete example per variant, so the abstraction is grounded.
    for i, (name, tasks) in enumerate(variants.items()):
        pick = next((t for t in tasks if 0.0 < float((np.asarray(t.y) >= 0.5).mean()) < 1.0), None)
        if pick is None:
            continue
        rates = cohort_rates(np.asarray(pick.y))
        ax_example.plot(np.arange(1, rates.size + 1), rates, marker="o", markersize=3,
                        color=style.variant_colour(name, i),
                        label=f"one {style.variant_label(name)} task")
    _labelled: set[str] = set()
    for y in list(_real_targets("pd", real).values())[:2]:
        rates = cohort_rates(y)
        # Label ONCE. Plotting two real datasets each with `label="real"` put "real" in the
        # legend twice, which reads as two different things.
        ax_example.plot(np.arange(1, rates.size + 1), rates, marker="*", markersize=7,
                        color=style.STAR, linestyle=":",
                        label=None if (real and "real" not in _labelled) else "_nolegend_")
        _labelled.add("real")
    ax_example.set_xlabel("contiguous row block")
    ax_example.set_ylabel("positive-label rate in the block")
    ax_example.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax_example.set_ylim(bottom=0)
    style.title(ax_example, "Label rate by row block")
    style.legend_below(fig, ncol=4)
    return fig


# ---------------------------------------------------------------------------
# 4. Is the synthetic task the right difficulty?
# ---------------------------------------------------------------------------


def plot_difficulty_calibration(
    variants: dict[str, list[Any]],
    real_scores: dict[str, float] | None = None,
    task: str = "lgd",
    max_datasets: int = 40,
):
    """Is the synthetic task as hard as the real one?

    INVISIBLE IN EVERY OTHER FIGURE, and it decides whether the prior teaches anything useful.
    A prior whose tasks are trivially easy teaches the model that features determine the target
    almost exactly; one whose tasks are noise teaches it to predict the mean. Real credit data
    is neither — it is *low signal but not zero*, and that is the band the prior should land in.

    Measured with a small ExtraTrees, the same family TabICL uses for its own predictability
    filter, so "difficulty" here means the same thing it does upstream.
    """
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.48))

    scores_per_variant: dict[str, list[float]] = {}
    for name, tasks in variants.items():
        scores = []
        for t in list(tasks)[:max_datasets]:
            score = _quick_score(t, task)
            if score is not None:
                scores.append(score)
        if scores:
            scores_per_variant[name] = scores

    positions = np.arange(len(scores_per_variant))
    for i, (name, scores) in enumerate(scores_per_variant.items()):
        colour = style.variant_colour(name, i)
        jitter = (np.random.default_rng(i).random(len(scores)) - 0.5) * 0.30
        ax.scatter(np.full(len(scores), i) + jitter, scores, s=16, color=colour, alpha=0.75,
                   linewidths=0, zorder=3)
        # Median bar no wider than the jitter it summarises. A bar spanning the whole column
        # dwarfed the points and read as the subject rather than the summary.
        ax.plot([i - 0.17, i + 0.17], [np.median(scores)] * 2, color=style.INK, linewidth=1.6,
                zorder=4, solid_capstyle="butt")

    shown = [s for scores in scores_per_variant.values() for s in scores if np.isfinite(s)]
    if real_scores:
        values = np.asarray(list(real_scores.values()), dtype=float)
        # 10th-90th percentile, not min-max: one small real dataset scores R^2 = -4.8, and a
        # min-max band would cover everything and therefore say nothing. In the LEGEND, never as
        # text on the axes, where it sat on the points.
        lo, hi = (float(np.percentile(values, 10)), float(np.percentile(values, 90)))
        ax.axhspan(lo, hi, color=style.REAL, alpha=0.12, zorder=1,
                   label="real datasets, 10th-90th percentile")
        ax.axhline(float(np.median(values)), color=style.REAL, linestyle="--", linewidth=1.0,
                   zorder=3, label="real datasets, median")
        shown += [lo, hi]
    # The view is the data's range — never a fixed one (it ran to 20 before, flattening every
    # point onto the zero line) and never below -1, where one outlier would do the same.
    if shown:
        top = min(1.02, max(shown) + 0.05)
        bottom = max(-1.0, min(shown) - 0.05)
        ax.set_ylim(bottom, top if top > bottom else bottom + 1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels([style.variant_label(n) for n in scores_per_variant])
    ax.set_ylabel("R²" if task == "lgd" else "ROC-AUC")
    style.legend_below(ax, ncol=2)
    return fig


def _quick_score(task_obj: Any, task: str) -> float | None:
    """One cheap predictability score for a synthetic dataset. `None` if it cannot be scored."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import r2_score, roc_auc_score

    X = np.asarray(task_obj.X, dtype=float)
    y = np.asarray(task_obj.y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] < 40:
        return None
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    cut = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    if len(yte) < 10:
        return None
    try:
        if task == "pd":
            ytr_b, yte_b = (ytr >= 0.5).astype(int), (yte >= 0.5).astype(int)
            if len(np.unique(ytr_b)) < 2 or len(np.unique(yte_b)) < 2:
                return None
            model = ExtraTreesClassifier(n_estimators=25, max_depth=6, random_state=0)
            model.fit(Xtr, ytr_b)
            return float(roc_auc_score(yte_b, model.predict_proba(Xte)[:, 1]))
        model = ExtraTreesRegressor(n_estimators=25, max_depth=6, random_state=0)
        model.fit(Xtr, ytr)
        return float(r2_score(yte, model.predict(Xte)))
    except Exception:  # noqa: BLE001 — one unscoreable dataset must not kill the figure
        return None


# ---------------------------------------------------------------------------
# 5. What does the model actually see?
# ---------------------------------------------------------------------------


def _texture_columns(X: np.ndarray, n_cols: int) -> list[int]:
    """The `n_cols` columns with the most distinct values, in their original order.

    The first `n_cols` columns of a real table are often one-hot flags or IDs, which render as a
    flat white panel; a generated table is zero-padded, so its tail columns are empty. Neither says
    anything about texture. The columns that vary most are the fair sample of both.
    """
    counts = []
    for c in range(X.shape[1]):
        col = X[:, c]
        col = col[np.isfinite(col)]
        counts.append(len(np.unique(col)) if col.size else 0)
    order = np.argsort(counts)[::-1][:n_cols]
    return sorted(int(c) for c in order if counts[c] > 1)


def plot_side_by_side_tables(
    synthetic: Any,
    real: Any | None = None,
    n_rows: int = 12,
    n_cols: int = 7,
    task: str = "lgd",
    real_name: str | None = None,
):
    """One real table and one synthetic table, same layout, a few rows each.

    THE MOST CONVINCING FIGURE FOR A READER, and the simplest. Every other figure here is a
    summary statistic; this is the thing itself. If the synthetic table obviously does not look
    like the real one, no distance metric will rescue it — and if it does, a reader believes the
    rest of the paper more readily.

    Values are shown as a heatmap with the target as a separate final column, because the point
    is the *texture* — how much variation, how many repeated values, where the missing cells
    are — not the individual numbers.
    """
    style.apply()
    panels = [("credit prior, one generated task", synthetic, style.CREDIT)]
    if real is not None:
        panels.append((f"{real_name or 'real credit data'}, real", real, style.REAL))

    fig, axes = plt.subplots(1, len(panels), figsize=style.grid_figsize(len(panels), 1,
                                                                        panel_ratio=1.05),
                             squeeze=False)
    for i, (label, obj, colour) in enumerate(panels):
        ax = axes[0][i]
        X = np.asarray(obj.X, dtype=float)
        y = np.asarray(obj.y, dtype=float).ravel()
        # A fixed random sample of rows (seed 0), not the head: a real table's first rows encode
        # whatever its file order encodes.
        rows = np.sort(np.random.default_rng(0).choice(len(y), size=min(n_rows, len(y)),
                                                       replace=False))
        cols = _texture_columns(X[rows], n_cols) or list(range(min(n_cols, X.shape[1])))
        X = X[np.ix_(rows, cols)]
        y = y[rows]
        # Per-column rank normalisation, so one wide-scale column does not flatten the rest to
        # a single shade. The texture is what matters, not the units.
        shown = np.zeros_like(X)
        for c in range(X.shape[1]):
            col = X[:, c]
            finite = np.isfinite(col)
            if finite.sum() > 1 and np.ptp(col[finite]) > 0:
                shown[:, c] = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col) + 1e-12)
        # The target on the same 0-1 shade scale as the features: LGD is on [0, 1] already, a
        # default flag is 0 or 1, and anything else is min-max scaled like a feature.
        yr = np.ptp(y[np.isfinite(y)]) if np.isfinite(y).any() else 0.0
        y_shown = (y - np.nanmin(y)) / yr if yr > 0 and (np.nanmin(y) < 0 or np.nanmax(y) > 1) else y
        grid = np.column_stack([shown, y_shown])
        ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        # A line before the last column: the target is not a feature and should not read as one.
        ax.axvline(X.shape[1] - 0.5, color=style.INK, linewidth=1.2)
        ax.set_xticks(list(range(X.shape[1])) + [X.shape[1]])
        ax.set_xticklabels([f"f{c}" for c in cols] + ["y"],
                           fontsize=mpl.rcParams["xtick.labelsize"] * 0.85)
        ax.set_xlabel("feature (the most varied columns) and target y",
                      fontsize=mpl.rcParams["xtick.labelsize"])
        # Row NUMBERS, not target values. The target is already the last column, so printing
        # it again down the side said the same thing twice and invited the reader to think the
        # left-hand numbers were a different quantity.
        ax.set_yticks(range(len(y)))
        ax.set_yticklabels([str(r + 1) for r in range(len(y))],
                           fontsize=mpl.rcParams["ytick.labelsize"] * 0.85)
        ax.set_ylabel("row (random sample)", fontsize=mpl.rcParams["ytick.labelsize"])
        ax.grid(visible=False)
        # All four spines, so the coloured frame closes. The project style hides top and right,
        # which left these panels framed on two sides and looking unfinished.
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(colour)
            sp.set_linewidth(1.2)
        ax.set_title(label, loc="left", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# 6. Where does the boundary mass come from?
# ---------------------------------------------------------------------------


def plot_boundary_mass_sources(
    variants: dict[str, list[Any]],
    real: dict[str, Any] | None = None,
):
    """Raw credit mass at economic 0/1, real datasets as stars; no generic-scale points.

    A refinement of a figure that already existed rather than a new idea. The reason to keep it
    is that "total boundary mass" hides the asymmetry that matters: a portfolio where most
    defaults recover in full (mass at 0) is a completely different book from one where most are
    a total loss (mass at 1), and both can share a total. Splitting the axes shows which of the
    two a prior actually produces, and whether it lands where the real datasets do.
    """
    style.apply()
    names = [name for name in variants if name.startswith("credit")]
    if not names:
        from src.visualize.campaign import availability
        return availability("Raw credit loss targets are required for the economic endpoint-mass view.")
    n = len(names)
    if any(float(t.y.min()) < 0 or float(t.y.max()) > 1 for name in names for t in variants[name]):
        raise ValueError("The economic endpoint panel requires raw [0,1] credit targets")
    fig, axes = plt.subplots(1, n, figsize=style.grid_figsize(n, 1, panel_ratio=1.0),
                             squeeze=False, sharex=True, sharey=True)
    real_points = [
        (float((np.asarray(y) <= 1e-4).mean()), float((np.asarray(y) >= 1-1e-4).mean()))
        for y in _real_targets("lgd", real).values()
    ]
    real_names = [s.split(".", 1)[-1].removeprefix("lgd_") for s in _real_targets("lgd", real)]
    for i, name in enumerate(names):
        ax = axes[0][i]
        colour = style.variant_colour(name, i)
        pts = [(float((np.asarray(t.y) <= 1e-4).mean()), float((np.asarray(t.y) >= 1-1e-4).mean()))
               for t in variants[name]]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=14, color=colour, alpha=0.6, linewidths=0, zorder=2,
                       label="one synthetic task" if i == 0 else None)
        # STARS ON TOP, IN MAGENTA. Two separate visibility failures were stacked here: at
        # s=55 behind s=26 dots the stars were buried, and `style.REAL` is orange — the same
        # hue as the credit variant's own points, so where they overlapped nothing was
        # distinguishable. `style.STAR` appears nowhere else in the palette.
        for k, (rx, ry) in enumerate(real_points):
            ax.scatter([rx], [ry], marker="*", s=90, color=style.STAR, zorder=5,
                       edgecolors="white", linewidths=0.7,
                       label="one real dataset" if (i == 0 and k == 0) else None)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Raw loss mass at 0 · full recovery")
        ax.set_title(f"Credit tasks (n={len(variants[name])}) and real books", loc="left")
    axes[0][0].set_ylabel("Raw loss mass at 1 · total loss")
    from matplotlib.ticker import PercentFormatter
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    style.legend_below(fig, ncol=3)
    if real_points:
        for ax in axes.flat:
            style.place_labels(ax, *np.asarray(real_points).T, real_names,
                               fontsize=style.PANEL_TEXT, background=True)
    return fig


def plot_standardised_endpoint_ties(variants):
    """Endpoint ties after a common context-fitted affine map, with no loss semantics."""
    from matplotlib.ticker import PercentFormatter
    import torch
    from src.prior.generator import standardise_on_context

    style.apply()
    if not variants:
        raise ValueError("At least one prior is required for endpoint ties")
    fig, axes = plt.subplots(1, len(variants), squeeze=False, sharex=True, sharey=True,
                             figsize=style.figsize(style.WIDTH_FULL, .52))
    for i, ((name, tasks), ax) in enumerate(zip(variants.items(), axes.flat)):
        points = []
        for task in tasks:
            if "train_size" not in task.meta:
                raise ValueError("A recorded context split is required for comparable standardised ties")
            n_context = int(task.meta["train_size"])
            z = np.asarray(standardise_on_context(torch.as_tensor(task.y), n_context), dtype=float)
            # A shared 1e-4 tolerance on the standardised view, not economic zero/one.
            points.append((float(np.mean(np.abs(z-z.min()) <= 1e-4)),
                           float(np.mean(np.abs(z-z.max()) <= 1e-4))))
        if points:
            x, y = np.asarray(points).T
            ax.scatter(x, y, color=style.variant_colour(name, i), alpha=.45,
                       s=style.SEED_POINT, marker=style.SOURCE_MARKERS[i % 2])
        ax.set(xlabel="Rows at the table minimum", xlim=(-.02, 1.02), ylim=(-.02, 1.02),
               xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
        style.title(ax, f"{style.variant_label(name).capitalize()} · {len(points)} tables")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].set_ylabel("Rows at the table maximum")
    fig.suptitle("Repeated values at each target extreme")
    return fig
