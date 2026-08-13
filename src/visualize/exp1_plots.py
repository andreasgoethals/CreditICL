"""Figures for Exp1, whose question is: **which of our 32 priors is best?**

WHAT THIS REPLACED, AND WHY

The earlier figures showed the prior in general — 100 target histograms, correlation heatmaps of
random synthetic features, a target histogram for PD. They looked like analysis and answered
nothing:

* **a PD target histogram** is a bar at 0 and a bar at 1. That is one number, the default rate,
  drawn as a picture.
* **correlation heatmaps of synthetic features** show that random graphs produce random
  correlations. There is nothing in them to learn.
* **100 thumbnail histograms** cannot be compared by eye, which is the only thing they permit.

Every figure here answers a question that changes what we do next:

1. `plot_prior_realism_ranking` — which priors even look like real credit data?
2. `plot_mechanism_decomposition` — do LGD's boundary atoms *come from* the loan economics?
3. `plot_default_clustering` — do PD defaults arrive in waves, as real ones do?
4. `plot_difficulty_calibration` — is the synthetic task the right difficulty?
5. `plot_side_by_side_tables` — what does the model actually see?
6. `plot_boundary_mass_sources` — where does the boundary mass come from?
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
    """THE MONEY FIGURE. One row per prior, sorted by how close its targets are to real data.

    Exp1 exists to rank 32 priors. This is that ranking, as a single readable chart: for each
    prior, the distance between its target distribution and each real dataset's, with the mean
    marked. A reader learns in one glance which priors are candidates and which are not.

    A distance is used rather than side-by-side histograms because 32 histograms cannot be
    compared by eye — which is exactly why the old figures answered nothing.
    """
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
        # what the variant produces ON AVERAGE, not what one draw did.
        pooled = np.concatenate([np.asarray(t.y, dtype=float).ravel() for t in tasks])
        pooled = np.clip(pooled, 0.0, 1.0)
        per_real = {r: distribution_distance(pooled, y) for r, y in reals.items()}
        rows.append((name, float(np.mean(list(per_real.values()))), per_real))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=style.row_figsize(len(rows), per_row=0.30, base=1.3))
    labels = [r[0] for r in rows]
    ypos = np.arange(len(rows))
    for i, (_, mean_d, per_real) in enumerate(rows):
        values = list(per_real.values())
        # A guide line per row. Without it the eye cannot carry a dot at x=0.6 back to its
        # label three rows up, which is the one thing this chart is for.
        ax.plot([min(values), max(values)], [i, i], color=style.GRID, linewidth=3.0,
                solid_capstyle="round", zorder=1)
        # Every real dataset as a dot, so the SPREAD is visible: a prior that matches one
        # dataset and misses six is not a good prior, and a mean alone would hide that.
        ax.scatter(values, np.full(len(values), i), s=18, color=style.MUTED, alpha=0.85,
                   zorder=2, linewidths=0)
        ax.scatter([mean_d], [i], s=58, color=style.CREDIT, zorder=3, marker="D",
                   edgecolors="white", linewidths=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("distance from real credit targets  (total variation, 0 = identical)")
    ax.set_xlim(left=0)
    ax.grid(visible=True, axis="x")
    ax.grid(visible=False, axis="y")
    # Subtitle deliberately short. A long one wraps to three bold lines and eats a third of a
    # 2.2 in figure; the detail belongs in the note underneath, where it costs one grey line.
    style.title(ax, "Which prior looks most like real credit data?", "Lower is better")
    style.figure_note(
        fig, "Diamond = mean over the real datasets, dots = one per dataset. Total variation "
             "on pooled targets, 40 fixed bins.")
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
    fig.suptitle("Where the boundary atoms come from")
    counts_note = ", ".join(f"{k}: {len(v)}" for k, v in groups.items())
    style.figure_note(
        fig, f"Targets pooled per mechanism ({counts_note}). Mass at 0 and 1 is a consequence "
             f"of the loss story, not a parameter.")
    return fig


# ---------------------------------------------------------------------------
# 3. Do PD defaults cluster, as real ones do?
# ---------------------------------------------------------------------------


def plot_default_clustering(
    variants: dict[str, list[Any]],
    real: dict[str, Any] | None = None,
    n_cohorts: int = 12,
):
    """FAR MORE INFORMATIVE FOR PD THAN ANY TARGET HISTOGRAM.

    Real credit's defining feature is that defaults arrive in **waves**: a recession lifts
    everyone's risk at once, so the default rate moves between cohorts far more than
    independent coin flips would. An i.i.d. prior cannot produce that, and our Vasicek
    one-factor mechanism is there precisely to.

    So: split each dataset into cohorts, and plot the SPREAD of the default rate across them.
    A flat line means independence; a wide spread means clustering. The reference is what the
    spread would be if rows were independent at the same base rate — drawn as a dashed line,
    because "more than chance" is the whole claim and needs a yardstick.
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
            rows.append((name, observed, expected, style.SERIES[i % len(style.SERIES)]))

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
                np.full(len(real_ratios), len(rows)), real_ratios, marker="*", s=70,
                color=style.REAL, zorder=4, label="real datasets",
            )
    ax_spread.axhline(1.0, color=style.MUTED, linestyle="--", linewidth=0.9)
    ax_spread.annotate("independent rows", (0.02, 1.0), xycoords=("axes fraction", "data"),
                       fontsize=7, color=style.MUTED, va="bottom")
    labels = [r[0] for r in rows] + (["real"] if real else [])
    ax_spread.set_xticks(range(len(labels)))
    ax_spread.set_xticklabels(labels, rotation=20, ha="right")
    ax_spread.set_ylabel("cohort spread / chance")
    style.title(ax_spread, "Do defaults cluster?", "Above 1 = waves, not coin flips")

    # RIGHT: one concrete example per variant, so the abstraction is grounded.
    for i, (name, tasks) in enumerate(variants.items()):
        pick = next((t for t in tasks if 0.0 < float((np.asarray(t.y) >= 0.5).mean()) < 1.0), None)
        if pick is None:
            continue
        rates = cohort_rates(np.asarray(pick.y))
        ax_example.plot(np.arange(1, rates.size + 1), rates, marker="o", markersize=3,
                        color=style.SERIES[i % len(style.SERIES)], label=name)
    for y in list(_real_targets("pd", real).values())[:2]:
        rates = cohort_rates(y)
        ax_example.plot(np.arange(1, rates.size + 1), rates, marker="*", markersize=6,
                        color=style.REAL, linestyle=":", label="real")
    ax_example.set_xlabel("cohort")
    ax_example.set_ylabel("default rate")
    ax_example.legend(loc="best", fontsize=7)
    style.title(ax_example, "One dataset each", "Flat = independent; jagged = shared shocks")
    style.figure_note(fig, f"Cohorts are {n_cohorts} contiguous blocks of rows.")
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
    for i, (_name, scores) in enumerate(scores_per_variant.items()):
        colour = style.SERIES[i % len(style.SERIES)]
        jitter = (np.random.default_rng(i).random(len(scores)) - 0.5) * 0.30
        ax.scatter(np.full(len(scores), i) + jitter, scores, s=16, color=colour, alpha=0.75,
                   linewidths=0, zorder=3)
        # Median bar no wider than the jitter it summarises. A bar spanning the whole column
        # dwarfed the points and read as the subject rather than the summary.
        ax.plot([i - 0.17, i + 0.17], [np.median(scores)] * 2, color=style.INK, linewidth=1.6,
                zorder=4, solid_capstyle="butt")

    if real_scores:
        values = np.asarray(list(real_scores.values()), dtype=float)
        # 10th-90th percentile, not min-max. One small real dataset scores R^2 = -4.8 under a
        # contiguous 70/30 split, and a min-max band would stretch from -4.8 to 0.7 — which
        # covers everything and therefore says nothing. The outlier is reported in the text
        # summary instead of being allowed to flatten the figure.
        lo, hi = (float(np.percentile(values, 10)), float(np.percentile(values, 90)))
        ax.axhspan(lo, hi, color=style.REAL, alpha=0.12, zorder=1)
        ax.axhline(float(np.median(values)), color=style.REAL, linestyle="--", linewidth=1.0,
                   zorder=3)
        # Anchored INSIDE the axes. At x=0.99 with ha="right" the text ran off the right edge
        # of the figure, because `constrained_layout` sizes to the axes and not to an
        # annotation hanging outside them.
        ax.annotate("real credit data", (0.985, hi), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color=style.REAL,
                    annotation_clip=False,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
        n_out = int(np.sum((values < lo) | (values > hi)))
        if n_out:
            ax.annotate(f"{n_out} real dataset(s) outside the band",
                        (0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=6.5, color=style.MUTED)
        # Clip the view to the interesting region. An R^2 of -4.8 on the axis compresses every
        # real difference into a few pixels.
        finite = [s for scores in scores_per_variant.values() for s in scores if np.isfinite(s)]
        if finite:
            floor = min(-0.2, float(np.percentile(finite, 2)))
            ax.set_ylim(bottom=max(floor, -1.0))

    ax.set_xticks(positions)
    ax.set_xticklabels(list(scores_per_variant), rotation=20, ha="right")
    ax.set_ylabel("R²" if task == "lgd" else "ROC AUC")
    style.title(ax, "Is the synthetic task the right difficulty?",
                "Shaded band = real credit data")
    style.figure_note(
        fig, "One point per synthetic dataset, bar = median. Small ExtraTrees on a 70/30 "
             "split — the same family as TabICL's own predictability filter.")
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


def plot_side_by_side_tables(
    synthetic: Any,
    real: Any | None = None,
    n_rows: int = 8,
    n_cols: int = 7,
    task: str = "lgd",
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
    panels = [("our synthetic prior", synthetic, style.CREDIT)]
    if real is not None:
        panels.append(("real credit data", real, style.REAL))

    fig, axes = plt.subplots(1, len(panels), figsize=style.grid_figsize(len(panels), 1,
                                                                        panel_ratio=1.05),
                             squeeze=False)
    for i, (label, obj, colour) in enumerate(panels):
        ax = axes[0][i]
        X = np.asarray(obj.X, dtype=float)
        y = np.asarray(obj.y, dtype=float).ravel()
        X = X[:n_rows, :n_cols]
        y = y[:n_rows]
        # Per-column rank normalisation, so one wide-scale column does not flatten the rest to
        # a single shade. The texture is what matters, not the units.
        shown = np.zeros_like(X)
        for c in range(X.shape[1]):
            col = X[:, c]
            finite = np.isfinite(col)
            if finite.sum() > 1 and np.ptp(col[finite]) > 0:
                shown[:, c] = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col) + 1e-12)
        grid = np.column_stack([shown, y])
        ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        # A line before the last column: the target is not a feature and should not read as one.
        ax.axvline(X.shape[1] - 0.5, color=style.INK, linewidth=1.2)
        ax.set_xticks(list(range(X.shape[1])) + [X.shape[1]])
        ax.set_xticklabels([f"f{c}" for c in range(X.shape[1])] + ["y"],
                           fontsize=mpl.rcParams["xtick.labelsize"] * 0.85)
        # Row NUMBERS, not target values. The target is already the last column, so printing
        # it again down the side said the same thing twice and invited the reader to think the
        # left-hand numbers were a different quantity.
        ax.set_yticks(range(len(y)))
        ax.set_yticklabels([str(r + 1) for r in range(len(y))],
                           fontsize=mpl.rcParams["ytick.labelsize"] * 0.85)
        ax.set_ylabel("row", fontsize=mpl.rcParams["ytick.labelsize"])
        ax.grid(visible=False)
        # All four spines, so the coloured frame closes. The project style hides top and right,
        # which left these panels framed on two sides and looking unfinished.
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(colour)
            sp.set_linewidth(1.2)
        style.title(ax, label, f"{X.shape[1]} features + target")
    fig.suptitle("What the model actually sees")
    style.figure_note(fig, f"First {n_rows} rows. Each feature min-max normalised within its "
                           f"own column, so shade shows relative value, not units. The rule "
                           f"separates the target from the features.")
    return fig


# ---------------------------------------------------------------------------
# 6. Where does the boundary mass come from?
# ---------------------------------------------------------------------------


def plot_boundary_mass_sources(
    variants: dict[str, list[Any]],
    real: dict[str, Any] | None = None,
):
    """Mass at 0 against mass at 1, one panel per variant, real datasets as stars.

    A refinement of a figure that already existed rather than a new idea. The reason to keep it
    is that "total boundary mass" hides the asymmetry that matters: a portfolio where most
    defaults recover in full (mass at 0) is a completely different book from one where most are
    a total loss (mass at 1), and both can share a total. Splitting the axes shows which of the
    two a prior actually produces, and whether it lands where the real datasets do.
    """
    style.apply()
    names = list(variants)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=style.grid_figsize(n, 1, panel_ratio=1.0),
                             squeeze=False, sharex=True, sharey=True)
    real_points = [
        (target_stats(y)["frac_at_min"], target_stats(y)["frac_at_max"])
        for y in _real_targets("lgd", real).values()
    ]
    for i, name in enumerate(names):
        ax = axes[0][i]
        colour = style.SERIES[i % len(style.SERIES)]
        pts = [(s["frac_at_min"], s["frac_at_max"])
               for s in (target_stats(t.y) for t in variants[name])]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=26, color=colour, alpha=0.75, linewidths=0, zorder=3)
        # Stars are the REFERENCE, so they are drawn smaller and behind. At s=80 they buried
        # the synthetic points completely, which inverted the figure: the subject vanished
        # under its own yardstick.
        for rx, ry in real_points:
            ax.scatter([rx], [ry], marker="*", s=55, color=style.REAL, zorder=2,
                       edgecolors="white", linewidths=0.5)
        # The line where the two atoms are equal. Above it a book is loss-heavy, below it
        # recovery-heavy, and which side a prior sits on is the readable fact.
        ax.plot([0, 1], [0, 1], color=style.MUTED, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("mass at 0 (full recovery)")
        style.title(ax, name)
    axes[0][0].set_ylabel("mass at 1 (total loss)")
    fig.suptitle("Which boundary does the mass sit on?")
    counts = ", ".join(f"{k}: {len(v)}" for k, v in variants.items())
    style.figure_note(fig, f"One point per synthetic dataset ({counts}); stars are the real "
                           f"LGD datasets. The dotted line is equal mass at both ends.")
    return fig
