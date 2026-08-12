"""Look at what the prior actually generates.

The point of these plots is to catch, by eye, things a summary statistic hides:
a target that is secretly discrete, features that are all copies of each other, a
"credit" prior that is producing the same shape every time. Numbers in a log tell
you the mean boundary mass; a grid of 100 histograms tells you whether the family
is genuinely varied or whether one setting dominates.

Every function samples with **exactly the same code path the training run uses** —
`TaskGenerator` from a config — so what you see is what the model trains on, not a
demonstration built separately.

`notebooks/prior_visualisation.ipynb` calls these and holds no logic of its own.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG
from src.utils.config import expand_with_seeds, load
from src.utils.target_stats import target_stats
from src.visualize import style


def sample_tasks(
    config_path: str,
    n: int = 100,
    *,
    credit_fraction: float | None = None,
    seed: int = 0,
    grid_index: int = 0,
) -> tuple[list[Any], dict[str, Any]]:
    """Draw `n` tasks exactly as training would, plus a summary dict.

    `credit_fraction=None` uses whatever the config says. Override it to compare
    the original prior (0.0) against ours (1.0) side by side.
    """
    cfg = expand_with_seeds(load(config_path))[grid_index]
    task = cfg["task"]
    if credit_fraction is not None:
        cfg["prior"]["credit_fraction"] = credit_fraction

    gen = TaskGenerator(cfg["prior"], task, PriorRNG(seed))
    tasks = [gen.sample() for _ in range(n)]

    info = {
        "config": config_path,
        "task": task,
        "credit_fraction": cfg["prior"]["credit_fraction"],
        "n_sampled": n,
        "sources": {s: sum(t.source == s for t in tasks) for s in ("base", "credit")},
        "filter": gen.filter_summary(),
    }
    return tasks, info


# ---------------------------------------------------------------------------
# The target — the thing this project is actually about
# ---------------------------------------------------------------------------



def plot_boundary_mass(tasks: list[Any], real_reference: dict[str, tuple[float, float]] | None = None):
    """Mass at each boundary, one point per task, with the real datasets overlaid.

    This is the plot that answers "does the prior look like the data?". The real
    datasets are the target; the cloud is what we generate.
    """
    style.apply()
    stats = [target_stats(t.y) for t in tasks]
    at_min = np.array([s["frac_at_min"] for s in stats])
    at_max = np.array([s["frac_at_max"] for s in stats])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.40))

    ax1.scatter(at_min, at_max, s=34, alpha=0.5, color=style.CREDIT,
                edgecolor="white", linewidth=0.4, label="sampled tasks", zorder=3)
    if real_reference:
        for name, (m0, m1) in real_reference.items():
            ax1.scatter([m0], [m1], marker="*", s=300, color=style.REAL,
                        edgecolor="white", linewidth=0.8, zorder=5)
            ax1.annotate(name, (m0, m1), fontsize=8, color=style.REAL, weight="semibold",
                         xytext=(7, 5), textcoords="offset points")
        ax1.scatter([], [], marker="*", s=200, color=style.REAL, label="real datasets")
    ax1.set_xlabel("mass at the low boundary (full recovery)")
    ax1.set_ylabel("mass at the high boundary (total loss)")
    ax1.grid(axis="x")
    style.title(ax1, "Where the boundary mass sits",
                "The cloud is what we generate; the stars are what we must cover")
    ax1.legend(loc="upper right")

    total = at_min + at_max
    ax2.hist(total, bins=25, color=style.CREDIT)
    ax2.axvline(float(total.mean()), color=style.INK, lw=1.6, ls="--")
    ax2.annotate(f"mean {total.mean():.3f}", (float(total.mean()), ax2.get_ylim()[1] * 0.94),
                 fontsize=9, color=style.INK, xytext=(6, 0), textcoords="offset points")
    ax2.set_xlabel("total boundary mass")
    ax2.set_ylabel("number of tasks")
    style.title(ax2, "Total boundary mass",
                "A spread, not a spike — the prior samples a family")
    return fig


# ---------------------------------------------------------------------------
# Shape of the tables
# ---------------------------------------------------------------------------


def plot_table_shapes(tasks: list[Any]):
    """Rows, features, and how many distinct values the target takes.

    The distinct-value plot is worth more than it looks: a target with very few
    distinct values is effectively discrete, which happens when a tree or
    discretisation function ends up driving it. That is a real property of the
    prior, not a bug, but you want to know how often it happens.
    """
    style.apply()
    rows = np.array([t.n_rows for t in tasks])
    feats = np.array([t.n_features for t in tasks])
    distinct = np.array([len(np.unique(t.y.numpy())) / max(t.n_rows, 1) for t in tasks])

    panels = [
        (rows, "rows per task", "context length the model must handle"),
        (feats, "features per task", "table width"),
        (distinct, "distinct target values / rows", "low = effectively discrete"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=style.figsize(style.WIDTH_FULL, 0.32))
    for ax, (data, label, hint) in zip(axes, panels):
        ax.hist(data, bins=25, color=style.CREDIT)
        ax.axvline(float(np.median(data)), color=style.INK, lw=1.5, ls="--")
        ax.set_xlabel(label)
        ax.set_ylabel("number of tasks")
        style.title(ax, f"median {np.median(data):.3g}", hint)
    fig.suptitle("Shape of the generated tables")
    return fig



def plot_correlation_spectrum(tasks: list[Any]):
    """Eigenvalue spectrum of the feature correlation matrix, pooled.

    The same quantity O'Prior compares against real tables. A spectrum that decays
    fast means a few directions explain everything (strong shared structure); a
    flat spectrum means near-independent features.
    """
    spectra = []
    for t in tasks:
        X = t.X.numpy()
        if X.shape[1] < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.nan_to_num(np.corrcoef(X, rowvar=False))
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        spectra.append(ev / max(ev[0], 1e-9))

    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, 0.62))
    for ev in spectra[:60]:
        ax.plot(np.arange(1, len(ev) + 1) / len(ev), ev,
                color=style.CREDIT, alpha=0.22, linewidth=0.9)
    if spectra:
        # The median curve, so the eye has something to hold on to in the spaghetti.
        grid = np.linspace(0, 1, 50)
        resampled = np.array([
            np.interp(grid, np.arange(1, len(ev) + 1) / len(ev), ev) for ev in spectra
        ])
        ax.plot(grid, np.median(resampled, axis=0), color=style.INK, lw=2.4, label="median")
        ax.legend()
    ax.set_xlabel("eigenvalue rank (normalised)")
    ax.set_ylabel("eigenvalue / largest")
    ax.grid(axis="x")
    style.title(
        ax, f"Correlation spectrum, {len(spectra)} tasks",
        "Fast decay = strong shared structure; flat = near-independent features",
    )
    return fig


def plot_feature_target_relation(tasks: list[Any], n_show: int = 8):
    """Target against its most-correlated feature, per task.

    Shows what the model is actually asked to learn: a clean trend, a step from a
    threshold rule, or a cloud. Horizontal bands at 0 and 1 are the censoring.
    """
    style.apply()
    n_show = min(n_show, len(tasks))
    ncols = 4
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=style.grid_figsize(ncols, nrows, panel_ratio=0.86))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_show:
            ax.axis("off")
            continue
        X, y = tasks[i].X.numpy(), tasks[i].y.numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
        j = int(np.argmax(np.abs(corr)))
        ax.scatter(X[:, j], y, s=7, alpha=0.28, color=style.source_color(tasks[i].source),
                   edgecolor="none")
        # Mark the boundaries: horizontal bands sitting exactly on these lines are
        # the censoring, and that is the thing worth seeing in this plot.
        for edge in (float(y.min()), float(y.max())):
            ax.axhline(edge, color=style.REAL, lw=0.7, ls=":", alpha=0.6)
        style.title(ax, f"feature {j}", f"r = {corr[j]:+.2f}")
        ax.set_xticks([])
        ax.set_ylabel("target")
    fig.suptitle("Target vs its most correlated feature")
    style.figure_note(
        fig, "Flat bands on the dotted lines are the boundary atoms. A clean trend, a "
        "step from a threshold rule, and a cloud are all things the prior should produce."
    )
    return fig


def compare_priors(config_path: str, n: int = 100, seed: int = 0):
    """Original prior vs ours, side by side. The motivating figure.

    Returns (fig, summary_dict). The summary is what belongs in a results file;
    the figure is what belongs in the paper.
    """
    base_tasks, base_info = sample_tasks(config_path, n, credit_fraction=0.0, seed=seed)
    ours_tasks, ours_info = sample_tasks(config_path, n, credit_fraction=1.0, seed=seed)
    task = base_info["task"]

    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, 0.66))
    arms = [
        (base_tasks, "original TabICL prior", style.ORIGINAL),
        (ours_tasks, "our prior", style.CREDIT),
    ]
    for row, (tasks, label, colour) in enumerate(arms):
        ys = [t.y.numpy() for t in tasks]
        pooled = np.concatenate(ys)
        axes[row, 0].hist(pooled, bins=60, color=colour)
        axes[row, 0].set_ylabel("count")
        axes[row, 0].set_xlabel("target value")
        in_unit = float(((pooled >= 0) & (pooled <= 1)).mean())
        style.title(axes[row, 0], f"{label} — pooled target",
                    f"{in_unit:.0%} of values inside [0, 1]")

        stats = [target_stats(t.y) for t in tasks]
        total = np.array([s["frac_at_min"] + s["frac_at_max"] for s in stats])
        axes[row, 1].hist(total, bins=25, color=colour)
        axes[row, 1].set_xlabel("total boundary mass")
        axes[row, 1].set_ylabel("number of tasks")
        with_atoms = float((total > 0.01).mean())
        style.title(axes[row, 1], f"{label} — boundary mass",
                    f"mean {total.mean():.3f}; {with_atoms:.0%} of tasks have any")

    fig.suptitle(f"{task.upper()}: what changes when we swap the prior")
    style.figure_note(
        fig, "Top row is the control (TabICL unchanged); bottom row is ours. "
        "The difference in the right-hand column is the mechanism this project adds."
    )

    def _summary(tasks, info):
        stats = [target_stats(t.y) for t in tasks]
        return {
            **info,
            "frac_at_min_mean": float(np.mean([s["frac_at_min"] for s in stats])),
            "frac_at_max_mean": float(np.mean([s["frac_at_max"] for s in stats])),
            "tasks_in_unit_interval": float(np.mean([s["in_unit_interval"] for s in stats])),
            "tasks_with_any_atom": float(np.mean([s["has_atom_at_min"] or s["has_atom_at_max"] for s in stats])),
        }

    return fig, {"original": _summary(base_tasks, base_info), "ours": _summary(ours_tasks, ours_info)}
