"""Read-only diagnostic instrumentation of the actual accepted synthetic stream.

Unlike the binary illustrative draws, this calls production ``plan_batch`` and
``sample_slot`` without overriding class counts, filtering or target encoding.
The observer is local to these CPU diagnostics; training code is not modified.
"""
from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG
from src.prior.stream import plan_batch
from src.visualize import style


class _ObservedGenerator(TaskGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []
        original_accept = self.filter.accept

        def observe_accept(X, y, *, is_classif):
            before = len(self.filter.stats.pseudo_r2)
            accepted = original_accept(X, y, is_classif=is_classif)
            row = self.observations[-1]
            row["statistically_judged"] = True
            if len(self.filter.stats.pseudo_r2) > before:
                row["difficulty"] = self.filter.stats.pseudo_r2[-1]
            return accepted

        self.filter.accept = observe_accept

    def _candidate(self, plan, rng, seed):
        task = super()._candidate(plan, rng, seed)
        self.observations.append({"task": task, "source": "credit" if plan.use_credit else "generic",
                                  "slot": plan.slot, "accepted": False, "valid": False,
                                  "statistically_judged": False, "difficulty": np.nan})
        return task

    def _valid(self, task, plan):
        valid = super()._valid(task, plan)
        self.observations[-1]["valid"] = valid
        return valid


def illustrative_variants(config_path, *, n=500, seed=0):
    """Fresh unfiltered binary/raw-loss illustrations, never stale cached pools."""
    from src.utils.config import expand_with_seeds, load
    from src.visualize.draw import draw

    config = expand_with_seeds(load(config_path))[0]
    loaded = {}
    for name, mixture in (("original", 0.), ("credit", 1.)):
        prior = copy.deepcopy(config["prior"])
        prior["credit_fraction"] = mixture
        prior["filter"]["mode"] = "off"
        loaded[name], _ = draw(config["task"], prior, n, seed, credit_reading=True)
    return loaded, "fresh live unfiltered diagnostic; binary PD / raw credit LGD"


def accepted_stream(config, *, n=48, seed=1729, mixtures=(0., .5, 1.)):
    """Candidate and accepted task observations for a small matched slot audit."""
    from src.prior.filters import predictability

    prior = config.get("prior", config)
    rows = []
    for mixture in mixtures:
        cfg = copy.deepcopy(prior)
        cfg["credit_fraction"] = mixture
        generator = _ObservedGenerator(cfg, "pd", PriorRNG(seed))
        plans = plan_batch(seed, 0, cfg, n, regression=False, credit_fraction=mixture)
        for plan in plans.slots:
            accepted = generator.sample_slot(plan)
            next(r for r in reversed(generator.observations) if r["task"] is accepted)["accepted"] = True
        for event in generator.observations:
            task = event.pop("task")
            row = {**event, "mixture": mixture, "seed": seed, "n_accepted_requested": n,
                   "classes": np.nan, "minority_share": np.nan,
                   "features": np.nan, "low_cardinality_share": np.nan}
            if task is not None:
                X, y = task.X.numpy(), task.y.numpy()
                _, counts = np.unique(y, return_counts=True)
                width = int(task.meta.get("d", X.shape[1]))
                row.update(classes=len(counts), minority_share=float(counts.min()/counts.sum()),
                           features=width, low_cardinality_share=float(np.mean([
                               np.unique(X[:, j]).size <= 20 for j in range(width)])) if width else np.nan)
                if row["valid"] and not np.isfinite(row["difficulty"]):
                    # Same OOB surrogate; the small bootstrap count here does
                    # not alter its pseudo-R² or any production acceptance decision.
                    row["difficulty"] = predictability(task.X, task.y, is_classif=True, n_boot=20)[1]
            rows.append(row)
    return pd.DataFrame(rows)


def _ecdf(ax, values, *, label, color, linestyle="-"):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if len(values):
        ax.step(values, np.arange(1, len(values)+1)/len(values), where="post",
                color=color, linestyle=linestyle, label=label)


def class_exposure(frame):
    style.apply()
    accepted = frame[frame.accepted]
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, .55))
    handles = []
    for mixture, rows in accepted.groupby("mixture", sort=True):
        color = style.credit_fraction_colour(mixture)
        counts = rows.classes.value_counts(normalize=True).reindex(range(2, 11), fill_value=0)
        name = style.prior_mix_label(mixture)
        axes[0].plot(counts.index, counts, marker="o", color=color)
        binary = rows[rows.classes.eq(2)]
        _ecdf(axes[1], binary.minority_share, label="_nolegend_", color=color)
        handles.append(Line2D([], [], color=color, marker="o",
                              label=f"{name}: {len(rows)} tasks, binary n={len(binary)}"))
    axes[0].set(xlabel="Observed classes per accepted table", ylabel="Share of accepted tables",
                xticks=[2, 4, 6, 8, 10], title="Production class-count exposure")
    axes[1].set(xlabel="Minority share, binary tables only", ylabel="Cumulative share of tables",
                xlim=(0, .5), title="Binary slice of the same stream")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    style.legend_below(fig, handles, ncol=1)
    return fig


def acceptance_and_difficulty(frame):
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, .58))
    mixtures = sorted(frame.mixture.unique())
    for source, offset, color in (("generic", -.16, style.ORIGINAL), ("credit", .16, style.CREDIT)):
        for i, mixture in enumerate(mixtures):
            rows = frame[frame.source.eq(source) & frame.mixture.eq(mixture)]
            if len(rows):
                rate = rows.accepted.sum()/len(rows)
                axes[0].bar(i+offset, rate, .28, color=color)
                axes[0].text(i+offset, rate+.025, f"{int(rows.accepted.sum())}/{len(rows)}",
                             ha="center", fontsize=style.PANEL_TEXT)
        valid = frame[frame.source.eq(source) & frame.valid]
        _ecdf(axes[1], valid.difficulty, label=f"{source}: candidates n={len(valid)}", color=color, linestyle="--")
        _ecdf(axes[1], valid[valid.accepted].difficulty,
              label=f"{source}: accepted n={int(valid.accepted.sum())}", color=color)
    axes[0].set(xticks=range(len(mixtures)), xticklabels=[f"{m:.0%}" for m in mixtures],
                xlabel="Configured credit mixture", ylabel="Accepted / all candidates", ylim=(0, 1.12),
                title="Validity + configured acceptance")
    axes[0].set_yticks([0, .25, .5, .75, 1])
    axes[1].set(xlabel="ExtraTrees OOB pseudo-R²", ylabel="Cumulative share of valid tasks",
                title="Candidate versus delivered difficulty")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    style.legend_below(fig, ncol=2)
    return fig


def feature_exposure(frame):
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, .5))
    for mixture, rows in frame[frame.accepted].groupby("mixture"):
        color = style.credit_fraction_colour(mixture)
        label = f"{style.prior_mix_label(mixture)}, n={len(rows)}"
        _ecdf(axes[0], rows.features, label=label, color=color)
        _ecdf(axes[1], rows.low_cardinality_share, label=label, color=color)
    axes[0].set(xlabel="Non-padding features per table", ylabel="Cumulative share of accepted tables")
    axes[1].set(xlabel="Share of features with ≤20 distinct values", ylabel="Cumulative share of accepted tables")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    style.legend_below(fig, ncol=3)
    return fig


def summary(frame):
    accepted = frame[frame.accepted]
    return ("ACCEPTED PRODUCTION-STREAM AUDIT\n"
            f"Seed {int(frame.seed.iloc[0])}; {int(frame.n_accepted_requested.iloc[0])} accepted slots per mixture; "
            "same slot shapes and native 2–10-class control route.\n"
            + accepted.groupby('mixture').agg(tables=('accepted', 'size'), binary=('classes', lambda x: int(x.eq(2).sum()))).to_string()
            + "\nCandidates include invalid attempts; credit bypasses only the statistical filter, not validity. "
            "Difficulty is an OOB pseudo-R² diagnostic, not transfer accuracy. "
            "Low-cardinality share is a feature-type proxy, not a categorical schema. "
            "These are new diagnostic draws, not retained production training samples.")
