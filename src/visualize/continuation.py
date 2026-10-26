"""Readable CPT/retention panels: never pool backbones or learning rates."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.experiments.design import MILESTONES, RATES
from src.experiments import results
from src.utils import paths
from src.utils.config import expand_with_seeds, load
from src.visualize import style, campaign, training_plots as training

METHODS = {"full_wd": "Full CPT + decay (reused)", "no_decay": "Full CPT, no decay",
           "l2sp": "L2-SP anchor", "diagscale": "DiagScale", "lora": "Q/V LoRA"}
BACKBONES = {"tabicl": "TabICLv2", "tabpfn3": "TabPFN-3"}


def design_figure(experiment=2):
    """A registered design, explicitly separated from measured outcomes."""
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, .46))
    if experiment == 2:
        panels = [
            ("What varies?", ["2 released backbones", "PD and LGD", "0%, 25%, 50%, 100% credit", "2 peak rates per backbone", "3 paired training seeds", "96 training runs"]),
            ("What is measured?", ["10,000 updates per run", "8 retained checkpoints", "125 → 250 → 500 → 1,000", "2,500 → 5,000 → 7,500 → 10,000", "Full-table credit + native OOD", "Same five folds for every model"])]
    else:
        panels = [
            ("Fixed comparison", ["50% credit / 50% generic", "Same tables within each backbone", "10,000 updates; 3 seeds", "12 full-CPT baselines reused", "48 additional main runs", "36 development-only pilots"]),
            ("Four additional methods", ["Full CPT without weight decay", "L2-SP: anchor to released weights", "DiagScale: query scales only", "LoRA: rank-16 Q/V updates", "Select settings on development", "Report credit + OOD at 8 steps"])]
    for ax, (title, lines) in zip(axes, panels):
        ax.axis("off")
        ax.set_title(title)
        for i, line in enumerate(lines):
            ax.text(0, .91-i*.145, line, transform=ax.transAxes, fontsize=style.PANEL_TEXT,
                    color=style.INK, va="top")
    fig.suptitle(f"Experiment {experiment} · registered design, not results")
    return fig


def training_curves(track, backbone, *, metric="loss"):
    """Two rate panels with paired seeds; losses never mix architectures."""
    cfg = load(paths.config_path(f"Exp2_{track.upper()}"))
    configs = {r["_run_name"]: r for r in expand_with_seeds(cfg) if r["architecture"] == backbone}
    runs = training.load_loss_logs(track, "exp2") if metric == "loss" else training.load_progress(track, "exp2")
    if not runs:
        return campaign.availability("No training traces are available for this continuation recipe.")
    runs = {n: d for n, d in runs.items() if n in configs}
    datasets = training.comparable_datasets(runs, metric)[0] if metric != "loss" else []
    style.apply()
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=style.figsize(style.WIDTH_FULL, .6))
    for lr_index, ax in enumerate(axes):
        for fraction in (0, .25, .5, 1):
            curves = []
            color = style.credit_fraction_colour(fraction)
            for name, frame in runs.items():
                r = configs[name]
                if r["lr_index"] != lr_index or r["prior"]["credit_fraction"] != fraction:
                    continue
                value = frame.train_loss if metric == "loss" else training._mean_metric(frame, metric, datasets=datasets)
                if len(value) != len(frame):
                    continue
                series = pd.Series(value.to_numpy(), index=frame.step).dropna()
                if series.empty:
                    continue
                ax.plot(series.index, series, color=color, alpha=.25, linewidth=style.TRACE_LINE)
                curves.append(series.rename(r["seed"]))
            if len(curves) == 3:
                common = pd.concat(curves, axis=1, join="inner").dropna()
                ax.plot(common.index, common.mean(axis=1), color=color, linewidth=style.EFFECT_LINE)
        ax.set(xlabel="Optimizer updates", title=f"Peak LR {RATES[backbone][lr_index]:g}")
    axes[0].set_ylabel("Synthetic query loss" if metric == "loss" else f"Development {style.metric_label(metric)}")
    fig.suptitle(BACKBONES[backbone] + (" · optimization" if metric == "loss" else " · small-context monitoring"))
    style.legend_below(fig, [Line2D([], [], color=style.credit_fraction_colour(f), label=style.prior_mix_label(f))
                             for f in (0, .25, .5, 1)], ncol=4)
    return fig


def checkpoint_curves(track, backbone, *, experiment=2, metric=None):
    """Saved-checkpoint benchmarks; explicit steps and complete seed triplets."""
    metric = metric or ("roc_auc" if track == "pd" else "r2")
    frames = {d: results.benchmark_frames(track, experiment, domain=d) for d in ("credit", "ood")}
    scores = {d: results.complete_scores(f, metric, experiment=experiment) for d, f in frames.items()}
    if all(f.empty for f in scores.values()):
        return campaign.availability("Checkpoint curves require complete five-fold tables for all three training seeds. "
                                     "Monitoring values do not populate these panels.")
    style.apply()
    nrows = 2 if experiment == 2 else 1
    fig, axes = plt.subplots(nrows, 2, squeeze=False,
                             figsize=style.grid_figsize(2, nrows, panel_ratio=1.0))
    labels = (0, .25, .5, 1) if experiment == 2 else tuple(METHODS)
    reference_present = False
    for i, row in enumerate(axes):
        for domain, ax in zip(("credit", "ood"), row):
            frame = scores[domain]
            if not frame.empty:
                frame = frame[frame.architecture.eq(backbone)]
                if experiment == 2:
                    frame = frame[frame.lr_index.eq(i)]
                for j, label in enumerate(labels):
                    part = frame[frame["mixture" if experiment == 2 else "treatment"].eq(label)].sort_values("checkpoint_step")
                    color = style.credit_fraction_colour(label) if experiment == 2 else style.METHOD_COLORS[label]
                    ax.plot(part.checkpoint_step, part["mean"], color=color,
                            marker=style.MILESTONE_MARKER, markersize=style.CURVE_MARKER)
                    ax.fill_between(part.checkpoint_step, part.minimum, part.maximum, color=color, alpha=.12)
            title = "Designated credit holdout" if domain == "credit" else "Native OOD"
            if experiment == 2:
                title += f" · LR {RATES[backbone][i]:g}"
            ax.set(title=title, xlabel="Optimizer updates", xscale="log")
            ax.set_xticks([125, 500, 2500, 10000], ["125", "500", "2,500", "10,000"])
            reference = results.released_score(track, backbone, domain, metric)
            if reference is not None:
                ax.axhline(reference, color=style.REFERENCE, linestyle="--")
                reference_present = True
        row[0].set_ylabel(style.metric_label(metric))
    fig.suptitle(f"{BACKBONES[backbone]} · full-table five-fold {style.metric_label(metric)}")
    handles = [Line2D([], [], color=style.credit_fraction_colour(x) if experiment == 2 else style.METHOD_COLORS[x],
                      label=style.prior_mix_label(x) if experiment == 2 else METHODS[x]) for x in labels]
    if reference_present:
        handles.append(Line2D([], [], color=style.REFERENCE, linestyle="--", label="Released weights (step 0)"))
    style.legend_below(fig, handles,
                       ncol=2 if experiment == 3 else 4)
    return fig


def retention_effects(track, backbone, *, metric=None):
    metric = metric or ("roc_auc" if track == "pd" else "r2")
    tables = {}
    for domain in ("credit", "ood"):
        frame = results.benchmark_frames(track, 3, domain=domain, final_only=True)
        if not frame.empty:
            frame = frame[frame.architecture.eq(backbone)]
        tables[domain] = results.paired_retention(frame, metric)
    if any(f.empty for f in tables.values()):
        return campaign.availability("The retention comparison requires all four methods and the reused full-CPT "
                                     "baseline, with three seeds and complete credit/OOD folds.")
    style.apply()
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=style.figsize(style.WIDTH_FULL, .62))
    methods = list(METHODS)[1:]
    for (domain, frame), ax in zip(tables.items(), axes):
        for i, method in enumerate(methods):
            row = frame[frame.configuration.eq(f"exp3_{method}")].iloc[0]
            ax.plot([row.ci_low, row.ci_high], [i, i], color=style.METHOD_COLORS[method], linewidth=style.EFFECT_LINE)
            ax.scatter(row.mean_difference, i, color=style.METHOD_COLORS[method], s=style.EFFECT_POINT)
        ax.axvline(0, color=style.MUTED, linestyle=":")
        ax.set(title="Credit holdout" if domain == "credit" else "Native OOD",
               xlabel=f"Δ {style.metric_label(metric)} vs full CPT + decay")
        ax.set_yticks(range(4), [METHODS[m] for m in methods])
    fig.suptitle(f"{BACKBONES[backbone]} · retention methods at 10,000 updates")
    return fig


def pilot_scores(track, backbone):
    """All pilot candidates, never a winner-only plot."""
    from src.experiments.selection import load_selection
    receipt = load_selection("retention", optional=True, for_analysis=True)
    if receipt is None:
        return campaign.availability("The development-only pilot comparison has not been frozen.")
    df = pd.DataFrame(receipt["candidates"])
    df = df[df.track.eq(track) & df.model.eq(backbone)]
    style.apply()
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=style.figsize(style.WIDTH_FULL, .55))
    for ax, method in zip(axes, ("l2sp", "diagscale", "lora")):
        part = df[df.method.eq(method)].sort_values("value")
        ax.plot(part.value, part.score, color=style.METHOD_COLORS[method], marker=style.MILESTONE_MARKER)
        ax.set(title=METHODS[method], xlabel="Anchor strength α" if method == "l2sp" else "Peak adapter LR", xscale="log")
        ax.set_xticks(part.value, [f"{v:g}" for v in part.value])
    axes[0].set_ylabel(f"Development {style.metric_label('roc_auc' if track == 'pd' else 'r2')}")
    fig.suptitle(f"{BACKBONES[backbone]} · 2,500-update pilots, seed 991")
    return fig


def retention_summary():
    from src.experiments.selection import load_selection
    lines = ["A · REGISTERED DESIGN", "Fixed 50% mixture; 48 new 10,000-update runs plus 12 reused Exp2 controls.",
             "B · DEVELOPMENT SELECTION", "36 pilots use seed 991 and only development datasets; 2,500 updates on a 10,000-update schedule."]
    for kind in ("retention_pilot", "retention"):
        saved = load_selection(kind, optional=True, for_analysis=True)
        lines.append(f"{kind}: " + (str(saved["choices"]) if saved else "not frozen"))
    lines += ["C · CREDIT AND OOD RETENTION", "Paired final contrasts use the same training seeds and five-fold assignments."]
    trajectories = []
    for track in ("pd", "lgd"):
        for domain in ("credit", "ood"):
            frame = results.benchmark_frames(track, 3, domain=domain)
            for backbone in BACKBONES:
                part = frame[frame.architecture.eq(backbone)] if not frame.empty else frame
                metric = "roc_auc" if track == "pd" else "r2"
                scores = results.complete_scores(part, metric, experiment=3)
                final = part[part.checkpoint_step.eq(10000)] if not part.empty else part
                effects = results.paired_retention(final, metric)
                name = f"{track.upper()} / {BACKBONES[backbone]} / {domain}"
                lines.append(name + ": " + ("final paired results not complete" if effects.empty
                                           else "\n" + effects.to_string(index=False)))
                trajectories.append(name + ": " + ("no complete three-seed checkpoint groups" if scores.empty
                                    else "\n" + scores.to_string(index=False)))
    lines += ["D · ADAPTATION TRAJECTORIES", "Eight retained checkpoints; incomplete seed/fold groups are withheld.",
              *trajectories, "Holdout/OOD do not select hyperparameters. Historical holdout exposure remains disclosed."]
    return "\n".join(lines)
