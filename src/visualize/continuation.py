"""Readable CPT/retention panels: never pool backbones or learning rates."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.experiments import results
from src.experiments.design import (
    MAIN_STEPS,
    MILESTONES,
    RATES,
    RETENTION_RECIPES,
    SEEDS,
    retention_baseline,
)
from src.utils import paths
from src.utils.config import expand_with_seeds, load
from src.visualize import campaign, style
from src.visualize import training_plots as training

METHODS = {"full_wd": "Full CPT · primary LR", "full_wd_lower": "Full CPT · lower LR",
           "no_decay": "Full CPT · no decay", "l2sp": "L2-SP · alpha 0.003",
           "diagscale_low": "DiagScale · LR 1e-4", "diagscale_high": "DiagScale · LR 1e-3",
           "lora_low": "Q/V LoRA · LR 3e-6", "lora_high": "Q/V LoRA · LR 1e-5"}
RETENTION_PANELS = (("full_wd", "full_wd_lower", "no_decay", "l2sp"),
                    ("full_wd", "full_wd_lower", "diagscale_low", "diagscale_high"),
                    ("full_wd", "full_wd_lower", "lora_low", "lora_high"))
BACKBONES = {"tabicl": "TabICLv2", "tabpfn3": "TabPFN-3"}


def checkpoint_description():
    return ", ".join(f"{step:,}" for step in MILESTONES)


def trajectory_caption(track, metric, *, experiment=2):
    domains = ("credit benchmark only; native-OOD CRPS has different target units across datasets"
               if track == "lgd" and metric == "crps" else "credit benchmark and native OOD")
    groups = "four mixtures at each fixed peak rate" if experiment == 2 else "six recipes and both reused full-CPT rates, separated into full-update, DiagScale and LoRA panels"
    return (f"Full-table five-fold {style.metric_label(metric)} against optimizer updates {checkpoint_description()} "
            f"for {groups}, on the {domains}. Points average the {len(SEEDS)} registered training seeds; "
            "shading spans their dataset-mean scores, not a confidence interval. Missing checkpoints "
            "are not connected; the final update is the primary endpoint.")


def design_figure(experiment=2):
    """A registered design, explicitly separated from measured outcomes."""
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, .46))
    if experiment == 2:
        panels = [
            ("What varies?", ["2 released backbones", "PD and LGD", "0%, 25%, 50%, 100% credit", "2 predetermined learning rates", "2 paired training seeds", "64 runs; both rates reported"]),
            ("What is measured?", [f"{MAIN_STEPS:,} updates per run", f"{len(MILESTONES)} retained checkpoints", " → ".join(f"{s:,}" for s in MILESTONES), "Final step is the primary endpoint", "Full-table credit + native OOD", "Same five folds for every model"])]
    else:
        panels = [
            ("Fixed comparison", ["50% credit / 50% generic", "Same tables within each backbone", f"{MAIN_STEPS:,} updates; 2 paired seeds", "16 full-CPT baselines reused", "48 additional runs", "Both baseline rates reported"]),
            ("Six additional recipes", ["Full CPT without weight decay", "L2-SP: anchor to released weights", "DiagScale: LR 1e-4 and 1e-3", "Q/V LoRA: LR 3e-6 and 1e-5", "Compare with both full-CPT rates", f"Report credit + OOD at {len(MILESTONES)} steps"])]
    for ax, (title, lines) in zip(axes, panels):
        ax.axis("off")
        ax.set_title(title)
        for i, line in enumerate(lines):
            ax.text(0, .91-i*.145, line, transform=ax.transAxes, fontsize=style.PANEL_TEXT,
                    color=style.INK, va="top")
    fig.suptitle(f"Experiment {experiment} · registered design, not results")
    return fig


def training_curves(track, backbone, *, metric="loss"):
    """Separate predetermined rates; losses never mix architectures."""
    cfg = load(paths.config_path(f"Exp2_{track.upper()}"), allow_placeholders=True)
    configs = {r["_run_name"]: r for r in expand_with_seeds(cfg) if r["architecture"] == backbone}
    runs = training.load_loss_logs(track, "exp2") if metric == "loss" else training.load_progress(track, "exp2")
    if not runs:
        return campaign.availability("No training traces are available for this continuation recipe.")
    runs = {n: d for n, d in runs.items() if n in configs}
    if not runs:
        return campaign.availability("No training traces match the current fixed-rate design.")
    datasets = training.comparable_datasets(runs, metric)[0] if metric != "loss" else []
    log_loss_axis = False
    if metric == "loss":
        values = np.concatenate([f.train_loss.to_numpy() for f in runs.values()])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite recorded loss cannot be silently removed from the training figure")
        magnitudes = np.abs(values[values != 0])
        log_loss_axis = bool(len(magnitudes) and magnitudes.max() / magnitudes.min() > 1e4)
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
                ax.plot(series.index, series, color=color, alpha=.25, linewidth=style.TRACE_LINE,
                        linestyle=style.credit_fraction_line(fraction))
                curves.append(series.rename(r["seed"]))
            if len(curves) == len(SEEDS):
                common = pd.concat(curves, axis=1, join="inner").dropna()
                ax.plot(common.index, common.mean(axis=1), color=color, linewidth=style.EFFECT_LINE,
                        linestyle=style.credit_fraction_line(fraction))
        ax.set(xlabel="Optimizer updates", title=f"Fixed peak LR {RATES[backbone][lr_index]:g}")
        if log_loss_axis:
            ax.set_yscale("symlog", linthresh=1)
    if log_loss_axis and magnitudes.max() > 1e12:
        # The default symlog locator crowds -1/0/+1 into the same pixels when
        # finite tail losses span hundreds of decades. Keep zero and use a
        # sparse set of decades; retain every observation on the shared scale.
        exponent = int(np.ceil(np.log10(magnitudes.max())))
        spacing = max(1, int(np.ceil(exponent / 6)))
        powers = 10. ** np.arange(spacing, exponent + 1, spacing)
        ticks = (list(-powers[::-1]) if values.min() < 0 else []) + [0.]
        ticks += list(powers) if values.max() > 0 else []
        axes[0].set_yticks(ticks)
        if values.min() >= 0:
            axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(("Synthetic query loss" + ("\n(symlog; linear within ±1)" if log_loss_axis else ""))
                      if metric == "loss" else f"Development {style.metric_label(metric)}")
    fig.suptitle(BACKBONES[backbone] + (" · optimization" if metric == "loss" else " · small-context monitoring"))
    style.legend_below(fig, [Line2D([], [], color=style.credit_fraction_colour(f),
                             linestyle=style.credit_fraction_line(f), label=style.prior_mix_label(f))
                             for f in (0, .25, .5, 1)], ncol=4)
    return fig


def checkpoint_curves(track, backbone, *, experiment=2, metric=None):
    """Saved-checkpoint benchmarks; explicit steps and complete registered seed pairs."""
    metric = metric or ("roc_auc" if track == "pd" else "r2")
    # CRPS is in target units: averaging arbitrary native regression scales would
    # let the largest-scale dataset dominate. Credit LGD shares the [0,1] scale.
    domains = ("credit",) if track == "lgd" and metric == "crps" else ("credit", "ood")
    frames = {d: results.benchmark_frames(track, experiment, domain=d) for d in domains}
    scores = {d: results.complete_scores(f, metric, experiment=experiment) for d, f in frames.items()}
    scores = {d: f[f.architecture.eq(backbone)] if not f.empty else f for d, f in scores.items()}
    if all(f.empty for f in scores.values()):
        return campaign.availability("Checkpoint curves require complete five-fold tables for both registered training seeds. "
                                     "Monitoring values do not populate these panels.")
    style.apply()
    nrows = 2 if experiment == 2 else 3
    fig, axes = plt.subplots(nrows, len(domains), squeeze=False, sharey="col",
                             figsize=style.grid_figsize(len(domains), nrows, panel_ratio=.78))
    labels = (0, .25, .5, 1) if experiment == 2 else tuple(METHODS)
    reference_present = False
    for i, row in enumerate(axes):
        panel_labels = labels if experiment == 2 else RETENTION_PANELS[i]
        for domain, ax in zip(domains, row):
            frame = scores[domain]
            if not frame.empty:
                frame = frame[frame.architecture.eq(backbone)]
                if experiment == 2:
                    frame = frame[frame.lr_index.eq(i)]
                for label in panel_labels:
                    part = frame[frame["mixture" if experiment == 2 else "treatment"].eq(label)]
                    # Reindex before plotting so absent milestones break the
                    # curve rather than implying unmeasured intermediate scores.
                    part = part.set_index("checkpoint_step").reindex(MILESTONES)
                    color = style.credit_fraction_colour(label) if experiment == 2 else style.METHOD_COLORS[label]
                    ax.plot(part.index, part["mean"], color=color,
                            marker=style.credit_fraction_marker(label) if experiment == 2 else style.METHOD_MARKERS[label],
                            linestyle=style.credit_fraction_line(label) if experiment == 2 else style.METHOD_LINES[label],
                            markersize=style.CURVE_MARKER, label=str(label))
                    ax.fill_between(part.index, part.minimum, part.maximum, color=color, alpha=.12)
            if frame.empty:
                ax.text(.5, .5, "Complete seed pairs pending", ha="center", va="center",
                        transform=ax.transAxes, color=style.MUTED, fontsize=style.PANEL_TEXT)
            title = "Credit benchmark" if domain == "credit" else "Native OOD"
            if experiment == 2:
                title += f" · LR {RATES[backbone][i]:g}"
            if experiment == 3:
                title += " · " + ("Full updates", "DiagScale", "Q/V LoRA")[i]
            ax.set(title=title, xlabel="Optimizer updates", xscale="log")
            ax.set_xticks(MILESTONES, [f"{s:,}" for s in MILESTONES])
            ax.minorticks_off()
            reference = results.released_score(track, backbone, domain, metric) if not frame.empty else None
            if reference is not None:
                ax.axhline(reference, color=style.REFERENCE, linestyle="--")
                reference_present = True
        row[0].set_ylabel(style.metric_label(metric))
    fig.suptitle(f"{BACKBONES[backbone]} · full-table five-fold {style.metric_label(metric)}")
    handles = [Line2D([], [], color=style.credit_fraction_colour(x) if experiment == 2 else style.METHOD_COLORS[x],
                      marker=style.credit_fraction_marker(x) if experiment == 2 else style.METHOD_MARKERS[x],
                      linestyle=style.credit_fraction_line(x) if experiment == 2 else style.METHOD_LINES[x],
                      label=style.prior_mix_label(x) if experiment == 2 else (f"Full CPT · LR {RATES[backbone][int(x == 'full_wd_lower')]:g}"
                            if x in {"full_wd", "full_wd_lower"} else METHODS[x])) for x in labels]
    if reference_present:
        handles.append(Line2D([], [], color=style.REFERENCE, linestyle="--", label="Released weights (step 0)"))
    style.legend_below(fig, handles,
                       ncol=3)
    return fig


def trajectory_ready(track, backbone, *, experiment=2, metric=None):
    """A measured trajectory needs at least two checkpoints in a fixed group.

    One complete final endpoint is useful, but it cannot describe change with
    training time. Exp3's reused baselines alone are not retention-method results.
    """
    metric = metric or ("roc_auc" if track == "pd" else "r2")
    for domain in ("credit", "ood"):
        if track == "lgd" and domain == "ood" and metric == "crps":
            continue
        frame = results.complete_scores(results.benchmark_frames(track, experiment, domain=domain),
                                        metric, experiment=experiment)
        if frame.empty:
            continue
        frame = frame[frame.architecture.eq(backbone)]
        if experiment == 3:
            frame = frame[frame.treatment.isin(RETENTION_RECIPES)]
        groups = ["lr_index", "mixture"] if experiment == 2 else ["treatment"]
        if not frame.empty and frame.groupby(groups).checkpoint_step.nunique().ge(2).any():
            return True
    return False


def retention_ready(track, backbone, *, metric=None):
    """A final retention panel needs both domains, not just a finished credit suite."""
    metric = metric or ("roc_auc" if track == "pd" else "r2")
    for domain in ("credit", "ood"):
        frame = results.benchmark_frames(track, 3, domain=domain, final_only=True)
        if not frame.empty:
            frame = frame[frame.architecture.eq(backbone)]
        if results.paired_retention(frame, metric).empty:
            return False
    return True


def retention_effects(track, backbone, *, metric=None):
    from src.visualize.paper_plots import seed_differences

    metric = metric or ("roc_auc" if track == "pd" else "r2")
    raw = {}
    for domain in ("credit", "ood"):
        frame = results.benchmark_frames(track, 3, domain=domain, final_only=True)
        if not frame.empty:
            frame = frame[frame.architecture.eq(backbone)]
        if results.paired_retention(frame, metric).empty:
            return campaign.availability("Final comparisons need six recipes and both full-CPT rates, "
                                         "with both seeds and complete credit/OOD folds.")
        raw[domain] = frame.assign(info_run_name=[f"exp3_{m}__s{s}" for m, s in
                                                   zip(frame.treatment, frame.training_seed)])
    style.apply()
    fig, axes = plt.subplots(2, 2, sharey="row", sharex="col",
                             figsize=style.grid_figsize(2, 2, panel_ratio=.95))
    methods = list(RETENTION_RECIPES)
    for panel, ax in enumerate(axes.flat):
        domain = ("credit", "ood")[panel % 2]
        baseline = ("full_wd", "full_wd_lower")[panel // 2]
        pairs = results.retention_pairs(baseline)
        frame = results.paired_retention(raw[domain], metric, baseline=baseline)
        for i, method in enumerate(methods):
            row = frame[frame.configuration.eq(f"exp3_{method}")].iloc[0]
            ax.plot([row.ci_low, row.ci_high], [i, i], color=style.METHOD_COLORS[method], linewidth=style.EFFECT_LINE)
            ax.scatter(row.mean_difference, i, color=style.METHOD_COLORS[method],
                       marker=style.METHOD_MARKERS[method], s=style.EFFECT_POINT, zorder=4)
            for j, (seed, value) in enumerate(seed_differences(raw[domain], f"exp3_{method}", pairs[f"exp3_{method}"], metric)):
                ax.scatter(value, i + (j - .5) * .28, color=style.MUTED,
                           marker=style.SEED_MARKERS[seed], s=style.SEED_POINT, zorder=3)
        ax.axvline(0, color=style.MUTED, linestyle=":")
        ax.set(title="Credit benchmark" if domain == "credit" else "Native OOD",
               xlabel=f"Δ {style.metric_label(metric)} · " + ("primary contrasts" if panel < 2 else "vs lower-rate full CPT"))
        labels = [METHODS[m] for m in methods]
        if panel < 2:
            labels[1] = "L2-SP vs no decay"
        ax.set_yticks(range(len(methods)), labels)
    axes[0, 0].invert_yaxis()
    axes[1, 0].invert_yaxis()
    fig.suptitle(f"{BACKBONES[backbone]} · retention methods at {MAIN_STEPS:,} updates")
    style.legend_below(fig, [Line2D([], [], color=style.MUTED, linestyle="none",
        marker=style.SEED_MARKERS[s], label=f"paired seed {s}") for s in SEEDS], ncol=len(SEEDS))
    return fig


def checkpoint_summary(track, *, experiment=2):
    """Numerical trajectory summary, respecting native regression target units."""
    lines = [f"Retained updates: {checkpoint_description()}; final {MAIN_STEPS:,} is primary.",
             f"Each point requires all {len(SEEDS)} training seeds and complete five-fold dataset cells."]
    metrics = ("roc_auc", "log_loss", "brier") if track == "pd" else ("r2", "crps")
    for domain in ("credit", "ood"):
        frame = results.benchmark_frames(track, experiment, domain=domain)
        for metric in metrics:
            if track == "lgd" and domain == "ood" and metric == "crps":
                lines.append("Native-OOD CRPS remains per dataset in the benchmark files; different target units are not averaged.")
                continue
            table = results.complete_scores(frame, metric, experiment=experiment)
            lines.append(f"{domain} / {style.metric_label(metric)}: " +
                         ("no complete seed/fold groups" if table.empty else "\n" + table.to_string(index=False)))
    return "\n".join(lines)


def results_summary(track):
    """Exp2's final cell follows its A/B/C/D/E section order exactly."""
    from src.visualize import paper_plots as paper

    lines = ["A · COMPLETE SCORES AND COMPARISON CONTRACT",
             "All four mixtures and both fixed rates are reported; development scores select no settings.",
             campaign.continuation_recipe(track).to_string(index=False),
             f"Complete final credit table available: {paper.final_ready(track, 'exp2')}.",
             "\nB · CREDIT EFFECTS"]
    absolute = paper.score_table(track, "exp2")
    if not absolute.empty:
        lines.insert(-1, "Final credit scores (equal dataset weights):\n" + absolute.to_string(index=False))
    credit = paper.final_credit(track, "exp2")
    metrics = ("roc_auc", "log_loss", "brier") if track == "pd" else ("r2",)
    for reference in (False, True):
        lines.append("Comparator: released weights" if reference else "Comparator: matched 0% continuation")
        for metric in metrics:
            table = pd.DataFrame() if credit is None else paper._table(credit, metric, track, reference=reference)
            lines.append(style.metric_label(metric) + ": " + ("not completely benchmarked" if table.empty
                                                                else "\n" + table.to_string(index=False)))
    lines += ["\nC · CHECKPOINT TRAJECTORIES", checkpoint_summary(track),
              "\nD · NATIVE OOD", f"Complete final OOD table available: {paper.ood_ready(track, 'exp2')}."]
    from src.eval import analysis
    ood = analysis.load_ood_results(track, "exp2")
    for metric in (("roc_auc", "log_loss") if track == "pd" else ("r2",)):
        table = pd.DataFrame() if ood is None else paper._table(ood, metric, track)
        lines.append(style.metric_label(metric) + ": " + ("not completely benchmarked" if table.empty
                                                            else "\n" + table.to_string(index=False)))
    if track == "lgd":
        lines.append("\nE · DISTRIBUTIONAL ENDPOINTS")
        for metric in ("crps", "boundary_mass_abs_err"):
            table = pd.DataFrame() if credit is None else paper._table(credit, metric, track)
            lines.append(style.metric_label(metric) + ": " + ("not completely benchmarked" if table.empty
                                                                else "\n" + table.to_string(index=False)))
        lines.append("Credit LGD shares the [0,1] scale. Endpoint-mass agreement is aggregate, not conditional calibration.")
    lines += ["Dataset-bootstrap intervals are descriptive; seeds and folds are not extra datasets.",
              "Historical benchmark exposure remains disclosed; no rate, mixture or checkpoint is selected here."]
    return "\n".join(lines)


def retention_recipe():
    """Fixed settings are descriptive metadata, never selected from benchmark scores."""
    rows = []
    for backbone in BACKBONES:
        baselines = [r for r in expand_with_seeds(load(paths.config_path("Exp2_PD")))
                        if r["architecture"] == backbone and r["seed"] == SEEDS[0]
                        and retention_baseline(r)]
        methods = [r for r in expand_with_seeds(load(paths.config_path("Exp3_PD")))
                   if r["architecture"] == backbone and r["seed"] == SEEDS[0]]
        for run in [*[dict(b, method="full_wd" if b["lr_index"] == 0 else "full_wd_lower") for b in baselines], *methods]:
            t = run["train"]
            rows.append({"Backbone": BACKBONES[backbone], "Method": METHODS[run["method"]],
                         "Peak LR": f"{t['lr']:g}", "Weight decay": t["weight_decay"],
                         "L2-SP alpha": t.get("l2sp_alpha", 0.), "Updates": t["max_steps"]})
    return pd.DataFrame(rows)


def retention_summary():
    lines = ["A · REGISTERED DESIGN", f"Fixed 50% mixture; 48 new {MAIN_STEPS:,}-update runs plus 16 reused Exp2 controls.",
             "B · FIXED ADAPTATION RECIPES", retention_recipe().to_string(index=False),
             "No benchmark-driven learning-rate or adapter selection.",
             "C · CREDIT AND OOD RETENTION", "Paired final contrasts use the same training seeds and five-fold assignments."]
    trajectories = []
    for track in ("pd", "lgd"):
        for domain in ("credit", "ood"):
            frame = results.benchmark_frames(track, 3, domain=domain)
            for backbone in BACKBONES:
                part = frame[frame.architecture.eq(backbone)] if not frame.empty else frame
                metric = "roc_auc" if track == "pd" else "r2"
                scores = results.complete_scores(part, metric, experiment=3)
                final = part[part.checkpoint_step.eq(MAIN_STEPS)] if not part.empty else part
                effects = pd.concat([results.paired_retention(final, metric, baseline=b)
                                     for b in ("full_wd", "full_wd_lower")], ignore_index=True)
                name = f"{track.upper()} / {BACKBONES[backbone]} / {domain}"
                lines.append(name + ": " + ("final paired results not complete" if effects.empty
                                           else "\n" + effects.to_string(index=False)))
                trajectories.append(name + ": " + ("no complete two-seed checkpoint groups" if scores.empty
                                    else "\n" + scores.to_string(index=False)))
    lines += ["D · ADAPTATION TRAJECTORIES", f"{len(MILESTONES)} retained checkpoints; incomplete seed/fold groups are withheld.",
              *trajectories, "Settings and primary endpoint are fixed in advance; both credit and OOD outcomes are reported. Historical benchmark exposure is disclosed."]
    return "\n".join(lines)
