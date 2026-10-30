"""Readable CPT/retention panels: never pool backbones or learning rates."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from src.experiments import results
from src.experiments.design import MAIN_STEPS, MILESTONES, RATES, SEEDS
from src.utils import paths
from src.utils.config import expand_with_seeds, load
from src.visualize import campaign, style
from src.visualize import training_plots as training

METHODS = {"full_wd": "Full CPT + decay (reused)", "no_decay": "Full CPT, no decay",
           "l2sp": "L2-SP anchor", "diagscale": "DiagScale", "lora": "Q/V LoRA"}
BACKBONES = {"tabicl": "TabICLv2", "tabpfn3": "TabPFN-3"}


def checkpoint_description():
    return ", ".join(f"{step:,}" for step in MILESTONES)


def trajectory_caption(track, metric, *, experiment=2):
    domains = ("credit benchmark only; native-OOD CRPS has different target units across datasets"
               if track == "lgd" and metric == "crps" else "credit benchmark and native OOD")
    groups = "four mixtures at each fixed peak rate" if experiment == 2 else "four methods and the reused full-CPT baseline"
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
            ("Fixed comparison", ["50% credit / 50% generic", "Same tables within each backbone", f"{MAIN_STEPS:,} updates; 2 paired seeds", "8 full-CPT baselines reused", "32 additional runs", "Settings fixed before benchmarking"]),
            ("Four additional methods", ["Full CPT without weight decay", "L2-SP: anchor to released weights", "DiagScale: query scales only", "LoRA: rank-16 Q/V updates", "Compare with ordinary full CPT", f"Report credit + OOD at {len(MILESTONES)} steps"])]
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
    axes[0].set_ylabel("Synthetic query loss" if metric == "loss" else f"Development {style.metric_label(metric)}")
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
    nrows = 2 if experiment == 2 else 1
    fig, axes = plt.subplots(nrows, len(domains), squeeze=False, sharey="col",
                             figsize=style.grid_figsize(len(domains), nrows, panel_ratio=.78))
    labels = (0, .25, .5, 1) if experiment == 2 else tuple(METHODS)
    reference_present = False
    for i, row in enumerate(axes):
        for domain, ax in zip(domains, row):
            frame = scores[domain]
            if not frame.empty:
                frame = frame[frame.architecture.eq(backbone)]
                if experiment == 2:
                    frame = frame[frame.lr_index.eq(i)]
                for label in labels:
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
                      label=style.prior_mix_label(x) if experiment == 2 else METHODS[x]) for x in labels]
    if reference_present:
        handles.append(Line2D([], [], color=style.REFERENCE, linestyle="--", label="Released weights (step 0)"))
    style.legend_below(fig, handles,
                       ncol=2 if experiment == 3 else 3)
    return fig


def retention_effects(track, backbone, *, metric=None):
    from src.visualize.paper_plots import seed_differences

    metric = metric or ("roc_auc" if track == "pd" else "r2")
    tables, raw = {}, {}
    for domain in ("credit", "ood"):
        frame = results.benchmark_frames(track, 3, domain=domain, final_only=True)
        if not frame.empty:
            frame = frame[frame.architecture.eq(backbone)]
        tables[domain] = results.paired_retention(frame, metric)
        if not frame.empty:
            raw[domain] = frame.assign(info_run_name=[f"exp3_{m}__s{s}" for m, s in
                                                       zip(frame.treatment, frame.training_seed)])
    if any(f.empty for f in tables.values()):
        return campaign.availability("The retention comparison requires all four methods and the reused full-CPT "
                                     "baseline, with both registered seeds and complete credit/OOD folds.")
    style.apply()
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=style.figsize(style.WIDTH_FULL, .62))
    methods = list(METHODS)[1:]
    for (domain, frame), ax in zip(tables.items(), axes):
        for i, method in enumerate(methods):
            row = frame[frame.configuration.eq(f"exp3_{method}")].iloc[0]
            ax.plot([row.ci_low, row.ci_high], [i, i], color=style.METHOD_COLORS[method], linewidth=style.EFFECT_LINE)
            ax.scatter(row.mean_difference, i, color=style.METHOD_COLORS[method],
                       marker=style.METHOD_MARKERS[method], s=style.EFFECT_POINT, zorder=4)
            for j, (seed, value) in enumerate(seed_differences(raw[domain], f"exp3_{method}", "exp3_full_wd", metric)):
                ax.scatter(value, i + (j - .5) * .28, color=style.MUTED,
                           marker=style.SEED_MARKERS[seed], s=style.SEED_POINT, zorder=3)
        ax.axvline(0, color=style.MUTED, linestyle=":")
        ax.set(title="Credit benchmark" if domain == "credit" else "Native OOD",
               xlabel=f"Δ {style.metric_label(metric)} vs full CPT + decay")
        ax.set_yticks(range(4), [METHODS[m] for m in methods])
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

    lines = ["A · DEVELOPMENT COMPARISON",
             "All four mixtures and both fixed rates are reported; development scores select no settings.",
             campaign.continuation_recipe(track).to_string(index=False),
             f"Complete final credit table available: {paper.final_ready(track, 'exp2')}.",
             "\nB · CREDIT EFFECTS"]
    dev = paper.final_credit(track, "exp2", "development")
    if dev is not None and not dev.empty:
        from src.eval.comparisons import identities
        metric = "roc_auc" if track == "pd" else "r2"
        means = dev.assign(run=identities(dev)).groupby(["run", "dataset"])[metric].mean().groupby("run").mean()
        lines.insert(-1, "Development dataset means per training run:\n" + means.to_string())
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
        baseline = next(r for r in expand_with_seeds(load(paths.config_path("Exp2_PD")))
                        if r["architecture"] == backbone and r["seed"] == SEEDS[0]
                        and r["lr_index"] == 0 and r["prior"]["credit_fraction"] == .5)
        methods = [r for r in expand_with_seeds(load(paths.config_path("Exp3_PD")))
                   if r["architecture"] == backbone and r["seed"] == SEEDS[0]]
        for run in [dict(baseline, method="full_wd"), *methods]:
            t = run["train"]
            rows.append({"Backbone": BACKBONES[backbone], "Method": METHODS[run["method"]],
                         "Peak LR": f"{t['lr']:g}", "Weight decay": t["weight_decay"],
                         "L2-SP alpha": t.get("l2sp_alpha", 0.), "Updates": t["max_steps"]})
    return pd.DataFrame(rows)


def retention_summary():
    lines = ["A · REGISTERED DESIGN", f"Fixed 50% mixture; 32 new {MAIN_STEPS:,}-update runs plus 8 reused Exp2 controls.",
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
                effects = results.paired_retention(final, metric)
                name = f"{track.upper()} / {BACKBONES[backbone]} / {domain}"
                lines.append(name + ": " + ("final paired results not complete" if effects.empty
                                           else "\n" + effects.to_string(index=False)))
                trajectories.append(name + ": " + ("no complete two-seed checkpoint groups" if scores.empty
                                    else "\n" + scores.to_string(index=False)))
    lines += ["D · ADAPTATION TRAJECTORIES", f"{len(MILESTONES)} retained checkpoints; incomplete seed/fold groups are withheld.",
              *trajectories, "Settings and primary endpoint are fixed in advance; both credit and OOD outcomes are reported. Historical benchmark exposure is disclosed."]
    return "\n".join(lines)
