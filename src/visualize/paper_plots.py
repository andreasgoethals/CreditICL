"""Paper-facing final contrasts. Complete benchmark loaders remain the data gates.

No monitoring curve is substituted for a final credit or native OOD result.
Uncertainty is over paired datasets; seed marks describe training variability.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from src.eval import analysis
from src.eval.comparisons import (
    identities,
    paired_ablation_effects,
    paired_prior_effects,
    paired_reference_effects,
    require_balanced_cells,
)
from src.eval.metrics import METRIC_DIRECTIONS
from src.eval.protocol import PROTOCOL
from src.eval.results import load_results
from src.utils import paths
from src.utils.config import load
from src.visualize import campaign, style


def _role(frame, track, exp, role="holdout"):
    cfg = load(paths.config_path(f"{exp.capitalize()}_{track.upper()}"), allow_placeholders=True)
    if exp in {"exp2", "exp3"} and role != "development":
        # Fixed-design CPT reports the complete credit suite. Historical project
        # exposure is disclosed rather than claiming untouched confirmation.
        datasets = cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"]
        return frame[frame.dataset.isin(datasets)].copy()
    return frame[frame.dataset.isin(cfg["eval"][f"{'dev' if role == 'development' else 'holdout'}_datasets"])].copy()


def _subset(frame, subset):
    if subset == "all":
        return frame
    column = next((c for c in ("n_classes", "info_n_classes") if c in frame), None)
    if column is None:
        raise ValueError("Native OOD class counts are required for the binary/multiclass diagnostic")
    if subset not in ("binary", "multiclass"):
        raise ValueError(f"Unknown OOD diagnostic subset {subset!r}")
    counts = pd.to_numeric(frame[column], errors="raise")
    if counts.isna().any():
        raise ValueError("Native OOD class counts must be recorded for every benchmark row")
    mask = counts.eq(2) if subset == "binary" else counts.gt(2)
    return frame[mask].copy()


def final_credit(track, exp="exp1", role="holdout"):
    frame = load_results(track, exp=exp, protocol=PROTOCOL)
    return None if frame is None else _role(frame, track, exp, role)


def final_ready(track, exp="exp1") -> bool:
    return load_results(track, exp=exp, protocol=PROTOCOL) is not None


def _components(track):
    frame = analysis.load_ablation_results(track)
    if frame is not None and not frame.protocol.eq(PROTOCOL).all():
        return None
    return frame


def ood_ready(track, exp="exp1"):
    return analysis.load_ood_results(track, exp) is not None


def components_ready(track):
    return _components(track) is not None


def development_selection(track, exp="exp1", *, backbone=None):
    frame = final_credit(track, exp, "development")
    if frame is None:
        return campaign.availability("Complete current-protocol development results are required for this descriptive comparison.")
    frame = _backbone(frame, backbone)
    part = frame.assign(identity=identities(frame))
    part["configuration"] = part.identity.str.replace(r"__s\d+$", "", regex=True)
    metric = "roc_auc" if track == "pd" else "r2"
    cells = part.groupby(["configuration", "identity", "dataset"])[metric].mean()
    seeds = cells.groupby(level=[0, 1]).mean()
    order = seeds.groupby(level=0).mean().sort_values().index
    style.apply()
    fig, ax = plt.subplots(figsize=style.row_figsize(len(order), per_row=.33, base=1.3))
    for i, name in enumerate(order):
        values = seeds.loc[name].to_numpy()
        ax.plot([values.min(), values.max()], [i, i], color=_mass_colour(name), linewidth=style.EFFECT_LINE)
        ax.scatter(values, np.full(len(values), i), color=_mass_colour(name), s=style.SEED_POINT)
    ax.set_yticks(range(len(order)), [_compact_label(name, backbone) if name.startswith("exp") else
                                    style.reference_label(name, track) for name in order])
    ax.set(xlabel=f"Development {style.metric_label(metric)}", title="Complete final development results")
    return fig


def label(configuration):
    if not configuration.startswith("exp"):
        return style.reference_label(configuration)
    if "ablations" in configuration:
        key = configuration.split("__arm=", 1)[-1]
        return {"no_macro_target": "No macro term", "no_informative_missingness": "Uncoupled gaps",
                "no_shift": "No shift", "no_tail_warp": "No tail warp",
                "base_cardinality": "Generic cardinalities"}.get(key, key)
    match = re.search(r"credit_fraction=([0-9p.]+)", configuration)
    mix = style.prior_mix_label(float(match.group(1).replace("p", "."))) if match else configuration
    if configuration.startswith("exp2"):
        from src.experiments.design import RATES
        backbone = "tabpfn3" if "tabpfn3" in configuration else "tabicl"
        index = re.search(r"lr_index=(\d+)", configuration)
        rate = RATES[backbone][int(index.group(1))] if index else None
        return ("TabPFN-3" if backbone == "tabpfn3" else "TabICLv2") + " · " + mix + (f" · LR {rate:g}" if rate else "")
    return mix


def _backbone(frame, backbone):
    if backbone is None or frame is None:
        return frame
    ids = identities(frame)
    own = ids.str.startswith("exp") & ids.str.contains("tabpfn3").eq(backbone == "tabpfn3")
    ref = ids.eq("tabpfn3" if backbone == "tabpfn3" else "tabiclv2")
    return frame[own | ref].copy()


def _compact_label(name, backbone):
    text = label(name)
    return text.split(" · ", 1)[-1] if backbone is not None and name.startswith("exp2") else text


def _table(frame, metric, track, *, ablations=False, reference=False):
    if ablations:
        return paired_ablation_effects(frame, metric, track)
    return (paired_reference_effects if reference else paired_prior_effects)(frame, metric)


def _order(name):
    """Scientific dose/rate order, independent of array IDs or observed scores."""
    fraction = re.search(r"credit_fraction=([0-9p.]+)", name)
    rate = re.search(r"lr_index=(\d+)", name)
    return (int(name.startswith("exp")), int("tabpfn3" in name),
            int(rate.group(1)) if rate else 0,
            float(fraction.group(1).replace("p", ".")) if fraction else -1, name)


def score_table(track, exp="exp1", *, domain="credit", role="holdout", subset="all"):
    """Exact absolute scores without ranking or pooling architectures/rates.

    Folds are averaged within each dataset and training seed, then datasets
    equally, then seeds. Point-only references cannot supply CRPS or predictive
    endpoint masses. Missing expected scores are errors, not smaller means.
    """
    frame = (final_credit(track, exp, role) if domain == "credit"
             else analysis.load_ood_results(track, exp))
    if frame is None or frame.empty:
        return pd.DataFrame()
    if domain == "ood":
        frame = _subset(frame, subset)
    require_balanced_cells(frame)
    part = frame.assign(identity=identities(frame))
    part["configuration"] = part.identity.str.replace(r"__s\d+$", "", regex=True)
    metrics = (("roc_auc", "log_loss", "brier") if track == "pd" else
               ("r2", "crps", "boundary_mass_abs_err") if domain == "credit" else ("r2",))
    rows = []
    for name in sorted(part.configuration.unique(), key=_order):
        cells = part[part.configuration.eq(name)]
        row = {"Configuration": label(name), "Datasets": cells.dataset.nunique(),
               "Training seeds": cells.identity.nunique() if name.startswith("exp") else 0}
        for metric in metrics:
            values = cells.get(metric, pd.Series(dtype=float))
            if values.empty or values.isna().all():
                if name in {"linear", "catboost"} and metric in {"crps", "boundary_mass_abs_err"}:
                    row[style.metric_label(metric)] = "unavailable"
                    continue
                raise ValueError(f"Missing expected {metric} for {name}")
            if not np.isfinite(values).all():
                raise ValueError(f"Partial/non-finite {metric} for {name}")
            means = cells.groupby(["identity", "dataset"])[metric].mean().groupby("identity").mean()
            row[style.metric_label(metric)] = f"{means.mean():.4f}" + (
                f" [{means.min():.4f}, {means.max():.4f}]" if len(means) > 1 else "")
        rows.append(row)
    return pd.DataFrame(rows)


def contrast_table(track, exp="exp1", *, domain="credit", subset="all", reference=False,
                   metrics=None):
    """Compact supporting effects with the same pairing as the main figures."""
    frame = final_credit(track, exp) if domain == "credit" else analysis.load_ood_results(track, exp)
    if frame is None or frame.empty:
        return pd.DataFrame()
    if domain == "ood":
        frame = _subset(frame, subset)
    metrics = metrics or (("roc_auc", "log_loss", "brier") if track == "pd" else
                          ("r2", "crps", "boundary_mass_abs_err") if domain == "credit" else ("r2",))
    rows = []
    for metric in metrics:
        for row in _table(frame, metric, track, reference=reference).itertuples():
            rows.append({"Configuration": label(row.configuration), "Metric": style.metric_label(metric),
                         "Mean difference": row.mean_difference, "95% dataset interval":
                         f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]", "Datasets": row.datasets})
    return pd.DataFrame(rows)


def seed_differences(frame, configuration, control, metric):
    """One mean paired-dataset difference per training seed; folds averaged first."""
    part = frame.assign(_identity=identities(frame))
    part["_configuration"] = part._identity.str.replace(r"__s\d+$", "", regex=True)
    treatment = part[part._configuration.eq(configuration)].copy()
    baseline = part[part._configuration.eq(control)].copy()
    treatment["_training_seed"] = treatment._identity.str.extract(r"__s(\d+)$")[0].astype(int)
    output = []
    for seed, rows in treatment.groupby("_training_seed"):
        ref = baseline[baseline._identity.str.endswith(f"__s{seed}")] if control.startswith("exp") else baseline
        a = rows.groupby("dataset")[metric].mean()
        b = ref.groupby("dataset")[metric].mean()
        if set(a.index) != set(b.index):
            raise ValueError("Seed marks require the same paired datasets")
        output.append((int(seed), float((a-b).mean())))
    return output


def paired_scores(track, exp="exp1", *, domain="credit", ablations=False,
                  reference=False, subset="all", metrics=None, backbone=None):
    metrics = metrics or (("roc_auc", "log_loss") if track == "pd" else ("r2", "crps"))
    if ablations:
        frame = _components(track)
        frame = None if frame is None else _role(frame, track, exp)
    elif domain == "credit":
        frame = final_credit(track, exp)
    else:
        frame = analysis.load_ood_results(track, exp)
        frame = None if frame is None else _subset(frame, subset)
    if frame is None or frame.empty:
        return campaign.availability("Complete final benchmark results required for paired comparisons. "
                                     "Training monitoring and incomplete folds are not used here.")
    frame = _backbone(frame, backbone)
    tables = [_table(frame, m, track, ablations=ablations, reference=reference) for m in metrics]
    if any(t.empty for t in tables):
        return campaign.availability("No complete matched treatment and comparator cells are available.")
    order = sorted(tables[0].configuration, key=_order)
    tables = [t.set_index("configuration").loc[order].reset_index() for t in tables]
    style.apply()
    fig, axes = plt.subplots(1, len(metrics), sharey=True,
                             figsize=style.row_figsize(len(tables[0]), per_row=.37, base=1.65), squeeze=False)
    seed_ids = set()
    for ax, metric, table in zip(axes[0], metrics, tables):
        for i, row in enumerate(table.itertuples()):
            color = style.CREDIT if ablations else _mass_colour(row.configuration)
            if np.isfinite([row.ci_low, row.ci_high]).all():
                ax.plot([row.ci_low, row.ci_high], [i, i], color=color, linewidth=style.EFFECT_LINE)
            ax.scatter([row.mean_difference], [i], color=color, s=style.EFFECT_POINT, zorder=4)
            seeds = seed_differences(frame, row.configuration, row.control, metric)
            seed_ids.update(seed for seed, _ in seeds)
            offsets = np.linspace(-.23, .23, len(seeds)+1)[:len(seeds)]
            for offset, (seed, value) in zip(offsets, seeds):
                ax.scatter(value, i+offset, color=style.MUTED, marker=style.SEED_MARKERS[seed % 3],
                           s=style.SEED_POINT, alpha=.75, zorder=3)
        ax.axvline(0, color=style.REFERENCE, linestyle=":")
        ax.set_title(style.metric_label(metric), loc="left")
        direction = "higher" if METRIC_DIRECTIONS[metric] else "lower"
        ax.set_xlabel(f"Treatment − comparator\n{direction} favors treatment")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_yticks(range(len(table)), [_compact_label(r.configuration, backbone) for r in table.itertuples()])
        ax.tick_params(axis="y", labelsize=style.PANEL_TEXT)
    axes[0, 0].invert_yaxis()
    counts = [int(v) for v in tables[0].datasets.unique()]
    comparator = "full 50% package" if ablations else "released weights" if reference else "equally trained 0% control"
    fig.suptitle(("TabPFN-3 · " if backbone == "tabpfn3" else "TabICLv2 · " if backbone else "") + f"Versus {comparator} · n={','.join(map(str, counts))} datasets",
                 fontsize=style.PANEL_TEXT)
    handles = [Line2D([], [], color=style.INK, marker="o", label="mean / dataset interval")]
    handles += [Line2D([], [], color=style.MUTED, linestyle="none", marker=style.SEED_MARKERS[s % 3],
                       label=f"seed {s}") for s in sorted(seed_ids)]
    style.legend_below(fig, handles, ncol=min(4, len(handles)))
    return fig


def retention_frontier(track, exp="exp1", *, reference=False, subset="all"):
    credit = final_credit(track, exp)
    ood = analysis.load_ood_results(track, exp)
    if credit is None or ood is None:
        return campaign.availability("The frontier requires complete final credit AND native OOD benchmarks. "
                                     "The two domains contain different datasets; training monitors do not populate this figure.")
    ood = _subset(ood, subset)
    metric = "roc_auc" if track == "pd" else "r2"
    x = _table(credit, metric, track, reference=reference)
    y = _table(ood, metric, track, reference=reference)
    joined = x.merge(y, on=["configuration", "control"], suffixes=("_credit", "_ood"), validate="one_to_one")
    if joined.empty:
        return campaign.availability("No matched mixture/backbone occurs in both final domains.")
    style.apply()
    groups = ("tabicl", "tabpfn3") if exp == "exp2" else ("all",)
    fractions = set()
    fig, axes = plt.subplots(1, len(groups), squeeze=False,
                             figsize=style.figsize(style.WIDTH_FULL, .64), sharex=True, sharey=True)
    for group, ax in zip(groups, axes[0]):
        rows = joined if group == "all" else joined[joined.configuration.str.contains("tabpfn3").eq(group == "tabpfn3")]
        for row in rows.itertuples():
            match = re.search(r"credit_fraction=([0-9p.]+)", row.configuration)
            fraction = float(match.group(1).replace("p", "."))
            fractions.add(fraction)
            color = style.credit_fraction_colour(fraction)
            ax.errorbar(row.mean_difference_credit, row.mean_difference_ood,
                    xerr=[[max(0, row.mean_difference_credit-row.ci_low_credit)],
                          [max(0, row.ci_high_credit-row.mean_difference_credit)]],
                    yerr=[[max(0, row.mean_difference_ood-row.ci_low_ood)],
                          [max(0, row.ci_high_ood-row.mean_difference_ood)]],
                        fmt=style.RATE_MARKERS[int(re.search(r"lr_index=(\d+)", row.configuration).group(1))] if "lr_index=" in row.configuration else style.credit_fraction_marker(fraction),
                        color=color, markersize=style.FRONTIER_MARKER,
                        label=_compact_label(row.configuration, group) if exp == "exp2" else style.prior_mix_label(fraction))
            xs = dict(seed_differences(credit, row.configuration, row.control, metric))
            ys = dict(seed_differences(ood, row.configuration, row.control, metric))
            for seed in sorted(set(xs) & set(ys)):
                # Mean markers encode the rate; small seed points cannot reuse
                # those shapes with a conflicting meaning in the same panel.
                ax.scatter(xs[seed], ys[seed], color=color,
                           marker="." if exp == "exp2" else style.SEED_MARKERS[seed % 3],
                           s=style.SEED_POINT, alpha=.6)
        ax.axhline(0, color=style.MUTED, linestyle=":")
        ax.axvline(0, color=style.MUTED, linestyle=":")
        title = f"Credit versus generality · {subset} OOD" if group == "all" else ("TabPFN-3" if group == "tabpfn3" else "TabICLv2")
        title += (f"\nn={int(joined.datasets_credit.iloc[0])} credit; "
                  f"n={int(joined.datasets_ood.iloc[0])} OOD")
        if exp == "exp2":
            from src.experiments.design import RATES
            high, low = RATES[group]
            title += f"\nPeak LRs: {high:g} / {low:g}"
        ax.set(xlabel=f"Credit Δ {style.metric_label(metric)}\nHigher is better on both axes", title=title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0, 0].set_ylabel(f"Native OOD Δ {style.metric_label(metric)}")
    fig.suptitle("Versus released weights" if reference else "Versus equally trained 0% control",
                 fontsize=style.PANEL_TEXT)
    if exp == "exp2":
        handles = [Line2D([], [], color=style.credit_fraction_colour(f),
                           label=style.prior_mix_label(f)) for f in sorted(fractions)]
        handles += [Line2D([], [], color=style.INK, linestyle="none", marker=m, label=label)
                    for m, label in zip(style.RATE_MARKERS, ("Higher peak LR", "Lower peak LR"))]
        style.legend_below(fig, handles, ncol=3)
    else:
        style.legend_below(fig, ncol=2)
    return fig


def distributional_scorecard(exp="exp1"):
    return paired_scores("lgd", exp, metrics=("r2", "crps", "boundary_mass_abs_err"))


def dataset_effects(track, exp="exp1", *, backbone=None):
    """Every designated-holdout dataset, same matched control as the headline."""
    frame = final_credit(track, exp)
    if frame is None:
        return campaign.availability("Complete final benchmark cells are required for per-dataset effects.")
    metric = "roc_auc" if track == "pd" else "r2"
    frame = _backbone(frame, backbone)
    contrasts = _table(frame, metric, track)
    contrasts = contrasts.set_index("configuration").loc[sorted(contrasts.configuration, key=_order)].reset_index()
    part = frame.assign(identity=identities(frame))
    part["configuration"] = part.identity.str.replace(r"__s\d+$", "", regex=True)
    scores = part.groupby(["configuration", "identity", "dataset"])[metric].mean()
    scores = scores.groupby(level=[0, 2]).mean().unstack("dataset")
    values = np.column_stack([(scores.loc[r.configuration]-scores.loc[r.control]).to_numpy()
                              for r in contrasts.itertuples()])
    style.apply()
    fig, ax = plt.subplots(figsize=style.row_figsize(len(scores.columns), per_row=.4, base=1.6))
    limit = max(float(np.abs(values).max()), 1e-6)
    image = ax.imshow(values, cmap=style.CMAP_DIV, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(scores.columns)), [s.split(".", 1)[-1] for s in scores.columns])
    ax.set_xticks(range(len(contrasts)), [_compact_label(c, backbone).replace(" · ", "\n") for c in contrasts.configuration])
    ax.tick_params(axis="x", labelsize=style.PANEL_TEXT)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:+.4f}" if exp == "exp2" else f"{value:+.3f}", ha="center", va="center", fontsize=style.PANEL_TEXT,
                color=style.LIGHT_INK if abs(value) > .6*limit else style.INK)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label=f"Δ {style.metric_label(metric)} vs matched 0% control")
    return fig


def endpoint_mass(exp="exp1", *, backbone=None):
    frame = final_credit("lgd", exp)
    fields = ["true_mass_at_0", "true_mass_at_1", "pred_mass_at_0", "pred_mass_at_1"]
    if frame is None or not set(fields).issubset(frame):
        return campaign.availability("Complete final LGD endpoint-mass metrics are required. "
                                     "Per-row quantiles and conditional calibration are not inferred from aggregate rows.")
    part = frame.assign(identity=identities(frame))
    if backbone is not None:
        matching = part.identity.str.startswith("exp") & part.identity.str.contains("tabpfn3").eq(backbone == "tabpfn3")
        released = part.identity.eq("tabpfn3" if backbone == "tabpfn3" else "tabiclv2")
        part = part[matching | released]
    part = part[part[fields].notna().all(axis=1)].copy()
    if part.empty:
        return campaign.availability("No method with predictive endpoint masses has been benchmarked completely.")
    part["configuration"] = part.identity.str.replace(r"__s\d+$", "", regex=True)
    means = part.groupby(["configuration", "identity", "dataset"])[fields].mean().reset_index()
    # Paired axes keep the same probability range, focused on observed values.
    # This is a scatter diagnostic, not a bar whose length needs the full unit axis.
    limit = min(1., max(.1, float(means[fields].to_numpy().max()) * 1.15))
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, .58), sharex=True, sharey=True)
    configurations = sorted(means.configuration.unique())
    for endpoint, ax in enumerate(axes):
        for name in configurations:
            rows = means[means.configuration.eq(name)]
            for identity, values in rows.groupby("identity"):
                found = re.search(r"credit_fraction=([0-9p.]+)", name)
                marker = (style.credit_fraction_marker(float(found.group(1).replace("p", ".")))
                          if found else ("D" if "tabicl" in name else "P"))
                ax.scatter(values[f"true_mass_at_{endpoint}"], values[f"pred_mass_at_{endpoint}"],
                           color=_mass_colour(name), marker=marker,
                           s=style.SEED_POINT, alpha=.75, label=label(name) if identity == rows.identity.iloc[0] else None)
        ax.plot([0, limit], [0, limit], color=style.MUTED, linestyle=":")
        ax.set(xlim=(-.03*limit, limit), ylim=(-.03*limit, limit), xlabel="Observed portfolio mass",
               title="Full recovery (0)" if endpoint == 0 else "Total loss (1)")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        named = means.groupby("dataset")[[f"true_mass_at_{endpoint}", f"pred_mass_at_{endpoint}"]].mean()
        style.place_labels(ax, named.iloc[:, 0], named.iloc[:, 1],
                           [s.split(".", 1)[-1].removeprefix("lgd_") for s in named.index],
                           fontsize=style.PANEL_TEXT)
    axes[0].set_ylabel("Average predicted mass")
    style.legend_below(fig, ncol=3)
    return fig


def _mass_colour(name):
    match = re.search(r"credit_fraction=([0-9p.]+)", name)
    if match:
        return style.credit_fraction_colour(float(match.group(1).replace("p", ".")))
    return style.REFERENCE if "tabicl" in name else style.RELEASED_TABPFN


def final_summary(track, exp="exp1"):
    ready = final_ready(track, exp)
    return (f"FINAL PAIRED EVIDENCE — {track.upper()}, {exp}\n"
            f"Complete credit results: {ready}. Comparators remain separate: matched 0% control; "
            "architecture-matched released weights.\n"
            "OOD binary/multiclass subsets are diagnostics, not replacement selection endpoints.\n"
            "Intervals describe paired datasets after averaging folds/seeds; seed marks are not independent datasets.\n"
            "LGD: aggregate endpoint matching is not conditional calibration; tolerance 1e-4, 99-level quantile approximation.")


def results_summary(track, exp="exp1"):
    """Numbers and availability in the results notebook's section order."""
    metrics = ("roc_auc", "log_loss") if track == "pd" else ("r2", "crps", "boundary_mass_abs_err")
    dev = final_credit(track, exp, "development")
    lines = ["A · DEVELOPMENT AND COMPARISON CONTRACT",
             ("Both backbones and all four mixtures (0%, 25%, 50%, 100%) are fixed; no Experiment 1 winner is selected for Experiment 2."
              if exp == "exp2" else "Exp2 retains all four mixtures; Exp3 uses the fixed 50% mixture, independent of this ranking."),
             ("Both predetermined learning rates are reported; no pilot or best-on-test selection. Historical benchmark exposure remains disclosed."
              if exp in {"exp2", "exp3"} else
              "The designated holdout has historical project exposure; it is not untouched confirmation."),
             f"Development final table available: {dev is not None}.",
             "\nB · " + ("CREDIT BENCHMARK EFFECTS" if exp in {"exp2", "exp3"}
                            else "DESIGNATED-HOLDOUT PACKAGE EFFECTS")]
    credit = final_credit(track, exp)
    for metric in metrics:
        table = pd.DataFrame() if credit is None else _table(credit, metric, track)
        lines.append(metric + ": " + ("not yet completely benchmarked" if table.empty
                                      else "\n" + table.to_string(index=False)))
    if exp == "exp2":
        lines.append("B2 · RELEASED WEIGHTS — separate comparator, separately within each backbone")
        for metric in metrics:
            table = pd.DataFrame() if credit is None else _table(credit, metric, track, reference=True)
            lines.append(metric + ": " + ("not yet completely benchmarked" if table.empty
                                          else "\n" + table.to_string(index=False)))
    lines += ["\nC · FINAL NATIVE OOD AND RETENTION FRONTIER", analysis.summary(track, exp),
              ("Binary/multiclass OOD subsets are diagnostics, not replacement selection endpoints."
               if track == "pd" else
               "Native regression OOD retention is descriptive and does not select the prior mixture.")]
    if exp == "exp1":
        lines.append("\nD · CONDITIONAL COMPONENT REMOVALS")
        components = _components(track)
        components = None if components is None else _role(components, track, exp)
        for metric in metrics:
            table = pd.DataFrame() if components is None else _table(components, metric, track, ablations=True)
            lines.append(metric + ": " + ("not yet completely benchmarked" if table.empty
                                          else "\n" + table.to_string(index=False)))
    if track == "lgd":
        lines += [f"\n{'D' if exp == 'exp2' else 'E'} · DISTRIBUTIONAL ENDPOINTS",
                  "Endpoint matching is aggregate, not conditional calibration; tolerance 1e-4; 99 quantile levels."]
    lines.append("Dataset bootstrap intervals are descriptive; training seeds/folds are not extra datasets.")
    return "\n".join(lines)
