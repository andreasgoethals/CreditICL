"""Dated local evidence for notebooks; never interpret files as a live queue."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.utils import paths
from src.utils.config import expand_with_seeds, load
from src.visualize import style


COMPONENTS = {
    "no_macro_target": "Without macro target term",
    "no_informative_missingness": "Without covariate-coupled gaps",
    "no_shift": "Without context/query shift",
    "no_tail_warp": "Without marginal tail warps",
    "base_cardinality": "With generic cardinality sampling",
}


def snapshot_metadata() -> dict:
    file = paths.outputs_dir() / "manifests" / "notebook_snapshot.json"
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}


def coverage(track: str, exp: str = "exp1", *, components: bool = False) -> pd.DataFrame:
    variant = "_ablations" if components else ""
    cfg = load(paths.config_path(f"{exp.capitalize()}_{track.upper()}{variant}"),
               allow_placeholders=True)
    rows = []
    root = paths.runs_dir(int(exp.removeprefix("exp")))
    for run in expand_with_seeds(cfg):
        folder = root / run["_run_name"]
        summary = folder / "summary.json"
        saved = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
        steps = int(saved.get("steps", 0))
        progress = folder / "progress.csv"
        if progress.is_file():
            frame = pd.read_csv(progress)
            if "step" in frame and len(frame):
                steps = max(steps, int(frame.step.max()))
        complete = bool(saved.get("completed")) and steps >= run["train"]["max_steps"]
        label = (COMPONENTS[run["arm"]] if run.get("arm") in COMPONENTS
                 else style.prior_mix_label(run["prior"]["credit_fraction"]))
        if exp == "exp2" and not components:
            label = ("TabPFN-3" if "tabpfn3" in run["_run_name"] else "TabICLv2") + " · " + label + f" · LR {run['train']['lr']:g}"
        rows.append({"run": run["_run_name"], "seed": run["seed"],
                     "label": label, "credit_fraction": float(run["prior"]["credit_fraction"]),
                     "architecture": str(run.get("architecture", "tabicl")), "lr_index": run.get("lr_index", 0),
                     "state": "completed" if complete else "partial" if folder.is_dir() else "absent",
                     "steps": steps, "budget": run["train"]["max_steps"]})
    frame = pd.DataFrame(rows)
    # The quarter-mixture is appended in the launch grid to preserve old array
    # identities; figures show the scientific dose order, not submission order.
    if not components and len(frame):
        frame = frame.sort_values(["architecture", "credit_fraction", "seed"], kind="stable")
    return frame.reset_index(drop=True)


def provenance(track: str, exp: str = "exp1") -> str:
    from src.eval.protocol import PROTOCOL
    from src.utils.preflight import source_fingerprint

    main = coverage(track, exp)
    extra = coverage(track, exp, components=True) if exp == "exp1" else pd.DataFrame()
    meta = snapshot_metadata()
    lines = ["LOCAL EVIDENCE — not a live scheduler query",
             f"Experiment {exp.removeprefix('exp')}, {track.upper()}; current benchmark protocol {PROTOCOL}.",
             f"Local notebook-code fingerprint: {source_fingerprint()}."]
    if meta:
        lines += [f"Downloaded archive: {meta.get('archive_name')}; local download time {meta.get('downloaded_at')}.",
                  f"Archive SHA-256: {meta.get('sha256')}",
                  f"Recorded executable fingerprint: {meta.get('source_fingerprint')}",
                  f"Latest training record in that snapshot: {meta.get('latest_training_record', 'not recorded')}."]
    else:
        lines.append("No imported snapshot manifest; counts below describe files currently available locally.")
    for label, frame in (("Main comparison", main), ("Component tests", extra)):
        if len(frame):
            counts = frame.state.value_counts()
            lines.append(f"{label}: {counts.get('completed', 0)}/{len(frame)} downloaded completions; "
                         f"{counts.get('partial', 0)} partial traces; {counts.get('absent', 0)} without local output.")
    lines += ["Partial traces do not establish that a job is still running; absent output does not establish that it was never queued.",
              "Final credit/OOD benchmarking is a separate completion gate; monitoring is not its substitute."]
    return "\n".join(lines)


def component_coverage(track: str):
    frame = coverage(track, components=True)
    return _coverage_map(frame, "Conditional component-run coverage")


def main_coverage(track: str, exp: str = "exp1", *, backbone=None):
    frame = coverage(track, exp)
    if backbone is not None:
        frame = frame[frame.architecture.eq(backbone)].copy()
        frame["label"] = frame.label.str.split(" · ", n=1).str[-1]
    return _coverage_map(frame, "Main-run coverage in local records" + (f" · {backbone}" if backbone else ""))


def _coverage_map(frame, title):
    labels = frame.label.drop_duplicates().tolist()
    seeds = sorted(frame.seed.unique())
    style.apply()
    fig, ax = plt.subplots(figsize=style.row_figsize(len(labels), per_row=.45, base=1.5))
    colors = {"completed": style.CREDIT, "partial": style.EXTERNAL, "absent": style.GRID}
    for row in frame.itertuples():
        x, y = seeds.index(row.seed), labels.index(row.label)
        ax.add_patch(plt.Rectangle((x-.45, y-.4), .9, .8, color=colors[row.state], alpha=.85))
        text = "complete" if row.state == "completed" else f"{100*row.steps/row.budget:.0f}% recorded" if row.state == "partial" else "no output"
        ax.text(x, y, text, ha="center", va="center", fontsize=style.PANEL_TEXT,
                color=style.LIGHT_INK if row.state == "completed" else style.INK)
    ax.set(xticks=range(len(seeds)), xticklabels=[f"seed {s}" for s in seeds],
           yticks=range(len(labels)), yticklabels=labels, xlim=(-.6, len(seeds)-.4),
           ylim=(len(labels)-.5, -.5), title=title)
    ax.grid(False)
    style.legend_below(fig, [Patch(color=colors[k], label=v) for k,v in
                            (("completed", "completed"), ("partial", "partial trace"), ("absent", "no local output"))], ncol=3)
    return fig


def training_summary(track: str, exp: str = "exp1") -> str:
    """Notebook section order, without interpreting file absence as queue state."""
    from src.visualize import training_plots as tp

    lines = ["A · PROVENANCE AND RUN COVERAGE", provenance(track, exp)]
    if exp == "exp1":
        lines += ["Four main mixtures: 0%, 25%, 50%, 100%; three seeds each.",
                  "The 25% mixture is an extension approved after the initial 48-run campaign; "
                  "six new runs bring the planned total to 54, preserving completed results."]
    if exp == "exp2":
        lines += ["PLANNED CONTINUATION (not completed training)",
                  continuation_recipe(track).to_string(index=False),
                  "Matched mixtures within each backbone; row/task budgets differ between backbones.",
                  "0% is an extra-training control, not a guarantee against forgetting."]
    lines.append("\nB · OPTIMISATION HEALTH")
    runs = tp.load_progress(track, exp)
    if not runs:
        lines += ["No downloaded progress CSVs. Training health cannot be measured from this checkout.",
                  "\nC · DEVELOPMENT MONITORING", "No local curves; no mixture is selected here."]
        return "\n".join(lines)
    for name, frame in sorted(runs.items()):
        losses = frame.get("train_loss", pd.Series(dtype=float)).dropna()
        lines.append(f"{name}: {int(frame.step.max())} updates recorded; "
                     f"last recorded training loss {losses.iloc[-1]:.5g}." if len(losses)
                     else f"{name}: no finite training loss recorded.")
    metric = tp.HEADLINE[track]
    kept, excluded = tp.comparable_datasets(runs, metric)
    lines += ["\nC · DEVELOPMENT MONITORING",
              f"Common monitored development datasets: {', '.join(kept) or 'none'}.",
              f"Excluded from common monitoring mean: {', '.join(excluded) or 'none'}.",
              "Monitoring uses short diagnostic contexts. It is not the full-row, five-fold final benchmark.",
              "Different synthetic-task losses do not rank downstream prediction quality."]
    return "\n".join(lines)


def continuation_recipe(track: str) -> pd.DataFrame:
    """Display the actual expanded recipe, including backbone-specific shape overrides."""
    import math

    runs = expand_with_seeds(load(paths.config_path(f"Exp2_{track.upper()}")))
    rows = []
    for architecture in ("tabicl", "tabpfn3"):
        selected = [r for r in runs if r["architecture"] == architecture]
        p, t = selected[0]["prior"], selected[0]["train"]
        lo, hi = p["n_rows_range"]
        law = p.get("n_rows_sampling", "uniform")
        mean = lo if lo == hi else ((hi-lo)/math.log(hi/lo) if law == "log_uniform" else (lo+hi)/2)
        tasks = t["max_steps"] * t["batch_size"]
        rows.append({"Backbone": "TabICLv2" if architecture == "tabicl" else "TabPFN-3",
                     "Runs per track": len(selected), "Peak LRs": ", ".join(f"{lr:g}" for lr in sorted({r["train"]["lr"] for r in selected}, reverse=True)),
                     "Weight decay": t["weight_decay"], "Updates": t["max_steps"], "Tables per run": f"{tasks:,}",
                     "Rows per table": f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,} ({law.replace('_', '-')})",
                     "Context share": "–".join(f"{v:.0%}" for v in p["train_frac_range"]),
                     "Expected row exposures": f"{tasks*mean/1e6:,.1f} million"})
    return pd.DataFrame(rows)


def availability(message: str, *, title: str = "Evidence not yet available"):
    style.apply()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, .42))
    ax.axis("off")
    ax.text(.04, .76, title, transform=ax.transAxes, fontsize=style.NOTE_TITLE,
            weight="bold", ha="left")
    import textwrap
    ax.text(.04, .57, textwrap.fill(message, 84), transform=ax.transAxes,
            ha="left", va="top", fontsize=style.PANEL_TEXT)
    return fig
