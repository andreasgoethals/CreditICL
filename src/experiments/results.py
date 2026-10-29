"""Complete benchmark cells for continuation curves and reused Exp3 baselines.

These are analysis readers, not launch gates. Launch/selection uses the stronger
checkpoint/input/package receipts in benchmark_status. Downloaded CSVs can be
analysed without copying hundreds of GB of checkpoints to the laptop.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.eval.protocol import PROTOCOL, N_FOLDS, checkpoint_steps
from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name


def released_score(track, backbone, domain, metric):
    """Shared step-zero reference, with the same datasets as the CPT curves."""
    from src.eval.comparisons import identities
    from src.eval.results import load_results
    from src.eval.analysis import load_ood_results

    frame = load_results(track, exp="exp2", protocol=PROTOCOL) if domain == "credit" else load_ood_results(track, "exp2")
    if frame is None or frame.empty or metric not in frame:
        return None
    part = frame[identities(frame).eq("tabiclv2" if backbone == "tabicl" else "tabpfn3")]
    if domain == "credit":
        cfg = load(paths.config_path(f"Exp2_{track.upper()}"), allow_placeholders=True)
        part = part[part.dataset.isin(cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"])]
    if part.empty or not np.isfinite(part[metric]).all():
        return None
    return float(part.groupby("dataset")[metric].mean().mean())


def read_cells(path, datasets, *, identity, step=None):
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        keys = list(df[["dataset", "seed", "fold"]].itertuples(index=False, name=None))
        expected = {(d, 0, fold) for d in datasets for fold in range(N_FOLDS)}
        from src.eval.comparisons import identities
        valid = (len(keys) == len(expected) and set(keys) == expected
                 and df.status.eq("ok").all() and df.protocol.eq(PROTOCOL).all()
                 and identities(df).eq(identity).all())
        if step is not None:
            valid &= "info_checkpoint_step" in df and df.info_checkpoint_step.eq(step).all()
        if not valid:
            raise ValueError("wrong identity/step/protocol or incomplete five-fold coverage")
        return df
    except (ValueError, KeyError, OSError) as exc:
        warnings.warn(f"Withheld {path.name}: {exc}", stacklevel=2)
        return None


def benchmark_frames(track: str, experiment: int, *, domain="credit", final_only=False):
    """Every available complete checkpoint; an absent seed is never averaged away."""
    cfg = load(paths.config_path(f"Exp{experiment}_{track.upper()}"), allow_placeholders=True)
    runs = expand_with_seeds(cfg)
    if any(r.get("design_unresolved") for r in runs):
        return pd.DataFrame()
    selections = [(experiment, r, r["_grid"]["array_index"], r.get("method", "full_wd")) for r in runs]
    if experiment == 3:
        from src.experiments.selection import load_selection
        saved = load_selection("retention", for_analysis=True)
        baselines = expand_with_seeds(load(paths.config_path(f"Exp2_{track.upper()}"), allow_placeholders=True))
        for item in saved["baselines"]:
            if item["track"] == track:
                run = baselines[item["array_index"]]
                if run_name(run) != item["run_name"]:
                    raise ValueError("Frozen baseline identity changed")
                selections.append((2, run, item["array_index"], "full_wd"))
    if domain == "credit":
        datasets = cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"]
    elif domain == "ood":
        from src.eval.ood import list_ood_datasets
        datasets = [d.name for d in list_ood_datasets("classification" if track == "pd" else "regression")]
    else:
        raise ValueError("Unknown benchmark domain")
    if not datasets:
        return pd.DataFrame()
    frames = []
    prefix = "results" if domain == "credit" else "ood_results"
    folder_track = track if domain == "credit" else "ood"
    for exp, run, index, method in selections:
        steps = checkpoint_steps(run)[:1] if final_only else checkpoint_steps(run)
        for step in steps:
            tail = "" if step == run["train"]["max_steps"] else f"_s{step}"
            path = paths.benchmark_dir(exp, folder_track) / f"{prefix}_exp{exp}bench_{track}_a{index}{tail}.csv"
            frame = read_cells(path, datasets, identity=run_name(run), step=step)
            if frame is None:
                continue
            frames.append(frame.assign(architecture=run["architecture"], treatment=method,
                evaluation_role="tuning_benchmark" if domain == "credit" else "ood_not_used_for_selection",
                learning_rate=run["train"]["lr"], lr_index=run.get("lr_index", 0),
                mixture=run["prior"]["credit_fraction"], training_seed=run["seed"], checkpoint_step=step))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def complete_scores(frame, metric, *, experiment):
    """Equal dataset weights, paired seeds 0/1/2; no partial-group summaries."""
    if frame.empty or metric not in frame:
        return pd.DataFrame()
    keys = ["architecture", "lr_index", "mixture"] if experiment == 2 else ["architecture", "treatment"]
    rows = []
    for group, part in frame.groupby(keys + ["checkpoint_step"]):
        if set(part.training_seed.unique()) != {0, 1, 2} or not np.isfinite(part[metric]).all():
            continue
        datasets = [set(g.dataset) for _, g in part.groupby("training_seed")]
        if any(ds != datasets[0] for ds in datasets):
            continue
        scores = part.groupby(["training_seed", "dataset"])[metric].mean()
        by_seed = scores.groupby(level=0).mean()
        rows.append(dict(zip(keys + ["checkpoint_step"], group), mean=by_seed.mean(),
                         minimum=by_seed.min(), maximum=by_seed.max(), datasets=len(datasets[0])))
    return pd.DataFrame(rows)


def paired_retention(frame, metric):
    """Four methods versus the reused, seed-matched 50% full-CPT baseline."""
    from src.eval.comparisons import paired_prior_effects
    if frame.empty:
        return pd.DataFrame()
    counts = frame.groupby("treatment").training_seed.agg(lambda s: set(s))
    methods = {"full_wd", "no_decay", "l2sp", "diagscale", "lora"}
    if set(counts.index) != methods or any(s != {0, 1, 2} for s in counts):
        return pd.DataFrame()
    copy = frame.copy()
    copy["info_run_name"] = [f"exp3_{m}__s{s}" for m, s in zip(copy.treatment, copy.training_seed)]
    pairs = {f"exp3_{m}": "exp3_full_wd" for m in methods - {"full_wd"}}
    return paired_prior_effects(copy, metric, pairs=pairs)
