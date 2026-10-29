"""Complete benchmark cells for continuation curves and reused Exp3 baselines.

These are analysis readers, not launch gates. Launch uses the stronger
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
    if track not in {"pd", "lgd"} or backbone not in {"tabicl", "tabpfn3"}:
        raise ValueError("Unknown benchmark track or backbone")
    model = "tabiclv2" if backbone == "tabicl" else "tabpfn3"
    if domain == "credit":
        cfg = load(paths.config_path(f"Exp2_{track.upper()}"), allow_placeholders=True)
        datasets = cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"]
        path = paths.benchmark_dir(paths.REFERENCE, track) / f"results_reference_{track}_{model}.csv"
    elif domain == "ood":
        from src.eval.ood import list_ood_datasets
        datasets = [d.name for d in list_ood_datasets("classification" if track == "pd" else "regression")]
        path = paths.benchmark_dir(paths.REFERENCE, "ood") / f"ood_results_reference_{track}_{model}.csv"
    else:
        raise ValueError("Unknown benchmark domain")
    if not datasets:
        return None
    # A complete shared reference is useful before the CPT campaign finishes.
    # Validate its own cells instead of depending on every final trained arm.
    part = read_cells(path, datasets, identity=model)
    if part is None or part.empty or metric not in part:
        return None
    scores = pd.to_numeric(part[metric], errors="coerce")
    if not np.isfinite(scores).all():
        return None
    return float(scores.groupby(part.dataset).mean().mean())


def read_cells(path, datasets, *, identity, step=None):
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        keys = list(df[["dataset", "seed", "fold"]].itertuples(index=False, name=None))
        expected = {(d, 0, fold) for d in datasets for fold in range(N_FOLDS)}
        from src.eval.comparisons import identities
        valid = (len(keys) == len(expected) and set(keys) == expected
                 and df["status"].eq("ok").all() and df["protocol"].eq(PROTOCOL).all()
                 and identities(df).eq(identity).all())
        if step is not None:
            valid &= "info_checkpoint_step" in df and df.info_checkpoint_step.eq(step).all()
        if not valid:
            raise ValueError("wrong identity/step/protocol or incomplete five-fold coverage")
        # CSV-only analysis cannot verify remote input receipts, but must never
        # relabel explicitly capped/subsampled evidence as a full-table result.
        for column in ("context_cap", "info_context_cap", "info_row_cap"):
            if column in df and df[column].notna().any():
                raise ValueError(f"full-table benchmark has a recorded {column}")
        if "subsampled" in df and not df["subsampled"].astype(str).str.lower().isin(("false", "0")).all():
            raise ValueError("full-table benchmark records subsampling")
        for columns in (("n_train", "n_train_used"),
                        ("prep_features_in", "prep_features_used", "n_features_used")):
            if any(c in df for c in columns):
                counts = df[list(columns)].apply(pd.to_numeric, errors="coerce")
                if (not np.isfinite(counts.to_numpy()).all() or not counts.gt(0).all().all()
                        or not counts.eq(counts.round()).all().all()
                        or not counts.eq(counts.iloc[:, 0], axis=0).all().all()):
                    raise ValueError("full-table benchmark did not retain all training rows/features")
        if "pred_nonfinite_frac" in df and not pd.to_numeric(df["pred_nonfinite_frac"], errors="coerce").eq(0).all():
            raise ValueError("benchmark discarded non-finite predictions")
        return df
    except (ValueError, KeyError, OSError) as exc:
        warnings.warn(f"Withheld {path.name}: {exc}", stacklevel=2)
        return None


def benchmark_frames(track: str, experiment: int, *, domain="credit", final_only=False):
    """Every available complete checkpoint; an absent seed is never averaged away."""
    cfg = load(paths.config_path(f"Exp{experiment}_{track.upper()}"), allow_placeholders=True)
    runs = expand_with_seeds(cfg)
    selections = [(experiment, r, r["_grid"]["array_index"], r.get("method", "full_wd")) for r in runs]
    if experiment == 3:
        baselines = expand_with_seeds(load(paths.config_path(f"Exp2_{track.upper()}"), allow_placeholders=True))
        for run in baselines:
            if run["prior"]["credit_fraction"] == .5 and run["lr_index"] == 0:
                selections.append((2, run, run["_grid"]["array_index"], "full_wd"))
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
                evaluation_role="fixed_design_credit" if domain == "credit" else "ood_not_used_for_selection",
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
