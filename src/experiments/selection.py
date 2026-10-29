"""Freeze Experiment 2A learning rates on all credit and Exp3 adapters on development.

python -m src.experiments.selection --stage learning-rate
python -m src.experiments.selection --stage adapters
Neither command submits a job or changes the runnable YAML. Receipts record all
candidate scores and input hashes. Credit outcomes in Exp2/3 are tuning-benchmark
results, not independent holdout evidence. OOD metrics never enter a score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths


def selection_path(design: str) -> Path:
    if design in {"cpt", "retention_pilot"}:
        return paths.experiment_dir(2) / "selection" / "learning_rate.json"
    return paths.experiment_dir(3) / "selection" / "adapters.json"


def load_selection(design: str, *, optional=False, for_analysis=False) -> dict | None:
    """Strict launch gate by default; downloaded receipts may be read for plots.

    Analysis mode still checks source, schema and values, but cannot attest that
    VSC paths/packages are present on the laptop. It never authorizes training.
    """
    from src.eval.benchmark_status import input_digest
    from src.utils.preflight import source_fingerprint

    path = selection_path(design)
    if not path.exists() and optional:
        return None
    if not path.exists():
        stage = "learning-rate" if design in {"cpt", "retention_pilot"} else "adapters"
        raise ValueError(f"Experiment selection is not frozen. Run python -m "
                         f"src.experiments.selection --stage {stage} after the required benchmarks.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected_design = "cpt" if design == "retention_pilot" else design
    if (saved.get("schema") != 3 or saved.get("design") != expected_design
            or saved.get("source_fingerprint") != source_fingerprint()
            or set(saved.get("choices", {})) != {f"{t}/{m}" for t in ("pd", "lgd") for m in ("tabicl", "tabpfn3")}
            or not saved.get("inputs")
            or saved.get("credit_evaluation_role") != "tuning_benchmark"
            or saved.get("selection_scope") != ("all_credit" if expected_design == "cpt" else "development")
            or saved.get("ood_used_for_selection") is not False):
        raise ValueError("Invalid or stale experiment selection receipt")
    from src.experiments.design import RATES, CANDIDATES
    for identity, choice in saved["choices"].items():
        if choice.get("full_lr") not in RATES[identity.split("/")[1]]:
            raise ValueError("Invalid selected base learning rate")
        if design == "retention" and any(type(choice.get(m)) is not int or choice[m] not in range(len(v))
                                         for m, v in CANDIDATES.items()):
            raise ValueError("Invalid selected adapter candidate")
    if not for_analysis:
        from src.utils.environment import environment_fingerprint
        from src.validation.campaign import input_inventory
        if (saved.get("environment_fingerprint") != environment_fingerprint()
                or saved.get("validation_inputs") != input_inventory()):
            raise ValueError("Selection environment or benchmark inputs changed")
        for name, digest in saved["inputs"].items():
            if input_digest(Path(name)) != digest:
                raise ValueError(f"Selection input changed: {name}")
    saved["receipt_digest"] = input_digest(path)
    return saved


def development_score(frame: pd.DataFrame, datasets: list[str], name: str, metric: str) -> float:
    """Equal dataset/fold weights; complete five folds with no hidden row budget."""
    from src.eval.protocol import N_FOLDS

    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Development datasets must be nonempty and unique")
    part = frame[frame.dataset.isin(datasets)].copy()
    expected = {(d, 0, f) for d in datasets for f in range(N_FOLDS)}
    keys = list(part[["dataset", "seed", "fold"]].itertuples(index=False, name=None))
    if (set(keys) != expected or len(keys) != len(expected)
            or not part.status.eq("ok").all() or not part.info_run_name.eq(name).all()
            or not np.isfinite(part[metric]).all()):
        raise ValueError(f"Incomplete or mismatched development evidence: {name}")
    return float(part.groupby("dataset")[metric].mean().mean())


def selection_datasets(cfg: dict, stage: str) -> list[str]:
    """Explicit tuning scope; legacy partition names do not imply a test holdout."""
    evaluation = cfg["eval"]
    if stage == "learning-rate":
        if evaluation.get("selection_scope") != "all_credit":
            raise ValueError("Experiment 2A requires the declared all-credit tuning scope")
        datasets = evaluation["dev_datasets"] + evaluation["holdout_datasets"]
    elif stage == "adapters":
        datasets = evaluation["dev_datasets"]
    else:
        raise ValueError(f"Unknown selection stage: {stage}")
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("Selection datasets must be nonempty and unique")
    return datasets


def choose(rows: list[dict]) -> dict:
    """Highest primary metric; exact ties prefer the smaller numeric candidate."""
    if not rows or any(not np.isfinite(r["score"]) for r in rows):
        raise ValueError("No complete finite candidates")
    return sorted(rows, key=lambda r: (-r["score"], r["value"]))[0]


def freeze(stage: str) -> Path:
    from src.eval import benchmark_status as bs
    from src.experiments.design import CANDIDATES
    from src.utils.config import expand_with_seeds, load, run_name
    from src.utils.preflight import source_fingerprint
    from src.validation.campaign import write_json
    from src.validation.campaign import input_inventory
    from src.utils.environment import environment_fingerprint

    original_source = source_fingerprint()
    original_environment = environment_fingerprint()
    original_inputs = input_inventory()
    pilot = stage == "adapters"
    exp, variant = (3, "pilots") if pilot else (2, "pilots")
    previous = load_selection("retention_pilot") if pilot else None
    choices = {k: dict(v) for k, v in previous["choices"].items()} if previous else {}
    inputs, rankings, baselines, dataset_inventory = {}, [], [], {}
    if previous:
        path = selection_path("retention_pilot")
        inputs[str(path.resolve())] = bs.input_digest(path)
        baselines = previous["baselines"]
    for track in ("pd", "lgd"):
        cfg = load(bs.config_path(exp, track, variant))
        datasets = selection_datasets(cfg, stage)
        dataset_inventory[track] = datasets
        runs = expand_with_seeds(cfg)
        scores = {}
        for index, run in enumerate(runs):
            # Final checkpoint slots precede learning-curve slots.
            if not bs.complete(exp, track, index, variant=variant):
                raise ValueError(f"Required final benchmark is incomplete: Exp{exp} {track} slot {index}")
            tag, _, files = bs.request(exp, track, index, variant=variant)
            for path in [files["credit"], bs.receipt_path(tag)]:
                inputs[str(path.resolve())] = bs.input_digest(path)
            metric = "roc_auc" if track == "pd" else "r2"
            score = development_score(pd.read_csv(files["credit"]), datasets,
                                      run_name(run), metric)
            key = (run["architecture"], run.get("method", "full"),
                   run["candidate"] if pilot else run["lr_index"])
            scores.setdefault(key, []).append((score, run))
        for model in ("tabicl", "tabpfn3"):
            identity = f"{track}/{model}"
            choices.setdefault(identity, {})
            for method in CANDIDATES if pilot else ("full",):
                rows = []
                for (backbone, treatment, candidate), observations in scores.items():
                    if (backbone, treatment) != (model, method):
                        continue
                    if pilot:
                        valid = len(observations) == 1 and observations[0][1]["seed"] == 991
                    else:
                        valid = (len(observations) == 2 and
                                 {(r["prior"]["credit_fraction"], r["seed"]) for _, r in observations}
                                 == {(0.0, 990), (0.5, 990)})
                    if not valid:
                        raise ValueError("Missing paired pilot mixtures or seed")
                    value = CANDIDATES[method][candidate] if pilot else observations[0][1]["train"]["lr"]
                    rows.append(dict(track=track, model=model, method=method, candidate=candidate,
                                     value=value, score=float(np.mean([x[0] for x in observations])),
                                     training_seeds=sorted({x[1]["seed"] for x in observations}),
                                     mixtures=sorted({x[1]["prior"]["credit_fraction"] for x in observations}),
                                     metric=metric))
                if len(rows) != (3 if pilot else 2):
                    raise ValueError(f"Missing candidates for {identity}/{method}")
                winner = choose(rows)
                rankings.extend(rows)
                choices[identity][method if pilot else "full_lr"] = winner["candidate"] if pilot else winner["value"]
        if not pilot:
            # Baselines are the separate three-seed main runs, never pilot weights.
            main = expand_with_seeds(load(bs.config_path(2, track), allow_placeholders=True))
            baselines.extend({"track": track, "model": r["architecture"], "run_name": run_name(r),
                              "array_index": r["_grid"]["array_index"], "seed": r["seed"]}
                             for r in main if r["prior"]["credit_fraction"] == .5)
    design = "retention" if pilot else "cpt"
    destination = selection_path(design)
    payload = dict(schema=3, design=design, source_fingerprint=source_fingerprint(), inputs=inputs,
                   environment_fingerprint=original_environment, validation_inputs=original_inputs,
                   choices=choices, candidates=rankings, baselines=baselines,
                   selection_scope="development" if pilot else "all_credit",
                   selection_datasets=dataset_inventory, credit_evaluation_role="tuning_benchmark",
                   ood_used_for_selection=False,
                   criterion=("development" if pilot else "all-credit") +
                   " ROC-AUC (PD) / R2 (LGD); equal dataset and pilot-mixture weights; lower value on exact tie")
    if (source_fingerprint() != original_source or environment_fingerprint() != original_environment
            or input_inventory() != original_inputs
            or any(bs.input_digest(Path(p)) != h for p, h in inputs.items())):
        raise ValueError("Code or selection evidence changed while selecting; no receipt written")
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Selection already frozen with different evidence; preserve it and review before replacing")
    else:
        write_json(destination, payload)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("learning-rate", "adapters"))
    args = parser.parse_args(argv)
    try:
        print(f"Selection frozen: {freeze(args.stage)}")
    except (ValueError, OSError, KeyError) as exc:
        print(f"Selection blocked: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
