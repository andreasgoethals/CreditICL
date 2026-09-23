"""Completion receipts for paired credit/OOD benchmarks, including their provenance.

A CSV's existence is not completion. A receipt is written only after every requested
dataset/model/seed cell succeeds in both domains; changed inputs or results invalidate it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 2
REFERENCE_MODELS = "tabiclv2,tabpfn3,catboost,linear"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tag_for(exp: int, track: str, index: int, n_arms: int) -> str:
    if not 0 <= index <= n_arms:
        raise ValueError(f"Index {index} is outside 0..{n_arms}")
    return f"reference_{track}" if index == n_arms else f"exp{exp}bench_{track}_a{index}"


def receipt_path(tag: str) -> Path:
    return paths.results_dir() / "receipts" / f"{tag}.json"


def request(exp: int, track: str, index: int, context_cap: int = 1024,
            seeds: tuple[int, ...] = (0, 1, 2),
            reference_models: str = REFERENCE_MODELS) -> tuple[str, dict, dict[str, Path]]:
    from src.data.discovery import list_datasets
    from src.eval.ood import N_PER_TASK, list_ood_datasets, manifest_path

    cfg = load(ROOT / "config" / f"Exp{exp}_{track.upper()}.yaml")
    runs = expand_with_seeds(cfg)
    tag = tag_for(exp, track, index, len(runs))
    reference = index == len(runs)
    models = reference_models.split(",") if reference else ["crediticl"]
    kind = "regression" if track == "lgd" else "classification"
    credit = sorted(set(cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"]))
    ood = sorted(d.name for d in list_ood_datasets(kind))
    if not credit or not set(credit).issubset(list_datasets(track)):
        raise ValueError("Not all configured credit datasets are available")
    if len(ood) < N_PER_TASK:
        raise ValueError(f"OOD cache needs at least {N_PER_TASK} {kind} datasets")
    # Hash executable evaluation code, so old receipts cannot silently survive a fix.
    sources = list((ROOT / "src/eval").rglob("*.py")) + [
        ROOT / "scripts/evaluate.py", ROOT / "scripts/evaluate_ood.py",
        ROOT / "scripts/slurm/benchmark.slurm",
    ]
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}
    data = {"protocol": PROTOCOL, "track": track, "kind": kind, "models": models,
            "seeds": list(seeds), "context_cap": context_cap, "credit": credit, "ood": ood,
            "ood_manifest": digest(manifest_path()), "code": code}
    if not reference:
        run = runs[index]
        name = run_name(run)
        summary = json.loads(paths.run_summary_path(name).read_text(encoding="utf-8"))
        steps = int(run["train"]["max_steps"])
        if not summary.get("completed") or summary.get("steps") != steps:
            raise ValueError(f"Training is not complete for {name}")
        checkpoint = paths.checkpoints_dir() / name / f"step-{steps}.ckpt"
        stat = checkpoint.stat()
        data.update(run_name=name, config=run, checkpoint=str(checkpoint.resolve()),
                    checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns)
    files = {"credit": paths.results_dir(track, "eval") / f"results_{tag}.csv",
             "ood": paths.results_dir("ood", "eval") / f"ood_results_{tag}.csv"}
    return tag, data, files


def validate_cells(df: pd.DataFrame, datasets: list[str], models: list[str], seeds: list[int],
                   metric: str, context_cap: int) -> None:
    required = {"dataset", "model", "seed", "status", "context_cap", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing result columns: {sorted(required - set(df.columns))}")
    keys = list(df[["dataset", "model", "seed"]].itertuples(index=False, name=None))
    expected = set(product(datasets, models, seeds))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete, unexpected or duplicate dataset/model/seed cells")
    if not df["status"].eq("ok").all() or not np.isfinite(df[metric].astype(float)).all():
        raise ValueError("Failed cells or non-finite headline metrics")
    if not df["context_cap"].eq(context_cap).all():
        raise ValueError("Context cap differs from the requested benchmark protocol")


def check(exp: int, track: str, index: int, *, write: bool = False, **kwargs) -> bool:
    tag, wanted, files = request(exp, track, index, **kwargs)
    receipt = receipt_path(tag)
    if not write:
        if not receipt.is_file():
            return False
        old = json.loads(receipt.read_text(encoding="utf-8"))
        return old.get("request") == wanted and old.get("files") == {
            k: digest(p) for k, p in files.items()
        }
    for domain, file in files.items():
        df = pd.read_csv(file)
        metric = ("roc_auc" if track == "pd" else "r2")
        validate_cells(df, wanted[domain], wanted["models"], wanted["seeds"], metric,
                       wanted["context_cap"])
        if (domain == "credit" and "run_name" in wanted
                and ("info_run_name" not in df or not df["info_run_name"].eq(wanted["run_name"]).all())):
            raise ValueError("Credit result belongs to a different training arm")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"request": wanted, "files": {k: digest(p) for k, p in files.items()}},
                              indent=2), encoding="utf-8")
    tmp.replace(receipt)
    return True


def complete(exp: int, track: str, index: int, **kwargs) -> bool:
    try:
        return check(exp, track, index, **kwargs)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--track", required=True, choices=("pd", "lgd"))
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--context-cap", type=int, default=1024)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--reference-models", default=REFERENCE_MODELS)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    try:
        ok = check(args.exp, args.track, args.index, write=args.write,
                   context_cap=args.context_cap, seeds=tuple(map(int, args.seeds.split(","))),
                   reference_models=args.reference_models)
        return 0 if ok else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Benchmark incomplete: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
