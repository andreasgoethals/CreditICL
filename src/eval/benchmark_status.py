"""Completion receipts for paired credit/OOD benchmarks, including their provenance, and the map
from a SLURM array index to what that index scores.

A CSV's existence is not completion. A receipt is written only after every requested
dataset/model/seed/fold cell succeeds in both domains; changed inputs or results invalidate it.

THE ARRAY (evaluation protocol 4, `src/eval/protocol.py`)

    index = position x n_arms + arm     for every saved checkpoint of every arm, where
                                        position 0 is the FINAL step, then 2,500, 5,000, ...
    index = n_arms x n_steps + k        the reference models, one per slot (k-th of
                                        `REFERENCE_MODELS`)
    then, per cap in `eval.context_caps`  every arm's FINAL checkpoint and every reference model
                                        that has a context (`CONTEXT_MODELS`), scored again with the
                                        context capped at that many rows — the prior's effect inside
                                        the training distribution (30-09-2026; tag suffix `_ctx<cap>`)

Final checkpoints come first, so the headline table exists before the learning curves. A
final checkpoint writes `experiment_<N>/benchmark/<track>/results_exp<N>bench_<track>_a<arm>.csv`
(the name the results notebooks read); an earlier one adds `_s<step>`. Each reference model
writes `reference/benchmark/<track>/results_reference_<track>_<model>.csv`, once for all
experiments (`paths.owner_of`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.protocol import N_FOLDS, PROTOCOL, checkpoint_steps
from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MODELS = "tabiclv2,tabpfn3,catboost,linear"
#: The models whose context can be capped: in-context learners. CatBoost and the linear model have
#: no context — their "context" is the training set, and capping it is a different experiment.
CONTEXT_MODELS = ("crediticl", "tabiclv2", "tabpfn3")


def config_path(exp: int, track: str, variant: str | None = None) -> Path:
    """`config/Exp<N>_<TRACK>[_<variant>].yaml` — a variant is an extra config beside the main one."""
    return ROOT / "config" / f"Exp{exp}_{track.upper()}{'_' + variant if variant else ''}.yaml"


def arm_model(run: dict) -> str:
    """The evaluation wrapper for an arm's checkpoint: ours (`crediticl`, TabICL architecture) or
    TabPFN's own (`tabpfn3`, which loads the checkpoint the TabPFN trainer writes)."""
    return "tabpfn3" if str(run.get("architecture", "tabicl")) == "tabpfn3" else "crediticl"


def context_caps(cfg: dict) -> list[int]:
    """The capped-context evaluations a config asks for, besides the full one (`eval.context_caps`)."""
    return [int(c) for c in ((cfg.get("eval") or {}).get("context_caps") or [])]


def reference_models_for(cfg: dict, reference_models: str) -> list[str]:
    """The reference slots of a config. NONE for a configuration that scores only the development
    datasets (no holdout list): its reference results would overwrite the shared `reference/`
    files, which hold every dataset, with a development-only subset."""
    if not (cfg.get("eval") or {}).get("holdout_datasets"):
        return []
    return [m for m in reference_models.split(",") if m]


@dataclass(frozen=True)
class Slot:
    """What one array index scores."""

    kind: str  # "arm" | "reference" | "none"
    tag: str = ""
    arm: int | None = None
    step: int | None = None
    final: bool = False
    model: str | None = None
    context: int | None = None  # rows of context, when capped; None = the whole training pool


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def slot_for(exp: int, track: str, index: int, reference_models: str = REFERENCE_MODELS,
             variant: str | None = None) -> Slot:
    # Placeholders allowed: only the grid size and the checkpoint steps are read here.
    cfg = load(config_path(exp, track, variant), allow_placeholders=True)
    runs = expand_with_seeds(cfg)
    steps = checkpoint_steps(runs[0])
    n_arms, n_steps = len(runs), len(steps)
    refs = reference_models_for(cfg, reference_models)
    if index < 0:
        raise ValueError(f"Index {index} is negative")
    if index < n_arms * n_steps:
        position, arm = divmod(index, n_arms)
        step, final = steps[position], position == 0
        # A variant is its own benchmark: `exp2<variant>bench_...` cannot collide with `exp2bench_...`.
        tag = (f"exp{exp}{variant or ''}bench_{track}_a{arm}" + ("" if final else f"_s{step}"))
        return Slot("arm", tag, arm=arm, step=step, final=final, model=arm_model(runs[arm]))
    k = index - n_arms * n_steps
    if k < len(refs):
        return Slot("reference", f"reference_{track}_{refs[k]}", model=refs[k])
    k -= len(refs)
    capped_refs = [m for m in refs if m in CONTEXT_MODELS]
    for cap in context_caps(cfg):
        if k < n_arms:
            tag = f"exp{exp}{variant or ''}bench_{track}_a{k}_ctx{cap}"
            return Slot("arm", tag, arm=k, step=steps[0], final=True, model=arm_model(runs[k]), context=cap)
        k -= n_arms
        if k < len(capped_refs):
            return Slot("reference", f"reference_{track}_{capped_refs[k]}_ctx{cap}", model=capped_refs[k],
                        context=cap)
        k -= len(capped_refs)
    return Slot("none")


def n_slots(exp: int, track: str, reference_models: str = REFERENCE_MODELS,
            variant: str | None = None) -> int:
    cfg = load(config_path(exp, track, variant), allow_placeholders=True)
    runs = expand_with_seeds(cfg)
    refs = reference_models_for(cfg, reference_models)
    capped = len(context_caps(cfg)) * (len(runs) + len([m for m in refs if m in CONTEXT_MODELS]))
    return len(runs) * len(checkpoint_steps(runs[0])) + len(refs) + capped


def receipt_path(tag: str) -> Path:
    return paths.benchmark_dir_for(tag, "receipts") / f"{tag}.json"


def request(exp: int, track: str, index: int, seeds: tuple[int, ...] = (0,), n_folds: int = N_FOLDS,
            reference_models: str = REFERENCE_MODELS, variant: str | None = None
            ) -> tuple[str, dict, dict[str, Path]]:
    from src.data.discovery import list_datasets
    from src.eval.ood import N_PER_TASK, list_ood_datasets, manifest_path

    cfg = load(config_path(exp, track, variant))
    runs = expand_with_seeds(cfg)
    slot = slot_for(exp, track, index, reference_models, variant)
    if slot.kind == "none":
        raise ValueError(f"Index {index} scores nothing")
    models = [slot.model]
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
            "seeds": list(seeds), "n_folds": n_folds, "credit": credit, "ood": ood,
            "ood_manifest": digest(manifest_path()), "code": code, "context_cap": slot.context}
    if slot.kind == "arm":
        run = runs[slot.arm]
        name = run_name(run)
        summary = json.loads(paths.run_summary_path(name).read_text(encoding="utf-8"))
        final = int(run["train"]["max_steps"])
        if not summary.get("completed") or summary.get("steps") != final:
            raise ValueError(f"Training is not complete for {name}")
        checkpoint = paths.run_checkpoints_dir(name) / f"step-{slot.step}.ckpt"
        stat = checkpoint.stat()
        data.update(run_name=name, config=run, step=slot.step, checkpoint=str(checkpoint.resolve()),
                    checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns)
    files = {"credit": paths.benchmark_dir_for(slot.tag, track) / f"results_{slot.tag}.csv",
             "ood": paths.benchmark_dir_for(slot.tag, "ood") / f"ood_results_{slot.tag}.csv"}
    return slot.tag, data, files


def validate_cells(df: pd.DataFrame, datasets: list[str], models: list[str], seeds: list[int],
                   n_folds: int, metric: str, context: int | None = None) -> None:
    required = {"dataset", "model", "seed", "fold", "status", "protocol", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing result columns: {sorted(required - set(df.columns))}")
    keys = list(df[["dataset", "model", "seed", "fold"]].itertuples(index=False, name=None))
    expected = set(product(datasets, models, seeds, range(n_folds)))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete, unexpected or duplicate dataset/model/seed/fold cells")
    if not df["status"].eq("ok").all() or not np.isfinite(df[metric].astype(float)).all():
        raise ValueError("Failed cells or non-finite headline metrics")
    if not df["protocol"].eq(PROTOCOL).all():
        raise ValueError(f"Rows from another evaluation protocol (want {PROTOCOL})")
    if context is None:
        if "context_cap" in df and df["context_cap"].notna().any():
            raise ValueError("A context cap was applied; the protocol gives every model the whole pool")
    elif "context_cap" not in df or not pd.to_numeric(df["context_cap"], errors="coerce").eq(context).all():
        raise ValueError(f"This slot scores contexts capped at {context} rows; the file was not")


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
        if domain == "ood" and "status" in df:
            # An LGD checkpoint declines classification suites and vice versa; only its own kind counts.
            df = df[df["status"] != "skipped"]
        metric = ("roc_auc" if track == "pd" else "r2")
        validate_cells(df, wanted[domain], wanted["models"], wanted["seeds"], wanted["n_folds"], metric,
                       context=wanted.get("context_cap"))
        if (domain == "credit" and "run_name" in wanted
                and ("info_run_name" not in df or not df["info_run_name"].eq(wanted["run_name"]).all())):
            raise ValueError("Credit result belongs to a different training arm")
        if (domain == "credit" and "step" in wanted
                and ("info_checkpoint_step" not in df
                     or not df["info_checkpoint_step"].astype(int).eq(wanted["step"]).all())):
            raise ValueError("Credit result belongs to a different checkpoint")
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=int, required=True, choices=(0, 1, 2, 3))
    ap.add_argument("--track", required=True, choices=("pd", "lgd"))
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--reference-models", default=REFERENCE_MODELS)
    ap.add_argument("--variant", default=None, help="config variant: config/Exp<N>_<TRACK>_<variant>.yaml")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--describe", action="store_true",
                    help="print what the index scores (kind tag arm step model context) and exit")
    ap.add_argument("--count", action="store_true", help="print the number of array slots and exit")
    args = ap.parse_args(argv)
    variant = args.variant or None
    if args.count:
        print(n_slots(args.exp, args.track, args.reference_models, variant))
        return 0
    if args.describe:
        s = slot_for(args.exp, args.track, int(args.index), args.reference_models, variant)
        print(s.kind, s.tag or "-", "-" if s.arm is None else s.arm, "-" if s.step is None else s.step,
              s.model or "-", "-" if s.context is None else s.context)
        return 0
    try:
        ok = check(args.exp, args.track, int(args.index), write=args.write,
                   seeds=tuple(map(int, args.seeds.split(","))), n_folds=args.folds,
                   reference_models=args.reference_models, variant=variant)
        return 0 if ok else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Benchmark incomplete: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
