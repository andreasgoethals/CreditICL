"""PIPELINE 4 — score the baselines on every real credit dataset.

    python scripts/evaluate.py --task lgd
    python scripts/evaluate.py --task both --seeds 0,1,2
    python scripts/evaluate.py --task pd --models catboost,tabiclv2 --datasets 0008.german

Preprocesses anything missing first, so one command is enough from a fresh clone.

Writes to results/<task>/eval/:
    results_<timestamp>.csv    one row per (dataset, model, seed)
    summary_<timestamp>.csv    mean of the headline metrics per model
Logs go to logs/ and contain nothing you need to keep.

A model that fails leaves a row with status="failed" and the error; the sweep
continues. Twenty results plus one explained failure beats zero results.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.baselines import DEFAULT_BASELINES, availability_report  # noqa: E402
from src.eval.crediticl_baseline import register_or_warn as register_crediticl  # noqa: E402
from src.eval.crediticl_baseline import resolve_our_checkpoint  # noqa: E402
from src.eval.runner import EvalConfig, run, summarise  # noqa: E402
from src.utils.logging_setup import log_environment, log_section, setup_logging  # noqa: E402
from src.utils.paths import describe, logs_dir, results_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("pd", "lgd", "both"), default="both")
    ap.add_argument("--datasets", default=None, help="comma-separated slugs; default all")
    ap.add_argument("--models", default=",".join(DEFAULT_BASELINES))
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--split", default="random", choices=("random", "temporal"))
    ap.add_argument("--tag", default=None, help="suffix for the output filenames")
    ap.add_argument(
        "--checkpoint", default=None,
        help="path to one of OUR step-*.ckpt files, required by --models crediticl. "
             "Omit it and the single checkpoint matching --task is used; with several, "
             "the run refuses to guess rather than scoring an arbitrary arm.",
    )
    args = ap.parse_args()

    tasks = ["pd", "lgd"] if args.task == "both" else [args.task]
    stamp = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    log, _, log_path = setup_logging(f"evaluate_{stamp}", logs_dir(), level="INFO", console=True)
    log_section(log, "CreditICL — EVAL PIPELINE")
    log_environment(log, {"pipeline": "eval", "split": args.split, "test_size": args.test_size})
    log.info("storage:\n%s", "\n".join(f"    {k} = {v}" for k, v in describe().items()))
    log.info("log file -> %s", log_path)

    # `crediticl` — OUR checkpoints — is added to the registry explicitly rather than at import
    # time, so a missing training dependency cannot stop the external baselines from running.
    # THIS CALL HAS TO BE HERE. Nothing in the production path made it, so `--models crediticl`
    # died with an unknown-baseline error inside a SLURM job that then reported success.
    register_crediticl(log)

    for name, (ok, err) in availability_report().items():
        log.info("baseline %-10s %s", name, "available" if ok else f"UNAVAILABLE — {err}")

    exit_code = 0
    models = args.models.split(",")
    for task in tasks:
        # Registering `crediticl` only makes the NAME resolvable. It also needs a checkpoint,
        # and passing none is why every one of its cells failed on 14-08-2026 while the run
        # still reported "25/50 cells OK".
        model_kwargs = {}
        if "crediticl" in models:
            ckpt = resolve_our_checkpoint(args.checkpoint, task, log)
            if ckpt is not None:
                model_kwargs["crediticl"] = {"checkpoint": str(ckpt)}

        cfg = EvalConfig(
            task=task,
            datasets=args.datasets.split(",") if args.datasets else None,
            models=models,
            seeds=[int(s) for s in args.seeds.split(",")],
            test_size=args.test_size,
            split=args.split,
            model_kwargs=model_kwargs,
        )
        df = run(cfg)

        out = results_dir(task, "eval")
        out.mkdir(parents=True, exist_ok=True)
        res_path = out / f"results_{stamp}.csv"
        df.to_csv(res_path, index=False)
        log.info("wrote %s (%d rows)", res_path, len(df))

        summary = summarise(df, task)
        if not summary.empty:
            sum_path = out / f"summary_{stamp}.csv"
            summary.to_csv(sum_path, index=False)
            log.info("wrote %s", sum_path)
            log.info("summary for %s:\n%s", task, summary.to_string(index=False))

        if not df.empty and (df["status"] != "ok").any():
            exit_code = 1  # non-zero so a SLURM job surfaces partial failure

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
