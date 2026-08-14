"""PIPELINE 4b — score models on NON-CREDIT tasks, to check we did not break them.

    python scripts/evaluate_ood.py --models linear,catboost
    python scripts/evaluate_ood.py --models linear,catboost,tabiclv2 --seeds 0,1,2

WHY: we deliberately bend the prior toward credit risk. If that buys credit performance
by losing general performance, and we only ever measure credit datasets, we would never
see it. Three outcomes are all worth reporting — no degradation (the gain was free), mild
degradation (a quantified trade-off), or severe degradation (the prior is destructive).

Needs the OOD cache, fetched once on a LOGIN node:
    python -m src.utils.fetch_ood
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging_setup import log_environment, log_section, setup_logging  # noqa: E402
from src.utils.paths import logs_dir, results_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="linear,catboost")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--kinds", default="classification,regression")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=10_000)
    ap.add_argument("--reference", default=None,
                    help="model to report deltas against, e.g. the control checkpoint")
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--checkpoint", default=None,
        help="path to one of OUR step-*.ckpt files, required by --models crediticl",
    )
    ap.add_argument(
        "--task", choices=("pd", "lgd"), default=None,
        help="which of our checkpoints to score. Required with --models crediticl and no "
             "--checkpoint: a regression checkpoint cannot score classification datasets.",
    )
    args = ap.parse_args()

    log, _, log_path = setup_logging("evaluate_ood", logs_dir(), console=True)
    log_section(log, "CreditICL — OUT-OF-DOMAIN EVALUATION")
    log_environment(log, {"pipeline": "eval_ood", "models": args.models})

    # See scripts/evaluate.py — `crediticl` is registered explicitly, and forgetting it here
    # meant the out-of-domain check silently ran WITHOUT our own model, which is the only
    # model the check exists to interrogate.
    from src.eval.crediticl_baseline import register_or_warn as register_crediticl
    from src.eval.crediticl_baseline import resolve_our_checkpoint

    register_crediticl(log)

    from src.eval.ood_runner import OODEvalConfig, ood_text_summary, run_ood, summarise_ood

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]

    # Registration alone leaves `crediticl` unable to score anything — it needs a checkpoint.
    # The task defaults from the kinds being scored, since the two are tied: a regression
    # (LGD) checkpoint has no class head and cannot touch a classification dataset.
    model_kwargs: dict[str, dict[str, object]] = {}
    if "crediticl" in models:
        task = args.task or ("pd" if kinds == ["classification"] else "lgd")
        ckpt = resolve_our_checkpoint(args.checkpoint, task, log)
        if ckpt is not None:
            model_kwargs["crediticl"] = {"checkpoint": str(ckpt)}

    cfg = OODEvalConfig(
        models=models,
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        kinds=kinds,
        test_size=args.test_size,
        max_rows=args.max_rows,
        model_kwargs=model_kwargs,
    )
    df = run_ood(cfg)

    # OOD results are their OWN pipeline directory, never mixed with the credit results.
    out = results_dir("ood", "eval")
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    df.to_csv(out / f"ood_results{suffix}.csv", index=False)
    summarise_ood(df).to_csv(out / f"ood_summary{suffix}.csv", index=False)

    text = ood_text_summary(df, reference_model=args.reference)
    (out / f"ood_summary{suffix}.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    log.info("results -> %s", out)
    log.info("log file -> %s", log_path)
    return 0 if not df.empty and (df["status"] == "ok").any() else 1


if __name__ == "__main__":
    sys.exit(main())
