"""PIPELINE 2 — generate a pool of synthetic datasets and write it to disk.

    # one shard (what an array task runs)
    python scripts/generate_prior.py --config config/Exp1_LGD.yaml --variant original \
        --shard 0 --n-shards 20

    # everything, serially (fine locally, slow on a cluster)
    python scripts/generate_prior.py --config config/Exp1_LGD.yaml --variant original --all

    # check what exists
    python scripts/generate_prior.py --config config/Exp1_LGD.yaml --status

VARIANTS. `--variant original` forces `credit_fraction=0` and generates the
unmodified TabICL prior — the pool EVERY arm shares. `--variant credit_v1` forces
`credit_fraction=1` and generates only our credit datasets. Training then mixes the
two at whatever ratio its config asks for, which is why the two pools are generated
separately rather than pre-mixed.

CPU ONLY, on purpose. Generation is CPU-bound, so it belongs on cheap CPU nodes in
parallel — not on a GPU that would sit idle waiting for ExtraTrees fits.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import expand_with_seeds, load  # noqa: E402
from src.utils.logging_setup import log_environment, log_section, setup_logging  # noqa: E402
from src.utils.paths import describe, logs_dir, results_dir  # noqa: E402

#: What each variant means. `credit_fraction` is FORCED, so a pool can never end up
#: being a silent mixture of the two.
VARIANTS = {
    "original": 0.0,
    "credit_v1": 1.0,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="original", help=f"one of {sorted(VARIANTS)}")
    ap.add_argument("--shard", type=int, default=None, help="shard index; defaults to $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--n-shards", type=int, default=100)
    ap.add_argument("--n-datasets", type=int, default=400000, help="TOTAL across all shards")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="generate every shard in this process")
    ap.add_argument("--status", action="store_true", help="report what exists and exit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = expand_with_seeds(load(args.config))[0]
    task = cfg["task"]

    log, _, log_path = setup_logging(f"generate_prior_{task}_{args.variant}", logs_dir(), console=True)
    log_section(log, f"CreditICL — PRIOR GENERATION — {task.upper()} / {args.variant}")

    if args.status:
        from src.prior.pool import verify_pools

        report = verify_pools(task, sorted(VARIANTS), expect=args.n_datasets)
        out = results_dir(task, "prior")
        out.mkdir(parents=True, exist_ok=True)
        (out / "pool_status.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("wrote %s", out / "pool_status.json")
        return 0 if report["ok"] else 1

    if args.variant not in VARIANTS:
        ap.error(f"--variant must be one of {sorted(VARIANTS)}")

    log_environment(log, {"pipeline": "prior", "variant": args.variant, "task": task})
    log.info("storage:\n%s", "\n".join(f"    {k} = {v}" for k, v in describe().items()))

    # Force the mixture so a pool is unambiguously one thing or the other.
    prior_cfg = copy.deepcopy(cfg["prior"])
    prior_cfg["credit_fraction"] = VARIANTS[args.variant]
    log.info(
        "variant %r forces credit_fraction=%s (config said %s)",
        args.variant, prior_cfg["credit_fraction"], cfg["prior"].get("credit_fraction"),
    )
    log.info("budget: %d datasets across %d shards", args.n_datasets, args.n_shards)
    log.info(
        "for scale: TabICLv2 itself used about 35M synthetic datasets "
        "(500K+40K+10K steps at batch 64). We use O'Prior's controlled-comparison "
        "budget instead, and every arm gets exactly this many."
    )

    from src.prior.pool import generate_shard, pool_status

    shards = (
        range(args.n_shards)
        if args.all
        else [args.shard if args.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))]
    )

    for shard in shards:
        if not 0 <= shard < args.n_shards:
            log.error("shard %d out of range for --n-shards %d", shard, args.n_shards)
            return 2
        generate_shard(
            prior_cfg,
            task,
            args.variant,
            shard_index=shard,
            n_shards=args.n_shards,
            n_datasets_total=args.n_datasets,
            seed=args.seed,
            force=args.force,
        )

    status = pool_status(task, args.variant)
    log.info("pool now holds %d datasets in %s shards", status["n_datasets"], status["shards"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
