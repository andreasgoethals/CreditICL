"""Pretrain NanoTabICLv2 on one grid point of a prior config.

One invocation == one grid point == one SLURM array task == one checkpoint dir.

    python scripts/pretrain.py --config config/LGD.yaml --index 0

The config's lever grid is expanded deterministically (see src/utils/config.py),
so `--index` maps to the same configuration on every call. That is what makes a
resubmitted or resumed array task land on the run it was meant to.

Useful before submitting anything:

    python scripts/pretrain.py --config config/LGD.yaml --list
    python scripts/pretrain.py --config config/LGD.yaml --index 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import expand_with_seeds, load_yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a prior config YAML")
    ap.add_argument("--index", type=int, default=None, help="grid index; defaults to $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--out-root", default=None, help="output root; defaults to config.out_root or ./res")
    ap.add_argument("--list", action="store_true", help="print the expanded grid and exit")
    ap.add_argument("--dry-run", action="store_true", help="resolve the config and exit without training")
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps (for smoke tests)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument(
        "--prior-source",
        choices=["generate", "pool"],
        default=None,
        help="override prior.pool.source. 'pool' reads pre-generated shards and is "
        "what the SLURM chain uses; 'generate' builds datasets live.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    runs = expand_with_seeds(cfg)

    if args.list:
        print(f"{len(runs)} runs from {args.config}")
        for r in runs:
            g = r["_grid"]
            print(f"  [{g['array_index']:>4}] seed={r['seed']} {g['tag']}")
        print(f"\nSLURM: #SBATCH --array=0-{len(runs) - 1}")
        return 0

    index = args.index
    if index is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            ap.error("--index is required when SLURM_ARRAY_TASK_ID is not set")
        index = int(env)

    if not 0 <= index < len(runs):
        print(f"index {index} out of range: config expands to {len(runs)} runs", file=sys.stderr)
        return 2

    run = runs[index]
    if args.max_steps is not None:
        run.setdefault("train", {})["max_steps"] = args.max_steps
    if args.prior_source is not None:
        run.setdefault("prior", {}).setdefault("pool", {})["source"] = args.prior_source

    # Two storage tiers, same split CreditPFN uses (see src/utils/paths.py):
    #   metrics / logs / resolved config -> $VSC_DATA  (small, backed up)
    #   checkpoints                      -> project staging (big, ~1 TB)
    from src.utils import paths

    if args.out_root:
        out_dir = Path(args.out_root) / run["_run_name"]
        ckpt_dir = out_dir / "checkpoints"
        log_dir = Path(args.out_root) / "logs"
    elif paths.on_vsc():
        out_dir = paths.outputs_dir() / run["_run_name"]
        log_dir = paths.logs_dir()
        ckpt_dir = paths.resolve_writable(
            paths.checkpoints_dir() / run["_run_name"],
            fallback=paths.outputs_dir() / run["_run_name"] / "checkpoints",
        )
    else:
        out_dir = paths.outputs_dir() / run["_run_name"]
        ckpt_dir = out_dir / "checkpoints"
        log_dir = ROOT / "logs"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")

    print(f"=== {run['_run_name']} ===", flush=True)
    print(f"grid {index + 1}/{len(runs)}  levers: {json.dumps(run['_grid']['assignments'], default=str)}", flush=True)
    print(f"outputs     : {out_dir}", flush=True)
    print(f"checkpoints : {ckpt_dir}", flush=True)
    print(f"logs        : {log_dir}", flush=True)
    _pool = run.get("prior", {}).get("pool", {}) or {}
    print(f"prior source: {_pool.get('source', 'generate')}", flush=True)

    if args.dry_run:
        print("dry run: config resolved, nothing trained.", flush=True)
        return 0

    from src.train.loop import Trainer  # imported late so --list/--dry-run need no torch

    trainer = Trainer(run, out_dir, device=args.device, ckpt_dir=ckpt_dir, log_dir=log_dir)
    trainer.maybe_resume()
    summary = trainer.train()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"done": run["_run_name"], **summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
