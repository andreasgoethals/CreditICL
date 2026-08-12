"""Fast end-to-end check: does the prior generate, and does the model train?

Run this BEFORE any real submission. It is also the pre-flight step inside the
SLURM scripts, so a broken environment costs seconds instead of a queue wait
plus a dead array.

    python -m src.utils.smoke_test --task lgd --steps 2
    python -m src.utils.smoke_test --task pd  --steps 2 --report

`--report` also prints what the prior actually produced: boundary mass for LGD,
base rate for PD, plus the filter's rejection statistics. That is the quickest
way to see whether the credit path is doing what it claims. It is a smaller
version of scripts/measure_prior.py.

Tables are tiny, so this is safe on a login node and free on the `interactive`
partition. It DOES use the GPU, AMP and a multi-worker DataLoader when a GPU is
present, because those paths are used by the real runs and are covered nowhere
else — otherwise the first GPU step ever taken would be inside the paid array.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import expand_with_seeds, load  # noqa: E402

DEFAULT_CONFIG = {"lgd": "config/Exp1_LGD.yaml", "pd": "config/Exp1_PD.yaml"}


def shrink(cfg: dict, steps: int) -> dict:
    """Make the config tiny so the check is fast but still exercises every path."""
    cfg = copy.deepcopy(cfg)
    # The point of a smoke test is "does the pipeline run on THIS machine", so it must not be
    # blocked by a missing package. Prefer the real architecture and say loudly when falling
    # back, because a pass on the fallback does NOT mean the real model works.
    from src.models.architecture import DEFAULT, is_available

    if not is_available(DEFAULT):
        print(
            f"  NOTE: `{DEFAULT}` is not installed — smoke-testing on the vendored fallback.\n"
            f"        This checks the PIPELINE, not the architecture the experiments use.\n"
            f"        Install it before any real run:  pip install \"tabicl>=2.0\""
        )
        cfg["architecture"] = "nanotabicl"
    prior = cfg.setdefault("prior", {})
    prior["n_rows_range"] = [96, 128]
    prior["n_features_range"] = [4, 8]
    prior["max_features"] = 16
    prior["n_nodes_range"] = [2, 6]
    prior["max_filter_attempts"] = 8
    # Force the credit path so the new code is actually exercised, whatever the
    # config's mixture setting happens to be.
    prior["credit_fraction"] = 1.0

    model = cfg.setdefault("model", {})
    model.update(
        {
            "embed_dim": 32,
            "col_num_blocks": 1,
            "row_num_blocks": 1,
            "icl_num_blocks": 2,
            "col_nhead": 2,
            "row_nhead": 2,
            "icl_nhead": 2,
            "n_cls_rows": 16,
        }
    )

    train = cfg.setdefault("train", {})
    train.update(
        {
            "max_steps": steps,
            "batch_size": 2,
            "micro_batch_size": 2,
            "num_quantiles": 32,
            "log_every": 1,
            "save_temp_every": 0,
            "save_perm_every": 0,
        }
    )
    # Exercise the SAME code paths the real runs use, where the hardware allows.
    # AMP (bfloat16 autocast + GradScaler) and a multi-worker DataLoader are both
    # used by the real configs and are NOT covered anywhere else, so forcing them
    # off here would mean the first GPU step ever taken is inside the paid array.
    import torch

    has_cuda = torch.cuda.is_available()
    train["amp"] = has_cuda
    # Two workers, not the config's 12: enough to exercise the multiprocess
    # DataLoader path (worker seeding, pickling, persistent_workers) without
    # spawning a dozen processes for a three-step check.
    train["num_workers"] = 2
    cfg["seeds"] = [0]
    return cfg


def report_prior(cfg: dict, task: str, n: int = 24) -> dict:
    """Sample a few tasks and report what the target distribution looks like."""
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.utils.target_stats import summarise, target_stats

    gen = TaskGenerator(cfg["prior"], task, PriorRNG(0))
    stats, rates, widths, rows = [], [], [], []
    for _ in range(n):
        t = gen.sample()
        widths.append(t.n_features)
        rows.append(t.n_rows)
        stats.append(target_stats(t.y))
        if task == "pd":
            rates.append(float(t.y.mean()))

    out: dict = {
        "sampled": n,
        "rows_min_max": [min(rows), max(rows)],
        "features_min_max": [min(widths), max(widths)],
        "filter": gen.filter_summary(),
    }
    if task == "pd":
        # NOT target_stats here. For a binary target every value sits on a
        # "boundary", so boundary_mass_mean is always exactly 1.0 and
        # distinct_fraction is 2/n — numbers that look meaningful and are not.
        # The base rate is the quantity that actually matters for PD.
        out["base_rate_mean"] = round(sum(rates) / len(rates), 4)
        out["base_rate_min"] = round(min(rates), 4)
        out["base_rate_max"] = round(max(rates), 4)
    else:
        out["target"] = summarise(stats)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("lgd", "pd"), required=True)
    ap.add_argument("--config", default=None, help="defaults to config/<TASK>.yaml")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--report", action="store_true", help="also summarise the prior's output")
    args = ap.parse_args()

    cfg_path = args.config or DEFAULT_CONFIG[args.task]
    cfg = shrink(expand_with_seeds(load(cfg_path))[0], args.steps)

    print(f"[1/4] config loaded and shrunk: {cfg_path}", flush=True)

    from src.prior.generator import TaskGenerator  # noqa: F401  (import check)
    from src.prior.rng import PriorRNG

    gen = TaskGenerator(cfg["prior"], args.task, PriorRNG(0))
    task = gen.sample()
    print(
        f"[2/4] prior OK: X={tuple(task.X.shape)} y={tuple(task.y.shape)} "
        f"source={task.source} y_range=({float(task.y.min()):.3f}, {float(task.y.max()):.3f})",
        flush=True,
    )
    left_unit = float(task.y.min()) < -1e-6 or float(task.y.max()) > 1 + 1e-6
    unscaled = cfg["prior"]["credit"]["target"].get("target_scaling", "none") == "none"
    if args.task == "lgd" and left_unit and unscaled:
        print("  FAIL: LGD target left [0,1] with target_scaling='none'.", file=sys.stderr)
        return 1

    import torch

    from src.train.loop import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/4] device={device}  amp={cfg['train']['amp']}  workers={cfg['train']['num_workers']}", flush=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = Trainer(cfg, tmp, device=device)
        print(
            f"      model OK: {trainer.freeze_report['trainable_params']:,} trainable / "
            f"{trainer.freeze_report['total_params']:,} total "
            f"(strategy={trainer.freeze_report['strategy']})",
            flush=True,
        )
        summary = trainer.train()
        print(f"[4/4] {args.steps} training steps OK: {json.dumps(summary['freeze'])}", flush=True)

    if args.report:
        print("\nprior report:")
        print(json.dumps(report_prior(cfg, args.task), indent=2))

    print("\nSMOKE TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
