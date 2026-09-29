"""Is our trainer upstream's trainer? Run both on the SAME batches from the SAME weights.

    python scripts/check_equivalence.py --config config/Exp1_PD.yaml --index 0 --steps 20 --device cuda
    python scripts/check_equivalence.py --config config/Exp1_PD.yaml --tiny --steps 5    # local, CPU

Our `Trainer.train_step` and upstream's `Trainer.run_batch` (pinned dump, `src/tabicl/train/_run.py`)
each take the identical batch — our prior's control arm, which is upstream's `GraphSCM` — starting
from identical weights, with upstream's optimizer and scheduler built by upstream's own
`configure_optimizer` / `get_scheduler`. After every step it prints both losses, both learning
rates and the largest parameter difference.

WHAT PASSING MEANS. On CPU in float32 the two must agree to rounding (the same operations in the
same order): any real difference in the update rule — optimizer, schedule, clipping, loss, the
micro-batch split, the trimming of padded columns — shows up as a jump, usually in the first step.
On a GPU with bf16 autocast they agree to bf16 rounding, and a small drift over steps is expected.

Upstream's module imports `wandb` and `transformers` at load time for logging and for schedules we
do not use; neither is installed here, so both are stubbed (a stub raises if it is ever called).
Nothing is installed and nothing touches `tfm-library/`.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_optional_imports() -> None:
    def unavailable(name):
        def fn(*_a, **_kw):
            raise RuntimeError(f"stub: transformers.{name} is not installed and was not expected to be called")
        return fn

    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        for name in ("get_constant_schedule", "get_linear_schedule_with_warmup",
                     "get_cosine_schedule_with_warmup", "get_polynomial_decay_schedule_with_warmup"):
            setattr(tf, name, unavailable(name))
        sys.modules["transformers"] = tf


def _tiny(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    p = cfg["prior"]
    p.update(n_rows_range=[128, 128], n_features_range=[2, 8], max_features=16, max_filter_attempts=4)
    cfg["train"].update(batch_size=8, micro_batch_size=4, max_steps=50)
    cfg.setdefault("model", {}).update(embed_dim=32, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=2,
                                        col_nhead=2, row_nhead=2, icl_nhead=2, col_num_inds=8)
    cfg["progress"] = {"every_datasets": 0}
    cfg.setdefault("logging", {}).update(log_hardware_every=0, log_grad_every=0, to_file=False, console=False)
    return cfg


def upstream_trainer(cfg: dict, model_kwargs: dict, state_dict: dict, device: str, regression: bool):
    """Upstream's Trainer, built without its DDP/wandb/prior/checkpoint setup, around a TabICL
    holding our initial weights. Its optimizer, scheduler and AMP context are upstream's own."""
    _stub_optional_imports()
    from tabicl._model.tabicl import TabICL
    from tabicl.train._run import Trainer as UpstreamTrainer
    from tabicl.train._train_config import build_parser

    t = cfg["train"]
    amp = bool(t.get("amp", True)) and device.startswith("cuda")
    args = [
        "--device", device, "--dtype", "bfloat16" if amp else "float32", "--amp", str(amp),
        "--max_steps", str(t["max_steps"]), "--batch_size", str(t["batch_size"]),
        "--micro_batch_size", str(t["micro_batch_size"]), "--lr", str(t["lr"]),
        "--muon", str(str(t.get("optimizer", "muon")).lower() == "muon"),
        "--beta1", str(t.get("beta1", 0.9)), "--beta2", str(t.get("beta2", 0.95)),
        "--weight_decay", str(t.get("weight_decay", 0.01)), "--use_cautious_wd", "False",
        "--scheduler", str(t.get("scheduler", "cosine_with_restarts")),
        "--warmup_proportion", str(t.get("warmup_proportion", 0.01)),
        "--cosine_num_cycles", str(t.get("cosine_num_cycles", 1)),
        "--cosine_amplitude_decay", str(t.get("cosine_amplitude_decay", 1.0)),
        "--cosine_lr_end", str(t.get("cosine_lr_end", 1e-7)),
        "--gradient_clipping", str(t.get("gradient_clipping", 10.0)),
        "--max_classes", "10",
    ]
    if regression:
        args += ["--regression_method", "quantile", "--num_quantiles", str(t.get("num_quantiles", 999))]
    up = UpstreamTrainer.__new__(UpstreamTrainer)
    up.config = build_parser().parse_args(args)
    up.ddp, up.master_process, up.ddp_world_size, up.ddp_rank = False, True, 1, 0
    up.regression = regression
    up.curr_step = 0
    up._disable_cudnn_sdp = True   # ours pins the same kernels (src/train/loop.py sdpa_context)
    up.model = TabICL(**model_kwargs).to(device)
    up.model.load_state_dict(state_dict)
    up.raw_model = up.model
    up.configure_optimizer()
    up.configure_amp()
    return up


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/Exp1_PD.yaml")
    ap.add_argument("--index", type=int, default=0, help="grid index; its prior is forced to the control")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true", help="shrink model and tables for a quick CPU run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from src.prior.dataset import PriorBatchDataset
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.train.loop import Trainer
    from src.utils.config import expand_with_seeds, load

    cfg = copy.deepcopy(expand_with_seeds(load(args.config))[args.index])
    cfg["prior"]["credit_fraction"] = 0.0          # the control: upstream's GraphSCM
    if args.tiny:
        cfg = _tiny(cfg)
    cfg["progress"] = {"every_datasets": 0}
    cfg.setdefault("logging", {}).update(log_hardware_every=0, log_grad_every=0)
    cfg["seed"] = args.seed
    task = cfg["task"]
    regression = task == "lgd"

    with tempfile.TemporaryDirectory() as tmp:
        ours = Trainer(cfg, Path(tmp) / "o", device=args.device, ckpt_dir=Path(tmp) / "c",
                       log_dir=Path(tmp) / "l", manifest_dir=Path(tmp) / "m")
        kwargs = dict(ours.model.creditcl_model_config)
        up = upstream_trainer(cfg, kwargs, copy.deepcopy(ours.model.state_dict()), args.device, regression)

        ds = PriorBatchDataset(cfg["prior"], task, int(cfg["train"]["batch_size"]), seed=args.seed)
        gen = TaskGenerator(cfg["prior"], task, PriorRNG(args.seed), stream_seed=args.seed)
        key = "pinball" if regression else "ce"
        worst_loss = worst_param = 0.0
        print(f"{'step':>4}  {'our loss':>12}  {'upstream':>12}  {'|dloss|':>9}  {'our lr':>10}  "
              f"{'upstream lr':>11}  {'max |dparam|':>12}")
        for step in range(args.steps):
            batch = ds.batch(gen, step)
            mine = ours.train_step(batch)
            theirs = up.run_batch([t.clone() for t in batch])
            dloss = abs(mine["loss"] - theirs[key])
            dparam = max(float((a.detach() - b.detach()).abs().max())
                         for a, b in zip(ours.model.parameters(), up.model.parameters()))
            worst_loss, worst_param = max(worst_loss, dloss), max(worst_param, dparam)
            print(f"{step:>4}  {mine['loss']:>12.6f}  {theirs[key]:>12.6f}  {dloss:>9.2e}  "
                  f"{ours.scheduler.get_last_lr()[0]:>10.3e}  {up.scheduler.get_last_lr()[0]:>11.3e}  "
                  f"{dparam:>12.2e}")
            if not math.isfinite(mine["loss"]):
                print("non-finite loss; stopping")
                return 1
        ours.close()

    exact = args.device == "cpu" and not (bool(cfg["train"].get("amp")) and args.device.startswith("cuda"))
    tol_loss, tol_param = (1e-4, 1e-4) if exact else (5e-2, 5e-2)
    verdict = "PASS" if worst_loss <= tol_loss and worst_param <= tol_param else "FAIL"
    print(f"\n{verdict}: largest |dloss| {worst_loss:.2e} (tolerance {tol_loss:g}), largest |dparam| "
          f"{worst_param:.2e} (tolerance {tol_param:g}) over {args.steps} steps on {args.device}"
          f"{'' if exact else ' with bf16 autocast (drift expected)'}.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
