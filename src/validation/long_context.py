"""Bounded GPU capacity checks for Experiment 2's actual TabICLv2 training path.

These are diagnostics, not research training runs. Each process starts from
released weights, consumes the configured synthetic prior, performs one update,
and discards its weights. A corner probe pads only constant zero columns to test
the declared maximum allocation; it never discards generated rows or features.
The representative batch keeps the real 64-task ragged distribution and workers.
No throughput extrapolation from one update is a reliable campaign ETA.
"""
from __future__ import annotations

import argparse
import copy
import math
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from src.utils.config import expand_with_seeds, load

# Endpoints of each upstream row-dependent feature ceiling, plus the shortest
# stage-3 table. These are physical capacity checks, not sampled effect estimates.
CAPACITY_CORNERS = ((400, 100), (10240, 100), (20000, 80), (30000, 60),
                    (40000, 40), (50000, 30), (60000, 20))


@contextmanager
def observe_fp32_attention():
    """Check actual Q/K/V dtypes through forward AND activation recomputation."""
    import torch
    from torch.nn import functional as F

    original = F.scaled_dot_product_attention
    observations = {"calls": 0, "qkv_dtype": "torch.float32"}

    def checked(query, key, value, *args, **kwargs):
        if any(t.dtype != torch.float32 for t in (query, key, value)):
            raise RuntimeError("Long-context attention silently changed from FP32")
        observations["calls"] += 1
        return original(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = checked
    try:
        yield observations
        if not observations["calls"]:
            raise RuntimeError("No SDPA calls observed; the precision check did not exercise attention")
    finally:
        F.scaled_dot_product_attention = original


def diagnostic_config(track: str, *, rows: int | None = None,
                      source: str = "mixed") -> dict:
    """Use the resolved production recipe, with explicit diagnostic overrides."""
    if track not in {"pd", "lgd"}:
        raise ValueError("track must be pd or lgd")
    if source not in {"base", "credit", "mixed"}:
        raise ValueError("Unknown synthetic source")
    if (rows is None) != (source == "mixed"):
        raise ValueError("Mixed probes require the full sampled batch; corners require a source")
    matches = [c for c in expand_with_seeds(load(f"config/Exp2_{track.upper()}.yaml"))
               if c["arm"] == "tabicl" and c["seed"] == 0
               and float(c["prior"]["credit_fraction"]) == .5]
    if len(matches) != 1:
        raise ValueError("Expected exactly one seed-zero mixed TabICL recipe")
    cfg = copy.deepcopy(matches[0])
    if cfg["prior"].get("batch_layout") != "ragged" or cfg["train"]["micro_batch_size"] != 1:
        raise ValueError("Long-context validation requires the production ragged micro-batch recipe")
    if not cfg["train"].get("activation_checkpointing") or not cfg["train"].get("require_efficient_attention"):
        raise ValueError("Long-context validation requires checkpointing and efficient attention")
    if rows is not None:
        widths = dict(CAPACITY_CORNERS)
        if rows not in widths:
            raise ValueError("Unsupported capacity corner")
        cfg["prior"]["n_rows_range"] = [rows, rows]
        cfg["prior"]["n_features_range"] = [widths[rows], widths[rows]]
        cfg["prior"]["credit_fraction"] = float(source == "credit")
        cfg["train"].update(batch_size=1, num_workers=0)
    else:
        # Match production's worker cap and prefetch bound, including DataLoader
        # process teardown; the actual heterogeneous batch is not shrunk.
        cfg["train"]["num_workers"] = min(int(cfg["train"]["num_workers"]),
                                           int(cfg["train"]["max_workers"]))
    cfg["experiment"] = "exp0_capacity"
    cfg["_run_name"] = f"exp0_capacity_{track}_{source}_{rows or 'batch'}"
    # A capacity probe must execute a nonzero update. The production first
    # warm-up step has LR zero, so explicitly test its configured peak LR here.
    cfg["train"]["warmup_proportion"] = 0.0
    cfg["progress"] = {"every_datasets": 0}
    cfg["logging"] = {"console": True, "to_file": False,
                       "log_hardware_every": 0, "log_grad_every": 0,
                       "log_weights_every": 0, "log_prior_every": 0}
    return cfg


def capacity_batch(batch, *, rows: int, width: int):
    """Pad constant columns to the capacity ceiling without losing any data."""
    import torch

    X, y, active, lengths, contexts = batch
    if not isinstance(X, (list, tuple)) or len(X) != 1 or len(y) != 1:
        raise ValueError("A capacity corner requires one ragged synthetic table")
    if int(lengths[0]) != rows or X[0].shape[0] != rows or y[0].shape[0] != rows:
        raise ValueError("The generated table did not retain the requested rows")
    if not 0 < int(contexts[0]) < rows or X[0].shape[1] > width:
        raise ValueError("Invalid context or generated width exceeds capacity corner")
    if int(active[0]) != X[0].shape[1] or int(active[0]) < 1:
        raise ValueError("Invalid active-column metadata")
    padded = X[0].new_zeros((rows, width))
    padded[:, :X[0].shape[1]] = X[0]
    return [padded], y, torch.tensor([width], dtype=active.dtype), lengths, contexts


def run_probe(track: str, out: Path, *, rows: int | None = None,
              source: str = "mixed") -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Long-context capacity validation requires a CUDA allocation")
    from src.prior.dataset import build_loader
    from src.prior.stream import plan_batch
    from src.train.loop import Trainer

    cfg = diagnostic_config(track, rows=rows, source=source)
    train = cfg["train"]
    planned = plan_batch(cfg["seed"], 0, cfg["prior"], train["batch_size"],
                         regression=track == "lgd", credit_fraction=cfg["prior"]["credit_fraction"])
    out.mkdir(parents=True, exist_ok=True)
    loader = build_loader(cfg["prior"], track, train["batch_size"], cfg["seed"],
                          num_workers=train["num_workers"], prefetch_factor=train["prefetch_factor"])
    iterator = None
    generated_at = time.perf_counter()
    try:
        iterator = iter(loader)
        batch = next(iterator)
        generation_seconds = time.perf_counter() - generated_at
        actual_widths = [int(v) for v in batch[2]]
        if rows is not None:
            batch = capacity_batch(batch, rows=rows, width=dict(CAPACITY_CORNERS)[rows])
        if len(batch[0]) != train["batch_size"] or batch[3].tolist() != planned.seq_lens:
            raise ValueError("Diagnostic batch differs from the complete configured task plan")
        if batch[4].tolist() != planned.train_sizes:
            raise ValueError("Diagnostic contexts differ from the configured task plan")
        # Work files are temporary and never enter production checkpoint paths.
        with tempfile.TemporaryDirectory(prefix="capacity-", dir=out) as work:
            with Trainer(cfg, Path(work), device="cuda", manifest_dir=Path(work)) as trainer:
                before = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters()
                          if p.requires_grad}
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with observe_fp32_attention() as precision:
                    stats = trainer.train_step(batch)
                torch.cuda.synchronize()
                update_seconds = time.perf_counter() - started
                peak_bytes = torch.cuda.max_memory_allocated()
                if not math.isfinite(float(stats["loss"])) or trainer.datasets_seen != train["batch_size"]:
                    raise ValueError("Non-finite loss or incomplete task accounting")
                changed = 0
                for name, param in trainer.model.named_parameters():
                    if not torch.isfinite(param).all():
                        raise ValueError(f"Non-finite model parameter after update: {name}")
                    if name in before and not torch.equal(before[name], param.detach().cpu()):
                        changed += 1
                if not changed:
                    raise ValueError("The diagnostic optimizer did not change any model parameter")
                for state in trainer.optimizer.state.values():
                    for name, value in state.items():
                        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                            raise ValueError(f"Non-finite optimizer state: {name}")
                return dict(track=track, source=source, tasks=train["batch_size"],
                            rows=batch[3].tolist(), context_rows=batch[4].tolist(),
                            active_features=actual_widths, capacity_features=batch[2].tolist(),
                            credit_tasks=sum(s.use_credit for s in planned.slots),
                            zero_padding_for_capacity=rows is not None,
                            generation_seconds=round(generation_seconds, 3),
                            update_seconds=round(update_seconds, 3),
                            peak_allocated_gib=round(peak_bytes / 2**30, 3),
                            loss=float(stats["loss"]), updated_parameter_tensors=changed,
                            workers=train["num_workers"], prefetch_factor=train["prefetch_factor"],
                            loaded_weights=trainer.load_report,
                            observed_attention=precision,
                            tf32_matmul_allowed=torch.backends.cuda.matmul.allow_tf32,
                            attention="FP32 SDPA; efficient backends required; upstream FA3 disabled",
                            activation_checkpointing=train["activation_checkpointing"],
                            diagnostic_peak_lr=train["lr"],
                            scope="One diagnostic update, not a performance or campaign-runtime estimate")
    finally:
        # The representative case deliberately exercises production workers. Do
        # not leave prefetched batches/processes alive during the next stage.
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def main(argv=None) -> int:
    from src.validation.checks import FAIL, PASS, Report, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("pd", "lgd"), required=True)
    parser.add_argument("--rows", type=int, choices=[r for r, _ in CAPACITY_CORNERS])
    parser.add_argument("--source", choices=("base", "credit", "mixed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = Report()
    try:
        detail = run_probe(args.track, args.out, rows=args.rows, source=args.source)
        report.add("long_context", f"{args.track}/{args.source}/{args.rows or 'batch'}", PASS, detail)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        report.add("long_context", f"{args.track}/{args.source}/{args.rows or 'batch'}", FAIL,
                   f"{type(exc).__name__}: {exc}")
    write(report, args.out)
    return int(report.failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
