"""Synthetic-only CPT mechanism probes; never campaign results or LR selection.

Released weights stay on VSC. Training uses a separate scratch directory, writes
no checkpoints and leaves the production recipes untouched. Short-table TabICL
probes cannot establish the best production learning rate or benchmark accuracy.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

OBJECTIVES = ("native", "crps", "crps_mae", "crps_huber")
TABICL_RATES = (1e-5, 2e-5, 5e-5, 1e-4)


def probe_config(backbone, task, lr, *, seed=0, steps=32, force_fp32=False):
    from src.utils.config import expand_with_seeds, load

    run = next(r for r in expand_with_seeds(load(f"config/Exp2_{task.upper()}.yaml", allow_placeholders=True))
               if r["architecture"] == backbone and r["seed"] == seed
               and r["prior"]["credit_fraction"] == .5 and r["lr_index"] == 0)
    cfg = copy.deepcopy(run)
    cfg.update(experiment="cpt_mechanism_probe", _run_name=f"probe_{backbone}_{task}_{seed}")
    cfg.pop("design", None)
    cfg.pop("design_revision", None)
    # The first 32 updates of the actual 1500-step schedule, not a compressed
    # 32-step cosine cycle. Retain production table count, features and split.
    cfg["train"].update(lr=lr, max_steps=steps, schedule_total_steps=1500,
                        num_workers=0, max_workers=0, log_every=1)
    # Preserve this historical objective comparison's target-space intervention.
    # The new standardization-only production policy has separate Exp0 tests.
    if backbone == "tabpfn3" and task == "lgd":
        cfg["train"]["regression_target_policy"] = "native"
    if backbone == "tabicl":
        cfg["prior"].update(n_rows_range=[1024, 1024], n_rows_sampling="uniform")
    if force_fp32:
        if backbone != "tabpfn3":
            raise ValueError("The precision intervention is for TabPFN only")
        cfg["train"]["amp"] = False
    cfg["progress"] = {"every_datasets": 0, "n_ood": 0}
    cfg["logging"] = dict(level="WARNING", console=False, to_file=False,
                          log_hardware_every=0, log_grad_every=0, log_weights_every=0)
    return cfg


def bar_objective(logits, targets, bar, objective):
    """Compare native MSE with bounded-slope alternatives, in the SAME space.

    Huber delta=1 is fixed in transformed target units; it is a diagnostic
    hypothesis, not an upstream default or a benchmark-tuned choice.
    """
    from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss

    if objective not in OBJECTIVES:
        raise ValueError(objective)
    loss = _compute_regression_loss(
        logits_BQL=logits, targets_BQ=targets, bardist_loss_fn=bar,
        ce_loss_weight=0., crps_loss_weight=1.,
        mse_loss_weight=float(objective == "native"),
        mae_loss_weight=float(objective == "crps_mae"))
    if objective == "crps_huber":
        # Avoid squaring astronomical residuals in the unselected branch.
        absolute = (bar.mean(logits) - targets).abs()
        loss = loss + (0.5 * absolute.clamp(max=1.).square()
                       + (absolute - 1.).clamp(min=0.)).mean()
    return loss


def regression_objective(est, batch, n_estimators, _weights, *, objective):
    """Diagnostic loss replacement, scoped to this process by patch()."""
    from tabpfn.finetuning.finetuned_regressor import _targets_in_estimator_space

    _, members, _ = est.forward(batch.X_query)
    logits = torch.stack(members, dim=2)
    q, b, e, width = logits.shape
    if b != 1 or e != n_estimators or width != batch.znorm_space_bardist.num_bars:
        raise ValueError("Unexpected native ensemble layout")
    targets = torch.stack([_targets_in_estimator_space(batch.y_query[0].double(), c[0])
                           for c in batch.configs])
    if not torch.isfinite(targets).all():
        raise FloatingPointError("Non-finite transformed query")
    dtype = torch.float64 if targets.abs().max() > math.sqrt(
        torch.finfo(torch.float32).max / (4 * max(1, targets.numel()))) else torch.float32
    bar = copy.deepcopy(batch.znorm_space_bardist).to(device=logits.device, dtype=dtype)
    return bar_objective(logits.permute(1, 2, 0, 3).reshape(b * e, q, width).to(dtype),
                         targets.to(device=logits.device, dtype=dtype), bar, objective)


def _write(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _gradient_norm(model):
    return float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad, dtype=torch.float64)
        for p in model.parameters() if p.grad is not None])))


def prepared_case(preparer, cfg, batch_id, slot_id, *, fraction):
    from tabpfn.finetuning.data_util import meta_dataset_collator
    from tabpfn.finetuning.finetuned_regressor import _targets_in_estimator_space

    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG, one_thread
    from src.prior.stream import plan_batch

    prior = copy.deepcopy(cfg["prior"])
    prior["credit_fraction"] = fraction
    seed = int(cfg["seed"])
    slot = plan_batch(seed, batch_id, prior, 16, regression=True, credit_fraction=fraction).slots[slot_id]
    with one_thread():
        task = TaskGenerator(prior, "lgd", PriorRNG(seed), stream_seed=seed).sample_slot(slot)
        item = preparer.prepare(task.X[:, :max(1, task.meta["d"])].numpy().astype(np.float64),
                                task.y.numpy().astype(np.float64), slot.train_size,
                                zlib.crc32(f"{seed}:{batch_id}:{slot_id}".encode()) & 0x7fffffff)
    batch = meta_dataset_collator([item["item"]])
    targets = torch.stack([_targets_in_estimator_space(batch.y_query[0].double(), c[0])
                           for c in batch.configs])
    record = dict(batch=batch_id, slot=slot_id, fraction=fraction,
                  raw_target_abs_max=float(task.y.abs().max()),
                  context_zscore_query_abs_max=float(batch.y_query.abs().max()),
                  transformed_query_abs_max=float(targets.abs().max()),
                  transforms=[type(c[0].target_transform).__name__ for c in batch.configs],
                  finite=bool(torch.isfinite(targets).all()))
    return item, record


def run_tail(directory, report, save):
    """Real production tail plus ordinary generic/credit tasks; identical weights."""
    from functools import partial

    from tabpfn.finetuning._torch_compat import sdpa_kernel_context

    from src.train.tabpfn_trainer import TabPFNTaskPreparer, TabPFNTrainer

    cfg = probe_config("tabpfn3", "lgd", 1e-5)
    trainer = TabPFNTrainer(cfg, directory, device="cuda")
    trainer.maybe_resume()
    preparer = TabPFNTaskPreparer("lgd", trainer.weights_path, 2, 0, target_policy="native")
    report["cases"] = []
    cases = [(685, 15, 0.)] + [(b, 0, f) for f in (0., 1.) for b in (0, 1, 2, 3)]
    try:
        for batch_id, slot_id, fraction in cases:
            item, case = prepared_case(preparer, cfg, batch_id, slot_id, fraction=fraction)
            case["objectives"] = []
            # FP32 forward isolates the mathematical objective from AMP overflow.
            for objective in OBJECTIVES:
                torch.manual_seed(442)
                torch.cuda.manual_seed_all(442)
                trainer.optimizer.zero_grad(set_to_none=True)
                replacement = partial(regression_objective, objective=objective)
                with trainer._full_precision(), sdpa_kernel_context(), patch(
                        "src.train.tabpfn_trainer.regression_loss", replacement):
                    loss = trainer._task_loss(item)
                    divisor = max(1., math.sqrt(abs(float(loss.detach()))) / 1e24)
                    (loss / divisor).backward()
                norm = _gradient_norm(trainer.model) * divisor
                case["objectives"].append(dict(objective=objective, loss=float(loss.detach()),
                    gradient_norm=norm, clip_coefficient=min(1., 1. / (norm + 1e-6))))
                trainer.optimizer.zero_grad(set_to_none=True)
            report["cases"].append(case)
            save()
            print(json.dumps(case), flush=True)
    finally:
        trainer.close()


def _validation_batches(cfg, backbone, preparer):
    from src.prior.dataset import PriorBatchDataset
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG

    batches = []
    for fraction in (0., 1.):
        prior = copy.deepcopy(cfg["prior"])
        prior["credit_fraction"] = fraction
        dataset = PriorBatchDataset(prior, cfg["task"], 8, seed=9137)
        generator = TaskGenerator(prior, cfg["task"], PriorRNG(9137), stream_seed=9137)
        batch = dataset.batch(generator, 0)
        batches.append((fraction, preparer(batch, 99137) if backbone == "tabpfn3" else batch))
    return batches


def _measure(trainer, batches, backbone):
    from src.train.loop import sdpa_context

    results = []
    # All variants evaluated with the SAME native objective, plus separate
    # robust CRPS for TabPFN. Held-out SYNTHETIC tasks, never credit test scores.
    with torch.no_grad():
        for fraction, batch in batches:
            losses = []
            if backbone == "tabpfn3":
                from functools import partial
                for objective in ("native", "crps"):
                    with trainer._full_precision(), patch("src.train.tabpfn_trainer.regression_loss",
                                                          partial(regression_objective, objective=objective)):
                        losses.append(float(np.mean([float(trainer._task_loss(p)) for p in batch])))
            else:
                X, y, d, n, ts = batch
                with sdpa_context(require_efficient=trainer.require_efficient_attention):
                    for i in range(len(X)):
                        table = X[i][:int(n[i]), :max(1, int(d[i]))][None].to(trainer.device)
                        target = y[i][:int(n[i])][None].to(trainer.device)
                        losses.append(float(trainer._loss_for(table, target, int(ts[i]))[0]))
                losses = [float(np.mean(losses))]
            results.append(dict(fraction=fraction, native_loss=losses[0],
                                **({"crps": losses[1]} if backbone == "tabpfn3" else {})))
    return results


def run_trajectory(directory, report, save, backbone, task, steps):
    from functools import partial

    from src.prior.dataset import PriorBatchDataset
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.train.loop import Trainer
    from src.train.tabpfn_trainer import TabPFNTaskPreparer, TabPFNTrainer

    variants = [(rate, "native") for rate in TABICL_RATES] if backbone == "tabicl" else [
        (1e-5, objective) for objective in OBJECTIVES]
    report["variants"] = []
    for rate, objective in variants:
        cfg = probe_config(backbone, task, rate, steps=steps)
        cls = Trainer if backbone == "tabicl" else TabPFNTrainer
        trainer = cls(cfg, directory / f"{rate}_{objective}", device="cuda")
        trainer.maybe_resume()
        initial = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters()}
        preparer = (TabPFNTaskPreparer(task, trainer.weights_path, 2, 0, target_policy="native")
                    if backbone == "tabpfn3" else None)
        validation = _validation_batches(cfg, backbone, preparer)
        dataset = PriorBatchDataset(cfg["prior"], task, cfg["train"]["batch_size"], seed=0)
        generator = TaskGenerator(cfg["prior"], task, PriorRNG(0), stream_seed=0)
        result = dict(lr=rate, objective=objective, seed=0, steps=[],
                      recipe={key: cfg["train"].get(key) for key in (
                          "optimizer", "batch_size", "warmup_proportion", "schedule_total_steps",
                          "weight_decay", "gradient_clipping", "amp")},
                      before=_measure(trainer, validation, backbone))
        report["variants"].append(result)
        save()
        started = time.monotonic()
        try:
            for step in range(steps):
                batch = dataset.batch(generator, step)
                if backbone == "tabpfn3":
                    batch = preparer(batch, step)
                    with patch("src.train.tabpfn_trainer.regression_loss",
                               partial(regression_objective, objective=objective)):
                        stats = trainer.train_step(batch)
                else:
                    stats = trainer.train_step(batch)
                trainer.step += 1
                result["steps"].append(dict(update=step + 1, lr=trainer.scheduler.get_last_lr()[0], **stats))
                save()
                print(f"{backbone}/{task} {rate} {objective}: {step + 1}/{steps} loss={stats['loss']:.6g}", flush=True)
            result["after"] = _measure(trainer, validation, backbone)
            result["elapsed_seconds"] = time.monotonic() - started
            with torch.no_grad():
                difference = sum(float((p.detach().cpu().double() - initial[n].double()).square().sum())
                                 for n, p in trainer.model.named_parameters())
                scale = sum(float(p.double().square().sum()) for p in initial.values())
            result["relative_parameter_drift"] = math.sqrt(difference / scale)
            save()
        finally:
            trainer.close()
        del trainer, initial, validation, preparer, batch
        gc.collect()
        torch.cuda.empty_cache()


def run_stress_update(directory, report, save, *, force_fp32=False):
    """One real outlier-containing batch at peak LR, with fresh optimizer state.

    This is a controlled intervention, not replay of production optimizer state.
    It checks the complete AMP/retry/clipping/update path and held-out synthetic
    losses. No claim about long-run retention follows from a single update.
    """
    from functools import partial

    from src.prior.dataset import PriorBatchDataset
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.train.tabpfn_trainer import TabPFNTaskPreparer, TabPFNTrainer

    report["variants"] = []
    report["intervention"] = "production generic batch 685, fresh AdamW, one update at 1e-5; no warm-up"
    for objective in OBJECTIVES:
        cfg = probe_config("tabpfn3", "lgd", 1e-5, steps=1, force_fp32=force_fp32)
        cfg["train"]["warmup_proportion"] = 0.
        cfg["prior"]["credit_fraction"] = 0.
        trainer = TabPFNTrainer(cfg, directory / objective, device="cuda")
        trainer.maybe_resume()
        preparer = TabPFNTaskPreparer("lgd", trainer.weights_path, 2, 0, target_policy="native")
        result = dict(objective=objective, lr=trainer.optimizer.param_groups[0]["lr"], amp=trainer.amp)
        report["variants"].append(result)
        try:
            validation = _validation_batches(cfg, "tabpfn3", preparer)
            result["before"] = _measure(trainer, validation, "tabpfn3")
            dataset = PriorBatchDataset(cfg["prior"], "lgd", 16, seed=0)
            generator = TaskGenerator(cfg["prior"], "lgd", PriorRNG(0), stream_seed=0)
            prepared = preparer(dataset.batch(generator, 685), 685)
            save()
            with patch("src.train.tabpfn_trainer.regression_loss",
                       partial(regression_objective, objective=objective)):
                result["update"] = trainer.train_step(prepared)
            result["after"] = _measure(trainer, validation, "tabpfn3")
            save()
            print(json.dumps(result), flush=True)
        finally:
            trainer.close()
        del trainer, preparer, prepared, validation
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tail", "trajectory", "stress"), required=True)
    parser.add_argument("--backbone", choices=("tabicl", "tabpfn3"), default="tabpfn3")
    parser.add_argument("--task", choices=("pd", "lgd"), default="lgd")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--full-precision", action="store_true",
                        help="Stress-mode precision control: disable FP16 and GradScaler for every objective")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 128:
        parser.error("Diagnostic budget must be between 1 and 128 updates")
    if args.mode in {"tail", "stress"} and args.backbone != "tabpfn3":
        parser.error("Target-tail interventions require the TabPFN regression backbone")
    if args.full_precision and args.mode != "stress":
        parser.error("The full-precision intervention is supported only in stress mode")
    if (args.mode in {"tail", "stress"} or args.backbone == "tabpfn3") and args.task != "lgd":
        parser.error("TabPFN objective probes require regression")
    if os.environ.get("PYTHONHASHSEED") != "0" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run in Slurm with PYTHONHASHSEED=0; released weights stay on VSC")
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required")
    if args.out.exists():
        raise FileExistsError("Preserve the preceding diagnostic; choose a fresh output path")
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    from src.utils.preflight import source_fingerprint, training_fingerprint

    report = dict(status="RUNNING", mode=args.mode, backbone=args.backbone, task=args.task,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source=source_fingerprint(), training=training_fingerprint(),
                  runtime_sha256=os.environ.get("CREDITICL_RUNTIME_SHA256"),
                  job=os.environ["SLURM_JOB_ID"], torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  scope="synthetic mechanism diagnostic, not benchmark evidence or hyperparameter selection",
                  rows=1024, updates=args.steps if args.mode == "trajectory" else int(args.mode == "stress"),
                  force_full_precision=args.full_precision,
                  unchanged_production_recipes=True)
    def save():
        _write(args.out, report)

    save()
    try:
        with tempfile.TemporaryDirectory(prefix="crediticl-cpt-probe-") as directory:
            if args.mode == "tail":
                run_tail(Path(directory), report, save)
            elif args.mode == "stress":
                run_stress_update(Path(directory), report, save, force_fp32=args.full_precision)
            else:
                run_trajectory(Path(directory), report, save, args.backbone, args.task, args.steps)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
