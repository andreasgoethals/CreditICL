"""Pretraining loop for TabICLv2 on a configured prior.

Written here because NanoTabICL ships **no pretraining code** (its README defers
to nanoTabPFN) and nanoTabPFN is classification-only, so it cannot host the LGD
regression arm. Structure follows TabICL's `train/_run.py`: micro-batching for
gradient accumulation, AMP with a GradScaler, gradient clipping after unscaling,
and temp/permanent checkpoint tiers.

The losses are transcribed exactly from `Trainer.run_micro_batch`:

* regression — pinball over `num_quantiles` levels at
  ``linspace(0, 1, num_quantiles + 2)[1:-1]``, which is the grid inference and
  `QuantileDistribution` assume. Getting these levels wrong would silently
  mis-scale the loss.
* classification — plain cross-entropy.

A note on the quantile head and boundary atoms: a 999-quantile grid represents a
point mass *exactly* (an atom at 0 with mass 0.11 is just levels 0.001..0.11 all
mapping to 0). Representability is therefore **not** the bottleneck for LGD; the
open questions are point-prediction extraction from a bimodal predictive and
support constraints at decode time. See docs/EXPERIMENTAL_DESIGN.md §1.5.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..prior.dataset import build_loader
from ..utils.logging_setup import close_logging, log_environment, log_section, setup_logging
from . import distributed as dist
from .adapt import STRATEGIES, apply_freezing, load_pretrained, set_training_mode
from .checkpoint import latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint
from .optim import build_optimizer, build_scheduler
from .progress import ProgressConfig, ProgressTracker
from .telemetry import Telemetry


def _slurm_seconds_left() -> float:
    """Remaining job walltime in seconds, or 0.0 when not under SLURM.

    Uses `squeue` rather than parsing `$SBATCH_TIMELIMIT`, because the time *left* is
    what matters for the overrun warning and only the scheduler knows it. Any failure
    returns 0.0 and simply disables the warning — a diagnostic must never be able to
    kill a training run.
    """
    import os
    import subprocess

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return 0.0
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%L"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — no diagnostic is worth a crash
        return 0.0
    if not out or "-" in out and out.count(":") == 0:
        return 0.0
    days, _, clock = out.partition("-")
    if not clock:
        clock, days = days, "0"
    parts = [int(x) for x in clock.split(":") if x.isdigit()]
    if not parts:
        return 0.0
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    try:
        d = int(days)
    except ValueError:
        d = 0
    return float(((d * 24 + h) * 60 + m) * 60 + sec)


def _hms(seconds: float) -> str:
    """Human duration. `3h 04m` beats `11072.4 s` when you are watching a log."""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def quantile_levels(num_quantiles: int, device=None, dtype=None) -> torch.Tensor:
    """The exact grid TabICL trains and predicts on."""
    return torch.linspace(0.0, 1.0, num_quantiles + 2, device=device, dtype=dtype)[1:-1]


def enforce_monotonic_quantiles(quantiles: Any, method: str = "sort") -> Any:
    """Fix quantile crossing along the last axis. Accepts a Tensor or an ndarray.

    A quantile head predicts each level independently, so nothing stops it from returning
    q_0.4 > q_0.6. Upstream fixes this — `enforce_monotonicity(quantiles, method="sort")`,
    applied inside `QuantileDistribution` before the quantiles are used for anything — and we
    did not, which is why it is here.

    It matters more than it looks. Every quantity we read off the predictive assumes the grid
    is sorted: the "median" is column Q//2 and is only the median if the row is ordered;
    coverage counts how many truths fall between two columns; PIT and CRPS integrate across
    them. On a crossed row all four are quietly wrong, and none of them looks wrong.

    `sort` is upstream's default. `cummax` is offered because it is the other cheap option
    and upstream names it, but it drags the whole row up to its running maximum and so
    distorts the distribution; prefer `sort` unless there is a reason.

    NOT applied to the training loss: pinball is evaluated per level independently, exactly as
    upstream trains it, and sorting there would change the gradient.
    """
    if method not in ("sort", "cummax"):
        raise ValueError(f"unknown method {method!r}; use 'sort' or 'cummax'")
    if isinstance(quantiles, torch.Tensor):
        if method == "cummax":
            return torch.cummax(quantiles, dim=-1).values
        return torch.sort(quantiles, dim=-1).values
    import numpy as _np

    arr = _np.asarray(quantiles)
    if method == "cummax":
        return _np.maximum.accumulate(arr, axis=-1)
    return _np.sort(arr, axis=-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, num_quantiles: int) -> torch.Tensor:
    """pred: (B, n_test, Q); target: (B, n_test)."""
    alphas = quantile_levels(num_quantiles, device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    errors = target.unsqueeze(-1) - pred
    return torch.maximum(alphas * errors, (alphas - 1) * errors).mean()


class Trainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        out_dir: str | Path,
        device: str | None = None,
        ckpt_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.task = cfg["task"]
        self.regression = self.task == "lgd"
        self.out_dir = Path(out_dir)
        # Checkpoints are the big artefact and belong on project staging; metrics
        # and logs are small and belong on $VSC_DATA. See src/utils/paths.py.
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else self.out_dir / "checkpoints"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        lcfg = cfg.get("logging", {})
        self.log, self.metrics, self.log_path = setup_logging(
            cfg.get("_run_name", cfg.get("experiment", "run")),
            log_dir or (self.out_dir / "logs"),
            level=str(lcfg.get("level", "INFO")),
            console=bool(lcfg.get("console", True)),
            # None => write files on the cluster, not locally.
            to_file=lcfg.get("to_file"),
        )
        self.log_prior_every = int(lcfg.get("log_prior_every", 0))
        self.log_prior_samples = int(lcfg.get("log_prior_samples", 16))

        tcfg = cfg.get("train", {})
        self.max_steps = int(tcfg.get("max_steps", 10_000))
        self.batch_size = int(tcfg.get("batch_size", 4))
        self.micro_batch_size = int(tcfg.get("micro_batch_size", self.batch_size))
        self.grad_clip = float(tcfg.get("gradient_clipping", 1.0))
        self.num_quantiles = int(tcfg.get("num_quantiles", 999))
        self.log_every = int(tcfg.get("log_every", 50))
        # Telemetry cadences live under `logging:`, not `train:`, because they change what gets
        # RECORDED and never what gets computed — switching them off must not alter a result.
        _lcfg = cfg.get("logging", {})
        self.log_hardware_every = int(_lcfg.get("log_hardware_every", 0) or 0)
        self.log_grad_every = int(_lcfg.get("log_grad_every", 0) or 0)
        self.save_temp_every = int(tcfg.get("save_temp_every", 500))
        self.save_perm_every = int(tcfg.get("save_perm_every", 5_000))
        self.max_temp_checkpoints = int(tcfg.get("max_temp_checkpoints", 2))
        self.num_workers = int(tcfg.get("num_workers", 0))
        self.seed = int(cfg.get("seed", 0))
        # Seconds of job walltime left to warn against. Read from SLURM when present so
        # the number is real rather than a guess; 0 disables the warning.
        self._walltime_warning_s = _slurm_seconds_left()

        # Multi-GPU: torchrun sets RANK/WORLD_SIZE. Absent, this is a no-op and the
        # single-GPU path is byte-for-byte what it was.
        self.dist = dist.detect()
        self.device = device or dist.setup(self.dist)
        # Each rank must draw DIFFERENT datasets, or an N-GPU run silently sees the
        # same batch N times at 1/N the real diversity.
        self.prior_seed = dist.rank_seed(self.seed, self.dist)
        # The batch is SPLIT across ranks so the effective batch — and therefore the
        # compute budget — matches the config regardless of GPU count.
        self.local_batch_size = dist.local_batch_size(self.batch_size, self.dist)
        torch.manual_seed(self.seed)

        # The header goes first, before anything that can log or fail. Otherwise a
        # crash during model construction leaves a log with no indication of which
        # run or machine produced it.
        log_section(self.log, f"CreditICL — {self.task.upper()} — {cfg.get('_run_name', '?')}")
        log_environment(self.log, {"device": self.device, "task": self.task, "seed": self.seed,
                                   **self.dist.describe()})
        grid_levers = cfg.get("_grid", {}).get("assignments", {})
        self.log.info("grid levers: %s", json.dumps(grid_levers, default=str))
        actual_cf = cfg["prior"].get("credit_fraction")
        grid_cf = grid_levers.get("prior.credit_fraction")
        note = ""
        if grid_cf is not None and grid_cf != actual_cf:
            # The smoke test overrides this after the grid is expanded. Say so,
            # rather than printing two different numbers and looking broken.
            note = f"  (OVERRIDDEN after grid expansion; grid said {grid_cf})"
        self.log.info(
            "credit_fraction IN USE: %s%s  — share of datasets from OUR prior, rest are original",
            actual_cf, note,
        )
        # Which prior source is in use is the first thing you want from a log when a
        # run looks wrong: "pool" and "generate" produce different task streams.
        pool_cfg = cfg["prior"].get("pool", {}) or {}
        source = pool_cfg.get("source", "generate")
        if source == "pool":
            from ..prior.pool import pool_status

            for label, variant in (
                ("original", pool_cfg.get("original_variant", "original")),
                ("credit", pool_cfg.get("credit_variant", "credit_v1")),
            ):
                st = pool_status(self.task, variant)
                self.log.info(
                    "prior source: POOL  %-8s %-12s %s datasets, %s shards, complete=%s",
                    label, variant, st["n_datasets"], st["shards"], st.get("complete"),
                )
        else:
            self.log.info("prior source: GENERATE (datasets built live on the fly)")
        self.log.info("outputs     -> %s", self.out_dir)
        self.log.info("checkpoints -> %s", self.ckpt_dir)
        self.log.info("log file    -> %s", self.log_path)
        self.log.info("metrics     -> %s", self.metrics.path)

        self.model = self._build_model()

        # --- adaptation strategy: scratch / full / icl_only / head_only --------
        icfg = cfg.get("init", {})
        self.strategy = str(icfg.get("strategy", "scratch"))
        if self.strategy not in STRATEGIES:
            raise ValueError(f"init.strategy={self.strategy!r} must be one of {STRATEGIES}")
        if self.strategy != "scratch":
            ckpt = icfg.get("pretrained_path")
            if not ckpt:
                raise ValueError(f"init.strategy={self.strategy!r} needs init.pretrained_path")
            self.load_report = load_pretrained(self.model, ckpt)
            self.log.info("loaded pretrained weights: %s", json.dumps(self.load_report))
        else:
            self.load_report = {"strategy": "scratch"}

        self.freeze_report = apply_freezing(self.model, self.strategy)
        self.log.info(
            "adaptation strategy=%s  trainable=%s / %s params (%.1f%%)",
            self.freeze_report["strategy"],
            f"{self.freeze_report['trainable_params']:,}",
            f"{self.freeze_report['total_params']:,}",
            100 * self.freeze_report["trainable_fraction"],
        )

        self.model = self.model.to(self.device)
        # build_optimizer BEFORE wrapping: it splits parameters by shape for Muon, and
        # DDP's `module.` prefix would not change shapes but the unwrapped model is the
        # clearer thing to hand it.
        self.optimizer = build_optimizer(self.model, tcfg)
        self.model = dist.wrap_model(self.model, self.dist)
        if self.dist.enabled:
            self.log.info(
                "DISTRIBUTED: world_size=%d  batch %d -> %d per rank  (effective batch unchanged)",
                self.dist.world_size, self.batch_size, self.local_batch_size,
            )
        self.scheduler = build_scheduler(self.optimizer, tcfg, self.max_steps)

        self.amp = bool(tcfg.get("amp", True)) and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.amp else nullcontext()
        )

        self.step = 0
        self.datasets_seen = 0
        # Initialised here as well as in `train`, so `train_step` can be called on its own
        # from a test or a benchmark without the loop having set it up.
        self._phase: dict[str, float] = defaultdict(float)
        self.resumed_at: int | None = None

        # Learning curve on REAL data. Only rank 0 measures and writes, so a multi-GPU
        # run produces one curve rather than N interleaved ones.
        pcfg = cfg.get("progress", {}) or {}
        from ..utils.paths import manifests_dir, on_vsc

        # Same tier rule as the logs: on the cluster the manifest joins the other small
        # durable output under $VSC_DATA/output/; locally it stays inside the run's own
        # directory, so a test with an explicit out_dir is self-contained.
        manifest_dir = manifests_dir() if on_vsc() else Path(self.out_dir) / "manifests"

        self.progress = ProgressTracker(
            ProgressConfig(
                every_datasets=int(pcfg.get("every_datasets", 0)),
                datasets=list(pcfg.get("datasets", []) or []),
                n_datasets=int(pcfg.get("n_datasets", 4)),
                n_ood=int(pcfg.get("n_ood", 4)),
                context_rows=int(pcfg.get("context_rows", 512)),
                max_test_rows=int(pcfg.get("max_test_rows", 2000)),
                seed=self.seed,
            ),
            self.task,
            cfg.get("_run_name", "run"),
            manifest_dir,
        ) if self.dist.is_main else None

        # Rank 0 only: every rank sits on the same node's GPUs, so eight ranks sampling
        # `nvidia-smi` would write eight near-identical rows and multiply the subprocess cost
        # by eight for no extra information.
        self.telemetry = Telemetry(
            cfg.get("_run_name", "run"),
            manifest_dir,
            hardware_every=self.log_hardware_every if self.dist.is_main else 0,
            grad_every=self.log_grad_every if self.dist.is_main else 0,
        )
        # Complements `log_environment()` above rather than repeating it: that one records the
        # run's identity (task, seed, rank, world size), this one records the MACHINE — GPU
        # model and capability, library versions, and the SLURM job/array ids. "Which GPU was
        # this on?" decides whether two runs are comparable at all, and it cannot be
        # reconstructed from the numbers afterwards.
        self.log.info("hardware: %s", json.dumps(self.telemetry.environment))
        if self.progress is not None and self.progress.enabled:
            self.log.info(
                "progress curve -> %s  (every %s datasets)",
                self.progress.path, f"{self.progress.cfg.every_datasets:,}",
            )

        self.log.info(
            "budget: %d steps x %d datasets/step = %d datasets  |  amp=%s  workers=%d",
            self.max_steps,
            self.batch_size,
            self.max_steps * self.batch_size,
            self.amp,
            self.num_workers,
        )

    # -- setup ---------------------------------------------------------------
    def _build_model(self) -> Any:
        """The model, from `architecture:` — TabICLv2's own in every real config.

        `model:` is NOT in any experiment config: the architecture is fixed and never swept,
        because the project's claim is about the prior. The block is honoured here only so a
        test can shrink the network to something that trains in a second.
        """
        from ..models.architecture import DEFAULT, build_model

        mcfg = dict(self.cfg.get("model", {}))
        architecture = self.cfg.get("architecture", DEFAULT)
        if self.regression:
            mcfg.setdefault("num_quantiles", self.num_quantiles)
        # `max_classes` IS NOT SET FROM THE PRIOR. It is part of the architecture, and the
        # architecture is TabICLv2's, unchanged — every one of upstream's classifier stage
        # scripts passes `--max_classes 10`. This line used to force it to `prior.n_classes`
        # (2), which made a 27,538,938-parameter model where upstream's is 27,552,258: a
        # different network, so "same architecture as TabICLv2" stopped being true and Exp3
        # could not warm-start from the released checkpoint (4 head tensors mismatched).
        # A binary task simply uses the first two of the ten logits — see `_loss_for`.
        return build_model(self.task, architecture=architecture, **mcfg)

    def maybe_resume(self) -> None:
        path = latest_checkpoint(self.ckpt_dir)
        if path is None:
            self.log.info("no checkpoint in %s — starting at step 0", self.ckpt_dir)
            return
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.step = int(payload.get("step", 0))
        self.datasets_seen = int(payload.get("extra", {}).get("datasets_seen", 0))
        self.resumed_at = self.step

        # A STALE CHECKPOINT LOOKS EXACTLY LIKE A FINISHED RUN. If the checkpoint is already
        # at (or past) `max_steps`, the training loop's condition is false immediately: it
        # writes "finished", reports `elapsed_s: 0.0`, exits 0, and the evaluation then scores
        # the OLD weights. That happened on 17-08-2026 to all four arms — `clean_run` protects
        # `checkpoints/`, which was right while our checkpoints fell back to `$VSC_DATA`, and
        # wrong the moment the staging permission fix let them land in `checkpoints/` for real.
        if self.step >= self.max_steps:
            self.log.warning(
                "=" * 78
                + "\n  THIS RUN WILL TRAIN NOTHING. The checkpoint in %s is already at step "
                  "%d and max_steps is %d, so the loop exits immediately and the evaluation "
                  "scores the OLD weights.\n"
                  "  If you meant to start fresh, delete that directory first:\n"
                  "      python -m src.utils.clean_run --clean --checkpoints\n"
                  "  If you meant to extend the run, raise train.max_steps.\n"
                + "=" * 78,
                self.ckpt_dir, self.step, self.max_steps,
            )
        self.log.warning(
            "RESUMED from %s at step %d. DataLoader worker RNGs are re-seeded on resume, so "
            "the dataset stream differs from an uninterrupted run; step and dataset counts "
            "are preserved exactly.",
            path.name, self.step,
        )

        # Guard against silently changing the experiment on resume. `LambdaLR`
        # does not persist its lambda, so it is rebuilt from the CURRENT config:
        # resuming with a different max_steps therefore reshapes the whole LR
        # schedule mid-run, and resuming with a different prior would mean the
        # checkpoint and the data no longer match. Both are invisible in the loss
        # curve, so shout about them.
        old_cfg = payload.get("config") or {}
        old_steps = old_cfg.get("train", {}).get("max_steps")
        if old_steps is not None and int(old_steps) != self.max_steps:
            self.log.warning(
                "max_steps CHANGED on resume: checkpoint had %s, this run has %d. The LR "
                "schedule is rebuilt from the current value, so its shape differs from an "
                "uninterrupted run. Fine when deliberately extending a run; a mistake if you "
                "meant to continue the same one.",
                old_steps, self.max_steps,
            )
        old_prior = old_cfg.get("prior", {})
        for key in ("credit_fraction",):
            if key in old_prior and old_prior[key] != self.cfg["prior"].get(key):
                self.log.error(
                    "prior.%s CHANGED on resume: checkpoint was trained with %s, this run says "
                    "%s. These weights and this data do not belong together — the arm is no "
                    "longer the arm. Use a fresh checkpoint directory.",
                    key, old_prior[key], self.cfg["prior"].get(key),
                )

    # -- one optimisation step ----------------------------------------------
    def _loss_for(self, X: torch.Tensor, y: torch.Tensor, train_size: int) -> tuple[torch.Tensor, dict[str, float]]:
        y_train, y_test = y[:, :train_size], y[:, train_size:]
        with self.amp_ctx:
            pred = self.model(X, y_train)
            if self.regression:
                loss = pinball_loss(pred, y_test, self.num_quantiles)
                extra = {}
            else:
                # Upstream's `_compute_batch_loss` does exactly this:
                #     n_classes = int(batch.y_train.max().item()) + 1
                #     logits_used = logits[..., :n_classes].reshape(-1, n_classes)
                #
                # MEASURED, so nobody has to guess: with `tabicl` this slice is a NO-OP.
                # `TabICL.forward` already returns `n_classes` columns, taken from `y_train`
                # — a 10-wide head with binary y returns (B, test, 2), and with 5 classes it
                # returns (B, test, 5). The slice is kept because upstream keeps it and it
                # costs nothing: it is the only thing standing between us and a silent
                # mis-shaped loss if that convention ever changes.
                n_classes = max(2, int(y_train.max().item()) + 1)
                flat = pred[..., :n_classes].flatten(end_dim=-2)
                true = y_test.long().flatten()
                loss = F.cross_entropy(flat, true)
                with torch.no_grad():
                    extra = {"accuracy": float((flat.argmax(dim=1) == true).float().mean())}
        return loss, extra

    def train_step(self, batch) -> dict[str, float]:
        # WHERE DOES THE TIME GO? Three explanations for the B200 running training 16x
        # slower than a benchmark of the same model, prior and loader have now been proposed
        # and two measured wrong — the card, then the optimiser. This splits the step so the
        # next run answers it instead of a fourth hypothesis.
        #
        # No `cuda.synchronize()` anywhere: it would cost more than it measures. Attribution
        # is still sound because `float(loss.detach())` below is itself a sync point, so GPU
        # work lands in `fwd_bwd` rather than leaking into the next phase. `data` is the one
        # to read first — if it dominates, the GPU is waiting for the prior.
        t_step = time.perf_counter()
        X, y, train_size = batch
        # Not model.train(): that is recursive and would switch dropout back on
        # inside frozen blocks. Mirrors TabICL's `_set_training_mode`.
        set_training_mode(self.model, True, self.strategy)
        self.optimizer.zero_grad(set_to_none=True)

        n_micro = math.ceil(X.shape[0] / self.micro_batch_size)
        totals: dict[str, float] = {"loss": 0.0}
        oom = 0

        for i in range(n_micro):
            sl = slice(i * self.micro_batch_size, (i + 1) * self.micro_batch_size)
            Xi = X[sl].to(self.device, non_blocking=True)
            yi = y[sl].to(self.device, non_blocking=True)
            if Xi.shape[0] == 0:
                continue
            try:
                loss, extra = self._loss_for(Xi, yi, train_size)
                self.scaler.scale(loss / n_micro).backward()
                # detach() before float(): torch warns that converting a tensor
                # still attached to the graph can behave unexpectedly, and we only
                # want the number for logging.
                totals["loss"] += float(loss.detach()) / n_micro
                for k, v in extra.items():
                    totals[k] = totals.get(k, 0.0) + v / n_micro
            except torch.cuda.OutOfMemoryError:
                self.log.warning("OOM in micro-batch %d/%d at step %d — skipping", i + 1, n_micro, self.step)
                torch.cuda.empty_cache()
                oom += 1
                continue

        self._phase["fwd_bwd"] += time.perf_counter() - t_step
        t_tail = time.perf_counter()

        if n_micro and oom / n_micro > 0.1:
            raise RuntimeError(
                f"{oom}/{n_micro} micro-batches OOMed at step {self.step}. "
                "Reduce micro_batch_size, n_rows_range or max_features."
            )

        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            # Trainable parameters only: frozen ones have no grad, and including
            # them would change the computed norm and so the effective clip.
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.grad_clip
            )

        # HERE and nowhere else: gradients exist, are unscaled (so the numbers are the real
        # ones rather than AMP's scaled copies), and have not been zeroed yet. One line later
        # they are gone and every norm would read 0.0 — which looks like a dead model instead
        # of a mistimed probe.
        t_telem = time.perf_counter()
        if self.telemetry.due_grad(self.step + 1):
            row = self.telemetry.sample_grads(self.model)
            row.update({"step": self.step + 1, "datasets_seen": self.datasets_seen, "kind": "grad"})
            self.telemetry.record(row)
            self.log.info(
                "grads step %6d  %s",
                self.step + 1,
                "  ".join(
                    f"{k.replace('gw_ratio_', '')}={row[k]:.2e}"
                    for k in sorted(row)
                    if k.startswith("gw_ratio_")
                ),
            )

        # Telemetry is measured separately because `nvidia-smi` is a subprocess, and on a node
        # with 24 GPUs it enumerates all of them — a plausible reason a B200 node would be
        # slower here than a 2-GPU one, and cheap to rule in or out.
        t_opt = time.perf_counter()
        self._phase["telemetry"] += t_opt - t_telem

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self._phase["optimizer"] += time.perf_counter() - t_opt
        self._phase["step_tail"] += (t_telem - t_tail)

        self.datasets_seen += int(X.shape[0])
        return totals

    # -- main ----------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        loader = build_loader(
            self.cfg["prior"],
            self.task,
            # Per-rank batch and per-rank seed: the batch is split so the effective
            # budget is unchanged, and the seeds differ so ranks draw distinct data.
            batch_size=self.local_batch_size,
            seed=self.prior_seed,
            num_workers=self.num_workers,
        )
        it = iter(loader)
        started = time.time()
        window: dict[str, float] = {}
        window_n = 0
        # Wall-clock per phase, accumulated since the last log line and reported as a share
        # of it. Reset each window so the numbers describe the recent past, not the average
        # since step 0 — a stall that starts at step 800 must be visible at step 900.
        self._phase = defaultdict(float)
        phase_since = time.perf_counter()

        while self.step < self.max_steps:
            # THE DATA WAIT. If this dominates, the GPU is idle waiting for the prior and no
            # faster card will help; if it is small, the time is inside the step.
            t_data = time.perf_counter()
            batch = next(it)
            self._phase["data"] += time.perf_counter() - t_data
            stats = self.train_step(batch)
            self.step += 1
            for k, v in stats.items():
                window[k] = window.get(k, 0.0) + v
            window_n += 1
            # Kept separately from `window`, which is reset every log_every steps: the
            # progress hook fires on a dataset count, not a step count, so the two are
            # not in phase and it would often read an empty window.
            last_loss = float(stats.get("loss", float("nan")))

            if self.step % self.log_every == 0:
                avg = {k: v / max(window_n, 1) for k, v in window.items()}
                record = {
                    "step": self.step,
                    "lr": self.scheduler.get_last_lr()[0],
                    "datasets_seen": self.datasets_seen,
                    **avg,
                }
                self.metrics.write(record)
                # Rate from the CURRENT run only. Dividing total steps by elapsed time
                # after a resume would count steps this process never ran and report a
                # rate several times too high, making the ETA useless exactly when a
                # long chained job most needs it.
                done_here = self.step - (self.resumed_at or 0)
                elapsed = time.time() - started
                steps_per_s = done_here / max(elapsed, 1e-9)
                remaining = self.max_steps - self.step
                eta_s = remaining / max(steps_per_s, 1e-9)
                pct = 100.0 * self.step / max(self.max_steps, 1)
                # WHERE THE TIME WENT, as a share of the window's wall clock. Printed on its
                # own line so the step line stays readable, and only when there is a window to
                # divide by.
                span = time.perf_counter() - phase_since
                if span > 0 and self._phase:
                    parts = "  ".join(
                        f"{name}={100.0 * secs / span:4.1f}%"
                        for name, secs in sorted(
                            self._phase.items(), key=lambda kv: -kv[1]
                        )
                    )
                    unaccounted = 100.0 * (span - sum(self._phase.values())) / span
                    data_pct = 100.0 * self._phase.get("data", 0.0) / span
                    # The verdict only when there is one. A hint appended unconditionally
                    # reads as "0% data means the GPU is WAITING", which is the opposite of
                    # what the number says.
                    verdict = ""
                    if data_pct >= 50.0:
                        verdict = "   <- STARVED: the GPU is waiting for the prior"
                    elif data_pct < 10.0:
                        verdict = "   <- compute-bound: the data pipeline is keeping up"
                    self.log.info(
                        "phases step %6d  %s  other=%4.1f%%%s",
                        self.step, parts, unaccounted, verdict,
                    )
                self._phase = defaultdict(float)
                phase_since = time.perf_counter()

                self.log.info(
                    "step %6d/%d [%5.1f%%]  %s  lr=%.3e  %.2f steps/s  "
                    "elapsed %s  eta %s  finishes ~%s  datasets %s",
                    self.step,
                    self.max_steps,
                    pct,
                    "  ".join(f"{k}={v:.5f}" for k, v in avg.items()),
                    record["lr"],
                    steps_per_s,
                    _hms(elapsed),
                    _hms(eta_s),
                    time.strftime("%a %H:%M", time.localtime(time.time() + eta_s)),
                    f"{self.datasets_seen:,}",
                )
                if self.telemetry.due_hardware(self.step):
                    hw = self.telemetry.sample_hardware(
                        step=self.step, datasets_seen=self.datasets_seen, steps_per_s=steps_per_s
                    )
                    hw["kind"] = "hardware"
                    hw["train_loss"] = last_loss
                    self.telemetry.record(hw)
                    # Utilisation and peak memory in the log line too, not only the CSV: the
                    # CSV has to be fetched from the cluster, and "was the GPU busy?" should be
                    # answerable from the log that gets pasted into a message.
                    parts = [
                        f"{k}={hw[k]:.1f}"
                        for k in ("gpu0_utilization_gpu", "mem_max_allocated_gb", "cpu_percent")
                        if isinstance(hw.get(k), (int, float))
                    ]
                    if parts:
                        self.log.info("hw   step %6d  %s", self.step, "  ".join(parts))

                # A 72h VSC job that will not finish should say so while there is still
                # time to react, rather than being discovered dead at the wall.
                if self._walltime_warning_s and eta_s > self._walltime_warning_s:
                    self.log.warning(
                        "PROJECTED OVERRUN: eta %s exceeds the remaining job walltime. "
                        "This run will be cut off and must be resumed (--resume auto), "
                        "or restarted with more GPUs (torchrun --nproc_per_node=N).",
                        _hms(eta_s),
                    )
                window, window_n = {}, 0

            if self.log_prior_every and self.step % self.log_prior_every == 0:
                self._log_prior_report()

            if self.progress is not None and self.progress.due(self.datasets_seen):
                self.progress.record(
                    dist.unwrap(self.model),
                    step=self.step,
                    datasets_seen=self.datasets_seen,
                    train_loss=last_loss,
                    elapsed_s=time.time() - started,
                )

            is_temp = self.save_temp_every > 0 and self.step % self.save_temp_every == 0
            is_perm = self.save_perm_every > 0 and self.step % self.save_perm_every == 0
            if is_temp or is_perm or self.step == self.max_steps:
                self._save()

        summary = {
            "steps": self.step,
            "datasets_seen": self.datasets_seen,
            "elapsed_s": round(time.time() - started, 1),
            "resumed_at": self.resumed_at,
            "freeze": self.freeze_report,
            "pretrained_load": self.load_report,
        }
        self.log.info("finished: %s", json.dumps({k: v for k, v in summary.items() if k != "freeze"}))
        # Last thing in the log, on purpose: this is what gets read first when a run comes back
        # slower than expected, and the starvation warning is the most common cause.
        if self.telemetry.enabled:
            self.log.info("%s", self.telemetry.summary())
            summary["telemetry_csv"] = str(self.telemetry.path)
        self.close()
        return summary

    def close(self) -> None:
        """Release the log file handle. Idempotent."""
        close_logging()

    # Usable as a context manager, so an exception mid-run still releases the
    # log file rather than leaving it locked.
    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.log.exception("run failed: %s", exc)
        self.close()

    def _log_prior_report(self) -> None:
        """Log what the prior is actually producing, mid-run.

        Cheap insurance: a config typo that silently switches the credit path off
        looks identical to a real null result in the loss curve. Seeing the
        boundary mass or base rate in the log makes the difference obvious.
        """
        try:
            from ..prior.generator import TaskGenerator
            from ..prior.rng import PriorRNG
            from ..utils.target_stats import target_stats

            gen = TaskGenerator(self.cfg["prior"], self.task, PriorRNG(self.seed + 999_983))
            vals: list[float] = []
            sources = {"base": 0, "credit": 0}
            for _ in range(self.log_prior_samples):
                t = gen.sample()
                sources[t.source] = sources.get(t.source, 0) + 1
                if self.regression:
                    # Scale-invariant: mass on the target's extreme values. The
                    # naive (y<=0)|(y>=1) version reports ~0.5 on a standard-scaled
                    # target, which is meaningless. See src/utils/target_stats.py.
                    vals.append(target_stats(t.y)["boundary_mass"])
                else:
                    vals.append(float(t.y.mean()))
            label = "boundary_mass" if self.regression else "base_rate"
            report = {
                "step": self.step,
                f"prior_{label}_mean": round(sum(vals) / len(vals), 4),
                "prior_sources": sources,
                "prior_filter": gen.filter_summary(),
            }
            self.metrics.write(report)
            self.log.info(
                "prior check @%d: %s=%.4f  sources=%s  filter_rejection=%.2f",
                self.step, label, report[f"prior_{label}_mean"], sources,
                gen.filter_summary().get("rejection_rate", 0.0),
            )
        except Exception as exc:  # never let a diagnostic kill a training run
            self.log.warning("prior report failed (continuing): %s", exc)

    def _save(self) -> None:
        # Only rank 0 writes, and it saves the UNWRAPPED model: a DDP state_dict has
        # every key prefixed with `module.` and would not load into a plain model at
        # evaluation time.
        if not self.dist.is_main:
            return
        save_checkpoint(
            self.ckpt_dir,
            step=self.step,
            model=dist.unwrap(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.cfg,
            extra={"datasets_seen": self.datasets_seen, "resumed_at": self.resumed_at},
        )
        removed = prune_checkpoints(
            self.ckpt_dir,
            save_perm_every=self.save_perm_every,
            max_temp=self.max_temp_checkpoints,
        )
        self.log.info(
            "checkpoint saved at step %d%s",
            self.step,
            f" (pruned {len(removed)} old temp checkpoints)" if removed else "",
        )
