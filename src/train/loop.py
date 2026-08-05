"""Pretraining loop for NanoTabICLv2 on a configured prior.

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
support constraints at decode time. See docs/experimental_design.md §1.5.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..models.nanotabiclv2 import NanoTabICLv2
from ..prior.dataset import build_loader
from ..utils.logging_setup import log_environment, log_section, setup_logging
from .adapt import STRATEGIES, apply_freezing, load_pretrained, set_training_mode
from .checkpoint import latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint
from .optim import build_optimizer, build_scheduler


def quantile_levels(num_quantiles: int, device=None, dtype=None) -> torch.Tensor:
    """The exact grid TabICL trains and predicts on."""
    return torch.linspace(0.0, 1.0, num_quantiles + 2, device=device, dtype=dtype)[1:-1]


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
        self.save_temp_every = int(tcfg.get("save_temp_every", 500))
        self.save_perm_every = int(tcfg.get("save_perm_every", 5_000))
        self.max_temp_checkpoints = int(tcfg.get("max_temp_checkpoints", 2))
        self.num_workers = int(tcfg.get("num_workers", 0))
        self.seed = int(cfg.get("seed", 0))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

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
        self.optimizer = build_optimizer(self.model, tcfg)
        self.scheduler = build_scheduler(self.optimizer, tcfg, self.max_steps)

        self.amp = bool(tcfg.get("amp", True)) and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.amp else nullcontext()
        )

        self.step = 0
        self.datasets_seen = 0
        self.resumed_at: int | None = None

        log_section(self.log, f"CreditICL — {self.task.upper()} — {cfg.get('_run_name', '?')}")
        log_environment(self.log, {"device": self.device, "amp": self.amp, "task": self.task})
        self.log.info("levers: %s", json.dumps(cfg.get("_grid", {}).get("assignments", {}), default=str))
        self.log.info("credit_fraction: %s", cfg["prior"].get("credit_fraction"))
        self.log.info("outputs     -> %s", self.out_dir)
        self.log.info("checkpoints -> %s", self.ckpt_dir)
        self.log.info("log file    -> %s", self.log_path)
        self.log.info(
            "budget: %d steps x %d datasets/step = %d datasets",
            self.max_steps,
            self.batch_size,
            self.max_steps * self.batch_size,
        )

    # -- setup ---------------------------------------------------------------
    def _build_model(self) -> NanoTabICLv2:
        mcfg = self.cfg.get("model", {})
        n_classes = int(self.cfg.get("prior", {}).get("n_classes", 2))
        # regression: max_classes=0 and out_dim=Q; classification: both = n_classes
        max_classes = 0 if self.regression else n_classes
        out_dim = self.num_quantiles if self.regression else n_classes
        return NanoTabICLv2(
            max_classes=max_classes,
            out_dim=out_dim,
            embed_dim=int(mcfg.get("embed_dim", 128)),
            col_num_blocks=int(mcfg.get("col_num_blocks", 3)),
            row_num_blocks=int(mcfg.get("row_num_blocks", 3)),
            icl_num_blocks=int(mcfg.get("icl_num_blocks", 12)),
            col_nhead=int(mcfg.get("col_nhead", 8)),
            row_nhead=int(mcfg.get("row_nhead", 8)),
            icl_nhead=int(mcfg.get("icl_nhead", 8)),
            feature_group_size=int(mcfg.get("feature_group_size", 3)),
            n_cls_cols=int(mcfg.get("n_cls_cols", 4)),
            n_cls_rows=int(mcfg.get("n_cls_rows", 128)),
        )

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
        self.log.warning(
            "RESUMED from %s at step %d. DataLoader worker RNGs are re-seeded on resume, so "
            "the dataset stream differs from an uninterrupted run; step and dataset counts "
            "are preserved exactly.",
            path.name, self.step,
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
                flat = pred.flatten(end_dim=-2)
                true = y_test.long().flatten()
                loss = F.cross_entropy(flat, true)
                with torch.no_grad():
                    extra = {"accuracy": float((flat.argmax(dim=1) == true).float().mean())}
        return loss, extra

    def train_step(self, batch) -> dict[str, float]:
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
                totals["loss"] += float(loss) / n_micro
                for k, v in extra.items():
                    totals[k] = totals.get(k, 0.0) + v / n_micro
            except torch.cuda.OutOfMemoryError:
                self.log.warning("OOM in micro-batch %d/%d at step %d — skipping", i + 1, n_micro, self.step)
                torch.cuda.empty_cache()
                oom += 1
                continue

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

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        self.datasets_seen += int(X.shape[0])
        return totals

    # -- main ----------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        loader = build_loader(
            self.cfg["prior"],
            self.task,
            batch_size=self.batch_size,
            seed=self.seed,
            num_workers=self.num_workers,
        )
        it = iter(loader)
        started = time.time()
        window: dict[str, float] = {}
        window_n = 0

        while self.step < self.max_steps:
            batch = next(it)
            stats = self.train_step(batch)
            self.step += 1
            for k, v in stats.items():
                window[k] = window.get(k, 0.0) + v
            window_n += 1

            if self.step % self.log_every == 0:
                avg = {k: v / max(window_n, 1) for k, v in window.items()}
                record = {
                    "step": self.step,
                    "lr": self.scheduler.get_last_lr()[0],
                    "datasets_seen": self.datasets_seen,
                    **avg,
                }
                self.metrics.write(record)
                steps_per_s = self.step / max(time.time() - started, 1e-9)
                eta_min = (self.max_steps - self.step) / max(steps_per_s, 1e-9) / 60
                self.log.info(
                    "step %d/%d  %s  lr=%.3e  %.2f steps/s  eta %.0f min",
                    self.step,
                    self.max_steps,
                    "  ".join(f"{k}={v:.5f}" for k, v in avg.items()),
                    record["lr"],
                    steps_per_s,
                    eta_min,
                )
                window, window_n = {}, 0

            if self.log_prior_every and self.step % self.log_prior_every == 0:
                self._log_prior_report()

            is_temp = self.save_temp_every > 0 and self.step % self.save_temp_every == 0
            is_perm = self.save_perm_every > 0 and self.step % self.save_perm_every == 0
            if is_temp or is_perm or self.step == self.max_steps:
                self._save()

        return {
            "steps": self.step,
            "datasets_seen": self.datasets_seen,
            "elapsed_s": round(time.time() - started, 1),
            "resumed_at": self.resumed_at,
            "freeze": self.freeze_report,
            "pretrained_load": self.load_report,
        }

    def _log_prior_report(self) -> None:
        """Log what the prior is actually producing, mid-run.

        Cheap insurance: a config typo that silently switches the credit path off
        looks identical to a real null result in the loss curve. Seeing the
        boundary mass or base rate in the log makes the difference obvious.
        """
        try:
            from ..prior.generator import TaskGenerator
            from ..prior.rng import PriorRNG

            gen = TaskGenerator(self.cfg["prior"], self.task, PriorRNG(self.seed + 999_983))
            vals: list[float] = []
            sources = {"base": 0, "credit": 0}
            for _ in range(self.log_prior_samples):
                t = gen.sample()
                sources[t.source] = sources.get(t.source, 0) + 1
                if self.regression:
                    vals.append(float(((t.y <= 0.0) | (t.y >= 1.0)).float().mean()))
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
        save_checkpoint(
            self.ckpt_dir,
            step=self.step,
            model=self.model,
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
