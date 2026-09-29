"""Continued pretraining of the released TabPFN-3 weights on the synthetic prior (Experiment 2).

TabPFN-3's pretraining code is not public, so this does NOT imitate it. It runs every synthetic
task through TabPFN's OWN fine-tuning path (`tabpfn.finetuning`, pinned 9.0.0): the batched fit
mode, `get_preprocessed_dataset_chunks` + `meta_dataset_collator` for the per-member
preprocessing, `fit_from_preprocessed`, and the training forward with the loss its own
`FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor` compute. A task therefore reaches the
weights in training exactly as a real table reaches them at prediction — TabPFN's preprocessing
in both places — which is the property `prediction_view` gives the TabICL arms.

WHAT IS TABPFN'S, ARGUMENT FOR ARGUMENT (`FinetunedTabPFNBase.__init__` / `_fit`):

* two ensemble members per training forward (`n_estimators_finetune=2`), activation
  checkpointing on (`force_recompute_layer`), cuDNN attention excluded (`sdpa_kernel_context`);
* float16 autocast with a `GradScaler` on CUDA, gradient norm clipped at 1.0;
* AdamW, a warm-up of 10 % of the steps then cosine decay (our `build_scheduler`, configured to
  match: `warmup_proportion: 0.1`, `beta2: 0.999` — torch's default, which TabPFN uses);
* classification loss: cross-entropy on the raw logits of every member, `(Q, B, E, L)` reshaped
  to `(B*E, L, Q)` (`_forward_with_loss`); the estimator's `n_classes_` is set per task first,
  because `forward` slices and un-permutes the logits with it;
* regression loss: TabPFN's `_compute_regression_loss` with its fine-tuner's defaults — CRPS on
  the bar distribution plus MSE of its mean, no NLL term (`ce 0, crps 1, mse 1`) — on targets
  z-scored with the context statistics (the collator's `znorm_space_bardist`).

WHAT IS OURS: the data. The official fine-tuner takes ONE real dataset and cuts it into
context/query chunks; this takes the synthetic stream (`src/prior/dataset.py`) — the same
tasks, in the same order, as the TabICL arms with the same seed, with credit tables in the raw
encoding (`prior.encoding: raw`: gaps as NaN, the LGD target on [0, 1]) because TabPFN's own
preprocessing does the rest. One task per forward pass, `train.batch_size` tasks per update
(TabPFN-Wide, the continued-pretraining precedent in `tfm-library/papers/2026/03_Kolberg_et_al.*`,
trains at 16). The per-task preprocessing runs in the data-loader workers
(`TabPFNTaskPreparer`), each holding its own CPU copy of the estimator.

TabPFN's prior is not public, so the control arm (`credit_fraction: 0`) continues TabPFN-3's
pretraining on TabICLv2's open prior, not on its own. Every arm shares that, so the credit
effect is still isolated; the paper must say it.
"""

from __future__ import annotations

import json
import os
import signal
import time
import zlib
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.logging_setup import close_logging, log_environment, log_section, setup_logging
from .checkpoint import latest_checkpoint, prune_checkpoints
from .optim import build_optimizer, build_scheduler
from .telemetry import Telemetry, WeightTracker

#: What TabPFN's own fine-tuner switches off for training (`_build_estimator_config`): the
#: training loop hands the model numbers, never text or dates, and preprocessing stays on the CPU.
TABPFN_INFERENCE_CONFIG = {"ENABLE_GPU_PREPROCESSING": False, "TRANSFORM_TEXT": False,
                           "TRANSFORM_DATES": False}
#: `FinetunedTabPFNRegressor`'s default loss weights (tabpfn 9.0.0).
REGRESSION_LOSS_WEIGHTS = {"ce": 0.0, "crps": 1.0, "crls": 0.0, "mse": 1.0, "mae": 0.0}


def resolve_tabpfn_weights(task: str, path: str | Path | None = None) -> Path:
    """The released TabPFN-3 checkpoint for `task`: an explicit path, else the local one
    `baselines.find_local_tabpfn_checkpoint` finds (the file the reference column scores)."""
    if path:
        from src.utils.paths import find_pretrained

        return find_pretrained(path)
    from src.eval.baselines import find_local_tabpfn_checkpoint

    found = find_local_tabpfn_checkpoint("classifier" if task == "pd" else "regressor")
    if found is None:
        raise FileNotFoundError(
            "no released TabPFN-3 checkpoint on disk (tabpfn-v3-{classifier,regressor}-*.ckpt in "
            "checkpoints/ or on project storage). Compute nodes cannot download it.")
    return found


def build_estimator(task: str, model_path: str | Path, *, n_estimators: int, device: str, seed: int):
    """A TabPFN-3 estimator in the batched fine-tuning mode, exactly as the official fine-tuner
    builds its training estimator (`_create_estimator` + `_build_estimator_config`)."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    cls = TabPFNClassifier if task == "pd" else TabPFNRegressor
    est = cls.create_default_for_version(
        version=ModelVersion.V3, model_path=str(model_path), fit_mode="batched",
        differentiable_input=False, n_estimators=int(n_estimators), device=device,
        ignore_pretraining_limits=True, random_state=int(seed),
        inference_config=dict(TABPFN_INFERENCE_CONFIG),
    )
    est._initialize_model_variables()
    return est


class _ContextQuerySplit:
    """`split_fn` for `get_preprocessed_dataset_chunks`: the first `train_size` rows are the
    context, the rest the query — the stream's own split, which a shift may have ordered."""

    def __init__(self, train_size: int):
        self.train_size = int(train_size)

    def __call__(self, X, y, stratify=None):  # noqa: ARG002 — the signature TabPFN calls
        ts = self.train_size
        return X[:ts], X[ts:], y[:ts], y[ts:]


class TabPFNTaskPreparer:
    """Turns one stream batch into TabPFN's preprocessed items, in a data-loader worker.

    Picklable: the estimator is built lazily in each process that calls it, on the CPU, from
    the same released checkpoint and settings as the training estimator. The preprocessing it
    runs (`_initialize_dataset_preprocessing` + the per-member transforms of `ds[0]`) depends
    only on the data, the settings and the seed — never on the weights being trained.
    """

    def __init__(self, task: str, model_path: str | Path, n_estimators: int, seed: int):
        self.task = task
        self.model_path = str(model_path)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self._est = None

    def _estimator(self):
        if self._est is None:
            from torch.utils.data import get_worker_info

            if get_worker_info() is not None:
                torch.set_num_threads(1)  # one core per worker: the workers are the parallelism
            self._est = build_estimator(self.task, self.model_path, n_estimators=self.n_estimators,
                                        device="cpu", seed=self.seed)
        return self._est

    def prepare(self, X: np.ndarray, y: np.ndarray, train_size: int, seed: int) -> dict[str, Any] | None:
        """One task -> `{"item", "n_classes"}`, or None when TabPFN cannot train on it (a query
        label the context never shows — the official fine-tuner skips those batches too)."""
        from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks

        classification = self.task == "pd"
        if classification:
            y = y.astype(np.int64)
            if not set(np.unique(y[train_size:]).tolist()) <= set(np.unique(y[:train_size]).tolist()):
                return None
        est = self._estimator()
        ds = get_preprocessed_dataset_chunks(
            est, [X], [y], _ContextQuerySplit(train_size), None,
            "classifier" if classification else "regressor",
            equal_split_size=False, data_shuffle_seed=seed, preprocessing_random_state=seed,
            shuffle=False,
        )
        item = ds[0]
        return {"item": item, "n_classes": int(est.n_classes_) if classification else None}

    def __call__(self, batch, b: int) -> list[dict[str, Any] | None]:
        X, y, d, seq_lens, train_sizes = batch
        out = []
        for i in range(X.shape[0]):
            n, w, ts = int(seq_lens[i]), max(1, int(d[i])), int(train_sizes[i])
            seed = zlib.crc32(f"{self.seed}:{b}:{i}".encode()) & 0x7FFFFFFF
            try:
                out.append(self.prepare(X[i, :n, :w].numpy().astype(np.float64),
                                        y[i, :n].numpy().astype(np.float64), ts, seed))
            except Exception as exc:  # noqa: BLE001 — one table must not stop the stream
                out.append({"error": f"{type(exc).__name__}: {exc}"})
        return out


def classification_loss(est, batch, n_estimators: int) -> torch.Tensor:
    """`FinetunedTabPFNClassifier._forward_with_loss`, verbatim in effect."""
    logits = est.forward(batch.X_query, return_raw_logits=True)  # Q, B, E, L
    q, b, e, n_logits = logits.shape
    if b != 1 or e != n_estimators or n_logits != est.n_classes_:
        raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)} "
                           f"(estimators {n_estimators}, classes {est.n_classes_})")
    logits_blq = logits.permute(1, 2, 3, 0).reshape(b * e, n_logits, q)
    targets = batch.y_query.repeat(b * e, 1).to(logits.device).long()
    return torch.nn.functional.cross_entropy(logits_blq, targets)


def regression_loss(est, batch, n_estimators: int, weights: dict[str, float]) -> torch.Tensor:
    """`FinetunedTabPFNRegressor._forward_with_loss`, through TabPFN's own loss function."""
    from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss

    _, per_estimator, _ = est.forward(batch.X_query)
    logits = torch.stack(per_estimator, dim=2)  # Q, B, E, L
    q, b, e, n_logits = logits.shape
    bardist = batch.znorm_space_bardist
    if b != 1 or e != n_estimators or n_logits != bardist.num_bars:
        raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
    logits_bql = logits.permute(1, 2, 0, 3).reshape(b * e, q, n_logits)
    targets = batch.y_query.repeat(b * e, 1).to(logits.device)
    return _compute_regression_loss(
        logits_BQL=logits_bql, targets_BQ=targets, bardist_loss_fn=bardist,
        ce_loss_weight=weights["ce"], crps_loss_weight=weights["crps"],
        crls_loss_weight=weights["crls"], mse_loss_weight=weights["mse"],
        mse_loss_clip=None, mae_loss_weight=weights["mae"], mae_loss_clip=None,
    )


class TabPFNTrainer:
    """`src.train.loop.Trainer`'s interface (`maybe_resume`, `train`, `close`) for TabPFN-3."""

    def __init__(self, cfg: dict[str, Any], out_dir: str | Path, device: str | None = None,
                 ckpt_dir: str | Path | None = None, log_dir: str | Path | None = None,
                 manifest_dir: str | Path | None = None):
        self.cfg = cfg
        self.task = cfg["task"]
        self.regression = self.task == "lgd"
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else self.out_dir / "checkpoints"
        manifest_dir = Path(manifest_dir) if manifest_dir is not None else self.out_dir
        self.run_name = str(cfg.get("_run_name", cfg.get("experiment", "run")))
        lcfg = cfg.get("logging", {}) or {}
        self.log, self.metrics, self.log_path = setup_logging(
            self.run_name, log_dir or (self.out_dir / "logs"), level=str(lcfg.get("level", "INFO")),
            console=bool(lcfg.get("console", True)), to_file=lcfg.get("to_file"))

        tcfg = cfg.get("train", {}) or {}
        self.tcfg = tcfg
        self.max_steps = int(tcfg.get("max_steps", 10_000))
        self.batch_size = int(tcfg.get("batch_size", 16))
        self.n_estimators = int(tcfg.get("n_estimators", 2))
        self.grad_clip = float(tcfg.get("gradient_clipping", 1.0))
        self.log_every = int(tcfg.get("log_every", 50))
        self.save_temp_every = int(tcfg.get("save_temp_every", 250))
        self.save_perm_every = int(tcfg.get("save_perm_every", 2500))
        self.max_temp_checkpoints = int(tcfg.get("max_temp_checkpoints", 2))
        self.num_workers = int(tcfg.get("num_workers", 0))
        self.seed = int(cfg.get("seed", 0))
        self.loss_weights = {**REGRESSION_LOSS_WEIGHTS, **(tcfg.get("regression_loss_weights") or {})}
        torch.manual_seed(self.seed)

        if (cfg.get("prior") or {}).get("encoding", "raw") != "raw":
            raise ValueError("TabPFN arms need prior.encoding: raw — TabPFN preprocesses the table "
                             "itself, so a table in TabICL's prediction encoding would be "
                             "preprocessed twice")
        if str(tcfg.get("optimizer", "adamw")).lower() != "adamw":
            raise ValueError("TabPFN-3 is fine-tuned with AdamW (its own fine-tuner); "
                             f"train.optimizer={tcfg.get('optimizer')!r} is not supported for it")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(tcfg.get("amp", True)) and self.device.startswith("cuda")
        icfg = cfg.get("init", {}) or {}
        self.weights_path = resolve_tabpfn_weights(self.task, icfg.get("pretrained_path"))

        log_section(self.log, f"CreditICL — TabPFN-3 continued pretraining — {self.task.upper()} — "
                              f"{self.run_name}")
        log_environment(self.log, {"device": self.device, "task": self.task, "seed": self.seed,
                                   "architecture": "tabpfn3"})
        self.log.info("released weights -> %s", self.weights_path)
        self.log.info("credit_fraction IN USE: %s", (cfg.get("prior") or {}).get("credit_fraction"))
        self.log.info("outputs -> %s | checkpoints -> %s", self.out_dir, self.ckpt_dir)

        self.est = build_estimator(self.task, self.weights_path, n_estimators=self.n_estimators,
                                   device=self.device, seed=self.seed)
        self.model = self.est.models_[0]
        self.model.train()
        n_params = sum(p.numel() for p in self.model.parameters())
        self.log.info("TabPFN-3 %s: %s parameters, %d members per forward, AMP %s",
                      "regressor" if self.regression else "classifier", f"{n_params:,}",
                      self.n_estimators, self.amp)
        self.optimizer = build_optimizer(self.model, tcfg)
        self.scheduler = build_scheduler(self.optimizer, tcfg, self.max_steps)
        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None

        from tabpfn.architectures.interface import PerformanceOptions

        self.performance = PerformanceOptions(force_recompute_layer=bool(tcfg.get("activation_checkpointing", True)),
                                              use_chunkwise_inference=False)
        self.step = 0
        self.datasets_seen = 0
        self.skipped_tasks = 0
        self.resumed_at: int | None = None
        self._train_seconds_before = 0.0
        self._segment_started: float | None = None
        self._stop_requested: str | None = None

        self.telemetry = Telemetry(self.run_name, manifest_dir,
                                   hardware_every=int(lcfg.get("log_hardware_every", 0) or 0),
                                   grad_every=int(lcfg.get("log_grad_every", 0) or 0))
        self.weights = WeightTracker(manifest_dir, every=int(lcfg.get("log_weights_every",
                                                                       lcfg.get("log_grad_every", 0)) or 0))
        pcfg = cfg.get("progress", {}) or {}
        from .progress import ProgressConfig
        from .tabpfn_progress import TabPFNProgressTracker

        self.progress = TabPFNProgressTracker(
            ProgressConfig(
                every_datasets=int(pcfg.get("every_datasets", 0)),
                datasets=list(pcfg.get("datasets") or cfg.get("eval", {}).get("dev_datasets", [])),
                allowed_datasets=cfg.get("eval", {}).get("dev_datasets"),
                n_datasets=int(pcfg.get("n_datasets", 4)), n_ood=int(pcfg.get("n_ood", 4)),
                context_rows=int(pcfg.get("context_rows", 512)),
                max_test_rows=int(pcfg.get("max_test_rows", 2000)), seed=int(pcfg.get("seed", 0)),
            ),
            self.task, self.run_name, manifest_dir, trainer=self,
        )

    # -- checkpoints ---------------------------------------------------------------------
    def save_weights(self, path: Path, extra: dict[str, Any] | None = None) -> Path:
        """The weights in TabPFN's own checkpoint format (`save_tabpfn_model`), which
        `TabPFNClassifier(model_path=...)` loads — how the benchmark scores it. Atomic."""
        from tabpfn.model_loading import save_tabpfn_model

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        save_tabpfn_model(self.est, tmp, additional_fields={
            "crediticl_run_name": self.run_name, "crediticl_step": self.step,
            "crediticl_config": self.cfg, **(extra or {})})
        os.replace(tmp, path)
        return path

    def _save(self) -> None:
        path = self.save_weights(self.ckpt_dir / f"step-{self.step}.ckpt", extra={
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": self.step,
            "extra": {"datasets_seen": self.datasets_seen, "resumed_at": self.resumed_at,
                      "train_seconds": self._train_seconds(), "skipped_tasks": self.skipped_tasks},
        })
        removed = prune_checkpoints(self.ckpt_dir, save_perm_every=self.save_perm_every,
                                    max_temp=self.max_temp_checkpoints)
        self.log.info("checkpoint saved at step %d -> %s%s", self.step, path.name,
                      f" (pruned {len(removed)} old temp checkpoints)" if removed else "")

    def maybe_resume(self) -> None:
        self.weights.start(self.model)  # the released weights: drift is measured from them
        path = latest_checkpoint(self.ckpt_dir)
        if path is None:
            self.log.info("no checkpoint in %s — starting from the released weights", self.ckpt_dir)
            self.telemetry.resume(0)
            self.weights.resume(0)
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = {k: v for k, v in payload["state_dict"].items() if not k.startswith("criterion.")}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected or missing:
            raise RuntimeError(f"{path}: state dict mismatch (missing {len(missing)}, "
                               f"unexpected {len(unexpected)})")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.step = int(payload.get("step", 0))
        extra = payload.get("extra", {}) or {}
        self.datasets_seen = int(extra.get("datasets_seen", 0))
        self.skipped_tasks = int(extra.get("skipped_tasks", 0))
        self._train_seconds_before = float(extra.get("train_seconds", 0.0) or 0.0)
        self.resumed_at = self.step
        self.telemetry.resume(self.step)
        self.weights.resume(self.step)
        if self.step >= self.max_steps:
            self.log.warning("THIS RUN WILL TRAIN NOTHING: the checkpoint in %s is at step %d and "
                             "max_steps is %d. To start fresh: python -m src.utils.clean_run --clean "
                             "--checkpoints", self.ckpt_dir, self.step, self.max_steps)
        self.log.warning("RESUMED from %s at step %d; the task stream restarts at batch %d.",
                         path.name, self.step, self.step)

    def _train_seconds(self) -> float:
        here = time.time() - self._segment_started if self._segment_started else 0.0
        return self._train_seconds_before + here

    # -- one update ----------------------------------------------------------------------
    def _task_loss(self, prepared: dict[str, Any]) -> torch.Tensor:
        from tabpfn.finetuning._torch_compat import sdpa_kernel_context
        from tabpfn.finetuning.data_util import meta_dataset_collator

        batch = meta_dataset_collator([prepared["item"]])
        if self.regression:
            self.est.raw_space_bardist_ = batch.raw_space_bardist
            self.est.bardist_ = batch.znorm_space_bardist
        else:
            # `forward` slices and un-permutes the logits with `n_classes_`, and our tasks do not
            # share one class count the way a single fine-tuning dataset does.
            self.est.n_classes_ = int(prepared["n_classes"])
            self.est.classes_ = torch.arange(self.est.n_classes_)
        self.est.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices,
                                       batch.configs, performance_options=self.performance)
        autocast = (torch.amp.autocast("cuda", enabled=True) if self.amp else nullcontext())
        with autocast, sdpa_kernel_context():
            if self.regression:
                return regression_loss(self.est, batch, self.n_estimators, self.loss_weights)
            return classification_loss(self.est, batch, self.n_estimators)

    def train_step(self, prepared: list[dict[str, Any] | None]) -> dict[str, float]:
        from tabpfn.finetuning._torch_compat import sdpa_kernel_context

        usable = [p for p in prepared if p is not None and "item" in p]
        self.skipped_tasks += len(prepared) - len(usable)
        for p in prepared:
            if p is not None and "error" in p:
                self.log.warning("task skipped at step %d: %s", self.step, p["error"])
        self.datasets_seen += len(prepared)
        if not usable:
            return {}
        total = 0.0
        for p in usable:
            loss = self._task_loss(p)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {self.step}")
            scaled = loss / len(usable)
            with sdpa_kernel_context():
                (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
            total += float(loss.detach())
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.telemetry.due_grad(self.step + 1):
            row = self.telemetry.sample_grads(self.model)
            row.update({"step": self.step + 1, "datasets_seen": self.datasets_seen, "kind": "grad"})
            self.telemetry.record(row)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                           self.grad_clip)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return {"loss": total / len(usable), "tasks": float(len(usable))}

    # -- the loop --------------------------------------------------------------------------
    def _install_stop_signals(self) -> None:
        def handler(signum, _frame):
            self._stop_requested = signal.Signals(signum).name
            self.log.warning("received %s — finishing the step, then saving and stopping",
                             self._stop_requested)
        for sig in (getattr(signal, "SIGUSR1", None), signal.SIGTERM):
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # not the main thread (tests)
                    pass

    def train(self) -> dict[str, Any]:
        from src.prior.dataset import build_loader

        self._install_stop_signals()
        prior_cfg = dict(self.cfg["prior"])
        preparer = TabPFNTaskPreparer(self.task, self.weights_path, self.n_estimators, self.seed)
        loader = build_loader(prior_cfg, self.task, batch_size=self.batch_size, seed=self.seed,
                              num_workers=self.num_workers, start_batch=self.step, transform=preparer)
        it = iter(loader)
        started = time.time()
        self._segment_started = started
        if self.progress.enabled:
            if self.step == 0:
                self.progress.record(self.model, step=0, datasets_seen=0, train_loss=float("nan"),
                                     elapsed_s=0.0)
            else:
                self.progress.resume_from(self.datasets_seen)
        if self.step == 0:
            self.weights.sample(self.model, step=0, datasets_seen=0)
        window: dict[str, float] = defaultdict(float)
        window_n = 0
        phase: dict[str, float] = defaultdict(float)
        phase_since = time.perf_counter()
        last_loss = float("nan")
        prog_sum, prog_n = 0.0, 0
        while self.step < self.max_steps:
            t0 = time.perf_counter()
            prepared = next(it)
            phase["data"] += time.perf_counter() - t0
            t1 = time.perf_counter()
            stats = self.train_step(prepared)
            phase["step"] += time.perf_counter() - t1
            self.step += 1
            if stats:
                last_loss = stats["loss"]
                window["loss"] += stats["loss"]
                window_n += 1
                prog_sum, prog_n = prog_sum + stats["loss"], prog_n + 1
            if self.weights.due(self.step):
                self.weights.sample(self.model, step=self.step, datasets_seen=self.datasets_seen)
            if self.step % self.log_every == 0:
                avg = window["loss"] / max(window_n, 1)
                lr = self.scheduler.get_last_lr()[0]
                self.metrics.write({"step": self.step, "lr": lr, "datasets_seen": self.datasets_seen,
                                    "loss": avg, "skipped_tasks": self.skipped_tasks})
                done_here = self.step - (self.resumed_at or 0)
                steps_per_s = done_here / max(time.time() - started, 1e-9)
                eta = (self.max_steps - self.step) / max(steps_per_s, 1e-9)
                span = time.perf_counter() - phase_since
                shares = {f"phase_{k}_pct": round(100.0 * v / span, 2) for k, v in phase.items()} if span > 0 else {}
                self.log.info("step %6d/%d  loss=%.5f  lr=%.3e  %.3f steps/s  eta %.1f h  %s",
                              self.step, self.max_steps, avg, lr, steps_per_s, eta / 3600,
                              "  ".join(f"{k[6:-4]}={v:.0f}%" for k, v in shares.items()))
                if self.telemetry.due_hardware(self.step):
                    hw = self.telemetry.sample_hardware(step=self.step, datasets_seen=self.datasets_seen,
                                                        steps_per_s=steps_per_s)
                    hw.update({"kind": "hardware", "train_loss": last_loss, "train_loss_window": avg,
                               "lr": lr, **shares})
                    self.telemetry.record(hw)
                window, window_n = defaultdict(float), 0
                phase, phase_since = defaultdict(float), time.perf_counter()
            if self.progress.due(self.datasets_seen):
                self.progress.record(self.model, step=self.step, datasets_seen=self.datasets_seen,
                                     train_loss=prog_sum / max(prog_n, 1),
                                     elapsed_s=time.time() - started)
                prog_sum, prog_n = 0.0, 0
                self.model.train()
            is_temp = self.save_temp_every > 0 and self.step % self.save_temp_every == 0
            is_perm = self.save_perm_every > 0 and self.step % self.save_perm_every == 0
            if is_temp or is_perm or self.step == self.max_steps or self._stop_requested:
                self._save()
            if self._stop_requested:
                self.log.warning("STOPPING EARLY on %s at step %d/%d; resubmit to resume.",
                                 self._stop_requested, self.step, self.max_steps)
                break
        summary = {
            "steps": self.step, "completed": self.step >= self.max_steps,
            "stopped_by_signal": self._stop_requested, "datasets_seen": self.datasets_seen,
            "skipped_tasks": self.skipped_tasks, "elapsed_s": round(time.time() - started, 1),
            "train_seconds_total": round(self._train_seconds(), 1), "resumed_at": self.resumed_at,
            "architecture": "tabpfn3", "released_weights": str(self.weights_path),
        }
        self.log.info("finished: %s", json.dumps(summary))
        if self.telemetry.enabled:
            self.log.info("%s", self.telemetry.summary())
            summary["telemetry_csv"] = str(self.telemetry.path)
        if self.weights.enabled:
            self.log.info("%s", self.weights.summary())
            summary["weights_csv"] = str(self.weights.path)
        self.close()
        return summary

    def close(self) -> None:
        close_logging()

