"""Checkpoint save / resume.

Why this is load-bearing rather than a convenience: the KU Leuven VSC
documentation contains **no Slurm requeue recipe** (the string `requeue` does not
appear in it), and its only documented checkpointing facility is the Torque-era
`csub`/BLCR framework. With a 72 h walltime ceiling, a long run must checkpoint,
resume, and resubmit itself. See `docs/VSC.md` §3.

Naming follows TabICL (`step-<n>.ckpt`) so `get_latest_checkpoint` logic is
familiar, and distinguishes *temporary* checkpoints (frequent, pruned) from
*permanent* ones (kept), as upstream does via `save_temp_every` /
`save_perm_every`.

Model, optimizer, scheduler and step counter are restored. The prior stream is
indexed by batch/slot (src/prior/stream.py), so resuming at the saved step replays
the same tasks independently of the worker count. A changed training implementation
or recipe must use a new run name; historical checkpoints remain readable for scoring.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

STEP_RE = re.compile(r"^step-(\d+)\.ckpt$")


def validate_training_fingerprint(payload: dict[str, Any]) -> None:
    from src.utils.preflight import training_fingerprint

    if payload.get("training_fingerprint") != training_fingerprint():
        raise ValueError("Checkpoint training code differs or has no recorded fingerprint. "
                         "Preserve it for evaluation; use a new run name for the current recipe.")


def validate_resume_config(old: dict[str, Any], current: dict[str, Any]) -> None:
    """Do not continue weights under a different prior, seed or optimiser recipe.

    Worker/micro-batch counts, output/logging settings and the maximum step count
    may change for recovery. Extending the schedule is warned about by the trainer.
    """
    changed = []
    missing = object()
    for key in ("task", "seed", "architecture", "model", "init", "prior"):
        if old.get(key, missing) != current.get(key, missing):
            changed.append(key)
    old_train, new_train = old.get("train", {}), current.get("train", {})
    for key in ("batch_size", "optimizer", "lr", "weight_decay", "beta1", "beta2", "momentum",
                "scheduler", "warmup_proportion", "gradient_clipping", "l2sp_alpha", "loss_weights",
                "cosine_num_cycles", "cosine_amplitude_decay", "cosine_lr_end", "n_estimators", "amp",
                "regression_loss_weights", "activation_checkpointing", "dtype",
                "cautious_weight_decay", "num_quantiles", "require_efficient_attention"):
        if old_train.get(key, missing) != new_train.get(key, missing):
            changed.append(f"train.{key}")
    if changed:
        raise ValueError("Resume configuration CHANGED: " + ", ".join(changed)
                         + ". Preserve these checkpoints and use a new run name for a different recipe.")


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in d.iterdir():
        m = STEP_RE.match(entry.name)
        if m:
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, entry)
    return None if best is None else best[1]


def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step-{step}.ckpt"
    # UPSTREAM-COMPATIBLE KEYS alongside ours. `TabICLClassifier(model_path=<path>)` reads
    # `state_dict`, `curr_step` and a `config` holding the TabICL kwargs; we already write
    # `state_dict`, so adding the other two makes our checkpoints loadable by upstream's own
    # wrapper — which is how our model can be scored through the SAME preprocessing and
    # ensembling as the released one instead of a hand-rolled single pass.
    inner = getattr(model, "module", model)   # unwrap DDP
    model_config = getattr(inner, "creditcl_model_config", None)
    # UPSTREAM'S SCHEMA, EXACTLY. `TabICLClassifier(model_path=<path>)` does:
    #
    #     assert "config" in checkpoint          # the TabICL kwargs
    #     assert "state_dict" in checkpoint
    #     self.model_ = TabICL(**checkpoint["config"])
    #     self.model_.load_state_dict(checkpoint["state_dict"])
    #
    # So `config` must be the ARCHITECTURE kwargs, not a project YAML — otherwise `TabICL(**...)`
    # is handed `task`, `prior`, `sweep` and dies. Our own metadata therefore moves to
    # `crediticl_config`. Writing the schema their loader expects is what lets our weights be
    # scored through upstream's own wrapper, which is the only way the comparison against the
    # released model is about weights rather than about two different inference pipelines.
    #
    # They load with `weights_only=True`, so everything here must be plain types or tensors.
    if model_config is None:
        # Warn, do not raise. A checkpoint without it is still ours to resume from; it just
        # cannot be handed to upstream's wrapper, which matters only for scoring. Raising here
        # would also break every test that checkpoints a plain `nn.Module`.
        from src.utils.logging_setup import get_logger

        get_logger().warning(
            "this model carries no `creditcl_model_config`, so the checkpoint will NOT be "
            "loadable by TabICLRegressor(model_path=...) and `crediticl` cannot be scored from "
            "it. Build the model through src.models.architecture.build_model, which records it."
        )
    from src.train.random_state import capture
    from src.utils.preflight import training_fingerprint

    payload = {
        "training_fingerprint": training_fingerprint(),
        "config": model_config,          # upstream's contract: kwargs to TabICL
        "state_dict": model.state_dict(),
        "curr_step": step,               # upstream's name for the step
        "step": step,                    # ours, kept so older readers still work
        "crediticl_config": config,      # OUR resolved YAML — task, prior, sweep, everything
        "optimizer_state": optimizer.state_dict(),
        # Recorded so a resume can refuse a changed optimizer instead of loading one
        # optimizer's state into another (torch would raise something unhelpful, or worse,
        # accept a structurally compatible state that means something else).
        "optimizer_class": type(optimizer).__name__,
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
        "random_state": capture(),
    }
    # Write to a temp file then rename: a job killed at the walltime limit
    # mid-write would otherwise leave a truncated checkpoint that resume picks up
    # as "latest" and fails on.
    tmp = path.with_suffix(".ckpt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
    current_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # weights_only=False because the payload carries the config dict; the file is
    # one we wrote ourselves on our own filesystem.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if current_config is not None:
        validate_training_fingerprint(payload)
        validate_resume_config(payload.get("crediticl_config") or {}, current_config)
    model.load_state_dict(payload["state_dict"])
    if optimizer is not None and payload.get("optimizer_state"):
        state = payload["optimizer_state"]
        written_by = payload.get("optimizer_class")
        if isinstance(state, dict) and "kind" in state:
            written_by = "_MuonWithAux"   # the two-optimizer facade used until 28-09-2026
        if written_by is not None and written_by != type(optimizer).__name__:
            raise ValueError(
                f"checkpoint was written by a different optimizer ({written_by}, this run "
                f"builds {type(optimizer).__name__}). Resuming would silently reset or "
                f"misread the optimizer state; start a new run instead."
            )
        optimizer.load_state_dict(state)
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])
    if optimizer is not None:
        from src.train.random_state import restore

        restore(payload.get("random_state"))
    return payload


def prune_checkpoints(ckpt_dir: str | Path, *, save_perm_every: int, max_temp: int) -> list[Path]:
    """Delete the oldest temporary checkpoints beyond `max_temp`.

    A checkpoint is temporary when its step is not a multiple of
    `save_perm_every` — the same rule TabICL uses in `manage_checkpoint`.
    """
    if max_temp <= 0:
        return []
    d = Path(ckpt_dir)
    temps: list[tuple[int, Path]] = []
    for entry in d.iterdir():
        m = STEP_RE.match(entry.name)
        if m:
            step = int(m.group(1))
            if save_perm_every > 0 and step % save_perm_every != 0:
                temps.append((step, entry))
    temps.sort()
    removed = []
    while len(temps) > max_temp:
        _, victim = temps.pop(0)
        try:
            victim.unlink()
            removed.append(victim)
        except OSError:
            pass
    return removed
