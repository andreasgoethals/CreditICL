"""Checkpoint save / resume.

Why this is load-bearing rather than a convenience: the KU Leuven VSC
documentation contains **no Slurm requeue recipe** (the string `requeue` does not
appear in it), and its only documented checkpointing facility is the Torque-era
`csub`/BLCR framework. With a 72 h walltime ceiling, a long run must checkpoint,
resume, and resubmit itself. See `docs/vsc.md` §3.

Naming follows TabICL (`step-<n>.ckpt`) so `get_latest_checkpoint` logic is
familiar, and distinguishes *temporary* checkpoints (frequent, pruned) from
*permanent* ones (kept), as upstream does via `save_temp_every` /
`save_perm_every`.

**What resume does and does not guarantee.** Model, optimizer, scheduler, step
counter and the *main-process* prior RNG are restored exactly. DataLoader worker
RNGs are not: workers are re-spawned on resume, so a resumed run draws a
different — but still reproducible-from-seed — task stream than an uninterrupted
one would have. That is a real limitation and is recorded in the checkpoint as
`resumed_at`, so any run whose stream was interrupted is identifiable rather than
silently assumed pristine. Matched-compute claims are made in *steps and datasets
consumed*, which resume preserves exactly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

STEP_RE = re.compile(r"^step-(\d+)\.ckpt$")


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
    payload = {
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "extra": extra or {},
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
) -> dict[str, Any]:
    # weights_only=False because the payload carries the config dict; the file is
    # one we wrote ourselves on our own filesystem.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])
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
