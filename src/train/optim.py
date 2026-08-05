"""Optimizer and LR schedule.

The cosine-with-restarts schedule is transcribed from TabICL's
`train/_optim.py` (`_get_cosine_with_restarts_lr_lambda`), including the
amplitude decay per cycle and the `lr_end` floor.

**Deviation: AdamW, not Muon.** TabICLv2 credits part of its gain to the Muon
optimizer, and its reference scripts pass `--muon True`. We use AdamW, which
upstream also supports (`--muon False`) as a first-class alternative. Reasons:

* Muon's correctness depends on details (Newton-Schulz iteration count, the
  `matched_adamw_rms=0.2` scaling, cautious weight decay) that we cannot verify
  without the implementation, and a subtly wrong optimizer would degrade every
  arm *equally and invisibly* — the worst possible failure mode for a controlled
  comparison.
* The optimizer is held fixed across arms, so it is not the experimental
  variable. Absolute performance suffers; the prior contrast does not.

Note upstream's own caveat about Muon: the released checkpoints were trained
*without* cautious weight decay even though the paper reports using it, because
the flag was left unwired. Another reason not to guess at a reimplementation.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "adamw")).lower()
    if name != "adamw":
        raise ValueError(
            f"optimizer={name!r} is not implemented. Only 'adamw' is available; see this module's "
            "docstring for why Muon is deliberately not reimplemented here."
        )
    # Only trainable parameters, so a frozen fine-tune does not carry optimizer
    # state for weights it never updates. TabICL filters the same way:
    # `params = [p for p in self.model_.parameters() if p.requires_grad]`.
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters — check the freeze strategy")
    return torch.optim.AdamW(
        params,
        lr=float(cfg.get("lr", 3e-4)),
        betas=(float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95))),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )


def _cosine_with_restarts_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    amplitude_decay: float,
    lr_init: float,
    lr_end: float,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return lr_end / lr_init  # LambdaLR multiplies by lr_init

    cycle_progress = (float(num_cycles) * progress) % 1.0
    current_cycle = int(float(num_cycles) * progress)
    amplitude = amplitude_decay**current_cycle

    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
    current_lr = lr_end + (lr_init - lr_end) * cosine_factor * amplitude
    return current_lr / lr_init


def _constant_lambda(current_step: int, *, num_warmup_steps: int) -> float:
    """Flat LR after warmup. TabICL v1 stage 3 uses `--scheduler constant`, which
    is the right choice for a frozen fine-tune: a decaying schedule over very few
    steps mostly just shrinks the update you were trying to make."""
    if num_warmup_steps > 0 and current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], max_steps: int) -> LambdaLR:
    warmup_proportion = float(cfg.get("warmup_proportion", 0.01))
    warmup_steps = int(max_steps * warmup_proportion)
    lr_init = float(cfg.get("lr", 3e-4))

    kind = str(cfg.get("scheduler", "cosine_with_restarts")).lower()
    if kind == "constant":
        return LambdaLR(optimizer, partial(_constant_lambda, num_warmup_steps=warmup_steps))
    if kind != "cosine_with_restarts":
        raise ValueError(f"unknown scheduler {kind!r}; expected 'cosine_with_restarts' or 'constant'")

    lr_lambda = partial(
        _cosine_with_restarts_lambda,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        num_cycles=int(cfg.get("cosine_num_cycles", 1)),
        amplitude_decay=float(cfg.get("cosine_amplitude_decay", 1.0)),
        lr_init=lr_init,
        lr_end=float(cfg.get("cosine_lr_end", 1e-7)),
    )
    return LambdaLR(optimizer, lr_lambda)
