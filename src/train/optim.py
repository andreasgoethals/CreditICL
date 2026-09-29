"""Optimizer and LR schedule.

The cosine-with-restarts schedule is transcribed from TabICL's
`train/_optim.py` (`_get_cosine_with_restarts_lr_lambda`), including the
amplitude decay per cycle and the `lr_end` floor.

**MUON, EXACTLY AS TABICLV2.** Upstream's `Trainer.configure_optimizer` (pinned dump,
`src/tabicl/train/_run.py`) builds:

    Muon(param_groups=[dict(params=list(model.parameters()), use_muon=True)],
         lr=config.lr, weight_decay=config.weight_decay, matched_adamw_rms=0.2,
         momentum=config.beta1, nesterov=True, ns_steps=5,
         adamw_betas=(config.beta1, config.beta2), adamw_eps=1e-8,
         use_cautious_wd=config.use_cautious_wd)

and so do we, with the vendored class (`_muon_vendored.py`). Every parameter — biases and
LayerNorm scales included — goes through Muon, and `train.lr` is THE learning rate (stage 1:
`--lr 8e-4`). There is no second rate: `train.muon_lr` is rejected.

*(Until 28-09-2026 this module preferred `torch.optim.Muon`, gave it only the weight matrices
with torch's defaults, and sent the rest to an AdamW at 3e-4. torch's default step scaling is
Keller's `sqrt(max(1, A/B))`, upstream's is Moonlight's `0.2*sqrt(max(A, B))`: the matrices
took steps 4.8x smaller than upstream's at the same nominal rate, with momentum 0.95 instead
of 0.9. Experiment 1 ran that way. See docs/AGENTS_MEMORY.md, 28-09-2026.)*

Muon's Newton-Schulz iteration is a chain of small matmuls per matrix per step, so it costs
more per step than AdamW on a big card (measured 16-08-2026; `scripts/benchmark_gpu.py`).

AdamW remains available (`optimizer: adamw`), and upstream supports it too (`--muon False`).
Pretraining from scratch (Exp1, Exp3) uses MUON, because that is how the released weights
were made. Continued pretraining (Exp2) currently uses ADAMW, following the continued-
pretraining recipes it takes its learning rates and L2-SP from (Real-TabPFN, TabPFN-Wide,
and upstream's own `_finetune` module) — a design choice, open for revisiting now that Muon's
`lr` is a single, sweepable rate. Whichever is chosen is held FIXED across the arms of an
experiment, so it changes absolute performance, never the contrast.

Upstream's own caveat: the released checkpoints were trained *without* cautious weight decay
even though the paper reports using it (the flag was left unwired), so it defaults to off.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW or Muon, per `train.optimizer`. Muon is built exactly as upstream builds it."""
    name = str(cfg.get("optimizer", "adamw")).lower()
    # Only trainable parameters, so a frozen fine-tune does not carry optimizer
    # state for weights it never updates. TabICL filters the same way:
    # `params = [p for p in self.model_.parameters() if p.requires_grad]`.
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters — check the freeze strategy")

    betas = (float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95)))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    lr = float(cfg.get("lr", 3e-4))

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    if name == "muon":
        if "muon_lr" in cfg:
            raise ValueError(
                "train.muon_lr no longer exists: under Muon, train.lr IS Muon's rate, as in "
                "upstream (`--muon True --lr 8e-4`). Our old optimizer had two rates — Muon on "
                "the matrices, an auxiliary AdamW on the rest — which upstream never had. Move "
                "the value to train.lr and delete train.muon_lr."
            )
        from ._muon_vendored import Muon

        # Upstream's `Trainer.configure_optimizer`, argument for argument. ONE group, every
        # parameter, `use_muon=True`: without the flag the vendored class silently routes the
        # group to its internal AdamW branch.
        return Muon(
            param_groups=[dict(params=params, use_muon=True)],
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=0.2,
            momentum=betas[0],
            nesterov=True,
            ns_steps=5,
            adamw_betas=betas,
            adamw_eps=1e-8,
            use_cautious_wd=bool(cfg.get("cautious_weight_decay", False)),
        )

    raise ValueError(f"optimizer={name!r} is not implemented; use 'adamw' or 'muon'")


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
    # A FLOAT, as upstream's `get_scheduler` computes it (`config.max_steps * warmup_proportion`).
    # Rounding it down (as until 28-09-2026) gave 0 warm-up steps whenever the product was below
    # 1, and a full-rate first step where upstream takes a zero-rate one.
    warmup_steps = max_steps * warmup_proportion
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
