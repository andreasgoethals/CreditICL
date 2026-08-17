"""Optimizer and LR schedule.

The cosine-with-restarts schedule is transcribed from TabICL's
`train/_optim.py` (`_get_cosine_with_restarts_lr_lambda`), including the
amplitude decay per cycle and the `lr_end` floor.

**We use MUON**, matching TabICLv2, whose reference scripts pass `--muon True` and which
credits part of its gain to it. `config/Exp1_*.yaml` sets `optimizer: muon`, and the
implementation is vendored from the pinned dump rather than reimplemented, so the
Newton-Schulz iteration count and the `matched_adamw_rms` scaling are upstream's own.

*(This docstring used to claim the opposite — that we had deviated to AdamW. That was true
before Muon was vendored and was never updated, so the module described the wrong optimizer
for weeks. `test_optim_docstring_matches_the_configs` now ties the two together.)*

**MUON IS THE REASON THE B200 LOOKED SLOW.** Measured 16-08-2026: with `optimizer: muon`,
B200 training ran at 0.53 steps/s while the free RTX 5000 Ada ran at 6.6 — but a benchmark of
the identical model, prior and data loader using plain SGD had the B200 1.8x *faster* end to
end (8.14 vs 4.50 steps/s). The optimiser was the only difference. Muon orthogonalises every
weight matrix with a Newton-Schulz iteration — a chain of small matmuls per matrix per step,
which is latency-bound, not throughput-bound, so a card with 1,392 bf16 TFLOP/s can still lose
badly. `scripts/benchmark_gpu.py` has a row for it.

AdamW remains available (`optimizer: adamw`), and upstream supports it too (`--muon False`).
Whichever is chosen is held FIXED across arms, so it is never the experimental variable:
it changes absolute performance, not the prior contrast.

Note upstream's own caveat: the released checkpoints were trained *without* cautious weight
decay even though the paper reports using it, because the flag was left unwired.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


def _split_muon_params(model: torch.nn.Module) -> tuple[list, list]:
    """Split into the matrices Muon orthogonalises and everything else.

    Muon only applies to parameters that are genuinely 2-D *matrices* — the linear and
    attention weights. Biases, LayerNorm scales and embedding lookups are 1-D (or are
    used as lookups rather than as linear maps), and orthogonalising them is undefined.
    Every Muon implementation therefore runs AdamW on that remainder, and so do we.
    """
    matrices, others = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_embedding = isinstance(module, torch.nn.Embedding)
            if p.ndim == 2 and not is_embedding and name != "bias":
                matrices.append(p)
            else:
                others.append(p)
    return matrices, others


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW or Muon, per `train.optimizer`.

    TabICLv2 uses Muon and credits part of its gain to it, so matching them means
    offering it. See this module's docstring for what is and is not verified.
    """
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
        muon_cls = _resolve_muon()
        matrices, others = _split_muon_params(model)
        if not matrices:
            raise ValueError("optimizer='muon' but the model has no 2-D weight matrices")
        # TabICLv2 uses 8e-4 for Muon against 1e-4 for AdamW in TabICL v1 — Muon
        # prefers a markedly higher rate, so `train.muon_lr` is separate rather than
        # reusing `lr`. Passing an AdamW-scale rate to Muon just wastes the run.
        muon_lr = float(cfg.get("muon_lr", 8e-4))
        muon = muon_cls(matrices, lr=muon_lr, weight_decay=weight_decay)
        aux = (
            torch.optim.AdamW(others, lr=lr, betas=betas, weight_decay=weight_decay)
            if others
            else None
        )
        return _MuonWithAux(muon, aux)

    raise ValueError(f"optimizer={name!r} is not implemented; use 'adamw' or 'muon'")


class _MuonWithAux(torch.optim.Optimizer):
    """Muon on the weight matrices, AdamW on everything else, as one optimizer.

    `torch.optim.Muon` raises on any parameter that is not 2-D ("Muon only supports 2D
    parameters"), so biases, LayerNorm scales and embeddings need a second optimizer.
    Every Muon deployment does this; the pairing is part of the method, not a
    workaround.

    Presented as a single `Optimizer` so the training loop, the LR scheduler and
    checkpoint save/load need no special case. `param_groups` is a concatenated view,
    which is what `LambdaLR` walks to scale learning rates — so the cosine schedule
    reaches both halves. Both keep their own base LR, so the 8x gap between the Muon
    rate and the AdamW rate survives scheduling.
    """

    def __init__(self, muon: torch.optim.Optimizer, aux: torch.optim.Optimizer | None):
        self.muon = muon
        self.aux = aux
        self._opts = [o for o in (muon, aux) if o is not None]
        # No super().__init__: this is a facade over real optimizers, and calling it
        # would create a third, empty set of param groups to keep in sync.
        self.param_groups = [g for o in self._opts for g in o.param_groups]
        self.defaults = dict(getattr(muon, "defaults", {}))

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # noqa: ANN001 — matches torch's signature
        loss = closure() if closure is not None else None
        for o in self._opts:
            o.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "muon_with_aux",
            "muon": self.muon.state_dict(),
            "aux": self.aux.state_dict() if self.aux is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "muon_with_aux":
            raise ValueError(
                "checkpoint was written by a different optimizer. Resuming a run with a "
                "changed train.optimizer would silently reset the optimizer state; "
                "start a new run instead."
            )
        self.muon.load_state_dict(state["muon"])
        if self.aux is not None and state["aux"] is not None:
            self.aux.load_state_dict(state["aux"])
        self.param_groups = [g for o in self._opts for g in o.param_groups]


def _resolve_muon():
    """Find a Muon implementation, preferring torch's own.

    Deliberately not reimplemented here. Muon's correctness depends on details
    (Newton-Schulz iteration count, the `0.2*sqrt(max(n,m))` Moonlight scaling,
    cautious weight decay) that are easy to get subtly wrong, and a subtly wrong
    optimizer degrades every arm *equally and invisibly* — the worst failure mode for
    a controlled comparison. A maintained implementation removes that risk; writing
    our own would reintroduce it.
    """
    if hasattr(torch.optim, "Muon"):  # torch >= 2.9 ships it in core
        return torch.optim.Muon
    try:
        from muon import MuonWithAuxAdam  # the reference `muon` package

        return MuonWithAuxAdam
    except ImportError:
        pass
    try:
        from pytorch_optimizer import Muon  # Schaipp's collection, cited by the paper

        return Muon
    except ImportError:
        pass
    # UPSTREAM'S OWN, vendored from the pinned dump. This is the branch the cluster takes:
    # VSC runs torch 2.8 (no `torch.optim.Muon`) and the published `tabicl` wheel does not
    # ship the training package, so before this existed every cluster run died at optimizer
    # construction. It is also the strictly better fallback — the exact optimizer that
    # produced the released TabICLv2 checkpoints, rather than a second implementation that
    # merely ought to agree.
    from ._muon_vendored import Muon as VendoredMuon

    return VendoredMuon


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
