"""Deterministic, checkpointable randomness for the prior.

Why this exists rather than using global `np.random` / `torch` RNGs the way
NanoTabICL's `prior.py` does:

1. **Resume correctness.** VSC gives us a 72 h walltime ceiling and no Slurm
   requeue facility, so long runs *must* checkpoint and resume. If the prior's
   RNG state is not part of the checkpoint, a resumed run draws a different task
   stream and the arm is no longer the arm it was. Global RNG state is awkward to
   capture per DataLoader worker; an explicit object is not.
2. **Worker independence.** Each DataLoader worker must draw a *different* task
   stream, but reproducibly so. Seeds are derived as (base_seed, worker_id).

Every sampling function in this package takes a `PriorRNG` as its first
argument. That is deliberately verbose; it is also the only way to be sure no
hidden global state leaks between arms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import torch

T = TypeVar("T")

# Large odd multiplier to decorrelate the numpy and torch streams from the same
# base seed without them ever coinciding.
_TORCH_SEED_OFFSET = 0x9E3779B97F4A7C15


class PriorRNG:
    """Paired numpy + torch generators with saveable state."""

    __slots__ = ("np", "torch", "_seed")

    def __init__(self, seed: int, worker_id: int = 0):
        self._seed = int(seed) + 1_000_003 * int(worker_id)
        self.np = np.random.default_rng(self._seed)
        self.torch = torch.Generator(device="cpu")
        self.torch.manual_seed((self._seed + _TORCH_SEED_OFFSET) % (2**63 - 1))

    @property
    def seed(self) -> int:
        return self._seed

    # -- state, for checkpointing -------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"seed": self._seed, "np": self.np.bit_generator.state, "torch": self.torch.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._seed = state["seed"]
        self.np.bit_generator.state = state["np"]
        self.torch.set_state(state["torch"])

    # -- scalars -------------------------------------------------------------
    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self.np.uniform(low, high))

    def lognum(self, low: float, high: float) -> float:
        """Log-uniform draw in [low, high]. NanoTabICL's `randlognum`."""
        return float(np.exp(self.np.uniform(np.log(low), np.log(high))))

    def logint(self, low: float, high: float) -> int:
        """Log-uniform integer in [low, high-1]. NanoTabICL's `randlogint`."""
        return int(np.clip(np.floor(self.lognum(low, high)).astype(int), low, high - 1))

    def randint(self, low: int, high: int) -> int:
        """Uniform integer in [low, high-1]."""
        return int(self.np.integers(low, high))

    def boolean(self, p: float = 0.5) -> bool:
        """NanoTabICL's `randbool` when p=0.5."""
        return bool(self.np.uniform() < p)

    def choice(self, options: Sequence[T]) -> T:
        """NanoTabICL's `randchoice` — uniform over a sequence."""
        return options[self.randint(0, len(options))]

    def weighted_choice(self, options: Sequence[T], weights: Sequence[float]) -> T:
        w = np.asarray(weights, dtype=float)
        total = w.sum()
        if total <= 0:
            return self.choice(options)
        return options[int(self.np.choice(len(options), p=w / total))]

    # -- tensors -------------------------------------------------------------
    # `*_like` helpers exist because torch's `randn_like` / `rand_like` do not
    # accept a generator, so they would silently escape to the global RNG.
    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.torch)

    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.torch)

    def randn_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape, generator=self.torch, dtype=x.dtype)

    def rand_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand(x.shape, generator=self.torch, dtype=x.dtype)

    def randperm(self, n: int) -> torch.Tensor:
        return torch.randperm(n, generator=self.torch)

    def torch_randint(self, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(high, size=shape, generator=self.torch)

    def cauchy(self, n: int) -> torch.Tensor:
        return torch.empty(n).cauchy_(generator=self.torch)

    def standard_cauchy(self) -> float:
        return float(self.np.standard_cauchy())

    def multinomial(self, probs: torch.Tensor, num_samples: int, replacement: bool = True) -> torch.Tensor:
        return torch.multinomial(probs, num_samples=num_samples, replacement=replacement, generator=self.torch)
