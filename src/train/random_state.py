"""Checkpoint stochastic model state using only weights-only-loader-safe types."""
import random

import numpy as np
import torch


def capture() -> dict:
    state = np.random.get_state()  # noqa: NPY002 -- restore upstream code's legacy global RNG
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4])}


def restore(state: dict | None) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    random.setstate(state["python"])
    name, keys, position, gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gauss, cached))  # noqa: NPY002
