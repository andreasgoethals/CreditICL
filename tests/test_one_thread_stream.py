"""The task stream does not depend on how many threads generate it (30-09-2026).

A multi-threaded reduction sums in another order; the last bits differ; a later threshold (a class
boundary, the predictability filter) flips; a different table comes out. DataLoader workers run on
one thread and the main process on many, so the stream used to depend on `num_workers` — a requeued
arm on a node with another CPU count would have continued on another stream. Every task is now
generated on one thread (`rng.one_thread`); the cluster check (`exp0_checks.py determinism`) also
compares 0 against 2 workers and a fresh process.
"""

from __future__ import annotations

import copy
import hashlib

import pytest

torch = pytest.importorskip("torch")

from src.prior.dataset import PriorBatchDataset  # noqa: E402
from src.prior.generator import TaskGenerator  # noqa: E402
from src.prior.rng import PriorRNG, one_thread  # noqa: E402
from src.utils.config import expand_with_seeds, load  # noqa: E402


def _prior(track: str) -> dict:
    prior = copy.deepcopy(expand_with_seeds(load(f"config/Exp1_{track.upper()}.yaml"))[0]["prior"])
    prior["credit_fraction"] = 0.5
    prior["n_rows_range"] = [256, 256]
    return prior


def _digest(batch) -> str:
    h = hashlib.sha256()
    for tensor in batch:
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_a_batch_is_the_same_with_one_thread_or_many(track):
    prior = _prior(track)
    ds = PriorBatchDataset(prior, track, 4, seed=0)
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        many = _digest(ds.batch(TaskGenerator(prior, track, PriorRNG(0), stream_seed=0), 0))
        torch.set_num_threads(1)
        one = _digest(ds.batch(TaskGenerator(prior, track, PriorRNG(0), stream_seed=0), 0))
    finally:
        torch.set_num_threads(before)
    assert many == one


def test_the_thread_counts_are_restored():
    from threadpoolctl import threadpool_info

    before = torch.get_num_threads()
    pools = [p["num_threads"] for p in threadpool_info()]
    with one_thread():
        assert torch.get_num_threads() == 1
        assert all(p["num_threads"] == 1 for p in threadpool_info())
    assert torch.get_num_threads() == before
    assert [p["num_threads"] for p in threadpool_info()] == pools
