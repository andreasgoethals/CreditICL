"""`src/visualize/draw.py` — the parallel task draws that make the prior notebooks fast.

The prior notebooks drew ~1,500 tasks each, one at a time, for ~12 minutes a notebook. `draw`
spreads a large request over worker processes; a small one must stay exactly what it was.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

from src.prior.generator import TaskGenerator  # noqa: E402
from src.prior.rng import PriorRNG  # noqa: E402
from src.utils.config import expand_with_seeds, load  # noqa: E402
from src.visualize import draw as D  # noqa: E402


def _prior(track: str = "PD") -> dict:
    prior = expand_with_seeds(load(f"config/Exp1_{track}.yaml"))[0]["prior"]
    prior["n_rows_range"] = [128, 128]
    prior["filter"]["mode"] = "off"
    return prior


def test_a_small_draw_is_the_single_process_draw_it_always_was():
    """Below `MIN_PARALLEL` nothing is spawned, and the tasks are bit-identical to drawing them
    one by one — so every small draw in the tests and the plotting helpers is unchanged."""
    prior, n = _prior(), D.MIN_PARALLEL - 1
    gen = TaskGenerator(prior, "pd", PriorRNG(3))
    expected = [gen.sample() for _ in range(n)]
    got, summary = D.draw("pd", prior, n, seed=3)
    assert len(got) == n
    assert all(torch.equal(a.X, b.X) and torch.equal(a.y, b.y) for a, b in zip(expected, got))
    assert summary["attempts"] >= n


@pytest.mark.slow
def test_a_large_draw_is_split_across_workers():
    """Each worker draws its own chunk (`PriorRNG(seed, worker_id=k)`, as the training DataLoader
    seeds its workers); together they return exactly `n` tasks and one merged filter summary."""
    prior, n = _prior(), D.MIN_PARALLEL + 3
    tasks, summary = D.draw("pd", prior, n, seed=0, workers=2)
    assert len(tasks) == n
    assert {tuple(t.X.shape) for t in tasks} == {(128, prior["max_features"])}
    assert summary["accepted"] == n  # filter off: every valid candidate is accepted


@pytest.mark.slow
def test_parallel_map_keeps_order_and_values():
    items = list(range(-D.MIN_PARALLEL - 5, 0))
    assert D.parallel_map(abs, items, workers=2) == [abs(x) for x in items]


def test_the_worker_count_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("CREDITICL_DRAW_WORKERS", "3")
    assert D.default_workers() == 3
    monkeypatch.delenv("CREDITICL_DRAW_WORKERS")
    assert 1 <= D.default_workers() <= 8
