"""Draw many prior tasks at once, in parallel worker processes.

The prior notebooks (0.2, 0.3) draw about 1,500 tasks each — 500 per prior with the filter on,
then smaller batches per figure — and drawing them one at a time took ~12 minutes a notebook.
`draw` splits a request into one chunk per worker process. Chunk k is drawn with
`PriorRNG(seed, worker_id=k)`, the per-worker seeding the training DataLoader uses, so the
result is a proper draw from the prior; which tasks come out depends on the worker count, as
the draws already did on the process (see the reproducibility caveat in 0.2).

A request smaller than `MIN_PARALLEL` stays in this process, exactly as before: a worker costs
~10 s of imports, which only pays for itself on a large draw — and the tests' small draws keep
their single-process results.

Worker processes are SPAWNED. Notebooks run in a Jupyter kernel — interactively or through
`run_notebooks` — whose main module is not a script, so a worker re-runs nothing; a plain script
calling `draw` needs the usual `if __name__ == "__main__":` guard. Workers run torch on one
thread each — they are many, and tables are small.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

#: Below this many tasks a draw stays in-process. (24, so `correlated_defaults`' 30-task
#: draws per correlation still go parallel.)
MIN_PARALLEL = 24

_POOL: ProcessPoolExecutor | None = None
_POOL_WORKERS = 0


def default_workers() -> int:
    """`CREDITICL_DRAW_WORKERS`, else half the cores (two prior notebooks run side by side), ≤ 8."""
    env = os.environ.get("CREDITICL_DRAW_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _pool(workers: int) -> ProcessPoolExecutor:
    """One pool per process, kept for its lifetime, so the imports are paid once per notebook."""
    global _POOL, _POOL_WORKERS
    if _POOL is None or _POOL_WORKERS != workers:
        if _POOL is not None:
            _POOL.shutdown()
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                    initializer=_one_thread)
        _POOL_WORKERS = workers
    return _POOL


def _one_thread() -> None:
    """Each worker on one torch thread: there are many of them, and the tables are small. Set in
    the WORKER only — an in-process draw leaves the caller's thread count alone."""
    import torch

    torch.set_num_threads(1)


def _draw_chunk(task: str, prior: dict, n: int, seed: int, worker_id: int):
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG

    gen = TaskGenerator(prior, task, PriorRNG(seed, worker_id=worker_id))
    tasks = [gen.sample() for _ in range(n)]
    return tasks, gen.filter.stats, gen.invalid_candidates, gen.filter_fallbacks


def _merge_filter(parts: list[tuple]) -> dict[str, Any]:
    """The filter summary of the whole draw, as `TaskGenerator.filter_summary` reports one."""
    from src.prior.filters import FilterStats

    total = FilterStats()
    invalid = fallbacks = 0
    for _, stats, inv, fb in parts:
        for field in ("attempts", "accepted", "rejected_unpredictable", "rejected_trivial",
                      "rejected_band"):
            setattr(total, field, getattr(total, field) + getattr(stats, field))
        total.pseudo_r2 += stats.pseudo_r2
        total.pvalues += stats.pvalues
        invalid += inv
        fallbacks += fb
    return {**total.summary(), "invalid_candidates": invalid, "filter_fallbacks": fallbacks}


def draw(task: str, prior: dict, n: int, seed: int = 0,
         workers: int | None = None) -> tuple[list[Any], dict[str, Any]]:
    """`n` tasks from `TaskGenerator(prior, task)`, and the draw's filter summary."""
    workers = min(workers or default_workers(), max(1, n))
    if workers <= 1 or n < MIN_PARALLEL:
        parts = [_draw_chunk(task, prior, n, seed, 0)]
    else:
        # About four chunks per worker, so a worker that finishes early takes the next chunk
        # rather than idling while the slowest one ends (a task's cost varies with its graph).
        size = max(4, -(-n // (4 * workers)))
        sizes = [min(size, n - start) for start in range(0, n, size)]
        futures = [_pool(workers).submit(_draw_chunk, task, prior, s, seed, k)
                   for k, s in enumerate(sizes)]
        parts = [f.result() for f in futures]
    return [t for part in parts for t in part[0]], _merge_filter(parts)


def parallel_map(fn: Callable[[Any], Any], items: Iterable[Any],
                 workers: int | None = None) -> list[Any]:
    """`[fn(x) for x in items]`, spread over the draw pool when there are enough items.

    `fn` must be a module-level function (workers import it by name). Results keep the order
    of `items`, so a deterministic `fn` returns exactly what the loop would.
    """
    items = list(items)
    workers = min(workers or default_workers(), max(1, len(items)))
    if workers <= 1 or len(items) < MIN_PARALLEL:
        return [fn(x) for x in items]
    return list(_pool(workers).map(fn, items, chunksize=max(1, len(items) // (4 * workers))))
