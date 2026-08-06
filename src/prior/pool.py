"""Pre-generated pools of synthetic datasets, and the mixture that reads them.

WHY PRE-GENERATE INSTEAD OF GENERATING ON THE FLY

1. **Every arm sees the same original datasets.** The 70-90% that comes from the
   unmodified TabICL prior is *literally the same files* for every arm. So a
   difference between arms cannot come from the luck of the draw — only from the
   credit datasets. When you are hunting a small effect across 48 runs, that
   removes most of the noise.
2. **It is much cheaper.** Generation is CPU-bound (TabICLv2 runs it with
   `--prior_device cpu --n_jobs 16`), so generating on the fly leaves an expensive
   GPU waiting on ExtraTrees fits. Pools are built on CPU nodes, in parallel, and
   the GPU then does nothing but train.
3. **Resume becomes exact.** A resumed run reads the same files, so the task
   stream is identical rather than merely reproducible-from-seed.

LAYOUT — one directory per variant, on project storage:

    prior_cache/
      lgd__original/     the unmodified TabICL prior      <- SHARED by every arm
      lgd__credit_v1/    our credit-targeted datasets
      pd__original/
      pd__credit_v1/
        shard_00000.pt   … each holds `datasets_per_shard` episodes
        manifest.json    counts, the config used, the code version

SHARDS EXIST FOR PARALLELISM. Array task *i* writes shard *i*, so 40,000 datasets
across 20 CPU tasks is 20 independent jobs with no coordination and no locking.

BUDGET. TabICLv2 itself used **≈35M** synthetic datasets (500K+40K+10K steps at
batch 64) against TabICL v1's ≈83M and TabPFNv2's ≈130M. We use 40,000 — 0.1% of
that — because we are running O'Prior's controlled comparison, not reproducing
TabICLv2. Every arm gets exactly the same count, which is what makes the
comparison fair; `verify_pools` checks it rather than trusting it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG
from src.utils.logging_setup import get_logger
from src.utils.paths import prior_cache_dir

#: Bump when the on-disk shard layout changes, so stale pools are rejected.
POOL_VERSION = 1


@dataclass
class PoolManifest:
    """What is in a pool. Written per shard, aggregated by `pool_status`."""

    variant: str
    task: str
    shard_index: int
    n_shards: int
    n_datasets: int
    seed: int
    pool_version: int = POOL_VERSION
    credit_fraction: float = 0.0
    sources: dict[str, int] = field(default_factory=dict)
    filter_stats: dict[str, float] = field(default_factory=dict)
    group_stats: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def variant_dir(task: str, variant: str) -> Path:
    return prior_cache_dir(f"{task}__{variant}")


def _atomic_torch_save(path: Path, obj: Any) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def generate_shard(
    prior_cfg: dict[str, Any],
    task: str,
    variant: str,
    *,
    shard_index: int,
    n_shards: int,
    n_datasets_total: int,
    seed: int = 0,
    force: bool = False,
) -> Path:
    """Generate one shard of a pool and write it.

    The shard's RNG seed is derived from (seed, shard_index), so shards are
    independent but the whole pool is reproducible from one number. Two array tasks
    can never draw the same datasets.
    """
    log = get_logger()
    out_dir = variant_dir(task, variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{shard_index:05d}.pt"
    manifest_path = out_dir / f"shard_{shard_index:05d}.json"

    if shard_path.is_file() and manifest_path.is_file() and not force:
        log.info("[prior] %s shard %d already present, skipping", variant, shard_index)
        return shard_path

    # Split the total as evenly as possible; the first shards absorb the remainder
    # so the pool total is EXACT rather than approximately right.
    base, extra = divmod(n_datasets_total, n_shards)
    n_here = base + (1 if shard_index < extra else 0)

    started = time.time()
    rng = PriorRNG(seed, worker_id=shard_index)
    gen = TaskGenerator(prior_cfg, task, rng)

    log.info(
        "[prior] %s shard %d/%d: generating %d datasets (credit_fraction=%s)",
        variant, shard_index, n_shards, n_here, prior_cfg.get("credit_fraction"),
    )

    episodes: list[dict[str, Any]] = []
    sources = {"base": 0, "credit": 0}
    for i in range(n_here):
        t = gen.sample()
        sources[t.source] = sources.get(t.source, 0) + 1
        # float32 for X and y; anything else doubles the pool for no benefit.
        episodes.append({"X": t.X.contiguous(), "y": t.y.contiguous(), "source": t.source})
        if (i + 1) % 500 == 0:
            log.info("[prior] %s shard %d: %d/%d", variant, shard_index, i + 1, n_here)

    _atomic_torch_save(shard_path, episodes)

    manifest = PoolManifest(
        variant=variant,
        task=task,
        shard_index=shard_index,
        n_shards=n_shards,
        n_datasets=len(episodes),
        seed=seed,
        credit_fraction=float(prior_cfg.get("credit_fraction", 0.0)),
        sources=sources,
        filter_stats=gen.filter_summary(),
        group_stats=gen.group_summary(),
        seconds=round(time.time() - started, 1),
    )
    tmp = manifest_path.parent / f".{manifest_path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)

    mb = shard_path.stat().st_size / 1e6
    log.info(
        "[prior] %s shard %d DONE: %d datasets, %.1f MB, %.0fs, filter rejection %.2f",
        variant, shard_index, len(episodes), mb, manifest.seconds,
        manifest.filter_stats.get("rejection_rate", 0.0),
    )
    return shard_path


def pool_status(task: str, variant: str) -> dict[str, Any]:
    """How complete a pool is. Cheap — reads only the manifests."""
    d = variant_dir(task, variant)
    if not d.is_dir():
        return {"variant": variant, "exists": False, "n_datasets": 0, "shards": 0}

    manifests = sorted(d.glob("shard_*.json"))
    total, expected, srcs = 0, None, {"base": 0, "credit": 0}
    for m in manifests:
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a torn manifest means an unfinished shard
            continue
        if data.get("pool_version") != POOL_VERSION:
            continue
        if not (d / f"shard_{data['shard_index']:05d}.pt").is_file():
            continue  # manifest without payload: incomplete
        total += int(data["n_datasets"])
        expected = int(data["n_shards"])
        for k, v in (data.get("sources") or {}).items():
            srcs[k] = srcs.get(k, 0) + int(v)

    return {
        "variant": variant,
        "task": task,
        "exists": True,
        "path": str(d),
        "shards": len(manifests),
        "shards_expected": expected,
        "complete": expected is not None and len(manifests) == expected,
        "n_datasets": total,
        "sources": srcs,
    }


def verify_pools(task: str, variants: list[str], *, expect: int | None = None) -> dict[str, Any]:
    """Check every pool is complete and that they all hold the SAME count.

    Equal counts are what make "matched compute" true rather than assumed, so this
    is a hard check and not a log line.
    """
    log = get_logger()
    stats = {v: pool_status(task, v) for v in variants}
    counts = {v: s["n_datasets"] for v, s in stats.items()}

    problems = []
    for v, s in stats.items():
        if not s["exists"]:
            problems.append(f"{v}: missing")
        elif not s["complete"]:
            problems.append(f"{v}: incomplete ({s['shards']}/{s['shards_expected']} shards)")
    distinct = set(counts.values())
    if len(distinct) > 1:
        problems.append(f"pools hold DIFFERENT counts {counts} — matched compute would be a fiction")
    if expect is not None and distinct and distinct != {expect}:
        problems.append(f"expected {expect} datasets per pool, got {counts}")

    for v, s in stats.items():
        log.info("[prior] pool %-22s %s datasets, %s shards, complete=%s",
                 v, s["n_datasets"], s["shards"], s.get("complete"))
    if problems:
        for p in problems:
            log.error("[prior] POOL PROBLEM: %s", p)
    return {"stats": stats, "counts": counts, "problems": problems, "ok": not problems}


class PoolReader:
    """Reads episodes from one or more pools, lazily, one shard at a time.

    Shards are loaded on demand and only one is held at a time, so a 40,000-dataset
    pool costs one shard of memory rather than all of it.
    """

    def __init__(self, task: str, variant: str):
        self.task = task
        self.variant = variant
        self.dir = variant_dir(task, variant)
        self.shards = sorted(self.dir.glob("shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(
                f"no shards in {self.dir}. Generate the pool first:\n"
                f"  python scripts/generate_prior.py --config config/{task.upper()}.yaml "
                f"--variant {variant}"
            )
        self._loaded_index: int | None = None
        self._loaded: list[dict[str, Any]] = []

    def _shard_for(self, index: int) -> tuple[int, int]:
        """Map a flat index onto (shard, offset). Assumes near-equal shards."""
        # Cheap and adequate: shard sizes differ by at most one.
        per = max(1, len(self._peek_first_shard()))
        return min(index // per, len(self.shards) - 1), index % per

    def _peek_first_shard(self) -> list[dict[str, Any]]:
        if self._loaded_index != 0:
            self._loaded = torch.load(self.shards[0], weights_only=False)
            self._loaded_index = 0
        return self._loaded

    def sample(self, rng: PriorRNG) -> dict[str, Any]:
        """A uniformly random episode from the pool."""
        shard_i = rng.randint(0, len(self.shards))
        if self._loaded_index != shard_i:
            self._loaded = torch.load(self.shards[shard_i], weights_only=False)
            self._loaded_index = shard_i
        return self._loaded[rng.randint(0, len(self._loaded))]

    def __len__(self) -> int:
        return pool_status(self.task, self.variant)["n_datasets"]


class MixedPoolSampler:
    """Draws episodes from the original pool and ours, per `credit_fraction`.

    This is the training-time counterpart of `TaskGenerator`: same mixture
    semantics, but reading pre-generated files instead of generating live.
    """

    def __init__(
        self,
        task: str,
        credit_fraction: float,
        rng: PriorRNG,
        *,
        original_variant: str = "original",
        credit_variant: str = "credit_v1",
    ):
        self.task = task
        self.credit_fraction = float(credit_fraction)
        self.rng = rng
        self.original = PoolReader(task, original_variant)
        # A pure-control arm needs no credit pool, so do not demand one.
        self.credit = PoolReader(task, credit_variant) if self.credit_fraction > 0 else None
        self.counts = {"base": 0, "credit": 0}

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, str]:
        use_credit = self.credit is not None and self.rng.boolean(self.credit_fraction)
        pool = self.credit if use_credit else self.original
        ep = pool.sample(self.rng)
        source = "credit" if use_credit else "base"
        self.counts[source] += 1
        return ep["X"], ep["y"], source

    def describe(self) -> dict[str, Any]:
        return {
            "credit_fraction": self.credit_fraction,
            "original_pool": len(self.original),
            "credit_pool": len(self.credit) if self.credit else 0,
            "drawn": dict(self.counts),
        }
