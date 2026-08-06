"""Pre-generated prior pools.

The tests that matter here are about **fairness of the comparison**: every arm must
draw its original-prior share from the same files, and every pool must hold the same
number of datasets. If either breaks, "matched compute" becomes a fiction and no
arm-to-arm difference means anything.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch


@pytest.fixture
def tiny_prior(lgd_cfg):
    p = copy.deepcopy(lgd_cfg["prior"])
    p.update(
        {
            "n_rows_range": [64, 80],
            "n_features_range": [3, 5],
            "max_features": 8,
            "n_nodes_range": [2, 3],
            "max_filter_attempts": 4,
        }
    )
    return p


@pytest.fixture
def pool_env(tmp_path, monkeypatch):
    """Point the pool cache at a temp dir and reload the modules that read it."""
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as paths

    importlib.reload(paths)
    import src.prior.pool as pool

    importlib.reload(pool)
    yield pool
    importlib.reload(paths)
    importlib.reload(pool)


def _build(pool, prior, variant, credit_fraction, *, n=12, shards=3, seed=0):
    cfg = copy.deepcopy(prior)
    cfg["credit_fraction"] = credit_fraction
    for i in range(shards):
        pool.generate_shard(
            cfg, "lgd", variant, shard_index=i, n_shards=shards, n_datasets_total=n, seed=seed
        )


def test_shards_sum_to_the_exact_requested_total(pool_env, tiny_prior):
    """13 datasets over 4 shards must be exactly 13, not 12 or 16. Uneven splits are
    where an off-by-one turns into unequal pools."""
    pool = pool_env
    _build(pool, tiny_prior, "original", 0.0, n=13, shards=4)
    assert pool.pool_status("lgd", "original")["n_datasets"] == 13


def test_pools_are_verified_equal(pool_env, tiny_prior):
    """The check that makes matched compute true instead of assumed."""
    pool = pool_env
    _build(pool, tiny_prior, "original", 0.0, n=12)
    _build(pool, tiny_prior, "credit_v1", 1.0, n=12)
    report = pool.verify_pools("lgd", ["original", "credit_v1"], expect=12)
    assert report["ok"], report["problems"]
    assert set(report["counts"].values()) == {12}


def test_unequal_pools_are_rejected(pool_env, tiny_prior):
    """If one pool is bigger, the comparison is unfair and must fail loudly."""
    pool = pool_env
    _build(pool, tiny_prior, "original", 0.0, n=12)
    _build(pool, tiny_prior, "credit_v1", 1.0, n=8)
    report = pool.verify_pools("lgd", ["original", "credit_v1"])
    assert not report["ok"]
    assert any("DIFFERENT counts" in p for p in report["problems"])


def test_incomplete_pool_is_detected(pool_env, tiny_prior):
    """A run killed part-way leaves missing shards; that must not read as complete."""
    pool = pool_env
    cfg = copy.deepcopy(tiny_prior)
    cfg["credit_fraction"] = 0.0
    pool.generate_shard(cfg, "lgd", "original", shard_index=0, n_shards=3, n_datasets_total=12, seed=0)
    status = pool.pool_status("lgd", "original")
    assert status["shards"] == 1 and status["shards_expected"] == 3
    assert status["complete"] is False


def test_variant_pools_contain_only_their_own_source(pool_env, tiny_prior):
    """`original` must be 100% base and `credit_v1` 100% credit. A pool that is
    secretly a mixture would make the mixture lever meaningless."""
    pool = pool_env
    _build(pool, tiny_prior, "original", 0.0)
    _build(pool, tiny_prior, "credit_v1", 1.0)
    assert pool.pool_status("lgd", "original")["sources"]["credit"] == 0
    assert pool.pool_status("lgd", "credit_v1")["sources"]["base"] == 0


def test_shards_are_independent_draws(pool_env, tiny_prior):
    """Two array tasks must not generate the same datasets."""
    pool = pool_env
    _build(pool, tiny_prior, "original", 0.0, n=12, shards=3)
    d = pool.variant_dir("lgd", "original")
    a = torch.load(d / "shard_00000.pt", weights_only=False)
    b = torch.load(d / "shard_00001.pt", weights_only=False)
    assert not torch.equal(a[0]["y"], b[0]["y"]), "different shards must differ"


def test_regenerating_is_idempotent(pool_env, tiny_prior):
    """Re-running an array task must not duplicate or corrupt a finished shard."""
    pool = pool_env
    _build(pool, tiny_prior, "original", 0.0)
    before = pool.pool_status("lgd", "original")["n_datasets"]
    _build(pool, tiny_prior, "original", 0.0)  # again
    assert pool.pool_status("lgd", "original")["n_datasets"] == before


def test_mixture_honours_credit_fraction(pool_env, tiny_prior):
    pool = pool_env
    from src.prior.rng import PriorRNG

    _build(pool, tiny_prior, "original", 0.0)
    _build(pool, tiny_prior, "credit_v1", 1.0)
    s = pool.MixedPoolSampler("lgd", 0.25, PriorRNG(0))
    for _ in range(400):
        s.sample()
    drawn = s.describe()["drawn"]
    frac = drawn["credit"] / sum(drawn.values())
    assert 0.15 < frac < 0.35, f"asked 0.25, drew {frac:.2f}"


def test_control_arm_needs_no_credit_pool(pool_env, tiny_prior):
    """credit_fraction=0 must work with only the original pool present, so the
    control arm can run before any credit pool has been built."""
    pool = pool_env
    from src.prior.rng import PriorRNG

    _build(pool, tiny_prior, "original", 0.0)
    s = pool.MixedPoolSampler("lgd", 0.0, PriorRNG(0))
    assert s.credit is None
    for _ in range(20):
        _, _, src = s.sample()
        assert src == "base"


def test_missing_pool_raises_with_the_fix_in_the_message(pool_env, tiny_prior):
    pool = pool_env
    with pytest.raises(FileNotFoundError, match="generate_prior.py"):
        pool.PoolReader("lgd", "does_not_exist")


def test_pooled_episodes_are_finite_and_bounded(pool_env, tiny_prior):
    """LGD targets from our path must still be in [0,1] after the disk round trip."""
    pool = pool_env
    from src.prior.rng import PriorRNG

    cfg = copy.deepcopy(tiny_prior)
    cfg["credit"]["target"]["target_scaling"] = "none"
    _build(pool, cfg, "credit_v1", 1.0)
    reader = pool.PoolReader("lgd", "credit_v1")
    rng = PriorRNG(0)
    for _ in range(20):
        ep = reader.sample(rng)
        assert torch.isfinite(ep["X"]).all() and torch.isfinite(ep["y"]).all()
        assert float(ep["y"].min()) >= 0.0 and float(ep["y"].max()) <= 1.0


def test_training_from_a_pool(pool_env, tiny_prior, lgd_cfg, tmp_path):
    """The end-to-end path: pools on disk, model trains from them."""
    pool = pool_env
    from src.train.loop import Trainer

    _build(pool, tiny_prior, "original", 0.0, n=12)
    _build(pool, tiny_prior, "credit_v1", 1.0, n=12)

    cfg = copy.deepcopy(lgd_cfg)
    cfg["prior"] = copy.deepcopy(tiny_prior)
    cfg["prior"]["credit_fraction"] = 0.3
    cfg["prior"]["pool"] = {"source": "pool"}
    trainer = Trainer(cfg, tmp_path / "o", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=tmp_path / "l")
    summary = trainer.train()
    assert summary["steps"] == cfg["train"]["max_steps"]
    assert summary["datasets_seen"] > 0


def test_unknown_pool_source_is_rejected(lgd_cfg):
    from src.prior.dataset import PriorBatchDataset

    cfg = copy.deepcopy(lgd_cfg["prior"])
    cfg["pool"] = {"source": "magic"}
    with pytest.raises(ValueError, match="must be 'generate' or 'pool'"):
        PriorBatchDataset(cfg, "lgd", batch_size=2, seed=0)
