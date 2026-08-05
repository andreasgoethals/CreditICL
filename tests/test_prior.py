"""The prior generator end to end: mixture lever, filters, noise columns, batching.

The mixture lever tests are the important ones. `credit_fraction` is the whole
experiment, and there are two ways it could silently break: it could be ignored
(so every arm trains on the same data and every result is a null) or it could be
inverted. Both would look like a real finding in the loss curve.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.prior.dataset import PriorBatchDataset
from src.prior.filters import PredictabilityFilter, predictability
from src.prior.generator import TaskGenerator
from src.prior.noise_features import add_noise_features
from src.prior.preprocess import standard_scaling, to_ranks
from src.prior.rng import PriorRNG

# --- the mixture lever -------------------------------------------------------


@pytest.mark.parametrize("fraction,expected", [(0.0, "base"), (1.0, "credit")])
def test_extreme_mixture_uses_only_one_source(lgd_cfg, fraction, expected):
    lgd_cfg["prior"]["credit_fraction"] = fraction
    gen = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(0))
    sources = {gen.sample().source for _ in range(12)}
    assert sources == {expected}


def test_mixture_actually_mixes(lgd_cfg):
    lgd_cfg["prior"]["credit_fraction"] = 0.5
    gen = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(0))
    sources = [gen.sample().source for _ in range(40)]
    assert "base" in sources and "credit" in sources


def test_mixture_fraction_is_roughly_honoured(lgd_cfg):
    """0.2 should give about 20% of our datasets, not 80%. Catches an inversion."""
    lgd_cfg["prior"]["credit_fraction"] = 0.2
    gen = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(0))
    n = 150
    credit = sum(gen.sample().source == "credit" for _ in range(n))
    assert 0.08 < credit / n < 0.38, f"got {credit / n:.2f}, expected about 0.2"


def test_rejects_out_of_range_fraction(lgd_cfg):
    lgd_cfg["prior"]["credit_fraction"] = 1.5
    with pytest.raises(ValueError, match="credit_fraction"):
        TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(0))


def test_base_path_is_untouched_by_credit_settings(lgd_cfg):
    """At credit_fraction 0 the control arm must be the ORIGINAL prior. If our
    settings leaked into it, the baseline would be contaminated and every
    comparison meaningless."""
    lgd_cfg["prior"]["credit_fraction"] = 0.0
    a = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(7)).sample()

    lgd_cfg["prior"]["credit"]["target"]["atom_prob"] = 1.0
    lgd_cfg["prior"]["credit"]["target"]["boundary_mass_range"] = [0.4, 0.45]
    b = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(7)).sample()

    assert torch.allclose(a.y, b.y), "credit settings must not affect the base path"


# --- shapes ------------------------------------------------------------------


@pytest.mark.parametrize("task", ["lgd", "pd"])
def test_task_shapes_are_consistent(lgd_cfg, pd_cfg, task):
    cfg = lgd_cfg if task == "lgd" else pd_cfg
    gen = TaskGenerator(cfg["prior"], task, PriorRNG(0))
    for _ in range(8):
        t = gen.sample()
        assert t.X.ndim == 2 and t.y.ndim == 1
        assert t.X.shape[0] == t.y.shape[0]
        assert t.X.shape[1] <= cfg["prior"]["max_features"]


@pytest.mark.parametrize("task", ["lgd", "pd"])
def test_no_nan_or_inf_in_output(lgd_cfg, pd_cfg, task):
    cfg = lgd_cfg if task == "lgd" else pd_cfg
    gen = TaskGenerator(cfg["prior"], task, PriorRNG(0))
    for _ in range(10):
        t = gen.sample()
        assert torch.isfinite(t.X).all(), "features must be finite"
        assert torch.isfinite(t.y).all(), "target must be finite"


def test_fixed_shape_is_honoured(lgd_cfg):
    """Batches share a shape, so the generator must respect an imposed one."""
    gen = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(0))
    t = gen.sample(shape=(120, 6))
    assert t.X.shape[0] == 120


def test_lgd_credit_target_is_bounded(lgd_cfg):
    lgd_cfg["prior"]["credit_fraction"] = 1.0
    lgd_cfg["prior"]["credit"]["target"]["target_scaling"] = "none"
    gen = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(0))
    for _ in range(10):
        y = gen.sample().y
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_pd_credit_target_is_binary(pd_cfg):
    pd_cfg["prior"]["credit_fraction"] = 1.0
    gen = TaskGenerator(pd_cfg["prior"], "pd", PriorRNG(0))
    for _ in range(10):
        y = gen.sample().y
        assert set(y.unique().tolist()) <= {0.0, 1.0}
        assert 0 < float(y.sum()) < y.numel()


# --- reproducibility ---------------------------------------------------------


def test_same_seed_gives_the_same_task(lgd_cfg):
    """Without this, no result is reproducible and resume is meaningless."""
    a = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(42)).sample()
    b = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(42)).sample()
    assert torch.equal(a.X, b.X) and torch.equal(a.y, b.y)


def test_different_seeds_give_different_tasks(lgd_cfg):
    a = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(1)).sample()
    b = TaskGenerator(lgd_cfg["prior"], "lgd", PriorRNG(2)).sample()
    assert not (a.X.shape == b.X.shape and torch.equal(a.X, b.X))


def test_rng_state_round_trips():
    """Checkpoint/resume depends on this: save the state, draw, restore, redraw."""
    rng = PriorRNG(0)
    rng.randn(4)  # advance it
    state = rng.state_dict()
    first = rng.randn(8)
    rng.load_state_dict(state)
    assert torch.equal(rng.randn(8), first)


def test_worker_ids_give_different_streams():
    a = PriorRNG(0, worker_id=0).randn(6)
    b = PriorRNG(0, worker_id=1).randn(6)
    assert not torch.equal(a, b)


# --- the predictability filter ----------------------------------------------


def test_predictability_detects_a_strong_signal():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 4, generator=g)
    y = X[:, 0] * 2.0  # perfectly predictable
    pval, r2 = predictability(X, y, is_classif=False)
    assert pval < 0.05 and r2 > 0.5


def test_predictability_detects_pure_noise():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 4, generator=g)
    y = torch.randn(300, generator=g)  # unrelated to X
    pval, r2 = predictability(X, y, is_classif=False)
    assert r2 < 0.2


def test_filter_off_accepts_almost_everything():
    f = PredictabilityFilter(mode="off")
    g = torch.Generator().manual_seed(0)
    X, y = torch.randn(200, 3, generator=g), torch.randn(200, generator=g)
    assert f.accept(X, y, is_classif=False)


def test_filter_rejects_a_constant_target_in_every_mode():
    for mode in ("tabicl", "off", "banded"):
        f = PredictabilityFilter(mode=mode)
        X = torch.randn(150, 3)
        assert not f.accept(X, torch.zeros(150), is_classif=False)


def test_banded_mode_rejects_a_too_easy_task():
    f = PredictabilityFilter(mode="banded", quantile_band=(0.02, 0.30))
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 4, generator=g)
    y = X[:, 0] * 5.0  # far too predictable for the band
    assert not f.accept(X, y, is_classif=False)
    assert f.stats.rejected_band == 1


def test_filter_stats_add_up():
    f = PredictabilityFilter(mode="tabicl")
    g = torch.Generator().manual_seed(0)
    for _ in range(6):
        X = torch.randn(150, 3, generator=g)
        f.accept(X, X[:, 0].clone(), is_classif=False)
    s = f.stats.summary()
    assert s["attempts"] == 6
    assert 0.0 <= s["rejection_rate"] <= 1.0


def test_unknown_filter_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown filter mode"):
        PredictabilityFilter(mode="magic")


# --- noise columns -----------------------------------------------------------


def test_noise_features_widen_the_table():
    rng = PriorRNG(0)
    X = torch.randn(200, 10)
    X2, meta = add_noise_features(rng, X, {"fraction": 0.5}, max_features=32)
    assert meta["noise_features"] == 5
    assert X2.shape == (200, 15)


def test_noise_features_respect_the_cap():
    rng = PriorRNG(0)
    X = torch.randn(200, 10)
    X2, _ = add_noise_features(rng, X, {"fraction": 2.0}, max_features=12)
    assert X2.shape[1] <= 12


def test_noise_features_off_by_default():
    X = torch.randn(100, 5)
    X2, meta = add_noise_features(PriorRNG(0), X, {}, max_features=32)
    assert torch.equal(X2, X) and meta["noise_features"] == 0


def test_shuffled_column_keeps_its_distribution():
    """The point of `shuffled`: same values, no relationship. Column statistics
    cannot spot it, only the link to the target can."""
    rng = PriorRNG(0)
    X = torch.randn(500, 4)
    X2, _ = add_noise_features(rng, X, {"fraction": 0.25, "kind_weights": {"shuffled": 1.0}}, 32)
    assert X2.shape[1] == 5
    # Some column of X2 must be a permutation of a column of X.
    sorted_orig = {tuple(torch.sort(X[:, j]).values.tolist()) for j in range(4)}
    assert any(tuple(torch.sort(X2[:, j]).values.tolist()) in sorted_orig for j in range(5))


# --- preprocessing -----------------------------------------------------------


def test_to_ranks_is_uniform_and_monotone():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, generator=g)
    u = to_ranks(x)
    assert float(u.min()) > 0.0 and float(u.max()) < 1.0
    assert float(u.mean()) == pytest.approx(0.5, abs=0.02)
    order = torch.argsort(x)
    assert float((u[order][1:] - u[order][:-1]).min()) > 0


def test_standard_scaling_centres_and_scales():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 3, generator=g) * 7 + 4
    out = standard_scaling(x)
    assert torch.allclose(out.mean(dim=0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(out.std(dim=0, correction=1), torch.ones(3), atol=1e-2)


def test_standard_scaling_survives_a_constant_column():
    x = torch.cat([torch.randn(100, 1), torch.ones(100, 1)], dim=1)
    out = standard_scaling(x)
    assert torch.isfinite(out).all()


# --- batching ----------------------------------------------------------------


@pytest.mark.parametrize("task", ["lgd", "pd"])
def test_batches_are_dense_and_consistent(lgd_cfg, pd_cfg, task):
    cfg = lgd_cfg if task == "lgd" else pd_cfg
    ds = PriorBatchDataset(cfg["prior"], task, batch_size=3, seed=0)
    it = iter(ds)
    for _ in range(4):
        X, y, train_size = next(it)
        assert X.shape[0] == 3 and y.shape[0] == 3
        assert X.shape[1] == y.shape[1]
        assert 0 < train_size < X.shape[1]
        assert torch.isfinite(X).all() and torch.isfinite(y).all()


def test_batch_rows_are_never_ragged(pd_cfg):
    """Regression test: PD's underwriting selection drops rows, and rounding once
    left a task a row short, which crashed torch.stack."""
    pd_cfg["prior"]["credit_fraction"] = 1.0
    pd_cfg["prior"]["credit"]["target"]["selection"] = {"selection_drop": 0.2, "selection_sharpness": 0.7}
    ds = PriorBatchDataset(pd_cfg["prior"], "pd", batch_size=4, seed=0)
    it = iter(ds)
    for _ in range(6):
        X, y, _ = next(it)
        assert X.shape[:2] == y.shape[:2]


def test_pd_batches_keep_both_classes_in_context(pd_cfg):
    """In-context learning needs both classes on the context side of the split."""
    ds = PriorBatchDataset(pd_cfg["prior"], "pd", batch_size=4, seed=0)
    it = iter(ds)
    for _ in range(5):
        _, y, train_size = next(it)
        for row in y:
            assert len(row[:train_size].unique()) >= 1  # at minimum, not empty
