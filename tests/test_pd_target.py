"""The PD target family: rare positives, weak signal, and credit-shaped structure."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.prior.rng import PriorRNG
from src.prior.targets.pd import (
    apply_informative_missingness,
    apply_pd_target,
    apply_threshold_rules,
    apply_underwriting_selection,
)


def features(n: int = 600, d: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def latent(n: int = 600, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


BASE = {
    "base_rate_range": [0.05, 0.15],
    "signal_strength": 1.0,
    "flip_pos_to_neg": 0.0,
    "flip_neg_to_pos": 0.0,
    "rules": {"n_rules": 0},
    "selection": {"selection_drop": 0.0},
    "missingness": {"missing_col_fraction": 0.0},
}


# --- base rate ---------------------------------------------------------------


def test_base_rate_lands_in_the_requested_range():
    rng = PriorRNG(0)
    for _ in range(30):
        _, y, meta = apply_pd_target(rng, features(), latent(), BASE, max_features=16)
        assert 0.02 <= meta["realised_base_rate"] <= 0.25


def test_narrow_range_gives_the_requested_rate():
    rng = PriorRNG(1)
    cfg = {**BASE, "base_rate_range": [0.08, 0.08]}
    for _ in range(10):
        _, y, meta = apply_pd_target(rng, features(1000), latent(1000), cfg, max_features=16)
        assert meta["realised_base_rate"] == pytest.approx(0.08, abs=0.02)


def test_both_classes_always_present():
    """Cross-entropy needs two classes, and so does the in-context split."""
    rng = PriorRNG(2)
    cfg = {**BASE, "base_rate_range": [0.005, 0.01]}  # deliberately extreme
    for _ in range(25):
        _, y, _ = apply_pd_target(rng, features(200), latent(200), cfg, max_features=16)
        assert 0 < float(y.sum()) < y.numel()


def test_labels_are_binary():
    rng = PriorRNG(3)
    _, y, _ = apply_pd_target(rng, features(), latent(), BASE, max_features=16)
    assert set(y.unique().tolist()) <= {0.0, 1.0}


# --- signal strength ---------------------------------------------------------


def test_signal_dilution_makes_the_task_harder():
    """Diluting the label-assigning score must lower how well the features
    separate the classes. Measured as the gap in mean feature value."""

    def separation(rho: float) -> float:
        X, z = features(1200), latent(1200)
        # Make one feature the true driver so separation is measurable.
        X = X.clone()
        X[:, 0] = z
        _, y, _ = apply_pd_target(PriorRNG(4), X, z, {**BASE, "signal_strength": rho}, 16)
        pos, neg = X[y == 1, 0], X[y == 0, 0]
        return abs(float(pos.mean()) - float(neg.mean()))

    assert separation(1.0) > separation(0.2)


def test_signal_strength_is_recorded():
    _, _, meta = apply_pd_target(PriorRNG(5), features(), latent(), {**BASE, "signal_strength": 0.6}, 16)
    assert meta["signal_strength"] == 0.6


# --- label noise -------------------------------------------------------------


def test_asymmetric_flipping_lowers_the_positive_rate():
    """Cures (1 -> 0) are more common than the reverse, so heavy one-sided
    flipping should reduce the share of positives."""
    rng = PriorRNG(6)
    clean, flipped = [], []
    for _ in range(15):
        _, _, m1 = apply_pd_target(rng, features(800), latent(800), {**BASE, "base_rate_range": [0.2, 0.2]}, 16)
        clean.append(m1["realised_base_rate"])
        _, _, m2 = apply_pd_target(
            rng, features(800), latent(800),
            {**BASE, "base_rate_range": [0.2, 0.2], "flip_pos_to_neg": 0.5}, 16,
        )
        flipped.append(m2["realised_base_rate"])
    assert sum(flipped) / len(flipped) < sum(clean) / len(clean)


# --- threshold rules ---------------------------------------------------------


def test_rules_change_the_latent():
    rng = PriorRNG(7)
    X, z = features(), latent()
    out, meta = apply_threshold_rules(rng, X, z, {"n_rules": 3, "rule_weight_range": [1.0, 2.0]})
    assert meta["n_rules"] == 3
    assert not torch.allclose(out, z)


def test_zero_rules_is_a_no_op():
    X, z = features(), latent()
    out, meta = apply_threshold_rules(PriorRNG(8), X, z, {"n_rules": 0})
    assert meta["n_rules"] == 0
    assert torch.equal(out, z)


def test_rules_create_step_changes():
    """A hard rule should make the latent take a jump, unlike a smooth function.
    Check the biggest gap between neighbouring sorted values is large."""
    rng = PriorRNG(9)
    X = features(800, 4)
    z = torch.zeros(800)  # no smooth signal, so any structure is from the rule
    out, _ = apply_threshold_rules(rng, X, z, {"n_rules": 1, "rule_weight_range": [3.0, 3.0], "conjunction_prob": 0.0})
    assert len(out.unique()) <= 4, "a single hard rule on a flat latent gives few distinct values"


# --- underwriting selection --------------------------------------------------


def test_selection_removes_rows():
    rng = PriorRNG(10)
    X, z = features(1000), latent(1000)
    X2, z2, meta = apply_underwriting_selection(rng, X, z, {"selection_drop": 0.2})
    assert X2.shape[0] == pytest.approx(800, abs=2)
    assert z2.shape[0] == X2.shape[0]
    assert meta["selection_drop"] == 0.2


def test_selection_is_a_no_op_at_zero():
    X, z = features(), latent()
    X2, z2, meta = apply_underwriting_selection(PriorRNG(11), X, z, {"selection_drop": 0.0})
    assert torch.equal(X2, X) and torch.equal(z2, z)
    assert meta["selection_drop"] == 0.0


def test_selection_truncates_the_risky_tail():
    """The point of the through-the-door problem: the observed book is missing its
    riskiest applicants, so the surviving latent should have a lower mean."""
    rng = PriorRNG(12)
    X, z = features(2000), latent(2000)
    _, z2, _ = apply_underwriting_selection(rng, X, z, {"selection_drop": 0.3, "selection_sharpness": 1.0})
    assert float(z2.mean()) < float(z.mean())


def test_selection_keeps_rows_in_order():
    """Row order matters: the model splits context from query by position, so a
    reordered sample would silently change what is in the context."""
    rng = PriorRNG(13)
    X = torch.arange(500, dtype=torch.float32).reshape(500, 1)
    z = latent(500)
    X2, _, _ = apply_underwriting_selection(rng, X, z, {"selection_drop": 0.2})
    col = X2[:, 0]
    assert float((col[1:] - col[:-1]).min()) > 0, "surviving rows must stay in their original order"


# --- informative missingness -------------------------------------------------


def test_missingness_adds_indicator_columns():
    rng = PriorRNG(14)
    X = features(400, 6)
    y = (latent(400) > 0).float()
    X2, meta = apply_informative_missingness(
        rng, X, y, {"missing_col_fraction": 0.5, "missing_indicators": True}, max_features=32
    )
    assert meta["missing_cols"] == 3
    assert X2.shape[1] > X.shape[1]


def test_missingness_respects_the_feature_cap():
    """Batches are dense tensors, so the width must stay bounded."""
    rng = PriorRNG(15)
    X = features(400, 8)
    y = (latent(400) > 0).float()
    X2, _ = apply_informative_missingness(
        rng, X, y, {"missing_col_fraction": 1.0, "missing_indicators": True}, max_features=8
    )
    assert X2.shape[1] <= 8


def test_missingness_is_a_no_op_at_zero():
    X = features()
    y = (latent() > 0).float()
    X2, meta = apply_informative_missingness(PriorRNG(16), X, y, {"missing_col_fraction": 0.0}, 32)
    assert torch.equal(X2, X) and meta["missing_cols"] == 0


def test_no_nans_survive_imputation():
    """The model standardises its input; a NaN would poison the whole column."""
    rng = PriorRNG(17)
    X = features(400, 6)
    y = (latent(400) > 0).float()
    X2, _ = apply_informative_missingness(rng, X, y, {"missing_col_fraction": 0.5}, 32)
    assert not bool(torch.isnan(X2).any())


# --- metadata ----------------------------------------------------------------


def test_metadata_records_every_component():
    rng = PriorRNG(18)
    cfg = {
        **BASE,
        "rules": {"n_rules": 2},
        "selection": {"selection_drop": 0.1},
        "missingness": {"missing_col_fraction": 0.3},
    }
    _, _, meta = apply_pd_target(rng, features(), latent(), cfg, max_features=32)
    for key in ("target", "n_rules", "selection_drop", "signal_strength",
                "target_base_rate", "realised_base_rate", "missing_cols"):
        assert key in meta, f"missing {key}"
    assert meta["target"] == "pd"
