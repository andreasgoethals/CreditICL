"""The LGD target family — the heart of the LGD experiment.

Four properties matter, and each has a test:

1. the target never leaves [0,1];
2. the requested boundary mass is what actually comes out;
3. the transform is MONOTONE in the latent, so it changes the target's shape
   without changing how predictable it is (that is what makes the intervention
   clean rather than confounded);
4. the family really does cover all three shapes we measured in the real data —
   U-shaped, one-sided, and fully interior.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.prior.rng import PriorRNG
from src.prior.targets.lgd import apply_lgd_target, kumaraswamy_icdf, sample_lgd_shape


def latent(n: int = 800, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


BASE = {
    "mode": "quantile",
    "shape_ab_range": [0.3, 4.0],
    "boundary_mass_range": [0.10, 0.20],
    "atom_prob": 1.0,
    "max_total_boundary_mass": 0.60,
    "signal_strength": 1.0,
    "round_to": 0,
    "target_scaling": "none",
}


# --- 1. bounded --------------------------------------------------------------


@pytest.mark.parametrize("mode", ["quantile", "censor"])
def test_target_stays_in_unit_interval(mode):
    rng = PriorRNG(0)
    for _ in range(20):
        y, _ = apply_lgd_target(rng, latent(), {**BASE, "mode": mode})
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 1.0


def test_clipping_happens_even_with_odd_settings():
    """The clip is unconditional: LGD is a fraction and cannot leave [0,1]."""
    rng = PriorRNG(1)
    cfg = {**BASE, "boundary_mass_range": [0.4, 0.45], "shape_ab_range": [0.2, 0.25]}
    y, _ = apply_lgd_target(rng, latent(), cfg)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


# --- 2. the requested mass is the delivered mass -----------------------------


def test_quantile_mode_hits_the_requested_boundary_mass():
    rng = PriorRNG(2)
    cfg = {**BASE, "boundary_mass_range": [0.15, 0.15]}
    for _ in range(10):
        y, meta = apply_lgd_target(rng, latent(2000), cfg)
        assert meta["realised_p0"] == pytest.approx(0.15, abs=0.02)
        assert meta["realised_p1"] == pytest.approx(0.15, abs=0.02)


def test_atom_prob_zero_gives_no_atoms():
    rng = PriorRNG(3)
    y, meta = apply_lgd_target(rng, latent(), {**BASE, "atom_prob": 0.0})
    assert meta["target_p0"] == 0.0 and meta["target_p1"] == 0.0
    # An interior-only target should have almost nothing sitting on the boundary.
    assert meta["realised_p0"] < 0.01 and meta["realised_p1"] < 0.01


def test_total_boundary_mass_is_capped():
    """Without the cap a draw can collapse to two points, which the degeneracy
    check would then throw away — wasting generation instead of training."""
    rng = PriorRNG(4)
    cfg = {**BASE, "boundary_mass_range": [0.45, 0.5], "max_total_boundary_mass": 0.5}
    for _ in range(20):
        _, meta = apply_lgd_target(rng, latent(), cfg)
        assert meta["target_p0"] + meta["target_p1"] <= 0.5 + 1e-9


# --- 3. monotone, so predictability is preserved -----------------------------


def test_transform_is_monotone_in_the_latent():
    """This is the property that makes the LGD arm a clean intervention: the
    target's SHAPE changes but the information the features carry does not.
    Spearman correlation with the latent must be 1 (up to ties in the atoms)."""
    rng = PriorRNG(5)
    z = latent(500)
    y, _ = apply_lgd_target(rng, z, {**BASE, "atom_prob": 0.0})

    order = torch.argsort(z)
    y_sorted = y[order]
    diffs = y_sorted[1:] - y_sorted[:-1]
    assert float(diffs.min()) >= -1e-6, "target must be non-decreasing in the latent"


def test_atoms_preserve_ordering_outside_the_atoms():
    rng = PriorRNG(6)
    z = latent(500)
    y, _ = apply_lgd_target(rng, z, BASE)
    order = torch.argsort(z)
    y_sorted = y[order]
    assert float((y_sorted[1:] - y_sorted[:-1]).min()) >= -1e-6


def test_signal_strength_reduces_correlation_with_the_latent():
    """Diluting the signal must actually make the task harder, not just noisier
    looking. Compare rank correlation at full vs half strength."""
    z = latent(1500)
    y_full, _ = apply_lgd_target(PriorRNG(7), z, {**BASE, "signal_strength": 1.0})
    y_weak, _ = apply_lgd_target(PriorRNG(7), z, {**BASE, "signal_strength": 0.3})

    def rank_corr(a, b):
        ra = torch.argsort(torch.argsort(a)).float()
        rb = torch.argsort(torch.argsort(b)).float()
        return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])

    assert rank_corr(z, y_full) > rank_corr(z, y_weak)


# --- 4. the family covers the shapes we actually measured --------------------


def test_family_covers_u_shaped_and_interior():
    """Real data disagrees: Freddie is U-shaped with 11.4%/8.1% on the boundaries,
    LendingClub is one-sided with ~1.8%, AXA has none. One fixed shape would
    overfit to Freddie, so the family must produce all three."""
    rng = PriorRNG(8)
    cfg = {**BASE, "atom_prob": 0.75, "boundary_mass_range": [0.0, 0.25]}
    two_sided = one_sided = interior = 0
    for _ in range(120):
        _, meta = apply_lgd_target(rng, latent(400), cfg)
        p0, p1 = meta["target_p0"], meta["target_p1"]
        if p0 > 0.01 and p1 > 0.01:
            two_sided += 1
        elif p0 > 0.01 or p1 > 0.01:
            one_sided += 1
        else:
            interior += 1
    assert two_sided > 0, "no U-shaped (Freddie-like) draws"
    assert one_sided > 0, "no one-sided (LendingClub-like) draws"
    assert interior > 0, "no interior-only (AXA-like) draws"


def test_shape_range_spans_u_and_hump():
    """a<1,b<1 gives a U; a>1,b>1 gives a hump. The default range must reach both."""
    rng = PriorRNG(9)
    below = above = 0
    for _ in range(300):
        s = sample_lgd_shape(rng, BASE)
        if s["a"] < 1 and s["b"] < 1:
            below += 1
        if s["a"] > 1 and s["b"] > 1:
            above += 1
    assert below > 0 and above > 0


# --- the Kumaraswamy curve itself -------------------------------------------


def test_kumaraswamy_icdf_is_a_valid_inverse_cdf():
    u = torch.linspace(0.001, 0.999, 200)
    for a, b in [(0.5, 0.5), (2.0, 3.0), (0.4, 2.5), (1.0, 1.0)]:
        y = kumaraswamy_icdf(u, a, b)
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
        assert float((y[1:] - y[:-1]).min()) >= -1e-6, "must be non-decreasing in u"


def test_kumaraswamy_matches_its_own_cdf():
    """Round-trip: applying the CDF to the inverse CDF should give back u."""
    u = torch.linspace(0.01, 0.99, 50)
    a, b = 1.7, 0.6
    y = kumaraswamy_icdf(u, a, b)
    back = 1.0 - (1.0 - y**a) ** b  # the Kumaraswamy CDF
    assert torch.allclose(back, u, atol=1e-4)


def test_a_below_one_pushes_mass_toward_zero():
    u = torch.linspace(0.001, 0.999, 2000)
    low_a = kumaraswamy_icdf(u, 0.4, 1.0)
    high_a = kumaraswamy_icdf(u, 3.0, 1.0)
    assert float(low_a.mean()) < float(high_a.mean())


# --- scaling -----------------------------------------------------------------


def test_standard_scaling_preserves_shape_but_not_range():
    """standard_scaling is a straight line, so it keeps the two humps and only
    destroys the [0,1] range. This is the whole basis for the project claiming a
    'frequency and range' gap rather than a structural absence."""
    rng = PriorRNG(10)
    z = latent(1000)
    y_raw, _ = apply_lgd_target(PriorRNG(10), z, {**BASE, "target_scaling": "none"})
    y_std, meta = apply_lgd_target(PriorRNG(10), z, {**BASE, "target_scaling": "standard"})

    # Range is gone.
    assert float(y_std.min()) < 0.0
    # Shape survives: the count of distinct values is unchanged by an affine map.
    assert len(y_raw.unique()) == len(y_std.unique())
    # And the realised mass was recorded BEFORE scaling, so it is still meaningful.
    assert meta["realised_p0"] > 0.05


def test_metadata_reports_what_actually_happened():
    rng = PriorRNG(11)
    _, meta = apply_lgd_target(rng, latent(), BASE)
    for key in ("target", "mode", "kuma_a", "kuma_b", "target_p0", "realised_p0", "target_scaling"):
        assert key in meta
    assert meta["target"] == "lgd"


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown LGD target mode"):
        apply_lgd_target(PriorRNG(0), latent(), {**BASE, "mode": "nonsense"})


def test_rejects_unknown_scaling():
    with pytest.raises(ValueError, match="unknown target_scaling"):
        apply_lgd_target(PriorRNG(0), latent(), {**BASE, "target_scaling": "nonsense"})
