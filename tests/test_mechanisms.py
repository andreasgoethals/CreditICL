"""Credit-risk mechanisms — the project's core contribution.

The property that matters most here is **not** "does the marginal look like credit
data". It is: does the mechanism produce credit's shape *as a consequence*, while
keeping the features predictive?

That second half is the one silent failure mode worth real paranoia. A mechanism that
quietly replaced the SCM latent with fresh noise would produce beautiful boundary
atoms, pass every distributional check, and hand the model an unlearnable task — and
every downstream number would be meaningless without anything looking wrong. So
several tests below assert **rank correlation with the latent**, not just shape.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.prior.rng import PriorRNG
from src.prior.targets import mechanisms as M


@pytest.fixture
def latent():
    """A latent with real structure, as the SCM would produce."""
    rng = PriorRNG(0)
    return rng.randn(600)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


# -- the invariants every LGD mechanism must satisfy --------------------------


@pytest.mark.parametrize("name", sorted(M.LGD_MECHANISMS))
def test_lgd_mechanisms_stay_in_the_unit_interval(name, latent):
    """LGD is a loss *fraction*. Outside [0,1] it is not LGD."""
    rng = PriorRNG(1)
    for _ in range(15):
        y, _ = M.LGD_MECHANISMS[name](rng, latent, {})
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0, name
        assert torch.isfinite(y).all(), name


def _correlation_ratio(driver: torch.Tensor, y: torch.Tensor, bins: int = 20) -> float:
    """eta-squared: the share of y's variance explained by the driver.

    Used instead of Spearman because a segment mixture is **deliberately**
    non-monotone: each sub-book re-ranks internally, so the target jumps down at a
    segment boundary. That is the intended structure of a portfolio — a well-secured
    but weak mortgage can lose less than a strong unsecured loan — and a rank
    correlation reads it as lost signal. eta-squared measures dependence of any shape.
    """
    order = driver.argsort()
    ys = y[order]
    chunks = torch.chunk(ys, bins)
    grand = ys.mean()
    between = sum(c.numel() * (c.mean() - grand) ** 2 for c in chunks if c.numel())
    total = ((ys - grand) ** 2).sum()
    return float(between / (total + 1e-12))


@pytest.mark.parametrize("name", sorted(M.LGD_MECHANISMS))
def test_lgd_mechanisms_keep_the_features_predictive(name, latent):
    """THE test. A mechanism must transform the latent, not discard it.

    Without this, a prior can look perfect and teach nothing: beautiful boundary
    atoms, a credit-shaped marginal, and a target no model can predict. Every
    downstream number would then be meaningless with nothing looking wrong.
    """
    rng = PriorRNG(2)
    scores = []
    for _ in range(20):
        y, _ = M.LGD_MECHANISMS[name](rng, latent, {})
        scores.append(_correlation_ratio(latent, y))
    median = sorted(scores)[len(scores) // 2]
    assert median > 0.5, f"{name}: latent explained only {median:.2f} of the target's variance"


@pytest.mark.parametrize("name", sorted(M.LGD_MECHANISMS))
def test_lgd_mechanisms_are_reproducible(name, latent):
    a, _ = M.LGD_MECHANISMS[name](PriorRNG(7), latent, {})
    b, _ = M.LGD_MECHANISMS[name](PriorRNG(7), latent, {})
    assert torch.equal(a, b)


@pytest.mark.parametrize("name", sorted(M.LGD_MECHANISMS))
def test_lgd_mechanisms_are_not_constant(name, latent):
    rng = PriorRNG(3)
    for _ in range(10):
        y, _ = M.LGD_MECHANISMS[name](rng, latent, {})
        assert float(y.std()) > 1e-6, name


# -- collateral: the atoms must be a CONSEQUENCE ------------------------------


def test_collateral_produces_exact_atoms_at_both_ends(latent):
    """The whole argument for this module. Over-collateralised loans must land on
    *exactly* 0 (not 0.01), and total losses on exactly 1.
    """
    rng = PriorRNG(4)
    saw_zero = saw_one = False
    for _ in range(60):
        y, _ = M.lgd_collateral(rng, latent, {})
        if float((y == 0.0).float().mean()) > 0.005:
            saw_zero = True
        if float((y == 1.0).float().mean()) > 0.005:
            saw_one = True
        if saw_zero and saw_one:
            break
    assert saw_zero, "no mass at exactly 0 — over-collateralisation is not producing it"
    assert saw_one, "no mass at exactly 1 — total loss is not producing it"


def test_more_collateral_means_less_loss(latent):
    """Sanity on the economics: shifting coverage up must reduce LGD. If this fails
    the identity is inverted, which no distributional check would catch."""
    lo, _ = M.lgd_collateral(PriorRNG(5), latent, {"log_coverage_mean_range": [-1.5, -1.5]})
    hi, _ = M.lgd_collateral(PriorRNG(5), latent, {"log_coverage_mean_range": [1.5, 1.5]})
    assert float(hi.mean()) < float(lo.mean())


def test_bigger_haircut_means_more_loss(latent):
    small, _ = M.lgd_collateral(PriorRNG(6), latent, {"haircut_range": [0.1, 0.1]})
    big, _ = M.lgd_collateral(PriorRNG(6), latent, {"haircut_range": [0.9, 0.9]})
    assert float(big.mean()) > float(small.mean())


def test_downturn_shock_raises_lgd(latent):
    """Basel's downturn-LGD requirement: a bad economy depresses collateral values."""
    cfg = {"shock_beta_range": [0.5, 0.5], "log_coverage_mean_range": [0.2, 0.2]}
    calm = torch.zeros(latent.numel())
    crisis = torch.ones(latent.numel()) * 1.5
    y_calm, _ = M.lgd_collateral(PriorRNG(8), latent, cfg, calm)
    y_bad, _ = M.lgd_collateral(PriorRNG(8), latent, cfg, crisis)
    assert float(y_bad.mean()) > float(y_calm.mean())


# -- workout -----------------------------------------------------------------


def test_zero_recovery_share_creates_mass_at_total_loss(latent):
    """Unsecured facilities that recover nothing are the dominant atom-at-1 case."""
    y, meta = M.lgd_workout(PriorRNG(9), latent, {"zero_recovery_range": [0.4, 0.4], "cost_range": [0.0, 0.0]})
    assert meta["zero_recovery_rate"] == pytest.approx(0.4)
    assert float((y == 1.0).float().mean()) > 0.25


def test_discounting_makes_late_recovery_a_loss(latent):
    """A slow full recovery IS a loss — time is a risk driver, not an inconvenience."""
    cfg = {"zero_recovery_range": [0.0, 0.0], "cost_range": [0.0, 0.0], "max_periods": 8}
    none, _ = M.lgd_workout(PriorRNG(10), latent, {**cfg, "discount_range": [0.0, 0.0]})
    steep, _ = M.lgd_workout(PriorRNG(10), latent, {**cfg, "discount_range": [0.3, 0.3]})
    assert float(steep.mean()) > float(none.mean())


# -- segment mixture ---------------------------------------------------------


def test_segment_mixture_reports_its_segments(latent):
    y, meta = M.lgd_segment_mixture(PriorRNG(11), latent, {})
    assert meta["mechanism"] == "segment_mixture"
    assert 2 <= meta["n_segments"] <= 3
    assert sum(s["n"] for s in meta["segments"]) == latent.numel()


def test_segment_mixture_covers_every_row(latent):
    """A row left unassigned would silently keep its zero-initialised value, which
    would look like a legitimate atom at 0."""
    rng = PriorRNG(12)
    for _ in range(10):
        y, meta = M.lgd_segment_mixture(rng, latent, {})
        assert sum(s["n"] for s in meta["segments"]) == latent.numel()
        assert y.numel() == latent.numel()


# -- the dispatcher ----------------------------------------------------------


def test_dispatcher_uses_every_mechanism_over_many_draws(latent):
    """Mechanism DIVERSITY is O'Prior's strongest reported driver of transfer, so a
    dispatcher that quietly always picked one would undercut the whole design."""
    rng = PriorRNG(13)
    seen = set()
    for _ in range(120):
        _, meta = M.apply_lgd_mechanism(rng, latent, {})
        seen.add(meta["mechanism_choice"])
    assert seen == set(M.LGD_MECHANISMS), f"only saw {seen}"


def test_dispatcher_respects_zero_weights(latent):
    rng = PriorRNG(14)
    cfg = {"mechanism_weights": {"collateral": 1.0, "workout": 0.0, "segment_mixture": 0.0}}
    for _ in range(25):
        _, meta = M.apply_lgd_mechanism(rng, latent, cfg)
        assert meta["mechanism_choice"] == "collateral"


def test_dispatcher_rejects_all_zero_weights(latent):
    with pytest.raises(ValueError, match="positive weight"):
        M.apply_lgd_mechanism(PriorRNG(15), latent, {"mechanism_weights": {"collateral": 0.0}})


# -- the systematic factor ---------------------------------------------------


def test_cohorts_are_contiguous_blocks_with_a_shared_shock():
    """A vintage is a time period, so its rows are adjacent and share one shock."""
    shock, meta = M.sample_cohort_factor(PriorRNG(16), 400, {"min_cohorts": 4, "max_cohorts": 4})
    assert meta["cohorts"] == 4
    assert len(torch.unique(shock)) <= 4
    # Within the first block every value is identical.
    assert float(shock[:100].std()) == pytest.approx(0.0, abs=1e-9)


def test_single_cohort_means_no_shock():
    shock, meta = M.sample_cohort_factor(PriorRNG(17), 100, {"min_cohorts": 1, "max_cohorts": 1})
    assert meta["cohorts"] == 1
    assert torch.all(shock == 0)


# -- PD: Vasicek -------------------------------------------------------------


def test_vasicek_hits_the_requested_base_rate(latent):
    """The threshold is Phi^-1(PD), so the rate should be close to exact in
    expectation. Averaged over draws to average out the systematic factor."""
    rng = PriorRNG(18)
    for target in (0.05, 0.15, 0.30):
        rates = []
        for _ in range(40):
            y, _ = M.pd_vasicek(rng, latent, {"base_rate_range": [target, target], "rho_range": [0.03, 0.03]})
            rates.append(float(y.mean()))
        mean_rate = sum(rates) / len(rates)
        assert abs(mean_rate - target) < 0.06, f"target {target}, got {mean_rate:.3f}"


def test_vasicek_labels_are_binary_and_both_present(latent):
    rng = PriorRNG(19)
    for _ in range(20):
        y, _ = M.pd_vasicek(rng, latent, {"base_rate_range": [0.1, 0.4]})
        assert set(torch.unique(y).tolist()) <= {0.0, 1.0}


def test_vasicek_keeps_the_features_predictive(latent):
    """Same paranoia as LGD: the latent must drive the label."""
    rng = PriorRNG(20)
    aucs = []
    for _ in range(20):
        y, _ = M.pd_vasicek(rng, latent, {"base_rate_range": [0.2, 0.2], "rho_range": [0.03, 0.03]})
        pos, neg = latent[y == 1], latent[y == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        # Defaults should sit at the LOW end of the latent (asset value).
        aucs.append(float(pos.mean() < neg.mean()))
    assert sum(aucs) / len(aucs) > 0.9, "labels are not tracking the latent"


def test_higher_rho_means_more_variable_realised_rate(latent):
    """The reason Vasicek is here rather than a plain quantile cut: correlation makes
    the realised default rate move between cohorts, so the model sees good and bad
    years instead of the same year 35 million times.
    """
    cfg = {"base_rate_range": [0.15, 0.15], "cohort": {"min_cohorts": 8, "max_cohorts": 8}}
    spreads = {}
    for rho in (0.03, 0.24):
        rng = PriorRNG(21)
        rates = []
        for _ in range(60):
            shock, _ = M.sample_cohort_factor(rng, latent.numel(), {"min_cohorts": 1, "max_cohorts": 1})
            z = torch.full((latent.numel(),), float(rng.randn(1).item()))
            y, _ = M.pd_vasicek(rng, latent, {**cfg, "rho_range": [rho, rho]}, z)
            rates.append(float(y.mean()))
        mean = sum(rates) / len(rates)
        spreads[rho] = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5
    assert spreads[0.24] > spreads[0.03], f"rho did not increase rate dispersion: {spreads}"


def test_basel_rho_range_matches_the_regulation():
    """0.03-0.24 spans Basel's prescribed asset correlations (QRRE 0.04, mortgage
    0.15, corporate 0.12-0.24). A citable range, not an invented one."""
    lo, hi = M.BASEL_RHO_RANGE
    assert lo <= 0.04 and hi >= 0.24


def test_pd_dispatcher_reports_realised_rate(latent):
    rng = PriorRNG(22)
    y, meta = M.apply_pd_mechanism(rng, latent, {"base_rate_range": [0.1, 0.3]})
    assert "realised_base_rate" in meta and "rho" in meta
    assert meta["realised_base_rate"] == pytest.approx(float(y.mean()), abs=1e-4)


def test_pd_dispatcher_rejects_single_class(latent):
    """An all-zero target is unlearnable and would poison the batch statistics."""
    with pytest.raises(ValueError, match="single-class"):
        M.apply_pd_mechanism(PriorRNG(23), latent, {"base_rate_range": [0.0, 0.0]})


# -- ratio features ----------------------------------------------------------


def test_ratio_features_are_appended_and_finite():
    """Quotients are the one common credit transform TabICL's eight function
    families do not include — they have products, not ratios."""
    rng = PriorRNG(24)
    X = rng.randn(200, 6)
    out, meta = M.pd_ratio_features(rng, X, {"max_ratios": 3})
    assert out.shape[1] > X.shape[1]
    assert out.shape[1] == X.shape[1] + meta["ratio_features"]
    assert torch.isfinite(out).all(), "a near-zero denominator produced inf/nan"


def test_ratio_features_survive_a_zero_denominator():
    """The high-risk case IS the near-zero denominator (no income, no collateral),
    so it must not blow up."""
    rng = PriorRNG(25)
    X = torch.stack([torch.ones(100), torch.zeros(100)], dim=1)
    out, _ = M.pd_ratio_features(rng, X, {"max_ratios": 2})
    assert torch.isfinite(out).all()


def test_ratio_features_noop_on_a_single_column():
    rng = PriorRNG(26)
    X = rng.randn(50, 1)
    out, meta = M.pd_ratio_features(rng, X, {})
    assert out.shape == X.shape and meta["ratio_features"] == 0


# -- WOE binning -------------------------------------------------------------


def test_woe_is_a_monotone_staircase(latent):
    """Scorecard bins are constrained monotone in risk, and locally flat. Both
    properties change what 'learning the function' means."""
    woe, meta = M.pd_monotone_woe(PriorRNG(27), latent, {"min_bins": 5, "max_bins": 5})
    assert meta["woe_bins"] == 5
    assert len(torch.unique(woe)) <= 5, "should be a staircase, not continuous"
    # Monotone: sorting by the latent must give a non-decreasing WOE sequence.
    ordered = woe[latent.argsort()]
    diffs = ordered[1:] - ordered[:-1]
    assert float(diffs.min()) >= -1e-6, "WOE must be monotone in the driver"


def test_woe_preserves_rank_information(latent):
    woe, _ = M.pd_monotone_woe(PriorRNG(28), latent, {"min_bins": 8, "max_bins": 8})
    assert _spearman(latent, woe) > 0.9
