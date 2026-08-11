"""Credit-risk *mechanisms* for the prior — the project's core contribution.

WHY THIS MODULE EXISTS, AND WHY IT IS DIFFERENT FROM `lgd.py` / `pd.py`

`lgd.py` shapes a target's **marginal distribution**: rank the latent, push it
through a Kumaraswamy inverse CDF, then place atoms at the boundaries with a dialled
-in probability. That gives *exact* control, which is what a clean ablation needs.
But the atoms are a **parameter**, not a **consequence**. Real LGD has mass at zero
because loans are over-collateralised, not because someone chose `p0 = 0.2`.

O'Prior's headline finding is that **structural mechanism diversity is the strongest
driver of transfer** — stronger than observational realism. If that is right, then a
prior that reproduces credit's *marginal* is doing the weaker of the two things, and
one that reproduces credit's *generating mechanism* is doing the stronger one. This
module does the stronger one.

Every mechanism here is a standard credit-risk model, not an invention:

| mechanism | what it is | why the shape follows |
|---|---|---|
| `collateral` | LGD = shortfall after selling collateral | atoms at 0/1 **emerge** from over-collateralisation and total loss |
| `workout` | recovery as a discounted, censored cashflow stream | atom at 1 from no-recovery cases, common in unsecured lending |
| `segment_mixture` | secured / partially-secured / unsecured sub-books | bimodality as a genuine **mixture**, which is what a portfolio is |
| `vasicek` | Merton/Vasicek one-factor default model | the basis of the Basel IRB formula; gives **correlated** defaults |
| `cohort_factor` | vintage-level systematic shock | default rates move together over time; couples PD and LGD |

WHAT IS DOMAIN THEORY AND WHAT IS OUR CHOICE. The functional forms below are
textbook (Merton 1974 / Vasicek 2002 for the factor model; the Basel IRB asset
correlations; the standard workout-LGD definition). The **parameter ranges** are ours,
chosen to bracket what we measured in `data/raw` — see `notebooks/data_exploration`.
Where a range comes from a published regulatory value it says so inline. Nothing here
is fitted to the evaluation data: these are priors, and fitting them to the test sets
would be leakage.

CRUCIALLY, FEATURES STAY PREDICTIVE. Each mechanism consumes the SCM latent as its
driver (collateral coverage, asset value, segment assignment), so the DAG's features
still explain the target. A mechanism that replaced the latent with fresh noise would
produce beautiful marginals with no signal, which is the one failure mode that would
silently ruin every downstream number.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..preprocess import to_ranks
from ..rng import PriorRNG

# Basel II/III IRB prescribed asset correlations, by exposure class. Used as the
# sampling range for `vasicek` so the correlation structure is regulatory-plausible
# rather than arbitrary.
#   revolving retail (QRRE)  rho = 0.04
#   other retail             rho = 0.03 - 0.16
#   residential mortgage     rho = 0.15
#   corporate                rho = 0.12 - 0.24
BASEL_RHO_RANGE = (0.03, 0.24)


def _standardise(z: torch.Tensor) -> torch.Tensor:
    return (z - z.mean()) / (z.std() + 1e-8)


def _normal_icdf(u: torch.Tensor) -> torch.Tensor:
    """Inverse standard normal CDF. `torch.special.ndtri` with a safe clamp."""
    return torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6))


def _uniform_from_latent(y_latent: torch.Tensor) -> torch.Tensor:
    """Latent -> Uniform(0,1) by ranking.

    Rank-transforming is monotone, so Spearman correlation with the latent is exactly
    1: the mechanism changes the target's scale and shape but not *how predictable* it
    is from the features. That separation is what makes an effect attributable.
    """
    return to_ranks(y_latent).clamp(1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Shared: the systematic factor. Credit's defining feature is that losses are
# CORRELATED — one economy drives every borrower.
# ---------------------------------------------------------------------------


def sample_cohort_factor(
    rng: PriorRNG, n_rows: int, cfg: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assign rows to vintages and give each a shared shock.

    Real books are stacks of monthly or quarterly vintages, and a recession hits a
    whole vintage at once. That makes rows **not** independent given the features —
    which is exactly the structure an i.i.d. prior cannot teach.

    Returns `(shock_per_row, meta)` where `shock_per_row` is a standard-normal-scaled
    systematic factor, constant within a cohort.
    """
    n_cohorts = int(rng.randint(int(cfg.get("min_cohorts", 1)), int(cfg.get("max_cohorts", 12)) + 1))
    if n_cohorts <= 1:
        return torch.zeros(n_rows), {"cohorts": 1, "cohort_sd": 0.0}

    sd = float(rng.uniform(*cfg.get("cohort_sd_range", [0.2, 1.0])))
    # Contiguous blocks, not random assignment: a vintage is a time period, so rows
    # in the same cohort are adjacent when the table is sorted by origination date.
    # A model can in principle pick this up, which is the point.
    edges = torch.linspace(0, n_rows, n_cohorts + 1).long()
    shock = torch.zeros(n_rows)
    draws = rng.randn(n_cohorts) * sd
    for c in range(n_cohorts):
        shock[edges[c] : edges[c + 1]] = draws[c]
    return shock, {"cohorts": n_cohorts, "cohort_sd": round(sd, 4)}


# ---------------------------------------------------------------------------
# LGD mechanisms
# ---------------------------------------------------------------------------


def lgd_collateral(
    rng: PriorRNG, y_latent: torch.Tensor, cfg: dict[str, Any], shock: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """LGD as the shortfall left after selling the collateral.

    The standard secured-lending identity:

        recovery = min(EAD, coverage x (1 - haircut) x EAD)
        LGD      = clip(1 - recovery/EAD + costs, 0, 1)

    where `coverage` is collateral value over exposure (the reciprocal of LTV) and
    `haircut` is the fire-sale discount. Two things fall out for free:

    * **an atom at exactly 0** whenever coverage x (1 - haircut) >= 1 + costs — the
      loan is over-collateralised, the bank is made whole, and the loss is not
      "small", it is *zero*. This is the mass that TabICL's prior cannot produce.
    * **an atom at exactly 1** when coverage is near zero and workout costs eat the
      rest — total loss, again exactly, not approximately.

    So the boundary mass is a *consequence* of the loan economics. `atom_prob` in the
    marginal-shaping path has to be told what the answer is; this derives it.
    """
    n = y_latent.numel()
    u = _uniform_from_latent(y_latent)

    # Coverage is right-skewed and strictly positive: most loans cluster near
    # full coverage with a long tail of very well-secured ones. A lognormal is the
    # usual choice for LTV-like ratios.
    mu = float(rng.uniform(*cfg.get("log_coverage_mean_range", [-0.35, 0.55])))
    sigma = float(rng.uniform(*cfg.get("log_coverage_sd_range", [0.35, 1.10])))
    coverage = torch.exp(mu + sigma * _normal_icdf(u))

    # A downturn depresses collateral values — the "downturn LGD" Basel requires.
    if shock is not None:
        beta = float(rng.uniform(*cfg.get("shock_beta_range", [0.0, 0.5])))
        coverage = coverage * torch.exp(-beta * shock)

    # Some facilities carry no collateral at all — most unsecured consumer lending.
    # This is a genuine point mass at zero coverage, not a small value, and it is what
    # produces the atom at LGD = 1: nothing to sell, so the loss is total.
    unsecured = float(rng.uniform(*cfg.get("unsecured_share_range", [0.0, 0.45])))
    coverage = torch.where(u < unsecured, torch.zeros(n), coverage)

    haircut = float(rng.uniform(*cfg.get("haircut_range", [0.10, 0.55])))
    cost_mean = float(rng.uniform(*cfg.get("cost_range", [0.0, 0.20])))
    # Costs vary per loan (legal fees, servicing, time in workout).
    costs = (cost_mean * (0.5 + rng.rand(n))).clamp(0.0, 0.6)

    # Costs come out of the collateral proceeds, and only THEN is the result capped at
    # the exposure. Adding costs after the cap would mean an over-collateralised loan
    # still lost the legal fees, so LGD could never reach exactly 0 — which silently
    # removed the atom this whole mechanism exists to produce. Surplus collateral
    # covers the workout costs in reality, and the bank is made whole.
    net_recovery = coverage * (1.0 - haircut) - costs

    # INTERIOR-ONLY draws: a book with no total losses and no fully-recovered loans.
    # `atom_prob` needs some datasets to have NO boundary mass at all, because
    # lgd_lendingclub has only 1.8% — effectively an interior distribution.
    #
    # Simply narrowing the parameter ranges did not achieve that. Coverage is lognormal,
    # so its right tail always produced some rows with net_recovery > 1, and those clip
    # to exactly 0. The leak was in EVERY draw, which is why atom_prob=0 still yielded
    # atoms in 57% of datasets. Clamping the ranked coverage per row is what actually
    # guarantees an interior target: net_recovery is held inside (eps, 1-eps), so the
    # clip below never binds.
    if cfg.get("interior_only", False):
        eps = float(cfg.get("interior_margin", 0.02))
        net_recovery = net_recovery.clamp(min=eps).clamp(max=1.0 - eps)

    y = (1.0 - net_recovery).clamp(0.0, 1.0)

    meta = {
        "mechanism": "collateral",
        "log_coverage_mean": round(mu, 4),
        "log_coverage_sd": round(sigma, 4),
        "haircut": round(haircut, 4),
        "cost_mean": round(cost_mean, 4),
        "unsecured_share": round(unsecured, 4),
    }
    return y, meta


def lgd_workout(
    rng: PriorRNG, y_latent: torch.Tensor, cfg: dict[str, Any], shock: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """LGD from a discounted, censored recovery cashflow stream.

    The regulatory definition of realised LGD is a *workout* calculation: recoveries
    arrive over months or years and are discounted back to default date.

        LGD = clip( (EAD - sum_t r_t / (1+d)^t + costs) / EAD , 0, 1 )

    Two structural features come out of this that a smooth marginal cannot express:

    * **a hard atom at 1** for the many facilities that recover essentially nothing
      (unsecured, borrower disappeared). Discounting cannot rescue a zero.
    * **discount-driven compression**: a slow full recovery is a *loss*, because the
      money arrived late. Time is a risk driver, not just an inconvenience.
    """
    n = y_latent.numel()
    u = _uniform_from_latent(y_latent)

    n_periods = int(rng.randint(1, int(cfg.get("max_periods", 8)) + 1))
    discount = float(rng.uniform(*cfg.get("discount_range", [0.0, 0.15])))

    # Total recoverable fraction is driven by the latent, so features predict it.
    # The floor is a partial-guarantee / minimum-realisation level: with a floor
    # above the workout costs, EVERY facility recovers something and the book has no
    # total losses at all. Without it the low tail always lands on exactly 1, so the
    # mechanism could not reach the ~2% boundary mass we measured on lgd_lendingclub —
    # it bottomed out near 11%, and the prior has to span the observed range, not just
    # its ugly end.
    floor = float(rng.uniform(*cfg.get("recovery_floor_range", [0.0, 0.5])))
    total = floor + (1.0 - floor) * u
    # A share of facilities recover nothing at all — the dominant unsecured case.
    zero_rate = float(rng.uniform(*cfg.get("zero_recovery_range", [0.0, 0.45])))
    total = torch.where(u < zero_rate, torch.zeros(n), total)

    if shock is not None:
        beta = float(rng.uniform(*cfg.get("shock_beta_range", [0.0, 0.5])))
        total = (total * torch.exp(-beta * shock.clamp(min=0))).clamp(0.0, 1.5)

    # Spread it over the workout period and discount. Front- or back-loaded.
    weights = rng.rand(n_periods) + 0.05
    weights = weights / weights.sum()
    discounted = torch.zeros(n)
    for t, w in enumerate(weights.tolist(), start=1):
        discounted = discounted + total * w / (1.0 + discount) ** t

    cost_mean = float(rng.uniform(*cfg.get("cost_range", [0.0, 0.15])))
    costs = cost_mean * (0.5 + rng.rand(n))
    loss = 1.0 - discounted + costs
    # Same interior guarantee as `lgd_collateral`. With a recovery floor this path is
    # already interior in practice, but the flag has to mean the same thing everywhere or
    # `segment_mixture` (which mixes the two) could reintroduce atoms through one branch.
    if cfg.get("interior_only", False):
        eps = float(cfg.get("interior_margin", 0.02))
        loss = loss.clamp(min=eps).clamp(max=1.0 - eps)
    y = loss.clamp(0.0, 1.0)

    meta = {
        "mechanism": "workout",
        "n_periods": n_periods,
        "discount": round(discount, 4),
        "zero_recovery_rate": round(zero_rate, 4),
        "cost_mean": round(cost_mean, 4),
    }
    return y, meta


def lgd_segment_mixture(
    rng: PriorRNG, y_latent: torch.Tensor, cfg: dict[str, Any], shock: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """A portfolio of 2-3 sub-books, each with its own loss mechanism.

    This is the honest explanation of "LGD is bimodal". A real book is not one weird
    distribution — it is a **mixture**: mortgages recover almost everything, unsecured
    cards recover almost nothing, and the aggregate has a mode at each end. Modelling
    the aggregate directly teaches the shape; modelling the mixture teaches the
    *reason*, and lets the model discover which segment a row belongs to from the
    features, which is what a credit analyst actually does.

    Segment membership is driven by a slice of the latent, so it is **learnable**
    rather than random noise.
    """
    n = y_latent.numel()
    n_seg = int(rng.randint(2, int(cfg.get("max_segments", 3)) + 1))
    u = _uniform_from_latent(y_latent)

    # Split the latent's range into segments with random shares. Membership is a
    # function of the latent, hence predictable from the features.
    cuts = torch.sort(rng.rand(n_seg - 1)).values if n_seg > 1 else torch.tensor([])
    bounds = torch.cat([torch.zeros(1), cuts, torch.ones(1)])
    assign = torch.bucketize(u, bounds[1:-1].contiguous())

    y = torch.zeros(n)
    per_segment = []
    for s in range(n_seg):
        mask = assign == s
        if not bool(mask.any()):
            continue
        # Re-rank inside the segment so each sub-book spans its own full range.
        sub_latent = y_latent[mask]
        mech = "collateral" if rng.boolean(0.5) else "workout"
        fn = lgd_collateral if mech == "collateral" else lgd_workout
        sub_shock = None if shock is None else shock[mask]
        y_sub, m = fn(rng, sub_latent, cfg, sub_shock)
        y[mask] = y_sub
        per_segment.append({"segment": s, "n": int(mask.sum()), **m})

    return y, {"mechanism": "segment_mixture", "n_segments": n_seg, "segments": per_segment}


LGD_MECHANISMS = {
    "collateral": lgd_collateral,
    "workout": lgd_workout,
    "segment_mixture": lgd_segment_mixture,
}


def apply_lgd_mechanism(
    rng: PriorRNG, y_latent: torch.Tensor, cfg: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pick a mechanism and apply it.

    Sampling *which* mechanism, rather than fixing one, is the point: O'Prior's
    finding is that mechanism **diversity** drives transfer, so a prior offering one
    mechanism — however faithful — is the weaker design.
    """
    weights = cfg.get("mechanism_weights") or {
        "collateral": 0.4,
        "workout": 0.35,
        "segment_mixture": 0.25,
    }
    names = [n for n in LGD_MECHANISMS if weights.get(n, 0) > 0]
    if not names:
        raise ValueError("no LGD mechanism has positive weight")
    choice = rng.weighted_choice(names, [float(weights[n]) for n in names])

    shock, cohort_meta = sample_cohort_factor(rng, y_latent.numel(), cfg.get("cohort", {}))

    # atom_prob is the share of datasets that carry boundary atoms AT ALL. It matters
    # because `lgd_lendingclub` has only 1.8% boundary mass — effectively an interior
    # distribution — so a prior where EVERY dataset has atoms cannot represent it.
    # Previously this lever was only read by the marginal-shaping path and did nothing
    # here, which made it an inert entry in the experiment grid.
    atom_prob = float(cfg.get("atom_prob", 0.8))
    wants_atoms = rng.boolean(atom_prob)
    mech_cfg = dict(cfg)
    if not wants_atoms:
        # A book with no total losses and no fully-recovered loans — partially secured
        # lending at moderate LTV, which is a real portfolio type, not a contrivance.
        #
        # `interior_only` is an explicit guarantee rather than an attempt to arrange it
        # by narrowing parameter ranges. The range-narrowing version leaked: coverage is
        # lognormal, its right tail always pushed some rows past full recovery, and those
        # clipped to exactly 0. See `lgd_collateral`.
        mech_cfg["interior_only"] = True
        mech_cfg["zero_recovery_range"] = [0.0, 0.0]
        mech_cfg["unsecured_share_range"] = [0.0, 0.0]
        mech_cfg["recovery_floor_range"] = [0.25, 0.6]

    y, meta = LGD_MECHANISMS[choice](rng, y_latent, mech_cfg, shock)

    # A mechanism that produced a constant target would pass every distributional
    # check and teach nothing. Catch it here rather than in the filter.
    if float(y.std()) < 1e-6:
        raise ValueError(f"LGD mechanism {choice!r} produced a constant target")

    meta.update(cohort_meta)
    meta["mechanism_choice"] = choice
    meta["atom_prob"] = atom_prob
    meta["wants_atoms"] = wants_atoms
    return y.float(), meta


# ---------------------------------------------------------------------------
# PD mechanisms
# ---------------------------------------------------------------------------


def pd_vasicek(
    rng: PriorRNG, y_latent: torch.Tensor, cfg: dict[str, Any], shock: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Default via the Merton/Vasicek one-factor model — the Basel IRB foundation.

    Each borrower has a latent "asset value" made of a systematic and an idiosyncratic
    part, and defaults when it falls below a threshold:

        A_i = sqrt(rho) * Z + sqrt(1 - rho) * eps_i
        default_i = 1 if A_i < Phi^-1(PD)

    Three properties this gives that a plain quantile cut on a latent does not:

    * **the base rate is exact**, because the threshold is `Phi^-1(PD)`;
    * **defaults are correlated** through `Z`, so the realised rate varies between
      cohorts the way it does in a real book. A model trained only on independent
      labels has never seen a bad year;
    * **`rho` is a regulatory quantity** (0.03-0.24 across Basel exposure classes),
      so the correlation is plausible rather than invented.

    `eps_i` is the **SCM latent**, not fresh noise — that is what keeps the features
    predictive. Using noise here would give a perfect-looking base rate and an
    unlearnable task.
    """
    n = y_latent.numel()
    rho = float(rng.uniform(*cfg.get("rho_range", list(BASEL_RHO_RANGE))))
    pd_rate = float(rng.uniform(*cfg.get("base_rate_range", [0.02, 0.30])))

    # Idiosyncratic part: the latent, mapped to a standard normal through its ranks
    # so the Vasicek algebra is on the right scale.
    eps = _normal_icdf(_uniform_from_latent(y_latent))
    z = shock if shock is not None else torch.zeros(n)
    if float(z.std()) < 1e-9:
        # No cohort structure requested: draw one scalar systematic factor so the
        # realised rate still differs from the target rate, as a real sample does.
        z = torch.full((n,), float(rng.randn(1).item()))

    asset = math.sqrt(rho) * z + math.sqrt(1.0 - rho) * eps
    threshold = float(_normal_icdf(torch.tensor([pd_rate])).item())
    y = (asset < threshold).float()

    realised = float(y.mean())
    meta = {
        "mechanism": "vasicek",
        "rho": round(rho, 4),
        "target_base_rate": round(pd_rate, 4),
        "realised_base_rate": round(realised, 4),
        "threshold": round(threshold, 4),
    }
    return y, meta


def pd_ratio_features(
    rng: PriorRNG, X: torch.Tensor, cfg: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Append credit-style **ratio** features (DTI, LTV, utilisation, coverage).

    Credit datasets are dense with ratios — debt-to-income, loan-to-value, credit
    utilisation, interest coverage. Tanna et al. 2026 report 50+ derived features on
    Home Credit, "affordability ratios" among them.

    A ratio is a specific non-linearity: it has a pole where the denominator
    approaches zero, and it is scale-free. TabICL's function families (linear,
    quadratic, MLP, tree, GP, discretisation, softmax, product) include **products**
    but not **quotients**, so this is a genuinely absent functional form, not a
    re-labelling of one already there.
    """
    n, m = X.shape
    if m < 2:
        return X, {"ratio_features": 0}
    k = int(rng.randint(1, min(int(cfg.get("max_ratios", 4)), m) + 1))
    cols = []
    pairs = []
    for _ in range(k):
        i = int(rng.randint(0, m))
        j = int(rng.randint(0, m))
        if i == j:
            continue
        num, den = X[:, i], X[:, j]
        # Shift the denominator off zero rather than clamping the ratio: keeping the
        # heavy tail is the point, since a near-zero denominator is exactly the
        # high-risk case (no income, no collateral).
        scale = den.abs().median() + 1e-3
        ratio = num / (den + torch.sign(den + 1e-12) * 0.1 * scale)
        cols.append(ratio.clamp(-1e4, 1e4).unsqueeze(1))
        pairs.append((i, j))
    if not cols:
        return X, {"ratio_features": 0}
    return torch.cat([X, *cols], dim=1), {"ratio_features": len(cols), "ratio_pairs": pairs}


def pd_monotone_woe(
    rng: PriorRNG, y_latent: torch.Tensor, cfg: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Push the latent through a monotone step function — a scorecard binning.

    Credit scorecards bin every driver and assign a weight-of-evidence value per bin,
    with the bins constrained to be **monotone** in risk. The result is a latent that
    is a coarse staircase in the underlying driver: locally flat, globally monotone.

    That matters because it changes what "learning the function" means. A smooth
    latent rewards interpolation; a staircase rewards finding the cut points. Real
    scored portfolios look like the staircase, because a scorecard produced them.
    """
    n_bins = int(rng.randint(int(cfg.get("min_bins", 3)), int(cfg.get("max_bins", 12)) + 1))
    u = _uniform_from_latent(y_latent)
    edges = torch.linspace(0, 1, n_bins + 1)[1:-1]
    binned = torch.bucketize(u, edges.contiguous()).float()
    # Monotone but unevenly spaced WOE values: real bins are not equally risky.
    steps = torch.cumsum(rng.rand(n_bins) + 0.15, dim=0)
    steps = (steps - steps.mean()) / (steps.std() + 1e-8)
    woe = steps[binned.long().clamp(0, n_bins - 1)]
    return woe, {"woe_bins": n_bins}


PD_MECHANISMS = {"vasicek": pd_vasicek}


def apply_pd_mechanism(
    rng: PriorRNG, y_latent: torch.Tensor, cfg: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Vasicek default assignment, optionally after a scorecard-style binning."""
    meta: dict[str, Any] = {}
    latent = y_latent
    if rng.boolean(float(cfg.get("woe_prob", 0.3))):
        latent, woe_meta = pd_monotone_woe(rng, latent, cfg.get("woe", {}))
        meta.update(woe_meta)

    shock, cohort_meta = sample_cohort_factor(rng, latent.numel(), cfg.get("cohort", {}))
    y, mech_meta = pd_vasicek(rng, latent, cfg, shock)
    meta.update(cohort_meta)
    meta.update(mech_meta)

    # A single-class target is unlearnable and would poison the batch statistics.
    if float(y.mean()) in (0.0, 1.0):
        raise ValueError("Vasicek mechanism produced a single-class target")
    return y, meta
