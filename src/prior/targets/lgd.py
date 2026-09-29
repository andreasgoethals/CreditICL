"""LGD target family: bounded on [0,1] with a controlled mass at the boundaries.

WHAT REAL LGD LOOKS LIKE (measured from all 7 processed LGD datasets,
2026-08-06 — total mass at *exactly* the min or max; see
`notebooks/0. General/0.1_data_exploration.ipynb`, which regenerates these numbers):

    0001.heloc              n=58,862   73.0%   dominated by the boundaries
    0003.axa                n= 2,545   34.2%
    0005.base_modelisation  n=   594   27.6%
    0004.base_model         n=   762   22.4%
    0006.lgd_freddie        n=16,002   19.5%   genuinely U-shaped
    0002.loss2              n= 4,637    7.3%
    0007.lgd_lendingclub    n= 5,627    1.8%   effectively interior

So boundary mass spans **1.8% to 73%** — a factor of forty. The premise "bimodal
with point masses at 0 and 1" is strong for about four of the seven and weak for
the rest. A prior that hard-codes any single one of these shapes would overfit to
it and lose on the others. This module therefore samples a **family over boundary
mass and interior shape**, spanning the whole observed range continuously.

(An earlier version of this docstring reported only three datasets and concluded
the premise held for "one of three". That was measured before the other four were
preprocessed, and understated the case.)

THE DESIGN, and why each choice.

1. **Rank-transform the latent first** (`to_ranks`). Strictly monotone, so
   Spearman rho with the SCM latent is exactly 1 and the features carry the same
   information as before. This is the key property: we change the target's
   *marginal* without changing its *predictability*, so any downstream effect is
   attributable to shape alone and not to a confounded change in signal strength.

2. **Shape the interior with a Kumaraswamy inverse CDF.** Closed form
   (no scipy), monotone, and it spans the whole Beta-like family with two
   parameters:

       a<1, b<1  ->  U-shaped                (Freddie-like interior)
       a>1, b>1  ->  unimodal interior       (AXA-like)
       a>1, b<1  ->  mass pushed toward 1    (LendingClub-like)
       a<1, b>1  ->  mass pushed toward 0

   It is also the primitive TabICL already uses (`rand_kumaraswamy_act`), so the
   base prior and this arm share a lineage rather than colliding.

3. **Create the atoms by censoring, not by squashing.** Two modes:

   ``quantile`` (default) — the target is the ICDF of the mixture
   ``p0*delta_0 + p1*delta_1 + (1-p0-p1)*Kumaraswamy(a,b)``. Boundary masses come
   out **exactly** p0 and p1, which is what an experiment needs.

   ``censor`` — build a latent loss fraction on a *wider* interval and clip it to
   [0,1]. This is the honest economic story: recoveries can exceed exposure
   (LGD <= 0, booked as 0) and workout costs can exceed it (LGD >= 1, booked as
   1). Boundary mass is emergent rather than dialled in.

   Both destroy information in the tails — once a row is booked at 0 you cannot
   tell a 100% from a 120% recovery — which is a real feature of the task, not a
   defect of the simulation.

4. **`target_scaling` is a lever, not a default.** Arm A standard-scales the
   target unconditionally. Because that map is affine it preserves shape but
   destroys the [0,1] support, so "does the model benefit from seeing the actual
   bounded scale?" is a separate, cheap question from "does the shape help?".
   Keep them separable.

To make *every* dataset carry boundary atoms (a reasonable strong-arm setting),
set ``atom_prob: 1.0`` and give ``boundary_mass_range`` a positive lower bound.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ...utils.target_stats import target_stats
from ..preprocess import standard_scaling, to_ranks
from ..rng import PriorRNG


def kumaraswamy_icdf(u: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Inverse CDF of Kumaraswamy(a, b) on (0, 1).

    CDF is F(y) = 1 - (1 - y^a)^b, hence y = (1 - (1-u)^(1/b))^(1/a).
    """
    u = u.clamp(1e-7, 1 - 1e-7)
    return (1.0 - (1.0 - u).pow(1.0 / b)).pow(1.0 / a)


def sample_lgd_shape(rng: PriorRNG, cfg: dict) -> dict:
    """Draw one point from the LGD target family.

    The interior is Kumaraswamy(a, b) with `a` and `b` drawn log-uniformly from their own
    ranges — `interior_a_range`, `interior_b_range` — so the interior can lean towards small
    losses (a < b: most recoveries nearly complete, a long tail of large losses — the shape
    of most real LGD data) without losing the other shapes. Both fall back to
    `interior_shape_range`, then to the old `shape_ab_range`. (Until 28-09-2026 the configs set
    `interior_shape_range`, which nothing read: the interior used the default [0.3, 4.0].)

    Each boundary atom has its own chance of existing (`atom_prob_0`, `atom_prob_1`) and its own
    mass range (`boundary_mass_0_range`, `boundary_mass_1_range`), both falling back to the
    shared `atom_prob` / `boundary_mass_range`: real LGD tables range from no atoms at all to
    half the rows at exactly 1.
    """
    shared = cfg.get("interior_shape_range", cfg.get("shape_ab_range", [0.3, 4.0]))
    a_lo, a_hi = cfg.get("interior_a_range", shared)
    b_lo, b_hi = cfg.get("interior_b_range", shared)
    a = rng.lognum(float(a_lo), float(a_hi))
    b = rng.lognum(float(b_lo), float(b_hi))

    m_lo, m_hi = cfg.get("boundary_mass_range", [0.02, 0.25])
    lo0, hi0 = cfg.get("boundary_mass_0_range", [m_lo, m_hi])
    lo1, hi1 = cfg.get("boundary_mass_1_range", [m_lo, m_hi])
    atom_prob = float(cfg.get("atom_prob", 0.75))
    max_total = float(cfg.get("max_total_boundary_mass", 0.60))
    # `lo + (hi - lo) * U ** power`: 1 is uniform; above 1 most tables get a small atom and a few a
    # large one, which is how the real atoms spread (at 1: 0.3 %, 4 %, 6 %, 8 %, 12 %, 16 %, 52 %). A
    # uniform draw wide enough to reach `heloc`'s 52 % put the median far above the real 8 %.
    pow0 = float(cfg.get("boundary_mass_0_power", cfg.get("boundary_mass_power", 1.0)))
    pow1 = float(cfg.get("boundary_mass_1_power", cfg.get("boundary_mass_power", 1.0)))

    def mass(lo: Any, hi: Any, power: float) -> float:
        return float(lo) + (float(hi) - float(lo)) * rng.uniform(0.0, 1.0) ** power

    p0 = mass(lo0, hi0, pow0) if rng.boolean(float(cfg.get("atom_prob_0", atom_prob))) else 0.0
    p1 = mass(lo1, hi1, pow1) if rng.boolean(float(cfg.get("atom_prob_1", atom_prob))) else 0.0

    # Keep genuine interior mass; without this, a draw can degenerate to a
    # two-point distribution, which the trivial/degenerate filters would bin.
    if p0 + p1 > max_total:
        scale = max_total / (p0 + p1)
        p0, p1 = p0 * scale, p1 * scale

    return {"a": a, "b": b, "p0": p0, "p1": p1}


def apply_lgd_target(
    rng: PriorRNG,
    y_latent: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict]:
    """Map an SCM latent to a bounded LGD-like target.

    Returns (y, meta). `meta` records the realised boundary masses so the prior's
    actual output distribution can be reported rather than assumed.
    """
    mode = cfg.get("mode", "quantile")
    shape = sample_lgd_shape(rng, cfg)
    a, b, p0, p1 = shape["a"], shape["b"], shape["p0"], shape["p1"]

    # Optional signal dilution: mix noise into the latent BEFORE ranking. This
    # decouples "what shape is the target" from "how predictable is it", so the
    # two can be varied independently instead of moving together. A `[lo, hi]`
    # `signal_strength_range` draws it per table, as real tables differ in difficulty.
    s_range = cfg.get("signal_strength_range")
    rho = (float(rng.uniform(float(s_range[0]), float(s_range[1]))) if s_range is not None
           else float(cfg.get("signal_strength", 1.0)))
    if rho < 1.0:
        z = (y_latent - y_latent.mean()) / (y_latent.std() + 1e-8)
        noise = rng.randn_like(z)
        y_latent = (rho**0.5) * z + ((1.0 - rho) ** 0.5) * noise

    # MECHANISM mode: derive the target from credit economics (collateral coverage,
    # workout cashflows, portfolio segments) instead of shaping its marginal. The
    # boundary atoms then EMERGE from over-collateralisation and total loss rather
    # than being dialled in via atom_prob. See src/prior/targets/mechanisms.py; this
    # is the arm O'Prior's "mechanism diversity beats observational realism" finding
    # points at.
    if mode == "mechanism":
        from .mechanisms import apply_lgd_mechanism

        y, mech_meta = apply_lgd_mechanism(rng, y_latent, cfg.get("mechanism", {}))
        stats = target_stats(y)
        meta = {
            "target": "lgd",
            "mode": mode,
            "frac_at_0": stats["frac_at_min"],
            "frac_at_1": stats["frac_at_max"],
            **mech_meta,
        }
        if cfg.get("target_scaling", "none") == "standard":
            y = standard_scaling(y.unsqueeze(1)).squeeze(1)
            meta["target_scaling"] = "standard"
        return y.float(), meta

    u = to_ranks(y_latent)

    if mode == "quantile":
        interior = 1.0 - p0 - p1
        y = torch.zeros_like(u)
        mid = (u >= p0) & (u <= 1.0 - p1)
        if interior > 1e-6:
            u_mid = ((u[mid] - p0) / interior).clamp(1e-7, 1 - 1e-7)
            y[mid] = kumaraswamy_icdf(u_mid, a, b)
        y[u > 1.0 - p1] = 1.0
        # rows with u < p0 stay 0.0

    elif mode == "censor":
        # Stretch onto (-m0, 1+m1), then clip. Boundary mass is emergent.
        v = kumaraswamy_icdf(u, a, b)
        m0 = p0 / max(1.0 - p0 - p1, 1e-3)
        m1 = p1 / max(1.0 - p0 - p1, 1e-3)
        y = (-m0 + (1.0 + m0 + m1) * v).clamp(0.0, 1.0)

    else:
        raise ValueError(
            f"unknown LGD target mode {mode!r}; expected 'quantile', 'censor' or 'mechanism' "
            f"('zoib' is applied by the generator through `apply_lgd_zoib`, which needs the features)"
        )

    # Optional recording granularity: real LGD is derived from currency amounts
    # and is often stored rounded, which clusters mass on a lattice.
    grid = cfg.get("round_to", 0) or 0
    if grid and rng.boolean(float(cfg.get("round_prob", 0.25))):
        y = torch.round(y * grid) / grid

    # Hard clip to [0,1] on every credit dataset, unconditionally.
    #
    # This is the cheapest and probably the single most useful thing this whole
    # module does. LGD is a *fraction* — it cannot leave [0,1] — and nothing in
    # the base prior knows that. Clipping alone encodes "this target is bounded",
    # which is the one property every LGD dataset shares, whatever its shape.
    # Both modes above already land inside [0,1]; this guards against rounding
    # and against any future mode that does not.
    y = y.clamp(0.0, 1.0)

    realised_p0 = float((y <= 0.0).float().mean())
    realised_p1 = float((y >= 1.0).float().mean())

    scaling = cfg.get("target_scaling", "none")
    if scaling == "standard":
        # What `TabICLRegressor.fit` does at prediction time: standardise, nothing else. Affine,
        # so the shape survives and only the support moves. NO outlier clipping first (it used to
        # be here): with a rare atom — 1 % of rows at 1 — the atom sits more than 4 SD out and
        # clipping erases it, and prediction never clips the target. The generator applies this
        # step itself, last, so missingness can couple to the raw [0, 1] target first.
        y = standard_scaling(y.unsqueeze(-1)).squeeze(-1)
    elif scaling != "none":
        raise ValueError(f"unknown target_scaling {scaling!r}; expected 'none' or 'standard'")

    meta = {
        "target": "lgd",
        "mode": mode,
        "kuma_a": a,
        "kuma_b": b,
        "target_p0": p0,
        "target_p1": p1,
        "realised_p0": realised_p0,
        "realised_p1": realised_p1,
        "signal_strength": rho,
        "target_scaling": scaling,
    }
    return y.float(), meta


# ---------------------------------------------------------------------------
# ZOIB mode — the literature's LGD data-generating process (added 29-09-2026)
# ---------------------------------------------------------------------------
#
# The zero-and-one inflated beta (ZOIB) regression of Ospina & Ferrari (2010), in the
# parameterisation Li, Zhang & Zhao use to SIMULATE LGD data ("Modeling Loss Given Default
# Regressions", OCC/FDIC working paper, version of 11-05-2020, section 2.1, eqs. 1-6; following
# Li et al. 2016 and Yashkir & Yashkir 2013):
#
#     P(LGD = 0) = e^{x a} / (1 + e^{x a} + e^{x b})        (a full recovery)
#     P(LGD = 1) = e^{x b} / (1 + e^{x a} + e^{x b})        (a total loss)
#     LGD | interior ~ Beta(mu phi, (1 - mu) phi),  mu = logistic(x g)
#
# with a macroeconomic factor shared by every default of a period (their x2, the quarterly
# unemployment rate, one value per 10,000 defaults) among the explanatory variables. Their true
# values: intercepts a0 = -0.54, b0 = -1.46, g0 = 0; precision phi = 5; nine covariates with
# coefficients +0.4 (a), -0.1 (b), -0.1 (g) on N(0, 0.5^2) variables; macro coefficients -5, +6,
# +0.5 on the unemployment rate. A covariate that raises the chance of a full recovery lowers the
# chance of a total loss and the interior mean; a downturn does the opposite (Altman et al. 2005:
# recoveries fall when defaults rise).
#
# That is ONE portfolio. A prior needs a distribution over portfolios, so each table draws its
# own parameters (`sample_zoib_params`): intercepts normal around the published values, the
# precision log-uniform around phi = 5, covariate and macro loadings uniform over ranges that
# contain the published effect sizes, with the published signs. Nothing here is set from the
# evaluation datasets (docs/PRIORS.md).


def sample_zoib_params(rng: PriorRNG, cfg: dict) -> dict[str, float]:
    """One portfolio's ZOIB parameters. Defaults are the published values, spread as documented."""

    def normal(key: str, default: tuple[float, float]) -> float:
        mean, sd = cfg.get(key, default)
        return float(mean) + float(sd) * float(rng.randn(1).item())

    def uni(key: str, default: tuple[float, float]) -> float:
        lo, hi = cfg.get(key, default)
        return float(rng.uniform(float(lo), float(hi)))

    phi_lo, phi_hi = cfg.get("phi_range", [1.0, 20.0])
    return {
        "a0": normal("alpha0_mean_sd", (-0.54, 1.0)),
        "b0": normal("beta0_mean_sd", (-1.46, 1.0)),
        "g0": normal("gamma0_mean_sd", (0.0, 1.0)),
        "phi": float(rng.lognum(float(phi_lo), float(phi_hi))),
        # Covariate loadings: the SD of the covariate part of each linear predictor. Published:
        # 0.4 x 0.5 x sqrt(9) = 0.6 (P0), 0.1 x 0.5 x 3 = 0.15 (P1 and mu).
        "a_s": uni("loading_p0_range", (0.2, 1.5)),
        "b_s": uni("loading_p1_range", (0.1, 1.5)),
        "g_s": uni("loading_mu_range", (0.1, 1.0)),
        # Macro loadings per SD of the period factor. Published: -5, +6, +0.5 times the SD of the
        # 2006-2015 unemployment rate (about 0.018) = -0.09, +0.11, +0.01.
        "a_m": uni("macro_p0_range", (-0.3, 0.0)),
        "b_m": uni("macro_p1_range", (0.0, 0.3)),
        "g_m": uni("macro_mu_range", (0.0, 0.1)),
        "overlap": uni("component_overlap_range", (0.3, 1.0)),
        "signal_share": uni("signal_share_range", (0.1, 1.0)),
    }


def _period_factor(rng: PriorRNG, n: int, cfg: dict) -> tuple[torch.Tensor, int]:
    """A standard-normal macro value per period, constant within the period's contiguous block of
    rows (Li, Zhang & Zhao: one unemployment rate per quarter's defaults). One period = none."""
    n_periods = int(rng.randint(int(cfg.get("min_cohorts", 1)), int(cfg.get("max_cohorts", 40)) + 1))
    if n_periods <= 1:
        return torch.zeros(n), 1
    edges = torch.linspace(0, n, n_periods + 1).long()
    draws = rng.randn(n_periods)
    factor = torch.zeros(n)
    for c in range(n_periods):
        factor[edges[c]: edges[c + 1]] = draws[c]
    return factor, n_periods


def _standardised(z: torch.Tensor) -> torch.Tensor:
    return (z - z.mean()) / (z.std() + 1e-8)


def apply_lgd_zoib(
    rng: PriorRNG,
    X: torch.Tensor,
    y_latent: torch.Tensor,
    cfg: dict,
    max_features: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Draw an LGD target from a ZOIB regression on the table's own features.

    THE DRIVERS. The graph's target node — the SCM latent, rank-mapped to a standard normal so
    only its ordering matters — is the table's main recovery driver `s`. `signal_share` of it is
    kept and the rest replaced by noise: the drivers a lender never records (Li, Zhang & Zhao
    drop four or eight of their nine covariates when fitting, i.e. keep 11-56 % of the covariate
    signal). The three parts of the model each read their own driver (Ospina & Ferrari give P0,
    P1 and mu separate coefficient vectors): `overlap` of `s` plus a random projection of the
    features, so collateral can decide full recoveries while seniority moves the interior.

    THE PERIOD FACTOR is shared by every row of a period and enters all three predictors with
    the published signs. With `macro_feature_prob` it is also a column — as the unemployment rate
    is an explanatory variable in Li, Zhang & Zhao — otherwise it is an unrecorded driver.

    Returns the raw target on [0, 1]; the generator standardises it on the context afterwards
    when the encoding asks for it, as `TabICLRegressor` does to a real one.
    """
    from .mechanisms import _normal_icdf, _uniform_from_latent

    n = y_latent.numel()
    p = sample_zoib_params(rng, cfg)

    s = _normal_icdf(_uniform_from_latent(y_latent))
    share = p["signal_share"]
    if share < 1.0:
        s = math.sqrt(share) * s + math.sqrt(1.0 - share) * rng.randn(n)
    s = _standardised(s)

    feats = X.float()
    if feats.shape[1]:
        feats = feats[:, feats.std(0) > 1e-8]
    if feats.shape[1]:
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)

    def driver() -> torch.Tensor:
        """`overlap` of the main driver plus a random direction in feature space."""
        if feats.shape[1] == 0 or p["overlap"] >= 1.0:
            return s
        proj = _standardised(feats @ rng.randn(feats.shape[1]))
        return _standardised(math.sqrt(p["overlap"]) * s + math.sqrt(1.0 - p["overlap"]) * proj)

    d0, d1, dmu = driver(), driver(), driver()
    macro, n_periods = _period_factor(rng, n, cfg.get("cohort", {}))

    # The published signs: a better recovery driver raises P(0) and lowers P(1) and mu; a downturn
    # (macro > 0) lowers P(0) and raises P(1) and mu.
    eta0 = p["a0"] + p["a_s"] * d0 + p["a_m"] * macro
    eta1 = p["b0"] - p["b_s"] * d1 + p["b_m"] * macro
    eta_mu = p["g0"] - p["g_s"] * dmu + p["g_m"] * macro
    probs = torch.softmax(torch.stack([torch.zeros(n), eta0, eta1], dim=1), dim=1)  # interior, P0, P1
    mu = torch.sigmoid(eta_mu).clamp(1e-4, 1 - 1e-4)

    u = rng.rand(n)
    p0, p1 = probs[:, 1], probs[:, 2]
    at0 = u < p0
    at1 = (~at0) & (u < p0 + p1)
    phi = p["phi"]
    draws = rng.np.beta((mu * phi).double().numpy(), ((1.0 - mu) * phi).double().numpy())
    # A beta draw can round to exactly 0 or 1 in float32; an interior loss is strictly inside.
    interior = torch.from_numpy(draws).float().clamp(1e-6, 1.0 - 1e-6)
    y = torch.where(at0, torch.zeros(n), torch.where(at1, torch.ones(n), interior))

    meta: dict[str, Any] = {
        "target": "lgd",
        "mode": "zoib",
        **{f"zoib_{k}": round(v, 4) for k, v in p.items()},
        "periods": n_periods,
        "frac_at_0": round(float(at0.float().mean()), 4),
        "frac_at_1": round(float(at1.float().mean()), 4),
        "expected_p0": round(float(p0.mean()), 4),
        "expected_p1": round(float(p1.mean()), 4),
    }
    if n_periods > 1 and rng.boolean(float(cfg.get("macro_feature_prob", 0.5))) and X.shape[1] < max_features:
        X = torch.cat([X, macro.unsqueeze(1).to(X.dtype)], dim=1)
        meta["macro_feature"] = True
    return X, y.float(), meta
