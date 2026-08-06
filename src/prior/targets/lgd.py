"""LGD target family: bounded on [0,1] with a controlled mass at the boundaries.

WHAT REAL LGD LOOKS LIKE (measured from all 7 processed LGD datasets,
2026-08-06 — total mass at *exactly* the min or max; see
`notebooks/data_exploration.ipynb`, which regenerates these numbers):

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

import torch

from ..preprocess import outlier_removing, standard_scaling, to_ranks
from ..rng import PriorRNG


def kumaraswamy_icdf(u: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Inverse CDF of Kumaraswamy(a, b) on (0, 1).

    CDF is F(y) = 1 - (1 - y^a)^b, hence y = (1 - (1-u)^(1/b))^(1/a).
    """
    u = u.clamp(1e-7, 1 - 1e-7)
    return (1.0 - (1.0 - u).pow(1.0 / b)).pow(1.0 / a)


def sample_lgd_shape(rng: PriorRNG, cfg: dict) -> dict:
    """Draw one point from the LGD target family."""
    a_lo, a_hi = cfg.get("shape_ab_range", [0.3, 4.0])
    a = rng.lognum(a_lo, a_hi)
    b = rng.lognum(a_lo, a_hi)

    m_lo, m_hi = cfg.get("boundary_mass_range", [0.02, 0.25])
    atom_prob = float(cfg.get("atom_prob", 0.75))
    max_total = float(cfg.get("max_total_boundary_mass", 0.60))

    p0 = rng.uniform(m_lo, m_hi) if rng.boolean(atom_prob) else 0.0
    p1 = rng.uniform(m_lo, m_hi) if rng.boolean(atom_prob) else 0.0

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
    # two can be varied independently instead of moving together.
    rho = float(cfg.get("signal_strength", 1.0))
    if rho < 1.0:
        z = (y_latent - y_latent.mean()) / (y_latent.std() + 1e-8)
        noise = rng.randn_like(z)
        y_latent = (rho**0.5) * z + ((1.0 - rho) ** 0.5) * noise

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
        raise ValueError(f"unknown LGD target mode {mode!r}; expected 'quantile' or 'censor'")

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
        # Arm A's path. Affine, so the shape survives and only the support moves.
        y = standard_scaling(outlier_removing(y.unsqueeze(-1), threshold=4.0)).squeeze(-1)
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
