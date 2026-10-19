"""PD target family: imbalanced binary labels with credit-shaped generating structure.

WHAT ARM A ALREADY DOES, and why that changes the intervention. TabICL's
`_prior_config.py` sets ``"balanced": False``, so `BalancedBinarize` (median
split) is *not* the active path — `MulticlassAssigner` is, and in ``mode="rank"``
it cuts the latent at a **uniformly random data row**:

    boundary_indices = torch.randint(0, T, (num_classes - 1,))
    boundaries = input[boundary_indices]

For binary targets that makes the minority rate roughly Uniform(0,1), so on the
order of 10% of arm A's binary tasks already sit below 5% minority. Prior-side
imbalance in TabICL is therefore **uncontrolled and unmeasured, not missing** —
the opposite of O'Prior, which explicitly rejects tasks with *"collapsed or
severely imbalanced support classes"*.

So this module **reshapes an existing distribution** rather than adding a missing
mechanism. That is a cleaner intervention, and it means the arm-A base-rate
distribution is itself a reportable measurement.

The base-rate parameter describes applicants before selection. Realised accepted-book rates
can differ; invalid single-class tables are rejected and redrawn without fabricating labels.
"""

from __future__ import annotations

import math

import torch

from ..rng import PriorRNG


def _per_table(rng: PriorRNG, cfg: dict, key: str, default: float) -> float:
    """`<key>_range: [lo, hi]` drawn uniformly per table, else the fixed `<key>`."""
    span = cfg.get(f"{key}_range")
    if span is not None:
        return float(rng.uniform(float(span[0]), float(span[1])))
    return float(cfg.get(key, default))


def max_selection_drop(cfg: dict) -> float:
    """The largest share of rows the screen can remove — what the generator oversamples for."""
    span = cfg.get("selection_drop_range")
    return float(span[1]) if span is not None else float(cfg.get("selection_drop", 0.0))


def _quantile_cut(latent: torch.Tensor, positive_rate: float) -> torch.Tensor:
    """Label the top `positive_rate` fraction of `latent` as 1. Exact by construction."""
    n = latent.numel()
    n_pos = max(1, min(n - 1, int(round(positive_rate * n))))
    order = torch.argsort(latent, descending=True)
    y = torch.zeros(n, dtype=torch.float32)
    y[order[:n_pos]] = 1.0
    return y


def apply_threshold_rules(
    rng: PriorRNG,
    X: torch.Tensor,
    latent: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict]:
    """Overlay hard axis-aligned decision rules on the latent.

    Thresholds are placed at *quantiles* of each column, so they bite regardless
    of the column's scale, and weights are large enough to matter against a
    standardised latent.
    """
    # `n_rules_range: [lo, hi]` draws the count per table (inclusive), so some books have no hard
    # cut-offs at all; `n_rules` is a fixed count.
    counts = cfg.get("n_rules_range")
    n_rules = (int(rng.randint(int(counts[0]), int(counts[1]) + 1)) if counts is not None
               else int(cfg.get("n_rules", 0)))
    if n_rules <= 0 or X.shape[1] == 0:
        return latent, {"n_rules": 0}

    w_lo, w_hi = cfg.get("rule_weight_range", [0.5, 2.5])
    q_lo, q_hi = cfg.get("rule_quantile_range", [0.1, 0.9])
    conj_prob = float(cfg.get("conjunction_prob", 0.3))

    latent = (latent - latent.mean()) / (latent.std() + 1e-8)
    applied = 0
    for _ in range(n_rules):
        j = rng.randint(0, X.shape[1])
        col = X[:, j]
        thr = torch.quantile(col, rng.uniform(q_lo, q_hi))
        indicator = (col >= thr).float()

        if rng.boolean(conj_prob) and X.shape[1] > 1:
            # Conjunctive policy: "high utilisation AND short history".
            k = rng.randint(0, X.shape[1])
            thr_k = torch.quantile(X[:, k], rng.uniform(q_lo, q_hi))
            indicator = indicator * (X[:, k] <= thr_k).float()

        weight = rng.uniform(w_lo, w_hi) * (1.0 if rng.boolean() else -1.0)
        latent = latent + weight * indicator
        applied += 1

    return latent, {"n_rules": applied}


def apply_underwriting_selection(
    rng: PriorRNG,
    X: torch.Tensor,
    latent: torch.Tensor,
    cfg: dict,
    *,
    higher_is_risk: bool = True,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Drop the rows an underwriting screen would have rejected.

    This is **support truncation**: the observed book is a biased sample, selected on a
    score correlated with the outcome, so estimates learned from it are biased and no
    general-purpose prior produces that. It is NOT reject inference — it removes the rejected
    rows from the whole table, so context and query are truncated identically and the model
    is never asked to extrapolate across the acceptance boundary. That extrapolation is posed
    separately by the `selection` shift kind (`shift.py`), which partitions on
    observed covariates without using labels. Both target modes call this screen.
    """
    drop = _per_table(rng, cfg, "selection_drop", 0.0)
    if drop <= 0.0 or X.shape[0] < 32:
        result = (X, latent, {"selection_drop": 0.0})
        return (*result, torch.arange(len(latent))) if return_indices else result

    drop = min(drop, 0.5)
    # Screen on a noisy version of the latent: policy is informed but imperfect.
    sharpness = _per_table(rng, cfg, "selection_sharpness", 0.7)
    z = (latent - latent.mean()) / (latent.std() + 1e-8)
    if not higher_is_risk:
        z = -z
    score = sharpness * z + (1.0 - sharpness) * rng.randn_like(z)

    n_keep = int(round(X.shape[0] * (1.0 - drop)))
    keep = torch.argsort(score)[:n_keep]  # reject the highest-risk applicants
    keep = keep[torch.argsort(keep)]  # restore row order; the model is not order-invariant across train/test
    result = (X[keep], latent[keep], {"selection_drop": drop, "selection_sharpness": sharpness})
    return (*result, keep) if return_indices else result


def apply_informative_missingness(
    rng: PriorRNG,
    X: torch.Tensor,
    cfg: dict,
    max_features: int,
    fit_rows: int | None = None,
    fill: str = "mean",
) -> tuple[torch.Tensor, dict]:
    """Missingness driven by a shared borrower propensity, never the realised outcome.

    `fill="mean"` fills a gap as TabICL's wrapper does (the observed context mean); `fill="nan"`
    leaves it missing, for a model that handles missing values itself (TabPFN).

    A thin credit file — no bureau record, no income documentation — is itself a
    risk signal, so the *fact* of missingness carries information. Arm A's prior
    has no missingness at all, and TabICLv2 mean-imputes at inference, which
    discards exactly this signal.
    """
    rate_lo, rate_hi = cfg.get("missing_rate_range", [0.05, 0.35])
    frac = cfg.get("missing_col_fraction_range", cfg.get("missing_col_fraction", 0.0))
    # A `[lo, hi]` range draws the share per table: real credit tables go from one gappy column
    # to nearly all of them.
    frac_cols = float(rng.uniform(float(frac[0]), float(frac[1]))) if isinstance(frac, (list, tuple)) else float(frac)
    if frac_cols <= 0.0 or X.shape[1] == 0:
        return X, {"missing_cols": 0}

    n_cols = max(1, int(round(frac_cols * X.shape[1])))
    cols = rng.randperm(X.shape[1])[:n_cols]
    coupling = cfg.get("missing_risk_coupling_range", cfg.get("missing_risk_coupling", 1.0))
    # A broad synthetic dependence strength, not calibrated to benchmark outcomes.
    beta = float(rng.uniform(float(coupling[0]), float(coupling[1]))) if isinstance(coupling, (list, tuple)) else float(coupling)
    add_indicators = bool(cfg.get("missing_indicators", True))

    # A noisy covariate score can be associated with risk through the shared features.
    # No y argument: flipping outcome noise cannot change the missing-value mask.
    from .mechanisms import _normal_icdf, _uniform_from_latent

    z = torch.nan_to_num(X.float())
    z = (z - z.mean(0)) / z.std(0).clamp_min(1e-6)
    direction = rng.randn(z.shape[1])
    score = _normal_icdf(_uniform_from_latent(z @ direction))
    propensity = 0.5 * score + (0.75 ** 0.5) * rng.randn(len(X))
    # One borrower-level draw shared by all columns: adding columns cannot provide
    # an ever-growing set of independent votes for the same hidden target.
    row_uniform = rng.rand(len(X))
    X = X.clone()
    indicators = []
    for j in cols.tolist():
        target_rate = rng.uniform(rate_lo, rate_hi)
        # Logit intercept chosen so the marginal missing rate hits target_rate.
        signed = propensity
        if not 0 < target_rate < 1:
            raise ValueError("missing_rate_range must be inside (0, 1)")
        lower, upper = -40.0 - abs(beta), 40.0 + abs(beta)
        for _ in range(35):
            intercept = (lower + upper) / 2
            if float(torch.sigmoid(beta * signed + intercept).mean()) < target_rate:
                lower = intercept
            else:
                upper = intercept
        logits = beta * signed + (lower + upper) / 2
        mask = row_uniform < torch.sigmoid(logits)
        if mask.all() or not mask.any():
            continue
        # Impute with the observed mean, as TabICLv2 does at inference — of the CONTEXT rows
        # (`fit_rows`) when given, which is where upstream's imputer is fitted.
        observed = ~mask
        if fit_rows is not None:
            observed = observed & (torch.arange(len(mask)) < int(fit_rows))
            if not observed.any():
                continue
        X[mask, j] = float("nan") if fill == "nan" else X[observed, j].mean()
        if add_indicators:
            indicators.append(mask.float())

    if indicators and X.shape[1] + len(indicators) <= max_features:
        X = torch.cat([X, torch.stack(indicators, dim=-1)], dim=-1)

    return X, {"missing_cols": len(cols), "missing_indicators": len(indicators)}


def apply_label_noise(rng: PriorRNG, y: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """Asymmetric label noise on a 0/1 default flag: cures (1 -> 0, `flip_pos_to_neg`) are far
    commoner than defaults that go unrecorded (0 -> 1, `flip_neg_to_pos`). Each can be a fixed
    rate or a `[lo, hi]` range drawn per table."""
    def rate(key: str) -> float:
        # `<key>_range: [lo, hi]` draws per table (a `_range` key is never a sweep axis);
        # `<key>: value` is a fixed rate.
        v = cfg.get(f"{key}_range", cfg.get(key, 0.0))
        return float(rng.uniform(*v)) if isinstance(v, (list, tuple)) else float(v)

    q10, q01 = rate("flip_pos_to_neg"), rate("flip_neg_to_pos")
    if q10 > 0 or q01 > 0:
        u = rng.rand_like(y)
        flip_pos = (y == 1) & (u < q10)
        flip_neg = (y == 0) & (u < q01)
        y = y.clone()
        y[flip_pos] = 0.0
        y[flip_neg] = 1.0
    return y, {"flip_pos_to_neg": round(q10, 4), "flip_neg_to_pos": round(q01, 4)}


def apply_pd_target(
    rng: PriorRNG,
    X: torch.Tensor,
    y_latent: torch.Tensor,
    cfg: dict,
    max_features: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Turn an SCM latent into an imbalanced, credit-shaped binary target."""
    meta: dict = {"target": "pd"}

    # MECHANISM mode: assign defaults with the Merton/Vasicek one-factor model — the
    # basis of the Basel IRB formula — so defaults are CORRELATED through a systematic
    # factor instead of independent given the features. A prior of independent labels
    # has never shown the model a bad year.
    if str(cfg.get("mode", "quantile")) == "mechanism":
        from .mechanisms import apply_pd_mechanism

        latent, rule_meta = apply_threshold_rules(rng, X, y_latent, cfg.get("rules", {}))
        meta.update(rule_meta)
        # Selection: the observed book holds only the approved applicants. Until 28-09-2026 this
        # branch returned before selection and before the label noise below, so Exp1's credit
        # tasks had neither although the config set both (docs/AGENTS_MEMORY.md, 28-09-2026).
        y, mech_meta = apply_pd_mechanism(rng, latent, cfg.get("mechanism", {}))
        period = mech_meta.pop("_period_factor", None)
        meta.update(mech_meta)
        # Set applicant outcomes BEFORE screening. Re-ranking only accepted
        # applicants otherwise resets their default rate and erases selection.
        X, latent, sel_meta, keep = apply_underwriting_selection(
            rng, X, latent, cfg.get("selection", {}), higher_is_risk=False, return_indices=True)
        y = y[keep]
        period = period[keep] if period is not None else None
        meta.update(sel_meta)
        # The period factor as a column, with `macro_feature_prob`: macroeconomic variables are
        # explanatory variables in credit models (economic cycle dependency; Bellotti & Crook
        # 2012, as surveyed by Baesens et al. 2026). Otherwise it stays an unrecorded driver.
        mcfg = cfg.get("mechanism", {}) or {}
        if (period is not None and meta.get("cohorts", 1) > 1 and X.shape[1] < max_features
                and rng.boolean(float(mcfg.get("macro_feature_prob", 0.0)))):
            X = torch.cat([X, period.unsqueeze(1).to(X.dtype)], dim=1)
            meta["macro_feature"] = True
        y, noise_meta = apply_label_noise(rng, y, cfg)
        meta.update(noise_meta)
        # Invalid single-class tables are rejected by TaskGenerator, never relabelled.
        meta["applicant_base_rate"] = mech_meta["realised_base_rate"]
        meta["accepted_base_rate"] = float(y.mean())
        meta["mode"] = "mechanism"
        meta["realised_base_rate"] = float(y.mean())
        return X, y.float(), meta

    # 1. Hard policy rules on top of the smooth SCM latent.
    latent, rule_meta = apply_threshold_rules(rng, X, y_latent, cfg.get("rules", {}))
    meta.update(rule_meta)

    # 3. Signal dilution — applied to the LABEL-assigning latent only, so the
    #    features keep their distribution and only learnability changes.
    rho = float(cfg.get("signal_strength", 1.0))
    z = (latent - latent.mean()) / (latent.std() + 1e-8)
    if rho < 1.0:
        z = (rho**0.5) * z + ((1.0 - rho) ** 0.5) * rng.randn_like(z)
    meta["signal_strength"] = rho

    # 4. Base rate.
    br_lo, br_hi = cfg.get("base_rate_range", [0.01, 0.30])
    base_rate = rng.lognum(br_lo, br_hi)
    flipped = rng.boolean()
    if flipped:
        z = -z  # randomise which tail is "default", as upstream permutes labels
    y = _quantile_cut(z, base_rate)
    meta["target_base_rate"] = base_rate
    X, latent, sel_meta, keep = apply_underwriting_selection(
        rng, X, latent, cfg.get("selection", {}), higher_is_risk=not flipped, return_indices=True)
    z, y = z[keep], y[keep]
    meta.update(sel_meta)

    # 5. Asymmetric label noise: cures (1->0) are far more common than
    #    unobserved-default relabelling (0->1).
    y, noise_meta = apply_label_noise(rng, y, cfg)
    meta.update(noise_meta)

    # 6. Informative missingness USED TO BE APPLIED HERE, from `credit.target.missingness`.
    # It moved to `TaskGenerator._sample_candidate` on 25-08-2026 so that BOTH tracks get it:
    # `apply_lgd_target` is never handed X, so LGD could not have applied it, and the
    # `credit.missingness` block in the LGD configs was read by nothing at all — four settings
    # that looked active and were dead. docs/PRIORS.md lists missingness under "Both tracks".
    # The call site is unchanged in ORDER (after the target, before noise features), so PD's
    # output distribution is the same.

    meta["realised_base_rate"] = float(y.mean())
    return X, y, meta
