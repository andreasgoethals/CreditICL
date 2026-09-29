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

MEASURED BASE RATES in data/raw/pd/ (2026-08-05): GMSC 6.7%, Home Credit 8.1%,
HMEQ 20.0%, Taiwan 22.1% — i.e. roughly 6-22%, not the 1-5% often assumed. The
default `base_rate_range` brackets that with a little tail on each side.

FIVE COMPONENTS, each independently gated so they can be attributed separately.

1. **Base rate** — cut the latent at its (1 - pi) empirical quantile, so the
   positive rate is exactly pi. Direction is randomised, matching upstream's
   label permutation.

2. **Signal dilution** — noise is mixed into the latent used to *assign labels*,
   while the features still reflect the clean latent. That lowers the achievable
   AUC smoothly, which is how you reach credit's 0.70-0.85 regime without
   crippling the task. Diluting what the *model sees* instead would change the
   feature distribution too, confounding the two effects.

3. **Asymmetric label noise** — defaults that cure get booked as non-default far
   more often than the reverse. Symmetric flipping misses that asymmetry, which
   is exactly the part that biases a PD model.

4. **Threshold rules** — Klein & Hoffart 2026's charge is that a statistical
   prior can only fuzzily approximate a hard ``if amount >= 5000`` rule, and
   credit data is shaped by underwriting cutoffs and Basel/IFRS9 thresholds. This
   adds sharp axis-aligned and conjunctive rules to the latent. Note this follows
   from a *position* paper with no experiments, so it is the most speculative
   component here.

5. **Underwriting selection and informative missingness** — you only observe
   approved applicants (support truncation), and a thin file is itself a risk
   signal, so missingness is MNAR and target-linked rather than MCAR.
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
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Drop the rows an underwriting screen would have rejected.

    This is **support truncation**: the observed book is a biased sample, selected on a
    score correlated with the outcome, so estimates learned from it are biased and no
    general-purpose prior produces that. It is NOT reject inference — it removes the rejected
    rows from the whole table, so context and query are truncated identically and the model
    is never asked to extrapolate across the acceptance boundary. That extrapolation is posed
    separately by the `selection` shift kind (`shift.py`), which withholds the rejected region
    from the context while keeping it in the query. Only the quantile-mode path calls this;
    Both target modes call it (the mechanism mode only since 28-09-2026).
    """
    drop = _per_table(rng, cfg, "selection_drop", 0.0)
    if drop <= 0.0 or X.shape[0] < 32:
        return X, latent, {"selection_drop": 0.0}

    drop = min(drop, 0.5)
    # Screen on a noisy version of the latent: policy is informed but imperfect.
    sharpness = _per_table(rng, cfg, "selection_sharpness", 0.7)
    z = (latent - latent.mean()) / (latent.std() + 1e-8)
    score = sharpness * z + (1.0 - sharpness) * rng.randn_like(z)

    n_keep = int(round(X.shape[0] * (1.0 - drop)))
    keep = torch.argsort(score)[:n_keep]  # reject the highest-risk applicants
    keep = keep[torch.argsort(keep)]  # restore row order; the model is not order-invariant across train/test
    return X[keep], latent[keep], {"selection_drop": drop, "selection_sharpness": sharpness}


def apply_informative_missingness(
    rng: PriorRNG,
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: dict,
    max_features: int,
    fit_rows: int | None = None,
    fill: str = "mean",
) -> tuple[torch.Tensor, dict]:
    """MNAR missingness whose rate depends on the target, plus optional indicators.

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
    coupling = cfg.get("missing_target_coupling_range", cfg.get("missing_target_coupling", 1.0))
    # A `[lo, hi]` range draws the strength per table: in real data it runs from none to
    # strong (hmeq: whether DEBTINC is missing alone gives an AUC of 0.78).
    beta = float(rng.uniform(float(coupling[0]), float(coupling[1]))) if isinstance(coupling, (list, tuple)) else float(coupling)
    add_indicators = bool(cfg.get("missing_indicators", True))

    X = X.clone()
    indicators = []
    for j in cols.tolist():
        target_rate = rng.uniform(rate_lo, rate_hi)
        # Logit intercept chosen so the marginal missing rate hits target_rate.
        signed = 2.0 * y - 1.0
        logits = beta * signed + math.log(target_rate / (1.0 - target_rate))
        mask = rng.rand_like(logits) < torch.sigmoid(logits)
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
        X, latent, sel_meta = apply_underwriting_selection(rng, X, latent, cfg.get("selection", {}))
        meta.update(sel_meta)
        y, mech_meta = apply_pd_mechanism(rng, latent, cfg.get("mechanism", {}))
        period = mech_meta.pop("_period_factor", None)
        meta.update(mech_meta)
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
        # SAME GUARD AS THE QUANTILE PATH BELOW: the model and the loss need two classes. Rescue
        # a single-class draw by quantile-cutting the scored latent to a floor base rate.
        if float(y.sum()) < 2 or float(y.sum()) > y.numel() - 2:
            zc = (latent - latent.mean()) / (latent.std() + 1e-8)
            y = _quantile_cut(zc, 3.0 / max(y.numel(), 1))
            meta["mechanism_single_class_rescued"] = True
        meta["mode"] = "mechanism"
        meta["realised_base_rate"] = float(y.mean())
        return X, y.float(), meta

    # 1. Hard policy rules on top of the smooth SCM latent.
    latent, rule_meta = apply_threshold_rules(rng, X, y_latent, cfg.get("rules", {}))
    meta.update(rule_meta)

    # 2. Selection: the observed book is only the approved applicants.
    X, latent, sel_meta = apply_underwriting_selection(rng, X, latent, cfg.get("selection", {}))
    meta.update(sel_meta)

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
    if rng.boolean():
        z = -z  # randomise which tail is "default", as upstream permutes labels
    y = _quantile_cut(z, base_rate)
    meta["target_base_rate"] = base_rate

    # 5. Asymmetric label noise: cures (1->0) are far more common than
    #    unobserved-default relabelling (0->1).
    y, noise_meta = apply_label_noise(rng, y, cfg)
    meta.update(noise_meta)

    # Guard: the model and the loss both need two classes present.
    if float(y.sum()) < 2 or float(y.sum()) > y.numel() - 2:
        y = _quantile_cut(z, max(base_rate, 3.0 / max(y.numel(), 1)))

    # 6. Informative missingness USED TO BE APPLIED HERE, from `credit.target.missingness`.
    # It moved to `TaskGenerator._sample_candidate` on 25-08-2026 so that BOTH tracks get it:
    # `apply_lgd_target` is never handed X, so LGD could not have applied it, and the
    # `credit.missingness` block in the LGD configs was read by nothing at all — four settings
    # that looked active and were dead. docs/PRIORS.md lists missingness under "Both tracks".
    # The call site is unchanged in ORDER (after the target, before noise features), so PD's
    # output distribution is the same.

    meta["realised_base_rate"] = float(y.mean())
    return X, y, meta
