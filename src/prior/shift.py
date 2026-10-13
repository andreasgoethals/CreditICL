"""Distribution shift between the context rows and the rows being predicted.

THE GAP THIS FILLS

O'Prior's ablations report that three things contribute **independently** to transfer:
mechanism diversity, realism composition, and **shift-aware stress**. Our prior had the
first two. This is the third, and it is the one that matters most for credit. (O'Prior
measured that independence on classification tasks only; the LGD track carries it over as a
stated extrapolation, per docs/EXPERIMENTAL_DESIGN.md.)

Every other part of the prior draws context and query rows from the *same* table, so a
model can succeed by interpolating inside one population. Real credit scoring never
works that way: a scorecard is fitted on loans originated up to some date and then
applied to next quarter's applications. The population moves — the economy turns, the
lender changes its cut-offs, the product mix drifts. That is why banks re-calibrate
scorecards, and why Basel requires a *downturn* LGD rather than an average one.

A prior without shift teaches the model to interpolate. A prior with it teaches the
model to notice when the query rows do not look like the context, which is exactly the
skill an in-context model needs on a real book.

FOUR KINDS, all standard in the credit literature:

* `cohort`   — context = early vintages, query = late ones. Uses the systematic factor
               already in `mechanisms.py`, so the query genuinely sits under a different
               economic state rather than a jittered copy.
* `covariate` — the feature distribution moves (applicants get younger, loans larger)
               while the relationship between features and target holds. This is
               "covariate shift" proper, and it is the benign case.
* `prior_prob` — the base rate itself changes: the query book defaults more, or less,
               than the context did. This is the dangerous case, because a model that
               anchors on the context's base rate is systematically wrong.
* `selection` — a noisy feature-based acceptance policy puts approved applicants
               in context and rejected applicants in query. It changes feature
               support without consulting future labels. It does not guarantee
               that rejected applicants are riskier or model MNAR reject inference.

WHAT IS DELIBERATELY *NOT* DONE: the feature-to-target relationship is never broken.
Shifting the inputs is a solvable problem a good model should handle; changing the
underlying function between context and query would make the task unlearnable rather
than hard, and the model would just learn to ignore the context.

ROW ORDER IS THE MECHANISM. `dataset.py` takes the first `train_size` rows as context,
so arranging the rows *is* how the shift is applied. Nothing downstream needs to change.
"""

from __future__ import annotations

from typing import Any

import torch

from .rng import PriorRNG

SHIFT_KINDS = ("cohort", "covariate", "prior_prob", "selection")


def _split_point(n_rows: int, train_frac: float) -> int:
    """Where the context ends. Kept away from the extremes so both halves are usable."""
    return max(8, min(n_rows - 8, int(round(n_rows * train_frac))))


def apply_shift(
    rng: PriorRNG,
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: dict[str, Any],
    train_frac: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Maybe introduce a context-to-query shift. Returns (X, y, meta).

    Applied to a `shift_prob` fraction of datasets, not all of them: a model that only
    ever saw shifted tasks would learn to distrust its context everywhere, which is just
    as wrong as trusting it blindly. The mix is the point.
    """
    shift_prob = float(cfg.get("shift_prob", 0.0))
    if shift_prob <= 0 or not rng.boolean(shift_prob):
        return X, y, {"shift": "none"}

    weights = cfg.get("kind_weights") or {
        "cohort": 1.0,
        "covariate": 1.0,
        "prior_prob": 1.0,
        "selection": 1.0,
    }
    kinds = [k for k in SHIFT_KINDS if weights.get(k, 0) > 0]
    if not kinds:
        return X, y, {"shift": "none"}
    kind = rng.weighted_choice(kinds, [float(weights[k]) for k in kinds])

    n = int(X.shape[0])
    cut = _split_point(n, train_frac)

    if kind == "cohort":
        return _cohort_shift(rng, X, y, cfg, cut)
    if kind == "covariate":
        return _covariate_shift(rng, X, y, cfg, cut)
    if kind == "selection":
        return _selection_shift(rng, X, y, cfg, cut)
    return _prior_prob_shift(rng, X, y, cfg, cut)


def _cohort_shift(
    rng: PriorRNG, X: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], cut: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Context = early vintages, query = late ones.

    `mechanisms.sample_cohort_factor` already assigns rows to contiguous vintage blocks
    with a shared shock, so the rows arrive in time order. Leaving them in that order
    and splitting means the query sits under a *later* economic state than the context —
    which is the actual deployment condition for a scorecard.

    Rows are shuffled WITHIN each half so the model cannot exploit position; the shift
    should come from the distribution, not from a row index it could memorise.
    """
    n = int(X.shape[0])
    ctx = torch.arange(0, cut)[rng.randperm(cut)]
    qry = torch.arange(cut, n)[rng.randperm(n - cut)]
    order = torch.cat([ctx, qry])
    return X[order], y[order], {"shift": "cohort", "shift_cut": cut}


def _covariate_shift(
    rng: PriorRNG, X: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], cut: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The feature distribution moves; the feature-to-target relationship does not.

    Implemented by sorting on one feature and splitting: the context then covers the low
    end of that feature and the query the high end, so the query is genuinely outside the
    range the model was shown. Sorting rather than adding a constant offset matters —
    an offset is a shift the model can undo by re-centring, whereas an unseen *range*
    requires actual extrapolation.
    """
    n, d = int(X.shape[0]), int(X.shape[1])
    if d == 0:
        return X, y, {"shift": "none"}
    usable = torch.nonzero(torch.isfinite(X).all(dim=0) & (X.std(dim=0) > 1e-9)).flatten()
    if not len(usable):
        return X, y, {"shift": "none"}
    col = int(usable[int(rng.randint(0, len(usable)))])
    values = X[:, col]
    if not bool(torch.isfinite(values).all()) or float(values.std()) < 1e-9:
        return X, y, {"shift": "none"}

    order = torch.argsort(values, descending=bool(rng.boolean(0.5)))
    # Shuffle within halves, for the same reason as `_cohort_shift`.
    ctx = order[:cut][rng.randperm(cut)]
    qry = order[cut:][rng.randperm(n - cut)]
    order = torch.cat([ctx, qry])
    return X[order], y[order], {"shift": "covariate", "shift_feature": col, "shift_cut": cut}


def _prior_prob_shift(
    rng: PriorRNG, X: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], cut: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The base rate (or mean target) differs between context and query.

    The dangerous shift: a model that anchors on the context's default rate is
    systematically miscalibrated on the query. Real books do this constantly — a
    recession, a change in acceptance policy, a new product.

    Built by over-sampling high-target rows into one half and low-target rows into the
    other. Rows are only REARRANGED, never invented, so the marginal of the table as a
    whole is untouched and the shift is purely about how it was split.

    The size of the move is a RATE RATIO: the query's share of high rows (for PD, its default
    rate) is the context's times r, log-uniform in `prior_prob_ratio_range`, r or 1/r with equal
    chance. A ratio is how a default rate moves: under the Vasicek model this prior already
    uses (`mechanisms.py`), a one-to-two standard-deviation move of the systematic factor
    multiplies a book's default rate by 1.1 to 6.1 over the prior's rho in [0.03, 0.24] and
    default rates of 1-50 % (docs/PRIORS.md). Until 29-09-2026 the knob set the context's share
    of high rows directly (`prior_prob_range`), which assumes that group is half the table; for
    a binary target it is the default rate, so a 7 % book had almost every default put in the
    context and the query kept one.
    """
    median = float(y.median())
    high = torch.nonzero(y > median, as_tuple=False).flatten()
    low = torch.nonzero(y <= median, as_tuple=False).flatten()
    if len(high) < 8 or len(low) < 8:
        return X, y, {"shift": "none"}

    ratio = rng.lognum(*(float(v) for v in cfg.get("prior_prob_ratio_range", [1.25, 4.0])))
    if rng.boolean(0.5):
        ratio = 1.0 / ratio  # the query can default more, or less, than the context
    # The shares that keep every row: context c and query ratio * c, with
    # c * cut + ratio * c * (n - cut) = len(high); a query cannot be more than all high rows.
    n = int(y.shape[0])
    ctx_share = len(high) / (cut + ratio * (n - cut))
    if ratio * ctx_share > 1.0:
        ctx_share = (len(high) - (n - cut)) / cut

    # Each half keeps at least one row of each group, so both still hold both classes.
    fewest, most = max(1, cut - (len(low) - 1)), min(len(high) - 1, cut - 1)
    if fewest > most:
        return X, y, {"shift": "none"}
    want_high = max(fewest, min(most, int(round(cut * ctx_share))))
    want_low = cut - want_high

    high = high[rng.randperm(len(high))]
    low = low[rng.randperm(len(low))]
    ctx = torch.cat([high[:want_high], low[:want_low]])
    qry = torch.cat([high[want_high:], low[want_low:]])
    ctx = ctx[rng.randperm(len(ctx))]
    qry = qry[rng.randperm(len(qry))]
    order = torch.cat([ctx, qry])

    ctx_rate = float((y[ctx] > median).float().mean())
    qry_rate = float((y[qry] > median).float().mean())
    return X[order], y[order], {
        "shift": "prior_prob",
        "shift_cut": cut,
        "rate_ratio": round(ratio, 4),
        "context_high_rate": round(ctx_rate, 4),
        "query_high_rate": round(qry_rate, 4),
    }


def _selection_shift(
    rng: PriorRNG, X: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], cut: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Split by a noisy feature policy, independently of realised outcomes.

    All rows are retained; low policy scores form context, high scores form query.
    Risk means in the metadata are diagnostics, never inputs to the policy.
    """
    n = int(X.shape[0])
    usable = torch.isfinite(X).all(dim=0) & (X.std(dim=0) > 1e-9)
    if not usable.any():
        return X, y, {"shift": "none"}
    # A policy can inspect applicant features, not the future realised outcome.
    features = X[:, usable]
    features = (features - features.mean(dim=0)) / features.std(dim=0).clamp_min(1e-8)
    direction = rng.randn_like(features[0])
    proxy = features @ direction
    z = (proxy - proxy.mean()) / proxy.std().clamp_min(1e-8)
    sharpness = float(cfg.get("selection_sharpness", 0.35))
    score = sharpness * z + (1.0 - sharpness) * rng.randn_like(z)
    order = torch.argsort(score)  # ascending: approved (low policy score) first
    ctx = order[:cut][rng.randperm(cut)]
    qry = order[cut:][rng.randperm(n - cut)]

    order = torch.cat([ctx, qry])
    return X[order], y[order], {
        "shift": "selection",
        "shift_cut": cut,
        "context_risk_mean": round(float(y[ctx].float().mean()), 4),
        "query_risk_mean": round(float(y[qry].float().mean()), 4),
    }
