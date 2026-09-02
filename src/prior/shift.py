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
* `selection` — reject inference: the context is the APPROVED book (an underwriting screen
               kept the low-risk applicants) while the query reaches into the high-risk
               region the screen turned away. The model must rank applicants the historical
               policy never booked — the credit problem `apply_underwriting_selection`'s
               support truncation only gestures at, because there the rejected region leaves
               the whole table, whereas here it is withheld from the context but kept in the
               query, which is what makes it an extrapolation.

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
    col = int(rng.randint(0, d))
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
    """
    median = float(y.median())
    high = torch.nonzero(y > median, as_tuple=False).flatten()
    low = torch.nonzero(y <= median, as_tuple=False).flatten()
    if len(high) < 8 or len(low) < 8:
        return X, y, {"shift": "none"}

    # How lopsided each half is. 0.5 would be no shift at all.
    ctx_high_frac = float(rng.uniform(*cfg.get("prior_prob_range", [0.15, 0.45])))
    if rng.boolean(0.5):
        ctx_high_frac = 1.0 - ctx_high_frac  # shift can go either direction

    want_high = int(round(cut * ctx_high_frac))
    want_high = max(1, min(len(high) - 1, min(want_high, cut - 1)))
    want_low = cut - want_high
    if want_low < 1 or want_low > len(low) - 1:
        return X, y, {"shift": "none"}

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
        "context_high_rate": round(ctx_rate, 4),
        "query_high_rate": round(qry_rate, 4),
    }


def _selection_shift(
    rng: PriorRNG, X: torch.Tensor, y: torch.Tensor, cfg: dict[str, Any], cut: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Reject inference: context = approved applicants, query = the through-the-door book.

    An underwriting policy approves the low-risk applicants and rejects the high-risk ones,
    so a book only ever holds approved loans — yet a scorecard fitted on it must still rank
    the applicants the old policy turned away. We pose exactly that: score every row on a
    NOISY risk proxy (the screen is informed but imperfect), put the lowest-risk (approved)
    rows in the context, and let the query reach into the higher-risk region the context
    barely covers. No row is dropped — the rejected region is absent from the *context* but
    present in the *query*, so the task is extrapolation across the acceptance boundary, not
    the support truncation `apply_underwriting_selection` does.

    The feature-to-target relationship is untouched: only the split is selective, so the
    region is unseen but learnable — the principle every kind here follows.
    """
    n = int(X.shape[0])
    if float(y.std()) < 1e-9:
        return X, y, {"shift": "none"}  # no risk gradient to select on

    # Higher y = higher risk. A soft screen: risk proxy plus noise, so the approved book is
    # not a clean cut and still contains some of the bad outcomes.
    sharpness = float(cfg.get("selection_sharpness", 0.6))
    z = (y - y.mean()) / (y.std() + 1e-8)
    score = sharpness * z + (1.0 - sharpness) * rng.randn_like(z)
    order = torch.argsort(score)  # ascending: approved (low risk) first
    ctx = order[:cut][rng.randperm(cut)]
    qry = order[cut:][rng.randperm(n - cut)]

    # An approved book with none of the rare class teaches nothing about it. Decline the
    # extreme (a hard screen on a low base rate) rather than hand over a single-class context.
    if bool(((y == 0) | (y == 1)).all()):
        pos = float(y[ctx].sum())
        if pos < 1 or pos > len(ctx) - 1:
            return X, y, {"shift": "none"}

    order = torch.cat([ctx, qry])
    return X[order], y[order], {
        "shift": "selection",
        "shift_cut": cut,
        "context_risk_mean": round(float(y[ctx].float().mean()), 4),
        "query_risk_mean": round(float(y[qry].float().mean()), 4),
    }
