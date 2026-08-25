"""Metrics for LGD (bounded regression) and PD (imbalanced classification).

The choice of metric is not neutral here, and two traps are worth naming.

**LGD: never lead with a single number from a point prediction.** The target is
often two-humped, with mass at 0 and at 1 and little in between. The mean and the
median of such a distribution both land in the empty middle, so a point prediction
is a value the data almost never takes. R^2 and RMSE are still reported, because
the LGD literature reports them and we need comparability (published R^2 is about
0.04-0.15 for linear models, 0.10-0.25 for beta models, 0.20-0.43 for tree
ensembles), but they are reported *alongside* distributional metrics, and the
decoding rule that produced the point prediction is always recorded.

`boundary_mass_error` is the metric that most directly tests this project's
hypothesis, and no paper in the library reports it: does the model predict the
right *amount* of mass at 0 and at 1?

**PD: never lead with accuracy.** At a 7% base rate, predicting "no default" for
everyone scores 93%. Accuracy is computed and reported only so that a reader who
looks for it sees it next to the base rate and understands why it is useless here.
Ranking (ROC-AUC, PR-AUC) and calibration (Brier, ECE, log-loss) are what matter,
and PR-AUC is the more honest of the two ranking metrics under heavy imbalance.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# LGD
# ---------------------------------------------------------------------------


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, levels: np.ndarray) -> float:
    """Mean pinball loss. `q_pred` is (n, n_levels) aligned with `levels`."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    q_pred = np.asarray(q_pred, dtype=float)
    levels = np.asarray(levels, dtype=float).reshape(1, -1)
    err = y_true - q_pred
    return float(np.maximum(levels * err, (levels - 1.0) * err).mean())


def crps_from_quantiles(y_true: np.ndarray, q_pred: np.ndarray, levels: np.ndarray) -> float:
    """CRPS approximated from a quantile grid.

    The mean pinball loss over a uniform grid of levels converges to CRPS/2 as the
    grid gets finer, so this is 2 x mean pinball. With 999 levels the
    approximation is tight; the factor is applied so the number is comparable to
    CRPS values reported elsewhere.
    """
    return 2.0 * pinball_loss(y_true, q_pred, levels)


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of observations inside [lo, hi]. Compare to the nominal level."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= np.asarray(lo)) & (y_true <= np.asarray(hi))))


def boundary_mass_error(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 1e-6) -> dict[str, float]:
    """Does the model predict the right amount of mass at 0 and at 1?

    The metric this project exists to move, and one nothing in the library
    reports. Positive error means the model predicts too much boundary mass.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_0 = float(np.mean(y_true <= tol))
    true_1 = float(np.mean(y_true >= 1.0 - tol))
    pred_0 = float(np.mean(y_pred <= tol))
    pred_1 = float(np.mean(y_pred >= 1.0 - tol))
    return {
        "true_mass_at_0": true_0,
        "true_mass_at_1": true_1,
        "pred_mass_at_0": pred_0,
        "pred_mass_at_1": pred_1,
        "boundary_mass_err_0": pred_0 - true_0,
        "boundary_mass_err_1": pred_1 - true_1,
        "boundary_mass_abs_err": abs(pred_0 - true_0) + abs(pred_1 - true_1),
    }


def lgd_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantiles: np.ndarray | None = None,
    levels: np.ndarray | None = None,
    decoding: str = "unknown",
) -> dict[str, float | str]:
    """Everything we report for an LGD prediction."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # NON-FINITE PREDICTIONS ARE A FAILURE, AND MUST NOT BE DISGUISED AS ONE.
    # Every metric below silently evaluates to NaN on a NaN input, and `np.var` of an
    # all-NaN array is NaN, which fails the `> EPS` test further down and lands in the
    # CONSTANT-prediction branch. On 14-08-2026 three out-of-domain datasets were reported
    # as `constant_prediction=1.0` when the model had in fact emitted NaN — a numerical
    # blow-up reported as a modelling quirk. Record the fraction explicitly so the two are
    # never confused again.
    nonfinite = ~np.isfinite(y_pred)
    nonfinite_frac = float(np.mean(nonfinite)) if y_pred.size else 0.0

    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    out: dict[str, float | str] = {
        "n_test": int(y_true.size),
        "decoding": decoding,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        # Can be negative when the model is worse than predicting the mean. That is
        # informative on low-signal credit data, not an error — do not clip it.
        "r2": float(1.0 - ss_res / (ss_tot + EPS)),
        "pred_min": float(y_pred.min()),
        "pred_max": float(y_pred.max()),
        "pred_out_of_unit": float(np.mean((y_pred < -1e-6) | (y_pred > 1 + 1e-6))),
    }
    out.update(boundary_mass_error(y_true, y_pred))

    # -- systematic error -----------------------------------------------------
    # `bias` separates "wrong on average" from "noisy": an RMSE of 0.2 made of a constant
    # 0.2 overprediction is a different defect from one made of symmetric scatter, and only
    # the first is fixable by recalibration.
    out["bias"] = float(np.mean(y_pred - y_true))
    out["brier"] = float(np.mean(resid**2))  # same as MSE on a [0,1] target; the credit name

    # `calibration_slope` regresses truth on prediction. 1.0 is calibrated; below 1 means the
    # model is over-confident at the extremes (predicting 0.9 when the truth averages 0.7),
    # which is the characteristic failure on a bounded target and invisible in RMSE.
    out["pred_nonfinite_frac"] = nonfinite_frac
    var_pred = float(np.var(y_pred))
    if nonfinite_frac > 0.0:
        # NOT a constant prediction — a broken one. Kept separate so a sweep can be filtered
        # on it, and so `constant_prediction` keeps meaning what its name says.
        out["calibration_slope"] = float("nan")
        out["nan_predictions"] = 1.0
    elif var_pred > EPS:
        out["calibration_slope"] = float(np.cov(y_true, y_pred)[0, 1] / var_pred)
    else:
        # A model that outputs one value for everything. Not an error worth raising, but it
        # must not be reported as perfectly calibrated.
        out["calibration_slope"] = float("nan")
        out["constant_prediction"] = 1.0

    # -- ranking, which is what a loss-forecasting model is often used for --------
    # A bank ranking exposures by expected loss cares about order more than level, and rank
    # metrics survive a monotone miscalibration that destroys R^2.
    if y_true.size > 2 and np.unique(y_pred).size > 1:
        from scipy.stats import kendalltau, spearmanr

        out["spearman"] = float(spearmanr(y_true, y_pred).statistic)
        out["kendall"] = float(kendalltau(y_true, y_pred).statistic)

    # -- where the error lives ------------------------------------------------
    # THE debugging split for LGD. A model can score a good overall RMSE while being useless
    # exactly on the boundary atoms — the part of the distribution this whole project is about.
    at_boundary = (y_true <= 1e-6) | (y_true >= 1 - 1e-6)
    for label, mask in (("boundary", at_boundary), ("interior", ~at_boundary)):
        if mask.any():
            out[f"mae_{label}"] = float(np.mean(np.abs(resid[mask])))
            out[f"n_{label}"] = int(mask.sum())

    if quantiles is not None and levels is not None:
        q = np.asarray(quantiles, dtype=float)
        lv = np.asarray(levels, dtype=float)
        out["pinball"] = pinball_loss(y_true, q, lv)
        out["crps"] = crps_from_quantiles(y_true, q, lv)
        for nominal in (0.5, 0.8, 0.9):
            lo_l, hi_l = (1 - nominal) / 2, 1 - (1 - nominal) / 2
            lo = q[:, int(np.argmin(np.abs(lv - lo_l)))]
            hi = q[:, int(np.argmin(np.abs(lv - hi_l)))]
            out[f"coverage_{int(nominal * 100)}"] = interval_coverage(y_true, lo, hi)
        # PIT: where each true value falls in its own predicted distribution. If the
        # predictive is right, these are uniform on [0,1], so the distance from uniform is a
        # single number for "is the whole distribution right", not just its middle. Reported
        # as the mean absolute deviation of the PIT histogram from flat.
        pit = np.mean(q <= y_true[:, None], axis=1)
        hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
        out["pit_uniformity_error"] = float(np.mean(np.abs(hist / max(pit.size, 1) - 0.1)))
        out["pit_mean"] = float(pit.mean())  # 0.5 when centred; away from it means biased
    return out


# ---------------------------------------------------------------------------
# PD
# ---------------------------------------------------------------------------


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins.

    Reported because calibration is, per the library synthesis, "claimed
    everywhere and measured almost nowhere" — and for a PD model the probability
    is the product, not the ranking.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def maximum_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """MCE — the WORST bin, where ECE is the average of them weighted by occupancy.

    The pair matters for credit. ECE can look respectable while one bin is badly wrong, and in
    a PD model the bins that matter most — the high-score tail where the defaults are — are
    the sparsest, so occupancy weighting hides exactly the error a lender would care about.
    Report both or report neither.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    worst = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        worst = max(worst, abs(float(y_true[m].mean()) - float(p[m].mean())))
    return float(worst)


def threshold_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """The hard-label family: F1, precision, recall, MCC, balanced accuracy.

    All of them depend on a THRESHOLD, which is why the ranking metrics come first everywhere
    else in this file — 0.5 is arbitrary on an imbalanced target and a model can be excellent
    at ranking while scoring 0 here. Reported anyway, because they are what a downstream credit
    policy actually applies, and because a benchmark table that omits F1 gets asked for it.
    """
    y_true = np.asarray(y_true, dtype=float).ravel() >= 0.5
    yhat = np.asarray(p, dtype=float).ravel() >= threshold
    tp = float(np.sum(yhat & y_true))
    fp = float(np.sum(yhat & ~y_true))
    fn = float(np.sum(~yhat & y_true))
    tn = float(np.sum(~yhat & ~y_true))
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    specificity = tn / max(tn + fp, EPS)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), EPS))
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / max(precision + recall, EPS)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "mcc": float((tp * tn - fp * fn) / denom),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def ks_statistic(y_true: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the largest gap between the score distributions of defaults and
    non-defaults.

    The standard discrimination measure in credit scoring, reported alongside AUC because it
    answers a different question: AUC averages separation over every threshold, KS reports the
    single best one — which is the threshold a cut-off policy would actually use.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    pos, neg = p[y_true >= 0.5], p[y_true < 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    grid = np.unique(p)
    # Empirical CDFs on a shared grid; `searchsorted` keeps it O(n log n) rather than looping.
    cdf_pos = np.searchsorted(np.sort(pos), grid, side="right") / pos.size
    cdf_neg = np.searchsorted(np.sort(neg), grid, side="right") / neg.size
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def recall_at_top_k(y_true: np.ndarray, p: np.ndarray, k_frac: float) -> float:
    """Of all true defaults, how many are in the riskiest `k_frac` of the book?

    The metric a credit team actually acts on: if you can only review 5% of
    applications, what share of the bad ones do you catch?
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    n_pos = float(y_true.sum())
    if n_pos == 0:
        return float("nan")
    k = max(1, int(round(k_frac * y_true.size)))
    top = np.argsort(-p)[:k]
    return float(y_true[top].sum() / n_pos)


def pd_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Everything we report for a PD prediction."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), EPS, 1 - EPS)
    base_rate = float(y_true.mean())

    out: dict[str, float] = {
        "n_test": int(y_true.size),
        "base_rate": base_rate,
        # Reported next to the base rate ON PURPOSE. At a 7% base rate, 0.93
        # accuracy is what you get for predicting "never defaults", so seeing the
        # two side by side is the only honest way to show this number.
        "accuracy_at_0.5": float(np.mean((p >= 0.5) == (y_true >= 0.5))),
        "majority_class_accuracy": max(base_rate, 1 - base_rate),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, p),
        "mce": maximum_calibration_error(y_true, p),
        # A model that just predicts the base rate for everyone. Any log-loss
        # above this is worse than knowing nothing but the average.
        "log_loss_base_rate": float(
            -(base_rate * np.log(max(base_rate, EPS)) + (1 - base_rate) * np.log(max(1 - base_rate, EPS)))
        ),
        "mean_predicted": float(p.mean()),
        "calibration_bias": float(p.mean() - base_rate),
    }

    # Brier decomposed. `brier` alone cannot say WHY a model is wrong; the split says whether
    # it is miscalibrated (fixable by rescaling) or simply cannot separate the classes
    # (not fixable that way). Standard Murphy decomposition: brier = reliability - resolution
    # + uncertainty, where uncertainty is the base rate's own variance and is a property of the
    # dataset rather than the model.
    out["brier_uncertainty"] = float(base_rate * (1 - base_rate))
    out["brier_skill_score"] = float(1.0 - out["brier"] / max(out["brier_uncertainty"], EPS))

    # Hard labels at 0.5, and at the base rate. A single threshold on an imbalanced target is
    # close to meaningless on its own — at a 2 % base rate almost nothing crosses 0.5 and F1
    # collapses to 0 for a model that ranks perfectly — so the base-rate threshold is reported
    # beside it as the version a credit policy would actually pick.
    out.update(threshold_metrics(y_true, p, 0.5))
    for key, value in threshold_metrics(y_true, p, max(base_rate, EPS)).items():
        if key not in ("true_positives", "false_positives", "false_negatives", "true_negatives"):
            out[f"{key}_at_base_rate"] = value

    # `calibration_slope` from a logistic regression of the outcome on the log-odds. 1.0 is
    # calibrated, below 1 over-confident. Named in every credit-scoring validation standard,
    # and the reason it is here rather than only ECE: it gives a DIRECTION, not just a size.
    log_odds = np.log(p / (1 - p))
    if float(np.var(log_odds)) > EPS:
        try:
            from sklearn.linear_model import LogisticRegression

            # C=inf, not penalty=None: unregularised is what a calibration slope means (any
            # shrinkage would bias it toward 0 and make a calibrated model look over-confident),
            # and `penalty=None` is deprecated from sklearn 1.8.
            fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
            fit.fit(log_odds.reshape(-1, 1), (y_true >= 0.5).astype(int))
            out["calibration_slope"] = float(fit.coef_[0][0])
            out["calibration_intercept"] = float(fit.intercept_[0])
        except Exception:  # noqa: BLE001 — a diagnostic must not fail an eval pass
            out["calibration_slope"] = float("nan")
    else:
        out["calibration_slope"] = float("nan")

    if 0.0 < base_rate < 1.0:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["pr_auc"] = float(average_precision_score(y_true, p))
        # Gini = 2*AUC - 1. Redundant with AUC, and included anyway because credit-risk
        # reporting is written in Gini and a reader should not have to convert.
        out["gini"] = float(2.0 * out["roc_auc"] - 1.0)
        # PR-AUC of a random model equals the base rate, so the lift over it is
        # the interpretable version under heavy imbalance.
        out["pr_auc_lift"] = out["pr_auc"] / max(base_rate, EPS)
        out["ks"] = ks_statistic(y_true, p)
        for k in (0.01, 0.05, 0.10, 0.20):
            out[f"recall_at_top_{int(k * 100)}pct"] = recall_at_top_k(y_true, p, k)
    else:
        # A test fold with one class present. Reported as NaN rather than omitted, so a
        # missing column never reads as "this metric was not requested".
        for key in ("roc_auc", "pr_auc", "gini", "ks"):
            out[key] = float("nan")
    return out
