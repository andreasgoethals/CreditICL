"""The evaluation protocol, in one place: how a dataset is split, how a threshold is chosen,
which checkpoints are scored. Every model — ours, released TabICLv2, TabPFN-3, CatBoost, the
linear floor — goes through exactly this.

THE PROTOCOL (version 7, 13-10-2026)

* **5-fold cross-validation**, stratified on the default flag for PD, shuffled with a fixed
  seed, so every model sees the same folds. Each row is scored exactly once, by a model whose
  context (or training set) did not contain it.
* **The whole training pool is the context.** No row cap, no context selection, no
  subsampling: the four training folds go to the model as they are.
* **All eligible feature columns, the SAME ones for every model.** No width cap, association
  ranking or near-constant filter. The prediction-time feature policy still excludes targets,
  identifiers and information unavailable at the declared prediction time.
* **No training-row deletion.** Rare/extreme fitted rows are retained. Test rows are never
  removed; numeric test values are winsorised using training-fold bounds.
* **PD: the decision threshold maximises F1 on a validation split.** 20 % of the training pool
  (stratified) is held out, the model is fitted on the other 80 %, and the threshold with the
  highest validation F1 is kept. The model is then refitted on the WHOLE training pool and the
  test fold is scored at that threshold. The test fold never touches the choice.
* **Every metric at once**: threshold-free (ROC-AUC, PR-AUC, Brier, log-loss, ECE, KS, ...) and
  at the tuned threshold (F1, precision, recall, MCC, balanced accuracy); LGD adds the
  predictive distribution (CRPS, pinball, interval coverage) where the model gives one.
* **Every saved checkpoint**, not only the final one: `checkpoint_steps` lists them.

Version 2 (until 28-09-2026) scored one 80/20 split per seed with the context capped at 1,024
rows and the foundation models' training set capped at 10,000 rows, and reported F1 only at the
fixed thresholds 0.5 and the base rate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Bumped whenever the protocol changes, so a completion receipt written under an older one is
#: never mistaken for a current result (`benchmark_status`). 4 (30-09-2026): TabPFNCredit's
#: fold preprocessing (below) and the feature cap it replaces PCA with.
PROTOCOL = 7  # full rows AND features; no project-level selection or TabPFN feature subsampling
N_FOLDS = 5
#: Share of the training pool held out to choose the PD threshold.
VALIDATION_FRACTION = 0.2
#: The synthetic training width is NOT a benchmark input limit.
MAX_FEATURES: int | None = None
#: Quantile levels requested from models with a predictive distribution: 99, fine enough that
#: `2 x mean pinball` approximates CRPS closely.
QUANTILE_LEVELS = np.round(np.linspace(0.01, 0.99, 99), 2)


# ---------------------------------------------------------------------------------------------
# FOLD PREPROCESSING — TabPFNCredit's post-split steps (`1. TabPFN/TabPFNCredit/src/data/
# data_feeder.py`, `DataFeeder._apply_post_split_preprocessing`, as its main benchmark ran them),
# with optional legacy filters for explicit ablations only. Everything is learned on the rows a model
# is FITTED on and applied to the rows it predicts, so the test fold never shapes it. Identical
# for every model.
# ---------------------------------------------------------------------------------------------

#: A column whose single most common value covers more than this share of the fitted rows goes
#: (missing values count in the denominator; a column with no observed value stays).
NEAR_CONSTANT_THRESHOLD = 0.99
#: A fitted row goes when any numeric value is BOTH in the 0.1 % tails AND more than 5 IQR from
#: the median — rare and extreme; validation and test rows are never removed.
OUTLIER_TAIL = 0.001
OUTLIER_IQR = 5.0
MIN_ROWS_FOR_OUTLIERS = 100
MIN_ROWS_AFTER_OUTLIERS = 20
#: Statistics of a column need at least this many observed values.
MIN_OBSERVED = 20
#: Predicted rows are clipped to the fitted rows' 0.1st and 99.9th percentiles (numeric columns).
WINSORIZE = (0.001, 0.999)
#: Ranking method for an explicitly requested legacy feature-budget ablation only.
FEATURE_SELECTION = "univariate"
#: A categorical column is scored through its per-category mean target, shrunk toward the overall
#: mean with this many pseudo-rows, so a rare category cannot look perfectly predictive.
CATEGORY_SMOOTHING = 10.0


def _rank_score(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per column of `Z` (no missing values): |AUC - 0.5| for a 0/1 target, |Spearman rho| for any
    other. Both depend on the ORDER of a column's values only — no unit, no scale, no outlier can
    move them."""
    from scipy.stats import rankdata

    R = rankdata(Z, axis=0)
    binary = np.isin(y, (0.0, 1.0)).all()
    if binary:
        pos = y == 1
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 == 0 or n0 == 0:
            return np.zeros(Z.shape[1])
        auc = (R[pos].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        return np.abs(auc - 0.5)
    ry = rankdata(y)
    Rc = R - R.mean(axis=0)
    ryc = ry - ry.mean()
    denom = np.sqrt((Rc ** 2).sum(axis=0) * (ryc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (Rc * ryc[:, None]).sum(axis=0) / denom
    return np.abs(np.nan_to_num(rho, nan=0.0))


def rank_columns(X: np.ndarray, y: np.ndarray, cat_indices: list[int], *,
                 method: str = FEATURE_SELECTION) -> np.ndarray:
    """Column ranking for explicit legacy ablations, computed on fitted rows only.

    `univariate` (protocols 4–6): each column's rank association with the target —
    |AUC - 0.5| for PD, |Spearman rho| for LGD. The filter family foundation-model pipelines use
    for too-wide tables (Mitra v2's F-test top-256, TabForestPFN's F-test, the TabPFN docs'
    mutual information), in its rank form: indifferent to units and to any monotone transform,
    which raw variance is not (it ranks a balance in cents above one in thousands), and robust to
    the heavy tails of credit data. A missing value ranks below every observed one, so informative
    missingness counts. A categorical column is scored through its category means, smoothed by
    `CATEGORY_SMOOTHING`. Refitted on every set of fitted rows, so it cannot leak the test fold.

    `variance`: the raw variance of the observed values (label-free; kept for comparison)."""
    X = np.asarray(X, dtype=np.float64)
    if method == "variance":
        with warnings.catch_warnings():  # an all-missing column warns; it ranks last below anyway
            warnings.simplefilter("ignore", RuntimeWarning)
            score = np.nanvar(np.where(np.isfinite(X), X, np.nan), axis=0)
    elif method == "univariate":
        y = np.asarray(y, dtype=np.float64)
        Z = X.copy()
        cats = set(int(c) for c in cat_indices)
        for j in range(Z.shape[1]):
            col = Z[:, j]
            if j in cats:
                codes = np.where(col >= 0, col, -1.0)  # a missing category is a category
                overall = float(y.mean())
                values, inverse = np.unique(codes, return_inverse=True)
                sums = np.bincount(inverse, weights=y, minlength=len(values))
                counts = np.bincount(inverse, minlength=len(values))
                Z[:, j] = ((sums + CATEGORY_SMOOTHING * overall) / (counts + CATEGORY_SMOOTHING))[inverse]
            else:
                missing = ~np.isfinite(col)
                if missing.all():
                    Z[:, j] = 0.0
                elif missing.any():
                    col[missing] = np.nanmin(col[~missing]) - 1.0  # below every observed value
        score = _rank_score(Z, y)
    else:
        raise ValueError(f"unknown feature selection {method!r}")
    score = np.where(np.isfinite(score), score, -np.inf)
    # stable: equal scores keep their column order, so the choice is deterministic
    return np.argsort(-score, kind="stable")


@dataclass
class FoldPreprocessing:
    """What was learned on one set of fitted rows, to apply to the rows being predicted."""

    keep: np.ndarray  # kept column indices, in their original order
    cat_indices: list[int]  # categorical columns, renumbered to `keep`
    lower: np.ndarray  # per kept column: winsorising bounds, NaN where a column is not clipped
    upper: np.ndarray
    report: dict[str, Any] = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The predicted rows: the kept columns, numeric ones clipped. Missing stays missing."""
        Z = np.array(np.asarray(X, dtype=np.float32)[:, self.keep], dtype=np.float32)
        clip = np.isfinite(self.lower)
        if clip.any():
            Z[:, clip] = np.clip(Z[:, clip], self.lower[clip], self.upper[clip])
        return Z


def _near_constant(X: np.ndarray, cat_indices: set[int]) -> np.ndarray:
    """Boolean per column: one value covers more than `NEAR_CONSTANT_THRESHOLD` of the rows.

    A value covering more than half of the rows is the median of the observed values, so it is
    enough to count the median — one vectorised pass instead of TabPFNCredit's `np.unique` per
    column, with the same answer. Categorical codes use -1 for a missing category."""
    n = len(X)
    out = np.zeros(X.shape[1], dtype=bool)
    if n == 0:
        return out
    num = np.array([j for j in range(X.shape[1]) if j not in cat_indices], dtype=int)
    if num.size:
        with warnings.catch_warnings():  # an all-missing column: median NaN, count 0, kept
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X[:, num], axis=0)
        out[num] = (X[:, num] == med).sum(axis=0) / n > NEAR_CONSTANT_THRESHOLD
    for j in cat_indices:
        col = X[:, j]
        observed = col[col >= 0]
        if observed.size:
            out[j] = (col == np.median(observed)).sum() / n > NEAR_CONSTANT_THRESHOLD
    return out


def _outlier_rows(X: np.ndarray, numeric: list[int]) -> np.ndarray:
    """Boolean per row: some numeric value is in the 0.1 % tails and beyond 5 IQR of the median."""
    n = len(X)
    rows = np.zeros(n, dtype=bool)
    if n < MIN_ROWS_FOR_OUTLIERS:
        return rows
    for j in numeric:
        col = X[:, j]
        observed = ~np.isnan(col)
        if observed.sum() < MIN_OBSERVED:
            continue
        lo, q1, med, q3, hi = np.percentile(col[observed], [OUTLIER_TAIL * 100, 25, 50, 75,
                                                             (1 - OUTLIER_TAIL) * 100])
        iqr = q3 - q1
        if iqr == 0:
            continue
        with np.errstate(invalid="ignore"):
            rows |= observed & ((col < lo) | (col > hi)) & (np.abs(col - med) > OUTLIER_IQR * iqr)
    if rows.any() and n - rows.sum() < MIN_ROWS_AFTER_OUTLIERS:
        return np.zeros(n, dtype=bool)
    return rows


def fit_fold_preprocessing(X_fit: np.ndarray, y_fit: np.ndarray, cat_indices: list[int], *,
                           max_features: int | None = MAX_FEATURES,
                           selection: str | None = None,
                           remove_near_constant: bool = False,
                           remove_outliers: bool = False) -> tuple[FoldPreprocessing, np.ndarray, np.ndarray]:
    """Learn the fold preprocessing on the fitted rows. Returns the preprocessing and the fitted
    rows ready for the model: `(prep, X_fit_clean, y_fit_clean)`.

    All rows and columns are retained by default. Legacy filters are opt-in for explicit
    ablations. Prediction winsorisation learns only from fitted rows."""
    X = np.asarray(X_fit, dtype=np.float32)
    y = np.asarray(y_fit)
    cats = {int(c) for c in cat_indices}
    d = X.shape[1]

    constant = _near_constant(X, cats) if remove_near_constant else np.zeros(d, dtype=bool)
    keep = np.flatnonzero(~constant)
    numeric = [int(j) for j in keep if j not in cats]

    outliers = _outlier_rows(X, numeric) if remove_outliers else np.zeros(len(X), dtype=bool)
    if outliers.any():
        X, y = X[~outliers], y[~outliers]

    selected_from = int(keep.size)
    if max_features is not None and keep.size > max_features:
        kept_cats = [pos for pos, j in enumerate(keep) if j in cats]
        order = rank_columns(X[:, keep], y, kept_cats, method=selection or FEATURE_SELECTION)
        keep = np.sort(keep[order[:max_features]])

    lower = np.full(keep.size, np.nan, dtype=np.float32)
    upper = np.full(keep.size, np.nan, dtype=np.float32)
    for pos, j in enumerate(keep):
        if j in cats:
            continue
        col = X[:, j]
        observed = col[~np.isnan(col)]
        if observed.size >= MIN_OBSERVED:
            lower[pos], upper[pos] = np.percentile(observed, [WINSORIZE[0] * 100, WINSORIZE[1] * 100])

    position = {int(old): new for new, old in enumerate(keep.tolist())}
    prep = FoldPreprocessing(
        keep=keep,
        cat_indices=[position[c] for c in sorted(cats) if c in position],
        lower=lower,
        upper=upper,
        report={
            "features_in": d,
            "near_constant_dropped": int(constant.sum()),
            "outlier_rows_removed": int(outliers.sum()),
            "features_used": int(keep.size),
            **({"features_selected_from": selected_from,
                "feature_selection": selection or FEATURE_SELECTION} if selected_from > keep.size else {}),
        },
    )
    return prep, np.array(X[:, keep], dtype=np.float32), y


def cv_folds(y: np.ndarray, task: str, *, n_folds: int = N_FOLDS, seed: int = 0,
             groups: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """`(train_idx, test_idx)` for each fold. Stratified on the class for PD (so a 7 %-default
    table keeps defaults in every fold), plain shuffled folds for a continuous target."""
    from sklearn.model_selection import KFold, StratifiedKFold

    y = np.asarray(y)
    index = np.arange(len(y))
    if groups is not None:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
        if len(groups) != len(y) or len(np.unique(groups)) < n_folds:
            raise ValueError("Cross-validation requires aligned groups and at least one group per fold")
        splitter = (StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed) if task == "pd"
                    else GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed))
        return list(splitter.split(index, y, groups))
    if task == "pd":
        labels = (y >= 0.5).astype(int)
        if np.bincount(labels, minlength=2).min() >= n_folds:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return [(tr, te) for tr, te in splitter.split(index, labels)]
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(index)]


def validation_split(y: np.ndarray, task: str, *, fraction: float = VALIDATION_FRACTION,
                     seed: int = 0, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`(fit_idx, val_idx)` inside a training pool, stratified for PD."""
    from sklearn.model_selection import train_test_split

    index = np.arange(len(y))
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        # Choose among deterministic group-disjoint splits. Exact stratification
        # is not always possible when an entire minority class belongs to one group.
        candidates = list(GroupShuffleSplit(n_splits=32 if task == "pd" else 1,
                                           test_size=fraction, random_state=seed).split(index, y, groups))
        if task == "pd":
            labels = np.asarray(y) >= 0.5
            valid = [(a, b) for a, b in candidates
                     if len(np.unique(labels[a])) == len(np.unique(labels[b])) == 2]
            if not valid:
                raise ValueError("No group-disjoint threshold split contains both classes")
            candidates = sorted(valid, key=lambda pair: (
                abs(float(labels[pair[1]].mean()) - float(labels.mean()))
                + abs(len(pair[1]) / len(y) - fraction)))
        fit_idx, val_idx = candidates[0]
        return np.sort(fit_idx), np.sort(val_idx)
    strat = (np.asarray(y) >= 0.5).astype(int) if task == "pd" else None
    if strat is not None and np.bincount(strat, minlength=2).min() < 2:
        strat = None
    fit_idx, val_idx = train_test_split(index, test_size=fraction, random_state=seed, stratify=strat)
    return np.sort(fit_idx), np.sort(val_idx)


def f1_optimal_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """The probability threshold with the highest F1 for the positive class (default = 1), and
    that F1. Scanned over every distinct predicted probability (`precision_recall_curve`)."""
    from sklearn.metrics import precision_recall_curve

    y = (np.asarray(y, dtype=float).ravel() >= 0.5).astype(int)
    p = np.asarray(p, dtype=float).ravel()
    if y.min() == y.max():
        return 0.5, float("nan")
    precision, recall, thresholds = precision_recall_curve(y, p)
    # The last precision/recall pair has no threshold (recall 0 by construction).
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def checkpoint_steps(cfg: dict[str, Any]) -> list[int]:
    """Every checkpoint a run keeps — the multiples of `save_perm_every` up to `max_steps`, and
    `max_steps` itself — FINAL FIRST, so the headline result exists before the learning curve."""
    tcfg = cfg.get("train") or {}
    final = int(tcfg["max_steps"])
    every = int(tcfg.get("save_perm_every", 0) or 0)
    kept = list(range(every, final, every)) if every > 0 else []
    return [final] + [s for s in kept if s != final]
