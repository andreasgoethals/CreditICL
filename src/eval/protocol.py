"""The evaluation protocol, in one place: how a dataset is split, how a threshold is chosen,
which checkpoints are scored. Every model — ours, released TabICLv2, TabPFN-3, CatBoost, the
linear floor — goes through exactly this.

THE PROTOCOL (version 3, 28-09-2026)

* **5-fold cross-validation**, stratified on the default flag for PD, shuffled with a fixed
  seed, so every model sees the same folds. Each row is scored exactly once, by a model whose
  context (or training set) did not contain it.
* **The whole training pool is the context.** No row cap, no context selection, no
  subsampling: the four training folds go to the model as they are.
* **At most `MAX_FEATURES` = 500 columns, the SAME ones for every model** (29-09-2026): a table
  wider than that (`0011.loan_default`, 759; `0014.algorithmwatch`, 2,986) keeps its 500
  highest-variance columns, chosen on the training fold alone, for the foundation models, CatBoost
  and the linear floor alike. The foundation models accept no more; giving the others more would
  compare inputs, not models. Variance, not a target-aware score, so the choice cannot leak.
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

from typing import Any

import numpy as np

#: Bumped whenever the protocol changes, so a completion receipt written under an older one is
#: never mistaken for a current result (`benchmark_status`).
PROTOCOL = 3
N_FOLDS = 5
#: Share of the training pool held out to choose the PD threshold.
VALIDATION_FRACTION = 0.2
#: Column cap for every model (see the protocol above).
MAX_FEATURES = 500
#: Quantile levels requested from models with a predictive distribution: 99, fine enough that
#: `2 x mean pinball` approximates CRPS closely.
QUANTILE_LEVELS = np.round(np.linspace(0.01, 0.99, 99), 2)


def select_columns(X_train: np.ndarray, max_features: int = MAX_FEATURES) -> np.ndarray | None:
    """The `max_features` highest-variance columns of the TRAINING rows, in their original order,
    or None when the table is already narrow enough. Missing values are ignored; a column with no
    observed variance ranks last. Label-free, so it cannot leak the target into the choice."""
    X_train = np.asarray(X_train, dtype=float)
    if X_train.shape[1] <= max_features:
        return None
    import warnings

    with warnings.catch_warnings():  # an all-missing column warns; it ranks last below anyway
        warnings.simplefilter("ignore", RuntimeWarning)
        var = np.nanvar(np.where(np.isfinite(X_train), X_train, np.nan), axis=0)
    var = np.where(np.isfinite(var), var, -np.inf)
    # stable: equal variances keep their column order, so the choice is deterministic
    order = np.argsort(-var, kind="stable")
    return np.sort(order[:max_features])


def apply_columns(keep: np.ndarray | None, cat_indices: list[int], *arrays: np.ndarray):
    """`arrays` restricted to `keep`, and the categorical indices renumbered to match."""
    if keep is None:
        return (*arrays, list(cat_indices))
    position = {int(old): new for new, old in enumerate(keep.tolist())}
    cats = [position[c] for c in cat_indices if c in position]
    return (*(a[:, keep] for a in arrays), cats)


def cv_folds(y: np.ndarray, task: str, *, n_folds: int = N_FOLDS, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """`(train_idx, test_idx)` for each fold. Stratified on the class for PD (so a 7 %-default
    table keeps defaults in every fold), plain shuffled folds for a continuous target."""
    from sklearn.model_selection import KFold, StratifiedKFold

    y = np.asarray(y)
    index = np.arange(len(y))
    if task == "pd":
        labels = (y >= 0.5).astype(int)
        if np.bincount(labels, minlength=2).min() >= n_folds:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return [(tr, te) for tr, te in splitter.split(index, labels)]
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(index)]


def validation_split(y: np.ndarray, task: str, *, fraction: float = VALIDATION_FRACTION,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`(fit_idx, val_idx)` inside a training pool, stratified for PD."""
    from sklearn.model_selection import train_test_split

    index = np.arange(len(y))
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
