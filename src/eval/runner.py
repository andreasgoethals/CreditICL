"""PIPELINE 4 of 4 — score every model on every dataset, by the protocol in `protocol.py`.

Loop: for each dataset, for each of the 5 cross-validation folds, for each model — fit on the
training folds, score the held-out fold, append one row. Nothing clever; the value is in being
complete and in logging enough that a failure on the cluster is diagnosable from the log alone.

DESIGN CHOICES THAT MATTER

**It preprocesses on demand.** If a dataset is not in the processed cache, it is
preprocessed first (`ensure_processed`). So a fresh clone with raw data present
needs one command, not two.

**One failure never kills the run.** A model that OOMs or a dataset with a broken
recipe produces a row with `status="failed"` and an error message, and the loop
continues. Twenty results plus one explained failure beats zero results.

**Every row records the conditions, not just the score.** Rows carry the fold, the rows
actually used, the device, fit and predict seconds, the tuned threshold and the decoding rule.
A number without those is not reproducible, and "model X is worse" is usually "model X was
quietly given less data".

**Splits are random and that is a known weakness.** Purucker 2026 shows TFM rankings change
once splits become temporal or grouped, and credit data is exactly where that bites.
`split="temporal"` needs a date column per dataset, which we do not yet have — see
docs/EXPERIMENTAL_DESIGN.md §5.4.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.discovery import list_datasets
from src.data.pipeline import ensure_processed, load_processed
from src.eval.baselines import DEFAULT_BASELINES, availability_report, build
from src.eval.metrics import lgd_metrics, pd_metrics, threshold_metrics
from src.eval.protocol import (N_FOLDS, PROTOCOL, VALIDATION_FRACTION, apply_columns, cv_folds,
                               f1_optimal_threshold, select_columns, validation_split)
from src.utils.logging_setup import get_logger, log_section

#: Distribution metrics that describe a BOUNDED loss target. Dropped for out-of-domain
#: regression, where a mass at the edge of a min-max-scaled wine rating means nothing.
BOUNDARY_KEYS = ("true_mass_", "pred_mass_", "boundary_mass_", "mae_boundary", "n_boundary",
                 "mae_interior", "n_interior")


@dataclass
class EvalConfig:
    task: str
    datasets: list[str] | None = None
    models: list[str] = field(default_factory=lambda: list(DEFAULT_BASELINES))
    n_folds: int = N_FOLDS
    #: Seeds of the fold assignment; one 5-fold CV per seed. The protocol is one: `[0]`.
    seeds: list[int] = field(default_factory=lambda: [0])
    val_fraction: float = VALIDATION_FRACTION
    split: str = "random"  # "random" | "temporal" (temporal needs a date column)
    model_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Cap rows per dataset before splitting. `None` = use everything, which is what a REAL
    #: result must do — this exists for debug runs on small-memory partitions.
    #:
    #: `0014.algorithmwatch` is 158,700 x 2,986 float32 = **1.8 GB**, and the 14 PD datasets
    #: total 2.4 GB. Loading one, splitting it, and imputing it makes three more copies, which
    #: is how four arms hit `OUT_OF_MEMORY` on a 30 GB partition on 14-08-2026 — and only once
    #: preprocessing had succeeded, because before that the evaluation had nothing to load.
    #: The subsample is SEEDED AND RANDOM, never the head: taking the head of a sorted file
    #: misread one dataset's base rate as 49.5% against a true 37.8%.
    max_rows: int | None = None
    #: Cap on CONTEXT rows given to a TFM. `None` — the protocol: the whole training pool.
    max_context_rows: int | None = None


def check_split(split: str) -> None:
    if split == "temporal":
        # Deliberately not silently falling back to random: that would report a temporal
        # result that was not temporal.
        raise NotImplementedError(
            "temporal splits need a per-dataset date column, which the registry does "
            "not carry yet. See docs/EXPERIMENTAL_DESIGN.md §5.4."
        )
    if split != "random":
        raise ValueError(f"unknown split {split!r}")


def _release_memory() -> None:
    """Free the fold's model before the next one: a 400k-row context leaves gigabytes behind."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _fit_predict(model_name: str, task: str, seed: int, X_fit: np.ndarray, y_fit: np.ndarray,
                 X_pred: np.ndarray, cat_indices: list[int], model_kwargs: dict[str, Any] | None,
                 max_context_rows: int | None, *, quantiles: bool = False):
    """Build, fit on `(X_fit, y_fit)`, predict `X_pred`. Returns `(pred, quantiles, report)`."""
    model = build(model_name, task, seed=seed, **(model_kwargs or {}))
    # SET ON THE INSTANCE, not per-model: every TFM baseline in this run gets the same
    # cap (none, in the protocol), so the comparison stays about the weights.
    if max_context_rows is not None and hasattr(model, "max_context_rows"):
        model.max_context_rows = int(max_context_rows)
    report = model.fit(X_fit, y_fit, cat_indices)
    pred = np.asarray(model.predict(X_pred), dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite predictions")
    q = model.predict_quantiles(X_pred) if quantiles else None
    if q is not None and not np.isfinite(q[0]).all():
        raise ValueError("Non-finite predicted quantiles")
    del model
    return pred, q, report


def score_fold(task: str, model_name: str, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
               y_te: np.ndarray, cat_indices: list[int], *, seed: int = 0, fold: int = 0,
               val_fraction: float = VALIDATION_FRACTION, model_kwargs: dict[str, Any] | None = None,
               max_context_rows: int | None = None, bounded_target: bool = True) -> dict[str, Any]:
    """One model on one fold: every metric, as a dict. Raises on failure (the caller records it).

    PD: choose the threshold that maximises F1 on a validation split of the training pool
    (model fitted on the rest), then refit on the WHOLE pool and score the test fold at it.
    LGD: fit on the whole pool, score the point prediction and, where the model gives one, the
    predictive distribution."""
    out: dict[str, Any] = {"n_train": int(len(y_tr)), "n_test": int(len(y_te))}
    if task == "pd":
        fit_idx, val_idx = validation_split(y_tr, "pd", fraction=val_fraction, seed=1000 * seed + fold)
        p_val, _, _ = _fit_predict(model_name, task, seed, X_tr[fit_idx], y_tr[fit_idx], X_tr[val_idx],
                                   cat_indices, model_kwargs, max_context_rows)
        threshold, val_f1 = f1_optimal_threshold(y_tr[val_idx], p_val)
        _release_memory()
        p, _, report = _fit_predict(model_name, task, seed, X_tr, y_tr, X_te, cat_indices,
                                    model_kwargs, max_context_rows)
        out.update(pd_metrics(y_te, p))
        for key, value in threshold_metrics(y_te, p, threshold).items():
            out[f"{key}_tuned"] = value
        out.update({"n_val": int(len(val_idx)), "val_f1": val_f1})
        headline = (f"auc={out.get('roc_auc', float('nan')):.4f} f1={out['f1_tuned']:.4f} "
                    f"(thr {threshold:.3f}) brier={out['brier']:.4f}")
    else:
        pred, q, report = _fit_predict(model_name, task, seed, X_tr, y_tr, X_te, cat_indices,
                                       model_kwargs, max_context_rows, quantiles=True)
        if q is not None:
            m = lgd_metrics(y_te, pred, quantiles=q[0], levels=q[1], decoding="mean")
        else:
            m = lgd_metrics(y_te, pred, decoding="point")
        if not bounded_target:
            m = {k: v for k, v in m.items() if not k.startswith(BOUNDARY_KEYS)}
        out.update(m)
        headline = f"r2={out['r2']:.4f} rmse={out['rmse']:.4f} crps={out.get('crps', float('nan')):.4f}"
    if not np.isfinite(out["roc_auc" if task == "pd" else "r2"]):
        raise ValueError("Non-finite headline metric")
    out.update({
        "n_train_used": report.n_train_used,
        "subsampled": report.subsampled,
        "fit_seconds": report.fit_seconds,
        "predict_seconds": report.predict_seconds,
        **{f"info_{k}": v for k, v in report.extra.items()},
    })
    out["_headline"] = headline
    return out


def load_arrays(task: str, dataset: str, *, max_rows: int | None = None, seed: int = 0):
    """`(X, y, cat_indices, info)` for one processed dataset, capped only if asked."""
    log = get_logger()
    ds = load_processed(task, dataset)
    X, y = ds.X, ds.y
    info: dict[str, Any] = {"n_rows": ds.n_rows, "n_features": ds.n_features,
                            "n_categorical": len(ds.cat_indices)}
    # CAP BEFORE SPLITTING, so no copy of the full array is ever made. Seeded and random,
    # never the head — see `EvalConfig.max_rows`. `row_cap` goes into the results so a
    # capped number is never mistaken for a full-data one.
    if max_rows is not None and len(X) > max_rows:
        keep = np.random.default_rng(seed).choice(len(X), size=max_rows, replace=False)
        X, y = X[keep], y[keep]
        info["row_cap"] = max_rows
        info["n_rows_full"] = int(ds.n_rows)
        log.info("[eval] %s capped %d -> %d rows (max_rows)", dataset, ds.n_rows, max_rows)
    return X, y, list(ds.cat_indices), info


def evaluate_dataset(task: str, dataset: str, models: list[str], seed: int, *, n_folds: int = N_FOLDS,
                     val_fraction: float = VALIDATION_FRACTION, split: str = "random",
                     model_kwargs: dict[str, dict[str, Any]] | None = None,
                     max_rows: int | None = None, max_context_rows: int | None = None) -> list[dict[str, Any]]:
    """Every model on every fold of one dataset: `n_folds x len(models)` rows."""
    log = get_logger()
    check_split(split)
    base = {"task": task, "dataset": dataset, "seed": seed, "split": split, "protocol": PROTOCOL,
            "n_folds": n_folds, "context_cap": max_context_rows}
    try:
        X, y, cat_indices, info = load_arrays(task, dataset, max_rows=max_rows, seed=seed)
        # The folds are fixed by (dataset, seed) alone, so every model sees the same ones.
        folds = cv_folds(y, task, n_folds=n_folds, seed=seed)
    except Exception as exc:  # noqa: BLE001 — a broken dataset costs its own rows only
        log.error("[eval] %s/%s could not be loaded: %s", task, dataset, exc)
        return [{**base, "model": m, "fold": f, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                for m in models for f in range(n_folds)]
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        # THE SAME COLUMNS FOR EVERY MODEL: a table wider than `MAX_FEATURES` keeps its
        # highest-variance columns of this training fold, for all models alike.
        keep = select_columns(X[train_idx])
        X_tr, X_te, cats = apply_columns(keep, cat_indices, X[train_idx], X[test_idx])
        fold_info = ({} if keep is None else
                     {"features_selected_from": int(X.shape[1]), "features_used": int(len(keep)),
                      "feature_selection": "top-variance, training fold"})
        for model_name in models:
            row = {**base, **info, **fold_info, "model": model_name, "fold": fold, "status": "failed"}
            started = time.time()
            try:
                result = score_fold(task, model_name, X_tr, y[train_idx], X_te, y[test_idx],
                                    cats, seed=seed, fold=fold, val_fraction=val_fraction,
                                    model_kwargs=(model_kwargs or {}).get(model_name, {}),
                                    max_context_rows=max_context_rows)
                headline = result.pop("_headline")
                row.update(result)
                row["status"] = "ok"
                log.info("[eval]     fold %d  %-10s OK   %s", fold, model_name, headline)
            except Exception as exc:  # noqa: BLE001 — one model must not kill the sweep
                row["error"] = f"{type(exc).__name__}: {exc}"
                log.error("[eval]     fold %d  %-10s FAILED on %s/%s: %s", fold, model_name, task, dataset, row["error"])
                log.error("[eval]     traceback:\n%s", traceback.format_exc())
            finally:
                _release_memory()
            row["total_seconds"] = round(time.time() - started, 2)
            rows.append(row)
    return rows


def run(cfg: EvalConfig) -> pd.DataFrame:
    """Run the whole sweep and return one row per (dataset, model, seed, fold)."""
    log = get_logger()
    log_section(log, f"EVAL — {cfg.task.upper()}  (protocol {PROTOCOL}: {cfg.n_folds}-fold CV, whole training pool as context)")

    avail = availability_report()
    for name, (ok, err) in avail.items():
        log.info("[eval] baseline %-10s %s", name, "available" if ok else f"UNAVAILABLE — {err}")
    models = [m for m in cfg.models if avail.get(m, (False, "unknown"))[0]]
    skipped = [m for m in cfg.models if m not in models]
    if skipped:
        log.warning("[eval] skipping unavailable baselines: %s", ", ".join(skipped))
    if not models:
        raise RuntimeError("no baselines available — check the install")

    datasets = cfg.datasets if cfg.datasets is not None else list_datasets(cfg.task)
    log.info("[eval] %d datasets x %d models x %d folds x %d seeds = %d cells",
             len(datasets), len(models), cfg.n_folds, len(cfg.seeds),
             len(datasets) * len(models) * cfg.n_folds * len(cfg.seeds))

    # Preprocess on demand, so one command is enough from a fresh clone.
    ready = ensure_processed(cfg.task, datasets)
    usable = [d for d in datasets if ready.get(d) is not None]
    if len(usable) < len(datasets):
        log.warning(
            "[eval] %d datasets unusable and excluded: %s",
            len(datasets) - len(usable),
            ", ".join(d for d in datasets if d not in usable),
        )

    rows: list[dict[str, Any]] = []
    for dataset in usable:
        log.info("[eval] --- %s/%s ---", cfg.task, dataset)
        for seed in cfg.seeds:
            rows.extend(evaluate_dataset(
                cfg.task, dataset, models, seed, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction,
                split=cfg.split, model_kwargs=cfg.model_kwargs, max_rows=cfg.max_rows,
                max_context_rows=cfg.max_context_rows,
            ))

    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == "ok").sum()) if not df.empty else 0
    log.info("[eval] finished: %d/%d cells OK", n_ok, len(df))
    if n_ok < len(df):
        for _, bad in df[df["status"] != "ok"].iterrows():
            log.warning("[eval] FAILED %s/%s %s fold %s: %s", bad["task"], bad["dataset"], bad["model"],
                        bad.get("fold"), bad.get("error"))
    return df


def summarise(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Mean of the headline metrics per model: over folds within a dataset first, then over
    datasets, so a dataset counts once whatever its fold count."""
    if df.empty:
        return df
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return ok
    metrics = (
        ["roc_auc", "pr_auc", "f1_tuned", "brier", "log_loss", "ece"]
        if task == "pd"
        else ["r2", "rmse", "mae", "crps", "boundary_mass_abs_err"]
    )
    present = [m for m in metrics if m in ok.columns]
    per_dataset = ok.groupby(["model", "dataset"])[present].mean()
    out = per_dataset.groupby(level="model").mean().reset_index()
    out["n_datasets"] = per_dataset.groupby(level="model").size().values
    return out.sort_values(present[0], ascending=(task != "pd"))
