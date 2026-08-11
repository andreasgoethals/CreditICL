"""PIPELINE 4 of 4 — score every model on every dataset.

Loop: for each dataset, for each model, fit on train, predict on test, compute
metrics, append one row. Nothing clever; the value is in being complete and in
logging enough that a failure on the cluster is diagnosable from the log alone.

DESIGN CHOICES THAT MATTER

**It preprocesses on demand.** If a dataset is not in the processed cache, it is
preprocessed first (`ensure_processed`). So a fresh clone with raw data present
needs one command, not two.

**One failure never kills the run.** A model that OOMs or a dataset with a broken
recipe produces a row with `status="failed"` and an error message, and the loop
continues. Twenty results plus one explained failure beats zero results.

**Every row records the conditions, not just the score.** Rows carry the row count
actually used, whether the training set was subsampled, the device, fit and predict
seconds, and the decoding rule. A number without those is not reproducible, and
"model X is worse" is usually "model X was quietly given less data".

**Splits are random by default and that is a known weakness.** Purucker 2026 shows
TFM rankings change once splits become temporal or grouped, and credit data is
exactly where that bites. `split="temporal"` is wired but needs a date column per
dataset, which we do not yet have — see docs/EXPERIMENTAL_DESIGN.md §5.4.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.discovery import list_datasets
from src.data.pipeline import ensure_processed, load_processed
from src.eval.baselines import DEFAULT_BASELINES, availability_report, build
from src.eval.metrics import lgd_metrics, pd_metrics
from src.utils.logging_setup import get_logger, log_section


@dataclass
class EvalConfig:
    task: str
    datasets: list[str] | None = None
    models: list[str] = field(default_factory=lambda: list(DEFAULT_BASELINES))
    test_size: float = 0.2
    split: str = "random"  # "random" | "temporal" (temporal needs a date column)
    seeds: list[int] = field(default_factory=lambda: [0])
    model_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


def make_split(
    n: int,
    *,
    test_size: float,
    seed: int,
    y: np.ndarray | None = None,
    task: str = "pd",
    split: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx)."""
    if split == "temporal":
        # Deliberately not silently falling back to random: that would report a
        # temporal result that was not temporal.
        raise NotImplementedError(
            "temporal splits need a per-dataset date column, which the registry does "
            "not carry yet. See docs/EXPERIMENTAL_DESIGN.md §5.4."
        )
    if split != "random":
        raise ValueError(f"unknown split {split!r}")

    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_size * n)))

    # Stratify classification, so a 7%-positive dataset keeps positives on both
    # sides. Without this, small imbalanced datasets can land a test set with zero
    # positives and ROC-AUC becomes undefined.
    if task == "pd" and y is not None and 0 < y.sum() < n:
        pos = np.flatnonzero(y >= 0.5)
        neg = np.flatnonzero(y < 0.5)
        n_pos_test = max(1, int(round(test_size * len(pos))))
        n_neg_test = max(1, n_test - n_pos_test)
        test_idx = np.concatenate(
            [
                rng.choice(pos, size=min(n_pos_test, len(pos) - 1), replace=False),
                rng.choice(neg, size=min(n_neg_test, len(neg) - 1), replace=False),
            ]
        )
    else:
        test_idx = rng.choice(n, size=n_test, replace=False)

    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    return np.flatnonzero(mask), np.sort(test_idx)


def evaluate_one(
    task: str,
    dataset: str,
    model_name: str,
    seed: int,
    *,
    test_size: float = 0.2,
    split: str = "random",
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one model on one dataset and return one result row."""
    log = get_logger()
    row: dict[str, Any] = {
        "task": task,
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "split": split,
        "status": "failed",
    }
    started = time.time()

    try:
        ds = load_processed(task, dataset)
        X, y = ds.X, ds.y
        row.update(
            {
                "n_rows": ds.n_rows,
                "n_features": ds.n_features,
                "n_categorical": len(ds.cat_indices),
            }
        )

        train_idx, test_idx = make_split(
            X.shape[0], test_size=test_size, seed=seed, y=y, task=task, split=split
        )
        model = build(model_name, task, seed=seed, **(model_kwargs or {}))
        report = model.fit(X[train_idx], y[train_idx], ds.cat_indices)
        preds = model.predict(X[test_idx])
        y_test = y[test_idx]

        if task == "pd":
            row.update(pd_metrics(y_test, preds))
        else:
            q = model.predict_quantiles(X[test_idx])
            if q is not None:
                row.update(lgd_metrics(y_test, preds, quantiles=q[0], levels=q[1], decoding="median"))
            else:
                row.update(lgd_metrics(y_test, preds, decoding="point"))

        row.update(
            {
                "status": "ok",
                "n_train_used": report.n_train_used,
                "n_train_available": report.n_train_available,
                "subsampled": report.subsampled,
                "fit_seconds": report.fit_seconds,
                "predict_seconds": report.predict_seconds,
                **{f"info_{k}": v for k, v in report.extra.items()},
            }
        )
        headline = (
            f"pr_auc={row.get('pr_auc', float('nan')):.4f} roc_auc={row.get('roc_auc', float('nan')):.4f} "
            f"ece={row.get('ece', float('nan')):.4f}"
            if task == "pd"
            else f"r2={row.get('r2', float('nan')):.4f} rmse={row.get('rmse', float('nan')):.4f} "
            f"bnd_err={row.get('boundary_mass_abs_err', float('nan')):.4f}"
        )
        log.info("[eval]     %-10s OK   %s", model_name, headline)

    except Exception as exc:  # noqa: BLE001 — one model must not kill the sweep
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.error("[eval]     %-10s FAILED on %s/%s: %s", model_name, task, dataset, row["error"])
        log.error("[eval]     traceback:\n%s", traceback.format_exc())

    row["total_seconds"] = round(time.time() - started, 2)
    return row


def run(cfg: EvalConfig) -> pd.DataFrame:
    """Run the whole sweep and return one row per (dataset, model, seed)."""
    log = get_logger()
    log_section(log, f"EVAL — {cfg.task.upper()}")

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
    log.info("[eval] %d datasets x %d models x %d seeds = %d runs",
             len(datasets), len(models), len(cfg.seeds), len(datasets) * len(models) * len(cfg.seeds))

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
            for model_name in models:
                rows.append(
                    evaluate_one(
                        cfg.task,
                        dataset,
                        model_name,
                        seed,
                        test_size=cfg.test_size,
                        split=cfg.split,
                        model_kwargs=cfg.model_kwargs.get(model_name, {}),
                    )
                )

    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == "ok").sum()) if not df.empty else 0
    log.info("[eval] finished: %d/%d runs OK", n_ok, len(df))
    if n_ok < len(df):
        for _, bad in df[df["status"] != "ok"].iterrows():
            log.warning("[eval] FAILED %s/%s %s: %s", bad["task"], bad["dataset"], bad["model"], bad.get("error"))
    return df


def summarise(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Mean of the headline metrics per model, across datasets."""
    if df.empty:
        return df
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return ok
    metrics = (
        ["pr_auc", "roc_auc", "brier", "ece", "log_loss"]
        if task == "pd"
        else ["r2", "rmse", "mae", "boundary_mass_abs_err"]
    )
    present = [m for m in metrics if m in ok.columns]
    out = ok.groupby("model")[present].mean().reset_index()
    out["n_datasets"] = ok.groupby("model")["dataset"].nunique().values
    return out.sort_values(present[0], ascending=(task != "pd"))
