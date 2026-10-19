"""Score baselines and our checkpoints on the out-of-domain suites.

Kept separate from `runner.py` because the two answer different questions and must not
be averaged together. `runner.py` asks *"is our model better at credit?"*;
this asks *"did we break everything else?"*. Mixing them into one table invites exactly
the mistake of quoting a single mean across both.

Same protocol as the credit datasets (`src/eval/protocol.py`): 5-fold cross-validation, the
whole training pool as context, and every metric — for a binary task the PD set (ROC-AUC, Brier,
log-loss, ECE, F1 at the threshold tuned on a validation split, ...), for native multiclass
tasks macro-OVR AUC, log-loss, accuracy, macro F1 and multiclass Brier, and for regression R²,
RMSE, MAE, rank correlation and, where the model gives one, CRPS and interval coverage. The
boundary-mass metrics are dropped: a mass at the edge of a min-max-scaled wine rating means
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

import numpy as np
import pandas as pd

from src.eval.ood import OODDataset, list_ood_datasets, load_ood_dataset, ood_status
from src.eval.protocol import N_FOLDS, PROTOCOL, VALIDATION_FRACTION, cv_folds
from src.eval.runner import _release_memory, prepare_fold, score_fold
from src.utils.logging_setup import get_logger, log_section


@dataclass
class OODEvalConfig:
    models: list[str] = field(default_factory=lambda: ["linear", "catboost"])
    #: Fold-assignment seeds; one full CV per seed. The protocol is one: `[0]`.
    seeds: list[int] = field(default_factory=lambda: [0])
    n_folds: int = N_FOLDS
    val_fraction: float = VALIDATION_FRACTION
    kinds: list[str] = field(default_factory=lambda: ["classification", "regression"])
    datasets: list[str] | None = None  # explicit subset for diagnostic probes; never a row cap
    #: Row cap per dataset. `None` — the protocol: every row of the full-table cache.
    max_rows: int | None = None
    max_context_rows: int | None = None
    model_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: OUR checkpoint scores only its own task's OOD kind. `crediticl` wraps ONE checkpoint:
    #: an LGD (regression, 999-quantile) net has no class head and cannot score a
    #: classification dataset, and vice versa. On 25-08 this was unset and every OOD
    #: classification cell for an LGD checkpoint crashed with a (N,999)-vs-(N,2) broadcast
    #: error. lgd -> regression, pd -> classification.
    crediticl_task: str | None = None


def _prepare(entry: OODDataset, seed: int, cfg: OODEvalConfig) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    """Load one OOD dataset, cap it only if asked, encode native classification labels."""
    X, y, cat_indices = load_ood_dataset(entry)
    info: dict[str, Any] = {}
    if cfg.max_rows is not None and len(X) > cfg.max_rows:
        keep = np.random.default_rng(seed).choice(len(X), size=cfg.max_rows, replace=False)
        X, y = X[keep], y[keep]
        info["row_cap"] = cfg.max_rows
    if entry.kind == "classification":
        # Encode every original class, never replace the native task by one-vs-rest.
        _, y = np.unique(y, return_inverse=True)
        info["n_classes"] = int(len(np.unique(y)))
        info["classification_task"] = "binary" if info["n_classes"] == 2 else "multiclass"
    return X, y, cat_indices, info


def _fold_arrays(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                 regression: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Preserve model-specific missing-value handling and rescale the regression target, with TRAINING-fold statistics only."""
    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    info: dict[str, Any] = {}
    if regression:
        # Retain training-fold affine normalisation for comparable error units.
        # OOD wrappers receive bounded_target=False: a test value beyond the
        # training range is allowed, and its prediction must not be clipped as LGD.
        lo, hi = float(np.min(ytr)), float(np.max(ytr))
        if hi - lo < 1e-12:
            raise ValueError("training target is constant after splitting")
        # Test values outside the training range remain outside [0,1].
        ytr = ((ytr - lo) / (hi - lo)).astype(np.float32)
        yte = ((yte - lo) / (hi - lo)).astype(np.float32)
        info["target_scaled_from"] = [round(lo, 6), round(hi, 6)]
    return Xtr, Xte, ytr, yte, info


def evaluate_ood_dataset(entry: OODDataset, models: list[str], seed: int, cfg: OODEvalConfig,
                         *, journal=None) -> list[dict[str, Any]]:
    """Every model on every fold of one OOD dataset. Never raises — failures become rows."""
    log = get_logger()
    is_clf = entry.kind == "classification"
    task = "pd" if is_clf else "lgd"
    base = {"suite": entry.source_suite or "pinned OpenML benchmark", "kind": entry.kind,
            "dataset": entry.name, "openml_id": entry.openml_id, "seed": seed,
            "protocol": PROTOCOL, "n_folds": cfg.n_folds, "context_cap": cfg.max_context_rows}
    cached = {(fold, model): journal.get({**base, "fold": fold, "model": model})
              for fold in range(cfg.n_folds) for model in models} if journal else {}
    if cached and all(cached.values()):
        log.info("[ood] REUSE %s: all %d cells already saved", entry.name, len(cached))
        return list(cached.values())
    try:
        X, y, cat_indices, info = _prepare(entry, seed, cfg)
        folds = cv_folds(y, task, n_folds=cfg.n_folds, seed=seed)
    except Exception as exc:  # noqa: BLE001 — one dataset must not kill the sweep
        log.warning("[ood] FAILED to load %s: %s", entry.name, exc)
        return [{**base, "model": m, "fold": f, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                for m in models for f in range(cfg.n_folds)]
    base.update(info, n_features=int(X.shape[1]))
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        if all(cached.get((fold, model)) for model in models):
            rows.extend(cached[(fold, model)] for model in models)
            continue
        # The same fold preprocessing as the credit benchmark, once per fold for every model.
        try:
            Xtr, Xte, ytr, yte, fold_info = _fold_arrays(X, y, train_idx, test_idx, regression=not is_clf)
            stages = prepare_fold(task, Xtr, ytr, Xte, cat_indices, seed=seed, fold=fold,
                                  val_fraction=cfg.val_fraction)
            prep_error = ""
        except Exception as exc:  # noqa: BLE001 — recorded per model below
            stages, prep_error = None, f"{type(exc).__name__}: {exc}"
        for model_name in models:
            row = {**base, "model": model_name, "fold": fold, "status": "failed"}
            if cached.get((fold, model_name)):
                rows.append(cached[(fold, model_name)])
                continue
            # OUR checkpoint cannot cross task types; skip rather than crash.
            if model_name == "crediticl" and cfg.crediticl_task and cfg.crediticl_task != task:
                row.update(status="skipped",
                           error=f"{cfg.crediticl_task} checkpoint cannot score a {entry.kind} task")
                rows.append(row)
                continue
            if journal:
                journal.begin(row)
            started = time.monotonic()
            try:
                if stages is None:
                    raise RuntimeError(f"fold preparation failed: {prep_error}")
                row.update(fold_info)
                result = score_fold(task, model_name, Xtr, ytr, Xte, yte, cat_indices, seed=seed, fold=fold,
                                    val_fraction=cfg.val_fraction,
                                    model_kwargs=cfg.model_kwargs.get(model_name, {}),
                                    max_context_rows=cfg.max_context_rows, bounded_target=False,
                                    stages=stages)
                result.pop("_headline", None)
                row.update(result)
                row["status"] = "ok"
            except Exception as exc:  # noqa: BLE001 — one cell must not kill the sweep
                row["error"] = f"{type(exc).__name__}: {exc}"
                log.warning("[ood] FAILED %s / %s fold %d: %s", entry.name, model_name, fold, row["error"])
            finally:
                _release_memory()
            row["total_seconds"] = round(time.monotonic() - started, 2)
            rows.append(row)
            if journal:
                journal.save(row)
    return rows


def run_ood(cfg: OODEvalConfig, *, journal=None) -> pd.DataFrame:
    """Score every model on every cached OOD dataset."""
    log = get_logger()
    log_section(log, "OUT-OF-DOMAIN EVAL")

    status = ood_status()
    log.info("[ood] cache: %s", status["manifest"])
    log.info("[ood] %d datasets cached %s", status["n_datasets"], status["by_kind"])
    if not status["exists"]:
        raise FileNotFoundError(
            "no out-of-domain cache. Fetch it on a LOGIN node first (compute nodes have "
            "no internet):\n    python -m src.utils.fetch_ood"
        )
    if not status["complete"]:
        log.warning(
            "[ood] fewer than the target datasets per kind — usable, but a thin "
            "out-of-domain set is weak evidence in either direction. Say so when reporting."
        )

    entries = [d for d in list_ood_datasets() if d.kind in cfg.kinds]
    if cfg.datasets is not None:
        if not cfg.datasets or not set(cfg.datasets) <= {d.name for d in entries}:
            raise ValueError("Requested OOD diagnostic datasets are missing or empty")
        entries = [d for d in entries if d.name in cfg.datasets]
    log.info("[ood] %d datasets x %d models x %d folds x %d seeds = %d cells",
             len(entries), len(cfg.models), cfg.n_folds, len(cfg.seeds),
             len(entries) * len(cfg.models) * cfg.n_folds * len(cfg.seeds))

    rows: list[dict[str, Any]] = []
    for entry in entries:
        log.info("[ood] --- %s (%s, %d x %d) ---",
                 entry.name, entry.kind, entry.n_rows, entry.n_features)
        for seed in cfg.seeds:
            rows.extend(evaluate_ood_dataset(entry, cfg.models, seed, cfg,
                                            **({"journal": journal} if journal else {})))

    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == "ok").sum()) if not df.empty else 0
    n_skip = int((df["status"] == "skipped").sum()) if not df.empty else 0
    log.info("[ood] finished: %d ok, %d skipped (task type the checkpoint cannot score), "
             "%d failed (of %d cells)", n_ok, n_skip, len(df) - n_ok - n_skip, len(df))
    return df


def summarise_ood(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metric per (model, kind).

    Classification and regression are summarised **separately**, never pooled: a mean of
    ROC-AUC and R² is not a number.
    """
    if df.empty:
        return df
    from src.eval.comparisons import require_balanced_cells

    for _, cells in df.groupby("kind"):
        if "fold" in cells:
            require_balanced_cells(cells[cells.status.ne("skipped")])
    ok = df[df["status"] == "ok"]
    out = []
    # The headline metric of each kind first (the text summary reads the first row per kind).
    per_kind = {"classification": ("roc_auc", "f1_tuned", "brier", "log_loss"),
                "regression": ("r2", "rmse", "crps")}
    for kind, metrics in per_kind.items():
        part = ok[ok["kind"] == kind]
        for metric in metrics:
            if part.empty or metric not in part:
                continue
            # Folds within a dataset first, so every dataset counts once.
            per_dataset = part.groupby(["model", "dataset"])[metric].mean()
            g = per_dataset.groupby(level="model").agg(["mean", "std", "count"]).reset_index()
            g.insert(0, "kind", kind)
            g.insert(1, "metric", metric)
            out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def ood_text_summary(df: pd.DataFrame, reference_model: str | None = None) -> str:
    """Plain-text summary — the question is 'did we break anything?'.

    Reports each model's out-of-domain mean, and if a reference is given, the *delta*
    against it, because the absolute AUC on CC18 is not interesting on its own. The
    delta against the unmodified-prior checkpoint is the whole point.
    """
    lines = ["=" * 78, "OUT-OF-DOMAIN SUMMARY (non-credit tasks)", "=" * 78]
    summary = summarise_ood(df)
    if summary.empty:
        lines.append("no successful cells — check the log")
        return "\n".join(lines)

    for kind in summary["kind"].unique():
        part = summary[summary["kind"] == kind]
        metric = part["metric"].iloc[0]  # the headline metric of this kind
        part = part[part["metric"] == metric]
        lines.append(f"\n{kind.upper()}  (metric: {metric}, higher is better)")
        ref_val = None
        if reference_model is not None:
            match = part[part["model"] == reference_model]
            ref_val = float(match["mean"].iloc[0]) if len(match) else None
        for _, r in part.sort_values("mean", ascending=False).iterrows():
            delta = "" if ref_val is None else f"   delta vs {reference_model}: {r['mean'] - ref_val:+.4f}"
            lines.append(
                f"  {r['model']:<14} {r['mean']:.4f} +/- {float(r['std'] or 0):.4f} "
                f"(n={int(r['count'])}){delta}"
            )

    # A "skipped" cell is a deliberate task-type mismatch — an LGD (regression) checkpoint
    # declining a classification OOD task, or a PD net declining a regression one — NOT a
    # failure. Report the two apart, so a clean run does not read as "75 cells failed".
    skipped = df[df["status"] == "skipped"]
    if len(skipped):
        lines.append(
            f"\n{len(skipped)} cells skipped — the checkpoint's task type cannot score them "
            f"(e.g. a regression net on a classification task). Expected, not a failure."
        )
    failed = df[~df["status"].isin(("ok", "skipped"))]
    if len(failed):
        lines.append(f"\n{len(failed)} cells FAILED:")
        for _, r in failed.head(8).iterrows():
            lines.append(f"  {r['dataset']:<24} {r['model']:<12} {r.get('error', '')[:60]}")

    lines.append(
        "\nHOW TO READ THIS\n"
        "These tasks have nothing to do with credit. If a credit-tailored prior scores\n"
        "about the same here as the unmodified one, the credit gain came for free. If it\n"
        "scores clearly worse, we bought credit performance by giving up generality —\n"
        "still a real finding, but it must be reported as a trade-off, not hidden.\n"
        "Classification and regression are kept apart: averaging ROC-AUC with R^2 would\n"
        "produce a number that means nothing."
    )
    return "\n".join(lines)
