"""Score baselines and our checkpoints on the out-of-domain suites.

Kept separate from `runner.py` because the two answer different questions and must not
be averaged together. `runner.py` asks *"is our model better at credit?"*;
this asks *"did we break everything else?"*. Mixing them into one table invites exactly
the mistake of quoting a single mean across both.

The metrics here are deliberately **plain** — ROC-AUC for classification, R² for
regression — not the credit-specific ones. Boundary-mass calibration is meaningless on a
wine-quality dataset, and reporting it would be noise dressed as rigour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.eval.baselines import build
from src.eval.ood import OODDataset, list_ood_datasets, load_ood_dataset, ood_status
from src.utils.logging_setup import get_logger, log_section


@dataclass
class OODEvalConfig:
    models: list[str] = field(default_factory=lambda: ["linear", "catboost"])
    seeds: list[int] = field(default_factory=lambda: [0])
    test_size: float = 0.2
    kinds: list[str] = field(default_factory=lambda: ["classification", "regression"])
    max_rows: int = 10_000
    #: CC18 contains multiclass tasks, but every baseline here is binary (they were
    #: built for PD). True = score them one-vs-rest on the majority class and record
    #: that in the row; False = fail loudly rather than mis-score.
    binarise_multiclass: bool = True
    model_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: OUR checkpoint scores only its own task's OOD kind. `crediticl` wraps ONE checkpoint:
    #: an LGD (regression, 999-quantile) net has no class head and cannot score a
    #: classification dataset, and vice versa. On 25-08 this was unset and every OOD
    #: classification cell for an LGD checkpoint crashed with a (N,999)-vs-(N,2) broadcast
    #: error. lgd -> regression, pd -> classification.
    crediticl_task: str | None = None


def _split(X: np.ndarray, y: np.ndarray, seed: int, test_size: float, stratify: bool):
    from sklearn.model_selection import train_test_split

    strat = y if (stratify and len(np.unique(y)) > 1) else None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    except ValueError:
        # A class with a single member makes stratification impossible. Fall back rather
        # than dropping the dataset — an unstratified split is still a valid measurement.
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def evaluate_one_ood(
    entry: OODDataset, model_name: str, seed: int, cfg: OODEvalConfig
) -> dict[str, Any]:
    """One (dataset, model, seed) cell. Never raises — failures become rows."""
    log = get_logger()
    row: dict[str, Any] = {
        "suite": "OpenML-CC18" if entry.kind == "classification" else "OpenML-CTR23",
        "kind": entry.kind,
        "dataset": entry.name,
        "openml_id": entry.openml_id,
        "model": model_name,
        "seed": seed,
        "status": "ok",
    }
    try:
        X, y, cat_indices = load_ood_dataset(entry)
        # NaNs: the tree and linear baselines differ in tolerance, so impute here once
        # rather than per baseline, keeping the comparison on identical inputs.
        if not np.isfinite(X).all():
            col_median = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            col_median = np.nan_to_num(col_median)
            X = np.where(np.isfinite(X), X, col_median)

        if len(X) > cfg.max_rows:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(X), size=cfg.max_rows, replace=False)
            X, y = X[keep], y[keep]
            row["row_cap"] = cfg.max_rows

        is_clf = entry.kind == "classification"
        if is_clf:
            n_classes = int(len(np.unique(y)))
            if n_classes > 2:
                if not cfg.binarise_multiclass:
                    raise ValueError(
                        f"{entry.name} has {n_classes} classes and the baselines are "
                        f"binary-only (they exist for PD). Set binarise_multiclass=True "
                        f"to score it one-vs-rest on the majority class, or drop it."
                    )
                # One-vs-rest on the MAJORITY class. Stated in the row, because a
                # binarised multiclass AUC is not comparable to a native binary one and
                # must not be pooled with it silently.
                majority = int(np.bincount(y.astype(int)).argmax())
                y = (y == majority).astype(np.int64)
                row["binarised_from_n_classes"] = n_classes
        Xtr, Xte, ytr, yte = _split(X, y, seed, cfg.test_size, stratify=is_clf)
        row.update({"n_train": len(Xtr), "n_test": len(Xte), "n_features": X.shape[1]})

        if not is_clf:
            # THE REGRESSION BASELINES CLIP TO [0,1], because LGD is a loss fraction
            # (see LinearBaseline._predict). Out-of-domain targets have arbitrary
            # scales, so feeding them raw makes every model look terrible for a reason
            # that has nothing to do with generality — a standard-normal target scored
            # R^2 = 0.34 when the true relationship was perfectly linear.
            #
            # So min-max the target into [0,1]. R^2 is invariant to an affine transform
            # of y applied consistently to train and test, so this changes nothing about
            # what is being measured. Statistics come from TRAIN ONLY: using the test
            # range would leak the test distribution into the fit.
            lo = float(np.min(ytr))
            hi = float(np.max(ytr))
            span = hi - lo
            if span < 1e-12:
                raise ValueError("training target is constant after splitting")
            ytr = ((ytr - lo) / span).astype(np.float32)
            # Test values outside the training range land outside [0,1] and take a small
            # clipping penalty. That is honest — the model genuinely never saw that range.
            yte = ((yte - lo) / span).astype(np.float32)
            row["target_scaled_from"] = [round(lo, 6), round(hi, 6)]

        task = "pd" if is_clf else "lgd"
        model = build(model_name, task, seed=seed, **cfg.model_kwargs.get(model_name, {}))
        model.fit(Xtr, ytr, cat_indices=cat_indices)

        if is_clf:
            from sklearn.metrics import accuracy_score, roc_auc_score

            # `Baseline.predict` returns the POSITIVE-CLASS PROBABILITY for task="pd"
            # (see LinearBaseline._predict). There is no predict_proba on the interface,
            # and the baselines are all built for binary credit targets.
            p1 = np.ravel(model.predict(Xte))
            row["roc_auc"] = float(roc_auc_score(yte, p1))
            row["accuracy"] = float(accuracy_score(yte, (p1 >= 0.5).astype(int)))
            row["n_classes"] = 2
        else:
            from sklearn.metrics import mean_squared_error, r2_score

            pred = np.ravel(model.predict(Xte))
            row["r2"] = float(r2_score(yte, pred))
            row["rmse"] = float(np.sqrt(mean_squared_error(yte, pred)))

    except Exception as exc:  # noqa: BLE001 — one cell must not kill the sweep
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("[ood] FAILED %s / %s: %s", entry.name, model_name, row["error"])
    return row


def run_ood(cfg: OODEvalConfig) -> pd.DataFrame:
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
    log.info("[ood] %d datasets x %d models x %d seeds = %d cells",
             len(entries), len(cfg.models), len(cfg.seeds),
             len(entries) * len(cfg.models) * len(cfg.seeds))

    rows = []
    for entry in entries:
        log.info("[ood] --- %s (%s, %d x %d) ---",
                 entry.name, entry.kind, entry.n_rows, entry.n_features)
        for seed in cfg.seeds:
            for model_name in cfg.models:
                # OUR checkpoint cannot cross task types; skip rather than crash.
                if model_name == "crediticl" and cfg.crediticl_task:
                    wanted = "regression" if cfg.crediticl_task == "lgd" else "classification"
                    if entry.kind != wanted:
                        rows.append({
                            "dataset": entry.name, "kind": entry.kind, "model": model_name,
                            "seed": seed, "status": "skipped",
                            "error": f"{cfg.crediticl_task} checkpoint cannot score a "
                                     f"{entry.kind} task",
                        })
                        continue
                rows.append(evaluate_one_ood(entry, model_name, seed, cfg))

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
    ok = df[df["status"] == "ok"]
    out = []
    for kind, metric in (("classification", "roc_auc"), ("regression", "r2")):
        part = ok[ok["kind"] == kind]
        if part.empty or metric not in part:
            continue
        g = part.groupby("model")[metric].agg(["mean", "std", "count"]).reset_index()
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
        metric = part["metric"].iloc[0]
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
