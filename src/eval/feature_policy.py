"""Prediction-time exclusions for credit evaluation, shared by every model.

PD is prospective (application); LGD is predicted when default starts. The raw
and processed source tables are preserved. This policy excludes known outcome
components and conservatively excludes workout fields without a verified
default-time snapshot. It is NOT certification of the remaining source fields:
anonymised columns and observation dates still need their source dictionaries.
"""
from __future__ import annotations

import json

import numpy as np

from src.data.dataset_registry import canonical_slug, is_proprietary
from src.data.private_data import site

FEATURE_POLICY = "application-pd_default-lgd_v5"

EXCLUDED = {
    ("pd", "0006.hackerearth"): (
        "collection_recovery_fee", "last_week_pay", "recoveries", "total_rec_int", "total_rec_late_fee",
    ),
}


def prediction_features(task: str, dataset: str, X: np.ndarray, cat_indices: list[int],
                        feature_names: list[str] | None = None):
    """Return X, remapped categorical indices and an auditable exclusion report.

    Selection depends only on source column semantics, never on labels or a
    measured correlation. Every model and monitoring hook uses this same view.
    """
    key = canonical_slug(dataset)
    configured = site(required=True) if is_proprietary(key) else None
    private_names = configured.EXCLUSIONS.get(key, ()) if configured else ()
    markers = configured.MACRO_MARKERS.get(key) if configured else None
    excluded = {name.casefold() for name in (*EXCLUDED.get((task, key), ()), *private_names)}
    names = feature_names
    if (excluded or markers) and (names is None or len(names) != X.shape[1]):
        raise ValueError(f"{task}/{dataset}: column names required for prediction-time exclusions")
    names = names if names is not None else [str(i) for i in range(X.shape[1])]
    def eligible(name):
        normalized = name.removesuffix("__cat").casefold()
        future_macro = markers and (normalized.startswith(markers[0]) or markers[1] in normalized)
        return normalized not in excluded and not future_macro

    keep = [i for i, name in enumerate(names) if eligible(name)]
    kept = set(keep)
    dropped = [name for i, name in enumerate(names) if i not in kept]
    if not keep:
        raise ValueError(f"{task}/{dataset}: no features remain under {FEATURE_POLICY}")
    cats = set(cat_indices)
    report = {"feature_policy": FEATURE_POLICY,
              "n_features_before_policy": int(X.shape[1]),
              "excluded_features": json.dumps(dropped),
              "n_features_excluded": len(dropped)}
    return (X[:, keep] if dropped else X,
            [new for new, old in enumerate(keep) if old in cats], report)
