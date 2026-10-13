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

from src.data.dataset_registry import canonical_slug

FEATURE_POLICY = "application-pd_default-lgd_v2"

# 29 September source-data audit, applied after the prediction-time decision on
# 12 October. Exact source names, including categorical copies, are excluded.
# Exclude both numerator and denominator: these are outcome-construction inputs,
# not a separately verified exposure-at-default snapshot.
EXCLUDED = {
    ("pd", "0006.hackerearth"): {
        "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "last_week_pay",
    },
    ("lgd", "0005.base_modelisation"): {
        "numerateur", "denominateur", "C_flag_crc_clos_f", "annee_in_dft",
    },
    ("lgd", "0004.base_model"): {
        "FCLT_UnpaidInterest", "flag_clos", "duree_defaut", "DFLT_duree",
        "cat_DFLT_duree", "classe_DFLT_duree", "classe_duree",
        "DFLT_RationalEndDefault", "DFLT_RecoveryApproach",
        "FCLT_duree", "FCLT_duree_cat", "cat_FCLT_duree", "classe_FCLT_duree",
        "rating_1yad", "cat_rating_1yad",
        "DFLT_SIREN", "classe_DFLT_dicho_2A", "borne_1", "diff",
    },
    ("lgd", "0002.loss2"): {
        "UPB_At_Resolution", "Unpaid_Interest", "Total_Debt", "REO_Appraisal_Amount",
        "_Miclaimbal", "_Mirecovery", "_adv_interest1M", "_adv_interest", "_Accrued_int", "_Accruad_int", "_EAD",
    },
}


def prediction_features(task: str, dataset: str, X: np.ndarray, cat_indices: list[int],
                        feature_names: list[str] | None = None):
    """Return X, remapped categorical indices and an auditable exclusion report.

    Selection depends only on source column semantics, never on labels or a
    measured correlation. Every model and monitoring hook uses this same view.
    """
    excluded = {name.casefold() for name in EXCLUDED.get((task, canonical_slug(dataset)), set())}
    names = feature_names
    if excluded and (names is None or len(names) != X.shape[1]):
        raise ValueError(f"{task}/{dataset}: column names required for prediction-time exclusions")
    names = names if names is not None else [str(i) for i in range(X.shape[1])]
    keep = [i for i, name in enumerate(names)
            if name.removesuffix("__cat").casefold() not in excluded]
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
