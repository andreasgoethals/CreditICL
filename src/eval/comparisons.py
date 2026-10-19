"""Paired comparisons with datasets, not CV folds, as the units of uncertainty."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.eval.protocol import N_FOLDS


def identities(df: pd.DataFrame) -> pd.Series:
    runs = df.get("info_run_name", pd.Series("", index=df.index)).fillna("")
    return runs.where(runs.ne(""), df["model"])


def require_balanced_cells(df: pd.DataFrame, datasets: list[str] | None = None) -> None:
    """Refuse duplicate, missing or failed cells, including entirely missing datasets.

    Checkpoint curves are separate identities at each step. Rows from older
    protocols must be filtered before calling this full five-fold check.
    """
    required = {"dataset", "model", "seed", "fold", "status"}
    if not required.issubset(df) or df.empty or not df.status.eq("ok").all():
        raise ValueError("A comparison requires successful five-fold benchmark cells")
    part = df.assign(_identity=identities(df))
    columns = ["_identity"]
    if "info_checkpoint_step" in part:
        columns.append("info_checkpoint_step")
    expected = {(name, seed, fold) for name in (datasets or sorted(part.dataset.unique()))
                for seed in part.seed.unique() for fold in range(N_FOLDS)}
    for identity, cells in part.groupby(columns, dropna=False):
        keys = list(cells[["dataset", "seed", "fold"]].itertuples(index=False, name=None))
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError(f"Incomplete or duplicate benchmark cells for {identity}")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm's step-down family-wise adjustment, returned in the original order."""
    p = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("P-values must be finite probabilities")
    order = np.argsort(p, kind="stable")
    out = np.empty_like(p)
    out[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out.tolist()


def paired_prior_effects(df: pd.DataFrame, metric: str, *, seed: int = 0,
                        repetitions: int = 10_000, pairs: dict[str, str] | None = None) -> pd.DataFrame:
    """Each mixture versus its matching 0% control, separately per architecture.

    Average folds and training seeds *within* each dataset, then bootstrap paired
    dataset differences. The CI describes variation across these benchmark tasks;
    folds and training seeds are not counted as independent datasets. The paired
    sign test is exact and two-sided; Holm corrects the returned comparison family.
    """
    from scipy.stats import binomtest

    require_balanced_cells(df)
    part = df.assign(identity=identities(df))
    part["configuration"] = part.identity.str.replace(r"__s\d+$", "", regex=True)
    if pairs is None:
        pairs = {name: re.sub(r"credit_fraction=[0-9p.]+", "credit_fraction=0", name)
                 for name in part.configuration.unique()
                 if (match := re.search(r"credit_fraction=([0-9p.]+)", name))
                 and float(match.group(1).replace("p", ".")) > 0}
    involved = set(pairs) | set(pairs.values())
    # Point-only references cannot supply CRPS; they are not participants in a
    # contrast between our distributional models and must not veto that metric.
    part = part[part.configuration.isin(involved)]
    if repetitions < 1 or metric not in part or not np.isfinite(part[metric]).all():
        raise ValueError("Finite scores and a positive bootstrap count are required")
    cells = part.groupby(["configuration", "identity", "dataset"])[metric].mean()
    scores = cells.groupby(level=[0, 2]).mean().unstack("dataset")
    rng = np.random.default_rng(seed)
    rows = []
    for name in scores.index:
        if name not in pairs:
            continue
        control = pairs[name]
        if control not in scores.index:
            raise ValueError(f"Missing comparison control {control} for {name}")
        treatment_seeds = set(part.loc[part.configuration.eq(name), "identity"].str.extract(r"__s(\d+)$")[0])
        control_seeds = set(part.loc[part.configuration.eq(control), "identity"].str.extract(r"__s(\d+)$")[0])
        if control.startswith("exp") and treatment_seeds != control_seeds:
            raise ValueError(f"Unmatched training seeds for {name}")
        delta = (scores.loc[name] - scores.loc[control]).to_numpy(dtype=float)
        if not np.isfinite(delta).all():
            raise ValueError(f"Unmatched datasets for {name}")
        nonzero = delta[delta != 0]
        p = binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) else 1.
        if len(delta) > 1:
            samples = rng.choice(delta, size=(repetitions, len(delta)), replace=True).mean(axis=1)
            low, high = np.quantile(samples, [.025, .975])
        else:
            low = high = float("nan")  # one dataset cannot support across-dataset uncertainty
        rows.append({"configuration": name, "control": control, "mean_difference": delta.mean(),
                     "ci_low": low, "ci_high": high, "datasets": len(delta), "p_value": p,
                     "interval_method": "descriptive paired-dataset percentile bootstrap (nominal 95%)",
                     "small_sample": len(delta) < 10,
                     "minimum_sign_p": min(1., 2. ** (1 - len(nonzero)))})
    result = pd.DataFrame(rows)
    if rows:
        result["p_holm"] = holm_adjust(result.p_value.tolist())
    return result


def paired_ablation_effects(df: pd.DataFrame, metric: str, track: str,
                            **kwargs) -> pd.DataFrame:
    """Every component removal versus the full 50% prior, matched by training seed."""
    names = identities(df).str.replace(r"__s\d+$", "", regex=True)
    pairs = {name: f"exp1_{track}__prior-credit_fraction=0p5"
             for name in names.unique() if name.startswith(f"exp1ablations_{track}__")}
    return paired_prior_effects(df, metric, pairs=pairs, **kwargs)


def paired_reference_effects(df: pd.DataFrame, metric: str, **kwargs) -> pd.DataFrame:
    """Final checkpoint minus its released architecture; datasets remain the paired units."""
    names = identities(df).str.replace(r"__s\d+$", "", regex=True)
    pairs = {name: "tabpfn3" if "tabpfn3" in name else "tabiclv2"
             for name in names.unique() if name.startswith("exp")}
    return paired_prior_effects(df, metric, pairs=pairs, **kwargs)
