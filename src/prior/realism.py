"""How much does a prior's table look like a real credit table? The same statistics for both.

The credit prior is meant to resemble the credit datasets it will be scored on. This module
measures that, statistic by statistic, with ONE definition applied to a real table and to a
generated one:

* **shape** — rows, columns;
* **feature types** — share of *discrete* columns (at most 20 distinct values in 1,024 rows: the
  categorical codes, flags and small counts that dominate credit files), median absolute
  skewness of the other columns, share of heavy-tailed ones (|skew| > 2);
* **missingness** — share of columns with missing values and their missing rate (a real table
  carries NaN; the prior's missing cells are mean-imputed, so its configured rates are reported);
* **target** — PD: the default rate. LGD: the mass at exactly 0 and exactly 1, the interior's mean,
  SD and skewness, on the [0, 1] scale;
* **difficulty** — the pseudo-R² of the predictability filter (a 25-tree ExtraTrees, out-of-bag)
  on 1,024-row samples, the axis the filter acts on (`src/prior/filters.predictability`).

`compare(task)` prints the real datasets beside the credit prior of `config/Exp1_<task>.yaml`.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch

#: A column with at most this many distinct values in 1,024 rows counts as DISCRETE.
DISCRETE_MAX_VALUES = 20
N_ROWS = 1024


def _skew(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 8 or np.std(x) == 0:
        return 0.0
    z = (x - x.mean()) / x.std()
    return float(np.mean(z ** 3))


def feature_stats(X: np.ndarray) -> dict[str, float]:
    """Column-type statistics of a (rows x columns) matrix; NaN is treated as missing."""
    n_cols = X.shape[1]
    if n_cols == 0:
        return {"n_features": 0}
    discrete, skews = 0, []
    for j in range(n_cols):
        col = X[:, j]
        obs = col[np.isfinite(col)]
        if obs.size == 0 or np.unique(obs).size <= 1:
            continue
        if np.unique(obs).size <= DISCRETE_MAX_VALUES:
            discrete += 1
        else:
            skews.append(abs(_skew(obs)))
    miss = np.isnan(X).mean(axis=0)
    cols_missing = miss > 0
    return {
        "n_features": int(n_cols),
        "share_discrete": discrete / n_cols,
        "median_abs_skew": float(np.median(skews)) if skews else 0.0,
        "share_heavy_tailed": float(np.mean(np.asarray(skews) > 2.0)) if skews else 0.0,
        "share_cols_missing": float(cols_missing.mean()),
        "missing_rate_in_those": float(miss[cols_missing].mean()) if cols_missing.any() else 0.0,
    }


def prediction_view_of_real(X: np.ndarray) -> np.ndarray:
    """A real sample as upstream's prediction path hands it to the model: NaN mean-imputed
    (`SimpleImputer`), constant columns dropped (`UniqueFeatureFilter`), then
    `PreprocessingPipeline("none")` — here fitted on the whole sample, which stands in for a
    context of the same rows."""
    from tabicl._sklearn.preprocessing import PreprocessingPipeline

    X = np.array(X, dtype=np.float64)
    means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(np.nan_to_num(means), np.nonzero(bad)[1])
    X = X[:, [j for j in range(X.shape[1]) if np.unique(X[:, j]).size > 1]]
    if X.shape[1] == 0:
        return X
    return PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0).fit(X).transform(X)


def view_stats(X: np.ndarray) -> dict[str, float]:
    """The features AS THE MODEL SEES THEM (after the prediction preprocessing): how skewed the
    continuous columns still are, how spread their bulk is (IQR / 1.349: 1 for a normal column,
    near 0 when a few extreme values set the scale and the soft clip squeezed them), and how much
    of a table lies beyond 3 SD."""
    skews, bulk = [], []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.unique(col).size <= DISCRETE_MAX_VALUES:
            continue
        skews.append(abs(_skew(col)))
        q1, q3 = np.quantile(col, [0.25, 0.75])
        bulk.append((q3 - q1) / 1.349)
    return {
        "view_median_abs_skew": float(np.median(skews)) if skews else float("nan"),
        "view_share_heavy_tailed": float(np.mean(np.asarray(skews) > 2.0)) if skews else float("nan"),
        "view_bulk_sd": float(np.median(bulk)) if bulk else float("nan"),
        "view_share_beyond_3sd": float(np.mean(np.abs(X) > 3.0)) if X.size else float("nan"),
    }


def target_stats(y: np.ndarray, task: str) -> dict[str, float]:
    y = np.asarray(y, dtype=float).ravel()
    if task == "pd":
        # The DEFAULT rate is the minority share: a prior table's labels have random identity
        # (as upstream's do), so "share of ones" would read 70 % half the time.
        p1 = float(np.mean(y >= 0.5))
        return {"base_rate": min(p1, 1.0 - p1)}
    # Within BOUNDARY_TOL of 0 or 1 counts as the atom: `axa` stores its atoms at 1e-5 and 1 - 1e-5.
    from src.eval.metrics import BOUNDARY_TOL as tol

    interior = y[(y > tol) & (y < 1 - tol)]
    return {
        "mass_at_0": float(np.mean(y <= tol)),
        "mass_at_1": float(np.mean(y >= 1.0 - tol)),
        "interior_mean": float(interior.mean()) if interior.size else float("nan"),
        "interior_sd": float(interior.std()) if interior.size else float("nan"),
        "interior_skew": _skew(interior) if interior.size else float("nan"),
        "share_below_0.1": float(np.mean(y < 0.1)),
        "share_above_0.9": float(np.mean(y > 0.9)),
    }


def _pseudo_r2(X: np.ndarray, y: np.ndarray, task: str) -> float:
    from src.prior.filters import predictability

    Xt = torch.as_tensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    if task == "pd":
        yt = (yt >= 0.5).float()
        if float(yt.std()) == 0.0:
            return float("nan")
    return float(predictability(Xt, yt, is_classif=task == "pd")[1])


def _oob_auc(X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC of the filter's ExtraTrees (same settings) on its out-of-bag predictions: PD
    difficulty on the scale credit people read, beside the Brier-based pseudo-R^2."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import roc_auc_score

    Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    yb = (np.asarray(y) >= 0.5).astype(float)
    if yb.min() == yb.max():
        return float("nan")
    et = ExtraTreesRegressor(n_estimators=25, bootstrap=True, oob_score=True, n_jobs=1,
                             random_state=1, max_depth=6).fit(Xn, yb)
    p = et.oob_prediction_
    ok = np.isfinite(p)
    auc = roc_auc_score(yb[ok], p[ok]) if ok.sum() > 8 and len(set(yb[ok])) == 2 else float("nan")
    return float(max(auc, 1.0 - auc)) if np.isfinite(auc) else auc


def describe_real(task: str, datasets: list[str] | None = None, repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """One row per real dataset: the statistics above, on `repeats` random 1,024-row samples
    (averaged) for everything but the shape and the target, which use the whole table."""
    from src.data.discovery import list_datasets
    from src.data.pipeline import load_processed

    rng = np.random.default_rng(seed)
    rows = []
    for name in datasets or list_datasets(task):
        ds = load_processed(task, name)
        X, y = np.asarray(ds.X, dtype=np.float64), np.asarray(ds.y, dtype=np.float64)
        row: dict[str, Any] = {"dataset": name.split(".", 1)[-1], "n_rows": len(y),
                               "n_features_full": X.shape[1], "share_categorical": len(ds.cat_indices) / max(X.shape[1], 1)}
        row.update(target_stats(y, task))
        samples = []
        for _ in range(repeats):
            idx = rng.choice(len(y), size=min(N_ROWS, len(y)), replace=False)
            Xs = X[idx].copy()
            # A categorical code of -1 is pandas' "missing category": count it as missing.
            for j in ds.cat_indices:
                Xs[Xs[:, j] < 0, j] = np.nan
            s = feature_stats(Xs)
            s.update(view_stats(prediction_view_of_real(Xs)))
            s["pseudo_r2"] = _pseudo_r2(Xs, y[idx], task)
            if task == "pd":
                s["oob_auc"] = _oob_auc(Xs, y[idx])
            samples.append(s)
        for k in samples[0]:
            row[k] = float(np.nanmean([s[k] for s in samples]))
        rows.append(row)
    return pd.DataFrame(rows)


def describe_prior(config_path: str, *, n: int = 80, credit_fraction: float = 1.0,
                   overrides: dict[str, Any] | None = None, seed: int = 0,
                   filtered: bool = False) -> pd.DataFrame:
    """One row per generated task, on the raw target scale (LGD in [0, 1]) so it reads like a
    real target. `filtered=False` shows the population the filter chooses from; `True` applies
    the config's filter as training does (mode `tabicl`, and its `apply_to`)."""
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.utils.config import expand_with_seeds, load

    cfg = expand_with_seeds(load(config_path))[0]
    task = cfg["task"]
    prior = copy.deepcopy(cfg["prior"])
    prior["credit_fraction"] = credit_fraction
    if not filtered:
        prior.setdefault("filter", {})["mode"] = "off"
    if task == "lgd":
        prior.setdefault("credit", {}).setdefault("target", {})["target_scaling"] = "none"
    for dotted, value in (overrides or {}).items():
        node = prior
        *path, last = dotted.split(".")
        for key in path:
            node = node.setdefault(key, {})
        node[last] = value
    gen = TaskGenerator(prior, task, PriorRNG(seed))
    rows = []
    for _ in range(n):
        t = gen.sample(shape=(N_ROWS, int(gen.sample_shape()[1])), num_classes=2 if task == "pd" else None)
        d = int(t.meta.get("d", t.X.shape[1]))
        X, y = t.X[:, :d].numpy().astype(np.float64), t.y.numpy().astype(np.float64)
        row: dict[str, Any] = {"source": t.source}
        row.update(feature_stats(X))
        # A generated table is already in the model's view: credit tables end in the prediction
        # preprocessing, control tables in upstream's training encoding.
        row.update(view_stats(X))
        if t.source == "credit":
            # The prior's gaps are mean-imputed, so they are read from the task's own record.
            miss = prior.get("credit", {}).get("missingness", {}) or {}
            n_miss = int(t.meta.get("missing_cols", 0) or 0)
            row["share_cols_missing"] = n_miss / max(d, 1)
            lo, hi = miss.get("missing_rate_range", [0.0, 0.0])
            row["missing_rate_in_those"] = (float(lo) + float(hi)) / 2 if n_miss else 0.0
        row.update(target_stats(y, task))
        row["pseudo_r2"] = _pseudo_r2(X, y, task)
        if task == "pd":
            row["oob_auc"] = _oob_auc(X, y)
        rows.append(row)
    return pd.DataFrame(rows)


def compare(task: str, config_path: str | None = None, n: int = 80, overrides: dict | None = None) -> str:
    """Real datasets beside the credit prior (and TabICL's prior), as quantiles per statistic."""
    config_path = config_path or f"config/Exp1_{task.upper()}.yaml"
    real = describe_real(task)
    credit = describe_prior(config_path, n=n, credit_fraction=1.0, overrides=overrides)
    base = describe_prior(config_path, n=n, credit_fraction=0.0)
    stats = [c for c in real.columns if c in credit.columns and c not in ("dataset", "source")]

    def q(s: pd.Series) -> str:
        s = s.dropna()
        return "   -" if s.empty else f"{s.quantile(0.1):6.2f} {s.median():6.2f} {s.quantile(0.9):6.2f}"

    lines = [f"{task.upper()} — p10 / median / p90", f"{'statistic':<24}{'real data':>21}{'credit prior':>21}{'TabICL prior':>21}"]
    for st in stats:
        lines.append(f"{st:<24}{q(real[st]):>21}{q(credit[st]):>21}{q(base[st]) if st in base else '   -':>21}")
    return "\n".join(lines)
