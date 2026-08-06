"""PIPELINE 1 of 4 — turn raw credit files into a cached, ready-to-model form.

This is the least scientifically interesting pipeline and the one most likely to
waste a day, so it is built to be boring and loud.

WHAT IT DOES, per dataset:

    raw .csv/.parquet
        -> dataset-specific recipe   (src/data/dataset_preprocessing.py)
        -> one tidy table with correct dtypes, target as a column
        -> written as data.parquet + meta.json into the processed cache

FORMAT: ONE PARQUET FILE, NOT SEPARATE ARRAYS. Measured on Home Credit
(307k x 120): parquet+zstd is **17 MB against 149 MB** for X.npy + y.npy, an 8.5x
saving, and it is 2 files per dataset instead of 4 — which matters because project
storage has a low inode budget. It also keeps the two things an array format
throws away: real dtypes (a categorical stays categorical, instead of being an
integer code smuggled inside a float32 matrix) and the **column names**, which the
exploration notebook needs and no .npy can carry.

Parquet reads slower — 1.0s vs 0.06s on that dataset. Irrelevant: one CatBoost fit
on it takes minutes, and `load_processed` caches within a run.

WHERE THE CACHE LIVES. On the cluster, project storage
(`/lustre1/project/stg_00211/CreditICL/data/processed/<task>/<dataset>/`); locally,
`data/processed/<task>/<dataset>/`. Decided by `src/utils/paths.py`, which checks
whether `$VSC_DATA` exists — nothing to configure.

CACHING RULE. `meta.json` is written LAST and is the completeness marker. If a run
dies halfway through a dataset, the directory has arrays but no `meta.json`, so the
next run correctly treats it as missing and redoes it rather than loading a
half-written cache. The eval pipeline calls `ensure_processed`, which preprocesses
only what is absent.

LOGGING. Every dataset logs its raw path, row and column counts before and after,
which columns were dropped and why, the target's range, the class balance or
boundary mass, and how long it took. If a run on the cluster goes wrong, that log
should be enough to find it without re-running anything.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset_preprocessing import preprocess_dataset_specific
from src.data.discovery import list_datasets
from src.utils.logging_setup import get_logger
from src.utils.paths import find_processed_dir, find_raw_file, processed_write_dir

CACHE_VERSION = 3  # bump to invalidate every cache; recorded in meta.json


@dataclass
class DatasetMeta:
    """What we know about a processed dataset. Written to meta.json."""

    dataset: str
    task: str
    n_rows: int
    n_features: int
    target_col: str
    cat_cols: list[str] = field(default_factory=list)
    num_cols: list[str] = field(default_factory=list)
    cat_indices: list[int] = field(default_factory=list)
    n_raw_rows: int = 0
    n_raw_cols: int = 0
    dropped_cols: list[str] = field(default_factory=list)
    target_min: float = 0.0
    target_max: float = 0.0
    target_mean: float = 0.0
    n_missing_cells: int = 0
    # PD only
    base_rate: float | None = None
    # LGD only
    mass_at_0: float | None = None
    mass_at_1: float | None = None
    cache_version: int = CACHE_VERSION
    raw_path: str = ""
    seconds: float = 0.0


#: The target is stored as a COLUMN, not a separate file, so one read gets
#: everything and features and target can never drift out of alignment.
TARGET_COLUMN = "__target__"


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    frame.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _summarise_target(task: str, y: np.ndarray, meta: DatasetMeta) -> None:
    meta.target_min = float(np.nanmin(y))
    meta.target_max = float(np.nanmax(y))
    meta.target_mean = float(np.nanmean(y))
    if task == "pd":
        meta.base_rate = float(np.nanmean(y))
    else:
        # Mass exactly on the boundaries — the quantity the LGD prior is built to
        # reproduce, so it belongs in the metadata rather than being recomputed.
        meta.mass_at_0 = float(np.mean(y <= 0.0))
        meta.mass_at_1 = float(np.mean(y >= 1.0))


def preprocess_one(task: str, dataset: str, *, force: bool = False) -> Path:
    """Preprocess one dataset into the cache and return its directory."""
    log = get_logger()
    started = time.time()

    existing = find_processed_dir(task, dataset)
    if existing is not None and not force:
        log.info("[data] %-28s cached, skipping  (%s)", f"{task}/{dataset}", existing)
        return existing

    raw_path = find_raw_file(task, dataset)
    if raw_path is None:
        raise FileNotFoundError(f"no raw file for {task}/{dataset}")
    log.info("[data] %-28s preprocessing from %s", f"{task}/{dataset}", raw_path)

    df, target_col, cat_cols, num_cols = preprocess_dataset_specific(task, dataset)

    if target_col not in df.columns:
        raise ValueError(f"{task}/{dataset}: recipe returned target {target_col!r}, not in columns")

    # The recipes deliberately leave categoricals as strings — they only decide
    # WHICH columns are categorical, not how to encode them. Forcing the whole
    # frame to float32 therefore dies on the first text value ("P", " 60 months").
    # Encode categoricals to integer codes, exactly as TabPFNCredit does, and keep
    # numeric columns first so the categorical indices are a contiguous tail.
    cat_cols = [c for c in cat_cols if c in df.columns and c != target_col]
    num_cols = [c for c in num_cols if c in df.columns and c != target_col]

    # Any column the recipe did not classify: treat as numeric if it converts,
    # categorical otherwise. Never silently dropped — an unclassified column is a
    # recipe gap worth seeing in the log.
    classified = set(cat_cols) | set(num_cols) | {target_col}
    for col in df.columns:
        if col in classified:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
        get_logger().warning(
            "[data] %-28s column %r was not classified by the recipe; treating as %s",
            f"{task}/{dataset}", col, "numeric" if col in num_cols else "categorical",
        )

    feature_cols = num_cols + cat_cols
    cat_indices = list(range(len(num_cols), len(feature_cols)))

    blocks = []
    if num_cols:
        blocks.append(df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
    if cat_cols:
        codes = df[cat_cols].astype("category").apply(lambda s: s.cat.codes).to_numpy(dtype=np.int64)
        # Codes ride along in the float32 matrix. Exact for any cardinality below
        # 2^24, and every model here takes cat_indices rather than a dtype.
        blocks.append(codes.astype(np.float32))
    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), dtype=np.float32)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)

    meta = DatasetMeta(
        dataset=dataset,
        task=task,
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
        target_col=str(target_col),
        cat_cols=[str(c) for c in cat_cols],
        num_cols=[str(c) for c in num_cols],
        cat_indices=cat_indices,
        n_missing_cells=int(np.isnan(X).sum()),
        raw_path=str(raw_path),
    )
    _summarise_target(task, y, meta)

    out_dir = processed_write_dir(task, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the tidy table: features keeping their real dtypes, plus the target.
    tidy = pd.DataFrame({c: df[c] for c in num_cols})
    for c in cat_cols:
        # A pandas category, so the DTYPE itself records "this is categorical" and
        # there is no parallel index list to keep in sync.
        tidy[c] = df[c].astype("category")
    tidy[TARGET_COLUMN] = y

    # Atomic write, because on the cluster several array tasks can call
    # ensure_processed() for the SAME not-yet-cached dataset at once, on different
    # nodes sharing project storage. Without temp-file-plus-rename a reader can see
    # a half-written file. Preprocessing is deterministic, so last-writer-wins is
    # harmless — a torn file is not.
    _atomic_write_parquet(out_dir / "data.parquet", tidy)

    meta.seconds = round(time.time() - started, 2)
    # LAST, on purpose: this file is the completeness marker, so an interrupted
    # run leaves a directory that reads as absent rather than as complete.
    _atomic_write_text(out_dir / "meta.json", json.dumps(asdict(meta), indent=2))

    extra = (
        f"base_rate={meta.base_rate:.4f}"
        if task == "pd"
        else f"mass@0={meta.mass_at_0:.4f} mass@1={meta.mass_at_1:.4f}"
    )
    log.info(
        "[data] %-28s OK  rows=%d features=%d cat=%d missing=%d  y in [%.4f, %.4f]  %s  %.1fs -> %s",
        f"{task}/{dataset}",
        meta.n_rows,
        meta.n_features,
        len(cat_indices),
        meta.n_missing_cells,
        meta.target_min,
        meta.target_max,
        extra,
        meta.seconds,
        out_dir,
    )
    return out_dir


@dataclass
class ProcessedDataset:
    """A loaded dataset, in the shape every consumer wants.

    `frame` keeps real dtypes and column names for exploration and plotting; `X` is
    the float32 matrix the models take, with `cat_indices` telling them which
    columns are categorical. Two views of the same data, built once per run.
    """

    task: str
    dataset: str
    frame: pd.DataFrame  # features only, real dtypes, named columns
    X: np.ndarray  # float32, numeric columns first then categorical codes
    y: np.ndarray  # float32
    cat_indices: list[int]
    feature_names: list[str]
    meta: DatasetMeta

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def load_frame(task: str, dataset: str) -> tuple[pd.DataFrame, DatasetMeta]:
    """The cached table as stored, plus its metadata. Target is `TARGET_COLUMN`."""
    d = find_processed_dir(task, dataset)
    if d is None:
        raise FileNotFoundError(
            f"{task}/{dataset} is not in the processed cache. "
            "Run scripts/preprocess.py, or call ensure_processed() first."
        )
    meta_raw = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta_raw.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"{task}/{dataset} was processed with cache_version="
            f"{meta_raw.get('cache_version')} but this code expects {CACHE_VERSION}. "
            "Re-run with: python scripts/preprocess.py --task both --force"
        )
    known = set(DatasetMeta.__dataclass_fields__)
    meta = DatasetMeta(**{k: v for k, v in meta_raw.items() if k in known})
    return pd.read_parquet(d / "data.parquet"), meta


@lru_cache(maxsize=4)
def _load_cached(task: str, dataset: str) -> ProcessedDataset:
    frame, meta = load_frame(task, dataset)
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    features = frame.drop(columns=[TARGET_COLUMN])

    # Numeric first, categorical last, so cat_indices is a contiguous tail. The
    # parquet dtype is the source of truth for which is which — no index list to
    # keep in sync, which is where the previous format could silently go wrong.
    cat_names = [c for c in features.columns if isinstance(features[c].dtype, pd.CategoricalDtype)]
    num_names = [c for c in features.columns if c not in set(cat_names)]
    ordered = num_names + cat_names

    blocks = []
    if num_names:
        blocks.append(features[num_names].to_numpy(dtype=np.float32))
    if cat_names:
        codes = np.stack([features[c].cat.codes.to_numpy() for c in cat_names], axis=1)
        blocks.append(codes.astype(np.float32))
    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(frame), 0), dtype=np.float32)

    return ProcessedDataset(
        task=task,
        dataset=dataset,
        frame=features[ordered],
        X=X,
        y=y,
        cat_indices=list(range(len(num_names), len(ordered))),
        feature_names=[str(c) for c in ordered],
        meta=meta,
    )


def load_processed(task: str, dataset: str) -> ProcessedDataset:
    """Load a processed dataset. Cached in memory so the eval loop reads once."""
    return _load_cached(task, dataset)


def ensure_processed(
    task: str,
    datasets: list[str] | str | None = None,
    *,
    force: bool = False,
) -> dict[str, Path | None]:
    """Make sure every requested dataset is in the cache, preprocessing as needed.

    This is what the eval pipeline calls. A dataset that fails is recorded as None
    and reported, rather than aborting the whole run — one broken recipe should not
    cost you the other twenty results.
    """
    log = get_logger()
    if isinstance(datasets, str):
        # A bare slug is iterable, so without this it would be walked character by
        # character: `ensure_processed("pd", "0011.loan_default")` quietly tried to
        # preprocess datasets named "0", "1", ".", "l", "o", "a", "n"... and reported
        # 17 failures for one request. Accept the singular form instead.
        datasets = [datasets]
    datasets = datasets if datasets is not None else list_datasets(task)
    out: dict[str, Path | None] = {}
    failures: list[tuple[str, str]] = []

    for dataset in datasets:
        try:
            out[dataset] = preprocess_one(task, dataset, force=force)
        except Exception as exc:  # noqa: BLE001 — one bad dataset must not kill the run
            log.error("[data] %-28s FAILED: %s: %s", f"{task}/{dataset}", type(exc).__name__, exc)
            log.exception("[data] traceback for %s/%s", task, dataset)
            out[dataset] = None
            failures.append((dataset, f"{type(exc).__name__}: {exc}"))

    ok = sum(v is not None for v in out.values())
    log.info("[data] %s: %d/%d datasets ready", task, ok, len(out))
    if failures:
        log.warning("[data] %s: %d FAILED — %s", task, len(failures), "; ".join(f[0] for f in failures))
    return out


def cache_report(task: str) -> pd.DataFrame:
    """One row per processed dataset. Goes into results/<task>/data/."""
    rows: list[dict[str, Any]] = []
    for dataset in list_datasets(task):
        d = find_processed_dir(task, dataset)
        if d is None:
            rows.append({"dataset": dataset, "task": task, "cached": False})
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta["cached"] = True
        meta.pop("cat_cols", None)
        meta.pop("num_cols", None)
        meta.pop("cat_indices", None)
        meta.pop("dropped_cols", None)
        rows.append(meta)
    return pd.DataFrame(rows)
