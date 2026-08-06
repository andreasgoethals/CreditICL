"""The data pipeline: path contracts, caching, and the registry."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import numpy as np

from src.data.dataset_registry import REGISTRY, datasets_for_task
from src.data.discovery import list_datasets
from src.utils.paths import find_raw_file, find_raw_path, raw_file_for


def test_registry_covers_both_tasks():
    assert len(REGISTRY) == 21
    assert len(datasets_for_task("pd")) == 14
    assert len(datasets_for_task("lgd")) == 7


def test_every_raw_dataset_is_in_the_registry():
    """A dataset on disk but not registered would be silently skipped."""
    for task in ("pd", "lgd"):
        for slug in list_datasets(task):
            assert slug in REGISTRY, f"{task}/{slug} is on disk but not registered"


def test_raw_file_for_appends_and_never_replaces():
    """THE pathlib trap. Slugs look like `0001.gmsc`, so pathlib thinks `.gmsc` is
    the suffix and `with_suffix` would give `0001.csv`. In TabPFNCredit that
    silently broke every raw lookup.
    """
    from pathlib import Path

    stem = Path("/data/raw/pd/0001.gmsc")
    assert raw_file_for(stem, ".csv").name == "0001.gmsc.csv"
    assert raw_file_for(stem, ".csv").name != "0001.csv"


def test_find_raw_path_returns_a_stem_not_a_file():
    """dataset_preprocessing.py, copied from TabPFNCredit, appends the extension
    itself. Returning a full filename here produced `...csv.csv` — a real bug that
    made all 7 LGD datasets fail.
    """
    stem = find_raw_path("lgd", "0006.lgd_freddie")
    if stem is None:
        pytest.skip("raw data not present")
    assert stem.suffix == ".lgd_freddie", "should be the bare slug, no extension"
    assert not stem.is_file(), "a stem is not a file"
    assert find_raw_file("lgd", "0006.lgd_freddie").is_file()


def test_find_raw_path_none_for_unknown():
    assert find_raw_path("lgd", "9999.nonexistent") is None
    assert find_raw_file("lgd", "9999.nonexistent") is None


@pytest.mark.slow
def test_preprocess_and_load_roundtrip():
    from src.data.pipeline import load_processed, preprocess_one

    if find_raw_path("lgd", "0003.axa") is None:
        pytest.skip("raw data not present")
    preprocess_one("lgd", "0003.axa")
    ds = load_processed("lgd", "0003.axa")
    assert ds.X.shape[0] == ds.y.shape[0] == ds.n_rows
    assert ds.X.shape[1] == ds.n_features == len(ds.feature_names)
    assert ds.X.dtype == np.float32 and ds.y.dtype == np.float32
    assert ds.meta.task == "lgd"
    assert ds.meta.mass_at_0 is not None, "LGD metadata must record boundary mass"
    assert ds.meta.base_rate is None, "base_rate is a PD-only field"
    # The frame keeps names and dtypes; the exploration notebook needs both.
    assert list(ds.frame.columns) == ds.feature_names


@pytest.mark.slow
def test_cached_second_call_is_fast():
    """The cache must actually be used, or every eval run redoes 21 datasets."""
    import time

    from src.data.pipeline import preprocess_one

    if find_raw_path("lgd", "0003.axa") is None:
        pytest.skip("raw data not present")
    preprocess_one("lgd", "0003.axa")
    t0 = time.time()
    preprocess_one("lgd", "0003.axa")
    assert time.time() - t0 < 0.5, "second call should hit the cache"


@pytest.mark.slow
def test_pd_metadata_records_base_rate():
    from src.data.pipeline import load_processed, preprocess_one

    if find_raw_path("pd", "0008.german") is None:
        pytest.skip("raw data not present")
    preprocess_one("pd", "0008.german")
    ds = load_processed("pd", "0008.german")
    assert ds.meta.base_rate == pytest.approx(float(ds.y.mean()), abs=1e-6)
    assert set(np.unique(ds.y)) <= {0.0, 1.0}


@pytest.mark.slow
def test_german_credit_keeps_all_1000_rows():
    """German Credit has no header row. Reading it with an inferred header eats
    the first record, and the mangled column names look plausible enough that
    nobody notices — TabPFNCredit scored 999 rows for a while.
    """
    from src.data.pipeline import load_processed, preprocess_one

    if find_raw_path("pd", "0008.german") is None:
        pytest.skip("raw data not present")
    preprocess_one("pd", "0008.german")
    ds = load_processed("pd", "0008.german")
    assert ds.n_rows == 1000, f"expected 1000 rows, got {ds.n_rows}"


@pytest.mark.slow
def test_categorical_indices_point_at_a_contiguous_tail():
    """Numeric columns first, categoricals last. CatBoost and TabPFN both take
    indices, and a wrong mapping silently treats a category code as a magnitude.
    """
    from src.data.pipeline import load_processed, preprocess_one

    if find_raw_path("pd", "0008.german") is None:
        pytest.skip("raw data not present")
    preprocess_one("pd", "0008.german")
    ds = load_processed("pd", "0008.german")
    if ds.cat_indices:
        assert ds.cat_indices == list(range(min(ds.cat_indices), ds.n_features))


@pytest.mark.slow
def test_parquet_preserves_dtypes_and_names():
    """The reason for choosing parquet over .npy: an array format cannot carry
    which columns are categorical, nor what they are called."""
    import pandas as pd

    from src.data.pipeline import TARGET_COLUMN, load_frame, preprocess_one

    if find_raw_path("pd", "0008.german") is None:
        pytest.skip("raw data not present")
    preprocess_one("pd", "0008.german")
    frame, meta = load_frame("pd", "0008.german")
    assert TARGET_COLUMN in frame.columns, "target travels with the features"
    cats = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]
    assert cats, "german has categorical columns; the dtype must survive the round trip"
    assert all(isinstance(c, str) for c in frame.columns), "column names must survive"


@pytest.mark.slow
def test_loader_is_memory_cached():
    """The eval loop asks for the same dataset once per model; parquet is slower to
    read than .npy, so the cache is what makes that free."""
    import time

    from src.data.pipeline import load_processed, preprocess_one

    if find_raw_path("pd", "0008.german") is None:
        pytest.skip("raw data not present")
    preprocess_one("pd", "0008.german")
    load_processed("pd", "0008.german")
    t0 = time.time()
    load_processed("pd", "0008.german")
    assert time.time() - t0 < 0.05, "second load should come from the in-memory cache"
