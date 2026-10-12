"""The processed datasets are TabPFNCredit's, value for value (30-09-2026).

The per-dataset recipes were developed and validated in the sibling project
`1. TabPFN/TabPFNCredit`; this project copies them and must feed the models the SAME arrays:
the same numeric block, the same categorical codes, the same targets, the same column set.
Where that project is on disk (a laptop), every dataset is compared; elsewhere (the cluster) the
comparison is skipped and the storage round trip below still runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SIBLING = Path(__file__).resolve().parents[3] / "1. TabPFN" / "TabPFNCredit" / "data" / "processed"


def test_integer_labelled_categories_survive_the_parquet_round_trip(tmp_path):
    """Parquet returns an integer-labelled category as a NUMBER: `vehicle_loan`'s
    `manufacturer_id` was scored as one until 30-09-2026. Stored with string labels in the same
    order, it comes back a category with TabPFNCredit's codes."""
    from src.data.pipeline import stored_category

    s = pd.Series([120, 45, 48, 45, None])
    expected = s.astype("category").cat.codes.to_numpy()
    pd.DataFrame({"x": s.astype("category")}).to_parquet(tmp_path / "a.parquet", index=False)
    assert not isinstance(pd.read_parquet(tmp_path / "a.parquet")["x"].dtype, pd.CategoricalDtype), (
        "if this ever passes, the workaround below is no longer needed"
    )
    pd.DataFrame({"x": stored_category(s)}).to_parquet(tmp_path / "b.parquet", index=False)
    back = pd.read_parquet(tmp_path / "b.parquet")["x"]
    assert isinstance(back.dtype, pd.CategoricalDtype)
    assert np.array_equal(back.cat.codes.to_numpy(), expected)


def test_two_values_that_print_alike_stay_two_categories():
    from src.data.pipeline import stored_category

    cat = stored_category(pd.Series([1, "1", 2], dtype=object))
    assert len(set(cat.categories)) == len(cat.categories) == 3


def _datasets():
    if not SIBLING.is_dir():
        return []
    return [(t, d.name) for t in ("pd", "lgd") if (SIBLING / t).is_dir() for d in sorted((SIBLING / t).iterdir())]


@pytest.mark.skipif(not SIBLING.is_dir(), reason="the TabPFNCredit project is not on this machine")
@pytest.mark.parametrize("task, dataset", _datasets())
def test_every_dataset_matches_tabpfncredit(task, dataset):
    from src.data.loaders import load_processed
    from src.utils.paths import find_processed_dir

    if find_processed_dir(task, dataset) is None:
        pytest.skip(f"{task}/{dataset} is not processed here (python scripts/preprocess.py)")
    base = SIBLING / task / dataset
    info = json.loads((base / "info.json").read_text(encoding="utf-8"))
    N = np.load(base / "N.npy", mmap_mode="r") if (base / "N.npy").exists() else None
    C = np.load(base / "C.npy", mmap_mode="r") if (base / "C.npy").exists() else None
    y = np.load(base / "y.npy")
    ds = load_processed(task, dataset)
    n_num = N.shape[1] if N is not None else 0
    n_cat = C.shape[1] if C is not None else 0
    assert ds.X.shape == (len(y), n_num + n_cat), "the same column set"
    assert ds.cat_indices == list(range(n_num, n_num + n_cat)), "the same categorical block"
    rows = np.arange(len(y)) if ds.X.shape[1] < 500 else np.random.default_rng(0).choice(len(y), 5000, replace=False)
    if N is not None:
        assert np.array_equal(np.asarray(N[rows], np.float64), ds.X[rows, :n_num].astype(np.float64), equal_nan=True)
    if C is not None:
        assert np.array_equal(np.asarray(C[rows], np.int64), ds.X[rows, n_num:].astype(np.int64))
    assert np.array_equal(y.astype(np.float32), ds.y, equal_nan=True), "the same target (at float32)"
    assert info["n_samples"] == ds.n_rows
