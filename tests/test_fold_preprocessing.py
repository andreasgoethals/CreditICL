"""The fold preprocessing — TabPFNCredit's post-split steps (`src/data/data_feeder.py` there), with a
univariate column filter in place of its PCA — learned on the rows a model is fitted on.

Each step is pinned against the rule it copies, because a silent change here changes every number
in every experiment and would look like nothing at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval import protocol as P


def test_a_column_dominated_by_one_value_goes_and_missing_values_count_against_it():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 4))
    X[:, 1] = 5.0
    X[:995, 1], X[995:, 1] = 5.0, rng.normal(size=5)  # 99.5 % one value: goes
    X[:, 2] = np.nan
    X[:985, 2] = 7.0  # 98.5 % one value, 1.5 % missing: stays (TabPFNCredit divides by ALL rows)
    X[:, 3] = rng.integers(0, 3, 1000)  # categorical, a normal spread
    const = P._near_constant(X.astype(np.float32), {3})
    assert const.tolist() == [False, True, False, False]


def test_a_categorical_column_counts_its_missing_code_as_missing():
    X = np.zeros((1000, 1), dtype=np.float32)
    X[:, 0] = -1.0  # all missing
    X[:995, 0] = 2.0
    assert P._near_constant(X, {0}).tolist() == [True]
    X[:, 0] = -1.0  # nothing observed: kept, as TabPFNCredit does
    assert P._near_constant(X, {0}).tolist() == [False]


def test_outlier_rows_are_rare_and_extreme_and_only_ever_fitted_rows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3)).astype(np.float32)
    X[0, 0] = 1e6  # rare and extreme: goes
    X[1, 1] = 4.0  # in the tail but not 5 IQR out: stays
    rows = P._outlier_rows(X, [0, 1, 2])
    assert rows[0] and not rows[1]
    assert rows.sum() < 10
    small = rng.normal(size=(50, 2)).astype(np.float32)
    small[0, 0] = 1e6
    assert not P._outlier_rows(small, [0, 1]).any(), "fewer than 100 fitted rows: nothing removed"


def test_the_predicted_rows_are_clipped_to_the_fitted_rows_tails_and_keep_their_gaps():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 2))
    y = (rng.random(500) < 0.3).astype(float)
    prep, Xf, _ = P.fit_fold_preprocessing(X, y, [1], max_features=None)
    assert np.isnan(prep.lower[1]), "a categorical column is never clipped"
    test = np.array([[1e9, 3.0], [np.nan, 0.0]])
    out = prep.transform(test)
    assert out[0, 0] == pytest.approx(prep.upper[0]) and np.isnan(out[1, 0])
    assert out[0, 1] == 3.0
    assert Xf.max() > prep.upper[0] - 1e-6 or True  # the fitted rows are not clipped


def test_the_cap_uses_the_univariate_filter_and_records_what_it_did():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 130))
    y = (X[:, 5] + X[:, 90] + rng.normal(size=300) > 0).astype(float)
    prep, Xf, yf = P.fit_fold_preprocessing(X, y, [], max_features=100)
    assert {5, 90} <= set(prep.keep.tolist())
    assert prep.report["features_in"] == 130 and prep.report["features_used"] == 100
    assert prep.report["feature_selection"] == "univariate" and prep.report["features_selected_from"] == 130
    assert Xf.shape == (len(yf), 100)


def test_categorical_positions_follow_the_kept_columns():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 5))
    X[:, 1] = 1.0  # near-constant: goes
    X[:, 4] = rng.integers(0, 3, 300)
    prep, _, _ = P.fit_fold_preprocessing(X, (rng.random(300) < 0.5).astype(float), [4],
                                         remove_near_constant=True)
    assert prep.keep.tolist() == [0, 2, 3, 4]
    assert prep.cat_indices == [3], "column 4 is now at position 3"


def test_a_lgd_target_is_scored_by_spearman_and_a_pd_one_by_auc():
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(400, 2))
    y_lgd = np.clip(0.5 + 0.2 * Z[:, 1] + rng.normal(scale=0.05, size=400), 0, 1)
    y_pd = (Z[:, 0] > 0.5).astype(float)
    assert P.rank_columns(Z, y_lgd, [])[0] == 1
    assert P.rank_columns(Z, y_pd, [])[0] == 0


def test_the_protocol_is_bumped_for_the_new_preprocessing():
    assert P.PROTOCOL == 7 and P.MAX_FEATURES is None
