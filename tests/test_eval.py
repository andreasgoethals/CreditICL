"""The eval pipeline: metrics, splits, and baseline plumbing."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import numpy as np

from src.eval.baselines import BASELINES, availability_report, build
from src.eval.metrics import (
    boundary_mass_error,
    expected_calibration_error,
    lgd_metrics,
    pd_metrics,
    pinball_loss,
    recall_at_top_k,
)
from src.eval.runner import make_split

# --- metrics: LGD ------------------------------------------------------------


def test_perfect_prediction_gives_r2_one():
    y = np.linspace(0, 1, 100)
    m = lgd_metrics(y, y.copy())
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_r2_can_go_negative():
    """A model worse than the mean must report negative R^2, not a clipped zero.
    On low-signal credit data that genuinely happens and it is informative."""
    y = np.array([0.0, 0.5, 1.0, 0.2])
    m = lgd_metrics(y, np.array([1.0, 0.0, 0.0, 1.0]))
    assert m["r2"] < 0


def test_boundary_mass_error_detects_the_real_failure():
    """The metric this project exists to move: 20% of the truth sits at zero and
    the model predicts none there. Measured on Freddie, that is exactly what
    CatBoost and ridge both do."""
    y_true = np.concatenate([np.zeros(20), np.linspace(0.1, 0.9, 80)])
    y_pred = np.linspace(0.1, 0.9, 100)  # never predicts exactly 0
    e = boundary_mass_error(y_true, y_pred)
    assert e["true_mass_at_0"] == pytest.approx(0.2)
    assert e["pred_mass_at_0"] == pytest.approx(0.0)
    assert e["boundary_mass_err_0"] == pytest.approx(-0.2)
    assert e["boundary_mass_abs_err"] == pytest.approx(0.2)


def test_pinball_zero_when_quantiles_are_the_truth():
    y = np.zeros(10)
    q = np.zeros((10, 5))
    assert pinball_loss(y, q, np.linspace(0.1, 0.9, 5)) == pytest.approx(0.0)


def test_out_of_unit_predictions_are_reported():
    m = lgd_metrics(np.array([0.5, 0.5]), np.array([-0.3, 1.4]))
    assert m["pred_out_of_unit"] == pytest.approx(1.0)


def test_decoding_rule_is_recorded():
    """A point prediction from a two-humped distribution depends entirely on the
    decoding rule, so the number is meaningless without it."""
    m = lgd_metrics(np.array([0.2, 0.8]), np.array([0.3, 0.7]), decoding="median")
    assert m["decoding"] == "median"


# --- metrics: PD -------------------------------------------------------------


def test_pd_reports_the_majority_baseline_next_to_accuracy():
    """At a 7% base rate, 0.93 accuracy is what predicting nothing scores. The two
    numbers must appear together or accuracy actively misleads."""
    y = np.zeros(100)
    y[:7] = 1
    m = pd_metrics(y, np.full(100, 0.07))
    assert m["base_rate"] == pytest.approx(0.07)
    assert m["majority_class_accuracy"] == pytest.approx(0.93)


def test_perfect_ranking_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    m = pd_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == pytest.approx(1.0)


def test_ece_small_for_a_calibrated_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_large_for_an_overconfident_model():
    y = np.zeros(1000)
    y[:100] = 1  # 10% base rate
    assert expected_calibration_error(y, np.full(1000, 0.9)) > 0.5


def test_recall_at_top_k():
    y = np.zeros(100)
    y[:10] = 1
    p = np.concatenate([np.linspace(0.9, 0.8, 10), np.zeros(90)])
    assert recall_at_top_k(y, p, 0.10) == pytest.approx(1.0)


def test_pr_auc_lift_is_relative_to_the_base_rate():
    y = np.zeros(200)
    y[:20] = 1
    m = pd_metrics(y, np.concatenate([np.full(20, 0.9), np.full(180, 0.1)]))
    assert m["pr_auc_lift"] > 1.0


def test_log_loss_baseline_is_reported():
    """Predicting the base rate for everyone is the floor. Without it you cannot
    tell whether a log-loss is good or worse than knowing nothing."""
    y = np.zeros(1000)
    y[:100] = 1
    m = pd_metrics(y, np.full(1000, 0.1))
    assert m["log_loss"] == pytest.approx(m["log_loss_base_rate"], abs=1e-6)


# --- splits ------------------------------------------------------------------


def test_split_is_disjoint_and_complete():
    tr, te = make_split(100, test_size=0.2, seed=0)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == 100


def test_split_is_deterministic():
    a = make_split(500, test_size=0.2, seed=7)
    b = make_split(500, test_size=0.2, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_stratified_split_keeps_positives_on_both_sides():
    """Otherwise an imbalanced dataset can land a test set with zero positives,
    which makes ROC-AUC undefined and wastes the run."""
    y = np.zeros(500)
    y[:15] = 1  # 3% positives
    tr, te = make_split(500, test_size=0.2, seed=0, y=y, task="pd")
    assert y[tr].sum() > 0 and y[te].sum() > 0


def test_temporal_split_refuses_rather_than_faking_it():
    """Silently falling back to random would report a temporal result that was not
    temporal, which is worse than an error."""
    with pytest.raises(NotImplementedError, match="date column"):
        make_split(100, test_size=0.2, seed=0, split="temporal")


# --- baselines ---------------------------------------------------------------


def test_all_baselines_registered():
    assert set(BASELINES) == {"linear", "catboost", "tabpfn3", "tabiclv2"}


def test_availability_report_explains_every_absence():
    for name, (ok, err) in availability_report().items():
        assert isinstance(ok, bool)
        assert ok or isinstance(err, str), f"{name}: unavailable must say why"


def test_linear_baseline_fits_and_clips_lgd():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.clip(0.5 + 0.3 * X[:, 0], 0, 1)
    m = build("linear", "lgd", seed=0)
    m.fit(X, y, [])
    p = m.predict(X)
    assert p.shape == (200,)
    assert p.min() >= 0.0 and p.max() <= 1.0, "LGD is a fraction; predictions must be clipped"


def test_logistic_baseline_returns_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(float)
    m = build("linear", "pd", seed=0)
    m.fit(X, y, [])
    p = m.predict(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert m.name == "logistic", "the PD variant renames itself so results are clear"


def test_baseline_handles_nans():
    """Credit data is full of them and several models cannot take NaN."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    X[::10, 0] = np.nan
    y = np.clip(rng.uniform(size=200), 0, 1)
    m = build("linear", "lgd", seed=0)
    m.fit(X, y, [])
    assert np.isfinite(m.predict(X)).all()


def test_imputation_uses_train_statistics_only():
    """Imputing from the test set is leakage. Verified by checking the stored
    statistic comes from the fitted data, not the data being predicted."""
    X_train = np.array([[1.0], [2.0], [3.0]])
    X_test = np.array([[np.nan]])
    m = build("linear", "lgd", seed=0)
    m.fit(X_train, np.array([0.1, 0.2, 0.3]), [])
    filled, _ = m._impute(X_test, m._impute_stats)
    assert filled[0, 0] == pytest.approx(2.0), "should use the TRAIN median"


def test_tfm_row_cap_subsamples_and_records_it():
    """A silent subsample makes a model look worse for a reason nothing in the
    output explains."""
    from src.eval.baselines import TFM_MAX_TRAIN_ROWS, _TFMBaseline

    class Dummy(_TFMBaseline):
        name = "dummy"

        def _fit(self, X, y, cat_indices):
            self.seen = X.shape

        def _predict(self, X):
            return np.zeros(X.shape[0])

    n = TFM_MAX_TRAIN_ROWS + 5000
    X = np.zeros((n, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    rep = Dummy(task="lgd", seed=0).fit(X, y, [])
    assert rep.subsampled is True
    assert rep.n_train_used == TFM_MAX_TRAIN_ROWS
    assert rep.n_train_available == n


def test_tfm_row_cap_is_stratified_for_pd():
    """A uniform draw from a 3%-positive dataset can lose the positives."""
    from src.eval.baselines import TFM_MAX_TRAIN_ROWS, _TFMBaseline

    class Dummy(_TFMBaseline):
        name = "dummy"

        def _fit(self, X, y, cat_indices):
            self.y_seen = y

        def _predict(self, X):
            return np.zeros(X.shape[0])

    n = TFM_MAX_TRAIN_ROWS + 20000
    y = np.zeros(n, dtype=np.float32)
    y[: int(0.03 * n)] = 1.0
    m = Dummy(task="pd", seed=0)
    rep = m.fit(np.zeros((n, 3), dtype=np.float32), y, [])
    assert rep.extra["subsample_strategy"] == "stratified"
    assert m.y_seen.sum() > 0, "positives must survive the subsample"


def test_tfm_feature_cap_applies_and_records():
    """algorithmwatch has 2,986 features; TabPFN will not accept that."""
    from src.eval.baselines import TFM_MAX_FEATURES, _TFMBaseline

    class Dummy(_TFMBaseline):
        name = "dummy"

        def _fit(self, X, y, cat_indices):
            self.n_seen = X.shape[1]

        def _predict(self, X):
            return np.zeros(X.shape[0])

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, TFM_MAX_FEATURES + 200)).astype(np.float32)
    y = np.zeros(100, dtype=np.float32)
    m = Dummy(task="lgd", seed=0)
    rep = m.fit(X, y, [])
    assert m.n_seen == TFM_MAX_FEATURES
    assert rep.extra["features_capped_from"] == TFM_MAX_FEATURES + 200
    assert m.predict(X).shape == (100,), "predict must apply the same column subset"


# -- one evaluation harness, shared by all three experiments -------------------


def test_our_own_checkpoints_are_scorable_like_any_other_model():
    """The gap this closed: for a while nothing could score a checkpoint WE produced, so a
    finished run had no number attached to it. `CreditICLBaseline` puts our models in the same
    registry as CatBoost and TabPFN, which is what makes the comparison a single table."""
    from src.eval.baselines import BASELINES
    from src.eval.crediticl_baseline import register

    register()
    assert "crediticl" in BASELINES, "our own models must be a registered baseline"


def test_the_same_metrics_are_used_by_every_experiment():
    """Three experiments whose numbers are computed differently cannot be compared, and the
    whole point of Exp2 vs Exp3 is a comparison."""
    from src.utils.config import load

    for track in ("LGD", "PD"):
        metrics = {
            exp: tuple(load(f"config/{exp}_{track}.yaml", allow_placeholders=True)["eval"]["metrics"])
            for exp in ("Exp1", "Exp2", "Exp3")
        }
        assert len(set(metrics.values())) == 1, f"{track}: experiments disagree — {metrics}"


def test_the_same_dev_holdout_split_is_used_by_every_experiment():
    """A different split per experiment would let Exp3 look better than Exp2 purely by being
    scored on easier datasets."""
    from src.utils.config import load

    for track in ("LGD", "PD"):
        splits = {
            exp: (
                tuple(load(f"config/{exp}_{track}.yaml", allow_placeholders=True)["eval"]["dev_datasets"]),
                tuple(load(f"config/{exp}_{track}.yaml", allow_placeholders=True)["eval"]["holdout_datasets"]),
            )
            for exp in ("Exp1", "Exp2", "Exp3")
        }
        assert len(set(splits.values())) == 1, f"{track}: split differs between experiments"


def test_out_of_domain_is_measured_during_training_not_only_at_the_end():
    """A prior that helps credit by destroying generality must be visible while the run is
    still going, not discovered afterwards. `progress.n_ood` is what makes the OOD suites part
    of the training curve rather than a final afterthought."""
    from src.utils.config import load

    for exp in ("Exp1", "Exp2", "Exp3"):
        for track in ("LGD", "PD"):
            progress = load(f"config/{exp}_{track}.yaml", allow_placeholders=True)["progress"]
            assert progress["every_datasets"] > 0, f"{exp}_{track}: no progress curve"
            assert progress["n_ood"] > 0, f"{exp}_{track}: out-of-domain not scored during training"
            assert progress["n_datasets"] > 0, f"{exp}_{track}: credit data not scored during training"


def test_the_progress_tracker_records_both_credit_and_out_of_domain():
    """Both halves in one CSV, so the trade-off is one plot rather than a join."""
    import inspect

    from src.train import progress

    source = inspect.getsource(progress)
    assert "_ood_datasets" in source and "list_ood_datasets" in source
    assert "n_ood" in source
