"""Full-width benchmark contract, including real upstream probes reused by Exp0 GPU."""
import numpy as np
import pytest


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_every_fold_stage_retains_wide_and_near_constant_features(task):
    from src.eval.runner import prepare_fold
    rng = np.random.default_rng(17)
    X = rng.normal(size=(120, 601)).astype(np.float32)
    X[:, 450] = 1  # retain even a constant: the upstream model may handle it internally
    X[:, 600] = rng.integers(0, 3, len(X))
    y = (np.arange(len(X)) % 2).astype(float) if task == "pd" else rng.random(len(X))
    stages = prepare_fold(task, X[:100], y[:100], X[100:], [600])
    for name in ("final", "threshold") if task == "pd" else ("final",):
        stage = stages[name]
        assert stage.X_fit.shape[1] == stage.X_pred.shape[1] == 601
        assert stage.cat_indices == [600]
        assert stage.report["near_constant_dropped"] == 0
        assert "features_selected_from" not in stage.report


def assert_released_full_width(model_name, task, width=601):
    """Small row count but > both TabPFN member budgets; real weights, no network or training."""
    from src.eval.baselines import build
    rng = np.random.default_rng(17)
    X = rng.normal(size=(44, width)).astype(np.float32)
    X[:, -1] = rng.integers(0, 3, len(X))
    y = (np.arange(len(X)) % 2).astype(float) if task == "pd" else rng.random(len(X))
    model = build(model_name, task, seed=0, n_estimators=2)
    report = model.fit(X[:40], y[:40], [width - 1])
    pred = model.predict(X[40:])
    assert report.n_features == width and report.n_train_used == 40
    assert pred.shape == (4,) and np.isfinite(pred).all()
    if model_name == "tabpfn3":
        assert report.extra["feature_subsampling"] is False
        assert all(i is None for i in model._model.ensemble_preprocessor_.subsample_feature_indices)
    if task == "lgd":
        q, levels = model.predict_quantiles(X[40:])
        assert q.shape == (4, len(levels)) and np.isfinite(q).all()
    return {"model": model_name, "task": task, "features": report.n_features,
            "fit_seconds": report.fit_seconds, "predict_seconds": report.predict_seconds}


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_classical_baselines_use_wide_features(task):
    from src.eval.baselines import build
    rng = np.random.default_rng(11)
    X = rng.normal(size=(64, 216)).astype(np.float32)
    y = np.arange(64) % 2 if task == "pd" else rng.random(64)
    for family in ("linear", "catboost"):
        model = build(family, task, iterations=2, depth=2)
        report = model.fit(X[:54], y[:54], [])
        assert report.n_features == 216
        assert np.isfinite(model.predict(X[54:])).all()
