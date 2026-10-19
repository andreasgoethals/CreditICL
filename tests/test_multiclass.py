"""Native multiclass OOD prediction, including the real released wrappers."""
import numpy as np
import pytest


def assert_multiclass_wrapper(model_name, classes, *, checkpoint=None):
    from src.eval.baselines import build
    if model_name == "crediticl":
        from src.eval.crediticl_baseline import register
        register()
    rng = np.random.default_rng(69)
    n = classes * 5
    X = rng.normal(size=(n + classes, 5)).astype(np.float32)
    y = np.arange(n) % classes
    kwargs = dict(n_estimators=1, iterations=3, depth=2)
    if checkpoint:
        kwargs["checkpoint" if model_name == "crediticl" else "model_path"] = str(checkpoint)
    m = build(model_name, "pd", **kwargs)
    report = m.fit(X[:n], y, [])
    p = m.predict(X[n:])
    assert p.shape == (classes, classes) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1, rtol=1e-5, atol=1e-5)
    assert report.n_train_used == n and report.n_features == 5


@pytest.mark.parametrize("model_name", ["linear", "catboost", "tabiclv2", "tabpfn3"])
@pytest.mark.parametrize("classes", [3, 11])
def test_native_multiclass_real_wrappers(model_name, classes):
    assert_multiclass_wrapper(model_name, classes)


def test_multiclass_cross_validation_preserves_all_classes_and_rows():
    from src.eval.protocol import cv_folds
    y = np.arange(150) % 3
    seen = np.zeros(len(y), int)
    for train, test in cv_folds(y, "pd"):
        assert set(y[train]) == set(y[test]) == {0, 1, 2}
        assert np.array_equal(np.bincount(y[test]), [10, 10, 10])
        seen[test] += 1
    assert np.all(seen == 1)


@pytest.mark.parametrize("model", ["linear", "catboost"])
def test_multiclass_five_fold_scoring_does_not_tune_a_binary_threshold(model):
    from src.eval.protocol import cv_folds
    from src.eval.runner import score_fold
    rng = np.random.default_rng(19)
    X = rng.normal(size=(150, 5)).astype(np.float32)
    y = np.arange(150) % 3
    for fold, (train, test) in enumerate(cv_folds(y, "pd")):
        row = score_fold("pd", model, X[train], y[train], X[test], y[test], [], fold=fold,
                         model_kwargs={"iterations": 3, "depth": 2})
        assert row["n_classes"] == 3 and row["n_val"] == 0 and row["n_train_used"] == 120
        assert "f1_tuned" not in row and np.isfinite(row["log_loss"])


def test_ood_preparation_keeps_native_classes(monkeypatch):
    from types import SimpleNamespace
    from src.eval import ood_runner as ood
    X = np.zeros((30, 4))
    y = np.resize([10, 20, 50], 30)
    monkeypatch.setattr(ood, "load_ood_dataset", lambda e: (X, y, []))
    _, encoded, _, info = ood._prepare(SimpleNamespace(kind="classification"), 0, ood.OODEvalConfig())
    assert set(encoded) == {0, 1, 2} and info["n_classes"] == 3
    assert "binarised_from_n_classes" not in info
