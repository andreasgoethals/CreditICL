"""Monitoring must preserve the final benchmark's target and feature semantics."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.protocol import QUANTILE_LEVELS
from src.train.progress import ProgressConfig, ProgressTracker
from src.train.tabpfn_progress import TabPFNProgressTracker


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bounded", [True, False])
def test_benchmark_rejects_invalid_predictions_before_clipping(value, bounded):
    from src.eval.baselines import Baseline

    model = Baseline("lgd", bounded_target=bounded)
    with pytest.raises(ValueError, match="Non-finite predictions"):
        model._regression_output(np.array([0.5, value]))


@pytest.mark.parametrize("bounded,expected", [(False, 2.0), (True, 1.0)])
def test_monitor_clips_only_credit_losses(tmp_path, monkeypatch, bounded, expected):
    tracker = ProgressTracker(ProgressConfig(), "lgd", "test", tmp_path)
    monkeypatch.setattr(tracker, "_split", lambda *a, **kw: (np.arange(24), np.arange(24, 40)))
    monkeypatch.setattr(tracker, "_fit_predict", lambda m, X, y, Xt, **kw: {
        "point": np.full(len(Xt), 2.0),
        "quantiles": np.full((len(Xt), len(QUANTILE_LEVELS)), 2.0),
    })
    observed = {}

    def metrics(y, point, *, quantiles, **kwargs):
        observed.update(point=point, quantiles=quantiles)
        return {"rmse": 0.0}

    monkeypatch.setattr("src.eval.metrics.lgd_metrics", metrics)
    tracker._score(None, "test", np.arange(80).reshape(40, 2), np.linspace(0, 1, 40),
                   regression=True, bounded_target=bounded)
    assert np.all(observed["point"] == expected)
    assert np.all(observed["quantiles"] == expected)


def test_clipping_cannot_hide_infinite_predictions(tmp_path, monkeypatch):
    tracker = ProgressTracker(ProgressConfig(context_rows=16, max_test_rows=16), "lgd", "test", tmp_path)
    monkeypatch.setattr(tracker, "_fit_predict", lambda m, X, y, Xt, **kw: {
        "point": np.full(len(Xt), np.inf),
        "quantiles": np.full((len(Xt), len(QUANTILE_LEVELS)), np.inf),
    })
    with pytest.raises(ValueError, match="Non-finite predictions"):
        tracker._score(None, "test", np.arange(80).reshape(40, 2), np.linspace(0, 1, 40),
                       regression=True)


def test_tabpfn_monitor_receives_remapped_categories_and_target_bounds(tmp_path, monkeypatch):
    tracker = TabPFNProgressTracker(ProgressConfig(), "lgd", "test", tmp_path, trainer=None)
    tracker._weights = tmp_path / "weights.ckpt"
    monkeypatch.setattr(tracker, "_split", lambda *a, **kw: (np.arange(24), np.arange(24, 40)))
    observed = {}

    def fit(X, y, cats):
        observed["cats"] = cats
        observed["features"] = X.shape[1]

    model = SimpleNamespace(
        fit=fit, predict=lambda X: np.full(len(X), 0.5),
        predict_quantiles=lambda X: (np.full((len(X), len(QUANTILE_LEVELS)), 0.5), QUANTILE_LEVELS),
    )

    def build(*args, **kwargs):
        observed.update(kwargs)
        return model

    monkeypatch.setattr("src.eval.baselines.build", build)
    X = np.column_stack([np.ones(40), np.arange(40) % 3, np.arange(40)])
    tracker._score(None, "ood/test", X, np.linspace(0, 1, 40), regression=True,
                   bounded_target=False, cat_indices=[1])
    assert observed["cats"] == [0]  # the constant first column was removed
    assert observed["features"] == 2
    assert observed["bounded_target"] is False


def test_ood_monitor_preserves_loader_categories(tmp_path, monkeypatch):
    import torch

    tracker = ProgressTracker(ProgressConfig(), "lgd", "test", tmp_path)
    monkeypatch.setattr(tracker, "_real_datasets", lambda: [])
    monkeypatch.setattr(tracker, "_ood_datasets", lambda: [SimpleNamespace(name="sample")])
    monkeypatch.setattr("src.eval.ood.load_ood_dataset", lambda entry: (
        np.arange(80).reshape(40, 2), np.linspace(0, 1, 40), [1]))
    observed = {}

    def score(*args, **kwargs):
        observed.update(kwargs)
        return {"r2": 0.2}

    monkeypatch.setattr(tracker, "_score", score)
    tracker.record(torch.nn.Linear(2, 1), step=0, datasets_seen=0, train_loss=0.0, elapsed_s=0.0)
    assert observed["cat_indices"] == [1]
    assert observed["bounded_target"] is False
