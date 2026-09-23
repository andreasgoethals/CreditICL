"""Regressions reproduced from the September 2026 Exp1 output audit."""
import csv

import numpy as np
import pandas as pd
import pytest
import torch

from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_invalid_upstream_sentinel_never_escapes_filter_fallback(monkeypatch, track):
    gen = TaskGenerator({"base": {"implementation": "transcribed"}, "max_filter_attempts": 2,
                         "max_invalid_attempts": 3}, track, PriorRNG(0))
    invalid = gen._finish(torch.zeros(16, 3), torch.full((16,), -100.0), {"source": "base"})
    monkeypatch.setattr(gen, "_sample_candidate", lambda *a, **kw: invalid)
    with pytest.raises(RuntimeError, match="3 invalid tasks"):
        gen.sample()


def test_fallback_keeps_last_valid_task_not_last_invalid_task(monkeypatch):
    gen = TaskGenerator({"base": {"implementation": "transcribed"}, "max_filter_attempts": 2,
                         "max_invalid_attempts": 2}, "pd", PriorRNG(0))
    valid = gen._finish(torch.randn(16, 3), torch.arange(16) % 2, {"source": "base"})
    invalid = gen._finish(torch.zeros(16, 3), torch.full((16,), -100.), {"source": "base"})
    candidates = iter([valid, invalid, invalid])
    monkeypatch.setattr(gen, "_sample_candidate", lambda *a, **kw: next(candidates))
    monkeypatch.setattr(gen.filter, "accept", lambda *a, **kw: False)
    result = gen.sample()
    assert result is valid and result.meta["filter_fallback"]
    assert result.meta["invalid_attempts"] == 2


def test_negative_regression_targets_remain_valid_and_nan_targets_are_not_repaired():
    gen = TaskGenerator({"base": {"implementation": "transcribed"}}, "lgd", PriorRNG(0))
    task = gen._finish(torch.ones(16, 3), torch.linspace(-10, -1, 16), {"source": "base"})
    assert gen._valid(task)
    task.y[0] = float("nan")
    assert not gen._valid(task)


def test_cached_negative_labels_are_rejected_before_model_or_cuda_access():
    from src.train.loop import Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = {"prior": {"n_classes": 2}}
    trainer.regression = False
    # No model/device/optimizer exists: the validation must happen before any of them.
    with pytest.raises(ValueError, match="Invalid synthetic class labels"):
        trainer.train_step((torch.zeros(1, 16, 3), torch.full((1, 16), -100.), 8))


def test_progress_resumption_preserves_header_order_and_adds_columns(tmp_path):
    from src.train.progress import ProgressConfig, ProgressTracker
    tracker = ProgressTracker(ProgressConfig(), "pd", "run", tmp_path)
    tracker._append({"progress_protocol": 2, "step": 1, "score": 0.4})
    resumed = ProgressTracker(ProgressConfig(), "pd", "run", tmp_path)
    resumed._append({"step": 2, "progress_protocol": 2, "new": 0.6, "score": 0.5})
    with resumed.path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["step"], r["score"]) for r in rows] == [("1", "0.4"), ("2", "0.5")]
    assert rows[1]["new"] == "0.6"


def test_monitor_uses_float32_and_reports_nonfinite_prediction_as_error(tmp_path):
    from src.train.progress import ProgressConfig, ProgressTracker

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, X, y, inference_config=None):
            assert not inference_config.COL_CONFIG.use_amp
            assert not inference_config.ROW_CONFIG.use_amp
            assert not inference_config.ICL_CONFIG.use_amp
            return torch.full((1, X.shape[1] - y.shape[1], 3), float("nan"))

    tracker = ProgressTracker(ProgressConfig(context_rows=16), "lgd", "run", tmp_path)
    with pytest.raises(ValueError, match="Non-finite predictions in float32"):
        tracker._score(Model(), np.ones((40, 2)), np.linspace(0, 1, 40), np.random.default_rng(0))


def test_ood_target_scaling_does_not_use_query_extremes(tmp_path):
    from src.train.progress import ProgressConfig, ProgressTracker

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.context = None

        def forward(self, X, y):
            self.context = y.clone()
            return torch.full((1, X.shape[1] - y.shape[1], 3), 0.5)

    y = np.linspace(0, 1, 40)
    query = np.random.default_rng(0).permutation(40)[16:]
    y[query[0]] = 1000
    model = Model()
    tracker = ProgressTracker(ProgressConfig(context_rows=16), "lgd", "run", tmp_path)
    tracker._score(model, np.ones((40, 2)), y, np.random.default_rng(0), scale_target=True)
    assert float(model.context.min()) == 0 and float(model.context.max()) == 1


def test_benchmark_requires_all_finite_cells_and_matching_context():
    from src.eval.benchmark_status import validate_cells
    df = pd.DataFrame([{"dataset": "a", "model": "m", "seed": i, "status": "ok",
                        "context_cap": 1024, "r2": 0.3} for i in range(3)])
    validate_cells(df, ["a"], ["m"], [0, 1, 2], "r2", 1024)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_cells(df.iloc[:2], ["a"], ["m"], [0, 1, 2], "r2", 1024)
    df.loc[0, "r2"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_cells(df, ["a"], ["m"], [0, 1, 2], "r2", 1024)


def test_benchmark_receipt_requires_both_domains_and_invalidates_changed_inputs(tmp_path, monkeypatch):
    from src.eval import benchmark_status as B
    files = {d: tmp_path / f"{d}.csv" for d in ("credit", "ood")}
    wanted = {"credit": ["a"], "ood": ["b"], "models": ["linear"], "seeds": [0], "context_cap": 1024}
    monkeypatch.setattr(B, "request", lambda *a, **kw: ("ref", wanted, files))
    monkeypatch.setattr(B, "receipt_path", lambda tag: tmp_path / "receipt.json")
    for domain, file in files.items():
        pd.DataFrame([{"dataset": wanted[domain][0], "model": "linear", "seed": 0,
                       "status": "ok", "r2": 0.5, "context_cap": 1024}]).to_csv(file, index=False)
    assert not B.complete(1, "lgd", 45)
    assert B.check(1, "lgd", 45, write=True)
    assert B.complete(1, "lgd", 45)
    files["ood"].unlink()
    assert not B.complete(1, "lgd", 45)


def test_prior_ranking_uses_dev_only_and_aggregates_training_seeds(monkeypatch):
    from src.eval.selection import ROOT, development_ranking
    from src.utils.config import expand_with_seeds, load, run_name
    cfg = load(ROOT / "config/Exp1_PD.yaml")
    runs = expand_with_seeds(cfg)
    rows = []
    for run in runs:
        name = run_name(run)
        credit = run["prior"]["credit_fraction"] > 0
        for ds in cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"]:
            dev = ds in cfg["eval"]["dev_datasets"]
            score = (0.9 if credit else 0.6) if dev else (0.1 if credit else 1.0)
            for seed in range(3):
                rows.append({"dataset": ds, "model": "crediticl", "info_run_name": name,
                             "seed": seed, "status": "ok", "roc_auc": score})
    ranking = development_ranking(pd.DataFrame(rows), "pd")
    assert len(ranking) == 15
    assert set(ranking.training_seeds) == {3}
    assert ranking.iloc[0]["mean"] == pytest.approx(0.9)
    assert ranking.iloc[-1]["mean"] == pytest.approx(0.6)
    import matplotlib.pyplot as plt

    from src.visualize import results_plots
    monkeypatch.setattr(results_plots, "load_results", lambda *a: pd.DataFrame(rows))
    fig = results_plots.overall_ranking("pd")
    assert len(fig.axes[0].patches) == 15
    plt.close(fig)


def test_benchmark_cannot_be_complete_with_missing_configured_data(monkeypatch):
    from types import SimpleNamespace

    from src.data import discovery
    from src.eval import benchmark_status as B
    from src.eval import ood
    monkeypatch.setattr(B, "load", lambda *a: {"eval": {"dev_datasets": ["a"], "holdout_datasets": ["b"]}})
    monkeypatch.setattr(B, "expand_with_seeds", lambda *a: [{}])
    monkeypatch.setattr(discovery, "list_datasets", lambda *a: ["a"])
    monkeypatch.setattr(ood, "list_ood_datasets", lambda *a: [SimpleNamespace(name="ood")])
    with pytest.raises(ValueError, match="Not all configured credit"):
        B.request(1, "pd", 1)
    monkeypatch.setattr(discovery, "list_datasets", lambda *a: ["a", "b"])
    with pytest.raises(ValueError, match="OOD cache needs"):
        B.request(1, "pd", 1)


def test_results_loader_excludes_obsolete_grid_indices(tmp_path, monkeypatch):
    from src.utils import paths
    from src.visualize import results_plots
    monkeypatch.setattr(paths, "results_dir", lambda *a: tmp_path)
    for index in (0, 45, 74):
        pd.DataFrame([{"dataset": "example", "model": f"arm-{index}", "roc_auc": 0.5,
                       "status": "ok"}]).to_csv(tmp_path / f"results_exp1bench_pd_a{index}.csv", index=False)
    result = results_plots.load_results("pd")
    assert result["model"].tolist() == ["arm-0"]


def test_ood_imputation_and_context_cap_use_training_protocol(monkeypatch):
    from types import SimpleNamespace

    from src.eval import ood_runner as O
    seen = {}
    class Model:
        max_context_rows = None
        def fit(self, X, y, **kwargs):
            seen["X"] = X.copy()
            seen["cap"] = self.max_context_rows
        def predict(self, X):
            return np.zeros(len(X))
    monkeypatch.setattr(O, "load_ood_dataset", lambda *a: (np.ones((5, 1)), np.arange(5), []))
    monkeypatch.setattr(O, "_split", lambda *a, **kw: (
        np.array([[1.], [np.nan], [3.]]), np.array([[10000.], [2.]]),
        np.array([0., 0.5, 1.]), np.array([0.25, 0.75])))
    monkeypatch.setattr(O, "build", lambda *a, **kw: Model())
    entry = SimpleNamespace(name="example", kind="regression", openml_id=0)
    result = O.evaluate_one_ood(entry, "linear", 0, O.OODEvalConfig(max_context_rows=1024))
    assert result["status"] == "ok"
    assert seen["X"][1, 0] == 2.0 and seen["cap"] == 1024
