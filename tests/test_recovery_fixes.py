"""Regressions reproduced from the September 2026 Exp1 output audit."""
import csv

import numpy as np
import pandas as pd
import pytest
import torch

from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG


def _task(X, y, **meta):
    from src.prior.base import SyntheticTask

    return SyntheticTask(X=X.float(), y=y.float(), source="base", meta={"source": "base", "d": X.shape[1], **meta})


def _plan(regression):
    from src.prior.stream import SlotPlan

    return SlotPlan(batch=0, slot=0, group=0, seq_len=16, train_size=8, num_features=3,
                    num_classes=None if regression else 2, use_credit=False)


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_invalid_upstream_sentinel_never_escapes_filter_fallback(monkeypatch, track):
    gen = TaskGenerator({"base": {"implementation": "transcribed"}, "max_filter_attempts": 2,
                         "max_invalid_attempts": 3}, track, PriorRNG(0))
    invalid = _task(torch.zeros(16, 3), torch.full((16,), -100.0))
    monkeypatch.setattr(gen, "_candidate", lambda *a, **kw: invalid)
    with pytest.raises(RuntimeError, match="3 invalid tasks"):
        gen.sample_slot(_plan(track == "lgd"))


def test_fallback_keeps_last_valid_task_not_last_invalid_task(monkeypatch):
    gen = TaskGenerator({"base": {"implementation": "transcribed"}, "max_filter_attempts": 2,
                         "max_invalid_attempts": 2}, "pd", PriorRNG(0))
    valid = _task(torch.randn(16, 3), torch.arange(16) % 2)
    invalid = _task(torch.zeros(16, 3), torch.full((16,), -100.))
    candidates = iter([valid, invalid, invalid])
    monkeypatch.setattr(gen, "_candidate", lambda *a, **kw: next(candidates))
    monkeypatch.setattr(gen.filter, "accept", lambda *a, **kw: False)
    result = gen.sample_slot(_plan(False))
    assert result is valid and result.meta["filter_fallback"]
    assert result.meta["invalid_attempts"] == 2


def test_negative_regression_targets_remain_valid_and_nan_targets_are_not_repaired():
    gen = TaskGenerator({"base": {"implementation": "transcribed"}}, "lgd", PriorRNG(0))
    task = _task(torch.randn(16, 3), torch.linspace(-10, -1, 16))
    assert gen._valid(task, _plan(True))
    task.y[0] = float("nan")
    assert not gen._valid(task, _plan(True))


def test_a_class_missing_from_the_context_is_repaired_or_rejected_as_upstream_does():
    """`cls_sanity_check`: the context and the query must hold the same classes. A table whose
    only positives sit in the query is repaired by permuting rows (upstream tries 10)."""
    gen = TaskGenerator({"base": {"implementation": "transcribed"}}, "pd", PriorRNG(0))
    y = torch.zeros(16)
    y[12:] = 1.0                                   # every positive in the query
    task = _task(torch.randn(16, 3), y)
    assert gen._valid(task, _plan(False))
    ctx, qry = set(task.y[:8].tolist()), set(task.y[8:].tolist())
    assert ctx == qry == {0.0, 1.0}
    shifted = _task(torch.randn(16, 3), y.clone(), shift="cohort")
    assert not gen._valid(shifted, _plan(False)), "a shifted table's row order is the shift"


def test_cached_negative_labels_are_rejected_before_model_or_cuda_access():
    from src.train.loop import Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = {"prior": {"max_classes": 2}}
    trainer.regression = False
    # No model/device/optimizer exists: the validation must happen before any of them.
    with pytest.raises(ValueError, match="Invalid synthetic class labels"):
        trainer.train_step((torch.zeros(1, 16, 3), torch.full((1, 16), -100.), 8))


def test_progress_resumption_preserves_header_order_and_adds_columns(tmp_path):
    from src.train.progress import ProgressConfig, ProgressTracker
    from src.train.progress import PROGRESS_PROTOCOL

    tracker = ProgressTracker(ProgressConfig(), "pd", "run", tmp_path)
    tracker._append({"progress_protocol": PROGRESS_PROTOCOL, "step": 1, "score": 0.4})
    resumed = ProgressTracker(ProgressConfig(), "pd", "run", tmp_path)
    resumed._append({"step": 2, "progress_protocol": PROGRESS_PROTOCOL, "new": 0.6, "score": 0.5})
    with resumed.path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["step"], r["score"]) for r in rows] == [("1", "0.4"), ("2", "0.5")]
    assert rows[1]["new"] == "0.6"


class _FakeRegressor:
    """Stands in for `TabICLRegressor` around the live weights: records the context it is fitted
    on and returns a fixed prediction."""

    def __init__(self, value: float):
        self.value, self.context = value, None

    def fit(self, X, y):
        self.context = np.asarray(y, dtype=float).copy()
        return self

    def predict(self, X, output_type=None, alphas=None):
        n = len(X)
        return {"mean": np.full(n, self.value), "quantiles": np.full((n, len(alphas)), self.value)}


def test_monitor_reports_nonfinite_prediction_as_error(tmp_path, monkeypatch):
    from src.train.progress import ProgressConfig, ProgressTracker

    tracker = ProgressTracker(ProgressConfig(context_rows=16, max_test_rows=16), "lgd", "run", tmp_path)
    monkeypatch.setattr(tracker, "_wrapper", lambda model, regression: _FakeRegressor(float("nan")))
    with pytest.raises(ValueError, match="Non-finite predictions"):
        tracker._score(torch.nn.Linear(1, 1), "unit/x", np.ones((40, 2)), np.linspace(0, 1, 40),
                       regression=True)


def test_ood_target_scaling_does_not_use_query_extremes(tmp_path, monkeypatch):
    from src.train.progress import ProgressConfig, ProgressTracker

    tracker = ProgressTracker(ProgressConfig(context_rows=16, max_test_rows=16), "lgd", "run", tmp_path)
    est = _FakeRegressor(0.5)
    monkeypatch.setattr(tracker, "_wrapper", lambda model, regression: est)
    y = np.linspace(0, 1, 40)
    ctx, query = tracker._split("unit/x", y, classification=False)
    y[query[0]] = 1000  # an extreme in the QUERY must not set the scale
    tracker._score(torch.nn.Linear(1, 1), "unit/x", np.ones((40, 2)), y, regression=True,
                   scale_target=True, bounded_target=False)
    assert float(est.context.min()) == 0 and float(est.context.max()) == 1


def _cells(n_folds=5, **extra):
    return pd.DataFrame([{"dataset": "a", "model": "m", "seed": 0, "fold": f, "status": "ok",
                          "protocol": 3, "context_cap": np.nan, "r2": 0.3, **extra} for f in range(n_folds)])


def test_benchmark_requires_every_fold_finite_uncapped_and_current():
    from src.eval.benchmark_status import validate_cells
    validate_cells(_cells(), ["a"], ["m"], [0], 5, "r2")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_cells(_cells().iloc[:4], ["a"], ["m"], [0], 5, "r2")
    df = _cells()
    df.loc[0, "r2"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_cells(df, ["a"], ["m"], [0], 5, "r2")
    with pytest.raises(ValueError, match="context cap"):
        validate_cells(_cells(context_cap=1024), ["a"], ["m"], [0], 5, "r2")
    with pytest.raises(ValueError, match="protocol"):
        validate_cells(_cells(protocol=2), ["a"], ["m"], [0], 5, "r2")


def test_benchmark_receipt_requires_both_domains_and_invalidates_changed_inputs(tmp_path, monkeypatch):
    from src.eval import benchmark_status as B
    files = {d: tmp_path / f"{d}.csv" for d in ("credit", "ood")}
    wanted = {"credit": ["a"], "ood": ["b"], "models": ["linear"], "seeds": [0], "n_folds": 5}
    monkeypatch.setattr(B, "request", lambda *a, **kw: ("ref", wanted, files))
    monkeypatch.setattr(B, "receipt_path", lambda tag: tmp_path / "receipt.json")
    for domain, file in files.items():
        pd.DataFrame([{"dataset": wanted[domain][0], "model": "linear", "seed": 0, "fold": f,
                       "status": "ok", "r2": 0.5, "protocol": 3} for f in range(5)]).to_csv(file, index=False)
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
    n_configs = len({r["_grid"]["tag"] for r in runs})  # Exp1: 3 credit mixes (was 15 priors)
    assert len(ranking) == n_configs
    assert set(ranking.training_seeds) == {3}
    assert ranking.iloc[0]["mean"] == pytest.approx(0.9)
    assert ranking.iloc[-1]["mean"] == pytest.approx(0.6)
    import matplotlib.pyplot as plt

    from src.visualize import results_plots
    monkeypatch.setattr(results_plots, "load_results", lambda *a: pd.DataFrame(rows))
    from matplotlib.collections import PathCollection

    fig = results_plots.overall_ranking("pd")
    points = [c for c in fig.axes[0].collections if isinstance(c, PathCollection)]
    assert sum(len(c.get_offsets()) for c in points) == n_configs  # one point per configuration
    plt.close(fig)


def test_benchmark_cannot_be_complete_with_missing_configured_data(monkeypatch):
    from types import SimpleNamespace

    from src.data import discovery
    from src.eval import benchmark_status as B
    from src.eval import ood
    monkeypatch.setattr(B, "load", lambda *a, **kw: {"eval": {"dev_datasets": ["a"], "holdout_datasets": ["b"]}})
    monkeypatch.setattr(B, "expand_with_seeds", lambda *a: [{}])
    monkeypatch.setattr(B, "slot_for", lambda *a, **kw: B.Slot("reference", "reference_pd_linear", model="linear"))
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
    monkeypatch.setattr(paths, "benchmark_dir", lambda *a: tmp_path)
    # An old 45-arm benchmark (no `protocol` column: protocol 2) left behind a45 and a74.
    for index in (45, 74):
        pd.DataFrame([{"dataset": "example", "model": f"arm-{index}", "roc_auc": 0.5,
                       "status": "ok"}]).to_csv(tmp_path / f"results_exp1bench_pd_a{index}.csv", index=False)
    old = results_plots.load_results("pd")
    assert sorted(old["model"]) == ["arm-45", "arm-74"] and set(old["protocol"]) == {2}
    # Once the current benchmark writes, it alone is read — never a mixture of the two.
    pd.DataFrame([{"dataset": "example", "model": "arm-0", "roc_auc": 0.5, "status": "ok", "fold": 0,
                   "protocol": 3}]).to_csv(tmp_path / "results_exp1bench_pd_a0.csv", index=False)
    pd.DataFrame([{"dataset": "example", "model": "arm-0", "roc_auc": 0.4, "status": "ok", "fold": 0,
                   "protocol": 3, "info_checkpoint_step": 2500}]).to_csv(
        tmp_path / "results_exp1bench_pd_a0_s2500.csv", index=False)
    result = results_plots.load_results("pd")
    assert result["model"].tolist() == ["arm-0"], "an earlier checkpoint is not a final result"
    assert set(result["protocol"]) == {3}


def test_ood_imputation_and_target_scaling_use_the_training_fold_only():
    from src.eval import ood_runner as O

    X = np.array([[1.0], [np.nan], [3.0], [10000.0], [2.0]])
    y = np.array([0.0, 0.5, 1.0, 0.25, 0.75])
    Xtr, Xte, ytr, yte, info = O._fold_arrays(X, y, np.array([0, 1, 2]), np.array([3, 4]), regression=True)
    assert Xtr[1, 0] == 2.0, "the training fold's median (of 1 and 3), untouched by the test fold"
    assert info["target_scaled_from"] == [0.0, 1.0]
    assert yte.tolist() == [0.25, 0.75]


def test_a_debug_context_cap_reaches_the_model(monkeypatch):
    from src.eval import runner

    seen = {}

    class Model:
        max_context_rows = None

        def fit(self, X, y, cat_indices=None):
            seen["cap"] = self.max_context_rows
            from src.eval.baselines import FitReport
            return FitReport(model="fake", n_train_used=len(X))

        def predict(self, X):
            return np.full(len(X), 0.5)

        def predict_quantiles(self, X):
            return None

    monkeypatch.setattr(runner, "build", lambda *a, **kw: Model())
    X = np.random.default_rng(0).normal(size=(50, 2))
    y = np.linspace(0.0, 1.0, 50)
    runner.score_fold("lgd", "fake", X[:40], y[:40], X[40:], y[40:], [], max_context_rows=1024)
    assert seen["cap"] == 1024

def test_prior_ranking_reads_protocol_three_fold_rows():
    """Protocol 3 writes one row per (dataset, fold) at fold seed 0. The ranking must count those
    cells — expecting protocol 2's evaluation seeds 0/1/2 would exclude every configuration."""
    from src.eval.protocol import N_FOLDS
    from src.eval.selection import ROOT, development_ranking
    from src.utils.config import expand_with_seeds, load, run_name
    cfg = load(ROOT / "config/Exp1_PD.yaml")
    rows = []
    for run in expand_with_seeds(cfg):
        credit = run["prior"]["credit_fraction"] > 0
        for ds in cfg["eval"]["dev_datasets"]:
            for fold in range(N_FOLDS):
                rows.append({"dataset": ds, "model": "crediticl", "info_run_name": run_name(run),
                             "seed": 0, "fold": fold, "status": "ok", "protocol": 3,
                             "roc_auc": 0.8 if credit else 0.7})
    ranking = development_ranking(pd.DataFrame(rows), "pd")
    assert len(ranking) == 3 and set(ranking.training_seeds) == {3}
    assert ranking.iloc[0]["mean"] == pytest.approx(0.8)
    # One fold missing makes that configuration incomplete, never quietly averaged.
    ranking = development_ranking(pd.DataFrame(rows[1:]), "pd")
    assert len(ranking) == 2


def test_the_learning_curve_draws_every_saved_checkpoint(tmp_path, monkeypatch):
    """One line per credit share over the saved checkpoints, the reference models as lines."""
    import matplotlib.pyplot as plt

    from src.utils import paths
    from src.visualize import results_plots
    monkeypatch.setattr(paths, "benchmark_dir", lambda *a: tmp_path)
    monkeypatch.setattr(results_plots, "_roles", lambda *a: {})
    for arm, share in enumerate(("0", "0p5")):
        for step, suffix in ((12500, ""), (2500, "_s2500")):
            rows = [{"dataset": d, "model": "crediticl", "fold": f, "seed": 0, "status": "ok",
                     "protocol": 3, "roc_auc": 0.6 + step / 1e5 + 0.01 * arm,
                     "info_checkpoint_step": step,
                     "info_run_name": f"exp1_pd__prior-credit_fraction={share}__s0"}
                    for d in ("a", "b") for f in range(5)]
            pd.DataFrame(rows).to_csv(tmp_path / f"results_exp1bench_pd_a{arm}{suffix}.csv", index=False)
    pd.DataFrame([{"dataset": d, "model": "catboost", "fold": f, "seed": 0, "status": "ok",
                   "protocol": 3, "roc_auc": 0.8} for d in ("a", "b") for f in range(5)]).to_csv(
        tmp_path / "results_reference_pd_catboost.csv", index=False)
    fig = results_plots.checkpoint_curve("pd", role=None)
    ax = fig.axes[0]
    curves = [ln for ln in ax.get_lines() if sorted(map(float, ln.get_xdata())) == [2500.0, 12500.0]]
    assert len(curves) == 2, "one line per credit share over the two saved checkpoints"
    assert any(abs(ln.get_ydata()[0] - 0.8) < 1e-9 for ln in ax.get_lines()), "the reference line"
    text = results_plots.results_summary("pd")
    assert "protocol 3" in text and "2,500:" in text
    plt.close(fig)


def test_training_curves_never_mix_two_sweeps(tmp_path, monkeypatch):
    """The redesigned Exp1's run names differ from the 45-arm sweep's, so both sets of manifests can
    sit in one folder. The curves read the current grid once it has output, the old one until then."""
    from src.utils import paths
    from src.utils.config import expand_with_seeds, load, run_name
    from src.visualize import training_plots as tp
    monkeypatch.setattr(paths, "outputs_dir", lambda: tmp_path)

    def write(name, frame):
        paths.run_dir(name).mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths.run_file(name, "progress"), index=False)

    old = "exp1_pd__prior-credit_fraction=0p5__prior-filter-mode=banded__s0"
    write(old, pd.DataFrame({"step": [1, 2], "train_loss": [0.5, 0.4]}))
    assert list(tp.load_progress("pd")) == [old]
    new = run_name(expand_with_seeds(load("config/Exp1_PD.yaml"))[0])
    write(new, pd.DataFrame({"step": [0, 500], "train_loss": [0.7, 0.3]}))
    assert list(tp.load_progress("pd")) == [new]
