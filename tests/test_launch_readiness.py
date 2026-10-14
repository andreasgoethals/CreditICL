"""Regressions from the 12 October failed Exp0 job and the compute-safety audit."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.utils import preflight


def test_old_tabpfn_is_rejected_before_initializing_an_estimator(monkeypatch):
    from src.train.tabpfn_trainer import build_estimator

    monkeypatch.setattr(preflight.metadata, "version", lambda _: "8.2.0")
    with pytest.raises(RuntimeError, match=r"requires tabpfn==9.1.0; found 8.2.0"):
        build_estimator("pd", "does-not-exist.ckpt", n_estimators=2, device="cpu", seed=0)


def test_tabicl_version_string_alone_cannot_identify_the_prior(monkeypatch):
    def distribution(name):
        if name == "tabpfn":
            return SimpleNamespace(version=preflight.TABPFN_VERSION)
        return SimpleNamespace(version="2.1.1", read_text=lambda _: None)

    monkeypatch.setattr(preflight.metadata, "distribution", distribution)
    rows = preflight.dependency_checks()
    assert rows[0].ok and not rows[1].ok


def test_missing_catboost_blocks_dependency_preflight(monkeypatch):
    original = preflight.metadata.distribution

    def distribution(name):
        if name == "catboost":
            raise preflight.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(preflight.metadata, "distribution", distribution)
    row = next(r for r in preflight.dependency_checks() if r.name == "catboost")
    assert not row.ok
    assert preflight.main(["--dependencies-only"]) == 1


def test_failed_preflight_cannot_submit_an_experiment(monkeypatch):
    from src.utils import run_experiment as runner

    monkeypatch.setattr(runner, "unconfigured_tracks", lambda *a: [])
    monkeypatch.setattr(runner, "plan", lambda *a: [
        runner.Stage("exp1-pd-train", 1, "pd", "train", total=9, commands=[["sbatch", "never"]])])
    monkeypatch.setattr(preflight, "main", lambda _: 1)
    calls = []
    monkeypatch.setattr(runner.subprocess, "call", lambda *a, **kw: calls.append(a))
    assert runner.main(["1", "--submit"]) == 1
    assert calls == []


@pytest.mark.parametrize("stale", [True, False])
def test_preprocessing_rebuilds_only_stale_or_incomplete_caches(tmp_path, monkeypatch, stale):
    from src.data import pipeline as p

    (tmp_path / "meta.json").write_text(json.dumps({"cache_version": 3 if stale else p.CACHE_VERSION}))
    (tmp_path / "data.parquet").write_bytes(b"old cache")
    monkeypatch.setattr(p, "find_processed_dir", lambda *a: tmp_path)
    monkeypatch.setattr(p, "find_raw_file", lambda *a: tmp_path / "raw.csv")
    monkeypatch.setattr(p, "processed_write_dir", lambda *a: tmp_path)
    seen = []

    def recipe(*args):
        seen.append(args)
        return pd.DataFrame({"x": [0., 1., 2.], "y": [0., 1., 0.]}), "y", [], ["x"]

    monkeypatch.setattr(p, "preprocess_dataset_specific", recipe)
    p.preprocess_one("pd", "fake")
    assert bool(seen) == stale
    assert preflight.check_cache(tmp_path, p.CACHE_VERSION)[0]
    if stale:
        assert len(pd.read_parquet(tmp_path / "data.parquet")) == 3


def test_legacy_ood_manifest_is_reused_without_moving_files(tmp_path, monkeypatch):
    from src.utils import paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "staging_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "datasets_dir", lambda: tmp_path / "data")
    legacy = tmp_path / "prior_cache" / "ood"
    legacy.mkdir(parents=True)
    (legacy / "ood_manifest.json").write_text('{"version":2,"datasets":[]}')
    assert paths.ood_cache_dir() == legacy
    preferred = tmp_path / "data" / "ood"
    preferred.mkdir(parents=True)
    (preferred / "ood_manifest.json").write_text('{"version":2,"datasets":[]}')
    assert paths.ood_cache_dir() == preferred
    assert (legacy / "ood_manifest.json").is_file()


def test_default_tabpfn3_weights_never_select_an_older_generation(tmp_path, monkeypatch):
    from src.eval.baselines import find_local_tabpfn_checkpoint

    monkeypatch.setenv("CREDITICL_TABPFN_DIR", str(tmp_path))
    for name in ("tabpfn-v2-classifier-default.ckpt", "tabpfn-v3-classifier-v3_default.ckpt",
                 "tabpfn-v3-classifier-other.ckpt"):
        (tmp_path / name).write_bytes(b"weights")
    assert find_local_tabpfn_checkpoint("classifier").name == "tabpfn-v3-classifier-v3_default.ckpt"


@pytest.mark.parametrize("prepared", [[], [None, None], [{"error": "bad preprocessing"}]])
def test_tabpfn_cannot_count_an_empty_or_failed_batch_as_a_training_step(prepared):
    from src.train.tabpfn_trainer import TabPFNTrainer

    trainer = TabPFNTrainer.__new__(TabPFNTrainer)
    trainer.step = 42
    with pytest.raises(RuntimeError, match="step 42"):
        trainer.train_step(prepared)


def test_full_context_preserves_one_million_rows_including_extremes():
    from src.eval.baselines import _TFMBaseline
    from src.eval.protocol import cv_folds, fit_fold_preprocessing

    n = 1_000_000
    X = np.arange(n, dtype=np.float32).reshape(-1, 1)
    X[-1, 0] = 1e12
    y = (np.arange(n) % 2).astype(np.float32)
    prep, Xfit, yfit = fit_fold_preprocessing(X, y, [])
    assert len(Xfit) == len(yfit) == n
    assert prep.report["outlier_rows_removed"] == 0
    for task in ("pd", "lgd"):
        model = _TFMBaseline(task)
        Xgiven, ygiven = model._maybe_subsample(Xfit, yfit)
        assert len(Xgiven) == len(ygiven) == n
    folds = cv_folds(y, "pd")
    np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(n))
    assert all(len(tr) == 800_000 and len(te) == 200_000 for tr, te in folds)


def test_ood_linear_predictions_can_extrapolate_outside_training_target_range():
    from src.eval.baselines import LinearBaseline

    X = np.linspace(0, 1, 200).reshape(-1, 1)
    y = X[:, 0]
    raw = LinearBaseline("lgd", bounded_target=False)
    raw.fit(X, y)
    pred = raw.predict(np.array([[-1.], [2.]]))
    assert pred[0] < 0 and pred[1] > 1
    bounded = LinearBaseline("lgd")
    bounded.fit(X, y)
    np.testing.assert_array_equal(bounded.predict(np.array([[-1.], [2.]])), [0., 1.])


def test_prediction_features_remove_outcomes_and_remap_categories_without_changing_rows():
    from src.eval.feature_policy import prediction_features

    names = ["income", "recoveries", "purpose", "recoveries__cat", "last_week_pay"]
    X = np.arange(30).reshape(6, 5)
    original = X.copy()
    filtered, cats, report = prediction_features("pd", "0006.hackerearth", X, [2, 3], names)
    np.testing.assert_array_equal(filtered, X[:, [0, 2]])
    np.testing.assert_array_equal(X, original)
    assert cats == [1]
    assert json.loads(report["excluded_features"]) == ["recoveries", "recoveries__cat", "last_week_pay"]


def test_lgd_target_components_are_not_benchmark_features(monkeypatch):
    from src.eval import runner

    ds = SimpleNamespace(X=np.arange(30).reshape(10, 3), y=np.ones(10), cat_indices=[],
                         feature_names=["numerateur", "denominateur", "EAD_TOTAL_EURO"],
                         n_rows=10, n_features=3)
    monkeypatch.setattr(runner, "load_processed", lambda *a: ds)
    X, y, cats, info = runner.load_arrays("lgd", "0005.base_modelisation")
    assert X.shape == (10, 1) and len(y) == 10 and cats == []
    assert info["n_features_excluded"] == 2
    np.testing.assert_array_equal(X[:, 0], ds.X[:, 2])


def test_rebuilt_project_cache_is_not_shadowed_by_a_repo_copy(tmp_path, monkeypatch):
    from src.utils import paths

    old, new = tmp_path / "repo" / "fake", tmp_path / "project" / "fake"
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
    monkeypatch.setattr(paths, "processed_task_dirs", lambda *a: [old.parent, new.parent])
    monkeypatch.setattr(paths, "processed_write_dir", lambda *a: new)
    assert paths.find_processed_dir("pd", "fake") == new


@pytest.mark.parametrize("change", ["seed", "prior", "lr", "gradient_clipping"])
def test_resume_rejects_a_changed_training_recipe(change):
    from copy import deepcopy
    from src.train.checkpoint import validate_resume_config

    old = {"seed": 0, "prior": {"credit_fraction": .5}, "train": {"lr": .001, "gradient_clipping": 10}}
    new = deepcopy(old)
    if change in old:
        new[change] = 1 if change == "seed" else {"credit_fraction": 1.0}
    else:
        new["train"][change] *= 2
    with pytest.raises(ValueError, match="Resume configuration CHANGED"):
        validate_resume_config(old, new)


def test_partial_or_stale_exp0_checks_cannot_unlock_paid_training(tmp_path, monkeypatch):
    from src.utils import paths
    from src.utils.environment import environment_fingerprint

    root = tmp_path / "experiment_0"
    (root / "report").mkdir(parents=True)
    (root / "logs").mkdir()
    monkeypatch.setattr(paths, "experiment_dir", lambda *a: root)
    monkeypatch.setattr(preflight, "source_fingerprint", lambda: "current")
    (root / "report" / "checks.json").write_text(json.dumps({
        "source_fingerprint": "current", "rows": [{"check": "env", "status": "PASS"}]}))
    (root / "logs" / "exp0_checks_20261012.log").write_text(
        "equivalence PD exit code: 0\nequivalence LGD exit code: 0\nEND status=OK\n")
    assert not preflight.experiment_zero_checks(include_runs=False)[0].ok
    checks = [dict(check=c, status="PASS") for c in (
        "env", "paths", "storage", "data", "ood", "weights", "prior", "determinism",
        "configs", "orchestration", "train", "benchmark")]
    report = {"source_fingerprint": "old", "rows": checks,
              "environment_fingerprint": environment_fingerprint()}
    file = root / "report" / "checks.json"
    file.write_text(json.dumps(report))
    assert not preflight.experiment_zero_checks(include_runs=False)[0].ok
    report["source_fingerprint"] = "current"
    file.write_text(json.dumps(report))
    assert all(c.ok for c in preflight.experiment_zero_checks(include_runs=False))
    # Preserving a historical failed log must not poison a later successful check.
    (root / "logs" / "exp0_checks_20261001.log").write_text("END status=FAILED\n")
    assert all(c.ok for c in preflight.experiment_zero_checks(include_runs=False))


def test_nonfinite_training_loss_cannot_update_weights(tmp_path, lgd_cfg, monkeypatch):
    import torch
    from src.train.loop import Trainer

    trainer = Trainer(lgd_cfg, tmp_path / "run", device="cpu", ckpt_dir=tmp_path / "ck")
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    monkeypatch.setattr(trainer, "_loss_for", lambda *a: (torch.tensor(float("nan")), {}))
    try:
        with pytest.raises(FloatingPointError, match="Non-finite training loss"):
            trainer.train_step((torch.randn(2, 32, 4), torch.rand(2, 32), 16))
        assert trainer.step == 0
        assert all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items())
    finally:
        trainer.close()


def test_oom_discards_the_entire_update_instead_of_counting_partial_work(tmp_path, lgd_cfg, monkeypatch):
    import torch
    from src.train.loop import Trainer

    lgd_cfg["train"]["micro_batch_size"] = 1
    trainer = Trainer(lgd_cfg, tmp_path / "run", device="cpu", ckpt_dir=tmp_path / "ck")
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    param = next(p for p in trainer.model.parameters() if p.requires_grad)
    calls = []

    def loss(*args):
        calls.append(1)
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("simulated exhausted device")
        return param.square().sum(), {}

    monkeypatch.setattr(trainer, "_loss_for", loss)
    try:
        with pytest.raises(RuntimeError, match="incomplete update was discarded"):
            trainer.train_step((torch.randn(2, 32, 4), torch.rand(2, 32), 16))
        assert trainer.step == 0 and len(calls) == 2
        assert all(p.grad is None for p in trainer.model.parameters())
        assert all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items())
    finally:
        trainer.close()


@pytest.mark.parametrize("classification", [True, False])
def test_quality_filter_matches_pinned_upstream_on_the_same_padded_tables(classification):
    import torch
    from tabicl.prior._dataset import should_filter
    from src.prior.filters import PredictabilityFilter

    settings = SimpleNamespace(remove_trivial_datasets=False, filter_unpredictable_datasets=True)
    rng = np.random.default_rng(19)
    for noise in (0., .2, 20.):
        X = torch.from_numpy(rng.normal(size=(128, 6)).astype(np.float32))
        y = X[:, 0] + torch.from_numpy((noise * rng.normal(size=128)).astype(np.float32))
        if classification:
            y = (y > 0).float()
        X = torch.cat((X, torch.zeros(128, 26)), dim=1)
        ours = PredictabilityFilter("tabicl").accept(X, y, is_classif=classification)
        assert ours == (not should_filter(X, y, settings, is_classif=classification))


def test_input_digest_changes_when_a_benchmark_input_changes(tmp_path):
    from src.eval.benchmark_status import input_digest

    file = tmp_path / "data.parquet"
    file.write_bytes(b"old")
    before = input_digest(file)
    file.write_bytes(b"new input")
    assert input_digest(file) != before


def test_current_benchmark_rejects_a_summary_from_different_training_code(tmp_path, monkeypatch):
    from src.data import discovery
    from src.eval import benchmark_status as benchmark, ood
    from src.utils import paths

    run = {"task": "lgd", "seed": 0, "prior": {}, "train": {"max_steps": 10},
           "eval": {"dev_datasets": ["credit"], "holdout_datasets": []}}
    monkeypatch.setattr(benchmark, "load", lambda *a, **kw: run)
    monkeypatch.setattr(benchmark, "expand_with_seeds", lambda cfg: [cfg])
    monkeypatch.setattr(benchmark, "run_name", lambda cfg: "exp1_lgd_test")
    monkeypatch.setattr(benchmark, "slot_for", lambda *a: benchmark.Slot(
        "arm", "exp1bench_lgd_a0", arm=0, step=10, final=True, model="crediticl"))
    monkeypatch.setattr(discovery, "list_datasets", lambda task: ["credit"])
    monkeypatch.setattr(ood, "N_PER_TASK", 1)
    monkeypatch.setattr(ood, "list_ood_datasets", lambda kind: [SimpleNamespace(
        name="ood", slug="ood", n_rows=125, cache_file=None)])
    monkeypatch.setattr(ood, "manifest_path", lambda: tmp_path / "manifest.json")
    monkeypatch.setattr(ood, "ood_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "find_processed_dir", lambda *a: tmp_path)
    monkeypatch.setattr(paths, "run_config_path", lambda name: tmp_path / "config.json")
    monkeypatch.setattr(paths, "run_summary_path", lambda name: tmp_path / "summary.json")
    monkeypatch.setattr(paths, "run_checkpoints_dir", lambda name: tmp_path)
    monkeypatch.setattr(paths, "benchmark_dir_for", lambda *a: tmp_path)
    for filename in ("manifest.json", "meta.json", "data.parquet", "ood.npz", "step-10.ckpt"):
        (tmp_path / filename).write_bytes(b"immutable input")
    (tmp_path / "meta.json").write_text(json.dumps({"n_rows": 125}), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(run), encoding="utf-8")
    summary = {"completed": True, "steps": 10, "training_fingerprint": preflight.training_fingerprint()}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(benchmark, "package_fingerprint", lambda: "validated packages")
    first_request = benchmark.request(1, "lgd", 0)[1]
    assert first_request["step"] == 10
    assert first_request["credit_rows"] == {"credit": 125}
    assert first_request["ood_rows"] == {"ood": 125}
    monkeypatch.setattr(benchmark, "package_fingerprint", lambda: "changed packages")
    assert benchmark.request(1, "lgd", 0)[1] != first_request
    for fingerprint in (None, "different implementation"):
        summary["training_fingerprint"] = fingerprint
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match="training code differs"):
            benchmark.request(1, "lgd", 0)


def test_resume_rejects_unknown_or_changed_training_implementation():
    from src.train.checkpoint import validate_training_fingerprint
    from src.utils.preflight import training_fingerprint

    for saved in ({}, {"training_fingerprint": "different implementation"}):
        with pytest.raises(ValueError, match="training code differs"):
            validate_training_fingerprint(saved)
    validate_training_fingerprint({"training_fingerprint": training_fingerprint()})


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_catboost_uses_native_categories_including_missing_and_unseen_values(task):
    pytest.importorskip("catboost")
    from src.eval.baselines import CatBoostBaseline

    rng = np.random.default_rng(42)
    X = np.column_stack([rng.normal(size=80), np.tile([0., 1., 2., -1.], 20)])
    y = (X[:, 1] == 1).astype(float)
    before = X.copy()
    model = CatBoostBaseline(task, iterations=3, depth=2)
    report = model.fit(X, y, [1])
    assert model._model.get_cat_feature_indices() == [1]
    assert report.n_train_used == len(X) and report.extra["categorical_features"] == 1
    np.testing.assert_array_equal(X, before)
    predictions = model.predict(np.array([[0., 99.], [np.nan, -1.], [1., np.nan]]))
    assert predictions.shape == (3,) and np.isfinite(predictions).all()


def test_missing_tabicl_reference_cannot_fall_through_to_an_online_download(monkeypatch):
    from src.eval.baselines import TabICLBaseline
    from src.utils import paths

    def missing(*args):
        raise FileNotFoundError("test has no weights")

    monkeypatch.setattr(paths, "find_pretrained", missing)
    with pytest.raises(FileNotFoundError, match="Missing local released TabICLv2 weights"):
        TabICLBaseline("pd")._wrapper_kwargs()


def test_tabpfn_regression_loss_uses_each_members_transformed_targets(monkeypatch):
    import torch
    from tabpfn.finetuning import finetuned_regressor as upstream
    from src.train.tabpfn_trainer import regression_loss

    class ShiftTarget:
        def transform(self, y):
            return y + 10

    y = torch.tensor([[0., 1., 2.]])
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    batch = SimpleNamespace(X_query=None, y_query=y,
                            znorm_space_bardist=FullSupportBarDistribution(torch.linspace(-3, 3, 5)),
                            configs=[[SimpleNamespace(target_transform=ShiftTarget())],
                                     [SimpleNamespace(target_transform=None)]])
    est = SimpleNamespace(forward=lambda _: (None, [torch.zeros(3, 1, 4), torch.zeros(3, 1, 4)], None))
    seen = {}

    def loss(**kw):
        seen.update(kw)
        return torch.tensor(0.)

    monkeypatch.setattr(upstream, "_compute_regression_loss", loss)
    regression_loss(est, batch, 2, dict(ce=0., crps=1., crls=0., mse=1., mae=0.))
    torch.testing.assert_close(seen["targets_BQ"], torch.tensor([[10., 11., 12.], [0., 1., 2.]]))
