"""Regression tests for the fresh full-table campaign, run in Exp0's CPU job."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("module", ["catboost", "tabpfn", "tabicl", "tabicl.prior._graph_scm",
                                     "openml", "transformers", "wandb", "xgboost", "pyarrow",
                                     "scipy", "sklearn", "pandas", "yaml", "filelock", "jupyter_client"])
def test_required_packages_import_without_optional_skips(module):
    importlib.import_module(module)


def test_declared_dependency_tree_is_actually_installed():
    from src.utils.environment import requirement_audit
    _, issues = requirement_audit()
    assert not issues, issues


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_catboost_real_categorical_fit_and_prediction(task):
    from src.eval.baselines import build
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 3)).astype(np.float32)
    X[:, 2] = rng.integers(-1, 4, len(X))
    X[0, 0] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(float) if task == "pd" else rng.random(len(X))
    model = build("catboost", task, seed=0, iterations=5, depth=2)
    report = model.fit(X[:70], y[:70], [2])
    pred = model.predict(X[70:])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    assert report.n_train_used == 70


@pytest.mark.parametrize("exp", [0, 1, 2, 3])
@pytest.mark.parametrize("track", ["PD", "LGD"])
def test_production_configs_never_request_row_or_context_caps(exp, track):
    from src.utils.config import load
    cfg = load(f"config/Exp{exp}_{track}.yaml", allow_placeholders=True)
    assert not cfg["eval"].get("context_caps")
    assert cfg["eval"].get("max_rows") is None
    assert cfg["eval"].get("max_context_rows") is None
    if exp != 3:
        from src.eval.benchmark_status import n_slots, slot_for
        assert all(slot_for(exp, track.lower(), i).context is None for i in range(n_slots(exp, track.lower())))


def test_full_million_rows_are_used_exactly_once_as_test_rows():
    from src.eval.protocol import cv_folds
    y = np.arange(1_000_000) % 2
    folds = cv_folds(y, "pd")
    counts = np.zeros(len(y), dtype=np.uint8)
    for train, test in folds:
        assert len(train) == 800_000 and len(test) == 200_000
        counts[test] += 1
        assert not np.intersect1d(train, test).size
    assert np.all(counts == 1)


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_repeated_entities_never_cross_outer_or_inner_split(task):
    from src.eval.protocol import cv_folds, validation_split
    groups = np.repeat(np.arange(40), 4)
    y = np.arange(len(groups)) % 2 if task == "pd" else np.linspace(0, 1, len(groups))
    seen = []
    for train, test in cv_folds(y, task, groups=groups):
        assert not set(groups[train]) & set(groups[test])
        fit, val = validation_split(y[train], task, groups=groups[train])
        assert not set(groups[train][fit]) & set(groups[train][val])
        seen.extend(test)
    assert sorted(seen) == list(range(len(y)))


def test_quantile_loss_preserves_fp32_levels_under_bfloat16():
    from src.train.loop import pinball_loss
    pred = torch.ones(2, 999, dtype=torch.bfloat16, requires_grad=True)
    target = torch.zeros(2)
    loss = pinball_loss(pred, target, 999)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(pred.grad).all() and loss.item() == pytest.approx(.5)


def test_equal_latents_have_equal_ranks_after_any_permutation():
    from src.prior.preprocess import to_ranks
    x = torch.tensor([1., 0., 1., 2., 1., 0.])
    order = torch.tensor([4, 5, 2, 0, 3, 1])
    assert torch.equal(to_ranks(x)[order], to_ranks(x[order]))
    assert torch.all(to_ranks(x)[x == 1] == .5833333)


def test_boundary_mass_comes_from_distribution_not_the_mean():
    from src.eval.metrics import boundary_mass_error, lgd_metrics
    y = np.tile([0., 1.], 500)
    q = np.tile([0., 1.], (len(y), 1))
    scores = lgd_metrics(y, np.full(len(y), .5), quantiles=q, levels=np.array([.25, .75]))
    assert scores["pred_mass_at_0"] == scores["pred_mass_at_1"] == .5
    assert scores["boundary_mass_abs_err"] == 0
    assert scores["pit_uniformity_error"] < .02
    assert "boundary_mass_abs_err" not in lgd_metrics(y, np.full(len(y), .5))
    with pytest.raises(ValueError, match="quantiles"):
        boundary_mass_error(y, np.full(len(y), .5))


def test_perfect_lgd_calibration_slope_is_exactly_one():
    from src.eval.metrics import lgd_metrics
    y = np.array([0., .3, .7, 1.])
    assert lgd_metrics(y, y)["calibration_slope"] == pytest.approx(1.)


@pytest.mark.parametrize("text,months", [("1yrs11mon", 23), ("1 yrs 11 mon", 23), ("0yrs0mon", 0)])
def test_credit_duration_parser(text, months):
    from src.data.dataset_preprocessing import parse_credit_age
    assert parse_credit_age(text) == months


def test_large_numeric_values_within_safe_reduction_range_are_not_flattened():
    import pandas as pd
    from src.data.dataset_preprocessing import _sanitize_numeric_values
    frame = pd.DataFrame({"x": [1e14, 2e14, np.inf]})
    result = _sanitize_numeric_values(frame, ["x"])
    assert result.x.iloc[1] == 2e14 and result.x.iloc[0] == 1e14
    assert np.isnan(result.x.iloc[2])


def test_missingness_coupling_keeps_requested_marginal_rate():
    from src.prior.targets.pd import apply_informative_missingness
    from src.prior.rng import PriorRNG
    y = (torch.arange(20000) % 10 == 0).float()
    X = torch.ones(len(y), 1)
    cfg = {"missing_rate_range": [.3, .3], "missing_col_fraction": 1.,
           "missing_target_coupling": 1., "missing_indicators": False}
    missing, _ = apply_informative_missingness(PriorRNG(1), X, y, cfg, 1, fill="nan")
    mask = torch.isnan(missing[:, 0])
    assert abs(mask.float().mean().item() - .3) < .02
    assert mask[y == 1].float().mean() > mask[y == 0].float().mean()


def test_cpu_gpu_validation_have_disjoint_expensive_stages():
    from src.utils.validation import plan
    cpu, gpu = plan("cpu"), plan("gpu")
    assert any(s.name == "unit" for s in cpu) and not any(s.name == "unit" for s in gpu)
    assert any(s.name == "vsc" for s in gpu) and not any(s.name == "vsc" for s in cpu)
    assert not any(s.phase == "training" for s in cpu)
    assert all("--max-context-rows" not in s.command for s in cpu + gpu)
    assert cpu[0].name == "preprocess"


def test_reset_output_reaches_both_tiers_but_preserves_inputs(isolated_output, monkeypatch):
    from src.utils import paths
    from src.utils.clean_run import reset_output
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("src.utils.run_experiment.queued_job_names", lambda: set())
    files = []
    for root in (paths.outputs_dir(), paths.big_outputs_dir()):
        file = root / "experiment_0/validation/old/checkpoints/arm/step-600.ckpt"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"old")
        files.append(file)
    released = paths.pretrained_dir() / "keep.ckpt"
    released.parent.mkdir(parents=True, exist_ok=True)
    released.write_bytes(b"reference")
    assert reset_output()[0] == len(set(files))
    assert all(p.exists() for p in files)
    reset_output(clean=True)
    assert all(not p.exists() for p in files) and released.read_bytes() == b"reference"


def test_reset_output_removes_readonly_empty_directories(isolated_output, monkeypatch):
    import stat
    from src.utils import paths
    from src.utils.clean_run import reset_output

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("src.utils.run_experiment.queued_job_names", lambda: set())
    folder = paths.outputs_dir() / "figures" / "old figure"
    folder.mkdir(parents=True)
    folder.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        reset_output(clean=True)
        assert not folder.exists()
    finally:
        if folder.exists():
            folder.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)


def test_old_ood_cache_cannot_satisfy_full_table_campaign(tmp_path, monkeypatch):
    from src.eval import ood
    monkeypatch.setattr(ood, "ood_root", lambda: tmp_path)
    (tmp_path / "ood_manifest.json").write_text(json.dumps({"version": 2, "datasets": [{"old": 1}]}))
    assert ood.list_ood_datasets() == []
    with pytest.raises(ValueError, match="full tables"):
        ood.fetch_ood_datasets(max_rows=50_000)


def test_ood_refresh_keeps_the_pinned_list_and_all_rows(tmp_path, monkeypatch):
    import sys
    import pandas as pd
    from types import SimpleNamespace
    from src.eval import ood
    monkeypatch.setattr(ood, "ood_root", lambda: tmp_path)
    entry = {"name": "full_table", "openml_id": 123, "kind": "regression",
             "n_rows": 50000, "n_features": 2, "n_classes": None}
    manifest = tmp_path / "ood_manifest.json"
    manifest.write_text(json.dumps({"version": 2, "datasets": [entry]}))
    ds = SimpleNamespace(name="full_table", dataset_id=123, default_target_attribute="target",
                         get_data=lambda **_: (pd.DataFrame({"x": np.arange(60003), "z": 1.}),
                                               pd.Series(np.arange(60003)), None, None))
    requests = []
    def get_dataset(identifier, **_):
        requests.append(identifier)
        return ds
    monkeypatch.setitem(sys.modules, "openml", SimpleNamespace(datasets=SimpleNamespace(get_dataset=get_dataset)))
    found = ood.fetch_ood_datasets()
    assert requests == [123] and len(found) == 1
    assert found[0].n_rows == found[0].source_rows == 60003
    X, y, _ = ood.load_ood_dataset(found[0])
    assert len(y) == len(X) == 60003 and y[-1] == 60002
    # A second invocation cannot empty or duplicate an already cached manifest.
    assert ood.fetch_ood_datasets() == found
    assert len(ood.list_ood_datasets()) == 1
    before = manifest.read_bytes()
    def unavailable(*_a, **_k):
        raise RuntimeError("download failed")
    ds.get_data = unavailable
    with pytest.raises(RuntimeError, match="manifest was preserved"):
        ood.fetch_ood_datasets(force=True)
    assert manifest.read_bytes() == before


def test_initial_ood_download_does_not_pin_an_incomplete_list(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.eval import ood
    monkeypatch.setattr(ood, "ood_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "openml", SimpleNamespace(study=SimpleNamespace(
        get_suite=lambda _: SimpleNamespace(tasks=[]))))
    with pytest.raises(RuntimeError, match="no partial list was pinned"):
        ood.fetch_ood_datasets(n_per_task=1)
    assert not ood.manifest_path().exists()


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_cache_publication_retries_windows_locks_without_deleting_old_data(tmp_path, monkeypatch, winerror):
    from src.eval import ood
    source, destination = tmp_path / "new.tmp", tmp_path / "manifest.json"
    source.write_text("new")
    destination.write_text("old")
    replace = Path.replace
    attempts = []
    def locked_once(path, target):
        assert destination.read_text() == "old"
        attempts.append(path)
        if len(attempts) == 1:
            error = PermissionError("locked")
            error.winerror = winerror
            raise error
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr(ood.time, "sleep", lambda _: None)
    ood._replace_cache(source, destination)
    assert len(attempts) == 2 and destination.read_text() == "new" and not source.exists()


@pytest.mark.parametrize("winerror,attempts_expected", [(5, 6), (None, 1)])
def test_cache_publication_preserves_old_manifest_when_replacement_is_refused(
        tmp_path, monkeypatch, winerror, attempts_expected):
    from src.eval import ood
    source, destination = tmp_path / "new.tmp", tmp_path / "manifest.json"
    source.write_text("new")
    destination.write_text("old")
    attempts = []
    def denied(path, target):
        attempts.append(path)
        error = PermissionError("denied")
        if winerror is not None:
            error.winerror = winerror
        raise error
    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(ood.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        ood._replace_cache(source, destination)
    assert len(attempts) == attempts_expected
    assert source.read_text() == "new" and destination.read_text() == "old"


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_scratch_initialization_and_head_freezing_on_real_architecture(task):
    from src.models.architecture import build_model
    from src.train.adapt import apply_freezing
    model = build_model(task, embed_dim=32, col_num_blocks=1, col_num_inds=8,
                        row_num_blocks=1, row_num_cls=2, icl_num_blocks=1,
                        col_nhead=4, row_nhead=4, icl_nhead=4)
    projections = [p for name, p in model.named_parameters() if "out_proj.weight" in name]
    assert projections and all(torch.count_nonzero(p) > 0 for p in projections)
    apply_freezing(model, "head_only")
    decoder = dict(model.named_parameters())["icl_predictor.decoder.2.weight"]
    assert decoder.requires_grad
    assert not dict(model.named_parameters())["row_interactor.tf_row.blocks.0.attn.out_proj.weight"].requires_grad
    loss = model.icl_predictor.decoder(torch.ones(2, decoder.shape[1] // 2)).square().mean()
    loss.backward()
    assert decoder.grad is not None and torch.count_nonzero(decoder.grad) > 0


def test_upstream_categorical_correction_actually_expands_cardinality():
    from src.prior.upstream import prior_config, seeded
    from tabicl.prior.graph_lib._base import Context
    from tabicl.prior.graph_lib._properties import sample_categorical_sizes
    maximum = {}
    for corrected in (False, True):
        with seeded(27):
            context = Context(config=prior_config({"use_corrected_cat_sizes": corrected}), device="cpu")
            sizes = [n for _ in range(40) for n in sample_categorical_sizes(100, context, max_cat_size=500)]
        maximum[corrected] = max(sizes)
    assert maximum[False] <= 9 < maximum[True] <= 500


def test_decision_base_rate_is_never_inferred_from_test_labels():
    from src.eval.metrics import pd_metrics, threshold_metrics
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .2, .3, .8])
    assert "f1_at_base_rate" not in pd_metrics(y, p)
    result = pd_metrics(y, p, decision_base_rate=.2)
    assert result["f1_at_base_rate"] == threshold_metrics(y, p, .2)["f1"]


def test_full_reset_refuses_active_jobs_and_unexpected_roots(isolated_output, monkeypatch):
    from src.utils import paths
    from src.utils.clean_run import reset_output
    root = paths.outputs_dir()
    root.mkdir(parents=True, exist_ok=True)
    file = root / "keep-until-idle.log"
    file.write_text("active")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="login node"):
        reset_output(clean=True)
    assert file.exists()
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setattr(paths, "outputs_dir", lambda: root.parent)
    with pytest.raises(RuntimeError, match="Unexpected"):
        reset_output(clean=True)
