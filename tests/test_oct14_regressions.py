"""Behavioural regressions from the 14 October VSC output and independent audit."""
import json
import logging
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


def test_float32_reductions_remain_finite_on_extreme_numeric_inputs():
    from src.data.dataset_preprocessing import _sanitize_numeric_values
    frame = pd.DataFrame({"x": np.resize([1e41, -9e41, 1e15, np.nan], 100_000)})
    frame = _sanitize_numeric_values(frame, ["x"])
    x = frame.x.to_numpy(dtype=np.float32)
    assert np.isfinite(np.nanmean(x)) and np.isfinite(np.nanstd(x))


@pytest.mark.parametrize("case", ["large_columns", "percentages", "employment_sentinel", "score_scale"])
def test_source_recipes_preserve_rows_and_numeric_meaning(monkeypatch, case):
    from src.data import dataset_preprocessing as recipes
    if case == "large_columns":
        task, name = "pd", "0011.loan_default"
        raw = pd.DataFrame({"f391": [-9e41, 0., 7e41], "f627": [5e41, 2e20, 8e30], "loss": [0, 1, 2]})
    elif case == "percentages":
        task, name = "lgd", "0007.lgd_lendingclub"
        raw = pd.DataFrame({"lgd": [.1, .2, .3], "int_rate": ["9%", "20%", "11%"],
                            "revol_util": ["5.5%", "40%", None]})
    elif case == "employment_sentinel":
        task, name = "pd", "0012.home_credit"
        raw = pd.DataFrame({"TARGET": [0, 1, 0], "DAYS_EMPLOYED": [365243, -700, -2000]})
    else:
        task, name = "pd", "0009.bank_status"
        raw = pd.DataFrame({"Loan Status": ["Fully Paid", None, "Charged Off"],
                            "Credit Score": [710, None, 7100]}, index=[4, 8, 12])
    monkeypatch.setattr(recipes, "_load_dataset_file", lambda path: raw.copy())
    frame, _, cat, num = recipes.preprocess_dataset_specific(task, name, raw_dir=Path("unused"))
    if case == "large_columns":
        assert len(frame) == 3 and {"f391", "f627"} <= set(num)
        assert frame.f391.is_monotonic_increasing
        assert np.isfinite(frame[["f391", "f627"]].to_numpy(dtype=np.float32).std(axis=0)).all()
    elif case == "percentages":
        assert len(frame) == 3 and {"int_rate", "revol_util"} <= set(num)
        assert not {"int_rate", "revol_util"} & set(cat)
        assert frame.int_rate.tolist() == [9., 20., 11.]
        assert pd.isna(frame.revol_util.iloc[2])
    elif case == "employment_sentinel":
        assert len(frame) == 3 and np.isnan(frame.DAYS_EMPLOYED.iloc[0])
        assert frame.DAYS_EMPLOYED.iloc[1:].tolist() == [-700, -2000]
    else:
        assert frame.index.tolist() == [4, 12]  # source IDs still align with the group sidecar
        assert frame["Credit Score"].tolist() == [710., 710.]


@pytest.mark.parametrize("rho", [.03, .12, .24, .30])
def test_removing_macro_factor_preserves_marginal_default_rate(rho):
    from src.prior.rng import PriorRNG
    from src.prior.targets.mechanisms import pd_vasicek
    latent = torch.linspace(-3, 3, 20_000)
    y, meta = pd_vasicek(PriorRNG(0), latent, {"rho_range": [rho, rho],
        "base_rate_range": [.05, .05], "macro_scale": 0})
    assert float(y.mean()) == pytest.approx(.05, abs=.001)
    assert meta["target_base_rate"] == .05


def test_approved_population_keeps_its_original_outcomes():
    from src.prior.rng import PriorRNG
    from src.prior.targets.pd import apply_pd_target
    latent = torch.linspace(-3, 3, 4000)
    X = latent[:, None]
    common = {"mode": "mechanism", "mechanism": {"rho_range": [.1, .1],
        "base_rate_range": [.1, .1], "macro_scale": 0, "signal_share_range": [1., 1.]}}
    _, all_y, _ = apply_pd_target(PriorRNG(4), X, latent, common, 1)
    _, accepted_y, _ = apply_pd_target(PriorRNG(4), X, latent,
        {**common, "selection": {"selection_drop": .5, "selection_sharpness": 1.}}, 1)
    assert len(accepted_y) == 2000
    assert accepted_y.mean() < all_y.mean() / 4


def test_monitoring_keeps_entities_disjoint(tmp_path):
    from src.train.progress import ProgressConfig, ProgressTracker
    tracker = ProgressTracker(ProgressConfig(context_rows=80, max_test_rows=50), "pd", "test", tmp_path)
    groups = np.repeat(np.arange(60), 4)
    y = np.arange(len(groups)) % 2
    context, test = tracker._split("repeated borrowers", y, True, groups)
    assert not set(groups[context]) & set(groups[test])
    assert len(context) <= 80 and len(test) <= 50
    assert len(np.unique(y[context])) == len(np.unique(y[test])) == 2


def test_model_width_is_measured_after_fit():
    from src.eval.baselines import Baseline
    model = Baseline("pd")
    model._fit = lambda X, y, cat: setattr(model, "_model", SimpleNamespace(n_features_in_=2))
    with pytest.raises(ValueError, match="fitted 2 of 3"):
        model.fit(np.ones((20, 3)), np.arange(20) % 2)


def test_empty_selection_has_no_winner():
    from src.eval.selection import development_ranking
    assert development_ranking(None, "pd").empty


@pytest.mark.parametrize("kind", ["SIGTERM", "SIGUSR1"])
def test_duplicate_signals_keep_cancellation_priority(monkeypatch, kind):
    from src.train.loop import Trainer
    from src.train.tabpfn_trainer import TabPFNTrainer
    for cls in (Trainer, TabPFNTrainer):
        handlers = {}
        monkeypatch.setattr(signal, "signal", lambda number, fn, handlers=handlers: handlers.setdefault(number, fn))
        trainer = object.__new__(cls)
        trainer._stop_requested = None
        trainer.step = 1
        trainer.log = logging.getLogger("signal-test")
        trainer._install_stop_signals()
        term = signal.SIGTERM
        usr = getattr(signal, "SIGUSR1", term)
        handlers[term](term, None)
        handlers[term if kind == "SIGTERM" else usr](term if kind == "SIGTERM" else usr, None)
        assert trainer._stop_requested == "SIGTERM"


@pytest.mark.parametrize("patch,segment", [({"stopped_by_signal": "SIGTERM"}, 0),
    ({"steps": 40, "resumed_at": 40}, 0), ({"completed": True}, 0), ({}, 32)])
def test_no_cancelled_empty_or_unbounded_auto_continuation(patch, segment):
    from src.utils.resume_guard import require_progress
    summary = {"completed": False, "stopped_by_signal": "SIGUSR1", "steps": 60, "resumed_at": 40}
    with pytest.raises(ValueError):
        require_progress({**summary, **patch}, segment)
    require_progress(summary, 0)


@pytest.mark.parametrize("resumed_at", [None, 0, 40])
def test_first_and_later_walltime_stops_can_continue(resumed_at):
    from src.utils.resume_guard import require_progress

    # Both trainers write JSON null before their first checkpoint resume.
    require_progress({"completed": False, "stopped_by_signal": "SIGUSR1",
                      "steps": 60, "resumed_at": resumed_at}, 0)
    with pytest.raises(ValueError, match="No automatic continuation"):
        require_progress({"completed": False, "stopped_by_signal": "SIGUSR1",
                          "steps": resumed_at or 0, "resumed_at": resumed_at}, 0)


def test_linked_cleanup_refuses_before_deleting_other_files(tmp_path, monkeypatch):
    from src.utils import clean_run
    root = tmp_path / "output"
    root.mkdir()
    ordinary = root / "keep-until-validated.txt"
    ordinary.write_text("evidence")
    link = root / "junction"
    link.mkdir()
    monkeypatch.setattr(clean_run, "_linked", lambda p: p == link)
    with pytest.raises(RuntimeError, match="linked"):
        clean_run.wipe(root, [])
    assert ordinary.read_text() == "evidence"


def test_unwritable_checkpoint_storage_never_falls_back(tmp_path, monkeypatch):
    from src.utils.paths import resolve_writable
    preferred, fallback = tmp_path / "staging", tmp_path / "small"
    preferred.mkdir()
    original = Path.write_text
    def denied(path, *args, **kwargs):
        if path.parent == preferred:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", denied)
    with pytest.raises(OSError, match="no fallback"):
        resolve_writable(preferred, fallback)
    assert not fallback.exists()


def test_gpfs_working_checkpoints_publish_only_at_safe_boundaries(tmp_path, monkeypatch):
    from src.utils.working_storage import checkpoint_work_dir, publish_checkpoint
    permanent = tmp_path / "permanent"
    permanent.mkdir()
    (permanent / "step-10.ckpt").write_bytes(b"resume")
    monkeypatch.setenv("CREDITICL_WORK_ROOT", str(tmp_path / "scratch"))
    work = checkpoint_work_dir(permanent)
    assert work != permanent and (work / "step-10.ckpt").read_bytes() == b"resume"
    for step in (11, 20, 21, 22):
        (work / f"step-{step}.ckpt").write_bytes(str(step).encode())
        publish_checkpoint(work, step, permanent_every=10, final=step == 22, stopped=step == 21)
    assert not (permanent / "step-11.ckpt").exists()
    assert all((permanent / f"step-{s}.ckpt").is_file() for s in (10, 20, 21, 22))


def test_failed_atomic_copy_preserves_previous_complete_checkpoint(tmp_path, monkeypatch):
    from src.utils import working_storage as storage
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"old complete")
    def fail(source, temporary):
        temporary.write_bytes(b"partial")
        raise OSError("quota")
    monkeypatch.setattr(storage.shutil, "copy2", fail)
    with pytest.raises(OSError, match="quota"):
        storage.atomic_copy(source, target)
    assert target.read_bytes() == b"old complete"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source", "target"]


def test_staged_input_is_reused_and_invalidated_on_change(tmp_path, monkeypatch):
    from src.utils import working_storage as storage
    source = tmp_path / "data.parquet"
    source.write_bytes(b"one")
    monkeypatch.setenv("CREDITICL_WORK_ROOT", str(tmp_path / "scratch"))
    first = storage.local_input(source)
    assert first != source and first.read_bytes() == b"one"
    source.write_bytes(b"updated")
    assert storage.local_input(source).read_bytes() == b"updated"


def test_mindwell_does_not_silently_use_project_storage(monkeypatch):
    from src.utils.working_storage import work_root
    for name in ("CREDITICL_WORK_ROOT", "VSC_SCRATCH", "VSC_SCRATCH_GPFS1"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="GPFS"):
        work_root()


@pytest.mark.parametrize("published", ["absent", "corrupt", "valid"])
def test_success_cleanup_never_loses_an_unpublished_checkpoint(tmp_path, monkeypatch, published):
    from src.utils import working_storage as storage
    monkeypatch.delenv("CREDITICL_WORK_ROOT", raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path / "scratch"))
    permanent = tmp_path / "permanent"
    work = storage.checkpoint_work_dir(permanent)
    (work / "step-10.ckpt").write_bytes(b"trained state")
    if published != "absent":
        storage.publish_checkpoint(work, 10, permanent_every=10, final=True, stopped=False)
    if published == "corrupt":
        (permanent / "step-10.ckpt").write_bytes(b"broken copy")
    if published == "valid":
        storage.cleanup_successful_job()
        assert not storage.work_root().exists()
        assert (permanent / "step-10.ckpt").read_bytes() == b"trained state"
    else:
        with pytest.raises(RuntimeError, match="not safely published"):
            storage.cleanup_successful_job()
        assert (work / "step-10.ckpt").read_bytes() == b"trained state"


def test_success_cleanup_leaves_explicit_work_root_untouched(tmp_path, monkeypatch):
    from src.utils.working_storage import cleanup_successful_job
    monkeypatch.setenv("CREDITICL_WORK_ROOT", str(tmp_path))
    (tmp_path / "keep").write_bytes(b"explicit external path")
    cleanup_successful_job()
    assert (tmp_path / "keep").is_file()


def test_ood_generation_paths_cannot_escape_cache(tmp_path, monkeypatch):
    from src.eval import ood
    monkeypatch.setattr(ood, "ood_root", lambda: tmp_path)
    entry = ood.OODDataset("test", 1, "regression", 50, 2, cache_file="../other.npz")
    with pytest.raises(ValueError, match="within"):
        ood.cache_path(entry)


def test_benchmark_csv_publication_is_atomic(tmp_path, monkeypatch):
    from src.utils.atomic_output import write_csv
    target = tmp_path / "results.csv"
    target.write_text("previous")
    frame = pd.DataFrame({"metric": [1.]})
    def fail(path, **kwargs):
        path.write_text("partial")
        raise OSError("quota")
    monkeypatch.setattr(frame, "to_csv", fail)
    with pytest.raises(OSError):
        write_csv(frame, target)
    assert target.read_text() == "previous"
    assert len(list(tmp_path.iterdir())) == 1


def test_failed_forced_ood_refresh_preserves_all_previous_tables(tmp_path, monkeypatch):
    import sys
    from dataclasses import asdict

    from src.eval import ood
    monkeypatch.setattr(ood, "ood_root", lambda: tmp_path)
    entries = [ood.OODDataset(f"example_{i}", i, "regression", 60, 2) for i in (1, 2)]
    for entry in entries:
        np.savez_compressed(ood.cache_path(entry), X=np.ones((60, 2)), y=np.arange(60), cat_indices=[])
    manifest = {"version": ood.OOD_VERSION, "datasets": [asdict(e) for e in entries]}
    ood.manifest_path().write_text(json.dumps(manifest), encoding="utf-8")
    before = {p: p.read_bytes() for p in [ood.manifest_path(), *map(ood.cache_path, entries)]}
    def get_dataset(identity, **kwargs):
        if identity == 2:
            raise OSError("download failed")
        return SimpleNamespace(name="example_1", dataset_id=1, default_target_attribute="target",
            get_data=lambda **kw: (pd.DataFrame(np.zeros((60, 2))), pd.Series(np.arange(60)), None, None))
    monkeypatch.setitem(sys.modules, "openml", SimpleNamespace(datasets=SimpleNamespace(get_dataset=get_dataset)))
    with pytest.raises(RuntimeError, match="incomplete"):
        ood.fetch_ood_datasets(force=True, n_per_task=1)
    assert {p: p.read_bytes() for p in before} == before
    assert np.array_equal(ood.load_ood_dataset(entries[0])[0], np.ones((60, 2)))


def test_reference_and_continued_tabpfn_use_explicit_equal_preprocessing(tmp_path):
    from src.eval.baselines import full_feature_tabpfn_config
    first, second = tmp_path / "released.ckpt", tmp_path / "continued.ckpt"
    transforms = [{"name": "quantile_uni_coarse", "categorical_name": "ordinal_shuffled",
                   "append_original": False, "subsample_features": -1, "max_features_per_estimator": 500}]
    for path, setting in ((first, True), (second, False)):
        torch.save({"inference_config": {"ENABLE_GPU_PREPROCESSING": setting,
                    "PREPROCESS_TRANSFORMS": transforms}}, path)
    assert full_feature_tabpfn_config(first, 600) == full_feature_tabpfn_config(second, 600)
    assert full_feature_tabpfn_config(second, 600)["ENABLE_GPU_PREPROCESSING"] is True


def test_repeated_entities_and_future_fields_are_covered():
    from src.data.pipeline import GROUP_SOURCES
    from src.eval.feature_policy import prediction_features
    assert GROUP_SOURCES["pd", "0009.bank_status"] == "Customer ID"
    assert GROUP_SOURCES["lgd", "0005.base_modelisation"] == "spm"
    names = ["spm", "PY1_unemployment", "DT_DECHEANCE_TERME", "topbaloispur_cor", "income_at_default"]
    X, cats, _ = prediction_features("lgd", "0005.base_modelisation", np.ones((20, 5)), [], names)
    assert X.shape == (20, 1) and not cats
