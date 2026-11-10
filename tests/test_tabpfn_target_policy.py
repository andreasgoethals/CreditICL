"""Native target preprocessing, without released weights on the laptop."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.train.tabpfn_trainer import (
    TabPFNTaskPreparer,
    _ContextQuerySplit,
    configure_synthetic_targets,
)


def collection(y, train_size, *, seed=0):
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    from tabpfn.finetuning.data_util import (
        DatasetCollectionWithPreprocessing,
        RegressorDatasetConfig,
    )
    from tabpfn.preprocessing.configs import PreprocessorConfig
    from tabpfn.preprocessing.ensemble import generate_regression_ensemble_configs
    from tabpfn.preprocessing.steps.reshape_feature_distribution_step import (
        get_all_reshape_feature_distribution_preprocessors,
    )
    configs = generate_regression_ensemble_configs(
        num_estimators=2, add_fingerprint_feature=False, polynomial_features="no",
        feature_shift_decoder="shuffle", preprocessor_configs=[PreprocessorConfig("none")],
        target_transforms=[None, get_all_reshape_feature_distribution_preprocessors(len(y))["safepower"]],
        random_state=seed, num_models=1, outlier_removal_std=None,
    )
    X = np.random.default_rng(20).normal(size=(len(y), 4))
    config = RegressorDatasetConfig(config=configs, X_raw=X, y_raw=np.asarray(y, dtype=np.float64),
                                   cat_ix=[], znorm_space_bardist_=FullSupportBarDistribution(torch.linspace(-3, 3, 11)))
    return DatasetCollectionWithPreprocessing(_ContextQuerySplit(train_size), seed, [config])


@pytest.mark.parametrize("seed", [0, 1, 77])
def test_native_preparation_keeps_context_only_standardization_and_member_features(seed):
    y = np.r_[np.linspace(-2., 3., 30) ** 3, [-1e4, 1e4, 0.3]]
    original = collection(y, 30, seed=seed)
    repaired = copy.deepcopy(original)
    feature_configs = [vars(c).copy() for c in repaired.configs[0].config]
    configure_synthetic_targets(repaired, "standardize")
    native, fixed = original[0], repaired[0]
    for before, after in zip(feature_configs, repaired.configs[0].config):
        assert {k: v for k, v in vars(after).items() if k != "target_transform"} == {
            k: v for k, v in before.items() if k != "target_transform"}
    for a, b in zip(native.X_context + native.X_query, fixed.X_context + fixed.X_query):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    expected = torch.tensor((y - y[:30].mean()) / y[:30].std(), dtype=torch.float32)
    for context in fixed.y_context:
        torch.testing.assert_close(context, expected[:30], rtol=0, atol=0)
    torch.testing.assert_close(fixed.y_query, expected[30:], rtol=0, atol=0)
    assert all(c.target_transform is None for c in fixed.configs)
    np.testing.assert_array_equal(repaired.configs[0].y_raw, y)
    # Query targets must not influence fitted statistics or the context representation.
    changed = collection(np.r_[y[:30], [2e5, -2e5, -10]], 30, seed=seed)
    configure_synthetic_targets(changed, "standardize")
    for a, b in zip(changed[0].y_context, fixed.y_context):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_real_worker_entrypoint_removes_transform_before_dataset_materialization(monkeypatch):
    import tabpfn.finetuning.data_util as native
    ds = collection(np.arange(40.) ** 2, 30)
    monkeypatch.setattr(native, "get_preprocessed_dataset_chunks", lambda *a, **k: ds)
    prep = TabPFNTaskPreparer("lgd", "unused", 2, 0)
    prep._est = SimpleNamespace()
    result = prep.prepare(np.ones((40, 2)), np.arange(40.), 30, 12)
    assert all(c.target_transform is None for c in result["item"].configs)
    assert result["n_classes"] is None


def test_classification_does_not_enter_regression_policy(monkeypatch):
    import tabpfn.finetuning.data_util as native

    from src.train import tabpfn_trainer
    marker = object()
    monkeypatch.setattr(native, "get_preprocessed_dataset_chunks", lambda *a, **k: [marker])
    monkeypatch.setattr(tabpfn_trainer, "configure_synthetic_targets",
                        lambda *a: pytest.fail("Classification entered regression preprocessing"))
    prep = TabPFNTaskPreparer("pd", "unused", 2, 0)
    prep._est = SimpleNamespace(n_classes_=2)
    assert prep.prepare(np.ones((40, 2)), np.arange(40.) % 2, 30, 12) == {
        "item": marker, "n_classes": 2}


@pytest.mark.parametrize("policy", ["typo", None, False])
def test_invalid_policy_is_not_silently_native(policy):
    with pytest.raises(ValueError, match="target policy"):
        TabPFNTaskPreparer("lgd", "unused", 2, 0, target_policy=policy)


@pytest.mark.parametrize("constant", [False, True])
def test_small_context_variance_keeps_upstream_floor_without_clipping_query(constant):
    context = np.ones(30) + (0 if constant else np.linspace(0, 1e-10, 30))
    y = np.r_[context, [2., -1.]]
    ds = collection(y, 30)
    configure_synthetic_targets(ds, "standardize")
    with pytest.warns(UserWarning, match="constant or near-constant"):
        batch = ds[0]
    torch.testing.assert_close(batch.y_query, torch.tensor((y[30:] - context.mean()) / 1e-8).float())
    assert torch.isfinite(batch.y_query).all()  # not a bound on residual magnitude


def test_native_historical_control_is_explicit_and_unchanged():
    ds = collection(np.arange(40.) ** 2, 30)
    before = [c.target_transform for c in ds.configs[0].config]
    configure_synthetic_targets(ds, "native")
    assert [c.target_transform for c in ds.configs[0].config] == before
    assert any(c is not None for c in before)


@pytest.mark.parametrize("batch_id,slot_id", [(511, 1), (527, 9), (685, 15)])
def test_known_synthetic_tails_are_linear_after_native_preprocessing(batch_id, slot_id):
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG, one_thread
    from src.prior.stream import plan_batch
    from src.utils.config import expand_with_seeds, load
    run = next(r for r in expand_with_seeds(load("config/Exp2_LGD.yaml"))
               if r["architecture"] == "tabpfn3" and r["seed"] == 0
               and r["prior"]["credit_fraction"] == 0 and r["lr_index"] == 0)
    slot = plan_batch(0, batch_id, run["prior"], 16, regression=True, credit_fraction=0.).slots[slot_id]
    with one_thread():
        task = TaskGenerator(run["prior"], "lgd", PriorRNG(0), stream_seed=0).sample_slot(slot)
    y = task.y.numpy().astype(np.float64)
    ds = collection(y, slot.train_size)
    configure_synthetic_targets(ds, "standardize")
    batch = ds[0]
    expected = (y[slot.train_size:] - y[:slot.train_size].mean()) / max(y[:slot.train_size].std(), 1e-8)
    torch.testing.assert_close(batch.y_query, torch.tensor(expected).float(), rtol=0, atol=0)
    assert batch.y_query.abs().max() < 1000


def test_only_lgd_tabpfn_runs_change_identity_and_cannot_resume_native_targets():
    from src.train.checkpoint import validate_resume_config
    from src.utils.config import expand_with_seeds, load, run_name
    for exp in (2, 3):
        for task in ("PD", "LGD"):
            runs = expand_with_seeds(load(f"config/Exp{exp}_{task}.yaml"))
            for run in runs:
                affected = task == "LGD" and run["architecture"] == "tabpfn3"
                assert ("regression_target_policy" in run["train"]) == affected
                assert ("target-standardize-v1" in run_name(run)) == affected
                if affected:
                    old = copy.deepcopy(run)
                    old["train"].pop("regression_target_policy")
                    assert run_name(old) != run_name(run)
                    with pytest.raises(ValueError, match="regression_target_policy"):
                        validate_resume_config(old, run)


@pytest.mark.parametrize("experiment", [2, 3])
@pytest.mark.parametrize("task", ["PD", "LGD"])
def test_unchanged_completed_campaigns_remain_usable_but_never_resume(experiment, task):
    from src.train.checkpoint import validate_training_fingerprint
    from src.utils import completed_runs, preflight
    from src.utils.config import expand_with_seeds, load
    for run in expand_with_seeds(load(f"config/Exp{experiment}_{task}.yaml")):
        old = copy.deepcopy(run)
        old["train"].pop("regression_target_policy", None)
        summary = dict(completed=True, steps=1500, datasets_seen=1500*run["train"]["batch_size"],
                       training_fingerprint=preflight.training_fingerprint_for(completed_runs.REVIEWED_IMPLEMENTATION))
        if task == "LGD" and run["architecture"] == "tabpfn3":
            with pytest.raises(ValueError):
                completed_runs.validate_completed_training(summary, run, old)
        else:
            completed_runs.validate_completed_training(summary, run, old)
            changed = copy.deepcopy(run)
            changed["train"]["lr"] = .12345
            with pytest.raises(ValueError):
                completed_runs.validate_completed_training(summary, changed, changed)
            with pytest.raises(ValueError):
                validate_training_fingerprint(summary, run)


def assert_production_targets_and_update(directory, batch_id, rate):
    """Actual released GPU weights, full 16-task updates at both fixed peak rates."""
    import math

    from src.prior.dataset import build_loader
    from src.train.tabpfn_trainer import TabPFNTrainer
    from src.utils.config import expand_with_seeds, load
    run = next(r for r in expand_with_seeds(load("config/Exp2_LGD.yaml"))
               if r["architecture"] == "tabpfn3" and r["seed"] == 0
               and r["prior"]["credit_fraction"] == 0 and r["train"]["lr"] == rate)
    run["train"].update(num_workers=0, warmup_proportion=0.)
    run["progress"] = {"every_datasets": 0, "n_ood": 0}
    run["logging"].update(console=False, to_file=False, log_hardware_every=0, log_grad_every=0)
    trainer = TabPFNTrainer(run, directory, device="cuda")
    try:
        prep = TabPFNTaskPreparer("lgd", trainer.weights_path, 2, 0)
        items = next(iter(build_loader(run["prior"], "lgd", batch_size=16, seed=0,
                                      num_workers=0, start_batch=batch_id, transform=prep)))
        assert len(items) == 16
        for item in items:
            assert item and "error" not in item
            assert all(c.target_transform is None for c in item["item"].configs)
            assert torch.isfinite(item["item"].y_query).all()
            assert item["item"].y_query.abs().max() < 1000
        stats = trainer.train_step(items)
        assert math.isfinite(stats["loss"])
        assert trainer.datasets_seen == 16 and trainer.skipped_tasks == 0
        assert trainer.scheduler.last_epoch == 1
        assert trainer.full_precision_retries == 0
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    finally:
        trainer.close()
