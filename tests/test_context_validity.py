"""Native TabPFN context validity, matched slot budgets, and the production batch-37 failure."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.prior.generator import SyntheticTask, TaskGenerator
from src.prior.rng import PriorRNG
from src.prior.stream import SlotPlan, plan_batch, slot_seed
from src.utils.config import expand_with_seeds, load


def generator(encoding="raw", task="pd"):
    return TaskGenerator({"encoding": encoding, "base": {"implementation": "transcribed"},
                          "filter": {"mode": "off"}}, task, PriorRNG(0), stream_seed=0)


def context_task(X, task="pd", shift="covariate"):
    y = torch.arange(len(X)).float() if task == "lgd" else torch.arange(len(X)).remainder(2).float()
    return SyntheticTask(X=X.clone(), y=y, source="credit",
                         meta={"d": X.shape[1], "shift": shift})


@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("missing", [False, True])
def test_query_variation_cannot_make_an_empty_native_context_valid(task, missing):
    X = torch.zeros(8, 2)
    X[4:] = torch.arange(8).reshape(4, 2).float()
    if missing:
        X[:4] = float("nan")
    plan = SlotPlan(0, 0, 0, 8, 4, 2, None if task == "lgd" else 2, True)
    gen = generator(task=task)
    assert not gen._valid(context_task(X, task), plan)
    assert gen.constant_context_rejections == 1


@pytest.mark.parametrize("values", [[0., 0., 0., 0.], [float("nan")] * 4,
                                    [0., 1., 0., 0.], [0., float("nan"), 0., 0.],
                                    [float("nan"), 0., float("nan"), 0.]])
def test_context_selection_matches_native_constant_feature_step(values):
    # Keep native NaN semantics: informative missingness is not discarded as constant.
    from tabpfn.errors import TabPFNValidationError
    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.steps.remove_constant_features_step import RemoveConstantFeaturesStep

    a = np.array(values, dtype=np.float32).reshape(-1, 1)
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=1)
    try:
        RemoveConstantFeaturesStep()._fit(a, schema)
        expected = True
    except TabPFNValidationError:
        expected = False
    X = torch.tensor(np.concatenate([a, [[3.], [4.], [5.], [6.]]]))
    assert generator()._usable_context(context_task(X), SlotPlan(0, 0, 0, 8, 4, 1, 2, True)) == expected


def test_context_check_runs_after_class_balance_row_repair(monkeypatch):
    gen = generator()
    X = torch.arange(8.).reshape(-1, 1)
    task = context_task(X, shift="none")
    task.y = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.])
    calls = []
    monkeypatch.setattr(gen, "_usable_context", lambda candidate, plan: calls.append(candidate.X.clone()) or False)
    assert not gen._valid(task, SlotPlan(0, 0, 0, 8, 4, 1, 2, True))
    assert len(calls) == 1


def test_rejection_retains_source_shape_and_budget_without_moving_other_slots(monkeypatch):
    gen = generator()
    plan = SlotPlan(37, 15, 3, 8, 4, 2, 2, True)
    seeds = []

    def candidate(slot, rng, seed):
        seeds.append(seed)
        X = torch.zeros(8, 2)
        X[4:] = 1
        if len(seeds) > 1:
            X[1, 0] = 2
        return context_task(X)

    monkeypatch.setattr(gen, "_candidate", candidate)
    task = gen.sample_slot(plan)
    assert seeds == [slot_seed(0, 37, 15, "credit", k) for k in (0, 1)]
    assert task.X.shape == (8, 2) and task.source == "credit"
    assert task.meta["invalid_attempts"] == gen.invalid_candidates == 1
    assert gen.filter_summary()["constant_context_rejections"] == 1
    assert plan.train_size == 4 and plan.slot == 15


def test_persistently_invalid_context_fails_at_the_existing_bounded_budget(monkeypatch):
    gen = generator()
    gen.max_invalid_attempts = 3
    X = torch.zeros(8, 1)
    X[4:] = 1
    monkeypatch.setattr(gen, "_candidate", lambda *args: context_task(X))
    with pytest.raises(RuntimeError, match="3 invalid tasks"):
        gen.sample_slot(SlotPlan(0, 0, 0, 8, 4, 1, 2, True))
    assert gen.constant_context_rejections == 3


def test_encoded_exp1_stream_keeps_the_preexisting_validity_rule():
    X = torch.zeros(8, 1)
    X[4:] = 1
    assert generator("tabicl")._valid(context_task(X), SlotPlan(0, 0, 0, 8, 4, 1, 2, True))


def test_actual_production_slot_is_redrawn_without_changing_either_learning_rate_stream():
    runs = expand_with_seeds(load("config/Exp2_PD.yaml"))
    run = runs[11]
    assert run["prior"] == runs[15]["prior"] and run["seed"] == runs[15]["seed"]
    plan = plan_batch(0, 37, run["prior"], 16, regression=False, credit_fraction=1.).slots[15]
    gen = TaskGenerator(run["prior"], "pd", PriorRNG(0), stream_seed=0)
    original = gen._candidate(plan, PriorRNG(slot_seed(0, 37, 15, "credit")),
                              slot_seed(0, 37, 15, "credit"))
    assert original is not None and not gen._usable_context(original, plan)
    assert original.X.shape[0] == 1024 and plan.train_size == 823
    fixed = gen.sample_slot(plan)
    repeated = TaskGenerator(run["prior"], "pd", PriorRNG(0), stream_seed=0).sample_slot(plan)
    assert gen._usable_context(fixed, plan) and fixed.meta["invalid_attempts"] >= 1
    torch.testing.assert_close(fixed.X, repeated.X, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(fixed.y, repeated.y, rtol=0, atol=0)


def assert_native_cuda_update_accepts_the_full_repaired_production_batch(tmp_path, index):
    from src.prior.dataset import PriorBatchDataset
    from src.train.tabpfn_trainer import TabPFNTaskPreparer, TabPFNTrainer
    from src.utils.paths import find_pretrained

    run = copy.deepcopy(expand_with_seeds(load("config/Exp2_PD.yaml"))[index])
    run["init"]["pretrained_path"] = str(find_pretrained(run["init"]["pretrained_path"]))
    run["progress"]["every_datasets"] = 0
    run["logging"] = {"console": False, "to_file": False, "log_hardware_every": 0,
                      "log_grad_every": 0, "log_weights_every": 0}
    batch = next(iter(PriorBatchDataset(run["prior"], "pd", 16, 0, start_batch=37)))
    prepare = TabPFNTaskPreparer("pd", run["init"]["pretrained_path"], 2, 0)
    prepared = prepare(batch, 37)
    assert len(prepared) == 16 and all(item is not None and "item" in item for item in prepared)
    trainer = TabPFNTrainer(run, tmp_path / "repair", device="cuda")
    try:
        stats = trainer.train_step(prepared)
        assert np.isfinite(stats["loss"]) and trainer.datasets_seen == 16
        assert trainer.skipped_tasks == 0
        assert all(bool(torch.isfinite(p).all()) for p in trainer.model.parameters())
    finally:
        trainer.close()
