"""Long synthetic tables preserve upstream padding and every requested row."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.prior import upstream
from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG, one_thread
from src.prior.stream import SlotPlan


def _plan(*, task="pd", rows=60_000, credit=False, width=20):
    return SlotPlan(batch=0, slot=0, group=0, seq_len=rows,
                    train_size=int(rows * 0.8), num_features=5,
                    num_classes=None if task == "lgd" else (2 if credit else 3),
                    use_credit=credit, max_features=width)


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_long_control_filter_sees_upstream_group_padding(pd_cfg, lgd_cfg, monkeypatch, task):
    """ExtraTrees' input width matters even when the added columns are constant."""
    cfg = copy.deepcopy((pd_cfg if task == "pd" else lgd_cfg)["prior"])
    cfg["max_features"] = 100
    cfg["filter"] = {"mode": "tabicl", "apply_to": "base"}
    seen = {}

    def graph(**kwargs):
        seen["graph_width"] = kwargs["max_features"]
        rows = kwargs["seq_len"]
        x = torch.zeros(rows, kwargs["max_features"])
        x[:, :5] = torch.arange(rows, dtype=torch.float32)[:, None]
        return x, torch.arange(rows, dtype=torch.float32) % 3

    def accept(x, y, **kwargs):
        seen["filter_shape"] = x.shape
        return True

    monkeypatch.setattr(upstream, "sample_control", graph)
    gen = TaskGenerator(cfg, task, PriorRNG(12))
    monkeypatch.setattr(gen.filter, "accept", accept)
    with one_thread():
        sample = gen.sample_slot(_plan(task=task))
    assert seen == {"graph_width": 20, "filter_shape": torch.Size([60_000, 20])}
    assert sample.X.shape == (60_000, 20)
    assert sample.meta["d"] == 5


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_exp1_slot_padding_default_remains_identical(pd_cfg, lgd_cfg, task):
    """Adding stage-3 padding must not change existing 1,024-row task streams."""
    cfg = copy.deepcopy((pd_cfg if task == "pd" else lgd_cfg)["prior"])
    cfg["max_features"] = 100
    plan = _plan(task=task, rows=1024, width=100)
    with one_thread():
        gen = TaskGenerator(cfg, task, PriorRNG(12))
        explicit = gen.sample_slot(plan)
        inherited = gen.sample_slot(replace(plan, max_features=None))
    assert torch.equal(explicit.X, inherited.X)
    assert torch.equal(explicit.y, inherited.y)


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_actual_long_control_passes_the_original_prior_filter(pd_cfg, lgd_cfg, task):
    cfg = copy.deepcopy((pd_cfg if task == "pd" else lgd_cfg)["prior"])
    cfg["max_features"] = 100
    cfg["max_filter_attempts"] = 40
    cfg["filter"] = {"mode": "tabicl", "apply_to": "base"}
    with one_thread():
        sample = TaskGenerator(cfg, task, PriorRNG(12)).sample_slot(_plan(task=task))
    assert sample.X.shape == (60_000, 20)
    assert sample.y.shape == (60_000,)
    assert torch.isfinite(sample.X).all() and torch.isfinite(sample.y).all()
    assert sample.meta["base_impl"] == "upstream.GraphSCM"
    assert sample.meta["filter_attempts"] >= 1


@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("shift", ["cohort", "covariate", "prior_prob", "selection"])
def test_long_credit_table_preserves_shape_split_and_signal(pd_cfg, lgd_cfg, task, shift):
    cfg = copy.deepcopy((pd_cfg if task == "pd" else lgd_cfg)["prior"])
    cfg["max_features"] = 100
    cfg["credit_fraction"] = 1.0
    cfg["credit"]["shift"] = {"shift_prob": 1.0, "kind_weights": {shift: 1.0}}
    cfg["credit"]["missingness"]["missing_prob"] = 1.0
    plan = _plan(task=task, credit=True)
    with one_thread():
        sample = TaskGenerator(cfg, task, PriorRNG(12)).sample_slot(plan)
    assert sample.X.shape == (60_000, 20)
    assert sample.y.shape == (60_000,)
    assert 1 <= sample.meta["d"] <= plan.num_features
    assert torch.isfinite(sample.X).all()
    assert torch.isfinite(sample.y).all()
    assert sample.meta["shift"] == shift
    assert sample.meta["shift_cut"] == plan.train_size
    if task == "pd":
        assert set(sample.y[:plan.train_size].unique().tolist()) == {0.0, 1.0}
        assert set(sample.y[plan.train_size:].unique().tolist()) == {0.0, 1.0}
    else:
        assert abs(float(sample.y[:plan.train_size].mean())) < 1e-5
        assert float(sample.y[:plan.train_size].std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)


def test_invalid_slot_width_cannot_silently_discard_features(pd_cfg):
    cfg = pd_cfg["prior"]
    gen = TaskGenerator(cfg, "pd", PriorRNG(12))
    with pytest.raises(ValueError, match="Slot feature count"):
        gen.sample_slot(_plan(rows=400, width=4))


def test_standalone_draws_follow_the_training_row_law(pd_cfg):
    cfg = copy.deepcopy(pd_cfg["prior"])
    cfg.update(n_rows_range=[400, 60_000], n_rows_sampling="log_uniform",
               n_features_range=[1, 100], max_features=100)
    gen = TaskGenerator(cfg, "pd", PriorRNG(12))
    shapes = np.asarray([gen.sample_shape() for _ in range(2000)])
    assert np.all((400 <= shapes[:, 0]) & (shapes[:, 0] < 60_000))
    # A uniform sampler would put its median near30k, not the geometric midpoint.
    assert 4000 < np.median(shapes[:, 0]) < 6000
    assert shapes[shapes[:, 0] > 50_000, 1].max() <= 20
