"""Stage-3 task shapes, bounded batches and deterministic replay without GPU work."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.prior.dataset import PriorBatchDataset, build_loader
from src.prior.generator import TaskGenerator
from src.prior.rng import PriorRNG
from src.prior.stream import SlotPlan, _sample_seq_len, adjust_max_features, plan_batch


def test_log_uniform_rows_match_the_upstream_inverse_cdf_and_keep_small_tasks():
    """A uniform 400..60k draw would overexpose large tables by about sixfold."""
    rng = np.random.default_rng(901)
    reference = np.random.default_rng(901)
    expected = [int(np.exp(reference.uniform(np.log(400), np.log(60000)))) for _ in range(20000)]
    actual = [_sample_seq_len(rng, 400, 60000, "log_uniform") for _ in range(20000)]
    assert actual == expected
    assert min(actual) >= 400 and max(actual) < 60000
    assert np.median(actual) == pytest.approx(np.sqrt(400 * 60000), rel=.035)
    assert np.mean(np.array(actual) < 1024) == pytest.approx(np.log(1024 / 400) / np.log(150), abs=.015)


def test_default_row_sampler_preserves_existing_uniform_and_fixed_rng_draws():
    rng, expected = np.random.default_rng(123), np.random.default_rng(123)
    assert [_sample_seq_len(rng, 96, 128) for _ in range(100)] == expected.integers(96, 128, 100).tolist()
    for mode in ("uniform", "log_uniform"):
        rng, expected = np.random.default_rng(123), np.random.default_rng(123)
        assert _sample_seq_len(rng, 1024, 1024, mode) == 1024
        assert rng.random() == expected.random(), "a fixed size must not consume the context-split draw"


@pytest.mark.parametrize("lo,hi,mode", [(0, 1024, "log_uniform"), (128, 64, "uniform"),
                                      (2, 2, "uniform"), (400, 60000, "log-normal")])
def test_invalid_row_laws_fail_before_generation(lo, hi, mode):
    with pytest.raises(ValueError, match="prior.n_rows"):
        _sample_seq_len(np.random.default_rng(0), lo, hi, mode)


@pytest.mark.parametrize("rows,cap", [(10240, 100), (10241, 80), (20000, 80), (20001, 60),
                                     (30000, 60), (30001, 40), (40000, 40), (40001, 30),
                                     (50000, 30), (50001, 20), (60000, 20)])
def test_stage3_feature_limit_reaches_the_actual_slot(rows, cap):
    cfg = {"n_rows_range": [rows, rows], "n_rows_sampling": "log_uniform",
           "n_features_range": [100, 100], "max_features": 100,
           "grouping": {"group_size": 1}}
    slots = plan_batch(0, 0, cfg, 4, regression=False, credit_fraction=.5).slots
    assert all(s.num_features == cap and s.max_features == cap for s in slots)
    from tabicl.prior._dataset import Prior
    assert adjust_max_features(rows, 100) == Prior.adjust_max_features(rows, 100)


def test_long_context_shapes_stay_paired_across_mixtures():
    cfg = {"n_rows_range": [400, 60000], "n_rows_sampling": "log_uniform",
           "train_frac_range": [.79, .81], "grouping": {"group_size": 1}}
    plans = [plan_batch(2, 9, cfg, 64, regression=False, credit_fraction=f).slots for f in (0., .5, 1.)]
    shape = lambda p: [(s.seq_len, s.train_size, s.num_features, s.max_features) for s in p]
    assert shape(plans[0]) == shape(plans[1]) == shape(plans[2])
    assert len({s.seq_len for s in plans[0]}) > 50
    assert all(.79 * s.seq_len - 1 < s.train_size <= .81 * s.seq_len for s in plans[0])


def _assert_batches_equal(first, second):
    for i in (0, 1):
        assert len(first[i]) == len(second[i])
        for a, b in zip(first[i], second[i]):
            assert torch.equal(a, b)
    for i in (2, 3, 4):
        assert torch.equal(first[i], second[i])


@pytest.mark.parametrize("track", ["pd", "lgd"])
@pytest.mark.parametrize("rows", [[128, 128], [96, 160]])
def test_dense_and_ragged_training_tables_are_identical(pd_cfg, lgd_cfg, track, rows):
    cfg = copy.deepcopy((pd_cfg if track == "pd" else lgd_cfg)["prior"])
    cfg.update(n_rows_range=rows, n_rows_sampling="log_uniform", credit_fraction=.5,
               grouping={"group_size": 1, "subgroup_size": 1})
    dense = PriorBatchDataset(cfg, track, 4, seed=5)
    ragged_cfg = {**cfg, "batch_layout": "ragged"}
    ragged = PriorBatchDataset(ragged_cfg, track, 4, seed=5)
    a = dense.batch(TaskGenerator(cfg, track, PriorRNG(5)), 7)
    b = ragged.batch(TaskGenerator(ragged_cfg, track, PriorRNG(5)), 7)
    assert isinstance(a[0], torch.Tensor) and isinstance(b[0], list)
    for j in range(4):
        n, d = int(a[3][j]), int(a[2][j])
        assert torch.equal(a[0][j, :n, :d], b[0][j])
        assert torch.equal(a[1][j, :n], b[1][j])
    for i in (2, 3, 4):
        assert torch.equal(a[i], b[i])


def test_ragged_batch_does_not_pad_short_wide_tables_to_the_largest_shape(monkeypatch):
    from src.prior import dataset
    slots = [SlotPlan(0, 0, 0, 60000, 48000, 20, None, False, 20),
             SlotPlan(0, 1, 1, 400, 320, 100, None, False, 100)]
    monkeypatch.setattr(dataset, "plan_batch", lambda *a, **k: SimpleNamespace(
        slots=slots, seq_lens=[60000, 400], train_sizes=[48000, 320]))
    gen = SimpleNamespace(sample_slot=lambda s: SimpleNamespace(
        X=torch.zeros(s.seq_len, s.num_features), y=torch.zeros(s.seq_len), meta={"d": s.num_features}))
    X, y, d, lengths, splits = PriorBatchDataset({"batch_layout": "ragged"}, "lgd", 2, 0).batch(gen, 0)
    assert [tuple(t.shape) for t in X] == [(60000, 20), (400, 100)]
    assert sum(t.numel() for t in X) < .11 * (2 * 60000 * 100)
    assert [len(t) for t in y] == lengths.tolist() == [60000, 400]


def _assert_ragged_worker_replay(cfg):
    cfg.update(batch_layout="ragged", n_rows_sampling="log_uniform",
               credit_fraction=.5, grouping={"group_size": 1, "subgroup_size": 1})
    direct = build_loader(cfg, "lgd", 2, 31, num_workers=0)
    resumed = build_loader(cfg, "lgd", 2, 31, num_workers=0, start_batch=1)
    workers = build_loader(cfg, "lgd", 2, 31, num_workers=2, prefetch_factor=1, start_batch=1)
    iterator = iter(direct)
    next(iterator)
    expected = next(iterator)
    _assert_batches_equal(expected, next(iter(resumed)))
    worker_iterator = iter(workers)
    try:
        _assert_batches_equal(expected, next(worker_iterator))
    finally:
        worker_iterator._shutdown_workers()


def test_ragged_loader_resume_and_worker_count_preserve_every_value(lgd_cfg, tmp_path):
    # Upstream orders feature groups using string hashes. Setting the environment
    # inside this interpreter cannot change its hash seed. Start the probe under
    # the same fixed seed as pretrain.py/Exp0, including Windows-spawned workers.
    # Keep the exact tensor equality and stop/resume assertions in the probe.
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "prior.json"
    config.write_text(json.dumps(lgd_cfg["prior"]), encoding="utf-8")
    probe = tmp_path / "worker_probe.py"
    probe.write_text(
        "import json, runpy, sys\n"
        "from pathlib import Path\n"
        "if __name__ == '__main__':\n"
        f"    sys.path.insert(0, {str(root)!r})\n"
        f"    checks = runpy.run_path({str(Path(__file__).resolve())!r})\n"
        "    checks['_assert_ragged_worker_replay'](\n"
        "        json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))\n",
        encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(probe), str(config)], cwd=root,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kwargs", [{"num_workers": -1}, {"prefetch_factor": 0},
                                    {"prefetch_factor": 1.5}, {"prefetch_factor": True}])
def test_invalid_loader_queue_settings_fail_before_starting_workers(kwargs):
    with pytest.raises(ValueError):
        build_loader({}, "pd", 2, 0, **kwargs)


@pytest.mark.parametrize("cfg", [{"batch_layout": "unknown"},
                                 {"batch_layout": "ragged", "pool": {"source": "pool"}}])
def test_unknown_or_legacy_ragged_layout_is_rejected(cfg):
    with pytest.raises(ValueError, match="batch_layout"):
        PriorBatchDataset(cfg, "pd", 2, 0)
