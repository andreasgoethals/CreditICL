"""Capacity diagnostics must cover the real long-context path without shrinking it."""
import copy
import sys

import pytest
import torch

from src.utils.config import expand_with_seeds, load
from src.validation import campaign
from src.validation import long_context as lc


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_full_batch_probe_keeps_production_tables_and_model_recipe(track):
    production = next(c for c in expand_with_seeds(load(f"config/Exp2_{track.upper()}.yaml"))
                      if c["arm"] == "tabicl" and c["seed"] == 0
                      and c["prior"]["credit_fraction"] == .5)
    probe = lc.diagnostic_config(track)
    assert probe["prior"] == production["prior"]
    assert probe["init"] == production["init"]
    assert probe["architecture"] == "tabicl"
    assert probe["train"]["batch_size"] == production["train"]["batch_size"] == 64
    for key in ("optimizer", "lr", "weight_decay", "gradient_clipping", "amp",
                "micro_batch_size", "activation_checkpointing", "require_efficient_attention"):
        assert probe["train"][key] == production["train"][key]
    assert probe["train"]["num_workers"] == 4
    assert probe["train"]["prefetch_factor"] == 1
    assert probe["train"]["warmup_proportion"] == 0  # explicit nonzero-update diagnostic


@pytest.mark.parametrize("source,fraction", [("base", 0.), ("credit", 1.)])
@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_corners_fix_shape_without_replacing_the_synthetic_mechanisms(track, source, fraction):
    production = lc.diagnostic_config(track)
    for rows, width in lc.CAPACITY_CORNERS:
        probe = lc.diagnostic_config(track, rows=rows, source=source)
        expected = copy.deepcopy(production["prior"])
        expected.update(n_rows_range=[rows, rows], n_features_range=[width, width],
                        credit_fraction=fraction)
        assert probe["prior"] == expected
        assert probe["train"]["batch_size"] == 1
        assert probe["train"]["num_workers"] == 0
        assert probe["init"]["strategy"] == "full"
        assert probe["init"]["strict_load"] is True


def test_mandatory_gpu_plan_covers_every_corner_and_source_in_separate_bounded_processes():
    stages = [s for s in campaign.plan("gpu") if s.name.startswith("long-context-")]
    assert len(stages) == 2 * (2 * len(lc.CAPACITY_CORNERS) + 1)
    assert len({s.command for s in stages}) == len(stages)
    for track in ("pd", "lgd"):
        for rows, _ in lc.CAPACITY_CORNERS:
            for source in ("base", "credit"):
                stage = next(s for s in stages if s.name == f"long-context-{track}-{source}-{rows}")
                assert stage.command[stage.command.index("--rows") + 1] == str(rows)
                assert stage.command[stage.command.index("--source") + 1] == source
        assert any(s.name == f"long-context-{track}-mixed-batch" for s in stages)
    assert all(s.phase == "capacity" and s.max_seconds == 7200 and s.report for s in stages)
    assert not any(s.name.startswith("long-context-") for s in campaign.plan("cpu"))


def test_capacity_padding_retains_all_generated_values_and_targets():
    X = torch.arange(400 * 73, dtype=torch.float32).reshape(400, 73)
    y = torch.arange(400, dtype=torch.float32)
    batch = ([X], [y], torch.tensor([73]), torch.tensor([400]), torch.tensor([320]))
    padded = lc.capacity_batch(batch, rows=400, width=100)
    assert padded[0][0].shape == (400, 100)
    assert torch.equal(padded[0][0][:, :73], X)
    assert torch.count_nonzero(padded[0][0][:, 73:]) == 0
    assert padded[1][0] is y
    assert padded[2].tolist() == [100]
    assert padded[3] is batch[3] and padded[4] is batch[4]
    assert batch[2].tolist() == [73]  # original metadata remains available in the report


@pytest.mark.parametrize("rows,width", [(399, 100), (400, 72)])
def test_capacity_probe_refuses_row_or_feature_truncation(rows, width):
    batch = ([torch.ones(400, 73)], [torch.ones(400)], torch.tensor([73]),
             torch.tensor([400]), torch.tensor([320]))
    with pytest.raises(ValueError):
        lc.capacity_batch(batch, rows=rows, width=width)


def test_gpu_validation_refuses_cpu_before_creating_work_files(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="CUDA allocation"):
        lc.run_probe("pd", output, rows=60000, source="base")
    assert not output.exists()


def test_slow_capacity_failure_is_not_relabelled_as_an_allocation_deferral(tmp_path):
    stage = campaign.Stage("long-context-test", (sys.executable, "-c", "import time; time.sleep(30)"),
                           "capacity", max_seconds=0.05)
    with pytest.raises(RuntimeError, match="capacity guard"):
        campaign.execute(stage, tmp_path / "capacity.log", timeout=20)
