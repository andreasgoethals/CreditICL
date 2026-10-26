"""Extend the completed campaign without reassigning its runs or benchmark files."""
import copy
import json

import pytest

from src.utils.config import expand_with_seeds, load


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_extension_preserves_old_run_recipes_and_benchmark_tags(track, monkeypatch):
    from src.eval import benchmark_status as bench
    from src.train.checkpoint import validate_resume_config

    cfg = load(f"config/Exp1_{track.upper()}.yaml")
    original = copy.deepcopy(cfg)
    original.pop("array_append", None)
    original["prior"]["credit_fraction"] = [0., .5, 1.]
    old_runs, new_runs = expand_with_seeds(original), expand_with_seeds(cfg)
    assert len(new_runs) == 12
    for before, after in zip(old_runs, new_runs[:9], strict=True):
        assert {k: v for k, v in before.items() if k != "_grid"} == {
            k: v for k, v in after.items() if k != "_grid"}
        assert before["_grid"]["array_index"] == after["_grid"]["array_index"]
        assert before["_grid"]["assignments"] == after["_grid"]["assignments"]
        validate_resume_config(before, after)
    assert {(r["prior"]["credit_fraction"], r["seed"]) for r in new_runs[9:]} == {
        (.25, seed) for seed in (0, 1, 2)}

    new_slots = [bench.slot_for(1, track, i) for i in range(bench.n_slots(1, track))]
    monkeypatch.setattr(bench, "load", lambda *a, **k: original)
    old_slots = [bench.slot_for(1, track, i) for i in range(bench.n_slots(1, track))]
    assert {s.tag: s for s in old_slots}.items() <= {s.tag: s for s in new_slots}.items()
    additions = [s for s in new_slots if s.tag not in {v.tag for v in old_slots}]
    assert len(additions) == 15
    assert {s.arm for s in additions} == {9, 10, 11}


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_completed_original_campaign_submits_only_the_three_new_arms(track, isolated_output, monkeypatch):
    from src.utils import paths, run_experiment
    from src.utils.completed_runs import HISTORICAL_IMPLEMENTATION
    from src.utils.preflight import training_fingerprint_for

    monkeypatch.setattr(run_experiment, "queued_job_names", set)
    runs = expand_with_seeds(load(f"config/Exp1_{track.upper()}.yaml"))
    for run in runs[:9]:
        summary = paths.run_summary_path(run["_run_name"])
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"steps": 12500, "completed": True,
                                       "datasets_seen": 800000,
                                       "training_fingerprint": training_fingerprint_for(HISTORICAL_IMPLEMENTATION)}))
        paths.run_config_path(run["_run_name"]).write_text(json.dumps(run))
    stage = next(s for s in run_experiment.plan(1, [track], single_cluster=True)
                 if s.name == f"exp1-{track}-train")
    assert stage.done == 9 and stage.total == 12
    assert len(stage.commands) == 1
    assert "--array=9-11%8" in stage.commands[0]


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_coverage_displays_dose_order_and_pending_extension(track, isolated_output):
    from src.utils import paths
    from src.visualize import campaign

    runs = expand_with_seeds(load(f"config/Exp1_{track.upper()}.yaml"))
    for run in runs[:9]:
        summary = paths.run_summary_path(run["_run_name"])
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"steps": 12500, "completed": True}))
    coverage = campaign.coverage(track)
    assert coverage.credit_fraction.drop_duplicates().tolist() == [0., .25, .5, 1.]
    assert coverage.state.value_counts().to_dict() == {"completed": 9, "absent": 3}
    assert coverage[coverage.credit_fraction.eq(.25)].state.eq("absent").all()
    assert "9/12" in campaign.provenance(track)


@pytest.mark.parametrize("append", [{}, [None], [{}], [{"prior.credit_fraction": .75}]])
def test_invalid_array_extension_is_rejected(append):
    cfg = load("config/Exp1_PD.yaml")
    cfg["array_append"] = append
    with pytest.raises(ValueError, match="array_append"):
        expand_with_seeds(cfg)
