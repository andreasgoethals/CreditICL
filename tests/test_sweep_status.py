"""Resuming a sweep the cluster cancelled.

A cluster-wide drain cancels every PENDING array task without signalling any of them, so the
job script's own resilience — SIGUSR1, checkpoint, exit 64, resubmit — cannot help those arms.
Resubmitting the whole array would redo everything already finished. These tests pin the tool
that resubmits only what is left.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.sweep_status import DONE, PARTIAL, TODO, ArmState, array_spec, report

ROOT = Path(__file__).resolve().parents[1]


def _arm(index: int, state: str) -> ArmState:
    return ArmState(index=index, name=f"arm{index}", state=state, max_steps=12_500)


def test_array_spec_collapses_into_ranges():
    """`0-4,9,12-74`, not 71 comma-separated numbers: a spec you can read before submitting
    300 GPU-hours is a spec you can check."""
    states = [_arm(i, DONE) for i in range(5, 9)] + [_arm(i, TODO) for i in (0, 1, 2, 3, 4)]
    states += [_arm(9, PARTIAL)] + [_arm(i, DONE) for i in (10, 11)]
    states += [_arm(i, TODO) for i in range(12, 75)]
    assert array_spec(states) == "0-4,9,12-74"


def test_a_finished_sweep_asks_for_nothing():
    assert array_spec([_arm(i, DONE) for i in range(75)]) == ""
    assert "EVERY ARM IS DONE" in report([_arm(i, DONE) for i in range(3)])


def test_partial_arms_are_resubmitted_not_skipped():
    """A `partial` arm has a checkpoint and no completed summary. It must appear in the spec —
    `maybe_resume` continues it — and it must NOT be treated as done."""
    states = [_arm(0, DONE), _arm(1, PARTIAL), _arm(2, TODO)]
    assert array_spec(states) == "1-2"


def test_state_is_read_from_the_filesystem_not_from_slurm(tmp_path, monkeypatch):
    """`sacct` forgets and a drained cluster loses its queue; the output tree survives both."""
    from src.utils import paths, sweep_status
    from src.utils.config import expand_with_seeds, load, run_name

    runs = expand_with_seeds(load(ROOT / "config" / "Exp1_LGD.yaml", allow_placeholders=True))
    del paths  # `out_root` keeps every file of an arm in <out_root>/<run name>/, like pretrain.py

    # arm 0 finished; arm 1 stopped early with a checkpoint; arm 2 never started.
    d0 = tmp_path / run_name(runs[0])
    d0.mkdir(parents=True)
    (d0 / "summary.json").write_text(json.dumps({"steps": 12_500, "completed": True}), encoding="utf-8")
    d1 = tmp_path / run_name(runs[1])
    (d1 / "checkpoints").mkdir(parents=True)
    (d1 / "checkpoints" / "step-3000.ckpt").write_bytes(b"x")
    (d1 / "summary.json").write_text(
        json.dumps({"steps": 3_000, "completed": False, "stopped_by_signal": "SIGUSR1"}),
        encoding="utf-8",
    )

    states = sweep_status.arm_states(ROOT / "config" / "Exp1_LGD.yaml", out_root=tmp_path)
    assert states[0].state == DONE
    assert states[1].state == PARTIAL and states[1].stopped_by == "SIGUSR1"
    assert states[2].state == TODO
    assert array_spec(states).startswith("1-")


def test_an_old_summary_without_the_completed_key_still_counts_as_done(tmp_path, monkeypatch):
    """`completed` was added on 24-08-2026. Treating its absence as "not done" would redo
    every arm that finished before then."""
    from src.utils import paths, sweep_status
    from src.utils.config import expand_with_seeds, load, run_name

    runs = expand_with_seeds(load(ROOT / "config" / "Exp1_LGD.yaml", allow_placeholders=True))
    del paths
    d0 = tmp_path / run_name(runs[0])
    d0.mkdir(parents=True)
    (d0 / "summary.json").write_text(json.dumps({"steps": 12_500}), encoding="utf-8")

    assert sweep_status.arm_states(ROOT / "config" / "Exp1_LGD.yaml", out_root=tmp_path)[0].state == DONE


def test_finished_shorter_schedule_does_not_count_as_finished_extended_run(tmp_path):
    from src.utils import sweep_status
    from src.utils.config import expand_with_seeds, load

    config = ROOT / "config" / "Exp1_LGD.yaml"
    run = expand_with_seeds(load(config))[0]
    directory = tmp_path / run["_run_name"]
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({"steps": 100, "completed": True}))
    assert sweep_status.arm_states(config, tmp_path)[0].state == PARTIAL


def test_finished_summary_cannot_hide_a_changed_training_recipe(tmp_path):
    from src.utils import sweep_status
    from src.utils.config import expand_with_seeds, load

    config = ROOT / "config" / "Exp1_LGD.yaml"
    run = expand_with_seeds(load(config))[0]
    directory = tmp_path / run["_run_name"]
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({"steps": 12500, "completed": True}))
    run["prior"]["credit_fraction"] = .75
    (directory / "config.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Resume configuration CHANGED: prior"):
        sweep_status.arm_states(config, tmp_path)


def test_finished_summary_cannot_hide_a_changed_training_implementation(tmp_path):
    from src.utils import sweep_status
    from src.utils.config import expand_with_seeds, load

    config = ROOT / "config" / "Exp1_LGD.yaml"
    run = expand_with_seeds(load(config))[0]
    directory = tmp_path / run["_run_name"]
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({
        "steps": 12500, "completed": True, "training_fingerprint": "older implementation"}))
    with pytest.raises(ValueError, match="training code differs"):
        sweep_status.arm_states(config, tmp_path)


def test_resubmit_uses_the_requested_experiment_and_checks_readiness(monkeypatch):
    from src.utils import sweep_status, preflight
    from src.utils import run_experiment
    from src.utils import submit_jobs

    monkeypatch.setattr(run_experiment, "queued_job_names", lambda: set())

    monkeypatch.setattr(sweep_status, "arm_states", lambda *a: [_arm(0, PARTIAL)])
    calls = []
    monkeypatch.setattr(preflight, "main", lambda args: calls.append(args) or 1)
    monkeypatch.setattr(submit_jobs, "submit_command", lambda cmd, receipt: pytest.fail("submitted despite failed gate"))
    assert sweep_status.main(["--config", "config/Exp2_LGD.yaml", "--resubmit"]) == 1
    monkeypatch.setattr(preflight, "main", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(submit_jobs, "submit_command", lambda cmd, receipt: calls.append(cmd) or [])
    assert sweep_status.main(["--config", "config/Exp2_LGD.yaml", "--resubmit"]) == 0
    assert "--export=ALL,EXP=2,VARIANT=" in calls[-1]
    assert calls[-1][-1] == "scripts/slurm/pretrain_lgd.slurm"
    assert '--array=0' in calls[-1]
    assert not any('%' in arg for arg in calls[-1])
    assert calls[0] == ["--require-exp0"]


@pytest.mark.parametrize("track", ["LGD", "PD"])
def test_the_tool_covers_the_whole_grid(track, tmp_path):
    """One state per array index, or the spec cannot be trusted."""
    from src.utils.config import expand_with_seeds, load
    from src.utils.sweep_status import arm_states

    cfg = ROOT / "config" / f"Exp1_{track}.yaml"
    n = len(expand_with_seeds(load(cfg, allow_placeholders=True)))
    states = arm_states(cfg, out_root=tmp_path)
    assert len(states) == n
    assert [s.index for s in states] == list(range(n))
