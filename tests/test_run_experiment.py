"""The driver that survives the cluster stopping.

Maintenance drains, node failures, the 72 h walltime, and running out of credits until the
balance is topped up — this cluster stops routinely, and a plan made of five sbatch commands
in a fixed order will be executed wrongly after a week's interruption. `pipeline` replaces the
commands: it reads what is finished off the output tree and submits only the rest.

The property every test here defends is the same one: **running it again is always the right
move, and never the wrong one.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.run_experiment import plan, render

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """An isolated output tree, and no `squeue`: nothing is queued unless a test says so."""
    from src.utils import paths
    from src.utils import run_experiment as pipeline

    monkeypatch.setattr(paths, "outputs_dir", lambda: tmp_path / "output")
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "output" / "results")
    monkeypatch.setattr(paths, "checkpoints_dir", lambda *a: tmp_path / "ckpt")
    monkeypatch.setattr(pipeline, "queued_job_names", set)
    return tmp_path


def _finish_arms(tmp_path, track, indices):
    """Write the `summary.json` a completed arm leaves behind."""
    from src.utils.config import expand_with_seeds, load, run_name

    runs = expand_with_seeds(load(ROOT / "config" / f"Exp1_{track.upper()}.yaml"))
    for i in indices:
        d = tmp_path / "output" / run_name(runs[i])
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(
            json.dumps({"steps": 12_500, "completed": True}), encoding="utf-8"
        )


def test_benchmark_is_blocked_until_every_arm_has_trained(fake_tree):
    """Phase 2 scores what phase 1 writes. Not a preference — there is nothing to score."""
    stages = {s.name: s for s in plan(1, ["lgd"])}
    assert stages["exp1-lgd-train"].state == "ready"
    assert stages["exp1-lgd-benchmark"].state == "blocked"
    assert stages["exp1-lgd-benchmark"].blocked_by == "exp1-lgd-train"
    assert not stages["exp1-lgd-benchmark"].commands, "a blocked stage must offer no command"


def test_one_arm_short_still_blocks_the_benchmark(fake_tree):
    """74 of 75 is not done. Scoring then would silently benchmark a partial sweep."""
    _finish_arms(fake_tree, "lgd", range(74))
    stages = {s.name: s for s in plan(1, ["lgd"])}
    assert stages["exp1-lgd-train"].done == 74
    assert stages["exp1-lgd-train"].state == "ready"
    assert stages["exp1-lgd-benchmark"].state == "blocked"


def test_a_finished_sweep_unblocks_the_benchmark(fake_tree):
    _finish_arms(fake_tree, "lgd", range(75))
    stages = {s.name: s for s in plan(1, ["lgd"])}
    assert stages["exp1-lgd-train"].state == "done"
    assert not stages["exp1-lgd-train"].commands, "a finished stage must offer no command"
    bench = stages["exp1-lgd-benchmark"]
    assert bench.state == "ready"
    assert any("EXP=1,TRACK=lgd" in " ".join(c) for c in bench.commands)


def test_after_a_drain_only_the_missing_arms_are_resubmitted(fake_tree):
    """THE point of the whole module. A cluster-wide drain cancels every pending task; blindly
    resubmitting the array would redo everything already trained."""
    from src.utils.run_experiment import plan as _plan

    _finish_arms(fake_tree, "lgd", list(range(40)) + [50, 51])
    train = next(s for s in _plan(1, ["lgd"]) if s.kind == "train")
    assert train.done == 42
    joined = " ".join(train.commands[0])
    assert "--array=40-49,52-74%8" in joined, joined
    # A scattered spec goes to ONE cluster: splitting it across two gains nothing.
    assert len(train.commands) == 1


def test_a_fresh_sweep_is_split_across_two_clusters_but_a_partial_one_is_not(fake_tree):
    """Two clusters means two queues, which halves the wall-clock on a fresh run. It is a
    scheduling choice and nothing about LGD requires wICE — `--single-cluster` turns it off."""
    fresh = next(s for s in plan(1, ["lgd"]) if s.kind == "train")
    assert len(fresh.commands) == 2
    assert any("wice" in " ".join(c) for c in fresh.commands)
    assert any("mindwell" in " ".join(c) for c in fresh.commands)

    single = next(s for s in plan(1, ["lgd"], single_cluster=True) if s.kind == "train")
    assert len(single.commands) == 1
    assert "wice" not in " ".join(single.commands[0])


def test_pd_and_lgd_are_treated_identically(fake_tree):
    """They were not, and the asymmetry was accidental: LGD got two sbatch commands and PD one
    because LGD was split across clusters. Same stages, same rules, same arm count."""
    stages = {s.name: s for s in plan(1, ["lgd", "pd"])}
    assert {s.total for s in stages.values() if s.kind == "train"} == {75}
    assert {s.total for s in stages.values() if s.kind == "benchmark"} == {76}
    for track in ("lgd", "pd"):
        assert stages[f"exp1-{track}-benchmark"].state == "blocked"


def test_nothing_queued_is_submitted_twice(fake_tree, monkeypatch):
    """A drain empties the queue; a slow day does not. Re-running while 8 arms are pending
    must not add 8 more."""
    from src.utils import run_experiment as pipeline

    monkeypatch.setattr(pipeline, "queued_job_names", lambda: {"crediticl-lgd"})
    train = next(s for s in plan(1, ["lgd"]) if s.kind == "train")
    assert train.state == "running"
    assert not train.commands


def test_a_broken_squeue_errs_towards_submitting(fake_tree, monkeypatch):
    """No `squeue` on a laptop, and a login node can have a broken module. An empty answer must
    mean "submit it" — a duplicate job can be cancelled, work that never starts cannot."""
    from src.utils import run_experiment as pipeline

    monkeypatch.setattr(pipeline, "queued_job_names", set)
    assert next(s for s in plan(1, ["lgd"]) if s.kind == "train").state == "ready"


def test_the_report_says_what_to_do_next(fake_tree):
    text = render(plan(1, ["lgd", "pd"]), 1)
    assert "READY TO SUBMIT" in text
    assert "waiting on exp1-lgd-train" in text
    assert "Safe to repeat" in text


def test_the_experiment_number_is_required_and_shown():
    """"Did I just submit Exp1 or Exp2?" must never be a question answered from memory. The
    number is a required positional and appears in the report header."""
    from src.utils.run_experiment import main, render

    # No number -> argparse exits (SystemExit), never a silent default.
    with pytest.raises(SystemExit):
        main([])

    assert "EXPERIMENT 2" in render([], 2)  # empty stage list, exp 2


def test_exp2_and_exp3_refuse_to_run_while_still_templates(fake_tree, capsys):
    """They ship with FILL_FROM_EXP1 where Exp1's winner goes. Submitting one would run for
    hours and measure the wrong arm, so `run_experiment 2` must refuse until it is filled in."""
    from src.utils.run_experiment import main, unconfigured_tracks

    assert unconfigured_tracks(2, ["lgd", "pd"]) == ["lgd", "pd"]
    assert unconfigured_tracks(3, ["lgd", "pd"]) == ["lgd", "pd"]
    assert unconfigured_tracks(1, ["lgd", "pd"]) == [], "Exp1 is fully configured"

    rc = main(["2", "--submit"])
    assert rc == 1, "a template experiment must refuse with a non-zero exit"
    out = capsys.readouterr().out
    assert "NOT CONFIGURED YET" in out
    assert "FILL_FROM_EXP1" in out
    assert "sbatch" not in out, "nothing may be submitted for an unconfigured experiment"


def test_one_invocation_drives_one_experiment_only():
    """`plan` takes a single experiment, not a list — the tool cannot be pointed at 'all
    experiments' by accident."""
    import inspect

    from src.utils.run_experiment import plan

    sig = inspect.signature(plan)
    assert list(sig.parameters)[0] == "exp"
    # a single int, so exp1 and exp2 can never be planned in the same call
    stages = plan(1, ["lgd"])
    assert all(s.name.startswith("exp1-") for s in stages)
