"""Output renames must preserve historical results even after git pull."""

import pytest

from src.utils.rename_output import apply_moves, main, plan_moves


def test_merge_preserves_old_and_current_files(tmp_path):
    old, new = tmp_path / "output", tmp_path / "output CreditICL"
    (old / "manifests").mkdir(parents=True)
    (old / "manifests" / "run.json").write_text("completed")
    (old / "README.md").write_text("historical")
    new.mkdir()
    (new / "README.md").write_text("current")
    moves = plan_moves(old, new)
    assert (old / "README.md").read_text() == "historical"  # preview is read only
    apply_moves(moves, old)
    assert not old.exists()
    assert (new / "manifests" / "run.json").read_text() == "completed"
    assert (new / "README.md").read_text() == "current"
    assert (new / "legacy-output" / "output" / "README.md").read_text() == "historical"
    assert plan_moves(old, new) == []  # repeat is a no-op


def test_atomic_rename_keeps_checkpoint_bytes(tmp_path):
    old, new = tmp_path / "output", tmp_path / "output CreditICL"
    old.mkdir()
    (old / "weights.ckpt").write_bytes(b"checkpoint")
    apply_moves(plan_moves(old, new), old)
    assert (new / "weights.ckpt").read_bytes() == b"checkpoint"


def test_collision_blocks_before_any_move(tmp_path):
    old, new = tmp_path / "output", tmp_path / "output CreditICL"
    old.mkdir()
    (old / "run.json").write_text("old")
    (new / "legacy-output" / "output").mkdir(parents=True)
    (new / "run.json").write_text("new")
    (new / "legacy-output" / "output" / "run.json").write_text("earlier")
    with pytest.raises(FileExistsError):
        plan_moves(old, new)
    assert (old / "run.json").read_text() == "old"


def test_both_tiers_are_checked_before_any_move(tmp_path, monkeypatch):
    from src.utils import rename_output as module

    small, big = tmp_path / "small", tmp_path / "big"
    for tier in (small, big):
        (tier / "output").mkdir(parents=True)
        (tier / "output" / "run.json").write_text("record")
    (big / "output CreditICL").write_text("not a directory")
    monkeypatch.setattr(module.paths, "outputs_dir", lambda: small / "output CreditICL")
    monkeypatch.setattr(module.paths, "big_outputs_dir", lambda: big / "output CreditICL")
    assert main(["--apply"]) == 1
    assert (small / "output" / "run.json").exists()


@pytest.mark.parametrize("job_name", ["crediticl-exp1-pd", "cicl-eval"])
def test_active_jobs_block_rename(tmp_path, monkeypatch, job_name):
    from src.utils import rename_output as module
    from src.utils import run_experiment

    (tmp_path / "output").mkdir()
    monkeypatch.setattr(module.paths, "outputs_dir", lambda: tmp_path / "output CreditICL")
    monkeypatch.setattr(module.paths, "big_outputs_dir", lambda: tmp_path / "output CreditICL")
    monkeypatch.setattr(module.paths, "on_vsc", lambda: True)
    monkeypatch.setattr(run_experiment, "queued_job_names", lambda: {job_name})
    assert main(["--apply"]) == 1
    assert (tmp_path / "output").exists()


def test_unavailable_scheduler_blocks_rename(tmp_path, monkeypatch):
    from src.utils import rename_output as module
    from src.utils import run_experiment

    (tmp_path / "output").mkdir()
    monkeypatch.setattr(module.paths, "outputs_dir", lambda: tmp_path / "output CreditICL")
    monkeypatch.setattr(module.paths, "big_outputs_dir", lambda: tmp_path / "output CreditICL")
    monkeypatch.setattr(module.paths, "on_vsc", lambda: True)

    def unavailable():
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(run_experiment, "queued_job_names", unavailable)
    assert main(["--apply"]) == 1
    assert (tmp_path / "output").exists()
