"""Cleanup must preserve research evidence even when outputs share the same storage tier."""

from pathlib import Path

import pytest

from src.utils import clean_run, paths, run_experiment


def write(root: Path, relative: str, content: bytes = b"evidence") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def cleanup_tree(isolated_output, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(run_experiment, "queued_job_names", lambda: set())
    return paths.outputs_dir(), paths.big_outputs_dir()


def test_only_explicit_failures_and_generated_outputs_are_deleted(cleanup_tree, capsys):
    small, big = cleanup_tree
    victims = [
        write(small, "experiment_0/logs/failed.log", b"traceback\nEND status=FAILED rc=1\n"),
        write(big, "experiment_1/logs/failed.log", b"END status=OK\nEND status=FAILED\n"),
        write(small, "figures/results/01_curve.pdf"),
        write(small, "figures/results/_figures.json"),
        write(small, "All_Results.md"),
        write(small, "CAPTIONS.md"),
    ]
    keep = [
        write(small, "experiment_1/logs/done.log", b"END status=OK\n"),
        write(small, "experiment_1/logs/resumed.log", b"END status=FAILED\nEND status=OK\n"),
        write(small, "experiment_1/logs/running.log", b"error in an earlier stage\n"),
        write(small, "experiment_1/runs/arm/summary.json"),
        write(small, "experiment_0/report/checks.json"),
        write(small, "logs/processed_summary_pd.csv"),
        write(big, "experiment_1/checkpoints/arm/step-12500.ckpt"),
        write(big, "experiment_0/validation/attempt/checkpoints/step-600.pt"),
        write(big, "experiment_0/validation/attempt/checkpoints/logs/debug.log", b"END status=FAILED\n"),
        write(big, "experiment_1/benchmark/pd/scores.csv"),
        write(big, "experiment_1/benchmark/receipts/arm.json"),
        write(big, "prior_cache/logs/build.log", b"END status=FAILED\n"),
        write(small, "figures/README.md"),
        write(small, "figures/important.ckpt"),
        write(small, "README.md"),
    ]
    preserved = {path: path.read_bytes() for path in keep}
    assert clean_run.main(["--stale"]) == 0
    assert all(path.exists() for path in victims)
    assert "Nothing was deleted" in capsys.readouterr().out
    assert {item.path for item in clean_run.stale_files()} == set(victims)
    assert clean_run.main(["--stale", "--clean"]) == 0
    assert not any(path.exists() for path in victims)
    assert {path: path.read_bytes() for path in keep} == preserved
    assert clean_run.stale_files() == []
    assert clean_run.main(["--stale", "--clean"]) == 0


def test_collapsed_storage_tiers_are_not_counted_twice(cleanup_tree, monkeypatch):
    small, _ = cleanup_tree
    monkeypatch.setattr(paths, "big_outputs_dir", lambda: small)
    victim = write(small, "logs/failed.log", b"END status=FAILED\n")
    plan = clean_run.stale_files()
    assert [item.path for item in plan] == [victim]
    assert clean_run.delete_stale_files(plan) == 1


@pytest.mark.parametrize("failure", ["queued", "scheduler_unavailable", "inside_job"])
def test_active_or_unverifiable_cluster_never_deletes(cleanup_tree, monkeypatch, failure):
    small, _ = cleanup_tree
    victim = write(small, "logs/failed.log", b"END status=FAILED\n")
    if failure == "queued":
        monkeypatch.setattr(run_experiment, "queued_job_names", lambda: {"crediticl-exp1-pd"})
    elif failure == "scheduler_unavailable":
        def unavailable():
            raise RuntimeError("squeue unavailable")
        monkeypatch.setattr(run_experiment, "queued_job_names", unavailable)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert clean_run.main(["--stale", "--clean"]) == 1
    assert victim.read_bytes() == b"END status=FAILED\n"


def test_changed_file_aborts_entire_plan_before_first_deletion(cleanup_tree):
    small, _ = cleanup_tree
    first = write(small, "logs/a.log", b"END status=FAILED\n")
    last = write(small, "logs/z.log", b"END status=FAILED\n")
    plan = clean_run.stale_files()
    last.write_bytes(b"END status=OK\n")
    with pytest.raises(RuntimeError, match="changed since listing"):
        clean_run.delete_stale_files(plan)
    assert first.exists()
    assert last.read_bytes() == b"END status=OK\n"


def test_linked_directories_and_files_are_skipped(cleanup_tree, monkeypatch):
    small, _ = cleanup_tree
    linked_dir_file = write(small, "figures/external/paper.pdf")
    linked_file = write(small, "figures/linked.pdf")
    # Exercise both symlink/junction detection without requiring Windows symlink privileges.
    original = clean_run._linked
    monkeypatch.setattr(clean_run, "_linked", lambda p: p in {linked_dir_file.parent, linked_file} or original(p))
    assert clean_run.stale_files() == []
    assert linked_dir_file.exists() and linked_file.exists()


def test_replaced_output_root_is_refused(cleanup_tree, monkeypatch):
    small, _ = cleanup_tree
    victim = write(small, "figures/paper.pdf")
    plan = clean_run.stale_files()
    original = clean_run._linked
    monkeypatch.setattr(clean_run, "_linked", lambda p: p == small or original(p))
    with pytest.raises(RuntimeError):
        clean_run.delete_stale_files(plan)
    assert victim.exists()


@pytest.mark.parametrize("option", [["--checkpoints"], ["--processed"], ["--prior-cache"], ["--experiment", "0"]])
def test_stale_mode_rejects_broader_cleanup_flags(option):
    with pytest.raises(SystemExit) as error:
        clean_run.main(["--stale", "--clean", *option])
    assert error.value.code == 2


def test_foreign_output_root_is_refused(cleanup_tree, monkeypatch, tmp_path):
    foreign = tmp_path / "output OtherProject"
    victim = write(foreign, "figures/paper.pdf")
    monkeypatch.setattr(paths, "big_outputs_dir", lambda: foreign)
    assert clean_run.main(["--stale", "--clean"]) == 1
    assert victim.exists()
