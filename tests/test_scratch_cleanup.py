"""Never trade a full shared scratch quota for loss of recoverable checkpoints."""

import pytest

from src.utils import scratch_cleanup as cleaner
from src.utils import working_storage as storage


def test_node_local_workspace_avoids_shared_gpfs_and_publishes_rolling_saves(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDITICL_WORK_ROOT", raising=False)
    monkeypatch.setenv("VSC_SCRATCH_NODE", str(tmp_path / "node"))
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path / "gpfs"))
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    permanent = tmp_path / "project"
    work = storage.checkpoint_work_dir(permanent)
    for step in (125, 250, 500, 750, 1000, 1250):
        (work / f"step-{step}.ckpt").write_bytes(str(step).encode())
        storage.publish_checkpoint(work, step, permanent_every=2500, final=False, stopped=False,
                                   keep_steps=(125, 250, 500, 1000, 2500))
        assert list(work.glob("step-*.ckpt")) == [work / f"step-{step}.ckpt"]
        assert (permanent / f"step-{step}.ckpt").read_bytes() == str(step).encode()
    assert all((permanent / f"step-{s}.ckpt").exists() for s in (125, 250, 500, 1000))
    assert not (tmp_path / "gpfs").exists()
    # Simulate losing the node entirely: a new job resumes from project storage.
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    restarted = storage.checkpoint_work_dir(permanent)
    assert (restarted / "step-1250.ckpt").read_bytes() == b"1250"


def test_failed_publish_does_not_remove_previous_working_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITICL_WORK_ROOT", str(tmp_path / "work"))
    work = storage.checkpoint_work_dir(tmp_path / "project")
    (work / "step-1.ckpt").write_bytes(b"previous")
    (work / "step-2.ckpt").write_bytes(b"new")
    def fail(*a):
        raise OSError("quota")
    monkeypatch.setattr(storage, "atomic_copy", fail)
    with pytest.raises(OSError, match="quota"):
        storage.publish_checkpoint(work, 2, permanent_every=10, final=False, stopped=False)
    assert (work / "step-1.ckpt").read_bytes() == b"previous"


@pytest.mark.parametrize("publication", ["valid", "missing", "corrupt"])
def test_scratch_cleanup_only_removes_verified_copies(publication, tmp_path, monkeypatch):
    scratch, permanent = tmp_path / "gpfs", tmp_path / "project" / "output CreditICL"
    permanent.mkdir(parents=True)
    job = scratch / "CreditICL" / "jobs" / "123"
    folder = job / "checkpoints" / "key"
    folder.mkdir(parents=True)
    (folder / "destination.txt").write_text(str(permanent))
    (folder / "step-10.ckpt").write_bytes(b"trained")
    if publication != "missing":
        (permanent / "step-10.ckpt").write_bytes(b"trained" if publication == "valid" else b"bad")
    (job / "inputs").mkdir()
    (job / "inputs" / "copy.bin").write_bytes(b"disposable")
    outsider = scratch / "JEVPFN" / "keep.bin"
    outsider.parent.mkdir()
    outsider.write_bytes(b"other project")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(scratch))
    monkeypatch.delenv("VSC_SCRATCH_LUSTRE1", raising=False)
    monkeypatch.setattr(cleaner.paths, "big_outputs_dir", lambda: permanent)
    monkeypatch.setattr(cleaner, "retired_jobs", lambda ids: {"123"})
    monkeypatch.setattr(cleaner.clean_run, "_require_idle_cluster", lambda: None)
    cleaner.clean_published()
    assert (folder / "step-10.ckpt").exists()
    cleaner.clean_published(clean=True)
    assert (folder / "step-10.ckpt").exists() == (publication != "valid")
    assert not (job / "inputs").exists()
    assert outsider.read_bytes() == b"other project"
    assert cleaner.clean_published(clean=True) == (0, 0)


def test_accounting_does_not_claim_other_projects_or_active_jobs(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(cleaner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0,
        stdout="1|crediticl-exp2-pd|COMPLETED\n2|jevpfn-run|COMPLETED\n3|crediticl-exp1-pd|RUNNING\n"))
    assert cleaner.retired_jobs({"1", "2", "3"}) == {"1"}


def test_missing_node_storage_never_falls_back_to_shared_gpfs(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDITICL_WORK_ROOT", raising=False)
    monkeypatch.delenv("VSC_SCRATCH_NODE", raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path))
    with pytest.raises(RuntimeError, match="No automatic fallback"):
        storage.work_root()
    assert list(tmp_path.iterdir()) == []
