"""GPU-only recovery must never erase CPU validation or another experiment."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils import clean_run, paths

ATTEMPT = "gpu_20261014T160518Z_j11689448_8352ce02"
CPU = "cpu_20261014T154239Z_j62319803_27c8e2cf"


def put(path: Path, content: bytes = b"evidence") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def alias(path: Path, target: Path) -> None:
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(path))
    else:
        path.symlink_to(target, target_is_directory=True)


@pytest.fixture
def idle(isolated_output, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("VSC_SCRATCH_GPFS1", raising=False)
    monkeypatch.delenv("VSC_SCRATCH", raising=False)
    monkeypatch.setattr("src.utils.run_experiment.queued_job_names", lambda: set())
    return isolated_output


def test_gpu_reset_deletes_both_tiers_and_preserves_cpu_and_inputs(idle):
    generated, protected = [], []
    for project in {paths.outputs_dir().parent, paths.big_outputs_dir().parent}:
        for output_name in (paths.OUTPUT_NAME, "output", "outputCreditICL", "output_CreditICL"):
            root = project / output_name
            for name in (f"validation/{ATTEMPT}/logs/credit-pd-3.log",
                         f"validation/{ATTEMPT}/checkpoints/probe/step-10.ckpt",
                         f"validation/{ATTEMPT}/reference/benchmark/pd/results.csv",
                         "logs/exp0_gpu_20261014_180508_j11689448.log",
                         "report/validation.json", "report/validation.json.tmp"):
                generated.append(put(root / "experiment_0" / name))
            for name in (f"experiment_0/validation/{CPU}/report/validation.json",
                         "experiment_0/report/cpu_validation.json",
                         "experiment_0/report/validation.lock",
                         "experiment_0/logs/exp0_cpu_20261014_174213_j62319803.log",
                         "experiment_1/checkpoints/run/step-12500.ckpt",
                         "reference/benchmark/pd/results.csv", "README.md"):
                protected.append(put(root / name))
        generated.append(put(project / "slurm-crediticl-exp0-gpu-11689448.err"))
        for name in ("slurm-crediticl-exp0-cpu-62319803.out",
                     "slurm-crediticl-exp1-pd-11689449.out",
                     "slurm-otherproject-exp0-gpu-11689448.err",
                     "data/raw/table.csv", "checkpoints/tabicl-classifier.ckpt"):
            protected.append(put(project / name))
    assert clean_run.reset_exp0_gpu() == (len(generated), len(generated) * len(b"evidence"))
    assert all(p.exists() for p in generated)
    assert clean_run.reset_exp0_gpu(clean=True)[0] == len(generated)
    assert all(not p.exists() for p in generated)
    assert all(p.read_bytes() == b"evidence" for p in protected)
    assert clean_run.reset_exp0_gpu() == (0, 0)


def test_gpu_reset_cleans_only_scratch_jobs_identified_as_gpu(idle, monkeypatch):
    scratch = idle / "gpfs"
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(scratch))
    unrelated_scratch = put(idle / "other-scratch/CreditICL/jobs/11689448/keep")
    monkeypatch.setenv("VSC_SCRATCH", str(idle / "other-scratch"))
    generated = [put(paths.outputs_dir() / "experiment_0/validation" / ATTEMPT / "report/validation.json"),
                 put(paths.outputs_dir().parent / "slurm-crediticl-exp0-gpu-11688276.err")]
    for job in ("11689448", "11688276"):
        for name in ("inputs/copy.npz", "checkpoints/a/step-10.ckpt", "tabicl-inference/data.mmap"):
            generated.append(put(scratch / "CreditICL/jobs" / job / name))
    protected = [unrelated_scratch, put(scratch / "CreditICL/jobs/62319803/cpu.txt"),
                 put(scratch / "CreditICL/jobs/11689449/production.ckpt"),
                 put(scratch / "OtherProject/jobs/11689448/weights.ckpt")]
    assert clean_run.reset_exp0_gpu(clean=True)[0] == len(generated)
    assert all(not p.exists() for p in generated)
    assert all(p.exists() for p in protected)


def test_gpu_reset_handles_aliases_without_duplicate_deletion(idle, monkeypatch):
    project = paths.outputs_dir().parent
    project.mkdir(parents=True)
    mount = idle / "mount-alias"
    alias(mount, project.parent)
    monkeypatch.setattr(paths, "big_outputs_dir", lambda: mount / project.name / paths.OUTPUT_NAME)
    victim = put(project / "slurm-crediticl-exp0-gpu-11689448.out")
    assert clean_run.reset_exp0_gpu(clean=True) == (1, len(b"evidence"))
    assert not victim.exists()


@pytest.mark.parametrize("kind", ["queued", "allocation"])
def test_gpu_reset_refuses_active_jobs_before_any_deletion(idle, monkeypatch, kind):
    victim = put(paths.outputs_dir() / "experiment_0/report/validation.json")
    if kind == "queued":
        monkeypatch.setattr("src.utils.run_experiment.queued_job_names", lambda: {"crediticl-exp0-cpu"})
    else:
        monkeypatch.setenv("SLURM_JOB_ID", "11689449")
    with pytest.raises(RuntimeError, match="queued/running|login node"):
        clean_run.reset_exp0_gpu(clean=True)
    assert victim.exists()


@pytest.mark.parametrize("relative", ["experiment_0/validation", f"experiment_0/validation/{ATTEMPT}",
                                      f"experiment_0/validation/{ATTEMPT}/linked"])
def test_gpu_reset_refuses_linked_containers_and_attempt_descendants(idle, relative):
    outside = idle / "outside"
    kept = put(outside / "keep.ckpt")
    root = paths.outputs_dir()
    linked = root / relative
    linked.parent.mkdir(parents=True, exist_ok=True)
    alias(linked, outside)
    other = put(paths.big_outputs_dir() / "experiment_0/report/validation.json")
    with pytest.raises(RuntimeError, match="[Ll]inked"):
        clean_run.reset_exp0_gpu(clean=True)
    assert kept.exists() and other.exists()


def test_gpu_reset_refuses_scratch_alias_into_unrelated_files(idle, monkeypatch):
    scratch = idle / "gpfs"
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(scratch))
    victim = put(paths.outputs_dir() / "experiment_0/validation" / ATTEMPT / "log.txt")
    outside = idle / "unrelated"
    kept = put(outside / "keep.ckpt")
    (scratch / "CreditICL/jobs").mkdir(parents=True)
    alias(scratch / "CreditICL/jobs/11689448", outside)
    with pytest.raises(RuntimeError, match="[Ll]inked"):
        clean_run.reset_exp0_gpu(clean=True)
    assert kept.exists() and victim.exists()


def test_gpu_reset_refuses_changed_logs_before_deleting_anything(idle, monkeypatch):
    victim = put(paths.outputs_dir().parent / "slurm-crediticl-exp0-gpu-11689448.out")
    other = put(paths.big_outputs_dir() / "experiment_0/report/validation.json")
    monkeypatch.setattr(clean_run, "_require_idle_cluster", lambda: victim.write_bytes(b"changed"))
    with pytest.raises(RuntimeError, match="changed since planning"):
        clean_run.reset_exp0_gpu(clean=True)
    assert victim.exists() and other.exists()


def test_gpu_reset_refuses_input_overlap(idle, monkeypatch):
    root = paths.big_outputs_dir() / "experiment_0/validation" / ATTEMPT
    victim = put(root / "inputs/raw.csv")
    monkeypatch.setattr(paths, "datasets_dir", lambda: root / "inputs")
    with pytest.raises(RuntimeError, match="Input data or released weights overlap"):
        clean_run.reset_exp0_gpu(clean=True)
    assert victim.exists()


def test_gpu_reset_cli_is_dry_by_default(idle, capsys):
    victim = put(paths.outputs_dir() / "experiment_0/report/validation.json")
    assert clean_run.main(["--reset-exp0-gpu"]) == 0
    assert victim.exists() and "Nothing deleted" in capsys.readouterr().out
    assert clean_run.main(["--reset-exp0-gpu", "--clean"]) == 0
    assert not victim.exists()


@pytest.mark.parametrize("option", ["--reset-output", "--stale", "--checkpoints", "--processed"])
def test_gpu_reset_cannot_be_combined_with_a_wider_scope(idle, option):
    victim = put(paths.outputs_dir() / "experiment_0/report/validation.json")
    with pytest.raises(SystemExit):
        clean_run.main(["--reset-exp0-gpu", "--clean", option])
    assert victim.exists()
