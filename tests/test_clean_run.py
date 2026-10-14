# Came with the template, and worth keeping: `src/utils/clean_run.py` is identical in every
# project, and it deletes things. The one behaviour worth pinning is that a wipe leaves the tracked
# `.gitkeep` markers and their directories behind — without them a fresh clone has nowhere to write.
"""`src/utils/clean_run.py` — the wipe."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils import clean_run


def test_lists_by_default_and_deletes_only_when_asked(isolated_output, capsys, monkeypatch) -> None:
    """A listing you meant as a deletion costs one more command; the reverse costs the run."""
    from src.utils.paths import logs_dir
    from src.utils import run_experiment

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(run_experiment, "queued_job_names", lambda: set())

    logs_dir().mkdir(parents=True, exist_ok=True)
    victim = logs_dir() / "run.log"
    victim.write_text("x" * 100, encoding="utf-8")

    clean_run.main([])
    assert victim.exists(), "the default must not delete anything"
    assert "Nothing was deleted" in capsys.readouterr().out

    clean_run.main(["--clean"])
    assert not victim.exists()


def test_a_wipe_keeps_the_directory_skeleton(isolated_output) -> None:
    """`rmtree` would take the tracked markers and the tree's README with it."""
    from src.utils.paths import figures_dir, logs_dir, outputs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    (logs_dir() / ".gitkeep").write_text("", encoding="utf-8")
    (outputs_dir() / "README.md").write_text("index", encoding="utf-8")
    per_notebook = figures_dir("1.3_pd_results")
    per_notebook.mkdir(parents=True, exist_ok=True)
    (per_notebook / "01_x.pdf").write_bytes(b"%PDF")
    (logs_dir() / "run.log").write_text("x", encoding="utf-8")

    removed = clean_run.wipe(clean_run.roots()[0])
    assert removed == 2                                  # the pdf and the log, not the markers
    assert (logs_dir() / ".gitkeep").is_file()
    assert (outputs_dir() / "README.md").is_file()
    assert not per_notebook.exists()                     # per-run, no marker, so it goes


def test_gitkeep_is_never_counted(isolated_output) -> None:
    """A directory holding only structure markers is already clean."""
    from src.utils.paths import logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    (logs_dir() / ".gitkeep").write_text("", encoding="utf-8")
    assert clean_run.measure(clean_run.roots()[0]) == (0, 0)


def test_both_storage_tiers_are_cleared_on_the_cluster(isolated_output) -> None:
    """Checkpoints live on project storage there, so `--checkpoints` has to reach the big tier;
    a plain clean touches only the small one."""
    assert [r for r in clean_run.roots() if "staging" in str(r)] == []
    big = [r for r in clean_run.roots(checkpoints=True) if "staging" in str(r)]
    assert len(big) == 4 and all(r.name == "checkpoints" for r in big)
    assert len(clean_run.roots(experiment=1, checkpoints=True)) == 2


def test_processed_is_opt_in(isolated_output) -> None:
    """Rebuilding the cache can cost far more than re-running the notebooks, so "clean the last
    run" must not silently throw it away."""
    from src.utils.paths import processed_dir

    assert processed_dir() not in clean_run.roots()
    assert processed_dir() in clean_run.roots(processed=True)



def test_clean_run_spares_the_ood_cache_and_the_released_weights(tmp_path, monkeypatch):
    """The out-of-domain cache cannot be rebuilt where the deletion happens (compute nodes have
    no outbound internet), and the released weights are a download, not ours. Neither may be
    counted or wiped, whatever the flags."""
    from src.utils import clean_run

    root = tmp_path / "tree"
    (root / "pool").mkdir(parents=True)
    (root / "pool" / "shard0.npz").write_bytes(b"pool")
    (root / "ood").mkdir(parents=True)
    (root / "ood" / "cache.npz").write_bytes(b"downloaded")
    (root / "weights").mkdir(parents=True)
    (root / "weights" / "tabicl.ckpt").write_bytes(b"RELEASED")
    monkeypatch.setattr(clean_run, "ood_cache_dir", lambda: root / "ood")
    monkeypatch.setattr(clean_run, "pretrained_dir", lambda: root / "weights")

    prot = clean_run.protected_paths(checkpoints=True, prior_cache=True)
    assert clean_run.measure(root, prot)[0] == 1, "only the pool is deletable"
    clean_run.wipe(root, prot)
    assert not (root / "pool" / "shard0.npz").exists(), "the pool should go"
    assert (root / "ood" / "cache.npz").is_file(), "the ood cache must survive"
    assert (root / "weights" / "tabicl.ckpt").is_file(), "the released weights must survive"


def test_checkpoints_are_opt_in_even_where_both_tiers_are_one_folder(tmp_path, monkeypatch):
    """The bug that cost job 11517891: a rerun that finds an old checkpoint resumes at
    `max_steps`, trains nothing, and exits 0. So `--checkpoints` must reach them — and a plain
    clean must not, even locally where they sit inside the tree it walks."""
    from src.utils import clean_run, paths

    for var in ("VSC_DATA", *paths.STAGING_ENV_VARS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)   # read at call time by every resolver
    assert paths.big_outputs_dir() == paths.outputs_dir() == tmp_path / "output CreditICL"

    run = paths.run_checkpoints_dir("exp1_lgd__a__s0")
    run.mkdir(parents=True)
    (run / "step-1500.ckpt").write_bytes(b"ours")
    log = paths.logs_dir(1) / "a.log"
    log.parent.mkdir(parents=True)
    log.write_text("x", encoding="utf-8")

    plain = clean_run.protected_paths()
    for root in clean_run.roots():
        clean_run.wipe(root, plain)
    assert (run / "step-1500.ckpt").is_file(), "a plain clean must not reach checkpoints"
    assert not log.exists()

    assert clean_run.run_checkpoint_dirs() == [run]
    opted = clean_run.protected_paths(checkpoints=True)
    for root in clean_run.roots(checkpoints=True):
        clean_run.wipe(root, opted)
    assert not (run / "step-1500.ckpt").exists()


def test_resuming_at_max_steps_warns_that_nothing_will_train():
    """`elapsed_s: 0.0`, exit 0, "finished" — a stale checkpoint is indistinguishable from a
    completed run unless the loop says so."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    assert "if self.step >= self.max_steps:" in src
    assert "THIS RUN WILL TRAIN NOTHING" in src
    assert "--clean --checkpoints" in src, "the warning must name the fix"


def test_checkpoint_discovery_is_by_structure_not_by_name(tmp_path, monkeypatch):
    """A run directory whose name nobody anticipated is the dangerous one: a rerun RESUMES from
    a surviving checkpoint and trains nothing. All four arms did exactly that on 17-08-2026."""
    from src.utils import clean_run

    base = tmp_path / "checkpoints"
    base.mkdir()
    monkeypatch.setattr(clean_run, "checkpoint_roots", lambda experiment=None: [base])
    for name in ("exp1_abc", "debug_exp1_lgd", "pilot_batch_64", "nested"):
        d = base / name
        (d / "inner").mkdir(parents=True)
        (d / "inner" / "step-500.ckpt").write_bytes(b"ours")
    (base / "not_a_run").mkdir()          # a directory with no checkpoint in it

    found = {d.name for d in clean_run.run_checkpoint_dirs()}
    assert found == {"exp1_abc", "debug_exp1_lgd", "pilot_batch_64", "nested"}


def test_no_tree_is_listed_twice():
    """Locally every tier collapses into the repo; listing a tree twice double-counts."""
    from src.utils import clean_run

    for kwargs in ({}, {"checkpoints": True}, {"prior_cache": True, "processed": True},
                   {"experiment": 0, "checkpoints": True}):
        found = clean_run.roots(**kwargs)
        assert len(found) == len({str(r) for r in found}), kwargs


def test_full_reset_includes_legacy_runs_and_names_on_both_tiers(isolated_output, monkeypatch):
    from src.utils import paths

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("src.utils.run_experiment.queued_job_names", lambda: set())
    generated, protected = [], []
    for project in {paths.outputs_dir().parent, paths.big_outputs_dir().parent}:
        for relative in ("output/results.csv", "outputCreditICL/logs/old.log",
                         "output_CreditICL/experiment_1/progress.csv",
                         "checkpoints/exp1_pd__mechanism-rho_range=[0.03,0.12]__s2/step-12500.ckpt",
                         "checkpoints/exp1_lgd__s0/step-12500.ckpt",
                         "checkpoints/debug_pd/step-3.ckpt",
                         "slurm-crediticl-exp0-cpu-62316841.out",
                         "slurm-crediticl-exp0-gpu-11688276.err",
                         "slurm-cicl-preprocess-1234.out"):
            p = project / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"generated")
            generated.append(p)
        for relative in ("checkpoints/tabicl-classifier-v2-20260212.ckpt",
                         "checkpoints/tabpfn-v3-regressor-v3_default.ckpt",
                         "checkpoints/released_bundle/weights.ckpt",
                         "data/raw/input.csv", "prior_cache/ood/table.npz",
                         "slurm-otherproject-1234.out", "slurm-crediticl-not-a-job.out"):
            p = project / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"input")
            protected.append(p)
    assert clean_run.reset_output()[0] == len(generated)
    assert all(p.exists() for p in generated)
    clean_run.reset_output(clean=True)
    assert all(not p.exists() for p in generated)
    assert all(p.read_bytes() == b"input" for p in protected)
    assert clean_run.reset_output() == (0, 0)


def test_full_reset_refuses_active_jobs_even_for_legacy_only(isolated_output, monkeypatch):
    import pytest
    from src.utils import paths

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("src.utils.run_experiment.queued_job_names", lambda: {"crediticl-exp0-cpu"})
    path = paths.pretrained_dir() / "exp1_pd__s0/step-12500.ckpt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="queued/running"):
        clean_run.reset_output(clean=True)
    assert path.exists()


def _directory_alias(alias: Path, target: Path) -> None:
    """Exercise real ancestor aliases on Windows and Linux, without admin privileges."""
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(alias))
    else:
        alias.symlink_to(target, target_is_directory=True)


@pytest.fixture
def aliased_projects(isolated_output, monkeypatch):
    from src.utils import paths

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("src.utils.run_experiment.queued_job_names", lambda: set())
    projects = (paths.outputs_dir().parent, paths.big_outputs_dir().parent)
    aliases = []
    for index, project in enumerate(projects):
        project.mkdir(parents=True)
        alias = isolated_output / f"storage-alias-{index}"
        _directory_alias(alias, project.parent)
        aliases.append(alias / project.name)
    monkeypatch.setattr(paths, "outputs_dir", lambda: aliases[0] / paths.OUTPUT_NAME)
    monkeypatch.setattr(paths, "big_outputs_dir", lambda: aliases[1] / paths.OUTPUT_NAME)
    return projects, aliases


def test_full_reset_handles_vsc_ancestor_aliases_on_both_tiers(aliased_projects):
    projects, aliases = aliased_projects
    generated, protected = [], []
    for project, alias in zip(projects, aliases):
        assert alias != alias.resolve() == project.resolve()
        for relative in ("slurm-crediticl-exp0-cpu-62313072.out",
                         "output CreditICL/experiment_0/logs/checks.log",
                         "output/old.log", "checkpoints/exp1_pd__s0/step-2.ckpt"):
            item = project / relative
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(b"generated")
            generated.append(item)
        for relative in ("checkpoints/released.ckpt", "data/raw/input.csv",
                         "slurm-otherproject-1234.out"):
            item = project / relative
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(b"input")
            protected.append(item)
    assert clean_run.reset_output() == (len(generated), len(generated) * len(b"generated"))
    assert all(p.exists() for p in generated)
    clean_run.reset_output(clean=True)
    assert all(not p.exists() for p in generated)
    assert all(p.read_bytes() == b"input" for p in protected)
    assert clean_run.reset_output() == (0, 0)


def test_full_reset_counts_one_project_once_through_two_aliases(aliased_projects, monkeypatch):
    from src.utils import paths

    projects, _ = aliased_projects
    monkeypatch.setattr(paths, "big_outputs_dir", lambda: projects[0] / paths.OUTPUT_NAME)
    log = projects[0] / "slurm-crediticl-exp0-cpu-62313072.out"
    log.write_bytes(b"log")
    output = projects[0] / paths.OUTPUT_NAME / "checks.log"
    output.parent.mkdir()
    output.write_bytes(b"output")
    assert clean_run.reset_output() == (2, 9)
    assert clean_run.reset_output(clean=True) == (2, 9)
    assert not log.exists() and not output.exists()


@pytest.mark.parametrize("change", ["append", "replace"])
def test_full_reset_still_refuses_changed_slurm_logs_with_ancestor_aliases(
        aliased_projects, monkeypatch, change):
    projects, _ = aliased_projects
    log = projects[0] / "slurm-crediticl-exp0-cpu-62313072.out"
    log.write_bytes(b"log")
    other = projects[1] / "slurm-crediticl-exp0-gpu-11688276.err"
    other.write_bytes(b"evidence")

    def change_after_planning():
        if change == "append":
            with log.open("ab") as stream:
                stream.write(b"still running")
        else:
            before = log.stat()
            replacement = log.with_suffix(".replacement")
            replacement.write_bytes(b"new")
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            replacement.replace(log)
            assert log.stat().st_ino != before.st_ino

    monkeypatch.setattr(clean_run, "_require_idle_cluster", change_after_planning)
    with pytest.raises(RuntimeError, match="Output changed since planning"):
        clean_run.reset_output(clean=True)
    assert log.exists() and other.read_bytes() == b"evidence"


@pytest.mark.parametrize("linked_name", ["slurm-crediticl-exp0-cpu-62313072.out", "output CreditICL"])
def test_full_reset_rejects_linked_outputs_beneath_allowed_ancestor_aliases(
        aliased_projects, tmp_path, linked_name):
    projects, _ = aliased_projects
    outside = tmp_path / "outside-output"
    outside.mkdir()
    evidence = outside / "keep.txt"
    evidence.write_bytes(b"keep")
    _directory_alias(projects[0] / linked_name, outside)
    other = projects[1] / "slurm-crediticl-exp0-gpu-11688276.err"
    other.write_bytes(b"evidence")
    with pytest.raises(RuntimeError, match="Unexpected"):
        clean_run.reset_output(clean=True)
    assert evidence.read_bytes() == b"keep"
    assert other.read_bytes() == b"evidence"
