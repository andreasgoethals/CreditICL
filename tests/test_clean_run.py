# Came with the template, and worth keeping: `src/utils/clean_run.py` is identical in every
# project, and it deletes things. The one behaviour worth pinning is that a wipe leaves the tracked
# `.gitkeep` markers and their directories behind — without them a fresh clone has nowhere to write.
"""`src/utils/clean_run.py` — the wipe."""

from __future__ import annotations

from pathlib import Path

from src.utils import clean_run


def test_lists_by_default_and_deletes_only_when_asked(isolated_output, capsys) -> None:
    """A listing you meant as a deletion costs one more command; the reverse costs the run."""
    from src.utils.paths import logs_dir

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
    assert paths.big_outputs_dir() == paths.outputs_dir() == tmp_path / "output_CreditICL"

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
