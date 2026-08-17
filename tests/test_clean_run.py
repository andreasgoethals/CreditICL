# Came with the template, and worth keeping: `src/utils/clean_run.py` is identical in every
# project, and it deletes things. The one behaviour worth pinning is that a wipe leaves the tracked
# `.gitkeep` markers and their directories behind — without them a fresh clone has nowhere to write.
"""`src/utils/clean_run.py` — the wipe."""

from __future__ import annotations

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
    """`rmtree` would take `output/figures/.gitkeep` with it, and the next clone would have
    nowhere to write."""
    from src.utils.paths import figures_dir, logs_dir

    for folder in (logs_dir(), figures_dir()):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".gitkeep").write_text("", encoding="utf-8")
    per_notebook = figures_dir("nb")
    per_notebook.mkdir(parents=True, exist_ok=True)
    (per_notebook / "01_x.pdf").write_bytes(b"%PDF")
    (logs_dir() / "run.log").write_text("x", encoding="utf-8")

    removed = clean_run.wipe(clean_run.roots()[0])
    assert removed == 2                                  # the pdf and the log, not the markers
    assert (logs_dir() / ".gitkeep").is_file()
    assert (figures_dir() / ".gitkeep").is_file()
    assert not per_notebook.exists()                     # per-run, no marker, so it goes


def test_gitkeep_is_never_counted(isolated_output) -> None:
    """A directory holding only structure markers is already clean."""
    from src.utils.paths import logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    (logs_dir() / ".gitkeep").write_text("", encoding="utf-8")
    assert clean_run.measure(clean_run.roots()[0]) == (0, 0)


def test_both_storage_tiers_are_cleared_on_the_cluster(isolated_output) -> None:
    """`output/results/` lives on project storage there, so clearing only `$VSC_DATA` would leave
    the largest files behind."""
    roots = clean_run.roots()
    assert len(roots) == 2
    assert any("staging" in str(r) for r in roots)


def test_processed_is_opt_in(isolated_output) -> None:
    """Rebuilding the cache can cost far more than re-running the notebooks, so "clean the last
    run" must not silently throw it away."""
    from src.utils.paths import processed_dir

    assert processed_dir() not in clean_run.roots()
    assert processed_dir() in clean_run.roots(processed=True)



def test_clean_run_spares_the_ood_cache_even_with_prior_cache(tmp_path, monkeypatch):
    """`--prior-cache` clears the prior pools, and the out-of-domain cache happens to live
    under the same root because that is where big things go. It is not a prior pool.

    It also cannot be rebuilt where the deletion happens: compute nodes have no outbound
    internet, so `fetch_ood` only works from a login node. A sweep that wiped it would report
    every out-of-domain column empty and never say why.
    """
    from src.utils import clean_run

    pool_root = tmp_path / "prior_cache"
    (pool_root / "credit_v1").mkdir(parents=True)
    (pool_root / "credit_v1" / "shard0.npz").write_bytes(b"pool")
    (pool_root / "ood").mkdir(parents=True)
    (pool_root / "ood" / "cache.npz").write_bytes(b"downloaded")
    (pool_root / "ood" / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(clean_run, "prior_cache_root", lambda: pool_root)

    n_before, _ = clean_run.measure(pool_root)
    assert n_before == 1, "the ood cache must not even be COUNTED as deletable"

    clean_run.wipe(pool_root)
    assert not (pool_root / "credit_v1" / "shard0.npz").exists(), "the pool should go"
    assert (pool_root / "ood" / "cache.npz").is_file(), "the ood cache must survive"
    assert (pool_root / "ood" / "manifest.json").is_file()
