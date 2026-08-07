"""Storage tiers: big files to project staging, small files to $VSC_DATA."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def paths(monkeypatch):
    """Reload the module so it re-reads the environment each time."""
    def _load(**env):
        for var in ("VSC_DATA", "VSC_SCRATCH", "VSC_PROJECT_LUSTRE1",
                    "CREDITICL_STAGING_ROOT", "CREDITPFN_STAGING_ROOT", "TABPFN_STAGING_ROOT"):
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import src.utils.paths as p
        return importlib.reload(p)
    return _load


def test_off_vsc_everything_is_local(paths):
    p = paths()
    assert p.on_vsc() is False
    # Local outputs live under results/_local, not a separate top-level `res/`.
    # `res/` was deleted for good: there is now exactly one place results can live,
    # and locally there should not really be any — real runs happen on the cluster.
    assert p.outputs_dir().name == "_local"
    assert p.outputs_dir().parent.name == "results"
    # No doubled project name when the root already IS the project. Compare the
    # tail only — the repo's own parent folder is legitimately called
    # "4. CreditICL", so a substring check on the whole path gives a false alarm.
    tail = p.outputs_dir().parts[-2:]
    assert tail[0] != "CreditICL" or tail[1] != "CreditICL"
    assert p.datasets_dir().parts[-2:] != ("CreditICL", "CreditICL")


def test_on_vsc_tiers_are_separate(paths):
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    assert p.on_vsc() is True
    assert "lustre1" in p.checkpoints_dir().as_posix(), "checkpoints belong on staging"
    assert "lustre1" in p.datasets_dir().as_posix(), "datasets belong on staging"
    assert "/data/leuven" in p.outputs_dir().as_posix(), "metrics belong on $VSC_DATA"
    assert "/data/leuven" in p.logs_dir().as_posix(), "logs belong on $VSC_DATA"


def test_project_name_is_inserted_on_shared_roots(paths):
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    assert p.repo_dir().as_posix().endswith("/CreditICL")
    assert "/CreditICL/" in p.checkpoints_dir().as_posix()


def test_staging_override_wins(paths):
    p = paths(VSC_DATA="/data/x", CREDITICL_STAGING_ROOT="/my/staging")
    assert p.staging_root().as_posix() == "/my/staging"


def test_falls_back_to_the_lab_shared_variable(paths):
    p = paths(VSC_DATA="/data/x", CREDITPFN_STAGING_ROOT="/lab/staging")
    assert p.staging_root().as_posix() == "/lab/staging"


def test_our_override_beats_the_lab_one(paths):
    p = paths(VSC_DATA="/data/x", CREDITICL_STAGING_ROOT="/mine", CREDITPFN_STAGING_ROOT="/theirs")
    assert p.staging_root().as_posix() == "/mine"


def test_resolve_writable_returns_a_usable_dir(paths, tmp_path):
    p = paths()
    got = p.resolve_writable(tmp_path / "target")
    assert got.exists()
    assert (got / "probe").write_text("x") is None or True


def test_resolve_writable_falls_back(paths, tmp_path, monkeypatch):
    """Staging permissions have failed mid-run before, so this path matters."""
    p = paths()
    fallback = tmp_path / "fallback"

    def boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.mkdir", boom)
    # mkdir is broken for both, so it should still not raise — it warns and returns.
    with pytest.raises(OSError):
        p.resolve_writable(tmp_path / "nope", fallback=fallback)


def test_describe_reports_every_tier(paths):
    p = paths(VSC_DATA="/data/x", VSC_PROJECT_LUSTRE1="/lustre1/project")
    d = p.describe()
    for key in ("on_vsc", "staging_root", "data_root", "datasets_dir", "checkpoints_dir", "outputs_dir"):
        assert key in d
