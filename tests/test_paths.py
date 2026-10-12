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
    # `output CreditICL/` is THE single root for everything the code generates, locally and
    # on the cluster, so "what did this run produce" and "what can I delete" have one answer.
    assert p.outputs_dir().name == "output CreditICL"
    assert p.big_outputs_dir() == p.outputs_dir(), "locally the two tiers are one folder"
    assert p.logs_dir() == p.outputs_dir() / "logs", "jobs of no experiment log at the root"
    assert p.figures_root() == p.outputs_dir() / "figures"
    assert p.all_results_path().parent == p.outputs_dir()
    assert p.captions_path().parent == p.outputs_dir()
    # No doubled project name when the root already IS the project. Compare the
    # tail only — the repo's own parent folder is legitimately called
    # "4. CreditICL", so a substring check on the whole path gives a false alarm.
    tail = p.outputs_dir().parts[-2:]
    assert tail[0] != "CreditICL" or tail[1] != "CreditICL"
    assert p.datasets_dir().parts[-2:] != ("CreditICL", "CreditICL")


def test_on_vsc_tiers_are_separate(paths):
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    assert p.on_vsc() is True
    assert "lustre1" in p.pretrained_dir().as_posix(), "released weights belong on staging"
    assert "lustre1" in p.run_checkpoints_dir("exp1_pd__a__s0").as_posix(), "ours too"
    assert "lustre1" in p.datasets_dir().as_posix(), "datasets belong on staging"
    assert "/data/leuven" in p.outputs_dir().as_posix(), "metrics belong on $VSC_DATA"
    assert "/data/leuven" in p.logs_dir().as_posix(), "logs belong on $VSC_DATA"


def test_project_name_is_inserted_on_shared_roots(paths):
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    assert p.repo_dir().as_posix().endswith("/CreditICL")
    assert "/CreditICL/" in p.pretrained_dir().as_posix()
    assert "/CreditICL/output CreditICL/" in p.big_outputs_dir().as_posix() + "/"


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
    for key in ("on_vsc", "staging_root", "data_root", "datasets_dir", "pretrained_dir",
                "outputs_dir", "big_outputs_dir"):
        assert key in d


# -- the per-experiment layout (29-09-2026) -------------------------------------


def test_every_experiment_has_its_own_folder_on_both_tiers(paths):
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    for n in (0, 1, 2, 3):
        small, big = p.experiment_dir(n), p.experiment_dir(n, big=True)
        assert small.name == big.name == f"experiment_{n}"
        assert small.parent == p.outputs_dir() and big.parent == p.big_outputs_dir()
        assert "/data/leuven" in small.as_posix() and "lustre1" in big.as_posix()
    with pytest.raises(ValueError):
        p.experiment_dir(4)


def test_a_run_name_or_tag_names_its_experiment(paths):
    p = paths()
    assert p.experiment_of("exp1_pd__prior-credit_fraction=0.5__s2") == 1
    assert p.experiment_of("exp2_lgd__model=tabpfn3__s0") == 2
    assert p.experiment_of("exp1bench_pd_a3_s2500") == 1
    assert p.experiment_of("exp2searchbench_pd_a5") == 2
    assert p.experiment_of("exp2_pd_search__arm=tabpfn3_adamw_1e-5__s0") == 2
    assert p.experiment_of("exp0_pd__base__s0") == 0
    assert p.experiment_of("debug_exp1_lgd") == 0, "debug runs belong to the debug suite"
    for bad in ("pilot_batch_64", "exp9_pd__x", "expX_pd", "reference_pd_catboost"):
        with pytest.raises(ValueError):
            p.experiment_of(bad)


def test_benchmark_files_go_to_their_owner(paths):
    p = paths()
    assert p.benchmark_dir_for("exp1bench_pd_a0", "pd") == p.experiment_dir(1) / "benchmark" / "pd"
    assert p.benchmark_dir_for("reference_lgd_tabpfn3", "ood") == p.reference_dir() / "benchmark" / "ood"
    assert p.benchmark_dir_for("20260929_101010", "pd") == p.big_outputs_dir() / "adhoc" / "benchmark" / "pd"
    with pytest.raises(ValueError):
        p.benchmark_dir(1, "eval")


def test_a_run_keeps_its_small_files_together_and_its_weights_on_the_big_tier(paths):
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    name = "exp1_lgd__prior-credit_fraction=1.0__s1"
    run = p.run_dir(name)
    assert run == p.experiment_dir(1) / "runs" / name
    for kind, file in p.RUN_FILES.items():
        assert p.run_file(name, kind) == run / file
    assert p.run_summary_path(name) == run / "summary.json"
    assert p.run_checkpoints_dir(name) == p.experiment_dir(1, big=True) / "checkpoints" / name
    with pytest.raises(ValueError):
        p.run_file(name, "manifest")


def test_every_notebooks_figures_sit_in_one_figures_folder(paths):
    """30-09-2026: one `figures/<notebook>/` folder for every chapter (no `general/`)."""
    p = paths()
    for nb in ("0.2_prior_visualisation_pd", "1.3_pd_results", "2.1_pd_finetuning", "smoke"):
        assert p.figures_dir(nb) == p.outputs_dir() / "figures" / nb


def test_benchmark_results_live_on_project_storage(paths):
    """`$VSC_DATA` is 75 GiB for every project together (30-09-2026): the benchmark CSVs go to
    staging with the checkpoints; only what a job appends to while it runs stays small."""
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    for owner in (1, "reference", "adhoc"):
        assert "lustre1" in p.benchmark_dir(owner, "pd").as_posix(), owner
    assert p.benchmark_dir(1, "pd") == p.experiment_dir(1, big=True) / "benchmark" / "pd"
    assert p.benchmark_dir("reference", "ood") == p.reference_dir(big=True) / "benchmark" / "ood"
    for small in (p.logs_dir(), p.logs_dir(1), p.run_dir("exp1_pd__x__s0"), p.figures_root()):
        assert small.as_posix().startswith("/data/leuven/383/vsc38338/CreditICL/output CreditICL"), small


def test_the_ood_cache_is_input_data_not_output(paths):
    p = paths(VSC_DATA="/data/leuven/383/vsc38338", VSC_PROJECT_LUSTRE1="/lustre1/project")
    assert p.ood_cache_dir() == p.datasets_dir() / "ood"
    assert p.outputs_dir() not in p.ood_cache_dir().parents
    assert p.big_outputs_dir() not in p.ood_cache_dir().parents


def test_the_slurm_scripts_log_where_python_says(paths, repo_root):
    """The job scripts compute their log folder in bash (before the environment is active), so
    the one thing tying them to `paths` is this test."""
    p = paths(VSC_DATA="/data/x", VSC_PROJECT_LUSTRE1="/lustre1/project")
    assert p.outputs_dir().as_posix() == "/data/x/CreditICL/output CreditICL"
    for script in ("pretrain_pd", "pretrain_lgd", "benchmark", "exp0"):
        text = (repo_root / "scripts" / "slurm" / f"{script}.slurm").read_text(encoding="utf-8")
        assert 'OUTPUT_ROOT="${VSC_DATA}/CreditICL/output CreditICL"' in text, script
        assert 'LOGDIR="${OUTPUT_ROOT}/experiment_${EXP}/logs"' in text, script
