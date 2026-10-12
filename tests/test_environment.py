"""Package mismatches must be caught on login, before training consumes credits."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import environment as env

ROOT = Path(__file__).resolve().parents[1]


def _dist(version="1.0", requires=(), origin=None, python=">=3.11"):
    return SimpleNamespace(version=version, requires=list(requires), metadata={"Requires-Python": python},
                           read_text=lambda _: json.dumps(origin) if origin else None)


def _install(monkeypatch, packages):
    def distribution(name):
        if name not in packages:
            raise env.metadata.PackageNotFoundError(name)
        return packages[name]
    monkeypatch.setattr(env.metadata, "distribution", distribution)


def test_project_bounds_and_transitive_extra_dependencies_are_checked(monkeypatch):
    _install(monkeypatch, {"model": _dist(requires=[
        "trees>=2; extra == 'pretrain'", "unused; extra == 'unselected'", "base>=1"]),
        "trees": _dist("1.0"), "base": _dist()})
    versions, issues = env.requirement_audit({"dependencies": ["model[pretrain]>=1"]})
    assert versions == {"base": "1.0", "model": "1.0", "trees": "1.0"}
    assert issues == ["model requires trees>=2; installed 1.0"]


def test_missing_evaluation_and_dev_requirements_cannot_pass(monkeypatch):
    _install(monkeypatch, {})
    _, issues = env.requirement_audit({"optional-dependencies": {
        "eval": ["catboost>=1.2"], "dev": ["pytest>=8"], "track": ["unused"]}})
    assert len(issues) == 2 and all("missing" in item for item in issues)
    assert not any("unused" in item for item in issues)


def test_version_number_does_not_substitute_for_required_git_revision(monkeypatch):
    wanted = "a" * 40
    _install(monkeypatch, {"model": _dist(origin={"url": "https://example.com/model.git",
                                               "vcs_info": {"commit_id": "b" * 40}})})
    project = {"dependencies": [f"model @ git+https://example.com/model.git@{wanted}"]}
    assert "source differs" in env.requirement_audit(project)[1][0]
    _install(monkeypatch, {"model": _dist(origin={"url": "https://example.com/model.git",
                                               "vcs_info": {"commit_id": wanted}})})
    assert not env.requirement_audit(project)[1]


def test_python_requirements_of_project_and_packages_are_checked(monkeypatch):
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.12.3")
    _install(monkeypatch, {"model": _dist(python=">=3.13")})
    _, issues = env.requirement_audit({"requires-python": "<3.12", "dependencies": ["model"]})
    assert len(issues) == 2 and all("Python" in item for item in issues)


def test_dependency_cycles_and_repeated_extras_terminate(monkeypatch):
    _install(monkeypatch, {"first": _dist(requires=["second", "extra; extra == 'train'"]),
                          "second": _dist(requires=["first[train]"]), "extra": _dist()})
    versions, issues = env.requirement_audit({"dependencies": ["first", "second"]})
    assert not issues and set(versions) == {"first", "second", "extra"}


def test_pip_failure_is_not_reported_as_a_healthy_environment(monkeypatch):
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="model requires numpy<2; installed 2.5", stderr=""))
    ok, detail = env.pip_consistency()
    assert not ok and "numpy<2" in detail


@pytest.mark.parametrize("version,ok", [("2.11.0+cu128", True), ("2.11.0", False),
                                       ("2.13.0+cu128", False)])
def test_vsc_requires_the_selected_cuda_build(monkeypatch, version, ok):
    from src.utils import paths, preflight
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setattr(env, "requirement_audit", lambda: ({}, []))
    monkeypatch.setattr(env, "pip_consistency", lambda: (True, "consistent"))
    monkeypatch.setattr(preflight.metadata, "version", lambda name: version)
    row = next(r for r in preflight.dependency_checks() if r.name == "VSC CUDA torch")
    assert row.ok is ok


def test_package_inventory_excludes_install_credentials_and_paths(monkeypatch):
    package = _dist(origin={"url": "https://private:secret@example.com/model.git",
                           "vcs_info": {"commit_id": "abc"}})
    package.metadata["Name"] = "Some_Model"
    monkeypatch.setattr(env.metadata, "distributions", lambda: [package])
    snapshot = env.environment_snapshot()
    assert snapshot["packages"] == {"some-model": {"version": "1.0", "commit": "abc"}}
    assert "secret" not in json.dumps(snapshot)
    changed = json.loads(json.dumps(snapshot))
    changed["packages"]["some-model"]["version"] = "2.0"
    assert env.environment_fingerprint(changed) != env.environment_fingerprint(snapshot)
    assert "some-model" in env.environment_changes(snapshot, changed)[0]


def test_package_change_invalidates_successful_exp0_checks(tmp_path, monkeypatch):
    from src.utils import paths, preflight
    snapshot = {"python": "3.12.3", "platform": "linux", "packages": {"numpy": {"version": "2.5.1"}}}
    monkeypatch.setattr(env, "environment_snapshot", lambda: snapshot)
    monkeypatch.setattr(preflight, "source_fingerprint", lambda: "code")
    monkeypatch.setattr(paths, "experiment_dir", lambda *a: tmp_path)
    (tmp_path / "report").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "exp0_checks_1.log").write_text(
        "equivalence PD exit code: 0\nequivalence LGD exit code: 0\nEND status=OK\n")
    (tmp_path / "report" / "checks.json").write_text(json.dumps({
        "source_fingerprint": "code", "environment": snapshot,
        "environment_fingerprint": env.environment_fingerprint(snapshot),
        "rows": [{"check": c, "status": "PASS"} for c in preflight.CHECK_SECTIONS]}))
    assert all(row.ok for row in preflight.experiment_zero_checks(include_runs=False))
    snapshot["packages"]["numpy"]["version"] = "2.5.2"
    failures = [r for r in preflight.experiment_zero_checks(include_runs=False) if not r.ok]
    assert len(failures) == 1 and "numpy" in failures[0].detail


@pytest.mark.parametrize("script", ["setup_venv.sh", "update_env.sh"])
def test_installation_is_refused_inside_a_slurm_job(script):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    result = subprocess.run([bash, "-c", f'export SLURM_JOB_ID=123; bash scripts/slurm/{script}'],
                            capture_output=True, text=True, timeout=20, cwd=ROOT)
    assert result.returncode == 1 and "login node" in result.stderr


def test_updater_cannot_install_into_a_sibling_project(tmp_path, monkeypatch):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    repo = tmp_path / "CreditICL"
    scripts = repo / "scripts" / "slurm"
    scripts.mkdir(parents=True)
    other = tmp_path / "sibling" / ".venv"
    other.mkdir(parents=True)
    (scripts / "_activate_env.sh").write_text(
        'export VIRTUAL_ENV="$CREDITICL_TEST_VENV"\n'
        'python() { echo "INSTALL MUST NOT BE REACHED"; return 99; }\n', encoding="utf-8")
    monkeypatch.setenv("VSC_DATA", tmp_path.as_posix())
    monkeypatch.setenv("CREDITICL_TEST_VENV", other.as_posix())
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    result = subprocess.run([bash, str(ROOT / "scripts/slurm/update_env.sh")],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1 and "Refusing to install outside" in result.stderr
    assert "INSTALL MUST NOT BE REACHED" not in result.stdout


@pytest.mark.parametrize("job", [True, False])
def test_activator_checks_jobs_but_allows_login_environment_repair(job):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    script = (ROOT / "scripts/slurm/_activate_env.sh").read_text(encoding="utf-8")
    function = script[script.index("_verify_job_environment() {"):].split("\n}", 1)[0] + "\n}\n"
    code = f"SLURM_JOB_ID={'123' if job else ''}\npython() {{ return 19; }}\n"
    result = subprocess.run([bash, "-s"], input=code + function + "_verify_job_environment\n",
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == (19 if job else 0)
