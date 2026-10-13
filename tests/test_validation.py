"""Execution and provenance tests for the GPU Experiment 0 gate."""
import json
import os
import sys

import pytest

from src.utils import paths
from src.utils import validation as v


def test_attempt_isolates_both_tiers_and_reference_without_touching_exp1(isolated_output):
    normal = (paths.experiment_dir(0), paths.reference_dir(big=True), paths.experiment_dir(1))
    with v.attempt_scope("new_attempt"):
        assert paths.experiment_dir(0) == normal[0] / "validation" / "new_attempt"
        assert paths.benchmark_dir_for("exp0bench_pd_a0", "pd").parent.parent == paths.experiment_dir(0, big=True)
        assert paths.benchmark_dir_for("reference_pd_tabpfn3", "pd").parent.parent == paths.reference_dir(big=True)
        assert paths.reference_dir(big=True).parent == paths.experiment_dir(0, big=True)
        assert paths.experiment_dir(1) == normal[2]
    assert paths.reference_dir(big=True) == normal[1]


@pytest.mark.parametrize("name", ["../output", "", "x/y", "x\\y", "a" * 81])
def test_attempt_rejects_unsafe_names(name):
    with pytest.raises(ValueError), v.attempt_scope(name):
        pass


@pytest.mark.parametrize("task", ["PD", "LGD"])
@pytest.mark.parametrize("choice", [0.5, 1.0])
def test_planned_exp3_selection_preserves_validation_but_recipe_edits_do_not(tmp_path, monkeypatch, task, choice):
    from src.utils.preflight import source_fingerprint

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("")
    folder = tmp_path / "config"
    folder.mkdir()
    file = folder / f"Exp3_{task}.yaml"
    template = "sweep:\n  prior.credit_fraction: [0.0, FILL_FROM_EXP1]\ntrain:\n  max_steps: 200000\n"
    file.write_text(template)
    original = source_fingerprint()
    file.write_text(template.replace("FILL_FROM_EXP1", str(choice)))
    assert source_fingerprint() == original
    file.write_text(template.replace("200000", "200001"))
    assert source_fingerprint() != original
    file.write_text(template.replace("FILL_FROM_EXP1", "0.75"))
    assert source_fingerprint() != original
    file.write_text(template.replace("FILL_FROM_EXP1", "0.0"))
    assert source_fingerprint() != original
    file.write_text(template)
    other = folder / f"Exp1_{task}.yaml"
    other.write_text(template)
    before = source_fingerprint()
    other.write_text(template.replace("FILL_FROM_EXP1", str(choice)))
    assert source_fingerprint() != before


def test_complete_manifest_covers_all_routes_budgets_and_benchmarks():
    stages = v.plan()
    assert len({s.name for s in stages}) == len(stages)
    assert sum(s.name.startswith("smoke-") for s in stages) == 6
    assert sum(s.stop_run is not None for s in stages) == 6
    assert sum(s.name.startswith("resume-") for s in stages) == 6
    assert sum(s.name.startswith("equivalence-") for s in stages) == 2
    assert sum(s.name.startswith("receipt-") for s in stages) == 14
    assert stages[0].name == "preflight"
    phases = [s.phase for s in stages]
    assert phases == sorted(phases, key=("diagnostics", "training", "benchmark").index)
    assert all("--max-rows" not in s.command for s in stages)
    assert all("sbatch" not in s.command for s in stages)


@pytest.mark.parametrize("skipped,mandatory,ok", [(0, True, True), (1, True, False), (1, False, True)])
def test_mandatory_vsc_tests_cannot_pass_by_skipping(tmp_path, skipped, mandatory, ok):
    report = tmp_path / "junit.xml"
    report.write_text(f'<testsuites><testsuite tests="5" skipped="{skipped}" failures="0" errors="0"/></testsuites>')
    if ok:
        assert v.check_junit(report, mandatory=mandatory)["tests"] == 5
    else:
        with pytest.raises(ValueError, match="coverage"):
            v.check_junit(report, mandatory=mandatory)


@pytest.mark.parametrize("exit_code", [0, 1, 64])
def test_child_exit_status_cannot_be_masked(tmp_path, exit_code):
    stage = v.Stage("child", (sys.executable, "-c", f"raise SystemExit({exit_code})"))
    if exit_code == 0:
        assert v.execute(stage, tmp_path / "log", timeout=20)["seconds"] >= 0
    else:
        with pytest.raises(RuntimeError, match="exit="):
            v.execute(stage, tmp_path / "log", timeout=20)


def test_timeout_terminates_child(tmp_path):
    stage = v.Stage("timeout", (sys.executable, "-c", "import time; time.sleep(60)"))
    with pytest.raises(TimeoutError):
        v.execute(stage, tmp_path / "timeout.log", timeout=.1)


def fake_campaign(monkeypatch, isolated_output):
    import filelock  # noqa: F401 -- select this host's lock backend before mocking sys.platform

    import src.utils.environment as environment
    import src.utils.preflight as preflight

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SLURM_JOB_ID", "test123")
    monkeypatch.setattr(preflight, "source_fingerprint", lambda: "source")
    monkeypatch.setattr(environment, "environment_fingerprint", lambda: "env")
    monkeypatch.setattr(v, "input_inventory", lambda: {"bytes": "inputs"})
    artifact = isolated_output / "evidence"
    artifact.write_text("valid")
    from src.eval.benchmark_status import digest
    monkeypatch.setattr(v, "collect_artifacts", lambda _, phase="gpu": {str(artifact): digest(artifact)})
    monkeypatch.setattr(v, "plan", lambda phase="gpu": [v.Stage("preflight", ()), v.Stage("check", ()),
                                          v.Stage("train", (), phase="training")])
    monkeypatch.setattr(v, "execute", lambda *_a, **_k: {})
    assert v.run_validation(phase="cpu") == 0
    return artifact


def test_failed_diagnostic_blocks_training_and_cannot_reuse_older_pass(monkeypatch, isolated_output):
    fake_campaign(monkeypatch, isolated_output)
    ran = []

    def execute(stage, *_args, **_kwargs):
        ran.append(stage.name)
        if stage.name == "check":
            raise RuntimeError("actual GPU failure")
        return {}

    v.write_json(v.receipt_path(), {"status": "PASS", "attempt": "old"})
    monkeypatch.setattr(v, "execute", execute)
    assert v.run_validation() == 1
    assert ran == ["preflight", "check"]
    receipt = json.loads(v.receipt_path().read_text())
    assert receipt["status"] == "FAILED" and receipt["attempt"] != "old"
    assert not v.readiness()[0].ok


def test_gpu_refuses_to_start_without_a_current_cpu_pass(monkeypatch, isolated_output):
    fake_campaign(monkeypatch, isolated_output)
    v.receipt_path("cpu").unlink()
    ran = []
    monkeypatch.setattr(v, "execute", lambda stage, *_a, **_k: ran.append(stage.name))
    assert v.run_validation(phase="gpu") == 1
    assert ran == []


def test_rerunning_cpu_invalidates_the_older_gpu_pass(monkeypatch, isolated_output):
    fake_campaign(monkeypatch, isolated_output)
    assert v.run_validation(phase="gpu") == 0
    assert v.readiness()[0].ok
    assert v.run_validation(phase="cpu") == 0
    assert not v.readiness()[0].ok


@pytest.mark.parametrize("changed", ["nothing", "artifact", "input", "source", "environment", "stage", "running"])
def test_readiness_requires_current_complete_evidence(monkeypatch, isolated_output, changed):
    import src.utils.environment as environment
    import src.utils.preflight as preflight

    artifact = fake_campaign(monkeypatch, isolated_output)
    monkeypatch.setattr(v, "execute", lambda *_a, **_k: {})
    assert v.run_validation() == 0
    if changed == "artifact":
        artifact.write_text("modified")
    if changed == "input":
        monkeypatch.setattr(v, "input_inventory", lambda: {"bytes": "other"})
    if changed == "source":
        monkeypatch.setattr(preflight, "source_fingerprint", lambda: "changed")
    if changed == "environment":
        monkeypatch.setattr(environment, "environment_fingerprint", lambda: "changed")
    if changed in ("stage", "running"):
        receipt = json.loads(v.receipt_path().read_text())
        if changed == "stage":
            receipt["stages"].pop()
        else:
            receipt["status"] = "RUNNING"
        v.write_json(v.receipt_path(), receipt)
    assert v.readiness()[0].ok == (changed == "nothing")
    assert v.ATTEMPT_ENV not in os.environ


def test_diagnostic_pass_cannot_overwrite_production_reference(monkeypatch, isolated_output):
    fake_campaign(monkeypatch, isolated_output)
    target = paths.big_outputs_dir() / "reference/benchmark/pd/results_reference_pd_linear.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("complete publication benchmark")
    assert v.run_validation(phase="gpu") == 0
    assert target.read_text() == "complete publication benchmark"


def test_download_verifier_reports_failed_and_unrun_stages(tmp_path):
    from src.utils.exp0_verify import FAIL, verify

    v.write_json(tmp_path / "experiment_0/report/validation.json", {
        "schema": v.SCHEMA, "attempt": "failed_attempt", "status": "FAILED",
        "stages": [{"name": "preflight", "status": "FAIL", "error": "missing package"}],
        "error": "Dependent work blocked",
    })
    result = verify(tmp_path)
    assert any(status == FAIL and "missing package" in detail for status, _, detail in result.items)
    assert any(status == FAIL and what == "GPU Experiment 0" for status, what, _ in result.items)


def test_invalid_manifest_replaces_old_pass_with_failed_receipt(monkeypatch, isolated_output):
    fake_campaign(monkeypatch, isolated_output)
    v.write_json(v.receipt_path(), {"status": "PASS", "attempt": "previous"})

    def broken_plan(phase="gpu"):
        raise ValueError("configuration is broken")

    monkeypatch.setattr(v, "plan", broken_plan)
    assert v.run_validation() == 1
    assert json.loads(v.receipt_path().read_text())["status"] == "FAILED"


def test_exp1_default_plan_includes_main_and_component_runs(monkeypatch, isolated_output, capsys):
    from src.utils import run_experiment

    monkeypatch.setattr(run_experiment, "queued_job_names", lambda: set())
    assert run_experiment.main(["1", "--phase", "train"]) == 0
    output = capsys.readouterr().out
    for task in paths.TASKS:
        assert f"exp1-{task}-train" in output and f"exp1ablations-{task}-train" in output
    assert output.count("0/9") == 2 and output.count("0/15") == 2
    assert "VARIANT=ablations" in output


@pytest.mark.parametrize("selected", [False, True])
def test_cluster_planner_check_accepts_exp3_before_and_after_selection(monkeypatch, selected):
    from types import SimpleNamespace

    from scripts.exp0_checks import Report, check_orchestration

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr("src.utils.run_experiment.unconfigured_tracks",
                        lambda exp, tasks: ["pd"] if exp == 3 and not selected else [])

    def run(command, **kwargs):
        blocked = command[-1] == "3" and not selected
        return SimpleNamespace(returncode=int(blocked), stdout="NOT CONFIGURED YET" if blocked else "ready")

    monkeypatch.setattr("scripts.exp0_checks.subprocess.run", run)
    report = Report()
    check_orchestration(report)
    assert len(report.rows) == 4 and report.failed == 0
