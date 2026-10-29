"""Execution and provenance tests for the GPU Experiment 0 gate."""
import json
import os
import signal
import sys

import pytest

from src.utils import paths
from src.validation import campaign as v


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
def test_exp3_selection_receipts_do_not_change_code_but_any_recipe_edit_does(tmp_path, monkeypatch, task):
    from src.utils.preflight import source_fingerprint
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "scripts/slurm").mkdir(parents=True)
    (tmp_path / "scripts/slurm/container_python").write_text("#!/bin/bash\n")
    folder = tmp_path / "config"
    folder.mkdir()
    file = folder / f"Exp3_{task}.yaml"
    template = "design: retention\nprior:\n  credit_fraction: 0.5\ntrain:\n  max_steps: 10000\n"
    file.write_text(template)
    original = source_fingerprint()
    receipts = tmp_path / "output CreditICL" / "experiment_3" / "selection"
    receipts.mkdir(parents=True)
    (receipts / "adapters.json").write_text('{"choices": {}}')
    assert source_fingerprint() == original
    for before, after in (("0.5", "0.25"), ("10000", "10001"), ("retention", "cpt")):
        file.write_text(template.replace(before, after))
        assert source_fingerprint() != original


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
    assert phases == sorted(phases, key=("diagnostics", "capacity", "training", "benchmark").index)
    assert all("--max-rows" not in s.command for s in stages)
    assert all("sbatch" not in s.command for s in stages)
    capacity = [s for s in stages if s.phase == "capacity"]
    # Existing full-table benchmarks plus synthetic long-context corners and
    # one full heterogeneous production update for each track.
    from src.validation.long_context import CAPACITY_CORNERS
    assert len(capacity) == 2 * 4 * 3 + 2 * (2 * len(CAPACITY_CORNERS) + 3)
    assert all("--checkpoint" not in s.command for s in capacity)
    assert all("--max-context-rows" not in s.command for s in capacity)
    assert len({s.name for s in stages}) == len(stages)  # no repeated benchmark work


@pytest.mark.parametrize("skipped,mandatory,ok", [(0, True, True), (1, True, False), (1, False, False)])
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


def test_interruption_cleans_up_running_child(tmp_path, monkeypatch):
    processes = []
    popen = v.subprocess.Popen

    def launch(*args, **kwargs):
        processes.append(popen(*args, **kwargs))
        return processes[-1]

    def interrupt(_seconds):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(v.subprocess, "Popen", launch)
    monkeypatch.setattr(v.time, "sleep", interrupt)
    previous = signal.getsignal(signal.SIGTERM)
    stage = v.Stage("interrupted", (sys.executable, "-c", "import time; time.sleep(60)"))
    with pytest.raises(v.ValidationDeferred, match="SIGTERM"), v.interruption_signals():
        v.execute(stage, tmp_path / "interrupted.log", timeout=30)
    assert len(processes) == 1 and processes[0].poll() is not None
    assert signal.getsignal(signal.SIGTERM) == previous


def test_slow_benchmark_cell_fails_early_instead_of_repeating_allocation(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text("import os,time\nfrom src.eval.progress import atomic_json,PROGRESS_ENV\n"
                      "atomic_json(os.environ[PROGRESS_ENV],dict(cell='wide fold',started=time.time()-7201))\n"
                      "time.sleep(60)\n")
    stage = v.Stage("credit-pd-3", (sys.executable, str(script)), phase="capacity")
    with pytest.raises(RuntimeError, match="two-hour"):
        v.execute(stage, tmp_path / "logs" / "slow.log", timeout=30)


def test_expired_allocation_does_not_launch_a_new_process(monkeypatch, tmp_path):
    monkeypatch.setattr(v.subprocess, "Popen", lambda *a, **k: pytest.fail("launched after deadline"))
    with pytest.raises(v.ValidationDeferred):
        v.execute(v.Stage("unused", ()), tmp_path / "log", timeout=0)


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
    from src.utils.preflight import experiment_zero_checks
    assert not experiment_zero_checks(include_runs=False)[0].ok


@pytest.mark.parametrize("name, phase, cause", [
    ("vsc", "diagnostics", "CUDA resume failed"),
    ("unit", "diagnostics", "portable tests failed"),
    ("slurm-client", "diagnostics", "site credit check failed"),
    ("credit-pd-4", "capacity", "full-table capacity failed"),
])
def test_mandatory_gpu_failure_stops_dependent_work_and_prints_cause(
        monkeypatch, isolated_output, capsys, name, phase, cause):
    fake_campaign(monkeypatch, isolated_output)
    stages = [v.Stage("preflight", ()), v.Stage(name, (), phase=phase),
              v.Stage("train-pd", (), phase="training")]
    monkeypatch.setattr(v, "plan", lambda phase="gpu": stages)
    # Update the CPU receipt to the mocked manifest before testing the GPU phase.
    assert v.run_validation(phase="cpu") == 0
    ran = []

    def execute(stage, log, **kwargs):
        ran.append(stage.name)
        if stage.name == name:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"Traceback\nRuntimeError: {cause}\n", encoding="utf-8")
            raise RuntimeError("exit=1")
        return {}

    monkeypatch.setattr(v, "execute", execute)
    assert v.run_validation(phase="gpu") == 1
    assert ran == ["preflight", name]
    receipt = json.loads(v.receipt_path().read_text())
    failed = receipt["stages"][-1]
    assert failed["seconds"] >= 0 and cause in failed["log_tail"]
    assert cause in capsys.readouterr().out


def test_failure_details_include_failed_tests_and_bound_large_logs(tmp_path):
    log, junit = tmp_path / "unit.log", tmp_path / "unit.xml"
    log.write_text("older output\n" * 2000 + "last failure\n", encoding="utf-8")
    junit.write_text('<testsuites><testsuite><testcase classname="tests.test_x" name="test_bad">'
                     '<failure message="no"/></testcase></testsuite></testsuites>')
    stage = v.Stage("unit", (), junit=str(junit))
    details = v.failure_details(stage, log)
    assert details["failed_tests"] == ["tests.test_x::test_bad"]
    assert details["test_failures"] == [{"test": "tests.test_x::test_bad", "message": "no"}]
    assert details["log_tail"].endswith("last failure")
    assert len(details["log_tail"]) <= 8000 and len(details["log_tail"].splitlines()) <= 40
    log.unlink()
    junit.write_text("incomplete")
    assert v.failure_details(stage, log) == {"log": str(log)}


def test_test_assertion_survives_a_tail_full_of_warnings(monkeypatch, isolated_output, capsys):
    fake_campaign(monkeypatch, isolated_output)
    junit = v.receipt_path().parent / "vsc.xml"
    stages = [v.Stage("preflight", ()), v.Stage("vsc", (), junit=str(junit))]
    monkeypatch.setattr(v, "plan", lambda phase="gpu": stages)
    assert v.run_validation(phase="cpu") == 0

    def execute(stage, log, **kwargs):
        if stage.name == "vsc":
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("DeprecationWarning: unrelated warning\n" * 1000, encoding="utf-8")
            junit.write_text('<testsuites><testsuite><testcase classname="tests.test_vsc" name="precision">'
                             '<failure message="prediction mismatch"/></testcase></testsuite></testsuites>')
            raise RuntimeError("exit=1")
        return {}

    monkeypatch.setattr(v, "execute", execute)
    assert v.run_validation(phase="gpu") == 1
    assert "prediction mismatch" in capsys.readouterr().out
    status = v.readiness("gpu")[0]
    assert not status.ok
    assert "Failed stage: vsc" in status.detail
    assert "tests.test_vsc::precision" in status.detail
    assert "vsc.log" in status.detail


def test_test_assertions_are_bounded_even_with_many_large_failures(tmp_path):
    junit, log = tmp_path / "vsc.xml", tmp_path / "vsc.log"
    junit.write_text('<testsuites><testsuite>' + ''.join(
        f'<testcase name="test_{i}"><error>{"error " * 1000}</error></testcase>'
        for i in range(30)) + '</testsuite></testsuites>')
    details = v.failure_details(v.Stage("vsc", (), junit=str(junit)), log)
    assert len(details["failed_tests"]) == 20
    assert len(details["test_failures"]) == 3
    assert all(len(f["message"]) == 2000 for f in details["test_failures"])


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


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_publishes_incomplete_receipt_and_preserves_completed_stages(monkeypatch, isolated_output, sig):
    fake_campaign(monkeypatch, isolated_output)
    previous = signal.getsignal(sig)
    original = v.execute

    def interrupt(stage, log, **kwargs):
        if stage.name == "check":
            signal.getsignal(sig)(sig, None)
        return original(stage, log, **kwargs)

    monkeypatch.setattr(v, "execute", interrupt)
    assert v.run_validation() == 64
    saved = json.loads(v.receipt_path().read_text())
    assert saved["status"] == "DEFERRED"
    assert signal.Signals(sig).name in saved["error"]
    assert [s["name"] for s in saved["stages"]] == ["preflight"]
    assert not v.readiness()[0].ok
    assert signal.getsignal(sig) == previous
    monkeypatch.setattr(v, "execute", original)
    assert v.run_validation(resume=True) == 0
    assert json.loads(v.receipt_path().read_text())["attempt"] == saved["attempt"]


def test_old_running_receipt_is_identified_as_old_evidence_not_a_live_job(isolated_output):
    v.write_json(v.receipt_path(), {"schema": 2, "phase": "gpu", "status": "RUNNING",
                                  "attempt": "gpu_20261015T084304Z_j11692660_c29a3510"})
    row = v.readiness()[0]
    assert not row.ok
    assert "11692660" in row.detail and "old validation format" in row.detail
    assert "not a live scheduler status" in row.detail


def test_preflight_does_not_forbid_the_validation_continuation_it_recommends(monkeypatch, capsys):
    from src.utils import preflight

    monkeypatch.setattr(preflight, "run_checks", lambda **kw: [])
    monkeypatch.setattr(preflight, "experiment_zero_checks", lambda **kw: [
        preflight.Check("Experiment 0 gpu", False, "DEFERRED: resubmit the same GPU script")])
    assert preflight.main(["--require-exp0"]) == 1
    text = capsys.readouterr().out
    assert "resubmit the same GPU script" in text
    assert "Do not launch Experiments 1-3" in text
    assert "Do not submit GPU jobs" not in text


@pytest.mark.parametrize("hard_kill", [False, True])
def test_walltime_continues_same_campaign_and_reuses_verified_stages(monkeypatch, isolated_output, hard_kill):
    fake_campaign(monkeypatch, isolated_output)
    ran, pause = [], [True]

    def execute(stage, log, **kwargs):
        ran.append(stage.name)
        if stage.name == "check" and pause[0]:
            pause[0] = False
            raise v.ValidationDeferred("allocation ended")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"passed {stage.name}")
        return {}

    monkeypatch.setattr(v, "execute", execute)
    assert v.run_validation() == 64
    previous = json.loads(v.receipt_path().read_text())
    assert previous["status"] == "DEFERRED" and len(previous["stages"]) == 1
    assert not v.readiness()[0].ok
    assert "resubmit the same GPU script" in v.readiness()[0].detail
    if hard_kill:
        # Slurm SIGKILL can prevent the final DEFERRED publication. The last
        # atomic RUNNING receipt still has the same verified completed prefix.
        previous["status"] = "RUNNING"
        v.write_json(v.receipt_path(), previous)
    assert v.run_validation(resume=True) == 0
    current = json.loads(v.receipt_path().read_text())
    assert current["attempt"] == previous["attempt"]
    assert len(current["allocations"]) == 2
    assert ran == ["preflight", "check", "check", "train"]
    assert v.readiness()[0].ok
    assert v.run_validation(resume=True) == 0
    assert ran == ["preflight", "check", "check", "train"]


@pytest.mark.parametrize("change", ["source", "environment", "inputs", "cpu", "artifact", "failed"])
def test_continuation_refuses_changed_campaign_without_spending_compute(monkeypatch, isolated_output, change):
    fake_campaign(monkeypatch, isolated_output)

    def execute(stage, log, **kwargs):
        if stage.name == "check":
            raise v.ValidationDeferred("walltime")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("completed")
        return {}

    monkeypatch.setattr(v, "execute", execute)
    assert v.run_validation() == 64
    saved = json.loads(v.receipt_path().read_text())
    if change == "source":
        monkeypatch.setattr("src.utils.preflight.source_fingerprint", lambda: "changed")
    elif change == "environment":
        monkeypatch.setattr("src.utils.environment.environment_fingerprint", lambda: "changed")
    elif change == "inputs":
        monkeypatch.setattr(v, "input_inventory", lambda: {})
    elif change == "cpu":
        v.receipt_path("cpu").write_text("changed")
    elif change == "artifact":
        from pathlib import Path
        Path(next(iter(saved["stages"][0]["artifacts"]))).write_text("changed")
    else:
        saved["status"] = "FAILED"
        v.write_json(v.receipt_path(), saved)
    monkeypatch.setattr(v, "execute", lambda *a, **k: pytest.fail("spent compute on stale campaign"))
    assert v.run_validation(resume=True) == 1
    assert json.loads(v.receipt_path().read_text())["attempt"] == saved["attempt"]


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
    from src.validation.verify import FAIL, verify

    v.write_json(tmp_path / "experiment_0/report/validation.json", {
        "schema": v.SCHEMA, "attempt": "failed_attempt", "status": "FAILED",
        "stages": [{"name": "preflight", "status": "FAIL", "error": "missing package"}],
        "error": "Dependent work blocked",
    })
    result = verify(tmp_path)
    assert any(status == FAIL and "missing package" in detail for status, _, detail in result.items)
    assert any(status == FAIL and what == "GPU Experiment 0" for status, what, _ in result.items)


def test_download_verifier_never_trusts_success_footer_over_incomplete_receipt(tmp_path):
    from src.validation.verify import FAIL, WARN, verify

    v.write_json(tmp_path / "experiment_0/report/validation.json", {
        "schema": v.SCHEMA, "attempt": "interrupted_attempt", "status": "RUNNING",
        "stages": [{"name": "preflight", "status": "PASS"}],
    })
    log = tmp_path / "experiment_0/logs/exp0_gpu_old.log"
    log.parent.mkdir(parents=True)
    log.write_text("END status=OK\n")
    result = verify(tmp_path)
    assert any(status == FAIL and what == "GPU Experiment 0" for status, what, _ in result.items)
    assert any(status == WARN and "not a live scheduler check" in detail for status, _, detail in result.items)


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
    assert output.count("0/12") == 2 and output.count("0/15") == 2
    assert "VARIANT=ablations" in output


@pytest.mark.parametrize("selected", [False, True])
def test_cluster_planner_check_reports_configuration_state(monkeypatch, selected):
    from types import SimpleNamespace

    from src.validation.checks import Report, check_orchestration

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr("src.utils.run_experiment.unconfigured_tracks",
                        lambda exp, tasks: ["pd"] if exp == 3 and not selected else [])

    def run(command, **kwargs):
        blocked = command[-1] == "3" and not selected
        return SimpleNamespace(returncode=int(blocked), stdout="HAS INVALID CONFIGURATION" if blocked else "ready")

    monkeypatch.setattr("src.validation.checks.subprocess.run", run)
    report = Report()
    check_orchestration(report)
    assert len(report.rows) == 4 and report.failed == 0
