"""No paid submission can be hidden inside the runtime compatibility probe."""
import os
import subprocess
from types import SimpleNamespace

import pytest

from src.validation import slurm_client


@pytest.mark.parametrize("failure", [False, True])
def test_submission_probe_is_dry_and_rejects_plugin_errors(monkeypatch, failure):
    monkeypatch.setenv("CREDITICL_IN_CONTAINER", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] == 90
        failed = failure and args[0] == "sbatch"
        return SimpleNamespace(returncode=int(failed), stdout="OK", stderr="serializer" if failed else "")

    monkeypatch.setattr(slurm_client.subprocess, "run", run)
    if failure:
        with pytest.raises(RuntimeError, match="serializer"):
            slurm_client.check()
    else:
        slurm_client.check()
        assert len(calls) == 3
    assert calls[0][0] == "sam-balance"
    assert all("--test-only" in args and "--array=0" in args for args in calls[1:])


def test_both_validation_phases_include_submission_client_check():
    from src.validation.campaign import plan
    for phase in ("cpu", "gpu"):
        assert any(s.name == "slurm-client" for s in plan(phase))


def test_submission_warning_about_disabled_credit_check_is_not_success(monkeypatch):
    monkeypatch.setenv("CREDITICL_IN_CONTAINER", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(slurm_client.subprocess, "run", lambda args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="OK", stderr="credit balance check is not enforced"))
    with pytest.raises(RuntimeError, match="not enforced"):
        slurm_client.check()


def test_dry_run_capacity_refusal_waits_for_actual_success(monkeypatch, capsys):
    monkeypatch.setenv('CREDITICL_IN_CONTAINER', '1')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('CREDITICL_EXP0_ATTEMPT', 'cpu_fixture')
    calls, sleeps = [], []
    def run(args, **kwargs):
        calls.append(args)
        refused = len(calls) == 3
        return SimpleNamespace(returncode=int(refused), stdout='',
            stderr='QOSMaxSubmitJobPerUserLimit\nallocation failure: Job violates accounting/QOS policy'
            if refused else 'OK') if args[0] == 'sbatch' else SimpleNamespace(returncode=0, stdout='balance', stderr='')
    monkeypatch.setattr(slurm_client.subprocess, 'run', run)
    monkeypatch.setattr(slurm_client.time, 'sleep', sleeps.append)
    slurm_client.check()
    assert sleeps == [60]
    assert calls[2] == calls[3] and '--test-only' in calls[3]
    assert '[WAIT]' in capsys.readouterr().out


@pytest.mark.parametrize('cpu_current', [False, True])
def test_paid_gpu_never_idles_for_capacity_and_requires_current_cpu_proof(monkeypatch, cpu_current, capsys):
    from src.validation import campaign
    monkeypatch.setenv('CREDITICL_IN_CONTAINER', '1')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('CREDITICL_EXP0_ATTEMPT', 'gpu_fixture')
    monkeypatch.setattr(campaign, 'readiness', lambda phase: [SimpleNamespace(ok=cpu_current)] if phase == 'cpu' else [])
    monkeypatch.setattr(slurm_client.subprocess, 'run', lambda args, **kwargs:
        SimpleNamespace(returncode=1, stdout='', stderr='QOSMaxSubmitJobPerUserLimit') if args[0] == 'sbatch'
        else SimpleNamespace(returncode=0, stdout='balance', stderr=''))
    monkeypatch.setattr(slurm_client.time, 'sleep', lambda _: pytest.fail('paid GPU must not wait for admission'))
    if cpu_current:
        slurm_client.check()
        assert 'admission refused' in capsys.readouterr().out
    else:
        with pytest.raises(RuntimeError, match='CPU resource validation is missing'):
            slurm_client.check()


@pytest.mark.parametrize('error', ["Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
                                  'QOSMaxSubmitJobPerUserLimit; sam-balance failed'])
def test_ambiguous_resource_or_credit_failure_is_never_retried(monkeypatch, error):
    monkeypatch.setenv('CREDITICL_IN_CONTAINER', '1')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(slurm_client.subprocess, 'run', lambda args, **kwargs:
        SimpleNamespace(returncode=1 if args[0] == 'sbatch' else 0, stdout='balance', stderr=error))
    monkeypatch.setattr(slurm_client.time, 'sleep', lambda _: pytest.fail('unsafe retry'))
    with pytest.raises(RuntimeError, match='Slurm submission client failed'):
        slurm_client.check()


def test_cpu_validation_fits_verified_free_debug_allocation():
    script = (slurm_client.REPO_ROOT / "scripts/slurm/exp0_cpu.slurm").read_text()
    assert "#SBATCH --partition=interactive" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=48G" in script
    assert "--gpus-per-node" not in script


def test_container_uses_current_host_configuration_not_inherited_container_path(tmp_path):
    wrapper = slurm_client.REPO_ROOT / "scripts/slurm/container_python"
    subprocess.run(["bash", "-n", str(wrapper)], check=True, capture_output=True)
    text = wrapper.read_text()
    selection = "unset APPTAINERENV_SLURM_CONF" + text.split("unset APPTAINERENV_SLURM_CONF", 1)[1].split("for name in", 1)[0]
    (tmp_path / "slurm.conf").write_text("ClusterName=test\n")
    env = dict(os.environ, SLURM_CONF="slurm.conf", APPTAINERENV_SLURM_CONF="/obsolete/login/slurm.conf")
    result = subprocess.run(["bash", "-c", selection + '\nprintf "%s\\n%s" "$SLURM_CONF" "${APPTAINERENV_SLURM_CONF:-unset}"'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert lines[0].endswith("/slurm.conf") and lines[1] == "unset"


@pytest.mark.parametrize("allocated,queued,ok", [(False, "", True), (True, "", False),
                                                (False, "crediticl-exp2-pd", False),
                                                (False, "another-project", True)])
def test_writable_maintenance_requires_login_and_no_project_consumers(allocated, queued, ok):
    text = (slurm_client.REPO_ROOT / "scripts/slurm/container_python").read_text()
    guard = "_checkout_mode=ro" + text.split("_checkout_mode=ro", 1)[1].split("runner=(", 1)[0]
    script = 'squeue() { printf "%s" "$TEST_QUEUE"; };\n' + guard + '\nprintf "%s" "$_checkout_mode"'
    env = dict(os.environ, CREDITICL_MAINTENANCE="1", TEST_QUEUE=queued)
    env.pop("SLURM_JOB_ID", None)
    if allocated:
        env["SLURM_JOB_ID"] = "123"
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == ok
    if ok:
        assert result.stdout == "rw"


@pytest.mark.parametrize("args,mode", [([], "ro"), (["-m", "src.utils.clean_run"], "ro"),
    (["-m", "src.utils.clean_run", "--reset-exp0", "--clean"], "rw")])
def test_only_explicit_cleanup_opens_idle_checkout_for_deletion(args, mode):
    text = (slurm_client.REPO_ROOT / "scripts/slurm/container_python").read_text()
    guard = "_checkout_mode=ro" + text.split("_checkout_mode=ro", 1)[1].split("runner=(", 1)[0]
    script = 'squeue() { return 0; };\n' + guard + '\nprintf "%s" "$_checkout_mode"'
    env = dict(os.environ)
    env.pop("CREDITICL_MAINTENANCE", None)
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", "-c", script, "bash", *args], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout == mode


def test_child_allocation_discards_parent_container_mounts_and_env():
    text = (slurm_client.REPO_ROOT / "scripts/slurm/container_python").read_text()
    setup = text.split("image=${CREDITICL_RUNTIME_IMAGE", 1)[0]
    env = dict(os.environ, APPTAINER_BIND="/tmp/parent-allocation", APPTAINERENV_SLURM_JOB_ID="old",
               SINGULARITYENV_PATH="/old/venv", SLURM_JOB_ID="current")
    script = setup + '\nprintf "%s|%s|%s|%s" "${APPTAINER_BIND:-unset}" "${APPTAINERENV_SLURM_JOB_ID:-unset}" "${SINGULARITYENV_PATH:-unset}" "$SLURM_JOB_ID"'
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
    assert result.stdout == "unset|unset|unset|current"
