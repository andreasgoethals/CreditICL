"""No paid submission can be hidden inside the runtime compatibility probe."""
from types import SimpleNamespace
import os
import subprocess

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
        return SimpleNamespace(returncode=int(failure), stdout="", stderr="serializer" if failure else "OK")

    monkeypatch.setattr(slurm_client.subprocess, "run", run)
    if failure:
        with pytest.raises(RuntimeError, match="serializer"):
            slurm_client.check()
    else:
        slurm_client.check()
        assert len(calls) == 2
    assert all("--test-only" in args and "--array=0" in args for args in calls)


def test_both_validation_phases_include_submission_client_check():
    from src.validation.campaign import plan
    for phase in ("cpu", "gpu"):
        assert any(s.name == "slurm-client" for s in plan(phase))


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
