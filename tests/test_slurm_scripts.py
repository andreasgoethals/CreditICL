"""The SLURM layer is the only code in this project that nothing runs before the cluster does.

A typo in a job script is not caught by an import, a linter, or any other test: it surfaces as a
job that queues, starts, and exits in twenty seconds with a usage message no one reads, having
burnt a scheduling slot and taught nothing. That happened on 14-08-2026 — `debug_exp1.slurm`
passed `--resume auto` to `scripts/pretrain.py`, which defines no such flag, so argparse exited 2
before a single training step ran. Eight jobs across two partitions did nothing at all.

These tests parse the job scripts as text and check them against the Python they invoke.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "scripts" / "slurm"

JOB_SCRIPTS = sorted(SLURM.glob("*.slurm")) + sorted(SLURM.glob("*.sh"))


def _accepted_flags(script: Path) -> set[str]:
    """Every long option the script's argparse defines.

    Read as text rather than imported: importing pulls in torch and the whole `src` tree for
    what is a one-line question, and several of these scripts have import-time side effects.
    """
    src = script.read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*\n?\s*"(--[a-z0-9-]+)"', src))


def _invocations() -> list[tuple[Path, str, set[str], int]]:
    """Every `python scripts/<name>.py …` call in the SLURM layer, with the flags it passes."""
    found: list[tuple[Path, str, set[str], int]] = []
    for job in JOB_SCRIPTS:
        raw = job.read_text(encoding="utf-8")
        # join backslash continuations so a multi-line call is one string
        joined = re.sub(r"\\\s*\n\s*", " ", raw)
        for lineno, line in enumerate(joined.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or "python " not in stripped:
                continue
            m = re.search(r"python\s+(scripts/[a-z_]+\.py)(.*)", stripped)
            if not m:
                continue
            flags = set(re.findall(r"(--[a-z0-9-]+)", m.group(2)))
            found.append((job, m.group(1), flags, lineno))
    return found


def test_there_are_invocations_to_check():
    """A regex that silently matches nothing would make every test below vacuously pass."""
    calls = _invocations()
    assert len(calls) >= 5, f"only found {len(calls)} python calls in {SLURM} — regex broken?"


@pytest.mark.parametrize(
    "job,target,flags",
    [(j, t, f) for j, t, f, _ in _invocations()],
    ids=[f"{j.name}:{t.split('/')[-1]}:{n}" for j, t, _, n in _invocations()],
)
def test_job_scripts_only_pass_flags_that_exist(job: Path, target: str, flags: set[str]):
    """THE test in this file. `--resume auto` cost eight jobs and two partitions."""
    script = ROOT / target
    assert script.exists(), f"{job.name} calls {target}, which does not exist"
    unknown = flags - _accepted_flags(script)
    assert not unknown, (
        f"{job.name} passes {sorted(unknown)} to {target}, which does not define "
        f"{'them' if len(unknown) > 1 else 'it'}. argparse exits 2 and the job dies before "
        f"doing any work. Accepted: {sorted(_accepted_flags(script))}"
    )


def _requested_models() -> set[str]:
    """Every name passed to `--models` anywhere in the SLURM layer."""
    names: set[str] = set()
    for job in JOB_SCRIPTS:
        joined = re.sub(r"\\\s*\n\s*", " ", job.read_text(encoding="utf-8"))
        for m in re.finditer(r'--models\s+"?([a-z0-9_,]+)"?', joined):
            names.update(n for n in m.group(1).split(",") if n)
    return names


def test_job_scripts_only_ask_for_models_that_exist():
    """A flag can exist and its VALUE still be wrong.

    `--models crediticl,tabiclv2` passes the flag check above, but `crediticl` is added to the
    registry by an explicit `register()` that no production caller made — so the evaluation
    died with an unknown-baseline error, masked by the job script's `|| echo WARNING`. The
    training would have finished and our own model would never have been scored.
    """
    from src.eval import baselines
    from src.eval.crediticl_baseline import register_or_warn

    # BASELINES is module-level state, so registering into it leaks across the whole test
    # session — `test_all_baselines_registered` asserts on the exact set and fails from a
    # dozen files away. Restore it.
    before = dict(baselines.BASELINES)
    try:
        register_or_warn()  # exactly what the entry points do
        requested = _requested_models()
        assert requested, "no --models found in the SLURM scripts — regex broken?"
        unknown = requested - set(baselines.BASELINES)
        assert not unknown, (
            f"the SLURM scripts ask for {sorted(unknown)}, which no entry point registers. "
            f"Registered: {sorted(baselines.BASELINES)}"
        )
    finally:
        baselines.BASELINES.clear()
        baselines.BASELINES.update(before)


@pytest.mark.parametrize("entry", ["evaluate.py", "evaluate_ood.py"])
def test_evaluation_entry_points_register_our_own_baseline(entry: str):
    """The registration is explicit by design, which makes it easy to forget — and it was."""
    text = (ROOT / "scripts" / entry).read_text(encoding="utf-8")
    assert "register_or_warn" in text, (
        f"scripts/{entry} never registers the 'crediticl' baseline, so --models crediticl "
        f"cannot resolve and OUR model is silently left out of its own experiment"
    )
    assert "register_crediticl(log)" in text, f"scripts/{entry} imports it but never calls it"


def test_every_job_script_is_valid_bash():
    """`bash -n` parses without executing. Catches an unclosed quote or a broken `case`."""
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash on this machine")
    for job in JOB_SCRIPTS:
        result = subprocess.run(
            [bash, "-n", str(job)], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"{job.name} does not parse:\n{result.stderr}"


def test_training_failure_does_not_abort_the_debug_script():
    """`set -e` would kill the script at a failed training call, so `STATUS=$?` would be dead
    code, the evaluation guard meaningless, and the artefact summary — the part you most want
    when a run has just failed — never printed."""
    text = (SLURM / "debug_exp1.slurm").read_text(encoding="utf-8")
    assert "set +e\npython scripts/pretrain.py" in text, (
        "the training call must run with `set +e` so STATUS can be inspected"
    )
    assert "STATUS=$?\nset -e" in text, "`set -e` must be restored immediately after"


def test_debug_job_passes_its_own_checkpoint_to_the_evaluation():
    """Registering the baseline only makes the NAME resolvable; it also needs a checkpoint.

    On 14-08-2026 none was passed, so every `crediticl` cell failed with "needs
    checkpoint=<path>" while the run reported "25/50 cells OK" and contained nothing at all
    about our own model — the only model the experiment is about.
    """
    text = (SLURM / "debug_exp1.slurm").read_text(encoding="utf-8")
    assert "--dry-run" in text and "CKPT_DIR=" in text, (
        "the job must read the checkpoint directory from --dry-run so it cannot drift "
        "from the directory training actually wrote to"
    )
    assert '"${CKPT_ARG[@]}"' in text, "the resolved checkpoint must reach both evaluations"
    assert text.count('"${CKPT_ARG[@]}"') >= 2, "credit AND out-of-domain evaluation need it"
