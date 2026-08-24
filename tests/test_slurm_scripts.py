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


def test_no_two_sweep_arms_do_the_same_thing():
    """At `credit_fraction: 0.0` no dataset comes from our prior, so every credit lever is
    dead — and the grid was scheduling EIGHT identical control runs per seed. 24 arms of 96
    where 3 would do: bit-identical output for 22 % of the compute budget."""
    from src.utils.config import effective_fingerprint, expand_with_seeds, load

    for name in ("Exp1_LGD.yaml", "Exp1_PD.yaml"):
        runs = expand_with_seeds(load(ROOT / "config" / name))
        prints = [effective_fingerprint(r) for r in runs]
        assert len(prints) == len(set(prints)), f"{name} still schedules duplicate arms"

        controls = [r for r in runs if float(r["prior"].get("credit_fraction", 0)) == 0.0]
        seeds = {r["seed"] for r in controls}
        assert len(controls) == len(seeds), (
            f"{name}: {len(controls)} control arms across {len(seeds)} seeds — "
            f"one per seed is the most that can differ"
        )


def test_debug_arms_point_where_the_comments_say():
    """`ARMS` is hard-coded, and the comment above it claims what each index is. Deduplicating
    the sweep moved the quantile arm from 11 to 9; an index that silently points at a
    different arm is worse than a crash, so the pairing is checked rather than trusted."""
    import re

    from src.utils.config import expand_with_seeds, load

    text = (SLURM / "debug_exp1.slurm").read_text(encoding="utf-8")
    claimed = {
        int(m.group(1)): (m.group(2), m.group(3))
        for m in re.finditer(r"^#\s+(\d+)\s+cf=([\d.]+)\s+(mechanism|quantile)\b", text, re.M)
    }
    assert claimed, "the ARMS mapping comment is missing or reformatted"

    arms = [int(x) for x in re.search(r"^ARMS=\(([^)]*)\)", text, re.M).group(1).split()]
    assert set(arms) == set(claimed), f"ARMS={arms} does not match the documented {sorted(claimed)}"

    runs = expand_with_seeds(load(ROOT / "config" / "Exp1_LGD.yaml"))
    for index, (cf, mode) in claimed.items():
        assert index < len(runs), f"index {index} is past the end of a {len(runs)}-run sweep"
        run = runs[index]
        assert float(run["prior"]["credit_fraction"]) == float(cf), (
            f"index {index} claims cf={cf}, grid says {run['prior']['credit_fraction']}"
        )
        # the control ignores mode entirely, so only check it where it can act
        if float(cf) > 0:
            assert run["prior"]["credit"]["target"]["mode"] == mode, (
                f"index {index} claims {mode}, grid says "
                f"{run['prior']['credit']['target']['mode']}"
            )


def test_debug_job_sets_the_micro_batch_from_the_partition():
    """This edit silently failed to apply TWICE — a heredoc whose backslash-continuations did
    not match, and then one that wrote a literal `\n`. `bash -n` passed both times because the
    result was still valid shell, and no test looked, so a run went out at micro_batch_size 4
    on a 183 GB card.

    It matters: at batch 4 the B200 sat at 3.5 % utilisation and 2.2 datasets/s; at batch 64 it
    reached 89 % and 25.6. Launch size is the whole story on a big GPU.
    """
    text = (SLURM / "debug_exp1.slurm").read_text(encoding="utf-8")
    assert "--micro-batch-size" in text, "the job must pass a micro-batch size"
    literal_backslash_n = chr(92) + "n"   # the two characters, not a newline
    assert literal_backslash_n not in text, (
        "a literal backslash-n crept into the script — a heredoc wrote the escape "
        "instead of a line break, and bash -n still accepted it"
    )
    # Upstream's 4. MEASURED: both the 3.5 %-utilisation run and the 88.7 % one used
    # micro 4 — what fills the GPU is the number of consecutive passes before a
    # synchronising optimiser step (1 vs 16), not the size of each pass. At 88.7 %
    # there is ~1.1x left, so deviating from upstream would buy almost nothing.
    assert 'MICRO="${MICRO:-4}"' in text, "the micro-batch should match upstream's 4"

    # every continuation must end the line, or the next flag is swallowed as an argument
    call = text[text.index("python scripts/pretrain.py \\") :]
    call = call[: call.index("STATUS=$?")]
    for line in call.splitlines()[:-1]:
        assert line.rstrip().endswith("\\"), f"broken continuation: {line!r}"


def test_debug_steps_fit_the_walltime_at_the_configured_batch():
    """The debug budget is `steps x batch_size`. batch_size went 4 -> 64, so 1,500 steps became
    96,000 datasets and job 11518236 was KILLED at the one-hour wall on step 1,422 — losing the
    evaluation and the checkpoint after 62 minutes. A debug run has to finish."""
    import yaml

    text = (SLURM / "debug_exp1.slurm").read_text(encoding="utf-8")
    steps = int(re.search(r'DEBUG_STEPS="\$\{DEBUG_STEPS:-(\d+)\}"', text).group(1))
    batch = yaml.safe_load(
        (ROOT / "config" / "Exp1_LGD.yaml").read_text(encoding="utf-8")
    )["train"]["batch_size"]

    # 0.4 steps/s measured on the B200 at batch 64; leave a third of the hour for evaluation.
    budget_s = steps / 0.4
    assert budget_s < 2400, (
        f"{steps} steps x batch {batch} is ~{budget_s / 60:.0f} min of training at the "
        f"measured 0.4 steps/s, leaving nothing for the evaluation inside a 1 h walltime"
    )


# ---------------------------------------------------------------------------------------
# The Exp1 launchers. Each of these pinned a real defect found on 24-08-2026: the scripts
# said "TARGET: Mindwell B200" in a banner and then submitted `--clusters=wice
# --partition=gpu_a100`, with wICE's core and memory limits and `--array=0-47` for a sweep
# that expands to 75.
# ---------------------------------------------------------------------------------------

#: Mindwell gpu_b200, from the VSC "CPU resource limits in GPU jobs" table.
GPU_B200_MAX_CORES = 24
GPU_B200_MAX_MEM_MIB = 194_400
#: Mindwell's general cap. Only `*_long` partitions may exceed it, and gpu_b200 is not one.
MINDWELL_MAX_WALLTIME_H = 72

LAUNCHERS = {"pretrain_lgd.slurm": "LGD", "pretrain_pd.slurm": "PD"}


def _directives(name: str) -> dict[str, str]:
    text = (ROOT / "scripts" / "slurm" / name).read_text(encoding="utf-8")
    out = {}
    for line in text.splitlines():
        if line.startswith("#SBATCH "):
            body = line[len("#SBATCH "):].strip()
            key, _, val = body.partition("=")
            out[key.lstrip("-")] = val
    return out


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_launcher_targets_the_cluster_its_banner_claims(name):
    d = _directives(name)
    assert d["clusters"] == "mindwell"
    assert d["partition"] == "gpu_b200"


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_launcher_stays_inside_the_per_gpu_resource_limits(name):
    """Asking for more than the documented share per GPU earns a warning from VSC."""
    d = _directives(name)
    assert int(d["cpus-per-task"]) <= GPU_B200_MAX_CORES
    mem_gb = int(d["mem"].rstrip("Gg"))
    assert mem_gb * 1024 <= GPU_B200_MAX_MEM_MIB
    hours = int(d["time"].split(":")[0])
    assert hours <= MINDWELL_MAX_WALLTIME_H


@pytest.mark.parametrize("name, track", sorted(LAUNCHERS.items()))
def test_array_range_matches_the_grid_it_submits(name, track):
    """`--array=0-47` against a 75-arm sweep silently drops 27 arms and nobody notices until
    the results table is short."""
    # `expand_with_seeds`, the same call `pretrain.py --list` makes: it crosses the grid,
    # multiplies by seeds, AND collapses the arms that would do the same thing.
    from src.utils.config import expand_with_seeds, load

    cfg = load(ROOT / "config" / f"Exp1_{track}.yaml", allow_placeholders=True)
    n_runs = len(expand_with_seeds(cfg))
    d = _directives(name)
    span, _, throttle = d["array"].partition("%")
    lo, _, hi = span.partition("-")
    assert int(lo) == 0
    assert int(hi) == n_runs - 1, f"{name}: --array=0-{hi} but the grid expands to {n_runs}"
    assert throttle and 0 < int(throttle) <= 24, "throttle to at most the 24 B200s that exist"


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_launcher_survives_a_kill(name):
    """A 75-arm sweep spanning days WILL meet the walltime, a node failure, or an
    unannounced maintenance drain. All three arrive as a signal; none may cost an arm."""
    d = _directives(name)
    text = (ROOT / "scripts" / "slurm" / name).read_text(encoding="utf-8")
    assert "requeue" in d, "Slurm must be allowed to put a killed task back in the queue"
    assert d["signal"].startswith("B:USR1@"), "the trainer needs warning before the walltime"
    # Slurm signals the batch script, not python: the trap must forward it, and the child
    # must be backgrounded or `wait` never returns in time.
    assert "kill -USR1" in text
    # Backgrounded, or `wait` would not return until the child exited anyway.
    assert "TRAIN_PID=$!" in text
    assert any(line.rstrip().endswith("&") for line in text.splitlines())
    # Exit 64 = "saved, not finished". Resubmitting is what makes the arm outlast the wall.
    assert 'RC" -eq 64' in text and "sbatch --clusters=" in text


def test_the_trainer_actually_produces_exit_64():
    """The job script's resubmission branch is dead code unless pretrain.py emits the code."""
    src = (ROOT / "scripts" / "pretrain.py").read_text(encoding="utf-8")
    assert "return 64" in src
    assert 'summary.get("completed", True)' in src


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_workers_come_from_the_allocation_not_the_config(name):
    """The prior is generated on the CPU, so `num_workers` should be whatever the partition
    gave us: 23 on gpu_b200, 17 on gpu_a100, 7 on interactive. The config's 12 would waste
    half a B200 node and oversubscribe a smaller card."""
    text = (ROOT / "scripts" / "slurm" / name).read_text(encoding="utf-8")
    assert "SLURM_CPUS_PER_TASK:-8} - 1" in text
    assert '--num-workers "${WORKERS}"' in text


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_a_resumed_arm_goes_back_to_the_same_cluster(name):
    """`--clusters=mindwell` was hard-coded in the resubmit line, so an arm started on wICE
    would have resumed on Mindwell — a different GPU mid-run, which makes its own timings
    incomparable and can meet a different memory ceiling."""
    text = (ROOT / "scripts" / "slurm" / name).read_text(encoding="utf-8")
    assert 'sbatch --clusters="${SLURM_CLUSTER_NAME:-mindwell}"' in text
    assert '--partition="${SLURM_JOB_PARTITION:-gpu_b200}"' in text
    assert "sbatch --clusters=mindwell --array=" not in text
