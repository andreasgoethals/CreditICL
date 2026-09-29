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
    """Every name passed to `--models` anywhere in the SLURM layer — literal, or through the
    benchmark's `${MODEL}`, which is a reference model or `benchmark_status.arm_model`'s choice."""
    from src.eval.benchmark_status import REFERENCE_MODELS, arm_model

    names: set[str] = set()
    for job in JOB_SCRIPTS:
        joined = re.sub(r"\\\s*\n\s*", " ", job.read_text(encoding="utf-8"))
        for m in re.finditer(r'--models\s+"?([a-z0-9_,]+)"?', joined):
            names.update(n for n in m.group(1).split(",") if n)
        if '--models "${MODEL}"' in joined:
            ref = re.search(r'REFERENCE_MODELS="\$\{REFERENCE_MODELS:-([a-z0-9_,]+)\}"', joined)
            names.update((ref.group(1) if ref else REFERENCE_MODELS).split(","))
            names.update(arm_model({"architecture": a}) for a in ("tabicl", "tabpfn3"))
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


def test_no_two_sweep_arms_do_the_same_thing():
    """At `credit_fraction: 0.0` no dataset comes from our prior, so every credit lever is
    dead — and the grid was scheduling EIGHT identical control runs per seed. 24 arms of 96
    where 3 would do: bit-identical output for 22 % of the compute budget."""
    from src.utils.config import effective_fingerprint, expand_with_seeds, load

    for name in ("Exp1_LGD.yaml", "Exp1_PD.yaml"):
        runs = expand_with_seeds(load(ROOT / "config" / name))
        prints = [effective_fingerprint(r) for r in runs]
        assert len(prints) == len(set(prints)), f"{name} still schedules duplicate arms"

        # cf=0 kills every credit lever, so a control can still differ only by the FILTER
        # (which applies to the base prior too) and the SEED — 3 filter modes x the seeds.
        controls = [r for r in runs if float(r["prior"].get("credit_fraction", 0)) == 0.0]
        keys = {(r["prior"]["filter"]["mode"], r["seed"]) for r in controls}
        assert len(controls) == len(keys), (
            f"{name}: {len(controls)} control arms but only {len(keys)} distinct (filter, seed) — "
            f"a credit lever is wrongly varying the control"
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


# ---------------------------------------------------------------------------------------
# Two phases, and the order is a fact about the data: phase 2 scores what phase 1 wrote.
# ---------------------------------------------------------------------------------------

BENCH = "benchmark.slurm"


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_phase_one_only_trains(name):
    """Evaluation lived in the training launcher for one day (25-08-2026) and was moved out.

    Scoring inside the training job means each arm is benchmarked by whatever the code looked
    like the hour it happened to finish, against a reference scored on a different day. Every
    comparison this project got wrong, it got wrong exactly that way.
    """
    text = (ROOT / "scripts" / "slurm" / name).read_text(encoding="utf-8")
    assert "evaluate.py" not in text
    assert "evaluate_ood.py" not in text
    assert BENCH in text, "it must point the reader at the phase that does score"


def test_phase_two_covers_every_checkpoint_plus_the_reference_models():
    """One slot per (arm, saved checkpoint) — final checkpoints first — then one per reference
    model (released TabICLv2, TabPFN-3, CatBoost, linear). The `--array` default must cover
    exactly Exp1's slots; every index past them must exit rather than race."""
    from src.eval.benchmark_status import n_slots, slot_for

    text = (ROOT / "scripts" / "slurm" / BENCH).read_text(encoding="utf-8")
    spec = next(ln for ln in text.splitlines() if ln.startswith("#SBATCH --array="))
    hi = int(spec.split("=")[1].split("%")[0].split("-")[1])
    for track in ("lgd", "pd"):
        total = n_slots(1, track)
        assert hi == total - 1, f"--array=0-{hi} should be 0-{total - 1} for Exp1 {track}"
        first = slot_for(1, track, 0)
        assert first.kind == "arm" and first.final and first.tag == f"exp1bench_{track}_a0"
        assert slot_for(1, track, total - 1).kind == "reference"
        assert slot_for(1, track, total).kind == "none"
    assert '"$SLOT_KIND" == "none"' in text, "indices past the last slot must exit"


@pytest.mark.parametrize("exp,track", [(1, "LGD"), (1, "PD"), (2, "LGD"), (3, "LGD")])
def test_phase_two_is_shared_by_all_three_experiments(exp, track):
    """One benchmark, three experiments. The config is chosen by `EXP` and `TRACK`, so Exp3 and
    Exp2 are scored by the same code and against the same reference column as Exp1."""
    from src.utils.config import expand_with_seeds, load

    text = (ROOT / "scripts" / "slurm" / BENCH).read_text(encoding="utf-8")
    assert 'CONFIG="config/Exp${EXP}_' in text
    # and the config it would pick must exist and expand
    cfg = ROOT / "config" / f"Exp{exp}_{track}.yaml"
    assert cfg.is_file()
    assert len(expand_with_seeds(load(cfg, allow_placeholders=True))) > 0


def test_the_reference_column_is_scored_once_and_reused():
    """CatBoost, TabPFN-3, released TabICLv2 and logistic/linear do not depend on our prior, so
    their numbers are identical across Exp1/2/3. Rescoring per experiment would waste GPU time
    AND produce three slightly different reference columns to compare against."""
    from src.eval.benchmark_status import n_slots, slot_for

    text = (ROOT / "scripts" / "slurm" / BENCH).read_text(encoding="utf-8")
    assert "src.eval.benchmark_status" in text and "FORCE_REFERENCE" in text
    assert "tabiclv2,tabpfn3,catboost,linear" in text
    total = n_slots(1, "pd")
    refs = [slot_for(1, "pd", i) for i in range(total - 4, total)]
    assert [r.tag for r in refs] == [f"reference_pd_{m}" for m in ("tabiclv2", "tabpfn3", "catboost", "linear")]
    assert all("exp" not in r.tag for r in refs), "the tag must not mention the experiment"


def test_phase_two_gives_every_model_the_whole_training_pool():
    """Protocol 3: no context cap anywhere, the same fold seed in all four scoring calls."""
    text = (ROOT / "scripts" / "slurm" / BENCH).read_text(encoding="utf-8")
    assert "--max-context-rows" not in text and "CONTEXT_CAP" not in text
    assert text.count('--seeds "${SEEDS}"') == 4


def test_phase_two_refuses_an_arm_that_never_finished():
    """A SIGUSR1 checkpoint loads perfectly and is not a result. The arm summary is the
    authority, the same one `sweep_status` reads (the arm's run folder)."""
    text = (ROOT / "scripts" / "slurm" / BENCH).read_text(encoding="utf-8")
    assert "run_summary_path" in text
    assert 'data.get("completed")' in text
    assert "is NOT complete" in text


def test_phase_two_scores_our_checkpoints_not_the_released_ones():
    """The whole point. `--models crediticl` with an explicit `--checkpoint`, and a hard error
    when the directory is empty rather than a silent fall-through to the download."""
    from src.eval.benchmark_status import arm_model

    text = (ROOT / "scripts" / "slurm" / BENCH).read_text(encoding="utf-8")
    # The wrapper follows the arm's architecture: ours for TabICL, TabPFN's own for TabPFN-3.
    assert '--models "${MODEL}"' in text and "--models crediticl" not in text
    assert arm_model({"architecture": "tabicl"}) == "crediticl"
    assert arm_model({"architecture": "tabpfn3"}) == "tabpfn3"
    assert '--checkpoint "$CKPT"' in text
    # The slot's own saved step, never a rolling checkpoint picked by a lexical sort.
    assert "SKIPPED: no checkpoint under" in text
    assert 'CKPT="${CKPT_DIR}/step-${STEP}.ckpt"' in text
    assert 'if [ ! -f "$CKPT" ]' in text


# ---------------------------------------------------------------------------------------
# Experiment 0 — the cluster debug suite (replaced debug_exp1.slurm on 29-09-2026)
# ---------------------------------------------------------------------------------------


def test_exp0_runs_every_check_and_the_equivalence_on_the_production_gpu():
    d = _directives("exp0.slurm")
    assert d["clusters"] == "mindwell" and d["partition"] == "gpu_b200"
    assert int(d["cpus-per-task"]) <= GPU_B200_MAX_CORES
    text = (SLURM / "exp0.slurm").read_text(encoding="utf-8")
    assert "scripts/exp0_checks.py" in text
    assert "scripts/check_equivalence.py" in text and "for TRACK in PD LGD" in text
    # a failing check must not skip the equivalence runs, and the job still reports the failure
    run = text.index("python -u scripts/exp0_checks.py")
    assert text.index("\nset +e\n") < run < text.index("\nset -e\n", run)
    assert 'exit "${CHECKS_RC}"' in text
    assert "EXP=0" in text


def test_exp0_configs_cover_every_training_path():
    """One tiny arm per path the experiments use — or a path could first break in a real run."""
    from src.utils.config import expand_with_seeds, load

    for track in ("PD", "LGD"):
        runs = expand_with_seeds(load(ROOT / "config" / f"Exp0_{track}.yaml"))
        kinds = {(r["architecture"], r["init"]["strategy"]) for r in runs}
        assert kinds == {("tabicl", "scratch"), ("tabicl", "full"), ("tabpfn3", "full")}, kinds
        for r in runs:
            assert r["_run_name"].startswith(f"exp0_{track.lower()}__"), "must land in experiment_0/"
            assert r["train"]["max_steps"] <= 1000, "a check, not an experiment"
            assert r["progress"]["every_datasets"] > 0, "the progress curve must be exercised"
            assert r["logging"]["log_weights_every"] > 0
            if r["architecture"] == "tabpfn3":
                assert r["prior"]["encoding"] == "raw"
        # every dataset: the benchmark check measures a real slot's cost
        assert runs[0]["eval"]["holdout_datasets"], "Exp0's benchmark must score every dataset"


def test_every_training_launcher_can_run_the_exp2_search():
    for name in LAUNCHERS:
        text = (SLURM / name).read_text(encoding="utf-8")
        assert 'VARIANT="${VARIANT:-}"' in text
        assert "${VARIANT:+_${VARIANT}}" in text
        assert "VARIANT=${VARIANT}" in text, "a resubmitted search arm must stay in the search"
    bench = (SLURM / BENCH).read_text(encoding="utf-8")
    assert '--variant "$VARIANT"' in bench and "${VARIANT:+_${VARIANT}}" in bench
