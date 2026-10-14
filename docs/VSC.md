# Running CreditICL on KU Leuven VSC

How to submit, what it costs, and how a multi-day sweep survives a cluster that stops routinely.
Distilled from `tfm-library/repositories/VSC Documentation.txt` (pin `de324efcca55be392feb0c01830b3b6ce364587f`, checked 13-10-2026) plus what we learned
running on it; where the docs are silent this file says so. **Verify charge rates and partitions
against the live docs before a large run.** The recovery procedure after *any* interruption is one
command — [§4.4](#44-recovery-is-one-command).

Job logs remain under `output CreditICL/`. A small `slurm-<name>-<job>.out/.err` in the
submission directory now captures startup failures before that logging can be established
(for example, an unwritable output directory). These fallback files are gitignored.

## Contents

1. [Cost, and which partition to use](#1-cost-and-which-partition-to-use)
2. [Account and cluster setup](#2-account-and-cluster-setup)
3. [Cores and memory per GPU](#3-cores-and-memory-per-gpu)
4. [Surviving a long sweep](#4-surviving-a-long-sweep)
5. [Storage: three tiers, and the Lustre-vs-GPFS rule](#5-storage)
6. [The Python environment](#6-the-python-environment)
7. [Job-script rules that bite](#7-job-script-rules-that-bite)
8. [Submitting: the scripts](#8-submitting-the-scripts)
9. [Compute budget](#9-compute-budget)
10. [Experiment 0: run it first](#10-experiment-0-run-it-first)
11. [Rename existing output, preserving completed work](#11-rename-existing-output-preserving-completed-work)
12. [Experiments 1, 2 and 3: the commands](#12-experiments-1-2-and-3-the-commands)

---

## 1. Cost, and which partition to use

**Quote every job before submitting** — `sam-quote sbatch …` returns the worst-case cost (job runs to
its full time limit). Billing is `(CPU_weight·effective_cores + GPU_weight·num_gpus) · walltime_minutes`,
where `effective_cores` is what you *block*, not what you request: request one core but a node's whole
memory and you are billed for every core.

| partition | cluster | GPUs (queue) | cores/GPU | credits/GPU-h | max walltime |
|---|---|---|---|---|---|
| `interactive` | mindwell | 4 × RTX 5000 Ada 32 GiB | 8 | **free** | 16 h |
| **`gpu_b200`** | mindwell | **24** × B200 192 GiB | **24** | 26,250 | 72 h |
| `gpu_a100` | wICE | 16 × A100 80 GiB | 18 | **8,500** | 72 h |
| `gpu_h100` | wICE | 20 × H100 80 GiB | 16 | 34,167 | 72 h |
| `gpu_a100_debug` | wICE | 1 × A100 | 18 | — | 1 h (one job only) |

GPU count describes capacity; queue wait depends on current demand. Our generator uses CPU cores,
so GPU speed alone does not predict the time or cost of an arm.

- **`gpu_b200` remains the production default:** 24 cores per GPU and 192 GiB of GPU memory.
  The H100 GPU charge is about **1.30×** the B200 charge, not 4×; its 16-core allocation may
  constrain generation. Choose between them using measured throughput and current queue state.
- **`gpu_a100` is cheaper per allocated hour**, but total cost depends on how much longer a job
  takes. Measure with `scripts/slurm/gpubench.slurm` and `src/utils/compare_gpubench.py`; use
  `sam-quote` for the actual allocation price. The H100 CPU estimate uses the published Zen4
  Genoa rate (3.472 credits/core-minute), not the A100 Ice Lake rate (2.546).
- The published table does not separately list the B200 hosts' Turin CPU rate. The launcher's
  existing B200 CPU estimate is provisional; the live `sam-quote` price is authoritative.

`interactive` costs nothing and on Mindwell is a *full* GPU (not wICE's ~1/7 MIG slice) — the right
home for debugging, data staging, and Lustre↔GPFS transfers. **The P100/V100 partitions, now on wICE, cannot run this
code**: the wheel is compiled for sm_75+ (P100 is sm_60, V100 sm_70), and bf16 AMP (a measured ~2×)
needs Ampere or later.

## 2. Account and cluster setup

- **Account `lp_verbekelab`** (has Mindwell access; billed under it since 14-08-2026).
- **SSH to `login.hpc.kuleuven.be`**, which now selects a wICE login node. Mindwell shares
  these login nodes. Genius retired at the end of September 2026; its P100/V100 nodes moved to
  wICE. Submit with explicit `#SBATCH --clusters=wice` or `mindwell`; every `.slurm` here does.
- The relevant library source sections are `source/leuven/tier2_hardware/tier2_login_nodes.rst`,
  `source/leuven/old_hardware/genius/genius_hardware.rst`, and `source/leuven/slurm_specifics.rst`.
  Current B200/A100/H100 partitions and resource limits still match this project's launchers.
- **Compute nodes have no outbound internet** — so `setup_venv.sh` and `fetch_ood` must run on a login
  node.

## 3. Cores and memory per GPU

Requesting more than the per-GPU ceiling raises a warning; on repeat it draws administrator attention.
This drives `--num-workers` for prior generation (the CPU-bound half of every training job), which the
job reads from `$SLURM_CPUS_PER_TASK` so one script is correct on 8 cores and on 24.

| partition | max cores/GPU | max mem/GPU | suggested `--n_jobs` |
|---|---|---|---|
| `interactive` | 8 | 32,000 MiB | — |
| `gpu_b200` | **24** | 194,400 MiB | 20 (of 24) |
| `gpu_a100` | 18 | 126,000 MiB | 16 (of 18) |
| `gpu_h100` | 16 | 187,200 MiB | 14 (of 16) |

`gpu_b200` is only **3 nodes / 24 GPUs total**, shared with every other user, so a throttle above ~8
is optimistic regardless of what the QoS allows. `nvidia-smi` memory is not utilisation — a 28.5M-param
model at micro-batch 4 uses ~13 GB of a B200's 192 GB but still runs at ~80–89% of the card's compute,
so there is no headroom to reclaim by enlarging the batch (and `micro_batch` is pinned at `group_size`
anyway).

## 4. Surviving a long sweep

### 4.1 The three limits

- **Walltime — 72 h, and `gpu_b200` cannot exceed it** (only `*_long` partitions reach 7 days). The
  default when you omit `--time` is 1 h — always set it. Billing is on *actual* time, so a generous
  request costs nothing, but a shorter one backfills sooner; the resume machinery makes an underestimate
  cheap.
- **Concurrency — two caps** (pending+running jobs, and total running resources), both per-partition
  QoS. The docs give the mechanism, not the numbers, so read them live:
  `sacctmgr show qos normal format=Name%20,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30`. We have hit both.
  **Use the array throttle `%N` rather than counting by hand** — `--array=0-44%8` keeps ≤8 active
  whatever the limit is.
- **Unannounced maintenance** — Mindwell drains without warning and cancels running jobs. Not often,
  but often enough that a multi-day sweep meets it.

### 4.2 The resilience mechanism (one chain covers all three)

A walltime kill, a node failure, and a maintenance drain all arrive as a signal, so one chain handles
them: `--signal=B:USR1@600` fires 600 s before the wall → a `trap` forwards it to python (Slurm does
**not**) → `_install_stop_signals` in `src/train/loop.py` catches it, finishes the step, writes a
checkpoint → `pretrain.py` returns **exit 64** ("saved, not finished") → the job script **resubmits its
own array index** → `maybe_resume` picks up the newest checkpoint. `--requeue` covers a node failure
where no signal arrives; `save_temp_every: 250` bounds an ungraceful kill to ~10 min. The trained child
must be backgrounded and `wait`ed on — a foreground child blocks the trap until too late.

**Consequence: an arm may take longer than the walltime and still complete** — the 72 h ceiling is a
scheduling detail, not a design constraint.

> The self-resubmit only fires on the graceful exit 64. A hard crash (a process abort) does **not**
> self-resubmit and leaves no summary — those arms are recovered by `sweep_status` below.

### 4.3 When the whole cluster drains — `sweep_status`

The chain above saves a *running* arm; it does nothing for a *pending* one that a drain cancels.
`sweep_status` reads the **output tree** (not `sacct`, which forgets, nor the queue, which is gone) and
classifies each array index — `done` (`summary.json` `completed: true`, not resubmitted), `partial`
(a checkpoint but no completed summary → resumes), `todo` (nothing → starts at 0):

```bash
python -m src.utils.sweep_status --config config/Exp1_PD.yaml            # report + resubmit spec
python -m src.utils.sweep_status --config config/Exp1_PD.yaml --resubmit # actually submit what's left
```

It prints a collapsed spec (`0-4,9,12-44`) so it can be checked before committing GPU-hours, and is
**safe to run repeatedly** — it never touches a `done` arm. Cross-check `squeue --me --clusters=mindwell`
first and don't resubmit an index already queued (two jobs would race one checkpoint).

### 4.4 Recovery is one command

For the whole experiment (both phases, both tracks), `run_experiment` reads the output tree and submits
only what is missing — **running it again is the recovery procedure after any interruption**:

```bash
python -m src.utils.run_experiment 1            # report: done / ready / running / blocked
python -m src.utils.run_experiment 1 --submit   # submit whatever is ready
```

The experiment number is required (never a default) and printed at the top, so "Exp1 or Exp2?" is
answered on the page. `run_experiment 3` refuses to run while its config still holds the
placeholder `FILL_FROM_EXP1` (Experiment 1's mix). Guarantees, each tested: benchmark cannot start before training finishes; a drain
costs only what was pending; a re-run never doubles a queued job (matched on the Slurm job name); a
unavailable `squeue` blocks submission until the queue can be checked, preventing duplicate jobs.

## 5. Storage

Three tiers, resolved automatically by [`src/utils/paths.py`](../src/utils/paths.py):

| tier | path | holds | backup | quota |
|---|---|---|---|---|
| **project staging** | `/lustre1/project/stg_00211/CreditICL` | inputs: `data/` (raw, processed, the OOD cache `ood/`) and `checkpoints/` (the released TabICLv2 / TabPFN-3 weights); the big half of `output CreditICL/`: `{reference,experiment_<N>}/benchmark/`, `experiment_<N>/checkpoints/<run>/`, `prior_cache/` | no | ≥1 TB, **low inode budget** |
| **personal data** | `$VSC_DATA/CreditICL` | the repo, and the small half of `output CreditICL/`: `logs/`, `{reference,experiment_<N>}/logs/`, `experiment_<N>/{runs,report}/` | **yes** | **75 GiB for every project together** — kept low |
| scratch | `$VSC_SCRATCH` | working scratch only | no | 500 GiB, **purged after 30 days without access** |

**One tree, two tiers, the same layout** (`src/utils/paths.py`; the tree is described in
`output CreditICL/README.md`). Since 30-09-2026 only what a job APPENDS TO while it runs stays on
`$VSC_DATA` — its log and a run's progress/telemetry CSVs, kilobytes to a few MB each; the benchmark
CSVs moved to staging with the checkpoints. Two reasons: `$VSC_DATA` is shared by every project, and
Mindwell jobs keep frequent small writes off Lustre (the rule below), while a benchmark writes its CSV
once, at the end. Downloading both halves (§10) gives every log, run record and benchmark score; the
figures are drawn locally. `paths.resolve_writable()` probes staging with a real write at job start and
fails before training if project storage is unwritable; it never redirects checkpoint output.
On Mindwell, `working_storage.py` copies repeated-read inputs once per job to GPFS. Rolling
checkpoints stay there; permanent snapshots, interruption checkpoints and final checkpoints
are copied atomically to project storage before completion is recorded. Successful jobs and
saved walltime continuations release their job-specific scratch after verifying the latest
checkpoint's permanent copy by hash; failed/cancelled jobs retain scratch for diagnosis.
A hard node loss may
lose work since the last published checkpoint; scratch is recoverable only while it still exists.

**The Lustre-vs-GPFS rule, enforced:** wICE jobs must do I/O on **Lustre**, Mindwell jobs on
**GPFS** — `$VSC_SCRATCH` already resolves to the right one, so *use `$VSC_SCRATCH`, never a hard-coded
path*; non-complying jobs can be cancelled without notice. **Purge trap:** scratch deletes by *last
access*, and `mv`/`rsync -t` carry an old atime — so **copy (not move) then `touch`**, and copy
checkpoints back to project storage. The copier refreshes access time and preserves modification
time for input identity. Small logs and run summaries remain on `$VSC_DATA`.

## 6. The Python environment

**There is no PyTorch module on VSC** — we install torch into a venv, **one per microarchitecture**
(wICE and Mindwell differ; a venv built on one is not reliably usable on the other), suffixed
`.venv-${VSC_ARCH_LOCAL}`. A venv with torch is ~5–8 GB across tens of thousands of small files, so it
lives on `$VSC_DATA` (staging's inode budget would exhaust first; `$VSC_HOME`'s 3 GiB can't hold torch).

**Build it once, on a login node** (compute nodes have no internet):

```bash
cd $VSC_DATA/CreditICL && bash scripts/slurm/setup_venv.sh      # idempotent; --recreate starts over
```

It builds from `pyproject.toml`, installs torch from the CUDA index, `pip install -e ".[dev,eval]"`,
then verifies every import *and* that our model matches the released checkpoints.

**Refresh an existing environment after dependency changes:**

```bash
bash scripts/slurm/update_env.sh
```

This only updates a `.venv` or `.venv-*` inside this project and refuses to install inside a
Slurm job. VSC torch is pinned to **2.11.0+cu128**, the installed build observed on 12-10-2026;
TabPFN is **9.1.0**, and TabICL uses its pinned Git commit. All runtime/dev/eval requirements,
including nested extras, are checked along with `pip check`. An inventory of all installed
package versions goes to `output CreditICL/general/report/environment.json`. Every job checks
the environment it actually selects. Exp0 records a package fingerprint; production submissions
require that exact validated environment, and benchmark receipts include it. A package change
requires revalidation, never a silent continuation under different dependencies. The inventory
is platform-specific; do not install a Windows freeze onto VSC. These metadata checks supplement
the GPU tests; they do not prove CUDA/runtime compatibility by themselves.

**Auto-activate on `cd`** (interactive shells): `bash scripts/slurm/shell_hook.sh --install` appends one
line to `~/.bashrc` that sources the in-repo hook (so `git pull` keeps it current; `--status`/`--uninstall`
manage it). It uses `PROMPT_COMMAND` (works after `pushd`/subshells/symlinks, unlike an overridden `cd`)
and stands down another project's active venv first — which mattered, because a sibling venv kept winning
the PATH race and `pip` reported "already satisfied" for packages we never had. **Jobs need none of
this** — `scripts/slurm/_activate_env.sh` finds and activates the venv explicitly, because `~/.bashrc`
is not reliably sourced in a batch job.

Two gotchas that read like a corrupt install: **the module load is not optional** (`.venv/bin/python` is
a thin link to the Lmod interpreter; the module name is read from `.python_module` beside the venv,
because module trees are per-architecture) and **never `module --force purge`** (it removes the sticky
`cluster/*` modules that set up `MODULEPATH`, collapsing the tree so even a real module won't load —
plain `module purge` is correct).

## 7. Job-script rules that bite

The ones that cost time or credits if you follow generic Slurm advice:

1. **Shebang `#!/bin/bash -l`** (login shell) — without `-l`, `module load` fails in the job.
2. **`--clusters=<name>` is mandatory** (KU Leuven made it required); monitoring commands need it too
   (`-M all` works for those).
3. **`--account` is mandatory** and must hold credits.
4. **Never `module --force purge`** (see §6); plain `purge` is fine.
5. **Memory requests silently multiply cores and cost:** per-core memory is a *max*, so
   `--mem-per-cpu` above it allocates `ceil()` more cores — `5G` on a 2500 MB/core partition = 3× the
   credits. Beware the `G` multiplier.
6. **`torchrun`, not `srun`, for multi-GPU** — this Slurm has no PMI, so MPI-via-`srun` may hang;
   `torchrun` uses its own rendezvous.
7. **Environment variables are not propagated** from the submitting shell — pass what you need with
   `--export`, or read it from a file.

Useful KU Leuven tooling: `slurm_jobinfo <jobid>`, `slurmtop`, `sam-balance`,
`python scripts/check_storage.py` (is staging writable?).

## 8. Submitting: the scripts

Use the two Experiment 0 scripts below. After both pass, the receipt-aware launcher reads
actual completion and queue state:

```bash
python -m src.utils.run_experiment 1             # inspect, submit nothing
python -m src.utils.run_experiment 1 --submit    # missing main/component training or benchmarks
```

Repeat after a stage finishes or a resumable interruption. Complete/current runs are skipped.
The retired `submit.sh` and `submit_pipeline.sh` refuse submission; their old pool/dependency chain
does not implement the current experiment design. Production scripts also enforce the Exp0 gate.

## 9. Compute budget

Measured on B200, batch 64, micro-batch 4, the config's prior shape:

| | value |
|---|---|
| B200 GPU-hour | 26,250 GPU credits plus host CPU/memory charges; verify current Turin billing |
| one Exp1 arm, 12,500 steps | ~5.5 GPU-h (0.63 steps/s, measured 24-09-2026 on the arms without the `banded` filter, which the redesign dropped) |
| **Exp1, 24 arms/track (9 main + 15 component)** (45 until 28-09-2026) | measure the revised 48-run campaign in Exp0; the 24-09 run took ~1,354 GPU-h for 90 arms, its banded-filter arms 5-7x the rest |
| one Exp3 arm, 200,000 steps | ~88 GPU-h at the same speed; 4 arms/track (control + Exp1's mix, 2 seeds) ≈ 700 GPU-h |
| TabICLv2's own full pretraining | 24.5 GPU-days per model (the paper's figure) |

The revised configurations generate tasks live in every experiment. Exp1 uses 12,500 updates,
2.5% of upstream's 500k-step first stage; Exp3 increases updates while retaining 1,024-row tasks.
Neither reproduces upstream's complete curriculum. The historical speeds above are planning
references, not measurements of the corrected pipeline. Queue delays, simultaneous allocations,
and the new full-row benchmarks must be added to training time.

For Experiment 2 (measured in Experiment 0 before it is committed): 18 arms per track (2 models × 3
mixes × 3 seeds), 10,000 steps each, using upstream TabICL and an adapted TabPFN-Wide recipe — no search
since 30-09-2026. A TabICLv2 arm costs about what an Exp1 arm costs per step (same shapes, batch 64), a
TabPFN-3 arm 16 tasks per step through TabPFN's own preprocessing. The old ~400 GPU-hour estimate
is provisional. `python -m src.utils.exp0_verify` extrapolates training hours from the new six
diagnostic runs. Benchmark probes measure selected tables only: their times do not establish
the runtime of the complete publication benchmark matrix.

## 10. Experiment 0: run it first

Experiment 0 consists of **two sequential Slurm jobs**. The CPU job rebuilds preprocessing,
checks every credit/OOD table and all five folds, audits dependencies and runs the portable tests.
The GPU job tests mandatory CUDA behavior, six isolated training routes, two upstream-equivalence
checks, and six 600-step runs with forced stop/resume. It scores every model route on the largest,
widest and median-size tables in each domain/track (duplicates removed), using all eligible rows
and five folds. It is a capacity and correctness gate, not the publication benchmark matrix.

```bash
cd "$VSC_DATA/CreditICL"
git pull --ff-only
source scripts/slurm/_activate_env.sh
# Optional intentional clean restart; deletes generated output and our trained checkpoints:
python -m src.utils.clean_run --reset-output --clean
bash scripts/slurm/update_env.sh
# Refresh the existing pinned OOD list without the old 50,000-row download cap:
python -m src.utils.fetch_ood
sbatch --clusters=wice scripts/slurm/exp0_cpu.slurm
# Wait for CPU COMPLETED/PASS before submitting the GPU job:
python -m src.utils.preflight --require-exp0-cpu && sbatch --clusters=mindwell scripts/slurm/exp0.slurm
```

`update_env.sh` installs this project's requirements/extras and verifies all transitive dependencies
and `pip check`. CPU and GPU jobs must use that same environment. Preprocessing cache v7 adds
borrower groups, safe large-value transforms and percentage/sentinel corrections. OOD cache v3 retains the same pinned dataset IDs
but restores full rows; a failed refresh cannot silently substitute another dataset.
The CPU reservation is 4 hours (controller 3.8), GPU 24 hours (controller 23.5); these are limits,
not runtime promises. No production array or automatic retry follows a failed validation.

Attempts live under `experiment_0/validation/<attempt>/` on both tiers. The canonical receipt is
`output CreditICL/experiment_0/report/validation.json`; it is RUNNING, FAILED or PASS. Previous
attempts remain until an explicit reset. CPU completion is recorded in `cpu_validation.json`;
the GPU receipt binds that exact CPU receipt. Rerunning CPU invalidates an older GPU PASS.
Exp0 reference probes are isolated and are never published as complete production results.
The launcher refuses production work if tests, code, configurations, packages, inputs or artifacts
changed after validation. The sole planned exception is filling Exp3's `FILL_FROM_EXP1` with the
already-tested 50% or 100% mixture after development-set selection; all other recipe changes still
invalidate validation. Each training run records its fully resolved choice.

After completion:

```bash
python -m src.utils.preflight --require-exp0
python -m src.utils.exp0_verify
```

Only PASS authorizes proceeding with the separate Experiment 1 submission. The verifier estimates
training hours from the short runs; it cannot predict queue delay. Download the small tier plus the
benchmark/report files from project storage, retaining the attempt directory structure. Checkpoint
binaries can remain on VSC; the local verifier reports that omission and VSC preflight verifies them
in place. Run `python -m src.utils.exp0_verify --root "/path/to/parent"`, where that parent contains
the downloaded `output CreditICL/` directory.

## 11. Rename existing output, preserving completed work

The output root is **`output CreditICL`**, including the space. This is a deliberate naming
change from the template's `output/`, requested by Andreas. The same name applies on both
`$VSC_DATA/CreditICL/` and project storage. Renaming a directory does not alter scientific results.

After the local changes have been committed and pushed, on a wICE login node:

```bash
cd "$VSC_DATA/CreditICL"
git pull
source scripts/slurm/_activate_env.sh
python -m src.utils.rename_output
python -m src.utils.rename_output --apply
```

The command checks both storage tiers before moving files, refuses to proceed while CreditICL
jobs are queued or running, and preserves collisions with files supplied by `git pull` under
`legacy-output/output/`. It never overwrites files. Repeating it is safe. Jobs with custom names
must also be stopped before renaming. No automatic migration of historical run layouts is made:
September's `manifests/` and `results/` remain readable as historical records under the new root;
its trained checkpoints remain at `<staging>/CreditICL/checkpoints/<run>/`.

**Do not delete these old checkpoints or resubmit the September sweep.** That 45-arm-per-track
campaign completed, including its benchmark. Today's Exp1 has 9 main and 15 component runs per track
and uses a changed prior,
optimizer settings, dataset partition, and evaluation protocol, so the old outputs do not count
as completion of the new design. Validate the revised pipeline through Exp0 before launching it.

To download the missing September benchmark CSVs after renaming (no model weights included):

```bash
STG=$(python -c "from src.utils.paths import big_outputs_dir; print(big_outputs_dir())")
python -m zipfile -c "$VSC_DATA/CreditICL/benchmark-results.zip" "$STG/results"
```

Download `benchmark-results.zip` from the repository directory. The run logs alone are not a
substitute for checking every result cell in those CSVs.

## 12. Experiments 1, 2 and 3: the commands

```bash
# Experiment 1 (after Experiment 0 passed): train, then benchmark (full rows, five folds).
python -m src.utils.run_experiment 1 --submit          # again later: it submits what is left

# Experiment 2: continued pretraining using the controlled TabICL and adapted TabPFN recipes.
python -m src.utils.run_experiment 2 --submit

# Experiment 3: once Experiment 1 has chosen its mix on the development datasets, fill FILL_FROM_EXP1
# in config/Exp3_PD.yaml and config/Exp3_LGD.yaml, commit, pull on the cluster, then:
python -m src.utils.run_experiment 3 --submit
```
