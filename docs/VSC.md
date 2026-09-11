# Running CreditICL on KU Leuven VSC

How to submit, what it costs, and how a multi-day sweep survives a cluster that stops routinely.
Distilled from `tfm-library/repositories/VSC Documentation.txt` (pin `52dab01`) plus what we learned
running on it; where the docs are silent this file says so. **Verify charge rates and partitions
against the live docs before a large run.** The recovery procedure after *any* interruption is one
command — [§4.4](#44-recovery-is-one-command).

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

**GPU count is queue length; cores/GPU is our real bottleneck**, because TabICL generates its prior on
the CPU (`--prior_device cpu`). Two non-obvious conclusions follow:

- **`gpu_b200` is the best production target on *both* axes** — most GPUs (shortest queue) and most
  cores/GPU. This inverts the naive ranking: `gpu_h100` is the fastest chip but has the *fewest* cores
  and costs 4× a B200, so it is never the rational choice here.
- **`gpu_a100` is ~2.7× cheaper all-in** and, since much of a training job's GPU-minutes are spent
  waiting on the CPU generator, buys the most *independent runs per credit* — but only if it is less
  than 2.7× slower, which is plausible but unmeasured. Measure with `scripts/slurm/gpubench.slurm` +
  `src/utils/compare_gpubench.py` before committing a sweep to it.

`interactive` costs nothing and on Mindwell is a *full* GPU (not wICE's ~1/7 MIG slice) — the right
home for debugging, data staging, and Lustre↔GPFS transfers. **Genius (P100/V100) cannot run this
code**: the wheel is compiled for sm_75+ (P100 is sm_60, V100 sm_70), and bf16 AMP (a measured ~2×)
needs Ampere or later.

## 2. Account and cluster setup

- **Account `lp_verbekelab`** (has Mindwell access; billed under it since 14-08-2026).
- **Neither wICE nor Mindwell has a login node** — SSH into **Genius** and submit with
  `#SBATCH --clusters=<name>`. Every `.slurm` here already does.
- **Compute nodes have no outbound internet** — so `setup_venv.sh` and `fetch_ood` must run on a login
  node.

## 3. Cores and memory per GPU

Requesting more than the per-GPU ceiling raises a warning; on repeat it draws administrator attention.
This drives `--num-workers` for prior generation (the CPU-bound half of every training job), which the
job reads from `$SLURM_CPUS_PER_TASK` so one script is correct on 8 cores and on 24.

| partition | max cores/GPU | max mem/GPU | suggested `--n_jobs` |
|---|---|---|---|
| `interactive` | 8 | ~60,000 MiB | — |
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
answered on the page. `run_experiment 2`/`3` refuse to run while their configs still hold
`FILL_FROM_EXP1`. Guarantees, each tested: benchmark cannot start before training finishes; a drain
costs only what was pending; a re-run never doubles a queued job (matched on the Slurm job name); a
broken `squeue` errs toward submitting (a duplicate can be cancelled; work that never starts cannot).

## 5. Storage

Three tiers, resolved automatically by [`src/utils/paths.py`](../src/utils/paths.py):

| tier | path | holds | backup | quota |
|---|---|---|---|---|
| **project staging** | `/lustre1/project/stg_00211` | big files: datasets, checkpoints, prior pools, result CSVs | no | ≥1 TB, **low inode budget** |
| **personal data** | `$VSC_DATA` | the repo + small durable output: logs, manifests, figures, configs | **yes** | **75 GiB** — tight |
| scratch | `$VSC_SCRATCH` | working scratch only | no | 500 GiB, **purged after 30 days without access** |

Big-and-regenerable → staging (its low inode budget wants few big files, not thousands of per-step
metrics — those go to `$VSC_DATA`). `paths.resolve_writable()` probes staging with a real write at job
start and falls back to `$VSC_DATA` with a loud warning (a sibling project lost checkpoints to
unwritable staging).

**The Lustre-vs-GPFS rule, enforced:** Genius/wICE jobs must do I/O on **Lustre**, Mindwell jobs on
**GPFS** — `$VSC_SCRATCH` already resolves to the right one, so *use `$VSC_SCRATCH`, never a hard-coded
path*; non-complying jobs can be cancelled without notice. **Purge trap:** scratch deletes by *last
access*, and `mv`/`rsync -t` carry an old atime — so **copy (not move) then `touch`**, and copy
checkpoints you care about back to `$VSC_DATA` when a run finishes.

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

Don't hand-roll `#SBATCH` — the launchers pick the partition and read `$SLURM_CPUS_PER_TASK`, so one
script is correct everywhere. `#SBATCH` directives are only defaults, overridden on the command line.

```bash
bash scripts/slurm/submit.sh --list                 # the partition inventory, from the shell
bash scripts/slurm/submit.sh b200 pd                # <where> <track> — one debug arm
sbatch --clusters=mindwell --array=0-44%8 scripts/slurm/pretrain_pd.slurm   # a full phase-1 sweep
bash scripts/slurm/submit_pipeline.sh both          # the whole chain, submit and log out
```

**The track is an argument, never an environment variable:** `CONFIG=… bash submit.sh` sets the var for
the *calling* shell, not sbatch's — so the job never saw it and a run meant as PD went out as a second
LGD job. `submit.sh` prints `where / track / config / script` before submitting and refuses an unknown
track.

`submit_pipeline.sh` chains five stages, each on the hardware it needs, with `--dependency`: preprocess
(1 CPU job) → **prior pools (2 arrays × 20 CPU tasks** — the two variants are independent, so submitting
separately makes 40 tasks eligible at once) → **verify gate** (`afterok` on the arrays — equal, complete
pools or the chain stops) → GPU training array → evaluation (`afterany` — keep the numbers from arms
that finished). Prior generation pins `OMP_NUM_THREADS=1` (the parallelism is the array, not BLAS) and
each task sleeps 1–20 s to avoid a Lustre metadata storm.

## 9. Compute budget

Measured on B200, batch 64, micro-batch 4, the config's prior shape:

| | value |
|---|---|
| B200 GPU-hour, all-in | ~30,625 credits (26,250 + ~4,375 for 24 cores) |
| one Exp1 arm, 12,500 steps | ~16–25 h (the `banded` filter's ~90% rejection makes some arms slower) |
| **Exp1, 45 arms/track** | order of tens of millions of credits |
| TabICLv2's own full pretraining | 24.5 GPU-days per model (the paper's figure) |

At throttle `%8`, 45 arms of ~20 h is `ceil(45/8) × 20 ≈ 120 h` wall-clock — **the throttle, not the
credit balance, sets how long the sweep takes**, so decide it against the QoS limits (§4.1). Our 12,500
steps is 2.5% of upstream's 500k-step stage 1 (which would be ~347 h *per arm*) — the whole reason Exp1
is a *screening* tier: it ranks priors; only Exp3 runs the winner long enough for the number to stand
alone. **Storage rules out pooling at full scale** (35M datasets ≈ 3.5–4.7 TB/variant vs a ~1 TB quota;
upstream sees each dataset once, so there is no corpus), which is why Exp3 uses `--prior-source generate`
and pools remain only for Exp1, where they remove draw-luck between short arms.
