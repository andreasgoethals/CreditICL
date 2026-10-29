# Running CreditICL on KU Leuven VSC

Shared infrastructure reference: [VSC Library](../vsc-library/README.md). The parent Git commit pins its revision; inspect it with `git submodule status -- vsc-library`.
Its [official repository snapshot](<../vsc-library/VSC Repository.txt>),
[guides](../vsc-library/guides/) and [Slurm templates](<../vsc-library/slurm templates/README.md>)
are read-only. Keep this project's actual paths, environment, resource choices,
submission/resume steps and run evidence in this file or the project's own run logs.
Check [upstream review flags](../vsc-library/UPSTREAM_CHANGES.md) and the
[live Leuven status](https://status.vscentrum.be/tier2_leuven.html) before applying a changed policy.

How to submit, what it costs, and how a multi-day sweep survives a cluster that stops routinely.
Distilled from `vsc-library/VSC Repository.txt` (pin `289ed81df5d101512f91ae1ea3b0083a2de08a96`, checked 20-10-2026) plus what we learned
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

| partition | cluster | GPUs (queue) | cores/GPU | GPU-component credits/h | max walltime |
|---|---|---|---|---|---|
| `interactive` | mindwell | 4 × RTX 5000 Ada 32 GiB | 8 | **free** | 16 h |
| **`gpu_b200`** | mindwell | **24** × B200 192 GiB | **24** | 26,250 | 72 h |
| `gpu_a100` | wICE | 16 × A100 80 GiB | 18 | **8,500** | 72 h |
| `gpu_h100` | wICE | 20 × H100 80 GiB | 16 | 34,167 | 72 h |
| `gpu_a100_debug` | wICE | 1 × A100 | 18 | — | 1 h (one job only) |

GPU count describes capacity; queue wait depends on current demand. Our generator uses CPU cores,
so GPU speed alone does not predict the time or cost of an arm.
The verified 29-10 quote for this project's B200 + 24 CPU + 180G allocation is **30,600 credits/h**,
including host resources; use that allocation price for the current campaign estimates in §9.

- **`gpu_b200` remains the production default:** 24 cores per GPU and 192 GiB of GPU memory.
  The H100 GPU charge is about **1.30×** the B200 charge, not 4×; its 16-core allocation may
  constrain generation. Choose between them using measured throughput and current queue state.
- **`gpu_a100` is cheaper per allocated hour**, but total cost depends on how much longer a job
  takes. Measure with `scripts/slurm/gpubench.slurm` and `src/validation/compare_gpu.py`; use
  `sam-quote` for the actual allocation price. The H100 CPU estimate uses the published Zen4
  Genoa rate (3.472 credits/core-minute), not the A100 Ice Lake rate (2.546).
- The published table does not separately list the B200 hosts' Turin CPU rate. The launcher's
  existing B200 CPU estimate is provisional; the live `sam-quote` price is authoritative.

`interactive` costs nothing and on Mindwell is a *full* GPU (not wICE's ~1/7 MIG slice) — the right
home for debugging, data staging, and Lustre↔GPFS transfers. **The P100/V100 partitions, now on wICE, cannot run this
code**: the wheel is compiled for sm_75+ (P100 is sm_60, V100 sm_70), and bf16 AMP (a measured ~2×)
needs Ampere or later.

## 2. Account and cluster setup

**Access adoption, 29-10-2026.** `docs/TEMPLATE.md` matches the canonical template byte-for-byte
(SHA-256 `285c7c321b2cd52d8d415e30f41fe775ceb47113f39007c2490fdb0ec23fc8d8`). No additional
template synchronization or per-project SSH installation is needed. The shared VSC reference pin
for this review is `0ed4ffa89d312b9bb6411681b6a14f81672439f3`.

Andreas renews access manually in **Windows PowerShell**:

```powershell
step ssh login --context VSC
```

He completes KU Leuven Authenticator approval. Agents reuse this Windows user's existing native
OpenSSH agent; they do not start login flows, configure renewal hooks or schedule reminders.
If the certificate is unavailable/expired, ask Andreas to run the command. Keep certificate
contents, private keys and authentication URLs out of reports and Git.

For relevant tasks, directly inspect bounded VSC files, logs, job/accounting state and quotas
using the tested method in [TEMPLATE.md](TEMPLATE.md#vsc-access-windows-and-agents): native
`System32/OpenSSH/ssh.exe`, noninteractive options, strict host-key checking, and an LF-delimited
script on stdin with **no trailing remote command**. Verify the channel with the deliberate
exit-7 stdout/stderr probe before trusting results. Missing markers, even with exit 0, are a channel
failure; do not interpret them as empty results or retry uncertain mutations. Network/host-key
failures require their own diagnosis, not automatic authentication renewal. Remote computational
debugging belongs in an explicitly authorized Slurm allocation; a working SSH connection grants
no authorization to submit, cancel, install, relocate or delete.

**Verified read-only access, 29-10-2026.** The native LF-stdin probe returned its stdout marker
and exit 7, but its raw stderr marker did not reach the client. The verified capture fallback
runs the bounded Bash payload through remote Python 3 `subprocess.run(..., capture_output=True)`
and emits a JSON envelope on stdout containing separate `stdout`, `stderr` and `returncode`.
That envelope contained `VSC_ACCESS_OK`, `VSC_ERROR_CHECK` and return code 7; SSH also exited 7.
Parse the envelope and verify its status rather than treating the absent raw stderr stream as
success. This is a route-specific transport workaround, not an authentication or shell change.

The live checkout resolves to `/vsc-hard-mounts/leuven-data/383/vsc38338/CreditICL`, at commit
`be6de6e`, with tracked notebook/figure changes. Preserve those changes before any update.
The `.venv-skylake` runtime is still on DATA; persistent assets remain under
`/lustre1/project/stg_00211/CreditICL`. No PROJECT_ROOT overrides were set, and no jobs were
listed for this user on Mindwell or wICE at inspection time. Quota usage was DATA
56,233M/76,800M; PROJECT 538.2G/1T; GPFS scratch 8.634G/500G; Lustre scratch 28.5G/500G.
These are account/allocation totals, not CreditICL-only sizes. No authentication, installation,
migration, workload submission or deletion was performed; this read does not validate GPU code.

- **Account `lp_verbekelab`** (has Mindwell access; billed under it since 14-08-2026).
- **SSH to `login.hpc.kuleuven.be`**, which now selects a wICE login node. Mindwell shares
  these login nodes. Genius retired at the end of September 2026; its P100/V100 nodes moved to
  wICE. Submit with explicit `#SBATCH --clusters=wice` or `mindwell`; every `.slurm` here does.
- The relevant library source sections are `source/leuven/tier2_hardware/tier2_login_nodes.rst`,
  `source/leuven/old_hardware/genius/genius_hardware.rst`, and `source/leuven/slurm_specifics.rst`.
  Current B200/A100/H100 partitions and resource limits still match this project's launchers.
- **Compute nodes have no outbound internet** — prepare package/input artifacts before an authorized
  offline container build/validation allocation. `fetch_ood` and exceptional legacy environment
  maintenance need a networked preparation route; neither is an automatic job-start action.

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

**Submission policy, 29-10-2026:** current launch plans contain no fixed percentage throttle.
Inspect the account/QoS admission ceiling before submitting; Slurm controls running resources.
The reviewed 64-run Exp2 grid fits the site's verified submission limits. Explicit memory/CPU
requests remain measured project choices: these launchers execute Python directly and do not
introduce an `srun`/MPI step affected by the site's CPU/memory-variable workaround.

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
where no signal arrives; `save_temp_every: 250` bounds an ungraceful kill to 250 updates (roughly
ten minutes for the observed short-table runs, not for long-context Exp2). The trained child
must be backgrounded and `wait`ed on — a foreground child blocks the trap until too late.

**Consequence: an arm may take longer than the walltime and still complete** — the 72 h ceiling is a
scheduling detail, not a design constraint.

Before launching long-context Exp2, compare the measured 64-task update time plus checkpoint-copy
time with the 600-second warning interval. The stop handler completes the current update first;
the old short-table timing does not establish that this fits. A capacity PASS alone is insufficient
to choose checkpoint cadence, warning time or a campaign budget.

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
answered on the page. `run_experiment 3` resolves its predeclared settings without any selection receipt. Guarantees, each tested: benchmark cannot start before training finishes; a drain
costs only what was pending; a re-run never doubles a queued job (matched on the Slurm job name); a
unavailable `squeue` blocks submission until the queue can be checked, preventing duplicate jobs.

## 5. Storage

Three tiers, resolved automatically by [`src/utils/paths.py`](../src/utils/paths.py):

The actual checkout, including `.git`, stays on DATA. Large persistent assets and new installed
dependencies belong on PROJECT, with dependencies consolidated into immutable SIF images as
described in §6. Scratch contains temporary work only. The existing DATA venv is an explicitly
documented migration exception; it has not been relocated or replaced by this policy update.
Do not pull, update submodules or mutate environments while active/queued jobs use that checkout.
Use a separate actual DATA checkout for a new revision, or wait for those jobs to finish.

| tier | path | holds | backup | quota |
|---|---|---|---|---|
| **project staging** | `/lustre1/project/stg_00211/CreditICL` | inputs: `data/` (raw, processed, the OOD cache `ood/`) and `checkpoints/` (the released TabICLv2 / TabPFN-3 weights); the big half of `output CreditICL/`: `{reference,experiment_<N>}/benchmark/`, `experiment_<N>/checkpoints/<run>/`, `prior_cache/` | no | ≥1 TB, **low inode budget** |
| **personal data** | `$VSC_DATA/CreditICL` | the repo, and the small half of `output CreditICL/`: `logs/`, `{reference,experiment_<N>}/logs/`, `experiment_<N>/{runs,report}/` | **yes** | **75 GiB for every project together** — kept low |
| scratch | `$VSC_SCRATCH` | working scratch only | no | 500 GiB, **purged after 30 days without access** |

Canonical allocation-root overrides are `CREDITICL_PROJECT_ROOT`, then `VSC_PROJECT_ROOT`.
They name the **shared allocation root**, and the resolver appends `CreditICL`. Existing
`CREDITICL_STAGING_ROOT`, `CREDITPFN_STAGING_ROOT` and `TABPFN_STAGING_ROOT` remain lower-priority
aliases, followed by the existing Lustre allocation default. Unset new overrides preserve current
paths; setting one is a deliberate cutover decision, not a copy/migration operation. Pass it explicitly
through the submission environment; arbitrary login exports are not assumed to reach a Leuven job.
Unavailable persistent storage fails instead of redirecting large writes to DATA or scratch.

Before migration/build/staging, check both bytes **and file counts** with `myquota` and a bounded
inspection of this project's directories. Verify PROJECT protection independently; DATA backups
do not protect project checkpoints. Keep independent backups of irreplaceable results.

**One tree, two tiers, the same layout** (`src/utils/paths.py`; the tree is described in
`output CreditICL/README.md`). Since 30-09-2026 only what a job APPENDS TO while it runs stays on
`$VSC_DATA` — its log and a run's progress/telemetry CSVs, kilobytes to a few MB each; the benchmark
CSVs moved to staging with the checkpoints. Two reasons: `$VSC_DATA` is shared by every project, and
Mindwell jobs keep frequent small writes off Lustre (the rule below), while a benchmark writes its CSV
once, at the end. Downloading both halves (§10) gives every log, run record and benchmark score; the
figures are drawn locally. `paths.resolve_writable()` probes staging with a real write at job start and
fails before training if project storage is unwritable; it never redirects checkpoint output.
On Mindwell, `working_storage.py` now uses **`$VSC_SCRATCH_NODE/CreditICL/jobs/<jobid>`**
for staged inputs, prediction offload and one latest working checkpoint. There is no automatic
shared-GPFS fallback. Every checkpoint save, including rolling recovery states, is copied atomically
to project storage before older working copies are removed. Permanent milestones and two rolling
states are retained on project storage. Node-local disk is temporary and removed at job end;
a hard loss can therefore lose at most the work since the last successfully published save.
Successful jobs verify the latest publication before releasing their working directory.

**Native I/O:** Mindwell repeated I/O belongs on GPFS or node-local scratch; wICE uses Lustre.
Project-Lustre access from Mindwell is restricted here to input staging and checkpoint publication,
not repeated training reads or offload. This follows VSC library pin
`0ed4ffa89d312b9bb6411681b6a14f81672439f3`. Small logs remain on `$VSC_DATA`.
`CREDITICL_WORK_ROOT` is an explicit override; leave it unset for the node-local default.

To remove old shared-scratch copies without deleting persistent experiment results, while no
CreditICL job is queued/running:

```bash
python -m src.utils.scratch_cleanup
python -m src.utils.scratch_cleanup --clean
myquota
```

This only considers configured GPFS/Lustre `CreditICL/jobs` directories positively identified
by Slurm accounting as terminated CreditICL jobs. Disposable staged inputs/offload may be removed;
checkpoint folders require byte-identical persistent copies. Unknown jobs, unpublished/corrupt
checkpoints and other projects are retained. Preview messages identify anything requiring review.
The 29-10 Exp2 reset already removed 95.010 GB; the remaining shared-GPFS usage is not all attributed
to this project. Do not delete other projects to recover it.

## 6. The Python environment

**Target policy: immutable dependencies in versioned Apptainer/SIF images on PROJECT.** This is
the project's adopted template policy, not a claim that VSC mandates it. The container avoids the
file-count cost of an unpacked PyTorch environment. Do not create an unpacked venv/Conda tree or
persistent container sandbox on PROJECT, and do not move a populated DATA environment there.
Local Windows venvs remain supported.

**CPU validation allocation, 29-10-2026.** Exp0 CPU uses wICE `interactive`, eight CPUs, 48G and four hours, with no GPU request. Live QoS permits eight CPUs; the previous full CPU validation peaked at 21.1 GiB. This is computational debugging, not production training. Both CPU/GPU validation explicitly exercise `sbatch --test-only` for wICE and Mindwell. The host Slurm serializer/data-parser libraries, site Lua modules and real config-cache directory are bound into the image; an inherited login-node configuration path is never blindly reused.

**Runtime migration authorized, 29-10-2026.** The candidate is built through `scripts/slurm/build_runtime.slurm`; `src.utils.runtime_image` archives distribution payloads and hashes, and the image installs them into a fresh `/opt/venv`. Console scripts are regenerated and original Git origins are retained. The DATA manifest `.runtime-build/current.json` selects the PROJECT SIF. `_activate_env.sh` now selects that image by default; `CREDITICL_LEGACY_RUNTIME=1` is an explicit recovery exception. The wrapper binds checkout code read-only, output/assets/scratch writable, and site-managed Git/Slurm clients with their non-glibc dependencies. Site credit tools use their host interpreter and native loader under a separate read-only mount; the scientific interpreter remains `/opt/venv/bin/python`. Missing or disabled credit checks fail validation. No training job installs packages. `clean_run --clean` alone can open the checkout for deletion on an idle login node, so old root-level Slurm fallback logs remain cleanable. The wrapper rejects that mode in an allocation or while any CreditICL job is queued/running; previews and all computation retain the read-only code mount.

**Before acceptance the legacy DATA venv is retained for rollback.** `_activate_env.sh` continues to discover existing
`.venv-${VSC_ARCH_LOCAL}` / `.venv` environments and their recorded Python modules so existing
recovery paths remain usable. This repository has not yet built or accepted a SIF runtime. There is
no automatic container switch, environment relocation, package installation or cleanup in this update.

Migration must be proposed and approved before execution:

1. Freeze the current DATA checkout, runtime inventory/locks, CUDA build, source revision and
   submodule pins. Check queued/running/recoverable jobs and both quota types. Identify the allocated
   PROJECT destination and verify accessibility from the intended wICE and Mindwell partitions.
2. Prepare an immutable image definition and pinned offline dependencies from the recorded VSC
   environment. Keep definitions/manifests in DATA Git, build caches/temp files on scratch and the
   final versioned SIF on PROJECT. Build only in an authorized allocation with sufficient resources.
3. Record the image digest and explicit code/data/results/cache binds. Validate CPU imports,
   dependencies and filesystem publication, then actual GPU precision, forward/backward,
   checkpoint/resume and benchmark behavior through Experiment 0, including `--nv`/host-driver
   compatibility. Metadata or a local container test alone does not certify the VSC GPU path.
4. Switch only a separate/new validated campaign to the accepted image. Preserve the old runtime,
   checkout and checkpoints until no active, queued or resumable job needs them. Retire exact
   project-owned obsolete files only after verified publication and an approved cleanup.

Until that migration, explicitly authorized legacy repair while all consumers are idle is available:

```bash
bash scripts/slurm/update_env.sh --legacy-maintenance
# Exceptional rebuild, only if reviewed and needed:
bash scripts/slurm/setup_venv.sh --legacy-maintenance --recreate
```

Both scripts otherwise stop before any installation and explain the new policy. The explicit
exception preserves the old DATA location and is not a container migration. Never run repair as
routine preparation for a healthy campaign. Existing shell auto-activation hooks may remain, but
do not install a new hook during this adoption; jobs explicitly source `_activate_env.sh`.

VSC torch is pinned to **2.11.0+cu128**, the installed build observed on 12-10-2026;
TabPFN is **9.1.0**, and TabICL uses its pinned Git commit. All runtime/dev/eval requirements,
including nested extras, are checked along with `pip check`. An inventory of all installed
package versions goes to `output CreditICL/general/report/environment.json`. Every job checks
the environment it actually selects. Exp0 records a package fingerprint; production submissions
require that exact validated environment, and benchmark receipts include it. A package change
requires revalidation, never a silent continuation under different dependencies. The inventory
is platform-specific; do not install a Windows freeze onto VSC. These metadata checks supplement
the GPU tests; they do not prove CUDA/runtime compatibility by themselves.

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
`python -m src.utils.check_storage` (is staging writable?).

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

Completed Exp1 training on B200, batch 64 and micro-batch 4, with explicit projections below:

| | value |
|---|---|
| B200 allocation-hour | **30,600 credits**, live `sam-quote` on 29-10-2026 for one B200, 24 CPUs, 180G; GPU component 26,250, host charges additional |
| one Exp1 arm, 12,500 steps | **6.43–7.38 training GPU-h**, measured in the completed 24-10-2026 snapshot |
| **Completed original Exp1, 24 arms/track (9 main + 15 component)** | **323.616 recorded training GPU-h** for all 48 completed arms, excluding startup and benchmarks; see [RUNS, 24-10-2026](RUNS.md#24-10-2026--exp1-training-complete--all-48-runs-successful) |
| **Completed Exp1 extension: 3 more arms/track** | 25% credit, three seeds, indices 9–11; all six completed on 28-10, making 54 completed training runs. Their recorded training totals about 40.7 GPU-h; quarter-mixture benchmarks remain pending in the 29-10 snapshot. |
| Superseded 96-run Exp2, 10k updates | Production TabICL timing projects 210–225 GPU-h/arm; combined with the provisional TabPFN subtotal, **12,189–12,909 GPU-h / 373–395M training credits**, excluding benchmarking. Replaced by the fixed-rate budget below. |
| Fixed-rate Exp2 | 64 runs × 1,500 updates: approximately **37.3–39.5M training credits**, linearly extrapolated from earlier throughput; excludes validation and benchmarking. |
| Fixed-recipe Exp3 | 32 new runs × 1,500 updates + 8 reused baselines: **18.6–19.8M full-update-equivalent credits** for planning only; adapter throughput is unmeasured. |
| Current Exp3 | Retention-method runtime remains unmeasured; do not extrapolate scratch throughput. The old 200k scratch Exp3 is abandoned. |
| TabICLv2's own full pretraining | 24.5 GPU-days per model (the paper's figure) |

The revised configurations generate tasks live in every experiment. Exp1 uses 12,500 updates,
2.5% of upstream's 500k-step first stage; Exp2/3 adapt released weights for 1,500 updates.
Neither reproduces upstream's complete curriculum. Exp1 times describe its frozen completed
campaign; they do not measure the new Exp2/Exp3 recipes. Wall-clock completion also depends
on queue delays and concurrent allocations, and benchmarks add their own runtime.

For Experiment 2 (capacity checked in Experiment 0 before launch): 32 arms per track (2 models × 4
mixes × 2 fixed rates × 2 seeds), 1,500 steps each, using upstream TabICL and TabPFN's differentiable
fine-tuning interface on synthetic tasks. The 24-10 revision changes TabICLv2 to 400–60,000 log-uniform rows, 79–81%
context and one task per microbatch, retaining 64 tasks per update. Its expected row exposure is
about 1,142 million per main arm, 11.6 times a matched 1,024-row proposal; this is **not** a runtime ratio.
TabPFN-3 retains 16 tasks per update and 1,024 rows through its own preprocessing. The old ~400
GPU-hour estimate is **obsolete**. Never extrapolate TabICLv2 Exp2 cost from Exp0's 1,024-row
route checks. Fresh capacity and representative-batch probes must establish feasibility first;
one representative update gives a timing observation, not a reliable campaign ETA.
Benchmark probes measure selected tables only: their times do not establish
the runtime of the complete publication benchmark matrix.

The 29-10 live CSV audit provides an inference planning anchor. Summed fold `total_seconds`
for a complete credit+OOD suite were 2.3824 hours PD / 0.1319 hours LGD for released TabICL,
and 1.9298 hours PD / 0.1388 hours LGD for released TabPFN. The 45 Exp1 TabICL checkpoint
evaluations per track had medians of 2.5337 hours PD / 0.1268 hours LGD. These timers include
preprocessing and calibration; they are not Slurm elapsed time or billed GPU-hours. At comparable inference
speed, all five Exp2 checkpoints imply approximately 367–378 timed hours (11.22–11.58M
credits); Exp3 adds 183–189 hours (5.61–5.79M), conditional on adapter inference speed.
Together with training, the planning total is roughly 73–77M credits, excluding validation,
retries and remaining Exp1 benchmarks. The live available balance was 95,812,429 credits. This reserve
motivates the 1,500-update design; the estimate is not a spending guarantee.

The 29-10 quote uses an actual jobscript (this VSC `sam-quote` rejects `--wrap` alone):

```bash
sam-quote sbatch --clusters=mindwell --partition=gpu_b200 --account=lp_verbekelab \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=24 --mem=180G \
  --time=01:00:00 --array=0 scripts/slurm/pretrain_pd.slurm
```

Available balance in the supplied transcript is 95,945,277 credits. This is a dated balance;
use `sam-balance` again before submitting. The fixed 1,500-update grid is implemented locally; this does not authorize a new VSC workload. Training estimates exclude validation and full-table benchmarking.

For the authorized clean Exp2 restart, preview then delete only its generated run output:

```bash
python -m src.utils.clean_run --reset-exp2
python -m src.utils.clean_run --reset-exp2 --clean
myquota
```

Run on a login node with no CreditICL jobs queued/running. Both permanent output tiers are
included; configure `VSC_SCRATCH_GPFS1` and `VSC_SCRATCH_LUSTRE1` to inspect their job directories.
Accounting records and checkpoint destination markers identify Exp2 scratch. Unidentified jobs
are retained and printed; another project's scratch is never a cleanup target. The same command
works locally on the local output tree. Downloads ZIPs remain evidence. Current Exp0 and all
Exp1 outputs survive. After code or the final approved grid changes, validate that revision
before production; do not repeatedly rerun Exp0 while the budget design is still changing.

## 10. Experiment 0: run it first

For the 26-10-2026 two-rate Exp2 and retention-method Exp3 revision, start a fresh validation pair after a successful
pull and with no CreditICL jobs queued or running. Preserve generated tracked notebook edits with
`git stash push -- notebooks "output CreditICL"` if they block the pull; do not
reset the entire repository or stash/untrack the expensive experiment outputs. Chain the pull,
source check and submission with `&&` (or use a Bash subshell with `set -e`) so a failed update
cannot be followed by a submission on old code.

The scoped cleanup is `python -m src.utils.clean_run --reset-exp0` to inspect, followed by
`python -m src.utils.clean_run --reset-exp0 --clean` to remove **only Experiment 0** on DATA and
PROJECT storage (including diagnostic checkpoints and CPU/GPU Slurm logs), plus scratch jobs
identified by those validation names. It refuses an active allocation/queue, links and escaped
paths. Production experiments, shared reference benchmarks, datasets and released weights survive.
Locally the same option targets the one local output tree; Downloads ZIPs are separate evidence.
This deliberately removes old successful validation too, for a clean current-code pair; their
historical outcome remains recorded in RUNS. Do this before CPU validation, never between CPU
and GPU. After a diagnosed GPU failure with unchanged code, `--reset-exp0-gpu` is the narrower
option that preserves CPU validation. Do not reset a DEFERRED attempt: it should resume.

Experiment 0 consists of **CPU then GPU validation** (two scripts). The CPU job rebuilds preprocessing,
checks every credit/OOD table and all five folds, audits dependencies and runs the portable tests.
The GPU job tests mandatory CUDA behavior, six isolated training routes, two upstream-equivalence
checks, and six 600-step runs with forced stop/resume. It scores every model route on the largest,
widest and median-size tables in each domain/track (duplicates removed), using all eligible rows
and five folds. It is a capacity and correctness gate, not the publication benchmark matrix.

The revised TabICLv2 recipes add **34 mandatory long-context GPU stages**: seven row/feature capacity
corners for both tracks and both priors (28), followed by a complete, sampled 64-task update per
track (2), and a 60k-row stage for both DiagScale and LoRA on each track (4).
Another **16 native method round trips** cover training, forced stop/resume, frozen weights and
checkpoint prediction for four methods, two tasks and both backbones. The long-context stages
use released weights, activation recomputation, float32 efficient SDPA, four
loader workers at most and one prefetched batch per worker. No mathematical-attention fallback
or implicit FA3 float16 conversion is allowed. Each long-context stage requires a finite, nonzero
optimizer update, records memory and timing, and has a two-hour guard; the smaller method round
trips have a 30-minute guard. Diagnostic weights are discarded.

**While any CreditICL job is queued or running, do not pull changes, reset output or rerun these
setup commands in its checkout.** Finish and preserve that campaign first. Validate the revised
code for Exp2 in a separate checkout/output root or after preserving Exp1's code and receipts.
The previous Exp0 PASS remains evidence for its own frozen Exp1 revision; it cannot validate
the new larger-table implementation. No Exp1 retraining is required by this change.

```bash
cd "$VSC_DATA/CreditICL" &&
git stash push -m "Saved notebook outputs before Exp2-3 update" -- notebooks "output CreditICL/figures" "output CreditICL/CAPTIONS.md" "output CreditICL/All_Results.md" &&
git pull --ff-only &&
git submodule update --init --recursive &&
source scripts/slurm/_activate_env.sh &&
python -m src.utils.preflight &&
python -m src.utils.clean_run --reset-exp0 &&
python -m src.utils.clean_run --reset-exp0 --clean &&
sbatch --clusters=wice scripts/slurm/exp0_cpu.slurm
# Wait for CPU COMPLETED/PASS before submitting the GPU job:
python -m src.utils.preflight --require-exp0-cpu && sbatch --clusters=mindwell scripts/slurm/exp0.slurm
```

The 26-10 implementation adds no dependencies. Do not update working packages or refresh inputs
as a routine step: the latest VSC attempt already passed its dependency audit. If the prerequisite
check reports a real package mismatch, review the PROJECT SIF migration or explicitly authorize
`update_env.sh --legacy-maintenance`; verify all transitive dependencies and `pip check` before CPU validation.
CPU and GPU jobs must use the same environment. Preprocessing cache v8 retains
borrower groups and percentage corrections, fixes Freddie missing-value codes and preserves distinct
extreme feature values. OOD cache v3 retains the same pinned dataset IDs
but restores full rows; a failed refresh cannot silently substitute another dataset.
The CPU reservation is 4 hours (controller 3.8), GPU 24 hours (controller 23.5); these are limits,
not runtime promises. No production array or automatic retry follows a failed validation.
The GPU phase may span allocations under the same frozen code, dependencies and inputs. Each
successful benchmark fold is immediately saved in an atomic journal and CSV. At the allocation
deadline, `DEFERRED`/exit 64 preserves finished stages and folds; resubmit the same GPU script.
Slurm may label exit 64 FAILED, so check the controller's `END status=DEFERRED` message. A receipt
left RUNNING by a hard scheduler kill also supports verified continuation. A genuine FAILED
receipt refuses continuation: inspect its cause instead of blindly repeating it. Keep the same
CPU receipt; no `git pull`, package update, input refresh or output reset between allocations.

RUNNING in the JSON is only the last saved controller state; use `sacct` to learn whether
the allocation is still alive. A cancelled job may leave it behind. The launchers forward
TERM/INT/HUP, report `END status=INTERRUPTED` and retain scratch/evidence; the controller
publishes DEFERRED when it can catch termination. Only a complete, matching PASS receipt
unlocks production. A shell footer, including an old `END status=OK`, is insufficient.
Local edits must first be committed and pushed (agent pushes authorized by Andreas on 26-10-2026) before VSC's `git pull` can
retrieve them; compare `python -c "from src.utils.preflight import source_fingerprint; print(source_fingerprint())"`
on both machines before the clean CPU submission.

Each benchmark cell has a two-hour Exp0 runtime guard. Crossing it fails validation and blocks
production, even if the mathematics might eventually finish. No rows/features/folds/ensemble
members are removed to meet this guard. TabICL uses upstream ISAB key/value caches plus bounded
row blocks for large embeddings, retaining the entire training context. Mandatory CUDA tests
compare this path with the unmodified upstream model before the capacity probes. Logs expose
column/row progress, model timings and process-tree RAM; improved full-table runtime must be
measured on the actual B200 before quoting a campaign ETA.

Attempts live under `experiment_0/validation/<attempt>/` on both tiers. The canonical receipt is
`output CreditICL/experiment_0/report/validation.json`; it is RUNNING, DEFERRED, FAILED or PASS. Previous
attempts remain until an explicit reset. CPU completion is recorded in `cpu_validation.json`;
the GPU receipt binds that exact CPU receipt. Rerunning CPU invalidates an older GPU PASS.
Exp0 reference probes are isolated and are never published as complete production results.
The launcher refuses production work if tests, code, configurations, packages, inputs or artifacts
changed after validation. Exp3's settings are prescribed in the versioned configuration;
no YAML edit is required or exempted. Each training run records its fully resolved choice.

After completion:

```bash
python -m src.utils.preflight --require-exp0
python -m src.validation.verify
```

Only a current PASS authorizes new production submissions. Preserve already completed Exp1 runs;
they do not need retraining because Exp2's recipe changed. The verifier estimates short-route
training hours only; long-context TabICL Exp2 cost stays unknown until representative throughput
is measured, and queue delay is never predicted. Download the small tier plus the
benchmark/report files from project storage, retaining the attempt directory structure. Checkpoint
binaries can remain on VSC; the local verifier reports that omission and VSC preflight verifies them
in place. Run `python -m src.validation.verify --root "/path/to/parent"`, where that parent contains
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
campaign completed, including its benchmark. Today's Exp1 has 12 main and 15 component runs per track
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

## 12. Experiments 1, 2 and 3: current commands

After pulling a reviewed change, run CPU Exp0 and then GPU Exp0 for that revision. Never pull
while a campaign is active. `exp0.slurm` continues only matching code/packages/inputs/CPU receipts;
a stale previous campaign starts a fresh isolated attempt. `DEFERRED` can be resubmitted without
changes; a current `FAILED` needs diagnosis. Preserve Exp1 outputs throughout.

If old cluster notebook executions block a pull, preserve just those tracked artifacts first:

```bash
git stash push -m "VSC notebook renders before update" -- notebooks "output CreditICL/All_Results.md" "output CreditICL/CAPTIONS.md" "output CreditICL/figures"
git pull --ff-only
git submodule update --init --recursive
```

The stash excludes untracked experiment results/checkpoints. Keep it until any handwritten
notebook changes have been recovered; do not automatically pop old rendered artifacts back.
New VSC notebook execution writes copies, figures and summaries under the current Exp0
attempt's `notebooks/` directory. Local publishing still saves the tracked artifacts in place.
This deliberate cluster-only template adaptation avoids changing checkout files during validation.

```bash
sbatch --clusters=wice scripts/slurm/exp0_cpu.slurm
# Once CPU completes successfully:
python -m src.utils.preflight --require-exp0-cpu && sbatch --clusters=mindwell scripts/slurm/exp0.slurm
# Once GPU completes successfully:
python -m src.utils.preflight --require-exp0 && python -m src.validation.verify
python -m src.utils.run_experiment 1 --phase benchmark --submit
```

Each experiment planner is safe to repeat: it skips completed/queued arms and waits for training
before benchmarking. Exp1 includes only missing quarter-mixture work if the other 48 runs and
benchmarks are already complete. All 54 training runs were complete in the 29-10 snapshot.
Run the fixed Exp2 grid after Exp1 benchmarking and current validation.
The explicitly authorized one-off `scripts/slurm/deploy.slurm` controller performs that
sequence and stops immediately after submitting Exp2; its durable JSON receipt records
all accepted job IDs. It stops on failed validation or incomplete Exp1 benchmark receipts.
Exp2 retains five checkpoint benchmarks per run. All rows/features/five folds remain.
For Exp2/Exp3, keep `--single-cluster`: these commands use the Mindwell B200 allocation validated
by Exp0. The planner's optional historical LGD split across wICE A100s is not validated by a B200
capacity result and is not the launch recipe here.

```bash
python -m src.utils.run_experiment 2 --single-cluster --submit
# Repeat after training to submit missing benchmarks. Then proceed with Exp3:
python -m src.utils.run_experiment 3 --single-cluster --submit
# Repeat after training for benchmarks.
```

The former 200k-update scratch Exp3 is abandoned. Current Exp3 has 32 new runs and 8 reused Exp2 baselines, all at 1,500 updates.
Both CPT rates and all adapter settings are fixed before benchmarking; no pilot selects them.
The 125/250/500/1,000/1,500 checkpoints describe adaptation trajectories, with 1,500 as primary endpoint. Never delete Exp1 or Exp2 baseline checkpoints while
these comparisons are active. Check project capacity before launching: retained checkpoints,
rolling resume files and full-table benchmark journals all consume project storage. The 24-hour
Exp0 allocation is a limit, not a prediction of total campaign duration.
