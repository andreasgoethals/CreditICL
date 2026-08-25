# Running CreditICL on KU Leuven VSC

Distilled from `tfm-library/repositories/VSC Documentation.txt` (library pin
`21d555a`, extracted 2026-08-05). **Every fact below was read out of that
document**; where the documentation is silent, this file says so explicitly
rather than guessing. Verify against the live docs before a large run —
partitions and charge rates change.

---

## 1. Cost a job *before* submitting it

Use `sam-quote`. Prefix your real submission command with it:

```bash
sam-quote sbatch --account=lp_myproject --clusters=wice --partition=gpu_a100 \
                 --nodes=1 --gpus-per-node=1 --time=24:00:00 job.slurm
```

It returns the credit cost assuming the **worst case** — that the job runs
to its full time limit.

The billing formula for GPU jobs is:

```
(CPU_weight * effective_cores + GPU_weight * num_gpus) * walltime_minutes
```

`effective_cores` is **not** what you requested. If a job requests one core
but the full memory of a node, the whole node is blocked and you are billed
for all of its cores. The CPU term is floored: `floor(weight * cores)`.

### Charge rates (credits per unit per **minute**)

| Cluster | Resource | Type | Weight | ≈ credits/GPU-**hour** |
|---|---|---|---|---|
| Genius | P100 | GPU | 41.6667 | 2,500 |
| Genius | V100 | GPU | 59.5833 | 3,575 |
| wICE | **A100** | GPU | **141.667** | **8,500** |
| wICE | H100 | GPU | 569.444 | 34,167 |
| Mindwell | **B200** | GPU | **437.50** | **26,250** |
| wICE | Icelake | CPU core | 2.54630 | — |
| wICE | Sapphire Rapids / Zen4 Genoa | CPU core | 3.47222 | — |
| wICE | Icelake bigmem/hugemem | CPU core | 4.39815 | — |
| Mindwell | Graniterapids | CPU core | 2.60416667 | — |
| Mindwell | Graniterapids bigmem / B200 host | CPU core | 3.03819444 | — |

### The non-obvious conclusion

**B200 is cheaper than H100** — 437.50 vs 569.444 credits/GPU-minute, a 23%
saving — *and* it is a newer, faster architecture, *and* it gives more CPU
cores per GPU (24 vs 16). There is no regime in this table where H100 is the
rational choice over B200 for our workload.

**A100 is ~3.1× cheaper per hour than B200 and ~4× cheaper than H100.**
Since TabICL's prior generator runs on **CPU** (`--prior_device cpu`), a
prior-generation-bound job spends much of its GPU-minutes idle. For the
nano-scale ablation sweep, A100 buys the most *independent runs* per credit,
which matters more than single-run speed when we have four-plus prior arms
to train.

**Rule of thumb for this project:**

| what | where | why |
|---|---|---|
| debugging, data staging, quick prior sanity checks | `interactive` | **free** |
| the prior-arm ablation sweep (many parallel runs) | wICE `gpu_a100` | cheapest per run; 18 cores/GPU feeds the CPU-bound generator |
| a full-scale confirmation run | Mindwell `gpu_b200` | fastest, 24 cores/GPU, cheaper than H100, NVLink for multi-GPU |
| anything | wICE `gpu_h100` | avoid — dominated by B200 on price *and* cores |

---

## 2. The `interactive` partition is free

At KU Leuven the `interactive` partition **costs no credits**, deliberately,
to keep light work off the billed partitions. Limits:

- max **1 node**
- max **8 cores**
- max **1 virtual GPU slice** on wICE — a MIG slice ≈ 1/7 of an A100's CUDA
  cores and memory — **or 1 full GPU on Mindwell**
- max **16 h** walltime

Mindwell's interactive nodes carry **2× NVIDIA RTX 5000 (Ada), 32 GiB** each
— a *full* GPU, free, for up to 16 hours. That is a genuinely usable
prototyping target for nano-scale prior work, and it is where all data
staging and Lustre↔GPFS transfer jobs should run.

```bash
# Free interactive shell with a full GPU on Mindwell
srun --account=lp_myproject --clusters=mindwell --partition=interactive \
     --ntasks-per-node=8 --gpus-per-node=1 --time=08:00:00 --pty bash -l
```

Note: `--gpu_cmode` is irrelevant on wICE `interactive` because MIG
partitioning is in use there.

---

## 2a. Which partition to submit to — the inventory

Read out of `tfm-library/repositories/VSC Documentation.txt`. **GPU count is queue length**, and
**cores per GPU is our actual bottleneck**, because TabICL generates its prior on the CPU
(`--prior_device cpu` in upstream's own stage scripts).

| target | cluster | partition | **GPUs** | **cores/GPU** | mem/GPU | credits/GPU-h | max walltime |
|---|---|---|---|---|---|---|---|
| `free` | mindwell | `interactive` | 4 × RTX 5000 Ada 32 GiB | 8 | ~32 G | **none** | 16 h |
| `b200` | mindwell | `gpu_b200` | **24** × B200 192 GiB | **24** | 194400 M | 26,250 | 72 h |
| `a100` | wICE | `gpu_a100` | 16 × A100 80 GiB | 18 | 126000 M | 8,500 | 72 h |
| `h100` | wICE | `gpu_h100` | 20 × H100 80 GiB | 16 | 187200 M | 34,167 | 72 h |
| `dbg1h` | wICE | `gpu_a100_debug` | **1** × A100 80 GiB | 18 | 126000 M | — | **1 h** |

```bash
bash scripts/slurm/submit.sh --list          # the table above, from the shell
bash scripts/slurm/submit.sh free lgd        # <where> <track>
bash scripts/slurm/submit.sh b200 pd
```

**The track is an argument, never an environment variable.** `sbatch [options] script args…`
passes trailing arguments straight to the job script, so `submit.sh` hands it the config path
and the job reads it as `$1`. The earlier `CONFIG=… bash submit.sh` form was broken in a way
that is easy to miss: it sets the variable for the *calling* shell, not for `sbatch`'s
environment, so the job never saw it and a run meant as PD went out as a second LGD job.
`submit.sh` now prints `where / track / config / script` before submitting, and refuses a track
it does not recognise.

**`gpu_a100` has the fewest GPUs of any GPU partition available to us.** The first debug
submission sat there behind `Reason: Priority` and never started.

**`gpu_b200` is the best production target on both axes**: the most GPUs (24, so the shortest
queue) *and* the most cores per GPU (24, which is what feeds the prior generator). Note this
inverts the naive ranking — `gpu_h100` is the fastest chip but has the **fewest cores per GPU**
(16) and costs 4× a B200 credit-for-credit, so it is the worst choice for this workload.

**`interactive` costs no credits**, and on Mindwell it is a *full* GPU rather than wICE's MIG
slice. 8 cores is the trade, which is ample for a 1,500-step debug run.

**`gpu_a100_debug`** takes ≤ 1 h and Slurm allows only **one queued job at a time**, so a
4-task array cannot use it. Single-arm checks only.

**Job scripts must not hard-code any of this.** `#SBATCH` directives are defaults;
`submit.sh` overrides them on the command line, and the job reads `$SLURM_CPUS_PER_TASK` to set
`num_workers` so one script is correct on 8 cores and on 24.

## 2b. Our actual account and cluster setup

Confirmed from the sibling **CreditPFN** project, which already runs on this
setup (`3. CreditPFN/CreditPFN/docs/VSC_GUIDE.md`):

- **Account: `lp_verbekelab`.** It has Mindwell access.
- **Neither wICE nor Mindwell has its own login node.** You always SSH into
  **Genius** and submit to the target cluster with `#SBATCH --clusters=<name>`.
  Every `.slurm` file in this repo already does that.
- CreditPFN's split, which we follow: **Mindwell `gpu_b200` for training**,
  wICE `gpu_h100` / `gpu_a100` for evaluation, wICE `batch` for CPU data prep.

## 3. The three limits, and how a long sweep survives them

Everything in this section is verified against `tfm-library/repositories/VSC Documentation.txt`
except the third limit, which is not documented anywhere and comes from experience.

### 3.1 Walltime — 72 h, and `gpu_b200` cannot exceed it

> "In general, the maximum walltime for Mindwell jobs is 3 days (72 hours). Only jobs submitted
> to the `*_long` partitions are allowed to have walltimes up to 7 days."

`gpu_b200` is not a `*_long` partition, so **72 h is the ceiling for every GPU job we run.**
The `interactive` partition allows 16 h with 8 cores and one RTX 5000 Ada, and **costs no
credits**. The default if you omit `--time` is 1 hour — always set it.

Do not request 72 h out of habit. Billing is on *actual* time, so a long request costs nothing
extra, but the backfill scheduler starts short jobs sooner. Request what the arm needs plus a
margin; the resume machinery below makes an underestimate cheap.

### 3.2 Concurrency — two separate caps, and the numbers are not in the docs

VSC caps the number of jobs a user may have **pending plus running**, and separately the total
resources their running jobs may occupy. Both are set per partition QoS; `gpu_b200` uses the
`normal` QoS. The documentation gives the mechanism but not the values, so read them off the
live system:

```bash
sacctmgr show qos normal format=Name%20,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30
```

We have hit both already — `QOSMaxSubmitJobPerUserLimit` and `QOSMaxCpuPerUserLimit` — so
neither is theoretical. **Use the array throttle `%N` rather than trying to stay under the cap
by hand**: `--array=0-74%8` keeps at most 8 tasks active whatever the limit turns out to be,
and Slurm releases the next one as each finishes.

There are **24 B200 GPUs in total** (3 nodes x 8), shared with every other user on Mindwell, so
a throttle above ~8 is optimistic regardless of what the QoS allows.

### 3.3 Unannounced maintenance — the one that is not written down

**Mindwell goes into maintenance without warning, and running jobs are cancelled.** Not often,
but often enough that a multi-day sweep will meet it. The documentation only alludes to this,
under reservations: capacity may be taken back "in case of unforeseen circumstances, such as
hardware failures, critical security vulnerabilities or urgent maintenance."

### 3.4 What we do about all three — the same mechanism

A walltime kill, a node failure and a maintenance drain all arrive as a **signal**. So:

| piece | where | what it does |
|---|---|---|
| `--signal=B:USR1@600` | job script | Slurm signals the batch script 600 s before the wall |
| `trap ... kill -USR1 $TRAIN_PID` | job script | forwards it to python, which Slurm does **not** do |
| `_install_stop_signals` | `src/train/loop.py` | catches USR1/TERM, sets a flag |
| the step loop | `src/train/loop.py` | finishes the step, writes a checkpoint, breaks |
| `return 64` | `scripts/pretrain.py` | exit 64 = "saved, not finished" |
| resubmit on 64 | job script | `sbatch --array=<this task>` — the arm continues |
| `--requeue` | job script | covers node failure, where no signal arrives in time |
| `maybe_resume` | `src/train/loop.py` | picks up the newest checkpoint on start |
| `save_temp_every: 250` | the configs | bounds what an *ungraceful* kill costs to ~10 min |

The child must be **backgrounded and `wait`ed on**: a foreground child would block the trap
until it exited, which is exactly too late. `SLURM_RESTART_COUNT` is printed in the job banner,
so a log tells you whether it is a first attempt or a resumption.

**Consequence: an arm may take longer than the walltime and still complete.** That is what
makes the 72 h ceiling a scheduling detail rather than a design constraint.

---

### 3.5 When the whole cluster goes down — the case the job script cannot cover

The signal chain in 3.4 saves an arm that is **running**. It does nothing for an arm that is
**pending**: a cluster-wide drain cancels every queued task at once, with no signal to anyone.
Resubmitting the array afterwards would redo every finished arm.

`sweep_status` reads the OUTPUT TREE — not `sacct`, which forgets, and not the queue, which is
gone — and classifies each array index:

| state | evidence on disk | what happens on resubmission |
|---|---|---|
| `done` | `summary.json` with `completed: true` | not resubmitted |
| `partial` | a checkpoint, but no completed summary | resumes from the checkpoint |
| `todo` | nothing | starts at step 0 |

```bash
python -m src.utils.sweep_status --config config/Exp1_LGD.yaml
python -m src.utils.sweep_status --config config/Exp1_LGD.yaml --resubmit
```

It prints a collapsed spec — `0-4,9,12-74` rather than 71 numbers — so it can be checked before
committing hundreds of GPU-hours. **Running it repeatedly is safe**: it never touches a `done`
arm, and a `partial` arm continues rather than restarting.

`summary.json` is the authority rather than the checkpoint, because a checkpoint exists from
step 250 onwards and says nothing about whether the arm finished.

---

## 3.6 Which GPU to actually use

`nvidia-smi` memory is not utilisation. Job 11523286 measured **13.08 GB of 178 GB** on a
B200 — but also **0.791 steps/s against a 0.986 ceiling, i.e. 80 % of the compute the card can
deliver** for this model, and the 17-08 training run reported 88.7 % on `utilization.gpu`.
The card is busy; a 28.5M-parameter model at micro-batch 4 simply does not need memory.

**So there is no headroom to reclaim by making the batch bigger**, and `micro_batch` is pinned
at `group_size` = 4 by `validate_micro_batch` regardless. What the idle memory does mean is
that **we are paying for a card we do not need**:

| partition | credits/GPU-h | + CPU | all-in | vs B200 |
|---|---|---|---|---|
| mindwell `gpu_b200` | 26,250 | 24 cores, 4,375 | **30,625** | 1.00x |
| wICE `gpu_a100` | 8,500 | 18 cores, 2,750 | **11,250** | **0.37x** |
| wICE `gpu_h100` | 34,167 | 16 cores, 2,222 | 36,389 | 1.19x |
| mindwell `interactive` | 0 | 0 | **free** | 8 cores, 16 h, 1 job |

An A100 80 GB holds 13 GB six times over and costs **2.7x less all-in**. It is cheaper per arm
as long as it is less than 2.7x slower — which is plausible but **not measured**, and this
workload is latency-bound rather than throughput-bound (which is exactly why the memory is
idle), so raw FLOP ratios do not predict it. Measure before assuming:

```bash
bash scripts/slurm/submit.sh a100 lgd scripts/slurm/gpubench.slurm
python -m src.utils.compare_gpubench $VSC_DATA/CreditICL/output/logs/gpubench_*.json
```

Compare `end_to_end.projected_hours_12500_steps`. If the A100 is under 2.7x the B200's 4.39 h
(i.e. under ~11.9 h), it is the cheaper card for Exp1 — and it frees the B200s for other work.

The launchers take the partition from the command line and read `SLURM_CPUS_PER_TASK` for the
worker count, so one script is correct on 24 cores and on 18. A resumed arm returns to the
cluster and partition it started on.

---

### 3.7 Genius (P100 / V100) is not an option

Worth stating with the evidence, because the queue on Genius is separate from Mindwell's and
wICE's, so it looks attractive when another project is occupying those.

**It cannot run this code.** Our own benchmark output settles it in one line:

    compiled_for  ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']

P100 is **sm_60** and V100 is **sm_70**. Neither is in the installed wheel, so
`has_kernels_for_this_card` would come back False and every kernel would fall back or fail.

Two further reasons, each independently disqualifying:

- **No bfloat16.** bf16 needs Ampere (sm_80) or later. Our AMP path is
  `autocast("cuda", dtype=torch.bfloat16)` and AMP is a measured **2.07x** — losing it doubles
  every arm.
- **Memory.** Peak is 13.08 GB *with* bf16; in fp32 it is roughly double, against a P100's
  16 GB.

So the real choice is **Mindwell `gpu_b200` or wICE `gpu_a100` / `gpu_h100`** — and those are
two different clusters with two different queues, which is the flexibility that mattered. A100
is sm_80 and H100 is sm_90; both are in the wheel.

---

### 3.8 The recovery procedure is one command

This cluster stops routinely — unannounced maintenance drains, node failures, the 72 h
walltime, and running out of credits until the balance is topped up. A plan made of five
`sbatch` commands in a fixed order will be executed wrongly after a week's interruption,
because nobody remembers which of the five already happened.

```bash
python -m src.utils.run_experiment 1            # Exp1: what is done, what is next, what is blocked
python -m src.utils.run_experiment 1 --submit   # Exp1: submit whatever is ready
```

**The experiment number is required** — never a default — and is printed at the top of the
report, so "am I looking at Exp1 or Exp3?" is answered on the page, not from memory. One
invocation drives exactly ONE experiment's two phases (train -> benchmark) across both tracks.
`run_experiment 2` and `run_experiment 3` **refuse to run** while their configs still hold
`FILL_FROM_EXP1`, so you cannot start a later experiment before Exp1 has chosen a winner.

It reads the **output tree** — not `sacct`, which forgets, and not the queue, which a drain
empties — and submits only what is missing. **Running it repeatedly is the intended usage.**
After any interruption, whatever the cause, the recovery procedure is to run it again.

```
==============================================================================
 CREDITICL — EXPERIMENT 1 — read from the output tree, not from memory
==============================================================================
  [x] exp1-lgd-train             done        75/75
  [ ] exp1-lgd-benchmark         ready        0/76
  [~] exp1-pd-train              running     42/75
  [-] exp1-pd-benchmark          blocked      0/76  waiting on exp1-pd-train
```

Four guarantees, each with a test:

| | |
|---|---|
| `benchmark` cannot start early | phase 2 scores what phase 1 writes; 74 of 75 arms still blocks it |
| a drain costs only what was pending | a partial sweep resubmits `--array=40-49,52-74`, not `0-74` |
| a re-run never doubles a queued job | matched on the Slurm job name via `squeue` |
| a broken `squeue` errs towards submitting | a duplicate can be cancelled; work that never starts cannot |

**LGD is split across two clusters and PD is not** — on a FRESH sweep only. Two clusters means
two queues and roughly half the wall-clock; there is nothing about LGD that needs wICE, and
`--single-cluster` turns it off. A partial sweep goes back to one cluster, because the
surviving indices are scattered and splitting a scattered spec across two partitions gains
nothing but confusion.

---

## 4. Storage: three tiers, and the Lustre-vs-GPFS rule

### Where CreditICL puts things

Same split as CreditPFN, implemented in
[`src/utils/paths.py`](../src/utils/paths.py):

| tier | path | what goes here | backup | quota |
|---|---|---|---|---|
| **project staging** | `/lustre1/project/stg_00211` (`$VSC_PROJECT_LUSTRE1/stg_00211`) | the **big files**: datasets, **trained checkpoints**, result CSVs | no | large (≥1 TB), **low inode budget** |
| **personal data** | `$VSC_DATA` | the **repo**, plus small durable output: logs, `metrics.jsonl`, resolved configs, figures | **yes** | **75 GiB** — tight |
| scratch | `$VSC_SCRATCH` | working scratch only | no | 500 GiB, **purged after 30 days without access** |

Why datasets and checkpoints go to **staging**: they are the biggest artefacts,
`$VSC_DATA`'s 75 GiB cannot hold them, and scratch gets purged. Why per-step
metrics do **not**: staging has a low inode budget, so it wants few big files,
not thousands of small ones.

Override the staging root with `$CREDITICL_STAGING_ROOT` if it ever moves.
`paths.resolve_writable()` probes staging with a real write at job start and falls
back to `$VSC_DATA` with a loud warning — CreditPFN lost a run's checkpoints to
unwritable staging on 2026-07-03, so this is a real failure mode, not a
hypothetical one.

### The raw VSC picture

| Variable | Path | Type | Where | Backup | Quota |
|---|---|---|---|---|---|
| `$VSC_HOME` | `/user/leuven/3../vsc3....` | NFS | all clusters | **yes** | **3 GiB** |
| `$VSC_DATA` | `/data/leuven/3../vsc3....` | NFS | all clusters | **yes** | **75 GiB** |
| `$VSC_SCRATCH` | `/scratch/leuven/3../vsc3....` | **Lustre** | Genius, wICE | no | 500 GiB |
| `$VSC_SCRATCH` | (same variable) | **GPFS** | Mindwell | no | 500 GiB |
| `$VSC_SCRATCH_NODE` | `/tmp` | ext4 | Genius (node-local) | no | 200 GiB |

**The rule, and it is enforced:** Genius and wICE jobs must do their I/O on
**Lustre**; Mindwell jobs must use **GPFS**. `$VSC_SCRATCH` already resolves
to the correct one for the node you are on — so *use `$VSC_SCRATCH`, never a
hard-coded path*. The documentation states that non-complying jobs **can be
cancelled by administrators without prior notice**.

Cross-filesystem access, for staging between clusters:

- `$VSC_SCRATCH_LUSTRE1` and `$VSC_SCRATCH_GPFS1` point at the "other" one.
- Lustre is reachable from Mindwell; GPFS is reachable from wICE and from
  the Genius **login** nodes — but **not from Genius compute nodes**.
- Do transfers as jobs on the `interactive` partition (free), not on a login
  node, unless they take under a couple of minutes.

### Staging data without tripping the purge

**`$VSC_SCRATCH` deletes files not *accessed* for more than 30 days.**

The trap: `mv` does not count as an access, and `rsync` with timestamp
preservation copies an old atime. A dataset moved to scratch can therefore
be purged almost immediately because its last-access timestamp is historic.

**Do this instead** — copy (not move), then touch:

```bash
cp -r "$VSC_DATA/crediticl/data" "$VSC_SCRATCH/crediticl/data"
find "$VSC_SCRATCH/crediticl" -type f -exec touch {} +
```

Keep the authoritative copy of `data/` in `$VSC_DATA` (75 GiB, backed up)
and treat scratch as a working cache that may vanish. **Checkpoints we care
about must be copied back to `$VSC_DATA` when a run finishes** — scratch is
not backed up and is purged.

---

## 5. Python environment on VSC

**One command, once:**

```bash
cd $VSC_DATA/CreditICL && bash scripts/slurm/setup_venv.sh
```

On a **login node** — compute nodes have no outbound internet, so pip cannot reach PyPI there.
It builds `$VSC_DATA/CreditICL/.venv` from `pyproject.toml`, installs torch from the CUDA index
first, then `pip install -e ".[dev,eval]"`, then verifies every import *and* that our model
matches the released TabICLv2 checkpoints. Idempotent; `--recreate` starts over.

### Auto-activate it when you `cd` into the repo

**Two commands, once:**

```bash
bash scripts/slurm/shell_hook.sh --install
```

```bash
source ~/.bashrc
```

That appends **one line** to `~/.bashrc` which sources
[`scripts/slurm/shell_hook.sh`](../scripts/slurm/shell_hook.sh). The hook itself lives in the
repo, so `git pull` keeps it current and `~/.bashrc` never needs editing again. Re-running
`--install` is safe; `--uninstall` removes it (with a backup) and `--status` shows what is wired
up and which venvs exist.

Check it took:

```bash
cd $VSC_DATA/CreditICL && python -c "import sys; print(sys.prefix)"
```

It must print a path ending in `.venv-<arch>`. If it prints another project's venv, the hook is
not registered — run `bash scripts/slurm/shell_hook.sh --status`.

The hook also **stands down an already-active venv from another project** before taking over.
That mattered in practice: `TabPFNCredit/tabpfncreditvenv` was active and kept winning the PATH
race, so `pip install` reported *"already satisfied"* for packages CreditICL had never had.

<details>
<summary>What the hook does, if you want to read it rather than install it</summary>

The same code as `scripts/slurm/shell_hook.sh`. Prefer installing the file — a copy pasted into
`~/.bashrc` stops receiving fixes.

```bash
# --- CreditICL: auto-activate the project venv -----------------------------
_crediticl_auto_venv() {
    local repo="${VSC_DATA}/CreditICL"
    local venv="${repo}/.venv-${VSC_ARCH_LOCAL:-generic}"
    # Fall back to any venv in the repo, so this keeps working on a login node
    # whose $VSC_ARCH_LOCAL differs from the one that built it.
    if [[ ! -x "$venv/bin/python" ]]; then
        local found
        for found in "$repo"/.venv-* "$repo/.venv"; do
            [[ -x "$found/bin/python" ]] && { venv="$found"; break; }
        done
    fi
    case "$PWD/" in
        "$repo"/*)
            if [[ -x "$venv/bin/python" && "${VIRTUAL_ENV:-}" != "$venv" ]]; then
                # The module that BUILT this venv, recorded by setup_venv.sh.
                local mod
                mod="$(cat "$venv/.python_module" 2>/dev/null)"
                [[ -n "$mod" ]] && module load "$mod" 2>/dev/null
                source "$venv/bin/activate"
            fi
            ;;
        *)
            # Only deactivate OUR venv. Someone else's stays alone.
            if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" == "$repo"/* ]]; then
                deactivate 2>/dev/null
            fi
            ;;
    esac
}
PROMPT_COMMAND="_crediticl_auto_venv${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
```

</details>

**`PROMPT_COMMAND`, not an overridden `cd`.** It fires before every prompt, so it also works
after `pushd`, a subshell, or landing in the directory from a symlink — all of which an
overridden `cd` misses.

**The module load is not optional, and the name is read from disk.**
`.venv/bin/python` is a thin link to the Lmod interpreter; without the module its shared
library is absent and the venv fails in a way that reads like a corrupt install. The module
name is read from `.python_module` beside the venv rather than hard-coded, because module trees
on VSC are **per-architecture** — the first version of this pinned
`Python/3.12.3-GCCcore-13.3.0`, which exists on skylake and made Lmod report *"exist but cannot
be loaded as requested"* on a login node with a different tree.

**Never `module --force purge`.** The `cluster/*` modules are *sticky* and set up the
architecture-specific `MODULEPATH`; force-purging removes them too, which collapses the tree so
that even a module that genuinely exists cannot be loaded.

**Why one venv per project at all.** Packages were being installed into
`TabPFNCredit/tabpfncreditvenv`, so `pip install` reported *"already satisfied"* for things
CreditICL had never had, and the two projects could silently disagree about the torch version
a result was produced with.

### Jobs need none of this

`scripts/slurm/_activate_env.sh` finds `$VSC_DATA/CreditICL/.venv` itself, loads the module,
and verifies the imports. `~/.bashrc` is **not reliably sourced in a batch job**, so a job must
never depend on the hook above — which is why the activator does the work explicitly.

Historically that activator expected a **conda** env, and warned that an auto-activated venv
would shadow conda silently. It now prefers the project venv and falls back to conda, so the
hook above and the job path agree instead of fighting.

### Why the venv lives on `$VSC_DATA`

A venv with torch is ~5–8 GB across **tens of thousands of small files**. Project storage
(`/lustre1/...`) has a **low inode budget** — it is sized for few big files — so a venv there
exhausts inodes long before space. `$VSC_DATA` is 75 GiB and handles the file count.

---

There is **no PyTorch module** in the VSC documentation — the string does
not appear. We install torch ourselves into a venv.

### Toolchains

Available toolchains run **2023a through 2025a**. The 2023a toolchain gives
**`Python/3.11.3-GCCcore-12.3.0`**, which matches this project's
`requires-python = ">=3.11,<3.12"` — that alignment is deliberate.

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a     # numpy, scipy, pandas
```

### One venv **per microarchitecture**

wICE (Icelake / Sapphire Rapids / Zen4 Genoa) and Mindwell (Granite Rapids /
Turin) are **different microarchitectures**. A venv built on one is not
reliably usable on the other. Follow the documented convention of suffixing
with `$VSC_ARCH_LOCAL`:

```bash
cd "$VSC_DATA/crediticl"
python3 -m venv "venv-${VSC_ARCH_LOCAL}" --system-site-packages
source "venv-${VSC_ARCH_LOCAL}/bin/activate"
pip install --upgrade pip
pip install torch --no-cache-dir            # CUDA build for the cluster
pip install -e . --no-cache-dir --no-build-isolation
```

`--system-site-packages` reuses the module-provided numpy/scipy/pandas
instead of recompiling them.

### Rules that will bite you

- **Never put a venv in `$VSC_HOME`** — 3 GiB, and a torch install alone
  exceeds a large fraction of it. Use `$VSC_DATA`.
- Venvs contain tens of thousands of small files, which parallel filesystems
  handle badly. If this becomes a problem, the documented answer is
  containerising via **`hpc-container-wrapper` (Tykky)**.
- If using Conda instead: load the **centrally installed `Miniforge3`
  module** — the docs explicitly say *do not* install Miniforge yourself.
  Then redirect its directories, or it will fill `$VSC_HOME`:
  ```bash
  conda config --append envs_dirs $VSC_DATA/conda_envs
  conda config --append pkgs_dirs $VSC_SCRATCH/conda_pkgs
  ```
  Avoid `conda init bash` — it makes Conda always-on and conflicts with the
  module system.

---

## 6. KU Leuven job-script rules that contradict the generic VSC pages

These are the ones that cost time or credits if you follow generic Slurm
advice instead.

1. **Shebang must be `#!/bin/bash -l`**, not `#!/bin/bash`. The `-l` (login
   shell) is what makes `~/.bashrc` run and the sticky `cluster` module load.
   Without it, `module load` fails inside the job.

2. **`-M/--clusters` is mandatory when submitting.** KU Leuven has made it
   required, unlike stock Slurm. It also defaults by *where you run the
   command*, so from a Genius login node you must pass `-M wice` or
   `-M mindwell`. Monitoring commands (`squeue`, `sacct`, `scontrol`) need it
   too; `-M all` works for those.

3. **`-A/--account` is mandatory** and must hold enough credits.

4. **Never `module --force purge` in a job.** It unloads the sticky `cluster`
   modules and breaks the job environment. Plain `module purge` is correct —
   sticky modules survive it.

5. **Memory requests silently multiply your core count and your bill.** Each
   partition's default memory-per-core is also its *maximum*. Asking for
   more allocates more cores, via `ceil()`. On a 2500 MB/core partition:
   `--mem-per-cpu=4000M` → `ceil(4000/2500) = 2` cores; `--mem-per-cpu=5G`
   → `ceil(5*1024/2500) = 3` cores, i.e. **triple the credits**. Beware the
   `G` multiplier.

6. **Use `mpirun`, not `srun`, for MPI.** This Slurm install has **no PMI
   support**, so MPI launched via `srun` may hang. `srun`'s purpose here is
   interactive jobs. *For us this means `torchrun` rather than `srun` for
   multi-GPU* — `torchrun` uses its own rendezvous and is unaffected.

7. **Environment variables are not propagated** from the submitting shell.
   Jobs start from your login environment plus `~/.bashrc` only. If you need
   to pass one, `--export` requires re-listing the minimum:
   `--export=HOME,USER,TERM,PATH=/bin:/sbin,FOO=bar`.

8. **Respect the CPU-per-GPU caps** (§7) or you get a warning and, on repeat,
   attention from the administrators.

Useful KU Leuven-specific tooling: `slurm_jobinfo <jobid>` (readable job
summary), `slurmtop` (cluster state), `sacctmgr show qos ...` (your
concurrent-job limits — partitions map to QoS `debug` / `interactive` /
`long` / `normal`).

---

## 7. CPU cores and memory available **per GPU**

Requesting more than this raises a warning. This table drives our `--n_jobs`
for prior generation, which is the CPU-bound half of every training job.

| Cluster | Partition | Max cores / GPU | Max memory / GPU (MiB) |
|---|---|---|---|
| Genius | `gpu_p100*` | 9 | 45,000 |
| Genius | `gpu_v100*` | 4 | 84,000 |
| wICE | `interactive` | 8 | 60,000 |
| wICE | `gpu` / `gpu_a100` | **18** | 126,000 |
| wICE | `gpu_h100` | 16 | 187,200 |
| Mindwell | `gpu_b200` | **24** | 194,400 |

TabICLv2's reference scripts use `--n_jobs 16` for prior generation
alongside 4 GPUs. Scale it to the partition: **A100 → `--n_jobs 16`** (of
18, leaving headroom for the training process), **B200 → `--n_jobs 20`** (of
24), **H100 → `--n_jobs 14`** (of 16 — one more reason to skip H100).

### Hardware worth knowing

- **Mindwell `gpu_b200`:** only **3 nodes, 24 B200 total**. 8× B200 SXM6
  (Blackwell, 192 GiB HBM each, NVLink), 2× AMD EPYC 9655 (Turin, 96 cores
  each → 192 cores/node), 1536 GiB RAM, 960 GB local NVMe. Few nodes means
  real queueing — do not put the critical path on it without slack.
- **wICE `gpu_a100`:** 4 GPUs/node, so a 1-GPU job should stay within ~1/4
  of the node's CPU and memory.
- A `gpu_a100_debug` partition exists for full-GPU testing, but **only one
  such job may be queued at a time**.

---

## 8. Job-script skeletons

### Single-GPU ablation run (wICE A100 — the default for prior arms)

```bash
#!/bin/bash -l
#SBATCH --account=lp_myproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=cicl-prior
#SBATCH --output=%x-%A_%a.out

module purge                      # NOT --force purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
source "$VSC_DATA/crediticl/venv-${VSC_ARCH_LOCAL}/bin/activate"

# $VSC_SCRATCH resolves to Lustre here; do all I/O against it.
export CICL_SCRATCH="$VSC_SCRATCH/crediticl"
mkdir -p "$CICL_SCRATCH/checkpoints"

cd "$VSC_DATA/crediticl"
python -m scripts.pretrain --config "config/priors/${PRIOR_ARM}.yaml" \
                           --out "$CICL_SCRATCH/checkpoints/${PRIOR_ARM}" \
                           --n-jobs 16 \
                           --resume auto
```

### The whole pipeline as one dependency chain

One command submits everything and you can log out:

```bash
bash scripts/slurm/submit_pipeline.sh both
```

Five stages, each on the hardware it actually needs:

| # | stage | where | shape | why |
|---|---|---|---|---|
| 1 | preprocess | 1 CPU node | single job | 21 datasets in pandas; minutes, no GPU |
| 2 | prior pools | CPU nodes | **2 arrays × 20 tasks** | generation is CPU-bound (ExtraTrees fits); a GPU here would idle |
| 3 | verify | 1 CPU task | single job | the gate: equal, complete pools or the chain stops |
| 4 | pretrain | GPU | array over the config grid | the only stage that needs a GPU |
| 5 | evaluate | 1 GPU node | single job | baselines + our checkpoints on real data |

Three scheduling choices worth knowing about:

* **The two variants are separate array jobs.** `original` and `credit_v1` do not
  depend on each other, so submitting them separately makes 40 tasks eligible at
  once instead of 20-then-20. Slurm backfills small CPU tasks readily, so this is
  close to free parallelism.
* **`afterok` on an array job waits for every task.** That is what makes stage 3
  meaningful: if one of 40 shards dies, training never starts, instead of training
  on a pool that is quietly 2,000 datasets short.
* **Stage 5 uses `afterany`, not `afterok`.** If 2 of 48 training runs fail we still
  want the numbers for the 46 that finished.

Prior generation pins `OMP_NUM_THREADS=1`. The parallelism is the array, not the
maths; without the pin, 8 concurrent shards each spawning 8 BLAS threads
oversubscribe the node and every task slows down together.

Each array task also sleeps a random 1–20 s before starting. Twenty tasks opening
the same repo simultaneously is a metadata storm, and Lustre punishes many small
concurrent opens far harder than it punishes reads.

### Fanning the prior arms out in parallel (array job)

Since credits are not the binding constraint and parallelism is, one array
task per prior arm is the right shape — each arm is fully independent.

```bash
#SBATCH --array=0-3
ARMS=(arm_a_default arm_b_general_realism arm_c_credit arm_d_complex)
export PRIOR_ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
```

Check your concurrent-job limit first, since array tasks count against it:

```bash
sacctmgr show qos normal format=Name%20,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30
```

### Multi-GPU, single node (Mindwell B200 — confirmation runs)

Use `torchrun`, not `srun` (§6.6). TabICLv2's reference recipe used 4 GPUs.

```bash
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --time=72:00:00

torchrun --standalone --nproc_per_node=4 -m scripts.pretrain \
         --config "config/priors/${PRIOR_ARM}.yaml" --n-jobs 20 --resume auto
```

Note this job runs on **GPFS** scratch, and it needs a
`venv-${VSC_ARCH_LOCAL}` built on Mindwell — not the wICE one.

---

## 9. Settled against the live system

Answered by running on it. Kept because "we checked" is worth as much as the answer.

| question | answer | how we know |
|---|---|---|
| Which account? | `lp_verbekelab` | jobs submitted and billed under it since 14-08-2026 |
| Is `gpu_b200` reachable under it? | yes | job 11521108 and others ran there |
| Per-GPU core/memory ceiling | 24 cores, 194,400 MiB | VSC "CPU resource limits in GPU jobs" |
| B200 charge rate | 437.50 credits per GPU-**minute** (26,250/h) | VSC charge-rate table |
| CPU charge rate on Mindwell | 3.038 credits per core-minute | same table |
| Do compute nodes have outbound internet? | **no** | why `prior_cache/ood` can only be built on a login node |
| `MaxSubmitJobsPerUser` / `MaxTRESPerUser` | still unread — use `%N` throttling | see §3.2 |

Still worth re-checking before a big submission: the credit balance (`sam-balance`), and
whether the staging directory is writable (`python scripts/check_storage.py`). A run that
cannot write where it thinks it can reroutes silently to `$VSC_DATA` and dies when the 75 GiB
quota goes.

---

## 10. Compute budget: what is actually affordable

Measured, not estimated. B200, batch 64, micro-batch 4, the config's prior shape.

| | value |
|---|---|
| B200 GPU-hour | **26,250 credits** (+ ~2,200 for 24 cores) |
| one Exp1 arm, 12,500 steps | **~16-22 h** (bracketed; the benchmark pins it) |
| **Exp1, 75 arms** | **~1,200-1,650 GPU-hours = 33-46 M credits** |
| TabICLv2's own full pretraining | 24.5 GPU-days per model, H100 (the paper's figure) |

At the array throttle `%8`, 75 arms of ~20 h is `ceil(75/8) x 20 = 200 h`, about **8 days of
wall-clock**. At `%16` it is 4 days. The throttle, not the credit balance, is what sets how
long the sweep takes — so decide it against the QoS limits in §3.2 rather than by feel.

### Why the full upstream budget cannot be the grid

Upstream's stage 1 is 500,000 steps. At our measured rate that is ~347 h **per arm**, and 75
arms would be 26,000 GPU-hours. Our 12,500 steps is 2.5 % of stage 1, and that is the whole
reason Exp1 is a *screening* tier: it ranks priors, and only Exp2 runs the winner long enough
for the number to mean anything on its own.

### Storage rules out pooling at full scale

At the measured 97 KB (LGD) / 131 KB (PD) per dataset, 35M datasets is **3.5-4.7 TB per
variant**, ~16 TB for four pools, against a ~1 TB staging quota. TabICLv2 never stored its
datasets either — 550K steps x batch 64 means **each dataset is seen once**, so there are no
epochs and no corpus. Exp2 therefore uses `--prior-source generate`.

Pools remain useful for Exp1, where they remove draw-luck between arms that are only 12,500
steps long.
