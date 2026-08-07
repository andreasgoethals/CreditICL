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

## 2b. Our actual account and cluster setup

Confirmed from the sibling **CreditPFN** project, which already runs on this
setup (`3. CreditPFN/CreditPFN/docs/VSC_GUIDE.md`):

- **Account: `lp_verbekelab`.** It has Mindwell access.
- **Neither wICE nor Mindwell has its own login node.** You always SSH into
  **Genius** and submit to the target cluster with `#SBATCH --clusters=<name>`.
  Every `.slurm` file in this repo already does that.
- CreditPFN's split, which we follow: **Mindwell `gpu_b200` for training**,
  wICE `gpu_h100` / `gpu_a100` for evaluation, wICE `batch` for CPU data prep.

## 3. Walltime ceilings and the checkpoint/resume gap

- **wICE and Mindwell: 72 h (3 days), and that is a hard ceiling for GPUs.**
  There is **no `gpu_*_long` partition on wICE or Mindwell** — only the CPU
  partitions have 7-day `_long` variants (`batch_long`,
  `batch_icelake_long`, `batch_sapphirerapids_long`). Genius's older
  `gpu_p100_long` / `gpu_v100_long` do allow 7 days, but those are P100/V100 and
  far too slow to be worth it.
- **Default walltime if you omit `--time` is 1 hour** (30 minutes on
  `*_debug`). Always set `--time`.

### The gap, stated plainly

**The VSC documentation contains no Slurm requeue recipe.** The string
`requeue` does not appear anywhere in it. The only documented checkpointing
facility is the **`csub` / BLCR framework**, which is a Torque-era tool, not
a Slurm-native mechanism, and not applicable to our GPU jobs.

**Consequence for us:** application-level checkpointing is **mandatory, not
optional**. TabICLv2's reference regressor stage 1 is **500,000 steps** —
far beyond any 72 h ceiling. `src/train/` must therefore:

1. write a checkpoint every *N* steps to `$VSC_SCRATCH`, including optimizer
   and LR-scheduler state and the prior's RNG state (otherwise a resumed run
   does not resample the same task stream);
2. detect and resume from the newest checkpoint on startup, idempotently;
3. self-resubmit, since nothing will do it for us. The pattern is for the
   job script to `sbatch` its own successor after the training process exits
   cleanly, guarded by a step-count check so the chain terminates.

Budget checkpoint time inside the walltime: a job killed at the limit loses
everything since the last write.

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
bash scripts/submit_pipeline.sh both
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

## 9. Open questions to resolve against the live system

The documentation does not settle these; check before the first big run.

- Which `lp_*` credit accounts exist for this project, and their balances.
- Whether Mindwell `gpu_b200` is accessible under those accounts (it is a
  newer cluster; the docs mention a `lp_mindwell_pilot` account in examples,
  which suggests gated access).
- Actual `MaxSubmitJobsPerUser` and `MaxTRESPerUser` per QoS, which caps how
  wide the array fan-out can go.
- Whether compute nodes have outbound internet — assume **not**, and
  pre-download everything (including any HuggingFace assets) to `$VSC_DATA`.
  This is why `wandb` is an opt-in extra and must run offline.
- Current charge rates via `scontrol show partitions --clusters=wice`, since
  the table in §1 is a snapshot.

---

## 10. Compute budget: what is actually affordable

Verified numbers, not estimates:

| | source |
|---|---|
| TabICLv2 full pretraining | **24.5 GPU-days per model** (20 + 2.5 + 2 across three stages), H100 80GB — the paper's own figure |
| TabICL v1 | 60 A100-days |
| O'Prior's controlled study | **40,000 datasets per prior**, batch 4, 1,000 steps × 10 epochs |

Our `config/*.yaml` replicates **O'Prior's** budget exactly, not TabICLv2's, and that is
deliberate. O'Prior states the reason: *"at reduced scale, pretraining requires only tens
of thousands of synthetic datasets rather than millions, making controlled prior ablations
computationally feasible without sacrificing the qualitative conclusions."*

### Why the full budget cannot be the grid

| checkpoints | GPU-days |
|---|---|
| 2 | 49 |
| 6 | 147 |
| 48 (the current grid) | **1,176 = 3.2 GPU-years** |

Unlimited credits do not help: wall-clock is the constraint. Hence two phases.

**Phase 1 — the science.** 40K datasets, all 48 arms, 1 GPU each, pools on. Answers
*which* prior mix wins. ~2 hours per arm; the whole grid fits in a day of queueing.

**Phase 2 — the headline.** Full TabICLv2 budget on **2 checkpoints only** (winning arm +
unmodified control), on-the-fly generation, 4–8 GPU DDP, 2–3 chained 72h jobs each.
Answers *is it actually good*.

### Storage rules out pooling at full scale

At the measured 97 KB (LGD) / 131 KB (PD) per dataset, 35M datasets is **3.5–4.7 TB per
variant**, ~16 TB for four pools, against a ~1 TB staging quota. TabICLv2 never stored
its datasets either — 550K steps × batch 64 ≈ 35M means **each dataset is seen once**,
so there are no epochs and no corpus. Phase 2 therefore uses `--prior-source generate`.

At 35M draws the pooling benefit is also negligible: the point of pools is removing
draw-luck between arms, and that noise is ~1/√35M. Pools are for Phase 1.
