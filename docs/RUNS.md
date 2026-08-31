# RUNS.md — every cluster run, written up

One entry per run on the VSC. This is the project's laboratory notebook: what was submitted,
what came back, what was wrong with it, and what we think it means.

**Newest first. Dates `DD-MM-YYYY`.** Never delete an entry — a run that failed is evidence,
and a run nobody wrote up will be repeated.

Related, and deliberately separate:

| file | holds |
|---|---|
| **this file** | the full write-up of each run: config, numbers, bugs, interpretation |
| [`AGENTS_MEMORY.md`](AGENTS_MEMORY.md) | the one-line index of runs, and the **dead ends** |
| [`CHANGELOG.md`](CHANGELOG.md) | changes to the *repository*, not to what we know |

---

## The workflow

**1. Before submitting.** Read this file and `AGENTS_MEMORY.md`. Then, locally:

```powershell
.CreditICL\Scripts\python.exe -m src.utils.smoke_test --task lgd --steps 3
.CreditICL\Scripts\python.exe -m pytest -q
```

And **on the login node**, because a run that cannot write where it thinks it can will
silently reroute to `$VSC_DATA` and only fail once the 75 GiB quota is gone:

```bash
python scripts/check_storage.py
```

**1b. Copy down the last run BEFORE clearing it.** Outputs accumulate and a rerun mixes old
files with new, but a checkpoint deleted before it was copied is a bug you cannot diagnose —
the LGD NaN was lost exactly that way.

```bash
# `--ignore-failed-read` + a glob: manifests does not exist until a run has written one, and
# plain tar exits 1 on a missing path, so the archive that was supposed to protect the run
# never gets created.
tar czf ~/crediticl_$(date +%Y%m%d).tar.gz --ignore-failed-read -C "$VSC_DATA/CreditICL" output
python -m src.utils.clean_run            # LISTS what is there, deletes nothing
python -m src.utils.clean_run --clean    # delete it
```

`clean_run` never touches `data/raw/`, `checkpoints/` or `tfm-library/`, and never
`prior_cache/ood/` even under `--prior-cache` — compute nodes have no outbound internet, so
that cache can only be rebuilt from a login node.

**2. Submit**, and **immediately add a stub entry here** with the date, the config and the job
id — before the run finishes. A job killed at the walltime never comes back to write its own
entry, and the submission details are exactly what is lost.

**3. When it finishes, Andreas downloads the output and uploads it back into a chat session** so
it can be read and debugged. What to collect:

```bash
# on the login node — small enough to pull down whole
tar czf run_<date>.tar.gz \
    $VSC_DATA/CreditICL/output/logs/ \
    $VSC_DATA/CreditICL/output/manifests/
```

That is the whole diagnostic surface: `logs/*.log` (what happened), `manifests/*__progress.csv`
(the real-data curve), `manifests/*__telemetry.csv` (GPU and gradients). `output/results/` lives
on project storage and is only worth pulling when the numbers themselves are needed.

**4. Fill in the entry.** Results, bugs, interpretation. Add a row to `AGENTS_MEMORY.md`, and a
**Dead ends** entry there for anything that cost more than a couple of minutes.

---

## What gets logged, and why each thing is there

A cluster run cannot be watched. By the time anything is known, the job is over and only what
it wrote down survives — so the rule is **log generously**. Four files per run:

### `output/logs/<run>_<timestamp>.log` — the narrative

| line | what it answers |
|---|---|
| `environment:` | the run's identity — task, seed, rank, world size |
| `hardware:` | the **machine** — GPU model and capability, torch/CUDA/numpy versions, SLURM job and **array task id**. Two runs on different GPUs are not comparable, and this is the only record of which |
| `grid levers:` | exactly which arm of the sweep this is |
| `credit_fraction IN USE:` | printed separately, and loudly when overridden — the single most important number in the experiment |
| `prior source:` | `GENERATE` or `POOL`. A silent switch here would change what the run measures |
| `budget:` | steps × datasets/step, resolved, so a truncated run is obvious |
| `step N/M [%] loss=… lr=… steps/s … eta … finishes ~<clock time>` | progress, with a wall-clock finish estimate |
| `PROJECTED OVERRUN` | fires while there is still time to react, rather than the job being found dead at the 72 h wall |
| `hw step N …` | GPU utilisation, peak memory, CPU % — see below |
| `grads step N …` | gradient-to-weight ratio per architecture block |
| `OOM in micro-batch` | a skipped micro-batch; >10 % raises |
| `finished:` + `telemetry:` | the closing summary, including a **starvation warning** |

### `output/manifests/<run>__progress.csv` — does it work yet?

One row per `progress.every_datasets` synthetic datasets, scoring the **live model on real
data** mid-training. This turns a run from one end-of-run number into a curve, which is what
answers *"does the credit prior help early, late, or only at convergence?"* Columns cover the
real credit datasets **and** the out-of-domain suites, so a prior that helps credit by
destroying generality is visible immediately rather than at the end.

### `output/manifests/<run>__telemetry.csv` — was the machine busy, and is the model learning?

| group | columns | the question it settles |
|---|---|---|
| GPU | `gpu*_utilization_gpu`, `utilization_memory`, `memory_used/total`, `temperature`, `power_draw`, `clocks_sm` | **Was the GPU actually working?** A compute-bound run and a data-starved run look identical from outside — same wall-clock, same loss curve — and the fix is opposite (more GPUs vs more `num_workers`). Guessing costs another run |
| torch memory | `mem_allocated_gb`, `mem_reserved_gb`, and the `max_` high-water marks | what to size a batch against. A large reserved-vs-allocated gap means fragmentation, which is how an OOM appears at step 40,000 after 39,999 identical steps |
| host | `cpu_percent`, `ram_used_gb`, `ram_percent` | the prior is generated on the CPU; pinned CPU with an idle GPU is a `num_workers` problem |
| throughput | `steps_per_s`, `datasets_per_s`, `elapsed_s` | comparable across runs and machines |
| gradients | `grad_{col,row,icl,head,other}`, `grad_global` | is every stack receiving signal? |
| weights | `weight_{...}` and **`gw_ratio_{...}`** | the ratio is the interpretable one — a gradient of 0.01 is tiny against weights of 0.1 and enormous against 1e-5. A block whose ratio sits orders of magnitude below the rest is effectively frozen, and the loss curve will never say so |

Cadences are `logging.log_hardware_every` and `logging.log_grad_every`. They sit under
`logging:` and not `train:` on purpose: **switching them off must not change a result.**
Rank 0 samples only — eight ranks on one node would write eight identical rows.

Everything here is best-effort: no `nvidia-smi`, no CUDA, no `psutil`, or an unwritable
directory all degrade to a missing column. *A diagnostic that can kill a three-day run is worse
than no diagnostic.*

### `output/logs/<run>_<timestamp>.metrics.jsonl` — the machine-readable curve

One JSON object per `log_every` steps. This is what gets plotted.

---

## Entry template

Copy this. Every heading earns its place; `n/a` is a fine answer, a missing heading is not.

```markdown
## DD-MM-YYYY — <config> — <one-line outcome>

**Submitted** DD-MM-YYYY HH:MM | **job** <id> (array <a>-<b>) | **cluster** wICE `<partition>`
**Commit** <sha> | **tfm-library pin** <sha> | **GPUs** <n> x <model> | **walltime** <req> / <used>

### Configuration
- **Config file** `config/ExpN_TRACK.yaml`, arm <index> of <total>
- **The levers** — copy the `grid levers:` line from the log, verbatim
- **Budget** <steps> steps x <batch> = <datasets> datasets
- **Init** scratch | full from `<checkpoint>`
- **Anything edited by hand since the last run** — the thing that always turns out to matter

### Results
| metric | dev | holdout | control arm | delta |
|---|---|---|---|---|
| | | | | |

- **Progress curve** still climbing at the end? plateaued? where?
- **Out-of-domain** did generality hold? (a credit gain paid for with OOD loss is not a result)
- **Throughput** <steps/s>, mean GPU utilisation <%>, peak memory <GB>
- **Files** `output/logs/<...>`, `output/manifests/<...>`

### Bugs and anomalies
Anything wrong with the run, however small — a NaN, a skipped micro-batch, a metric that
cannot be right. **A number that looks too good goes here, not in Results.**

### Interpretation
What we think happened and why. Explicitly separate:
- what the numbers **show**
- what we **think** explains it
- what would **test** that explanation

### Next
The one thing to do differently.
```

---

## Runs

## 29-08-2026 — Exp1 full sweep, 150 arms — **142/150 trained clean; only the two known bugs cost the other 8**

**Submitted** 25-08-2026 22:38 (LGD), 26-08-2026 04:04 (PD), resubmitted through 29-08 | **jobs** 11529826 + 61791522 (LGD), 11529827 (PD) | **cluster** mindwell `gpu_b200` (arms 0–39) + wICE `gpu_a100` (arms 40–74)
**Commit** da78f34 (early arms, dirty) → Vasicek fix pulled mid-run | **tfm-library pin** 21d555a6a24e | **GPUs** 1 × B200 or 1 × A100 per arm | **walltime** 12 h / 5.3–11.8 h used

### Configuration
- **Config file** `config/Exp1_{LGD,PD}.yaml`, all 75 arms per track (full prior grid × 3 seeds)
- **The levers** — LGD: `atom_prob {0.6, 0.8} × mode {mechanism, quantile} × target_scaling {none, standard} × credit_fraction {0, 0.1, 0.2, 0.3}`; PD: `category_frequency {balanced, power_law} × mode {mechanism, quantile} × signal_strength {0.6, 1.0} × credit_fraction {0, 0.1, 0.2, 0.3}` (the `credit_fraction: 0` control collapses to one prior × 3 seeds)
- **Budget** 12,500 steps × 64 = 800,000 datasets per arm
- **Init** scratch, Muon — upstream stage 1
- **Edited by hand mid-run** the Vasicek single-class guard (pulled ~28-08); the wICE resubmit fix landed *after* this download

### Results
| metric | LGD | PD |
|---|---|---|
| arms completed | 73 / 75 | 69 / 75 |
| final train loss (range) | 0.051 – 0.070 | 0.136 – 0.186 |
| wall-time / arm | 5.3 h (B200) – 11.8 h (A100) | 5.4 – 5.7 h |

- **Prior ranking** — n/a yet. This is the TRAINING phase; *which prior wins* is Phase 2 (`benchmark.slurm`), not run.
- **Progress curve** converged and flat by the end on every completed arm; no divergence.
- **Throughput** ~0.63 steps/s (B200), ~0.29 (A100); mean GPU util ~88–90 %, compute-bound (fwd_bwd ≈ 97 %).
- **Cost** 980 GPU-hours across 142 completed arms.
- **Files** `output/<arm>/{summary,config.resolved}.json`, `output/{logs,manifests}/`.

### Bugs and anomalies
- **6 PD arms (a1–a6) crashed** `ValueError: Vasicek mechanism produced a single-class target` — all `mode=mechanism`, all *before* the guard was deployed. No mechanism arm has crashed since; the fix is confirmed live (a17–a74 mechanism arms completed clean).
- **2 LGD wICE arms (a40, a73) stuck at 99.9 %** — drained at steps 12,460 / 12,487, then the self-resubmit was rejected by wICE's 18-core cap (see the 29-08 dead end). Fixed; they resume via `run_experiment`.
- **False alarms, checked and cleared:** a first log scan flagged "299 CUDA errors / 299 NaN". Both were regex artifacts — `Inf` inside `INFO`, `cuDNN` in the startup banner. Strict re-scan: **0** CUDA errors, **0** NaN losses, **0** non-finite gradients, **0** walltime kills.

### Interpretation
- **Show:** all 150 priors train stably to a tight, converged loss band; the sweep machinery (prior mixing, two-cluster split, drain→resume) works. The only failures are two mechanical bugs, both now fixed.
- **Think:** the tight LGD band (0.051–0.070) across very different priors means the per-arm budget is saturated — differences between priors will surface in Phase-2 *real-data* scores, not synthetic training loss.
- **Test:** Phase 2 — score all 150 checkpoints on the real credit datasets against the reference column (released TabICLv2, TabPFN-v3, CatBoost, logistic/linear). That is where the Exp1 question is actually answered.

### Next
Finish the 8 remaining arms (`run_experiment 1 --submit`), then launch Phase 2 (`benchmark.slurm`, `EXP=1`) for the prior ranking.

---

## 24-08-2026 (evening) — **the resilience chain worked, and arm 0 of Exp1 is finished**

**Jobs** 11524582 (killed at 20 min) -> 11524593 (resumed, completed) | `gpu_b200`, node
`r11g17` | **arm** 0 of 75, LGD, `credit_fraction 0.0` — the control

A deliberate kill test: one arm, `--time=00:20:00`, so `--signal=B:USR1@600` fires ten minutes
in. **Every link fired, in order**, and the arm then finished on its own.

    Dataloader workers: 23 (from SLURM_CPUS_PER_TASK=24)     <- allocation, not config
    SIGUSR1 -> forwarding to trainer 165362                  <- the trap
    SIGUSR1 received at step 446 - finishing this step       <- the handler
    checkpoint saved at step 447
    STOPPING EARLY on SIGUSR1 at step 447/12500
    INCOMPLETE: stopped at step 447 by SIGUSR1. Exit 64      <- the exit code
    Arm 0 incomplete - resubmitting to resume.
    Submitted batch job 11524593
    END status=INCOMPLETE-RESUBMITTED

Job 11524593 picked up at step 447 and ran to **step 12,500 / 800,000 datasets** in 4.02 h.
**Arm 0 of Exp1 LGD is done**, across two jobs, through a mid-run kill, with no human action.

### Health of the run itself
- **0.77-0.85 steps/s, mean 0.82** — against the benchmark's 0.791. The projection held.
- **GPU utilisation 86-92 %, mean 88.75 %.** Peak memory 12.3 GB.
- `phases` is `fwd_bwd=97.2%` once the workers warm up: compute-bound, not starved.
- **Loss 0.343 -> 0.0587** over 12,500 steps, smooth, no spikes. Gradients live in all three
  stacks at step 250 (`col=8.05e-03 icl=7.73e-04 row=1.58e-02`).
- Prior composition exactly as configured: `sources={'base': 16, 'credit': 0}` at
  `credit_fraction 0.0`; 23.8 % of candidates rejected as unpredictable.

### Three defects it exposed
1. **`PROJECTED OVERRUN` still told the reader to use `--resume auto`** — the flag that has
   never existed and that killed eight jobs with argparse exit 2 on 14-08-2026. It had survived
   ten days inside a WARNING string that fires every hundred steps. The advice was also simply
   wrong now: the run resumes itself. Rewritten to say so.
2. **The preflight smoke test writes its manifests into the REAL arm's results directory.** It
   runs with a temp `out_dir`, but `manifest_dir` resolves to the shared
   `$VSC_DATA/.../manifests/` on the cluster, under the real run name — a two-step toy run
   filing telemetry under a 12,500-step arm. It never actually wrote a row (4 datasets never
   reach `every_datasets`), but nothing stopped it. `Trainer` now takes `manifest_dir` and the
   smoke test pins its own.
3. **The job banner printed the wrong walltime.** `SBATCH_TIMELIMIT` is not set inside a job,
   so it fell back to the `#SBATCH` default and said `12:00:00` while the real limit was
   20 minutes — the one number a reader checks the ETA against. Now computed from
   `SLURM_JOB_END_TIME - SLURM_JOB_START_TIME`.

### One thing to fix before the sweep, outside the code
The MACHINE block reports `commit 1797f48  *** UNCOMMITTED CHANGES ***`. The cluster is running
working-tree edits on top of a commit from days ago, so **this arm is not reproducible from
any commit**. Fine for a kill test; not acceptable for 75 arms whose numbers go in a paper.

---

## 25-08-2026 — A100 vs B200: the A100 is 29 % cheaper per arm

**Job** 61776784, wICE `gpu_a100`, A100-SXM4-80GB, node `k28g32`, 18 cores.

| | B200 (11523286) | A100 (61776784) | ratio |
|---|---|---|---|
| matmul bf16 | 1,363 | 258 TFLOP/s | 5.28x |
| one micro-pass, fwd+bwd | 128.75 | 258.92 ms | 2.01x |
| AMP micro-pass | 62.19 | 132.37 ms | 2.13x |
| prior, one worker | 11.41 | 4.02 datasets/s | 2.84x |
| **end to end** | **0.791** | **0.409 steps/s** | **1.93x** |
| peak GPU memory | 13.08 | 11.72 GB | |
| % of its own ceiling | 80 % | **89 %** | |
| **hours per arm** | **4.39** | **8.49** | **1.93x** |

**The A100 is 1.93x slower and 2.72x cheaper per hour, so it is 29 % cheaper per arm.**

| | credits/arm | 75 arms | wall-clock at %8 |
|---|---|---|---|
| B200 | 134,443 | **10.08 M** | 1.8 days |
| A100 | 95,510 | **7.16 M** | 3.5 days |

Note the raw-FLOP ratio is 5.28x but the end-to-end ratio is only 1.93x — this workload is
latency-bound, not throughput-bound, exactly as the idle memory suggested. Peak FLOPs do not
predict it.

The A100's **prior generator is 2.84x slower per worker** (Icelake vs AMD Turin). With 17
workers it still supplies 1.44 steps/s against the 0.46 needed — 3.1x headroom, so the smaller
core count is not a problem.

**cuDNN works on both cards** at [4, 1024, 50], and flash is fastest on both. The exclusion
stands on the "buys nothing" reason.

### Recommendation
Split the sweep across both clusters: they have separate queues, so 40 arms on `gpu_b200` and
35 on `gpu_a100` finishes in **~1.8 days for 8.7 M credits**, and leaves neither cluster
monopolised. `sweep_status` reconciles the two halves afterwards.

---

## 24-08-2026 — benchmark, third attempt — **one arm is 4.4 hours, not 20**

**Submitted** 24-08-2026 11:54 | **job** 11523286 | **cluster** mindwell `gpu_b200`, node
`r11g22` | **used** 2 min 50 s | every rung produced a number, no failures

The first benchmark with no known blind spot: config-driven shape, the real micro-batch, AMP
on, the attention kernel pinned, and a rung that times every kernel.

### Results

| rung | value |
|---|---|
| matmul bf16 / fp32 | 1,363.32 / 63.23 TFLOP/s |
| one micro-pass, fwd | 28.55 ms |
| one micro-pass, fwd+bwd | 128.75 ms |
| a whole step (16 passes, fp32, SGD) | 2,060 ms |
| AMP: micro-pass fp32 -> bf16 | 128.89 -> **62.19 ms (2.07x faster)** |
| Muon extra, per optimiser step | **18.83 ms** |
| prior, one worker | 11.41 datasets/s |
| **end to end, 23 workers, AMP** | **0.791 steps/s, 50.62 datasets/s** |
| peak GPU memory | **13.08 GB of 178** |
| **projected hours per 12,500-step arm** | **4.39** |

### Attention backends — all four work, at the shape training uses

| backend | fp32 | bf16 |
|---|---|---|
| flash | 27.75 | **23.01** |
| mem-efficient | 27.90 | 23.23 |
| **cuDNN** | 28.79 | **24.11** |
| math | 49.45 | 46.56 |

**This retracts a claim.** On 20-08 cuDNN raised `mha_graph.execute(...).is_good()` and the
write-up said it "would have hit the 75-arm run". It would not have. That failure was at
[64, 1024, 50] — the whole batch in one pass, which only the broken benchmark ever built.
At [4, 1024, 50], the only shape training runs, cuDNN works fine. The exclusion stays, but on
the honest reason: flash is 4.8 % faster and cuDNN has a demonstrated failure at a nearby
shape, so dropping it costs nothing.

Also worth keeping: **math is 2x slower than flash.** The fallback is real but not free.

### The bug this run exposed: a micro-pass is not a step

The printed verdict said **"reaching 6 % of the ceiling — STARVED BY THE PRIOR: more cores"**.
That is the opposite of the truth.

`amp_bf16_step_ms` held the time of ONE forward/backward pass (62.19 ms), and the verdict
divided into it as though it were a whole step. A step is **16 passes plus one optimiser
step**:

    16 x 62.19 ms  +  18.83 ms Muon  =  1,013.9 ms  ->  0.986 steps/s

Measured 0.791 against 0.986 is **80 % of the ceiling: compute-bound**. And the prior supplies
11.41 x 23 = 262 datasets/s = 4.10 steps/s against the 0.99 the GPU can take — **4.2x
headroom**. Every "STARVED, more cores" line this tooling has printed has now been wrong.

Fixed: per-pass numbers are named `_micro_ms`, whole-step numbers `_step_ms`, the optimiser
rung reports `optimizer_overhead_ms_per_step` separately, and two tests forbid the verdict
from dividing into a per-pass number. A percentage above 105 % now prints "THE CEILING IS
WRONG" rather than a confident verdict.

### Muon, for the third and final time
Measured 1.15x in this rung, which calls `opt.step()` every pass. Training calls it once per
16 passes, so in training Muon is **1.019x** — 18.83 ms on a 1,014 ms step. The benchmark now
reports `muon_slowdown_in_training` rather than the raw rung ratio.

### What it means for Exp1

**~4.4 h per arm, not the 16-22 h bracket.** AMP is the difference: it was never enabled in an
end-to-end measurement before.

| | before this run | measured |
|---|---|---|
| hours per arm | 16-22 | **4.39** |
| 75 arms, credits | 33-46 M | **~10 M** |
| wall-clock at throttle %8 | ~8 days | **1.8 days** |

The seed-staging plan is withdrawn: at 10 M credits and under two days there is no reason to
run 25 arms first and add seeds later. **Run all 75.**

Two caveats on 4.39 h: the rung measures 20 training steps and excludes the final evaluation
and the mid-run progress scoring. `progress.every_datasets` was still 5,000, which was "10
points" when an arm was 50,000 datasets and had silently become **160 measurements** at
800,000. Raised to 40,000 -> 20 points. Walltime set to 12 h, ~2.7x the measured training
time, which covers both.

### Next
Submit Exp1.

---

## 20-08-2026 (15:34) — benchmark at the REAL batch — **it found a bug that would have killed Exp1**

**Submitted** 20-08-2026 15:33 | **job** 11521108 | **cluster** mindwell `gpu_b200`, node
`r11g17`, NVIDIA B200 | **used** 3 min 6 s | **ceiling quoted at submit** 30,600 credits

The rerun of the fixed benchmark. It now reads the config — `batch 64, rows 1024, features 50`,
printed at the bottom — and the first honest look at the real batch size broke in three places.

### Results

| rung | value |
|---|---|
| matmul bf16 / fp32 | 1,358.88 / 63.3 TFLOP/s (unchanged) |
| attention @1,024 | 0.026 ms |
| model fwd / fwd+bwd @batch 64 | 290.44 / **1,617.47** ms |
| AMP @batch 64 | fp32 1,617.24 ms, bf16 **RuntimeError inside cuDNN** |
| AdamW / Muon @batch 64 | 1,618.02 / 1,636.16 ms -> **1.01x** |
| prior, 1 worker | **11.78** datasets/s (640-dataset sample) |
| end to end | **CUDA OOM** — 176.30 GiB allocated of 178.34 |

### The three findings

**1. AMP crashes on this card, and training has `amp: true`.**

    RuntimeError: Expected mha_graph.execute(handle, variant_pack,
                  workspace_ptr.get()).is_good() to be true

That is `torch/csrc/cudnn/MHA.cpp` — the **fused cuDNN multi-head-attention graph**, one of the
four backends `scaled_dot_product_attention` picks between. PyTorch chooses silently, per call,
from the shapes and dtype, so the same model takes a different kernel at a different batch size
and only then fails. **Nothing in this project had ever pinned the choice.** A 75-arm run would
have discovered this one arm at a time. Upstream treats the choice as load-bearing too:
`--use_flash_attn3 False` in stage 1, `True` in stages 2-3.

Fixed: `src/models/backends.py` excludes cuDNN, keeps flash / mem-efficient / math, wraps every
forward pass in training, and logs which kernels are in use in the run card.

**2. The OOM is the BENCHMARK's bug, not training's.** It pushed all 64 datasets through one
forward pass. Training runs `ceil(64/4) = 16` passes of 4 and accumulates — `micro_plan` now
models that, and the end-to-end rung micro-batches exactly as `Trainer.train_step` does. Same
root cause as yesterday's: the instrument did not model the thing it was measuring.

**3. Muon is FREE at the real batch size — 1.01x, not 1.40x.** This corrects a number recorded
three times. Newton-Schulz runs once per weight matrix per step, so its ~18 ms is FIXED while
the forward/backward grows with the batch: 40 % of a batch-1 step, 1 % of a batch-64 step. Both
measurements were right; only one was at a batch size we train at.

### What is solid
**The prior is not the bottleneck and never was.** 11.78 datasets/s x 23 workers = **271
datasets/s supplied** against the **25.6 datasets/s** training consumed on 17-08 at 88.7 % GPU
utilisation — **10x headroom**. Every "STARVED, more cores" line the tooling has printed since
14-08 was an artefact of benchmarking at batch 4.

### Bugs fixed
1. Attention backend never pinned -> `src/models/backends.py`, wired into training and all
   five GPU rungs, with an `attention_backends` rung that times every kernel in both
   precisions and reports the failures as data.
2. GPU rungs ran one pass of `batch_size` -> `micro_plan(task)`, and a test forbidding
   `torch.randn(batch_size, ...)` anywhere in a timing rung.
3. Failed rungs were silently omitted from the summary, so a report with an OOM and a cuDNN
   crash in it ended on a tidy line and looked clean -> failures now print FIRST, before any
   verdict.
4. `bench_optimizers` still claimed Muon "explains the B200". Corrected in place.

### Interpretation
Still no per-arm cost — the run that was supposed to produce it crashed twice. But it bought
something worth more: **the cuDNN failure would have hit the 75-arm run**, and at one arm per
crash that is the kind of thing that costs a week rather than three minutes.

### Next
Rerun. `projected_hours_12500_steps` is now computed and printed by the end-to-end rung, so the
next run states the arm cost directly instead of leaving it to be inferred.

---

## 20-08-2026 — GPU benchmark after the prior-shape fix — **the instrument could not see the change**

**Submitted** 20-08-2026 14:45 | **job** 11520989 | **cluster** mindwell `gpu_b200`, node
`r11g12`, NVIDIA B200 178.3 GB, 148 SMs, driver 595.71.05 | **used** 55 s
**torch** 2.11.0+cu128, CUDA 12.8, `sm_100` shipped, `exact_arch_shipped: true`

Submitted to answer one question: what does matching upstream's stage-1 prior shape cost
(rows [512, 1024] -> exactly 1,024, features [3, 50] -> [1, 100])? **It answered a different
question, because only 2 of its 6 rungs read the config.**

### What it measured

| rung | value | read the new config? |
|---|---|---|
| matmul 4096 bf16 | **1,366.72** TFLOP/s | n/a |
| matmul 4096 fp32 | 63.18 TFLOP/s | n/a |
| attention @1,024 | 0.026 ms | n/a |
| model fwd | 16.14 ms | **NO** — hardcoded 1024x40, batch 1 |
| model fwd+bwd | 44.03 ms -> 22.71 steps/s | **NO** — same |
| AMP @batch 4 | fp32 119.32 -> bf16 **58.21** ms | **NO** — same |
| Muon vs AdamW | 63.25 vs 45.06 ms, **1.40x** | **NO** — same |
| prior, 1 worker | **10.88** datasets/s (0.0919 s each) | yes |
| end to end, 23 workers | **4.589** steps/s, 18.35 datasets/s | yes |

### The one real comparison, and the one real conclusion

`end_to_end` is the only rung that saw the new shape *and* has a matching earlier measurement:
17-08 on the same card at the same batch gave **8.26 steps/s**, today **4.589** — the prior-shape
fix costs **1.80x** here. But that number is not the training cost, for two reasons:

1. **It is measured at batch 4, and training runs at batch 64.** Batch 4 is the exact regime the
   17-08 investigation showed puts a B200 at 3.5 % utilisation. `--batch-size` defaulted to 4
   from before `train.batch_size` became 64.
2. **The GPU rungs were blind to the change**, so 1.80x is mostly the CPU prior generator
   getting slower on bigger datasets — not the GPU getting slower on wider tensors.

What IS solid: **at batch 64 the prior has ~10x headroom and is not the bottleneck.** 10.88
datasets/s x 23 workers = 250 datasets/s supplied, against the 25.6 datasets/s that training
consumed on 17-08 at 88.7 % GPU utilisation. AMP is a genuine **2.05x speed-up** (58.21 vs
119.32 ms), and Muon's 1.40x is unchanged and confirmed a third time.

### Bugs
1. **`rows, feats, train_size = 1024, 40, 768` hardcoded in four places.** Only `bench_prior`
   and `bench_end_to_end` called `_prior_cfg`. Fixed: one `bench_shape(task)` reads
   `n_rows_range`, `n_features_range` and `train_frac_range` from the config, and a test asserts
   the assignment is gone.
2. **`--batch-size` default 4**, stale since the config moved to 64. Now defaults to
   `train.batch_size`.
3. **The verdict divided a batch-4 rate by a batch-1 ceiling** — `bench_model` ran at batch 1 —
   and reported "20 % of the GPU, STARVED, more cores". It also contradicted itself: the prior
   extrapolation printed two lines above says 23 workers can supply 62.56 steps/s, 13x what was
   measured. Fixed: the ceiling is measured at the real batch size, includes AMP and Muon, and
   when neither ceiling explains the gap the verdict now says so instead of guessing "more
   cores".

### Interpretation
The arm cost is still not pinned. The FLOP model says the new shape costs ~2.5x per step; the
one measurement available says 1.80x in a regime we do not train in. So one Exp1 arm is
**somewhere between 15.6 h and 21.7 h**, and 75 arms between **33 M and 46 M credits**. A rerun
of the now-fixed benchmark settles it in under a minute of GPU time.

**The lesson is the same one as the NaN.** Six wrong theories then, because the instrument was
built after the guessing; today a benchmark submitted specifically to measure a config change
could not read the config. Check what the instrument reads BEFORE trusting what it says.

### Next
Rerun the benchmark, then decide `max_steps` against the real per-arm cost.

---

## 19-08-2026 — **THE PIPELINE IS FAIR. Every arm positive, gap is now budget alone**

**Job** 11520343 (Exp1 LGD debug, `gpu_b200`) — first run with `crediticl` scored through
upstream's wrapper and a shared context cap.

### The result
Same seven LGD datasets, same run, both models through `TabICLRegressor`, both capped at 1,024
context rows:

| dataset | `crediticl` (600 steps) | `tabiclv2` (released) | ratio |
|---|---|---|---|
| 1 | **0.343** | 0.743 | 2.2x |
| 2 | **0.222** | 0.677 | 3.0x |
| 3 | **0.182** | 0.662 | 3.6x |
| 4 | **0.174** | 0.402 | 2.3x |
| 5 | **0.163** | 0.446 | 2.7x |
| 6 | **0.142** | 0.343 | 2.4x |
| 7 | **0.066** | 0.225 | 3.4x |

**Every arm is positive.** The previous run, with a hand-rolled inference path and no context
cap, was **-1.437 to -0.246 — worse than predicting the mean on all seven.** Nothing about the
weights changed between the two runs; only the pipeline did.

### What that means
The remaining gap is a **uniform 2-4x ratio with no sign flips and no outliers** — the shape of
a model that is simply undertrained, not one that is broken. Our budget is 38,400 datasets
against upstream's 35,200,000: **0.11 %**. A model at 0.11 % of the data reaching 25-45 % of the
reference R² is a coherent, reportable position.

Two fixes produced it, both of them removals of OUR differences rather than changes to the model:

1. **`crediticl` subclasses `TabICLBaseline`** — scored through `TabICLRegressor`, so upstream's
   preprocessing, 8-member ensemble, context construction and decoding are inherited and cannot
   drift. Only `name`, `_wrapper_kwargs` and `_fit` (metadata) are overridden; `_predict` is
   inherited.
2. **A shared context cap at 1,024**, matched to `n_rows_range`. The wrapper does not cap
   context, so it had been feeding 47,089 rows on `heloc` to a model trained on <=1,024 — 46x.
   Applied to both columns from one setting, so it is part of the measurement.

### Bugs and anomalies
- None. Training completed, evaluation completed, exit 0, no NaN, no failures.
- The results CSV lives on `/lustre1` staging and was not in the upload, so the per-row
  `context_cap` / `n_context_full` provenance columns are unverified from this log. Worth
  confirming once, since they are what makes a capped number legible later.

### Interpretation
- **What the numbers show:** the measurement chain is sound and the comparison is now
  weights-only. They do NOT yet show anything about the prior — this is a 600-step control arm.
- **What is left before Exp1:** settle batch size on the control arm (3 runs), then run the 75
  arms. Nothing else is blocking.

### Next
Batch pilot, then Exp1.

---

## 18-08-2026 — **THE NaN IS FIXED. First clean run, and the first honest comparison**

**Jobs** 11519444/45 (GPU NaN diagnosis, both cards), 11519507–13 (Exp1 LGD debug, `gpu_b200`)

### The NaN: solved, and it was missing feature standardisation
The GPU walk named module **#0** — `col_embedder.in_linear`, the first layer — identically on
both cards and under all four attention backends. Its output `absmax` was **6.550e+04**, which
is `float16`'s largest finite value.

| dataset | input absmax | before | after |
|---|---|---|---|
| `axa` | 1.98 | finite | finite |
| `heloc` | 820 | finite | finite |
| `loss2` | **2.4e6** | NaN | **finite** |
| `base_modelisation` | **8.1e7** | NaN | **finite** |
| `base_model` | **9.6e8** | NaN | **finite** |

We fed raw currency amounts into the network. Upstream's `PreprocessingPipeline.fit` begins
with `CustomStandardScaler().fit_transform(X)`; we had median imputation and no scaling.
`standardise_from_context()` — context-only, so it cannot leak — now runs in all three
inference paths. **All 7 LGD datasets score and there is not one NaN in the log.**

Six explanations preceded this one and all six were wrong: the data, the architecture,
divergence, feature width, CPU-vs-CUDA, the attention kernel. The answer came from the first
tool that measured rather than theorised.

### The first honest comparison, and it is sobering
Same seven datasets, same evaluation run:

| | R² |
|---|---|
| **Released TabICLv2** | **+0.224 … +0.770** |
| Ours, 600 steps | −1.437 … −0.246 |

Ours is worse than predicting the mean, everywhere. Two causes, and they must not be conflated:

1. **Budget.** 600 steps x 64 = 38,400 datasets against upstream's 35.2 million — 0.11 %.
2. **An unfair inference pipeline, which is OUR bug.** `tabiclv2` is scored through
   `TabICLRegressor`, the official wrapper: upstream preprocessing *and* an 8-member
   feature-shuffled ensemble. Ours went through a hand-rolled single pass with no ensemble.
   **That is not a weights-only comparison**, so the gap above overstates the deficit by an
   unknown amount.

**Fixed:** checkpoints now carry `curr_step` and `model_config` alongside `state_dict`, which is
the schema `TabICLClassifier(model_path=<path>)` reads — so our weights can be scored through
upstream's own wrapper and the comparison becomes weights-only.

### Bugs and anomalies
- Nothing non-finite anywhere. Run completed, evaluation completed, exit 0.
- **Still open:** our checkpoints are now upstream-loadable but `CreditICLBaseline` does not yet
  USE the wrapper — that is the next change, and it is what makes the headline table fair.

### Next
Score through upstream's wrapper, then the batch pilot, then Exp1.

---

## 17-08-2026 — **THE B200 MYSTERY IS SOLVED: the batch was too small for the card**

**Jobs** 11518234/35 (benchmark, both cards), 11518236–40 (Exp1 LGD debug, `gpu_b200`, the
first run at upstream's `batch_size: 64`)

### The answer

| | GPU utilisation | datasets/s |
|---|---|---|
| B200, batch 4 | **3.5 %** | 2.2 |
| B200, batch 64 | **88.7 %** | **25.6** |
| RTX 5000 Ada (free), batch 4 | 69 % | 26.5 |

**Batch 4 was too little work to fill a B200.** One short forward pass, then a synchronising
optimiser step, so the card idled between steps — 3.5 % utilisation. The RTX 5000 Ada only ever
looked faster because it is small enough to be near-saturated at batch 4. Raising the batch to
upstream's 64 lifted the same card to 89 % and **12x the throughput per dataset**, and it was
changed for an unrelated reason: matching upstream.

**Five explanations were proposed before this one and every one was wrong** — the card, the
prior generator, Muon, the wheel's kernels, AMP. Each was inferred from a difference someone
noticed. The per-phase timing (`fwd_bwd=98.7 %`, `data=0.0 %`) and the benchmark's optimiser
and AMP rows are what finally cornered it:

| ruled out by measurement | |
|---|---|
| the card | B200 ahead on every benchmark rung |
| starvation | `data = 0.0 %` of step time |
| Muon | 1.42x on **both** cards |
| missing kernels | `sm_100` shipped; the false alarm was on the healthy card |
| AMP | **0.49x — it makes the B200 *faster*** |

### Also
- **The job was killed at the walltime**, step 1,422 of 1,500, after 62 minutes — losing the
  evaluation and the final checkpoint. The debug budget is `steps x batch_size`, and batch_size
  went 4 -> 64, so 1,500 steps quietly became 96,000 datasets. `DEBUG_STEPS` is now 600.
- **The micro-batch override never reached the job.** Two heredoc edits silently failed —
  one whose continuations did not match, one that wrote a literal `
` — and `bash -n` accepted
  both. The run used micro_batch_size 4 on a 183 GB card. Now set per partition, with a test.
- **Peak memory was 6.6 GB of 183 GB**, so there is a great deal left: micro 32 on `gpu_b200`.
- **The NaN survives `max_features: 100`.** `base_modelisation` (256 features) still returns
  all-NaN, so the training-range hypothesis is not sufficient on its own. Still needs a
  checkpoint.
- Learning is healthy at batch 64: loss 0.331 -> 0.122 by step 1,400, gradients live in all
  three stacks.

### Next
600-step debug to confirm it finishes end to end, then the batch pilot, then Exp1.

---

## 17-08-2026 (earlier) — the optimiser row — **Muon is not the cause either**

**Jobs** 11517858 (`interactive`), 11517859 (`gpu_b200`) — the benchmark rerun with
`bench_optimizers`, which times the optimiser the configs actually use.

| | RTX 5000 Ada | B200 |
|---|---|---|
| AdamW step | 73.5 ms | **44.5 ms** |
| **Muon step** | 98.7 ms | **62.2 ms** |
| **Muon / AdamW** | **1.34×** | **1.40×** |
| end to end (SGD) | 4.50 steps/s | **8.26** |

**Muon costs ~35–40 %, on both cards equally, and the B200 is faster at it in absolute terms.**
It cannot explain a 12× gap. The 16-08 hypothesis is dead.

**What is left, and it is the real question.** On the B200, the benchmark's end-to-end loop —
real prior, real DataLoader, 23 workers — runs at **8.26 steps/s**. Real training on the same
partition runs at **0.53**. A 16× gap, with the optimiser now accounting for at most 1.4× of
it. Everything the benchmark does *not* do is now suspect:

- **AMP** (`amp=True` in training; the benchmark runs plain fp32)
- **micro-batching** inside `train_step`
- **telemetry** — `nvidia-smi` on a node with 24 GPUs may be far slower than on one with 2
- **node contention** — the two ran hours apart, and CPU is shared

Note the free card shows the opposite sign: real training there (6.6 steps/s) is *faster* than
its own benchmark (4.50), so none of that machinery is inherently slow. Whatever it is, it is
specific to the B200 node.

**Next:** a per-phase timing breakdown inside the training loop — data wait, forward, backward,
optimiser, telemetry — logged every N steps. One run then answers it instead of another
hypothesis. Do not guess a third time.

Also confirmed: `has_kernels_for_this_card` is now clean for both cards, and `sm_100` is
shipped, so the B200 was never JIT-falling-back.

---

## 16-08-2026 — GPU benchmark + LGD debug — the evaluation chain works end to end

> **PARTLY SUPERSEDED 17-08-2026.** The heading claimed "it is MUON". Measured, Muon costs
> 1.34–1.40× on both cards — not the 15× blamed here. See the entry above. The benchmark
> results and the NaN finding in this entry stand.

**Jobs** 11517081 (benchmark, `interactive`), 11517082 (benchmark, `gpu_b200`),
11517370/71/72 + 11517008 (Exp1 LGD debug, 4 arms, `gpu_b200`)

### The benchmark overturns the previous conclusion

| measurement | RTX 5000 Ada (free) | B200 | |
|---|---|---|---|
| matmul fp32 | 36.1 TFLOP/s | **63.2** | B200 1.7× |
| matmul bf16 | 181 TFLOP/s | **1,392** | B200 **7.7×** |
| attention 1024 | 0.051 ms | **0.026** | B200 2.0× |
| model fwd+bwd | 69.3 ms | **44.1** | B200 1.6× |
| prior, 1 worker | 10.5 ds/s | **15.5** | B200 1.5× |
| **end to end (SGD)** | 4.50 steps/s | **8.14** | **B200 1.8×** |

**The B200 is faster on every single rung.** The 14-08 conclusion — "the B200 is the
bottleneck, use the free GPU" — was **wrong**, and wrong because the benchmark used plain SGD
while `config/Exp1_*.yaml` sets `optimizer: muon`.

| | optimiser | steps/s |
|---|---|---|
| B200, real training (11517370) | **muon** | 0.53 |
| B200, benchmark, same model/prior/loader | SGD | 8.14 |

**Muon costs a factor of ~15 on the B200 and nothing on the RTX 5000 Ada** (which trains at
6.6 steps/s *with* Muon, above its own 4.5 SGD benchmark). Muon orthogonalises each weight
matrix with a Newton-Schulz iteration — a chain of small matmuls per matrix per step, which is
latency-bound rather than throughput-bound, so peak FLOPs do not help.

Also corrected: the benchmark's `has_kernels_for_this_card` flag fired for the RTX 5000 Ada
(`sm_89` absent from a list containing `sm_86`). Minor versions are binary-compatible within a
major architecture, so that was a **false alarm on the healthy card** — the test is now on the
major version. `sm_100` *is* shipped, so the B200 was never JIT-falling-back.

### The evaluation chain now works end to end
`crediticl` is scored for the first time: `crediticl OK r2=-0.1144 rmse=0.2730
bnd_err=0.0204` on real LGD data. Registration, checkpoint resolution, architecture
reconstruction and the row cap all hold.

### Bugs and anomalies
1. **The LGD model emits 100 % NaN predictions on 3 of 4 real datasets** (`base_modelisation`,
   `base_model`, `loss2`) and on 5 out-of-domain regression datasets. `axa` — the only one
   with **zero missing values** and only 2 features — works, and gives sensible numbers
   (r2 −0.004, coverage 0.71/0.89/0.93, spearman 0.47).
   **Not the data:** the imputed input contains no non-finite value. **Not the architecture:**
   an untrained model on the same four datasets returns 0 % NaN. **Not divergence:** weight
   norms are flat across training (col 61.4→61.7, row 41.7, icl 174→175) and gradient ratios
   sit at 5e-3–1.5e-2 throughout. **Unexplained — needs the checkpoint to localise.**
   The new `pred_nonfinite_frac` / `nan_predictions` columns are what made it visible at all;
   previously it was reported as `constant_prediction`.
2. The progress curve is still one row per run: `progress.every_datasets` is 5,000 against a
   6,000-dataset debug budget. Expected, and it disappears at the real 50,000.

### Interpretation
- **What the numbers show:** the B200 is the better card on every measured axis, and Muon —
  not the hardware, not the prior — is what made it look 12× slow.
- **What we think explains it:** Newton-Schulz is a long chain of tiny GPU operations, so it
  is dominated by launch latency and per-op overhead, which a bigger card does not reduce.
  **Unverified**: the `bench_optimizers` row added today measures Muon against AdamW directly
  and will confirm or kill it on the next run.
- **What would test the NaN:** load the checkpoint and walk the stacks — `col_embedder`,
  `row_interactor`, `icl_predictor` — to find where the first non-finite value appears.

### Next
Rerun the benchmark on both cards with the optimiser row. Get one LGD checkpoint down to a
laptop to localise the NaN. **Exp1 cannot start until the NaN is understood** — it makes 3 of
4 LGD datasets unscoreable, which is most of the experiment.

---

## 14-08-2026 (afternoon) — Exp1 debug, PD on the FREE GPU — **the B200 is the bottleneck**

> **SUPERSEDED 16-08-2026.** The conclusion in this heading is wrong; see the entry above.
> The B200 is faster on every rung of the benchmark. The slowdown was Muon.

> **CORRECTION, added after reading `sacct`.** This entry originally described the run as
> clean. It was not: **all four arms ended `OUT_OF_MEMORY` (exit `0:125`)**, in the final
> evaluation, after training and the out-of-domain scoring had finished. The logs alone did
> not show it — they end mid-evaluation with no traceback, because the OOM killer sends
> SIGKILL and nothing gets to write a message. **Read `sacct` before calling a run clean; a
> log that stops is not a log that finished.** Cause and fix in Bugs 5 below. Everything
> about throughput below is unaffected — training completed on every arm.

**Submitted** 14-08-2026 14:56 | **jobs** 11517006/07/09/10 (PD, 4 arms) | **cluster** mindwell
`interactive`, node `l11i31`, NVIDIA RTX 5000 Ada 32 GiB | **preprocessing run first**

Submitted to split the confound left open by the morning run, where PD-vs-LGD and B200-vs-RTX
changed together. **Same PD config, same prior, different GPU.**

### Results — the confound is resolved

| | GPU | steps/s | GPU util | 1,500 steps |
|---|---|---|---|---|
| PD, this run | RTX 5000 Ada (**free**) | **4.4 – 6.9** | **65 – 75 %** | ~3–6 min |
| PD, this morning | B200 (26,250 credits/h) | 0.48 – 0.56 | 1.7 – 4.1 % | ~44–51 min |

**It is the hardware, not the task.** Identical PD configuration, ~12× faster on a free GPU that
costs nothing, with GPU utilisation an order of magnitude higher. The B200 conclusion from this
morning stands, and the "PD's prior is expensive" hypothesis is dead.

- **Preprocessing worked** — `all datasets ready`, 7/7 LGD and 14/14 PD, cached on `/lustre1`.
- **First real credit number from our own model**: `german` ROC-AUC **0.718**, PR-AUC **0.514**
  at step 1,250 of 1,500 from scratch. Not a result — 1,500 of 12,500 steps — but the
  measurement chain now works end to end.
- **Learning** loss 0.636 → ~0.109, accuracy 0.614 → 0.87.
- **The checkpoint is now passed**: `scoring our checkpoint: …/step-1500.ckpt`, and the eval log
  confirms `[eval] crediticl checkpoint: …`. Both of this morning's evaluation bugs are gone.

### Bugs and anomalies
1. **`crediticl` still scored nothing — third distinct cause.** `RuntimeError: checkpoint
   step-1500.ckpt does not match the architecture in its own config`, with
   `missing=['row_cls_tokens', 'x_embed.weight', …]` and
   `unexpected=['col_embedder.in_linear.weight', …]`. **`load_our_checkpoint` hard-coded
   `NanoTabICLv2`** while training builds upstream `TabICL` — the evaluation path was never
   migrated. All 14 credit datasets and all 25 out-of-domain cells failed.
2. **A NaN target in the second real dataset discarded every measurement after it.** One `try`
   wrapped both loops, so `ValueError: Input contains NaN.` on `0010.thomas` cost the remaining
   two credit datasets *and* all eight out-of-domain suites. The row holds one dataset and an
   error string — which is why this run's progress CSV has **no `ood__` columns at all**, while
   this morning's had eight. It looked like a smaller measurement; it was a masked failure.
3. **`/lustre1/…/checkpoints` still not writable** (`Errno 13`) — unchanged, still needs fixing
   before the 96-arm run.
5. **All four arms were killed by the OOM killer** at the end, on 30 GB. The 14 PD datasets
   total **2.4 GB** as float32 and `0014.algorithmwatch` alone is 158,700 × 2,986 = **1.8 GB**;
   the evaluation loads one, splits it and imputes it, making several more full-size copies.
   It appeared *only now* because it needs the datasets to exist: in the morning run
   preprocessing had not been done, so the evaluation skipped every credit dataset and used
   almost no memory. **A bug fixed upstream can reveal one downstream.**
   Fixed with `EvalConfig.max_rows` — a seeded random subsample taken *before* the split, with
   `row_cap` and `n_rows_full` recorded per row. The debug job sets 20,000; **the real Exp1
   must not**, and a test enforces that no config carries it.
4. **PD is trained with a 2-class head, the released TabICLv2 classifier has 10.** Found while
   writing the round-trip test, and confirmed by parameter count: 27,538,938 against
   27,552,258, a difference of exactly 13,320 — the four head tensors. `Trainer._build_model`
   sets `max_classes` from `prior.n_classes`, overriding `build_model`'s upstream default of 10.
   Exp1 and Exp3 are self-consistent, but **Exp2's PD warm start cannot load the released
   checkpoint.** Left unchanged deliberately: it is a decision about the experiment.

### Interpretation
- **What the numbers show:** the B200 is the wrong machine for this workload, at 3 % utilisation
  and 12× the wall-clock of a free GPU. The free interactive partition is both faster *and*
  free, and its only real cost is that an array runs serially there (`QOSMaxCpuPerUserLimit`).
- **What we think explains the B200:** unresolved. Blackwell is new and `torch 2.11.0+cu128`
  reports `gpu_capability: 10.0`, so a kernel-coverage or JIT-fallback problem is plausible but
  **unverified** — nothing in the log names it, and I am not going to assert it from the timing
  alone.
- **What would test it:** a single matmul-and-attention microbenchmark on both cards, which
  separates "the GPU is slow" from "the data pipeline stalls on that node".

### Next
Rerun on `free` with the architecture fix so `crediticl` finally scores.

**`max_classes` is now resolved, not open:** the head is TabICLv2's own 10, and the loss slices
to the classes present, per upstream's `_compute_batch_loss`. **Every checkpoint before
15-08-2026 is a 2-class model and is not comparable to anything trained after** — they are
different networks. Discard them rather than mixing.

The B200 question has a script now: `scripts/benchmark_gpu.py` via
`scripts/slurm/gpubench.slurm`, run on both cards and diffed with
`python -m src.utils.compare_gpubench`. It reports `torch.cuda.get_arch_list()`, so if the
wheel carries no `sm_100` kernels the answer will be on the first screen.

---

## 14-08-2026 (morning) — Exp1 debug, LGD + PD — pipeline runs end to end; our own model scored nothing

**Submitted** 14-08-2026 10:43 | **jobs** 11516954 (LGD, array 0–3), 11516956 (PD, array 0–3)
**Cluster** mindwell — LGD on `interactive` (RTX 5000 Ada), PD on `gpu_b200` (B200 183 GiB)
**Commit** e0c8750, with the `crediticl` registration fix pulled mid-run | **walltime** 1 h requested
**torch** 2.11.0+cu128, CUDA 12.8, driver 595.71.05, python 3.12.3

Preceded by jobs 11516936/11516938 at 10:30, which **all exited 2 in 21–60 s**: the job script
passed `--resume auto`, a flag `pretrain.py` does not define. See **Dead ends**.

### Configuration
- `config/Exp1_LGD.yaml` and `config/Exp1_PD.yaml`, arms 0/3/11/1 of 96
- **Budget** 1,500 steps × 4 datasets = 6,000 datasets — `--max-steps` overriding the config's
  12,500. **This is a plumbing test, not an experiment.**
- **Init** scratch. LGD 28,544,991 params; PD 27,538,938. All trainable.
- `prior source: GENERATE`, `amp=True`

### Results
Seven of eight arms completed 1,500 steps. LGD array task 3 never started — `ReqNodeNotAvail,
Reserved for maintenance`, deferred to 15-08 02:20, not a fault.

| arm | GPU | steps/s | mean GPU util | wall | final train loss |
|---|---|---|---|---|---|
| LGD cf=0.0 (control) | RTX 5000 Ada | 6.62 | 69.0 % | 222 s | 0.127 |
| LGD cf=0.3 mechanism | RTX 5000 Ada | 4.95 | 70.7 % | 302 s | — |
| LGD cf=0.3 quantile | RTX 5000 Ada | 5.15 | 65.9 % | 299 s | — |
| PD cf=0.0 (control) | B200 | 0.56 | **3.5 %** | 2,633 s | 0.171 (acc 0.95) |
| PD cf=0.1 mechanism | B200 | 0.48 | **3.5 %** | 3,041 s | — |
| PD cf=0.3 mechanism | B200 | 0.51 | **1.7 %** | 2,867 s | — |
| PD cf=0.3 quantile | B200 | 0.51 | **4.1 %** | 2,874 s | — |

- **Learning works.** PD loss 0.618 → 0.171, accuracy 0.649 → 0.95 over 1,500 steps. LGD loss
  0.339 → 0.127. Gradients reach all three stacks (`col≈5e-2`, `row≈5.6e-2`, `icl≈5.5e-3`).
- **The prior mixer works.** `prior check @500` reports `sources={'base': 16, 'credit': 0}` at
  cf=0.0 and `{'base': 12, 'credit': 4}` at cf=0.1 — exactly the requested 1-in-10 rounded to a
  16-dataset sample. Boundary mass 0.0513 on LGD; PD base rate 0.4230 under `balanced`.
- **Progress curve** one row per arm. `progress.every_datasets` is 5,000 and the debug budget is
  6,000, so only step 1,250 was ever reached. Not a bug, but it means there is no curve.
- **Out-of-domain** only `tabiclv2` was scored: **25 of 50 cells OK**, the other 25 being every
  `crediticl` cell. See Bugs.
- **Files** `output/logs/debug_exp1_task*_115169*.out`, `output/manifests/*__{progress,telemetry}.csv`

### Bugs and anomalies
1. **The B200 ran at 1.7–4.1 % GPU utilisation and was 12× slower per step than the free RTX
   5000 Ada.** CPU sat at 9–14 % of 24 cores with `num_workers=23`, so neither the GPU nor the
   CPU was busy. Our own telemetry caught it (`WARNING mean GPU utilisation 3% — the run is
   probably STARVED`) but its advice — raise `num_workers` — is wrong here, since 23 of 24 cores
   were already allocated. **Confounded:** PD-vs-LGD and B200-vs-RTX changed together, so this
   run cannot say which caused it. Resolving that is the first thing the next run must do.
2. **`crediticl` scored nothing, anywhere.** Two distinct causes, both visible in the logs:
   `unknown baseline 'crediticl'` at 10:48 (no production caller ever invoked `register()`),
   then `needs checkpoint=<path>` at 11:33 after the registration fix was pulled mid-run — the
   evaluation never passed a checkpoint path. **Every number in this run is about `tabiclv2`.**
3. **The real credit datasets were not preprocessed on the cluster**, so the progress-time
   evaluation logged `0001.gmsc … not in the processed cache` for all 14 PD datasets. The final
   evaluation calls `ensure_processed()` and got further; the progress-time one does not.
4. **Three out-of-domain regression datasets reported `constant_prediction=1.0` when the model
   had emitted NaN** (`Another-Dataset-on-used-Fiat-500`, `miami_housing`,
   `physiochemical_protein`). `np.var` of an all-NaN array is NaN, fails `> EPS`, and falls into
   the constant branch — a numerical blow-up reported as a modelling quirk.
5. **`/lustre1/project/stg_00211/CreditICL/checkpoints` is not writable** (`Errno 13`), so every
   arm fell back to `$VSC_DATA`, which has a **75 GiB quota**. Eight 1,500-step arms produced
   3.0 GB. The real Exp1 is 96 arms at 12,500 steps.
6. LGD out-of-domain R² is negative on most datasets (−0.46 to −0.83) and predictions are
   near-constant. At 1,500 of 12,500 steps from scratch this is expected, and is recorded only
   so it is not mistaken for a finding later.

### Interpretation
- **What the numbers show:** the pipeline runs end to end — prior generation, mixing, training,
  checkpointing, progress evaluation and both final evaluations all execute, and the model
  learns. They show nothing about whether a credit prior helps, and were never going to.
- **What we think explains the B200 result:** the GPU is waiting on data, and the CPU is not
  saturated either, which rules out "not enough workers" and points at latency inside dataset
  generation — a serial section, or per-dataset work that does not parallelise. PD's prior
  rejects 41 % of candidates (`filter_rejection=0.41`) against LGD's 27 %, which costs
  regeneration but cannot explain a 12× gap on its own.
- **What would test it:** run one PD arm on the free RTX 5000 Ada and one LGD arm on the B200.
  Two jobs, an hour, and the confound is gone. Until then, do not conclude the B200 is slow.

### Next
Split the confound, and re-run with the checkpoint now passed so `crediticl` actually scores.

---

Submission attempts were made on wICE before 11-08-2026; their logs were read and the bugs they
exposed are recorded under **Dead ends** in [`AGENTS_MEMORY.md`](AGENTS_MEMORY.md), but no
write-up was kept and the logs are gone.
