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

## 16-08-2026 — GPU benchmark + LGD debug — **it was never the GPU, it is MUON**

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
   Exp1 and Exp2 are self-consistent, but **Exp3's PD warm start cannot load the released
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
`scripts/slurm/benchmark.slurm`, run on both cards and diffed with
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
