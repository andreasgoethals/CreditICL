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

**1. Before submitting.** Read this file and `AGENTS_MEMORY.md`. Then:

```powershell
.CreditICL\Scripts\python.exe -m src.utils.smoke_test --task lgd --steps 3
.CreditICL\Scripts\python.exe -m pytest -q
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

## 14-08-2026 — Exp1 debug, LGD + PD — pipeline runs end to end; our own model scored nothing

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
