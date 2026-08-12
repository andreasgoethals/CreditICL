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

_No run has been written up yet._

Submission attempts were made on wICE before 11-08-2026; their logs were read and the bugs they
exposed are recorded under **Dead ends** in [`AGENTS_MEMORY.md`](AGENTS_MEMORY.md), but no
write-up was kept and the logs are gone. **The first entry here will be the first Exp1 run.**
