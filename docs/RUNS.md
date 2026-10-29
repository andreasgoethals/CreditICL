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

**Contents:** [The workflow](#the-workflow) · [What gets logged](#what-gets-logged-and-why-each-thing-is-there)
· [Entry template](#entry-template) · [Runs](#runs) (the dated write-ups, newest first).

---

## The workflow

**1. Before submitting.** Read this file and `AGENTS_MEMORY.md`. Then, locally:

```powershell
.CreditICL\Scripts\python.exe -m src.validation.smoke --task lgd --steps 3
.CreditICL\Scripts\python.exe -m pytest -q
```

And **on the login node**, to verify permanent storage before requesting compute. Unwritable storage now fails closed:

```bash
python -m src.utils.check_storage
```

**1b. Preserve the previous run.** Current Exp0 attempts have separate directories. Keep completed
checkpoints and their logs; a rerun does not require clearing them. Retained checkpoints/results
and any large backup archives belong on PROJECT, not the small HOME allocation. Verify the
allocation's backup policy and independent recovery copy before deleting irreplaceable evidence.
Inspect before any deliberate cleanup:

```bash
python -m src.utils.clean_run            # LISTS what is there, deletes nothing
```

`clean_run` never touches `data/raw/`, released files in `checkpoints/`, or `tfm-library/`, and never
`prior_cache/ood/` even under `--prior-cache` — compute nodes have no outbound internet, so
that cache can only be rebuilt from a login node.

**2. Submit**, and **immediately add a stub entry here** with the date, the config and the job
id — before the run finishes. A job killed at the walltime never comes back to write its own
entry, and the submission details are exactly what is lost.

**3. When it finishes, Andreas downloads the output and uploads it back into a chat session** so
it can be read and debugged. What to collect:

```bash
# on the login node — small enough to pull down whole
tar czf "crediticl_$(date +%Y%m%d).tar.gz" -C "$VSC_DATA/CreditICL" "output CreditICL"
```

Also collect the benchmark CSVs and receipts from project storage, retaining their paths relative
to `output CreditICL/`; the small tier alone cannot verify benchmark results. Keep checkpoint binaries
on VSC. The historical paths below describe older runs; current paths are in `docs/VSC.md`.

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

## 29-10-2026 — PROJECT runtime rebuild and 1,500-update deployment

Runtime mode repair 62468816 completed: Ruff 0.16.2 and Torch helper executability
verified. Candidate r3 SHA256 is
`83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`.
CPU probe 62468931 passed 97 checks and failed one child-process import test; adding
only the mounted checkout to the container's PYTHONPATH fixes that independent-cwd
import route. Compiler/model/plot caches are explicitly redirected to scratch.
The candidate remains unaccepted until the current full CPU/GPU Exp0 pair passes.


- **Authorization:** the user requested the 64-run/two-seed design, runtime migration,
  scoped stale-output cleanup, full CPU/GPU acceptance, remaining Exp1 benchmarking and Exp2 launch.
- **Build:** wICE interactive job **62468678**, eight CPUs, two-hour limit.
- **Inputs:** the accepted DATA `.venv-skylake` distribution payloads, Python 3.12.3
  base SIF copied read-only from the already staged official Python base, and GCC runtime libraries.
- **Method:** hash-recorded installed-artifact wheels; fresh `/opt/venv` inside a PROJECT SIF;
  original Git package origins retained. No populated environment is moved.
- **Candidate:** `/lustre1/project/stg_00211/CreditICL/runtimes/crediticl-20261029-py3123-cu128.sif`.
- **Build outcome:** 62468678 failed before GPU work on a universal wheel tag. The repaired
  62468733 image built successfully and matched the entire legacy dependency fingerprint.
  Follow-up 62468797 restores standalone CLI binaries; CPU/GPU acceptance remains pending.
  The old environment and every Exp1 output remain preserved.
- **New design:** Exp2 64×1,500 updates; Exp3 32 new + 8 reused controls; seeds 0/1;
  permanent checkpoints 125/250/500/1000/1500. Exp1 remains 54 original runs.


## 29-10-2026 — Live read-only reconciliation; no new workload

**Scope:** native Windows SSH agent, LF-stdin transport and verified JSON stdout/stderr/status
capture. No authentication flow, file mutation, checkpoint read, migration or job submission.
Checkout `be6de6e` on DATA; tracked rendered notebooks/PDFs remain modified. No jobs in the
user's Mindwell/wICE queues at inspection. TFM pin `26156f36eb7676e8d9050547817346db548b0e85`;
VSC reference pin `0ed4ffa89d312b9bb6411681b6a14f81672439f3`.

**Exp0:** existing CPU job 62441437 has 13 PASS stages; GPU job 11741580 has 116 PASS stages,
both bound to source `4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0`.
These are prior-code receipts, not acceptance of the new fixed-rate design.

**Exp1:** all 54 training summaries report completion at 12,500 updates. For each task, 45 main
checkpoint evaluations (the original nine arms × five steps) and 15 component evaluations are
complete: 120 credit CSVs and 120 OOD CSVs across both tasks. The six 25% arms (main indices
9–11 per task) still need their 30 checkpoint evaluations: 30 credit CSVs and 30 OOD CSVs.
All 240 Exp1 CSVs (21,300 fold rows) and 16 reference CSVs (1,420 rows)
have successful rows, finite primary metrics, exact dataset/five-fold coverage, no recorded
context cap/subsampling and matching training-row counts. All 256 CSV hashes match 128 receipts.
This checks saved result integrity; it is not a new prediction run or checkpoint-byte audit.

**Measured benchmark work:** full credit+OOD suites from summed fold `total_seconds`:

| Model | PD hours | LGD hours |
|---|---:|---:|
| Released TabICLv2 | 2.3824 | 0.1319 |
| Released TabPFN-3 | 1.9298 | 0.1388 |
| Exp1 TabICL median, 45 evaluations/task | 2.5337 | 0.1268 |
| Exp1 observed range | 2.2773–2.7445 | 0.0997–0.1477 |

These are work timers, not Slurm elapsed time or billing records. AlgorithmWatch alone took
1.6166 hours with released TabICL and 1.0653 hours with released TabPFN. Under comparable
throughput, the new four-step Exp2 benchmark is about 440–454 timed hours, and Exp3 adds
220–227; adapter inference remains unmeasured. At the earlier actual-script quote of 30,600
credits/hour, these imply 13.46–13.89M and 6.73–6.95M credits respectively, before startup/retries.

**Storage/account:** DATA 56,233M / 76,800M; PROJECT 538.2G / 1T; GPFS scratch 8.634G / 500G;
Lustre scratch 28.5G / 500G. These are account/allocation totals, not project-only sizes.
CreditICL's DATA venv is 9.8G and its PROJECT output is 36G. Available credits: 95,812,429.
The pending SIF migration is a separate explicit change. No files were deleted during this inspection.


## 29-10-2026 — Authorized Exp2 reset completed; Budget A not yet submitted

**Evidence:** VSC terminal attachment `00dae2ae-033f-49f8-8ee4-4792c8c616ec`, ending 11:39.
Andreas pulled be6de6e, activated the existing environment, previewed and applied
`python -m src.utils.clean_run --reset-exp2 --clean`. The utility reports **15,071 generated files,
95.010 GB removed**, preserving inputs and released reference weights. Targets include both output
tiers and classified Exp2 scratch jobs; Exp1 was outside the reset scope.

| Resource after cleanup | Used / limit |
|---|---|
| DATA | 56,261M / 76,800M |
| GPFS scratch | 422.4G / 500G |
| Lustre scratch | 28.5G / 500G |
| Project stg_00211 | 616.9G / 1T |
| Account credits | 96,335,099 balance; 521,235 reserved; 95,813,864 available |

No new scheduler submission or benchmark completion appears in this transcript. The prior snapshot
still establishes 54 completed Exp1 training runs, with the quarter-mixture benchmark work outstanding.
The remaining GPFS use has not been attributed to CreditICL; do not delete unrelated project data.

**Approved replacement:** Budget A has 48 × 2,500-update main Exp2 runs plus 16 × 500-update LR
pilots, development-only selection and no changes to Exp1. At the quoted 30,600 credits/allocation-hour,
the provisional training-only estimate is 50–53M credits including those pilots. Benchmark/validation
cost and Exp3 are additional. Exp3 keeps matched 2,500-update controls, 36 short method pilots and
48 new main runs. The implementation moves temporary Mindwell I/O to node-local disk and publishes
every saved checkpoint to project storage; hardware acceptance requires a new Exp0 CPU/GPU pair.


## 29-10-2026 — Exp1 training completed; Exp2 stopped on GPFS quota and TabPFN loss overflow

**Evidence** Downloads `output CreditICL.zip`, modified 29-10-2026 08:51:08, 28,122,513 bytes,
1,944 entries, 657,442,508 uncompressed bytes; SHA256
`38901be1f55cd06c655b5f8fa22a6518e51bd015e3682ac9923b2cedf2365c74`.
Andreas also supplied live `myquota`, Mindwell `squeue` and `du` output at 08:53–08:54 CET.
The ZIP retains compact output, not the project-tier checkpoint binaries/benchmark CSVs.
Runtime logs identify commit `673dba5`, TFM pin `702d9ab0b66f`, torch 2.11.0+cu128 and
TabPFN 9.1.0. That commit pins VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`.
The executable source fingerprint remains
`4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0` locally during this audit.

### 29-10 implementation follow-up (no new cluster submission)

Andreas subsequently authorizes discarding all Exp2 runs and requests the numerical fix before
a fresh rerun. Query-target transforms now use float64; the native CRPS+MSE arithmetic widens
when squared targets or their batch reduction would overflow FP32. In a full-precision replay, very wide losses use
a common backward divisor for all accumulated tasks and L2-SP, reversed during the existing
global clipping rule. This retains the same mathematical objective and task weights without
materializing an unrepresentable FP32 gradient. The native batch-685 task is covered through
released weights; mandatory VSC tests cover its loss/update and an even wider finite target.
This does not establish CUDA acceptance until Experiment 0 runs on the final chosen design.

Hash-only validation reads no longer stage artifacts into GPFS. Both checkpoint writers remove
failed temporary files, and quota/access exceptions explicitly report storage access. The new
`clean_run --reset-exp2` previews an Exp2-only reset on both permanent tiers and identified job
scratch; `--clean` applies it after idle-queue/path checks. It preserves Exp0/1/3, released weights,
data, shared references and unrelated projects. No user output was deleted during implementation.
Reviewed source/recipe compatibility preserves all 54 completed Exp1 runs; it does not authorize
resuming the numerically changed Exp2 implementation.

The actual one-task, one-hour B200/24-CPU/180G `sam-quote` returns **30,600 credits**; available
shared-account credits are **95,945,277**. The full 96×10k plan projects to 373–395M training
credits. The 48×2,500 one-rate and 96×1,000 two-rate alternatives project to 46.6–49.4M and
37.3–39.5M respectively, before benchmarks/validation. Neither budget is implemented while
Andreas compares the literature-grounded proposals. TFM pin `26156f36eb7676e8d9050547817346db548b0e85`;
VSC pin `1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87`. These submodules were not edited.

Local verification: full `python -m pytest -q` **1,957 passed / 36 mandatory VSC cases skipped**,
1,190.03 seconds. PYTHONHASHSEED=0, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1; Git Bash was added only
to the test process PATH. The production-case helper separately fixes its child hash seed to 0.
The focused numerical/storage/provenance checks pass 144/144. Current executable fingerprint:
`07f1b4e40e258e7d9748af8a4ee7c227cef1897561e2b324269001838a8022b5`.
The reviewed training implementation is
`01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`.
Full Ruff still reports 185 existing findings; comparison of diagnostic code/message counts to
HEAD shows no introduced findings. The new test modules lint clean. No notebooks/visualization
source was changed; no package installation, paid cluster run or user-output deletion occurred.

### Experiment 1: completed training, quarter-mixture benchmarks pending

All six quarter-mixture runs report completed=true, 12,500 updates and 800,000 synthetic tables.
They use the existing scratch-training recipe with 64 tables/update, 25% credit prior and seeds
0/1/2; per-arm resolved configs are in their `experiment_1/runs/.../config.json` records.
Their training fingerprint is `f78b9ab09ce66d3107c77c8505624842e8497bb1fefb64b7cf5785b4aae1a3f5`.

| Array / arm | Task / seed | Training seconds | Successful end, 28-10 CET |
|---|---|---:|---|
| 11747538 / 9 | LGD / 0 | 23,773.1 | 16:20:07 |
| 11747538 / 10 | LGD / 1 | 23,448.4 | 16:14:34 |
| 11747538 / 11 | LGD / 2 | 23,802.2 | 16:20:43 |
| 11747539 / 9 | PD / 0 | 25,175.1 | 16:43:34 |
| 11747539 / 10 | PD / 1 | 25,095.6 | 16:42:15 |
| 11747539 / 11 | PD / 2 | 25,252.5 | 16:45:02 |

Each new telemetry file has 175 rows; no non-finite post-zero recorded training losses were
found. Their wrappers end OK without scratch-cleanup warnings. The Python-module warning
precedes successful project-venv activation and did not stop these runs.
The complete Exp1 training count is **54** (24 main mixture runs and 30 component ablations).
The old 48 records remain present. Benchmark wrapper logs still number 128 and end OK;
no quarter-mixture benchmark jobs appear. This is not independent verification of project-tier
checkpoint bytes or detailed metric coverage.

### Experiment 2: what ran and what failed

Arrays 11747551 (LGD) and 11747552 (PD) use the resolved two-backbone, four-mixture, two-rate,
three-seed grid documented in the preceding submission entry: 48 arms per task, 10k updates.
All observed arms have seed 0. There are **19 resolved production run configs**, **zero completed
Exp2 runs**, and three partial summaries. The configured remaining 77 arms failed before training.

The 99 wrapper attempts classify as:

| Outcome | Attempts | Evidence |
|---|---:|---|
| Checkpointed and resubmitted | 3 | LGD TabICL primary-rate 0/25/50% arms, steps 551/550/550 |
| Checkpoint stream-write failure | 14 | PyTorch `basic_ios::clear: iostream error` / `unexpected pos` |
| Explicit quota failure | 79 | 77 initial arms plus two continuations; preflight/input-copy failures |
| Non-finite loss | 3 | LGD TabPFN primary-rate 0/25/50%, all at update 685 |

The three partial TabICL summaries retain approximately 42,500 training seconds each. Their
permanent stop checkpoints were logged as published. The 0% continuation subsequently logs
step 700 before failing on its next checkpoint write; the 25% continuation fails copying its
step-550 state to scratch. Other started TabICL arms log step 200 and saved step 125 before
the step-250 write fails. The TabPFN arms save step 500 before failing at 685. These are
recovery candidates, not verified intact checkpoint bytes: inspect both tiers before deleting.

Across 21 attempts with complete Date/END banners, their elapsed intervals sum to about
118.22 GPU-hours. This excludes startup outside those banners and the 78 preflight-only
attempts; it is neither Slurm billing nor a claim that all of that work is lost. Several states
were checkpointed and can potentially resume after integrity/provenance checks.

### Storage diagnosis and misleading readiness message

The live quota output reports:

| Tier | Used / quota | CreditICL footprint from supplied du |
|---|---|---|
| DATA | 56,221M / 76,800M | Not separately supplied |
| GPFS scratch | 497.7G / 500G | 84G, all under `CreditICL/jobs` |
| Lustre scratch | 27.25G / 500G | Not separately supplied |
| Shared project stg_00211 | 623.5G / 1T | 50G total: 47G output, 2.5G data, 640M released checkpoints, 28M prior cache |

GPFS scratch is the immediate exhausted tier. Roughly 414G of its usage belongs outside
CreditICL; its ownership/disposability is not established. Mindwell's live queue contains
only JEVPFN job 11750830, so no CreditICL job remains active there. Do not touch JEVPFN's
workspace or indiscriminately delete the shared scratch root.

An explicit failure copies a known project checkpoint into
`/gpfs1/scratch/383/vsc38338/CreditICL/jobs/11750317/checkpoints/.../.step-550.ckpt....copying`
and raises Errno 122. Other failures occur while staging Exp0 validation checkpoints into
per-job `inputs/`. `campaign.readiness` checks artifact hashes through `input_digest`, which
uses `working_storage.local_input` and copies each input per job. Its generic exception
message then says to rerun CPU/GPU Exp0. That instruction does not establish invalid Exp0:
both stored reports remain PASS, and all 23 CPU plus 205 available GPU artifact hashes match.
The other 48 GPU artifacts are absent from this ZIP. Failed copy attempts and failed training
jobs retain scratch; successful quarter-mixture jobs report no cleanup warning.

Per-job `inputs/` are staged read-only copies and are reconstructible. Per-job `checkpoints/`
may contain the newest state, including rolling checkpoints not yet on project storage.
Free only identified disposable caches after checking the queue; recover/verify checkpoints
before retiring a whole job directory. A full output reset would destroy useful results and
passing validation evidence without correcting the numerical issue.

### Numerical diagnosis: original-prior batch 685

The three LGD TabPFN arms fail with `FloatingPointError: non-finite loss at step 685`.
They also record very large finite training losses, reaching about 1.205e31 before failure,
and AMP retries/full-float32 replays on earlier batches. Their code raises immediately on a
non-finite forward loss; the existing FP32 replay path handles gradient overflow.

A CPU-only reproduction regenerated seed-0 batch 685 from the archived resolved config.
Slot 15 is an identical original-prior task in the 0/25/50% mixtures: 1,024 rows, 727 context
rows and 297 query rows. TabPFN's estimator-specific target transformation gives a maximum
absolute target of **1.0339849692701e25**, although the raw target remains finite. For finite
zero logits, native `_compute_regression_loss` with unchanged CRPS+MSE weights produces:

| Loss arithmetic | Result |
|---|---:|
| float32 | infinity |
| float64 | 1.7998735970984676e47, finite |

This isolates loss-range overflow without training or changing model weights. The local
environment differs from VSC: torch 2.13.0 rather than 2.11.0+cu128, NumPy 2.5.1 rather than
2.5.2; TabPFN 9.1.0, TabICL 2.1.1 and scikit-learn 1.9.0 match. It establishes a reproducible
failure mechanism, not bitwise equivalence of every VSC tensor or a validated GPU fix.
Wider target-transform/loss arithmetic is a candidate correction that retains the mathematical
objective; silently clipping targets or dropping the shared task would change the experiment.

### Cost, validation limits and next steps

Production TabICL logs show roughly 76–80 seconds per 64-table update, approximately
760,000 rows/update, with forward/backward accounting for over 99% of measured phases.
The first 200–500 updates imply roughly **210–225 GPU-hours per 10k-update arm**: about
10k GPU-hours for 48 TabICL arms, before TabPFN and benchmarking. This is a measured-throughput
extrapolation, not observed campaign completion, scheduler ETA or billing. Review the budget
before another full-grid submission; the old short-table timing was deliberately not used.

Exp0's native TabPFN stop/resume route ended at update 600, so it did not reach failing batch
685; its short tests also could not certify a full concurrent campaign under another project's
growing shared scratch usage. Add a direct regression probe for that batch and capacity/error
handling checks rather than claiming that the previous PASS tested every future synthetic task.

Preserve all Exp1 output, current Exp0 artifacts and Exp2 recovery candidates. Restore scratch
capacity, then the current unchanged Exp1 benchmark planner can fill the 25% slots. Correct and
validate the numerical path and storage/readiness behavior before resuming Exp2; any code
revision needs current validation without discarding compatible completed Exp1 results.
Exp3 remains unstarted and dependent on Exp2's development-only choices. This audit makes
no executable change, deletes no output and submits no jobs.

## 28-10-2026 — six Exp1 extension runs and 96 Exp2 CPT runs submitted

**Evidence** Terminal attachment `723611bd-b126-4ffc-a791-5f8c749844a7/Pasted text.txt`,
SHA256 `b9fb983ec5a2c3349b12cfe291cc7e97d2b4a3bf606ff256e469b828dc4cb382`.
At 09:09–09:10 CET the current Exp0 preflight and verifier pass (122 items, zero FAIL).
The transcript then confirms the following scheduler acceptances, ending at 09:47 CET:

| Job | Config | Array | Purpose |
|---|---|---|---|
| 11747538 | `config/Exp1_LGD.yaml` | `9-11%8` | Three 25%-credit scratch runs, seeds 0/1/2 |
| 11747539 | `config/Exp1_PD.yaml` | `9-11%8` | Three 25%-credit scratch runs, seeds 0/1/2 |
| 11747551 | `config/Exp2_LGD.yaml` | `0-47%8` | All 48 planned LGD continued-pretraining runs |
| 11747552 | `config/Exp2_PD.yaml` | `0-47%8` | All 48 planned PD continued-pretraining runs |

### Configuration and results so far

All submissions use the existing `pretrain_<task>.slurm` wrappers, `VARIANT=` and
`--single-cluster`: Mindwell B200, one GPU, 24 CPUs, 180 GiB and 12-hour segments per task.
The walltime continuation mechanism checkpoints and submits the unfinished arm again; a
production run is not necessarily finished within one segment. These are requested resources,
not a claim that a node was allocated or that training started.

Exp1 preserves the previous nine main runs per task and all 30 component runs; only indices
9–11 are submitted. Each added run uses the existing 64-table, 12,500-update recipe.
Exp2 crosses two backbones, four credit mixtures (0/25/50/100%), two model-specific rates and
three seeds for each task. TabICL uses 64 ragged tables/update, 400–60,000 log-uniform rows,
79–81% context, Muon and rates 2e-5/1e-5; TabPFN uses 16 tables/update, 1,024 rows,
30–90% context, AdamW and rates 1e-5/1e-6. Both use 10,000 updates, weight decay 0.01,
gradient clipping 1.0 and permanent checkpoints at 125/250/500/1,000/2,500/5,000/7,500/10,000.
The full resolved per-arm configs will be written with the production run artifacts. Package
versions and the current Exp0 receipt pass before both arrays; the validated source remains
`4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0`.

### Interpretation and next steps

No error appears in the submission transcript. Scheduler acceptance does not establish current
RUNNING/COMPLETED state, numerical correctness or final throughput. No new storage/quota output
was supplied. Keep code, packages and current output unchanged while these campaigns run; check
the live queue, quotas and credit balance. The `%8` cap applies separately to each Exp2 array,
subject to scheduler/user limits. After each experiment's training finishes, submit its missing
benchmarks with `python -m src.utils.run_experiment <1|2> --phase benchmark --single-cluster --submit`.
The training-only submissions above do not launch those benchmark arrays automatically.

## 28-10-2026 — current CPU/GPU validation passed; Exp1 extension remains unrun

**Evidence** Downloads `output CreditICL.zip`, 28-10-2026 09:10:54, 26,998,892 bytes,
1,600 entries, SHA256 `67ef250139ee0dd4357dc3123c562c9dd9da1389591a1ca2912b1fce111a6fcf`.
Andreas's on-VSC verification reports **122 items, zero FAIL** on 28-10 at 09:10.
No live scheduler query was made during this archive audit.

| Job | Resources and timing | Outcome |
|---|---|---|
| CPU 62441437 | wICE/batch, l33c20n1, 20 CPUs, 64 GiB; 27-10 08:35:55–09:05:11 CET | All 13 stages PASS; 1,907 unit tests passed, one external-project skip, 32 GPU tests deselected; notebook stage PASS |
| GPU 11741580 | Mindwell/gpu_b200, r11g22, B200, 24 CPUs, 180 GiB; 27-10 09:08:41–28-10 03:06:30 CET | All 116 stages PASS in 17h57m49s; 32 VSC tests passed, no skips |

### Configuration and evidence integrity

Both jobs ran commit `673dba5`, reporting a dirty checkout but identical executable source
fingerprint `4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0` and environment
fingerprint `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.
The source fingerprint still matches the local executable code on 28-10. The tested commit pins
TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465` and VSC
`3bcdf55e576d10161d6189b40ac39380bbba9eca`; subsequent local library-only updates do not alter this hash.
Environment checks pass for TabPFN 9.1.0, TabICL 2.1.1 at
`8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10, torch 2.11.0+cu128,
135 dependency requirements and pip check.

Canonical attempts are `cpu_20261027T073556Z_j62441437_3ac75552` and
`gpu_20261027T080841Z_j11741580_f4269428`. Their schema-3 reports contain PASS for every stage.
The GPU report binds the CPU receipt SHA256
`3bd13a45a9a060bd2942d735d51708e864f0d15a298b320322cf7c3884d82abd`.
All 23 CPU and 205 available GPU artifact hashes match. Another 48 GPU artifacts (six checkpoints,
28 CSVs, 14 receipt JSONs) live on project storage and are absent from this ZIP; local inspection
cannot independently verify those bytes. The user's on-VSC verifier is the evidence for that gate.

### Results and limits

Validation covers package/weight loading, native prediction equivalence, PD/LGD scratch and
continued-training stop/resume, benchmark receipts and focused credit/OOD evaluation. It also
passes all 16 backbone/task/retention-method round trips and 34 long-context probes, including
60,000-row base/credit tables, LoRA/DiagScale at that length, and mixed 64-table updates.
Resolved diagnostic configurations and observations remain in each attempt's run/report artifacts;
these short tests are not production efficacy measurements.

The mixed PD update took 90.166s generation plus 63.769s training, peak 19.093 GiB GPU memory;
LGD took 54.374s plus 63.438s, peak 19.055 GiB. Each contained 64 tables, 32 credit tables,
413–56,012 rows in this particular draw. Native training attention used FP32, TF32 enabled,
efficient SDPA, no FA3, activation checkpointing, four workers and prefetch one. These single
updates demonstrate capacity, not representative throughput across a 10,000-update campaign.

The verifier's short-run TabPFN projection is 1,227.6 GPU-hours for 24 PD runs and 881.8 for
24 LGD runs: **2,109.4 GPU-hours before any TabICL training or benchmarking**. Queue delays,
billing and long-run throughput are not established. Exp2 contains 96 total runs; this estimate
must not be presented as a complete campaign cost.

### Experiment 1 and storage

There are 48 completed Exp1 summaries, each at 12,500 updates and 800,000 synthetic tables:
9 main PD, 9 main LGD, 15 component PD and 15 component LGD. Their original training fingerprint
is `0d328e50ce06778731c5ba5218a61222b9cf396c8b8d767cb3e8a58f7473e69b`.
All 48 training and 128 benchmark wrapper logs end OK; the 240 inner evaluation logs have no
wrapper end marker and are not thereby failures. No tracebacks or ERROR markers occur in those
Exp1 logs. Full project-tier benchmark CSVs are not included, so this audit does not establish
numerical superiority or complete metric coverage. There are no quarter-mixture run paths and
no Exp2/Exp3 output: the six 25% runs and their benchmarks remain pending in the snapshot.

Only the current passing CPU/GPU Exp0 attempts occur in this ZIP. Its uncompressed size is
652.8 MB, mostly 622.5 MB of retained Exp1 output; Exp0 accounts for 24.0 MB. No cleanup was
performed, and this export cannot prove the absence of legacy files elsewhere on VSC.
The CPU storage probe on 27-10 measured 0.63 GB in the small output tier and 27.028 GB in the
large output tier. Its `myquota` output reports DATA 56,113M/76,800M, project stg_00211
559.8G/1T (44,784/150,000 files), GPFS scratch 151.7G/500G and Lustre scratch 26.2G/500G.
These are dated quotas; filesystem free-space probes are not quota headroom.

### Interpretation and next steps

No new validation failure was found. Keep this executable version and environment fixed; a fresh
Exp0 run is not required. Use the existing planner to finish only the six missing quarter-mixture
Exp1 arms and then missing benchmarks. Exp2's four-mixture grid is prespecified and does not wait
for an Exp1 winner, so its validation prerequisite is now satisfied. Check current quotas and
credits before the 96-run launch; the complete campaign ETA remains unknown. Preserve completed
Exp1 output and the current passing Exp0 reports and referenced artifacts on both tiers.

## 26-10-2026 — GPU 11739875 refused a stale continuation before training

**Evidence** Downloads `output CreditICL.zip`, downloaded 26-10-2026 21:42, 26,856,748 bytes,
1,486 entries; SHA256 `b64ae347e49e68b14a371292233c6650d6d2c8d925b8cf004ef306cf7c8dcc53`.
Andreas supplied `sacct`: **FAILED**, **1:0**, **00:00:29**, limit **24:00:00**.
**Job** GPU **11739875**, Mindwell B200 node r11g12; 24 CPUs, 180 GiB host RAM.
**Code** `19a76cc`; TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`,
VSC pin `3bcdf55e576d10161d6189b40ac39380bbba9eca`.

### Configuration and results

`scripts/slurm/exp0.slurm` requested continuation of the recorded GPU campaign. The main log
`experiment_0/logs/exp0_gpu_20261026_155903_j11739875.log` shows passing package checks:
TabPFN 9.1.0, TabICL 2.1.1 at the pinned source revision, CatBoost 1.2.10,
torch 2.11.0+cu128, 135 dependency requirements and pip check. Then validation stopped with:

`Experiment 0 continuation REFUSED: Code/config/tests/packages changed; a fresh CPU/GPU campaign is required`.

The selected old campaign was `gpu_20261021T123317Z_j11724030_f8a32d10`.
No new model-training or benchmark stage ran, and no new production hyperparameters were resolved.
The fresh CPU campaign `cpu_20261026T141139Z_j62431090_fbb2a984` passed its stages; its unit report
contains 1,876 tests, zero failures and one skip (999 seconds for that stage).
The archive contains 48 completed Experiment 1 training summaries at 12,500 steps, with no
completed 25% extension. These are metadata observations, not verification of remote checkpoint bytes
or complete project-tier benchmark metrics. Existing outputs were not modified during inspection.

### Cause and correction

The strict continuation guard correctly rejected stale evidence, but the job wrapper always asked
to resume. It now uses `--resume-if-current`: source/package/input-compatible attempts continue with
verified artifact reuse; stale attempts start a fresh isolated campaign. Explicit `--resume` still
refuses stale evidence. Old failed logs remain evidence rather than being relabelled as passing.

### Interpretation and next steps

This failure is an orchestration error before training, not numerical divergence or damaged Exp1
weights. The approved Exp2/Exp3 implementation now adds new training routes and therefore requires
a fresh CPU/GPU validation pair. Preserve Exp1, validate this source without changing it mid-run,
finish only missing 25% runs, then their benchmarks. Exp2 and Exp3 must wait for their new GPU gates;
local adapter checks do not establish B200 60k-row capacity or campaign runtime.

Post-fix local verification: 1,928 tests passed, 32 VSC-only tests skipped; all 12 notebooks ran
and saved 40 visually reviewed PDFs. The 16 released-weight CPU method round trips passed,
with the four L2-SP routes repeated after the final anchor-gradient correction. No new VSC
job was submitted from this checkout. Source fingerprint for the proposed fresh validation:
`4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0`.

## 26-10-2026 — current Exp0 CPU passed; GPU validation submitted

**Evidence** Terminal attachment `5dd82437-9175-46aa-92be-9fb495f153b4/Pasted text.txt`.
**Jobs** CPU **62431090** on wICE; GPU **11739875** on Mindwell.
**Commit** `19a76cc`; checked source fingerprint
`264c70dd8f5aff5acd72776366520de3765cf0a51383f9c761c8f8aef1eed5a0`.
**TFM pin** `702d9ab0b66f6639ba8d443ff9c4a91231964465`; **VSC pin**
`3bcdf55e576d10161d6189b40ac39380bbba9eca`. Runtime/resource accounting was not supplied.

### Configuration

Unmodified `scripts/slurm/exp0_cpu.slurm`, followed by the passing CPU prerequisite gate and
`scripts/slurm/exp0.slurm`. The checkout contains the six-run Exp1 quarter-mixture extension
and the existing single-learning-rate Exp2 design. No new production training was submitted in
this transcript. The proposed second learning rates and replacement Exp3 remain design work.

### Results

CPU receipt `cpu_20261026T141139Z_j62431090_fbb2a984` passed the current-code/all-artifacts gate.
The environment checks passed TabPFN 9.1.0, TabICL 2.1.1 at the required source commit,
CatBoost 1.2.10, torch 2.11.0+cu128, 135 dependency versions and pip check. All configured credit
data, released weights and 25 classification/25 regression OOD cache entries passed preflight.
The GPU submission succeeded; its live state, final verdict, throughput and memory are unknown.
No benchmark metrics or new resolved production-run configurations were supplied.

### Bugs and anomalies

No failure is shown in this transcript. Submission is not GPU validation success.
The scoped pre-pull stash preserves notebooks and generated figures; it is not an output reset.

### Interpretation

CPU validation for the extension is established. GPU validation still needs its own final receipt.
The local imported Exp1 metadata remains 48/48 completed at 12,500 steps; this is not a fresh
remote checkpoint-byte audit. Existing compatibility tests passed in a 129-test focused review.

### Next

Inspect GPU 11739875, then require Exp0 PASS before submitting only the missing three PD and
three LGD quarter-mixture runs. Benchmark their saved checkpoints after completion, reusing only
valid compatible prior receipts. Preserve all completed Exp1 outputs. New Exp2/Exp3 implementation
will need its own tests and validation and must not invalidate these results by broad reset.

## 26-10-2026 — GPU submission reused old validation after the pull failed

**Evidence.** Terminal attachment `990a68c9-e1da-4763-9b81-d427c8562787/Pasted text.txt`
and Downloads `output CreditICL.zip`, 28,497,143 bytes, SHA256
`8a919090384f00c60c8c6bbba442a1de8bcdced0b60761eb88c5e165d0b80b5b`.
The ZIP was read in place; no output was imported, deleted or overwritten.

**Job and revision.** GPU job **11737746**, Mindwell B200, 24 CPUs / 180 GiB requested,
logged commit **a073d7d**, dirty checkout. The preceding pull from a073d7d to d6cd9ce
aborted because executed notebooks and tracked `output CreditICL/All_Results.md` had local changes.
The separate submission command therefore ran the old code. Its log at 08:33:07 CET prints
the existing PASS for `gpu_20261021T123317Z_j11724030_f8a32d10`, then exits OK at
08:33:21 CET. It did not execute new validation stages. These log times are not scheduler billing.

**Configuration and results.** No new training configuration was executed. CPU job 62382230
and GPU job 11724030 remain the active PASS evidence for source `20f0e981...`.
The revised Exp2 source is `fb8521169c961e9c8edc8f464aa2ea44d802b05c39a3c499aebe6910d73376b7`
and has not been validated on VSC by this submission. The printed short-run Exp2 cost projections
come from the old verifier and do not estimate the revised 400–60,000-row TabICL configuration.
Dependencies passed (TabPFN 9.1.0, pinned TabICL 2.1.1, CatBoost 1.2.10, torch 2.11.0+cu128).

**Exp1 audit.** All 48 training summaries remain completed. The production log scan covers
496 files, with no traceback, failed footer or positive failed-cell count. The 176 successful
Exp1 job footers comprise 48 training jobs and 128 benchmark array tasks. Actual production
benchmark CSVs/journals/receipts are still absent: collect the project-tier experiment_1/benchmark
and reference/benchmark folders before numerical comparison or publication claims.

**Cleanliness.** Three Exp0 attempt directories are present: superseded CPU 62378918,
accepted CPU 62382230 and accepted GPU 11724030. Archive the superseded attempt and its log
outside active output once idle, retaining the accepted pair as Exp1 provenance. The new GPU
submission adds only a redundant successful top-level log. Empty lock files are coordination
artifacts, not failed runs; do not unlink them while processes could hold locks. The local working
output contains completed Exp1 records and current generated figures, no Exp0 attempt tree.
No broad reset is appropriate now that Exp1 is complete.

**Next steps.** Confirm no active CreditICL jobs, preserve tracked notebook/report edits with
a scoped Git stash, pull successfully, verify the intended source fingerprint and run CPU Exp0,
then GPU Exp0. Inspect long-context capacity/throughput results before launching Exp2 (36 runs at
the time of this audit; extended to 48 later on 26 October when the user requested 25% credit).
No executable source, configuration, environment or submodule was changed in this audit.
Read-only pins: TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465`,
VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`.

**Later 26-10-2026 follow-up.** The user requested a clean current-code validation pair. The new
`--reset-exp0` utility can remove the superseded/old successful Exp0 attempts on both tiers while
retaining all Exp1 outputs. The historical PASS outcomes above remain provenance; deleting their
active receipt files does not invalidate the completed Exp1 training. New precision checks and
the four-mixture Exp2 grid require fresh CPU/GPU validation. No remote deletion was performed here.

## 25-10-2026 — Exp1 benchmark jobs completed; project-tier results still to collect

**Evidence** Downloads `output CreditICL (1).zip`, downloaded 25-10-2026 10:28,
28,495,863 bytes, 1,359 files; SHA256
`b978b33762131959adb5adfa1b5fd4c63685b94f49e322483e090a93543d1ac9`.
This is an output snapshot, not a live scheduler query.

### Configuration and completion

| Job array | Work | Array indices | Successful log footers |
|---|---|---|---|
| 11734956 | main LGD checkpoints and references | 0–48 | 49/49 |
| 11734957 | main PD checkpoints and references | 0–48 | 49/49 |
| 11734971 | LGD component checkpoints | 0–14 | 15/15 |
| 11734972 | PD component checkpoints | 0–14 | 15/15 |

Each main grid covers nine arms at five saved checkpoints plus four released/classical
references; components use their final checkpoint. The logs report all 35 LGD credit cells,
70 PD credit cells and 125 OOD cells per respective evaluation completed successfully:
seven LGD / fourteen PD credit datasets and 25 native OOD datasets per track, five folds.
All 48 training summaries still report completion at 12,500 updates.
Environment checks record TabPFN 9.1.0, TabICL 2.1.1 at commit
`8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10 and CUDA torch 2.11.0+cu128.
The included Exp0 CPU/GPU PASS receipts retain source `20f0e98161f9e988b5eaf0d2775fb34406547844bafbbd90a653fb9bbe7e44c5`.

### Bugs and interpretation

No error traceback, ERROR marker, failed footer or positive failed-cell count was found
across 480 Experiment 1/reference logs. The 140 matched nonfinite metric log lines are
duplicated `crps=nan` entries for CatBoost/linear point predictions, where distributional
metrics are unavailable; this is not a failed prediction. Module fallback warnings are
followed by successful dependency checks. Full rows are explicitly logged, including
425,943/425,943 context rows for a large PD fold.

All 128 benchmark array tasks have successful completion evidence. However, the actual
benchmark CSVs, fold journals and receipts live in project storage and are absent from
this DATA-tier ZIP. Numerical completeness, paired effects and publication conclusions
still require those artifacts; log success alone does not replace their verification.
No code fix or Exp1 rerun is justified by this inspection.

### Next

Collect `experiment_1/benchmark` and `reference/benchmark` from the project-tier output root.
Confirm no CreditICL jobs remain active before updating the shared checkout. The revised
Exp2 long-context implementation needs fresh CPU and GPU Exp0 validation; the old PASS
receipts do not approve it. Review capacity timings before submitting its 36 runs.
No output was deleted and no cluster job was submitted by this audit. Local reference pins:
TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465`, VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`.

## 24-10-2026 — Exp1 training complete — all 48 runs successful

**Submitted** original arrays on 22–23 October; exact submission receipts not included | **jobs** 11728997 (main LGD), 11728998 (main PD), 11728999 (LGD components), 11729000 (PD components) | **cluster** Mindwell `gpu_b200`
**Validated commit** a073d7d | **source fingerprint** 20f0e98161f9e988b5eaf0d2775fb34406547844bafbbd90a653fb9bbe7e44c5 | **training fingerprint** 0d328e50ce06778731c5ba5218a61222b9cf396c8b8d767cb3e8a58f7473e69b | **GPUs** one B200 per task | **walltime** 12h requested; recorded training 6.43–7.38h per task

### Configuration

- **Config files** `Exp1_PD.yaml`, `Exp1_LGD.yaml`, `Exp1_PD_ablations.yaml`, `Exp1_LGD_ablations.yaml`. All 48 resolved configurations match the approved recipes, with the expected Slurm override from 12 to 23 data workers.
- **The levers** 0/0.5/1 credit fraction x three seeds x two tracks = 18 main runs. Five component removals at fraction 0.5 x three seeds x two tracks = 30 component runs.
- **Budget** every run completes 12,500 updates x batch 64 = 800,000 synthetic tables. Total 600,000 updates / 38,400,000 tables. Tables have 1,024 rows, 1–100 features, context fraction 0.3–0.9; Muon, peak LR 0.0008, micro-batch 4, float32.
- **Init** scratch TabICL throughout. Every summary reports `completed=true`, `resumed_at=null`, no stopping signal. No run settings were changed by this output audit.

### Results

| Group | Completed / planned | Median training hours | Range, hours | Sampled mean GPU utilization |
|---|---|---|---|---|
| Main LGD | 9 / 9 | 6.512 | 6.443–6.631 | 88.06% |
| Main PD | 9 / 9 | 6.981 | 6.848–7.376 | 84.22% |
| LGD components | 15 / 15 | 6.518 | 6.431–6.667 | 87.44% |
| PD components | 15 / 15 | 6.918 | 6.787–7.022 | 84.63% |

- **Completion evidence** all 48 job logs end `END status=OK`; each run logs a final checkpoint save at update 12,500. Last successful footer: 24-10-2026 01:21:15 CEST. Previously held `11728997_6` finished 23-10-2026 21:36:32 CEST. No fresh `sacct` output was supplied.
- **Progress curve** all 1,248 expected observations are present (26 per run, updates 0 through 12,500 every 500), with progress protocol 8 / feature policy `application-pd_default-lgd_v5`. Scored monitoring values, actual training losses, telemetry and weight diagnostics are finite. Losses on different synthetic mixtures are not directly comparable generalization measures.
- **Out-of-domain** monitoring contains the configured OOD tasks; production credit/OOD benchmark results are absent. The user reports launching the benchmark, but the ZIP contains no benchmark job ID, log or CSV to verify its live progress independently.
- **Throughput / cost** summed recorded training time is 323.616 GPU-hours, excluding activation/smoke overhead; this is not a SAM-credit accounting statement. Maximum sampled allocated GPU memory: 19.709 GiB.
- **Files** Downloads `output CreditICL.zip`, 8,126,272 bytes, SHA-256 `ddddb3b064c7d971520bffb626b2e7c67cfbceb6b81fab5133992e4dd4453c0e`; 847 files / 45,301,000 uncompressed bytes. No checkpoint binaries are included: their remote content was not independently verified.
- **Exp0 preservation** CPU/GPU PASS receipts remain bound to the frozen source/environment; all 128 available receipt-bound files match their hashes, while 48 project-tier artifacts remain outside this DATA-only archive. One older successful CPU attempt and 125 older frozen-source notebook PDFs remain; these are provenance/presentation artifacts, not duplicate Exp1 runs.

Development monitoring at the final update (equal weight per configured development dataset,
then mean and sample standard deviation across the three seeds):

| Credit-prior fraction | PD ROC-AUC, 6 development datasets | LGD R², 4 development datasets |
|---|---|---|
| 0% matched base-prior control | 0.7511 ± 0.0016 | 0.3356 ± 0.0026 |
| 50% credit | 0.7448 ± 0.0013 | 0.3311 ± 0.0027 |
| 100% credit | 0.7359 ± 0.0015 | 0.2982 ± 0.0059 |

The corresponding initial means were PD ROC-AUC 0.5052 and LGD R² −0.0042, identical
across mixtures within each seed. All three seeds have the same direction relative to
the control at the final update. These are fixed monitoring subsets with at most 4,096
context rows and 2,048 scored rows, **not the full-table five-fold benchmark**. They show
learning, but do not currently establish a gain from credit specialization. No confirmatory
test datasets enter this aggregation; the final benchmark and component analyses are pending.

### Bugs and anomalies

- No training exception, OOM, failed footer or nonfinite scored metric was found. The only nonfinite numeric cells are 48 initial `train_loss=nan` entries at update zero, before a training loss exists: expected missing observations.
- All 48 jobs report the known Python module-load fallback warning, followed by successful project-venv/dependency checks.
- Top-level logs contain 4,032 recurring sklearn probability-sum warnings in multiclass monitoring. The previously reproduced float32-to-float64 rounding mechanism remains applicable; finite/nonnegative/near-unit matrix checks pass. Raw probabilities are absent, so exact residuals cannot be reconstructed. This is not evidence requiring checkpoint deletion or retraining.

### Interpretation

Training completed as configured. Generalization gains and retention still require complete credit and native multiclass/regression OOD five-fold benchmarking, with paired seeds and complete cells. The new evidence does not justify retraining.

### Next

Keep the VSC checkout and outputs frozen during benchmarking; collect benchmark logs, CSVs and receipts from both tiers afterwards. Validate the revised Exp2 code separately before launching it. No cleanup, installation, submission or remote mutation was performed by this audit.

## 23-10-2026 — Exp1 first production snapshot — 21 complete, 13 running, one Slurm launch hold

**Submitted** exact submission timestamps not supplied; four arrays cover both main grids and component tests | **jobs** 11728997 (main LGD), 11728998 (main PD), 11728999 (LGD components), 11729000 (PD components) | **cluster** Mindwell `gpu_b200`; live jobs on r11g17/r11g22 at 09:14 CEST
**Validated commit** a073d7d | **source fingerprint** 20f0e98161f9e988b5eaf0d2775fb34406547844bafbbd90a653fb9bbe7e44c5 | **tfm-library gitlink** cbaff2f5fa0a6e910d7e7d1cd148437a3e0c217a | **vsc-library gitlink** 9e376b5ee9106e82d37eca86ca6acb861f580c93 | **GPUs** 1 B200 per task, %8 per array | **walltime** 12h requested per task

### Configuration

- **Config files** `config/Exp1_PD.yaml`, `config/Exp1_LGD.yaml`, `config/Exp1_PD_ablations.yaml`, `config/Exp1_LGD_ablations.yaml`; 9 main and 15 component runs per track, 48 total. Main mixture fractions 0/0.5/1, seeds 0/1/2; component tests at fraction 0.5 remove macro targeting, informative missingness, tail warps, shift or corrected cardinalities.
- **The levers** main `prior.credit_fraction`; components `arm` with the five named removals. All 34 saved resolved configurations pass comparison against their exact current expected run (prior, architecture, seed, initialization and optimizer recipe).
- **Budget** 12,500 updates x batch 64 = 800,000 synthetic tasks per run; micro-batch 4, scratch TabICL, 24 CPU cores / 180 GiB per GPU. Main permanent checkpoints every 2,500 updates; components retain the final checkpoint, rolling saves every 250 updates.
- **Init** scratch for every started arm. All 21 completed summaries have training fingerprint `0d328e50ce06778731c5ba5218a61222b9cf396c8b8d767cb3e8a58f7473e69b`, matching the validated TabICL implementation/environment. All 34 top-level logs first pass the exact current Exp0 GPU gate.
- **Anything edited by hand** no executable/package/config/notebook changes by this audit. The per-job two-update smoke test intentionally checks the credit path before the actual configured arm; its temporary credit-fraction override is not the production recipe. Saved production recipes and subsequent training logs match the declared fractions.

### Results

Snapshot combines the 09:12 download with the user's read-only `squeue` / `sacct` output at 09:14 CEST:

| Array / group | Completed | Running | Normally pending | Held |
|---|---|---|---|---|
| 11728997, main LGD | 8 | 0 | 0 | 1 |
| 11728998, main PD | 8 | 1 | 0 | 0 |
| 11728999, LGD components | 5 | 8 | 2 | 0 |
| 11729000, PD components | 0 | 4 | 11 | 0 |
| Total | 21 | 13 | 13 | 1 |

- **Completion evidence** all 21 completed summaries report `completed=true`, 12,500 updates and 800,000 tasks; 21 job logs end `END status=OK`, and supplied scheduler records show COMPLETED / exit 0:0 for the same tasks. No other completed task is inferred from absence in the queue.
- **Progress curve** 365,600 of 600,000 planned updates recorded (~60.9%); 43.75% of runs fully complete. The 13 active routes continue producing fresh metrics through 23-10 09:11:57 CEST. Monitored metrics are finite, all recorded training losses finite, and each started run's latest weight-drift observation is finite and positive. Training losses across different prior mixtures are not a like-for-like generalization comparison.
- **Out-of-domain** periodic monitoring is present; there are no production benchmark CSVs/receipts in this upload. Full credit/OOD five-fold evaluation follows training. No scientific winner or retention claim is established by these intermediate logs.
- **Throughput** completed LGD training median 6.51h (13 arms; range 6.44–6.67h); PD median 6.97h (8 arms; range 6.85–7.38h), before the short per-job activation/smoke overhead visible in scheduler elapsed times. All completed allocations fit the 12h request. Sampled GPU utilization means ~87.9% LGD and ~84.7% PD across available telemetry; these are sampled observations, not a billing efficiency measurement.
- **Remaining active work from the snapshot** main PD array 8 at 9,600/12,500, roughly 1.63h extrapolated remaining. LGD components: 5,000–10,200 updates, roughly 1.2–3.9h remaining per active task. PD components: 1,900–8,100 updates, roughly 2.5–5.9h. Each not-yet-started task still needs a normal full run plus queue time; no guaranteed calendar completion follows from these estimates.
- **Files** `C:\Users\U0152019\Downloads\output CreditICL.zip`, 6,476,160 bytes, SHA-256 `6713c528f221cf2f7dcd96693b56d1438f83ea4e175d2d33f3628ebeeb4e64dc`; 708 files / 37,728,169 uncompressed bytes, CRC valid, no duplicate or unsafe names. New production files are under `experiment_1/logs/` and `experiment_1/runs/` (34 resolved configs, 21 summaries). Project-tier checkpoint binaries are intentionally absent; their bytes were not independently hashed in this audit.
- **Exp0 preservation** the same CPU/GPU PASS receipts remain; all 23 CPU and 105 present GPU artifact hashes still match. The same six project checkpoint binaries, 28 benchmark CSVs and 14 receipts remain outside this DATA-tier download. The prior successful in-place VSC verification is historical evidence, not a new 23-10 check of those remote files.

### Bugs and anomalies

- **One held launch failure:** `11728997_6`, the main LGD control (`credit_fraction=0`, seed 2), is PENDING with reason `launch failed requeued held`, elapsed 0 and no corresponding training output in the ZIP. The supplied evidence establishes a launch hold, not its underlying filesystem/node/program cause; it contains no Python traceback for this task. Its held state blocks completion of the LGD main comparison until recovered.
- **Normal pending states:** LGD component indices 13–14 wait at the array's %8 concurrency cap (`JobArrayTaskLimit`); PD component indices 4–14 wait for scheduler priority. Every one of the 48 expected arms is accounted for between output and live queue.
- **Runtime audit:** no training traceback, fatal exception, failed footer, OOM or nonfinite monitored/training value found. Existing Mindwell Python-module fallback warnings occur before successful project-environment activation. Multiclass probability-rounding warnings recur in monitoring; they do not produce invalid monitored values. Counts from top-level and nested logs duplicate the same events, so they are not reported as independent failures.
- **No code fix is justified by this snapshot.** Do not invalidate a running validated campaign to remove cosmetic warnings, and do not confuse the smoke-test override with the actual control-run configuration.

### Interpretation

The production campaign is progressing as configured and within its requested per-task walltime. There is one operational recovery action, the held LGD control, rather than evidence requiring failed training arms or Exp0 to be rerun. Generalization and the prior-effect comparison still require the benchmark phase. A scheduler launch reason alone cannot prove the root cause is a particular node, filesystem or repository bug.

### Next

The agent advised `scontrol --clusters=mindwell release 11728997_6`, followed by a queue check for that exact task. This attempts to resume eligibility of the existing held task without submitting a duplicate or disturbing the other allocations. No release outcome has been supplied yet. If release is denied or the task returns to held, inspect `scontrol --clusters=mindwell show job 11728997_6`; an administrative hold may require VSC assistance. Do not repeatedly force release without diagnosing a recurring launch failure.

Keep the current source/packages and all ongoing jobs. After the training groups finish, submit missing benchmark slots with `python -m src.utils.run_experiment 1 --phase benchmark --single-cluster --submit`. No new jobs, package changes, cleanup or pushes were executed by the audit agent.

## 22-10-2026 — Exp0 GPU 11724030 — all 66 stages PASS on the frozen source

**Submitted** exact submission time not supplied; allocation recorded 21-10-2026 14:33:17 CEST | **job** 11724030 | **cluster** Mindwell `gpu_b200`, node r11g12
**Commit** a073d7d (`dirty=YES`; executable fingerprint matches) | **tfm-library gitlink** cbaff2f5fa0a6e910d7e7d1cd148437a3e0c217a | **vsc-library gitlink** 9e376b5ee9106e82d37eca86ca6acb861f580c93 | **GPUs** 1 x NVIDIA B200 | **walltime** 24h requested / 17h38m40s from banner to successful footer, not a scheduler billing record

### Configuration

- **Config files** `config/Exp0_PD.yaml`, `config/Exp0_LGD.yaml`; `scripts/slurm/exp0.slurm`, 24 cores / 180 GiB host RAM. GPU runtime reports Torch 2.11.0+cu128 and CUDA 12.8.
- **The levers** — six routes: scratch TabICL, continued TabICL and continued TabPFN-3, for each track, at the diagnostic 50% credit mixture. No new production sweep was submitted.
- **Budget** 600 updates per route; forced SIGUSR1 after update 100 stops at 101, then all six resume to 600. TabICL: batch 64, 38,400 synthetic tasks; TabPFN: batch 16, 9,600 tasks, zero skipped tasks.
- **Init** scratch or released classifier/regressor weights as appropriate; strict tensor matching and both upstream trainer-equivalence checks pass.
- **Frozen source** `20f0e98161f9e988b5eaf0d2775fb34406547844bafbbd90a653fb9bbe7e44c5`; environment `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`. GPU binds CPU receipt SHA-256 `d6e4a8bbc3509087af0197543ad9ef55b1463e9d05e5231fb548843e26b7f3e8` exactly; the full CPU/GPU input dictionaries are identical.
- **Packages** TabPFN 9.1.0; TabICL 2.1.1 at 8f1aa2098c894ab91dba15209cf2002ba4be6c6c; CatBoost 1.2.10; all 135 requirements and `pip check` pass. Dataset cache version 8, benchmark protocol 9.
- **Anything edited since validation** only these existing documentation records during the 22-10 audit; no executable, notebook-cell, dependency or input changes. The library pins above are verified checkout gitlinks; the uploaded runtime log does not include a separate `git submodule status` listing.

### Results

- **Validation** schema-3 receipt PASS, exact current 66-stage plan/order, every stage PASS; 28 CUDA tests passed with zero failures/errors/skips. Main log ends `END status=OK` at 22-10-2026 08:11:57 CEST. Supplied VSC preflight and verifier report all current prerequisites, 72 items and zero FAIL.
- **Full-table benchmark probes** four PD tables (HackerEarth, bank status, loan default, AlgorithmWatch), three LGD tables (HELOC, loss2, base_modelisation), five classification OOD tables and three regression OOD tables. All three diagnostic checkpoints and four references complete credit/OOD evaluation. Credit logs show 245 successful model/dataset/fold cells, with all folds 0–4. Largest PD table retains 532,428 rows; widest retains 158,700 rows and 2,986 features. Context caps remain `None`.
- **Interpretation of metrics** this is selected-dataset pipeline validation, not a completed Exp1 or a prior-effect claim. Reported reference means on those PD probes are AUC 0.711900 (TabICL), 0.724528 (TabPFN), 0.738023 (CatBoost), 0.697063 (linear); the 600-step scratch diagnostic is 0.646180. These summaries are from logs; the underlying project-tier CSVs are not in the download.
- **Progress curve** all six have points at steps 0/200/400/600; monitored metrics are finite. Only the deliberately uncomputed step-zero `train_loss` is NaN. This is not enough training to infer convergence or a scientific improvement.
- **Out-of-domain** multiclass and regression probes complete; full OOD retention comparisons remain part of Experiments 1–3.
- **Throughput** scratch training totals 1,286.3s PD / 1,260.1s LGD over 600 steps. Mean sampled GPU utilization is 86.4–87.2% across TabICL routes and 30.1% PD / 34.6% LGD for TabPFN; low TabPFN utilization is a performance limitation, not evidence by itself that adding workers fixes it. Largest observed process-tree RSS in heartbeat logs is 82.31 GiB, below the 180 GiB request; this is not scheduler MaxRSS.
- **Files** `experiment_0/report/validation.json`, attempt `gpu_20261021T123317Z_j11724030_f8a32d10`, job log `experiment_0/logs/exp0_gpu_20261021_143255_j11724030.log`. Download `C:\Users\U0152019\Downloads\output CreditICL.zip`: 3,733,314 bytes; SHA-256 `8dafa4663a67da647c5cf940ef303b8a6492871ee6a2734fed1206f31641292d`; 415 files, valid CRC, no duplicate/unsafe names.
- **Evidence completeness** 105/153 GPU artifacts are present and hash-identical; all 23 CPU artifacts also match. The omitted six checkpoint binaries, 28 benchmark CSVs and 14 benchmark receipts live on project storage. The supplied VSC verifier checks them in place successfully; independent local cell-by-cell CSV verification requires that tier as well. No omitted file is described as independently verified from this ZIP.
- **Projected training costs** Exp1 main comparison 132.6 GPU-h plus component tests 221.1 GPU-h (48 runs, about 353.7 GPU-h total); Exp2 about 886.9 GPU-h; Exp3 about 943.1 GPU-h. These extrapolate 600-step measurements, exclude production benchmarking/queue time and are not a billing statement or guaranteed walltime.

### Bugs and anomalies

- No failed stage, traceback, OOM termination or fatal runtime error in this successful attempt.
- The previously failing TabPFN LGD path now replays the same batches successfully in float32 at steps 217 and 511. Its cumulative summary records 50 AMP retries, two full-precision retries and zero skipped tasks; PD records three AMP retries, no full-precision retry and zero skipped tasks. Large finite synthetic regression losses are not downstream LGD errors; final monitoring remains finite.
- Multiclass OOD/monitoring emits probability-sum warnings. The metric path checks finite, nonnegative, near-unit rows; casting float32 probabilities to float64 can expose harmless rounding to sklearn's stricter warning threshold. A separate no-training reproduction using normalized float32 probabilities produced the same warning at a maximum row-sum error 1.29e-7 (log-loss change about 3.19e-9 after exact renormalization). This demonstrates a warning mechanism; the archive does not retain per-row probability matrices, so its exact residuals are unknown.
- One sklearn power-transform overflow warning occurs during LGD task preparation after step 520; the following updates, monitored predictions, checkpoint and all benchmark stages succeed. Its precise intermediate values are not retained. Do not silently claim every upstream operation was warning-free.
- Python module loading warns on Mindwell, then the project Python 3.12.3 environment, requirements and CUDA checks pass. Other warnings are upstream deprecations, deliberate signal/resume notices and utilization advisories.
- The ZIP retains old CPU attempt 62378918 (33 files / 500,191 bytes plus its main job log). There is one current CPU/GPU pair and no failed GPU attempt. Archive contents cannot establish that every legacy directory on both remote storage tiers is empty; no cleanup was performed.

### Interpretation

The frozen pipeline has passed its defined CPU and production-B200 validation, including the failure cases that previously blocked it. This supports starting Exp1 without changing source or rerunning Exp0. It does not prove every future synthetic task or production checkpoint will succeed, nor that the credit prior improves performance. Exp1 supplies that comparison. Keep existing receipts and outputs; source changes would require new validation.

### Next

Start Exp1 training on Mindwell using the current checkout and `python -m src.utils.run_experiment 1 --phase train --single-cluster --submit`; this includes the approved main grids and component ablations (24 runs per track). After training completes, use the same orchestrator with `--phase benchmark` to submit missing benchmark slots. Include project-tier CSVs/receipts with the next output download. No jobs were submitted by the audit agent.

## 21-10-2026 — Exp0 CPU 62382230 — current source passed all 13 stages

**Submitted** exact submission time not supplied; allocation recorded 21-10-2026 13:18:32 CEST | **job** 62382230 | **cluster** wICE `batch`, node m33c25n4
**Commit** a073d7d, `dirty=YES` with matching executable fingerprint | **tfm-library gitlink** cbaff2f5fa0a6e910d7e7d1cd148437a3e0c217a | **vsc-library gitlink** 9e376b5ee9106e82d37eca86ca6acb861f580c93 | **GPUs** none | **walltime** 4h requested / 21m04s banner-to-footer, not scheduler accounting

### Configuration

- **Config/entry point** `scripts/slurm/exp0_cpu.slurm`, 20 cores / 64 GiB, full 13-stage CPU campaign; source 20f0e981 and the same package/input inventories used by GPU 11724030.
- **The levers / budget / init** no production training arm; preprocessing, prerequisites, portable tests, benchmark/config/data/determinism/OOD/orchestration/path/prior/storage checks and all notebooks.
- **Anything edited by hand** previous executed notebook outputs and small generated summaries were preserved with a scoped Git stash before updating to a073d7d. The earlier Git lock/update failures occurred before submission and did not launch extra CPU jobs.

### Results

- PASS, 13/13 exact expected stages. Pytest: 1,617 passed, one expected missing sibling-TabPFNCredit identity skip, 28 CUDA cases deselected, zero failures/errors, 754.57s; GPU later executes all 28 CUDA cases. The PDF-bound test executes rather than skipping.
- Notebooks 11/11, 125 PDFs; notebook stage 225.04s. All 23 receipt-bound artifacts verified against the ZIP, and the GPU binds the exact resulting CPU receipt SHA-256.
- All processed data v8, full feature/no context-limit protocol 9, released weights and OOD inputs pass readiness. Small output tier at its storage check: 210 files / about 4 MB; current project output tree then empty. Shared project quota: 243G/1T, 36,730/150,000 files; DATA 54,256M/76,800M. These precede the GPU run and are not a current whole-filesystem cleanup certificate.
- **Progress / out-of-domain / throughput** no production learning curve or prior-effect estimate in this CPU job; its checks validate readiness. Main log ends `END status=OK` at 21-10-2026 13:39:36 CEST.
- **Files** `experiment_0/report/cpu_validation.json`, attempt `cpu_20261021T111832Z_j62382230_43923b09`, main log `experiment_0/logs/exp0_cpu_20261021_131814_j62382230.log`; archived in the 22-10 ZIP identified above.

### Bugs and anomalies

No CPU failures. One explicitly permitted missing-sibling test skip; upstream/test-fixture warnings remain in the log. The older CPU 62378918 attempt is still present separately and does not satisfy or invalidate this newer receipt.

### Interpretation

The current frozen source passed CPU validation and was subsequently consumed unchanged by the successful GPU campaign. No new CPU rerun is needed for this documentation-only audit.

### Next

Retain the CPU/GPU pair as the prerequisite evidence for Experiments 1–3; proceed to Exp1.

## 21-10-2026 — Exp0 CPU 62378918 — PASS; subsequent audit corrections require a new receipt

**Logged start/end** 09:56:44–10:19:34 CEST (22m50s; no sacct interval supplied)
| **job** 62378918 | **cluster** wICE `batch`, m33c22n1 | **commit** 576caea, dirty=YES
| **resources** 20 effective CPU cores, 65,536 MiB; requested four hours; no GPU.

### Configuration

- `scripts/slurm/exp0_cpu.slurm`, all 13 CPU stages; attempt
  `cpu_20261021T075644Z_j62378918_0028241f`, schema 3, source
  `f07e71e70eebd282b3401a9e745968ca47fc2edfec4821cc4608bf9eae21050a`.
- Environment fingerprint `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.
  TabPFN 9.1.0, TabICL 2.1.1 / commit 8f1aa2098c894ab91dba15209cf2002ba4be6c6c,
  CatBoost 1.2.10, torch 2.11.0+cu128; all 135 requirements and pip check passed.
- Cache/protocol 8; full-row/full-feature five-fold inventory. No production training/benchmark
  results expected. Run log does not independently record submodule pins; local audit pins are
  TFM cbaff2f5fa0a6e910d7e7d1cd148437a3e0c217a / VSC 9e376b5ee9106e82d37eca86ca6acb861f580c93.

### Results

- Canonical CPU receipt PASS, 13/13 stages and 129 individual checks PASS, no WARN/FAIL;
  all 23/23 recorded artifact hashes match. No traceback or failed footer; `END status=OK`.
- Portable suite: 1,590 passed, 2 expected skips, 28 GPU cases deselected, 879.26s. JUnit's
  1,592 includes skips: absent sibling-project identity comparison and PDF check before figures exist.
- 11/11 notebooks executed; 125 PDFs. Independent ZIP MediaBox inspection finds all 125 within
  the A4 figure bounds, covering the pre-generation PDF skip for this uploaded output.
- Current output storage at its check: 37 small-tier files/~1 MB and empty project output tree.
  Shared project quota 36,730/150,000 files, 243G/1T; DATA 54,245M/76,800M. The former nearly-full
  project inode condition is absent here. This does not enumerate every remote legacy/scratch path.
- ZIP: 2,581,682 bytes, SHA-256
  `e241c46e95c753155d16b473721f175ee499e390889d5989279719251a95a63e`, 178 files, one CPU attempt,
  no GPU attempt; CRC clean, no duplicate or unsafe entry paths.

### Bugs and anomalies

No CPU stage failed. Independent review nevertheless confirmed exclusion aliases/components,
missing ranking markers, architecture/mixture pooling, latent checkpoint-reference classification,
incomplete metric directions and stale notebook descriptions. Corrected those, strict portable-skip
checking, subprocess exit propagation and a current-child signal-ready handshake. Isolated LGD beta
draws so changing macro target loadings cannot alter the later macro-feature coin. Prior measurement
now separates binary positive-label rates from multiclass rarest-class rates. Protocol is now 9;
dataset membership and raw/cache tables stay unchanged.

Raw evidence: loss2's `clip((EAD - Proceeds + ELAO)/EAD, 0, 1)` matches all 4,638 complete targets
(maximum error ~5.6e-16). Base_modelisation's year alias matches 594/594, exposure alias 585/594,
and both sampling weights exactly identify the already excluded workout flags. Matched 5,671
unambiguous seven-feature loan signatures between Lending Club and hackerearth: all labels opposite;
all 14,705 hackerearth positive-recovery rows have source label zero. A standard amortisation fraction
explains R² 0.735 on 5,556 complete Lending Club LGD rows. These limit domain/target interpretation;
they do not authorize discarding datasets, flipping labels or rebuilding maturity cohorts.

### Interpretation and next

This CPU PASS was valid and unlocked GPU for f07e71e. The new corrections change the fingerprint:
publish/sync fixes, run CPU again, then require current CPU PASS before GPU. Preserve the uploaded ZIP;
no output was deleted here. Keep the approved grids, selection endpoints, full rows/features and
five folds. Do not adopt automatic restoration of AMP scale after every successful retry: backoff
calibrates GradScaler and repeatedly restoring an unsafe scale can repeat overflows. Existing
float32 replay and retry counters remain. No GPU PASS or production readiness is inferred.

### Final local verification of the corrected version

- Frozen source `20f0e98161f9e988b5eaf0d2775fb34406547844bafbbd90a653fb9bbe7e44c5`, identical
  before and after verification. No source edits during the final full suite.
- `python -m src.utils.run_notebooks --workers 2`: 11/11 notebooks OK, 125 PDFs; the slowest
  PD-prior notebook took 245s. Corrected the previously cropped positive-label-rate plot and
  removed the unsupported universal 10% collapse threshold; labels no longer certify every
  source target as default=1. Corrected Tanna's hybrid mean aggregation and range-midpoint note.
- Final `python -m pytest -q --junitxml <temporary report>`: **1,638 passed, 28 skipped**, zero
  failures/errors, 1,109.96s. All 28 skips are `tests.test_vsc`, requiring the actual allocation;
  the PDF check executes and passes after notebook generation. All 26 new audit regressions pass.
  Both ordinary and hard-kill continuation cases pass in this final invocation.
- An earlier trial had 1,634 passes, 29 skips and two failures: the Windows SIGUSR1 mock was
  corrected; a continuation case passed unchanged on isolated recheck and in the final serial
  suite (initial failure cause unconfirmed). An intermediate summary-string assertion was
  updated to the corrected positive-label wording. No failed trial is presented as a green run.
- `git diff --check` clean; TFM/VSC pins and the downloaded ZIP hash remain unchanged. Changes
  are local/uncommitted; no package install, VSC submission, output reset or Git publication
  performed here. Existing output figures/summaries were regenerated by the required notebook run.

## 21-10-2026 — audit of GPU 11715564 — 47 stages passed, TabPFN LGD gradient overflow

**Run** 20-10 13:28:50 through 21-10 00:07:55 CEST (main-log interval).
**Job** 11715564, Mindwell `gpu_b200`, node r11g22, one NVIDIA B200 (183,359 MiB),
24 CPU cores, 184,320 MiB host RAM; 24-hour request.
Andreas's `sacct` confirms **FAILED 1:0 after 10:39:48** (batch also FAILED 1:0).
**Commit** `d31770f`, dirty; runtime TFM pin `de324efcca55`, source
`4da9e67eb78b4e1a753b6fa8fe5580fb37b86c956df114db9e8879dab4677d8e`.
Primary-source inspection used read-only TFM pin `dd5091482404278f429e6f4b272d0314808f64d4`
and VSC pin `3a04134ab7786dc4158a389ac685acfe7b77b62b`; neither is evidence of a
different runtime pin in this job.
At the final check, other workspace activity had advanced the tracked library
pins to TFM `cbaff2f5fa0a6e910d7e7d1cd148437a3e0c217a` and VSC
`9e376b5ee9106e82d37eca86ca6acb861f580c93`; this audit did not modify either
submodule. The tested executable fingerprint remained unchanged.

### Configuration

- Same accepted CPU 62371055 receipt and environment fingerprint
  `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.
  TabPFN 9.1.0, pinned TabICL 2.1.1, CatBoost 1.2.10, torch 2.11.0+cu128;
  dependency/pip checks passed.
- Six 600-step diagnostic arms: scratch/released TabICL and continued TabPFN-3
  for PD/LGD. Each arm forcibly stops at step 101 and resumes. All use the
  synthetic 50% credit mixture; no production experiments were launched.
- Failing LGD arm: raw encoding, 1,024 synthetic rows, 1–100 features, batch 16,
  two TabPFN members, AdamW lr 1e-5, wd 1e-4, 10% warm-up/cosine, clip norm 1,
  FP16 AMP; upstream 9.1.0 CRPS+MSE objective and member target transforms.
  Checkpoint cadence 100; online monitoring every 3,200 tasks.
- Production/full-table validation retains all eligible rows/features and five folds.

### Results

- **47/66 stages PASS; stage 48 `resume-lgd-2` FAIL**. All **26/26 mandatory CUDA
  tests**, six smoke routes and both trainer-equivalence checks passed. The prior
  2,986-column streaming discrepancy is no longer present in its B200 test.
- Reference TabICLv2, TabPFN-3, CatBoost and linear models completed credit/OOD
  capacity checks. Credit logs show 20 successful folds/model on four PD tables
  (hackerearth, bank_status, loan_default, algorithmwatch) and 15/model on three
  LGD tables (heloc, loss2, base_modelisation). The 2,986-feature PD route passed.
- Five long arms completed step 600 after resuming at 101. All four TabICL arms
  reached 38,400 tasks each; TabPFN PD reached 9,600 with three AMP retries.
- TabPFN LGD resumed from 101, passed monitoring at 200 (lendingclub RMSE 0.1301,
  R2 0.7470; monitoring is not the publication benchmark), then failed at step 217.
  Sixteen backoffs reduced scale 128 to 0.001953125 without finite gradients.
  No successful update is counted for the failing batch. Last logged checkpoint
  is step 200; its summary still records the earlier deliberate stop at 101.
- No trained-checkpoint benchmark stages ran. The gate correctly blocks Exp1–3.

### Files and audit limits

- `Downloads/output CreditICL.zip`, 3,273,975 bytes, 373 files/442 entries,
  SHA-256 `8345cea09de1845bcc7969c9ba552fde54b688a5a2e7ace8e0c3a8de389141f1`;
  CRC valid. Attempt `gpu_20261020T112850Z_j11715564_4e63fa05`.
- CPU 23/23 artifact hashes match; the GPU's bound CPU receipt hash also matches.
  All 81 available GPU stage artifacts match. Forty referenced project-tier
  benchmark CSV/progress/receipt files and five recorded final checkpoints are outside the
  download; stage PASS records are available, but their numerical contents
  cannot be independently rechecked from this ZIP.
- Strict log scan finds the traceback only in `resume-lgd-2.log`. Scheduler
  and traceback establish a gradient failure, not walltime/OOM. No updated
  storage quota or directory-size output was supplied.

### Diagnosis and prepared fix

A local one-thread, PYTHONHASHSEED=0 replay of the same configuration and batch
index exposes task 6's safepower-transformed query magnitude **3.3905049e18**;
context/ordinary query targets are much smaller. A controlled upstream-loss
reproduction at that magnitude has finite loss **3.8576e34** and infinite FP16
gradients, versus finite float32 gradients. This establishes the failure mode;
the exact failed CUDA checkpoint remains on VSC and has not been replayed locally.
The pinned source is `TabPFN .txt`, symbols `_targets_in_estimator_space`,
`_compute_regression_loss`, `InferenceEngineBatchedNoPreprocessing.iter_outputs`.

The trainer now replays the entire unchanged batch once in float32 after bounded
loss-scale backoff, restoring RNG starting states and disabling both autocast owners. The
previously calibrated AMP scale is restored for the replay and following batches;
unsuccessful backoffs on an unrepresentable half-precision batch must not cause
underflow in subsequent ordinary batches. Persistent
failure still raises. The fallback's norm reduction uses float64 before applying
the same norm-1 clipping rule, preventing reduction overflow on large finite
gradients. Loss weights, target transforms, input tables and update budget are
unchanged. Replay counts are logged, summarized and restored from checkpoints.
Gradient telemetry now also reduces in float64: a direct probe with finite
1e30 gradients previously reported infinity instead of a norm of 1.4142e30.
Portable regressions cover the observed target magnitude, RNG/budget/scaler
state, persistent failure, autocast restoration and large finite norms. Two new
mandatory early CUDA checks cover the magnitude and real TabPFN replay/resume
paths before capacity/long-training stages.

All 22 recorded progress-curve points have finite headline metrics in the ZIP.
Local focused replay/resume tests initially passed 17/17. Two draft broad
invocations were deliberately stopped while completing the diagnostic-norm and
scale-recovery review; neither is counted as final verification.

Final local verification used fixed source
`f07e71e70eebd282b3401a9e745968ca47fc2edfec4821cc4608bf9eae21050a`:
**1,582 passed / 58 skipped**, zero failures, in **1,175.90s**. The skips comprise
29 Bash checks (Git Bash was installed but not on this process's PATH), 28 VSC
checks and one PDF-size check after output cleanup. With the existing Git Bash
enabled for the test process, the environment/Slurm/notebook-contract modules
passed **178 tests**, with only that PDF check skipped, in **28.94s**. This
recovers all 29 shell skips on unchanged source: **1,611 distinct local tests
passed**, without claiming a CUDA PASS. The extreme-target replay, retained
scale, next ordinary update and finite gradient reporting also passed a direct
focused follow-up. No package installation or VSC job was performed. Existing
notebooks/visualization sources were not changed; the fresh CPU job executes
the notebooks and generates the figures.

### Interpretation and next

CPU PASS certified the prior source; this training fix requires new CPU/GPU
receipts. Preserve the downloaded evidence, publish all prepared changes, and
use the guarded reset for the requested clean campaign before CPU then GPU.
Run the GPU-scoped reset before the full-output reset: the former uses existing
GPU attempt/log job IDs to find matching GPFS scratch directories; deleting
those identifiers first would leave the failed job's scratch copies behind.
Keep that revision/dependencies/inputs fixed throughout. A future DEFERRED run
can continue unchanged; a FAILED run must be diagnosed. Production remains gated.

## 20-10-2026 — Exp0 GPU 11715564 — submitted after corrected-source CPU PASS

**Submitted** 20-10-2026 around 09:21 CEST by Andreas; **job** 11715564, Mindwell.
The terminal records successful `--require-exp0-cpu` followed by `exp0.slurm`.
CPU receipt `cpu_20261019T131049Z_j62371055_f8ff58fe` was accepted as current.

### Configuration and results

- Current script requests one B200, 24 CPU cores, 180 GiB RAM and 24 hours;
  these are requested resources, not evidence of an allocated node.
- Current executable source is `4da9e67eb78b4e1a753b6fa8fe5580fb37b86c956df114db9e8879dab4677d8e`,
  local commit `d31770f`, library pin `de324efcca55be392feb0c01830b3b6ce364587f`.
- The 09:25 download contains no GPU attempt directory, main log or GPU receipt.
  Submission is confirmed; scheduler state, progress, elapsed runtime and outcome are unknown.

### Interpretation and next

Keep the accepted CPU receipt and freeze executable source, dependencies and inputs.
Inspect `sacct --clusters=mindwell -j 11715564`; do not reset or submit a duplicate.
GPU validation is still required before Experiments 1–3. A failed GPU result requires
diagnosis; a recorded DEFERRED continuation can reuse completed stages on unchanged source.

## 19-10-2026 — Exp0 CPU 62371055 — corrected source passed all 13 stages

**Audited** 20-10-2026 from `Downloads/output CreditICL.zip`.
**Job** 62371055, wICE `batch`, node m28c20n4; log reports 20 cores and 65,536 MiB RAM.
**Commit** `d31770f` (`dirty=YES` in log); executable fingerprint
`4da9e67eb78b4e1a753b6fa8fe5580fb37b86c956df114db9e8879dab4677d8e` matches the current checkout.
**tfm-library pin** `de324efcca55be392feb0c01830b3b6ce364587f` in the matching local source.
Main log 15:10:49–15:31:58 CEST (**21m09s**); this is not a scheduler billing total.

### Configuration

- CPU validation on protocol/cache v8: 21 credit datasets, 25 classification plus
  25 regression OOD tables, four released model weights; no production training.
- TabPFN 9.1.0, TabICL 2.1.1 at `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`,
  CatBoost 1.2.10, torch 2.11.0+cu128; 135 dependency checks and pip check passed.
- Environment fingerprint `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.

### Results

- **13/13 stages PASS**: preprocess 5s, preflight 3s, unit 741s, benchmark 37s,
  configs 2s, data 16s, determinism 63s, OOD 1s, orchestration 33s, paths 1s,
  prior 102s, storage 3s, notebooks 238s. Detailed check reports: 129 PASS, zero FAIL/WARN.
- Pytest **1,586 passed, one skipped, 26 deselected** in 738.06s. The skip is the
  sibling TabPFNCredit data-identity comparison, unavailable on VSC; the 26 GPU
  checks belong to the GPU phase. All eight new FP16/BF16 boundary regressions pass.
  The prior local setup-guard timeout did not recur: all 16 environment tests passed.
- **11/11 notebooks executed**, producing 125 PDFs and 11 figure manifests.
  Untrained experiment plots are placeholders, not production results.
  Notebook process-tree RSS sampled at 35.75 GiB, below the 64 GiB request.
- Canonical `experiment_0/report/cpu_validation.json` is PASS; **23/23 artifact
  SHA-256s verified** against the archive. Main log ends `END status=OK`.

### Output inventory and anomalies

- ZIP SHA-256 `045f65b49e31a8103b82d4ea9a87b497015c0f7a8e7cc60b5390f1ca72bd9cee`:
  213 files, 2,679,280 compressed bytes, 3,908,396 uncompressed bytes;
  CRC valid, no duplicate or unsafe paths.
- One previous CPU attempt, 62369106, remains: 33 attempt files plus its main and
  preprocessing logs, **35 files / 548,545 bytes**. It is isolated from the current
  canonical receipt. No previous GPU attempt or production checkpoint/result tree
  is present in this ZIP. Generated notebook figures/summaries are expected output.
- At 19-10 15:27 CEST, the storage check recorded **zero files** under the current
  project-tier output root. This does not inventory legacy paths elsewhere on VSC.
  The local checkout's output tree contains only its README; the downloaded ZIP
  was preserved and was not extracted into that tree.
- Shared Lustre project `stg_00211` reported **147,529/150,000 files** (2,471 slots
  remaining), despite using only 253.1 GiB of 1 TiB. This is a shared quota,
  not proof that CreditICL owns all those files. Review before larger campaigns.
- No new CPU failure was found. Deprecation and group-lookup warnings remain;
  passing CPU checks do not establish CUDA equivalence or full-table GPU capacity.

### Interpretation and next

The corrected code now has a valid VSC CPU result. GPU 11715564 is the next required
validation; no further CPU rerun or output reset is indicated. Keep this campaign
unchanged and inspect the GPU outcome before launching production experiments.
Remove only the isolated old CPU files later if a single-attempt archive is desired;
do not erase the current CPU evidence or active GPU output.

## 19-10-2026 — Exp0 GPU 11714993 — early streaming equivalence failure

**Job** 11714993, Mindwell `gpu_b200`, node r11g17, one NVIDIA B200 (183,359 MiB).
**Commit** `b351325` (`dirty=YES` in log); executable fingerprint
`6f2f530d982a95f9c03b352a44de80e2ede28b1c84952587fa544744f9350857`.
**tfm-library pin** `de324efcca55be392feb0c01830b3b6ce364587f` in the matching local source.
Main log 13:57:03–13:59:21 CEST (**2m18s**); allocation log began 13:56:37.
The archive establishes exit 1, not a live scheduler state or billing total.

### Configuration

- `scripts/slurm/exp0.slurm`: 24 CPU cores, 180 GiB RAM, one B200, 24-hour request.
- TabPFN 9.1.0, TabICL 2.1.1 at `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`,
  CatBoost 1.2.10, torch 2.11.0+cu128; 135 requirement checks and pip check passed.
- Same input inventory/environment as CPU 62369106; GPU's CPU receipt hash matches exactly.
- Validation has 66 stages; its mandatory CUDA suite uses `-x` before long training.
  The failing probe uses released TabICL PD weights, all eight ensemble members,
  32 context rows plus eight queries, 2,986 features, seed 11 and deterministic probe data.
  This diagnostic is not a five-fold production benchmark.

### Results

- Preflight passed; CUDA suite: **16 passed, 1 failed** in 123.43s (stage 128.03s).
- Passed: CUDA/backprop, RNG preservation, TabPFN PD/LGD checkpoint resume, storage,
  regression loss/quantiles, AMP retry, offline inputs, both model families at full
  PD/LGD widths and TabICL PD/LGD disk-offload equivalence.
- Failed: `test_tabicl_streaming_preserves_full_width_ensemble_predictions[pd-2986]`.
  Three of 16 probabilities exceeded unchanged `rtol=0.002, atol=0.0002`;
  maximum absolute discrepancy **0.00238492**, relative discrepancy **0.00992643**.
- No six-arm 600-step stop/resume runs, full-table capacity benchmarks or production
  experiments started. Process-tree host RSS sampled at 4.25 GiB; no OOM in this evidence.
- Attempt: `gpu_20261019T115703Z_j11714993_1f9c1620`, log `logs/vsc.log`;
  canonical `experiment_0/report/validation.json` correctly says FAILED.

### Bugs and anomalies

The local streaming code widened the column AMP result to `X.dtype` (float32)
before row attention. Upstream `InferenceManager.__call__` returns the computation
result directly for a single GPU batch, retaining half precision; buffer dtype is
not a universal compute rule. Changing that boundary changes CLS-token rounding,
normalization and residual arithmetic. Source checked in installed pinned TabICL
and `tfm-library/repositories/TabICL.txt`, symbols `InferenceManager.__call__`,
`InferenceManager._run_forward`, `RowInteraction._inference_forward`.

Explicit CPU FP16/BF16 tests reproduced six boundary failures (binary, regression,
many-class); the affine cases already retained float32 naturally. Removing the
conversion makes all eight new cases pass, with the original numeric tolerance.
A separate released-weight CPU FP16 probe (101 features, eight members) reduced
max probability difference from 0.0007531047 to 0.000371635; it is not a B200 replay.
All 39 focused memory/streaming tests pass. Exact B200 confirmation is still required.

Final local verification used unchanged executable fingerprint
`4da9e67eb78b4e1a753b6fa8fe5580fb37b86c956df114db9e8879dab4677d8e`.
The complete suite ran in 2,841.61s: **1,605 passed, 27 skipped, one failure**.
That failure was the setup-script guard's Git Bash subprocess exceeding its
20-second deadline, not a failing guard assertion. The unchanged case passed alone
in 2.61s; its entire environment module subsequently passed **16/16 in 1.97s**.
The exact timeout cause is unconfirmed; no checks or deadlines were weakened.
The skips are 26 VSC-only tests and the PDF-size check after local output cleanup.
All 115 focused validation/environment/readiness tests also passed before the full
run. A released-weight CPU FP16 check with eight-member stock batching versus
one-member streaming passed the original tolerance at 101 features. These checks
do not replace the full-width B200 probe. The archive SHA-256 remained unchanged.

The Python module-load warning did not prevent the venv or CUDA checks from passing.
Deprecation warnings obscured the assertion in the main log tail; reporting now
preserves bounded JUnit assertion text and readiness names the failed stage/test/log.

### Interpretation and next

The validation gate stopped early as designed; this was not another day-long run.
There is a confirmed precision bug, but no local CUDA device with which to claim the
recorded GPU discrepancy is fully resolved. Keep the strict full-width test and all
rows/features/five-fold benchmark settings. Publish fixes, reset failed GPU output
with the guarded utility (the archive is preserved), rerun CPU then GPU with fixed
source and environment. Production stays blocked until both receipts pass.

## 19-10-2026 — Exp0 CPU 62369106 — all 13 stages passed

**Job** 62369106, wICE `batch`, node l33c22n2; log reports 20 CPU cores and 65,536 MiB.
**Commit** `b351325`, executable source `6f2f530d982a95f9c03b352a44de80e2ede28b1c84952587fa544744f9350857`,
same library pin as the GPU entry above. Main log 13:10:23–13:45:33 CEST (**35m10s**).

### Configuration and results

- CPU validation, protocol/cache v8, 21 credit datasets and 25+25 native OOD tasks;
  full input/weight inventory and dependencies passed. No trained arms in this phase.
- **13/13 stages passed**: preprocess 85s, preflight 3s, unit 805s, benchmark 152s,
  configs 2s, data 154s, determinism 60s, OOD 1s, orchestration 31s, paths 1s,
  prior 99s, storage 3s, notebooks 693s.
- Pytest: **1,575 passed, 2 skipped**, 26 GPU tests deselected. The skips concern
  absent sibling TabPFNCredit data and no PDFs before notebooks execute.
- **11/11 notebooks executed, 125 figures**. These include empty-result placeholders;
  notebook success is not evidence of completed Experiment 1–3 results.
- Notebook process-tree RSS sampled up to 39.20 GiB within the 64 GiB request.
- Attempt `cpu_20261019T111023Z_j62369106_5e8158f1`; all **23 artifact hashes verified**.

### Bugs, interpretation and next

No failing CPU stages or new data/package errors in this archive. CPU-default AMP
was off, so these passes did not cover the precision issue exposed by the GPU.
The new explicit AMP regressions now run in this phase. Source/test changes require
renewed validation; no package reinstall or dataset deletion is indicated.

**Evidence for both jobs:** `Downloads/output CreditICL.zip`, modified 19-10-2026
14:03:40, 2,615,060 bytes, SHA-256
`be35eae7f31ca5d8ae1b797470c63bb6ab01d1004ba22815f4017381619ef317`.
All 217 ZIP entries pass CRC checks. Inspected in place; nothing extracted, reset
or deleted. GPU preflight artifact hash also matches; the failed test's log/XML
are retained in the archive, not falsely certified as successful artifacts.

## 19-10-2026 — Follow-up download of GPU 11692660: incomplete validation despite an OK shell footer

**Evidence:** `Downloads/output CreditICL.zip`, downloaded 19-10-2026 09:05:57 CEST,
SHA-256 `cd6f0481ea5052bc8a24d471b6a1e166c274df8c2575d2bcb9693d3739cb194d`.
CRC passes: 107 files, 43 directories, 1,400,107 uncompressed bytes. Inspected directly;
the archive was not extracted, modified or deleted. The download date is not the run date.

### Configuration and results

The newest attempts remain CPU **62324362** and GPU **11692660**, both started on
15-10-2026 with commit `cf0e539` and executable fingerprint
`adf308f6490eee2717ac9cf9ac87312376e0ef2bd89b8e2e785d8b9cfc347acb`.
Their configuration, packages and library pin are recorded in the 16-10 entry below.
All 22 CPU artifact hashes still match; the GPU-to-CPU receipt hash and input inventories
also match. CPU passed 12/12 stages, 1,437 unit tests (three skipped). GPU has 12/66
completed stages, including 17 CUDA checks, six smoke-training routes and both equivalence
checks. Its interrupted benchmark still has 16/20 completed fold messages: five each for
hackerearth, bank_status and loan_default, one for algorithmwatch. There are no new
tracebacks, failed model checks or nonfinite prediction reports in this download.

### Newly observed shutdown defect

The main GPU log now ends `END status=OK - Fri Oct 16 09:17:46 CEST 2026`, while
`validation.json` remains schema 2 / RUNNING and the benchmark log ends at 05:33:45
during AlgorithmWatch fold 1. This is not a successful validation. The supplied 19-10
sacct follow-up confirms **CANCELLED by 2538338**, elapsed **22:36:01**; batch step
CANCELLED, exit **0:15**, elapsed **22:36:10**. The parent job's 0:0 exit field therefore
does not establish success. The receipt is stale, not a job still running on 19 October.
No completed benchmark CSVs or diagnostic-training summaries were supplied.

The old shell printed OK solely from `$?` in its EXIT trap. It had no signal forwarding
or independent record that the validator had returned successfully. A shell interruption
could therefore print OK while the controller left a RUNNING receipt. The production
gate correctly refused that incomplete evidence, so Experiments 1–3 remain blocked.

### Fixes and interpretation

CPU/GPU launchers now share explicit signal forwarding, wait for validator termination,
and require a successful validator return before logging OK or cleaning scratch. Caught
controller termination publishes DEFERRED after stopping the child process tree; hard
SIGKILL remains necessarily uncatchable and cannot unlock training. Readiness and the
download verifier explicitly distinguish stored RUNNING from live scheduler status.

The streaming, per-cell persistence and continuation changes prepared on 16-10 remain
uncommitted in the local checkout at this inspection. VSC `git log` also confirms
`cf0e539` on main/origin/main and the old `adf308f...` source fingerprint. Its dirty
entries are untracked sentinels, environment, archive and output files; they are not the
new executable changes. Nothing in this archive tests the new adapter. Publish the reviewed local
changes first, verify the same executable fingerprint on VSC, then perform the requested
clean CPU/GPU campaign. Keep the archived evidence. Full rows, all eligible features and
five folds remain unchanged; B200 speed and complete GPU validation remain unmeasured.

**Local verification on 19-10:** focused lifecycle/readiness tests 160 passed; complete
suite **1,519 passed, 21 skipped, 508 warnings in 1,000.54 seconds**. Nineteen skips are
mandatory VSC/CUDA tests; two cover absent generated figures and an upstream-absent branch
when upstream is installed. The 24 additional test cases include actual Bash TERM/INT/HUP
delivery before/during validation, process cleanup, interrupted-receipt continuation,
old-status diagnostics and rejection of a false shell OK. Executable fingerprint:
`4348d3f6b1aab916ba94dca873009eb38a30bed4af8592a2b182ec092a133be1`.
No package changes, Git commit/push, cluster submissions or output deletion were performed.

## 16-10-2026 — Exp0 CPU 62324362 passed; GPU 11692660 cannot finish its full-width probes within 24 hours

**Evidence:** `Downloads/output CreditICL.zip`, modified 16-10-2026 08:59:17 CEST,
SHA-256 `06e7f519c1de9d07c754a0752c253d1dd3ca0c23e974fe269afea326234b0649`.
CRC passed; 107 files / 1,399,975 uncompressed bytes. Read directly, without extraction
or cleanup. This is the small output tier: no final benchmark CSVs or checkpoint binaries
were supplied. User accounting at 09:01 confirms GPU job RUNNING, elapsed 22:20:16,
limit 1 day, exit field 0:0; no MaxRSS was reported. Its final outcome is not yet known.

### Identity and configuration

- Both runs record commit `cf0e539`, dirty checkout. Executable fingerprint
  `adf308f6490eee2717ac9cf9ac87312376e0ef2bd89b8e2e785d8b9cfc347acb`
  matches the local checkout inspected on 16-10-2026. Dirty status alone is not evidence
  of different executable code; the identities of the dirty files were not supplied.
- Environment fingerprint `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`
  matches between phases: TabPFN 9.1.0, TabICL 2.1.1 at
  `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10, torch 2.11.0+cu128.
  All 135 requirements and `pip check` passed.
- Input inventories match; all 22 CPU artifact hashes match ZIP contents. The GPU receipt
  binds to CPU receipt SHA-256 `f04e918e92abf8e99b27aa0b1b216c01027c8b437eb43d0c997891b41b1e473d`.
- Read-only library pin `de324efcca55be392feb0c01830b3b6ce364587f`.
  Relevant official symbols in `tfm-library/repositories/TabICL.txt` and installed pinned
  source: `ColEmbedding._inference_with_feature_group`, `RowInteraction._aggregate_embeddings`,
  `InferenceManager.__call__`, `DiskTensor`, `TabICLClassifier._batch_forward`.

### CPU 62324362 — PASS

- wICE/batch, node `l33c20n3`; allocation banner 20 CPUs / 65,536 MiB.
  Attempt `cpu_20261015T073645Z_j62324362_2c36467d`.
- Log interval 15-10-2026 09:36:45–09:53:16 CEST, 16m31s (not scheduler accounting).
  All 12 stages passed; unit suite **1,437 passed, 3 skipped, 17 deselected** in 716.27s.
  Skips: absent sibling TabPFNCredit checkout, absent generated figures, and an upstream-absent
  error branch in an environment where upstream is installed. CUDA tests were deselected here.
- Benchmark contracts, configurations, 21 credit datasets, OOD cache, deterministic priors,
  orchestration, paths and storage checks passed.

### GPU 11692660 — RUNNING at the supplied snapshot, capacity problem

- Mindwell `gpu_b200`, node `r11g22`, one B200, 24 CPUs / 184,320 MiB host RAM.
  Attempt `gpu_20261015T084304Z_j11692660_c29a3510`.
- Allocation started 15-10-2026 10:41:39 CEST; declared end 16-10-2026 10:41:39.
  Validator began about 10:43:04 with its own 23.5-hour budget, so its timeout is expected
  around 16-10-2026 10:13. These are deadlines, not successful completion estimates.
- **12 of 66 stages passed**, in approximately 29 minutes. These include all **17 mandatory
  CUDA tests**, all six 10-step training/checkpoint-to-prediction routes, weight/environment
  checks and both 20-step upstream equivalence checks. CUDA tests include TabPFN PD/LGD
  resume and forced disk-offload equivalence. The six longer stop/resume runs remain untested
  in this attempt, as do all subsequent capacity/benchmark stages.
- Equivalence maximum loss/parameter differences: PD 6.63e-6 / 2.32e-6;
  LGD 3.11e-7 / 8.33e-6, within the configured tolerances.
- Stage 13, `credit-pd-3`, is released TabICL on four PD tables, five folds, one seed,
  protocol 7, with no row/context/feature cap. Three tables completed all five folds:
  hackerearth (532,428 rows / 30 eligible features), bank_status (100,000 / 15), and
  loan_default (105,471 / 759). The previous job's 759-feature OOM did not recur.
  Their logged AUC ranges are respectively 0.8445–0.8477, 0.6346–0.6488, 0.6897–0.7107.
- Algorithmwatch has 158,700 rows / 2,986 features. It began at 15-10 11:35:20;
  fold 0 finished at 16-10 01:09:18 (**13h33m58s**), AUC 0.6548, F1 0.5702,
  threshold 0.283, Brier 0.2206. Its inner/outer fits took 112.3s / 137.7s;
  almost all the elapsed time was in prediction (plus threshold selection), not fitting.
  Fold 1's outer fit finished at 05:33:45; both the ZIP and 09:01 live tail end there.
  Thus 16/20 cells in this one stage have completed; this is not overall completion.
- No new traceback, OOM, nonfinite prediction or failed stage was observed. AMP overflow
  retries recovered in the TabPFN smoke checks. These partial checks do not prove the entire
  suite correct or ready for production, and the ZIP cannot verify final prediction arrays.

### Confirmed implementation gaps and interpretation

- The full-width reference workload does not fit the single allocation at observed speed.
  Extrapolating only the first wide-table fold gives about 68 hours for its five folds alone;
  other stages remain, so no defensible completion time for the entire suite is available.
- The large table enters the expensive memory-bounded inference route. Column embeddings
  are laid out by feature and transposed for row attention; disk reads and full-width
  attention are plausible contributors. There is no stage-level GPU/I/O profile in the
  download to attribute the hours precisely; do not present disk thrashing as measured.
  The released checkpoint config confirms 128-dimensional embeddings, four CLS tokens and
  `col_feature_group="same"` (one group per feature). Float32 embedding storage is about
  181 GiB for the inner prediction and 226 GiB for the outer prediction per ensemble member,
  exceeding this B200's roughly 178 GiB. Eight members are evaluated sequentially.
- `src/eval/tabicl_memory.py` logs through `logging.getLogger(__name__)`, while
  `setup_logging` configures the non-propagating `crediticl` logger. INFO messages about
  offload decisions/temp storage are consequently absent from the normal run log. A local
  in-memory capture reproduced this: the control project message appeared, the memory
  manager message did not. No files or execution settings were changed by the probe.
- `evaluate_dataset` and `run` accumulate rows in memory; `scripts/evaluate.py` writes the
  results CSV only after `run` returns. A timeout loses the structured results for completed
  cells in this stage, although their headline log messages survive. `run_validation`
  creates a fresh attempt without stage/fold continuation, so unchanged resubmission repeats work.
- The supplied `sstat --clusters=mindwell` command was invalid and returned no resource
  evidence. Do not infer low GPU use or measured RAM use from that failed command.

### Next

Recommend stopping the still-running attempt because it cannot complete its remaining work
before its deadlines; no cancellation was performed by the assistant. Preserve this evidence.
Before another allocation, implement and verify durable per-fold results/continuation, expose
inference phase timings/storage, and address the full-width capacity path without reducing rows,
features, folds or ensemble members. Revised executable code needs fresh validation receipts.
Experiment 1 remains blocked. This audit changes existing run records only; it does not claim
that the newly identified gaps have been fixed or that the GPU validation has passed.

## 14-10-2026 — Exp0 CPU 62319803 passed; GPU 11689448 exhausted host RAM in the full-width benchmark

**Evidence:** `Downloads/output CreditICL.zip`, modified 18:47:19 CEST, SHA-256
`a7e2e882e9881ed2f29a6326841265d0b84024d6a087225bc4981f58647bcbd0`.
ZIP CRC passed; 73 files / 927,899 uncompressed bytes. Read directly without extracting or
deleting previous evidence. It contains the small output tier, not the benchmark CSVs or
checkpoint binaries on project storage. Finite fold metrics below are verified from logs,
not an audit of final prediction arrays or a completed benchmark receipt.

### Identity and configuration

- Both banners record `351d948`, dirty checkout. Their executable fingerprint
  `5bb68e06e4a5c55cfc08867e4a86e8111405cdd7d90189e2add291fb05dd856f`
  matched the clean local checkout before this audit's changes. Unspecified dirty files
  therefore do not establish a code mismatch or explain the failure.
- Both environments: TabPFN 9.1.0, TabICL 2.1.1 at
  `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10, torch 2.11.0+cu128.
  All 135 dependency checks and `pip check` passed. Environment fingerprint:
  `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.
- CPU/GPU input inventories match. The GPU receipt binds to the exact CPU receipt;
  all 22 CPU artifact hashes match the downloaded bytes.
- Local read-only library pin: `de324efcca55be392feb0c01830b3b6ce364587f`.
  Official code evidence: `tfm-library/repositories/TabICL.txt`, symbols
  `InferenceManager._resolve_offload_mode`, `InferenceManager._allocate_output_buffer`,
  `ColEmbedding._inference_without_feature_group`, plus the installed pinned source.
  The library's newer cgroup-aware helper is absent from the installed commit; merely
  reading/updating the library does not update the installed inference implementation.

### CPU 62319803 — PASS

- wICE/batch, `m28c22n1`, allocation banner 20 CPUs / 65,536 MiB.
  Attempt `cpu_20261014T154239Z_j62319803_27c8e2cf`.
  Main log 17:42:39–17:58:41 CEST: **16m02s**, not a supplied scheduler elapsed time.
- All 12 stages passed. Unit JUnit: 1,407 tests, **1,404 passed / 3 skipped**, zero
  failures/errors. Unit stage 687.05s. Skips: unavailable sibling TabPFNCredit,
  absent generated figures, and an inapplicable missing-upstream error path.
- All 21 credit datasets and 50 OOD tables validated. Benchmark split checks passed
  71 assertions; configurations 10, data 21, determinism 7, OOD 1, orchestration 4,
  paths 6, prior 6, storage 3. This is validation of inputs and CPU behavior, not
  evidence that every full-table GPU prediction fits the allocation.

### GPU 11689448 — FAILED during capacity validation

- Mindwell/gpu_b200, `r11g22`, one B200, 24 CPUs / 184,320 MiB host RAM.
  Attempt `gpu_20261014T160518Z_j11689448_8352ce02`.
  Main log 18:05:17–18:46:26 CEST: **41m09s**. Accounting supplied later at 19:37 CEST
  gives **41m32s**, approximately **0.692 GPU-hours allocated**, state `OUT_OF_MEMORY`,
  exit `0:125`, requested RAM `180G`, batch MaxRSS **188,735,484 KiB (~180 GiB)**.
  The `.err` supplied at 19:38 confirms one `oom_kill` in the batch step at 18:46:26.
  Actual charged credits have not been supplied.
- First **12 stages passed**: preflight; all **15 mandatory CUDA tests** with zero skips;
  environment and weights; all six ten-step training/checkpoint-to-prediction probes;
  both 20-step upstream training-equivalence checks.
- Real CUDA TabPFN PD and LGD checkpoint resume passed. This verifies the earlier
  Adam CPU/CUDA placement fix on the short probes. The six 600-step stop/resume runs
  were not reached. No production Experiment 1 training ran in this attempt.
- Equivalence maximum loss/parameter differences: PD 4.71e-6 / 3.18e-6;
  LGD 2.44e-7 / 1.97e-6. All smoke checkpoint tensors matched and predictions were finite.
- Stage `credit-pd-3` (released TabICL) exited **-9, SIGKILL**, after 718.06s. No Python
  traceback. The ZIP alone did not identify the external killer. The later `sacct`
  and `.err` evidence confirms Slurm host-memory exhaustion, rather than a CUDA exception.
- Five folds each finished with finite metrics on `0006.hackerearth`: 532,428 rows,
  30 eligible features, AUC .8445/.8447/.8475/.8477/.8464; and `0009.bank_status`:
  100,000 rows, 15 eligible features, AUC .6403/.6423/.6454/.6488/.6346.
  All training-fold rows were used. Removed columns were outcome/future-information exclusions.
- The next table, `0011.loan_default`, has 105,471 rows / 759 features. Its first inner
  PD fit used **67,500/67,500 rows x 759 features** (18.2s). Last logged activity was
  18:46:00, before prediction finished. `0014.algorithmwatch` (158,700 x 2,986) and
  all later capacity/training/benchmark stages were not reached.

### Diagnosis and next

The pinned TabICL manager sees total available node RAM rather than the Slurm limit;
the previous wrapper did not configure disk offloading. In addition, the classifier's
feature-shuffle shortcut calls `repeat` on the whole embedding tensor, defeating disk
backing even for a one-member batch. These are confirmed memory hazards; the later
scheduler accounting independently confirms that the job exceeded its host RAM limit.

The project adapter now keeps all eight ensemble members but executes them one at a
time, bounds column transfers, uses GPU or temporary disk storage, rejects whole-output
CPU/swap fallback and avoids the redundant single-member embedding copy. Linux disk
blocks are reserved before mmap writes to catch quota failures as exceptions. Mindwell
uses its job GPFS directory; temporary files are scoped to each prediction. No model
weights, preprocessing, benchmark row/feature policy, folds or ensemble count changed.
Released and trained TabICL share the adapter. No dependency or library files were edited.

Local tests compare PD probabilities and LGD mean/quantiles against all eight released
ensemble members and exercise real mmap writes, allocation failures, quota errors and
cleanup. Two new mandatory CUDA probes force disk offloading before the large-table
checks. Full-size behavior still requires a fresh CPU PASS followed by GPU PASS with
this changed source. Experiment 1 remains blocked. After the OOM diagnosis, Andreas
authorized removing stale Exp0 GPU output and rerunning. The GPU-only reset removes
recognized GPU attempts, results/checkpoints and logs across both output tiers and
their identified GPFS job directories, while preserving CPU records and permanent inputs.
It lists by default and refuses deletion while CreditICL jobs are queued/running.

**Local verification after the audit:** full suite **1,429 passed / 32 skipped**, 508 warnings,
923.18s. Thirteen skips were Bash missing from this tool shell's PATH; rerunning all three
affected modules with the already-installed Git Bash on PATH gave **127 passed / 1 expected
figure skip**, covering all thirteen. Thus **1,442 distinct local tests passed**, with 17
VSC-only tests and two expected non-GPU skips (absent figures and the missing-upstream path
when upstream is installed). The final GPU-allocation retry/periodic-flush adjustment also
passed all **17 memory regression cases**. Real CUDA forced-disk tests and the full-size
benchmark have not run on the revised code. No package installation, cluster submission,
commit, push, or deletion of real output occurred during this audit.

**Authorized cleanup after diagnosis:** the local GPU reset found zero generated files
in the repository output tree. The downloaded ZIP was then rewritten atomically to remove
its 34 GPU files, preserving all 39 remaining files (including every CPU record) byte-for-byte;
CRC and retained-content hashes were verified before replacing it. Its new SHA-256 is
`bd163a3cb5bbf70dd5f8e1f954964acc6e9d7afb8877dcbf4d1dcb378e5b4f47`.
The original archive hash above describes the evidence before this user-authorized cleanup.
No VSC deletion or new submission has been executed locally; the VSC cleanup is supplied
as `python -m src.utils.clean_run --reset-exp0-gpu --clean` after publishing/pulling the fix.
Final local verification with Git Bash on PATH: **1,458 passed / 19 skipped**, 508 warnings,
935.43s. Skips: 17 VSC-only checks, absent generated figures and the inapplicable missing-upstream
path. All 22 preserved CPU artifact hashes were rechecked against the original CPU receipt.

## 14-10-2026 — 17:11 reset refused before a new Exp0 submission

**Evidence:** Andreas's pasted login-node terminal output; `git pull --ff-only` advanced
`6054c50` to `16a8fd4`. Environment activation succeeded. This was a cleanup attempt, not a
compute job: the `&&` chain stopped before `sbatch`.

`python -m src.utils.clean_run --reset-output --clean` planned **186 generated files / 1.524 GB**
across `/vsc-hard-mounts/leuven-data/383/vsc38338/CreditICL/output CreditICL` and
`/lustre1/project/stg_00211/CreditICL/output CreditICL`. It then refused the root Slurm log
`/data/leuven/383/vsc38338/CreditICL/slurm-crediticl-exp0-cpu-62313072.out` during whole-plan
validation, before any unlink. No new CPU/GPU allocation or training resulted.

The root-log discovery stored an unresolved `/data/...` parent, but revalidation compared it
against a file resolved through `/vsc-hard-mounts/...`. This was a false containment failure,
not evidence of a concurrently changing log. Local tests using real filesystem directory aliases
reproduced it and the related duplicate-log count when two tiers alias the same project.
Resolve and deduplicate project parents before planning; keep active-job, input-overlap,
linked-output and file-change refusals. After the fix is published, repeat the existing reset
and CPU submission commands; GPU validation still follows a current CPU PASS.

**Local verification:** 111 focused tests passed; full suite **1,425 passed / 17 expected skips**
in 15m13s (15 VSC-only tests, absent notebook figures and an inapplicable missing-upstream path).
All six new filesystem regression cases passed; no real output was deleted by these tests.

## 14-10-2026 — Exp0 CPU 62316841: two tests read the actual failed GPU receipt

**Evidence:** Andreas's terminal attachment `fb75bab8-4986-42e0-8250-ef9d33060fbd/Pasted text.txt`
and the main/child logs pasted at 15:55 CEST. No new GPU job was submitted: readiness blocked it.
**Identity:** `6054c50`, dirty checkout (unspecified local changes), wICE/batch, node `m28c27n1`.
Local library pin `de324efcca55be392feb0c01830b3b6ce364587f`; the remote banner does not record its pin.

### Configuration and results

- CPU allocation banner: 20 CPUs and 65,536 MiB. The committed script requests 8 CPUs;
  the supplied banner is authoritative for this allocation. No training hyperparameters were exercised as production runs.
- Attempt `cpu_20261014T132749Z_j62316841_2e5d7c50`; main log
  `experiment_0/logs/exp0_cpu_20261014_152728_j62316841.log`.
  Banner 15:27:49, END 15:44:09 CEST: 16m20s log interval, not a supplied scheduler elapsed time.
- TabPFN 9.1.0, TabICL 2.1.1 at `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`,
  CatBoost 1.2.10, torch 2.11.0+cu128: all package checks passed, including 135 dependency versions and `pip check`.
- **11/12 stages passed:** preprocessing, preflight, benchmark partitions, configurations, data,
  determinism, OOD cache, orchestration, paths, prior and storage. Unit tests failed.
- Unit stage: **1,368 passed, 2 failed, 3 skipped, 13 deselected, 411 warnings in 703.39s**.
  Skips were the unavailable sibling TabPFNCredit checkout, absent generated figures, and
  the unavailable-upstream error path when upstream is installed. CUDA checks were deselected on CPU.
- Failing tests: `test_package_change_invalidates_successful_exp0_checks` and
  `test_partial_or_stale_exp0_checks_cannot_unlock_paid_training`.

### Diagnosis and next

Both fixtures created a temporary legacy `checks.json`, but only redirected `experiment_dir`.
The modern canonical receipt uses `outputs_dir`, so the tests read the real failed GPU receipt
from job 11688276 instead of their fixture. An isolated local reproduction with that outer failed
receipt produced the same two assertion failures. Using the existing fixture that isolates both
output tiers made the same reproduction pass (2/2). No production readiness check was bypassed.

This CPU failure is not evidence of another model failure, and does not validate the earlier GPU
resume fix. New local regressions cover restart settings, first walltime continuation, prior/model
random-state separation, uneven micro-batches, OOD fold memory release and inner PD validation
splits. Mandatory GPU tests include real PD/LGD TabPFN resume and CUDA random-state isolation;
the main job log now prints stage error tails and failing test names. Nonfinite PD monitoring
predictions now count as dataset errors, matching LGD, so the validation gate detects them;
the monitor itself still catches the error and allows training to continue.

The real-data partition audit passed all 21 local credit tables (five outer folds and the
inner PD threshold splits). Local dependency checks passed for 123 versions plus `pip check`.
The final complete local suite passed **1,419 tests, 17 skipped, 508 warnings in 950.81s**:
15 VSC-only checks, the absent-figures check, and the unavailable-upstream error case were skipped.
Critical static checks and `git diff --check` passed. No notebook or visualization code changed.
The 50-table OOD cache and CUDA are unavailable locally; their real checks remain mandatory on VSC.
Exp0 retains 12 CPU and 66 GPU stages; the existing full-table reference probes now precede
the longer training runs to detect capacity failures sooner.

At Andreas's request, after diagnosis and local verification, clear generated output on both
storage tiers and start a fresh CPU Exp0 followed by GPU Exp0. Keep datasets and released weights.
No cluster job, remote deletion, commit or push was performed in this debugging session.

## 14-10-2026 — Exp0 CPU 62313072 passed; GPU 11688276 failed after 01:58:32

**CPU:** submitted by 11:25 CEST, wICE, job `62313072`; PASS confirmed by 11:55 CEST.
**GPU:** submitted by 11:55 CEST, Mindwell, job `11688276`; FAILED 1:0 after 01:58:32,
confirmed by scheduler output supplied at 14:53 CEST.
**Code identity:** GPU banner reports `57d36cd`, dirty checkout (uncommitted changes unspecified).
The local checkout has the same HEAD and library pin `de324efcca55be392feb0c01830b3b6ce364587f`;
the supplied GPU banner does not separately identify the remote submodule pin.

### Configuration

- CPU command: `sbatch --clusters=wice scripts/slurm/exp0_cpu.slurm`. Current script requests
  8 CPUs, 64 GB and 4 hours; this transcript does not contain the allocation banner or elapsed time.
- GPU command: `python -m src.utils.preflight --require-exp0-cpu && sbatch --clusters=mindwell scripts/slurm/exp0.slurm`.
  GPU banner confirms one B200, 24 CPUs, 180 GB, node `r11g22`; Slurm confirms a 24-hour limit. Six short training/resume routes
  and selected full-table benchmarks are validation work, not production Exp1 results.
- Installed versions confirmed: TabPFN 9.1.0, TabICL 2.1.1 at
  `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10 and torch 2.11.0+cu128.

### Results and evidence

- All 135 dependency checks and `pip check` passed.
- All 21 processed credit datasets are version 7; all 50 pinned OOD tables and all four
  released model checkpoints are present.
- CPU receipt `cpu_20261014T092548Z_j62313072_c6ec3745` passed `readiness("cpu")`:
  every mandatory stage passed, with matching source, environment, inputs and artifact hashes.
  The excerpt does not provide per-stage timings or the detailed pytest counts.
- Slurm accepted GPU job `11688276`. Follow-up `sacct` reports the job and batch step
  FAILED with exit 1:0 after 01:58:32, below the 1-day limit. The extern helper's
  COMPLETED status does not establish successful GPU validation.
- GPU attempt: `gpu_20261014T095557Z_j11688276_9801e7fc`; main log
  `experiment_0/logs/exp0_gpu_20261014_115549_j11688276.log`, banner 11:55:57 and END 13:54:09 CEST.
- **17 stages passed:** preflight; 11/11 mandatory VSC tests (zero skipped); environment; weights;
  all six 10-update training/checkpoint/prediction smoke tests; both TabICL trainer equivalence
  checks; PD scratch/continued TabICL stop at 101 and resume to 600; TabPFN PD stop at 101.
- **Failure:** `resume-pd-2`, TabPFN PD, resumed step 101 at 13:51:14 then failed at the first
  `GradScaler.step` -> Adam `_multi_tensor_adam` / `_foreach_lerp_` with CPU/CUDA tensor mismatch.
  The LGD longer stop/resume runs and all full-table benchmarks were not reached.
- Weights/checkpoint checks matched every tensor (TabICL PD 391/LGD 347; TabPFN PD 510/LGD 504)
  and produced finite predictions. PD's one AMP retry and LGD's seven retries recovered and
  passed; grep hits on `0 FAILED` are successful reports, not additional failures.
- The Python module-loading warning did not prevent the venv, package audit or CUDA tests
  from passing. It is not the cause identified by this traceback.
- Confirmed allocation: approximately 1.976 B200 GPU-hours. The supplied `sam-balance` snapshot
  does not establish credits charged for this particular job.
- Evidence: Andreas's 14 October terminal attachment
  `28e6c50f-a5ae-44c5-b6bf-dfd5c465d7bb/Pasted text.txt`, 14:53 scheduler follow-up, and
  subsequently pasted main/child logs retrieved at 14:54–14:56 CEST.

### Interpretation and next

CPU validation passed for the submitted code/environment; GPU validation found a TabPFN
resume bug. In installed TabPFN 9.1.0, `estimator_to_device` configures devices but cannot move
an executor before one exists. `InferenceEngineBatchedNoPreprocessing._move_models_to_devices`
moves the model at first fit. Our optimizer was restored earlier, while the parameters were
still on CPU; PyTorch `Optimizer._process_value_according_to_param_policy` restores Adam moments
to each parameter's current device. Moving the model later left those moments behind. Fresh
training worked because Adam initialized its moments after the first forward; CPU resume
tests worked because no device changed.

The fix places the model on its training device **before optimizer construction**, following
the installed `tabpfn.finetuning.finetuned_base.FinetunedTabPFNBase` training path. It preserves
Adam's legitimate CPU scalar step counters. A local meta-device regression failed against the
old implementation. Added real-weight PD/LGD stop/save/recreate/resume/update probes exercise
optimizer moments, scheduler, scaler and task counters; the CUDA/AMP versions run in the early
mandatory VSC stage, before the six longer validation arms. They are not skipped on VSC.

Local verification: the old code failed the new device-placement regression; the fix passed
all 18 targeted tests, including PD/LGD real-weight CPU resume and checkpoint prediction.
The full suite then passed **1,391 tests / 15 skips / 505 warnings in 1,103.54 seconds**.
Skipped: 13 VSC-only checks, one generated-figure check because the output was cleared, and
one missing-upstream branch because upstream is installed. No notebook/visualization code
changed. This verifies CPU behavior and the placement contract, not B200 execution.

Keep the failed attempt's output. The code/test changes invalidate the old source-bound CPU
receipt, so publish the fixes, pull them on VSC, rerun CPU validation and then GPU validation.
Experiment 1 remains blocked until both receipts pass for that source. A CPU-only laptop cannot
certify the actual CUDA fix; the next VSC attempt must establish that. No production reruns or
further cleanup were submitted here.

## 14-10-2026 — Exp0 CPU 62311762 failed two path fixtures; GPU correctly blocked

**Submitted** 14-10-2026 08:44 CEST | **job** 62311762 | **cluster** wICE `batch`, node `l33c20n2`
**Commit** `2380e36` (log reports dirty checkout; cause not established) | **tfm-library pin**
`de324efcca55be392feb0c01830b3b6ce364587f` | **GPUs** 0 | **walltime** 4h requested / 14m53s log interval

### Configuration

- `scripts/slurm/exp0_cpu.slurm`, 8 CPUs, 64 GB; `validate_experiment.py --phase cpu`.
- Twelve stages: preprocess, portable tests, preflight and nine diagnostic sections.
- Protocol 7, all rows and eligible features, five-fold checks on 21 credit and 50 pinned OOD tables.
- TabPFN 9.1.0, TabICL 2.1.1 from commit `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10,
  torch 2.11.0+cu128. This job performed no production training or full model benchmark.
- Before submission, the reset removed 310 generated files / 60.785 GB from both storage tiers,
  including flat legacy `checkpoints/exp1_*`; the repeat dry run found zero generated files.
  Source data and four released checkpoints were retained.

### Results

- **FAILED exit 1:** 11/12 stages passed. Unit stage: 1,326 passed, 2 failed, 7 skipped, 10 deselected
  in 619.67s. Remaining stages completed; 135 dependency checks passed.
- Failures: `test_notebook_outputs` assumed a shared big/small-output parent; `test_paths`
  assumed a fresh OOD cache while the cluster retained the supported legacy cache.
- Four architecture tests silently skipped because their local checkpoint glob missed project storage.
- The subsequent GPU submission attempt at 09:08 was refused by the CPU receipt gate.
- Evidence: downloaded `output CreditICL.zip`, 39 files, and the 14 October terminal transcript;
  `experiment_0/logs/exp0_cpu_20261014_084405_j62311762.log`, attempt
  `cpu_20261014T064418Z_j62311762_cc1e460d`. No GPU job or new Exp1–3 result appears in this upload.

### Bugs and anomalies

The two unit failures concern fixtures, not failed training. Independent follow-up confirmed
additional production risks: large-value reduction overflow, incomplete borrower grouping,
post-default/future fields, the no-macro rate confound, unequal TabPFN preprocessing,
precision/equivalence mismatch, and repeated Mindwell working I/O on Lustre. These have separate
code repairs and regression checks; a CPU fixture repair alone would not validate them.

### Interpretation and next

The guard saved an unnecessary GPU allocation. This upload proves preprocessing, package and
data/split checks succeeded on wICE; it does not prove CUDA execution or full-table model capacity.
After publishing the local repairs, rerun CPU validation, then GPU validation. Preserve this
attempt for diagnosis; no fresh reset is needed. Only successful current receipts permit Exp1.

## 13-10-2026 — Exp0 CPU 62306531 submitted; legacy checkpoint cleanup gap confirmed

- **Job/config:** `scripts/slurm/exp0_cpu.slurm`, wICE batch, 8 CPUs, 64 GB, four-hour limit;
  submitted at 16:43 after `git pull --ff-only` reached `91a568a`. Library pin
  `de324efcca55be392feb0c01830b3b6ce364587f`. No training hyperparameters: CPU validation only.
- **Resolved setup:** project Python 3.12 environment; dependency update/audit succeeded with
  152 installed-package records, torch `2.11.0+cu128`; 25 classification + 25 regression OOD
  tables refreshed without changing the pinned dataset list. Protocol 6, five folds, full rows,
  but the then-current 100-feature cap remained. The job's resolved receipt has not been supplied.
- **Observed outcome:** user `sacct` at 16:53 reports RUNNING, elapsed 00:10:32. No CPU completion
  or GPU submission/result is established by this transcript.
- **Cleanup evidence:** reset removed 1,703 generated files (0.042 GB) from both current output
  roots. A subsequent `find` enumerated old `step-*.ckpt` files beneath project-storage
  `checkpoints/exp1_pd__*` and `checkpoints/exp1_lgd__*`, alongside four released reference files.
  Therefore the previous reset was incomplete for historical generated checkpoints.
- **Interpretation/action:** local correction adds legacy run/output discovery while preserving
  the released files, and protocol 7 removes shared and wrapper feature caps. Four real released
  model probes (TabPFN/TabICL × PD/LGD) passed with 601 input features, including LGD quantiles;
  both classifiers also passed at 2,986 features. This is local CPU evidence, not a VSC PASS.
  After the active job stops, update the checkout,
  reset with the corrected command, run CPU validation again, and only then submit GPU validation.
  No cluster command was executed by the agent; no GPU hours were consumed in this session.


## 13-10-2026 — Exp0 job 11681071 output audited; two GPU-specific failures

**Run date:** 12-10-2026, 18:31:35–18:52:36 CEST in the log (21m01s, not a billing record).
**Job:** 11681071, Mindwell B200, one GPU/24 CPUs/180 GiB, two-hour request.
**Input:** Downloads `output CreditICL.zip`, 7,753,774 bytes, 1,165 files, CRC valid;
SHA256 `83299ad94fe6cc81605346fccefc593522d964f2aa4bf04dbf43b51f1ab1697c`.
**Source fingerprint:** `68a5b2b001b4270a8c64f8f3e836497292758bb4dbd325c24339327f5a358a49`.
**Environment fingerprint:** `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.
The inspected library pin for this audit is `de324efcca55be392feb0c01830b3b6ce364587f`.

### Configuration and evidence

The report is `experiment_0/report/checks.json`; its main log is
`experiment_0/logs/exp0_checks_20261012_183135_j11681071.log`. TabPFN 9.1.0,
TabICL 2.1.1 at `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10 and
PyTorch 2.11.0+cu128 passed dependency checks. All 21 v5 credit caches, four released
weights and 50 OOD datasets passed. The six smoke configurations are Exp0's resolved
scratch-TabICL/continued-TabICL/continued-TabPFN routes, both tasks, with max_steps=10,
progress disabled, save_perm_every=10, save_temp_every=0, 23 workers and production
1,024-row shapes; batches are 64 for TabICL and 16 for TabPFN.

### Results, bugs and interpretation

95 PASS, 2 FAIL, 1 WARN (dirty checkout). TabICL's four smoke routes and checkpoint-to-wrapper
weight comparisons passed: PD 51.2/50.9 s, LGD 38.9/39.2 s for ten steps, peaks 12.11–13.05 GiB.
Both 20-step upstream-equivalence processes exited zero. These short times include startup and
are not reliable whole-campaign estimates.

PD TabPFN reported a nonfinite gradient norm at strict clipping. The old path prevented recovery
from scaled AMP overflow; the log alone cannot prove this was its only numerical cause.
LGD TabPFN failed in searchsorted because CPU-prepared bar-distribution
buffers accompanied CUDA predictions/targets. The fixes add device-local loss buffers and bounded
same-batch AMP retries (no silently skipped update or shortened task budget). Actual VSC success
of those fixes remains unverified until a new Exp0 run. No 600-step run, corrected Exp1 training
or new benchmark completion is evidenced by this upload.

The historical 90-run campaign remains preserved. On 13-10 the user chose three main mixtures
plus targeted components: 9 main and 15 ablation runs per task, 48 total. The former 15 settings
were filter/strength combinations, not 15 distinct mixture fractions. Their reduction appears
in `0bac424` (29-09-2026); git authorship alone does not establish who approved that decision.
The new validation workflow uses one Slurm allocation with separate attempt folders; it refuses
production launch until all required stages and artifact checks pass. No jobs submitted by this audit.

## 12-10-2026 — Prerequisites passed; revised Exp0 checks job accepted

**Submitted** 12-10-2026, terminal returned at 18:30 CEST, by Andreas | **job** `11681071`
**Cluster** Mindwell `gpu_b200` | **checkout pulled** `0ce444b` | **script** `exp0.slurm`
**Requested allocation** 1 B200, 24 CPUs, 180 GiB, 02:00:00; runtime and outcome not yet supplied.
The local library pin is `5b6c24dd9337c6e7bef45733730bf52ee280e61c`; the VSC submodule checkout
was not printed in this transcript.

### Configuration

The project environment updater completed in `.venv-skylake` (Python 3.12.3). Dependency
preflight passed TabPFN 9.1.0, TabICL 2.1.1 at the required source commit, CatBoost 1.2.10,
all 135 checked requirement versions including extras, `pip check`, and torch 2.11.0+cu128.
It saved 152 installed-package records to `general/report/environment.json`.
This job checks the GPU pipeline and runs both 20-step trainer-equivalence comparisons;
it does not submit the six 600-step Exp0 arms or the 18-arm Exp1 campaign.

### Results and anomalies

Post-completion preflight confirms **all 21 data caches at version 5**, all four released
weight files present, and 25 classification plus 25 regression OOD datasets without missing
files. This resolves the intermediate stale-cache observations during job `62297440`.
Slurm accepted job `11681071`; no running/completed state or GPU-check result was supplied.
There is no new failure in this transcript and no reason to repeat preprocessing.

### Interpretation and next

Wait for this checks job and inspect its exit status/log, then require
`python -m src.utils.preflight --require-exp0-checks` before the short-arm stage.
Preserve September's 90 checkpoints. They completed the old training recipe; they do not
count as the corrected Exp1, whose current configs expand to 9 PD and 9 LGD arms. A local
comparison against the configs at `2034cc2` confirms 45 old versus 9 current arms per track,
with no shared run names. No training code or configuration changed in this follow-up.

## 12-10-2026 — Environment updated on VSC; CPU cache rebuild completed

**Submitted** 12-10-2026 around 18:01 CEST by Andreas | **job** `62297440` | **cluster** wICE
**Checkout pulled** `57377b2` | **script** `preprocess.slurm`, `TASK=both` | **GPUs** none
**Requested allocation** 8 CPUs, 64 GiB, 02:00:00; raw data unchanged, stale caches rebuilt.

The installation transcript confirms Python 3.12.3 in `.venv-skylake`, torch 2.11.0+cu128,
TabPFN upgraded from 8.2.0 to **9.1.0**, TabICL 2.1.1 from commit
`8f1aa2098c894ab91dba15209cf2002ba4be6c6c`, CatBoost 1.2.10, NumPy 2.5.2, SciPy 1.18.1,
scikit-learn 1.9.0, pandas 2.3.3, PyArrow 25.0.1 and Pydantic 2.13.4. These are observed
installation versions, not a completed GPU-validation result. The local library pin after
Andreas's update is `5b6c24dd9337c6e7bef45733730bf52ee280e61c`; no library contents were edited.

An immediate preflight observed 13 rebuilt PD caches and 8 still at version 3 (the final PD
dataset and all seven LGD datasets). All four released weight files were found; the legacy
OOD cache supplied 25 datasets per kind, with no missing files. The command chain stopped
on the stale caches, so **no Exp0 GPU job was submitted**. The later `sacct` output reports
**COMPLETED, exit 0:0, elapsed 00:03:00**, including both job steps. Final cache versions must
still be confirmed by post-completion preflight; a successful scheduler exit alone is not that check.

Next: use the project-only package updater and full dependency audit, repeat preflight, then
submit the single Exp0 checks job. Preserve all earlier checkpoints/results; do not repeat
this successful preprocessing job unless the final cache check identifies a remaining problem.
GPU usage for this CPU job is zero; exact CPU billing was not provided.

The subsequent package-safeguard changes passed the local full suite (1,192 passed, 3 skipped,
781 s), plus all 16 environment-specific tests. The three skips remain the two unavailable
local CatBoost cases (tested separately earlier) and the inapplicable upstream-absent case.
This does not count as GPU validation of the newly installed VSC packages.

## 12-10-2026 — Exp0 checks failed on stale environment/data; trainer equivalence passed

**Submitted** 12-10-2026 14:03 CEST by Andreas | **job** `11680453` | **cluster** Mindwell `gpu_b200`
**Checkout pulled** `78f6b3f` | **GPUs** 1 x B200 | **allocation** 24 CPUs, 180 GiB, 02:00:00 limit

### Configuration

`sbatch --clusters=mindwell scripts/slurm/exp0.slurm`: environment, storage, data, released
weights, prior/trainer smoke checks and benchmark folds, followed by 20-step comparisons of
our trainer with upstream on identical batches for each track using `Exp1_PD.yaml` and
`Exp1_LGD.yaml`. No full training sweep is part of this job. The terminal confirms that both
old output roots were renamed successfully and checkpoints outside them were not moved.
The cluster's actual submodule checkout was not printed; the local pin is
`1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`.

### Results, anomalies and next step

The subsequent 14:47 terminal transcript reports **FAILED, exit 1:0, elapsed 00:17:27**,
ending at 14:21 CEST. There are 35 FAIL items; principal causes:

- All 21 processed datasets still use cache version 3, while the current loaders require 5.
- The new `data/ood` directory is absent; the pre-existing `prior_cache/ood` was not consulted.
- Installed TabPFN **8.2.0** rejects `TRANSFORM_TEXT` and `TRANSFORM_DATES` with two Pydantic
  validation errors. Andreas's 15:01 output confirms TabICL 2.1.1 and Pydantic 2.13.4.

Both PD and LGD trainer-equivalence subprocesses returned **0** on the B200. Determinism
across worker counts, processes and shared control slots passed. These successes do not
override the other failed sections. There is no supplied completion evidence for the six
600-step arms, the forced-resume test, or the Exp0 benchmark.

This job consumed **0.290833 GPU-hours** of allocation. The published GPU-only rate of
437.5 credits/minute implies ~7,634 GPU credits; that excludes CPU charges and is not a
SAM billing record. Exact credits require `sam-list-usagerecords`/account records. The old
campaign's ~1,354 GPU-hour figure is a historical estimate, not this job's cost.

Prepared fixes: dependency pins and read-only preflight, CPU rebuild of stale caches without
`--force`, legacy OOD lookup, version-checked local model loading, current-code Exp0 launch
gates and paired benchmark receipts. A current-code fingerprint prevents an old validation
report or old training summary from unlocking the revised campaign. Successful historical
outputs and checkpoints are preserved.

The same-day primary-source review found TabPFN 9.1.0 (released 02-10-2026) fixes regressor
loss targets after per-estimator transforms. The project now uses that helper and pins 9.1.0;
Andreas authorized installing this version in `.CreditICL` only. Both real-weight PD and LGD
save/resume/evaluation integration tests passed. A separate CPU probe drew 64 tables for each
PD/LGD x control/pure-credit combination at the configured 1,024 rows (256 tables total):
all passed, no fallback; control rejection rates were 28.1% PD and 20.0% LGD, at most four
attempts for a table. Credit tables are unfiltered under the configured base-only filter scope.
This is a small validation sample, not a guarantee for an 800,000-table arm.
The TabPFN change is a future TabPFN/Exp2 path fix, not a
cause of September's TabICL results. No cluster job was submitted by the agent.

Andreas also selected **PD at application / LGD at default**. Protocol 5 excludes known
post-outcome/target-construction fields, retains every eligible fitted row by default, and
does not clip OOD regression to [0,1]. These changes invalidate old evaluation receipts,
not the saved synthetic-training weights. Remaining anonymised feature timing and temporal/
grouped splits still require source-dictionary work before confirmatory publication claims.
The CatBoost wrapper now declares categorical columns and converts codes/missing values
consistently for its native handling. Two real CatBoost 1.2.10 tests (PD/LGD, missing/unseen
categories) passed using the already installed CreditPFN environment without modifying it;
CatBoost is absent from `.CreditICL` and is now a preflight dependency check.

Local validation with TabPFN 9.1.0: full pytest suite **1,175 passed, 3 skipped** (1,220 s).
Two skips were the CatBoost integrations verified separately above; the third tests the
upstream-absent path, inapplicable with TabICL installed. After the final provenance guards,
the affected launcher, benchmark, recovery and trainer integration modules passed **98 tests**,
with those same two CatBoost skips (199 s). `pip check` and `git diff --check` passed.
These checks do not replace validation of the current code on the B200 or a full-size benchmark.

Next: deploy fixes, repair packages and caches, pass preflight, repeat the checks job,
then complete short training/resume/benchmark validation before the revised **18-arm Exp1**.

## 12-10-2026 — September benchmark CSVs verified; rerun reasons clarified

**Source:** `Downloads/benchmark-results.zip`, 1,544,116 bytes, 552 files; SHA-256
`e7c472d994364c35f389d1ca6955fa5dee4ae3a6d8f5cb23168e1aa822e16898`.
All archive CRCs pass. This supplies the missing result tier for jobs `11614376` and
`11614377` of 24-09-2026, not a newly executed experiment.

### Configuration and results

All 92 receipts describe evaluation protocol 2. Each receipt's full dataset x model x seed
set is present, without duplicate keys; all 184 credit/OOD CSV SHA-256 values match their
receipts. Every one of the 12 recorded evaluation-source hashes matches commit `2034cc2`,
resolving the evaluation-code uncertainty left by the log's dirty-checkout marker.
Training-source provenance is not included in those hashes.

| Track | Credit rows OK | OOD rows OK | Best control dev | Best 50% credit dev |
|---|---:|---:|---:|---:|
| PD (ROC-AUC) | 2,058 | 3,675 | 0.732639 | 0.732451 |
| LGD (R2) | 1,029 | 3,675 | 0.519735 | 0.518260 |

Every headline metric is present and finite. All 945 trained-checkpoint LGD rows report
`pred_nonfinite_frac=0`, including all 135 rows on `0004.base_model`, the dataset affected by
historical monitoring NaNs. Mixed-credit and control remain close; pure-credit LGD remains
poor. Those old monitoring NaNs are not a reason to retrain or discard checkpoints.

### Limitations and chronology correction

- Context mismatch is confirmed: our and released TabICL used at most 1,024 rows, TabPFN up
  to 10,000, and CatBoost/linear the full pool. The outer `n_train_used` and requested
  `context_cap` columns alone do not describe the actual TabICL context; its inner cap is
  recorded separately. A fair cross-model comparison requires re-evaluation, not retraining.
- AXA's 147 rows have no boundary MAE because the old tolerance counted its near-zero/one
  targets as interior. Headline R2/RMSE remain finite; boundary diagnostics need recomputing
  with the current tolerance.
- The specific calibration of priors against all evaluation datasets was introduced during
  development on 28-09 and replaced by literature-based ranges on 29-09, after these runs.
  The saved September configs contain the earlier recipe. It was incorrect to present that
  later calibration change as an established flaw in these September checkpoints.
- The genuine training changes documented on 28-09 do apply: the old Muon setup differed
  from upstream, PD mechanism mode skipped configured selection/noise, and the task stream,
  LGD target encoding and prior design were subsequently revised. Existing weights cannot
  be relabelled as trained with the corrected recipe; the updated main Exp1 needs 9 arms per
  track, 18 total. The old 90 remain exploratory evidence.

The current optional capped benchmark still silently left TabPFN uncapped. A local regression
test reproduced 5,000 rows being fitted when 200 were requested for both tasks. The shared
TFM sampler now enforces identical capped rows and accurate fit counts; full-context behavior
is preserved. Validation: 1,143 tests passed, one expected skip. This fix is local until deployed.

Preserve all historical files. The subsequently supplied failure of the submitted Exp0 job is
recorded above; repair it and complete the short Exp0 runs, then revised Exp1 and its benchmark. No new cluster
job was submitted by the agent during this CSV audit.


## 12-10-2026 — downloaded September campaign re-audited; no new run

**Source:** `Downloads/output.zip`, 7,710,929 bytes; SHA-256
`537f8272f6530e9cc890bddcc839b3ebe64a70ee23a9d8ca99135e017a01b928`.
All ZIP CRCs pass. Extracted 1,156 files to `Downloads/output CreditICL`; renamed the archive
to `Downloads/output CreditICL.zip` without changing its hash.
Archive timestamps are copy times: actual logs end on **24-09-2026 at 22:00:27 CEST**.

### Configuration and evidence

- The saved resolved configs describe September's 45 arms per track, three seeds,
  credit fraction × filter × intensity, 12,500 steps and 800,000 datasets per arm.
  They differ from the current YAML files. See the original entries below for the full recipe.
- All 90 summaries report completed=true and the full budget; all 90 final training wrappers
  end OK. There are 58 normal incomplete/resubmitted wrappers, not unresolved failed runs.
- Benchmark arrays **11614376 (LGD)** and **11614377 (PD)** cover indices 0–45 without gaps.
  All 92 wrappers end OK: **3,087/3,087 credit cells**, **7,350/7,350 OOD cells** reported successful.
  Each of the 90 final checkpoint paths was loaded for both credit and OOD evaluation.
  Benchmark environment records identify commit `2034cc2`, `git_dirty=true`, and library pin
  `21d555a6a24e`; the exact uncommitted diff is not contained in this archive.
- No traceback, error-level line, failed wrapper, or non-finite logged training loss found in
  the supplied logs. No NaN/Inf tokens in the benchmark wrappers. The old LGD progress files
  still contain 605 non-finite base_model measurements in 30 arms; these are monitoring records,
  not the final benchmark. All benchmark wrappers warn that an old Python module did not load,
  then successfully run through the environment; recheck the environment after the VSC update.

### Results and limits

Reconstructed from rounded per-cell log scores using each saved config's development list:
PD best control ROC-AUC **0.73264**, best mixed-credit **0.73245**; LGD best control R²
**0.51974**, best mixed-credit **0.51825**. Pure-credit LGD configuration means range
**−0.08049 to 0.00882**. These reproduce the September interpretation, not a new ranking.
The archive lacks benchmark CSVs, receipts, and checkpoint binaries, so a complete metric-cell
and current checkpoint-integrity audit remains outstanding. Historical context-budget mismatch
between reference models and prior/data-partition problems documented below still apply.
Do not treat successful process exits as validation of the revised scientific protocol.

### Next

Preserve the old campaign; rename only its output roots to `output CreditICL`, retaining the
historical internal layout and checkpoint locations. Fetch the project-storage result CSVs.
The revised design needs its Exp0 gate, then Exp1 (9 arms/track), Exp2 (18/track), and Exp3
(4/track, its mix chosen on revised Exp1 development results). This audit submitted no jobs.
VSC guidance was checked against library pin `1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`:
Genius retired, wICE hosts login access, current production GPU targets remain supported.

---

## 24-09-2026 — Exp1 phase 2 benchmark — **COMPLETE 92/92; no credit prior beats the control on development, credit-only is worse on both tracks**

**Submitted** 24-09-2026 20:45 | **jobs** 11614376 (LGD), 11614377 (PD), each `--array=0-45%8`
(45 checkpoints + the reference column) | **cluster** Mindwell `gpu_b200`, 2 nodes | **Commit**
`2034cc2`, dirty checkout | **GPUs** 1 × B200 per task | **walltime** LGD 20:50–21:09, PD 21:06–22:00

### Configuration
- `python -m src.utils.run_experiment 1 --submit` → `scripts/slurm/benchmark.slurm`, `EXP=1`, both
  tracks, scoring the 90 checkpoints of the completed phase 1 (entry below). Context cap 1,024
  rows, evaluation seeds 0, 1, 2; reference column tabiclv2, tabpfn3, catboost, linear.
- **Anything edited by hand since the last run:** nothing in the evaluation code; `2034cc2` changed
  notebooks, docs and plotting only.

### Results
Read from the 92 task logs (the result CSVs stay on project storage), with `src/eval/selection.py`'s
rule: development datasets only, each arm's mean over its dataset × evaluation-seed cells, then the
mean and SD over training seeds. Holdout is shown for reference, never used to choose.

| | PD dev ROC-AUC | PD holdout | LGD dev R² | LGD holdout |
|---|---|---|---|---|
| best control (cf 0) | 0.7326 ± 0.0007 (`banded`) | 0.7281 | 0.5197 ± 0.0029 (`tabicl`) | 0.4590 |
| best credit prior | 0.7325 ± 0.0016 (cf 0.5 `banded` aggressive) | 0.7266 | 0.5182 ± 0.0022 (cf 0.5 `banded` aggressive) | 0.4479 |
| cf 0 and cf 0.5, all 12 configurations | 0.7303–0.7326 | 0.7208–0.7281 | 0.5143–0.5197 | 0.4440–0.4601 |
| cf 1, all 6 configurations | 0.7009–0.7151 | 0.6765–0.6901 | −0.081–0.009 | −0.219 to −0.033 |
| released TabICLv2 | 0.7322 | 0.7370 | 0.5337 | 0.4983 |
| TabPFN-3 | 0.7447 | 0.7665 | 0.6131 | 0.5515 |
| CatBoost | 0.7340 | 0.7762 | 0.5756 | 0.4952 |
| linear / logistic | 0.7246 | 0.7278 | 0.3874 | 0.3135 |

- **Out of domain:** PD ROC-AUC 0.947–0.952 for cf 0 / 0.5 against 0.931–0.940 for cf 1 (released
  TabICLv2 0.953); LGD R² 0.710–0.723 for cf 0 / 0.5 against −0.01–0.07 for cf 1 (TabICLv2 0.753).
- **Files** `output/logs/bench_exp1_{pd,lgd}_*_j1161437{6,7}_a*.log`; results CSVs under
  `/lustre1/project/stg_00211/CreditICL/output/results/{pd,lgd,ood}/eval/`.

### Bugs and anomalies
- None in the run: all 92 tasks `END status=OK`; every credit cell scored (42/42 per PD arm, 21/21
  per LGD arm, 168/168 and 84/84 for the reference column); 0 failed out-of-domain cells.
- **cf 1 on LGD collapses** — R² ≈ 0 on development, holdout and out-of-domain data alike, in all six
  configurations. Cause not established. *(25-09: most likely the target scale — `TabICLRegressor`
  standardises y, and cf 1 arms only ever trained on raw [0, 1] credit targets; calibration slope 4.1.
  Untested; see `AGENTS_MEMORY.md`, 25-09-2026.)*
- *(Found 25-09.)* **The 1,024-row cap binds only on our arms and the released TabICLv2**
  (`TabICLBaseline._cap_context`; for PD it draws 512 defaults and 512 non-defaults). TabPFN-3 goes
  through `_TFMBaseline._maybe_subsample` — up to 10,000 rows, in proportion — and CatBoost / linear
  see every row, although `benchmark.slurm` says the cap "applies to EVERY model". Ours vs the released
  TabICLv2 is matched; the other references had more data. The balanced PD context also inflates our
  arms' and TabICLv2's predicted default rate (mean 0.41 against a base rate of 0.22; calibration
  intercept ≈ −1.2, the log-odds shift from 22 % to 50 %), so their Brier score and calibration are not
  comparable with the references'. ROC-AUC is a ranking and unaffected by the shift.

### Interpretation
- **Show:** on development data every cf 0 and cf 0.5 configuration lies within 0.0023 ROC-AUC (PD)
  and 0.0054 R² (LGD), about one to two seed SDs, so filter mode and intensity make no measurable
  difference and the credit prior does not beat the control. Pure credit (cf 1) is clearly worse on
  both tracks. Our best PD models match the released TabICLv2 on development at 2.5 % of its budget,
  but trail it on holdout (0.728 against 0.737); on LGD they trail it on both.
- **Think:** the credit mechanisms add nothing the base prior lacks at this budget, and a prior made
  only of them removes something the models need — for LGD, apparently everything. A hypothesis.
- **Test:** Exp2 (continued pre-training from the released weights) asks the same question of a
  strong model.

### Next
Fill Exp2/Exp3's `FILL_FROM_EXP1` from notebooks 1.3/1.4 and submit Exp2.

---

## 24-09-2026 (evening) — Exp1 phase 1, final download — **COMPLETE: 90/90 arms trained; phase 2 not yet run**

**Snapshot** `C:\Users\U0152019\Downloads\output`, downloaded 19:41 CEST; logs 498 files (25 MB),
manifests 369 (7.6 MB). Last training line: PD a40, 17:34:13. **Final jobs** PD a25 `11604659`
(END OK 10:00), LGD a43 `11591463_43` (END OK 10:57), and two automatic continuations after the
noon walltime: PD a19 `11613402` (12:06 → 12:38) and PD a40 `11613401` (12:05 → 17:34).
**Cluster** Mindwell `gpu_b200`, 1 × B200 per arm | **Commit** final segments PD `d4f06c2`, LGD
`7800d08`, both with uncommitted changes (earlier segments `7d1cb02`, `9b752d2`, `6d716ec`) |
**tfm-library pin** `21d555a6a24e` | **Python** 3.12.3, Linux | **walltime** 12 h PD, 72 h LGD.

### Configuration
- Unchanged `config/Exp1_PD.yaml` and `config/Exp1_LGD.yaml`, 45 arms per track: credit fraction
  {0, 0.5, 1} × filter {tabicl, banded, off} × intensity {mild, aggressive} × seeds {0, 1, 2},
  the credit-fraction-0 control deduplicated.
- Budget 12,500 steps × 64 = 800,000 datasets per arm; scratch init, Muon.
- **Edited by hand since the morning snapshot:** nothing.

### Results
All **90 arms `completed: true`** at 12,500 steps and 800,000 datasets; the latest Slurm log of
every array index ends `END status=OK`. The four arms running this morning finished from steps
7,235 (a25), 2,500 (LGD a43, one 49-hour segment), 12,292 (a19) and 9,268 (a40).

Training monitor at the final step — **not the verdict**, and not comparable across protocols:

| | control (cf 0) | cf 0.5 | cf 1 |
|---|---|---|---|
| PD development ROC-AUC, `german` + `myhom` (the only dev sets all 45 arms monitored) | 0.672 | 0.669 | 0.646 |
| PD out-of-domain ROC-AUC, 8 sets | 0.997 | 0.992 | 0.975 |
| PD training loss (median over arms of each arm's last three logged losses, as notebook 1.1) | 0.295 | 0.199 | 0.066 |
| LGD development R², `base_model` (finite in 3 / 6 / 6 of 9 / 18 / 18 arms) | −0.03 | 0.19 | 0.02 |
| LGD out-of-domain R², 8 sets, all 45 arms (notebook 1.2 averages its 15 scored arms: control 0.58, credit 0.49) | 0.51 | 0.55 | 0.42 |
| LGD training loss (as notebook 1.2) | 0.093 | 0.064 | 0.032 |

- **Progress curve:** plateaued — PD development ROC-AUC moved by ≤ 0.002 over the last three
  monitor points (~1,900 steps). The nine protocol-2 PD arms (banded, cf 0 / 0.5; five dev sets)
  end at 0.725 (cf 0) and 0.718–0.723 (cf 0.5).
- **Out of domain:** held for the control and the blend; credit-only gives up 0.022 ROC-AUC (PD)
  and ~0.1 R² (LGD).
- **Throughput:** median 0.63–0.65 steps/s (`off`, `tabicl`) and 0.12–0.15 (`banded`); GPU
  utilisation 89–93 %; peak allocated memory 13.1 GB. **Compute** ≈ 587 GPU-h PD + 767 GPU-h LGD
  (logged segment time, excluding start-up, checkpoints and queueing); `banded` arms cost 28 (PD)
  and 39 (LGD) GPU-h each against ~5.5 for the others.
- **Files** `output/logs/`, `output/manifests/` of the snapshot.

### Bugs and anomalies
- **No new failures.** The four last arms ended `END status=OK`; nothing in their logs is an error.
- **The monitors cannot rank priors.** PD: 36 arms use the legacy protocol (`german`, `myhom` plus
  the holdout sets `hmeq`, `thomas`), nine use protocol 2 (five dev sets). LGD: all 45 are legacy,
  and `base_model` R² is non-finite in 30 of them (seeds 0 and 2, one non-finite prediction each),
  so the LGD row above rests on one seed per configuration.

### Interpretation
- **Show:** every checkpoint trained to a plateau. On PD's monitor, control ≈ blend > credit-only
  (0.672 / 0.669 / 0.646, seed SD ~0.01 within a configuration), and credit-only also loses some
  out-of-domain ability. The LGD monitor is too broken to read.
- **Think:** credit-only PD tasks are far more predictable than real default data (pseudo-R² ~0.6
  against ~0.06, notebook 0.2), so the cf-1 arms learn an easier task than the one they are scored
  on. A hypothesis, not tested.
- **Test:** phase 2 — every checkpoint on the development datasets, seeds paired, and the holdout
  only for the prior selected on development.

### Next
Submit phase 2 (`benchmark.slurm`, `EXP=1`) for both tracks — not submitted; it needs Andreas's go.

---

## 24-09-2026 — Exp1 retry progress — 86/90 complete; recovery working

**Snapshot:** `C:\Users\U0152019\Downloads\output CreditICL`, latest training line
24-09-2026 **09:30:48 CEST**; 873 files, 30,769,867 bytes. This is downloaded evidence,
not a live queue query. Andreas reports four jobs still running, matching the logs.
**Submitted:** PD parent `11598087` on 23-09 at 12:22; eight automatic continuations
`11604654`–`11604661` started just after midnight. Existing LGD `11591463_43` continues.
**Cluster/resources:** Mindwell `gpu_b200`, one B200, 24 cores, 180 GiB per task;
PD 12-hour allocations, existing LGD 72 hours. PD array concurrency is capped at eight.
**Recorded code:** PD `d4f06c2`, LGD `7800d08`; both report a dirty checkout and library
pin `21d555a6a24e`. The exact uncommitted cluster diff is unavailable.

### Configuration

- Unchanged `config/Exp1_PD.yaml` and `config/Exp1_LGD.yaml`, 45 arms per track.
  Resolved configurations are present in the downloaded `manifests/*__config.json`.
- Budget: 12,500 total optimizer steps × 64 datasets = 800,000 datasets per arm;
  micro-batch four, Muon 0.0008 / Adam 0.0003, AMP enabled, 23 data workers.
- PD retries: `1,4,10,16,19,25,31,34,40`, all `filter.mode=banded`, credit fraction
  zero or 0.5. Training resumes existing scratch-trained checkpoints; it does not reset.
- Remaining PD a19 is fraction 0.5, rho `[0.03,0.12]`, seed 1; a25/a40 are fraction 0.5,
  rho `[0.12,0.30]`, seeds 1/2. LGD a43 is fraction 1, boundary mass `[0.15,0.60]`,
  banded filter, seed 2. No configuration or code was changed during this inspection.

### Results

**PD 42/45; LGD 44/45; total 86/90 complete**, up from 80/90. All 86 completed summaries
agree with their latest Slurm logs: `completed=true`, 12,500 steps, 800,000 datasets,
a final checkpoint-save message and `END status=OK`. Checkpoint binaries are on cluster
storage and were not supplied for loading. Latest step evidence across all arms totals
1,118,150 / 1,125,000 steps (**99.39%** of the nominal update budget).

| Remaining job | Track/arm | Latest step | Progress | Conditional finish, 24-09 CEST |
|---|---|---:|---:|---|
| `11604659_25` | PD 25 | 12,250 | 98.0% | ~09:55–10:10 |
| `11591463_43` | LGD 43 | 12,200 | 97.6% | ~10:45–11:10 |
| `11604658_19` | PD 19 | 11,000 | 88.0% | ~12:30–13:15, plus any queue delay |
| `11598087_40` | PD 40 | 7,700 | 61.6% | ~17:30–18:30, plus any queue delay |

These estimates use step/timestamp slopes over the latest one, two and four hours,
allowing for final checkpoint work. PD a40 only started at 00:13 on 24-09, after a slot
in the eight-task array became available. PD a19 and a40 are likely to need another
automatic checkpoint/resubmission near their noon walltime limit.

Six retried PD arms completed: **1,4,10,16,31,34**, through child jobs respectively
`11604655`, `11604657`, `11604660`, `11604656`, `11604654`, `11604661`.
All eight overnight continuations loaded exactly the step saved by their parent:
1=10816, 4=8405, 10=10071, 16=11089, 19=6705, 25=7235, 31=11529, 34=8746.
The nine original retries also explicitly loaded their pre-existing checkpoints.

All four active arms have finite losses and nonzero finite column/row/ICL gradients.
Recent throughput is ~195–207 steps/hour for LGD a43, ~472–485 for PD a19,
~541–580 for PD a25, and ~581–598 for PD a40. Waiting for generated data accounts
for roughly **61–91%** of recent timed phases; sampled filter rejection is 85–93%.
Peak allocated GPU memory is about 13.1 GB. Data generation/filtering limits progress.

### Bugs and anomalies

- **No new training crashes:** 17 PD retry/continuation logs contain zero tracebacks,
  CUDA failures, error-level messages or non-finite training losses. Eight ended with
  the intended `INCOMPLETE-RESUBMITTED`, six with `OK`, and three remain active.
  The latest Slurm logs of all 90 arms contain no terminal training failure.
- The nine repaired PD progress CSVs contain **160 rows**, five development datasets
  and eight OOD datasets, with finite headline metrics, intact headers and increasing,
  unique steps. Nine legacy PD curves were retained separately by the resume code.
- **Old LGD monitoring remains unreliable:** 30/45 legacy curves contain non-finite
  headline R² values (605 recorded cells). Active a43 still reports NaNs for one credit
  dataset, although its loss is finite; its old progress history also repeats some steps.
  This run started before the monitor repair. These measurements cannot establish that
  a checkpoint is invalid or that a prior wins; the corrected final benchmark must decide.
- Thirty-six PD curves and all 45 LGD curves use the legacy monitoring protocol; nine PD
  curves use protocol 2. Their monitoring scores must not be pooled into a fair ranking.
- Partial summaries lag active progress: LGD a43 still has the old 1,622-step summary;
  PD a40 has no completed/interruption summary yet. Without the cluster checkpoint tree,
  the local status helper labels a40 `todo`; its live log proves resumption and 7,700 steps.
- Module-load fallback and projected-walltime warnings recur, but activation, training
  and automatic continuation succeeded. No benchmark logs or result CSVs were supplied;
  the large results storage tier is outside this downloaded folder.

### Interpretation

The retry and walltime recovery paths are operating correctly. Six previously failing
arms now have completed-run evidence; the other three PD arms and last LGD arm are advancing.
The recent rates put completion of the training phase in the late afternoon or early
evening of 24-09 if rates and queue availability hold. This does not yet establish a
benefit over released weights: final common-protocol credit/OOD evaluation is outstanding.

### Next

Allow the active runs and their automatic continuations to finish. Benchmark the completed
tracks through the corrected evaluation path, require finite and complete paired-domain
results, then select development-set priors before configuring Exp2/Exp3. Address the
generation bottleneck when planning long runs; do not change this sweep mid-run.

## 23-09-2026 — Exp1 scheduler confirmation and recovery repair

**Evidence:** Andreas's pasted login-node output at 10:30–10:39 CEST, following the downloaded
snapshot audited below. No additional cluster jobs were launched during this repair.

### Submission follow-up — 12:22 CEST

Andreas pulled the published fixes, activated the existing Python 3.12.3 project environment,
and submitted **PD array 11598087 on Mindwell**, indices `1,4,10,16,19,25,31,34,40`, at most eight
concurrently. It uses unchanged `config/Exp1_PD.yaml`, `EXP=1`, and automatic checkpoint resume;
the configured budget remains 12,500 total steps per arm, not 12,500 additional steps.
The launcher reported PD **36/45** complete and LGD **44/45**, still running; no LGD task was submitted.
At 12:23 the failed-training-log-only cleanup returned without an error. No deletion count was
printed; its scope excludes checkpoints, successful logs and manifests. Submission acceptance
is confirmed; runtime checkpoint loading, resolved configs, throughput and outcomes await new logs.

### Confirmed state

| Work | State |
|---|---|
| PD array `11591462`, indices `1,4,10,16,19,25,31,34,40` | All FAILED, exit 134 |
| LGD `11591463_13` / `_28` | COMPLETED, respectively 19:42:10 / 19:38:54 |
| LGD `11591463_43` | RUNNING at 25:06:34 elapsed; keep it running |
| Training totals | PD 36/45, LGD 44/45 |
| Benchmarks | Cluster planner reports 0/46 per track |

Hyperparameters and the resolved 45-arm grids are unchanged from the snapshot entry below.
The repair preserves trained checkpoints and resumes unfinished PD arms; it is not a new sweep.

### Code repairs

- Synthetic validity is separate from statistical predictability. Invalid/constant targets,
  including upstream's `-100` sentinel, are never a filter fallback. Invalid candidates have
  bounded retries; only a valid candidate can be returned. Legitimate negative regression
  targets remain allowed. Cached/generated batches are validated before CUDA transfer.
- Monitoring uses configured development datasets and a common evaluation seed. TabICL's
  internal inference AMP is explicitly disabled for the lightweight monitor; context scaling
  uses float64 statistics. Non-finite predictions are errors. OOD target scaling and OOD
  benchmark feature imputation use training/context rows only. Resumed CSVs preserve header
  order, legacy curves are archived on resume, cadence aligns to the dataset count, and a final
  monitor observation is requested at the last training step. These changes cannot repair old
  monitoring measurements retroactively or prove the exact cause of every historic NaN.
- The checkpoint diagnostic loader now uses the same architectural head width as the trainer
  (ten classes for PD by default). Round-trip tests call the actual training model builder;
  copying an obsolete two-class construction into both test sides had hidden the mismatch.
- Training submissions and self-resubmissions carry the experiment number and distinct job
  names. Fresh LGD splits respect the actual grid size. Scheduler lookup failure blocks
  submission. `--phase train|benchmark` makes the requested phase explicit.
- Benchmark jobs require the configured final checkpoint and apply the same 1,024-row TFM cap
  to credit and OOD evaluations, for both our checkpoints and reference models. Reference OOD
  work is restricted to its track's kind. Every configured credit dataset and the full cached
  OOD task set must be represented. A paired-domain receipt checks complete cell coverage,
  finite scores, protocol, run identity, checkpoint metadata and result/code fingerprints.
  Missing/failed cells cannot be counted as complete merely because a CSV exists.
- PD plots read production `roc_auc`/`pr_auc` names. Progress means no longer silently skip
  failed dataset values. Prior-selection rankings exclude holdouts and incomplete configurations,
  aggregate training seeds, and do not import legacy or other-experiment result files.

### Interpretation and next actions

The CPU reproduction demonstrates a real route to the observed classification abort, while the
actual failed CUDA batches remain unavailable. Completed checkpoints are retained: these logs do
not justify discarding the entire sweep. Resume the nine PD arms and allow LGD arm 43 to finish.
Then benchmark all 45 checkpoints plus the shared reference task per track on credit and general
datasets. Only after evaluation and development-set selection is Exp1 complete. Fill Exp2/Exp3
placeholders from that decision; previously inspected holdouts remain a reporting limitation.

### Validation

- Full `.CreditICL/Scripts/python.exe -m pytest -q`: **912 passed, 1 skipped, 3 warnings in
  538.38 s**. Git's installed Bash was placed on this process's PATH, so shell syntax coverage
  ran rather than being skipped. The remaining skip is the upstream-unavailable branch
  (upstream is installed); warnings are the three existing scheduler-order test fixtures.
- The subsequent stale-grid loader refinement passed **44 focused tests**. New modules
  pass Ruff, and all changed Python files pass Ruff's undefined/unused-name checks.
- Full notebook runner: **11/11 OK, 135 PDF figures**. All eight experiment notebooks also
  reran after the final plotting refinement; their text matches the complete combined summary.
  All 15 shell scripts passed Bash syntax checks.
- Snapshot replay reproduces exactly PD indices `1,4,10,16,19,25,31,34,40` and no LGD submission.
- All **826 original downloaded files** retain their original SHA-256 fingerprints.
  No data, checkpoint, submodule, installation, push or cluster job was modified/submitted.

## 23-09-2026 — Exp1 rerun audit — 80/90 complete; one LGD arm progressing, nine PD arms crashed again

**Snapshot** `C:\Users\U0152019\Downloads\output`; latest training line 23-09-2026 09:28:58 CEST.
This is downloaded evidence, not a live scheduler observation.
**Latest submissions** 22-09-2026 ~09:32 | **jobs** PD `11591462` (nine tasks), LGD `11591463`
(tasks 13, 28, 43) | **cluster** Mindwell `gpu_b200`, one B200/task, 24 cores, 180 GiB, 72 h.
**Recorded cluster commit** `7800d08`, dirty | **recorded library pin** `21d555a6a24e`.
**Local audit commit** `56d00da` | **local library pin** `e5ce01614eebe520af303f2b5bfd212298eab2be`.
Relevant generator, OOD loader, progress hook and launcher sources have no local diff from
`7800d08`; uncommitted cluster changes cannot be reconstructed from the download.

### Configuration

- `config/Exp1_{PD,LGD}.yaml`: 45 arms/track; credit fraction {0, 0.5, 1}, filter
  {tabicl, banded, off}, two intensities, three seeds, deduplicated no-credit controls.
  All 90 downloaded run names and array indices match the current expanded grids.
- Scratch initialization; 12,500 steps x 64 datasets = 800,000 nominal datasets/arm;
  micro-batch 4; Muon; AMP; live prior generation. Resumes reseed worker streams, so
  nominal dataset counts do not establish the number of distinct synthetic tasks.
- Active LGD arm 43: seed 2, full credit, banded filter, boundary range [0.15, 0.6].
  Logged levers, verbatim:
  `grid levers: {"prior.credit.target.boundary_mass_range": [0.15, 0.6], "prior.credit_fraction": 1.0, "prior.filter.mode": "banded"}`.
- No training configuration or source was changed during this audit.

### Results

| track | complete at 12,500 steps | incomplete in snapshot | latest log evidence |
|---|---:|---|---|
| LGD | 44/45 | arm 43 | 7,300/12,500 steps (58.4%), still progressing |
| PD | 36/45 | arms 1,4,10,16,19,25,31,34,40 | all nine ended `FAILED exit_code=134` on 22-09 |
| total | 80/90 (88.9%) | one progressing, nine crashed | 80 matching successful wrapper endings |

Every completed summary records 800,000 datasets. Parameter counts are 28,544,991 (LGD)
and 27,552,258 (PD), all trainable. All 12,326 training-metric JSONL records parse and their
numeric values are finite. Latest recorded gradients are finite and nonzero in the column,
row and ICL blocks for every arm. These checks establish recorded training health, not
validity of every generated batch or every evaluation prediction.

The active LGD job resumed from step 2,500. At 09:28:58 it had reached step 7,300, with its
last logged checkpoint at 7,250. Recent telemetry median: 0.0554 steps/s. The log estimates
25 h 55 min remaining, approximately **24-09-2026 11:24 CEST**, conditional on unchanged speed
and no interruption. The last phase breakdown assigns 89.3% to waiting for data. Instantaneous
GPU-utilization samples taken during work do not contradict this CPU/prior bottleneck.
Its old incomplete summary still says step 1,622; summaries are not a live progress counter.

There are **no phase-2 logs and no `results/` directory in this download**. That does not establish
whether the separate project-storage results tier is empty. `All_Results.md` is stale: it claims
no training CSVs exist, whereas the download contains 90.

Monitoring only, using completed **off/tabicl** arms so failed banded controls do not create
survivorship bias:

| credit fraction | PD mean AUC, same four monitor datasets | LGD mean R2, same three finite monitor datasets |
|---|---:|---:|
| 0 | 0.7206 | 0.2883 |
| 0.5 | 0.7230 | 0.4138 |
| 1 | 0.6899 | 0.3444 |

Equal-weight descriptive means over each arm's latest monitoring row, then over arms; six
control arms and twelve per credit share. **Not dev-set rankings, significance tests, or final
benchmark results.** The three LGD datasets in this sensitivity check are designated holdout;
two of the four PD monitor datasets are also holdout. Do not use these means to fill
`FILL_FROM_EXP1`. PD's half-credit gap is small; full-credit PD is weaker. LGD suggests a possible
benefit worth measuring properly. Non-banded OOD monitor means decline with credit share
(PD AUC 0.9970/0.9922/0.9748; LGD R2 0.5899/0.5317/0.4604). Those monitor suites have only eight
tasks, binary reductions for multiclass PD, and whole-table target scaling for LGD; they cannot
establish final OOD retention.

Most completed monitor curves change little over their last two intervals; this is not proof
of convergence on the final protocol. Twenty-one completed arms have their last monitor row
before step 12,500 because of restart timing. Phase 2 must score the final weights.

### Bugs and anomalies

1. **Confirmed invalid-task fallback; strongly supported explanation for PD aborts.** Upstream
   `GraphSCM.__call__` can return zero features and target `-100` when generation produces NaNs.
   `TaskGenerator._finish` preserves this finite sentinel. The filter rejects a constant target,
   but `TaskGenerator.sample` returns its last candidate after 40 rejections even if invalid.
   A controlled CPU reproduction exercised upstream's invalid-output branch, all 40 rejections,
   and its actual `OneHotAndLinear.forward`: the sampler returned `[-100.0]`, and the encoder
   raised `Class values must be non-negative.` No training was run. This behavior is present in
   the installed upstream code and `tfm-library/repositories/TabICL.txt` at the local pin above
   (symbols `GraphSCM.__call__`, `OneHotAndLinear.forward`). It matches the remaining failures:
   banded arms containing base-prior tasks (cf=0 or 0.5), while all six pure-credit PD banded arms
   completed. The actual crashed batches were not saved, so their labels are not directly known.
   Invalid regression fallbacks could also pass silently; their frequency and effect on completed
   LGD training are unmeasured.
2. **The earlier OOD-label explanation is insufficient.** The 22-09 jobs recorded `7800d08`,
   which includes the OOD `-1` sanitizer. Their progress evaluations succeeded hundreds of
   training steps before the aborts. The 07-09 entry's confident attribution should not be reused
   as an established explanation. Keep the OOD sanitizer, but fix and test the synthetic
   validity/fallback boundary before retrying. Do not clamp invalid labels to zero: that would
   turn a failure marker into fabricated training labels.
3. **Non-finite LGD monitoring predictions.** In 597 progress rows across all 30 seed-0/seed-2
   LGD arms, one monitored dataset reports `pred_nonfinite_frac=0.002625` (one of 381 predictions)
   and NaN R2/RMSE/pinball/CRPS. Seed-1 arms retain four finite R2 values; the other seeds retain
   three. NaN-skipping averages therefore change the dataset population between seeds. The table
   above explicitly uses the common three datasets. Cause is unlocalized without the relevant
   inputs/checkpoints; an absent exception or `n_errors` column does not make evaluation clean.
4. **Exp2/Exp3 dispatch still runs Exp1.** A read-only `_train_stage` reproduction for experiments
   1, 2 and 3 emitted the same training-script commands without an experiment/config argument.
   Both `scripts/slurm/pretrain_{pd,lgd}.slurm` hard-code `config/Exp1_*.yaml`, including on
   self-resubmit. Once placeholders are filled, the documented Exp2/Exp3 command would run or
   resume Exp1 instead. Fix initial submission and resubmission, and verify resolved commands.
5. **Selection and provenance gates.** The progress hook selects small datasets including
   holdouts, and `results_plots.overall_ranking` averages all datasets rather than the configured
   dev subset. A dev-only selection report is required. Phase-2 tags are reused across reruns;
   `_benchmark_stage` checks file existence, not checkpoint identity, valid cells or OOD completion.
   Verify/archive older results before scoring the revised grid. Stale cluster result files are a
   risk, not a fact established by this download. Temporal evaluation still raises
   `NotImplementedError` in `src.eval.runner.make_split` despite being offered by the CLI.

All 90 progress CSVs have consistent row widths. One failed PD arm has two step-625 rows after
restart; they are not independent measurements. Resume-time header handling was not exhaustively
validated here. `All_Results.md` and the absent result tier cannot settle the scientific outcome.

### Interpretation

The implementation has substantial working infrastructure and most training has finished, but
neither this sweep nor the research project is finished. One active job is compatible with nine
additional failed arms. Preserve completed checkpoints: the evidence justifies neither discarding
all 80 nor claiming that all are scientifically validated. Close the validity/evaluation gaps,
then use the frozen benchmark and dev-only selection to decide what deserves more compute.

### Next

**Now, on the VSC login node** (read-only; do not add `--submit` yet):

```bash
cd "$VSC_DATA/CreditICL"
source scripts/slurm/_activate_env.sh
squeue -M mindwell -u "$USER"
sacct -M mindwell -j 11591462,11591463 --format=JobID%24,State,ExitCode,Elapsed
python -m src.utils.run_experiment 1
```

Let LGD arm 43 continue if the scheduler and a fresh log tail still show progress. Do not duplicate
it, delete its checkpoints, change its prior, or blindly resubmit the nine PD failures. Last saved
PD steps in logs: 1000,3500,4000,1000,1750,1000,500,3250,2250 for indices
1,4,10,16,19,25,31,34,40 respectively; checkpoint files themselves were not downloaded.

After the invalid-task fix is tested and deployed, validate one failed PD arm with saved CPU-side
batch diagnostics and synchronous CUDA errors. A new cluster run needs Andreas's authorization.
Resume the other failed indices only after that validation.

Next, phase 2 per completed track: **45 final checkpoints plus one shared reference task**, matching
context cap, splits and evaluation seeds, and complete in-domain/OOD result rows. Check provenance
and LGD non-finite predictions before accepting rankings. LGD can be benchmarked independently
once its 45th arm finishes; it need not wait for PD training.

Retrieve the separate results tier too. From the login node, if that results directory exists:

```bash
tar -czf "$VSC_DATA/CreditICL/output/crediticl_results_$(date +%Y%m%d_%H%M%S).tar.gz" \
    -C /lustre1/project/stg_00211/CreditICL/output results
```

Select the prior on development datasets with paired comparisons across seeds. Exp2 is a
60-arm/track, one-seed continued-pretraining sweep from released weights (10,000 steps), followed
by seed confirmation of selected settings. Exp3 is the separate scratch-training confirmation:
winner versus control, 100,000 steps, five seeds (normally ten arms/track). Both still contain
`FILL_FROM_EXP1` and need the dispatch fix. With no clean Exp1 winner, explicitly choose whether
to stop or carry a small candidate set before spending more credits. Resolve temporal/grouped
evaluation and the already-inspected holdout issue before a confirmatory paper claim. Rebuild
notebooks and `All_Results.md` from verified outputs after the evaluation checks.

**Audit artifact:** `output/manifests/exp1_audit_20260923.json`: per-arm state, latest log/checkpoint
evidence, parsing counts, descriptive monitoring calculations and SHA-256 fingerprints for all
826 supplied files. Downloads was read without modification.
**Validation:** controlled CPU fallback/encoder reproduction and read-only launch-command
reproduction completed. `.CreditICL/Scripts/python.exe -m pytest -q`: **893 passed, 2 skipped,
3 warnings in 563.31 s**. Skips: upstream-unavailable case (upstream is installed), and bash syntax
check (bash unavailable locally). The three warnings come from scheduler-order test fixtures.
The existing suite passes despite the reproduced gaps; it does not validate the actual failed
CUDA batches. No notebooks or visualization source changed, so notebooks were not rerun.

## 07-09-2026 — Exp1 phase-1 re-run (90 arms) — **PARTIAL: 63/90 done; every unfinished arm is `filter=banded`; 15 PD banded arms crashed on an out-of-domain `-1` label**

**Submitted** ~03-09-2026 | **jobs** 11549669 (PD, `--array=0-44%8`), 61898850 (LGD) | **cluster** PD mindwell `gpu_b200`, LGD wICE `gpu_a100`
**Commit** 7d1cb02 (git_dirty=true) | **tfm-library pin** 21d555a6a24e | **GPUs** 1×B200 (PD) / 1×A100 (LGD) | **walltime** 12:00:00 req

The re-run after the notebook levelling and `selection_sharpness 0.6→0.35` change. Read from the
download of 07-09 (logs + manifests; results tier not pulled — phase 2 has not run).

### Configuration
- **Config** `config/Exp1_{PD,LGD}.yaml`, 45 arms/track = credit_fraction {0,0.5,1.0} × filter {tabicl,banded,off} × intensity {mild,aggressive} × seeds {0,1,2}, control-deduped at cf=0.
- **Budget** 12,500 steps × 64 datasets = 800,000 datasets/arm.
- **Edited by hand since 02-09:** notebooks restructured; `selection_sharpness` 0.6→0.35. Neither touches training or the model.

### Results
Completion (from `manifests/*__summary.json` `completed` flag, cross-checked against 63 `END status=OK` logs):

| track | completed | banded, still running (partial) | banded, **dead** (crashed) |
|---|---|---|---|
| PD | 30 / 45 | 6 (cf=1, graceful) | **9** |
| LGD | 33 / 45 | 12 | 0 |

- **All 27 unfinished arms are `filter=banded`.** `off`/`tabicl` all finished. Banded rejects ~90 % of
  candidate tables (by design — a strict predictability band), so it runs at **0.24 steps/s vs 0.63** and
  needs many checkpoint→requeue cycles; nothing wrong with it.
- **Progress-eval on the tracked real datasets** (monitoring only — small, 3–4 datasets, NOT the phase-2
  benchmark), completed arms, mean over seeds:
  - PD ROC-AUC: control cf=0 **0.721**, blend cf=0.5 **0.723**, full cf=1 **0.690**.
  - LGD R²: control cf=0 **0.220**, blend cf=0.5 **0.392**, full cf=1 **0.319**.
  - A half-credit blend leads on both; full credit trails (below control on PD). Suggestive of a
    "some domain structure helps, too much over-specialises" shape — but it is the training monitor,
    the cf=1 banded arms are undertrained, and phase 2 is the actual verdict.
- **Throughput** off/tabicl ~0.63 steps/s; banded ~0.24. No OOM anywhere; no NaN.

### Bugs and anomalies
- **15 PD banded arms crashed with a CUDA device-side assert** (`exit 134`,
  `ScatterGatherKernel.cu:409 idx_dim >= 0 && idx_dim < index_size`, surfacing async in the column
  encoder). 9 wrote no summary (killed before the graceful path); 6 (cf=1) wrote partial summaries.
  Because it is a hard abort, not the exit-64 signal path, **no self-resubmit fired** — the arms sat dead.
- **Root cause: a missing out-of-domain classification label.** The only scatter in the model is
  `F.one_hot(y_train.long(), max_classes=10)`; a `≥10` label diverts to mixed-radix, so the trigger is a
  **negative** label. Synthetic PD labels are strictly {0,1} (verified in code, both control and credit).
  The negative comes from the **progress-eval**: `src/eval/ood.py` coded a missing classification target
  with `pandas.cat.codes`, which returns **-1** for NaN. `-1` is finite, so the eval's `np.isfinite`
  filter passed it straight into `one_hot`, whose device-side assert corrupted the CUDA context and
  killed the run a few steps later. **Prior-independent** (so control cf=0 arms crashed too — confirmed:
  a1/a16/a31 are cf=0); **PD-only** (one_hot is the classification path; LGD's y-encoder is linear —
  explains 9 PD vs 1 LGD); **banded-concentrated** only because banded arms run 15–25 h and thus far more
  progress-eval cycles, each a fresh chance to draw a `-1` context row.
- Verified empirically: `pd.Series([...,NaN]).astype("category").cat.codes → [...,-1]`, `np.isfinite(-1)`
  is `True`, and `F.one_hot(tensor([-1]),10)` raises *"Class values must be non-negative"* (the CPU form
  of the device assert). **Fixed** in `src/eval/ood.py`: fetch drops unlabelled rows before coding;
  `load_ood_dataset` strips any surviving `-1` so the existing cache is safe without a re-fetch. Tests in
  `tests/test_ood.py`.
- The lone LGD casualty (a43, cf=1 banded) is a separate, rarer feature-side abort or coincidence; LGD
  cannot hit the one_hot path. Not chased — it finished on resubmit and LGD has no `one_hot`.

### Interpretation
- **Show:** 63/90 arms trained cleanly; the credit prior is sound; the crash is a data-plumbing bug in
  the OOD eval, not in the prior or the model, and not caused by `filter=banded`.
- **Think:** the -1-from-`cat.codes` reaches `one_hot` mainly via a cached OOD dataset with exactly two
  codes `{-1, c}` (one real class + NaN rows); the random per-eval context sample decides whether a `-1`
  row is drawn, which is why the crash step varies across arms. The accidental `np.bincount` guard in the
  progress hook catches the ≥2-real-class case but not this one.
- **Test:** resubmit the 9 dead PD arms with the fix in place; they resume from checkpoint and should now
  run through the OOD eval without aborting.

### Next
Resubmit only the unfinished arms (`sweep_status --resubmit`, after checking `squeue` to avoid
double-submitting a running one); the 9 dead PD arms are indices 1,4,10,16,19,25,31,34,40. Then phase 2.

## 02-09-2026 — Exp1 Phase 2 benchmark — **NULL RESULT: the credit prior does not beat a no-credit control on either track**

**Submitted** 01-09-2026 | **jobs** 61866150 (LGD), 61866151 (PD), preflight 61865246 | **cluster** wICE `gpu_a100` | ~154 tasks, **all END=OK**, ~17:25 → 01:49 (wICE shared, so slower than the ~1 h the compute alone needs)

### Configuration
- **Phase 2** `benchmark.slurm`, `EXP=1`, both tracks. Each of the 75 checkpoints/track scored on the 7 real credit datasets + the 50-dataset OOD suite, 3 eval seeds, 1024-row context cap. Task 75 = the shared reference column (released TabICLv2, TabPFN-3, CatBoost, linear/logistic), scored once.
- **Ranking unit** the 25 distinct priors/track (levers minus training seed), each averaged over its 3 training-seed checkpoints. (Results CSVs are on lustre; this ranking was reconstructed from the per-arm bench logs.)

### Results
LGD (credit R², higher better; typical seed SD **0.008**):

| | credit R² | OOD reg R² |
|---|---|---|
| best credit prior (cf 0.3, atom 0.8, mechanism, scale none) | 0.486 | 0.749 |
| **control (no credit)** | **0.485** | 0.753 |
| released TabICLv2 | 0.514 | 0.785 |
| TabPFN-3 | 0.578 | 0.790 |
| CatBoost | 0.530 | 0.727 |

PD (credit ROC-AUC, higher better; typical seed SD **0.0005**):

| | credit AUC | OOD cls AUC |
|---|---|---|
| **control (no credit) — rank 1** | **0.730** | 0.953 |
| best credit prior | 0.730 | 0.953 |
| released TabICLv2 | 0.736 | 0.956 |
| TabPFN-3 | 0.759 | 0.954 |
| CatBoost | 0.761 | 0.948 |

- **credit vs control:** LGD +0.0004 (1/20 of a seed SD); PD the control is literally rank 1. **No credit benefit on either track.** `credit_fraction` (0/0.1/0.2/0.3) has no effect.
- **0/25 priors beat the released TabICLv2** on either track.
- **The one real lever effect:** `target_scaling=standard` hurts LGD (0.42–0.44 vs 0.48 for `scale=none`; OOD 0.58–0.60 vs 0.75) — a preprocessing artefact (standardising a bounded [0,1] target), not a credit finding.
- **No OOD collapse:** the checkpoints keep general tabular ability (LGD OOD ~0.75 vs 0.785; PD OOD ~0.953 vs 0.956).

### Bugs and anomalies
- The quick log-parser's reference **OOD** number grabbed the classification AUC for the LGD regression row (cosmetic); the **credit** ranking — the actual result — is unaffected, and OOD is quoted here from the preflight reference log.
- The full-run task 75 for LGD correctly SKIPPED (reference already scored in the preflight), leaving an empty log — expected.

### Interpretation
- **Show:** at 12,500 steps from scratch the credit-tailored prior is indistinguishable from the base prior on real credit data, with enough seed precision (SD 0.008 / 0.0005) to have caught a ≥0.005 R² / ≥0.002 AUC effect. Our small-budget models also sit below the released TabICLv2 and the strong baselines.
- **Think:** either the budget (2.5 % of upstream) is too small for prior structure to express itself, or the credit mechanisms add no signal the base graph_scm prior lacks. The fair internal test (credit vs control, identical budget) is a clean null, so "a credit prior helps *from scratch*" is not supported.
- **Test:** Exp2 (continued pre-training from the released weights) is the discriminating experiment — it starts from 0.514 / 0.736 (already above our from-scratch best) and asks whether credit knowledge helps a *strong* model. Exp3 (winning prior run long) has no winner distinct from the control to confirm.

### Next
Decide Exp2 vs Exp3 in light of the null. Only clear Exp1 signal to carry forward: `target_scaling=none` for LGD.

---

## 01-09-2026 — Exp1 full sweep, 150 arms — **COMPLETE: 150/150 trained clean (the 8 stragglers finished on resubmit)**

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
| arms completed | 75 / 75 | 75 / 75 |
| final train loss (range) | 0.051 – 0.070 | 0.136 – 0.186 |
| wall-time / arm | 5.3 h (B200) – 11.8 h (A100) | 5.4 – 5.7 h |

- **Prior ranking** — n/a yet. This is the TRAINING phase; *which prior wins* is Phase 2 (`benchmark.slurm`), not run.
- **Progress curve** converged and flat by the end on every completed arm; no divergence.
- **Throughput** ~0.63 steps/s (B200), ~0.29 (A100); mean GPU util ~88–90 %, compute-bound (fwd_bwd ≈ 97 %).
- **Cost** 1,011 GPU-hours across all 150 arms.
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
