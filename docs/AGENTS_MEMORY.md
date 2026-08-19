# Agents' memory — runs and dead ends

What is worth carrying between sessions: **the cluster runs that have been done**, and **the things
that turned out not to work**. Read it before starting; add to it as you go.

Not the changelog — that records edits to the repository. This records experience: what was run,
what came out, and what is already known to fail.

**Keep it short.** One line per run, four per dead end. Newest first, dates `DD-MM-YYYY`. Never
delete an entry: a run you would otherwise repeat and a dead end you already paid for are both
evidence.

## How a run comes back to you

**After every cluster run, Andreas downloads the output and uploads it into a chat session so an
agent can read and debug it.** Expect to be handed a folder of `.log` files and `.csv` manifests
with no other context, and expect that to be the *only* record of the run — so:

- **Read them properly before theorising.** The `hardware:` line says which GPU it was; the
  `telemetry:` summary at the end of the log says whether the GPU was starved; `grads step` lines
  say whether every block was learning. Most "the model did not learn" questions are answered in
  those three places.
- **Write the run up in [`RUNS.md`](RUNS.md)** using its template, and add the one-line row to the
  table below. The upload is the only chance to capture it.
- **A number that looks too good is a bug until proven otherwise.** Check the config actually
  used (`grid levers:` and `credit_fraction IN USE:`), not the config on disk now.

## Runs

One row per cluster run worth remembering — which is most of them, because *"have we already tried
that configuration?"* is the question this table exists to answer. The **full** write-up of each
one lives in [`RUNS.md`](RUNS.md); this table is the index.

| Date | Run | Outcome | Notes |
|---|---|---|---|
| 17-08-2026 | 11518234/35 + 11518236–40 — benchmark + LGD debug at batch 64 | **B200 SOLVED: 3.5 % -> 88.7 % utilisation, 12x throughput** | The batch was too small to fill the card. Killed at the wall on step 1,422/1,500 |
| 14-08-2026 | 11516936/11516938 — Exp1 debug, 8 arms | **All failed, exit 2 in 21–60 s** | `--resume auto`, a flag `pretrain.py` never defined |
| 14-08-2026 | 11516954 — Exp1 debug LGD, `interactive` | 3/4 arms OK; task 3 deferred for maintenance | **6.6 steps/s, 69 % GPU**, loss 0.339→0.127 |
| 14-08-2026 | 11516956 — Exp1 debug PD, `gpu_b200` | 4/4 arms OK, 12× slower — **cause STILL UNKNOWN** | 0.5 steps/s, 3 % GPU; loss 0.618→0.171, acc 0.95 |
| 14-08-2026 | 11517006–10 — Exp1 debug PD, `interactive` | 6.9 steps/s, **but all 4 arms ended OUT_OF_MEMORY** in the evaluation | First real credit number: `german` ROC-AUC 0.718 |
| 16-08-2026 | 11517081/82 — GPU benchmark, both cards | **B200 faster on EVERY rung** (7.7× bf16 matmul, 1.8× end to end) | Overturns the 14-08 conclusion. Muon hypothesis raised here |
| 17-08-2026 | 11517858/59 — GPU benchmark + optimiser row | **Muon is only 1.34–1.40×, on BOTH cards** | Kills the Muon hypothesis too. B200 8.26 steps/s benchmarked vs 0.53 trained — gap unexplained |
| 16-08-2026 | 11517370–72 + 11517008 — Exp1 debug LGD, `gpu_b200` | `crediticl` **scored at last**; 0.53 steps/s under Muon | **NaN predictions on 3 of 4 LGD datasets — unexplained** |

The 14-08 afternoon run also gave the **first real credit number from our own model** —
`german` ROC-AUC 0.718 at step 1,250 — but `crediticl` still failed everywhere, on a *third*
distinct cause each time: unknown baseline → no checkpoint → **loader built Nano while training
built upstream TabICL**. Staging checkpoint directory still not writable. Full write-up in
[`RUNS.md`](RUNS.md).

## Dead ends

Anything that cost more than a couple of minutes and did not work — including what you eventually
fixed, because the fix is one changelog line and the dead end was the hour.

### 18-08-2026 - Why stages 2-3 are out of reach, and what to do instead
- **Question:** upstream trains in three stages (500k steps at <=1,024 rows, then 40k at 10,240,
  then 10k at 60,000). Should Exp1 do the same, or sweep it?
- **Measured:** no, and it is not close. Row attention is ~quadratic in sequence length and the
  micro-batch drops 4->1, so a stage-2 step costs ~400x a stage-1 step and a stage-3 step
  ~13,700x. **Stage 1 is only 0.3 % of upstream's total compute.** A proportionate curriculum on
  our 12,500-step stage 1 would cost **307x that stage 1** - per arm, times 75 arms.
- **Why it matters anyway:** once `crediticl` went through upstream's wrapper, the context stopped
  being capped, and the wrapper hands over the whole training split - 47,089 rows on `heloc`
  against the <=1,024 we train on. 5 of 7 LGD datasets exceed the training length.
- **Instead:** cap the context to what we trained on, on the SHARED base class so both columns
  get the same cap, stratified for PD, recorded per row. **When you cannot afford to match the
  reference at training time, match it at evaluation time** - and put the knob where it cannot
  be applied to one side only.

### 18-08-2026 - We had built a second inference pipeline without noticing
- **Tried:** scoring our checkpoints with a hand-rolled forward pass (`CreditICLBaseline`),
  while the released TabICLv2 was scored through `TabICLRegressor`.
- **Result:** not comparable. Ours had no preprocessing (which is what overflowed
  `col_embedder.in_linear` into all-NaN), no ensemble, and its own context rules. On the same
  seven LGD datasets in one run: released +0.22..+0.77 R2, ours -1.44..-0.25. No way to say how
  much of that was the prior and how much was our plumbing.
- **Why:** the module docstring said "None of them can load a checkpoint we trained", which was
  never checked. `TabICLClassifier(model_path=<path>)` has existed the whole time; the wrapper
  asserts `config` and `state_dict` and does `TabICL(**checkpoint["config"])`. All that was
  needed was writing their schema.
- **Instead:** `CreditICLBaseline` SUBCLASSES `TabICLBaseline` and overrides only the wrapper
  kwargs. Subclassing rather than duplicating is the point - preprocessing and ensembling
  cannot drift apart because there is only one copy. **Before writing a second implementation
  of anything, check whether the reference one takes a parameter.**

### 18-08-2026 - SOLVED: the LGD NaN was missing feature standardisation
- **Tried:** six explanations over four days - the data, the architecture, divergence, feature
  width, CPU-vs-CUDA, and the attention kernel. Every one wrong.
- **Result:** the GPU walk named it at module **#0**, `col_embedder.in_linear`, the FIRST layer,
  identically on both cards and under all four attention backends. Its output `absmax` was
  **6.550e+04** - `float16`'s largest finite value. The inputs explain the rest:

        axa                absmax 1.98    finite
        heloc              absmax 820     finite
        loss2              absmax 2.4e6   NaN
        base_modelisation  absmax 8.1e7   NaN
        base_model         absmax 9.6e8   NaN

- **Why:** we fed the network **raw, unscaled features** - currency amounts and loan balances up
  to 9.6e8. Upstream never does: `PreprocessingPipeline.fit` starts with
  `CustomStandardScaler().fit_transform(X)` and every prediction goes through
  `preprocessors_[...].transform(X)`. We had imputation but no scaling.
- **Instead:** `standardise_from_context()`, applied in all three inference paths, fitted on the
  CONTEXT rows only so it cannot leak. Verified: 4.7e8 -> 4.8.
- **The lesson about the process, not the bug:** every one of the six wrong answers was
  inferred from a correlation I noticed (width, device, backend). The right answer came from
  the FIRST tool that reported a measurement instead of a hypothesis - a hook on every module,
  printing where the number stopped being finite. **Build the instrument before the fourth
  theory, not after the sixth.**

### 18-08-2026 - The LGD NaN does not reproduce on CPU
- **Tried:** running `diagnose_nan.py` on the login node against the step-600 checkpoint, on
  the three datasets that report 100 % NaN plus `axa` as a control.
- **Result:** **every module finite on all four**, including `base_modelisation` (256 features,
  input absmax 8.1e7) with a sensible output (absmax 1.34). Weights are healthy too - largest
  parameter magnitude 1.31, none non-finite. So the checkpoint is fine and the data is fine.
- **Why the run still NaNs:** the only difference in the log is `device=cpu`. Training and
  evaluation run on CUDA. PyTorch picks the scaled-dot-product-attention backend itself on
  CUDA - flash, mem-efficient, cuDNN or math - and the choice depends on shape and dtype;
  **CPU only ever has `math`**. Upstream treats the choice as load-bearing:
  `--use_flash_attn3 False` in stage 1, `True` in stages 2-3. We do not control it at all.
- **Instead:** `--sdpa-backend` on the diagnostic tries every kernel in turn, and
  `scripts/slurm/diagnose.slurm` runs it on a real GPU. **UNCONFIRMED until that runs** - but
  the lesson is already banked: **reproduce a bug on the hardware that produced it.** A CPU
  "everything is finite" result did not refute a CUDA NaN, it only located it.

### 18-08-2026 - A login node kills unbounded torch on the second dataset
- **Tried:** `python scripts/diagnose_nan.py` on the Genius login node.
- **Result:** the first dataset finished, the second died with
  `terminate called ... std::system_error: Resource temporarily unavailable`.
- **Why:** that is `pthread_create` returning EAGAIN. Torch opens one OMP thread per core on a
  machine with dozens of cores shared by everyone logged in, and login nodes cap threads per
  user. Nothing to do with the model.
- **Instead:** the script sets `OMP_NUM_THREADS` and friends **before importing torch** (they
  are read at import and ignored afterwards), calls `torch.set_num_threads`, and collects
  between datasets. **Any CPU-side tool meant for a login node must bound its threads** -
  and a crash on the SECOND item, after the first succeeded, is the signature of a resource
  cap rather than bad input.

### 18-08-2026 - Why upstream's micro-batch is 4, and why "the GPU has room" is not a reason
- **Tried:** explaining upstream's `--micro_batch_size 4` as a memory limit, then as an
  artefact of small GPUs, then arguing our 88.7 % utilisation left only ~1.1x to gain.
- **Result:** all three wrong. From the pinned dump: `micro_batch_size == batch_size_per_gp`
  in **every** stage - 4/4 at seq 1,024, 1/1 at 10,240 and 1/1 at 60,000 - and
  `Trainer.validate_micro_batch` **raises**: *"All datasets in the micro batch must have the
  same sequence length."*
- **Why:** a micro-batch is stacked into ONE tensor, and datasets share a sequence length only
  within a GROUP. The micro-batch is therefore bounded by the group size, not by memory.
  Raising it past the group crashes; raising BOTH changes the data (fewer distinct sequence
  lengths per update), so it is never a free speed-up. The lever for speed is `batch_size` -
  more groups per update.
- **Also wrong:** `nvidia-smi`'s `utilization.gpu` is the **fraction of TIME a kernel was
  running**, not how much of the chip it used. 88.7 % does not mean 88.7 % of the card, so
  "only 1.1x left" did not follow from it.
- **Instead:** the trainer now REFUSES `micro_batch_size > group_size` with upstream's
  reasoning, and the run card prints both plus the micro-passes per update - which is the
  number that actually decides whether a big GPU stays busy (1 pass -> 3.5 % utilisation,
  16 passes -> 88.7 %). **Read the reference implementation's VALIDATION, not just its
  defaults**: the constraint was written down and I kept theorising instead of looking.

### 17-08-2026 - Two silent heredoc edits, and `bash -n` accepted both
- **Tried:** adding `--micro-batch-size` to `debug_exp1.slurm` with a python heredoc.
- **Result:** it never applied. The first attempt's backslash-continuations did not match the
  file, so `.replace()` was a no-op; the second wrote a **literal backslash-n** into the command. Both
  passed `bash -n` - the first because nothing changed, the second because
  `--micro-batch-size "$MICRO" (backslash-n) --num-workers ...` is still valid shell. A cluster run
  then went out at micro_batch_size 4 on a 183 GB card.
- **Why:** `.replace()` returns the input unchanged when the pattern is absent, and I checked
  only `bash -n` and the test suite, neither of which knew the flag was supposed to be there.
- **Instead:** assert the replacement landed (`assert old in t`), grep the file afterwards, and
  add a test for the thing being added. **A silent no-op edit is worse than a crash**, and
  `bash -n` proves syntax, never intent.

### 17-08-2026 - `git add -A` committed deletions nobody made on purpose
- **Tried:** the usual `git add -A; git commit; git push` at the end of a turn.
- **Result:** the commit **deleted** `output/All_Results.md` (272 lines),
  `output/figures/CAPTIONS.md` (the paper's captions, 133 lines) and every `.gitkeep`. It
  surfaced only when Andreas's `git pull` listed ten `delete mode` lines.
- **Why:** something run locally during the session removed them, and `git add -A` stages
  deletions as readily as additions. `output/` is *mostly* generated and gitignored, so it
  looks disposable - but a handful of files in it are tracked, and the `.gitkeep`s are what
  give a fresh clone somewhere to write.
- **Instead:** `test_tracked_output_files_still_exist` fails if any of them vanish, and
  **read `git status` for `D ` lines before committing** - especially under a directory that
  is mostly ignored, where a deletion does not look out of place.

### 17-08-2026 - How TabICL handles tables wider than it was trained on
- **Question:** our LGD model returns all-NaN on tables with 52-256 features, having trained
  on 3-50 with `max_features: 64`. Does upstream do something special at inference?
- **Answer, from the pinned dump: NO, and it does not need to.** The column encoder is a
  **shared set transformer** (`_tabicl.py`, "employs a shared set transformer"), which is
  permutation-equivariant and takes ANY number of columns - width is not an architectural
  limit. Ensembling permutes features for diversity (`feat_shuffle_method="latin"`); it does
  **not** subsample them. Upstream simply trains at `--max_features 100` and predicts wider.
- **So the NaN is numerical, not structural** - an untrained model returns 0 % NaN at 256
  features, so it comes from learned weights extrapolating far outside their training range.
- **Instead:** matched `max_features` to upstream's 100, which shortens the gap and removes an
  unintended deviation. **Check what the reference implementation actually does before
  assuming a limit is architectural** - "trained on <=64" and "cannot handle >64" are
  different claims, and only the first was true.

### 17-08-2026 - Two of my own fixes combined into a run that trained nothing
- **Tried:** rerunning the LGD debug after `clean_run --clean`, to get a checkpoint.
- **Result:** all four arms `RESUMED from step-1500.ckpt at step 1500`, skipped the loop
  entirely, logged `elapsed_s: 0.0` and **exited 0** — then evaluated the OLD weights.
- **Why:** a two-fix chain. Checkpoints used to land in `$VSC_DATA/output/<run>/checkpoints`
  because staging was mode 0500, and a normal clean removed them. Fixing that permission sent
  them to `checkpoints/`, which `clean_run` deliberately protects for the released weights.
  Neither fix was wrong; together they created the gap.
- **Instead:** `--checkpoints` clears our `exp*/` dirs only, and the trainer now shouts
  `THIS RUN WILL TRAIN NOTHING` when it resumes at `max_steps`. **When you change WHERE
  something is written, re-check everything that decides what gets deleted** - and a resume
  that skips the loop must never look like a completed run.

### 17-08-2026 - Muon was not the answer either, and I wrote a utility that already existed
- **Tried:** (a) blaming Muon for the B200 running training 16x slower than its own benchmark;
  (b) writing `scripts/clean_outputs.sh` to clear run output.
- **Result:** (a) **wrong again.** The `bench_optimizers` row measures Muon at **1.34x** on the
  RTX 5000 Ada and **1.40x** on the B200 - and the B200 is FASTER at Muon in absolute terms
  (62 ms vs 99 ms). It is ahead on every rung. (b) `src/utils/clean_run.py` already existed,
  mandated by `docs/TEMPLATE.md` and listed there as "filled in". The new script was deleted
  unused.
- **Why:** (a) I inferred the cause from the one difference I happened to notice between two
  runs, instead of measuring it - the same mistake as the B200 conclusion before it, one level
  down. (b) I did not check the template before adding a utility, which is step 1 of AGENTS.md.
- **Instead:** the B200 gap is **still open**: benchmark end-to-end 8.26 steps/s, real training
  0.53, same card, same loader, same worker count. The remaining suspects are inside the
  trainer - AMP, micro-batching, telemetry, node contention - and the way to settle it is a
  per-phase timing breakdown in the training loop, not another guess.
  **Two wrong causes in a row from the same habit: name the difference, then MEASURE it.**
  And **read `docs/TEMPLATE.md` before writing a utility** - `clean_run.py`, `paths.py` and
  `run_notebooks.py` are all already there.

### 14-08-2026 — I called a run clean from its logs; `sacct` said OUT_OF_MEMORY
- **Tried:** writing up run 11517006 from the `.out` logs and the manifests, which was all
  that had been uploaded.
- **Result:** reported as successful. `sacct` later showed **all four arms `OUT_OF_MEMORY`,
  exit `0:125`.** The logs give no hint: the OOM killer sends SIGKILL, so nothing writes a
  traceback and the file simply stops mid-evaluation.
- **Why:** a log that ends without an error looks like a log that finished. It is not the same
  thing, and only the scheduler knows the difference.
- **Instead:** **always read `sacct --format=JobID,State,ExitCode,Elapsed` before calling a run
  clean**, and ask for it alongside the logs. Also: fixing preprocessing is what *caused* this
  — with no processed datasets the evaluation had skipped them and used almost no memory.
  **A fix upstream can expose a bug downstream that was never reachable before.**

### 14-08-2026 — One staging directory was mode 0500, and every run silently rerouted
- **Tried:** writing checkpoints to `/lustre1/project/stg_00211/CreditICL/checkpoints`.
- **Result:** `[Errno 13] Permission denied` on every arm of every run, for days. Each one
  fell back to `$VSC_DATA` and finished, so nothing ever failed — which is why it survived.
- **Why:** the directory was **`dr-x------ vsc38338:lp_andreas`** — mode 0500. Owned by us,
  but with no write bit and no group access, while every sibling was `drwxrws--- … SETGID`.
  Something created it from a read-only source; a `cp`/`rsync` out of a HuggingFace cache
  does exactly this, since those files are 444. **The fix was `chmod u+rwx,g+rwxs`.**
- **Instead:** `scripts/check_storage.py`, run before submitting. It writes a real probe file
  (`os.access` lies on Lustre) and prints owner/group/mode for every level, which is what
  turned "permission denied" into "one directory is 0500" in a single screen.
  **A fallback that always works hides the fault it is compensating for** — `$VSC_DATA` is
  75 GiB and Exp1 is 96 arms, so the fallback would have failed exactly when it mattered.

### 14-08-2026 — Two claims I made from reading code, one of which was wrong
- **Tried:** asserting from the source that a 10-wide head would spread probability mass over
  eight absent classes, so Brier and calibration must be wrong.
- **Result:** **false.** `TabICL.forward` returns exactly the classes present in `y_train` —
  a 10-wide head with binary y returns 2 columns, with 5 classes it returns 5. Measured in
  thirty seconds once I bothered to run it. The slice upstream applies is defensive, not
  load-bearing, and no calibration metric was ever affected.
- **Why:** the head is *named* `max_classes` and the parameter count changes with it, so
  "the output is 10 wide" felt like it followed. It does not; the forward slices internally.
- **Instead:** `test_forward_width_follows_the_data_not_the_head` measures it. **A shape is
  one `print` away — never infer one from a constructor argument.** The `max_classes`
  architecture finding below was independently verified by parameter count and stands.

### 14-08-2026 — Quantile crossing was never fixed at decode time
- **Tried:** checking whether LGD's near-constant predictions and negative out-of-domain R²
  were a bug or just 1,500 of 12,500 steps.
- **Result:** undertraining explains the spread, but a real bug turned up alongside it: the
  predicted quantile rows are **not monotone**, and nothing sorted them. Confirmed on an
  untrained model — `np.all(np.diff(q, axis=1) >= 0)` is False.
- **Why:** a quantile head predicts each level independently; nothing ties q_0.4 below q_0.6.
  Everything downstream assumes otherwise — the median is column `Q//2`, coverage counts
  truths between columns, PIT and CRPS integrate across them. All four are wrong on a crossed
  row and none of them *looks* wrong.
- **Instead:** `enforce_monotonic_quantiles`, applied at all three decode points, matching
  upstream's `enforce_monotonicity(..., method="sort")` inside `QuantileDistribution`. NOT in
  the pinball loss — that is per-level by design. **When the library has a function for
  something, find out where it is called, not just what it does.**

### 14-08-2026 — The architecture was quietly made a function of the prior
- **Tried:** nothing — this was found while writing a round-trip test, not from a run.
- **Result:** PD trained a **27,538,938**-parameter model where TabICLv2's classifier is
  **27,552,258**. `Trainer._build_model` did
  `mcfg.setdefault("max_classes", prior.n_classes)`, so the head width came from the PRIOR —
  the one thing this project varies. The difference is exactly 13,320, the four head tensors.
- **Why:** `max_classes` reads like a data property (how many classes are there?) and is
  actually an architecture property (how wide is the head?). Upstream settles it: every
  classifier stage script passes `--max_classes 10`, and `_compute_batch_loss` slices
  `logits[..., :n_classes]` before cross-entropy. **Head width is architecture, class count is
  data.**
- **Instead:** `max_classes` is never set from config; the loss and both prediction paths slice
  to the classes actually present. Two tests pin it — the parameter count, and the absence of
  the `setdefault`. **An architecture that depends on the independent variable makes every
  result unattributable**, and nothing in a loss curve would ever show it.

### 14-08-2026 — Three bugs in a row between training and evaluation, each hidden by the last
- **Tried:** evaluating our own checkpoints, three submissions running.
- **Result:** `crediticl` scored nothing all three times, with a *different* error each time —
  (1) `unknown baseline`, (2) `needs checkpoint=<path>`, (3) `does not match the architecture`.
  Each fix revealed the next, and every failure was masked by `|| echo WARNING` so the job
  still reported success.
- **Why:** the training→evaluation seam had no end-to-end test. Registration, checkpoint
  resolution, and architecture reconstruction were each fine in isolation; nothing checked that
  a checkpoint written by the trainer could be read back by the evaluator.
- **Instead:** a **round-trip test** — build via `build_model`, save, `load_our_checkpoint`,
  assert identical keys and equal weights, for both tasks and both architectures. That single
  test would have caught bug 3 immediately and is the only kind that can.
  **When two halves of a pipeline are written separately, test the seam, not the halves.**

### 14-08-2026 — A NaN in dataset 2 discarded datasets 3, 4 and all eight OOD suites
- **Tried:** reading the progress curve from the PD-on-free run.
- **Result:** one real dataset in the CSV and an `error` cell reading `ValueError: Input contains
  NaN.` — no `ood__` columns at all, where the previous run had eight. It read as a smaller
  measurement rather than a failure.
- **Why:** a single `try` wrapped BOTH the real-dataset and out-of-domain loops, so the first
  exception skipped everything remaining. sklearn's message names neither the dataset nor
  whether the NaN was in the label or the prediction.
- **Instead:** per-dataset isolation, `n_errors` in the row, every error logged with its
  dataset name, non-finite targets dropped and counted, and non-finite *predictions* reported
  as `pred_nonfinite_frac` rather than raising. **An `except` around a loop converts one
  failure into total data loss — put it inside.**

**The table starts empty on purpose, and that is not the same as "nothing has run".** Submission
attempts were made on wICE before 11-08-2026 and their logs were read to fix the bugs recorded
below, but no row was ever written and those logs are no longer on disk, so there is nothing here
that can be trusted. **Add a row for every run from now on**, at submission time rather than
afterwards — a job that dies at the walltime never comes back to write its own entry.

- **Run** — the config or arm, and the commit if it matters.
- **Outcome** — `done` / `walltime` / `OOM` / `crashed` / `diverged` / `cancelled`.
- **Notes** — one line: the headline number, the output path if it is worth finding again, or the
  single thing the run showed. A number here saves re-reading `output/results/`.

### 16-08-2026 - The B200 was never slow. Muon was, and only on the B200
- **Tried:** concluding from two training runs that the B200 was 12x slower than a free
  RTX 5000 Ada, and telling Andreas to stop paying for it.
- **Result:** **wrong.** The benchmark has the B200 ahead on every rung - 7.7x on bf16
  matmul, 1.8x end to end. The training runs differed from the benchmark in exactly one
  respect: `config/Exp1_*.yaml` sets `optimizer: muon`, and my benchmark used plain SGD.
  B200 + Muon = 0.53 steps/s; B200 + SGD, same model, prior and loader = 8.14.
- **Why:** I built a six-rung ladder and left out the rung that mattered, because the
  optimiser did not feel like part of "is the GPU slow?". Muon's Newton-Schulz iteration is a
  chain of tiny matmuls per weight matrix per step - latency-bound, so peak FLOPs do not help.
- **Instead:** `bench_optimizers` times AdamW against Muon on the actual model. **A benchmark
  must run the CONFIGURED code path, not a simplified stand-in** - the simplification is
  exactly where the answer hid. Related: `src/train/optim.py`'s docstring still claimed "we
  use AdamW, not Muon" from before Muon was vendored, so the file documented the wrong
  optimizer for weeks; a test now ties it to the configs.

### 16-08-2026 - A diagnostic that cried wolf about the healthy card
- **Tried:** flagging a JIT fallback with `f"sm_{major}{minor}" in torch.cuda.get_arch_list()`.
- **Result:** printed `*** NO KERNELS FOR THIS CARD` for the RTX 5000 Ada (`sm_89`, list had
  `sm_86`) - the card that was *fine* - and stayed silent for the B200, the slow one.
- **Why:** cubins are binary-compatible across MINOR versions within a major architecture, so
  `sm_86` runs on `sm_89`. The exact-string test does not model that.
- **Instead:** test the major version. **A false alarm is worse than no alarm**: it points at
  the wrong suspect with an authoritative voice.

### 14-08-2026 — `CONFIG=` in front of a wrapper script does not reach the job
- **Tried:** `CONFIG=config/Exp1_PD.yaml bash scripts/slurm/submit.sh free ...`, then
  `bash scripts/slurm/submit.sh b200 ...` for the same PD run.
- **Result:** the first was rejected by a QoS limit; the second submitted a **second LGD run**.
  Nothing on screen said which config either job carried, so it looked like a PD job had gone
  out when none had.
- **Why:** `VAR=x bash script.sh` sets the variable for *that shell*, not for `sbatch`'s
  environment, so `--export=ALL` inside the job script had nothing to propagate. And
  `submit.sh` printed the partition and resources but never the config.
- **Instead:** the env var is gone. `sbatch [options] script args…` passes trailing arguments
  straight to the job script, so `submit.sh <where> <track>` resolves the track to a config path
  and passes it as `$1`; the job reads `CONFIG="${1:-config/Exp1_LGD.yaml}"`. An unrecognised
  track is refused rather than defaulting. Two lessons, both general: **prefer an argument over
  an environment variable** whenever a value must cross a process boundary, and **print every
  input that changes what a job does**, not just the ones passed as flags.

### 14-08-2026 — A flag the job script passes but the Python script does not define
- **Tried:** the first real debug array — 4 LGD arms on `interactive`, 4 PD arms on `gpu_b200`.
- **Result:** **all eight finished in 21–60 seconds** and the dashboard showed them as
  *Completed*. Nothing trained. 1,500 steps cannot run in 21 seconds; that is torch's import
  time and nothing else.
- **Why:** `debug_exp1.slurm` called `pretrain.py … --resume auto`, and `pretrain.py` defines no
  `--resume`. argparse prints a usage message and exits 2 before any of its own code runs.
  Resuming was **already automatic** — `trainer.maybe_resume()` is called unconditionally — so
  the flag was describing behaviour that needed no flag. Nothing catches this: the job scripts
  are shell text, so no import, linter, or test touched them.
- **Instead:** flag removed, and `tests/test_slurm_scripts.py` now parses every
  `python scripts/*.py` call in `scripts/slurm/` and checks each flag against that script's
  `add_argument` calls. Verified by reintroducing the bug and watching the test fail.
  **A very short "successful" job is a failed job** — check walltime against what the work
  should plausibly take before believing a green status.
- **Second bug found in the same block:** the training call ran under `set -e`, so a crash
  killed the script outright — `STATUS=$?` was dead code, the `if [ "$STATUS" -eq 0 ]`
  evaluation guard could never be false, and the artefact summary at the end (most useful
  exactly when a run has just failed) never printed. Now wrapped in `set +e` / `set -e`.

### 14-08-2026 — A registry entry with no production caller
- **Tried:** found while the rerun trained — checking that the evaluation half of
  `debug_exp1.slurm` would work, since it had still never executed.
- **Result:** it would not have. `--models crediticl,tabiclv2` names `crediticl`, which
  `src/eval/crediticl_baseline.register()` adds to the registry — and the **only caller of
  `register()` in the whole repository was `tests/test_eval.py`**. In the job it would have
  raised an unknown-baseline error, swallowed by `|| echo "WARNING: ..."`. The arm would have
  trained for 40 minutes and scored every model except ours.
- **Why:** `register()` is explicit by design so a broken checkpoint cannot stop the external
  baselines. Explicit means forgettable, and a test calling it hides that no one else does.
- **Instead:** `register_or_warn()`, called by both evaluation entry points, plus a test that
  every name any SLURM script passes to `--models` resolves against `BASELINES`.
  **A test that calls the setup itself proves nothing about the production path.**

### 14-08-2026 — There are TWO separate QoS limits, and the second one is on CPUs
- **Tried:** a 4-arm array on Mindwell `interactive` at `--cpus-per-task=8`, after the
  job-count limit below had already been worked around.
- **Result:** it was accepted, but only `_0` ran — `_[1-3]` sat on `QOSMaxCpuPerUserLimit`.
- **Why:** `QOSMaxSubmitJobPerUserLimit` caps how many jobs you may *queue*;
  `QOSMaxCpuPerUserLimit` caps how many cores you may *use at once*. On `interactive` the
  second cap is reached by a single 8-core task, so an array there is serial no matter how
  many GPUs the partition has.
- **Instead:** expect an N-arm array on `free` to take N × walltime end to end — acceptable,
  since it is free. When several arms must finish together, use `b200`, which is on the
  `normal` QoS and ran all four concurrently on one node. Both caps come from
  `sacctmgr show qos interactive,normal format=Name%20,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30`.

### 14-08-2026 — Queue limits are per-QoS, and an array counts as several jobs
- **Tried:** submitting an LGD 4-task array and a PD 4-task array to Mindwell `interactive`.
- **Result:** the second was refused with `QOSMaxSubmitJobPerUserLimit`.
- **Why:** each partition group has a QoS (`interactive`, `debug`, `long`, `normal`) capping
  how many jobs one user may have queued. The first array had already used the allowance.
- **Instead:** send the second job to a partition on a *different* QoS (`b200`/`a100` are on
  `normal`), or submit fewer arms with `--array=0-1`, or wait. The real numbers come from
  `sacctmgr show qos debug,interactive,long,normal format=Name%20,MaxSubmitJobsPerUser%15`.
  `submit.sh` now prints this when it sees the rejection.

### 13-08-2026 — `module --force purge`, and pinning a Python module by name
- **Tried:** `setup_venv.sh` opened with `module --force purge` and then
  `module load Python/3.12.3-GCCcore-13.3.0`.
- **Result:** Lmod: *"These module(s) or extension(s) exist but cannot be loaded as
  requested"* — on login-1, while the same module had worked on login-2.
- **Why:** two causes stacked. (1) The `cluster/*` modules on VSC are **sticky** and set up the
  architecture-specific `MODULEPATH`; `--force purge` removes them too, collapsing the tree so
  a module that exists cannot be resolved. (2) Module trees are **per-architecture**
  (`/apps/leuven/rocky9/<arch>/<toolchain>/…`), so a name present on skylake can be absent on
  another login node.
- **Instead:** plain `module purge`, then **discover** — try the preferred pins, else
  `module -t avail Python/3.12 Python/3.11` and take the newest in range. The chosen name is
  written to `.python_module` beside the venv so activation reloads exactly the interpreter the
  venv was built on. The venv is also arch-suffixed (`.venv-$VSC_ARCH_LOCAL`), which the
  project's own `docs/VSC.md` had already required and the first version ignored.
- **Cheap check that would have caught it:** `module -t avail Python` on the node in question,
  before writing any name into a script.

### 12-08-2026 — Three things the first cluster attempt found in ten minutes
- **Tried:** the documented first-run sequence on wICE — `fetch_ood`, then the smoke test.
- **Result:** (1) `fetch_ood` died on the FIRST dataset with `FileNotFoundError` on
  `3.kr-vs-kp.npz.tmp`; (2) the smoke test died at optimizer construction with
  `ModuleNotFoundError: pytorch_optimizer`; (3) a notebook cell that ran locally would have
  crashed with `unhashable type: 'dict'`.
- **Why:** (1) `np.savez_compressed` given a PATH not ending in `.npz` silently **appends** the
  extension, so it wrote `...npz.tmp.npz` and the rename found nothing. (2) VSC runs torch 2.8
  — no `torch.optim.Muon` — and the published `tabicl` wheel does not ship its training
  package, so there was no Muon at all. (3) a `{{...}}` left over from a `.format()` template
  is **valid Python** (a set containing a dict) until it executes.
- **Instead:** write through an open handle; vendor upstream's Muon from the pinned dump
  (`src/train/_muon_vendored.py`) so no pip package is needed and it is the exact optimizer
  that trained the released checkpoints; and `test_every_notebook_cell_compiles` now compiles
  every cell. **The lesson that generalises: none of the three could fail locally.** Local had
  a newer torch, a warm cache and a different notebook on disk. Run the documented sequence on
  the cluster before believing it works.

### 11-08-2026 — Vendoring NanoTabICL instead of TabICL itself
- **Tried:** built the model from `NanoTabICL.txt` (665 lines) because it is small and readable,
  then tried to load the released TabICLv2 checkpoint into it for the warm-start experiment.
- **Result:** **zero of 390 parameter names matched** any of the checkpoint's 347, and our model
  had 44 extra LayerNorm bias tensors. `load_state_dict` loads nothing and does not complain.
- **Why:** NanoTabICL is a *reimplementation*, not the code that wrote the checkpoint. The bias
  gap is the `--norm_type layernorm_nobias` flag from upstream's own stage script, which a
  reimplementation without that switch cannot express.
- **Instead:** `tfm-library/repositories/TabICL.txt` is the **real** repository — 127 files,
  32,794 lines, including `_model/tabicl.py`, the whole `prior/`, **and** `train/_run.py`,
  `train/_muon.py` and `scripts/train_v2_reg_stage{1,2,3}.sh`. Use their model and loop; keep our
  prior as the thing that changes. Cheap check: count matching `state_dict` keys before writing
  any adapter — one line, and it would have ended this in a minute.

### 11-08-2026 — Running anything with the bare `python` on this machine
- **Tried:** `python -m src.utils.run_notebooks` from the repo root.
- **Result:** all 3 notebooks failed with `ModuleNotFoundError: No module named 'torch'`.
- **Why:** bare `python` on this machine is `C:\Python314\python.exe` — **Python 3.14**, outside
  the project's own `>=3.11,<3.13` pin, and with none of the dependencies. The project venv is
  `.CreditICL/`, which was never on PATH.
- **Instead:** always `.CreditICL/Scripts/python.exe`. Check `sys.version` before believing an
  import error is a code bug.

### 11-08-2026 — Every figure drawn at roughly twice the page width
- **Tried:** hard-coded `figsize=(12, 4.6)`, `(13, 3.8)`, `(11.5, 7.4)` in the plot modules, which
  look right on screen.
- **Result:** the A4 text block is **6.30 in**. Every figure would be scaled to ~50 % in the
  document, taking 9pt text to ~4.5pt — below the ~7pt floor for print.
- **Why:** a screen has no fixed width, so nothing pushes back until the figure is on a page.
- **Instead:** sizes come only from `style.figsize` / `grid_figsize` / `row_figsize`.
  `test_no_plot_function_hard_codes_a_figure_size` and `test_every_saved_pdf_fits_the_a4_text_block`
  now measure the PDFs themselves.

### 11-08-2026 — `results_dir()` on the wrong storage tier
- **Tried:** `results_dir()` returning `outputs_dir() / "results"`.
- **Result:** per-row predictions would have gone to `$VSC_DATA` (75 GiB) instead of project
  storage, filling it — and once it is full, every job that writes a log fails too.
- **Why:** `outputs_dir()` is correct for logs, manifests and figures, which are small. `results/`
  is the one part of `output/` that is not.
- **Instead:** `results_dir()` honours the staging root. `test_results_are_the_one_part_of_output_on_project_storage`
  asserts both halves, because moving *everything* to staging is the opposite mistake.

### 11-08-2026 — A test writing into the real `output/` tree
- **Tried:** isolating a figure test with `monkeypatch.setenv("CREDITICL_STAGING_ROOT", tmp_path)`.
- **Result:** two 1-byte fake PDFs left in `output/figures/`, which the A4 size check then read
  as real figures with no MediaBox. The failure surfaced three test files away from its cause.
- **Why:** `figures_dir()` hangs off `outputs_dir()`, which **ignores staging** — only `results/`
  follows it. Setting the staging root alone isolates a quarter of the tree.
- **Instead:** the `isolated_output` fixture sets `VSC_DATA` *and* the staging root. Use it for any
  test that writes.

### Earlier — SLURM dependency chain spanning three clusters
- **Tried:** one submit script chaining wICE → Genius → Mindwell → wICE.
- **Result:** the chain broke; the user's actual error was a Genius stage with **no `--partition`**.
- **Why:** VSC job dependencies **do not cross clusters**, and Genius needs an explicit partition.
- **Instead:** all of Phase 1 stays on wICE. A chain that spans clusters has to be resubmitted by
  hand at the boundary, so do not build one.

### Earlier — `sbatch --parsable` in a dependency expression
- **Tried:** `dep=$(sbatch --parsable job.slurm)`.
- **Result:** malformed dependencies — the value is `jobid;cluster`, not `jobid`.
- **Instead:** `sbatch --parsable "$@" | cut -d';' -f1` (`scripts/slurm/submit_pipeline.sh`).

### Earlier — `head -3` of a traceback in the env check
- **Tried:** one combined import in `_activate_env.sh`, printing `head -3` on failure.
- **Result:** the three lines printed are exactly the boilerplate; the **fourth** line is the
  `ModuleNotFoundError` naming the package. The first real cluster run was undiagnosable.
- **Instead:** test each package separately and name every missing one.

### Earlier — Bounding `atom_prob` by narrowing the sampling ranges
- **Tried:** tightening the lognormal parameter ranges so recovery would not exceed exposure.
- **Result:** `atom_prob=0.0` still produced 57 % boundary mass; 0.8 gave 93 %.
- **Why:** a lognormal has an **unbounded right tail**. Narrowing ranges cannot bound it — the
  tail is always clipped to the boundary, so mass appears at a boundary that was asked for to be
  empty.
- **Instead:** an explicit `interior_only` clamp (`src/prior/targets/mechanisms.py`). Now
  0.0 → 0 %, 0.6 → 58 %, 1.0 → 99 %.

### Earlier — Out-of-domain regression scored against a `[0,1]` clip
- **Tried:** running the OOD regression suite with raw targets.
- **Result:** a standard-normal target scored **R² = 0.34 on a perfectly linear relationship**.
- **Why:** the regression baselines clip to `[0,1]` because LGD is a loss fraction. An
  arbitrary-scale target is destroyed by that clip, for a reason unrelated to generality.
- **Instead:** min-max the target from **train-only** statistics (`src/eval/ood_runner.py`); R² is
  invariant to a consistent affine transform. Test statistics would leak the test distribution.

### Earlier — Collateral recovery capped before costs
- **Tried:** capping recovery at exposure, then subtracting workout costs.
- **Result:** LGD could never reach exactly 0, so the full-recovery atom was unreachable.
- **Why:** economically backwards — costs come out of the proceeds, and only then is the result
  capped at what is owed.
- **Instead:** costs first, cap second.
