# Changelog

One chapter per date, `DD-MM-YYYY`, newest first. Terse: what changed, and why if the
reason is not obvious.

---

## 18-08-2026 — the micro-batch rule, a run card, and the first clean end-to-end run

- **`diagnose_nan.py` bounds its threads** — `OMP_NUM_THREADS` and friends set before torch is
  imported, plus `torch.set_num_threads`. Unbounded, it died on the second dataset with
  `std::system_error: Resource temporarily unavailable`, which is a login-node thread cap
  hitting `pthread_create`, not a model problem.
- **`scripts/diagnose_nan.py`** — walks a checkpoint with forward hooks and names the FIRST
  module whose output goes non-finite, with the activation magnitudes leading up to it and a
  weight-health report. It runs on the CLUSTER: a checkpoint is ~229 MB and awkward to move, a
  log is not. Verified against a planted NaN — it pointed at the right block, and reports a
  healthy model as healthy.

- **The trainer refuses `micro_batch_size > prior.grouping.group_size`.** Upstream keeps
  `micro_batch_size == batch_size_per_gp` in every stage and `validate_micro_batch` raises when
  a micro-batch mixes sequence lengths; we had no check. It is **not** a memory rule: datasets
  share a length only within a group, so the micro-batch is bounded by the group, and raising
  both changes the DATA rather than buying speed. `batch_size` is the speed lever.
- **Correction:** `nvidia-smi utilization.gpu` is the fraction of TIME a kernel was running,
  not how much of the chip was used — so the earlier "88.7 %, only ~1.1x left" did not follow.
- **A RUN CARD at startup** — architecture, parameters, batch/micro/micro-passes-per-update,
  optimiser and LR, the full prior configuration, a side-by-side against TabICLv2 stage 1
  (including the % of their data we see), and an explicit leakage section. The cluster log is
  the only channel back, so anything answerable at startup is now answered at startup.

## 17-08-2026 — Muon ruled out, 21 duplicate runs removed, batch and features matched to upstream

- **THE B200 IS SOLVED: batch 4 was too little work to fill it.** 3.5 % GPU utilisation and
  2.2 datasets/s at batch 4; **88.7 % and 25.6 datasets/s at batch 64**, same card. Five
  explanations were proposed and measured wrong first — the card, the prior, Muon, the wheel,
  AMP (which is 0.49x, i.e. it makes the B200 *faster*). Fixed as a side effect of matching
  upstream's batch.
- **`DEBUG_STEPS` 1500 -> 600.** The debug budget is `steps x batch_size`, so raising the batch
  to 64 silently turned 1,500 steps into 96,000 datasets; job 11518236 was killed at the
  one-hour wall on step 1,422, losing the evaluation and the checkpoint. A test now checks the
  step budget against the configured batch and the measured rate.
- **`--micro-batch-size` actually reaches the job now.** Two earlier heredoc edits silently
  failed to apply — one whose continuations did not match, one that wrote a literal
  backslash-n — and `bash -n` accepted both. (The per-partition values it originally set were
  themselves wrong; see 18-08 — the micro-batch is bounded by the prior's group size, so it
  stays at upstream's 4.)

- **Restored `output/All_Results.md`, `output/figures/CAPTIONS.md` and every `.gitkeep`.**
  Something run locally this session deleted them and a `git add -A` recorded the deletion, so
  the loss only showed up in someone else's `git pull`. `test_tracked_output_files_still_exist`
  now fails if any of them go missing, and the `.gitkeep` files matter beyond tidiness: without
  them a fresh clone has nowhere to write.
- **`scripts/slurm/pilot_batch.slurm`** — the control arm at batch 16 / 64 / 256, full 12,500
  steps, one seed, three array tasks. Settles batch size before Exp1 instead of sweeping it.

- **`batch_size: 64`, `micro_batch_size: 4`** — upstream's own `--batch_size 64
  --micro_batch_size 4`, in Exp1 and Exp2. It was 4 with no accumulation: an effective batch
  **sixteen times smaller than upstream's**, while using upstream's learning rate of 8e-4,
  which was tuned for 64. Large batches tolerate large rates; small ones do not, and that
  mismatch is the leading suspect for the NaN predictions.
- **`max_features: 100`**, also upstream's. It was 64 — a second unintended difference from
  "the same prior, plus credit structure", and it widened the extrapolation gap on real credit
  tables, which run to 256 columns.
- **`--profile`** on `benchmark_gpu.py` — `torch.profiler` over real steps, top ops by CUDA
  time, run with AMP and without. Four causes for the B200 gap have now been asserted and all
  four were wrong; a profile names the kernel instead of inferring it.
- **The telemetry summary reports memory headroom** and suggests a bigger micro-batch when
  peak allocation is under half the card. Utilisation says whether the GPU was busy; headroom
  says whether it was full, and only the second decides the micro-batch. The starvation
  warning now points at the `phases step` lines instead of always blaming `num_workers` —
  a high `fwd_bwd=` share means more workers will not help.
- **`--batch-size` on `pretrain.py`**, so batch size can be settled on the CONTROL ARM (3
  runs) rather than swept across all 75 (150 runs). Sweeping it would halve the power of the
  comparison Exp1 actually asks, and at a fixed learning rate it would be confounded anyway —
  batch size and LR are coupled. A test asserts no `Exp*.yaml` sweeps a batch size or a
  learning rate: optimisation settings are nuisance parameters, held fixed so the prior is the
  only thing that varies.
- **`--micro-batch-size`** on `pretrain.py`. The claim made here — "purely speed and memory,
  so raise it on a bigger card" — is **wrong** and was corrected on 18-08: the micro-batch is
  bounded by `prior.grouping.group_size`, because datasets share a sequence length only within
  a group. Gradient accumulation does average `loss / n_micro`, so the arithmetic is identical;
  what is not identical is which datasets can legally be stacked together.

- **The sweep no longer schedules duplicates: Exp1 is 75 runs, not 96.** At
  `credit_fraction: 0.0` nothing comes from our prior, so `atom_prob`, `mode` and
  `target_scaling` are dead — the grid was running **eight identical control arms per seed**.
  Same seed and same effective config is bit-identical output, so those 21 arms were 22 % of
  the budget for zero information. `effective_fingerprint()` drops `prior.credit` when the
  fraction is zero and dedupes on what a run will actually do.
- `debug_exp1.slurm` re-resolved: the quantile arm moved **11 -> 9**.
  `test_debug_arms_point_where_the_comments_say` now checks each pinned index against the
  grid, so the mapping cannot silently drift again.
- **Per-phase timing answered the B200 question halfway**: 98.4 % of a 1.8 s step is inside
  forward+backward, with data at 0.0 %, optimiser 1.4 %, telemetry 0.0 %. Not starvation, not
  Muon, not the loader. `bench_model` measured the same fwd+bwd at 43.7 ms — but at batch 1
  in fp32, where training uses batch 4 with bf16 autocast. **`bench_amp`** now measures both
  at the real batch size.

- **`clean_run --checkpoints`** clears OUR `exp*/` run directories under `checkpoints/`, never
  the released `*.ckpt`. Job 11517891's four arms all found a step-1500 checkpoint from the
  previous run, resumed at `max_steps`, trained nothing and **exited 0 in two seconds** while
  the evaluation scored the old weights. Caused by a chain of two of my own fixes: checkpoints
  used to fall back to `$VSC_DATA` (staging was mode 0500) where a normal clean removed them;
  fixing the permission sent them to the one directory `clean_run` protects.
- **The trainer refuses to be silent about it**: resuming at or past `max_steps` now logs
  `THIS RUN WILL TRAIN NOTHING`, names the checkpoint directory, and gives the fix.

- **`phases step N  data=..%  fwd_bwd=..%  optimizer=..%  telemetry=..%`**, logged every
  `log_every` steps, with a verdict when the data wait is decisive. Three explanations for the
  B200 gap have been proposed and two measured wrong; this replaces the fourth guess with a
  measurement. No `cuda.synchronize()` — `float(loss.detach())` already is one.
- The `RUNS.md` archive command now uses `--ignore-failed-read`: `output/manifests` does not
  exist until a run has written one, and plain `tar` exits 1, so the archive meant to protect
  a run was never created.

- **Correction: Muon costs 1.34× on the RTX 5000 Ada and 1.40× on the B200** — measured by the
  new `bench_optimizers` row, on both cards. It is *not* the 15× the previous entry blamed it
  for, and the B200 is faster at Muon in absolute terms (62 ms vs 99 ms per step). The B200 is
  ahead on every rung, including AdamW and Muon. **What real training does to lose 16× on that
  card is still unexplained** — see `RUNS.md`.
- **`clean_run.py` now protects `prior_cache/ood/`** even under `--prior-cache`. The
  out-of-domain cache lives under the prior-cache root only because that is where big things
  go; it is not a prior pool, and compute nodes have no outbound internet, so wiping it means
  a trip to a login node. `protected_paths()` + a test.
- Added a "copy down before clearing" step to the `RUNS.md` checklist: the LGD NaN is
  currently undiagnosable because the checkpoint was deleted before it was copied.
- *(A `scripts/clean_outputs.sh` written earlier this session was deleted unused —
  `src/utils/clean_run.py` already existed, per `docs/TEMPLATE.md`, and does the same job
  better. Check the template before adding a utility.)*

## 16-08-2026 — the B200 was never the problem (Muon suspected, wrongly — see 17-08)

- **`bench_optimizers`** in `scripts/benchmark_gpu.py` — times AdamW against Muon on the real
  model and reports the ratio. This rung was missing, and it was the answer: the first
  benchmark used plain SGD and had the B200 **1.8× faster** end to end, while real training
  with `optimizer: muon` ran at 0.53 steps/s against the free card's 6.6.
- **`src/train/optim.py` docstring corrected.** It claimed "Deviation: AdamW, not Muon" from
  before Muon was vendored, while every config sets `optimizer: muon`.
  `test_optim_docstring_matches_the_configs` now ties the two together.
- **`has_kernels_for_this_card` tests the MAJOR architecture.** The exact-string test called
  `sm_89` unsupported against a list containing `sm_86` and printed `NO KERNELS` for the
  *healthy* card while staying silent for the slow one. Cubins are compatible across minor
  versions. `exact_arch_shipped` is reported separately.
- `docs/RUNS.md` — 16-08 entry; the 14-08 "the B200 is the bottleneck" entry marked
  **SUPERSEDED**. `AGENTS_MEMORY.md` runs table corrected, and its dead-end entries moved into
  the Dead ends section where earlier edits had stranded them under Runs.

## 15-08-2026 — the evaluation ran out of memory once the data was actually there

- **`EvalConfig.max_rows`** — cap rows per dataset *before* the split, as a seeded random
  subsample, recording `row_cap` and `n_rows_full` per row. All four arms of job 11517006 were
  killed by the OOM killer on 30 GB: the 14 PD datasets are 2.4 GB and `0014.algorithmwatch`
  alone is 1.8 GB, which the loop then copies several times. It only became reachable once
  preprocessing succeeded — before that the evaluation had no datasets to load.
- **`--max-rows` on `evaluate.py`**, set to 20,000 by the debug job. **Default is `None`** and a
  test enforces that no `Exp*.yaml` carries it: a capped evaluation is a plumbing test, not a
  result.
- **`docs/RUNS.md` corrected** — that run was written up as clean from its logs, and `sacct`
  says `OUT_OF_MEMORY`. The OOM killer sends SIGKILL, so the log just stops.

## 14-08-2026 — The first cluster runs, and the pipeline debugged end to end

- **The staging `checkpoints/` directory was mode `0500`** (`dr-x------`), owned by us but with
  no write bit, while every sibling was `drwxrws--- SETGID`. Fixed on the cluster with
  `chmod u+rwx,g+rwxs`; `check_storage.py --fix` now does it. Runs no longer reroute to
  `$VSC_DATA`.
- **`check_storage.py` no longer probes a made-up `prior_cache/<name>` subdirectory** — it
  tested a path no run uses, and `--fix` then created it, leaving a stray directory on project
  storage. It checks the cache root instead.
- **A missing directory is no longer reported as broken** when its parent is writable: the
  pipeline creates it on first use, and flagging it sent the reader chmod-ing a path that was
  always going to work.
- **`compare_gpubench` recognises an unexpanded glob.** The shell passes the pattern through
  literally when nothing matches, so "need at least two JSON files, got 1" appeared while
  *neither* job had finished. It now says the files do not exist yet and prints the `squeue`
  and `sacct` commands to check. One result also prints its own column rather than nothing.
- `check_storage.py` added to the pre-submission checklist in `RUNS.md`.

- **`enforce_monotonic_quantiles`** in `src/train/loop.py`, applied at every LGD decode point
  (`predict`, `predict_quantiles`, and the progress scorer). A quantile head predicts each
  level independently, so `q_0.4 > q_0.6` happens — measured on an untrained TabICL LGD model,
  the raw rows are not monotone. Column `Q//2` is the median only once crossing is fixed, and
  coverage, PIT and CRPS all read the grid as ordered. Upstream does this inside
  `QuantileDistribution` (`enforce_monotonicity(..., method="sort")`); we did not.
  **Not** applied to the pinball loss — that is evaluated per level, as upstream trains it.
- **`scripts/check_storage.py`** — writes a real probe file into every directory the pipeline
  uses, walks the owner/group/mode of each level, and prints the exact `chmod`/`mv` to run.
  `--fix` repairs what it safely can.
- **Correction to the previous entry.** The logit slice is a **no-op** with `tabicl`:
  `forward` already returns exactly the classes present in `y_train` (measured — a 10-wide
  head with binary y returns 2 columns). Probability mass was never leaking, so Brier and
  calibration were **not** wrong. The slice stays as a guard, matching upstream, and
  `test_forward_width_follows_the_data_not_the_head` pins the behaviour it depends on.
  The `max_classes` architecture fix is unaffected and still stands.
- `0010.thomas` has **no NaN** in features or targets, and an untrained model produces no NaN
  on it. The `Input contains NaN` came from the trained weights at step 1,250, not the data;
  `pred_nonfinite_frac` will now say so instead of aborting the measurement.

- **`max_classes` is no longer set from the prior.** `Trainer._build_model` forced it to
  `prior.n_classes` (2), making a 27,538,938-parameter network where TabICLv2's classifier is
  27,552,258 — a *different architecture*, which voids the project's claim to be varying only
  the prior, and left Exp3 unable to warm-start. Now 10, as in every one of upstream's
  classifier stage scripts.
- **The classification loss slices the logits**, exactly as upstream's `_compute_batch_loss`
  does (`logits[..., :n_classes]` where `n_classes = y_train.max()+1`). Same slice added to
  both prediction paths: without it a softmax over ten logits gives part of the probability
  mass to eight classes PD never contains, so AUC survives but Brier, calibration slope and
  every threshold decision are wrong.
- **`scripts/benchmark_gpu.py`** — six-rung ladder from raw matmul to a real training step,
  separating "this chip is slow" from "the prior is starving it". Reports
  `torch.cuda.get_arch_list()` and whether the wheel has kernels for the card at all, which is
  the leading suspect for the B200.
- **`src/utils/compare_gpubench.py`** and **`scripts/slurm/benchmark.slurm`** — run it on two
  cards, diff row by row, and the tool names which layer diverges.

- **`load_our_checkpoint` now rebuilds through `src.models.architecture.build_model`**, reading
  `architecture:` from the checkpoint's own config. It hard-coded `NanoTabICLv2`, so once
  training moved to upstream `TabICL` every parameter name mismatched and every `crediticl`
  cell died — the third distinct cause in a row for the same symptom.
- **Round-trip test** (`build_model` → save → `load_our_checkpoint` → identical keys and equal
  weights), both tasks, both architectures. That seam had no test at all.
- **`test_pd_head_size_is_pinned`** records that our PD head is 2-class (27,538,938 params) while
  the released classifier is 10-class (27,552,258), so **Exp3's PD warm start cannot load the
  released checkpoint**. Deliberately not changed — that is a decision about the experiment.
- **Progress measurement isolates each dataset.** One `try` wrapped both loops, so a NaN target
  in the second real dataset discarded the other two *and* all eight out-of-domain suites. Adds
  `n_errors`, per-dataset error text, `n_dropped_nonfinite_target`, and `pred_nonfinite_frac`
  rather than letting sklearn raise a message naming neither the dataset nor which side the NaN
  was on.
- **`adapt.py` docstrings corrected** — they still said released-checkpoint compatibility was
  "IMPORTANT AND UNVERIFIED" and probably broken, which this project has since measured as
  347/347 and 391/391 exact.

- **`--checkpoint` on both evaluation entry points**, and `debug_exp1.slurm` passes the arm's own
  checkpoint (read from `--dry-run`, so it cannot drift from where training wrote). Registering
  `crediticl` only makes the name resolvable; with no checkpoint every one of its cells failed
  while the run still reported "25/50 cells OK".
- **`resolve_our_checkpoint()`** — explicit path wins, otherwise the single checkpoint matching
  the task; **refuses to guess** between arms rather than scoring an arbitrary one.
- **NaN predictions are no longer reported as `constant_prediction`.** `np.var` of an all-NaN
  array is NaN, fails `> EPS`, and fell into the constant branch — so a numerical blow-up on
  three out-of-domain datasets read as a modelling quirk. New `pred_nonfinite_frac` and
  `nan_predictions`.
- First entry in [`RUNS.md`](RUNS.md) and the first three rows in the runs table.

- **Removed `--resume auto` from `debug_exp1.slurm`.** `pretrain.py` defines no such flag, so
  argparse exited 2 and all eight jobs finished in under a minute having trained nothing.
  Resuming is automatic (`trainer.maybe_resume()` is unconditional).
- **`tests/test_slurm_scripts.py`** — checks every flag the SLURM scripts pass against the
  argparse options of the script they call, plus `bash -n` on each job script. Nothing else in
  the project ever reads this layer before the cluster does.
- **Training now runs under `set +e` in `debug_exp1.slurm`**, so a crash leaves `STATUS`
  meaningful, skips the evaluation deliberately, and still prints the artefact summary.
- **`evaluate.py` and `evaluate_ood.py` now register the `crediticl` baseline.** `register()`
  had no production caller — only a test — so `--models crediticl` could never resolve and our
  own checkpoints would have been left out of their own experiment, behind the job script's
  `|| echo WARNING`. Added `register_or_warn()` and a test that every `--models` name the SLURM
  scripts ask for actually resolves.

- **The config is an argument, not an environment variable**: `submit.sh <where> <track>`, e.g.
  `submit.sh free lgd`. `sbatch [opts] script args…` passes trailing arguments to the job script,
  which reads `CONFIG="${1:-…}"`. The old `CONFIG=… bash submit.sh` set the variable for the
  calling shell, not `sbatch`'s environment, so a run meant as PD went out as a **second LGD
  job** with nothing on screen to say so. `where / track / config / script` are now all printed
  before submission, and an unknown track is refused.
- **Debug walltime 4 h → 1 h.** Billing is on *actual* walltime, so a generous limit costs
  nothing directly — but the requested limit is what the scheduler backfills against and what
  `sam-quote` reports, and 1,500 steps does not need four hours. It cut the quoted ceiling from
  489,600 credits to a quarter of that. Override with `WALLTIME=`.
- **A QoS rejection is now explained.** `QOSMaxSubmitJobPerUserLimit` does not say which limit
  was hit; `submit.sh` names the QoS, gives the `sacctmgr` query, and lists the three ways out
  (wait, different QoS, fewer arms).
- The `sam-quote` banner now says the number is a **ceiling if it runs the full limit**, since
  the docs are explicit that charging follows actual time.

The first debug submission sat in `gpu_a100` behind `Reason: Priority` and never started.
Reading the hardware inventory in `tfm-library/repositories/VSC Documentation.txt` says why,
and that we had picked the worst partition available.

- **`gpu_a100` has 16 GPUs — the FEWEST of any GPU partition we can use.** `gpu_h100` has 20,
  and Mindwell's `gpu_b200` has **24**.
- **Cores per GPU is our real bottleneck**, not GPU speed: TabICL generates its prior on the
  **CPU** (`--prior_device cpu` in upstream's own stage scripts). The documented limits are
  B200 **24**, A100 18, H100 **16**. So `gpu_h100` is the worst fit for this workload despite
  being the fastest chip — fewest cores *and* 4× the credit cost — and **`gpu_b200` is the best
  on both axes**, most GPUs and most cores.
- **Mindwell `interactive` costs no credits** and gives a *full* RTX 5000 Ada (32 GiB) for up to
  16 h — wICE's interactive partition gives a MIG slice instead. Ample for a 1,500-step debug
  run, and it is not queueing behind production training.
- `gpu_a100_debug` exists (1 node, full A100, ≤ 1 h) but Slurm allows **one queued job at a
  time**, so a 4-task array cannot use it.

Changes:

- **`scripts/slurm/submit.sh`** — `submit.sh <target> <script>` with `free|b200|a100|h100|dbg1h`,
  each carrying the documented core and memory budget for that partition. `--list` prints the
  inventory; `DRY_RUN=1` shows the command; non-free targets run `sam-quote` first so the credit
  cost is on screen before anything queues.
- **`debug_exp1.slurm` now defaults to the free target**, so a bare `sbatch` also lands somewhere
  that starts. Command-line options override `#SBATCH`, which is what lets one script serve
  every partition.
- **`num_workers` comes from `$SLURM_CPUS_PER_TASK`**, via a new `pretrain.py --num-workers`.
  A fixed 12 oversubscribes an 8-core allocation and wastes half of a 24-core one.
- **The debug job evaluated the wrong task.** It hard-coded `--task lgd`, so
  `CONFIG=config/Exp1_PD.yaml` would have trained PD and evaluated LGD — passing, while
  measuring something else. Task and out-of-domain kind now follow the config.
- `docs/VSC.md` §2a carries the inventory table.

**New rule, applied everywhere: a figure carries data, axis labels and at most a short heading.**
Interpretation lives in the caption and the body text, where it can be edited without
re-rendering. Removed 10 `style.figure_note` calls, 21 title subtitles and 17 sentence-length
headings; two tests keep them out.

Explanations moved into **22 "How to read it" markdown cells** in the notebooks, above the
figure they describe.

Specific fixes:

- **`plot_lgd_targets` was unreadable.** Each panel had a wrapping two-line title, they
  collided with each other and the suptitle, and matplotlib gave up — *"axes sizes collapsed to
  zero"*. Now the dataset name only, shared axes, 7 panels laid out properly.
- **`plot_pd_base_rates`: log scale removed.** It was justified as showing two orders of
  magnitude; the rates run 6%–40%, inside one. Value labels moved inside the bars so the 40%
  label no longer sits on the 50% reference line, and the annotation is "TabICL prior" instead
  of a wrapped sentence.
- **`plot_base_rate_by_variant` rebuilt as two panels.** One axis could not hold both a
  histogram and 14 real datasets: names smeared, then a rug made them invisible and put its
  label across the bars, then the 50% line ran through the legend. The real datasets now have
  their **own strip** sharing the x-axis, so nothing can overlap by construction.
- **Reference markers are magenta (`style.STAR`), not orange.** Orange stars over the orange
  credit series were invisible. Dots also shrank and the stars moved on top.
- **`plot_feature_correlations` paginates** — it showed the 6 largest of 14 datasets while
  claiming to describe "real credit data". All of them now, 6 per page.
- `plot_shapes_by_variant`: one figure-level legend instead of two colliding per-axes ones.
- `plot_default_clustering`: "independent" label moved out of the violins; duplicate "real"
  legend entry removed.
- **The row cap was biasing every statistic.** `load_real_datasets` took the *first* 20,000
  rows, and the rows are not shuffled — `algorithmwatch`'s default rate read **49.5%** against
  a true **37.8%**. Now a seeded random subsample: error down to 0.6pp. It also dropped
  `cat_indices`, which broke any plot needing it for large datasets only.

## 13-08-2026

The second out-of-domain fetch worked — 25 classification + 25 regression — and exposed three
more bugs, one of them latent.

- **The credit filter ran AFTER the quota check.** `heloc` was skipped as *"quota already full"*,
  so the exclusion never ran on it: it stayed out of the cache purely because classification
  happened to be full. With room to spare it would have been cached again and the log would
  still have read clean. Exclusion is now checked first, always.
- **Overlapping suites cached the same table twice.** `airfoil_self_noise`,
  `concrete_compressive_strength`, `physiochemical_protein` and `superconductivity` are in both
  TabArena and CTR23 under different dataset ids, so 25 "regression" datasets were 21 distinct
  ones — four tables carrying double weight in the out-of-domain average. Deduplicated by name,
  seeded from the existing cache so a resumed fetch does not re-add.
- **`complete` was always False.** It iterated `SUITES` (suite *names*) instead of `KINDS`, so
  every lookup missed and the fetch warned *"fewer than 25 for at least one task type"* while
  holding exactly 25 of each. A warning that cries wolf is a warning nobody reads.
- `setup_venv.sh` pins `setuptools<82`, which torch 2.11 requires — otherwise every subsequent
  pip install reports a dependency conflict.

The venv built and the smoke test passed, but the out-of-domain fetch had two serious bugs.

- **`heloc` — one of our OWN LGD datasets — was cached as out-of-domain.** A dataset we select
  priors on cannot also be the evidence that generality survived. The keyword filter also missed
  `loss2`, `axa`, `base_model`, `base_modelisation` and `hackerearth`. Fixed by **deriving** the
  exclusion list from `data/raw/` (`our_dataset_names()`) instead of hand-maintaining names, so
  adding a dataset cannot silently reopen the hole.
- **Continuous targets were coded into thousands of classes.** `SUITES` was
  `{kind: [suites]}`, so every task inherited the kind of the list it sat in. TabArena carries
  both kinds, so `diamonds` (price), `houses`, `airfoil_self_noise` and `miami_housing` were
  cached as classification with their targets pushed through
  `.astype("category").cat.codes`. `SUITES` is now flat and `task_kind(task)` asks the task,
  with quotas per kind filled across suites, plus a belt-and-braces reject of any
  "classification" target with >100 classes.
- **`OOD_VERSION` → 2**, which retires the 50-dataset cache from the first fetch. Neither bug is
  repairable in place.
- **Zero regression datasets were cached** — every regression suite alias 404'd. Fixing the
  bucketing means TabArena's regression tasks now land correctly, and CTR23's numeric study id
  is listed alongside its alias.

The venv setup failed on the first attempt; two bugs, both mine.

- **`module --force purge` collapsed the module tree.** The `cluster/*` modules on VSC are
  *sticky* and set up the architecture-specific `MODULEPATH`, so force-purging them made Lmod
  report *"exist but cannot be loaded as requested"* for a module that genuinely exists. Now a
  plain `module purge`.
- **The Python module is discovered, not pinned.** Module trees are **per-architecture**, so
  `Python/3.12.3-GCCcore-13.3.0` resolves on one login node and not another. The script tries
  the preferred pins, then falls back to `module -t avail` and takes the newest 3.11/3.12 the
  node actually offers, and fails with the exact diagnostic commands if none exists. Verified
  against a simulated Lmod that refuses both pins.
- **The venv is arch-suffixed** — `.venv-$VSC_ARCH_LOCAL` — which `docs/VSC.md` already required
  ("one venv per microarchitecture") and the first version ignored. wICE and Mindwell are
  different microarchitectures and a venv built on one is not reliably usable on the other.
- **The chosen module name is written to `.python_module`** beside the venv, and both
  `_activate_env.sh` and the `~/.bashrc` hook read it instead of hard-coding a name, so the venv
  and its interpreter cannot drift apart.
- The hook was re-verified end to end: activates on entry, deactivates on exit, and falls back
  to whatever venv exists when `$VSC_ARCH_LOCAL` differs from the one that built it.

- **`scripts/slurm/setup_venv.sh`** — one command builds this project's own venv on the VSC
  from `pyproject.toml`, so pyproject is the single source of truth for what is installed.
  Installs torch from the CUDA index **first** (otherwise `pip install -e .` satisfies the
  dependency with a CPU-only wheel that never announces itself), then `.[dev,eval]`, then
  verifies every import and that our model still matches the released checkpoints. Idempotent.
- **`_activate_env.sh` now prefers the project venv**, falling back to conda. It previously
  expected conda and *warned that an auto-activated venv would shadow it silently* — so the
  shell hook and the job path were on a collision course. They agree now.
- **`docs/VSC.md` §5 rewritten** with the auto-activation hook for `~/.bashrc`. Uses
  `PROMPT_COMMAND` rather than overriding `cd`, so it also fires after `pushd`, a subshell, or
  arriving via a symlink; deactivates only *our* venv on leaving. Verified: the hook parses and
  toggles correctly.
- Recorded why the venv goes on `$VSC_DATA` and not project storage: ~5–8 GB across tens of
  thousands of small files, and project storage has a **low inode budget** — it would run out
  of inodes long before space.

Three bugs the first cluster attempt exposed, none of which could fail locally.

- **`fetch_ood` died on the first dataset.** `np.savez_compressed` given a *path* that does not
  end in `.npz` silently **appends** the extension, so the atomic write produced
  `x.npz.tmp.npz` and the rename raised `FileNotFoundError`. Now written through an open
  handle, which suppresses the renaming.
- **No Muon on the cluster.** VSC runs torch 2.8 (no `torch.optim.Muon`) and the published
  `tabicl` wheel does not ship its training package, so every run died at optimizer
  construction. Upstream's own Muon is now vendored from the pinned dump
  (`src/train/_muon_vendored.py`) — no new dependency, and it is the exact optimizer that
  trained the released checkpoints rather than a second implementation that ought to agree.
  `torch.optim.Muon` still wins when the installed torch has it.
- **A `{{...}}` left over from a `.format()` template shipped in the LGD notebook** and reached
  the cluster. It is *valid Python* — a set containing a dict — until it executes, so nothing
  caught it. `test_every_notebook_cell_compiles` now compiles every cell of every notebook.

Figures, after rendering and actually looking at each one:

- **Titles no longer eat a third of the figure.** Long subtitles wrapped to three bold lines;
  they are short now and the detail moved to the note. `style.figure_note` **wraps** — a note
  longer than the page ran off *both* edges and lost its first and last words.
- **Mechanism decomposition:** atom labels sat directly on the spikes they labelled; they now
  have headroom above the bars.
- **Difficulty calibration:** the "real credit data" label ran off the right edge; median bars
  dwarfed the points they summarised; `R^2` renders as R².
- **Boundary sources:** the reference stars were drawn at s=80 over s=13 data points, so the
  subject vanished under its own yardstick. Sizes swapped.
- **Side-by-side tables:** row labels repeated the target values that were already the last
  column; they are row numbers now, and the coloured frame closes on all four sides.

## 12-08-2026

- **The released TabICLv2 checkpoints now load with `strict=True`: 347/347 tensors for the
  regressor, 391/391 for the classifier.** Exp3's blocker is gone. Our LGD model is
  28,544,991 parameters — the checkpoint's exact count. Two fixes got there: the real
  constructor takes `bias_free_ln`, not `norm_type`; and `apply_freezing` hard-coded
  NanoTabICL's stack names, so Exp3's `icl_only` arm would have raised `AttributeError` only
  after the job had queued.
- **The test suite now runs on the real architecture** (`conftest` prefers it when installed),
  shrunk for speed. `test_gradients_actually_reach_the_weights` no longer reaches into one
  hard-coded layer — it checks that a majority of ALL trainable tensors moved, which would also
  catch a whole stack sitting frozen.
- **New Exp1 figures** in `src/visualize/exp1_plots.py`, each answering a question that changes
  what we do next: prior-realism ranking (the one to read first), LGD mechanism decomposition,
  PD default clustering against an independence reference, difficulty calibration against real
  data, side-by-side real/synthetic tables, and boundary-mass sources.
- **Removed:** `plot_target_grid` (100 thumbnails nobody can compare),
  `plot_feature_relationships` (random graphs make random correlations), and the PD target
  histogram (a bar at 0 and a bar at 1 — the default rate, drawn as a picture).
- **Metrics widened for debugging.** LGD gains `brier`, `bias`, `calibration_slope`,
  `spearman`, `kendall`, `mae_boundary`/`mae_interior` (the split that matters — a good overall
  RMSE can hide being useless exactly on the atoms), and PIT uniformity. PD gains `ks`,
  `gini`, `calibration_slope`/`intercept` and a Brier skill score. `ks` and
  `calibration_slope` were **named in the configs but never computed**.
- **`scripts/slurm/debug_exp1.slurm`** — four arms (control, mechanism, quantile, low-mix),
  1,500 steps, 4 h walltime, running training *and* both evaluations so the half of the
  pipeline that has never executed on the cluster gets exercised.
- The realism figure's reference band uses the 10th–90th percentile: one real LGD dataset
  scores R² = −4.8 under a contiguous split, and min-max made the band span everything.

- **One architecture for all three experiments: TabICLv2's own.** `tabicl>=2.0` is now a
  REQUIRED dependency and `src/models/architecture.py` is the single entry point; every config
  carries `architecture: tabicl`. NanoTabICL was a 665-line reimplementation vendored only so
  the repo had a model with nothing to install — the cost was that Exp3 could not warm-start
  and architectural identity was unverifiable. It survives as an explicitly-selected fallback
  for smoke tests and must never produce a result. **Needs installing before any real run.**
- **Each experiment owns its own prior; the shared `prior_file:` is gone.** It encoded a false
  claim: Exp1 sweeps **32** priors, Exp2 runs the **one** that won, Exp3 sweeps the **mixture**.
  One file cannot represent all three.
- **Two bugs found while doing it, both caught by tests written for the purpose:** the Exp2/Exp3
  winner placeholders were never inserted, so those configs would have silently trained on
  whatever default the prior carried; and a regex put PD's `category_frequency` placeholder
  under `prior.base` — the **control arm**, which must stay exactly TabICL's.
- **Exp3 rebuilt as continued pre-training, from the published precedent.** TabPFN-Wide
  (Kolberg et al. 2026) does exactly this and reports it matches the base model; its recipe is
  now Exp3's — **LR 1e-5** (not pretraining's 8e-4, which would destroy trained weights),
  batch 16, 10,000 steps, longer warmup, clip 1.0. Sweeps `credit_fraction` 0→1 (Mitra, Zhang
  et al. 2025: mixtures beat single priors, so expect an interior optimum) and `init.strategy`
  `full` vs `icl_only` (Tanna et al. 2026 find full fine-tuning can hurt calibration, which is
  disqualifying for PD).
- **Out-of-domain evaluation widened from 10 to ~150 datasets.** Added **TabArena** and
  **TALENT** — the two suites TabICLv2 itself reports on — beside OpenML-CC18/CTR23, which stay
  for comparability with O'Prior. 25 per suite, and an unresolvable suite is skipped with a
  warning rather than aborting the fetch.
- Tests now assert the evaluation harness is genuinely **shared**: identical metrics and
  identical dev/holdout split across all three experiments, our own checkpoints registered as a
  baseline beside CatBoost and TabPFN, and out-of-domain scored *during* training.
- `docs/PRIORS.md` cut from 268 to 185 lines.

- **Six config files, one per experiment per track** — `Exp{1,2,3}_{LGD,PD}.yaml`, replacing
  `LGD.yaml`/`PD.yaml`. Exp1 screens the prior grid (96 arms, 50,000 datasets each), Exp2 runs
  the winner long (400,000), Exp3 is Exp2 with only `init` changed.
- **`prior_file:`** — the prior lives once per track in `prior_{LGD,PD}.yaml` and each
  experiment names it, so all three sample the same distribution. `config.load()` deep-merges
  it; an experiment can override one nested key without deleting its siblings.
- **`FILL_FROM_EXP1` placeholders** in Exp2/Exp3, and `load()` **refuses** a config that still
  holds one — a config that runs with a placeholder burns GPU-hours measuring nothing.
- **`dev_datasets` / `holdout_datasets` filled in**, chosen to SPAN the range rather than
  cluster: LGD dev covers 1.8 %/22.4 %/73.0 % boundary mass, PD dev covers 6.7 %–40 % base
  rates and 1k–150k rows. Tests assert the splits are disjoint and cover every dataset on disk.
- **`src/train/telemetry.py`** — GPU utilisation, memory, power, clocks, CPU/RAM, throughput,
  and **per-block gradient/weight norms with their ratio**, sampled on two independent cadences
  (`logging.log_hardware_every`, `log_grad_every`). Answers the two questions every finished run
  raises: was the GPU actually busy, and was every stack learning. Every probe degrades to a
  missing column rather than raising; the closing summary warns when a run looks **starved**
  rather than compute-bound.
- **`docs/RUNS.md`** — the run write-up log: date, job id, hyperparameters, resolved config,
  results, bugs, interpretation, with a template and the collect-and-upload workflow.
- **`docs/PRIORS.md`** — how TabICLv2's own prior works, read out of `tfm-library`. Records that
  the classifier and regressor share **one** `graph_scm` generator (the only prior-side
  difference is `--max_classes 10` vs `--regression_method quantile`), that the regression
  target is standard-scaled, and that the original prior's boundary atoms are the ±4 SD
  **outlier clamp**, not economics.
- **Figure pagination.** `grid_figsize` bounds width but ten panels across A4 is 0.63 in each —
  page-correct and unreadable. `style.paginate`/`page_suffix` split them the way a paper does;
  `plot_target_shapes_by_variant` is now two figures of five, panels 1.26 in wide.
- **Titles wrap to their panel.** `axes.titlelocation` is `left` and matplotlib neither wraps a
  title nor counts its width, so adjacent panels' titles ran into each other.
- **Honest boundary labels.** The summary called the original prior's min/max ties "at 0 (full
  recovery)" and "at 1 (total loss)". Its target is not on `[0,1]`, so those are now named as
  scale-free ties at the extremes, with the clamp explained.
- `fetch_ood.py` and `smoke_test.py` moved to `src/utils/` (utilities, not experiments);
  `submit_pipeline.sh` moved into `scripts/slurm/`. `scripts/` now holds only experiments.

## 11-08-2026

Brought the repository in line with `docs/TEMPLATE.md`. Shared modules were **copied** from
the template rather than rewritten, so they stay identical across projects.

- **Figures are sized for A4.** `style.py` gains `WIDTH_FULL` (6.30 in = the 160 mm text
  block), `WIDTH_HALF`, `WIDTH_THIRD`, `MAX_HEIGHT`, `figsize()`, plus `grid_figsize()` and
  `row_figsize()` for panel grids and bar charts. Replaced 18 hard-coded sizes that were
  11–13 in wide — every figure would have been scaled to ~50 % in the document, taking 9pt
  text below the 7pt print floor. Two tests measure the PDFs themselves.
- **`style.use_style()` → `style.apply()`**; `style.savefig` removed, since it was a second
  way for a figure to reach disk without a caption or a manifest entry.
- **Figures are PDF only.** The notebook displays each one inline, so the PNGs were a second
  raster copy that could go stale. `savefig.bbox` is no longer `tight`: cropping to content
  makes two figures declared at the same width come out different widths.
- **Captions moved to the save site**, `FIGS.save(fig, name, caption=...)`, and out of the
  central `FIGURE_CAPTIONS` dict — a caption now cannot go stale when its figure is renamed.
- **Notebooks are discovered, not listed.** `run_notebooks.discover()` globs `notebooks/`
  alphabetically; the hard-coded tuple silently stopped covering anything added.
- **`results_dir()` now resolves to project storage**, as the two-tier layout always
  intended. It returned `outputs_dir()/results`, so per-row predictions would have filled
  `$VSC_DATA`'s 75 GiB and then every job that writes a log would fail too.
- **`paths.py`** gains `raw_dir`, `processed_dir`, `config_path`, `ensure`, `notebooks_dir`,
  `library_dir`, `prior_cache_root`, and `STAGING_ENV_VARS`; `results_dir`/`checkpoints_dir`
  are variadic.
- **`src/utils/clean_run.py`** and **`update_tfm_library.py`** copied in; `clean_run` gains
  `--prior-cache` because this project's largest artefact lives outside `output/`.
- **Utilities moved out of `scripts/`**, which now holds only experiments and `slurm/`:
  `clean_run.py` and `run_notebooks.py` deleted (superseded), `vendor_model.py` →
  `src/utils/`.
- **`CLAUDE.md`** added (one line, `@AGENTS.md`) — Claude Code reads that file and not
  `AGENTS.md`, so until now it read no rules at all. **`AGENTS.md`** replaced with the
  template's, which adds "read `AGENTS_MEMORY.md` first" and the notebook/figure rules.
- **`docs/AGENTS_MEMORY.md`** added and seeded with ten verified dead ends. The Runs table
  starts empty: earlier wICE logs were read but never recorded and are no longer on disk.
- **`src/data/loaders.py`** added as the documented entry point over `pipeline.py` and
  `discovery.py`, so the template's name resolves to something real.
- **`.vscode/settings.json`** replaced with the template's. Ours was **not valid JSON** —
  `"${workspaceFolder}\.CreditICL\Scripts\python.exe"` contains `\C` and `\S`, which are
  not legal escapes — and lacked `jupyter.notebookFileRoot`.
- **`.gitignore`**: dropped nine unanchored or stale rules (`logs/`, `res/`,
  `results/**/*.npy`, `results/_local/`, `output/**/_run.py`, …) already covered by anchored
  ones. `output/logs/` and `output/manifests/` now ignore their *contents*, so the tracked
  `.gitkeep` survives and a fresh clone has somewhere to write.
- **The test suite was destroying real output.** `test_run_artifacts.py`'s `tree` fixture set
  the staging root but *deleted* `VSC_DATA`, so `logs_dir()` and `manifests_dir()` resolved to
  the real repository `output/` — the cleanup tests then deleted from it, wiping
  `data_exploration/`'s eight figures and `CAPTIONS.md` on every run. Both tiers are now
  redirected, and an assertion in the fixture fails loudly if that ever stops holding.
- **`isolated_output` fixture** added to `conftest.py`, setting `VSC_DATA` *and* the staging
  root with the same guard. A test that set only staging wrote two fake PDFs into the real
  tree, because `figures_dir()` ignores staging.
- **`src/methods/`** deleted — an empty package nothing imported.
- `output/` tree created with tracked `.gitkeep` markers, `data/processed/.gitkeep` added,
  and `README.md` now ends with the template chapter.

## 09-08-2026

- **One output root.** Everything generated goes under `output/` — results, figures,
  logs, manifests — locally and on the cluster. Was spread over four top-level folders.
- **One shared `output/figures/CAPTIONS.md`**, grouped per notebook, figures in notebook
  order. Every figure written as PDF (300 dpi, print) and PNG (110 dpi, committable).
- **Captions rewritten as pure description.** No interpretation; a test rejects
  interpretive phrasing.
- **`output/All_Results.md`** collects every notebook's printed summary, alphabetically.
- **Config: a top-level `sweep:` block** holds every multi-value knob, so the experiment
  design is four lines instead of scattered brackets. One short comment per knob.
- **Fixed `atom_prob`.** It claimed to set the share of datasets with boundary atoms but
  leaked: 0 gave 57%, 0.6/0.8 gave 87%/93%. Cause: `lgd_collateral`'s coverage is
  lognormal and its right tail always pushed rows past full recovery, clipping to exactly
  0. Narrowing ranges could not fix an unbounded tail; interior draws now clamp net
  recovery explicitly. Now 0.0→0%, 0.6→58%, 0.8→79%.
- **Fixed: notebooks could not load a config.** Relative paths broke because Jupyter
  starts in `notebooks/`; they now fall back to the repo root.
- **Fixed: `figures/` in `.gitignore` matched at any depth**, so it also ignored
  `output/figures/`. Anchored with a leading slash.
- **Fixed: every figure was saved twice** — `plt.subplots` calls `plt.figure`, so both
  capture hooks fired.
- `docs/` files renamed to CAPITALS. `run_notebooks` moved to `src/utils/`. Transfer
  helpers grouped under `scripts/transfer/`.
- **`docs/TEMPLATE.md`** rewritten as a generic, reusable spec.

## 08-08-2026

- **Fixed the bug that made the first cluster run undiagnosable.** `_activate_env.sh`
  printed `head -3` of the traceback — exactly the three boilerplate lines, cutting off
  the `ModuleNotFoundError` that names the package. Now tests each package separately,
  lists every missing one, and checks `import src`.
- **Fixed: `fetch_ood.py` went silent for minutes** while downloading. Logs every attempt.
- **Shift stress** (`src/prior/shift.py`). O'Prior's ablations report three independent
  contributors: mechanism diversity, realism, and shift-aware stress. We had the first
  two. Three kinds — `cohort`, `covariate`, `prior_prob` — on 30% of our datasets only.
- **Training progress curve** (`src/train/progress.py`). One CSV row per 10,000 synthetic
  datasets with metrics on real and out-of-domain data, so a run is a curve rather than
  one end-of-run number. Never fatal.
- **Run cleanup** (`src/utils/run_artifacts.py`, `src/utils/clean_run.py`). Lists by
  default; expensive categories need naming explicitly; raw data and downloaded weights
  are unreachable by construction.
- **Verified:** our ExtraTrees filter matches TabICLv2 Appendix E.14 exactly.

## 07-08-2026

- **Fixed: the SLURM chain could never have run.** It spanned three clusters and Slurm
  dependencies do not cross clusters; the genius stages also had no `--partition`. All of
  Phase 1 now runs on wICE.
- **Fixed: `sbatch --parsable` returns `jobid;cluster`** on a multi-cluster site, so every
  dependency string was malformed.
- **`DRY_RUN=1`** validates the whole chain with `sbatch --test-only` and queues nothing.
- **Added the piece that made the project unmeasurable** (`src/eval/crediticl_baseline.py`):
  nothing could load a checkpoint *we* trained. LGD point predictions use the median, not
  the mean — on a bimodal predictive the mean lands where no loan sits.
- **Out-of-domain evaluation** (`src/eval/ood.py`, `ood_runner.py`): OpenML-CC18 + CTR23,
  credit-like datasets filtered out, suite IDs resolved from the API rather than
  hard-coded.
- **Fixed: OOD regression scores were crushed by a credit assumption.** The LGD baselines
  clip to [0,1]; a standard-normal target scored R²=0.34 on a linear relationship.

## 06-08-2026

- **Muon optimizer**, matching TabICLv2. `torch.optim.Muon` ships in torch ≥ 2.9, so no
  new dependency. Muon takes only 2-D parameters, so `_MuonWithAux` pairs it with AdamW
  for the rest and presents one optimizer.
- **Multi-GPU** (`src/train/distributed.py`). A full-budget checkpoint is 24.5 GPU-days
  against a 72-hour ceiling. The batch is split so the effective budget is unchanged, and
  each rank gets a distinct prior seed — without that, every GPU generates identical data.
- **Credit mechanisms live** (`prior.credit.target.mode: mechanism`). LGD boundary mass
  emerges from collateral coverage and workout cashflows; PD defaults come from the
  Merton/Vasicek one-factor model with Basel's 0.03–0.24 asset correlations.
- **Fixed: the collateral economics were inverted.** Recovery was capped at the exposure
  *before* costs, so an over-collateralised loan still lost the fees and LGD could never
  reach exactly 0 — the atom the mechanism exists to produce.
- **Pre-generated prior pools** (`src/prior/pool.py`): one folder per variant, so every arm
  draws its original share from the same files. `verify_pools` fails the run if counts
  differ.
- **The whole pipeline is one command** (`scripts/submit_pipeline.sh`), five stages chained
  with `--dependency=afterok`.
- **Data pipeline** (`src/data/`), 21 datasets to a parquet cache, recipes copied from
  TabPFNCredit. **Eval pipeline** (`src/eval/`) with ridge/logistic, CatBoost, TabPFN-3
  and TabICLv2.
- **Fixed: the metric bug that inverted the project's motivation.** `(y <= 0).mean()` on a
  standard-scaled target returns ~0.5, which made the original prior look like it had 54%
  mass at zero.
- Two notebooks with all logic in `src/visualize/`, plus a shared figure style.

## 05-08-2026

- Project scaffolding: repo layout, `pyproject.toml`, config expansion with the `_range`
  suffix rule, path resolver for the two VSC storage tiers, logging setup.
- Literature review against the pinned `tfm-library/`. Three claims in the original brief
  were weakened after reading the sources — see `docs/EXPERIMENTAL_DESIGN.md`
  §"Verified premises".
- Prior generator ported from TabICL: Cauchy random DAGs, eight function families,
  categorical converters, ExtraTrees predictability filter.
- LGD and PD target families added, plus the mixture lever `credit_fraction`.
