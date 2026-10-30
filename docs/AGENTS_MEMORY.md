# Agents' memory — runs and dead ends

## 30-10-2026 — notebook refresh; GPU inventory-test failure

**Tried:** Final exported-caption audit after deploying df14174 and starting CPU 62474587.
**Result:** Deliberately cancelled that CPU attempt after 2m24s; no GPU submitted.
**Why:** The Exp2 design caption still said four checkpoint evaluations although the recipe and figure correctly contained five; the earlier wording test missed that phrase.
**Instead:** Correct both captions, broaden the stale-count assertion, rerun the 86 affected checks and all notebooks, then deploy one frozen revision for fresh CPU/GPU acceptance.

**Tried:** Full GPU Exp0 11754388 at frozen 931d593 with PROJECT image r3.
**Result:** FAILED 1:0 after 1m52s: 19 tests passed; the input-inventory assertion expected eight entries but received nine. Controller 62471655 and its guarded continuation 62473912 stopped; no production work launched.
**Why:** Container validation correctly adds a runtime-image hash alongside eight reference-model requests. The older test counted and iterated over both record types together.
**Instead:** Check the exact eight model keys and runtime hash separately. Six portable cases exercise the actual GPU assertion through real inventory assembly, rejecting missing models, missing/changed runtime provenance and incomplete inputs. Training code, settings and dependencies are unchanged; require fresh current-source CPU/GPU acceptance before production.

Reporting: all 12 notebooks reran locally, saving 40 visually inspected PDFs. Another 38 synthetic-only figure previews were inspected outside the repository; they are layout tests, not results. A read-only VSC snapshot added 932 files and matched 288 existing files without overwriting any; all 54 Exp1 training runs, including the quarter-mixture extension, are now visible locally. Benchmark efficacy panels remain pending until receipts are complete. Preserve the empty legacy figure directory whose removal was blocked by automatic approval review; it holds no results.
Full local pytest: **2,148 passed, 36 VSC-only skips, zero failures**, 20m01s. Targeted continuation/caption-regression Ruff checks pass; the broader project still has 170 legacy lint findings, so no repository-wide lint-clean claim is made.

## 30-10-2026 — current CPU passed; guarded queue continuation

| Job | Configuration | Outcome |
|---|---|---|
| 62471657 | Full CPU Exp0, frozen 931d593, PROJECT image r3, wICE interactive 8 CPU/48G | COMPLETED 0:0, 1h55m04s; 14/14 stages PASS, 2,108 tests passed, two allowed skips; all 12 notebooks executed |
| 62474587 | Full CPU Exp0, df14174, same image/allocation | Deliberately CANCELLED after 2m24s to include the corrected five-checkpoint caption; preprocess, preflight and Slurm client passed |
| 11754388 | Full GPU Exp0, same source/image, B200 24 CPU/180G, 24h | FAILED 1:0, 1m52s; 19 tests passed before the stale eight-vs-nine inventory assertion |
| 62471655 | Deployment controller, frozen 931d593, wICE 1 CPU, 72h | FAILED 1:0 after GPU failure; no production benchmarks or training submitted |
| 62473912 | Operator continuation, wICE batch_long 1 CPU, 7 days, afterany:62471655 | FAILED 1:0, 39s; correctly rejected the parent's ordinary failure rather than treating it as TIMEOUT |

**Tried:** Wait under the original 72-hour deployment controller while the B200 queue is busy.
**Result:** GPU forecast moved to 01-11 23:10, only 18 minutes before the controller deadline.
**Why:** Queue delay can exceed the controller reservation; this is not a scientific failure.
**Instead:** Hash-pin a separate operator continuation in the ignored deployment ledger, with
afterany dependency, strict parent/source checks, receipt-backed submissions and fresh quota
gate before Exp2. Its local tests cover 18 continuation guards and seven quota cases; they are
operator tests, not additional completed GPU acceptance. The 7-day CPU reservation quotes
20,160 credits maximum; no additional GPU validation is submitted. All 2,204 Exp1 files still
match size/mtime, and all 1,110 JSON/CSV hashes match the pre-cleanup preservation record.
Andreas requested a 30-minute continuation on 30-10: Codex heartbeat
`finish-crediticl-validation-and-launch`, bound to this chat, watches these existing jobs and
continues authorized work. Delete it after all 64 Exp2 work IDs are accepted and cleanup/docs
are resolved; ask for manual SSH renewal only when needed. Do not pull documentation updates
into the frozen VSC checkout while jobs still consume commit 931d593.

## 29-10-2026 — approved 1,500-update deployment

**Tried:** CPU 62471633 before the reviewed unthrottled launch, at 500 Mindwell submissions.
**Result:** Packages, preprocessing and wICE dry-run client passed; Mindwell --test-only
returned explicit QOSMaxSubmitJobPerUserLimit. The fail-closed gate launched no GPU.
**Why:** Slurm applies admission limits even to dry runs; the node client/resource test
treated a definite transient refusal as a code/runtime failure.
**Instead:** CPU waits for a real successful dry run. On GPU, the same explicit daemon
reply proves node transport, while resources require the complete current CPU receipt;
this avoids idling a paid GPU at a full queue. Missing plugins/credits, generic QoS errors,
and absent/stale CPU evidence still fail. Tests exercise each boundary.

**Tried:** Bounded metadata monitoring with bare python3 in a persistent SSH login.
**Result:** No monitor output, despite a responsive channel; absolute /usr/bin/python3 worked.
**Why:** The long-lived shell had hashed a different project's Python path, modified meanwhile.
**Instead:** Use absolute host Python for metadata and the explicit PROJECT launcher for science;
never trust another project's login-path interpreter or treat empty exit-0 output as evidence.

| Job | Configuration | Outcome |
|---|---|---|
| 62471630 / 62471633 | Deployment / CPU Exp0, d210730, PROJECT image | CPU stopped at the dry-run client: definite Mindwell MaxSubmit refusal; preprocessing, packages and wICE client passed; no GPU launched |
| 62471598_14 / _15 | wICE interactive, 1 CPU/1G each, native requeue probe | Both COMPLETED 0:0; restart 0→1 with identical job, array, task, CPU, RAM and partition |
| 62471122 | Exp0 CPU, 79cdcd9, PROJECT image r3, 8 CPUs/48G | PASS: 14/14 stages; 2,069 portable tests passed, 2 allowed skips; all 12 notebooks executed |
| 62471116 | Deployment controller, same revision | CPU passed; GPU submission refused at 500 queued/running jobs; zero GPU time |

**Tried:** Submit the GPU acceptance job immediately after the successful CPU validation.
**Result:** Definite QOSMaxSubmitJobPerUserLimit refusal; no GPU job was accepted.
**Why:** Other campaigns already occupied all 500 admission slots. A grid smaller than 500
does not establish available capacity; a percentage running throttle does not free queue slots.
**Instead:** Adopt the canonical template's receipt-backed admission manager. Wait for real
capacity, split unthrottled arrays preserving original work IDs, retry only definite capacity
refusals, and stop on uncertain acceptance. Preserve other projects' jobs and the CPU evidence.


**Tried:** CPU validation 62470908 under controller 62470900, after alias-fixture repair.
**Result:** 954 portable tests passed before the notebook output-contract fixture failed;
no GPU launched. Preflight, site credit client and both clusters' submission dry runs passed.
**Why:** One older fixture edited os.environ directly using the legacy staging alias, so it
was missed by the first setenv-focused search and did not override the production PROJECT root.
**Instead:** Audit every alias reference, use the canonical PROJECT variable in that fixture,
and stop the campaign immediately after a unit or scheduler-client failure. Runtime wrapper
publication now uses a unique temporary symlink plus atomic rename for parallel arrays.


**Tried:** Fresh CPU Exp0 62470861 under controller 62470856, current PROJECT-root override.
**Result:** Preprocessing/preflight/native Slurm credit checks passed; portable tests exposed
one cleanup fixture before later test modules ran. SIGINT requested the stored traceback
through an overlapping Slurm step; 594 tests had passed. All 2,204 Exp1 files still matched
size/mtime and JSON/CSV hashes after the attempt.
**Why:** The local-layout fixture removed only legacy staging variables, leaving the new
CREDITICL_PROJECT_ROOT active. Related pool/planning fixtures needed the same isolation review.
**Instead:** Clear both root-alias families for local-layout fixtures and use the canonical
PROJECT override for temporary pool fixtures. Tests simulate inherited aliases and preserve
an unrelated sentinel. CPU pytest now fails fast on its first failure; a successful run still
executes every portable test, with GPU coverage unchanged.


**Tried:** Controller 62470838 submitted fresh CPU validation 62470841 through the image.
**Result:** CPU startup stopped in 30 s; the dependency gate kept GPU/production unsubmitted.
**Why:** Apptainer exported the controller's automatic bind list; sbatch --export=ALL carried
its node-local /tmp runtime path to a different node, where that source does not exist.
**Instead:** Clear inherited Apptainer/Singularity mount and override variables before rebuilding
this host's explicit mounts; discard inherited node-local image pointers. A direct Bash test
checks that parent mounts/env are dropped while the new SLURM_JOB_ID is retained.


**Tried:** Containerized deployment controller 62470743, before CPU/GPU Exp0.
**Result:** Submission failed (1:0, 98 s); no Experiment 0 or production GPU job was created.
**Why:** The host sbatch client dynamically needs serializer libraries and the compute node's
real config-cache directory, not the login node's inherited /run configuration path.
**Instead:** Bind serializer/data-parser dependencies and the resolved host configuration;
clear stale APPTAINERENV_SLURM_CONF and add mandatory sbatch --test-only checks in both phases.
The subsequent login dry run exposed the site filter's Lua POSIX module dependency; bind
the host Lua module directories and their non-glibc libraries as well. The site Lua filter
also calls sam-balance: preserve its host Python and native loader in an isolated site-client
mount, never substitute the scientific Python. Disabled credit-check warnings now fail validation.
Controller 62470810 and its CPU child 62470817 were deliberately stopped during infrastructure
acceptance, before GPU work, to repair that warning; keep their compact diagnosis in RUNS.
Host configuration probe 62470775 completed on wICE interactive and confirmed the configless
cache at /var/spool/slurmd/conf-cache. These are infrastructure probes, not research runs.

CPU validation now uses wICE interactive (8 CPUs, 48G, four hours; no GPU). Live QoS allows
8 CPUs per user and the partition allows 7,500 MiB per CPU. The last accepted CPU job
62441437 peaked at 22,133,236 KiB (~21.1 GiB), so this retains ample RAM. Batch's test-only
projection was next-day afternoon; the debug allocation avoids that scheduling delay without
changing any model, prior, GPU allocation or scientific budget.


CPU runtime probe **62468952 COMPLETED**: 98 passed; Python 3.12.3 and all 135 dependency
requirements, Git/Slurm clients, child imports and executable helpers passed inside the
PROJECT image. Whole-environment fingerprint remains identical to the old DATA venv.
The fresh full CPU/GPU Exp0 pair is still required for acceptance.
A login cleanup probe also caught that checkout-root Slurm fallback logs cannot be unlinked
through the normal read-only code bind. Explicit clean_run --clean now opens the checkout
only on a login node after confirming no CreditICL consumer; previews and computation stay
read-only. Compute-node and active-campaign refusals have direct Bash regression tests.

Cleanup preserved all 2,204 Exp1 files (metadata plus JSON/CSV hashes), archived compact
old diagnostics on PROJECT, reset 546 old Exp0 files (4.500 GB), retired 3.869 GB of
verified published scratch, 150 legacy sentinels and two superseded candidate images.
The four unmatched directories from failed diagnostic job 11688276 were separately
identified by their Exp0 destination markers and retired (17 files, 2.601 GB).
Exp1 planning confirms only 15 missing quarter-mixture checkpoint benchmarks per task;
all 54 training runs are complete. The DATA venv and build workspace remain rollback
material until full image acceptance. Other projects' active jobs are preserved.

**Tried:** Full local regression suite after requiring the container launcher in the source digest.
**Result:** 2,073 passed, 36 VSC-only skips, one miniature-repository fixture failed in 1,535 s.
**Why:** That fixture omitted the now mandatory extensionless container launcher.
**Instead:** Add the launcher to the fixture, retain the strict production digest, then rerun
all portable tests locally and in CPU Exp0 before GPU acceptance.


Runtime mode repair 62468816 completed: Ruff 0.16.2 and Torch helper executability
verified. Candidate r3 SHA256 is
`83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`.
CPU probe 62468931 passed 97 checks and failed one child-process import test; adding
only the mounted checkout to the container's PYTHONPATH fixes that independent-cwd
import route. Compiler/model/plot caches are explicitly redirected to scratch.
The candidate remains unaccepted until the current full CPU/GPU Exp0 pair passes.


The user chooses 64 Exp2 runs (two seeds 0/1, both fixed rates, four mixtures, two
backbones, PD/LGD), 1,500 updates and checkpoints 125/250/500/1000/1500. Exp3 uses
32 new runs and eight matched Exp2 baselines at the same horizon/seeds. Exp1 remains
54 completed runs with its original three seeds and identities. The user explicitly
authorizes the PROJECT runtime rebuild, verified stale-output cleanup, current-code
CPU/GPU Exp0, remaining Exp1 benchmarking and finally Exp2 submission. Stop after
Exp2 has been submitted. Runtime build 62468678 on wICE started from recorded installed
payload wheels and a Python 3.12.3 base image; acceptance is pending. Active jobs from
other projects were observed and preserved.


**Tried:** Runtime build 62468678 from the first installed-artifact snapshot.
**Result:** Build stopped during offline pip installation, before any GPU validation.
**Why:** The first tag of a py2/py3 universal wheel was selected instead of a supported tag.
**Instead:** Select using packaging's interpreter tags, retain a regression test, and reuse
hash-verified payloads in build 62468733. A follow-up build 62468797 restores standalone
ELF tools such as Ruff; Python entrypoints are recreated by pip.

The first image and legacy environment had identical dependency fingerprint
`d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.
Container probes verified Git and squeue after adding explicit cwd, configless Slurm,
user/group lookup and auth-plugin library binds. No model or protocol was changed.

## 29-10-2026 — fixed-rate design supersedes the unrun pilot proposal

The user withdrew benchmark-driven LR selection. Exp2 now has 96 runs: two backbones ×
four mixtures × two predetermined rates × three paired seeds × PD/LGD. Stop at 1,000
updates; retain 125/250/500/1,000, with 1,000 as the primary endpoint. This is a bounded
adaptation experiment, not established convergence. Report both rates rather than selecting
a best-on-test score. Exp3 fixes 50% and the primary rate, reuses 12 Exp2 baselines and adds
48 fixed no-decay/L2-SP/DiagScale/QV-LoRA runs. No pilot was launched in this revision;
prior design discussions below remain historical evidence. New run names include
`fixed-rates-v1`; all 54 Exp1 configurations, names and indices match the previous code.

Live read-only VSC inspection: DATA checkout at `be6de6e`; generated tracked notebook/PDF
changes still block an ordinary pull. No jobs on Mindwell/wICE. Both recorded Exp0 reports
PASS for source `4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0`
(13 CPU, 116 GPU stages); they do not validate this new revision. All 54 Exp1 summaries
are completed at 12,500 steps, including the six quarter-mixture runs. DATA venv is 9.8G;
PROJECT CreditICL output is 36G. Quota totals: DATA 56,233M / 76,800M, PROJECT 538.2G / 1T,
GPFS scratch 8.634G / 500G, Lustre scratch 28.5G / 500G. Available credits: 95,812,429.
No installation, migration, deletion, submission or remote update was performed.

Current TEMPLATE already matches canonical SHA256
`285c7c321b2cd52d8d415e30f41fe775ceb47113f39007c2490fdb0ec23fc8d8`.
Manual Windows `step ssh login --context VSC`, existing agent, native LF-stdin SSH only.
The raw stderr probe was absent; a verified remote subprocess JSON envelope preserved both
streams and exit code 7. See VSC.md for the tested fallback and pending PROJECT SIF migration.
The existing DATA venv remains a documented legacy deployment, not a completed migration.

Local verification: the full suite ran in 1,334 s with 2,008 passed, 36 VSC-only skips,
and two obsolete plotting assertions failing. Both failures pass after the assertions were
updated for the two-rate panels and retired pilot labels; the subsequent affected-code batch
passed all 117 tests. No B200 acceptance is implied by these CPU checks.
All 12 notebooks subsequently completed, with 40 saved PDFs, captions and final summaries;
112 saved-output/plot/design tests passed afterwards. Visually inspected the actual Exp2/3
design PDFs, LGD endpoint ties and PD covariate-missingness figure, plus populated two-rate
curve layouts using test data outside the publication outputs. New source fingerprint:
`cbe1c7ecde878906010a3f5bfcf93cc43f0c8e3f01d4b545cda9e587fdfe556d`.
Training implementation is unchanged:
`01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`.

**Tried:** Full-suite validation after replacing selected-rate plots with the fixed design.
**Result:** Two plotting-contract failures; training, resume and compatibility tests passed.
**Why:** Tests still expected two panels and a pilot count instead of four panels and fixed recipes.
**Instead:** Assert the new panel/legend/design behavior and rerun both failures and affected tests.


## 29-10-2026 — Exp2A scope amendment and notebook Git conflicts

Andreas requests LR selection on all 14 PD / 7 LGD datasets. This supersedes the
development-only LR pilot below, while keeping the 16 runs, two rates per backbone,
500-update schedule prefix, paired 0%/50% mixtures and independent pilot seed 990.
Retain the compact candidate evidence and selection receipt; later Exp2/3 credit scores
are tuning-benchmark results, not independent holdout evidence. OOD never selects the LR.
Exp3 adapter pilots still use development datasets; the inherited base LR was tuned on all credit.
Keep Exp2 weight decay .01 and Exp3's no-decay contrast; no Exp1 recipe or training code changes.

**Tried:** VSC pull be6de6e -> 065703e followed by scratch cleanup, 29-10 at 13:02.
**Result:** Pull aborted on modified tracked notebooks/PDFs; the chained cleanup never ran.
**Why:** CPU validation executed notebooks and saved generated figures into tracked publication files.
**Instead:** Preserve existing edits with a scoped Git stash; new VSC notebook renders go to
the current validation attempt's notebooks folder, including executed copies and summaries.

**Tried:** The full local suite from the desktop's default executable search path.
**Result:** 1,943 passed, 65 skipped, two shell tests failed after 1,296.51 s.
**Why:** The existing Git Bash was absent from PATH, repeating the already documented host prerequisite.
**Instead:** Add `C:/Users/U0152019/AppData/Local/Programs/Git/bin` to the test process PATH,
fix PYTHONHASHSEED=0 and one-thread BLAS, and rerun; both failed shell cases pass with Bash available.

Weight-decay probe: local Torch 2.13.0+cpu, FP32 foreach AdamW and zero gradients showed no
parameter change at peak LR 1e-6 / decay .01 (1e-5 did change nonzero entries). This is not CUDA
evidence. Check actual VSC arithmetic after LR selection before paying for an equivalent no-decay
Exp3 contrast. The executable grid remains unchanged; any resulting amendment must be recorded.

Final local suite: **1,974 passed, 36 VSC-only skipped**, 1,590.64 s, with the existing Git Bash
on PATH, PYTHONHASHSEED=0 and one-thread BLAS. No dependency or training implementation change;
the training implementation remains `01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`.
Final executable source fingerprint: `7c0ba9b8ad670883ccd9b32a12e1aac0afa475ac82b8d5857fe9810981ffa4c4`.
Fresh VSC acceptance is still required; local tests do not certify the B200 allocation.
All **12 notebooks passed**, producing 40 PDFs, captions and final printed summaries.
Visually inspected the regenerated Exp2/3 design PDFs, PD missingness and LGD endpoint ties,
plus the populated all-credit pilot layout using test data outside the published outputs.
No new research results were synthesized; the local imported campaign snapshot remains dated 24-10.
Final review corrected two summary sentences to describe the registered pilot rather than claim
that it had already run; all 100 relevant plotting, selection and saved-output checks passed afterward.
All nine Experiment 1–3 notebooks were then refreshed successfully; all 12 saved notebooks have
final printed summaries and no error outputs. Publication artifacts contain 40 PDFs.

## 29-10-2026 — Budget A, development-only pilots and node-local working storage

Andreas approves Budget A after the literature/cost comparison. Exp2 now has 48 main runs:
2 backbones × 2 tasks × 4 mixtures × 3 seeds, 2,500 updates; retain 125/250/500/1000/2500.
16 separate LR pilots use seed 990, the 0%/50% mixtures and two rates per backbone, stopping at
500 on the same 2,500-update schedule. Only development data enter the score; one frozen LR
per backbone/task applies to all main mixtures. Primary holdout/OOD results are separate;
historical holdout exposure remains disclosed. Main runs restart released weights, not pilot states.
Exp3 uses 36 development-only 500-update pilots at seed 991, 48 new 2,500-update main runs
(no decay, L2-SP, DiagScale, Q/V LoRA), and 12 reused Exp2 50% baselines. No L2-SP in Exp2.
No package or training-implementation change: Exp1's 54 runs and benchmark compatibility persist.

Latest VSC transcript (29-10, 11:39): Exp2-only reset at be6de6e removed 15,071 files / 95.010 GB.
GPFS scratch is now 422.4/500G, project 616.9G/1T, DATA 56,261/76,800M. Available credits 95,813,864.
No new job submission is evidenced. Other projects own unknown portions of the remaining GPFS usage.
Mindwell now stages to VSC_SCRATCH_NODE, with no automatic shared-GPFS fallback. Every save is
published atomically to project storage; only the latest working checkpoint stays on the node.
`src.utils.scratch_cleanup` only removes retired CreditICL caches/byte-verified checkpoint copies;
it preserves unpublished checkpoints, all persistent results and unrelated projects.
The source/config changes require fresh CPU/GPU Exp0 acceptance before production submission.
TFM pin 26156f36eb7676e8d9050547817346db548b0e85; VSC pin 1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87.

Final local acceptance: **1,970 passed, 36 VSC-only skipped**, 1,189.41 s, with PYTHONHASHSEED=0,
one-thread BLAS and the existing Git Bash on PATH. All 12 notebooks ran (40 figures); the five
Exp2/3 notebooks were rerun after the final changes (5/5, three design figures). Captions and final
printed summaries are saved. New design PDFs and populated trajectory/pilot layouts were inspected
visually; populated layouts used test fixtures, not invented research results. The local imported
notebook snapshot is still explicitly dated 24-10; the 54-completion statement uses the later VSC evidence.
Training implementation remains `01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`;
final executable source fingerprint is `c362bdf9b52134e3eb2727175db669037e7e1bd86c2a902072c87e7700710eb7`.
No cluster job, package installation or user-output deletion was performed in this turn.

**Tried:** full local pytest after replacing the 96-run/10k-step grid with Budget A.
**Result:** 1,964 passed, six failed, 36 VSC-only skipped; all six old-expectation cases passed in a 47-test rerun.
**Why:** curve fixtures, benchmark-slot counts and geometry-grid assertions still described two main rates and the old checkpoint list.
**Instead:** update those fixtures to the approved single frozen rate, 24 main runs per task and five milestones; retain geometry, pairing and reference assertions.


## 29-10-2026 — TabPFN loss-range correction and authorized Exp2 restart

Andreas authorizes fixing the numerical bug and discarding all Exp2 run output for a fresh
rerun. Exp1's 54 completed runs and its existing benchmarks must remain usable. The observed
numerical failure is confined to Exp2 LGD TabPFN primary-rate seed-0 arms 8/9/10 (0/25/50%
credit), all at update 685. Exp1 uses the TabICL trainer and its benchmarks do not call this loss.

The correction transforms query targets in float64 and widens loss arithmetic for tails whose
squared errors or their batch reduction would exceed FP32 range. It preserves the model forward precision, mathematical
CRPS+MSE objective, task stream and budgets. Wide-loss updates use a float64 norm reduction for
clipping. A common backward divisor also covers finite float64 losses whose true gradients exceed
FP32's range: earlier task gradients and L2-SP receive the same divisor, then global clipping
restores the mathematically equivalent update. Targets are not clipped and tasks are not discarded.
The actual seed-0 batch-685 original-prior task now passes a real released-weight CPU
optimizer update. New CPU and mandatory CUDA tests cover that task, finite gradients, AMP replay
and one committed update. GPU acceptance still requires new Exp0 validation.

Hash-only provenance reads no longer stage every input/validation checkpoint into per-job GPFS.
Quota/access failures are labelled storage access rather than instructions to retrain Exp0.
Both checkpoint writers remove partial temp files after a failed save. Completed Exp1 source
compatibility is bound to the exact new training digest and reviewed prior sources/recipes;
failed Exp2 checkpoints do not gain resume permission through that exception.

`python -m src.utils.clean_run --reset-exp2` previews; adding `--clean` removes Exp2 logs,
records, results, checkpoints, recognised legacy output and identified job scratch on configured
GPFS/Lustre roots. It checks all paths before deletion, refuses queued/running CreditICL jobs,
uses Slurm accounting for preflight-only job directories, and retains unclassified scratch.
Other experiments, shared references, datasets, released weights and Downloads are excluded.
No actual user output was deleted in this implementation turn; cleanup was tested in fixtures.

The corrected VSC quote with `--array=0` and the actual script returns **30,600 credits/hour**
for one B200, 24 CPUs, 180G. Available balance: **95,945,277**. At the measured/provisional
throughput the current 96×10k grid projects to **373–395M training credits**, excluding benchmarks.
The 48×2,500 single-rate alternative projects to 46.6–49.4M, and 96×1,000 to 37.3–39.5M.
These are proposals, not changed runnable grids. The user requests an extensive literature-based
comparison before choosing. Freeze that choice before a fresh CPU/GPU validation pair.
TFM research pin `26156f36eb7676e8d9050547817346db548b0e85`; VSC pin
`1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87`; neither submodule was edited.

Local acceptance: **1,957 passed, 36 VSC-only skipped**, 1,190.03 s, with PYTHONHASHSEED=0,
OMP/MKL threads=1 and the existing Git Bash on the process PATH. The focused checks pass 144/144.
Current source fingerprint `07f1b4e40e258e7d9748af8a4ee7c227cef1897561e2b324269001838a8022b5`;
reviewed training implementation `01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`.
Ruff reports 185 pre-existing findings, zero introduced (compared against HEAD); new test files lint clean.
No notebook/visualization source was changed, no package installed, and no cluster job submitted.

Tried: the first reset regression fixtures and new atomic-save failure probe.
Result: cleanup refused the recognised legacy Exp2 checkpoint folder; two test fixtures also needed correction.
Why: the released-weight parent guard was broader than the explicit legacy-run scope, and the save probe lacked the scheduler required by the existing API.
Instead: allow only the recognised `checkpoints/exp2_*` child exception, keep released files protected, repair the fixtures and rerun the suite before publication.

Tried: a complete local suite with the unbounded host thread default, then a wider finite-tail probe.
Result: the released-weight CPU replay became excessively slow; a finite FP64 loss can still produce FP32-infinite gradients above the original failure's range.
Why: native CPU attention is sensitive to oversubscription; widening the loss alone does not widen intermediate gradients.
Instead: bound threads in the CPU replay test, normalize the entire backward accumulation before the existing global clipping rule, and test against analytic FP64 gradients before the final suite.

Tried: the broader Windows suite with the default executable search path and the old exact-gradient assertion.
Result: two shell error-propagation tests could not find Bash; a transformed-target gradient differed from native FP32 by 2.24e-8.
Why: Git's installed Bash was outside PATH; the deliberately wider target transform changes FP32 rounding slightly.
Instead: expose the existing Git Bash only in the test process, retain exact classification/untransformed checks, bound transformed-gradient tolerance to FP32 roundoff, and rerun the full suite.

Tried: regenerating the exact failing production task in a test process without PYTHONHASHSEED=0.
Result: the nominal seed/slot could yield an ordinary transformed target instead of the reported extreme.
Why: upstream GraphSCM uses hash-dependent set ordering, already documented and fixed to hash seed 0 in production launchers.
Instead: regenerate this historical regression case in a fresh hash-seed-0 subprocess, not by inheriting the test runner's hash seed; also cover finite-element/overflowing-mean loss reduction.

## 29-10-2026 — all Exp1 training complete; Exp2 blocked by scratch quota and target-loss overflow

Downloads `output CreditICL.zip`, 29-10 08:51:08, 28,122,513 bytes, SHA256
`38901be1f55cd06c655b5f8fa22a6518e51bd015e3682ac9923b2cedf2365c74`.

| Job | Configuration | Observed outcome | Headline |
|---|---|---|---|
| 11747538, Mindwell | Exp1 LGD quarter mixture, indices 9–11 | All three completed | 12,500 updates and 800,000 tables each; 6.51–6.61 h |
| 11747539, Mindwell | Exp1 PD quarter mixture, indices 9–11 | All three completed | Same budget; 6.97–7.01 h |
| 11747551, Mindwell | Exp2 LGD, 48 arms | No completed arm | 11 configurations started; three stopped/resubmitted at 550/551; three TabPFN arms failed at 685; storage blocked the rest |
| 11747552, Mindwell | Exp2 PD, 48 arms | No completed arm | Eight TabICL configurations started; checkpoint writes failed; remaining 40 refused at preflight |

Exp1 now has **54/54 completed training summaries**, with finite post-zero recorded training
losses in the six new runs and no failed quarter-run logs. The same 128 old benchmark tasks
end OK; no quarter-mixture benchmark jobs appear. Project-tier CSV/checkpoint bytes are absent
from the ZIP. Exp0 reports remain PASS; all 228 available referenced artifact hashes still match.

Andreas's live 29-10 08:53–08:54 check shows GPFS scratch **497.7G/500G**, CreditICL/jobs 84G,
project stg_00211 623.5G/1T (CreditICL 50G, its output 47G), DATA 56,221M/76,800M.
No CreditICL job remains in Mindwell's queue; only unrelated JEVPFN 11750830 is RUNNING.
Do not delete other projects' work or reset validated/production output. Disposable per-job
`inputs/` copies can be rebuilt; `checkpoints/` may contain the newest recoverable state.

Tried: running the full Exp2 arrays after successful current Exp0 validation.
Result: 99 logged attempts: three checkpointed/resubmitted, 14 checkpoint-stream write failures, 79 explicit quota failures and three non-finite TabPFN losses.
Why: the shared GPFS scratch quota filled; readiness hashes stage validation inputs per job and then mislabel copy/quota errors as a need to rerun Exp0. The exact contributor to the other approximately 414G is not established.
Instead: recover scratch capacity without deleting checkpoints, preserve the passing Exp0 artifacts, and distinguish storage-access errors from stale/failed validation before resubmitting.

Tried: TabPFN LGD seed-0 arms at 0/25/50% credit with native transformed-target CRPS+MSE.
Result: all three fail at update 685; recorded finite losses also reach approximately 1.2e31 earlier.
Why: local regeneration of batch 685, original-prior slot 15, gives a transformed target around 1.034e25; native float32 MSE overflows even for finite logits. This slot is identical across those three mixtures.
Instead: validate wider-range loss/target-transform arithmetic without silently clipping targets or dropping tasks; add this exact batch as a regression probe. This audit implements no training change and establishes no GPU fix.

Observed TabICL throughput implies roughly 210–225 GPU-hours per 10k-update run, about
10k GPU-hours for its 48-arm subtotal. This is a production-log extrapolation, not measured
completion or billing. Review the campaign budget before full resubmission. Further details,
including reproduction package differences and recovery limits, are in RUNS.md.

## 28-10-2026 — Exp1 quarter-mixture extension and Exp2 submitted

Terminal attachment `723611bd-b126-4ffc-a791-5f8c749844a7/Pasted text.txt` supersedes the
submission state in the earlier archive audit below. Current Exp0 preflight passed before
both submissions; the planner preserved all 48 completed Exp1 runs.

| Job | Configuration | Observed outcome | Headline |
|---|---|---|---|
| 11747538, Mindwell | Exp1 LGD, indices 9–11, 25% credit, seeds 0/1/2 | Submission accepted | Three additional 12,500-update runs |
| 11747539, Mindwell | Exp1 PD, indices 9–11, 25% credit, seeds 0/1/2 | Submission accepted | Three additional 12,500-update runs |
| 11747551, Mindwell | Exp2 LGD, indices 0–47, throttle 8 | Submission accepted | 48 two-model/four-mixture/two-rate/three-seed CPT runs |
| 11747552, Mindwell | Exp2 PD, indices 0–47, throttle 8 | Submission accepted | 48 two-model/four-mixture/two-rate/three-seed CPT runs |

No runtime result, live queue or new quota evidence accompanies these submissions. Keep the
validated executable source and packages fixed; retain Exp0 artifacts and Exp1 output on both
tiers. Benchmarks were not submitted by these training-only commands. Run each experiment's
benchmark planner after its training completes; do not reset output during these campaigns.

## 28-10-2026 — current Exp0 passed; quarter-mixture Exp1 still pending

Downloads `output CreditICL.zip`, 28-10-2026 09:10, SHA256
`67ef250139ee0dd4357dc3123c562c9dd9da1389591a1ca2912b1fce111a6fcf`.

| Job | Configuration | Observed outcome | Headline |
|---|---|---|---|
| 62441437, wICE | Exp0 CPU, commit 673dba5 | PASS, 27-10 | 13 stages; 1,907 tests passed, one external-project skip; notebooks passed |
| 11741580, Mindwell B200 | Exp0 GPU, commit 673dba5 | PASS, 27–28-10 | 116 stages in 17h57m49s; 32 VSC tests, 16 adaptation routes, 34 long-context probes passed |

Both reports match source `4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0`
and environment `d4ed1fbe1397c0063fc12abffcb338fddd8038f21decb20c8c84accaf0bf0454`.
All 228 referenced artifacts present in the ZIP match their hashes; 48 project-tier GPU artifacts
are absent from this small-tier export. Andreas's on-VSC verifier reports 122 items, zero FAIL.
The GPU report binds the exact CPU receipt. No new Exp0 run is needed for this executable source.
Tested pins: TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465`, VSC
`3bcdf55e576d10161d6189b40ac39380bbba9eca`. Later local library-only updates preserve the source hash.

Exp1 remains 48 completed 12,500-step runs (18 main + 30 component), with 128 successful
benchmark task logs. There is no quarter-mixture output: six new runs remain, then their benchmarks.
The ZIP does not establish the current queue or independently verify project-tier numerical results.
Keep all completed runs and current Exp0 artifacts; do not reset output. Only the current passing
CPU/GPU attempt directories occur in this archive. No failed Exp1 end markers or tracebacks were found.

Exp2 is configured for 96 runs and no longer blocked by validation. Its TabPFN-only training
projection is 2,109.4 GPU-hours; TabICL campaign time and benchmarking remain unmeasured.
Single-update long-context capacity tests must not be turned into a reliable campaign ETA.
The 27-10 quota snapshot shows DATA 56,113M/76,800M and shared project 559.8G/1T; use `myquota`
for current quotas before launching. See RUNS.md for archive, cost and storage qualifications.

## 26-10-2026 — GPU continuation refusal and approved Exp2/Exp3 implementation

New evidence supersedes the earlier submission-only entry below: Downloads `output CreditICL.zip`
(26,856,748 bytes, SHA256 `b64ae347e49e68b14a371292233c6650d6d2c8d925b8cf004ef306cf7c8dcc53`)
and Andreas's scheduler output show GPU **11739875 FAILED 1:0 after 00:00:29**.
CPU **62431090 PASS** includes 1,876 tests, no failures, one skip. The GPU loaded packages and
then refused to resume the stale 21-10 campaign; it did not reach training or benchmarking.
The archive still contains 48 completed Exp1 summaries, 12,500 steps each, and no completed
25% extension. Checkpoint bytes and current remote job state were not independently read.

Tried: submitting the current GPU wrapper after current CPU validation.
Result: `Experiment 0 continuation REFUSED: Code/config/tests/packages changed`.
Why: unconditional `--resume` selected an old GPU receipt instead of starting a new campaign.
Instead: `--resume-if-current` resumes only matching campaigns, otherwise creates a new isolated
campaign. Explicit `--resume` remains strict; matching failed stages still need correction.

User now authorizes implementation. Exp2 = 96 runs (2 models × 2 tasks × 4 mixtures × 2 rates ×
3 seeds), 10k updates, permanent steps 125/250/500/1k/2.5k/5k/7.5k/10k. Exp3 fixes 50% credit:
36 development-only 2,500-update pilots (10k-schedule prefix), then 48 new 10k runs and 12 reused
Exp2 full-CPT baselines. Methods are no weight decay, L2-SP, query-only DiagScale and rank-16 Q/V
LoRA. Frozen hashed receipts choose rates/strengths using development only, without editing YAML.
Native OOD is evaluated unchanged; multiclass tasks retain their native labels.

Exp1 configs remain unchanged. Compatibility is bounded to the reviewed original completed
48-run source and previous `19a76cc` source/recipes/dependencies; only the latter admits exact
partial-run resume. Do not widen these exceptions or reset Experiment 1 output. All six 25%
extension identities remain append-only. Fresh Exp0 CPU and GPU validation is required.
TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`; VSC pin
`3bcdf55e576d10161d6189b40ac39380bbba9eca`. Both submodules remain unmodified.

Final portable suite: **1,928 passed, 32 VSC-only skipped**, 1,013.12 seconds. The 16
released-weight method/task/backbone CPU stop/resume/prediction probes also passed; all four
L2-SP routes were repeated after checking unused-parameter regularization. These do not certify
CUDA precision or long-table capacity. The launch recipe uses `--single-cluster` for Exp2/Exp3
so new runs stay on the Mindwell B200 allocation tested by Exp0. Reviewed executable source:
`4b31d59be2048f41f894db9a26c49919f759e65e9fdfae4305dd52e27ae164d0`.
All 12 notebooks executed successfully and saved 40 PDFs, captions and section-ordered printed
summaries. Every final PDF was rendered and pixel-identical to its visually reviewed version;
the three design figures and LGD endpoint-tie panel were inspected again. Nonempty future
result layouts were checked with explicit test fixtures, never published as research results.

Tried: the old configuration/plot assertions after replacing long-scratch Exp3 and expanding Exp2.
Result: the first complete suite exposed obsolete grid counts, placeholder assumptions and notebook names.
Why: tests still encoded the retired design; one new caption also missed the descriptive-caption rule.
Instead: retain the scientific safeguards, update the expected design, make the caption concrete,
and rerun the complete suite after the focused checks.

Tried: repeating all notebooks while the full suite and native model probes were running on Windows.
Result: one spawned LGD-plot worker raised WinError 6 and BrokenProcessPool; 11/12 notebooks completed.
Why: a multiprocessing handle failed; concurrent runs were present, but the exact OS cause is not established.
Instead: perform the final notebook run and full test suite sequentially and inspect the regenerated PDFs.

Tried: the full suite after successful sequential execution of all 12 notebooks.
Result: 1,922 tests passed; one benchmark-resume test hit Windows access denied during atomic JSON replacement.
Why: Windows denied replacing the existing file; sharing contention is likely, but no locking process was identified. The previous complete file survived.
Instead: use bounded retries for Windows sharing errors in JSON/CSV publication, test transient and
permanent failures explicitly, and rerun verification. This is not a reported VSC training failure.

## 26-10-2026 — current Exp0 CPU pass and GPU submission

Evidence: terminal attachment `5dd82437-9175-46aa-92be-9fb495f153b4/Pasted text.txt`.
The pull reached `19a76cc`; source fingerprint
`264c70dd8f5aff5acd72776366520de3765cf0a51383f9c761c8f8aef1eed5a0` was checked.

| Job | Configuration | Observed outcome | Headline |
|---|---|---|---|
| 62431090, wICE | `scripts/slurm/exp0_cpu.slurm` | CPU gate PASS | Receipt `cpu_20261026T141139Z_j62431090_fbb2a984`; all stages/artifacts current |
| 11739875, Mindwell | `scripts/slurm/exp0.slurm` | Submitted | No GPU completion result in this transcript |

The 48 local Exp1 summaries still report completed 12,500-step runs. The six-run quarter-mixture
extension must wait for the current GPU gate. This evidence does not validate the proposed second
Exp2 learning rates, early checkpoint schedule or replacement Exp3 methods, which are not implemented.
The user explicitly retained design/review-only scope. Targeted compatibility/training-contract checks:
129 passed. No runnable code, output or environment changed. Full run details are in `RUNS.md`.

## 26-10-2026 — completed Exp1 provenance during the extension

The reattached `exp0_gpu_20261026_083258_j11737746.log` confirms the already-recorded
old-checkout receipt reuse, not validation of the quarter-mixture extension. No new
cluster run is evidenced by this upload.

Tried: preparing the missing-arm commands against the actual completed Exp1 records.
Result: the planner rejected their recorded training fingerprint after Exp2-only source additions.
Why: completion and evaluation reused the strict checkpoint-resume check; the extension fixture
omitted the historical fingerprint, so it never exercised the real record format.
Instead: a narrowly reviewed completion/evaluation exception binds the a073d7d source to the
current dependency versions, the exact reviewed destination training-source digest, 16 original
PD/LGD recipes, seeds 0/1/2 and completed 12,500-step/800,000-table summaries. It requires the
saved configuration; unknown code, packages, recipes and partial runs still fail closed.
Checkpoint resume retains its original strict check. All 48 saved recipes were compared with
the current resolved recipes, and no output or checkpoint was changed. Benchmark reuse also
allows the reviewed a073d7d/50b3d7e/a18efa8 provenance-only transitions; scoring code, inputs,
packages, checkpoint bytes and result bytes must still agree. Current VSC validation remains
required before submissions. Andreas's read-only VSC check at 14:30 reports training fingerprint
`0d328e50ce06778731c5ba5218a61222b9cf396c8b8d767cb3e8a58f7473e69b`, torch 2.11.0+cu128,
TabICL 2.1.1, TabPFN 9.1.0, NumPy 2.5.2, SciPy 1.18.1 and scikit-learn 1.9.0.
Reconstructing the historical fingerprint from that metadata and the reviewed source matches
exactly. Reading the actual 48 imported summaries/configs with those dependency identities
recognises 9/12 main and 15/15 component runs per task, missing only main indices 9–11.
This was a local read-only reconstruction, not a fresh VSC validation or checkpoint-file audit.

CPT LR/L2-SP choices remain proposals for discussion; this fix does not change any training,
checkpoint frequency, precision, prior, benchmark protocol or notebook source.
Verification: focused provenance/extension checks, 121 passed. Full portable suite: 1,895 passed,
one changelog-format failure and 30 VSC-only skips in 20m57s; the duplicate date heading was
merged and all 69 affected documentation/notebook tests then passed. No training, evaluation
or orchestration test failed.
Executable fingerprint: `264c70dd8f5aff5acd72776366520de3765cf0a51383f9c761c8f8aef1eed5a0`.
TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`; VSC pin
`3bcdf55e576d10161d6189b40ac39380bbba9eca`. No notebook/visualization source changed.

Tried: the full portable suite with the newly added changelog entry.
Result: one formatting failure; all 1,895 other portable cases passed.
Why: the entry introduced a second heading for the existing 26-10-2026 chapter.
Instead: merge the bullet into that chapter and rerun the affected documentation/notebook checks.


## 26-10-2026 — approved Exp1 extension and CPT evidence review

User explicitly approved 25% credit for Exp1 after the original 48-run campaign. The main grid
now has 12 runs/track; components remain 15/track, for 54 total. `array_append` preserves the old
indices 0–8 and appends quarter-mixture seeds at 9–11. Resolved training recipes, checkpoint/run
names and benchmark tags for the existing runs remain unchanged. The metadata is stripped from
prior-file inheritance, so component studies stay at 50%. Display order is 0/25/50/100.

A narrowly reviewed receipt transition accepts the changed grid counts, selection messages and
default benchmark array size only when every scientific setting, other source digest, input,
package, checkpoint and result digest agrees. It cannot approve unknown historical code or
changed data. The 50b3d7e-to-extension source pairs are explicit; later edits fail closed again.
No completed training or benchmark output was removed, no training implementation/precision or
Exp2 optimizer setting changed, and no VSC job was submitted.

The expanded research review confirms 550k original TabICLv2 updates (500k/40k/10k), with
stage 3 already mixing 400–60k rows log-uniformly. The proposed 10k continuation budget is not
an established credit-CPT optimum. TabPFN3 publishes its architecture and prior description,
not its full original update/batch/curriculum schedule. TabPFN-Wide is v2 classification evidence.
Contemporary Mitra-v2, EXAONE, Xiaomi-TabLDM, LimiX2, TabDPT-Turbo and TabPFN2.5/3/3.5 reports
show materially different recipes. Main Exp2 stays at 48 full-update runs. L2-SP is recommended
as a separately validated method comparison before a broad LoRA/freezing grid; none is enabled.
Any pilot selection must exclude the final holdout/OOD benchmark.

Tried: append 25% to the sweep list without changing ordinary seed-outermost expansion.
Result: regression checks detected reassigned old array indices and benchmark arm identities.
Why: a fourth mixture inserts an arm inside every seed group rather than after the completed nine.
Instead: explicitly append the extension after seed expansion; test old resolved recipes and tags,
missing-arm submission, quarter-mixture selection, numeric plot order and strict receipt reuse.

Tried: the first full suite after extending the main grid (19m22s).
Result: 1,802 passes, six obsolete three-mixture/count assertions, one transient Windows rename denial,
and 30 expected VSC-only skips. No training/numerical regression was reported.
Why: fixtures still expected nine runs/two nonzero mixtures; Windows briefly denied a receipt replacement.
Instead: update explicit expected counts and reject incomplete four-mixture evidence; retry only
bounded Windows access/sharing errors during atomic receipt publication, retaining the old receipt.
Permanent permission failures still raise. Add success, bounded-failure and non-Windows-error tests.

Tried: full suite with unrestricted Python hash randomization (18m24s).
Result: 1,829 passed; the ragged-loader worker equality check failed on Windows; 30 VSC skips.
Why: upstream orders feature groups through a string set; newly spawned workers used different
hash seeds from the test interpreter. The same test passed with the production hash seed 0.
Instead: run this exact-equality probe in a fresh interpreter with PYTHONHASHSEED=0, inherited
by workers. Production pretrain/Exp0/notebook launchers already set this before Python starts;
do not change the upstream prior or weaken equality to hide the test-harness difference.

Verification: **1,830 passed, 30 VSC-only skips** in the final complete suite (17m53s), with
PYTHONHASHSEED=0 and Git Bash on PATH; the 64 focused checks also passed with the parent hash
seed unset. All 12 notebooks executed and saved 37 PDFs. Both changed coverage figures and
the prior endpoint-tie/missingness figures were inspected visually; PDF text comparison found
changes only in the two coverage figures. No template deviation.
Source fingerprint: `652fe453abc5d9c189e71433a006d41f0f96f92a7597c3bba81420fdffb2751a`.
Training fingerprint remains `cf48209f392f2cba239123eb408b942e39a9ce1df976de7db0b66b6b0d59330d`.
TFM pin: `702d9ab0b66f6639ba8d443ff9c4a91231964465`; VSC pin:
`3bcdf55e576d10161d6189b40ac39380bbba9eca`. Both submodules are unchanged. Fresh cluster
CPU/GPU validation and long-context capacity measurements remain required for revised Exp2.
Read-only pins: TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465`,
VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`.

## 26-10-2026 — four-mixture Exp2 and precision review

User requested 25% credit in Exp2 and authorized pushes. Exp2 now has 48 runs (two models,
four mixtures, three seeds, two tasks); completed Exp1 remains unchanged. Mixtures are Bernoulli
task probabilities, not exact per-update quotas. Full-parameter CPT remains the main design:
Rubachev's per-dataset TabPFNv2 PEFT comparison does not establish a LoRA advantage for synthetic
CPT of v3/TabICLv2. Reviewed `LoraMHA`/`replace_with_loras`, TabTune `inject_lora`, TabICL stage-3
scripts, and installed TabPFN 9.1.0 fine-tuning source; no adapter grid or dependency added.

Found and fixed: TabPFN `amp: false` could leave internal estimator autocast enabled. Both flags
now agree before executor construction, with an explicit outer FP16/disabled context. Exp0 tests
exercise both CUDA modes; long-context probes observe actual FP32 Q/K/V, including recomputation,
and record permitted TF32 matmul. This is not a claim to duplicate upstream FA3/AMP behavior.
`clean_run --reset-exp0` is the new full validation-only reset on both tiers, preserving Exp1-3
and shared references; use before a fresh CPU/GPU pair, never between them. No cluster run was
launched in this review; fresh GPU validation remains required.

Verification: all **1,795 portable tests** passed across the full suite and a Bash-enabled
recheck. The first full invocation reported 1,764 passes, 2 missing-Bash failures and 59 skips;
all 31 non-VSC failed/skipped cases then passed with installed Git Bash on PATH. The remaining
30 tests require the VSC/CUDA allocation. Separately, all 97 Slurm tests and all 19 scoped cleanup
tests passed. All 12 notebooks reran (37 PDF figures); changed rendered plots and the four-mixture
style key were visually inspected. Local Exp0 cleanup found zero files; completed Exp1 remains.
Final executable source fingerprint: `10356de6f4e26275f5607c5fcbed9f63f3f229adf1e3ec4e3df9d5451e0ea0c6`.
Read-only pins: TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465`, VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`.

Tried: the complete local pytest run from a PowerShell process without Git Bash on PATH.
Result: 1,764 passed, two shell-contract failures and 29 shell skips, plus 30 expected VSC skips.
Why: Bash was installed but unavailable to that Python process; the failing checks correctly refused to bypass Slurm validation.
Instead: prepend the existing Git `bin` directory to PATH, then rerun all 31 affected cases; all passed without changing or weakening the tests.

Tried: a new quarter-mixture regression test asserted exactly one quarter of every batch was credit.
Result: the focused suite found 17 credit tasks in a 64-task batch; 284 other tests passed.
Why: the established stream samples independent per-slot Bernoulli decisions, not fixed quotas.
Instead: test nested mixture membership, identical shared control tasks and configured fractions; preserve the production sampling law.

## 26-10-2026 — failed pull and reused validation receipt

| Job | Configuration | Outcome |
|---|---|---|
| 11737746 | GPU Exp0, old checkout `a073d7d` | reused GPU PASS from 11724030; exited OK at 08:33:21 CET without new validation stages |

The latest Downloads ZIP (`8a919090384f00c60c8c6bbba442a1de8bcdced0b60761eb88c5e165d0b80b5b`)
still contains 48 completed Exp1 summaries and successful benchmark logs, but no production
benchmark CSVs/journals/receipts. Old CPU/GPU receipts have fingerprint `20f0e981...`, not the
revised Exp2 `fb852116...`. Preserve the valid Exp1 evidence. Only CPU attempt 62378918 is
superseded among the three validation directories; empty lock files are normal coordination files.
Local output has no Exp0 attempts to clean. See RUNS for the audit and update sequence.

Tried: pull updated code, then submit GPU Exp0 and verify in separate command blocks.
Result: pull aborted; the later commands still succeeded against the old checkout and its receipts.
Why: executed tracked notebooks and All_Results.md blocked the merge; validation checks the installed revision, not whether origin contains a newer design.
Instead: wait for an idle checkout, stash only those generated tracked changes, pull, assert the intended source fingerprint, then submit fresh CPU followed by GPU validation.

## 25-10-2026 — Exp1 benchmark completion snapshot

| Jobs | Configuration | Outcome |
|---|---|---|
| 11734956 / 11734957 / 11734971 / 11734972 | main LGD/PD 49 slots each; component LGD/PD 15 each; five-fold credit + native OOD | 128/128 successful footers; no failed cells in logs |

Downloads `output CreditICL (1).zip` SHA256 `b978b33762131959adb5adfa1b5fd4c63685b94f49e322483e090a93543d1ac9`
contains the benchmark logs, but not project-tier benchmark CSVs/journals/receipts.
Collect those before numerical conclusions. Point-only CatBoost/linear `crps=nan` is expected.
Existing Exp0 PASS receipts are still for source `20f0e981...`; they do not validate revised
long-context Exp2. Confirm an idle checkout, update, then run CPU Exp0 and GPU Exp0 in order.
Preserve completed Exp1 outputs. Full evidence and limitations are in RUNS, 25-10-2026.

## 25-10-2026 — final Exp2 verification

The complete frozen-source suite passed: **1,787 passed, 28 VSC-only skips** in 20m41s.
No local skip substitutes for the mandatory CUDA allocation checks.
All 12 notebooks executed successfully after importing only the completed Exp1 run records.
All 37 exported PDFs were rendered and visually inspected, including the paired control-gap
curves; inline displays, captions, embedded/subset fonts and verbatim final summaries match.
The final rerun removed three stale availability PDFs; absent final benchmarks produce text
notices, not invented results. No benchmark result or checkpoint was deleted.
Local source fingerprint is `fb8521169c961e9c8edc8f464aa2ea44d802b05c39a3c499aebe6910d73376b7`.
The revised long-context CUDA route remains unvalidated on VSC. Preserve the running Exp1
benchmark revision and inspect future measured update/checkpoint time against Slurm's warning
interval before launching the more expensive Exp2 campaign.

Tried: complete-suite verification after importing completed Exp1 records and adding long-table Exp2.
Result: complete and targeted runs exposed five stale test assumptions; no production guard was relaxed.
Why: orchestration and sweep-grid tests read real downloaded output; a caption heuristic omitted score differences; prior-equality and run-card assertions still assumed all experiments used short tables.
Instead: isolate filesystem tests, retain semantic caption checks, test unchanged prior mechanisms separately from backbone-specific geometry, and require truthful run-card geometry/memory reporting; rerun the complete suite on frozen source.

## 24-10-2026 — completed Exp1 snapshot and long-context Exp2 revision

Downloads ZIP SHA256 `ddddb3b064c7d971520bffb626b2e7c67cfbceb6b81fab5133992e4dd4453c0e`
contains **48/48 completed Exp1 runs**, all 12,500 updates / 800,000 tasks, about **323.6 recorded
training GPU-hours**. No nonfinite scored monitoring values or failed training logs were found.
Development monitoring currently favours the 0% control; the full-table benchmark is not in this
ZIP. Its launch is user-reported. See RUNS for the complete evidence and scope limits.

This supersedes the earlier same-day unchanged-recipe note **for Exp2 TabICLv2 only**. Its proposed
continuation now matches upstream stage-3 geometry: 400–60,000 log-uniform rows, 79–81% context,
row-dependent feature ceilings, group/microbatch one and batch 64. Dense whole-batch padding is
replaced by per-table tensors; activation recomputation, four workers at most, prefetch one and
efficient FP32 SDPA bound memory. Optional upstream FA3 is disabled because it downcasts FP32
inputs to FP16. Exp1, Exp3 and TabPFN-3 task geometry remain unchanged. The 0% arm continues on
the installed original TabICL prior; no control is guaranteed not to degrade.

Expected TabICL row exposure is about 7.6 billion per run. Old short-table Exp2 GPU-hour estimates
are invalid. Before production, new Exp0 CPU/GPU receipts must include 28 row/feature/source
capacity corners and two real 64-task updates. Review measured memory, timing and cost before
authorizing the 36-run campaign. These new CUDA checks have not yet been run on VSC.
Keep the current Exp1 benchmark checkout frozen; no pull, cleanup or environment update while
it runs. This change does not justify retraining its completed checkpoints. Pins remain
TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465` and VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`.

Tried: targeted resume tests while independent agents were still editing shared training source.
Result: one checkpoint-resume test correctly rejected a changed training fingerprint.
Why: the implementation changed between checkpoint creation and loading during the test.
Instead: freeze all source edits and rerun the complete suite; do not weaken the resume guard.

## 24-10-2026 — figure readability and Experiment 2 audit

Exp2 remains **36 runs**: both released backbones, all three mixtures, three seeds, both tracks.
Parsed PD/LGD configurations are unchanged; only comments were corrected. The new TabPFN
guard rejects unsupported freezing/initialization and nonzero L2-SP before writing output.
Direct tests compare losses and gradients with the installed 9.1.0 fine-tuner, including
member-specific regression target transforms. Exp3 alone selects a nonzero Exp1 mixture from
complete current-protocol development results; OOD is descriptive, and seed spread is not a
significance test or tie-break rule. Historically inspected holdouts remain non-confirmatory.

Keep the active VSC checkout frozen through **Exp1 training and all credit/OOD/component
benchmarks**. The TabPFN guard changes the shared training fingerprint, and selection changes
the evaluation fingerprint, despite unchanged Exp1 numerical recipes. Preserve that checkout
and its receipts; validate the updated code separately before Exp2. These changes do not
justify retraining Exp1. No cluster job, installation, data change or submodule update occurred.
Read-only audit pins: TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465`;
VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`.

All 12 notebooks executed successfully. The 19 diagnostic PDFs were rendered and visually
inspected; 35 artificial result layouts plus four full-size result stress layouts were
checked only in TEMP. Captions, inline displays, embedded fonts and verbatim final summaries
match. No local production run/benchmark files were available: result notebooks print absence
notices rather than exporting empty PDFs. This is not a new VSC status assessment.
Full local suite: **1,701 passed / 28 VSC-only skips**, 1,112.59s. Final presentation changes
also passed 127 focused tests, with the two Bash checks passing separately after PATH setup.

Tried: recover the prior full-test session after the overnight continuation.
Result: its terminal handle had expired; a final pass/fail receipt was unavailable.
Why: process-session state did not survive the continuation; this is not evidence of a code failure.
Instead: repeat the full suite with a durable temporary log and JUnit report before publishing.

## 23-10-2026 — notebook presentation refresh

Twelve notebooks now follow the manuscript questions, with fresh seeded prior diagnostics,
native accepted-stream auditing, strict current-protocol final panels and a gated Exp3 chapter.
Training/evaluation/prior implementations and approved experiment grids are unchanged. The
active VSC checkout must remain at its validated source during the campaign: notebook/plot/test
edits change the full validation fingerprint, even though they do not change training weights.
The earlier 21/48 cluster snapshot below remains historical evidence. Its Downloads ZIP was no
longer present during notebook regeneration, and no local run/benchmark CSVs were available;
local availability notices must not be read as a new scheduler or training-failure report.

Local verification: **1,664 tests passed / 28 VSC-only skips**, 1,203.83s; all 12 notebooks
executed successfully. All 30 saved PDFs (21 diagnostics and nine availability notices) were
rendered and visually inspected, together with 35 artificial future-result layouts outside the
repository. Captions, inline figures, verbatim summaries and embedded fonts were checked.
Read-only library pins: TFM `702d9ab0b66f6639ba8d443ff9c4a91231964465`;
VSC `3bcdf55e576d10161d6189b40ac39380bbba9eca`. No production run was launched or changed.

## 20-10-2026

- Added the independent read-only VSC Library submodule and daily pin-update proposals; shared HPC guidance moved out of TFM Library, with project instructions retained in docs/VSC.md. No cluster workload was submitted.

What is worth carrying between sessions: **the cluster runs that have been done**, and **the things
that turned out not to work**. Read it before starting; add to it as you go.

Not the changelog — that records edits to the repository. This records experience: what was run,
what came out, and what is already known to fail.

**Keep it short.** One line per run, four per dead end. Newest first, dates `DD-MM-YYYY`. Never
delete an entry: a run you would otherwise repeat and a dead end you already paid for are both
evidence. The **full** write-up of each run is in [`RUNS.md`](RUNS.md); repository edits are in
[`CHANGELOG.md`](CHANGELOG.md).

**Contents:** [How a run comes back to you](#how-a-run-comes-back-to-you) ·
[Runs](#runs) (the index table) · [Dead ends](#dead-ends).

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

Latest upload (23-10-2026, 09:12 ZIP; 09:14 live queue): **Exp1 21/48 complete, 13 running,
13 normally pending, one held before training output**. Main PD 8/9 and LGD 8/9 complete;
component tests PD 0/15 and LGD 5/15 complete. All 34 started recipes match the approved grids;
all completed arms reach 12,500 updates / 800,000 synthetic tasks. Recorded updates total
365,600/600,000 (~60.9%). No training exception, failed footer, OOM, nonfinite loss or progress
metric found; sampled GPU means ~85% PD / 88% LGD. Frozen source remains 20f0e981, and the
same Exp0 PASS receipts remain intact. Benchmarking has not started; project-tier binaries absent.
**Held task 11728997_6** is LGD control seed 2, reason `launch failed requeued held`, elapsed 0.
Recommend one `scontrol --clusters=mindwell release 11728997_6`, then check its queue state;
this was advised, not executed or confirmed by this audit. If denied/held again, inspect its
`scontrol show job` record and seek VSC assistance. Leave all other jobs running; do not resubmit
whole arrays or change validated source. Normal pending reasons are Priority and the %8 array cap.
Measured finished training ~6.5h LGD / ~7h PD. Remaining walltime depends on pending allocations
and held-task recovery. Archive SHA-256 6713c528f221cf2f7dcd96693b56d1438f83ea4e175d2d33f3628ebeeb4e64dc.

Previous upload (22-10-2026, 09:11): **Exp0 CPU 62382230 and GPU 11724030 both PASS** for frozen
source `20f0e98161f9e988b5eaf0d2775fb34406547844bafbbd90a653fb9bbe7e44c5`, commit a073d7d.
CPU: 13/13 stages, 1,617 tests passed / one expected missing-sibling skip, 11 notebooks and 125 PDFs.
GPU: 66/66 stages, 28/28 CUDA tests, six 600-step stop/resume routes and all selected full-table
five-fold capacity benchmarks. CPU receipt hash, source, environment and input inventories match.
All 128 downloaded receipt-bound files match their hashes. Project-tier files (28 benchmark CSVs,
14 receipts and six checkpoints) are absent from this DATA-tier ZIP; the supplied VSC verifier
checked them in place and reports 72 items / zero FAIL. Download both tiers for independent CSV review.
LGD steps 217 and 511 replayed successfully in float32; zero skipped TabPFN tasks. No fatal errors;
multiclass rounding, upstream power-transform and utilization warnings remain disclosed in RUNS.md.
The ZIP retains one older CPU attempt (33 files, 500,191 bytes plus its job log), no failed GPU attempt.
**Proceed to Exp1 on this source; no Exp0 rerun or output reset is required.** No source/package edits,
training, installations, submissions or pushes were performed during this audit.

Previous upload (21-10-2026, 11:12): CPU 62378918 PASS for f07e71e: 13 stages, 1,590 tests passed,
2 expected skips, 28 CUDA cases deselected, 11 notebooks and 125 PDFs. All 23 hashes verified.
ZIP has one CPU attempt and no GPU/old campaign clutter. Subsequent independently confirmed audit
fixes change source and feature policy (protocol 9): publish/sync, then CPU once more before GPU.
No dataset dropped, labels flipped or output deleted. Shared project quota is now 36,730/150,000
files (243G/1T). The older near-full quota and GPU evidence below are historical.

Final corrected-source local verification (21-10-2026): `20f0e98161f9e988b5eaf0d2775fb34406547844bafbbd90a653fb9bbe7e44c5`
unchanged through the final run. **1,638 passed / 28 VSC-only skips**, zero failures/errors,
1,109.96s; 26 new behavioural regressions included. All 11 notebooks executed, 125 PDFs;
the post-generation PDF test passed. Source/config/notebook corrections include protocol 9,
paired beta RNG, signal readiness, strict skip/exit checks and truthful plotting/labels.
Sync this version, rerun CPU, then GPU; keep code, packages and inputs fixed between phases.

Current campaign decision (13-10-2026): Andreas requested a fresh output reset, superseding the
older preservation instructions below. Exp0 now requires CPU then GPU PASS receipts; the revised
Exp1 has 48 runs including component tests. Historical run entries remain an audit trail, not
reusable completion evidence. VSC submissions are made by Andreas; the rows below record the supplied evidence.

Previous VSC evidence (21-10-2026, 08:30 download): GPU 11715564 FAILED 1:0 after 10:39:48, at TabPFN LGD step 217 after 16 AMP retries. 47/66 stages passed, including 26/26 CUDA tests, wide streaming equivalence, all reference capacity benchmarks and five 600-step stop/resume arms. The local batch-217 replay exposes a safepower query target around 3.39e18; a finite loss overflows half-precision gradients. The prepared fix retries the identical batch once in float32 with stable norm clipping; fresh CPU/GPU receipts are required for changed source. Preserve this ZIP; do not resubmit the failed receipt or launch production.

Previous local final verification (21-10-2026), fixed source `f07e71e70eebd282b3401a9e745968ca47fc2edfec4821cc4608bf9eae21050a`: full suite **1,582 PASS / 58 SKIP**, zero failures, 1,175.90s. Existing Git Bash was absent from PATH; after enabling it for the test process, environment/Slurm/notebook-contract modules passed **178/178 executed tests**, one figure test skipped, in 28.94s. This covers all 29 Bash skips from the full run without source changes: 1,611 distinct local tests passed; 28 mandatory VSC tests and the missing-figure check remain unexecuted locally. Two added VSC cases test the replay and real checkpoint/resume before expensive stages. The prior CPU notebooks passed 11/11; the fresh CPU job will regenerate them. No VSC jobs, installs or pushes were performed here.

Previous snapshot (20-10-2026, 09:25): CPU 62371055 passed all 13 stages, 1,586 tests and 11 notebooks; all 23 artifact hashes verified for source `4da9e67`. GPU 11715564 was then only submitted, with no GPU files in that earlier ZIP. One old CPU attempt remained (35 files, 0.55 MB); current project-output inventory was empty at its CPU storage check. Shared file quota was 147,529/150,000; no later quota inventory was supplied.

Earlier VSC evidence (19-10-2026, 14:03 download): CPU 62369106 passed; GPU 11714993 failed its first wide streaming equivalence test after 16 passes. A local mixed-precision reproduction exposed an unintended float32 conversion before row attention. The corrected source was subsequently validated by CPU 62371055; the B200 result remains pending.

Local verification after the precision fix: fingerprint `4da9e67eb78b4e1a753b6fa8fe5580fb37b86c956df114db9e8879dab4677d8e` unchanged throughout. Full suite: 1,605 passed, 27 skipped, one Git Bash subprocess timed out at 20s (47m21s total). That test passed unchanged alone and all 16 environment tests then passed in 1.97s; the timeout's cause is unconfirmed. The skips are 26 mandatory VSC tests and one PDF check after local output cleanup. Do not describe the full invocation as green or infer a B200 PASS from these local checks.

Earlier local readiness (19-10-2026, before this GPU finding): source `6f2f530d982a95f9c03b352a44de80e2ede28b1c84952587fa544744f9350857`, library `de324ef`.
Full suite: 1,597 PASS / 26 VSC-only SKIP in 15m14s; notebooks: 11/11, 125 figures. The authorized local reset then removed 140 generated files and cleared 11 notebooks' outputs, with zero generated files remaining. All 71 input/weight files (3,294,397,860 bytes) and executable source were hash-verified unchanged. No CreditICL output ZIP remained in Downloads; the unrelated JEVPFN archive was untouched. Publish the complete moved/new-file change set before VSC cleanup and fresh CPU (13 stages) then GPU (66 stages) Exp0. Local CPU tests are not a VSC PASS receipt.

One row per cluster run worth remembering — which is most of them, because *"have we already tried
that configuration?"* is the question this table exists to answer. The **full** write-up of each
one lives in [`RUNS.md`](RUNS.md); this table is the index.

| Date | Run | Outcome | Notes |
|---|---|---|---|
| 23-10-2026 | Exp1 main LGD 11728997, Mindwell B200; started 22-10 | **8/9 complete, one held launch failure, no running task at 09:14** | All eight summaries 12,500 steps / 800,000 tasks; array 6 is control seed 2, no run output and elapsed 0, Slurm `launch failed requeued held`. Single-task release advised, outcome pending |
| 23-10-2026 | Exp1 main PD 11728998, Mindwell B200; started 22-10 | **8/9 complete, array 8 running at step 9,600/12,500 in snapshot** | Training configurations match; finite losses/monitoring and positive weight drift. ~1.6h estimated remaining for active task from 09:12, not a scheduler guarantee |
| 23-10-2026 | Exp1 LGD component tests 11728999, Mindwell B200; started 22-10 | **5/15 complete, eight running, two pending (%8 cap)** | Active updates 5,000–10,200; finished-arm training median ~6.5h. No training failures in uploaded evidence |
| 23-10-2026 | Exp1 PD component tests 11729000, Mindwell B200; started 23-10 | **0/15 complete, four running, eleven pending (Priority)** | Active updates 1,900–8,100; finite losses/monitoring. All 48 expected tasks accounted for by ZIP plus live queue |
| 22-10-2026 | Exp0 GPU 11724030, Mindwell B200/r11g12, a073d7d; started 21-10 | **PASS: 66/66 stages, 28/28 CUDA tests, all six stop/resume arms reach 600; 17h38m40s banner-to-footer** | Source 20f0e981, bound to CPU 62382230; all downloaded hashes match, 48 project-tier artifacts omitted from ZIP but verified in place by VSC (72 items / zero FAIL). LGD float32 replays at steps 217 and 511 succeeded; zero skipped tasks. Exp1 may proceed without source changes |
| 21-10-2026 | Exp0 CPU 62382230, wICE batch/m33c25n4, a073d7d; audited 22-10 | **PASS: 13/13 stages, 1,617 passed / one expected skip, 11/11 notebooks, 125 PDFs; 21m04s banner-to-footer** | Source 20f0e981; 23/23 artifact hashes verified, exact CPU receipt consumed by GPU 11724030. Input inventories and package fingerprint identical between phases |
| 21-10-2026 | Exp0 CPU 62378918, wICE batch/m33c22n1, `576caea` | **PASS: 13/13 stages, 1,590 passed / 2 expected skips, 11/11 notebooks, 125 PDFs; 22m50s log interval** | Source f07e71e, 23/23 artifact hashes match; one clean CPU attempt in ZIP. Subsequent protocol-9 audit corrections require a fresh CPU receipt before GPU |
| 20-10-2026 | Exp0 GPU 11715564, Mindwell B200/r11g22, `d31770f`; audited 21-10 | **FAILED 1:0 after 10:39:48; 47 stages PASS, resume-lgd-2 FAIL** | 26/26 CUDA checks and wide equivalence passed; four reference models completed full-row five-fold capacity checks, five 600-step arms completed. TabPFN LGD stopped at step 217 after 16 AMP retries; last recorded checkpoint step 200. Local replay exposes a transformed target 3.39e18; fixed-source CPU/GPU validation required |
| 19-10-2026 | Exp0 CPU 62371055, wICE batch/m28c20n4, `d31770f`; audited 20-10 | **PASS: 13/13 stages, 1,586 tests PASS/1 SKIP, 11/11 notebooks, 125 figures; 21m09s main-log interval** | Source `4da9e67` and 23/23 artifact hashes verified; all eight new mixed-precision cases pass on CPU, 26 CUDA checks deferred to GPU phase. No production arms started |
| 19-10-2026 | Exp0 GPU 11714993, Mindwell B200/r11g17, `b351325` | **FAILED exit 1; 16 CUDA tests PASS, 1 FAIL; 2m18s main-log interval** | PD 2,986-feature streaming predictions differed from stock by up to 0.00238492 (3/16 probabilities outside tolerance). No long training or capacity stages started. Local AMP reproduction confirms a float32 boundary conversion; remove it, retain tolerance, revalidate on B200 |
| 19-10-2026 | Exp0 CPU 62369106, wICE batch/l33c22n2, `b351325` | **PASS: 13/13 stages, 1,575 tests PASS/2 SKIP, 11/11 notebooks, 125 figures; 35m10s log interval** | All 23 CPU artifact hashes match; GPU binds this exact receipt and input inventory. All 21 caches v8 and 50 OOD tables present. Subsequent precision/test changes require a new CPU receipt |
| 19-10-2026 | Follow-up of GPU 11692660 and VSC checkout | **CANCELLED by 2538338 after 22:36:01; batch 0:15, 22:36:10** | ZIP still has schema-2 RUNNING / 12 of 66 stages and a false OK footer. All 22 CPU hashes match; no new model errors. VSC remains `cf0e539` / fingerprint `adf308f...`; 16-10 fixes remain unpublished locally. Add signal-aware status/tests, publish fixes, then clean CPU/GPU validation; do not resume the old slow code |
| 16-10-2026 | Exp0 GPU 11692660, Mindwell B200/r11g22, `cf0e539` dirty; started 15-10 | **RUNNING at 09:01, 22h20m/24h; 12/66 stages PASS; cannot finish at observed pace** | All 17 CUDA tests, six smoke routes and both equivalence checks passed. Released TabICL passed five folds on 759-feature loan_default; 2,986-feature algorithmwatch took 13h34 for fold 0, fold 1 still predicting. Current benchmark writes results only at stage end; validation has no continuation. Keep evidence, address runtime/persistence/logging before rerun; Exp1 blocked |
| 15-10-2026 | Exp0 CPU 62324362, wICE batch/l33c20n3, `cf0e539` dirty; audited 16-10 | **PASS: 12/12 stages, 1,437 passed / 3 skipped / 17 CUDA deselected; 16m31s log interval** | All 22 artifact hashes verified from ZIP; executable fingerprint matches local checkout and GPU, environment/inputs match, GPU binds exact CPU receipt. This is valid for the audited source; future executable fixes require renewed receipts |
| 14-10-2026 | Exp0 GPU 11689448, Mindwell B200/r11g22, commit `351d948` dirty | **OUT_OF_MEMORY, 0:125, 41m32s; 12 stages PASS** | Accounting/stderr supplied at 19:37–19:38 confirm host RAM OOM: MaxRSS 188,735,484 KiB against 180G, one oom_kill. All 15 CUDA checks, six smoke routes and both equivalence checks passed; full-table TabICL completed two datasets then died on 105,471 rows/759 features. Local GPU/disk memory fix needs fresh CPU then GPU validation; no Exp1 launch |
| 14-10-2026 | Exp0 CPU 62319803, wICE batch/m28c22n1, commit `351d948` dirty | **PASS: 12/12 stages, 1,404 tests PASS/3 SKIP; 16m02s log interval** | 21 credit and 50 OOD tables checked, package requirements passed, all 22 receipt artifact hashes verified from ZIP. CPU/GPU source and input fingerprints match. Subsequent inference-code changes require fresh CPU and GPU validation |
| 14-10-2026 | Exp0 CPU 62316841, wICE batch, node m28c27n1, commit `6054c50` dirty | **FAILED; 11/12 stages PASS; 1,368 tests PASS, 2 FAIL, 3 SKIP, 13 deselected; 16m20s log interval** | Two legacy readiness tests patched only `experiment_dir` and read the actual failed GPU receipt through `outputs_dir`. Reproduced locally with a failed outer receipt, then both passed using isolated output roots. Production gate correctly blocked GPU. Fresh CPU then GPU required after these source changes |
| 14-10-2026 | Exp0 GPU 11688276, Mindwell B200, commit `57d36cd` dirty, after CPU 62313072 | **FAILED 1:0, 01:58:32; 17 stages passed, `resume-pd-2` failed** | All six smoke tests, both equivalence checks, 11 mandatory CUDA tests and both TabICL PD stop/resumes passed. TabPFN PD restored CPU Adam moments before its first CUDA forward; first resumed update crashed. Place model before optimizer creation, add early PD/LGD CUDA resume probes, rerun CPU then GPU for changed source; no full-table benchmarks reached |
| 14-10-2026 | Exp0 CPU 62313072, wICE | **PASS confirmed by readiness check at 11:55 CEST** | Receipt `cpu_20261014T092548Z_j62313072_c6ec3745`; all mandatory stages/artifacts current. All 21 credit caches v7, all 50 OOD tables present, four released weights present, 135 dependency checks passed. Detailed test counts and elapsed time not supplied |
| 14-10-2026 | Exp0 CPU 62311762, wICE batch, commit `2380e36`, library `de324ef` | **FAILED exit 1, 14m53s; 11/12 stages PASS; 1,326 tests PASS, 2 FAIL, 7 SKIP, 10 deselected** | Both failures were test path assumptions; GPU submission was correctly blocked. All 21 credit / 50 OOD data checks and dependency checks passed. Reset removed 310 files / 60.785 GB, including legacy experiment checkpoints; second dry run found zero. New fixes require CPU then GPU validation, no further reset |
| 13-10-2026 | Exp0 CPU 62306531, wICE, commit `91a568a` | **RUNNING 00:10:32 at 16:53; no completion supplied** | 1,703 current-output files/0.042 GB removed on both tiers; subsequent listing proves flat `checkpoints/exp1_*` survived. Environment audit passed (152 packages), 50 full-row OOD tables refreshed. Protocol 7/full-feature changes require a new CPU PASS before GPU validation; do not update or reset a running checkout |
| 12-10-2026 | Exp0 checks 11681071, audited 13-10 from ZIP | **FAILED: 95 PASS, 2 FAIL, 1 WARN; both equivalence checks passed** | 21m01s log interval; PD TabPFN nonfinite scaled gradient; LGD TabPFN CPU bar-distribution buffers with CUDA logits. Input/package repairs succeeded. New single-job validation must pass before the corrected 48-run Exp1; preserve historical 90 |
| 12-10-2026 | Preprocessing, job 62297440, wICE, after `57377b2` | **COMPLETED 0:0, 00:03:00** | VSC installed TabPFN 9.1.0; correct TabICL commit and CatBoost 1.2.10 confirmed. Preflight ran while caches were rebuilding (13 current, 8 stale) and correctly prevented GPU submission. Repeat full preflight after completion; do not resubmit preprocessing |
| 12-10-2026 | Exp0 checks, job 11680453, Mindwell `gpu_b200`, after `78f6b3f` | **FAILED 1:0, 00:17:27; both trainer-equivalence checks passed** | 21 stale v3 data caches, missing new OOD path while legacy cache exists, installed TabPFN 8.2.0 rejecting v9 inference flags. Repair environment/cache, preflight, then repeat checks. No six-arm Exp0 or benchmark completion yet; 0.291 GPU-h used |
| 12-10-2026 | September benchmark CSV follow-up, `benchmark-results.zip` | **10,437/10,437 rows OK; no nonfinite headline metrics; 184/184 result hashes match receipts** | All 12 recorded evaluation source hashes match `2034cc2`. The 28-09 dataset-calibration development change postdates this campaign; do not attribute it to these checkpoints. Unequal contexts and AXA boundary diagnostics need re-evaluation only; the corrected optimizer/prior require a new 18-arm Exp1 |
| 12-10-2026 | Audit of downloaded September Exp1 archive, SHA-256 `537f8272f6530e9cc890bddcc839b3ebe64a70ee23a9d8ca99135e017a01b928` | **90/90 training complete; 92/92 benchmark jobs OK; latest event 24-09 22:00 CEST** | 3,087 credit and 7,350 OOD cells reported successful; no benchmark NaN/error lines. CSVs and checkpoint binaries absent. This is the old 45-arm grid, not the revised 9-arm grid; preserve it, download project-storage results, then validate the revised pipeline through Exp0. No jobs submitted |
| 24-09-2026 | Exp1 phase 2 benchmark: 11614376 (LGD), 11614377 (PD), Mindwell, arrays 0-45%8 | **COMPLETE 92/92, all OK; credit prior ≈ control on dev, cf 1 worse on both tracks** | PD dev ROC-AUC: best control 0.7326, best credit (cf 0.5 banded aggr) 0.7325, cf 1 0.70–0.72, released TabICLv2 0.7322. LGD dev R²: control 0.520, cf 0.5 ≤ 0.518, cf 1 ≈ 0 (collapse, also OOD). All cf 0/0.5 configs within 1–2 seed SDs. Commit `2034cc2` |
| 24-09-2026 | Exp1 final download, 19:41; last jobs PD 11604659 / 11613401 / 11613402, LGD 11591463_43 | **COMPLETE: 90/90 arms, all `END status=OK`; phase 2 not run** | ≈1,354 GPU-h (PD 587, LGD 767; banded arms 5–7× the rest). PD monitor (german+myhom, final step): control 0.672, cf 0.5 0.669, cf 1 0.646; OOD 0.997/0.992/0.975. LGD monitor unusable (base_model NaN in 30/45). Monitoring protocols differ — do not rank on them; phase 2 decides |
| 24-09-2026 | Exp1 snapshot at 09:31 CEST; PD 11598087 → 11604654–61, LGD 11591463_43 | **86/90 complete: PD 42/45, LGD 44/45; four advancing, no new training crashes** | Six PD retries completed; all eight overnight walltime continuations resumed at the exact saved step. Remaining PD a19=11000, a25=12250, a40=7700; LGD a43=12200 of 12500. Conditional finish today ~17:30–18:30 plus queue delay. New PD monitor: 160 finite rows; old LGD monitor still has NaNs in 30 arms. Final benchmark results were not in this download |
| 23-09-2026 | Exp1 PD retry 11598087, Mindwell; indices 1,4,10,16,19,25,31,34,40 | **SUBMITTED at 12:22 CEST; execution/results not yet supplied** | Updated launcher submitted only the nine unfinished PD arms; counts remain PD 36/45, LGD 44/45 with LGD still running. Failed-log cleanup returned without error at 12:23; checkpoints and successful logs were outside its scope |
| 23-09-2026 | Scheduler follow-up for 11591462 / 11591463 at 10:30–10:39 | **Confirmed: nine PD FAILED 134; LGD 13/28 completed, 43 RUNNING; benchmark 0/46 per track** | Local fixes prepared; no new cluster jobs submitted. Resume unfinished training; preserve all checkpoints and successful logs. This is follow-up evidence for the snapshot below |
| 23-09-2026 | Exp1 rerun snapshot; latest PD 11591462, LGD 11591463, Mindwell B200 | **80/90 complete: LGD 44/45, PD 36/45; nine PD crashes, one LGD progressing** | LGD a43 at 7300/12500 (09:28 CEST), conditional ETA 24-09 ~11:24. CPU reproduction proves invalid `-100` synthetic fallback reaches encoder; actual crashed batches unavailable. LGD monitor NaNs affect 30 arms. No results tier supplied; read RUNS.md before retrying |
| 07-09-2026 | 11549669 (PD, mindwell `gpu_b200`), 61898850 (LGD, wICE `gpu_a100`) — Exp1 phase-1 re-run, 90 arms | **PARTIAL 63/90; all 27 unfinished are `filter=banded`; 15 PD banded arms CRASHED (OOD `-1` → one_hot device assert)** | Root cause fixed in `src/eval/ood.py`. Non-banded 30/30 (PD) + 30/30 (LGD) done. Progress-eval only (phase-2 not run): cf=0.5 blend leads both tracks — PD AUC 0.723 vs control 0.721, LGD R² 0.392 vs 0.220; cf=1 trails. Suggestive, not the verdict |
| 02-09-2026 | 61866150 (LGD), 61866151 (PD) — Exp1 Phase 2, all 150 checkpoints scored, wICE `gpu_a100` | **NULL: credit prior ≈ no-credit control on BOTH tracks; 0/25 beat released TabICLv2** | LGD best 0.486 vs control 0.485 (seed SD 0.008) vs released 0.514; PD control IS rank 1 (0.730) vs released 0.736. `scale=standard` hurts LGD. No OOD loss. Fair internal test is a clean null |
| 01-09-2026 | 11529826 + 61791522 (LGD), 11529827 (PD), +resubmits 11540142/43 — Exp1 full sweep, 150 arms | **COMPLETE: 150/150 arms, all healthy** | 1,011 GPU-h. LGD loss 0.051–0.070, PD 0.136–0.186. 0 CUDA / NaN / walltime. The 6 PD (Vasicek) + 2 wICE stragglers finished on resubmit |
| 25-08-2026 | 61776784 — GPU benchmark, wICE `gpu_a100` | **A100 is 1.93x slower, 2.72x cheaper/hour -> 29 % cheaper per arm** | 8.49 h/arm, 89 % of its own ceiling. Raw FLOPs say 5.28x; this workload is latency-bound |
| 24-08-2026 | 11524582 -> 11524593 — Exp1 LGD arm 0, deliberate kill test | **Whole resilience chain fired; arm 0 COMPLETED across two jobs** | signal -> checkpoint -> exit 64 -> resubmit -> resume. 0.82 steps/s, 88.75 % GPU, loss 0.343 -> 0.059 |
| 24-08-2026 | 11523286 — GPU benchmark, clean | **One arm is 4.4 h, so Exp1 is ~10 M credits and 1.8 days** | AMP is 2.07x. All 4 attention kernels work at the real shape — the 20-08 cuDNN crash was the benchmark's own oversized pass |
| 20-08-2026 | 11521108 — GPU benchmark at batch 64, `gpu_b200` | **Found a cuDNN AMP crash that would have hit all 75 Exp1 arms** | Also: benchmark OOM (its own bug, no micro-batching), and Muon is 1.01x at batch 64 not 1.40x |
| 20-08-2026 | 11520989 — GPU benchmark, `gpu_b200` | **Wasted: 4 of 6 rungs ignored the config it was sent to measure** | Only prior + end-to-end read it. 8.26 -> 4.589 steps/s at batch 4 = 1.80x, but batch 4 is not what we train at |
| 19-08-2026 | 11520343 — Exp1 LGD debug, `gpu_b200` | **Pipeline fair at last: R2 positive on ALL 7 datasets** (was -1.44..-0.25) | Wrapper + shared 1,024 context cap. Gap to released now a uniform 2-4x — budget, not plumbing |
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

### 23-10-2026 — validate all declared captions before the slow suite

- **Tried**: run the full local test suite before auditing every caption in data-gated notebook branches.
- **Result**: 1,661 tests passed, 28 VSC tests skipped; one OOD caption contract failed after 18 minutes. Local CPU work only.
- **Why**: the binary/multiclass caption omitted its comparator, and did not match the concrete-quantity contract; the branch was not visible with missing benchmark data.
- **Instead**: name the equally trained 0% comparator explicitly and run all four caption checks over every declared save call before the full suite, including withheld future panels.

### 23-10-2026 — notebook gate and controlled-illustration settings

- **Tried**: read the Exp3 mixture from `load(...)["sweep"]`; retain the correlation illustration's scalar selection-disable setting.
- **Result**: the first local notebook pass finished 11/12; visual/source review found selection still active in that diagnostic. No cluster run was submitted.
- **Why**: `load` resolves/removes the sweep block; `selection_drop_range` takes precedence over the scalar. Old plotting prose did not establish the actual controlled settings.
- **Instead**: read the resolved `prior.credit_fraction`, override the range in the illustration, test observer equivalence/current-protocol gates/layout, and visually inspect rendered PDFs. Production generation and evaluation stay unchanged.

### 23-10-2026 — one production array task held after launch failure

- **Tried**: launch the nine-arm LGD main array 11728997 on Mindwell.
- **Result**: eight tasks completed; index 6 remains held at elapsed 0 with no training output while other arrays progress.
- **Why**: Slurm reports `launch failed requeued held`; its underlying node/filesystem/program cause is not established by the supplied queue/accounting records.
- **Instead**: release only 11728997_6 once and inspect its state; if denied or held again, read its detailed scheduler record and involve VSC support. Do not duplicate whole arrays or change the running validated code. Release outcome not yet supplied.

### 21-10-2026 — verify the entire corrected suite after draft failures

- **Tried**: full local verification before the final Windows fixture and reporting wording were settled.
- **Result**: first full trial had 1,634 passes / 29 skips / two failures; an intermediate focused run had one stale summary-wording assertion. The final serial suite is green (1,638/28).
- **Why**: one new mock assumed SIGUSR1 existed on Windows; the wording assertion encoded the old default-label claim. A one-off continuation failure passed unchanged in isolation and the final full run; its initial cause is unconfirmed.
- **Instead**: correct the portable mock and semantic expectation, retain strict assertions, execute notebooks, freeze source 20f0e981 and complete a new full suite with structured JUnit evidence. Never claim a failed trial passed.

### 21-10-2026 — portable test of the Unix stop-stage handshake

- **Tried**: fake a resumed training child with an existing checkpoint to test that its handler is ready before SIGUSR1.
- **Result**: the new local test failed first on an invalid fixture run name, then because Windows has no SIGUSR1; this is not a VSC runtime failure.
- **Why**: the mock used an unrecognised experiment prefix and assumed Linux's signal constants existed on Windows.
- **Instead**: use a valid Exp0 name and mock the signal number; exercise the same parent/child logic without skipping. All 25 new regressions pass in the focused run.

### 21-10-2026 — finish numerical recovery review before final verification

**Tried:** Broad local verification while reviewing the new full-precision recovery path.
**Result:** Two draft broad invocations were deliberately stopped while completing review; neither is counted as final verification. Direct telemetry reproduction reported infinity for finite 1e30 gradients.
**Why:** The old diagnostic norm/square overflowed in float32, and retaining unsuccessful scale backoffs after a float32 replay could underflow later ordinary AMP gradients.
**Instead:** Use stable diagnostic norms and restore the calibrated scale on replay; the final frozen-source suite passed 1,582 tests, with 29 Bash checks subsequently recovered unchanged from skips.

### 21-10-2026 — loss scaling cannot recover every finite regression loss

**Tried:** GPU 11715564 with bounded identical-batch AMP retries, after CPU PASS.
**Result:** Five arms finished; TabPFN LGD failed at step 217 despite scale 128 falling to 0.001953125, after 10h40m allocated.
**Why:** A locally regenerated task in that batch has a safepower query target around 3.39e18. The finite MSE overflows float16 gradients even at small scales; repeated scale backoff alone is insufficient. No CUDA replay of the failed checkpoint is available locally.
**Instead:** Replay the same batch once in float32, disable the estimator's nested autocast too, retain objective/RNG/budget, clip its norm with a float64 reduction, and test replay plus checkpoint/resume before long GPU stages. Persistent float32 failures still stop the run.

### 19-10-2026 — one local Git Bash deadline during the full suite

**Tried:** Ran the complete suite with fixed source, one BLAS/OpenMP thread and Git Bash available.
**Result:** 1,605 passed / 27 skipped; the setup-script guard test timed out in `subprocess.communicate` after 20s. Full run took 47m21s.
**Why:** A Bash subprocess exceeded its deadline; its exact cause was not established. The unchanged case passed alone (2.61s) and its entire 16-test environment module passed afterwards (1.97s).
**Instead:** Retain the guard, assertion and deadline; report the first-run timeout transparently, keep the corrected source fixed, and require fresh VSC CPU/GPU receipts.

### 19-10-2026 — CPU defaults did not exercise streaming mixed precision

**Tried:** Validated streamed TabICL on CPU with its automatic precision defaults, then submitted GPU 11714993.
**Result:** Sixteen CUDA tests passed; wide PD equivalence failed after two minutes, maximum probability difference 0.00238492.
**Why:** CPU auto disables AMP; streaming forcibly widened AMP column output to float32, unlike upstream's direct return. Explicit FP16/BF16 CPU probes reproduce the boundary mismatch in six cases; this does not yet prove the entire B200 discrepancy resolved.
**Instead:** Preserve the computation dtype, add binary/regression/radix/affine mixed-precision tests and diagnostic dtype reporting, keep the original GPU tolerance, and obtain new CPU/GPU receipts before production.

### 19-10-2026 — do not change caches or source during final verification

**Tried:** Ran the broad suite while completing review fixes and rebuilding notebook data caches.
**Result:** 1,572 passed, 15 failed, 27 skipped in 24m43s; this was not a valid final verification.
**Why:** Tests observed changing cache/source identities; other failures exposed old protocol/path fixtures, a real-scheduler fixture, an invalid upstream affine/grouping combination and a missing summary protocol label.
**Instead:** Finish preprocessing and corrections first; execute notebooks and the final test suite sequentially with fixed source and one BLAS/OpenMP thread.

### 19-10-2026 — native multiclass must exercise the streaming path

**Tried:** Native 3/11-class OOD wrappers plus forced streaming with released weights.
**Result:** Ordinary wrappers passed; the 11-class streamed path raised an unsupported mixed-radix error.
**Why:** The earlier streaming implementation covered binary/regression only; removing OOD binarisation exposed the missing path before VSC submission.
**Instead:** Use upstream digit decomposition/caches, average hidden states before affine/row attention, and test small/released CPU plus mandatory GPU equivalence; include trained-checkpoint multiclass probes in all PD smoke routes.

### 19-10-2026 — review changes invalidate earlier local and VSC receipts

**Tried:** Assess each Claude finding against the pinned code and actual analysis paths.
**Result:** Confirmed reporting/prior/edge-case defects; empirical shortcut percentages were not independently reproduced. Selection-induced class imbalance is intentional; primary AUC/R² selection need not change.
**Why:** Earlier tests did not cover native multiclass streaming or final OOD/ablation reporting. Broadening numeric prior ranges did not undo earlier benchmark-guided mechanism design.
**Instead:** Use protocol/cache v8, independent component RNGs and transparent provenance; run both fresh Exp0 phases on the same frozen source after publishing all moved/new files. Do not reuse pre-review PASS receipts.

### 19-10-2026 — Cancelled validation printed a successful shell footer

**Tried:** Interpreting the latest downloaded output and the VSC preflight failure.
**Result:** `END status=OK` contradicted a RUNNING receipt; sacct confirms cancellation after 22h36m, batch signal 15.
**Why:** The old EXIT trap trusted `$?` without a validator-completion flag or signal forwarding. VSC still ran the old fingerprint; the 16-10 fixes were uncommitted locally.
**Instead:** Forward signals, require successful validator completion for OK/cleanup, publish interrupted progress, test real shell signals, and label stored RUNNING separately from scheduler status. Publish reviewed fixes before one clean CPU/GPU campaign.

### 16-10-2026 — Restarted result files must retain their byte identity

**Tried:** Persist benchmark rows using JSON with recursively sorted keys, then recreate the CSV on restart.
**Result:** End-to-end CLI tests preserved the scores and avoided repeat predictions, but changed CSV column order.
**Why:** Sorted metric keys changed pandas' column order, invalidating hash-based benchmark receipts.
**Instead:** Preserve journal insertion order; both credit and OOD CLI restart tests require identical CSV bytes and no repeated model calls.

### 16-10-2026 — Memory-safe inference was not sufficient for a one-day validation allocation

**Tried:** GPU 11692660, the bounded GPU/disk adapter and the complete 66-stage Exp0 GPU plan in one 24-hour job.
**Result:** The earlier 759-feature OOM did not recur, but algorithmwatch's first full-width fold took 13h34; still in stage 13 after 22h20. No new numerical error or crash was observed at 09:01.
**Why:** Large-table prediction dominates, while the validation budget assumes all stages fit in one day. Completed benchmark cells are not saved until stage completion and the validator has no continuation. The offload logger is also disconnected from run logging; exact compute/I/O attribution remains unmeasured.
**Instead:** Preserve logs; recommend cancelling the infeasible attempt. The 16-10 fix streams upstream ISAB context into bounded row blocks, journals every completed fold, resumes verified Exp0 stages, connects memory logs and enforces a two-hour cell guard. Fresh CPU/GPU validation is required; B200 speed is still unmeasured. Keep full rows/features/folds; DEFERRED is continuation, never PASS. See RUNS.md for evidence and deadlines.

### 14-10-2026 — Full-row, full-width TabICL inference exceeded the tested resource envelope

**Tried:** GPU job 11689448 after CPU 62319803 passed; released TabICL on the selected full-table PD benchmarks, before the six longer diagnostic training runs.
**Result:** Five folds each completed for hackerearth (532,428 rows) and bank_status (100,000); loan_default prediction was killed after fitting 67,500 rows x 759 features. Later accounting/stderr confirmed OUT_OF_MEMORY at 41m32s, 188,735,484 KiB MaxRSS, one oom_kill.
**Why:** Host RAM reached the 180G allocation. Confirmed unsafe paths in pinned TabICL: node-level RAM estimation, no configured disk offload, and `ColEmbedding._inference_without_feature_group` copying the complete embedding with `repeat` even for one member.
**Instead:** Keep every row/feature/member, execute members sequentially, bound column transfers, preserve disk-backed embeddings and refuse CPU/swap fallback. Reserve disk blocks before writes on Linux and test real disk offload on CUDA. Andreas subsequently authorized removing stale GPU output; use `clean_run --reset-exp0-gpu --clean`, preserving CPU records. Changed code still requires fresh CPU/GPU receipts.

### 14-10-2026 — Full reset rejected VSC's storage alias

**Tried:** At 17:11 CEST, reset both output tiers after pulling `16a8fd4`, before submitting fresh Exp0 CPU validation.
**Result:** Planned 186 files / 1.524 GB, then refused `slurm-crediticl-exp0-cpu-62313072.out`; no file deleted and no new job submitted.
**Why:** Root Slurm logs retained `/data/...` as their containment root, while the final check resolved the file to `/vsc-hard-mounts/...`. Real ancestor-alias tests reproduced the refusal and duplicate counting.
**Instead:** Canonicalize project parents before discovery and deduplicate them; still reject linked outputs, changed files and active jobs. Retry the same reset/CPU submission chain after updating the code.

### 14-10-2026 — A new split test leaked an imported mock into later OOD tests

**Tried:** Run the full local suite after adding CPU checks for all five inner PD splits.
**Result:** Three later OOD tests saw an empty dataset list, although all 70 OOD tests passed alone.
**Why:** The new fixture first imported `ood_runner` while its source module's dataset-list function was mocked, retaining that alias after teardown.
**Instead:** Import and patch both bindings through `monkeypatch` so both are restored; the combined 72-test reproduction passed. Final full suite after all fixes: 1,419 passed / 17 expected skips in 15m50s.

### 14-10-2026 — CPU tests depended on real validation output

**Tried:** CPU job 62316841 after the TabPFN resume fix; legacy readiness tests mocked only `experiment_dir`.
**Result:** Two assertions failed; 1,368 tests and all 11 other validation stages passed. Unit stage took 703.39 seconds.
**Why:** The modern canonical receipt uses `outputs_dir`; VSC had a failed GPU receipt there, whereas the local checkout did not. The production refusal was correct, but the fixtures were not isolated.
**Instead:** Isolate both output tiers in those tests. Reproduce with an existing failed outer receipt; both now pass. Keep mandatory modern receipts and run CPU before GPU.

### 14-10-2026 — GPU validation 11688276 failed after CPU validation passed

**Tried:** Run GPU validation after CPU 62313072's verified PASS, including stop/resume of all six training routes.
**Result:** Job 11688276 failed after 01:58:32 on TabPFN PD's first update after restoring step 101; 17 preceding stages passed.
**Why:** The unfitted TabPFN estimator kept parameters on CPU during Adam state restoration; its first forward moved parameters to CUDA, leaving moments on CPU. CPU-only resume tests could not expose this mismatch.
**Instead:** Move the model before building AdamW, matching the installed TabPFN fine-tuner; cover placement locally and real PD/LGD AMP save/resume early in mandatory VSC tests. Preserve failed-attempt logs; new source requires fresh CPU/GPU receipts.

### 14-10-2026 — Automatic approval blocked temporary notebook-output cleanup

**Tried:** Remove only this session's 119 PDFs, 11 figure manifests and two generated summaries after notebook verification.
**Result:** Both the bounded recursive command and the narrower individual-file command were rejected before execution.
**Why:** Automatic approval returned only "blocked by policy"; this was not a filesystem error.
**Instead:** Leave the untracked artifacts intact and give Andreas exact manual cleanup instructions; no other deletion route was attempted.

### 14-10-2026 — Portable tests assumed the laptop storage layout

**Tried:** Run the complete CPU suite on wICE after the fresh-output reset.
**Result:** Notebook-output and OOD-path assertions failed; four released-weight tests skipped despite weights on project storage.
**Why:** Fixtures assumed big/small storage shared a parent and ignored the live legacy OOD directory; weight tests used a local glob.
**Instead:** Isolate the path fixtures and use the production weight resolver. CPU PASS must precede the GPU allocation.

### 14-10-2026 — Full local regression run still encoded superseded data assumptions

**Tried:** Full suite after preprocessing, grouping and storage repairs.
**Result:** First run: 1,372 passed, 7 failed, 13 skipped; the next full run found one remaining stale metrics-key assertion (1,387 passed).
**Why:** Fixtures expected old data values/config keys or inherited scheduler state; soft-clipped context means were treated as a centering oracle.
**Instead:** Compare unchanged sibling columns by name, separately test corrected recipes, isolate the scheduler, and compare shifted tables directly with upstream preprocessing. Final full suite: 1,388 passed / 12 expected skips in 16m35s; all 11 notebooks passed. VSC CPU/GPU validation is still pending.

### 14-10-2026 — Finite float32 values still overflowed full-table preprocessing

**Tried:** Clip source values at float32's maximum, checking that each value was finite.
**Result:** Loan-default columns f391/f627 still overflowed float32 mean/variance reductions.
**Why:** Representability of individual values does not ensure safe reductions across a large table.
**Instead:** Fixed signed-log transforms for those source columns, a conservative numeric bound, cache version 7, reduction regressions and a full-row Exp0 GPU probe.

### 13-10-2026 — Windows OOD manifest overwrite intermittently denied

**Tried:** Full local pytest after removing benchmark feature caps.
**Result:** 1,346 passed, 12 skipped; two failures: a stale monitor-test expectation and WinError 5 replacing an OOD manifest. Independent OOD reruns passed.
**Why:** Monitor now correctly retains all columns. Windows intermittently refused atomic file replacement; the external lock owner is not established.
**Instead:** Update the monitor expectation and retry Windows access/sharing failures for at most 3.1 seconds, preserving the old cache if retries fail; test both transient and permanent refusal. Prevent a monitor metric mock escaping through a lazy runner import. Final full local suite: 1,353 passed, 12 expected skips (10 VSC-only).

### 13-10-2026 — Reset missed pre-migration checkpoints; full-table benchmark still capped features

**Tried:** Full reset of both `output CreditICL` trees and full-row protocol 6.
**Result:** VSC removed only 42 MB; historical `checkpoints/exp1_*` remained. Benchmarks still selected 100 columns.
**Why:** Legacy run folders shared the released-weight parent outside the output tree; the row-only protocol change retained the old column budget. TabPFN also had internal per-member feature budgets.
**Instead:** Reset recognises only legacy experiment subdirectories, protecting released files. Protocol 7 retains all eligible columns, overrides TabPFN member budgets with a runtime coverage check, and adds local/CUDA wide-input tests. Revalidate CPU then GPU after updating; no cluster rerun submitted here.

### 13-10-2026 — Windows read-only empty output directories

**Tried:** The requested full local output reset after the complete tests and notebooks passed.
**Result:** Generated files were removed, but `rmdir` raised WinError 5 on an empty figure directory.
**Why:** Windows read-only directory attributes block `rmdir`; reproduced in a temporary directory.
**Instead:** Recheck directory boundaries and clear only that attribute on empty Windows output directories. Added a regression; 137 relevant tests passed, and cleanup completed with only the tracked output README retained.

### 13-10-2026 — Full-row protocol and fresh-output test migration

**Tried:** Full local suite after removing capped benchmarks and tightening complete-matrix reporting.
**Result:** 1,327 passed, 10 failed, 8 skipped; repaired focused rerun passed 425 tests.
**Why:** Legacy fixtures expected capped/one-fold results, generated summary files or pre-imputed OOD data; the benchmark Slurm default also retained obsolete array slots.
**Instead:** Preserve explicit protocol-5 fixtures for historical readers, test complete protocol-6 matrices, require only tracked scaffolding, preserve fold-local missing-value handling, and set the main Exp1 default to 49 slots. Final verification: 1,339 passed/7 expected skips; 11/11 notebooks, 119 figures.

### 13-10-2026 — New validation exposed stale test fixtures and monitor semantics

**Tried:** Full pytest after adding component configs and strict benchmark row receipts; targeted OOD monitoring regressions.
**Result:** Two old fixtures failed; three monitor assertions reproduced clipping/category faults.
**Why:** Fixtures bypassed config inheritance or omitted row counts; the monitor clipped OOD regression and dropped category metadata, while clipping could hide infinite predictions.
**Instead:** Resolve inherited configs in fixtures, require row accounting, carry category/bounds metadata and check finiteness before clipping; preserve old monitoring CSVs under their previous protocol.

### 13-10-2026 — CPU tests did not cover TabPFN's CUDA loss/scaler path

**Tried:** Exp0 job 11681071 after repairing packages and processed caches.
**Result:** Both TabPFN paths failed; 95 other checks and both trainer comparisons passed.
**Why:** CPU-prepared regression buffers were not moved to the GPU; strict clipping blocked AMP overflow recovery (the log cannot establish that overflow was the only numerical cause). CPU validation cannot establish B200 readiness.
**Instead:** Device-local loss buffers, bounded same-batch scale backoff, targeted CPU/CUDA regressions and one complete VSC validation receipt. No production rerun until it passes.


### 12-10-2026 — Checking caches before the CPU job finished
- **Tried:** full preflight at 18:01, immediately after submitting preprocessing job 62297440.
- **Result:** 13 current PD caches, one stale PD cache and seven stale LGD caches; no GPU job submitted.
- **Why:** the cache rebuild had not finished when it was inspected; the later scheduler record reports successful completion in three minutes.
- **Instead:** wait for the existing CPU job, then repeat preflight. Do not start a duplicate preprocessing job because an intermediate snapshot is stale.

### 12-10-2026 — A completed summary could hide changed training code
- **Tried:** mark an arm complete with the requested step count but an older training fingerprint.
- **Result:** sweep status incorrectly accepted it as done; regression reproduced before the fix.
- **Why:** completion checked the saved configuration and steps but not the recorded implementation.
- **Instead:** reject differing recorded fingerprints in sweep status and require current provenance for benchmark receipts; preserve legacy outputs for historical inspection. Final focused suite: 98 passed, two CatBoost skips tested separately.

### 12-10-2026 — Git pull did not update packages or generated data
- **Tried:** Exp0 job 11680453 after pulling the revised code.
- **Result:** 35 FAIL items, exit 1 after 17m27s; PD/LGD trainer equivalence both passed.
- **Why:** TabPFN 8.2.0 cannot accept `TRANSFORM_TEXT`/`TRANSFORM_DATES`; v3 caches and the old OOD location outlived the source update.
- **Instead:** install the pinned project requirements, rebuild only stale caches on a CPU node, reuse the legacy OOD path, and pass read-only preflight before requesting a GPU.

### 12-10-2026 — Tiny regression tests had depended on rejected prior tables
- **Tried:** reject exhausted filtering budgets, then run the existing tiny trainer, pool and stream fixtures.
- **Result:** the first complete 9.1.0 suite had 16 failures and 24 setup errors, all retry exhaustion in 64–128-row fixtures with 3–6 retries.
- **Why:** mechanics tests relied on the old last-rejected-table fallback, which silently changed the advertised prior.
- **Instead:** disable filtering in these tiny fixtures/CPU equivalence mode; keep dedicated upstream filter comparisons. All 256 production-size draws passed; control needed at most four attempts, no fallback.

### 12-10-2026 — Regressor target transforms changed upstream
- **Tried:** validation against the previously pinned TabPFN 9.0.0.
- **Result:** source audit found upstream's 9.1.0 transformed-target loss correction; interrupted the first full suite rather than treating it as final validation.
- **Why:** the ensemble's target transform must also be applied to each member's loss targets.
- **Instead:** use the official 9.1.0 helper, pin 9.1.0, and update only this project's package with Andreas's permission; real PD/LGD save/resume/evaluation tests pass.


### 12-10-2026 — TabPFN silently ignored the optional matched-context cap
- **Tried:** run the benchmark's capped path with 5,000 available rows and a 200-row cap for each TFM, observing the inputs passed to stubbed external estimators.
- **Result:** TabICL received 200 rows, TabPFN all 5,000; the requested cap appeared in the run metadata anyway. Full-context comparisons were unaffected.
- **Why:** only TabICL defined `max_context_rows` and applied the sampler; `_fit_predict` skipped setting the cap on TabPFN. TabICL's fit report also counted rows before its inner cap.
- **Instead:** cap in the shared TFM fit path before counting fitted rows; test PD/LGD, capped/full modes and identical sampled rows. This is an evaluation correction, not a training restart.

### 12-10-2026 — Exp0 could report success after a trainer equivalence failure
- **Tried:** use the Exp0 job's exit status as the readiness gate for the revised experiments.
- **Result:** final launcher review found that a PD or LGD equivalence failure could still end in `END status=OK`.
- **Why:** both equivalence exit codes were printed but discarded; the launcher returned only the environment-check status.
- **Instead:** return the first failed check's status while still checking both trainers; Bash execution tests cover each failure and the all-success case.

### 12-10-2026 — Concurrent notebook validation exhausted available memory
- **Tried:** the full pytest suite and `run_notebooks --workers 2` while other project notebook runners were active.
- **Result:** prior generation slowed for several minutes; only about 1.7 GiB of 64 GiB physical memory remained free.
- **Why:** each notebook starts its own draw pool (up to eight Python workers); imports and retained synthetic tables multiplied memory use. Stopping only this task's notebook process tree released about 26 GiB.
- **Instead:** run one notebook at a time with `CREDITICL_DRAW_WORKERS=2` and BLAS/OpenMP thread counts set to 1. Leave other projects' processes alone.

### 29-09-2026 — A finished notebook run looked hung for an hour
- **Tried:** `python -m src.utils.run_notebooks | tail`; each kernel stopped with `shutdown_kernel(now=True)`.
- **Result:** all 11 notebooks done in 4 minutes, but the job never ended; 32 orphaned draw workers were still alive after it was killed.
- **Why:** on Windows, killing a kernel does not kill the spawned workers of its `draw` pool; they kept the output pipe open.
- **Instead:** `draw.shutdown_pool()` at exit and before each kernel stops, workers exit with `mp.parent_process()`; send long runs' output to a file, not a pipe. A full run takes ~5 minutes — if it runs much longer, look for orphans.

### 29-09-2026 — The prior-probability shift emptied the query of a low-default book
- **Tried:** `prior_prob_range: [0.15, 0.45]` as the context's share of above-median rows ("0.5 = no shift"), unchanged since 24-09, when 0.2 already showed PD queries with < 1 % defaults.
- **Result:** for a binary target that share is the default rate itself, so a 7 % book put nearly every default in the context: 85 % of PD prior-probability tables kept ≤ 2 query defaults (LGD 19 %, at high train fractions).
- **Why:** "0.5 = no shift" holds only when the high group is half the table; the literature base rates (median 7 %) made the exception the rule.
- **Instead:** a rate ratio, query over context, log-uniform 1.25–4 either way, solved so every row is kept (`prior_prob_ratio_range`); ≤ 2 query defaults in 5 % (the ~1 % books), LGD 0 %. A figure that shows a defect is a bug report: fix it, or log it as open.

### 29-09-2026 — The credit prior was calibrated to the datasets it is scored on
- **Tried:** set every credit-prior range so the synthetic tables matched the 14 PD and 7 LGD datasets (`src/prior/realism.py`), holdout included.
- **Result:** a prior that looks like the test data by construction; any win would be unattributable to domain knowledge.
- **Why:** calibration is fitting, and the fitted-to set included the holdout — test leakage.
- **Instead:** every number from the credit-risk literature or deliberately wide (`docs/PRIORS.md` §5); `realism.py` stays as a descriptive comparison only.

### 29-09-2026 — A list-valued knob that is not a `_range` becomes a sweep axis
- **Tried:** `alpha0: [-0.54, 1.0]` for a normal's [mean, sd] in the ZOIB block.
- **Result:** the grid silently multiplied (each pair read as a two-point sweep).
- **Why:** the config treats any list under a non-literal key as a sweep (`is_literal_list`).
- **Instead:** `*_mean_sd` keys, literal like `*_range`; `--list` shows the arm count before anything runs.

### 29-09-2026 — A recorded period factor widened the table and a real column was zeroed
- **Tried:** append the period factor as a column after the target, on top of the slot's width.
- **Result:** `delete_constant_columns` scans only `plan.num_features` columns, so the last real column was zeroed and the rest misaligned; found by the train/predict round-trip test.
- **Why:** the width budget is fixed per slot (upstream's group structure).
- **Instead:** a target-added column takes a noise column's place (`_credit_candidate`).

### 29-09-2026 — The class repair ran after the context encoding
- **Tried:** leave upstream's row-permutation repair (context and query must hold the same classes) in `_valid`, after `prediction_view`.
- **Result:** with 1 % default rates the repair is common, and the repaired table's encoding was fitted on rows that were no longer the context.
- **Why:** anything fitted on the context must come after the row order is final.
- **Instead:** the repair runs in `_credit_candidate` before the gaps are filled and the table encoded.

### 29-09-2026 — Telemetry was rewritten from memory on every record
- **Tried:** `Telemetry.record` rewrote the CSV from its in-memory rows (for a union of columns).
- **Result:** a requeued arm's new process lost every earlier segment's rows.
- **Why:** the rows were never read back at start.
- **Instead:** read the existing CSV at start and trim to the resumed step (`_read_rows`, `resume`); `weights.csv` the same.

### 29-09-2026 — TabPFN's regressor fine-tuner does not train on the bar-distribution NLL
- **Tried:** assume continued pretraining of TabPFN-3's regressor uses its pretraining loss.
- **Result:** `FinetunedTabPFNRegressor` defaults to CRPS + MSE (`ce_loss_weight 0`); the loss function's own defaults (NLL 1) differ from the class's.
- **Why:** undocumented choice in tabpfn 9.0.0.
- **Instead:** `TabPFNTrainer` calls TabPFN's `_compute_regression_loss` with the fine-tuner's defaults (`train.regression_loss_weights`), and sets `n_classes_` per task for classification (the fine-tuner assumes one dataset).

Anything that cost more than a couple of minutes and did not work — including what you eventually
fixed, because the fix is one changelog line and the dead end was the hour.

### 28-09-2026 — Every credit PD table carried a cohort shift between context and query
- **Tried:** `shift_prob: 0.3` in `credit.shift`, read as "30 % of credit tables have a context/query shift".
- **Result:** the PD mechanism stacks vintages in contiguous row blocks (`sample_cohort_factor`) and nothing reshuffled the rows of the other 70 %, so every table with more than one vintage had its context from the early vintages and its query from the late ones.
- **Why:** the cohort shift (`shift._cohort_shift`) relies on that order on purpose; the unshifted path never undid it. The model cannot see row order, but it does see which rows are context.
- **Instead:** fixed — unshifted credit tables are randomly permuted (`TaskGenerator._credit_candidate`), as upstream's i.i.d. rows and a CV fold are; `test_without_a_shift_the_context_query_split_is_random_like_a_cv_fold`.

### 28-09-2026 — Credit tables were standardised on the whole table; prediction standardises on the context
- **Tried:** credit features through `process_features` (clip + standardise over all rows, as upstream's training does), checked against upstream's `PreprocessingPipeline` fitted on the context.
- **Result:** shifted tables came out up to 4 SD away from what prediction would show the model (`tests/test_train_predict_consistency.py`, first version).
- **Why:** whole-table statistics describe the context by numbers that include the shifted query rows — no real table can do that. Unshifted tables differ by sampling noise only, which is why upstream's own prior gets away with it.
- **Instead:** credit tables end in `prediction_view` (upstream's `UniqueFeatureFilter` + `PreprocessingPipeline("none")`, fitted on the context); the control keeps upstream's training encoding. Re-applying the pipeline is not a test: its soft clip is not idempotent.

### 28-09-2026 — Target-coupled missingness alone made credit PD tasks far too easy
- **Tried:** `missing_target_coupling: 1.0` in every table (Exp1 PD).
- **Result:** credit PD pseudo-R² far above real data; setting coupling to 0 alone brought most of the gap back.
- **Why:** with logit coupling 1.0 the missingness pattern of a few columns nearly encodes the label; real tables range from none to strong (hmeq's DEBTINC-missing alone: AUC 0.78).
- **Instead:** `missing_target_coupling_range: [0.0, 0.6]` drawn per table, gaps in half the tables (`missing_prob`), calibrated with `src/prior/realism.py`.

### 28-09-2026 — `interior_shape_range` was a dead key, and list-valued knobs became sweep axes
- **Tried:** (1) `interior_shape_range` in the LGD configs; (2) new per-table knobs written as lists (`flip_pos_to_neg: [0.05, 0.15]`, `boundary_mass_range_0: [...]`).
- **Result:** (1) nothing read it — the LGD interior used the default [0.3, 4.0]; (2) the config expander turned each list into a sweep axis: 51, then 27 runs instead of 9.
- **Why:** (1) `sample_lgd_shape` read `shape_ab_range`; (2) `expand_grid` treats any list-valued key not ending in `_range` as a lever.
- **Instead:** per-table ranges end in `_range` (`flip_pos_to_neg_range`, `boundary_mass_0_range`, ...), code reads both forms; `interior_a_range`/`interior_b_range` added, falling back to the old keys. Check `--list` after any config edit.

### 28-09-2026 — `axa`'s boundary atoms sit at 1e-5 and 1 − 1e-5, so exact 0/1 tests missed them
- **Tried:** LGD boundary mass as `y <= 0` / `y >= 1` (realism statistics) and `tol=1e-6` (`metrics.boundary_mass_error`).
- **Result:** `0003.axa` read as 0 % at both ends; it holds 28.8 % complete recoveries and 5.6 % total losses. The calibration targets and the benchmark's boundary-mass error for axa were wrong.
- **Why:** the source squeezes `lgd_time` into [1e-5, 1 − 1e-5], a beta-regression convenience; the recipe passes it through.
- **Instead:** `metrics.BOUNDARY_TOL = 1e-4` for every boundary test. With axa read correctly all 7 LGD tables have losses of exactly 0, so `atom_prob_0` is 1.0.

### 28-09-2026 — Upstream's graph sampler returns NaN graphs; the credit path kept them as zeros
- **Tried:** `upstream.sample_base_latent` for the credit tables, followed by `nan_to_num` in `_candidate`.
- **Result:** about 1 credit draw in 60 had NaN features and latent; it was zeroed and kept, rows of zeros with made-up labels.
- **Why:** `GraphSCM.__call__` marks such datasets invalid (y = -100) and they are redrawn; our credit path bypasses `GraphSCM.__call__` and never checked.
- **Instead:** a non-finite graph is an invalid candidate and the slot redraws it.

### 25-09-2026 — LGD 100 % credit arms collapse in the benchmark; the target scale, most likely
- **Tried:** Exp1 LGD with the credit target left raw in [0, 1] (`target_scaling` defaults to `none`; not set in `Exp1_LGD.yaml` or `Exp2_LGD.yaml`).
- **Result:** benchmark holdout R² −0.13 for every 100 % credit arm (0.46 at 0 %, 0.45 at 50 %), calibration slope 4.1. The training monitor ranks the other way round: 50 % credit ~0.19 on `base_model`, control ~−0.03.
- **Why (likely, untested):** TabICL's prior standardises regression targets and `TabICLRegressor.fit` standardises y (`y_scaler_`), so the benchmark feeds standardised targets — which a 100 % credit arm never saw; `progress._score` feeds raw [0, 1] targets — which a control arm never saw. Only 50 % arms saw both.
- **Instead:** read LGD credit-vs-control only at 0 % vs 50 % in the benchmark. Before Exp2 LGD: set `prior.credit.target.target_scaling: standard`, or score one 100 % checkpoint without y standardisation to confirm first (Andreas's call).
- **Since 28-09-2026:** `target_scaling: standard` (context statistics, as `y_scaler_`) in every LGD config, and the monitor scores through upstream's wrapper, so both see the benchmark's scale.

### 28-09-2026 — Three more differences from upstream's trainer, found only by running both
- **Tried:** reading our `Trainer.train_step` against upstream's `Trainer.run_batch` and calling them equivalent (a code comment even said the class slice was "MEASURED" to be a no-op).
- **Result:** `scripts/check_equivalence.py` (same batches, same weights, upstream's own optimizer and scheduler) showed: (1) PD loss differed at step 0 — we took cross-entropy over the classes present, upstream over all 10 logits the training-mode head returns; (2) the RoPE frequencies `row_interactor.tf_row.rope.freqs`, built with `requires_grad=False`, were trained, because `apply_freezing` first set everything trainable; (3) warm-up was rounded down to whole steps. Exp1 ran with (1) and (2).
- **Why:** the "no-op" was measured in eval mode, where the head does slice; the loss convention was copied from upstream's FINE-TUNING module; and "set all, then freeze some" never looked at what the architecture had fixed.
- **Instead:** fixed; the two trainers now agree bit for bit on CPU (`tests/test_equivalence.py`). Compare implementations by running them, never by reading them.

### 28-09-2026 — "Muon, as TabICLv2" was not TabICLv2's Muon
- **Tried:** `optimizer: muon`, `muon_lr: 8e-4` in Exp1, read as upstream's stage-1 optimiser.
- **Result:** the cluster (torch 2.11) took `torch.optim.Muon` with its defaults: Keller's `sqrt(max(1, A/B))` step scaling instead of upstream's Moonlight `0.2·sqrt(max(A, B))` — weight-matrix steps 4.8× smaller (parameter-weighted), momentum 0.95 instead of 0.9 — and biases, norms and embeddings went to AdamW at 3e-4, where upstream runs every parameter through Muon at 8e-4 (`Trainer.configure_optimizer`: one `use_muon=True` group, `matched_adamw_rms=0.2`, `momentum=beta1`).
- **Why:** `src/train/optim.py::_resolve_muon` prefers torch's class and `build_optimizer` passes only lr and weight decay; its docstring assumed torch 2.8 and the vendored class. The vendored fallback would have been worse: the matrices are passed as a plain list, no group carries `use_muon`, so its `step()` sends them to its internal AdamW branch.
- **Instead:** for a re-run, build upstream's optimiser exactly — the vendored `Muon` with `param_groups=[dict(params=all_params, use_muon=True)]`, `lr=8e-4`, `momentum=0.9`, `matched_adamw_rms=0.2` — or run upstream's own `tabicl.train` Trainer, and test that a weight matrix's update is orthogonalised.

### 28-09-2026 — Exp1's PD credit prior ran without its label noise and its underwriting selection
- **Tried:** reading `config/Exp1_PD.yaml`'s `credit.target` block (`flip_pos_to_neg: 0.10`, `flip_neg_to_pos: 0.01`, `selection: {selection_drop: 0.2, ...}`, `base_rate_range`) as part of the prior Exp1 trained on, as `docs/PRIORS.md` describes it.
- **Result:** 48 of 48 generated credit tasks carry no flip or selection metadata. Their median pseudo-R² is 0.57, against 0.06 on real PD data and 0.20 for TabICL's prior.
- **Why:** `apply_pd_target` returns straight after `apply_pd_mechanism` when `mode: mechanism`; steps 2–5 (selection, signal dilution, base rate, label noise) belong to `mode: quantile` only. Notebook 0.2 says so; the config and PRIORS.md do not.
- **Instead:** for a re-run, apply label noise and selection in mechanism mode too (or drop them from the config), and check the realised pseudo-R² of the credit tasks against real data before training.

### 25-09-2026 — "The context cap applies to every model" — it applies to two
- **Tried:** reading the Exp1 benchmark's reference gap and PD Brier scores as matched-context comparisons, as `benchmark.slurm` states ("applies to EVERY model in this array").
- **Result:** our arms and the released TabICLv2 predict a mean default rate of 0.41 against a base rate of 0.22 (TabPFN-3 0.21, CatBoost 0.22); Brier 0.19–0.22 against 0.12–0.14.
- **Why:** `--max-context-rows 1024` is read only by `TabICLBaseline._cap_context`, which for PD samples 50/50; TabPFN-3 gets `_TFMBaseline._maybe_subsample` (10,000 rows, proportional), CatBoost/linear every row. The results' `context_cap` column says 1024 for every model; `info_context_cap` is set only where it was applied.
- **Instead:** compare ours with the released TabICLv2 (matched); read the other references as having more data, and PD Brier/calibration of TabICL-architecture models as shifted by the 50 % context. Whether to cap TabPFN-3 too, or sample the cap in proportion, is Andreas's call (changes the benchmark).

### 25-09-2026 — `run_notebooks --only` erased the other notebooks' summaries
- **Tried:** re-running just the notebooks the new data touched (`--only 1.1_pd_training ...`).
- **Result:** `All_Results.md` and `CAPTIONS.md` came back holding only those notebooks; `--summaries-only` wrote "(no output captured)" for all eleven.
- **Why:** both documents were rebuilt from the executed notebooks alone, and each run deletes the captured text it read.
- **Instead:** fixed — a run always writes every notebook, keeping the previous block of those it did not execute (`run_notebooks.previous_blocks`).

### 24-09-2026 — A fixed seed does not reproduce the PD prior across processes
- **Tried:** quoting 0.2's PD prior numbers from one run and checking them against the next run with the same seed.
- **Result:** they moved (original prior median base rate 50.5 % → 50.0 %, share below 10 % 4.2 % → 3.0 %, predictability median 0.06 → 0.34); within one process the draws repeat exactly, across fresh processes two different sequences appear, and `PYTHONHASHSEED=0` selects one of them.
- **Why:** the installed upstream `tabicl/prior/graph_lib/_dataset.py` builds `feature_groups = list(set(...))` — a set of strings, ordered by the per-process hash seed. The column order of our own tables changes too, for a different reason: `TaskGenerator` shuffles them with torch's global generator, which is unseeded outside training.
- **Instead:** quote PD prior numbers only to the precision that survives a rerun (the notebooks now do). For reproducible training and figures, set `PYTHONHASHSEED` in the job scripts and the notebook runner, or sort the groups — not done: it changes what the cluster runs.

### 24-09-2026 — Pooled context/query histograms showed no distribution shift at all
- **Tried:** drawing each shift kind as ONE histogram of every task's context rows against one of every task's query rows (40 tasks pooled).
- **Result:** context and query coincided for cohort and prior-probability shift — the figure "showed" no shift — and its caption claimed the opposite.
- **Why:** a prior-probability shift goes up or down at random per task, so pooling cancels it; the cohort shift really is ~absent (PD mild ρ; LGD `quantile` mode has no vintages).
- **Instead:** one point per task, context mean against query mean, with the same prior's no-shift tasks as the noise yardstick (`mechanism_plots.shift_kinds`). It also exposed that PD's prior-probability shift leaves <1 % defaults in 82 % of queries.

### 24-09-2026 — Clipping a standardised target to [0, 1] invented boundary atoms
- **Tried:** comparing the original prior's pooled target with real LGD targets after `np.clip(y, 0, 1)`.
- **Result:** every negative standardised value became an exact 0, so the original prior looked closest to `axa` (the most mass at 0) and 0.1 further from real data than it is.
- **Why:** the original prior's regression target is standard-scaled (TabICLv2), not on [0, 1]; a clip is not a rescale.
- **Instead:** min-max scale a target that is not already on [0, 1] (`exp1_plots._unit_target`); the means become 0.34 (credit) against 0.36 (original).

### 24-09-2026 — A common step span truncated every PD curve to step 4,001
- **Tried:** averaging finished arms only over the span every arm logged (latest first log to earliest last log), to stop end-of-curve jumps.
- **Result:** all PD aggregates lost their first 4,000 steps.
- **Why:** three cf = 0.5 `banded` PD arms' progress logs were restarted under the development-only protocol and begin at steps 3,251-4,001.
- **Instead:** a union grid, and each GROUP's statistic drawn only where all of that group's arms have a value (`training_plots._complete`) — a group starts late or stops early instead of jumping.

### 24-09-2026 — Averaging Exp1 monitoring curves mixed holdout data into "which prior is best"
- **Tried:** averaging each arm's `real__*` progress columns into one training curve — first over every column (skipna off), then over the datasets valid in every arm.
- **Result:** the first silently dropped 30/45 LGD arms (`base_model` all-NaN); the second averaged only `axa`/`loss2`/`base_modelisation` for LGD — all three HOLDOUT datasets — and printed a +0.13 R² "credit wins" computed on the test set.
- **Why:** arms finished before the 23-09 development-only protocol monitored "the smallest few" datasets regardless of role (36 PD arms: `hmeq`, `thomas` are holdout); they have no `progress_protocol` column.
- **Instead:** average only the config's `eval.dev_datasets` that every arm carries (PD `german`, `myhom`; LGD `base_model`, 15 arms); show holdout curves labelled, never averaged; check `training_plots.monitoring_coverage` before reading any monitoring number.

### 23-09-2026 — Plot fixtures hid the PD metric-name mismatch
- **Tried:** rendering the repaired PD plots against existing synthetic plot fixtures.
- **Result:** three focused plot tests failed; fixtures used `auc`, while production CSVs contain `roc_auc`.
- **Why:** the old tests duplicated the wrong schema and therefore missed empty production charts; the ranking fixture also lacked complete development/seed coverage.
- **Instead:** test production column names and full development coverage; incomplete configurations must not produce a selection ranking. The fallback, monitor, launcher and benchmark fixes are now local code changes.

### 23-09-2026 — Checkpoint loader metadata still needs the task class count
- **Tried:** removing the prior class count while correcting the diagnostic loader's two-versus-ten architectural head mismatch.
- **Result:** all four checkpoint round-trip tests raised `NameError` while constructing returned metadata; the weights themselves loaded correctly.
- **Why:** `n_classes` also describes the task in the metadata, independently of architectural head width.
- **Instead:** retain it for metadata only; use the production model builder's head width. Round-trip tests now exercise the actual trainer builder, and all four pass.

### 23-09-2026 — PD retries still abort after the OOD-label sanitizer
- **Tried:** cluster retries of the nine base-containing PD banded arms, latest array 11591462 on 22-09, after the earlier OOD-label fix.
- **Result:** all nine failed again with exit 134; successful progress evaluations preceded training-time assertions. The older OOD attribution is insufficient.
- **Why:** confirmed CPU path: upstream `GraphSCM` invalid output uses `y=-100`; after 40 rejections `TaskGenerator.sample` returns that invalid last candidate, and the upstream encoder rejects negative classes. Actual failed batches were not captured.
- **Instead:** fix synthetic validity before fallback, preserve diagnostics, and validate one resumed PD arm before another array. Do not clamp invalid labels to zero. Also resolve LGD monitor NaNs and the Exp2/Exp3 scripts' hard-coded Exp1 config; details in RUNS.md. Source fixes are not part of this audit.

### 08-09-2026 - A missing OOD label (`cat.codes` → -1) crashed 15 PD banded arms via a GPU one_hot assert
- **Tried:** reading the re-run Exp1 phase-1 sweep (job 11549669) to find why 15 arms never finished.
- **Result:** every one was PD `filter=banded`; each died `exit 134`,
  `torch.AcceleratorError: CUDA error: device-side assert triggered` in `ScatterGatherKernel.cu`
  (`idx_dim >= 0 && idx_dim < index_size`), surfacing async inside the column encoder. A hard abort,
  so no summary and no self-resubmit — the arms sat dead, unlike the graceful walltime (exit 64) path.
- **Why:** the ONLY scatter in the model is `F.one_hot(y_train.long(), max_classes)`, and a `>=10`
  label diverts to mixed-radix, so the trigger is a **negative** label. Not the synthetic data (PD is
  strictly {0,1}, verified in code) — the **progress-eval**, which feeds real+OOD context labels to the
  model. `src/eval/ood.py` coded a missing classification target with `pandas.cat.codes`, which returns
  **-1** for NaN; -1 is *finite*, so the eval's `isfinite` filter waved it into `one_hot`. Prior-independent
  (control arms crashed too); PD-only (one_hot is classification; LGD's y-encoder is linear); banded-only
  because those arms run 15-25 h and so trigger far more progress-evals, **not** because of banded data.
- **Instead:** drop unlabelled rows before coding (fetch) and strip any surviving -1 (`load_ood_dataset`,
  so the existing cache is safe without a re-fetch). **An async CUDA assert cannot be localised by its
  Python stack; it took eliminating every other scatter and proving PD labels are {0,1} to reach the eval
  path. A finite sentinel defeats an `isfinite` guard — the two must agree on what "invalid" means.**

### 29-08-2026 - A drained wICE arm could not resume: the self-resubmit asked for b200 resources
- **Tried:** LGD arms 40-74 ran on wICE `gpu_a100`; on a drain the script checkpoints (exit 64)
  and `sbatch`-resubmits itself to resume from the checkpoint.
- **Result:** 2 arms (a40, a73) died at **99.9 %** (steps 12,460 and 12,487 of 12,500). The
  resubmit was REJECTED: *"you can at most request 18 cores and 126000 MiB"*. Both stuck, unresumed.
- **Why:** the exit-64 path re-ran the same script, inheriting its `#SBATCH` gpu_b200 header
  (24 cores, 180G). wICE `gpu_a100` caps at 18 / 126000 MiB — so only the RESUBMIT was rejected;
  the initial launch was fine because `run_experiment` sets per-partition resources on the sbatch line.
- **Instead:** the resubmit now echoes `--cpus-per-task=$SLURM_CPUS_PER_TASK --mem=$SLURM_MEM_PER_NODE`M,
  the grant this task actually holds. Fixed in both `pretrain_{lgd,pd}.slurm`; mindwell unchanged.

### 27-08-2026 - A rare single-class draw crashed 6 PD arms because the guard was on the wrong branch
- **Tried:** the overnight Exp1 run. 8 LGD arms finished clean; 6 PD `mode=mechanism` arms died
  with `ValueError: Vasicek mechanism produced a single-class target`.
- **Result:** `apply_pd_mechanism` raised on a single-class draw, and the raise propagated
  through the DataLoader worker and killed the arm (exit 1, no resume — it is not a signal).
- **Why:** `apply_pd_target` HAS a single-class guard (quantile-cut to a floor base rate), but
  the `mode: mechanism` branch returns EARLY, several lines before it. So the quantile path was
  protected and the mechanism path was not — and only the mechanism path can produce the
  degenerate Vasicek draw. A guard that protects one of two branches is worse than none,
  because the tests on the quantile path pass and hide it.
- **Instead:** the mechanism stops raising (reports `single_class` in meta) and its branch runs
  the same rescue. **When two code paths share a failure mode, the guard belongs before they
  fork, or on both — never on the one you happened to test first.**

### 26-08-2026 - Renaming a script to share it silently deleted the script it replaced
- **Tried:** making phase 2 shared across the three experiments by writing it to
  `scripts/slurm/benchmark.slurm` and removing the old `benchmark_exp1.slurm`.
- **Result:** `benchmark.slurm` was ALREADY the GPU hardware benchmark. The write overwrote it,
  and no test noticed because nothing pinned "benchmark.slurm runs benchmark_gpu.py". So the
  next `submit.sh b200 lgd benchmark.slurm` — habitually the card check — submitted a 76-task
  eval array (job 11528053). 74 of its tasks had no checkpoint and died in ~1 s each: the
  "little jobs that keep coming".
- **Why it hid:** the collision was a filename, not a code path. The phase-2 script was correct;
  the GPU script simply ceased to exist, and the only thing that would have caught it is a test
  that the GPU-benchmark filename still contains `benchmark_gpu.py`.
- **Instead:** distinct names (`gpubench.slurm` vs `benchmark.slurm`), and `pipeline.py` — never
  `submit.sh` — drives phase 2. **When you move a file onto a name, grep that name first.**

### 26-08-2026 - RESOLVED: the released `tabicl` does not contain the prior TabICLv2 was trained on
- **Tried:** importing upstream's `graph_scm` so the control arm stops being a transcription of
  NanoTabICL's transcription. Reported the day before as "not importable, would need a package
  upgrade".
- **Result:** it is importable, but from **none of the obvious places**. `pip install
  tabicl==2.1.1` and `pip install git+...@v2.1.1` both give a package with no `_graph_scm.py`
  and no `graph_lib/` - 29 files missing - while reporting version `2.1.1`. The **default
  branch** has them, at the same version string. Installing all three and listing the trees is
  what settled it.
- **Why it looked impossible:** the version number matched the pinned dump, so the natural
  conclusion was that the dump included files the package legitimately excluded. It does not:
  the release simply predates them.
- **Also required:** `xgboost`. `tabicl.prior._tree_scm` imports it at module scope, so
  `import tabicl.prior` fails without it even though nothing we use touches the tree prior.
  It is in upstream's `[pretrain]` extra.
- **And a second trap once it worked:** `graph_lib` draws from **numpy's** global RNG as well as
  torch's, so seeding `torch.manual_seed` alone left the prior non-reproducible under our own
  seed. Both are now snapshotted, seeded and restored together.
- **Instead:** `pip install "tabicl[pretrain] @ git+https://github.com/soda-inria/tabicl.git"`.
  **A version number is not a content guarantee - install it and list the tree.**

### 26-08-2026 - The control arm's base prior WAS transcribed from NanoTabICL (fixed same day)
- **Asked:** is the `credit_fraction: 0.0` arm running EXACTLY TabICLv2's prior?
- **Found:** `src/prior/base.py` opens with *"Transcribed from **NanoTabICL's** `prior.py`"* -
  a reimplementation, not upstream's `graph_scm`. And the installed `tabicl` package ships
  `_mlp_scm.py` and `_tree_scm.py` but **no `_graph_scm.py` and no `graph_lib/`**, both of
  which exist in the pinned 2.1.1 dump. So upstream's actual stage-1 prior is not importable
  in this environment; it would need a package upgrade plus `xgboost`.
- **One concrete divergence already found:** `max_cat_size` 100 in our control against
  `sample_categorical_sizes(..., max_cat_size=200)` in `_graph_scm.py`. Fixed. Others are
  unaudited.
- **What it does and does not invalidate.** Exp1 compares credit arms against a control that
  uses the SAME base implementation, so the RANKING is internally valid and the sweep is worth
  running. What it weakens is the separate claim that our control reproduces TabICLv2 - and
  therefore any absolute comparison against the released checkpoint, which is already framed
  as "2.5 % of their budget".
- **Instead:** run Exp1 on the ranking question, and before Exp3 - where the absolute number is
  the headline - install the pinned `tabicl` 2.1.1 with `xgboost` and drive
  `tabicl.prior.graph_scm` directly, exactly as the model class and the inference wrapper were
  switched from reimplementations to upstream. **"Transcribed from" is not "is".**

### 25-08-2026 - THE SWEEP WOULD HAVE PRODUCED 75 CHECKPOINTS AND ZERO NUMBERS
- **Tried:** a last read-through of `pretrain_lgd.slurm` before submitting 75 arms.
- **Result:** it trains and stops. **No evaluation anywhere in it.** Every result this project
  has ever produced came from `debug_exp1.slurm`, which trains AND scores; the production
  launcher never did. Submitting would have spent ~10 M credits and left 75 checkpoints with
  nothing to compare and no way to rank a single prior.
- **Why:** the two scripts were written weeks apart for different purposes, and every run since
  has gone through the debug path - so the gap was never exercised. The launcher's own tests
  checked its `#SBATCH` directives and its flags against argparse, never *what it does*.
- **Instead:** the eval block moved into both launchers, on the exit-0 path only (an arm stopped
  by SIGUSR1 has a checkpoint but is not finished, and scoring it would file a row that looks
  like a result). `--models crediticl` only, since `tabiclv2` does not vary by arm.
  **A test that a script is well-formed is not a test that it does the job.**

### 25-08-2026 - LGD's informative missingness was configured, documented, and never applied
- **Tried:** diffing the LGD and PD configs key by key, looking for asymmetries.
- **Result:** `missingness` sat under `credit.target` for PD and under `credit` for LGD. Only
  the PD path read it: `apply_informative_missingness` was called from inside
  `apply_pd_target`. **`apply_lgd_target` is never handed X**, so LGD could not have applied it
  under any config - and the four settings in the LGD files were read by nothing at all.
  `docs/PRIORS.md` lists missingness under **"Both tracks"**.
- **Why:** the mechanism was implemented for PD first and the function lives in `targets/pd.py`,
  so it was natural to call it there. Nothing connected "this is a both-tracks mechanism" to
  "it is invoked from a PD-only function". A config block that is never read raises no error.
- **Instead:** applied in `TaskGenerator._sample_candidate` for both tracks, from a single
  `credit.missingness`, at the same point in the pipeline (after the target, before the noise
  columns) so PD's distribution is unchanged. Two tests: one that it fires on both tracks, one
  that the config has one home. **When two tracks configure the same mechanism at different
  paths, one of them is dead.**

### 24-08-2026 - `--resume auto` was still in a warning string ten days after it killed eight jobs
- **Tried:** reading the first real arm's log, which prints `PROJECTED OVERRUN` every hundred
  steps while the ETA exceeds the walltime.
- **Result:** the warning said "must be resumed (--resume auto)". That flag has never existed;
  on 14-08-2026 it made argparse exit 2 and eight jobs did nothing at all. It was removed from
  the job scripts that day - and left in the WARNING TEXT, where it fired 4 times in 10 minutes.
- **Why:** the fix was applied where the bug bit, not everywhere the string appeared. A grep
  for `--resume auto` at the time would have found it; the test added that day only checked
  the SLURM scripts' flags against `pretrain.py`'s argparse.
- **Instead:** the message now describes what actually happens (checkpoint, exit 64, resubmit,
  resume) and a test asserts the dead flag is absent. **When you delete a flag, grep the whole
  tree for its NAME, not just the place it broke.**

### 24-08-2026 - "We only use 7 % of the GPU" is a memory reading, not a utilisation one
- **Question:** peak GPU memory on a B200 is 13.08 GB of 178. Can we raise the micro-batch,
  pack arms onto one card, or otherwise stop wasting 93 % of it?
- **Measured:** there is nothing to reclaim. The same run reaches **0.791 steps/s against a
  0.986 steps/s ceiling - 80 % of the compute the card can deliver** for this model, and
  `utilization.gpu` read 88.7 % during real training on 17-08. The card is busy. A
  28.5M-parameter model at micro-batch 4 x 1,024 rows is latency-bound, not memory-bound.
- **Why the levers do not exist:** `micro_batch <= group_size = 4` is enforced by
  `validate_micro_batch` (same sequence length AND same train/test split per pass), and raising
  `group_size` changes the prior. Packing several arms per GPU trades ~20 % against CPU
  contention and a lot of complexity, because the compute is already 80 % committed.
- **Instead:** the idle memory means we are paying for a card we do not need. An A100 80 GB
  holds the model six times over at **2.7x less credit all-in**, so it wins if it is less than
  2.7x slower - measurable in one minute with the benchmark. **Idle memory is a pricing
  signal, not a throughput opportunity.**

### 24-08-2026 - "STARVED BY THE PRIOR" was a unit error, and so were the two before it
- **Tried:** reading the benchmark's verdict, which said the run reached **6 %** of its GPU
  ceiling and was starved: "More cores, or a cheaper prior."
- **Result:** it reached **80 %**, and is compute-bound. `amp_bf16_step_ms` held the time of ONE
  forward/backward pass while the verdict divided into it as if it were a whole step. A step is
  16 passes plus one optimiser step: 16 x 62.19 + 18.83 = 1,013.9 ms = 0.986 steps/s, against
  0.791 measured. The prior meanwhile supplies 4.2x what the GPU can take.
- **Why:** the two quantities have the same shape - milliseconds - and differ only by a factor
  nobody carries in their head. Naming both `_step_ms` made them look interchangeable, and the
  same conflation is what made Muon read as 1.40x, then 1.01x, then 1.15x across three runs.
- **Instead:** per-pass keys are `_micro_ms`, whole-step keys are `_step_ms`, the optimiser cost
  is `optimizer_overhead_ms_per_step`, and a test forbids the verdict from touching a per-pass
  number. A ratio above 105 % now says "THE CEILING IS WRONG" instead of a verdict. **When two
  measurements share a unit but not a denominator, put the denominator in the NAME.**

### 24-08-2026 - RETRACTED: the cuDNN crash would NOT have hit the sweep
- **Claimed on 20-08:** cuDNN's fused MHA raised on a B200 under AMP, and "a 75-arm run would
  have discovered this one arm at a time."
- **Measured on 24-08 (job 11523286):** all four backends run fine at [4, 1024, 50], the only
  shape training builds - cuDNN at 24.11 ms bf16. The crash happened at [64, 1024, 50], the
  whole batch in one pass, which **only the broken benchmark ever built**. Training was never
  at risk.
- **Why the wrong conclusion:** the crash and the fix landed in the same session as the
  discovery that the benchmark was not micro-batching. Both were true; the causal link between
  them was assumed, not checked.
- **Instead:** the exclusion stays, on the reason that survives - flash is 4.8 % faster and
  cuDNN has a demonstrated failure at a nearby shape, so dropping it costs nothing. **A bug
  found in an instrument is a bug in the instrument until you reproduce it in the thing the
  instrument measures.**

### 24-08-2026 - The Exp1 launcher had been pointing at the wrong cluster all along
- **Tried:** reading `pretrain_lgd.slurm` before recommending it, while answering a question
  about cluster limits.
- **Result:** its banner said "TARGET: Mindwell - NVIDIA B200" and it printed "Mindwell B200"
  at runtime, but the directives said `--clusters=wice --partition=gpu_a100`, with wICE's
  18-core/100G per-GPU limits and `--array=0-47%12` against a sweep that expands to **75**. It
  would have run on the wrong hardware and silently dropped 27 arms.
- **Why:** every run so far has gone through `submit.sh` (which passes cluster and partition on
  the command line) or through `debug_exp1.slurm`. The real Exp1 launcher had not been executed
  since the project moved to Mindwell, and a comment saying the right thing kept it looking
  correct. **A banner is not a directive.**
- **Instead:** four tests parse the `#SBATCH` lines and check them against the VSC limit table
  and against `expand_with_seeds` - the same function `--list` uses, so the array range cannot
  drift from the grid again. **Test the job script against the thing it claims to match, not
  against a number someone typed twice.**

### 20-08-2026 - Muon costs 1.40x, and also 1.01x. Both measurements were right.
- **Tried:** using the optimiser rung, measured at batch 1, to explain why a B200 trained
  slowly. Recorded three times across 16-08 and 17-08 as "Muon is 1.34-1.42x on both cards".
- **Result:** at the REAL batch (micro 4 accumulated to 64) Muon costs **1.01x**. 1,636 ms
  against AdamW's 1,618.
- **Why:** Newton-Schulz runs once per WEIGHT MATRIX per STEP. Its ~18 ms is FIXED; the
  forward/backward is not. At batch 1 the step is 45 ms and Muon is 40 % of it; at batch 64 the
  step is 1,618 ms and Muon is 1 %. Nothing was mismeasured - the batch size was the variable
  nobody was holding.
- **Instead:** every timing rung now runs at the config's micro-batch. **A per-step fixed cost
  and a per-dataset cost look identical until you vary the batch**, so a ratio measured at one
  batch size says nothing about another.

### 20-08-2026 - PyTorch was choosing our attention kernel, and one of them crashes
- **Tried:** running the real model at the real batch (64) with AMP on a B200, to size Exp1.
- **Result:** `RuntimeError: Expected mha_graph.execute(...).is_good() to be true` -
  `torch/csrc/cudnn/MHA.cpp`, the fused cuDNN multi-head-attention graph. Training sets
  `amp: true`, so **this would have hit the 75-arm run**, one arm at a time.
- **Why:** `scaled_dot_product_attention` picks between four backends silently, per call, from
  the shapes and the dtype. Nothing in this project had ever pinned it, so the kernel was a
  property of whatever batch happened to come along - and the earlier batch-4 runs never
  selected the broken one. Upstream evidently regards the choice as load-bearing:
  `--use_flash_attn3 False` in stage 1, `True` in stages 2-3.
- **Instead:** `src/models/backends.py` excludes cuDNN and pins flash/mem-efficient/math,
  wrapped around every forward pass in training and logged in the run card. Plus an
  `attention_backends` rung that times every kernel in both precisions, **because a failure is
  a measurement**. **A silent per-call choice is a config setting you did not know you had.**

### 20-08-2026 - A benchmark submitted to measure a config change could not read the config
- **Tried:** one B200 job to measure what matching upstream's stage-1 prior shape costs, so the
  75-arm Exp1 budget could be sized from a measurement instead of a FLOP model.
- **Result:** 4 of its 6 rungs used `rows, feats, train_size = 1024, 40, 768`, hardcoded in four
  places, and `--batch-size` still defaulted to 4 from before `train.batch_size` became 64. Only
  `bench_prior` and `bench_end_to_end` called `_prior_cfg`. The verdict then divided a batch-4
  throughput by a batch-1 ceiling and announced "20 % of the GPU, STARVED, more cores" - while
  the prior extrapolation printed two lines above it said the workers could supply 13x what was
  measured. **The report contradicted itself and still sounded certain.**
- **Why:** the hardcoded shape was fine when written, because the config matched it. Nothing
  connected the two, so the config moved and the benchmark did not. Same for the batch default.
- **Instead:** `bench_shape(task)` reads the shape from the config, `--batch-size` defaults to
  `train.batch_size`, the ceiling is measured at the real batch with AMP and Muon, and the
  verdict admits when neither ceiling explains the gap. Two tests pin it. **Before trusting an
  instrument, check what it READS - a stale default answers as confidently as a live one.**

### 20-08-2026 - Two silent prior deviations that also made every arm cheaper
- **Tried:** answering "should Exp1 use upstream's three-stage curriculum, scaled down
  proportionally?" - which meant reading `train_v2_reg_stage{1,2,3}.sh` line by line for the
  first time rather than trusting the summary in the config comment.
- **Result:** two of OUR settings did not match stage 1 at all. `n_rows_range: [512, 1024]`
  against upstream's **exactly 1,024** - stage 1 passes `--max_seq_len 1024` and NO
  `--min_seq_len`, and `sample_seq_len` opens with `if min_seq_len is None: return max_seq_len`.
  And `n_features_range: [3, 50]` against `--min_features 1 --max_features 100`, while
  `max_features: 100` padded the tensors to 100 regardless - so the top HALF of the feature axis
  was never trained on anything.
- **Why:** both deviations made training CHEAPER (~2.5x per step combined), so nothing ever
  complained. A wrong setting that costs you money gets found; one that saves you money does
  not. Two rounds of "make everything the same" had already gone past them, because both times
  the check was against the config's own comment rather than against the script.
- **Instead:** `test_prior_shape_matches_upstream_stage_one` pins all five shape knobs to values
  read off the pinned script, with the flag names in the docstring. **When a claim is "we match
  upstream", the test should quote upstream's file, not our summary of it.**

### 18-08-2026 - Why stages 2-3 are out of reach, and what to do instead
- **Question:** upstream trains in three stages (500k steps at <=1,024 rows, then 40k at 10,240,
  then 10k at 60,000). Should Exp1 do the same, or sweep it?
- **Measured:** no for Exp1 - but the first numbers here were **wrong by ~30x** and are
  corrected below. **CORRECTED 20-08-2026:** a stage-2 step costs ~7x a stage-1 step and a
  stage-3 step ~120x, not 400x and 13,700x. The error was compounding the quadratic row scaling
  with the micro-batch drop 4->1, which changes the number of forward PASSES per step but not
  the datasets per step - total compute per step is unchanged by it. A proportional curriculum
  on our 12,500 steps costs **3.6x**, not 307x: ~9 h/arm becomes ~32 h, which is affordable for
  Exp3 and not for 75 arms.
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
