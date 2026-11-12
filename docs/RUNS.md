# Run record

Dated evidence from VSC: submissions, resolved configurations, outcomes, failures and
interpretation. Entries are newest first, with dates in `DD-MM-YYYY` format. Preserve failed
attempts and superseded recipes as history; their commands and settings are not current advice.

For current operations use [VSC.md](VSC.md); for the registered comparisons use
[EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md). [AGENTS_MEMORY.md](AGENTS_MEMORY.md) indexes
runs and recurring failure modes; [CHANGELOG.md](CHANGELOG.md) records repository changes.

**Contents:** [Recording workflow](#recording-workflow) · [Evidence to retain](#evidence-to-retain)
· [Entry template](#entry-template) · [Dated runs](#runs).

## Recording workflow

1. Read the relevant previous entries before submitting. Verify local tests, the accepted
   source/runtime/input identities, live jobs, resource limits and authorization through the
   VSC runbook. Passing local tests is not cluster acceptance.
2. Submit with a durable admission receipt and immediately record the revision, configuration,
   requested work IDs and accepted job IDs. Reconcile uncertain acknowledgment before retrying.
3. Inspect scheduler accounting, logs and artifacts directly. Preserve successful checkpoints
   and recovery state on PROJECT; a rerun is not a reason to clear them.
4. Collect only filtered analysis files when needed locally, with verified sizes/hashes.
   Weights, optimizer states, pools and runtime images stay on VSC.
5. Complete the outcome and interpretation below. Add a compact record to agent memory and
   **Tried / Result / Why / Instead** for any failure costing more than a few minutes.

## Evidence to retain

Paths below are relative to `output CreditICL/experiment_N/` on the tier selected by the path
resolver. Shared references and submission receipts have their own directories; see the
[output guide](<../output CreditICL/README.md>). Older entries retain their historical paths.

| Evidence | Purpose and interpretation |
|---|---|
| `runs/<run>/logs/` and experiment job logs | Resolved recipe, prior source, runtime/GPU/Slurm identity, progress, failures and final status |
| `runs/<run>/logs/*.metrics.jsonl` | Machine-readable optimization trace; preserve loss, LR, step and finite-gradient/replay diagnostics |
| `runs/<run>/progress.csv` | Small-context credit/OOD monitoring, not the full-table publication benchmark |
| `runs/<run>/telemetry.csv` and `weights.csv` | Sampled utilization, memory, throughput, gradient health and parameter displacement; missing optional telemetry is not proof of inactivity |
| Run configuration and finished summary | Exact arm, seed, updates, timings and completion identity |
| Checkpoints and publication records | Permanent milestones and rolling recovery states on PROJECT |
| Benchmark CSVs and receipts | Current protocol, dataset/model/seed/fold coverage, artifact hashes and row/feature accounting |
| Exp0 reports and attempt logs | Individual stage acceptance bound to source, runtime, inputs and artifacts |

An out-of-memory error, non-finite objective, skipped work or failed publication needs an
explicit explanation; optional telemetry failure must not be confused with a training failure.
A plausible score or Slurm exit code alone does not establish complete scientific output.
Record both credit gains and OOD losses, including adverse or negligible results.

## Entry template

Use the fields relevant to a training, benchmark or validation run; say `n/a` where appropriate.

~~~markdown
### DD-MM-YYYY — <campaign / allocation> — <outcome>

**Submitted:** <time and timezone> | **Jobs/work IDs:** <IDs>
**Source / runtime / library pins:** <identities>
**Allocation:** <cluster, partition, CPU/GPU/memory, requested and used walltime>
**Admission / validation receipts:** <paths>

**Configuration:** <config, resolved arm/seed, initialization, optimizer, LR,
steps, tables/update and geometry; link the complete saved config>

**Outcome:** <terminal accounting, completed artifacts/cells, remaining work;
credit and OOD metrics, throughput and measured resources where relevant>

**Bugs and anomalies:** <failure, impact, diagnosis and recovery; distinguish
expected warnings from missing/invalid work>

**Interpretation:** <what measurements show, hypotheses that explain them,
and limits of the inference>

**Next:** <remaining gates or authorized work; do not imply a submission happened>
~~~

## Runs

### 12-11-2026 — Exp3 completion audit and standardized-target deployment

**Authorization:** inspect/clean scoped stale artifacts, deploy the reviewed repair, run CPU then
GPU Exp0, and admit 16 Exp2 plus 12 Exp3 TabPFN LGD training replacements only after both pass.
No scheduled monitor, replacement benchmarks or additional research grid is authorized here.

**Completed campaign:** live accounting for arrays `11780823`/`11780839` confirms all 36
unaffected tasks completed `0:0`: 24 TabICL and 12 TabPFN PD. Each saved summary has 1,500
updates and the expected table count; all five requested checkpoints exist and are nonempty.
All 612 Exp3 metric records parse, with finite scalar values. The 24 TabICL runs took
31.50–33.47 GPU-hours including resumptions; TabPFN PD took 5.18–5.49 hours. Its 41 AMP retries
across twelve runs recovered without skipped tasks or full-precision replays.
Scanned 1,824 Exp1–3 logs with no traceback, fatal numerical, CUDA or storage-error markers.
Another 354 historical Exp0 logs contain four previously reviewed optional-cache warnings;
the explicit PROJECT weight checks pass. These checks support operational completion, not a
claim that training necessarily improves held-out performance. Exp3 benchmarking is still pending.

**Deployment:** no CreditICL tasks remained in either live queue. Production DATA fast-forwarded
from `0494e7a` to `37a0fba2cdc39e0d22e5f43dca3c95ceb26f7c22`; source fingerprint
`66527dc31b06f437b61e977115ccb792cca6362fedb8aa3b91b1fd3dd1d80ae9`.
Pins are TFM `2c5e65f2de14c7f11940c69306109b6db38b81ce` and VSC
`15e82af30221479f022ff3dbb89687709b961248`. The accepted PROJECT r3 SIF is unchanged
(`83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`).
The existing institution-only `.private-data/site.py` was provisioned outside Git with restricted
permissions and its original hash verified. Login preflight passes packages, processed inputs,
released weights and both OOD caches. This is readiness, not GPU acceptance.

**Cleanup and resources:** old validation evidence is preserved under
`PROJECT/CreditICL/recovery/20261112-before-target-standardize/`. The inventory records SHA256
for all 809 files. The 327 PROJECT files (13,427,087,184 bytes) were renamed on the same filesystem,
with inode/size/mtime preserved. The 482 DATA files were ZIP-archived and every member verified
before the scoped reset removed the originals. ZIP SHA256:
`1294e57409e01830ffa632489ace1940c934eba08ab0de45884e7a76b400a9c0`.
All 7,375 protected production files retained their sizes/mtimes. Both project shared-scratch
trees are empty; account-level scratch use belongs to other projects. Local output has no
checkpoint weights or partial files. Documented runtime rollback assets remain preserved.
Automatic approval review blocked removal of the empty local `memory/` directory; it remains.
At 08:42 CET: PROJECT 265.6G/1T (10,948 files), DATA 47,324M/76,800M (393k files), available
balance 26,378,091 credits. These are shared-allocation snapshots, not campaign billing.

**New validation:** CPU **62663287**, wICE interactive, 8 CPUs/48 GiB/four-hour limit, quote zero.
Receipt `output CreditICL/manifests/submissions/20261112-target-standardize-cpu.json` records
unambiguous acceptance. CPU completion and GPU validation are pending; no replacement training
has been submitted. Deployment/audit evidence is retained locally under
`.runtime-build/lgd-replacement-20261112/`.

### 10-11-2026 — Standardized synthetic targets and scoped retirement

**Scope:** local repair/tests and authorized deletion of old Exp2 TabPFN-3 LGD output.
No new Slurm workload, cancellation, source deployment or runtime change. User approval is
required before sending the new Exp0 jobs. TFM pin `2c5e65f2de14c7f11940c69306109b6db38b81ce`;
VSC-library pin `15e82af30221479f022ff3dbb89687709b961248`.

Reviewed source fingerprint: `66527dc31b06f437b61e977115ccb792cca6362fedb8aa3b91b1fd3dd1d80ae9`.
Training implementation: `9b92584bd26412aa049b632695417a9a16fe58e23e4ff3f025bd707e3c096eb4`.

**Implementation:** keep native context-only target mean/std and its 1e-8 standard-deviation floor.
Remove each regression ensemble member's power transform before materializing its training table;
choosing feature members first preserves their original preprocessing and RNG. Retain CRPS +
mean-MSE, gradient clipping, AMP/replay, affine target scaling, synthetic tables and inference.
New `target-standardize-v1` run identities and resume checks prevent old targets/optimizer states
from entering the replacement campaign. Historical synthetic objective probes explicitly retain
the native target policy. Unchanged Exp1/2/3 completions remain usable only under reviewed exact
recipe, source and package checks. Running jobs continue on their original frozen source.

**Coverage:** local native preprocessing tests check exact context/query z-scores, no query leakage,
unchanged feature members, constant/near-constant context handling, historical controls, PD
isolation, known production tail tables, identities and completion/resume compatibility. Six new
mandatory VSC tests use full sixteen-table batches 511/527/685 at both actual TabPFN peak rates,
checking finite targets/loss/parameters, one optimizer step, zero dropped tables and no FP32 replay.
The existing six-method LGD adapter train/stop/resume/inference stages inherit the new policy.
These GPU tests are configured, not executed evidence.

On the reproduced generic-prior tables (batch/slot 511/1, 527/9, 685/15), local context-only
standardization gives maximum absolute query targets 7.459, 17.944 and 9.169 respectively.
The native preprocessing regression tests verify these values against the explicit context
mean/std calculation. These are synthetic target checks, not model performance measurements.

**Local verification:** full suite 2,391 passed / 64 VSC-or-weight-dependent skips in 22m00s.
The later diagnostic-thread and Exp0 parity checks passed with their affected suites: 76 passed /
two weight-dependent skips. All twelve notebooks executed successfully and exported 43 PDFs;
the affected prior notebook was rerun after the thread guard. Final cells print their summaries.
All 43 retained final plots matched their preceding counterparts by rendered-pixel comparison;
the regenerated prior diagnostics and retained LGD panels were also visually inspected.
After overlapping partial reruns, regenerated the combined summaries verbatim from the final
executed notebooks to remove intermediate fingerprints. No local weights were fetched. This is not GPU
acceptance and does not authorize the replacement campaign.

The first full local invocation had 2,371 passes and eighteen failed assertions: sixteen stale
expectations allowed retired LGD completions, and two negative fixtures accidentally selected
the other registered TabICL rate. Updated those assertions/fixtures and reran the full suite;
no production compatibility gate was relaxed. Both logs remain in the local evidence directory.

**Additional review finding:** both Exp0 TabPFN arm overrides still used weight decay .0001,
while production uses .01. Aligned only the diagnostic overrides to .01 and added a PD/LGD
optimizer/target-policy parity test. Production recipes/results are unchanged; the earlier
diagnostic decay mismatch is separate from the target-transform failure.

**Notebook review finding:** the accepted-stream diagnostic called the production generator
without its loader's single-thread guard. A regression test reproduced the mismatch with a
two-thread caller. The diagnostic now uses the same guard, restores caller settings and verifies
identical records across caller thread counts. Training's generator and loader did not change.
Three regenerated prior diagnostic figures were visually reviewed; they are illustrative samples,
not changes to completed campaign measurements.

**Live evidence, 12:26 CET:** arrays 11780823/11780839 contain twelve completed, eighteen running
and six queued unaffected Exp3 tasks. Twelve TabPFN LGD tasks had already been stopped on 9 November
before training. All twelve completed TabPFN PD summaries record 1,500 steps, 24,000 tables and
zero skips. All 108 available Exp3 log files were read: no traceback, CUDA OOM or non-finite
markers. TabICL requeues retain graceful-stop/recovery records. No new validation gate was claimed.
Production DATA remains `0494e7a14bc320176de3a8665051630461f3c316`; the active Exp3 DATA checkout
remains `a3891f5cf1a4a6c7a4b917dad9906efbfc201e1b`.

**Retirement, 12:44 CET:** removed the old sixteen Exp2 LGD TabPFN runs (indices 8–15/24–31)
and their eighty benchmark slots from active output. Exact VSC inventory: 256 DATA job logs,
128 DATA run records, 80 PROJECT checkpoints, 800 PROJECT benchmark files; 1,264 files and
56,050,535,595 logical bytes. Local inventory: 1,008 analysis files, 49,566,175 bytes, no weights.
Preserved all 2,594 recorded Exp1 sizes/mtimes, unaffected Exp2 results and shared references;
240 Exp2 benchmark receipts remain. All 36 Exp3 initialization paths point to released weights,
with no dependency on retired assets. Manifests retain the removal inventory and outcome without
preserving the invalid results as current evidence. The immediately reported quota had not yet
reflected freed blocks; logical file bytes are not a claim about instantaneous quota recovery.

**Follow-up, 12:52 CET:** nineteen unaffected Exp3 jobs running, five queued, twelve completed;
all 110 logs remained free of fatal/non-finite/storage-error markers. Both frozen commits matched.
PROJECT quota then reported 256.1G/1T (down from 308.3G), consistent with the 52.2 GiB retirement.
DATA reported 47,304M/76,800M. Shared scratch usage includes other projects and was not cleaned.
At 13:18 CET both frozen commits matched. The final 13:37 CET queue/accounting and full
110-log recheck retained 12 completed/19 running/five queued, no nonzero exits among those 36
tasks, and no traceback, CUDA OOM, non-finite or storage-error markers.

**Evidence locations:** DATA `output CreditICL/manifests/retired-tabpfn-lgd-20261110-plan.json`
and `retired-tabpfn-lgd-20261110-done.json`; local `.runtime-build/target-standardize-20261110/`.
Historical synthetic diagnostic reports and old submission/validation receipts remain labelled
historical. Exp1 needs no rerun: its scratch models are TabICL; released TabPFN comparisons did
not train through the affected synthetic target path.

**Interpretation and next gate:** this removes the observed power-map amplification, not every
possible large standardized residual. A tiny context standard deviation can still amplify queries;
no arbitrary target/loss clipping was introduced. Better numerical behavior does not prove better
credit/OOD performance. After approval, validate a separate DATA checkout with new CPU/GPU Exp0
receipts, rerun all sixteen LGD TabPFN Exp2 models from released weights at 1e-5/1e-6, then benchmark
125/250/500/1,000/1,500 updates (80 slots). The twelve stopped LGD TabPFN Exp3 treatments must use
the same corrected policy and compare with its four 50% full-CPT baselines. All other branches
remain unchanged; no rates or treatments are selected using the existing benchmark scores.

**Local housekeeping limit:** automatic approval review rejected deletion of 134 temporary
PNG/PDF review files (11.4 MiB), stating only "blocked by policy". They remain ignored in this
task's `figure-review`/`figures` evidence folders. No retry or alternate deletion was attempted;
the completed local/VSC Exp2 retirement is unaffected, and these review files contain no weights.

### 09-11-2026 — Selective stop and CPT mechanism study

At 14:17 CET, cancelled pending LGD array `11780839` tasks `6-11,18-23`: the twelve
TabPFN-3 retention treatments. Verified owner, job name, pending state and matching
`Exp3_LGD.yaml`/`design.py` hashes before cancellation. The other twelve LGD tasks are
TabICLv2; all 24 PD tasks are unaffected. The stop receipt is retained beside the original
admission receipt in the frozen Exp3 DATA checkout. No completed checkpoint/result was removed.

Reason: native TabPFN regression target transforms and unbounded mean-MSE can produce
enormous finite losses on generic synthetic tasks. Existing global gradient clipping and
FP32 replay prevent numerical failure but do not bound a task's influence on gradient
direction. This does not establish that every observed deterioration has that cause:
the 100% credit arm also deteriorates without the largest transformed-target outliers.
TabICLv2 quantile loss and PD cross-entropy do not share this regression objective.

The isolated `src.validation.cpt_probe` compares native CRPS+MSE, CRPS alone, CRPS+MAE,
and CRPS+Huber (fixed delta 1 in transformed units), using the actual production tail
case and ordinary synthetic tables. Short paired training probes retain released weights,
production table counts/features/splits, and the first 32 updates of the 1,500-step schedule.
TabICLv2 diagnostic rows are fixed at 1,024, so they cannot establish the best rate for
the full long-context workload. Rates 1e-5, 2e-5, 5e-5 and 1e-4 test sensitivity rather
than selecting a benchmark winner. Evaluation uses separate synthetic tasks only.
Production recipes, outputs and the registered grids are unchanged.

**Local validation:** 2,364 passed, 58 skips requiring VSC/model weights, zero failures,
in 26m23s. No released/trained weights were downloaded. The isolated diagnostic module
and tests passed scoped Ruff. No paper folder was read or modified.

**GPU admissions:** diagnostic revision `c47fae81284d8c897713efcb5fac1fe041cdb88b` in
`$VSC_DATA/.crediticl-diagnostics-20261109/CreditICL`, PROJECT r3 image, TFM pin
`8f14e5945ca6b068ed0c76218e769d3e672d0d13`. B200 jobs `11781915-18` were cancelled
before starting at 14:38 CET because estimated starts were 13–14 November. Their original
receipts and `20261109-probes-queue-change.json` preserve the queue decision. Replacement
receipt `20261109-probes-interactive.json` admits tail `11781978` (30m), TabICL PD
`11781979` (2h), TabICL LGD `11781980` (2h), and TabPFN LGD `11781981` (2h).
Each requests one RTX 5000 Ada GPU, eight CPUs and default 16,000 MiB on Mindwell interactive.
The native quote was zero credits; the verified site limits allow four submitted jobs and
one concurrent GPU per user. These smaller-GPU measurements are not B200 capacity/throughput
acceptance. Reports are under PROJECT `experiment_0/diagnostics/cpt-20261109/`;
launcher logs and receipts are in the isolated DATA checkout.

**Tail result:** `11781978` completed 0:0 in 70s, nine tables × four FP32 forward/backward
objectives. At the real seed-0 generic batch 685, slot 15, raw target magnitude 7.664 became
9.338 after context standardization and 2.297e25 after the native fitted target transform.
All values remained finite. The native CRPS+MSE gradient norm was 1.315e24; CRPS alone
0.780, CRPS+MAE 3.029, CRPS+Huber 1.114. Corresponding losses were 8.881e47, 0.229,
3.867e22 and 3.867e22. A huge scalar loss need not imply a huge gradient: MAE/Huber retain
bounded residual derivatives. Huber delta 1 also changes small-error weighting relative
to unit MSE, so it is an alternative objective, not an isolated tail-only intervention.
This confirms an optimization hazard; it does not establish benchmark recovery or explain
all pure-credit deterioration. Native global clipping bounds the aggregate gradient norm,
not each task's contribution to its direction.

**Additional intervention:** one complete update on generic batch 685, with fresh AdamW
state and fixed peak LR 1e-5, tests AMP retry/global clipping and synthetic held-out losses.
It deliberately omits warm-up for that single stress update and is not a replay of the
production optimizer state. No checkpoint or real-benchmark tuning is involved.
Accepted as wICE interactive `62624028`, initially pending for Priority, on separate DATA
checkout `.crediticl-diagnostics-stress-20261109/CreditICL` at `b2173078efdc0207adac1d36df577de8a712d33b`.
Receipt `20261109-stress-wice.json`; one A100 1g.10gb slice, eight CPUs/default 16,000 MiB,
30 minutes, native quote zero credits. The follow-up passed 53 local checks (two VSC-weight
skips) and scoped Ruff. Neither production nor the earlier diagnostic checkout was changed.
It completed 0:0 in 6m57s on `k28i16` (16:05–16:12 CET). All four 16-table updates
were finite. Native CRPS+MSE needed 16 AMP retries and one FP32 replay; the three
alternatives needed neither. However, this did **not** establish a predictive benefit:
generic held-out native loss was 0.57532 before the update, 0.57352 after native,
0.63548 after CRPS, 0.66183 after CRPS+MAE and 0.66477 after CRPS+Huber.
Credit held-out loss was 1.48859 before, 1.48854 after native and 1.52653 after Huber.
The fresh-optimizer/no-warm-up intervention is not a long-run counterfactual and cannot
prove that transformed-target outliers caused the completed benchmark deterioration.
The first stress comparison also mixes native FP32 replay with alternative FP16 updates.
An explicit `--full-precision` stress control removes that confound without changing the
production configuration. Its local checks passed 56 tests with two VSC-weight skips;
diagnostic Ruff passed. The metric module retains one pre-existing import-format finding.
The FP32 control was admitted as wICE interactive `62624819` at 16:40 CET, on
`c8be115d96e3d4bb4e45978919087f64d0b1191e`, with the same 30m/MIG allocation and zero-credit
quote. A preceding admission `62624811` selected the production working directory because
activation was given the original DATA root. Its receipt exposed the mismatch; it was
cancelled while pending, with zero runtime. The corrected admission asserts isolated
container cwd, repository root and commit. Both receipts remain for audit; no production
checkout or running job was changed. This is an operator deployment error, not a model result.
The corrected wICE job remained pending with a 17:32 resource estimate. It was cancelled
before starting and replaced at 16:57 CET by Mindwell interactive `11783522` (FP32) and
`11783523` (matched AMP control), both on `c8be115`, with distinct reports/receipts, eight
CPUs, default 16,000 MiB and a 30m zero-credit quote each. Running both on RTX 5000 Ada
separates the precision intervention from the earlier stress test's A100 hardware.

**Interpretation checks:** the native TabPFN term called CRPS is bin-width-weighted ranked
probability error, with out-of-range targets mapped to the edge bin; it is not an exact
continuous tail integral. A local check gave identical losses/gradients for targets 10 and
1e25 beyond the same support. CRPS-only therefore changes the objective's tail sensitivity;
its smaller loss is not itself evidence of better predictions. The shared evaluation in the
trajectory probes keeps native loss and this distributional component separate.

Final Exp2 dataset-equal LGD R², averaged over the two training seeds, illustrates why neither
"all generic CPT degrades" nor "TabICL needs a higher rate" is supported:

| Model / prior | Higher-rate credit | Lower-rate credit | Higher-rate OOD | Lower-rate OOD |
|---|---:|---:|---:|---:|
| TabICLv2 / 0% credit | 0.34163 | 0.34103 | 0.79388 | 0.79391 |
| TabICLv2 / 100% credit | 0.34033 | 0.34037 | 0.78663 | 0.78929 |
| TabPFN-3 / 0% credit | -0.03773 | 0.34793 | 0.48928 | 0.79542 |
| TabPFN-3 / 100% credit | 0.29466 | 0.31887 | 0.27170 | 0.64310 |

Released credit/OOD baselines are 0.34038/0.79376 for TabICLv2 and 0.34288/0.79664 for
TabPFN-3. Higher/lower rates are respectively 2e-5/1e-5 and 1e-5/1e-6. These are already
observed benchmark results, not a held-out rate-selection exercise. R² is not the whole
outcome: pure-credit TabICL endpoint-mass error worsened by roughly 0.008 at both rates.
The TabPFN `video_transcoding` step-500 failure has finite, mostly positively biased predictions;
its negative R² must not be equated with an overflowing prediction or automatically attributed
to the generic synthetic-tail case. That run uses only credit tasks.

Completed Exp2 summaries record median accumulated training times of 33.31 GPU-hours
(TabICL PD), 32.44 (TabICL LGD), 5.50 (TabPFN PD) and 4.04 (TabPFN LGD), per 1,500-update
run. These are recorded training/monitoring times, not invoices or estimates for a new
objective. They favour a small, matched TabPFN LGD repair study over a blind TabICL rerun.
The LGD metric module's explanation was corrected: the conditional mean remains optimal
under squared error despite endpoint-heavy targets. Its executable AST is unchanged.
As a reviewer sensitivity check, averaging each known overlapping dataset pair into one family
left the sign of every final 100%-minus-0% contrast unchanged across two models, two tasks and
two rates. For TabICL LGD, dataset-equal differences -0.00130/-0.00066 became
family-equal -0.00174/-0.00100. This is a descriptive dependence check, not a new significance claim.

**TabICL LR interventions:** PD `11781979` completed 0:0 in 1h07m16s and LGD `11781980`
in 51m52s. Each compared four rates from identical released weights on the same 32 training
updates, with eight held-out generic and eight held-out credit synthetic tables. Every update
was finite. PD uses cross-entropy and LGD the native quantile loss; lower is better here.

| Peak LR | PD generic loss | PD credit loss | LGD generic loss | LGD credit loss |
|---|---:|---:|---:|---:|
| Before training | 0.468203 | 0.247156 | 0.100424 | 0.241681 |
| 1e-5 | 0.469151 | 0.245862 | 0.100469 | 0.241606 |
| 2e-5 | 0.469175 | 0.245409 | 0.100722 | 0.241348 |
| 5e-5 | 0.470196 | 0.244844 | 0.101116 | 0.240854 |
| 1e-4 | 0.470557 | 0.244899 | 0.101835 | 0.240714 |

Higher rates produced more adaptation and generic loss, with no further PD credit gain
from 5e-5 to 1e-4. These one-seed/short-table observations cannot select a rate for the
long-context production workload or establish real-benchmark improvement.

**TabPFN ordinary-task intervention:** `11781981` completed 0:0 in 26m37s. All four
32-update paths consumed the same 16-table batches and had identical initial held-out
losses. Native evaluation below uses CRPS+MSE for every training objective.

| Training objective | Generic native loss | Credit native loss | Generic CRPS | Credit CRPS |
|---|---:|---:|---:|---:|
| Before training | 0.575322 | 1.488586 | 0.222788 | 0.517531 |
| Native CRPS+MSE | 0.576405 | 1.472465 | 0.223179 | 0.512562 |
| CRPS | 0.576673 | 1.471503 | 0.223189 | 0.512187 |
| CRPS+MAE | 0.572119 | 1.490524 | 0.222309 | 0.517888 |
| CRPS+Huber | 0.576180 | 1.477558 | 0.223303 | 0.513985 |

No alternative dominates both populations. MAE improved generic loss while worsening credit
loss; native and CRPS produced modest credit gains with small generic losses. All updates were
finite. AMP backoffs were respectively 7/6/8/7, with no FP32 replay. These 32 steps are entirely
inside TabPFN's 150-update warm-up; they do not test its peak-rate stability or select a
production objective. The separate peak-rate stress update addresses a different question.

The all-FP32 stress control `11783522` completed 0:0 in 2m08s on RTX 5000 Ada. It confirms
that avoiding retries does not itself recover predictive performance: after the same outlier
batch, generic held-out native loss was 0.57352 for native, versus 0.66601/0.66852/0.67464
for CRPS/CRPS+MAE/CRPS+Huber. Corresponding credit losses were 1.48854 versus
1.52621/1.52736/1.52640. None needed AMP backoff because AMP was disabled.
Every variant started from the same weights/held-outs. This is still one update with fresh
optimizer state, not the long-run effect of intervening at production step 685.

The matched RTX AMP control `11783523` completed 0:0 in 3m39s. Native again needed
16 backoffs/one FP32 replay, and its final generic/credit losses (0.57351942/1.48854266)
matched the always-FP32 path (0.57351982/1.48854274) closely. All three alternative
objectives avoided those retries but had worse held-out losses under either precision.
Thus neither changing precision alone nor replacing MSE with these alternatives is a
demonstrated accuracy fix. Production training losses, metrics and checkpoints remain intact.

**Completed evidence:** seven diagnostic allocations, 396 complete training steps plus
36 single-table forward/backward objective checks; all scheduler exits 0:0 and all reports
PASS. These are bounded synthetic mechanism tests, not seven production/Exp0 acceptance runs.
The sum of recorded allocation elapsed times is 2h39m39s; native interactive quotes were zero.
Collected 23 analysis files (209,830 bytes) through `src.utils.result_manifest`, with every
size/hash verified after SFTP. Local proof: `manifests/cpt-20261109/verified-collection.json`.
No weights, optimizer state, pools or archives were transferred. Nine core trainer/generator
modules match the frozen Exp2 executable AST; both Exp2 configs match outside the private-alias
evaluation registry. This review changed diagnostic code/tests and explanations, not the
production objective, grid or synthetic generator.

**Campaign read at 17:14 CET:** Exp3 has nine completed PD tasks, fourteen running PD tasks,
one pending PD task and twelve pending TabICL LGD tasks; the twelve TabPFN LGD tasks remain
stopped. The frozen `a3891f5` science tree has no tracked changes. Read all 46 existing Exp3
logs (1,486,539 bytes): no traceback, runtime error, non-finite exception, CUDA OOM or ERROR
marker. The 76 AMP backoff messages are finite-update retries, not discarded tasks; completed
TabPFN PD metrics reach 1,500 with no FP32 replay. This is current execution evidence, not
proof of future convergence or benchmark quality. PROJECT usage was 292.7 GiB/1 TiB and
11,675/150,000 files; DATA 47,272/76,800 MiB. Available account balance: 51,315,537 credits.

**Next:** retain all completed Exp2 evidence and the 36 unaffected Exp3 runs. Do not treat a
loss-function substitution or higher TabICL LR as an established repair. A longer, matched
TabPFN LGD follow-up needs an explicit recipe and current Exp0 gates; if its objective changes,
its retention treatments need matching new baselines. The existing lower-rate baseline must
also be shown when judging whether a retention method helps. No new full Exp2 campaign or
Exp3 benchmark was launched by this diagnostic investigation.

### 09-11-2026 — Exp3 training admitted

Mindwell accepted PD array `11780823` and LGD array `11780839`, indices 0–23 each.
Every task requests one B200, 24 CPUs, 180G and a 12-hour allocation. The two arrays
cover all 48 registered new training runs. The six recipes use the fixed 50% prior,
two backbones and two paired seeds, with 1,500 updates and five permanent milestones.
The 16 matching Exp2 baselines are already measured. This admission covers training.

The frozen DATA checkout `$VSC_DATA/.crediticl-validation-20261106/CreditICL` is at
`a3891f5cf1a4a6c7a4b917dad9906efbfc201e1b`. The launcher explicitly restores that
DATA root and invokes its tracked Slurm script, with the same reviewed resource
directives, signal and requeue behavior. The existing production runtime manifest
selects the verified PROJECT r3 SIF. Both Slurm dry runs and the live `--require-exp0`
preflight passed. The 117 focused local tests passed. Local reporting executed all
12 notebooks, verified 52 PDFs and captions, and checked the four new trajectory plots.

The quote was 17,625,600 credits for the first allocations, with 57,946,762 available
at submission. PROJECT had 761.4 GiB and 138,455 files free. DATA had about 30 GiB free.
The receipt `output CreditICL/manifests/submissions/20261109-exp3-all-training.json`
under the revised checkout records both accepted arrays with no uncertain submission.
Initial scheduler reconciliation found all 48 task IDs. Fifteen PD tasks were running
and passed the allocation preflight and smoke test. Nine PD tasks awaited resources
and all 24 LGD tasks awaited priority. The startup logs contained no matched errors.

At 11:56 CET, a read-only check found 21 PD tasks running, three PD tasks pending and
all 24 LGD tasks pending. The active checkout still resolved to `a3891f5`. The later
publication revision applies the alias mapping to local reports and keeps the source-bound
Slurm and benchmark records at their original revisions.

### 09-11-2026 — Exp2 complete and verified; LGD retention outlier investigated

Mindwell arrays `11774787`, `11774788`, `11775176`, `11775177`, and `11775178`
completed all 320 trained checkpoint slots with exit `0:0`. All 320 receipts, 640 result
CSVs and 56,800 protocol-9 credit and OOD cells matched their hashes and expected coverage.
The revised Exp0 on `a3891f5` completed CPU 14/14 (wICE `62581713`) and GPU 125/125
(Mindwell `11774935`, 18h14m17s). The live preflight revalidated the source, environment,
inputs and artifact hashes using the PROJECT r3 image.

**TabPFN-3 LGD, 100% credit, peak LR `1e-5`, seed 0.** On `video_transcoding`, mean
five-fold R² was 0.998 for the released model and 0.994, 0.819, −8.149, 0.629 and 0.626
at updates 125, 250, 500, 1,000 and 1,500. At update 500, the four worst folds had a
positive mean prediction error of 0.223–0.230 on the training-fold-normalized target.
Squared mean error explained 92–93% of their MSE, and rank correlation also fell.
The checkpoint bytes matched the receipt, the target ranges and fold identities matched,
and the reference's evaluation-code differences are covered by reviewed compatible
transitions. The training loss at the 500-update log point was finite (1.555).
The available evidence indicates a checkpoint-specific degradation; the mechanism
behind the weight changes is unresolved. The second seed also deteriorated later on
this table (mean R² −3.009 at the final checkpoint).

At the final checkpoint, averaging folds and both seeds with equal weight per dataset:
TabICLv2 PD with 100% credit gained 0.0025–0.0034 credit AUC over matched generic CPT,
with mean OOD AUC nearly flat. Only four or five of the 14 credit tables improved,
depending on the rate. Vehicle Loan supplied the largest gain at both rates.
No TabPFN-3 PD recipe exceeded the released model's mean
credit AUC. TabICLv2 LGD showed no mean credit improvement from increasing the credit
share. The high-rate TabPFN-3 LGD 100% arm gained 0.332 credit R² against its degraded
generic control and lost 0.048 against released weights. Its mean OOD R² fell by 0.525
against release, or 0.456 with `video_transcoding` excluded as a sensitivity check.
Each of the 25 OOD regression datasets lost R² and had higher RMSE and CRPS.
Retain the original full-suite result as the primary comparison.

Collected 7,714 allowlisted analysis files (1,934,336,202 bytes), across production DATA,
PROJECT and the revised validation DATA checkout. Every member matched its SHA-256.
Local provenance is `output CreditICL/manifests/analysis_all_20261109.json`, with
experiment-specific manifests for notebook rendering. Revised validation and remote
rendered reports are separately preserved. The transfer archives and 405 empty imported
local lock files were removed after verification. Scoped VSC output contained no
partial publications. Model weights and runtime images remain on PROJECT.

### 06-11-2026 — Analysis collection and notebook review

Read-only collection from production `0494e7a` at **14:32:11 UTC**: 589 compact analysis
members, including 64 completed final-checkpoint credit/OOD receipt pairs and their 128 CSVs.
All member sizes and SHA-256 hashes matched the filtered inventory. No model/optimizer weights,
datasets, prior pools or runtime images were downloaded. The 4,097,080-byte transfer archive
had SHA-256 `694c492278eaf62c92e8c3bcc86cb5e0d0cf784652e075f7abd51294d2022ef3`;
its temporary DATA copy was verified and removed. Local provenance is
`output CreditICL/manifests/analysis_exp2_20261106.json`.

The notebook revision changes reporting only: exact scores accompany paired credit effects,
dataset maps and separate credit/OOD comparisons against trained controls and released weights.
This distinction matters particularly for TabPFN-3 LGD: improvement over a degraded continued
control does not establish retention of released-model performance. Synthetic loss tails remain
visible; descriptive dataset intervals do not turn historically exposed benchmarks into untouched
confirmation. Early-checkpoint trajectories and new Exp3 effects remain unavailable in this snapshot.

The established `figures/<notebook>` export layout is retained, as documented in the notebook
README. Research context uses the current implementation and paper notes, not the superseded
ArXiV grids. TFM pin `8f14e5945ca6b068ed0c76218e769d3e672d0d13`; VSC pin
`914ef4e57e1f153080e2686d03179c223db38c9a`. No campaign source or scientific setting changed.

Local acceptance: **2,333 passed, 58 skipped** in 18m06s (Git Bash on the test process PATH).
All 12 notebooks executed with fresh kernels; all 48 final PDFs were individually inspected
and the final rerender was pixel-identical. Inline figures, captions and printed summaries
remain saved. The local cleanup of temporary QA images/archive was blocked by automatic
approval review without a specific reason; that ignored QA workspace is retained.

At **15:42 UTC**, scheduler accounting covered all 320 Exp2 slots: **75 completed, 19 running,
226 pending**, no failed states. All 75 published receipts and 13,505 credit/OOD cells passed
hash, protocol, uniqueness, coverage, status and finite-primary-score checks: 64 final, eight
125-update and three 250-update checkpoints. Explicit admission receipts for both finals and
the three early-checkpoint arrays have no in-flight acknowledgments or missing slots. Revised
GPU `11774935` was running at 3h16m with 57 completed stages PASS; CPU remains PASS 14/14.
All Exp0/Exp2 work is admitted. Exp3 is conditionally authorized but awaits full GPU acceptance;
no Exp3 work or duplicate submission was launched. Production and validation checkouts stay frozen.
The earlier full scan of 85 benchmark-driver logs (148,214,824 bytes) found no fatal matches;
at 15:44 UTC a bounded tail scan of all 265 benchmark/evaluation logs also found none. A wider
full evaluation-log scan hit its read timeout, so that latter check covers tails, not every byte.

### 06-11-2026 — Remaining 256 Exp2 benchmarks admitted

Verified the completed training summaries, checkpoint archive directories and slot-to-model
mapping on `0494e7a`. The matching production Exp0 gate passed. All three scheduler dry runs
passed. The verified submission ceiling was 500 and the expanded queue had 188 tasks before
admission. The total quote was 16,646,400 credits with 68,770,309 available.

| Mindwell job | Earlier-checkpoint slots | Limit | Quote |
|---|---|---|---:|
| `11775176` | PD TabICL, 64 | 4h | 7,833,600 |
| `11775177` | PD TabPFN-3, 64 | 3h | 5,875,200 |
| `11775178` | LGD, both backbones, 128 | 45m | 2,937,600 |

The receipt at `output CreditICL/manifests/submissions/20261106-exp2-earlier-benchmark.json`
records the exact indices and successful admissions. At 10:35 CET the five arrays covered all
320 trained slots without duplicates. Eighteen final PD tasks were running, 46 finals and 256
earlier tasks were pending, and no benchmark completion receipts had been published yet.
The latest running finals were scoring AlgorithmWatch with no matched errors. The eight
verified shared reference receipts remain available for comparison.

Revised Exp0 CPU `62581713` completed 14/14. GPU `11774935` is queued for Priority and covers
the revised 125-stage manifest. The GPU receipt will appear under the isolated validation
checkout when the allocation starts. Keep both checkouts frozen through their consumers.

### 06-11-2026 — Shorter Exp2 benchmark allocations

On Mindwell at 09:57 CET verified shorter limits for all 46 queued final slots, using in-place
`scontrol` updates on the existing tasks. LGD `11774788_0–31` uses 45 minutes, PD TabICL
`11774787_18–23` four hours, and PD TabPFN-3 `11774787_24–31` three hours. The 18 running
PD tasks were outside the update scope. Exp0 uses its separate validation allocation.

Comparable historical benchmark scoring took 135–170 minutes for PD TabICL (75 logs),
118 minutes for the PD TabPFN-3 reference, and about 6–10 minutes for LGD TFMs. The widest PD
dataset, AlgorithmWatch, took up to 25 minutes per fold. Current PD tasks had started their
last credit dataset after about 51 minutes. Full rows, features, folds and OOD remain in scope.

Mindwell has `sched/backfill`. The LGD forecast moved from Sunday evening to Saturday, while
individual PD estimates spanned Friday evening through Saturday morning at readback. Use the
measured bounds as `--time` overrides for later submissions. VSC quoted 7,833,600 for 64 PD
TabICL, 5,875,200 for 64 PD TabPFN-3 and 2,937,600 for 128 LGD slots, 16,646,400 credits total.
Use current receipts and credits to plan admission of those earlier checkpoints.

### 06-11-2026 — Exp2 finals and revised Exp0 started concurrently

All 64 Exp2 runs completed 1,500 updates, including the last 14 PD jobs with Slurm exits 0:0.
Verified the 320 milestone archive directories and scanned 32 new or updated training logs.
Fourteen had successful job footers, with no fatal matches in the scan.

**Benchmark.** Mindwell arrays `11774787` (PD) and `11774788` (LGD) each score slots 0–31,
the final checkpoint of every arm for that track, on `0494e7a`. The live Exp0 gate accepted
all existing CPU/GPU artifacts. Full rows/features, five folds and native OOD remain in scope.
The first-wave quote is 23,500,800 credits at the existing 12-hour limit, versus 117,504,000 for
all 320 trained slots and 75,397,843 available at admission. The 256 earlier checkpoint slots
will need further credit-checked waves. The eight shared reference receipts were retained.
Submission receipts are under `output CreditICL/manifests/submissions/20261106-exp2-final-benchmark/`.

**Validation.** CPU `62581713` on wICE interactive, 8 CPUs/48G/4h, quote 0, runs the revised
`a3891f5` checkout at `$VSC_DATA/.crediticl-validation-20261106/CreditICL`. Its dedicated
receipt namespace keeps the production Exp0 gate usable throughout benchmarking. The batch
wrapper restores the isolated `VSC_DATA` after the VSC login profile, then checks the commit and
executes the tracked script. Runtime remains the verified PROJECT r3 SIF. Candidate source
`732101a15ec20602ac49341aa7c8684467265375baa83379f73f732b93e8e184`, TFM pin
`8f14e5945ca6b068ed0c76218e769d3e672d0d13`, VSC pin `914ef4e57e1f153080e2686d03179c223db38c9a`.
The CPU receipt lives in the candidate's `output CreditICL/experiment_0/report/` directory.

CPU `62581713` completed 0:0 in 34m43s. All 14 stages passed, including 2,320 tests (two
skips, 38 GPU deselections), all 14 released-weight integrations and 12/12 notebooks (38
figures). Attempt `cpu_20261106T075801Z_j62581713_003f9931`.

After the matching CPU gate passed, Mindwell accepted GPU `11774935`, one B200 with 24 CPUs,
180G and a 24-hour limit. Quote 734,400 credits, with 68,741,463 available. The receipt
`manifests/submissions/20261106-revised-exp0/gpu.json` under the candidate output root has
no uncertain admission. Both revised validation phases use that checkout and runtime.

At 09:36 CET 18 PD slots were running, 46 benchmark slots were pending, and the GPU
validation allocation was pending for Priority. Full-context
scoring advanced through its folds with no fatal log markers. The planner independently
verified all eight shared reference receipts. Check results and current credit before the
next benchmark wave. Keep both checkouts and validation artifact paths frozen until their
consumers finish. Exp3 remains gated on revised GPU acceptance.

### 05-11-2026 — Exp3 validation coverage audit; no new Slurm workload

Local review pins: TFM `8f14e5945ca6b068ed0c76218e769d3e672d0d13`;
VSC Library `914ef4e57e1f153080e2686d03179c223db38c9a` (both unchanged).

Direct read-only SSH rechecked deployed `0494e7a`: CPU attempt
`cpu_20261101T215620Z_j62516762_0d50fc3b` and GPU attempt
`gpu_20261102T074311Z_j11763662_5dc41327` remain PASS at source fingerprint
`571c1f6cb2bc968a8bc572fc6fd2b8f202f3bca6024db0a28aadb3268fd36d3b`.
Their stage counts are 14/117; the GPU report contains 16 successful adaptation routes using
the earlier no-decay/L2-SP/DiagScale/LoRA recipes. Current Exp3 requires 24 routes for six
recipes across two tasks/two backbones, within 14 CPU/125 GPU stages and 34 capacity probes.
It has not yet passed that revised VSC gate.

Local review reproduced an unnecessary rerun: adapter provenance was attached to all Exp3
requests, so the eight adapter-free references could not reuse identical Exp2 receipts. The
fix limits that extra source dependency to trained arms. Nine regressions cover all eight
reference request pairs, adapter changes and scientific-input mutations. Exp1/2 training
identities and implementation remain unchanged; no VSC checkout, job or environment changed.
No model weights were transferred. Evidence:
`.runtime-build/exp3-review-20261105-vsc-validation.json` and local review/test logs.
The review also found 14 local tests assuming released weights were downloaded. They now skip
only off VSC; all remain enabled for cluster CPU validation and the mandatory GPU counterparts
remain unchanged. A collection-only simulation verified this without executing a cluster workload.
Full local pytest: 2,322 passed, 58 resource/VSC skips, zero failures in 30m52s; 499 warnings
from upstream deprecations and deliberate test fixtures. Source fingerprint:
`732101a15ec20602ac49341aa7c8684467265375baa83379f73f732b93e8e184`.

Recommended sequence: after active Exp2 training consumers finish, deploy/validate the final
reviewed source once, then benchmark Exp2 and run authorized Exp3 without changing that source.
The benchmark's output artifacts alone do not require a second Exp0. A later source/runtime/
package/input change requires matching validation again; an older benchmark checkout can instead
be allowed to finish before the single Exp3 source cutover. No workload is submitted by this review.

### 05-11-2026 — 10:28–10:31 CET progress and output audit

No remote submissions or mutations. Production remains `0494e7a` and the accepted PROJECT r3 image;
scientific settings are unchanged. Exp0 CPU/GPU reports retain PASS (14/117 stages). Exp1
retains all 54 complete training runs and 158 paired benchmark receipts including shared references.

Exp2: **50/64 complete** (LGD 32/32, PD 18/32), **12 running + 2 Priority-pending**.
All unfinished configurations are PD/TabICL. Running: `11763811_2`, `11763815_3`,
`11763820_4`, `11763822_5`, `11763825_6`, `11763832_7`, `11763881_16`, `11763885_17`,
`11763887_18`, `11763890_19`, `11763895_20`, `11763897_21`. Pending: `11763899_22`,
`11763901_23`. Exact run-name/queue coverage accounts for all 64 without duplicates.
Latest recorded unfinished progress is 1059–1400 of 1500 updates; summaries from previous
checkpoint/requeue cycles can lag the current metrics. These resumptions are expected.

All 250 required milestone files for completed runs are nonempty and have readable Torch ZIP
directories with a data pickle. This is a structural check, not tensor loading or a full checksum
scan of the retained weights. No models were downloaded. Scanned 786 Exp1/2 logs (771,128,693
bytes): zero matched fatal exceptions, CUDA OOM, quota or checkpoint-write errors; no fallback
errors. All 1,003 recorded metric losses are finite, and no malformed metric record was found.
The 32 completed TabPFN runs report zero skipped tasks. Their handled AMP-overflow warnings
correspond to 1,062 retries overall (maximum 104/run); the validated same-batch replay path
recovered them. The evidence supports normal progress; downstream benchmarking is still pending.

Quota: DATA 44488M/76800M; Lustre PROJECT 253.3G/1T, 8145/150000 files. Available balance
76,052,575 credits. No cleanup or runtime/source cutover was performed. Exp2 benchmark receipts
remain zero; its 320 checkpoint evaluations and subsequent analysis remain outstanding.
Exp3 remains implemented locally and unsubmitted. Next: finish training, validate any source
cutover after consumers exit, then quote/authorize Exp2 credit/OOD benchmarking.
Evidence: `.runtime-build/audit-20261105/{status,logs}.json` (verified SSH stdout/stderr/status).

### 04-11-2026 — 17:28 CET training/storage audit; local model cleanup

No jobs submitted, cancelled or changed. VSC production remains frozen at `0494e7a` with
the accepted PROJECT r3 runtime. Exp0 CPU 62516762 and GPU 11763662 retain PASS reports
(14 and 117 stages). Exp1 retains 54 completed runs, 150 checkpoint/component receipts and
eight shared-reference receipts. This audit did not recompute those benchmark results.

Exp2: **47/64 complete at 1,500 updates** (PD16/32, LGD31/32), **17 running**, zero pending.
All 64 expected identities are present, and every completed run has checkpoints at
125/250/500/1000/1500. Remaining PD jobs: `11763811_[0-2]`, `11763815_3`, `11763820_4`,
`11763822_5`, `11763825_6`, `11763832_7`, `11763881_16`, `11763885_17`, `11763887_18`,
`11763890_19`, `11763895_20`, `11763897_21`, `11763899_22`, `11763901_23`.
Remaining LGD: `11763798_23`. Last recorded steps are PD700–1400 and LGD1400;
12-hour checkpoint/requeue cycles remain expected. The recipe/grid is unchanged.

Scanned 758 active-tree Exp1/2 logs (770,229,240 bytes): no matched crash, CUDA OOM, quota
or checkpoint-write failure, no nonfinite metric losses, no error-bearing fallback logs.
Historical scheduler failures/cancellations remain evidence of the earlier repaired campaign,
not new failures. Only the two accepted Exp0 attempt directories remain. No checkpoint file
is in the DATA output tree. This is a status/log audit, not a guarantee against unobserved bugs.

CreditICL PROJECT: 213.0 GiB including 197.8 GiB output and 12.1 GiB runtime/rebuild assets;
project-owned shared scratch ~21 KiB. Allocation usage 250.6G/1T, 8,132/150,000 files;
DATA 44357M/76800M. Available credit balance 31,811,358. No remote cleanup was needed or
performed; successful models and documented legacy environment recovery exceptions remain.

Locally removed 361 model files, 82.15 GiB; all 2,755 retained analysis files passed SHA256
verification. The temporary PowerShell helper's JSON-array and DirectoryInfo issues were
reproduced in PS5.1, fixed and tested before execution. Its executable code is retired.
Automatic review blocked deleting empty directories/helper files; harmless placeholders remain.
Evidence: `.runtime-build/local-cleanup-verified-20261104.json` and
`.runtime-build/audit-20261104/current-{status,health}.sh.result.json`.

Next: finish/verify training, safely update and validate the revised source after consumers exit,
then authorize/quote the outstanding Exp2 benchmarks (320 trained-checkpoint slots; eight shared
references may be reused after provenance checks). Exp3's 48 new runs/16 reused baselines remain
implemented locally and unsubmitted. Collect analysis only for subsequent notebook updates.

### 03-11-2026 — complete Exp0/Exp1 download and analysis snapshot

No new cluster computation was submitted. The PROJECT-image provenance planner on frozen
0494e7a accepts **54/54 Exp1 training runs and 158/158 benchmark slots**: 60 main checkpoint
slots plus four shared references per task, and 15 component slots per task. This includes
the 25% extension, all five folds and the native credit/OOD artifact gates.

The merged DATA/PROJECT scientific output trees `experiment_0/`, `experiment_1/` and
`reference/` are mirrored locally, including checkpoints, predictions, metrics, logs and
receipts: **3,332 files / 41,306,487,936 bytes**, with exact file-set and SHA256 agreement.
Breakdown: Exp0 642 files; Exp1 2,594; shared references 96. All Exp1 files were already
identical; 669 changed/missing Exp0/reference files (6,747,279,935 bytes) were downloaded
through native Windows SFTP. A bounded path guard stopped two initial transfer groups;
the 308 completed staged files were verified before resuming only the remaining 361.
All directories were then prepared/resolved sequentially; no guard was bypassed.

Preserved 547 superseded local Exp0 files and 52 replaced versions under
`.runtime-build/exp01-sync-20261103/local-before-sync/`; the active scientific tree contains
the current remote files only. Old hardlinked copies were never overwritten in place.
The full verification ledger and pre-sync inventory are beside that backup. Previously
rejected cleanup paths remain untouched. Input datasets and Exp2 output were outside the transfer.

Remote snapshot capture: 11:06 CET; its files remained unchanged at 11:20 CET. Inventory SHA256
`cbd5bff093daa60360feed95d6d39c08c74fceca8099c1939b8414cf314f31e3`.
Downloaded metadata archive `exp01-verified-file-manifest-20261103.zip`: 170,096 bytes,
SHA256 `a21cf7e5baed42e8bc057b93bd6260dd37ed1f3c3f158b59a0c1bd6c9912f8b2`.
It contains the file manifest; the payloads were separately synchronized by SFTP. The notebook
snapshot metadata now records this complete mirror and preserves the per-run historical provenance.
Copied Exp0 reports show CPU 62516762 PASS (14 stages) and GPU 11763662 PASS (117 stages).
Local reporting code is newer than the frozen scientific revision; no VSC pull or runtime change occurred.

All seven General/Exp1 notebooks executed locally, producing 53 PDFs plus captions and final printed
summaries. Their code cells are unchanged. Every figure was rendered and visually reviewed; all 53
match the preceding reviewed images pixel-for-pixel. The 47 loader/runner tests and 91 generated-output,
readability and visualization tests pass. The unchanged local source previously passed the full suite
(2,259 tests, 38 VSC-only skips); those CUDA cases passed in native GPU Exp0. Post-execution file-set
and SHA256 verification is recorded in `.runtime-build/exp01-sync-20261103/post-notebook-verified.json`.

Descriptively, 25%/50% mixtures remain close to the matched 0% control on credit; 100% is worse,
especially on OOD. At 100%, mean designated-holdout changes are PD ROC-AUC -0.0153 (eight datasets)
and LGD R² -0.0707 (three datasets); native OOD changes are -0.1221/-0.2498 across 25 datasets per task.
These small credit panels and their nominal dataset intervals do not establish a broadly superior mixture.
No learning rate, prior choice, experiment configuration or accepted job was changed from these results.

Task-owned temporary transfer helpers and the 53 rendered images/nine contact sheets were archived
locally with every member hash verified before retirement; the empty incoming staging tree was removed.
Evidence remains under `.runtime-build/exp01-sync-20261103/`; the pre-sync recovery copies are retained.

### 03-11-2026 — full GPU PASS and live campaign audit

**Production revision:** frozen `0494e7a14bc320176de3a8665051630461f3c316`;
source `571c1f6cb2bc968a8bc572fc6fd2b8f202f3bca6024db0a28aadb3268fd36d3b`;
training implementation `5300c294e4078273fa635f18b82714ebfdb17169046f25035e356aa2392a8e75`.
TFM pin `26156f36eb7676e8d9050547817346db548b0e85`; VSC pin
`0ed4ffa89d312b9bb6411681b6a14f81672439f3`. Accepted PROJECT image r3 remains SHA256
`83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`.
No scientific setting or active job was changed.

The current provenance planner independently accepted all Exp1 stages: 12/12 main
training and 64/64 benchmark slots per task; 15/15 component training and benchmark
slots per task. Exp2 training reports 16/32 LGD done and 0/32 PD done, both live with
no submission commands. Its operational ledger write correctly hit the image's read-only
checkout guard after these checks; the captured records were verified and saved natively
as `.runtime-build/audit-20261103/provenance-planner.json`, without relaxing the mount.

**Experiment 0:** CPU 62516762 passed all 14 stages. GPU 11763662 completed 0:0
in 18:05:41, from 02-11 08:42:43 to 03-11 02:48:24 CET, all **117/117 stages PASS**.
The 38 mandatory CUDA cases have zero failures/errors/skips, including both exact
production constant-context regressions. Explicit PROJECT-image login preflight
`--require-exp0` and `src.validation.verify` passed again: 123 items, zero failures.
This is actual Slurm/runtime acceptance, not a projection from local tests.

**Experiment 2, 09:07–09:23 CET:** all 64 configurations are accounted for: 16 completed
TabPFN LGD runs, 19 running replacements (16 TabICL LGD, three TabICL PD), and 29
PD replacements PENDING Priority. No dependency is stuck and no replacement failed.
Latest logged steps: LGD 600–800, PD 400–600; metrics are recorded every 100 updates,
so these are lower bounds. All 64 metric journals parse; all logged losses are finite.
The completed LGD runs have 1,500 updates, 24,000 tables, zero skipped tasks and all
five registered checkpoints. Their recorded mixed-precision retries are successful
fallbacks, not omitted tables. Resumed runs retain prior signal-stop summaries until
the current segment finishes; these do not mean the current job failed. Long runs use
the existing checkpoint/requeue mechanism at allocation boundaries. No duplicate
training, Exp2 benchmark, Exp3 or scheduled task was launched.

**Log audit:** every `.log/.out/.err` under both output tiers plus root CreditICL Slurm
fallbacks was read: 1,341 files, 793,531,171 bytes. Only expected offline-wrapper
weight warnings and the two historical failed deployment controllers were flagged.
The released local weight paths and successful reference probes are explicit in the
same logs. Inventories, hashes, metrics and scheduler snapshots are retained under
`.runtime-build/audit-20261103/` on DATA and locally.

**Cleanup:** 465 exact terminal root fallback files (20,585 bytes, mostly empty) were
archived, every member hash checked, and only then removed. All 38 active fallback
files were retained. Archive:
`/lustre1/project/stg_00211/CreditICL/recovery/20261103-audit/terminal-fallbacks.zip`,
89,477 bytes, SHA256 `833cdd9b99f290d0607b9ef78ce14453c2261fb8984d5425cbcdee690e96a5aa`.
No successful output or checkpoint was removed. Earlier rejected cleanup paths remain untouched.

**Storage:** canonical/local TEMPLATE matches SHA256
`9e113ac591a7d702bc61060a3bce703419cb7a296fcf9cf77283c1e5ffd27a20`.
DATA quota 46,390M/76,800M, 401k/10M files; PROJECT stg_00211 822.1G/1T,
58,721/150k files (shared allocation, about 201.9 GiB free). This is below the
250-GiB starting reserve for additional campaigns; existing accepted training is preserved.
CreditICL itself occupies about 138 GiB on PROJECT, including about 123 GiB outputs
and 13 GiB runtime/rebuild assets. Shared project scratch is only 17 KiB GPFS/4 KiB Lustre
scaffolding; account-wide scratch usage belongs partly to other projects. New GPFS project
prj_00013 appears in quota output, but its paths/access/cutover have not been validated;
no migration is inferred. Available account credits: 50,956,495 (snapshot, not final billing).
Current Exp2 checkpoints occupy 84.59 GiB; the five registered milestones for all 64
runs project to about 133.21 GiB from measured model checkpoint sizes (about 48.62 GiB
additional, plus rolling/recovery states and logs). The current reserve accommodates
that estimate; shared-allocation growth can still change available space.
The 9.77-GiB DATA venv remains a recovery exception pending other-login-host consumer
clearance; distinct Conda recovery is still unverified. No environment was moved/deleted.

**Validation maintenance:** reproduced the known full-queue delay with a real child process;
`execute()` now preserves the phase only for `slurm-client`, retaining portable test isolation.
Three capacity/CPU-proof cases and two isolation cases cover the child boundary. The affected
local suites passed 101 tests. A separate DATA checkout is required for VSC testing while
production consumes 0494e7a; this change does not alter the training fingerprint.
The new 330-file compact audit/Exp0 download was verified member-by-member (1,447,054 bytes,
SHA256 `a735b4e85eca4a9bdba858dfae427c481fa5e47743d8d39742896284364ee3af`).
Ruff reports the same two pre-existing import-order findings in the changed runner;
the changed test file has none. No lint-clean claim is made for the broader repository.

**Maintenance test 62527538:** published 00438dc was pulled with `--ff-only` into
`$VSC_DATA/CreditICL-validation-client-20261103`, not the consumed production checkout.
wICE interactive, two CPUs/8G/20 minutes, actual quote 0; site normal MaxSubmitPU 500
was reverified before receipt-backed admission. It failed 1:0 in 3m44s: 100 passed,
one temporary-script import fixture failed. The isolated launcher omitted PYTHONPATH,
which the production wrapper supplies. Add that explicit path and preserve child logs
before node-local cleanup, then retest. Original production jobs/source remain intact.

**Maintenance retry 62528019:** 169fff7 was pushed, then pulled into the now-idle test
checkout. Same allocation/quote 0, accepted through the separate
`manifests/submissions/validation-client-20261103-pythonpath.json` receipt. Slurm
COMPLETED 0:0 in 3m42s; JUnit confirms **101 passed, zero failures/errors/skips**
(187.09s; batch MaxRSS 685,384K), including the temporary-script guard and all five
new child-environment cases. Source fingerprint
`cf25443c2cc992947a9b8340f15f38cd3edd26befea0b79767f7f4b73867daa4`;
training implementation remains unchanged. These are targeted maintenance tests,
not replacement full Exp0 receipts for a production-source cutover.

Both maintenance jobs were terminal and no test-checkout consumer remained before
cleanup. All 36 test logs/reports/records were archived and hash-verified:
`/lustre1/project/stg_00211/CreditICL/recovery/20261103-audit/validation-client.zip`,
24,104 bytes, SHA256 `6e249a16afba6acce358b45d2996b5131fe99564b5d028dad826634e5fa8b712`.
The temporary DATA clone was removed; production HEAD remains 0494e7a.

**09:58 CET health recheck:** 16 complete, 19 RUNNING, 29 PENDING Priority; all 70
Exp2 logs remain free of error candidates, with finite current losses and advancing
metrics. Separately read 297 diagnostic CSVs/14,501 rows: no native `error`/`n_errors`
failures. Numeric PIT-uniformity error columns are performance metrics, not exceptions.
Nine retained operational logs contain only one already documented old bootstrap-test
failure, not a new production failure. Per-login-host retirement checks timed out at
the SSH banner; normal gateway accounting/log reads work. Keep both legacy environments
until consumer clearance/recovery requirements are resolved. No new authentication flow,
environment migration, production cancellation or scheduled monitor was initiated.

**Local closure:** `.CreditICL/Scripts/python.exe -m pytest -q` exited 0:
**2,259 passed, 38 VSC-only skips**, 510 warnings, 54m11s. The changed Python test file
passes Ruff; the runner's two import-order findings match the prior commit. Shell syntax
passes. Source and training fingerprints match those verified on the isolated VSC checkout.
The 10:21 CET production recheck still reports 16 complete/19 running/29 Priority-pending,
zero errors in 70 Exp2 logs; latest logged LGD steps 700–900, PD 400–700. Every approved
training ID is already accepted; no duplicate production submission is needed. Remaining
training continues through the existing requeue guard. Benchmarking and Exp3 remain later stages.
At 10:25 CET the final quota snapshot is DATA 46,393M/76,800M (401k files), shared
PROJECT 834.6G/1T (58,737 files; about 189.4 GiB free), and 50,980,457 available
credits. The production tracked worktree/pins are unchanged and clean. The temporary
maintenance clone is absent; no runtime/environment/checkpoint was relocated.
All 13 compact remote audit records were checksum-verified locally. After retaining a
verified local operator archive, 52 task-owned TEMP helper/result/log files were removed;
the full pytest log and cleanup ledger remain under `.runtime-build/audit-20261103/`.
Previously rejected cleanup targets remain untouched. Final reporting-only commits are
published without pulling them into the consumed production checkout.

### 02-11-2026 — all repaired admissions complete; GPU validation running

**10:35 CET final admission check.** All **48 effective replacements** are accepted and
observed PENDING Dependency in both expanded `squeue` and `sacct`: 16 LGD and 32 PD.
Together with the 16 preserved completed LGD runs, all **64 scientific configurations**
are covered. Original indices, two paired seeds, both fixed rates, four mixtures, both
backbones and the 1,500-update target remain unchanged. Initial allocations are Mindwell
B200 / one GPU / 24 CPU / 180G / 12h, with native checkpoint/requeue recovery.

Every replacement binds commit `0494e7a14bc320176de3a8665051630461f3c316`, PROJECT root
`/lustre1/project/stg_00211` and `afterok:11763662`. No missing or duplicate effective
logical slot remains. PD3 is **11763815_3**; its never-started predecessor **11763812_3**
is explicitly CANCELLED and superseded, not counted as active coverage. The same arguments
were verified in its separate restore receipt. All three receipt locks are absent,
all in-flight fields are null, and the finite submitter exited 0 at 10:30:11 CET; its PID
is gone. No scheduled task or deployment controller remains.

Canonical proof on DATA and locally:
`.runtime-build/repair-constant-context/all48-effective-admission-verified-20261102.json`,
SHA256 `c579251ed273e3370d0fce2463ab745a94eca3fa6d8e65e9c10e9aee9c9e4f00`.
The original two admission receipts remain unchanged; the additional PD3 restore receipt
and slot-handoff ledger document the substitution. The raw base acknowledgment list alone
includes the cancelled predecessor and must not be treated as effective live coverage.

GPU **11763662** is still **RUNNING** on r11g17: **46 completed stages PASS**, including
**38/38 CUDA tests**, zero failures/errors/skips. Both constant-context production-batch
regressions and long-context capacity/mixed-batch checks passed. Full GPU acceptance has
not finished; the largest full-row credit benchmark is active. The queued training tasks
also require current-source Exp0 readiness at startup, so they cannot train before PASS.
No Exp2 benchmark or Exp3 was submitted. Keep DATA at 0494e7a; these reporting changes
must not be pulled into the consumed checkout. Actual library pins remain TFM
`26156f36eb7676e8d9050547817346db548b0e85` and VSC
`0ed4ffa89d312b9bb6411681b6a14f81672439f3`; runtime image r3/SHA83f8145e... is unchanged.

Resource snapshot: DATA 46,336M/76,800M and 397k/10M files; PROJECT 767.7G/1T and
54,133/150k files (about 256.3 GiB free); account available 60,514,238 credits.
The admission guards passed before each submission. These are shared-allocation/account
snapshots, not CreditICL-only usage or an actual billing statement.

Effective parent/index mapping (array fragments reflect site admission capacity):

| Task | Parent job | Original indices |
|---|---|---|
| LGD | 11763693 | 0 |
| LGD | 11763738 | 1 |
| LGD | 11763742 | 2 |
| LGD | 11763744 | 3 |
| LGD | 11763756 | 4 |
| LGD | 11763761 | 5 |
| LGD | 11763769 | 6 |
| LGD | 11763782 | 7 |
| LGD | 11763797 | 16-22 |
| LGD | 11763798 | 23 |
| PD | 11763811 | 0-2 |
| PD | 11763815 | 3 |
| PD | 11763820 | 4 |
| PD | 11763822 | 5 |
| PD | 11763825 | 6 |
| PD | 11763832 | 7-8 |
| PD | 11763857 | 9 |
| PD | 11763860 | 10-11 |
| PD | 11763863 | 12 |
| PD | 11763866 | 13 |
| PD | 11763871 | 14 |
| PD | 11763878 | 15 |
| PD | 11763881 | 16 |
| PD | 11763885 | 17 |
| PD | 11763887 | 18 |
| PD | 11763890 | 19 |
| PD | 11763895 | 20 |
| PD | 11763897 | 21 |
| PD | 11763899 | 22 |
| PD | 11763901 | 23 |
| PD | 11763903 | 24 |
| PD | 11763907 | 25 |
| PD | 11763909 | 26 |
| PD | 11763912 | 27 |
| PD | 11763919 | 28-29 |
| PD | 11763923 | 30-31 |


**09:19 CET progress.** Service access recovered. GPU 11763662 is RUNNING on r11g17,
source 571c1f6c..., with 15 completed stages PASS. The mandatory CUDA suite passed all
38 cases in 192.67s, zero failures/errors/skips, including the two full production-batch
constant-context regressions. Full GPU acceptance is still pending. All 16 LGD replacements
and PD indices 0–3 are admitted; PD 4–31 (28 tasks) still await shared capacity.

The full queue also blocked the GPU's `sbatch --test-only`. Paused only our admission CLI
after checking no RPC was in flight, then cancelled only the unstarted dependent PD task
11763812_3 (runtime zero). Actual Mindwell dry-run acceptance passed at 09:07; the complete
Slurm-client stage took 1,445.19s. Restored the identical PD index 3 as **11763815_3** through
`constant-context-repair-20261102-pd3-slot-restore.json`, then resumed the main submitter at
09:09. No training work, source, runtime, grid or other project's job changed. The original
acknowledgment remains historical; final effective coverage must substitute 11763815_3 for
cancelled 11763812_3. Handoff ledger:
`.runtime-build/repair-constant-context/pd3-dryrun-slot-handoff-20261102.json`.

Driver follow-up for an idle checkout: `campaign.execute` removes `CREDITICL_EXP0_ATTEMPT`
from every `cpu=True` child, including GPU Slurm-client checks. Consequently its existing
GPU capacity-response branch cannot recognize the attempt. Preserve that scope for this
client and add an actual campaign-child environment test at the next validation maintenance.
The current frozen campaign instead obtained real dry-run acceptance; no gate was bypassed.

**08:22–08:40 CET continuation.** Reconciled eight accepted LGD replacements, indices
0–7, from parents 11763693, 11763738, 11763742, 11763744, 11763756, 11763761,
11763769 and 11763782. All retain source 0494e7a and `afterok:11763662`; no uncertain
acknowledgment or active owner lock remains. Forty tasks still need admission: LGD 16–23
and PD 0–31. The previous submitter's terminal response is unavailable, so its exact stop
reason is unconfirmed. The current-source GPU report and startup logs are absent; live
accounting cannot yet establish a new state.

Both login nodes return Slurm accounting/controller errors. Native DNS queries for the
configured accounting service return NXDOMAIN; clearing linker variables does not repair it.
The alternate login route preserved strict host-key checks and verified stdout/stderr/exit 7,
but showed the same scheduler failure. No system/SSH configuration or backend address was
changed. A single finite CLI now resumes the existing receipts, checks actual site limits and
reserves again at admission, and records its owner/lease/log under
`.runtime-build/repair-constant-context/`. It cannot submit while queue state is unknown.
No scheduled task is created; code/runtime and other campaigns remain untouched.

**Earlier, 00:35 CET.** Fresh CPU 62516762 is accepted (14/14 stages); GPU 11763662 remains queued for B200
validation. No repaired Exp2 task is claimed as submitted until its admission receipt exists.
The expanded Mindwell user queue is 500: 495 CreditPFN, four JEVPFN and this validation job.
The B200 partition's `normal` QoS has MaxSubmitPU 500; the account QoS does not override it.
The finite native metadata CLI waits through the existing receipt manager, preserving original
array indices and checking frozen source/runtime, validation state, quotas and credits again
at each admission opportunity. No recurring monitor, additional Slurm controller, running-job
throttle or change to another project was made. Replacements depend on `afterok:11763662`.

At 00:35 CET, **one of 48 replacements is accepted**: `11763693_0` (LGD index 0,
PENDING Dependency). The other 47 are not yet admitted. Its receipt verifies expected
commit 0494e7a, unchanged task index and dependency on 11763662, with no uncertain
acknowledgment. The single finite admission CLI continues waiting; no recurring task
was created. Latest checks: clean frozen DATA checkout, PROJECT 764.2G/1T and
53,981/150k files, DATA 46,273M/76,800M, available account balance 62,609,420 credits.
The fresh admission guard passed. GPU validation remains PENDING Priority, so its new
38-case CUDA suite has not run. Neither admission nor the predecessor's GPU PASS is
claimed as current-source GPU acceptance.

The first reserve check stopped before submission with less than 250 GiB free PROJECT space.
Retired only 32 unregistered rolling checkpoints at steps 750/1,250 from the 16 completed
TabPFN LGD runs, reclaiming 22,400,394,360 bytes. Verified all 80 registered milestones
(125/250/500/1,000/1,500) unchanged by size/mtime. Every Exp1 checkpoint and unfinished Exp2
recovery checkpoint remains. PROJECT then had approximately 256.6 GiB free; the same reserve
guard remains mandatory at actual submission. Ledger:
`.runtime-build/repair-constant-context/retired-completed-rolling-checkpoints.json`.

All 12 notebooks reran; all 56 exported PDFs were rendered and visually reviewed. Complete
Exp1 credit/OOD results, captions and printed summaries are published with the report changes.
These reporting-only changes must not be pulled into the VSC checkout consumed by job 11763662
and its dependent repairs, which stays at 0494e7a. Automatic approval review rejected removing
the local temporary transfer staging folders without a specific reason; they remain intact.

### 01-11-2026 — repair raw constant contexts; preserve valid CPT work

**Cause and scope.** The only failed task is PD `11762003_11` (TabPFN-3, LR 1e-5,
100% credit, seed 0), at batch 37/slot 15. Its 1,024-row/two-feature table varies overall,
but all 823 context rows are constant. Native TabPFN removes every feature and raises
`TabPFNValidationError`. PD index 15 has the same synthetic stream at the other fixed rate.
This is synthetic preprocessing validity, not a fault in the real credit datasets.

**Repair.** Match native constant-feature removal, including NaN behavior, on the raw
context after any class-balance row permutation. Redraw an unusable candidate using the
next deterministic attempt seed for the same source/slot/shape/split. Keep every update's
16 TabPFN tables, zero skipped tasks, 1,500 updates, both seeds/rates and all mixtures.
Encoded TabICL/Exp1 samples are unchanged. Persistently invalid sampling still fails at
the existing bounded attempt limit; no easy-task or downstream-performance filter was added.

Old implementation `01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`
to reviewed `5300c294e4078273fa635f18b82714ebfdb17169046f25035e356aa2392a8e75`:
only exact registered recipes, seeds 0/1, unchanged dependency versions and full table
accounting may resume. Completed pre-fix updates necessarily passed native preprocessing;
the repair changes no previously completed update. Original fingerprints are retained,
not relabelled. Changed/unknown recipes, packages, skipped budgets and source fail closed.

**Safe shutdown.** Pending Python would read files at startup, but Slurm caches batch scripts
and all old submissions require commit 56edade. Updating their checkout in place would mix
versions or fail that guard. Held/cancelled 30 pending tasks; sent USR1 to 17 running batch
scripts, verified each stopped summary's step/table count and nonempty PROJECT checkpoint,
then cancelled their requeued slots. No old CreditICL consumers remain. Preserve the 16
completed TabPFN LGD runs and every valid unfinished checkpoint; replace only the remaining
48 IDs (LGD 0–7,16–23; PD 0–31). Other projects' jobs were untouched.

**Evidence and cleanup.** Recovery root on PROJECT:
`CreditICL/recovery/20261101-constant-context/`.

- `failed-pd-11762003_11.zip`: eight files/63,013 bytes; SHA256
  `9645f894cef504e0899030136f9ce3e56d11719a3a0c51677b2916ebad824f37`.
- `retired-incomplete-segment-logs.zip`: 58 files/1,785,438 bytes; SHA256
  `990faceb06575735cda118b130ea033301e306d0200d134ae3545a7716e9ff8c`.
- `accepted-exp0/{data,project}`: previous accepted validation, moved out of active Exp0
  only after all 612 transferred files matched hashes and no consumers remained.

Archive members were verified before exact active files were removed. Configs, summaries,
metrics and valid checkpoints remain for resume. Operational ledgers are under DATA
`.runtime-build/repair-constant-context/`; all 64 original job/run mappings and the
pre-deployment recipe comparison are saved in output manifests.

**Local verification.** Full pytest: **2,254 passed, 38 VSC skips**, 23m48s, zero failures.
Final relocated regression selection: 90 passed/38 VSC skips; 164 broader affected checks
previously passed. Native preprocessing accepts the repaired full 16-table production
batch. Both rates have mandatory CUDA optimizer-update regressions in the actual
`tests/test_vsc.py` GPU suite (38 collected cases); local tests do not claim CUDA acceptance.
Fresh full source-bound CPU/GPU Exp0 and dependency-gated replacement admissions follow.

**Full output transfer.** User-authorized Exp0/Exp1 download includes checkpoints, not raw
datasets or released weights: **3,206 files, 41,295,352,039 bytes**, every source SHA256
verified locally. Inventory SHA256
`50a9d416ba5a59540888251f67fc916210caae72d4a0e1032312ee6b3f79deaa`.
Inventory/verification are in ignored `.runtime-build/output-transfer-20261101/`; local
canonical output receives a manifest and preserves older conflicting files before merging.
This provides an independent local Exp1 checkpoint copy; PROJECT remains the working store.

**Resources before deployment.** DATA 46,255M/76,800M and 394k/10M files; PROJECT
773.2G/1T and 53,051/150k files. Exp2 has 210 checkpoint files/108,453,609,594 bytes.
Account available 61,062,908 credits. Fresh quotes: CPU 0, GPU validation 734,400,
replacement LGD 5,875,200 and PD 11,750,400 credits maximum for the first allocations;
actual billing/requeues differ. No new scientific setting or recurring monitor is added.

**Deployment/validation admission.** GitHub push then safe DATA fast-forward to
`0494e7a14bc320176de3a8665051630461f3c316`; source
`571c1f6cb2bc968a8bc572fc6fd2b8f202f3bca6024db0a28aadb3268fd36d3b`.
The actual PROJECT image reports training fingerprint
`82dbe27fb913c0bfa180172a2970c90053c1f8a7984b1dc8027ad9aab00089fb`
and accepts 54 completed Exp1, 16 completed Exp2 and 29 partial Exp2 summaries.
CPU **62516762** completed 0:0 on wICE k28i14 in **41m16s**: 14/14 stages PASS,
2,232 tests/two expected skips, all 12 notebooks. PROJECT-image login CPU readiness accepts
all current artifacts. Receipt: `manifests/submissions/constant-context-repair-20261101-cpu.json`.
GPU **11763662** accepted on Mindwell (B200/24CPU/180G/24h, quote734,400), currently
PENDING Priority. Receipt: `manifests/submissions/constant-context-repair-20261101-gpu.json`.
Full new-source GPU acceptance is pending; do not pull into the consumed checkout.

**Further cleanup.** After independent local verification, retire only the 19 superseded
Exp0 diagnostic checkpoint binaries (6,711,814,477 bytes) from PROJECT recovery. Their
full local copies, compact remote reports and exact hash ledger remain; no production
checkpoint was removed in this diagnostic cleanup. The later retirement of obsolete rolling
states, with all registered production milestones retained, is recorded above.

Removed 61 empty terminal Exp2 root fallback files after checking no original Exp2 job
remained active; nonempty successful logs were retained. Local notebook rerender: **12/12
successful, 56 PDFs**, all visually reviewed from Poppler renders; final summaries/captions
and native OOD panels are populated. The supported single-worker drawing option avoids an
intermittent Windows pool-handle failure (its precise cause is unconfirmed); it changes only
illustrative diagnostic draws, whose worker-dependent seeding was already disclosed. Local
output checks: 105 passed, with the two Bash-dependent cases verified separately using the
existing Git Bash PATH. No visualization source or experimental design changed.

**Complete Exp1 interpretation.** Downloading the pinned 50-dataset OOD manifest (metadata
only, with all 56 shared-reference result files verified) makes the complete native OOD
tables available locally. Across the designated holdout datasets, mean treatment-minus-0%
differences are:

| Credit mixture | PD AUROC (8 datasets) | LGD R² (3 datasets) | Classification OOD AUROC (25) | Regression OOD R² (25) |
|---|---:|---:|---:|---:|
| 25% | +0.000515 | +0.000664 | −0.000508 | +0.002449 |
| 50% | −0.000011 | −0.002716 | −0.000724 | −0.003483 |
| 100% | −0.015259 | −0.070688 | −0.122099 | −0.249794 |

Mixed priors show no clear credit gain here. The pure-credit scratch arm performs worse
on credit and loses substantial general-domain performance; this measured outcome is not
the TabPFN context crash (Exp1 uses the unchanged encoded TabICL route). PD pure-credit
degradation also occurs on binary OOD (−0.022150 AUROC; eight tasks), and is larger on
multiclass OOD (−0.169134; 17 tasks). The whole-package treatment changes multiple exposures;
these contrasts alone do not identify one mechanism. LGD has only three holdout datasets,
so its nominal bootstrap intervals and discrete sign tests require particular restraint.
Seeds/folds are averaged within datasets, not treated as extra independent tasks; the
holdout has historical project exposure. No best mixture/rate is selected for Exp2.
Compact contrasts: `manifests/exp1_interpretation_20261101.json`; complete notebook summaries
and captions remain in `All_Results.md`/`CAPTIONS.md`.

### 01-11-2026 — Exp1 complete; Exp2 live audit and constant-context failure

**Acceptance.** GPU 11756026 completed 0:0 in **17h47m45s**, ending 31-10 at 05:29:29 CET:
117/117 stages PASS. CPU 62475000 passed 14 stages, 2,142 tests/two allowed skips and all
12 notebooks. PROJECT-image login preflight and final verifier passed on 31-10 (123 checks,
zero failures). Image r3 is accepted, with report hashes/job IDs in `.runtime-build/current.json`.
Frozen commit **56edadef4efe81120ea104870e6ba2e497f89b6a**, source
`0d9017bb83b5406362e6e81076220274a04aff081c6d99fe65fb196022d6f370`; image SHA256
`83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`.
TFM/VSC pins: `26156f36eb7676e8d9050547817346db548b0e85` /
`0ed4ffa89d312b9bb6411681b6a14f81672439f3`. No scientific setting changed.

**Admitted 31-10 at 10:36:57–58 CET:** Mindwell B200, one GPU/24 CPU/180G/12h per task,
no extra dependency or running throttle. All **94 original IDs** accepted exactly once:

| Job | Work | IDs | Initial allocation quote, credits |
|---|---|---|---:|
| 11762000 | Exp1 LGD benchmarks | 9–11, 21–23, 33–35, 45–47, 57–59 | 5,508,000 |
| 11762001 | Exp1 PD benchmarks | Same 15 IDs | 5,508,000 |
| 11762002 | Exp2 LGD training | 0–31 | 11,750,400 |
| 11762003 | Exp2 PD training | 0–31 | 11,750,400 |

Total initial quote **34,516,800** versus **80,174,739 available**; not a final bill or bound
on requeues. Quota/file headroom and live site ceiling 500 passed. Arguments/quotes/receipts:
output `manifests/production-*-20261031.json` and `manifests/submissions/production-20261031/`.
Exp2 retains the unchanged two-backbone/four-mixture/two-rate/two-seed grid: 64 × 1,500 updates;
milestones 125/250/500/1,000/1,500. Exact resolved hyperparameters remain in configs/run manifests.

**Exp1 complete.** All 30 new benchmarks completed 0:0 (LGD ~7–8 minutes, PD ~2h17m–2h34m).
Current provenance checks accept 64/64 main slots per task, including shared references, and
15/15 component slots per task: **158/158**, with requested credit/OOD cells and five folds.
All **54** training summaries remain compatible at 12,500 steps, including the six 25% extensions.

**Exp2 at 20:52 CET:** 16 TabPFN LGD runs finished 1,500 updates/zero skipped tasks; five
TabICL LGD running, eleven checkpointed/requeued at steps 545–554 on SIGUSR1. PD: thirteen
running, eighteen queued for priority, one failure. TabICL projects ~32–34 GPU-hours per arm;
further twelve-hour segments are expected. No Exp2 benchmark or Exp3 was launched.

**Failure:** `11762003_11` (underlying job 11763192), TabPFN PD, LR 1e-5, credit=1, seed 0,
failed 1:0 after 22m55s. At step 37 native preprocessing reported all features constant.
Local CPU reproduction identifies batch 37/slot 15: 1,024 rows/two columns, 823 context rows
with no varying feature, while query columns vary. Local TabPFN 9.1.0 matches VSC; local
Torch 2.13.0 differs from VSC 2.11.0+cu128. Index 15 uses the same task stream and is exposed.
This is a synthetic-context handling gap, not a real-dataset quality issue. No checkpoint yet;
preserve the run, define/test deterministic handling and matched task counts before restarting.
No active source, job or checkpoint was altered during this audit.

**Other observations:** 76 current batch plus 46 child logs (~77.5 MB) scanned; no quota/OOM
failure found. AMP backoff/FP32 replay and huge finite LGD target-tail loss windows recur.
Development prediction nonfinite fractions are zero. Some LGD credit/OOD monitoring scores
worsen; these are development diagnostics, not final five-fold results or grounds to select/change
the grid. Probability-sum warnings recur, while strict benchmark/result validation passed.
The dirty-code banner includes untracked artifacts; tracked source is clean and its hash matches.
DATA quota 46,277M/76,800M; PROJECT 760.3G/1T (~263.7 GiB and 97,585 files free). Own Exp1
PROJECT output ~31.5 GiB, Exp2 checkpoints ~89 GiB; available credits 63,016,807 at first snapshot.
Runtime/assets stay on PROJECT; legacy DATA environments remain documented recovery exceptions.
The monitor was deleted on 01-11 at Andreas's request. Evidence: `manifests/audit-20261101-*.json`;
compact local files under ignored `.runtime-build/audit-20261101/` were transferred with SHA256 checks.

### 30-10-2026 — CPU 62475000 accepted; GPU 11756026 started

During LGD resume, batch 217 exhausted AMP retries and used the existing FP32 replay.
The step-220 mean loss was 6.026e31; the next window was 0.95019. Local deterministic
reproduction found original-prior slot 6: raw target magnitude 8.4543, transformed query
magnitude 3.3905e18. With zero logits, the unchanged native MSE gives a comparable
20-update-window contribution of 6.027e31. This identifies a transformed-target tail,
not sustained divergence by itself. No code, task, target or objective was changed and
the job was not interrupted. At step 400 the ordinary loss is 0.87454, LendingClub
monitoring RMSE 0.130209/R2 0.746569 (initially 0.130918/0.743799), and a checkpoint
was saved. This shows no collapse on that monitoring table. Full benchmark/weight
checks remain required before GPU acceptance.

CPU **62475000** completed **0:0 in 32m51s**: all **14/14 stages PASS**, unit JUnit
**2,142 passed/two allowed skips/zero failures/errors**, and all 12 notebooks executed
(303.12s). Native `preflight --require-exp0-cpu` accepted the complete current receipt,
inputs, runtime identity and artifact hashes; the login/allocation defect is resolved.

GPU **11756026** was admitted through the receipt manager and is RUNNING on Mindwell
**r11g12**, B200, 24 CPU/180G/24h, unchanged image r3 and frozen **56edade**/source
`0d9017bb83b5406362e6e81076220274a04aff081c6d99fe65fb196022d6f370`.
Actual source-bound report is RUNNING, attempt `gpu_20261030T104216Z_j11756026_72bc0183`;
receipt `manifests/submissions/restart-runtime-context-20261030-gpu.json` records the
accepted ID with no uncertain in-flight acknowledgment. Startup and native accounting
were inspected; full GPU acceptance remains pending. No production work was launched.

Live site MaxSubmitPU **500**; full-allocation quote **734,400 credits maximum**, with
**90,907,892** available before admission. These are a quote/balance, not a billing record.
Quota then: DATA 45,912M/76,800M (390k/10M files), PROJECT 638.8G/1T (48,242/150k files).
Shared scratch account totals include other projects; no environment or checkpoint cleanup
was performed. Preserve both legacy environments until GPU/recovery/consumer checks pass.

### 30-10-2026 — runtime-context repair deployed; CPU 62475000

Git-first deployment froze **56edadef4efe81120ea104870e6ba2e497f89b6a**, source
`0d9017bb83b5406362e6e81076220274a04aff081c6d99fe65fb196022d6f370`, with
unchanged training implementation and image r3. On the actual login node, repaired
inventory equals all **nine** records saved by CPU 62474628. This verifies the context
defect directly; it does not substitute for fresh source-bound CPU/GPU acceptance.

CPU **62475000** is RUNNING on wICE k28i16, eight CPUs/48G/four hours; quote **0**,
dependencies and PROJECT `/opt/venv` startup checks PASS. Admission receipt:
`manifests/submissions/restart-runtime-context-20261030-cpu.json` under output.
Tracked checkout is unchanged; generated untracked records explain the job's dirty flag.
No GPU or production work was submitted. Wait for the complete CPU receipt before GPU admission.

Before reset, 103 files from the old CPU attempt were archived and each member hash checked:
`/lustre1/project/stg_00211/CreditICL/recovery/20261030-before-context-rerun/exp0-62474628.zip`
(2,843,468 bytes; SHA256 `967f4a67819b2cf933402f723ea0303e5a2ba765a779f829d432b185cdc81300`).
Scoped reset removed 101 generated Exp0 files. All 2,204 preserved Exp1 sizes/mtimes
and 1,110 recorded JSON/CSV hashes matched both before and after; all inputs remain.

### 30-10-2026 — CPU acceptance; cross-context readiness defect

CPU **62474628** completed **0:0 in 32m06s**, wICE k28i14, 8 CPU/48G, image r3 and
commit 79609ff/source 8b2be00 unchanged. **14/14 stages PASS**; JUnit has **2,126 passed,
two allowed skips, zero failures/errors**. All 12 notebooks executed (352.07s stage).
The login-node `preflight --require-exp0-cpu` then failed before any GPU submission.
Direct comparison found only `runtime_image` missing from the login inventory; all eight
shared data/weight records matched, as did the verified SIF hash. No dataset or weight
change caused this refusal. The GPU admission receipt does not exist; no GPU was submitted.

The defect was an allocation-only condition around runtime provenance. The fix records
the verified image for all VSC callers, including login-node readiness, and still rejects
missing/changed image identities; ordinary local venvs remain supported. New regression
cases exercise allocation-to-login equality and the complete readiness round trip. A
fresh current-source CPU/GPU campaign is required after the tested Git deployment;
the passed old receipt remains historical evidence. Training implementation is unchanged.
Local verification: the collected full suite passed **2,160 tests**, with 36 VSC-only
skips (24m20s); after adding GPU parametrization, all eight final CPU/GPU round-trip
cases passed separately. The earlier affected suite passed 135 tests. No notebook
or visualization implementation changed in this repair.

### 30-10-2026 — corrected reporting frozen for fresh validation

CPU **62474628** was submitted on wICE k28i14: eight CPUs, 48G, four-hour limit,
PROJECT image r3 unchanged. Frozen commit **79609ffc9c50d64c9fa21e824efb8c0d2cb0b0b1**,
source `8b2be00fc102b5e1a8f65809977008870676208300769e0bd5247a657d9ca732`.
Local evidence: full suite 2,148 passed/36 VSC-only skips, then 86 affected checks after
the caption-only correction; all 12 notebooks executed again, zero errors. All 40 final
PDF renderings match the visually reviewed images exactly. No GPU acceptance is claimed.

Before each reset, archives and every member hash were verified on PROJECT:
`recovery/20261030-before-inventory-test-rerun/exp0-62471657-11754388.zip`
(118 members, 2,861,843 bytes; SHA256 `3b88750c00f40eaa13483e2f0d6a16aaa0d0677c3651e586160b8dbc3f71dcea`)
and `recovery/20261030-before-caption-rerun/exp0-62474587.zip`
(12 members, 25,748 bytes; SHA256 `7623c591b36f56c6ed860659232f9d728f90f1236ff0d879a8fe9506f4807ccf`).
Only retired Exp0 output was cleared (112 then 10 files); all 2,204 Exp1 files retained
their sizes/mtimes and all 1,110 recorded JSON/CSV hashes matched before and after.
Datasets, released weights, completed checkpoints and production receipts were preserved.

The 30-minute heartbeat supervises the remaining stages instead of starting another
idle CPU controller. Require actual CPU PASS, then full GPU PASS and verified artifacts,
then finish only missing Exp1 benchmark slots; recheck native quotas before submitting
the 64 Exp2 training work IDs. Preserve all admission receipts and stop after acceptance,
without starting Exp2 benchmarks or Exp3. The DATA legacy environment remains rollback
until full GPU acceptance and verified recovery/no consumers. No active-checkout pull.

### 30-10-2026 — GPU inventory-test failure and notebook refresh

CPU **62474587**, revision df14174, was deliberately cancelled after **2m24s** during the
unit stage: final caption inspection caught an obsolete "four checkpoint evaluations"
phrase in both Exp2 design captions. Configuration and figures already used five. Both
captions and their regression check were corrected; 86 affected tests passed. No GPU
was submitted for that revision; its partial CPU evidence is retained before replacement.

**11754388**, Mindwell B200, 24 CPU/180G, 24-hour limit, commit 931d593 and the unchanged
PROJECT image r3: **FAILED 1:0 after 1m52s**. Preflight and native Slurm checks passed;
the VSC suite passed 19 tests before its input-inventory assertion failed (71.06s).
The expected eight reference requests were present; a ninth runtime-image hash record
was correctly included. The stale test expected only eight and would also have treated
runtime metadata as a model request. The fix validates both record types separately;
six portable regressions exercise the same assertion and its rejection cases. This is
a test repair, not evidence of defective trained checkpoints. Controllers **62471655**
and **62473912** failed closed; no Exp1 benchmark or Exp2 task was launched by them.
The earlier CPU PASS remains historical evidence, not acceptance of the repaired source.
Training implementation remains `01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`;
both library pins and the image digest are unchanged from the preceding entries.

**Local reporting:** all 12 notebooks executed; 40 exported PDFs and 38 synthetic-only
preview figures were visually inspected. Plots now match the 64-run/two-seed Exp2 and
fixed 50% Exp3, all five checkpoint steps, and separate models/rates. Missing milestones
break curves; native-OOD regression CRPS stays per dataset because target units differ.
All captions and section-ordered printed summaries were regenerated. No scientific
training, prediction or benchmark settings changed.
Full local pytest: **2,148 passed, 36 allocation-only VSC skips, zero failures**, 1,201.55s.
Targeted continuation/caption-regression Ruff checks pass. The full project scan retains 170 legacy lint
findings; unrelated lint cleanup is deferred rather than mixed into this scientific freeze.

The compact VSC snapshot captured at **07:58:56 UTC** contains 1,220 Exp1/reference
JSON/JSONL/CSV files: **932 new, 288 byte-identical, zero overwritten**. Archive SHA256
`ddd0184170fd454ec4d20d60e7a1aa0bd2abde3567fff0c42ad6a457ffb09a3a` and per-file hashes
are recorded in ignored `output CreditICL/manifests/notebook_snapshot*.json`.
All 54 training runs are complete; missing benchmark slots remain pending. No datasets
or checkpoints were downloaded, and the temporary transfer ZIP was removed after verification.

### 30-10-2026 — operator continuation for a long GPU queue

Job **62473912**, wICE `batch_long`, one CPU/default 3,400 MiB, seven days, has
`afterany:62471655`. Its maximum seven-day CPU reservation quotes **20,160 credits**.
Actual scheduler state was PENDING/Dependency. GPU **11754388** remains the sole GPU
validation job; its provisional forecast moved to 01-11 23:10, while parent controller
62471655 ends at 01-11 23:28. No other campaign was changed or cancelled.

This is an explicit operator-only deviation from deploying all executable files through Git:
the continuation lives in ignored `.runtime-build/20261029-two-seeds/`, leaving the current
scientific source and image unchanged. The Slurm-copied launcher verifies helper SHA256
`dc2a7f997d08df77123d8c0a82efece6d0bece4ae01554e650f4489a31442001`;
launcher SHA256 is `5f42c2592be788a55bfa4f21f821c183417d31d61e2912be3472c5b72c140cac`.
The helper pins commit 931d593 and source fingerprint f6a16dd, requires the parent's actual
TIMEOUT accounting and original GPU-wait phase, waits for that same GPU job, then requires
complete current Exp0 evidence and complete Exp1 receipts. It uses the normal admission
manager and retains submission receipts; an already-completed parent exits without duplication.
Other parent failures or later partial-submission phases stop for manual reconciliation.

Before submitting Exp2 it runs a native wICE quota child and requires at least 5 GiB/20,000
free DATA files and 250 GiB/2,500 free PROJECT files. Quota script SHA256:
`e12c86a5a079997f032079becfa380aaeca1aac6f924dafd29ffd1619ff1e1ca`.
Local operator tests passed **18 continuation guard cases + seven quota parser cases**;
both launchers passed Bash syntax checks. Actual site `sbatch --test-only` passed and the
receipt records acceptance 62473912. These operator checks do not replace GPU acceptance.
Scientific code, packages, experiment settings and library pins remain frozen as below.
The user-requested 30-minute Codex continuation `finish-crediticl-validation-and-launch`
was created and verified ACTIVE in this chat. It must delete itself after verified acceptance
of all 64 Exp2 training work IDs and final runtime/cleanup/documentation checks.

### 29-10-2026 — capacity-aware final validation

Controller **62471655** accepted CPU **62471657**, same allocations/image/design as the
previous entry. Frozen commit `931d5934811cb551ae8cc0a2294e0c7aee0198da`, source
`f6a16dd01ec25dc101c697616f480d01c48bbbf25be054b3d60ee776ab1c4728`.
Training implementation and both library pins remain unchanged. All seven files from
the preceding failed CPU attempt were archived and every member hash verified before
its nine generated files were removed; Exp1 and shared production references survived.
**CPU outcome, 30-10-2026:** COMPLETED 0:0, **1h55m04s**, including about 84 minutes waiting
for real Mindwell dry-run acceptance at the 500-job ceiling. All **14/14 stages passed**.
JUnit: **2,108 passed, two allowed skips, zero failures/errors**; the 36 VSC GPU cases were
properly deselected for CPU. Skips are the absent external TabPFNCredit checkout and fresh
PDF outputs before rendering. All **12 notebooks executed**, 338.05s. Portable tests took
1,062.11s including harness overhead (pytest reports 1,057.54s). Data/five-fold inventories,
determinism, priors, configurations, orchestration, storage and native site clients passed.
The entire dependency fingerprint still matches the preserved legacy environment.
The controller proceeds to GPU admission; GPU acceptance remains pending. Code and image
stay frozen. The archived earlier attempts are not reused as this revision's acceptance.

GPU job **11754388** was accepted on Mindwell and is pending. The post-CPU preservation
audit rechecked all **2,204 existing Exp1 files**: sizes/mtimes unchanged, **1,110 JSON/CSV
SHA256 hashes matched**, zero differences. The proof is stored in the ignored runtime
deployment ledger; completed checkpoints and production benchmark receipts are intact.


### 29-10-2026 — frozen final acceptance and authorized deployment

Controller **62471630** (wICE batch, one CPU, 72h) accepted CPU validation **62471633**
(wICE interactive, eight CPUs/48G, 4h). Commit
`d21073029916f0c637d50c86e0e7f2017603242c`; source fingerprint
`c3d596aea23be7639533b0380110344445aecd21dc5f97a668c6cdc0cf82660a`;
training implementation unchanged at
`01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4`.
The PROJECT image remains r3, SHA256
`83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`.
The previous CPU PASS was archived and all 99 archive member hashes verified before
resetting only its 101 generated files (4 MB). All production evidence survived.
Planned gates: current CPU PASS → current GPU PASS → all missing Exp1 receipts →
submit exactly 64 Exp2 training runs and stop. Queue admission now waits at the live
ceiling without changing another campaign. Outcome pending; the runtime remains a
candidate and DATA legacy environment remains rollback until full GPU acceptance.

**Outcome:** CPU 62471633 failed early at stage 3, before portable tests/GPU work.
Preprocessing, package/runtime preflight, native site balance and the wICE dry run passed.
Mindwell explicitly returned `QOSMaxSubmitJobPerUserLimit` even for `--test-only`;
the 500-slot queue, rather than a library/resource failure, caused this stop. The client
check now retries definite capacity refusals on CPU until actual dry-run acceptance.
GPU transport checks can use that same-current CPU resource proof after a definite
daemon capacity response, avoiding paid idle time. Generic policy errors and credit/plugin
failures still fail; missing or stale CPU proof is rejected. No resource acceptance is
inferred from a refusal. Dedicated tests cover these boundaries.


### 29-10-2026 — native Slurm continuation proof before production

Array **62471598**, wICE interactive, indices **14 and 15**, one CPU/1G and two minutes
per allocation, `--requeue`. Each initial task called `scontrol --clusters=wice requeue`
on its own explicit `array_task` identifier. Both restarted once and completed **0:0**.
JSONL records before and after confirmed identical job ID, array ID, task index, CPU,
RAM and partition; only restart count changed 0→1. Mindwell's live `JobRequeue` is 1;
the actual round trip was verified on wICE, not claimed as a Mindwell GPU measurement.
No model training or other campaign was changed by this free CPU infrastructure probe.

The two training launchers now use this existing-slot continuation after the unchanged
published-checkpoint/progress guard. Native restart counts plus legacy offsets retain
the 32-segment guard. Bash behavior tests exercise success, cancellation, failed progress
guard, rejected requeue and restart arithmetic. Legacy INCOMPLETE-RESUBMITTED logs remain
accepted alongside INCOMPLETE-REQUEUED. The recovery CLI uses the same receipt-backed
admission path as normal launch/controller submission, with no default percentage throttle.


### 29-10-2026 — full CPU acceptance; GPU admission full

CPU job **62471122**, controller **62471116**, source commit **79cdcd9** and fingerprint
`704cfdfc1ae5abc823786137bd8334bc7332ebc5f27a3f4b5e2a8bf817dfd096`:
**14/14 stages passed**, including native credit/submission clients, preprocessing,
2,069 portable tests (two allowed external/fresh-output skips), all data/prior/orchestration
checks and all 12 executed notebooks. Allocation: wICE interactive, eight CPUs, 48G,
four-hour limit. The notebook kernel was observed running `/opt/venv/bin/python` inside
the PROJECT image r3, SHA256 `83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`.

The controller then received a definite **QOSMaxSubmitJobPerUserLimit** refusal for GPU
submission: the user's other campaigns occupied 500 Mindwell slots (492 pending, eight
running). No CreditICL GPU job was accepted or billed. The CPU evidence is preserved.
The template's resumable queue-admission manager is being integrated before a final
same-revision CPU/GPU pair; it must wait for capacity, preserve task IDs when splitting
arrays, and record every accepted chunk without changing another campaign's jobs.

Runtime recovery job **62470983** completed, 0:0, **9m37s**. Its 7,233,483,401-byte
archive and all 167 input hashes were verified; SHA256
`c0b54ede3ce4f77c9cc47cd28c926c211df55a4fd0d8cb0f3c2718dc1c22a727`.
Only then was the verified 12,000,459,103-byte temporary build workspace removed.
The image and recovery bundle remain on PROJECT; DATA legacy runtime remains rollback
until GPU acceptance. The library pins and training implementation are unchanged.

### 29-10-2026 — container submission probes before clean validation

| Job | Allocation | Result |
|---|---|---|
| 62470743 | wICE batch, 1 CPU, deployment controller | FAILED, 1:0, 00:01:38; stopped before any Exp0/production job submission |
| 62470775 | wICE interactive, 1 CPU, 2G, five-minute limit | COMPLETED; host sbatch test-only and actual compute config path verified |
| 62470810 / 62470817 | wICE controller / interactive CPU child | Deliberately cancelled during runtime acceptance to fix the missing site credit client; no GPU launched |
| 62470838 / 62470841 | wICE controller / interactive CPU child | CPU startup FAILED (30 s): inherited parent container mount list; controller stopped before GPU |
| 62470856 / 62470861 | wICE controller / interactive CPU | Native runtime/credit/preflight passed; unit fixture leaked PROJECT-root alias; GPU remained blocked |
| 62470900 / 62470908 | wICE controller / interactive CPU | 954 tests passed; notebook path fixture used legacy alias; deliberately stopped after diagnosis; no GPU |
| 62470983 | wICE batch, 2 CPUs, 8G, one hour | COMPLETED, 0:0, 9m37s; recovery archive and all 167 content hashes verified |

The controller used source `cbd755ed9185403865b68a3d65f74e3c3045abb950c0a9841f476b7c3241d35f`
and image SHA256 `83f8145eafbbbcad0f25a5d109b0329aec12a159e5bc8ab9fc94a9f1e3ed0585`.
A login-node dry run revealed the missing serializer dependency; the compute-node failure
also showed that /run/slurm/conf resolves into /var/spool/slurmd/conf-cache. The wrapper now
binds that real directory, sanitizes inherited configuration and includes serializer/data-parser
libraries. Mandatory dry-run submission checks were added before long tests in both phases.
No trained weights, optimizer, prior or benchmark protocol changed. TFM pin
`26156f36eb7676e8d9050547817346db548b0e85`; VSC pin
`0ed4ffa89d312b9bb6411681b6a14f81672439f3`.


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
- **Full-table benchmark probes** four PD tables (HackerEarth, PropPD1, PropPD2, MicroFinance), three LGD tables (PropLGD1, PropLGD2, PropLGD5), five classification OOD tables and three regression OOD tables. All three diagnostic checkpoints and four references complete credit/OOD evaluation. Credit logs show 245 successful model/dataset/fold cells, with all folds 0–4. Largest PD table retains 532,428 rows; widest retains 158,700 rows and 2,986 features. Context caps remain `None`.
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

Restricted-source checks verified outcome-construction and workout-field exclusions. Matched 5,671
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
  (hackerearth, PropPD1, PropPD2, algorithmwatch) and 15/model on three
  LGD tables (PropLGD1, PropLGD2, PropLGD5). The 2,986-feature PD route passed.
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
hackerearth, PropPD1 and PropPD2, one for algorithmwatch. There are no new
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
  hackerearth (532,428 rows / 30 eligible features), PropPD1, and PropPD2. The previous host-memory failure did not recur.
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
  30 eligible features, AUC .8445/.8447/.8475/.8477/.8464; and `PropPD1`:
  AUC .6403/.6423/.6454/.6488/.6346.
  All training-fold rows were used. Removed columns were outcome/future-information exclusions.
- The next table, `PropPD2`, completed its first inner PD fit (18.2s). Last logged activity was
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
`pred_nonfinite_frac=0`, including all 135 rows on `PropLGD4`, the dataset affected by
historical monitoring NaNs. Mixed-credit and control remain close; pure-credit LGD remains
poor. Those old monitoring NaNs are not a reason to retrain or discard checkpoints.

### Limitations and chronology correction

- Context mismatch is confirmed: our and released TabICL used at most 1,024 rows, TabPFN up
  to 10,000, and CatBoost/linear the full pool. The outer `n_train_used` and requested
  `context_cap` columns alone do not describe the actual TabICL context; its inner cap is
  recorded separately. A fair cross-model comparison requires re-evaluation, not retraining.
- PropLGD3's 147 rows have no boundary MAE because the old tolerance counted its near-zero/one
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
  still contain 605 non-finite PropLGD4 measurements in 30 arms; these are monitoring records,
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
| LGD development R², `PropLGD4` (finite in 3 / 6 / 6 of 9 / 18 / 18 arms) | −0.03 | 0.19 | 0.02 |
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
  and `PropLGD4` R² is non-finite in 30 of them (seeds 0 and 2, one non-finite prediction each),
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
   context, so it had been feeding the full PropLGD1 training split to a model trained on at most 1,024 rows.
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
| `PropLGD3` | 1.98 | finite | finite |
| `PropLGD1` | 820 | finite | finite |
| `PropLGD2` | **2.4e6** | NaN | **finite** |
| `PropLGD5` | **8.1e7** | NaN | **finite** |
| `PropLGD4` | **9.6e8** | NaN | **finite** |

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
- **The NaN survives `max_features: 100`.** `PropLGD5` still returns
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
1. **The LGD model emits 100 % NaN predictions on 3 of 4 real datasets** (`PropLGD5`,
   `PropLGD4`, `PropLGD2`) and on 5 out-of-domain regression datasets. `PropLGD3` — the only one
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
