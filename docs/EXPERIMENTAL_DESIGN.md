# Experimental design

What the three experiments ask, why the design can fail, and the premises it rests on. This
describes what is **built** (the `config/Exp{1,2,3}_*.yaml` sweeps); the prior's mechanics are in
[PRIORS.md](PRIORS.md), every knob's value in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md), and what
each run actually did in [RUNS.md](RUNS.md). Original source-audit pin `de324efcca55be392feb0c01830b3b6ce364587f`; the 24-10-2026 Experiment 2 audit uses TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`.

## Contents

1. [The claim, and how it fails](#1-the-claim-and-how-it-fails)
2. [Verified premises — and two corrections](#2-verified-premises)
3. [The experiments](#3-the-experiments)
4. [Matched compute, seeds, and the two phases](#4-matched-compute-seeds-and-the-two-phases)
5. [Evaluation](#5-evaluation)
6. [Risks](#6-risks)
7. [Open decisions](#7-open-decisions)

---

## 1. The claim, and how it fails

> At fixed architecture and training budget, does the complete credit-targeted prior package
> change performance on this credit benchmark suite, and what happens to native OOD performance?

This is a package comparison. It does not identify credit structure separately from task format,
encoding, filtering or difficulty. Selective transfer and double dissociation are stronger,
future hypotheses that require additional controls, rather than claims established by this grid.

**The claim is wrong if:**

- the credit prior (`credit_fraction > 0`) does **not** beat the control (`credit_fraction = 0`) on
  the matching track — the intervention does not transfer;
- fine-tuning (Exp2) buys credit gains only by **eroding out-of-domain generality** — measured
  directly, not assumed.

Each is a publishable negative. The design is built to learn which is true, not so that credit wins.
Temporal deployment and crossed-task selectivity are not tested by the current five-fold protocol.

**One honest limitation.** A generic-realism control arm (O'Prior-style: realism *without* credit
targeting) was designed but **not built** — see [§3](#3-the-experiments). Without it we cannot
fully separate "credit-specific structure" from "any added realism". The proposed crossed
mechanism ablation ([§5.2](#52-the-double-dissociation)) is also **not implemented** by the current
four-mixture grid. Gains on PD and LGD alone do not establish a double dissociation or rule out
generic realism, difficulty or architecture interactions.

## 2. Verified premises

Everything here was read from source, not assumed. **Two corrections matter — the original framing
overstated two claims a code-reading reviewer would catch, and the honest versions are load-bearing.**

**2.1 The regression target *is* standard-scaled (verified).** `GraphSCM.__call__` clamps at ±4σ
(`outlier_removing`) then `standard_scaling`; `TabICLRegressor` mirrors this at inference and
inverse-transforms the predicted quantiles. Prior and inference agree.

**2.2 CORRECTION — bounded/bimodal targets are a *frequency* gap, not a structural absence.**
Standard-scaling is affine, therefore shape-preserving: a U-shaped target with two atoms arrives as a
U-shaped target with two atoms on a rescaled axis. Standardisation destroys the **[0,1] support**,
not the bimodality — and atom-producing mechanisms already exist (`outlier_removing` clamps → point
masses; piecewise-constant function families; `rand_kumaraswamy_act` maps to exactly [0,1] with
p≈0.5). The defensible claim: nothing in the prior *targets* boundary mass in real LGD's range, and
nothing aligns the output support with [0,1]. That is weaker than "absent" and, crucially,
**measurable** — the `0.3` prior-visualisation notebook measures it.

**2.3 CORRECTION — TabICL's prior is *not* necessarily balanced.** The active `GraphSCM`
uses `CategoricalConverter`, not the older `SCMPrior`'s `MulticlassAssigner`. Its realised
class-count and minority-frequency distributions must be measured directly, not inferred from
the older rank-binarisation path. Imbalance is uncontrolled rather than structurally absent;
we reshape the task distribution. O'Prior instead filters collapsed or severely imbalanced support.

**2.4 The head can already represent atoms.** The regressor is a **999-quantile** pinball head, which
represents a point mass exactly (an atom at 0 = quantile levels 0.001…mass all mapping to 0). So
representability is *not* the bottleneck; the real head-side issues are point-prediction extraction
from a bimodal predictive (median/mean land in the sparse interior) and the lack of a [0,1] support
constraint (clipping is a free baseline, not a contribution). This is why a ZOIB/mixture head is held
in reserve rather than built.

**2.5 The predictability filter — characterised.** Three filters, verified in code and against the
TabICLv2 paper: a **statistical** one (ExtraTrees OOB bootstrap test, reject if `pval ≥ 0.05`), a
**triviality** one (reject too-easy), and a **structural** one (reject DAGs where x, y share no
ancestors). Paper rate: "roughly 35% classification and 25% regression datasets are filtered". Two
nuances shape Exp1: it is a *significance* test, not an R² floor (it removes *no-signal* DGPs more
than *low-signal* ones), and the authors show filtering *improves* convergence — so the `banded`
variant contradicts a published result *with evidence behind it*, and must beat that argument.

**2.6 Our data does not match a single premise.** Measured from `data/raw/`: Freddie LGD is genuinely
U-shaped (11.4% at 0, 8.1% at 1); LendingClub LGD is unimodal, left-skewed, peaks at 0.80–0.85; AXA
has a substantial atom near zero (within the documented boundary tolerance). PD base rates span 6.7% (GMSC) to 40% (myhom). **Consequence:** the prior must be
a *family* over boundary mass and base rate, not a single shape — else it overfits one dataset.

## 3. The experiments

The 26-10-2026 implementation replaces the proposed long from-scratch Experiment 3 with
retention ablations. Experiment 1's resolved recipes, task budgets and array identities remain
unchanged, including the appended 25% mixture. This change does not require retraining Exp1.

| Experiment | Question | Production training runs |
|---|---|---|
| 0 | Do the current code, inputs, native model routes and hardware paths pass validation? | Diagnostic CPU job, then resumable GPU campaign; no research arms |
| 1 | Does the credit prior package help a scratch TabICL at a matched budget? | 54 = (4 mixtures + 5 component removals) × 3 seeds × 2 tasks |
| 2 | How does credit-biased continuation affect each released backbone? | 48 main = 2 backbones × 4 mixtures × 1 selected rate × 3 seeds × 2 tasks; 16 LR pilots |
| 3 | Which update constraints trade credit adaptation against OOD retention? | 48 new main runs + 12 reused Exp2 baselines; 36 separate pilots |

### 3.1 Experiment 1: preserved evidence

Four credit shares (0%, 25%, 50%, 100%) and five component removals at 50%, with three seeds
per PD/LGD task, use 12,500 updates of 64 tables of 1,024 rows. The original campaign contained
48 runs; six quarter-mixture runs were appended. Existing completed-run provenance is accepted
only for explicitly reviewed source/package/recipe combinations. Original historical completion
permission does not authorize arbitrary checkpoint resume. Unchanged Exp1 runs from `19a76cc`
may resume across this patch under an additional exact-source and recipe check.

### 3.2 Experiment 2: full continued pretraining

Every main run starts from released weights, updates the full model, and uses a fixed 2,500-update
budget. Four mixtures share one rate per backbone/task. Neither Exp1's winning mixture nor OOD
scores select the grid. The fixed primary endpoint is update 2,500. Updates
125, 250, 500, 1,000 and 2,500 are permanently retained and benchmarked.
Released weights are a shared step-zero reference. Earlier points describe trajectories, not
post-hoc alternative headline checkpoints.

| Setting | TabICLv2 | TabPFN-3 |
|---|---|---|
| Tables per update | 64, one-table gradient accumulation | 16, one preprocessed task at a time |
| Rows per synthetic table | 400–60,000, log-uniform | 1,024 |
| Context share | 79–81% | 30–90% |
| Pilot candidate peak rates; one is frozen | 2e-5 or 1e-5 | 1e-5 or 1e-6 |
| Optimizer | Native Muon configuration (all trainable parameters) | AdamW |
| Weight decay | 0.01 | 0.01 |
| Warm-up | 1% (25 updates) | 10% (250 updates) |
| Gradient norm limit | 1 | 1 |
| Precision | FP32, explicit TF32/efficient SDPA policy | FP32 weights, FP16 autocast and GradScaler |
| Objective | Native classification / 999-quantile regression | Native differentiable classification / distributional regression |

Synthetic width remains at most 100 with the upstream row-dependent ceiling; this is not a
benchmark feature limit. Every eligible benchmark row and feature is retained, with five common
folds, on 21 credit tables and 25 native OOD datasets per task. Small-context development/OOD
monitoring is labelled separately and never substitutes for the full-table benchmark.

**What is upstream and what is a research choice.** At TFM pin
`702d9ab0b66f6639ba8d443ff9c4a91231964465`, `repositories/TabICL.txt`, symbols
`train_v2_clf_stage3.sh`, `train_v2_reg_stage3.sh`, `Trainer.configure_optimizer`, and the task sampler document
stage 3's 10k updates, 64-table batch, long-table law, Muon, 2e-5 rate, 0.01 decay and 1% warm-up.
The lower rate is our sensitivity comparison. The preceding upstream stages use 500k and 40k
updates; this is continuation of a released model, not a recreation of all upstream pretraining.

TabPFN's public differentiable fine-tuning path (`repositories/TabPFN .txt`,
`FinetunedTabPFNClassifier`, `FinetunedTabPFNRegressor`, `fit_from_preprocessed`) processes
single-table context/query episodes. We reuse that path across fresh synthetic tables and average
the task losses before one optimizer update. AdamW, 1e-5 and 0.01 decay follow the public
fine-tuning interface. The 16-table batch and 1,024-row synthetic tasks are an explicit adaptation
recipe, informed by earlier TabPFN continuation work, not disclosed TabPFN-3 pretraining defaults.
The 1e-6 arm tests a more conservative update. TabPFN-3's original pretraining corpus and full
recipe are not public. Its 0% control therefore uses the generic TabICL prior, not a claimed
reconstruction of a native TabPFN-3 prior. No setting is claimed optimal in advance.

The synthetic task stream, warm-up fraction, precision and table geometry are matched across
mixtures within each backbone. Across backbones the adaptation budgets differ; the primary
contrast is the prior effect within each backbone, not a compute-controlled architecture contest.

**Learning-rate pilot (Budget A, approved 29-10-2026).** Sixteen separate runs cross both
backbones/tasks, both candidate rates and the 0%/50% mixtures at seed 990. Each stops at 500
updates on the **2,500-update schedule prefix**. Only development datasets are benchmarked,
with all rows/features and five folds; no holdout, OOD or monitoring-based selection. Average
folds within each dataset, then datasets and the two mixtures equally. Maximize ROC-AUC for PD
and R² for LGD; exact ties choose the smaller rate. Freeze one rate per backbone/task, shared
by every main mixture. Main seeds 0/1/2 restart from released weights. Immutable receipts bind
code, packages, benchmark inputs, all candidate scores and CSV hashes. No YAML edit is required.

Development performance is selection-exposed and reported descriptively. Primary comparisons
use designated holdout datasets and native OOD, excluded from the pilot. Five-fold CV on the
selection datasets does not make their selected scores unbiased. Previously inspected holdout
results remain disclosed; this is a controlled benchmark, not untouched external confirmation.

The 2,500-update endpoint is a compute-budget choice informed by measured costs, not a literature
claim of optimal convergence. Main storage is 240 retained snapshots, approximately 107 GB,
plus about 7.2 GB for the 16 pilots, rolling saves and benchmark output. Measured/provisional
training extrapolation is about 50–53M VSC credits including pilots at the quoted 30,600/hour;
benchmarking and fresh validation are additional and must be budgeted separately.

### 3.3 Experiment 3: constrained adaptation at a fixed 50% mixture

The primary experiment uses the same 2,500-update geometry and three paired seeds as Exp2.
The mixture is fixed in advance at 50%; it is not chosen from Exp1 holdout results.
Reuse the rate already frozen by the Exp2 LR pilot; do not select it again on main results.
Reuse the 12 main 50% full-CPT-with-decay runs as baselines.

| Additional method | Trainable parameters | Settings |
|---|---|---|
| No decay | Whole model | Selected full-model LR; weight decay 0 |
| L2-SP | Whole model | Selected LR; weight decay 0; alpha in {0.0003, 0.003, 0.03} |
| DiagScale | Query-scaling deltas only | AdamW; LR in {0.001, 0.01, 0.03}; adapter decay 0.01 |
| Q/V LoRA | Rank-16 Q/V factors only | Alpha 16, no dropout; AdamW; LR in {1e-5, 1e-4, 1e-3}; adapter decay 0.01 |

The three tuned methods each receive three 500-update pilots at seed 991 per backbone/task:
36 pilots total. Each pilot follows the first 500 updates of the main 2,500-update schedule,
not a compressed cosine cycle. Only complete development five-fold results choose a candidate.
Pilots neither benchmark holdout/OOD nor monitor them. All candidate scores are recorded.
The selected setting is then run from the released weights at seeds 0/1/2 for 2,500 updates;
pilots are never counted as independent main-seed observations. No-decay needs no pilot.

L2-SP adds alpha/2 times the squared displacement from the **released** parameters. The anchor
is reconstructed from the released checkpoint before resume; it never moves to the resumed
parameters. In TabPFN, its gradient is added once after unscaling/replaying the data gradient and
before clipping. Ordinary decay instead shrinks the current parameters toward zero. Both are
legitimate regularizers but express different assumptions; Exp3 measures that distinction.

DiagScale starts with identity query multipliers after each native query-scaling operation,
including TabPFN's `SoftmaxScalingMLP`; all original weights are frozen. Q/V LoRA starts with a
zero residual, preserves packed TabICL K weights exactly, and adapts native TabPFN Q/V projections.
Adapter checkpoints retain resumable factors; LoRA also exports merged native weights for normal
inference. DiagScale is installed before inference caches are built. Adapter optimizers use 10%
warm-up and clipping at 1. Adapter-only decay does not decay the frozen backbone.

These are adaptations of literature mechanisms, not claims that the papers measured this exact
multi-table CPT grid. Source anchors: `repositories/Test-Time Compute for Tabular Foundation Models.txt`,
`inject_diagscale`, and the associated Ning et al. paper; Real-TabPFN's L2-SP continuation;
Hu et al. LoRA. The DiagScale/LoRA paper uses single-table adaptation, and its all-linear LoRA
scope differs from this explicitly named Q/V variant. Pilot grids are bounded sensitivity checks,
not exhaustive optimization. Other retention techniques are outside this registered experiment.

`python -m src.experiments.selection --stage learning-rate` freezes the first receipt;
`--stage adapters` freezes the pilot selection. Both bind complete benchmark receipts, CSV hashes
and current source. They never rewrite YAML or submit jobs. A changed or missing receipt blocks
Exp3. Holdout/OOD are reporting outcomes only. The historical inspection of designated holdout
data is still disclosed; development-only selection now does not erase that exposure.

### 3.4 Experiment 0: implementation and capacity gates

The CPU job covers the portable suite, all dataset loads/splits, configurations, task replay,
selection isolation, checkpoint retention, receipts, output paths and notebook contracts.
The GPU campaign covers native scratch/full-CPT routes, optimizer stop/resume, model equivalence,
16 released-weight method/task/backbone round trips, 34 long-context stages (including 60k-row
LoRA and DiagScale), wide-feature inference and representative full-table five-fold benchmarks.
Passing this gate verifies those paths on that allocation; it cannot prove every future stochastic
batch or the entire production benchmark will succeed, nor establish a treatment's efficacy.

A resubmission continues an unchanged campaign with verified artifacts. When code/packages/inputs
or the CPU receipt change, `--resume-if-current` starts a fresh isolated campaign. An actual failed
current campaign is not silently skipped. Do not edit/pull during a campaign. Exp1 outputs are not
cleaned by this process.

## 4. Matched compute, seeds, and the two phases

**Matched within each backbone, not across papers.** Mixture arms of the same backbone receive
the same optimizer steps, task count and planned row/feature/context distribution. In Exp2,
TabICLv2 receives 160,000 tasks (batch 64, stage-3 geometry) and TabPFN-3 receives 40,000 tasks
(batch 16, 1,024 rows), each over 2,500 updates. Common random numbers pair shape draws across
mixtures within a backbone. Each task contributes equally, so longer tables do not implicitly
receive greater loss weight. Cross-backbone contrasts include different adaptation budgets;
the primary prior effect is the contrast between mixtures within each backbone. This is **not**
a match to O'Prior's protocol, nor a wall-clock match. **Two traps:** turning the filter *off*
makes generation cheaper, so wall-clock and credits are not comparable across filter arms (match on
steps/datasets, and report the rejection rate per arm); and TabICLv2's own ablations show a strong
prior×architecture interaction, so a prior change may need retuning — **log every arm's loss curve
and treat divergence as a reportable outcome, never tune it away silently.**
The current package comparison already filters native-prior tables but not credit tables;
generation cost and accepted-task difficulty can therefore differ across mixtures. The updated
long-context law preserves that declared comparison instead of changing filtering mid-study.

**Seeds.** 3 per main arm in Exp1, Exp2 and Exp3 (paired: a seed gives both arms the same base tasks
and initial weights within a backbone). Report mean ± spread; gaps smaller than the seed spread
indicate descriptive instability, not a significance test or an alternative selection rule.
Exp3 breaks exact development-score ties by the smaller numeric candidate, with no holdout tie-break.

**Two phases, and the order is a fact about the data.** Phase 1 (`pretrain_{pd,lgd}.slurm`) trains one
checkpoint per saved step per arm; phase 2 (`benchmark.slurm`) scores every saved checkpoint — one array
slot each, final checkpoints first — **plus a shared reference column** (released TabICLv2, TabPFN-3,
CatBoost, linear), one slot per model after them (`benchmark_status.slot_for`). Phase 2 cannot start
until phase 1 has written the checkpoints. The reference sits **inside the same array** so it is
scored by the same code, the same folds and the same protocol as our arms — every comparison
this project got wrong, it got wrong by scoring the two sides through different paths. Evaluation is
*not* inside the training job for the same reason: an arm would otherwise be benchmarked by whatever
the code looked like the hour it finished.

## 5. Evaluation

The designated split lives in every config's `eval:` block; selection uses `select_on: dev`.
The September results across these datasets have already been inspected. The designated holdout
is therefore not an untouched confirmatory set for subsequent design changes. Freeze this revised
protocol, report that history, and obtain an independent source/time cohort for confirmatory claims.

**5.1 What is compared, and on what.** Every checkpoint against the shared reference, on the real
credit datasets and the out-of-domain suites. The split is fixed:

| track | development (prior choice + iteration) | designated holdout (previously inspected) |
|---|---|---|
| **PD** (14) | gmsc, lendingclub, hackerearth, taiwan_creditcard, german, myhom | vehicle_loan, cobranded, bank_status, thomas, loan_default, home_credit, hmeq, algorithmwatch |
| **LGD** (7) | lgd_lendingclub, base_model, base_modelisation, heloc | loss2, axa, lgd_freddie |

`hackerearth` and `base_modelisation` moved to development on 30-09-2026: the first holds 5,696 of
`lendingclub`'s 9,578 loans, the second shares defaults with `base_model` (audit, §7 item 5), and a pair
split across the line would let a choice made on development be tested on the same loans.

The native `hackerearth.loan_status` label is retained. The 14-10-2026 review of the
[challenge repository](https://github.com/HackerEarth-Challenges/machine-learning-challenge-1)
did not establish whether 1 denotes default or repayment; correlations with recovery fields
do not settle that definition. Report it as the supplied binary lending benchmark target,
without claiming an independently verified default-event definition or reversing its labels.

**Why a holdout at all, when every model trains on synthetic data?** Because choices are still made
by looking at real-data results — which learning rate or adapter strength Exp3 uses. A choice
made on a set of datasets is fitted to them: the winning configuration's score there is biased upward
(the winner's curse), exactly as a hyperparameter chosen on test data is. Training on synthetic data
removes the leak through the weights, not the leak through the choice. Exp2 selects one rate per backbone/task on separate development-only pilots and reports
all four mixtures. Exp3 fixes the 50% mixture and reuses that frozen rate,
then selects method strengths on separate development-only pilots. Every Exp1 and Exp2 result
is reported on all eligible datasets, with the designated holdout shown separately. This separation
does not erase the effects of earlier inspection and must not be described as an unbiased fresh test.

Five-fold CV estimates each fixed model's performance; selecting a mixture using those same
assessment scores and then reporting only its winning score would still induce selection bias.
Exp2 therefore retains every predeclared mixture and the final checkpoint, with paired effects
against the matching continued-pretraining control and released model reported separately.
Only Exp3's rates and method strengths are selected, using development datasets and the primary metric alone.
OOD retention is an outcome to report, not an extra rule for re-ranking those choices. Existing
holdout exposure remains disclosed; this separation cannot make past inspection disappear.
These distinctions follow [Cawley & Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html)
and [Varma & Simon (2006)](https://doi.org/10.1186/1471-2105-7-91).

**The protocol** (`src/eval/protocol.py`, version 9 since 21-10-2026) is the same for every model —
our checkpoints, released TabICLv2, TabPFN-3, CatBoost, the linear floor — and for the out-of-domain
suites: **5-fold cross-validation** (stratified for PD, one fixed fold assignment); **the whole training
pool is the context** — no row cap or context selection; for PD, **the decision threshold maximises F1 on a validation split** (20 % of the
training pool), after which the model is refitted on the whole pool and the test fold is scored at that
threshold. Library defaults apply except the explicit full-row/full-feature overrides below.

**The base data caches** follow TabPFNCredit's per-dataset recipes (the sibling project
`1. TabPFN/TabPFNCredit`; `tests/test_data_identity.py` compares the processed arrays).
Since 12-10-2026, `src/eval/feature_policy.py` then excludes known outcome fields for the approved
prediction times: **PD at application, LGD at default**. It records excluded names and remaps
categorical columns identically for all models and monitoring. Raw/cache files remain unchanged.
Remaining anonymised fields and cohort observation dates still need source-dictionary validation;
behavioural card datasets cannot automatically be described as application underwriting tasks.
After those exclusions, protocol 9 retains **every eligible feature and every fitted row**, without
univariate selection or near-constant filtering. Validation/test numeric values are clipped to
the fitted rows' 0.1st and 99.9th percentiles, learned separately inside each fit. TabPFN retains
its released preprocessing transforms but receives a per-estimator budget large enough for all
input columns and their preprocessing expansion; a runtime check rejects internal feature
subsampling. All other models receive the same full input matrices. Result receipts require
matching input/preprocessing/model feature counts as well as complete row accounting.
The 100-column budget of protocols 4–6 was a project choice, not an architectural constraint;
it is removed from the main comparison. The synthetic training distribution remains 1–100
columns, so wide-table extrapolation is shared across prior arms. Source audit: `TabICL` and
`TabPFNEnsemblePreprocessor`/`PreprocessorConfig` in the library repository dumps, pin
`de324efcca55be392feb0c01830b3b6ce364587f`; real released PD/LGD models also passed local
601-feature prediction and regression-quantile probes, and both released classifiers passed
2,986-feature probes matching the widest local table's width. Exp0 repeats wide probes on CUDA. Every
saved permanent checkpoint is scored: Exp1 keeps its existing 2,500-step cadence; Exp2/Exp3 keep
125, 250, 500, 1,000 and 2,500. Both kinds of pilots benchmark only 500 on
development. Exp1 and Exp2/Exp3's TabPFN-3 arms train on 1,024-row tables;
Exp2/Exp3's TabICLv2 arms use 400–60,000 rows. All main runs are scored on full contexts of up to ~426,000 rows.
Length exposure is matched across mixtures within each backbone, but it differs across experiments
and backbones. Released TabICLv2 also completed its earlier training stages; extending our Exp2
row distribution does not make its cumulative training budget equal to theirs.
Matching table lengths across arms does not rule out a prior-by-length or prior-by-width
interaction. The estimand is performance in the requested full-table setting; this project does
not claim a length-invariant prior effect and does not replace the primary benchmark with a cap.

**Task-format limitation.** PD credit tasks are binary; the untouched upstream control draws
2–10 classes. Increasing the credit share therefore also increases the binary-task share.
A PD improvement cannot by itself be attributed to credit mechanisms rather than more binary
practice. Native binary and multiclass OOD results should be examined separately as well as in
the aggregate. A binary-only upstream control would be an additional attribution experiment,
not a repair needed to execute the currently approved package comparison.

**Inference scope.** Dataset-paired sign tests use datasets, not overlapping CV folds, as units.
Only three LGD datasets are in the holdout: even unanimous non-tied differences have a minimum
two-sided sign-test p-value of 0.25. LGD holdout and component results are consequently descriptive
at this sample size; effect sizes, all per-dataset scores and uncertainty must be reported.
Moving inspected development tasks into the confirmatory test or counting folds as independent
replicates would not solve that limitation. ROC-AUC/R² remain the predeclared selection metrics;
log-loss/Brier and CRPS/boundary calibration address different secondary questions, not post-hoc
replacement primary endpoints.

**Metrics** (all computed in one run). PD: ROC-AUC, PR-AUC, Brier, log-loss, KS, calibration slope, and
F1, precision, recall, MCC and balanced accuracy at the validation-tuned threshold —
**never accuracy alone** (at a 7% base rate "never defaults" already scores 0.93), and ECE with MCE
because the sparse high-score tail is where the defaults are. LGD: pinball, CRPS, interval coverage,
**boundary-mass calibration** (predicted vs observed P(y=0), P(y=1) — the metric that most directly
tests the hypothesis, which no cited paper reports), with R²/RMSE alongside and the decoding rule
stated (it changes the number materially). Success criteria are set to the LGD literature (published
R² ≈ 0.04–0.15 linear, 0.10–0.25 beta, 0.20–0.43 tree ensembles), not classification expectations.

**5.2 The double dissociation — proposed, not yet implemented.**

| mechanism enabled | predicted effect on **LGD** | predicted effect on **PD** |
|---|---|---|
| bounded target + boundary mass | **improves** | ~neutral |
| controlled base rate + Vasicek defaults | ~neutral | **improves** |

Testing this table requires separate mechanism toggles and a valid crossed-target construction.
The main Exp1/Exp2 grids change only the mixture fraction within each track. Exp1's separate
component removals test selected contributions, but do not supply this crossed-target design.
Add controlled target-only, feature-only and generic-realism comparisons before attributing
gains specifically to individual credit mechanisms; measure difficulty as well. Until then, a
positive result supports only the full prior package under the tested budget and architecture.

**5.3 Out-of-domain retention is a first-class axis, not an appendix.** After training on a
credit-targeted prior, score general/out-of-domain benchmarks to measure whether in-domain gains were
bought at out-of-domain cost. Report dataset-paired score changes; a specialization trade-off
is a hypothesis to test, not a conclusion borrowed from time-series adaptation. **If domain-targeting costs general performance, that is
the finding** — Exp2's `2.1`/`2.2` notebooks plot the credit-vs-out-of-domain curve directly.

**5.4 Temporal splits.** Distribution-shift work motivates testing time and borrower/source groups,
rather than generalising an i.i.d. result to deployment. Grouped five-fold splits are implemented
for HELOC loans, base_model obligors, base_modelisation clients and bank_status borrowers.
Temporal splits remain unimplemented; requesting temporal splitting raises an error. Feasibility depends on identifying
observation dates, label maturity and repeated borrowers in each source — see [§7](#7-open-decisions).
The presence of a date column alone does not establish an observation-time and mature-label protocol.

**5.5 Dataset semantics and the scope of the comparison.** Retain the existing benchmark inventory;
unusual correlations or high predictability alone are not grounds for dropping a dataset. The
source-defined `hackerearth` label has a repayment-oriented interpretation on matched Lending Club
loans and incomplete maturity information; do not call its positive label a verified default event
or silently flip it. `lgd_lendingclub`'s target is strongly explained by amortisation; without a
source definition it is a reported bounded-target benchmark, not independent evidence about realised
workout loss. These limitations restrict domain/calibration claims, not the execution of a matched
synthetic-pretraining comparison. No target or dataset is rebuilt by this audit.

Known target construction inputs are a different issue. On 21-10 the raw `loss2` target was
reconstructed as `clip((_EAD - _Proceeds + _ELAO) / _EAD, 0, 1)` for all 4,638 rows with finite terms.
Protocol 9 excludes `_ELAO` too. For `base_modelisation`, `an` exactly copies excluded
`annee_in_dft` (594/594); `EAD_BALOISE_REELLE` copies excluded `denominateur` (585/594); and
`Perim_poids`/`ponderation_11` encode the already excluded workout-group flags. Those aliases,
including categorical copies, are excluded consistently. Raw/cache tables and benchmark membership
are preserved. This closes known exclusions; it does not certify every anonymised field's timing.

## 6. Risks

| risk | why it is real | mitigation |
|---|---|---|
| our training loop is subtly wrong | it is ours; a bug invalidates all arms equally and invisibly | reproduce the control in the expected range; unit-test the pinball loss; verify a fixed seed reproduces exactly (all in `tests/`) |
| no generic-realism or crossed-mechanism arms | can't separate credit mechanisms from generic realism/difficulty | implement controlled ablations before making mechanism-specific claims; current mixture grid is insufficient |
| prior×architecture interaction | TabICLv2 ablations show divergence on the old prior | log all loss curves; report divergence as an outcome |
| nano→frontier gap | untested at scale (O'Prior's own caveat) | state as a limitation; scope any frontier confirmation separately |
| credit is not distinctive | if the default prior already covers credit's regime the premise is weak | the `0.2`/`0.3` notebooks measure this before credits are spent |
| meta-overfitting | 21 datasets, many iterations, historical holdout results already inspected | dev-only selection and frozen protocol now; independent source/time cohort for confirmatory claims |
| i.i.d.-only result | Purucker 2026 | temporal splits where the data allows ([§7](#7-open-decisions)) |
| outcome information in the features | the 29-09-2026 audit found outcome-construction and workout fields ([§7](#7-open-decisions) item 5) | known fields excluded under protocol 5; dictionaries needed for remaining fields. Giving every model leaked inputs does not remove the bias |
| the same loans in dev and holdout | 5,696 of `lendingclub`'s 9,578 loans are in `hackerearth`; `base_modelisation` shares defaults with `base_model` | fixed 30-09-2026: both pairs are on the development side ([§5](#5-evaluation)) |

## 7. Open decisions

1. **Temporal-split protocol** — which datasets carry a usable date/group column; needs a per-dataset
   audit before the protocol is fixed ([§5.4](#54-temporal-splits)).
2. **The generic-realism and mechanism ablations** — construct explicit controls at matched budgets
   after source review; the existing two-track mixture comparison is not a substitute.
3. **ZOIB/mixture LGD head** — held in reserve behind decoding + support constraints
   ([§2.4](#2-verified-premises)); revisit only if those are insufficient.
4. **Retention selection** — resolved 26-10-2026: fixed 50% mixture, two full-model rates
   tested in Exp2, and development-only candidate pilots in Exp3. The long scratch run is dropped.
5. **Outcome information in the evaluation data (audit of 29-09-2026; decision revised 12-10-2026).**
   The following historical audit numbers describe that cache/split, not the corrected benchmark. A column
   scan (ExtraTrees out-of-bag R² with and without the suspects, processed data) found:
   - `base_modelisation` (LGD holdout): `numerateur / denominateur` reproduces `lgd_defaut` to 0.001
     on 57 % of rows — the target is in the features. Also `DT_DECHEANCE_TERME` (acceleration date),
     `C_flag_crc_clos_f`, `annee_in_dft`. R² 0.48 → 0.39 without them.
   - `loss2` (LGD holdout): `REO_Appraisal_Amount`, `UPB_At_Resolution`, `_Mirecovery`,
     `_Miclaimbal`, `Unpaid_Interest` — values recorded at or after resolution. R² 0.63 → 0.43.
   - `base_model` (LGD dev): `flag_clos`, `duree_defaut`, `DFLT_duree` (and its binned copies),
     `DFLT_RationalEndDefault`, `DFLT_RecoveryApproach`, `FCLT_UnpaidInterest`. R² 0.48 → 0.45.
   - `lgd_lendingclub` (LGD dev): `months_since_origination` alone gives R² 0.58, everything else
     0.04 — the target appears to be the loss on the original amount, which the loan's age almost
     fixes. Known at default, so not a leak by timing, but the task is mostly amortisation; decide
     from the target's definition whether to keep it, re-define it or move it out of dev.
   - `hackerearth` (PD holdout): `recoveries`, `collection_recovery_fee`, `total_rec_int`,
     `total_rec_late_fee`, `last_week_pay` describe the loan after origination (weak on their own,
     univariate AUC ≤ 0.65, but not application data).
   - Dev/holdout overlap: `lendingclub` (PD dev) and `hackerearth` (PD holdout) share 5,696 loans
     (59.5 % of the dev set; exact on rate, DTI, revolving balance and utilisation, 99 % of those on
     income too); `base_model` (LGD dev) and `base_modelisation` (LGD holdout) share at least 16 % of
     the holdout's interior LGD values exactly, so likely the same defaults from two extracts.
   The 30-09 decision to keep all fields was unsound: equal access to leakage does not make the
   comparison valid. On 12-10 Andreas chose application-time PD/default-time LGD. Known post-outcome
   fields and target components are now excluded, including categorical copies. Some workout
   amounts are conservatively excluded without a verified default-time snapshot. Acceleration date
   (`DT_DECHEANCE_TERME`) is now conservatively excluded: many source values postdate default.
   Future `PY1_*` macro fields, workout flags and the client identifier are excluded too. The current `base_modelisation` cache reproduces the target from numerator/denominator
   within 0.001 on 49.2% of 594 rows, not the historical 57%. Source dictionaries and LGD target
   definitions remain necessary; weak univariate prediction does not certify absence of leakage.
6. **Exp2's reported checkpoint** — resolved 30-09-2026: with no search there is no chosen checkpoint;
   the final one (fixed at 2,500 steps after the 29-10 budget revision) is the result, the others a learning curve.
   The 26-10 review does not establish 10,000 as optimal; freeze any pilot-informed replacement
   budget for all mixtures of a backbone before the production comparison.
   `eval.headline_step`, if a config sets it, is honoured by `results_plots.load_results`.
7. **Full-row evaluation** — user decision 13-10-2026: no capped-context comparison. All eligible
   rows participate in five-fold CV. Every prediction-time-eligible column is retained, including
   inside TabPFN ensembles; there is no 100-feature benchmark cap. The 100-feature setting
   describes synthetic training only. Source identifiers and known post-outcome fields remain excluded.
