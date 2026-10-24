# Experimental design

What the three experiments ask, why the design can fail, and the premises it rests on. This
describes what is **built** (the `config/Exp{1,2,3}_*.yaml` sweeps); the prior's mechanics are in
[PRIORS.md](PRIORS.md), every knob's value in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md), and what
each run actually did in [RUNS.md](RUNS.md). Original source-audit pin `de324efcca55be392feb0c01830b3b6ce364587f`; later citation checks record their pins in `src/visualize/literature.py` and RUNS.md.

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
three-mixture grid. Gains on PD and LGD alone do not establish a double dissociation or rule out
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

All share the same real datasets and the same frozen evaluation ([§5](#5-evaluation)).
`credit_fraction` is the master switch throughout: the share of each batch drawn from our
credit-targeted path, the rest from the unmodified TabICL prior; `0.0` is the control. **The credit
prior is the same in every experiment.** Current ranges use literature and broad assumptions
([PRIORS.md](PRIORS.md) §5), but mechanism design was informed by these benchmark datasets.
The designated holdout was previously inspected. Removing the 28-09 numerical calibration on
29-09 does not undo that design exposure; external confirmation requires new data. That short-lived
calibration change postdates the September campaign and must not be attributed to its checkpoints.


| | **Exp0 — does it run?** | **Exp1 — does the prior help, all else equal?** | **Exp2 — is it useful for the real models?** | **Exp3 — run long** |
|---|---|---|---|---|
| start | the three paths below | from scratch | the released TabICLv2 and TabPFN-3 weights | from scratch |
| swept | `arm {scratch TabICL, fine-tuned TabICLv2, fine-tuned TabPFN-3}` | `credit_fraction {0, .5, 1} × 3 seeds` | `arm {TabICLv2, TabPFN-3} × credit_fraction {0, .5, 1} × 3 seeds` | `credit_fraction {0, Exp1's mix}` × 2 seeds |
| arms/track | 3 | 24 (9 main + 15 component) | 18 | 4 |
| steps | 600 | 12,500 | 10,000 | 200,000 |
| optimizer | as the path it checks | Muon (upstream stage 1) | upstream TabICL and an adapted TabPFN-Wide recipe | Muon |
| asks | every path works on the cluster; what a real arm and a real benchmark slot cost | the pure effect of the prior on what the model learns, at a small matched budget | whether the knowledge helps models trained at full scale, and at what out-of-domain cost | whether Exp1's effect holds at a longer budget |

**Exp0** uses two Slurm allocations: `exp0_cpu.slurm` checks every input, package, split and portable
test; `exp0.slurm` then checks CUDA, all six training/save/resume routes and both upstream trainer
equivalence checks. Each GPU route completes 600 steps with forced SIGUSR1 stop/resume. All model
routes are benchmarked on selected largest/widest/median-size credit and OOD tables, with full rows
and five folds. The complete benchmark matrix belongs to Experiments 1–3. CPU and GPU PASS receipts
bind code, packages, inputs and output hashes; either missing/currently failed receipt blocks production.
Diagnostic reference scores stay isolated from publication results.

**Exp1** measures the prior's effect in its purest form: the same architecture trained from scratch on
each mix, with the same compute. With so little training (2.5 % of upstream's stage 1) the arms are not
expected to reach the released model; the question is the contrast between mixes. Every training table
has 1,024 rows (TabICLv2's stage-1 budget), while evaluation uses every eligible row in the real
table with five-fold CV. Each fold fits on four folds and predicts the fifth; there is no scheduled
1,024-row context comparison. This tests transfer to larger contexts and must not be described as
reproducing upstream's full multi-stage, long-context training curriculum.

The component study was accepted on 13-10-2026: five removals at the fixed 50% mixture,
three paired seeds each, in `Exp1_{PD,LGD}_ablations.yaml`. Remove separately (1) the macroeconomic
term in the target, (2) the dependence of missingness on borrower covariates (retain missing values),
(3) context/query shifts, (4) the marginal tail warp, and (5) the credit categorical ceiling
(restore the upstream categorical sampler, whose legacy effective ceiling is nine). Architecture, optimizer and task budget stay fixed. These are
predeclared mechanism comparisons against the main 50% arm, not candidates selected on holdout
performance. Only final ablation checkpoints are scored; main arms supply the learning curves.
Random streams for targets, marginals, noise, shifts, row order and missingness are independent.
Removing a postprocessing component therefore does not advance unrelated random draws. The
categorical-ceiling removal changes the upstream sampled graph distribution and is a distributional
ablation; it does not promise identical latent graphs or labels. Selection can alter realised class
rates legitimately; the advertised PD range describes applicants, not accepted books.

Selection remains predeclared development ROC-AUC (PD) and R² (LGD). Log-loss and LGD
boundary-mass/CRPS comparisons are secondary hypothesis tests, reported alongside those primary
metrics rather than used to change the winning mixture after seeing results.

This adds 15 runs per track to the nine main runs, 48 in total. Component groups follow the
controlled-ablation principle in `tfm-library/papers/text/2026/05_Bouadi_et_al._Shaping_the_Prior_How_Synthetic_Task_Distributions_Determine_Tabular_Foundation_Model_Quality.txt`;
that paper studies general classification, not the credit-risk mechanisms here.

**Exp2 compares all three mixtures (0%, 50%, 100%) for both released models**, with three seeds:
18 runs per track, 36 total. It does not select a winning mixture from Exp1. Finish and benchmark
Exp1 first to assess its evidence and cost before the next campaign, while keeping this comparison
fixed. The submission gate requires validated Exp0, not an Exp1 selection artifact.

Exp2 asks whether the effect survives where it matters: the released, fully trained models, whose
pretraining continues on our prior. Two models, because a result for one architecture could be an
accident of it: TabICLv2 (open prior, trained through our own trainer, verified equal to upstream's)
and TabPFN-3 (trained through TabPFN's own fine-tuning path, `src/train/tabpfn_trainer.py`). **Recipes are fixed rather than searched (30-09-2026), with different evidential status:**
TabICLv2 uses its authors' stage-3 optimizer schedule at our controlled table size — Muon, lr 2e-5, weight
decay 0.01, 1 % warm-up, cosine to 1e-7, gradient clipping 1, batch 64, 10,000 steps
(`tfm-library/repositories/TabICL.txt`, `scripts/train_v2_clf_stage3.sh`; paper §4.1); TabPFN-3 uses an adaptation of how
TabPFN-Wide continued the released TabPFN-v2 classifier on the synthetic TabICL prior — AdamW, lr 1e-5, weight decay
1e-4, linear warm-up and cosine decay, clipping 1, batch 16, 10,000 steps, where its validation AUC
plateaus (`papers/2026/03_Kolberg`, "Continued Pre-Training"; they too fixed the length rather than
selecting a checkpoint). The stage-A search that used to choose among candidates bracketing these values
is dropped. TabPFN-Wide did not evaluate TabPFN-3 or LGD regression: extending its schedule to those models/tasks is our experimental choice, which Exp0 validates technically, not scientifically. The exact installed TabPFN 9.1.0 loss and preprocessing are used. In addition, CreditPFN's grid on real-data continued pretraining is
not finished and is a different setting (real tables), and a search would have cost as much as half of
the comparison. The same recipe for every mix keeps the comparison about the prior; every 2,500-step
checkpoint is scored as a learning curve, and the final one is the result. L2-SP and partial freezing are
not in the design (a later ablation). Two limits are shared by every arm and must be stated: TabPFN's
prior is not public, so its control continues on TabICL's prior; and all arms train on 1,024-row tables
where the released models also saw far longer ones, so the comparison with the released weights mixes
our prior's effect with that change (the control arm measures it), while the comparison between mixes
does not.

**Exp3** runs Experiment 1's best nonzero credit mix long, against its 0% control, from scratch:
the choice maximises development ROC-AUC (PD) or R² (LGD), averaging folds and training seeds.
Exact numerical ties choose the smaller credit share; seed spread is reported descriptively and
does not change that rule. If neither credit mix exceeds the control's development mean, the
candidate is not an established improvement: review whether the long-budget study merits its
cost, and describe a study that proceeds as testing whether an effect emerges at longer budgets.
Never substitute a duplicate 0% arm for the nonzero candidate.
The budget is 200,000 steps (16 × Exp1),
2 paired seeds (each seed gives both arms the same base tasks and initial weights; 5 seeds were too
expensive, 30-09-2026). The length: TabICLv2 reaches its reference in normalised accuracy before
100,000 steps but in log-loss only after ~200,000 (`papers/2026/02_Qu_TabICLv2` §7), our PD and LGD
metrics are probabilistic (Brier, log-loss, CRPS), and a prior's effect can appear late — the TabICL
prior degraded TabICLv2's validation loss only in the second half of its 280,000-step ablation. Its own
ablations are single runs at 280,000 steps, with the reference run twice to show the noise; two seeds
at 200,000 steps (~700 B200 GPU-hours for both tracks) follow that design at a price the project can pay.
100,000 steps with 2 seeds (~350) is the cheaper option, at the cost of stopping before log-loss has
settled. A stage-2-like end of training on longer tables (upstream's 400-10,240 rows) would shrink the
length gap at test time. It is not part of the current Exp3: Exp3 increases steps, not rows per task.
Any long-context training extension requires a separately specified budget and validation.

**What was designed but not built.** An **O'Prior-style generic-realism arm** (the ideal causal
control) and an **unrealistic-but-complex arm** (TabForestPFN's counter-hypothesis that complexity
beats realism). Both are deferred: O'Prior's realism engine is non-trivial to reimplement faithfully,
and a weak reimplementation would be a strawman that invalidates the headline. Their absence is the
limitation named in [§1](#1-the-claim-and-how-it-fails); evaluating both tracks does not replace
those controls or establish a double dissociation.

The prior mechanisms each experiment can switch on — PD's Vasicek one-factor defaults, controlled
base rate, underwriting rules, reject-inference selection, high-cardinality categoricals; LGD's
boundary atoms and interior shape; and the shared shift/missingness/noise — are described in
[PRIORS.md](PRIORS.md).

## 4. Matched compute, seeds, and the two phases

**Matched within each backbone, not across papers.** Mixture arms of the same backbone receive
the same optimizer steps and task count. In Exp2, TabICLv2 receives 640,000 tasks (batch 64) and
TabPFN-3 receives 160,000 tasks (batch 16), each over 10,000 updates. Consequently, cross-backbone
contrasts also include different adaptation budgets; the primary prior effect is the contrast
between mixtures within each backbone. This is **not** a match to O'Prior's protocol
(the configs follow TabICLv2 stage-1: 1,024 rows/table, 1–100 features), so absolute numbers are not
comparable to theirs — only our arm-to-arm contrasts are. **Two traps:** turning the filter *off*
makes generation cheaper, so wall-clock and credits are not comparable across filter arms (match on
steps/datasets, and report the rejection rate per arm); and TabICLv2's own ablations show a strong
prior×architecture interaction, so a prior change may need retuning — **log every arm's loss curve
and treat divergence as a reportable outcome, never tune it away silently.**

**Seeds.** 3 per arm in Exp1 and Exp2, 2 in Exp3 (paired: a seed gives both arms the same base tasks
and initial weights within a backbone). Report mean ± spread; gaps smaller than the seed spread
indicate descriptive instability, not a significance test or an alternative selection rule.
The Exp3 rule above breaks only exact development-mean ties by smaller credit share.

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
by looking at real-data results — which mix Exp3 runs, which checkpoint a curve is read at. A choice
made on a set of datasets is fitted to them: the winning configuration's score there is biased upward
(the winner's curse), exactly as a hyperparameter chosen on test data is. Training on synthetic data
removes the leak through the weights, not the leak through the choice. Since Experiment 2 dropped its
search (30-09-2026), the one choice left is Exp3's mix, made on development; every Exp1 and Exp2 result
is reported on all eligible datasets, with the designated holdout shown separately. This separation
does not erase the effects of earlier inspection and must not be described as an unbiased fresh test.

Five-fold CV estimates each fixed model's performance; selecting a mixture using those same
assessment scores and then reporting only its winning score would still induce selection bias.
Exp2 therefore retains every predeclared mixture and the final checkpoint, with paired effects
against the matching continued-pretraining control and released model reported separately.
Only Exp3's nonzero mix is selected, using development datasets and the primary metric alone.
OOD retention is an outcome to report, not an extra rule for re-ranking that choice. Existing
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
saved checkpoint (every 2,500 steps and the final one) is scored, so each arm has a learning curve on
real data as well as a final number. Our checkpoints train on 1,024-row tables and are scored on contexts
of up to ~426,000 rows: that length gap is identical for every arm, so the arm-vs-control contrast is
matched, but the released TabICLv2 was trained on up to 60,000 rows and is not — report the reference
gap with that caveat.
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
4. **Which mix fills Exp3's `FILL_FROM_EXP1`** — the predeclared development selection in
   `src/eval/selection.py`, at Exp1's final 12,500-step checkpoint. This is a budget-conditional
   choice, not proof that its ranking remains optimal at 200,000 steps; Exp3 tests persistence.
   Keep the current grid frozen. Exp2 compares every mix independently.
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
   the final one (10,000 steps, the recipes' own length) is the result, the others a learning curve.
   `eval.headline_step`, if a config sets it, is honoured by `results_plots.load_results`.
7. **Full-row evaluation** — user decision 13-10-2026: no capped-context comparison. All eligible
   rows participate in five-fold CV. Every prediction-time-eligible column is retained, including
   inside TabPFN ensembles; there is no 100-feature benchmark cap. The 100-feature setting
   describes synthetic training only. Source identifiers and known post-outcome fields remain excluded.
