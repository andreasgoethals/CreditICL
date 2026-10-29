# Experimental design

What the three experiments ask, why the design can fail, and the premises it rests on. This
describes what is **built** (the `config/Exp{1,2,3}_*.yaml` sweeps); the prior's mechanics are in
[PRIORS.md](PRIORS.md), every knob's value in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md), and what
each run actually did in [RUNS.md](RUNS.md). Original source-audit pin `de324efcca55be392feb0c01830b3b6ce364587f`; the 24-10-2026 Experiment 2 audit uses TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`. The fixed-rate design review of 29-10-2026 uses TFM pin `26156f36eb7676e8d9050547817346db548b0e85`.

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

The three research questions form a sequence rather than three searches for a winning score:

1. **RQ1 — prior usefulness and components:** at matched scratch architecture and training budget,
   how does the credit prior package change PD/LGD performance, and which removals change its effect?
2. **RQ2 — transfer and mixture:** when continuing released models, how do credit/generic mixture
   and two predeclared learning rates affect credit performance and native OOD retention?
3. **RQ3 — adaptation constraints:** at a fixed 50% mixture and predeclared reference rate, how do
   fixed regularization and parameter-efficient recipes change the adaptation–retention trade-off?

Experiment 0 validates implementation; it is not a fourth research experiment. Training trajectories,
cost, parameter drift and OOD retention are diagnostics within RQ2/RQ3. More backbones or additional
freezing methods are not necessary to answer these bounded questions.

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
Its [23-10-2026 revision](../tfm-library/papers/text/2026/10_Bouadi_et_al._Shaping_the_Prior_How_Synthetic_Task_Distributions_Determine_Tabular_Foundation_Model_Quality.txt)
also describes bounded target transforms and independently ablatable finance-style volatility shocks.
The novelty claim is therefore the controlled PD/LGD prior-package and adaptation study, not the
first domain-informed prior or a claim that existing generators cannot represent credit traits.

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
| 2 | How do mixture and fixed learning rate affect transfer and OOD retention? | 96 = 2 backbones × 4 mixtures × 2 rates × 3 seeds × 2 tasks; 1,000 updates each |
| 3 | Which fixed update constraints trade credit adaptation against OOD retention? | 48 new = 2 backbones × 4 methods × 3 seeds × 2 tasks; 1,000 updates each; 12 reused Exp2 baselines |

### 3.1 Experiment 1: preserved evidence

Four credit shares (0%, 25%, 50%, 100%) and five component removals at 50%, with three seeds
per PD/LGD task, use 12,500 updates of 64 tables of 1,024 rows. The original campaign contained
48 runs; six quarter-mixture runs were appended. Existing completed-run provenance is accepted
only for explicitly reviewed source/package/recipe combinations. Original historical completion
permission does not authorize arbitrary checkpoint resume. Unchanged Exp1 runs from `19a76cc`
may resume across this patch under an additional exact-source and recipe check.

### 3.2 Experiment 2: full continued pretraining

Every run starts from released weights, updates the full model, and uses a fixed **1,000-update**
budget. Both predeclared peak rates are crossed with all four mixtures and three paired seeds:
24 runs per backbone/task, 96 in total. There is no learning-rate pilot, candidate-ranking receipt,
or benchmark-driven rate choice. The earlier Exp2A proposal is superseded; historical attempts
remain described in [RUNS.md](RUNS.md), not silently relabelled as this design.

The primary endpoint is update 1,000. Updates 125, 250, 500 and 1,000 are permanently retained
and benchmarked. Released weights are a shared step-zero reference. Intermediate points describe
trajectories, not post-hoc alternative headline checkpoints. Report both rates separately, including
an unfavorable result, rather than the best rate per dataset or the maximum across rates.

| Setting | TabICLv2 | TabPFN-3 |
|---|---|---|
| Tables per update | 64, one-table gradient accumulation | 16, one preprocessed task at a time |
| Rows per synthetic table | 400–60,000, log-uniform | 1,024 |
| Context share | 79–81% | 30–90% |
| Fixed peak rates, both run | 2e-5 and 1e-5 | 1e-5 and 1e-6 |
| Optimizer | Native Muon configuration (all trainable parameters) | AdamW |
| Weight decay | 0.01 | 0.01 |
| Warm-up | 1% (10 updates) | 10% (100 updates) |
| Schedule | One warm-up/cosine cycle over 1,000 updates, floor 1e-7 | Same schedule family and endpoint |
| Gradient norm limit | 1 | 1 |
| Precision | FP32, explicit TF32/efficient SDPA policy | FP32 weights, FP16 autocast and GradScaler |
| Objective | Native classification / 999-quantile regression | Native differentiable classification / distributional regression |

Synthetic width remains at most 100 with the upstream row-dependent ceiling; this is not a
benchmark feature limit. Every eligible benchmark row and feature is retained, with five common
folds, on 21 credit tables and 25 native OOD datasets per task. Small-context development/OOD
monitoring is labelled separately and never substitutes for the full-table benchmark.

**What is upstream and what is a research choice.** The primary texts and repository snapshots
below were checked at TFM pin `26156f36eb7676e8d9050547817346db548b0e85`. Code-dump references
name symbols because snapshot line numbers are unstable.

| Evidence | Reported budget and geometry | Implication for this experiment |
|---|---|---|
| [TabICLv2](../tfm-library/papers/text/2026/09_Qu_et_al._TabICLv2_A_better_faster_scalable_and_open_tabular_foundation_model.txt), §4.1 and stage-3 scripts | Stage 1: 500k updates, 64 tables/update, 1,024 rows, peak 8e-4; stage 2: 40k, 400–10,240 rows, peak 1e-4; stage 3: 10k, 400–60,000 rows, peak 2e-5. Muon, cosine, decay .01. | Direct anchor for continuing the released model with stage-3 geometry and the higher rate. The native total is 550k updates / 35.2M tables; it does not establish an optimal credit-CPT budget. |
| [TabPFN-Wide](../tfm-library/papers/text/2026/03_Kolberg_et_al._TabPFN_Wide_Continued_Pre_Training_for_Extreme_Feature_Counts.txt), CPT and Appendix J | TabPFN-v2 classifier, 10k updates, batch 16 (8 for widest tasks), **40–400 rows**, widened features; AdamW 1e-5, decay 1e-4, warm-up/cosine. Duration followed a monitoring-set plateau. | Supports a synthetic-CPT rate and task batch, not a proven TabPFN-3 optimum. Neither its row geometry nor its decay is the public TabPFN fine-tuning default used here. |
| [Real-TabPFN](../tfm-library/papers/text/2025/07_Garg_et_al._Real_TabPFN_Improving_Tabular_Foundation_Models_via_Continued_Pre_training_With_Real_World_Data.txt) | TabPFN-v2 real-table CPT: 20k updates, batch 1, up to 20k rows, 60/40 context/query; AdamW 3e-7, warm-up/cosine, L2-SP .003. | Motivates cautious adaptation and the separate anchor penalty. Update count or LR cannot be transferred independently of its objective, rows and task batch. |
| [Ning et al.](../tfm-library/papers/text/2026/10_Ning_et_al._Test_Time_Compute_for_Tabular_Foundation_Models_Mechanisms_Gains_and_Limits.txt), B.2–B.3 | Single-table adaptation for at most 30 epochs, patience 8; selected full-FT rates 1e-5 for TabICLv2 and 1e-6 for TabPFN-3. | Supports the lower rates as external evidence, not their optimality for multi-table CPT. Their study-level LR selection on benchmark outcomes is disclosed in the paper; this experiment does not repeat that selection. |
| Public TabPFN fine-tuner, `repositories/TabPFN .txt`, `FinetunedTabPFNBase` and `get_cosine_schedule_with_warmup` | Default AdamW rate 1e-5, decay .01, 10% warm-up/cosine, clipping 1; a single-table, validation-stopped interface. | We reuse its differentiable episode path across synthetic tables, not its task-specific checkpoint selection. |
| [TabPFN-3 report](../tfm-library/papers/text/2026/05_Grinsztajn_et_al._TabPFN_3_Technical_Report.txt) | Synthetic pretraining exceeding 8T tokens, without a reproducible native optimizer/update/LR recipe. | Do not claim reproduction of its private original training or infer that TabICL's settings are optimal for it. |
| [EXAONE Tabular](../tfm-library/papers/text/2026/08_Eo_et_al._EXAONE_Tabular_1.0_Technical_Report.txt), [Mitra v2](../tfm-library/papers/text/2026/09_Tao_et_al._Mitra_v2_Technical_Report.txt), [KumoRFM 2](../tfm-library/papers/text/2026/04_Hudovernik_et_al._KumoRFM_2_Scaling_Foundation_Models_for_Relational_Learning.txt) | Different objectives, optimizer combinations, table batches and stage/adaptation schedules. | These demonstrate recipe dependence, not a universal continuation rate, seed count or update budget. |

In `repositories/TabICL.txt`, `train_v2_clf_stage3.sh`, `train_v2_reg_stage3.sh`,
`Trainer.configure_optimizer` and `Muon.step` establish the implemented stage-3 behavior:
regular .01 decay, 1% warm-up and clipping 1. Appendix D's cautious-decay discussion does not
establish that the released command enabled it. Continuing stage 3 preserves a distribution
containing both short and long tables; repeating stages 1–2 would introduce a different curriculum.

TabPFN's `FinetunedTabPFNClassifier`, `FinetunedTabPFNRegressor` and `fit_from_preprocessed`
supply differentiable single-table context/query episodes. We average losses from fresh synthetic
tables before an optimizer update. The 16-table batch and 1,024-row tables are a declared adaptation
recipe, not disclosed TabPFN-3 defaults. Its 0% control uses the generic TabICL prior, not a
reconstruction of the private native TabPFN prior.

**Rates, schedule and decay.** The TabICL pair compares its native stage-3 peak to half that rate.
The TabPFN pair brackets the public fine-tuning/synthetic-v2-CPT rate and a conservative rate with
actual v3 single-table evidence. AdamW's coordinate scaling and Muon's matrix transformation do
not remove the global LR multiplier. The 1,000-update schedule is a complete cycle, not an
early-stopped prefix of the earlier 10k or 2.5k proposals. General CPT research supports
re-warming/re-decaying with replay ([Ibrahim et al., 2024](https://arxiv.org/abs/2403.08763));
it does not prove an optimal schedule for this TFM task. No benchmark-driven plateau controller,
repeated restart or scheduler grid is added.

Weight decay multiplies current weights by `1 - lr * weight_decay`; it is not a reset to zero
and does not anchor the released checkpoint. With peak 2e-5 and this cosine schedule, its isolated
exact-arithmetic shrinkage is about .01% over 1,000 updates, ignoring gradients and rounding.
That is an illustration of scale, not a bound on optimization effects. Keep .01 in Exp2 to retain
the verified source/interface default and compare its removal separately in Exp3. L2-SP stays in
Exp3 so the main comparison does not add both a prior intervention and a checkpoint penalty.

A 29-10 local Torch 2.13.0+cpu probe with FP32 parameters, zero gradients and foreach AdamW
changed no entries at LR 1e-6 / decay .01, but changed nonzero entries at LR 1e-5. This is not CUDA
Torch 2.11 evidence. Exp3 uses the predetermined primary rate, not the lower-rate winner.
Validate effective decay on the actual runtime; numerically identical treatments should be
reported as such, not amplified after inspecting benchmark outcomes.

**Why 1,000 updates and three seeds.** This is a budget-limited adaptation study, not a claim
of convergence or optimality. It preserves four mixtures, both rates and three paired seeds.
Native stage 3, TabPFN-Wide and Real-TabPFN have different tables/update, rows, objectives and
feature counts; their update counts are not interchangeable measures of exposure. A negative
result means no improvement under this tested budget, not that longer credit CPT cannot work.
Three paired streams provide a modest measure of stochastic variation; more updates in one run
cannot replace replication. [Bouthillier et al. (2021), Appendix C](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)
motivates accounting for variance and paired comparisons, but does not prescribe three as a universal minimum.

There are 384 permanent Exp2 snapshots, approximately 170 GB by linear scaling of the earlier
checkpoint inventory, plus rolling saves, references and benchmark artifacts. The observed/provisional
48×2,500 training extrapolation of 46.6–49.4M credits scales to **37.3–39.5M for 96×1,000** at the
quoted 30,600 credits/hour. This excludes validation, benchmarking and Exp3, and assumes comparable
throughput. It is not a billing quote or a guarantee of affordability.

Every retained Exp2 snapshot on all credit and OOD datasets implies **68,160 dataset-fold-model
evaluations**: 48 PD runs × 4 steps × 39 datasets × 5 folds, plus 48 LGD runs × 4 × 32 × 5.
References and PD threshold-calibration refits add work. Exp3's 48 new runs add 34,080 such
evaluations. Full-table benchmarking can dominate total cost; price it from actual runtime evidence
before launching the entire campaign. Keeping few trainable parameters does not itself make
DiagScale/LoRA inference or fitting cheap. Preserve the full-row/full-feature/five-fold protocol
rather than silently reducing coverage to fit a budget.

A subsequent read-only VSC audit on 29-10 supplies a conditional inference anchor: existing
released-model and Exp1 fold timers project **13.46–13.89M benchmarking credits for Exp2**
and **6.73–6.95M for Exp3** at comparable speed. Including the training estimates gives roughly
76–80M credits, before validation, retries and remaining Exp1 benchmarking, versus 95.81M available
at inspection. These are stored-work timers rather than billing records; adapter inference
and adapted-model runtime remain unmeasured. See [VSC.md](VSC.md) and the
29-10 read-only audit in [RUNS.md](RUNS.md) for observed values.

### 3.3 Experiment 3: constrained adaptation at a fixed 50% mixture

The experiment uses the same 1,000-update geometry and three paired seeds as Exp2.
The mixture is fixed in advance at 50%. Full-model reference rates are **2e-5 for TabICLv2 and
1e-5 for TabPFN-3**, `lr_index=0`, regardless of which Exp2 rate scores better. Reuse those
12 main full-CPT-with-decay runs. There are 48 new runs and no pilot or candidate selection.

| Additional method | Trainable parameters | Fixed settings |
|---|---|---|
| No decay | Whole model | Predetermined full-model optimizer/LR; weight decay 0 |
| L2-SP | Whole model | Same optimizer/LR; weight decay 0; alpha .003 |
| DiagScale | Query-scaling deltas only | AdamW; LR .03; adapter decay .01 |
| Q/V LoRA | Rank-16 Q/V factors only | Alpha 16, no dropout; AdamW; LR 1e-4; adapter decay .01 |

These are four **fixed recipes**, not an exhaustive search for the best method. L2-SP .003 is
anchored in Real-TabPFN and DiagScale .03 in Ning et al.; transfer to this multi-table task is a
hypothesis. Q/V LoRA at rank/alpha 16 and LR 1e-4 is a declared engineering choice informed by
low-rank adaptation, not a rate proven best for these backbones. Ning's LoRA menu uses every
linear layer, so it does not establish equivalence to this Q/V-only implementation.

L2-SP adds alpha/2 times the squared displacement from the **released** parameters. The anchor
is reconstructed from the released checkpoint before resume; it never moves to resumed
parameters. In TabPFN, its gradient is added once after unscaling/replaying the data gradient and
before clipping. Ordinary decay instead shrinks current parameters toward zero. Both are
legitimate regularizers but express different assumptions. L2-SP versus no-decay isolates the
anchor addition; no-decay versus reused Exp2 isolates ordinary weight decay.

DiagScale starts with identity query multipliers after each native query-scaling operation,
including TabPFN's `SoftmaxScalingMLP`; all original weights are frozen. Q/V LoRA starts with a
zero residual, preserves packed TabICL K weights exactly, and adapts native TabPFN Q/V projections.
Adapter checkpoints retain resumable factors; LoRA also exports merged native weights for normal
inference. DiagScale is installed before inference caches are built. Adapter optimizers use 10%
warm-up (100 updates), clipping 1 and the same 1,000-update cosine endpoint. Adapter-only decay
does not decay the frozen backbone. All four milestones are benchmarked.

Source anchors: `repositories/Test-Time Compute for Tabular Foundation Models.txt`,
`inject_diagscale`, and the associated Ning et al. paper; Real-TabPFN's L2-SP continuation;
[Hu et al., LoRA](https://arxiv.org/abs/2106.09685). Single-table gains are not evidence that these
methods necessarily prevent general-domain forgetting during CPT. Credit gains, OOD changes,
parameter drift, trainable counts and measured runtime must all be reported.

Generic replay is already tested through Exp2's mixture factor. EWC, teacher distillation,
head-only or additional layer-freezing methods would change objectives or introduce more
hyperparameters; they remain optional future work rather than prerequisites for this study.
No learning-rate or adapter-selection receipt is part of the current launch path. Historical
pilot configs/receipts cannot override the predeclared rates or methods.

### 3.4 Experiment 0: implementation and capacity gates

The CPU job covers the portable suite, all dataset loads/splits, configurations, task replay,
fixed-rate and baseline identity, rejection of retired pilot selection, checkpoint retention,
receipts, output paths and notebook contracts.
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
TabICLv2 receives 64,000 tasks (batch 64, stage-3 geometry) and TabPFN-3 receives 16,000 tasks
(batch 16, 1,024 rows), each over 1,000 updates. Common random numbers pair shape draws across
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
Neither Exp2 nor Exp3 ranks seeds, rates or method settings using benchmark scores.

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

The legacy designated split lives in each config's `eval:` block and remains unchanged for Exp1
compatibility. Exp2 and Exp3 evaluate both partitions but neither partition selects their rates,
mixtures, checkpoint endpoint or method settings. The proposed all-credit Exp2A pilot is cancelled.
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

**Why distinguish development and assessment data when gradients use synthetic tables?**
Selecting a rate or method by real-data results still fits the procedure to those datasets: its
winning score is biased upward even when no real row supplied a training gradient. Exp2 avoids
that additional selection by reporting both externally motivated rates and all four mixtures;
Exp3 fixes its mixture, reference rate and method settings before their outcomes. Every arm is
reported on all eligible datasets. This does not erase historical benchmark inspection, and these
results must not be described as a newly untouched external confirmation.

Five-fold CV estimates each fixed model's performance; selecting a mixture using those same
assessment scores and then reporting only its winning score would still induce selection bias.
Exp2 therefore retains every predeclared rate and mixture and the final checkpoint, with paired effects
against the matching continued-pretraining control and released model reported separately.
Exp3 reuses only the predeclared primary-rate 50% baseline, not the best-performing Exp2 arm.
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
125, 250, 500 and 1,000. There are no Exp2/Exp3 pilots. Exp1 and Exp2/Exp3's TabPFN-3 arms train on 1,024-row tables;
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
Only three LGD datasets are in Exp1's designated holdout: even unanimous non-tied differences have a minimum
two-sided sign-test p-value of 0.25. LGD holdout and component results are consequently descriptive
at this sample size; effect sizes, all per-dataset scores and uncertainty must be reported.
Moving inspected development tasks into the confirmatory test or counting folds as independent
replicates would not solve that limitation. ROC-AUC/R² remain the predeclared primary score summaries;
log-loss/Brier and CRPS/boundary calibration address different secondary questions, not post-hoc
replacement primary endpoints.

**Metrics** (all computed in one run). PD: ROC-AUC, PR-AUC, Brier, log-loss, KS, calibration slope, and
F1, precision, recall, MCC and balanced accuracy at the validation-tuned threshold —
**never accuracy alone** (at a 7% base rate "never defaults" already scores 0.93), and ECE with MCE
because the sparse high-score tail is where the defaults are. LGD: pinball, CRPS, interval coverage,
**boundary-mass calibration** (predicted vs observed P(y=0), P(y=1)), with R²/RMSE alongside and
the decoding rule stated. The earlier quoted LGD R² ranges were not verified literature results;
they do not set a success threshold. Compare paired changes against matched controls under the
current protocol, rather than declaring success from an absolute score on one dataset.

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
| meta-overfitting | 21 datasets, many iterations, historical holdout results already inspected | both fixed rates and all recipes reported; no new pilot/test selection; independent source/time cohort needed for confirmatory credit claims |
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
4. **Retention settings** — resolved 29-10-2026: fixed 50% mixture, both full-model rates
   reported in Exp2, fixed method recipes and primary-rate baseline in Exp3; no pilots.
   The long scratch run remains dropped.
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
   the final one (fixed at 1,000 steps after the 29-10 fixed-rate revision) is the result, the others a learning curve.
   Neither 1,000 nor the original proposed 10,000 is established as optimal; the current horizon is
   a disclosed compute-budget constraint, frozen across all mixtures and rates.
   `eval.headline_step`, if a config sets it, is honoured by `results_plots.load_results`.
7. **Full-row evaluation** — user decision 13-10-2026: no capped-context comparison. All eligible
   rows participate in five-fold CV. Every prediction-time-eligible column is retained, including
   inside TabPFN ensembles; there is no 100-feature benchmark cap. The 100-feature setting
   describes synthetic training only. Source identifiers and known post-outcome fields remain excluded.
