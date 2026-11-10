# Configuration reference

The YAML files hold values; this document explains how they resolve and why the registered
recipes use them. [Experimental design](EXPERIMENTAL_DESIGN.md) owns the questions and evidence;
[PRIORS](PRIORS.md) owns generator mechanisms; [VSC](VSC.md) owns execution and resource planning.
Resolved configuration saved with a run is the authority for that run.

## Contents

1. [Sweep expansion and stable identities](#1-sweep-expansion-and-stable-identities)
2. [Training recipes](#2-training-recipes)
3. [Table geometry and feature limits](#3-table-geometry-and-feature-limits)
4. [Prior controls and encoding](#4-prior-controls-and-encoding)
5. [Budgets, checkpoints and interpretation](#5-budgets-checkpoints-and-interpretation)

## 1. Sweep expansion and stable identities

Inspect the resolved grid before submission:

~~~bash
python scripts/pretrain.py --config config/Exp2_PD.yaml --list
~~~

Scalar values specify one setting. Sweep lists are crossed; structural lists such as `_range`,
`_mean_sd`, `seeds`, `keep_steps` and dataset inventories retain their documented meaning.
`src/utils/config.py` defines the exceptions and expands `seeds` separately. Put a swept parameter
only in `sweep`, not also in the config body. Backbone presets in `arms` apply dotted overrides;
an unknown override fails rather than silently creating a new setting.

For Exp2, arm selects `tabicl` or `tabpfn3` and `lr_index` selects the corresponding numerical rate.
`src/experiments/design.py` owns `RATES`, `SEEDS`, `MAIN_STEPS`, `MILESTONES`, `FIXED_RETENTION` and
the design revisions. `apply_design` resolves these without reading benchmark scores.
Retired pilots, candidate fields and selection overrides are rejected.

Exp1 does not use that design resolver. Its appended 25% mixture uses
`array_append: [{prior.credit_fraction: 0.25}]` so indices 0–8 and their existing run names remain
unchanged; new indices 9–11 contain its three seeds. The metadata is removed from resolved
recipes and is not inherited through prior_file. Five component removals add 15 runs per task.
At 0% credit, unused credit knobs collapse to one effective control per seed.

Compatibility permits only reviewed source/package/recipe combinations. A completed historical
run being accepted for analysis does not grant arbitrary resume permission. Checkpoint, input,
environment, protocol and result hashes remain required.

## 2. Training recipes

### Experiment 1: scratch control

Four mixtures (0, 25, 50 and 100%) plus five component removals at 50%, each with three `seeds`
and two prediction tasks, give 54 runs. They use 12,500 updates and 64 tables/update at the
frozen small-table geometry. The component controls remove one mechanism at a time; they are
not additional mixture levels. Preserve these completed recipes when changing later experiments.

The optimizer is upstream Muon with peak LR `8e-4`, weight decay `.01`, gradient-norm limit
`10` and 1% warm-up (125 updates). One cosine cycle ends at `1e-7`. Parameters and computation
use FP32 with AMP disabled. Microbatches contain four tables. Main runs keep every 2,500th
update; component runs retain the 12,500-update endpoint.

### Experiment 2: full continued pretraining

**Target-space repair, 10-11-2026; pending VSC acceptance.** Exp2/3 TabPFN-3 LGD resolves
`train.regression_target_policy: standardize`. The native fine-tuning loader still fits mean/std
on context rows only (std floor 1e-8); per-member nonlinear target transforms are removed before
fitting. Feature-member choices, CRPS + mean-MSE, gradient clipping, precision, seeds, schedules
and inference preprocessing stay unchanged. New run names include `target-standardize-v1`;
old optimizer/checkpoint state is rejected. Historical objective probes explicitly use `native`.

The replacement comprises 16 Exp2 training runs and 80 checkpoint benchmarks. Twelve stopped
TabPFN LGD Exp3 treatments must use the same target policy and compare with the four corrected
50% Exp2 baselines. Other arms retain their original recipes. This prevents power-map
amplification; it neither bounds all standardized residuals nor establishes improved accuracy.

Two backbones × four mixtures × two rates × two `seeds` give **64 runs**, with 1,500 updates each.
No L2-SP or adapter is used. Both rates and all mixtures are reported separately.

| Setting | TabICLv2 | TabPFN-3 |
|---|---|---|
| `lr_index` 0 / 1 | 2e-5 / 1e-5 | 1e-5 / 1e-6 |
| Optimizer | Upstream all-parameter Muon | AdamW |
| Weight decay | .01 | .01 |
| Warm-up | .01 = 15 updates | .1 = 150 updates |
| Tables/update | 64 | 16 |
| Rows/table | 400–60,000, log-uniform | 1,024 |
| Context fraction | .79–.81 | .30–.90 |
| Precision | FP32 with explicit TF32/efficient-SDPA policy | FP32 weights, FP16 AMP and finite-gradient replay |

Both use global unscaled gradient-norm clipping at 1 and one warm-up/cosine schedule ending
at update 1,500, with LR floor 1e-7. Clipping limits the norm of the gradient passed to the
optimizer; it does not freeze weights or anchor them to the released checkpoint.

Muon follows the upstream parameter grouping and scaling, rather than torch.optim.Muon.
Its beta2 setting is retained for upstream API fidelity but affects only an AdamW branch, not
this all-Muon group. `TabPFNTrainer` accepts AdamW. Table losses are accumulated before the update;
different task batches and geometries mean compute is not matched **between backbones**.
Mixture comparisons within a backbone retain the same recipe.

The rates follow the native TabICLv2 stage-3 peak, public TabPFN fine-tuning/TabPFN-Wide
precedents and Ning et al.'s single-table adaptation evidence. They are externally motivated
choices, not proven optimal multi-table CPT rates. TabPFN-3's original pretraining recipe is
not public; its 0% arm here is additional generic **TabICL-prior** exposure.

### Experiment 3: constrained adaptation

The mixture is fixed at 50%. Six recipes × two backbones × two tasks × two `seeds` give
**48 new runs**. Sixteen Exp2 runs provide both full-CPT rates, without copying checkpoints; the four
TabPFN LGD baselines must first be replaced under the repaired target policy. The revision is
`retention-six-recipes-1500-v3`; Exp2 retains
fixed-rates-1500-two-seeds-v2.

| Recipe ID | Trainable weights / optimizer / peak rate | Regularization |
|---|---|---|
| `no_decay` | Full model; Muon 2e-5 for TabICL, AdamW 1e-5 for TabPFN | Decay 0 |
| `l2sp` | Same full-model primary rate | Decay 0; L2-SP alpha .003 |
| `diagscale_low` / `diagscale_high` | Query-scale deltas; AdamW 1e-4 / 1e-3 | .01 decay on deltas |
| `lora_low` / `lora_high` | Q/V factors; AdamW 3e-6 / 1e-5 | Decay 0; rank 16, alpha 16, dropout 0 |

Geometry, `seeds`, precision and data objectives remain those of Exp2 within each backbone.
Adapters use 10% warm-up, clipping 1 and the same cosine endpoint. Original weights are frozen
for adapters, and initialization preserves native predictions. DiagScale is injected before
inference caches; LoRA exports merged native weights while retaining resumable factors.
L2-SP anchors the **released** parameters, including after resume.

L2-SP versus `no_decay` tests the anchor addition. `no_decay` versus primary-rate Exp2 tests ordinary
decay. Adapter results are compared separately with both full-CPT rates. These are complete
recipe comparisons, not a best-rate search or an isolated parameter-count experiment.
The [design rationale](EXPERIMENTAL_DESIGN.md#33-experiment-3-constrained-adaptation-at-a-fixed-50-mixture)
records literature and CreditPFN archive evidence, plus their transfer limitations.

## 3. Table geometry and feature limits

Exp1 and Exp2/3's TabPFN `arms` use the following small-table law; the latter is a declared
adaptation, not a claim about private TabPFN-3 pretraining defaults:

| Parameter | Value |
|---|---|
| `n_rows_range` | [1024, 1024] |
| `n_features_range` / `max_features` | [1, 100] / 100 |
| `train_frac_range` | [.3, .9] |
| `group_size` / `subgroup_size` | 4 / 4 |
| `n_nodes_range` | [2, 33], exclusive upper bound in the transcribed backend |

Exp2/3's TabICL `arms` use [400, 60000], `log_uniform` row sampling, [.79, .81] context, group and
subgroup size 1, and one-table microbatches. This follows `Prior.sample_seq_len` and the stage-3
scripts in tfm-library/repositories/TabICL.txt. The median row count is about 4,899; uniform
sampling would be a different distribution. Row/feature/split draws are paired across mixtures.

With `base.implementation: upstream`, the native generator owns graph depth and control category
sampling. `n_nodes_range`, `base.max_cat_size` and `category_frequency` settings do not override that
production generator; they configure the transcribed test backend. Credit `max_cat_size` and
`use_corrected_cat_sizes` do configure the production credit path.

The upstream long-context ceiling, from `Prior.adjust_max_features`, is:

| Rows, up to | Maximum synthetic features |
|---:|---:|
| 10,240 | 100 |
| 20,000 | 80 |
| 30,000 | 60 |
| 40,000 | 40 |
| 50,000 | 30 |
| 60,000 | 20 |

The planned ceiling reaches native generation, filtering and credit augmentation. A macro
column replaces a noise column rather than increasing the planned width.

**These are training-distribution limits, not benchmark caps.** Protocol 9 retains all
prediction-time-eligible features and all fitted rows. It does not perform feature ranking or
near-constant removal. TabPFN's per-estimator budget includes its preprocessing expansion,
and a runtime check rejects internal feature subsampling. Exp0 checks wide-input predictions
and selected full-table capacity; that does not establish accuracy invariance at unseen widths.

Exp2 continues TabICLv2's stage-3 table law rather than restarting stages 1–2. TabPFN uses its
differentiable episode interface on fresh synthetic tasks. The models retain different prior
exposure histories, so matching a continuation budget does not match total pretraining.

## 4. Prior controls and encoding

| Control | Meaning |
|---|---|
| `prior.credit_fraction` | Per-slot probability of credit generation; 0/.25/.5/1 in Exp1/2, .5 in Exp3 |
| `prior.encoding` | `tabicl`: prediction-view preprocessing and context-scaled LGD; `raw`: native TabPFN preprocessing, NaNs retained and credit LGD initially in [0,1] |
| `prior.grouping` | Shared shape within groups for small-table recipes; independent long-table geometry for TabICL CPT |
| `prior.credit.target` | PD mechanism or LGD `zoib`; detailed equations/ranges in PRIORS |
| `prior.filter` | `tabicl`, `off` or `banded`; production `apply_to: base` leaves credit learnability unfiltered |
| `prior.pool.source` | `generate` for current campaigns; prior pools are not a prerequisite |

Credit additions include controlled PD base rates/common shocks, LGD endpoint masses, category
exposure, monotone marginal changes, covariate-dependent missingness and shifts. Their ranges,
source quality and qualifications are documented once in [PRIORS.md](PRIORS.md#5-what-we-change).
The control remains native TabICL generation/filtering with shared validity checks. A `raw`
TabPFN task must retain a usable feature on its context; query variation cannot rescue a
constant context. Invalid candidates are redrawn within the same bounded slot budget.

A 0% arm measures additional generic training, not guaranteed retention. Replay is a proposed
protection against forgetting, not proof of non-degradation. Credit PD is binary whereas the
native control can have 2–10 classes, so package effects also change task-format exposure.

## 5. Budgets, checkpoints and interpretation

| Experiment | Updates/run | Runs | Permanent checkpoints/run |
|---|---:|---:|---|
| Exp1 | 12,500 | 54 | Main-grid cadence 2,500; component study retains its configured endpoint |
| Exp2 | 1,500 | 64 | 125, 250, 500, 1,000, 1,500 |
| Exp3 | 1,500 | 48 new + 16 reused | Same five milestones |

`keep_steps` retains permanent checkpoints independently of rolling saves. `max_temp_checkpoints: 2`
applies only to nonpermanent recovery saves. Final-step benchmark slots remain first, preserving
existing identities. Full-row credit/native-OOD five-fold evaluation covers every retained milestone;
intermediate results describe trajectories, not alternative selected winners.

Exp2 retains 320 snapshots and Exp3 adds 240. Exp2 requires 56,800 dataset/fold/model evaluations,
and new Exp3 `arms` add 42,600, before references and PD calibration refits. Budget benchmarking
separately using actual timings and quotes in [VSC.md](VSC.md#resources-and-costs).

The 1,500-update horizon is a compute-budget choice, not an established optimum or convergence
claim. Upstream stage 3, TabPFN-Wide and Real-TabPFN used different rows, tables/update and
objectives; update counts alone do not compare exposure. Here a run sees 96,000 TabICL tasks
or 24,000 TabPFN tasks. Two paired `seeds` show spread but do not precisely estimate variability.

Early quoted real-data LGD R² bands were not verified source results. They neither calibrate
the prior nor establish a success threshold. Use current protocol-bound artifacts for results
and dataset-source evidence for feature/target claims.
