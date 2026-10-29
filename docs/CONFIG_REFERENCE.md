# Config reference — why each setting is what it is

The `config/Exp{1,2,3}_{PD,LGD}.yaml` files hold **values**, one note per knob; this file holds the
**reasoning** — why that value, what evidence backs it, what upstream does, what breaks if you change
it — under the same knob name. The split is deliberate: a config is read while deciding what to
submit, so prose in it makes the settings unfindable. If you catch yourself writing a paragraph above
a knob, it belongs here.

Neighbours: [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md) (what the experiments ask) ·
[PRIORS.md](PRIORS.md) (how the prior is built) · [VSC.md](VSC.md) (the cluster). Original source-audit pin `de324efcca55be392feb0c01830b3b6ce364587f`; the 24-10-2026 Experiment 2 audit uses TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`. The 29-10-2026 fixed-rate design uses TFM pin `26156f36eb7676e8d9050547817346db548b0e85`.

## Contents

1. [How the sweep works](#1-how-the-sweep-works)
2. [The prior's shape is not ours to choose](#2-the-priors-shape-is-not-ours-to-choose)
3. [`max_features` — 100, and where it bites](#3-max_features)
4. [Why one stage, not TabICLv2's three](#4-why-one-stage)
5. [Training budget: `max_steps`, `micro_batch_size`, `amp`](#5-training-budget)
6. [`optimizer` — Muon from scratch, AdamW to fine-tune](#6-optimizer)
7. [The prior knobs](#7-the-prior-knobs)
8. [Experiment 2 — the continued-pretraining knobs](#8-experiment-2-the-continued-pretraining-knobs)
9. [A note on the LGD R² band](#9-a-note-on-the-lgd-r-band)

---

## 1. How the sweep works

Any setting may be a single value or a list; a list means one run per value, and all lists are
**crossed**. Names ending in `_range` are literal `[low, high]` intervals sampled *from*, never swept
— to sweep one, nest it: `[[0.0, 0.1], [0.1, 0.3]]`; names ending in `_mean_sd` are literal
`[mean, sd]` pairs of a normal distribution. Inspect the expansion with
`python scripts/pretrain.py --config config/Exp1_PD.yaml --list`.

**Backbone presets** use `sweep.arm` and an `arms:` block of dotted overrides, applied after the
sweep (`apply_arm_preset`). An override must name a key the config body already has, so a typo
raises instead of doing nothing. For Exp2, `sweep.arm: [tabicl, tabpfn3]` crosses
`lr_index: [0, 1]`, mixtures and seeds. `src/experiments/design.py:apply_design` then maps the index
to the corresponding backbone's fixed numerical rate. Exp3 fixes `lr_index: 0` and crosses methods.

**Exp1 sweeps `credit_fraction {0, .25, .5, 1}` × 3 seeds = 12 main arms**. The 25% extension
was approved on 26-10-2026 after the initial nine-main-arm campaign; its three seeds are appended
at indices 9–11 so old run names and benchmark tags remain unchanged. The five component
removals still add 15 arms per task. The control is TabICLv2's prior and filter exactly;
the current ranges use the credit-risk literature and broad modelling choices ([PRIORS.md](PRIORS.md) §5).
Mechanism design was informed by the benchmark suite, including the designated holdout;
it is not untouched external validation. The credit mechanisms are shared across Exp0-Exp3;
table geometry is now backbone-specific in Exp2. At
`credit_fraction: 0.0` the credit knobs have no effect, so such combinations collapse to one control
per seed (`effective_fingerprint` in `src/utils/config.py`).

`array_append: [{prior.credit_fraction: 0.25}]` implements this extension as a stable partition
after crossing seeds. Without it, the ordinary seed-outermost expansion would shift the old
indices 3–8. This is launch metadata, removed from resolved recipes and not inherited through
`prior_file`; it does not change task generation or optimization. Invalid/unmatched patterns fail.
The receipt compatibility check permits only this reviewed change to grid counts, launcher array
size and selection messages. All data, checkpoint, package, evaluation-setting and result-file
hashes still have to match. Unknown code revisions remain invalid; this is not a force-reuse flag.

**One knob, one home:** a knob in `sweep:` must not also appear in the config body —
`apply_sweep_block` writes over the body, so a body literal is dead text that reads like a setting.
`test_a_swept_knob_has_exactly_one_home` checks all six files. **Open one group at a time:** with every
lever moving you cannot attribute an effect to any of them.

## 2. The prior's shape is not ours to choose

`n_rows_range`, `n_rows_sampling`, `n_features_range`, `max_features`, `n_nodes_range` and
`train_frac_range` specify the synthetic task distribution. Geometry is paired across mixtures
within each backbone, while the credit package also changes targets, encoding and filtering.
Exp1 retains the following stage-1 geometry from `scripts/train_v2_reg_stage1.sh`;
Exp2/Exp3's TabPFN-3 arms also retain it as a declared adaptation:

| knob | Exp1 / Exp2–3 TabPFN-3 | upstream stage 1 |
|---|---|---|
| `n_rows_range` | `[1024, 1024]` | `--max_seq_len 1024`, no `--min_seq_len` |
| `n_features_range` | `[1, 100]` | `--min_features 1 --max_features 100` |
| `n_nodes_range` | `[2, 33]` | `--min_n_nodes 2 --max_n_nodes 32` |
| `train_frac_range` | `[0.3, 0.9]` | `--min_train_size 0.3 --max_train_size 0.9` |

**These arms have 1,024 rows exactly, not a range**; only the train/test split varies.
Exp2/Exp3's TabICLv2 arms instead use **`[400, 60000]` with `n_rows_sampling: log_uniform`**,
`train_frac_range: [0.79, 0.81]`, group/subgroup size 1 and micro-batch size 1. This matches
`Prior.sample_seq_len` and the stage-3 scripts in `tfm-library/repositories/TabICL.txt`.
Log-uniform means equal probability per multiplicative interval, with median about 4,899 rows;
uniform sampling over the interval would be a materially different prior. The row/feature/split
draws remain paired between 0%, 25%, 50% and 100% mixtures for each seed.

With `base.implementation: upstream`, the upstream generator
owns graph depth and control categories. `n_nodes_range`, `base.max_cat_size` and the
`category_frequency` fields are only configurable in the small `transcribed` test backend;
they do not override the production upstream generator. Keep the documented upstream values
when using that backend.
`credit.max_cat_size` and `credit.use_corrected_cat_sizes` do configure the production credit path.
The benchmark metric set and pinned OOD inventory are fixed in the evaluation modules, rather
than configurable through unused `eval.metrics` / `eval.ood_suites` fields. `beta2` is passed to
Muon for upstream API fidelity but only affects its AdamW branch; our all-Muon group does not use it.
Two of these had drifted (`[512,1024]` rows, `[3,50]` features), both
making training *cheaper* — which is why nothing complained: a wrong setting that costs money gets
found; one that saves money does not. Stage-1 shape tests protect those experiments; the
long-context tests separately verify Exp2's sampling law, per-row feature caps and paired streams.

## 3. `max_features`

**100, upstream's, a training-distribution setting only** (TabICLv2 §4.1: "up to 100 features
throughout all stages"; every stage script passes `--max_features 100`). It caps the sampled width and
stops noise/missingness columns pushing past it. It does **not** limit inference: `TabICL.__init__`
takes no feature-count parameter — `col_embedder` is a permutation-equivariant induced-set transformer
with learned inducing points and no per-feature weights, so a trained checkpoint accepts any width.

For Exp2's long-context TabICLv2 training, upstream `Prior.adjust_max_features` additionally
limits width by row count: 100 through 10,240 rows, 80 through 20,000, 60 through 30,000,
40 through 40,000, 30 through 50,000, and 20 through 60,000. The resulting per-table ceiling
also reaches native generation and predictability filtering, where zero-padded width affects
the seeded ExtraTrees filter. Credit mechanisms stay within the same planned feature budget.

Protocol 9 retains the uncapped benchmark from 13-10-2026: `protocol.MAX_FEATURES = None`.
All prediction-time eligible columns and all training-fold rows reach every model. The wrappers
do not rank or select columns; the near-constant filter is also off. TabPFN's checkpoint-defined
per-estimator feature budget is enlarged to retain every input column, including preprocessing
expansion; fit fails if upstream still selects a subset. The other upstream transformations remain.
Exp0 tests real 2,986-column PD / 601-column LGD predictions and the widest selected full tables on the target GPU.

**Keep 100.** It is the control variable (a wider training distribution is a second difference from
TabICLv2); training wider and then reporting a win on a wide table would measure feature width, not the
prior. Both width and context-length extrapolation can occur at evaluation time, with the same
training geometry across mixtures of a backbone. Supporting wider input does not guarantee unchanged accuracy
or that every table fits GPU memory; Exp0 checks execution on the selected largest/widest tables.

## 4. Stage choice

Exp1 retains its fixed small-table scratch recipe. Exp2 continues released TabICLv2 with the
upstream stage-3 table law, including short and long tables in one log-uniform distribution;
it does not restart stages 1–2. TabPFN-3 uses the documented differentiable fine-tuning interface
on synthetic tasks. Its private original pretraining recipe is not claimed reproduced.

## 5. Budgets and checkpoint accounting

| Experiment | Updates / run | Main runs total | Permanent checkpoints / run |
|---|---:|---:|---|
| Exp1 | 12,500 | 54 | Existing cadence preserved |
| Exp2 main | 1,000 | 96 | 125, 250, 500, 1k |
| Exp3 main | 1,000 | 48 new + 12 reused Exp2 | Same four milestones as Exp2 |

The 29-10 fixed-rate revision cancels the learning-rate and adapter pilots. The 1,000-update
horizon is a compute-budget choice preserving four mixtures, two rates and three paired seeds;
it is not an evidence-based optimum or a convergence guarantee. Upstream stage 3 used 10,000
updates, TabPFN-Wide used 10,000 with much shorter tables, and Real-TabPFN used 20,000 with
one real table/update. Those budgets cannot be compared by update count alone. Full runs here
consume 64k/16k tasks for TabICL/TabPFN respectively. The full cosine schedule ends at 1,000,
rather than stopping an older longer schedule early. Report the fixed endpoint and all retained
trajectories, never the best checkpoint on the assessment data.

There are 384 permanent Exp2 snapshots and 192 new Exp3 snapshots. At the observed/provisional
training rates, Exp2 extrapolates to 37.3–39.5M credits, excluding benchmark/validation/Exp3.
Full evaluation of every saved checkpoint on all eligible rows/features and five folds creates
68,160 Exp2 dataset-fold-model evaluations and 34,080 for the new Exp3 arms before reference
models and threshold-calibration refits. Benchmarking must be budgeted separately; training
cost alone does not establish affordability. See the source evidence and cost assumptions in
[EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md) §3.2.

## 6. `optimizer`

**Muon for full-parameter TabICL updates; AdamW for TabPFN-3 and the new adapters.**
The optimizer is held fixed across mixture fractions within each backbone. Exp3's adapter
methods have their own predeclared AdamW rates; this is part of the fixed-recipe comparison,
not an optimum selected on this benchmark.

- **Muon** is what TabICLv2 uses, built exactly as upstream builds it (`src/train/optim.py`: one group,
  every parameter, Moonlight scaling; `train.lr` is the one rate — stage 1 `8e-4`, stage 3 `2e-5`). Its
  Newton–Schulz orthogonalisation is a *fixed* ~18 ms per weight matrix per step, so it is 1.40×
  overhead at batch 1 but **1.01× at batch 64** — free at the size we train.
- **AdamW for TabPFN-3**: TabPFN-Wide's continued pretraining and TabPFN's own fine-tuner both use it,
  and `TabPFNTrainer` accepts nothing else ([§8](#8-experiment-2-the-continued-pretraining-knobs)).

Within an experiment every mix of a model trains with the same optimizer, so it is never the variable
being measured.

## 7. The prior knobs

**`prior.credit_fraction`** — the main switch: the probability a synthetic dataset comes from our
credit path rather than the unmodified TabICL prior. `0.0` is the control. Exp1 and Exp2 sweep
`{0, .25, .5, 1}`; Exp3 fixes `0.5` and compares retention methods.
Mixing rather than replacing retains native-prior exposure, a plausible defence against forgetting;
it does not guarantee non-degradation. Tanna 2026 reports collapse under a different setting,
per-dataset supervised adaptation, which motivates measuring retention rather than predicting the
outcome of this synthetic continuation. The 0% arm measures additional generic TabICL-prior training against
the released weights; it is not assumed to improve or match them by construction.

**`prior.encoding`** — whose prediction path a credit table imitates: `tabicl` (TabICL's — gaps filled
with the context mean, `prediction_view`, an LGD target standardised on the context) or `raw` (the
table as recorded — gaps as NaN, the LGD target on [0, 1] — for TabPFN, whose own preprocessing does
the rest). The TabPFN-3 arms set `raw`; `TabPFNTrainer` refuses anything else. The control tables are
upstream's either way.

**`prior.grouping`** — upstream's batch structure (`GraphPrior.get_batch`): Exp1 and Exp2/Exp3
TabPFN-3 use groups/subgroups of 4 sharing row count, context/query split and feature count.
Exp2/Exp3 TabICLv2 uses groups/subgroups of 1, so each long-context task has its own shape and split.
The control draws 2–10 classes per table (`max_classes: 10`). `src/prior/stream.py` builds it;
mixture arms within the same backbone and seed share shape draws and identical native tasks in
unchanged native slots. Different backbone batch sizes do not produce an identical ordered stream.
A credit table keeps its slot's width even when it records the period factor as a column (the column
takes a noise column's place).

**`prior.credit.target` — LGD** (`mode: zoib`, 29-09-2026): the zero-and-one inflated beta regression
of Li, Zhang & Zhao (2020, §2.1), one portfolio per table. `alpha0_mean_sd` / `beta0_mean_sd` /
`gamma0_mean_sd` the intercepts of logit P(0), logit P(1) and the interior mean, normal around the
published −0.54 / −1.46 / 0; `phi_range` the beta precision (log-uniform, contains the published 5);
`loading_*_range` the effect of the recovery driver on each part and `macro_*_range` that of the period
factor (each range contains the published value; the signs are the published ones); `signal_share_range`
how much of the driver the features record; `component_overlap_range` how far the three parts share one
driver; `macro_feature_prob` how often the period factor is a column; `cohort` the number of periods.
The table [PRIORS.md](PRIORS.md) §5.2 gives every published value beside its range.
`target_scaling: standard` standardises the target on the context rows under the `tabicl` encoding, as
`TabICLRegressor` does to a real one (affine: the atoms stay two atoms). `mode: quantile` (dialled atoms)
and `mode: mechanism` (collateral, workout, segments) remain for ablations.

**`prior.credit.target` — PD** (`mode: mechanism`): `mechanism.base_rate_range` + `base_rate_log` the
default rate (log-uniform on [1 %, 50 %]; Brown & Mues 2012's most imbalanced setting to balanced),
`mechanism.rho_range` the Vasicek asset correlation (the Basel IRB range, 0.03 to 0.24),
`mechanism.signal_share_range` the share of risk the features explain (wide: [0.05, 0.95]),
`mechanism.cohort` the periods sharing the standard-normal systematic factor, `macro_feature_prob` how
often the factor is a column; `rules.n_rules_range` hard cut-offs per table (Klein & Hoffart 2026 — *a
position paper with zero experiments, the most speculative component, flag it*); `selection` (the
approved book only: `selection_drop_range`, `selection_sharpness_range`, drawn per table). No label
noise since 29-09-2026: its rates had no source.

**`prior.credit.marginals`** — long-tailed amounts and ratios: a monotone warp on a
`col_fraction_range` share of the continuous columns, strength per column from `strength_range` (both
wide). Monotone, so every split a tree could make — and the dependence on the target — is unchanged.

**`prior.credit.missingness`** — gaps in a `missing_prob` share of tables, in a
`missing_col_fraction_range` share of the columns, `missing_rate_range` missing each, their chance
coupled to observed borrower covariates with a strength drawn from `missing_risk_coupling_range` (all wide: the
literature says gaps are informative, not how often). Filled with the context mean under the `tabicl`
encoding, left NaN under `raw`; `missing_indicators: true` is refused, since TabICL's prediction path
adds no indicator.

**`prior.filter`** — `tabicl` (as shipped), `off` (keep everything), `banded` (keep only difficulty in
`quantile_band`). `apply_to: base` (every experiment) judges only TabICL's tables, as upstream does; the
credit tables are used as specified, since re-selecting them by learnability would change the specified
prior. `all` judges the credit tables too. Report the rejection rate per arm; it makes wall-clock
incomparable across filter settings.

## 8. Continued pretraining and retention

The registered questions, source evidence and limits are in
[EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md) §3. Resolved values are saved per run.
`src/experiments/design.py` owns `RATES`, `MAIN_STEPS`, `MILESTONES`, `FIXED_RETENTION` and
`DESIGN_REVISION`. `apply_design` resolves a predeclared recipe without reading scores or
selection receipts. Retired pilot designs, candidate fields and supplied selection overrides
are rejected. This module is not used by Exp1, whose existing resolved recipes remain unchanged.

**Exp2 has 48 runs per track:** 2 backbones × 4 mixtures × 2 rates × 3 seeds, or 96 overall.
Both rates are reported separately. Neither credit nor OOD results select a rate, mixture or
checkpoint. Legacy development/holdout names remain for Exp1 compatibility and reporting;
past benchmark exposure is disclosed, so these are not newly untouched datasets.

| Backbone | `lr_index=0` | `lr_index=1` | Optimizer | Decay | Warm-up |
|---|---:|---:|---|---:|---:|
| TabICLv2 | 2e-5 | 1e-5 | Native all-parameter Muon | .01 | .01 = 10 updates |
| TabPFN-3 | 1e-5 | 1e-6 | AdamW | .01 | .1 = 100 updates |

TabICLv2 keeps 64 tables/update, 400–60,000 log-uniform rows and 79–81% context.
TabPFN keeps 16 tables/update, 1,024 rows and 30–90% context. Both clip the global unscaled
gradient norm to 1 and use one warm-up/cosine cycle with floor 1e-7. TabICL remains FP32 under
the validated TF32/efficient-attention policy; TabPFN uses FP32 parameters with FP16 AMP and
finite-gradient replay. These are backbone-specific recipes, not matched cross-backbone compute.

The TabICL rates are anchored by its native stage-3 `2e-5` and the `1e-5` single-table result
in [Ning et al., Appendix B](../tfm-library/papers/text/2026/10_Ning_et_al._Test_Time_Compute_for_Tabular_Foundation_Models_Mechanisms_Gains_and_Limits.txt).
The TabPFN rates bracket the public fine-tuner/TabPFN-Wide `1e-5` and the same paper's TabPFN-3
`1e-6` result. Neither source proves an optimal multi-table credit-CPT recipe. Native TabPFN-3
pretraining settings remain undisclosed; the 0% control is generic TabICL-prior exposure.

`train.keep_steps: [125, 250, 500, 1000]` adds permanent milestones independently of rolling saves.
All four receive full-table credit/native-OOD five-fold evaluation; final-step slots remain first.
`max_temp_checkpoints: 2` applies only to nonpermanent saves. `max_steps`,
`schedule_total_steps` and the primary endpoint are all 1,000 for the production recipe.
Intermediate scores are trajectories, not selected winners.

**Exp3 has 24 new runs per track:** 2 backbones × 4 methods × 3 seeds, or 48 overall.
Its mixture is fixed at 50% and its full-model reference rate is always `lr_index=0`.
Twelve matching full-CPT-with-decay Exp2 runs are reused; the better-performing rate is never
chosen retrospectively. There are no `Exp3_*_pilots.yaml` jobs in the current design.

| Method | Optimizer and fixed peak LR | Decay / anchor |
|---|---|---|
| `no_decay` | Native Muon 2e-5 for TabICL; AdamW 1e-5 for TabPFN | Decay 0 |
| `l2sp` | Same full-model optimizer/rate | Decay 0; alpha .003 |
| `diagscale` | AdamW .03 | .01 on query-scale deltas only |
| `lora` | AdamW 1e-4 | .01 on rank-16 Q/V factors only; alpha 16 |

Adapter warm-up is 10% (100 updates), clipping 1, dropout 0, with the same 1,000-update cosine
endpoint and four saved milestones. Native weights are frozen for both adapters; initial
predictions are unchanged. L2-SP is supported by both trainers and always uses the released
checkpoint as its fixed anchor, including after resume. LoRA exports merged native inference
weights plus separate resumable factors. DiagScale injection precedes cache creation.

[Real-TabPFN](../tfm-library/papers/text/2025/07_Garg_et_al._Real_TabPFN_Improving_Tabular_Foundation_Models_via_Continued_Pre_training_With_Real_World_Data.txt)
supports the alpha .003 anchor; Ning et al. supports .03 for DiagScale in a different, single-table
setting. Q/V-only LoRA at 1e-4 is a fixed engineering recipe, not a reproduced optimum from
Ning's all-linear LoRA menu. Ordinary decay shrinks current weights; L2-SP penalizes displacement
from released weights. Compare L2-SP against no-decay to isolate anchoring, and no-decay against
reused Exp2 to isolate ordinary decay. These are fixed-recipe comparisons with no benchmark-driven
hyperparameter search. Keeping most weights frozen is not a guarantee of lower GPU time.

## 9. A note on the LGD R² band

The docs quote "published LGD R² ≈ 0.04–0.15 linear, 0.10–0.25 beta, 0.20–0.43 tree ensembles."
**That range came from this project's own brief, not a verified source** and must not be presented
as a literature result. Earlier notes recorded CatBoost R² of 0.376 on Freddie, 0.501 on heloc and
0.71–0.76 on `lgd_lendingclub`, before the subsequent feature/protocol corrections. Those historical
numbers neither describe the current benchmark nor establish leakage from score magnitude alone.
Use the current versioned benchmark artifacts for results and the dataset audit for feature claims.
