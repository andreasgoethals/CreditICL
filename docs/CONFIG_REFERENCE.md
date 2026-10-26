# Config reference — why each setting is what it is

The `config/Exp{1,2,3}_{PD,LGD}.yaml` files hold **values**, one note per knob; this file holds the
**reasoning** — why that value, what evidence backs it, what upstream does, what breaks if you change
it — under the same knob name. The split is deliberate: a config is read while deciding what to
submit, so prose in it makes the settings unfindable. If you catch yourself writing a paragraph above
a knob, it belongs here.

Neighbours: [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md) (what the experiments ask) ·
[PRIORS.md](PRIORS.md) (how the prior is built) · [VSC.md](VSC.md) (the cluster). Original source-audit pin `de324efcca55be392feb0c01830b3b6ce364587f`; the 24-10-2026 Experiment 2 audit uses TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`.

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

**Arms that are not a cross product** (Exp0, Exp2) are named presets: `sweep.arm` lists them and the
`arms:` block gives each one's dotted overrides (`arms.tabpfn3_adamw_1e-5: {architecture: tabpfn3,
train.lr: 1.0e-5, ...}`), applied after the sweep (`apply_arm_preset`). An override must name a key the
config body already has, so a typo raises instead of doing nothing.

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
| Exp2 | 10,000 | 96 | 125, 250, 500, 1k, 2.5k, 5k, 7.5k, 10k |
| Exp3 pilots | 2,500 (10k schedule prefix) | 36 | 2,500 |
| Exp3 main | 10,000 | 48 new + 12 reused Exp2 | Same eight milestones as Exp2 |

10,000 is a registered budget with upstream precedent, not an established optimum. Learning
curves assess under/over-adaptation. At 64/16 tables per update, full runs consume 640k/160k
tasks for TabICL/TabPFN respectively. Benchmark jobs are separate and can dominate total cost.

## 6. `optimizer`

**Muon for full-parameter TabICL updates; AdamW for TabPFN-3 and the new adapters.**
The optimizer is held fixed across mixture fractions within each backbone. Exp3's adapter
methods have their own development-selected AdamW rates; this is part of the method comparison.

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
outcome of this synthetic continuation. The 0% arm measures additional native-prior training against
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

The authoritative registered design and source limitations are in
[EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md) §3. Resolved values are saved per run.
`src/experiments/design.py` maps stable candidate indices to numeric settings; changing a numeric
setting invalidates validation/selection evidence rather than silently reusing a run.

Exp2: 48 runs per track (2 backbones × 4 mixtures × 2 rates × 3 seeds).
TabICLv2 uses Muon, peak 2e-5/1e-5, decay .01, warm-up .01, 64 tables/update,
400–60,000 log-uniform rows and 79–81% context. TabPFN-3 uses AdamW, peak 1e-5/1e-6,
decay .01, warm-up .1, 16 tables/update, 1,024 rows and 30–90% context. Both clip the global
unscaled gradient norm to 1. TabICL remains FP32 under the validated TF32/efficient-attention
policy; TabPFN uses FP32 parameters with FP16 AMP and finite-gradient replay. These are
backbone-specific recipes, not matched cross-backbone compute.

`train.keep_steps: [125,250,500,1000,2500,5000,7500,10000]` adds permanent milestones
independently of rolling saves. All eight receive full-table credit/native-OOD five-fold evaluation;
final-step slots remain first. `max_temp_checkpoints: 2` applies only to nonpermanent saves.
The primary endpoint is 10,000; intermediate scores are trajectories, not selected winners.

Exp3 replaces the old 200k scratch grid. `Exp3_*_pilots.yaml` has 18 pilots per task,
seed 991, max_steps 2500 and schedule_total_steps 10000, development benchmarks only.
`Exp3_*.yaml` has 24 main runs per task, max_steps 10000, three seeds, fixed 50% mixture.
Full-CPT-with-decay baselines reuse 12 selected Exp2 runs. Candidate settings:

| Method | Optimizer and peak LR | Decay / anchor |
|---|---|---|
| no_decay | Exp2-selected full-model optimizer/LR | decay 0 |
| l2sp | same full-model optimizer/LR | decay 0; alpha .0003/.003/.03 |
| diagscale | AdamW, LR .001/.01/.03 | .01 on query-scale deltas only |
| lora | AdamW, LR 1e-5/1e-4/1e-3 | .01 on rank-16 Q/V factors only; alpha 16 |

Adapter warm-up is 10%, clipping 1, dropout 0. Native weights are frozen for both adapters;
initial predictions are unchanged. L2-SP is supported by both trainers and always uses the
released checkpoint as its fixed anchor, including after resume. LoRA exports merged native
inference weights plus separate resumable factors. DiagScale injection precedes cache creation.

Selection receipts are created by `python -m src.experiments.selection --stage learning-rate`
and then `--stage adapters` after their required benchmarks. Missing/stale receipts block jobs.
`allow_placeholders=True` is for analysis/planning only and marks unresolved recipes explicitly.

## 9. A note on the LGD R² band

The docs quote "published LGD R² ≈ 0.04–0.15 linear, 0.10–0.25 beta, 0.20–0.43 tree ensembles."
**That range came from this project's own brief, not a verified source** and must not be presented
as a literature result. Earlier notes recorded CatBoost R² of 0.376 on Freddie, 0.501 on heloc and
0.71–0.76 on `lgd_lendingclub`, before the subsequent feature/protocol corrections. Those historical
numbers neither describe the current benchmark nor establish leakage from score magnitude alone.
Use the current versioned benchmark artifacts for results and the dataset audit for feature claims.
