# Config reference — why each setting is what it is

The `config/Exp{1,2,3}_{PD,LGD}.yaml` files hold **values**, one note per knob; this file holds the
**reasoning** — why that value, what evidence backs it, what upstream does, what breaks if you change
it — under the same knob name. The split is deliberate: a config is read while deciding what to
submit, so prose in it makes the settings unfindable. If you catch yourself writing a paragraph above
a knob, it belongs here.

Neighbours: [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md) (what the experiments ask) ·
[PRIORS.md](PRIORS.md) (how the prior is built) · [VSC.md](VSC.md) (the cluster). Library pin `52dab01`.

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

**Exp1 sweeps `credit_fraction {0, .5, 1}` × 3 seeds = 9 arms** (since 28-09-2026; before, it also
crossed three filters and two intensities, 45 arms). The control is TabICLv2's prior and filter exactly;
the credit prior is set from the credit-risk literature (29-09-2026, [PRIORS.md](PRIORS.md) §5), never
from the evaluation datasets, and is the same in Exp0-Exp3. At
`credit_fraction: 0.0` the credit knobs have no effect, so such combinations collapse to one control
per seed (`effective_fingerprint` in `src/utils/config.py`).

**One knob, one home:** a knob in `sweep:` must not also appear in the config body —
`apply_sweep_block` writes over the body, so a body literal is dead text that reads like a setting.
`test_a_swept_knob_has_exactly_one_home` checks all six files. **Open one group at a time:** with every
lever moving you cannot attribute an effect to any of them.

## 2. The prior's shape is not ours to choose

`n_rows_range`, `n_features_range`, `max_features`, `n_nodes_range`, `train_frac_range`. The project
rests on one sentence — *the only difference from TabICLv2 is the prior's credit structure* — so these
five are copied verbatim from `scripts/train_v2_reg_stage1.sh`:

| knob | ours | upstream stage 1 |
|---|---|---|
| `n_rows_range` | `[1024, 1024]` | `--max_seq_len 1024`, no `--min_seq_len` |
| `n_features_range` | `[1, 100]` | `--min_features 1 --max_features 100` |
| `n_nodes_range` | `[2, 33]` | `--min_n_nodes 2 --max_n_nodes 32` |
| `train_frac_range` | `[0.3, 0.9]` | `--min_train_size 0.3 --max_train_size 0.9` |

**Rows are 1,024 exactly, not a range** (`sample_seq_len` returns `max_seq_len` when no min is passed);
only the train/test split varies. Two of these had drifted (`[512,1024]` rows, `[3,50]` features), both
making training *cheaper* — which is why nothing complained: a wrong setting that costs money gets
found; one that saves money does not. `test_prior_shape_matches_upstream_stage_one` pins all five.

## 3. `max_features`

**100, upstream's, a training-distribution setting only** (TabICLv2 §4.1: "up to 100 features
throughout all stages"; every stage script passes `--max_features 100`). It caps the sampled width and
stops noise/missingness columns pushing past it. It does **not** limit inference: `TabICL.__init__`
takes no feature-count parameter — `col_embedder` is a permutation-equivariant induced-set transformer
with learned inducing points and no per-feature weights, so a trained checkpoint accepts any width.

What limits inference is the evaluation protocol, and it is the same number: `protocol.MAX_FEATURES =
100` (protocol 4, 30-09-2026) keeps a wider table's 100 columns with the strongest rank association with
the target, scored on each fit's own rows — for every model alike, ours, the released ones and the
baselines (`protocol.rank_columns`; docs/EXPERIMENTAL_DESIGN.md §5 for the literature behind the rule).
It binds on five datasets. Every model gets the whole training pool as context (the 1,024-row cap of
protocol 2 is gone); Exp0 and Exp1 are also scored with contexts capped at 1,024 rows
(`eval.context_caps`).

**Keep 100.** It is the control variable (a wider training distribution is a second difference from
TabICLv2); training wider and then reporting a win on a wide table would measure feature width, not the
prior; and nothing in the architecture wants it changed. Since the evaluation cap is the same 100, no
model meets a table wider than it trained on; the row count remains the exposure (`heloc`'s training
pool is 46× the training tables), matched across arms, and measured by the capped-context scores.

## 4. Why one stage

Upstream trains in three, and what changes between them is the row count and the LR (stage 1: 500k
steps, 1,024 rows, lr 8e-4, clip 10 → stage 3: 10k steps, up to 60k rows, lr 2e-5, clip 1). **Rows
enter the cost quadratically** through the 12-block ICL predictor, so a stage-3 step costs ~120× a
stage-1 step: keep upstream's 90.9/7.3/1.8 step split on our budget and stage 3 becomes ~227 steps —
1.8% of the steps but ~61% of the cost — and at 1/40th the LR those 227 steps would move the weights
almost not at all. Upstream's stages 2–3 are low-rate adaptation to long context, not more prior
learning. **So: stage 1 only.** Evaluation gives every model the whole training pool (protocol 4), so
our checkpoints meet contexts far longer than they trained on — the same for every arm, which keeps the
arm-vs-control contrast matched, but not for the released model, which trained to 60,000 rows. Long
context belongs in Exp3 on the winning prior — `init.strategy: full` + `pretrained_path`
is upstream's `--checkpoint_path … --only_load_model True`, no new code.

## 5. Training budget

**`max_steps`** is the compute lever, and the limit is credits: at batch 64, 12,500 steps ≈ 800k
datasets/arm (2.5% of upstream's stage 1); a B200 costs 437.5 credits/GPU-minute, so Exp1's 9 arms
run tens of millions of credits and doubling `max_steps` doubles that. The three experiments spend
deliberately differently — **Exp1 12,500** (a ranking), **Exp2 10,000** (continued pretraining, the
length of both published recipes), **Exp3 200,000** (Exp1's mix at length, 2 seeds; TabICLv2's log-loss
settles only after ~200,000 steps, EXPERIMENTAL_DESIGN §3). If the budget will not stretch, **cut
`max_steps`, never the prior shape** — a shorter run is just shorter; a cheaper prior is a confound.

**`micro_batch_size`** is a correctness constraint, not a memory setting: `Trainer.validate_micro_batch`
raises if datasets in one micro-batch disagree on sequence length or train/test split, both drawn per
group, so upstream keeps `micro == batch_size_per_gp` (Exp1/Exp3: micro 4 = group 4). Gradient
accumulation then runs `ceil(batch/micro)` passes and averages them — mathematically identical to one
big batch, only speed/memory differ. A bigger GPU does not buy a bigger micro-batch.

**`amp: true`** is a real ~2× speed-up (upstream uses it everywhere), and the attention backend is
**pinned** (`src/models/backends.py`): PyTorch's cuDNN fused MHA graph raised on a B200 at batch 64
under AMP, so it is excluded; flash, mem-efficient and math remain, and the run card logs which is used.

## 6. `optimizer`

**Muon for TabICL (Exp1, Exp3, and Exp2's TabICLv2 arms), AdamW for TabPFN-3 (Exp2)** — each model
trains with the optimizer its own recipe uses; neither is a free choice.

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
`{0, .5, 1}`; Exp3 runs `0.0` and Exp1's chosen mix.
Mixing rather than replacing is also the defence against the collapse Tanna 2026 reports for TabICL
under aggressive adaptation — a model that keeps seeing the original prior keeps the distribution it
was built for.

**`prior.encoding`** — whose prediction path a credit table imitates: `tabicl` (TabICL's — gaps filled
with the context mean, `prediction_view`, an LGD target standardised on the context) or `raw` (the
table as recorded — gaps as NaN, the LGD target on [0, 1] — for TabPFN, whose own preprocessing does
the rest). The TabPFN-3 arms set `raw`; `TabPFNTrainer` refuses anything else. The control tables are
upstream's either way.

**`prior.grouping`** — upstream's batch structure (`GraphPrior.get_batch`): groups of 4 datasets share
the row count and the context/query split, and (subgroup = group, `subgroup_size: 4`) the feature
count; the control draws 2–10 classes per table (`max_classes: 10`). `src/prior/stream.py` builds it,
and every arm with the same seed sees the same TabICL tables in the same slots (common random numbers).
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
coupled to the target with a strength drawn from `missing_target_coupling_range` (all wide: the
literature says gaps are informative, not how often). Filled with the context mean under the `tabicl`
encoding, left NaN under `raw`; `missing_indicators: true` is refused, since TabICL's prediction path
adds no indicator.

**`prior.filter`** — `tabicl` (as shipped), `off` (keep everything), `banded` (keep only difficulty in
`quantile_band`). `apply_to: base` (every experiment) judges only TabICL's tables, as upstream does; the
credit tables are used as specified, since re-selecting them by learnability would change the specified
prior. `all` judges the credit tables too. Report the rejection rate per arm; it makes wall-clock
incomparable across filter settings.

## 8. Experiment 2 — the continued-pretraining knobs

Exp2 continues the pretraining of the **released TabICLv2 and TabPFN-3 weights** on a mixture of the
original prior and ours. The recipe is a nuisance parameter that must be good enough not to confound
"does our prior help", so each model uses **the recipe its own literature uses for continued
pretraining, unchanged** (30-09-2026; the stage-A search that used to choose among candidates around
these values was dropped): TabICLv2 its authors' stage 3, TabPFN-3 TabPFN-Wide's. The published
recipes, all in the library or the installed package (bold: the one each model uses):

| | TabPFN's own fine-tuner (tabpfn 9.0.0) | **TabPFN-Wide (Kolberg 2026) → TabPFN-3** | Real-TabPFN (Garg 2025) | TabICLv2 stage 2 / **stage 3 → TabICLv2** |
|---|---|---|---|---|
| optimiser | AdamW (torch betas) | **AdamW** | AdamW | Muon / **Muon** |
| learning rate | 1e-5 | **1e-5** | 3e-7 | 1e-4 / **2e-5** |
| schedule | 10 % warm-up → cosine | **warm-up → cosine** | warm-up → cosine | 1 % warm-up → cosine to 1e-7 |
| weight decay | 0.01 | **1e-4** | — | 0.01 / **0.01** |
| grad clip | 1.0 | **1.0** | — | 10 / **1.0** |
| batch | 1 dataset | **16 datasets** | — | 64 / **64 datasets** |
| length | early stopping | **plateau by 10,000 steps** | — | 40,000 / **10,000** |

Kolberg's paper states "linear warm-up and cosine decay" without the warm-up's length; we use 10 %, the
TabPFN fine-tuner's. Every 2,500-step checkpoint is scored (a learning curve on real data); the final one
is the result, since both recipes fix the length. `eval.headline_step` can name another checkpoint as the
result, and `results_plots.load_results` honours it; no config sets it.

**`architecture`** — set per arm: `tabicl` (our `Trainer`, `src/train/loop.py`) or `tabpfn3`
(`TabPFNTrainer`, `src/train/tabpfn_trainer.py`, built on TabPFN's own fine-tuning path);
`scripts/pretrain.py` dispatches on it and the benchmark scores each with the wrapper of its kind.

**`init.pretrained_path`** + **`init.strict_load: true`** — the released file, repo-relative; found in the
repo or in `paths.pretrained_dir()` on project storage (`paths.find_pretrained`). Strict: every tensor
must load (391/391 classifier, 347/347 regressor), or the run stops. TabPFN's own loader is strict by
construction. **`init.strategy: full`** — every layer trains.

**TabPFN-3 only:** `train.batch_size: 16` (tasks per update, TabPFN-Wide's), `train.n_estimators: 2`
(ensemble members per training forward, the fine-tuner's `n_estimators_finetune`),
`train.activation_checkpointing` (default true, as the fine-tuner), `train.regression_loss_weights`
(default the fine-tuner's: CRPS 1 + MSE 1, no NLL term), `train.amp` (float16 autocast + GradScaler,
as the fine-tuner). AdamW only.

**`train.l2sp_alpha: 0`** — L2-SP (`Ω(w) = (α/2)‖w − w₀‖²` toward the released weights, Real-TabPFN's
α = 0.003) is implemented and tested but not in the design: neither recipe Exp2 follows uses it, and a
second axis would confound the first. A later ablation; forgetting is measured instead, on the
out-of-domain suites, in every progress curve and benchmark.
The drift L2-SP penalises is *measured* in every run instead (`weights.csv`: distance from the
released weights per block).

**Why no partial freezing or LoRA:** Rubachev 2025 found full fine-tuning matches every
parameter-efficient variant on TabPFNv2 while converging fastest; searching the whole TabICL dump for
"lora" returns zero matches; and fine-tuning TabICL is documented as fragile (Tanna 2026: full
per-dataset SFT drops TabZilla accuracy 0.873 → 0.567, while TabPFN survives) — which is why TabICLv2
continues at its authors' own stage-3 rate, the lowest upstream ever trained it at, and why the
out-of-domain suites are scored for every arm.

**Budget (agreed 29-09-2026: ~400 GPU-h):** 18 arms per track plus their benchmarks. Experiment 0
measures the per-arm and per-slot hours before Experiment 2 is submitted.

## 9. A note on the LGD R² band

The docs quote "published LGD R² ≈ 0.04–0.15 linear, 0.10–0.25 beta, 0.20–0.43 tree ensembles."
**That range came from this project's own brief, not a verified source** — it is used only as a sanity
band and must be cited or dropped before it appears in a paper. Our own measurements so far: CatBoost
0.376 on Freddie and 0.501 on heloc (inside the tree band), but `lgd_lendingclub` scores 0.71–0.76,
well outside it — most likely a feature derived from the recovery amount is surviving the recipe.
