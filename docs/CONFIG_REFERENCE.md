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
8. [Experiment 2 — the fine-tuning knobs](#8-experiment-2-the-fine-tuning-knobs)
9. [A note on the LGD R² band](#9-a-note-on-the-lgd-r-band)

---

## 1. How the sweep works

Any setting may be a single value or a list; a list means one run per value, and all lists are
**crossed**. Names ending in `_range` are literal `[low, high]` intervals sampled *from*, never swept
— to sweep one, nest it: `[[0.0, 0.1], [0.1, 0.3]]`. Inspect the expansion with
`python scripts/pretrain.py --config config/Exp1_PD.yaml --list`.

**Exp1 crosses `credit_fraction {0,.5,1} × filter {tabicl,banded,off} × intensity {mild,aggr}`, then
× 3 seeds.** At `credit_fraction: 0.0` the credit knobs have no effect, so those combinations collapse
to one control per seed (`effective_fingerprint` in `src/utils/config.py`) — which is what stops the
control being run under several names. Net: **45 arms**.

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

What limits inference is *our* eval code, at a different number: `TFM_MAX_FEATURES = 500` (keep the 500
highest-variance columns; `src/eval/ood.py` skips above it). Both limits live on the **shared** base
class, so `crediticl` and `tabiclv2` get identical treatment — same rule as the 1,024-row context cap.

**Keep 100.** It is the control variable (a wider training distribution is a second difference from
TabICLv2); training wider and then reporting a win on a 256-column table would measure feature width,
not the prior; and nothing in the architecture wants it changed. The honest exposure: `base_modelisation`
has 256 columns (2.6× the training width), so the model does extrapolate — but the exposure is
**matched** (both sides get the same 256 columns), and much milder than the 46× the row count was
before it was capped. Record feature width per dataset alongside `context_cap`; if a wide-table effect
ever shows, the answer is an Exp3 stage that trains wider, not a silent bump here.

## 4. Why one stage

Upstream trains in three, and what changes between them is the row count and the LR (stage 1: 500k
steps, 1,024 rows, lr 8e-4, clip 10 → stage 3: 10k steps, up to 60k rows, lr 2e-5, clip 1). **Rows
enter the cost quadratically** through the 12-block ICL predictor, so a stage-3 step costs ~120× a
stage-1 step: keep upstream's 90.9/7.3/1.8 step split on our budget and stage 3 becomes ~227 steps —
1.8% of the steps but ~61% of the cost — and at 1/40th the LR those 227 steps would move the weights
almost not at all. Upstream's stages 2–3 are low-rate adaptation to long context, not more prior
learning. **So: stage 1 only, and cap the *evaluation* context to 1,024 for both models from one shared
setting.** Long context belongs in Exp3 on the winning prior — `init.strategy: full` + `pretrained_path`
is upstream's `--checkpoint_path … --only_load_model True`, no new code.

## 5. Training budget

**`max_steps`** is the compute lever, and the limit is credits: at batch 64, 12,500 steps ≈ 800k
datasets/arm (2.5% of upstream's stage 1); a B200 costs 437.5 credits/GPU-minute, so Exp1's 45 arms
run tens of millions of credits and doubling `max_steps` doubles that. The three experiments spend
deliberately differently — **Exp1 12,500** (a ranking), **Exp2 10,000** (a fine-tune, matching Kolberg's
plateau), **Exp3 100,000** (the converged number on the winner). If the budget will not stretch, **cut
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

**Muon from scratch (Exp1/Exp3), AdamW to fine-tune (Exp2)** — and neither is a free choice.

- **Muon** is what TabICLv2 uses (`lr 8e-4`). Its Newton–Schulz orthogonalisation is a *fixed* ~18 ms
  per weight matrix per step, so it is 1.40× overhead at batch 1 but **1.01× at batch 64** — free at the
  size we train. (It took three runs to see this, because every earlier measurement was at batch 1.)
- **AdamW for Exp2**, for a mechanical reason: under `optimizer: muon`, `train.lr` is only the rate of
  Muon's *auxiliary* AdamW half, so sweeping it would move almost nothing and the LR sweep would falsely
  read as "learning rate doesn't matter". Both continued-pretraining papers use AdamW anyway ([§8](#8-experiment-2-the-fine-tuning-knobs)).

The optimizer is held fixed *within* each experiment, so it is never the variable being measured.

## 7. The prior knobs

**`prior.credit_fraction`** — the main switch: the probability a synthetic dataset comes from our
credit path rather than the unmodified TabICL prior. `0.0` is the control. Exp1 sweeps `{0, .5, 1}`;
Exp2 sweeps `{0, .25, .5, .75, 1}` (finer, because "how much of ours to add" *is* Exp2's question).
Mixing rather than replacing is also the defence against the collapse Tanna 2026 reports for TabICL
under aggressive adaptation — a model that keeps seeing the original prior keeps the distribution it
was built for.

**`prior.grouping`** — TabICL samples a meta-distribution per **group** of 4 datasets, values per
**subgroup** of 2, then a graph per dataset, so a batch is relatives (some uniformly small, some
uniformly hard). **On by default** — the control is supposed to *be* TabICL, so grouping makes it more
faithful; `group_size: 1` is the ablation. A group sharing hyperparameters *is* a small domain, which
is on-topic.

**`prior.credit.target` — LGD:** `mode: quantile` sets boundary mass exactly (best for a controlled
test); `mechanism` builds a latent loss fraction and lets the mass *emerge* (the true economic story).
`shape_ab_range` are Kumaraswamy shape params (`a,b<1` → U; `a,b>1` → hump); `boundary_mass_range` is
mass per atom; `atom_prob` whether each atom exists; `signal_strength` dilutes the feature link
*before* ranking so shape and difficulty move independently. The target is **always** hard-clipped to
[0,1] — LGD is a fraction and cannot leave it, and nothing in the original prior knows that. Boundary
mass has to be a *family*: our seven real files range from 73% mass (heloc) to 0% (axa).

**`prior.credit.target` — PD:** `base_rate_range` targets real data's 6.7%–40% (the original prior
sits at 0.500); `signal_strength` because credit PD lives at AUC 0.70–0.85; `flip_pos_to_neg`
(asymmetric — cures book as non-default far more than the reverse); `rules` (Klein & Hoffart 2026 — *a
position paper with zero experiments, the most speculative component, flag it*); `selection`
(reject-inference truncation, which no general prior produces); `missingness` (target-linked — a thin
file is a risk signal, which TabICLv2 mean-imputes away); `max_cat_size` (Purucker 2026: the
GBDT-over-TFM margin grows with high-cardinality columns, ρ=+0.47).

**`prior.filter`** — `tabicl` (as shipped), `off` (keep everything), `banded` (keep only difficulty in
`quantile_band`, aiming at credit's low-signal range). `banded` is a **removal**, so it cannot be
accused of adding capacity — but it contradicts a published convergence result and must beat that
argument. Report the rejection rate per arm; it makes wall-clock incomparable across filter settings.

## 8. Experiment 2 — the fine-tuning knobs

Exp2 starts from the **released TabICLv2 weights** and continues training on a mixture of the original
prior and ours. Everything here is a nuisance parameter that must be set well enough not to confound
"how much of ours to add". Two published continued-pretraining recipes bracket it, both in the library
(verified against `tfm-library` `52dab01`):

| | Real-TabPFN (Garg 2025) | TabPFN-Wide (Kolberg 2026) | TabICLv2 stage 3 |
|---|---|---|---|
| optimiser | AdamW | AdamW | Muon |
| learning rate | **3e-7** | **1e-5** | 2e-5 |
| schedule | warm-up → cosine | warm-up → cosine | cosine w/ restarts |
| weight decay | — | **1e-4** | — |
| grad clip | — | **1.0** | 1.0 |
| L2-SP | **α = 0.003** | — (plain wd) | — |

**The published rates span 3e-7 to 2e-5 — two orders of magnitude, and neither CPT paper tuned them.**
That disagreement is why `train.lr` is **swept** (`{1e-6, 1e-5}`), not picked; since we fine-tune on
*synthetic* data we sit in Kolberg's regime (the hotter end).

**`train.l2sp_alpha`** — `Ω(w) = (α/2)‖w − w₀‖²` toward the released checkpoint `w₀`, penalising the
drift that continued pretraining risks (learning credit by forgetting everything else). From Li et al.
2018; used by Real-TabPFN at α = 0.003, swept here against 0 (off). Correctness details in
`src/train/loop.py`: the penalty gradient is written directly onto `.grad` (not added to the loss —
otherwise the micro-batch loop applies it `n_micro` times and the effective α depends on micro-batch
size); applied after `scaler.unscale_` and before the clip; and `α > 0` with `init.strategy: scratch`
raises (there is no starting point).

**`init.strategy`** — `scratch` (random init; Exp1/Exp3), or three warm-start depths for Exp2:
`full` (every parameter — both CPT papers do this, so it is the default), `icl_only` (freeze the column
+ row encoders, train the ICL stack + target embeddings + head), `head_only` (the last layer alone).
Freezing mirrors TabICL's own `_finetune/base.py` (`freeze_col`/`freeze_row`/`freeze_icl`), with one
improvement: our change is to the *target* distribution, so `src/train/adapt.py` always keeps the
target-side parameters trainable (`y_embed_in`, `y_embed_icl`, `out_ln`, `out_mlp`, `row_ln`,
`row_cls_tokens`) even under a freeze — a naive "freeze the column stage" would freeze `y_embed_in`,
the very parameter that must adapt to a bounded target.

**Why no LoRA, and why "be gentle":** searching the whole TabICL dump for "lora" returns zero matches
(nothing upstream to validate against), Rubachev 2025 found full fine-tuning matches every
parameter-efficient variant on TabPFNv2 while converging fastest, and it would be a third confound.
And **fine-tuning TabICL is documented as dangerous** — Tanna 2026 reports full SFT drops TabZilla
accuracy 0.873 → 0.567 (TabPFN survives intact), a strong architecture×adaptation interaction on the
bad side for TabICL. That, plus the v2 authors' own very-low stage-3 LR, is *why* freeze depth is swept
and the LR is kept low. **Loading the released weights** uses the upstream `tabicl` package (they only
load into the code that saved them — see [PRIORS §7](PRIORS.md#7-open-item)); `load_pretrained` refuses
to run when fewer than half the tensors match, rather than silently training a random init.

**Budget:** `5 × 3 × 2 × 2 = 60` arms, one seed — at ~3.5 h/arm that is already ~210 GPU-hours, and
repeating a screen three times to denoise arms that will be discarded is the wrong place to spend
seeds. Once the mixture axis has a winner, re-run *that* configuration with three seeds.

## 9. A note on the LGD R² band

The docs quote "published LGD R² ≈ 0.04–0.15 linear, 0.10–0.25 beta, 0.20–0.43 tree ensembles."
**That range came from this project's own brief, not a verified source** — it is used only as a sanity
band and must be cited or dropped before it appears in a paper. Our own measurements so far: CatBoost
0.376 on Freddie and 0.501 on heloc (inside the tree band), but `lgd_lendingclub` scores 0.71–0.76,
well outside it — most likely a feature derived from the recovery amount is surviving the recipe.
