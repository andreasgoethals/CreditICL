# Config reference — why each setting is what it is

The six files in `config/` hold **values**; this file holds the **reasoning**. A knob in a
config carries at most a one-line note saying what it is. Everything else — why that value,
what evidence backs it, what upstream does, and what breaks if you change it — is here, under
the same knob name.

That split is deliberate and it has slipped twice. A config is read while deciding what to
submit, and 130 lines of prose in a 250-line settings file makes the settings unfindable. If
you catch yourself writing a paragraph above a knob, it belongs in this file.

| file | holds |
|---|---|
| `config/Exp{1,2,3}_{LGD,PD}.yaml` | the values, one note per knob |
| **this file** | the reasoning behind them |
| [`EXPERIMENTAL_DESIGN.md`](EXPERIMENTAL_DESIGN.md) | what the three experiments ask |
| [`PRIORS.md`](PRIORS.md) | how the credit prior is built |
| [`VSC.md`](VSC.md) | the cluster, its limits, and how to submit |

---

## How the sweep works

Any setting may be a single value or a list. A list means one run per value, and all lists are
**crossed**. Names ending in `_range` are literal `[low, high]` intervals sampled *from*, never
swept — to sweep one, nest it: `boundary_mass_range: [[0.0, 0.1], [0.1, 0.3]]`.

```bash
python scripts/pretrain.py --config config/Exp1_LGD.yaml --list
```

**Exp1 crosses to 32 combinations but expands to 25 runs x 3 seeds = 75.** At
`credit_fraction: 0.0` the three credit knobs have no effect, so those 8 combinations collapse
to one control arm. `effective_fingerprint` in `src/utils/config.py` does the collapsing; it is
what stops the control being run 8 times under 8 different names.

### One knob, one home

A knob that appears in `sweep:` **must not** also appear in the config body.
`apply_sweep_block` writes the sweep list over whatever is below it, so a body literal is dead
text that reads like a setting — and `prior.credit_fraction: 0.2` sat under `prior:` for months
labelled "SWEPT above; this value is only a fallback". There is no fallback: nothing reads a
config without expanding the grid first. `apply_sweep_block` now raises on the duplicate, and
`test_a_swept_knob_has_exactly_one_home` checks all six files.

The one consumer that genuinely read the raw block was the GPU benchmark. It now expands the
grid and takes arm 0, so it benchmarks the control prior **on purpose** rather than by falling
back to a default.

**Open one group at a time.** Crossing is multiplicative, and with every lever moving you
cannot attribute an effect to any of them.

---

## The prior's SHAPE is not ours to choose

`n_rows_range`, `n_features_range`, `max_features`, `n_nodes_range`, `train_frac_range`.

The whole project rests on one sentence: **the only difference from TabICLv2 is the prior's
credit structure.** These five knobs are the prior's shape, and they are copied from
`scripts/train_v2_reg_stage1.sh` in the pinned tfm-library:

| knob | ours | upstream stage 1 |
|---|---|---|
| `n_rows_range` | `[1024, 1024]` | `--max_seq_len 1024`, **no `--min_seq_len`** |
| `n_features_range` | `[1, 100]` | `--min_features 1 --max_features 100` |
| `max_features` | `100` | `--max_features 100` |
| `n_nodes_range` | `[2, 33]` | `--min_n_nodes 2 --max_n_nodes 32` |
| `train_frac_range` | `[0.3, 0.9]` | `--min_train_size 0.3 --max_train_size 0.9` |

**Rows are 1,024 EXACTLY, not a range.** `sample_seq_len` opens with
`if min_seq_len is None: return max_seq_len`, and stage 1 passes no `--min_seq_len`. Only the
train/test split varies. Two of these had drifted — `[512, 1024]` rows and `[3, 50]` features
— and both made training *cheaper*, which is why nothing ever complained. A wrong setting that
costs money gets found; one that saves money does not.
`test_prior_shape_matches_upstream_stage_one` now pins all five.

---

## `prior.max_features` — 100, and where that limit actually bites

**It is upstream's, stated in the paper, and it is a TRAINING-DISTRIBUTION setting only.**
TabICLv2 §4.1: *"We retain the three-stage structure of TabICL that progressively expands the
size of pretraining datasets, **with up to 100 features throughout all stages**."* All twelve
stage scripts in the pinned library pass `--max_features 100` — v1 and v2, classifier and
regressor, every stage.

### Where the number is applied

| where | what it does |
|---|---|
| `src/prior/generator.py` | caps the sampled width: `randint(lo, min(hi, max_features) + 1)` |
| `src/prior/noise_features.py` | `n_add = min(n_add, max_features - X.shape[1])` — junk columns cannot push past it |
| `src/prior/targets/pd.py` | missingness indicators are added only if they still fit under it |
| `src/prior/dataset.py` | **not applied.** The batch tensor is padded to the widest dataset IN THE BATCH, not to `max_features` |
| upstream's `adjust_max_features` | lowers it further for long sequences (80 at 20k rows, 20 at 60k) — a memory guard, never a raise |

### It does NOT limit inference, and here is why

**`TabICL.__init__` takes no feature-count parameter at all.** Its constructor takes
`max_classes`, `num_quantiles`, `embed_dim`, `col_num_inds` — nothing about feature width.
`col_embedder` is an **induced set transformer over the feature axis**: permutation-equivariant,
with 128 learned inducing points and no per-feature weights. There is therefore no
architectural maximum, and a trained checkpoint accepts any width.

What limits inference is **our** evaluation code, and it is a different number:

| where | value | behaviour above it |
|---|---|---|
| `src/eval/baselines.py` `TFM_MAX_FEATURES` | **500** | keep the 500 highest-variance columns |
| `src/eval/ood.py` `max_features` | **500** | skip the dataset entirely |
| upstream's sklearn wrapper | none | no feature cap anywhere |

Both evaluation limits live on the **shared** base class, so `crediticl` and `tabiclv2` get
the identical treatment — the same rule as the 1,024-row context cap.

### Is 100 optimal? Reassessed, and kept

**Keep it.** Three reasons, in order of weight:

1. **It is the experiment's control variable.** The project's claim is that the *only*
   difference from TabICLv2 is the prior's credit structure. A wider training distribution is
   a second difference, and the comparison stops being clean.
2. **It would bias the comparison in our favour.** The released `tabiclv2` we score against was
   trained at 100. Training ours at 200 and then reporting that ours does better on a
   256-column credit table would be measuring the feature width, not the prior.
3. **Nothing in the architecture wants it changed.** There is no parameter to change.

### The exposure this leaves, stated honestly

`base_modelisation` has **256 columns** — 2.6x the training width — so at inference the model
does extrapolate past what it saw. That is real, and it is worth being precise about how bad
it might be:

- TabPFN-Wide (Kolberg et al. 2026) reports TabICL "unable to reliably separate signal from
  noise, quickly converging toward random guessing" as feature dimensionality grows. **But
  that is TabICL v1, in an HDLSS needle-in-a-haystack regime with thousands of SNPs and ~1 %
  of them causal** — it does not predict what 256 credit columns do, and it should not be
  quoted as if it did.
- TabICLv2's own authors cite that paper for "extreme feature counts", i.e. they treat wide
  tables as a separate problem requiring continued pre-training — not something to fix by
  nudging `max_features`.
- The exposure is **matched**: both columns of the comparison get the same 256 columns. This
  is the same argument as the context cap, and it is much milder — 2.6x here against the 46x
  the row count was before it was capped.

**What to do instead of changing it:** record the feature width per evaluated dataset
alongside `context_cap`, so a reader can see which rows are extrapolating. If a wide-table
effect ever shows up, the answer is Exp2 with a stage that trains wider — not a silent change
to the screening tier.

---

## Why one stage, and not TabICLv2's three

Upstream trains in three, and what changes between them is the row count and the learning rate:

| stage | steps | share | rows | lr | clip |
|---|---|---|---|---|---|
| 1 | 500,000 | 90.9 % | 1,024 exactly | 8e-4 | 10 |
| 2 | 40,000 | 7.3 % | log-uniform 400-10,240 | 1e-4 | 10 |
| 3 | 10,000 | 1.8 % | log-uniform 400-60,000 | 2e-5 | 1 |

**Proportional in STEPS is wildly disproportionate in COMPUTE.** Rows enter the cost twice:
linearly through the column and row encoders, and *quadratically* through the 12-block ICL
predictor, which attends across rows. Calibrating `cost(n) = a.n + b.n^2` at n=1,024, a stage-2
step costs ~7x a stage-1 step and a stage-3 step ~120x. Keep the 90.9/7.3/1.8 split on 12,500
steps and stage 3 becomes 227 steps — **1.8 % of the steps and ~61 % of the budget**.

And those 227 steps would learn almost nothing: stage 3 runs at **one fortieth** of stage 1's
learning rate. By sum-of-learning-rates — a crude proxy for how far the weights move — stage 3
would contribute ~0.05 % of the run's optimisation for 61 % of its cost. **Upstream's stages
2-3 are low-rate adaptation to long context, not more prior learning.** Shrink their step
counts 40x and you pay for the adaptation without getting it.

**Instead:** stage 1 only, and cap the *evaluation* context to 1,024 for both models from one
shared setting, so the comparison is matched rather than approximate. Long context belongs in
Exp2 on the single winning prior, and it needs no new code — `init.strategy: full` plus
`pretrained_path` is upstream's `--checkpoint_path ... --only_load_model True`.

---

## `train.micro_batch_size` — a hard constraint, not a memory setting

`micro_batch_size <= prior.grouping.group_size`, full stop. A bigger GPU does not buy a bigger
micro-batch.

`Trainer.validate_micro_batch` **raises** when the datasets in one micro-batch disagree on
their sequence length *or* their train/test split, and both are drawn per group. Upstream keeps
`micro_batch_size == batch_size_per_gp` in every stage for exactly this reason: their 4 tracks
the group size, not the quality of their hardware.

Gradient accumulation runs `ceil(batch_size / micro_batch_size)` passes and averages them, so
the update is mathematically identical to one batch of 64 — **it does not change the result**,
only speed and memory. What keeps a big GPU busy is the number of micro-passes per sync, not
the micro-batch itself.

---

## `train.max_steps` — why 12,500 and not more

12,500 x 64 = **800,000 datasets per arm, 2.5 % of upstream's stage 1** (500,000 x 64 = 32 M).

The limit is credits. A B200 costs **437.50 credits per GPU-minute** (26,250/hour) plus ~3.04
per CPU-core-minute, and 75 arms at ~16-22 h each is **33-46 M credits**. Doubling `max_steps`
doubles that. Exp1 buys a **ranking**; Exp2 buys the converged number.

If the budget will not stretch, **cut `max_steps`, not the prior shape.** A shorter run is just
shorter; a cheaper prior is a confound.

---

## `train.amp` and the attention kernel

`amp: true`, and the attention backend is **pinned** — see `src/models/backends.py`.

PyTorch picks between four SDPA backends silently, per call, from the shapes and dtype. On
20-08-2026 the cuDNN fused multi-head-attention graph raised
`Expected mha_graph.execute(...).is_good() to be true` on a B200 at batch 64 under AMP. It is
excluded; flash, mem-efficient and math remain. The run card logs which are in use.

AMP itself is a real ~2x speed-up and upstream uses it in every stage.

---

## `train.optimizer` — Muon, and what it actually costs

Muon is what TabICLv2 uses, at `--lr 8e-4`. Its Newton-Schulz orthogonalisation runs **once per
weight matrix per step**, so its ~18 ms is a FIXED cost:

| batch | step time | Muon overhead |
|---|---|---|
| 1 | 45 ms | **1.40x** |
| 64 | 1,618 ms | **1.01x** |

Both measurements are right. At the batch size we train at, Muon is free — a conclusion that
took three runs to reach because every earlier measurement was at batch 1.

---

## `prior.credit_fraction` — the main switch

Probability that a given synthetic dataset comes from our credit path rather than
the unmodified TabICL prior. `0.0` is the control arm.

Defaults sweep `0.0 / 0.1 / 0.2 / 0.3` — i.e. 70–90% original. Keeping most of the
prior original is the natural defence against the collapse Tanna 2026 reports when
TabICL is adapted aggressively (TabZilla accuracy 0.873 → 0.567 under full
fine-tuning, while TabPFN survives it).

## `prior.grouping` — correlated hyperparameter sampling

TabICL samples a meta-distribution once per **group** of 4 datasets, concrete
values per **subgroup**, then a causal graph per dataset. Datasets in a batch are
therefore relatives: some batches come out uniformly small, some uniformly hard.

**On by default.** NanoTabICL removed it, which is why this project first shipped
without it — that was the wrong default. `credit_fraction=0.0` is the control arm
and the control is supposed to *be* TabICL, so having grouping makes it more
faithful, not less. Turning it off is the ablation.

It is also on-topic: a group sharing hyperparameters *is* a small domain, which is
the same object as "a domain-targeted prior". `group_size: 1` disables it.

## `prior.credit.target` — LGD

The target family, and the reason this project exists.

| setting | what it does |
|---|---|
| `mode` | `quantile` sets boundary mass exactly (best for a controlled test); `censor` builds a latent loss fraction on a wider range and clips to [0,1] — the true economic story, with emergent mass |
| `shape_ab_range` | Kumaraswamy shape parameters, log-uniform. `a<1,b<1` gives a U; `a>1,b>1` a hump; `a>1,b<1` leans toward 1 |
| `boundary_mass_range` | mass per atom when present |
| `atom_prob` | chance each atom exists at all; `1.0` forces both |
| `signal_strength` | dilutes the link to the features, applied **before** ranking so shape and difficulty move independently |
| `target_scaling` | `none` shows the model [0,1]; `standard` applies the original prior's affine scaling |

**Measured boundary mass in our own files** (2026-08-06) — this is why it has to be
a family and not one fixed shape:

| dataset | rows | mass at 0 | mass at 1 |
|---|---|---|---|
| heloc | 58,862 | 21.6% | **51.5%** |
| base_modelisation | 594 | 11.6% | 16.0% |
| base_model | 762 | 10.5% | 11.9% |
| lgd_freddie | 16,002 | 11.4% | 8.1% |
| loss2 | 4,637 | 3.6% | 3.7% |
| lgd_lendingclub | 5,627 | 1.5% | 0.3% |
| axa | 2,545 | 0% | 0% |

Four of seven have substantial boundary mass; one has 73%; one has none. A prior
fixed on the Freddie shape would fit Freddie and lose the rest.

The target is **always** hard-clipped to [0,1], whatever the settings. LGD is a
fraction and cannot leave [0,1]; nothing in the original prior knows that, and
encoding just that one fact is the cheapest useful thing in the module.

**Why the rank transform.** The target is replaced by its ranks before shaping.
Ranking is monotone, so it changes the target's *shape* without changing *how
predictable it is* from the features. Change both at once and a result could come
from either.

## `prior.credit.target` — PD

| setting | evidence |
|---|---|
| `base_rate_range` | real data is 6.7%–40% (GMSC 6.7, Home Credit 8.1, HMEQ 20.0, Taiwan 22.1, myhom 40.0). The measured original prior sits at **0.500** with only 5% of tasks under 10% positives |
| `signal_strength` | credit PD lives at AUC 0.70–0.85, so `1.0` is unrealistically easy |
| `flip_pos_to_neg` | defaults that cure get booked as non-default far more often than the reverse; symmetric flipping misses the asymmetry that biases a PD model |
| `rules` | Klein & Hoffart 2026 argue a statistical prior can only blur a hard `if amount >= 5000` rule. **A position paper with zero experiments** — the most speculative component here, and it should be flagged as such in any write-up |
| `selection` | you only observe approved applicants; no general-purpose prior produces this truncation |
| `missingness` | a thin file is itself a risk signal, so missingness is target-linked. The original prior has none at all, and TabICLv2 mean-imputes at inference, discarding exactly this |
| `max_cat_size`, `category_frequency` | Purucker 2026 finds the best-GBDT-over-best-TFM margin grows with high-cardinality columns (ρ=+0.47) |

## `prior.filter` — which generated datasets get discarded

TabICLv2 fits a shallow ExtraTrees and discards datasets it cannot beat a constant
on. Their paper reports *"roughly 35% classification and 25% regression datasets
are filtered"* in stage 1, and that filtering **improves** convergence (their
Fig. 10).

* `tabicl` — as shipped
* `off` — keep everything
* `banded` — keep only datasets whose difficulty lands in `quantile_band`, i.e.
  deliberately aim at credit's low-signal range instead of merely allowing it

`banded` is a **removal**, not an addition, so it cannot be accused of adding
capacity. It also contradicts a published convergence result, so it has to beat
that argument rather than ignore it.

## Experiment 3 — continued pre-training, and how its hyperparameters were chosen

Exp3 starts from the **released TabICLv2 weights** and keeps training on a mixture of the
original prior and ours. The question is how much of ours to add; everything else on this page
is a nuisance parameter that has to be set well enough not to confound the answer.

Two published recipes for continued pre-training of a tabular foundation model, both in the
pinned library:

| | Real-TabPFN (Garg et al. 2025) | TabPFN-Wide (Kolberg et al. 2026) |
|---|---|---|
| optimiser | AdamW | AdamW |
| learning rate | **3e-7** | **1e-5** |
| schedule | linear warm-up -> cosine annealing | linear warm-up -> cosine decay |
| weight decay | — | **1e-4** |
| grad clip | — | **1.0** |
| what is updated | all parameters | all parameters |
| **L2-SP** | **alpha = 0.003** | not used |

TabICLv2's own stage 3 — the closest thing to CPT inside TabICL — uses **lr 2e-5, clip 1.0**.

**The published rates span 3e-7 to 2e-5, nearly two orders of magnitude.** That disagreement is
the reason `train.lr` is swept rather than picked.

### `train.optimizer: adamw` — and why this is not a free choice

Both CPT papers use AdamW; TabICLv2 stage 3 uses Muon. The deciding argument is mechanical:
**under `optimizer: muon`, `train.lr` is only the rate of Muon's auxiliary AdamW half**, so
sweeping it would move almost nothing and the sweep would look like "learning rate does not
matter". Exp1 and Exp2 keep Muon, matching how the released weights were made.

### `train.l2sp_alpha` — pull toward the starting point

    Omega(w) = (alpha / 2) * || w - w0 ||^2       added to the loss

`w0` is the released checkpoint. It penalises drift from weights that already know the original
prior, which is exactly the risk continued pre-training runs: learning credit structure by
forgetting everything else. Introduced for transfer learning by Li et al. 2018; used for
continued pre-training of TabPFNv2 by Real-TabPFN at **alpha = 0.003**, which is the non-zero
value swept here against `0.0` (off).

Implementation notes that matter for correctness:

- The gradient is written **directly onto `.grad`** as `alpha * (w - w0)` rather than added to
  the loss. A parameter penalty is not per-example, and inside the micro-batch loop it would be
  applied `n_micro` times — the effective alpha would silently depend on the micro-batch size.
- It is applied **after `scaler.unscale_` and before the clip**: after, because an unscaled
  penalty on AMP-scaled gradients would make the effective alpha depend on the loss scaler;
  before, because L2-SP is part of the objective, so what gets clipped is the whole gradient.
- `l2sp_alpha > 0` with `init.strategy: scratch` **raises**. There is no starting point.

### `init.strategy` — three depths, not two

`full` (every parameter), `icl_only` (freeze the column and row encoders, train the ICL stack,
the y-embeddings and the head), `head_only` (the last layer alone). Both CPT papers update
everything, so `full` is the literature default and the other two test whether a cheaper
adaptation is enough. See the appendix at the bottom of this file for what the architecture
allows and why there is no LoRA.

### Budget

5 mixtures x 3 strategies x 2 alphas x 2 rates = **60 arms**, one seed. Seeds are deliberately
withheld: at ~3.5 h per arm this is already ~210 GPU-hours, and repeating a 60-arm screen three
times to reduce noise on arms that will be discarded is the wrong place to spend it. Once the
mixture axis has a winner, re-run that configuration with three seeds.

---

## `init.strategy` — how to start

`scratch` (random init), `full` (pretrained, train everything at a low LR), `icl_only` (freeze
the column and row blocks), `head_only` (freeze all blocks). Exp1 and Exp2 use `scratch`; Exp3
warm-starts from the released TabICLv2 weights.

All four come from what TabICL itself does. **No LoRA.** Full analysis in
[Appendix: full retraining or fine-tuning?](#appendix-full-retraining-or-fine-tuning) at the
bottom of this file.

## `model` — do not sweep

Held fixed across every run. That is the entire experiment: architecture,
optimizer and budget constant, only the prior varying. Sweeping these would make
the comparison meaningless.

## `train`

10,000 steps × 4 datasets = 40,000, matching O'Prior's budget so our numbers stay
comparable to theirs.

**"Matched compute" means steps and datasets, never wall-clock or credits.**
Turning the filter off makes generation cheaper, so time is not comparable across
filter settings.

`optimizer: adamw`, not Muon — TabICLv2 credits Muon for part of its gain, but a
subtly wrong reimplementation would degrade every arm equally and invisibly, which
is the worst failure mode for a controlled comparison. The optimizer is held fixed,
so it is not the variable.

## `logging`

Log files are written **on the cluster only**. Locally, output goes to the console
and nothing is left behind. `log_prior_every` periodically samples the prior and
records what it is actually producing — cheap insurance, because a config typo that
silently disables the credit path looks exactly like a real null result.

## A note on the published LGD R² figures

The configs and docs quote "published LGD R² ≈ 0.04–0.15 linear, 0.10–0.25 beta,
0.20–0.43 tree ensembles". **That range came from this project's own brief, not from
a source anyone has verified here.** It is used as a sanity band for our numbers,
and it should be replaced with a citation — or dropped — before it appears in a
paper. Our own measurements so far: CatBoost 0.376 on Freddie and 0.501 on heloc,
both inside the quoted tree band; but `lgd_lendingclub` scores 0.71–0.76, well
outside it, which most likely means a feature derived from the recovery amount is
surviving the recipe.

---

## Appendix: full retraining or fine-tuning?

Written 2026-08-05. Library pin `21d555a`.

**Short answer.** Four options are real, and one is not:

| option | what it does | where it comes from | recommendation |
|---|---|---|---|
| `scratch` | train every weight from random init | TabICLv2 stage 1 | **start here** |
| `full` | pretrained weights, train everything at a low LR | TabICLv2 stage 3 | best fine-tune option |
| `icl_only` | freeze the col + row blocks, train the ICL stack + y-embeddings + head | TabICL **v1** stage 3 | good second arm |
| `head_only` | freeze all three block stacks | our own floor | reference only |
| ~~LoRA~~ | low-rank adapters | — | **do not** — see §4 |

All four are implemented in [`src/train/adapt.py`](../src/train/adapt.py) and
selected by `init.strategy` in the configs.

---

### 1. What the architecture allows

TabICL has three stages in sequence, and the model file makes the split obvious:

```
TF_col   column embedding      InducedTransformerBlock x 3   "what is in this column"
TF_row   row attention         TransformerBlock x 3          "what is in this row"
TF_icl   in-context learning   TransformerBlock x 12         "what does y look like given the context"
```

The target enters **twice**: `y_embed_in` is added before the column blocks, and
`y_embed_icl` is added again before the ICL blocks. That detail matters for us and
is covered in §3.

TabICL ships freezing support for exactly these three stages. From
`_finetune/base.py`:

```python
def _frozen_submodules(self, model):
    out = []
    if self.freeze_col: out.append(model.col_embedder)
    if self.freeze_row: out.append(model.row_interactor)
    if self.freeze_icl: out.append(model.icl_predictor)
    return out
```

Two implementation details worth copying, both of which we did:

* `_apply_freezing` sets `requires_grad = False`, and `_set_training_mode`
  separately snaps frozen parts back to `eval()`. This is not redundant:
  `nn.Module.train()` is recursive, so calling it would switch dropout back on
  inside a frozen block.
* the optimizer only ever receives
  `[p for p in self.model_.parameters() if p.requires_grad]`.

### 2. What the TabICL authors actually did

This is the most useful evidence available, because it is the people who built the
model choosing a recipe rather than us guessing. Read straight off their training
scripts.

**TabICL v1, stage 3** (`scripts/train_stage3.sh`) — a partial fine-tune:

```
--freeze_col True          # column embedder frozen
--freeze_row True          # row encoder frozen
--lr 2e-6                  # very low
--scheduler constant       # no decay
--gradient_clipping 1.0    # down from 10.0 in stage 1
--only_load_model True     # weights only, not optimizer state
--max_steps 50 --batch_size 512
```

**TabICL v2, stage 3** (`scripts/train_v2_{clf,reg}_stage3.sh`) — **no freezing**:

```
--lr 2e-5
--scheduler cosine_with_restarts
--gradient_clipping 1.0
--only_load_model True
--max_steps 10000
```

So **the v2 authors moved away from freezing.** With the v2 architecture they
train everything at a low LR in the final stage. That is a real signal, and it is
why `full` rather than `icl_only` is the recommended fine-tune arm.

Note the pattern in both: the final stage is always a **low LR with tight
gradient clipping** (1.0 instead of stage 1's 10.0). Whatever you freeze, do not
fine-tune at stage-1 learning rates.

### 3. The one place we improve on a naive freeze

Our change is to the **target** distribution, not the feature distribution. So the
parts that most need to move are the target-side parts.

A naive "freeze the column stage" would also freeze `y_embed_in`, which is where
the target first enters the model — exactly the parameter that needs to adapt to a
bounded [0,1] target with mass at the boundaries. That would be
self-defeating.

So in [`src/train/adapt.py`](../src/train/adapt.py), freezing covers the **block
stacks** and always leaves these trainable, whatever the strategy:

```python
ALWAYS_TRAINABLE = ("y_embed_in", "y_embed_icl", "out_ln", "out_mlp",
                    "row_ln", "row_cls_tokens")
```

`icl_only` therefore means: keep the learned table representation, retrain
everything that touches the target.

### 4. Why not LoRA

Three reasons, in order of weight:

1. **TabICL has no LoRA.** Searching the entire repository dump for "lora" returns
   **zero matches**. There is nothing upstream to copy or validate against.
2. **Tanna 2025 found LoRA unstable on TabPFN** — batched-inference constraints
   force an automatic fallback to full fine-tuning. Rubachev 2025 independently
   found that for TabPFN v2, full fine-tuning matches LoRA and every other
   parameter-efficient variant on accuracy while converging fastest. So even where
   LoRA works, it buys nothing.
3. **It would be a third confound.** We are already varying the prior and the
   adaptation strategy. Adding an unvalidated adapter means a bad result could be
   the prior, the strategy, or the adapter, and we could not tell which.

If you later want a parameter-efficient option, the better-evidenced one is
BETA-style input adapters (Liu & Ye 2025): a small learnable input encoder,
0.6 MB of trainable parameters, weights otherwise frozen. But it is v1-only and
classification-only in the paper, so it is future work, not a tonight decision.

### 5. The warning that matters most

**Fine-tuning TabICL is documented as dangerous.** Tanna 2026 (*Exploring
Fine-Tuning for Tabular Foundation Models*) reports that full supervised
fine-tuning is **near-catastrophic for TabICL**: TabZilla accuracy drops from
**0.873 to 0.567**. TabPFN survives the same treatment essentially intact. This is
a strong architecture-by-adaptation interaction, and TabICL is on the bad side of
it.

Two honest caveats on that result: it is a four-page paper with **one
hyperparameter setting per strategy**, so "TabICL collapses" may partly be
under-tuning rather than a property of the architecture; and the authors are
comparing against their own models. But it points the same way as the v2 authors'
own choice of a very low LR, so the two independent signals agree: **be gentle**.

This is also a direct argument for the mixture lever. Keeping 70–90% of datasets
from the original prior means the model keeps seeing the distribution it was built
for, which is the natural defence against exactly the collapse Tanna documents.

### 6. The practical blocker

**Loading the released TabICLv2 checkpoints into NanoTabICL is untested and may
not work.** The released weights come from the full implementation, whose module
names differ (`col_embedder` / `row_interactor` / `icl_predictor` versus
NanoTabICL's `col_blocks` / `row_blocks` / `icl_blocks`). NanoTabICL's own README
points you at the main repository for pretrained weights, and notes a RoPE change
that "permutes the neurons", so a mismatch is plausible.

`load_pretrained` in `src/train/adapt.py` handles this by **refusing to run** when
fewer than half the tensors match, rather than quietly training a randomly
initialised model and reporting it as fine-tuned. That failure would be invisible
in the loss curve and would silently invalidate every comparison.

Three ways forward, best first:

1. **Pretrain your own base with `scratch`, then fine-tune from that.** The
   architecture is identical by construction, so there is no mapping problem at
   all. This is why `scratch` is the default and the recommended first run.
2. Write an explicit key mapping from the full checkpoint to NanoTabICL, and
   verify a forward pass reproduces the full model's output on the same input.
   Do not trust a mapping that has not been checked numerically.
3. Use the full `tabicl` package instead of NanoTabICL for the fine-tuning arms
   (`pip install -e ".[tabicl]"`), which loads its own checkpoints natively. Costs
   us the small, readable codebase.

### 7. Recommended sequence

1. **`scratch`, `credit_fraction` swept over 0.0 / 0.1 / 0.2 / 0.3.** This
   answers the actual research question — does adding our datasets to the prior
   help — with nothing else moving. Run this first.
2. **`full` from the best `scratch` checkpoint**, at `lr 2e-5`, cosine, clip 1.0.
   Answers "is it cheaper to fine-tune than to retrain?"
3. **`icl_only` from the same base**, at `lr 2e-6`, constant, clip 1.0. Answers
   "does keeping the table representation fixed protect against collapse?"

Steps 2 and 3 need step 1's checkpoint, so they cannot run tonight. That is fine —
step 1 is the one that answers the research question.

`recommended_hparams()` in `adapt.py` returns the LR / schedule / clipping each
strategy was actually used with upstream, so those numbers stay visible in the
config rather than being applied silently.

