# PRIORS.md — how TabICLv2 makes its synthetic data, and what we change

A tabular foundation model never sees a real table during pretraining. It sees millions of
**synthetic** ones from a generator called the **prior**. The prior decides what "a table" means
to the model, so it decides what the model is good at. This project changes that generator.

**Sources**, all inside this repository (pin `bbba8d4b`):

- paper — `tfm-library/papers/2026/02_Qu_et_al._TabICLv2_*.pdf`
- code — `tfm-library/repositories/TabICL.txt` (127 files), including **their own training
  commands**: `scripts/train_v2_{clf,reg}_stage{1,2,3}.sh`

Code is cited **by symbol name**, never line number — the dumps are re-snapshotted.

---

## 1. Both TabICLv2 models share ONE prior

Two released checkpoints, so you would expect two priors. There is one. Diffing their stage-1
commands, the whole difference is:

| | classifier | regressor |
|---|---|---|
| `--prior_type` | `graph_scm` | `graph_scm` — **identical** |
| every other prior flag | — | **identical** |
| head | `--max_classes 10` | `--regression_method quantile --num_quantiles 999` |
| layer norm | with biases | `layernorm_nobias` |

Only the **head** differs: discretise the last value into classes, or keep it continuous. So a
change to the prior affects both our tracks the same way.

## 2. What the generator does

`GraphSCM`, configured by `PriorConfig`. Per dataset:

1. **A random causal graph** — 2 to 32 nodes (`min_n_nodes`, `max_n_nodes`).
2. **A random function on every edge**, drawn from: `gp`, `generalized_gp`, `nn`, `tree`,
   `product`, `discretization`, `linear`, `flow`. The mix of smooth and step-like is the point —
   real tables are full of thresholds, and a smooth-only prior would never teach them.
3. **Not every node becomes a column** (`subsample_feature_nodes`), so features are correlated
   through causes you cannot see. Real data is like this.
4. **Random node importances**, so features differ in relevance.
5. **Some columns made categorical**, up to 256 levels.
6. **Numeric columns warped** by `KumaraswamyWarping`: min-max to `[0,1]`, then
   `1-(1-x^a)^b` with `a,b` log-uniform on `[0.2, 5]`. Worth knowing: this flexible `[0,1]`
   shape is **already in their prior**, and it is the same family we use for LGD's interior.
7. **Clean up** — `outlier_removing(threshold=4)` then `standard_scaling`.
8. **Maybe discard it** — both filters are ON in the released runs, and the regressor script
   notes **~25 % of stage-1 datasets get filtered** as unpredictable.

**`meta_sampling_mode='meta'`** correlates hyperparameters *within* a dataset, so one dataset is
coherently "smooth and narrow" and another "step-like and wide", instead of every dataset being
an average mush.

## 3. The regression target, and why "boundary mass" needs care

The regression branch is two lines:

```python
y = outlier_removing(y, threshold=4)   # clamp beyond ±4 SD
y = standard_scaling(y)                # mean 0, SD 1
```

**Their target has mean 0 and SD 1.** `0` and `1` are not meaningful points on it.

So the naive metric `(y <= 0).mean()` means *"below the mean"* → **≈0.5**. Measured that way the
unmodified prior looks like it puts 54 % of its mass at zero and has already solved LGD. It has
not — the metric was lying, in the direction that flatters this project.

The scale-free question is *"how much mass sits on the target's **own** extremes?"* —
`(y == y.min()).mean() + (y == y.max()).mean()`. Negligible for a continuous target, large for a
real atom, and valid at any scale. See `src/utils/target_stats.py`.

**The original prior does show atoms — 35 % of its datasets — and they are an artefact.**
`outlier_removing` sets every row beyond ±4 SD to *exactly* the same value, manufacturing a tie.
So the honest claim is not "they have no atoms" but:

> theirs has atoms **by accident, at an arbitrary scale**; ours has them **by construction at 0
> and 1, meaning full recovery and total loss**.

## 4. The classification target

`Reg2Cls` cuts a continuous latent into classes: `MulticlassAssigner` with `mode="rank"`
(roughly even counts) or `"value"` (uneven), chosen per dataset. `balanced` is **False**, so
nothing forces 50/50 — but nothing targets a *particular* rare rate either.

**That is the gap we use:** no mechanism aims at the 6.7 %–40 % default rates of real PD data,
and none represents a base rate as something with a cause.

> `DEFAULT_FIXED_HP`/`DEFAULT_SAMPLED_HP` configure the **v1** prior (`mix_probs (0.7, 0.3)`).
> v2 passes `--prior_type graph_scm`. `Reg2Cls` is shared machinery.

**One upstream quirk:** `Converter._transform` carries the comment *"this is the wrong version,
but it was used in the experiments"*, with `use_corrected_num_converters=False`. Upstream keeps a
known bug so the released checkpoints stay reproducible. **Leave every `use_corrected_*` at its
default** — "fixing" one makes our arms incomparable to the reference model.

---

## 5. What we change

Two rules:

1. **The control arm is exactly TabICLv2's prior.** `credit_fraction: 0.0` runs the original
   path with none of our code on it — enforced by a test, including that `prior.base` never
   carries one of our values.
2. **A mechanism must come from credit risk, not curve-fitting.** Anyone can add a parameter
   that makes a histogram match. The claim is that encoding *how losses arise* transfers.

`credit_fraction` **mixes**: at 0.2, one dataset in five is ours. The original prior is never
replaced — a model that only saw credit tables would be a credit model, not a foundation model,
which is what the out-of-domain suites check.

### LGD — a loss fraction in `[0,1]` with real mass at both ends

- **`mode: mechanism`** — derive the loss so the atoms *emerge*: `collateral` (0.40) where
  recovery is collateral minus costs capped at the debt, so over-collateralised loans land on
  **exactly 0** and unsecured ones on **exactly 1**; `workout` (0.35), discounted recovery
  cashflows; `segment_mixture` (0.25), a portfolio of segments, deliberately non-monotone.
  Plus `cohort`: 1–12 vintage blocks sharing a shock, so rows are **not i.i.d.**
- **`mode: quantile`** — shape the marginal directly: mass at the ends plus a Kumaraswamy
  interior.

### PD — a rare, correlated binary event

- **`mode: mechanism`** — the **Merton/Vasicek one-factor model**, with
  `rho_range: [0.03, 0.24]` from **Basel IRB asset correlations** (QRRE 0.04, mortgage 0.15,
  corporate 0.12–0.24). Defaults therefore **cluster**, which an i.i.d. prior cannot express.
- **`mode: quantile`** — cut the latent to hit a target base rate.
- **`selection`** — you only see *approved* applicants (20 % of rows dropped). Reject inference
  is not a nuisance here; it is the defining feature of credit data.
- **`rules`** — hard underwriting cut-offs and scorecard-style binning. *From Klein & Hoffart
  2026, a **position paper** — the most speculative mechanism here, and labelled so.*
- **Asymmetric label noise** — cures (10 %) are far commoner than missed defaults (1 %).

### Both tracks

- **`shift`** — context and prediction rows from different populations (`cohort`, `covariate`,
  `prior_prob`), 30 % of datasets. O'Prior's third contributor. Credit path only, so the control
  stays clean.
- **`missingness`** with `missing_target_coupling: 1.0` — missingness that **carries signal**,
  because a thin credit file is itself a risk indicator. TabICL's prior has none.
- **`noise_features`** — 30 % junk columns. Real credit tables are wide and mostly uninformative.
- **`max_cat_size: 500`** vs their 100 — credit has state, MSA, servicer.
- **`filter`** — `tabicl` reproduces their ExtraTrees filter exactly; `banded` instead keeps
  credit's weaker signal range.

---

## 6. Per experiment — three different prior designs

**Each config owns its own prior.** They are not shared, because the three experiments do
genuinely different things with it.

| | how many priors | budget/arm | init |
|---|---|---|---|
| **Exp1** | **32**, swept — this is the experiment | 50,000 datasets | scratch |
| **Exp2** | **1**, the Exp1 winner (+ control) | 400,000 | scratch |
| **Exp3** | 1 winner, **5 mixtures** swept | 160,000 | released checkpoint |

**Exp3 is continued pre-training, and it has a published precedent.** TabPFN-Wide (Kolberg et
al. 2026) *"extends existing models through continued pre-training on synthetic data sampled
from a customized prior"* and reports it matches or exceeds the base model. Its recipe is what
Exp3 follows: **LR 1e-5**, batch 16, **10,000 steps** (they saw no consistent gains beyond).
The learning rate is the important part — pretraining's 8e-4 applied to trained weights would
destroy them.

Exp3 sweeps `credit_fraction` from 0.0 to 1.0 because the model **already knows** the original
prior, so how much of ours to add *is* the question. Mitra (Zhang et al. 2025) selects a prior
mixture on performance, diversity and distinctiveness and finds mixtures beat single priors —
so expect an interior optimum, and `1.0` is included to measure what forgetting the original
prior costs. It also sweeps `init.strategy` (`full` vs `icl_only`) because Tanna et al. 2026
find full fine-tuning *"often reduces accuracy or calibration quality"*; their setting is
supervised fine-tuning on real data rather than continued pretraining, so that is a caution and
not a result — but miscalibration is disqualifying for PD, so it gets measured.

**For scale:** TabICLv2's own budget is `500,000 + 40,000 + 10,000` steps at batch 64 =
**35,200,000 datasets**, 24.5 GPU-days per model. Our Exp2 is ~1.1 % of that. Every result here
is a statement about priors **at a fixed small budget**, and must be written that way.

## 7. Open item

**Exp3 needs the upstream `tabicl` package installed** (`pip install "tabicl>=2.0"`). It is now
a required dependency: the released checkpoint only loads into the code that saved it. See
`docs/AGENTS_MEMORY.md`.
