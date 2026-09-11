# PRIORS — how TabICLv2 makes its synthetic data, and what we change

A tabular foundation model never sees a real table during pretraining; it sees millions of
**synthetic** ones from a generator called the **prior**. The prior decides what "a table" means to
the model, so it decides what the model is good at. This project changes that generator. Here is
what upstream does, what we add, and why every mechanism comes from credit risk rather than
curve-fitting.

Sources, in-repo (pin `52dab01`): paper `tfm-library/papers/2026/02_Qu_et_al._TabICLv2_*.pdf`; code
`tfm-library/repositories/TabICL.txt`, including their own `scripts/train_v2_{clf,reg}_stage{1,2,3}.sh`.
Code is cited **by symbol name**, never line number. The *values* these mechanisms take are in
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md); why they exist is [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md).

## Contents

1. [One prior, two heads](#1-one-prior-two-heads)
2. [What the generator does](#2-what-the-generator-does)
3. [The regression target, and why "boundary mass" needs care](#3-the-regression-target)
4. [The classification target](#4-the-classification-target)
5. [What we change](#5-what-we-change)
6. [Per experiment](#6-per-experiment)
7. [Open item](#7-open-item)

---

## 1. One prior, two heads

Two released checkpoints, but **one** prior. Diffing their stage-1 commands, the whole difference is
the head — the classifier discretises the last value into classes (`--max_classes 10`), the regressor
keeps it continuous (`--regression_method quantile --num_quantiles 999`, `layernorm_nobias`). Every
prior flag is identical. So a change to the prior affects both our tracks (PD, LGD) the same way.

## 2. What the generator does

`GraphSCM`, configured by `PriorConfig`. Per dataset: sample a **random causal graph** (2–32 nodes);
put a **random function on every edge** from `{gp, generalized_gp, nn, tree, product, discretization,
linear, flow}` — the mix of smooth and step-like is the point, since real tables are full of
thresholds; keep only **some nodes as columns** (`subsample_feature_nodes`), so features correlate
through unseen causes; assign **random importances**; make **some columns categorical** (up to 200
levels); **warp** numeric columns with `KumaraswamyWarping` (min-max to [0,1], then `1-(1-x^a)^b`) —
this flexible [0,1] shape is *already theirs*, and the same family we use for LGD's interior; then
**clean up** (`outlier_removing(threshold=4)` → `standard_scaling`) and **maybe discard** the dataset
(the filters; the regressor script notes ~25% of stage-1 datasets are filtered as unpredictable).

`meta_sampling_mode='meta'` correlates hyperparameters *within* a dataset, so one dataset is
coherently "smooth and narrow" and another "step-like and wide" rather than every dataset an average.

## 3. The regression target

The regression branch is two lines — `outlier_removing(threshold=4)` then `standard_scaling` — so
their target has **mean 0, SD 1**, and `0`/`1` are not meaningful points on it. This matters for how
we measure it: the naive `(y <= 0).mean()` means "below the mean" ≈ 0.5, which would make the
unmodified prior *look* like it already puts half its mass at zero. The scale-free question is *"how
much mass sits on the target's **own** extremes"* — `(y == y.min()).mean() + (y == y.max()).mean()`
(see `src/utils/target_stats.py`), negligible for a continuous target, large for a real atom.

The original prior *does* show atoms — ~35% of its datasets — but they are an artefact:
`outlier_removing` sets every row beyond ±4 SD to exactly the same value, manufacturing a tie. So the
honest claim is not "they have no atoms" but: **theirs has atoms by accident, at an arbitrary scale;
ours has them by construction at 0 and 1, meaning full recovery and total loss.**

## 4. The classification target

`Reg2Cls` cuts a continuous latent into classes (`MulticlassAssigner`, `mode="rank"` or `"value"`).
`balanced` is **False**, so nothing forces 50/50 — but nothing targets a *particular* rare rate
either. **That is the gap we use:** no mechanism aims at the 6.7%–40% default rates of real PD data,
and none represents a base rate as something with a cause.

> **Leave every `use_corrected_*` converter flag at its default.** `Converter._transform` carries a
> known upstream bug (*"this is the wrong version, but it was used in the experiments"*); upstream
> keeps it so the released checkpoints stay reproducible, and "fixing" it makes our arms incomparable
> to the reference model.

## 5. What we change

Two rules govern every addition:

1. **The control is exactly TabICLv2's prior.** `credit_fraction: 0.0` runs the original path with
   none of our code on it — enforced by a test, including that `prior.base` never carries our values.
2. **A mechanism must come from credit risk, not curve-fitting.** The claim is that encoding *how
   losses arise* transfers — not that a parameter was tuned until a histogram matched.

`credit_fraction` **mixes** rather than replaces: at 0.5, half of each batch is ours. A model that
only ever saw credit tables would be a credit model, not a foundation model — which is what the
out-of-domain suites check.

**LGD — a loss fraction in [0,1] with real mass at both ends.** `mode: mechanism` derives the loss so
the atoms *emerge*: `collateral` (recovery = collateral − costs, capped at the debt → over-secured
loans land on exactly 0, unsecured on exactly 1), `workout` (discounted recovery cashflows),
`segment_mixture` (a deliberately non-monotone portfolio of segments), plus `cohort` vintage blocks
sharing a shock so rows are not i.i.d. `mode: quantile` instead shapes the marginal directly — mass
at the ends plus a Kumaraswamy interior.

**PD — a rare, correlated binary event.** `mode: mechanism` is the **Merton/Vasicek one-factor
model**: asset value = √ρ·(systematic factor) + √(1−ρ)·(idiosyncratic), default when it falls below
the PD threshold, so defaults **cluster** through the shared factor — which an i.i.d. prior cannot
express. Around it: `selection` (you only see *approved* applicants — reject inference is the defining
feature of credit data, not a nuisance), `rules` (hard underwriting cut-offs and scorecard binning —
*from Klein & Hoffart 2026, a position paper, so the most speculative mechanism and labelled so*), and
asymmetric label noise (cures far commoner than missed defaults). `mode: quantile` instead cuts the
latent to a target base rate.

**Both tracks:** `shift` (context and prediction rows from different populations — cohort, covariate,
prior_prob, and PD's reject-inference `selection`); `missingness` with `missing_target_coupling: 1.0`
(missingness that *carries signal*, because a thin credit file is itself a risk indicator — TabICL's
prior has none); `noise_features` (junk columns, since real credit tables are wide and mostly
uninformative); a higher categorical cap (credit has state, MSA, servicer); and `filter` (`tabicl`
reproduces their ExtraTrees filter exactly, `banded` instead keeps only credit's weaker signal range,
`off` disables it).

## 6. Per experiment

Each config owns its prior inline — they are *not* shared, because the three experiments do genuinely
different things. The prior *design* differs as below; the training knobs (LR, steps, optimizer,
L2-SP, freeze strategy) are in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

| | prior | init | steps |
|---|---|---|---|
| **Exp1** | swept — 15 priors (`credit_fraction × filter × intensity`) × 3 seeds = 45 arms | scratch | 12,500 |
| **Exp2** | the winner, `credit_fraction` swept 0→1 alongside fine-tuning knobs — 60 arms | released checkpoint | 10,000 |
| **Exp3** | 1, the Exp1 winner (+ its control) | scratch | 100,000 |

**Exp2 is continued pre-training with a published precedent** — TabPFN-Wide (Kolberg et al. 2026)
extends a model through continued pretraining on a customised synthetic prior and reports it matches
or exceeds the base; its recipe is what Exp2 follows. It sweeps `credit_fraction` all the way to 1.0
because the model *already knows* the original prior, so how much of ours to add is the question, and
Mitra (Zhang et al. 2025) finds mixtures beat single priors — expect an interior optimum, with 1.0
included to measure what forgetting the original prior costs.

**For scale:** TabICLv2's own budget is `500k + 40k + 10k` steps at batch 64 ≈ 35M datasets, ~24.5
GPU-days per model; Exp3 is ~1% of that. **Every result here is a statement about priors at a fixed,
small budget, and must be written that way.**

## 7. Open item

**Exp2 needs the upstream `tabicl` package installed** (`pip install "tabicl>=2.0"`) — a required
dependency, because the released checkpoint only loads into the code that saved it (see
[AGENTS_MEMORY.md](AGENTS_MEMORY.md)).
