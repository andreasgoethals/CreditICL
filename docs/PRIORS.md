# Synthetic priors: upstream generator and credit intervention

This project's training stream contains **synthetic** tables from a generator called the
**prior**. That generator defines the task distribution on which the model learns. This project
changes it toward credit-risk mechanisms. The account below distinguishes the upstream code,
literature motivations and project design choices. Benchmark inspection helped motivate parts of
the design; this is not a claim that every setting was established independently of these datasets.

Source: [TabICLv2](../tfm-library/papers/text/2026/09_Qu_et_al._TabICLv2_A_better_faster_scalable_and_open_tabular_foundation_model.txt)
and `tfm-library/repositories/TabICL.txt`, including the stage scripts. The original source audit
used pin `de324efcca55be392feb0c01830b3b6ce364587f`; the current library pin is
`8f14e5945ca6b068ed0c76218e769d3e672d0d13`. Historical observations retain their original pins.
Code is cited **by symbol name**, never line number. Training settings are in
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md); why they exist is [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md).

## Contents

1. [One prior, two heads](#1-one-prior-two-heads)
2. [What the generator does](#2-what-the-generator-does)
3. [The regression target, and why "boundary mass" needs care](#3-the-regression-target)
4. [The classification target](#4-the-classification-target)
5. [What we change](#5-what-we-change)
6. [Per experiment](#6-per-experiment)
7. [Limitations and future work](#7-limitations-and-future-work)

---

## 1. One prior, two heads

The released classifier and regressor share the graph-based generator. Their target paths differ:
the classifier converts its latent target to classes (`--max_classes 10`), while the regressor
uses a continuous target and a 999-quantile head. Shared feature-generation mechanisms therefore
support both tasks, but a prior intervention need not have the same effect on PD and LGD.

## 2. What the generator does

`GraphSCM`, configured by `PriorConfig`, samples a causal graph, keeps a subset of nodes
as features and uses the installed `RandomFunction` mixture: MLP, tree, discretization,
linear, quadratic, GP, EM assignment and product. `sample_categorical_sizes` has nominal
`max_cat_size=200`, but the original `use_corrected_cat_sizes=False` limits feature
cardinalities to 2–9. Credit slots enable the corrected sampler and permit up to 500.

The legacy `NumericalConverter._transform` returns `(transformed_value, original_value)`:
the transformed value propagates through the graph, while the original becomes the dataset
column. Thus its Kumaraswamy warp does not directly give the observed columns a [0,1] shape.
Control tasks then use upstream cleanup and filtering. Stage 1 enables graph-unpredictability
and ExtraTrees filtering; its triviality filter is disabled.

`meta_sampling_mode='meta'` correlates hyperparameters *within* a dataset, so one dataset is
coherently "smooth and narrow" and another "step-like and wide" rather than every dataset an average.

## 3. The regression target

The regression branch is two lines — `outlier_removing(threshold=4)` then `standard_scaling` — so
their target has **mean 0, SD 1**, and `0`/`1` are not economic endpoints on it. This matters for how
we measure it: `(y <= 0).mean()` counts values below the mean, not mass at the zero-loss boundary.
That share is not necessarily 0.5 for a skewed distribution. The scale-free question is *"how
much mass sits on the target's **own** extremes"* — `(y == y.min()).mean() + (y == y.max()).mean()`
(see `src/utils/target_stats.py`), negligible for a continuous target, large for a real atom.

Upstream targets can contain endpoint ties: clipping sets multiple extreme rows to the same
value, and discrete mechanisms can create ties too. Their frequency depends on the sampled
generator and validity/filtering rules; use the current diagnostic notebook for measured rates.
Credit LGD targets instead place mass explicitly at economic endpoints before standardization.
Standardization preserves these ties but changes their numerical locations.

## 4. The classification target

`GraphSCM` creates classification targets with `CategoricalConverter`; `Reg2Cls` and
`MulticlassAssigner` belong to the older `SCMPrior` path. Categorical conversion can produce
imbalance, but an exact frequency of rare binary tasks must be measured, not inferred from
that older path. Our credit mechanism deliberately models a default rate and common shocks.

Preserve the upstream converter defaults on **control slots**. Credit slots deliberately enable
the corrected category-size sampler, a documented part of the treatment. Do not silently change
other converter semantics: upstream retains legacy behavior for reproducibility.

## 5. What we change

Three rules govern every addition:

1. **The control uses the native TabICL generator and filtering.** Credit mechanisms do not run
   on control slots. Shared validity checks, shape planning and model-specific preprocessing still
   apply; for example, TabPFN requires a usable context. These checks are tested explicitly.
2. **A mechanism must come from credit risk, not curve-fitting.** The claim is that encoding *how
   losses arise* transfers — not that a parameter was tuned until a histogram matched.
3. **Record provenance and prior exposure honestly.** Current ranges use cited literature and
   deliberately broad modelling choices. However, the family of mechanisms was developed after
   inspecting these same benchmark datasets, including the designated holdout. Removing numerical
   calibration on 29-09-2026 did not erase that earlier influence. The 28-09 calibration change
   postdates the completed September campaign; it must not be attributed to those checkpoints.
   The current benchmark supports controlled comparisons on a previously inspected suite, not a
   claim of untouched external validation. Independent source/time cohorts are needed for that.
   Notebook comparisons describe the prior; they must not be used to retune it after the freeze.


`credit_fraction` sets the probability that a task slot uses the credit path. At 0.5, half the
slots are credit tasks in expectation, not necessarily in every batch. Paired seeds reuse the
slot coins and geometry across mixtures. The OOD benchmark tests whether greater credit exposure
changes general-domain performance; neither replay nor 100% credit exposure determines that outcome
in advance.

### 5.1 PD — a rare, correlated binary event

`mode: mechanism` is the **Merton/Vasicek one-factor model** (Vasicek 2002), the model behind the
Basel IRB risk-weight formula: asset value = √ρ·Z + √(1−ρ)·ε, default when it falls below Φ⁻¹(PD).
Defaults **cluster** through the systematic factor Z, shared by borrowers in a period. This makes
cohort dependence explicit; it is not a claim that upstream graphs cannot represent dependence.
ε is the table's own SCM latent (so features can remain predictive), of
which `signal_share` is kept and the rest is risk no file records.

| knob | value | source |
|---|---|---|
| `rho_range` | 0.03 – 0.24 | Basel IRB asset correlations, from other retail (3 %) to corporate (24 %) (BCBS 2005) |
| `base_rate_range`, `base_rate_log` | 1 % – 50 %, log-uniform | motivated by the imbalanced settings of Brown & Mues (2012); the balanced upper bound and log-uniform sampling are project choices, not a guarantee of covering all upstream class frequencies |
| `signal_share_range` | 0.05 – 0.95 | **wide**: discrimination varies across portfolios and no value transfers |
| `cohort` | 1 – 40 periods, Z ~ N(0, 1) | the Vasicek model's standard-normal factor; 40 periods as in Li, Zhang & Zhao (2020) |
| `macro_feature_prob` | 0.5 | macroeconomic variables are explanatory variables in credit models (economic-cycle dependency; Bellotti & Crook 2012, surveyed in Baesens et al. 2026) — half the tables record the period factor as a column |
| `selection` | reject 0 – 50 %, sharpness 0.3 – 1.0 | only approved applicants are observed (support truncation, motivated by selection literature; Banasik & Crook 2007, surveyed in Baesens et al. 2026); acceptance rates are not published, so **wide** |
| `rules` | 0 – 3 cut-offs per table | hard underwriting cut-offs, **from Klein & Hoffart (2026), a position paper with no experiments — the most speculative mechanism, labelled so** |
| `woe_prob` | 0.5 | scorecard-style monotone binning of the risk driver; practitioner convention, no rate published — **uninformative** |
| label noise | **none** | asymmetric label noise (cures booked as non-default) was in the prior until 29-09-2026 with rates from nowhere; removed |

The base-rate range is the **unconditional applicant** PD parameter. A finite cohort,
macroeconomic shocks and acceptance selection can change its realised default rate.
Single-class tables are rejected and redrawn by the generator, not repaired by inventing defaults;
this validity conditioning is part of the training distribution. Tied latent scores use randomized
probability-integral transforms, preserving the normal marginal required by the Vasicek threshold.

Missingness uses no realised target. All affected columns share a noisy borrower propensity and
one borrower-level uniform draw, avoiding an increasing number of independent target votes as
width grows. This is an explicit synthetic modelling choice, not a calibrated empirical missingness
law. The component removal switches its covariate dependence off while retaining missing values.

### 5.2 LGD — a loss fraction in [0, 1] with real mass at both ends

`mode: zoib` draws the target from the **zero-and-one inflated beta regression** — the model the LGD
literature simulates from. Li, Zhang & Zhao (2020, section 2.1; OCC and FDIC economists) generate
LGD data as P(LGD=0) = e^{xα}/(1+e^{xα}+e^{xβ}), P(LGD=1) = e^{xβ}/(1+e^{xα}+e^{xβ}), and an interior
Beta(μφ, (1−μ)φ) with μ = logistic(xγ), following Ospina & Ferrari (2010), Li et al. (2016) and
Yashkir & Yashkir (2013); the explanatory variables include a macroeconomic factor shared by every
default of a quarter (the unemployment rate). The bimodal shape it produces is the documented one:
LGD values bounded on [0, 1] with modes near both ends (Asarnow & Edwards 1995; Qi & Zhao 2011; as
cited by Li, Zhang & Zhao), sometimes a third in the middle (Altman & Kalotay 2014, idem), arising from
portfolios that mix secured and unsecured loans (Baesens et al. 2026).

Their one portfolio becomes a prior over portfolios — each table draws its own parameters:

| knob | value | recorded source anchor (Li, Zhang & Zhao; see §5.4) |
|---|---|---|
| `alpha0_mean_sd` | normal(−0.54, 1) | α₀ = −0.54 |
| `beta0_mean_sd` | normal(−1.46, 1) | β₀ = −1.46 |
| `gamma0_mean_sd` | normal(0, 1) | γ₀ = 0 |
| `phi_range` | 1 – 20, log-uniform | φ = 5 |
| `loading_p0_range` / `_p1_` / `_mu_` | 0.2–1.5 / 0.1–1.5 / 0.1–1.0 (SD units) | 0.6 / 0.15 / 0.15 (nine N(0, 0.5²) covariates with coefficients 0.4 / −0.1 / −0.1) |
| `macro_p0_range` / `_p1_` / `_mu_` | −0.3–0 / 0–0.3 / 0–0.1 per SD | −0.09 / +0.11 / +0.01 (−5 / +6 / +0.5 times the SD of the 2006–2015 unemployment rate) — the signs: a downturn means fewer full recoveries and more total losses (Altman et al. 2005) |
| `signal_share_range` | 0.1 – 1 | their fits keep all nine covariates or drop four or eight (11 – 100 % of the covariate signal) |
| `cohort` | 1 – 40 periods | 40 quarters |
| `macro_feature_prob` | 0.5 | the unemployment rate is one of their explanatory variables |
| `component_overlap_range` | 0.3 – 1 | P(0), P(1) and μ have separate coefficient vectors in the ZOIB model (Ospina & Ferrari 2010); how far they share a driver is **wide** |

The spreads (SD 1 on the logit scale and ranges spanning the reported effects) are project choices
that turn one portfolio model into a distribution of tasks. They are not fitted to these benchmark
histograms. Predictive difficulty is measured, not calibrated to a target R². Earlier notes quoted
unverified real-data R² bands; those are not literature-established targets and must not be used
to justify the current ranges.

`mode: mechanism` (collateral, workout, segment mixtures) and `mode: quantile` (dialled atoms and a
Kumaraswamy interior) remain in the code for ablations; no experiment uses them.

### 5.3 Both tracks

| knob | value | source |
|---|---|---|
| `missingness` | half the tables; 5 – 100 % of columns; 1 – 50 % missing; coupling to a noisy covariate-based borrower propensity 0 – 1 | models correlated, covariate-dependent gaps (MAR), not outcome-driven missingness; target information is not guaranteed. The legacy `no_informative_missingness` ablation removes this coupling; it need not improve target AUC/R² |
| `marginals` | 0 – 100 % of continuous columns, skew 0.25 – 2.5 | money amounts, balances and incomes are positive and right-skewed; a monotone warp, so every split a tree could make — and the dependence on the target — is unchanged; magnitudes **wide** |
| `shift` | 30 % of tables | populations drift with the economic cycle and with acceptance policy (economic-cycle dependency and reject inference; Baesens et al. 2026); kinds **wide** |
| `prior_prob_ratio_range` | 1.25 – 4, log-uniform, either way | the prior-probability shift sets the query's default rate (LGD: its share of above-median losses) to the context's times this ratio. Under the Vasicek model of 5.1, a one-to-two standard-deviation move of the factor multiplies a book's default rate by 1.1 – 6.1 over ρ 0.03 – 0.24 and default rates 1 – 50 %; the range sits inside that. Until 29-09-2026 the knob (`prior_prob_range`) set the context's share of defaults to an absolute 15 – 85 %, which left the query of a low-default book a single default in 85 % of these tables |
| `noise_features` | add 30 % of the original width (about 23 % of the expanded table) | generic: real tables carry uninformative columns |
| `max_cat_size` | 500 | credit tables hold high-cardinality codes (region, product, servicer) |
| `filter.apply_to` | `base` | TabICLv2's filter judges TabICLv2's tables, as upstream; the credit tables are used as specified — re-selecting them by learnability would change the specified prior |

**The encoding is the prediction path of the model being trained.** For TabICL (`prior.encoding:
tabicl`, the default) a credit table ends in upstream's `UniqueFeatureFilter` + `PreprocessingPipeline
("none")` fitted on its context rows, gaps filled with the context mean without indicator columns, and
an LGD target standardised on the context as `TabICLRegressor`'s `y_scaler_` does — so a credit table
reaches the model in training exactly as a real table would at prediction
(`tests/test_train_predict_consistency.py`). For TabPFN (`prior.encoding: raw`, Exp2) the table goes
in as recorded — gaps as NaN, features untransformed, the LGD target on [0, 1]. TabPFN's native
feature preprocessing then runs in training and prediction. The revised synthetic LGD training
path retains context-only target standardization but disables nonlinear target power transforms;
released prediction preprocessing is unchanged. This applies equally to generic and credit tasks.
See the [target-space repair](CONFIG_REFERENCE.md#experiment-2-full-continued-pretraining).

Raw TabPFN tasks must retain at least one feature under its native constant-feature removal
on the **context rows**, including its NaN semantics. Query variation cannot make a constant
context usable. An unusable candidate is redrawn within the existing bounded validity budget,
with the same source, slot, table shape and context/query split; it never reduces tables per
update. This is a preprocessing validity condition, not a difficulty or downstream-score filter.
The encoded TabICL/Exp1 stream is unchanged (`tests/test_context_validity.py`).

### 5.4 Sources and verification limits

The pinned library contains the sources used for the broad domain motivation:

- [Baesens et al. (2026), current October revision](../tfm-library/papers/2026/10_Baesens_et_al._Tabular_Foundation_Models_for_Credit_Risk_Prediction_A_Dedicated_Review_and_Empirical_Comparison.pdf):
  credit imbalance, selection, economic cycles and LGD shape, with references to the underlying
  credit literature. This is a survey, not a calibration of this generator.
- [Klein & Hoffart (2026)](../tfm-library/papers/2026/01_Klein_and_Hoffart_Position_Foundation_Models_for_Tabular_Data_within_Systemic_Contexts_Need_Grounding.pdf):
  motivation for domain structure. It is a position paper, not experimental evidence for the
  cutoff frequencies or strengths used here.

The original prior review recorded a full reading of Li, Zhang & Zhao (2020),
*Modeling Loss Given Default Regressions*, section 2.1. Its recorded ZOIB coefficients anchor
the table above. That primary document is not in the current TFM snapshot, so this documentation
revision does not independently reverify the transcription. Recover and check the primary
version before presenting exact coefficients as a manuscript quotation.

Other source attributions remain indirect: BCBS (2005)/Vasicek (2002) for asset correlations;
Brown & Mues (2012) for class imbalance; Banasik & Crook (2007) for selection; Bellotti & Crook
(2012) for macroeconomic effects; and Ospina & Ferrari (2010), Li et al. (2016), Yashkir &
Yashkir (2013) and the recovery literature cited by Li, Zhang & Zhao for LGD models and shapes.
Verify their primary text and bibliographic metadata before citing exact numbers. The current
library does not supply those primary credit-model papers.

Earlier notes drew real-data R² ranges from a secondary preprint. Those ranges were not verified
against the primary studies and are not used as calibration targets or success thresholds.
The generator's broadening ranges, mixture shares and mechanism probabilities are declared
project choices; qualitative literature motivation does not make them empirically estimated.

## 6. Per experiment

Exp1 and Exp2 register their own prior geometry; Exp3 inherits Exp2 through `prior_file`.
The prior *design* differs as below; the training knobs (LR, steps, optimizer,
L2-SP, freeze strategy) are in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

| | prior | init | steps |
|---|---|---|---|
| **Exp0** | Prior, model, capacity and retention routes; diagnostics rather than research arms | scratch and released | stage-specific; six 600-step recovery routes plus bounded probes |
| **Exp1** | `credit_fraction` 0 / 0.25 / 0.5 / 1 of the literature-grounded credit prior × 3 seeds = 12 main arms per track; 25% added after the original campaign | scratch | 12,500 |
| **Exp2** | two backbones × four mixtures (0 / 0.25 / 0.5 / 1) × two fixed learning rates × two paired seeds = 32 arms per track | released TabICLv2 and TabPFN-3 | 1,500 |
| **Exp3** | fixed 50% mixture; no decay, L2-SP, DiagScale and Q/V LoRA, six recipes × two backbones × two seeds = 24 new arms per track; reuse eight Exp2 decay baselines | released TabICLv2 and TabPFN-3 | 1,500 |

**Exp2 is continued pre-training with a published precedent** — TabPFN-Wide (Kolberg et al. 2026)
extends TabPFN-v2 through continued pretraining on a customised synthetic prior. It is a precedent,
not a validated optimal recipe for TabPFN-3; the registered model-specific choices and limitations
are in CONFIG_REFERENCE.md. Exp2 sweeps `credit_fraction` all the way to 1.0
to measure the effect of increasing credit exposure. Mixture results in other TFMs motivate this
factor but do not establish an interior optimum here. All mixtures and both fixed rates are
reported; Exp1 does not select an Exp2 winner. Exp3 replaces the abandoned scratch-scale study.

**For scale:** TabICLv2's reported budget is `500k + 40k + 10k` updates at batch 64 = 35.2M
tables; Exp1 uses about 2.3% of that update count, at shorter fixed table lengths. This does
not imply the same fraction of compute or information exposure. Interpret results at their
tested budget, rather than as evidence of convergence.

## 7. Limitations and future work

**Signal strength is a limitation to measure, not an unresolved launch setting.** The current
prior ranges are frozen. Synthetic ExtraTrees pseudo-R² and real-data benchmark R² measure
different tasks and procedures; their difference alone is not a bug or a calibration target.
Any future signal-strength ablation needs a separately declared comparison and affected Exp0
validation. Do not retune a running campaign from its assessment results.

**Use the pinned upstream packages.** New local setup is described in README; VSC dependencies
come from the accepted immutable image. `python -m src.utils.preflight --dependencies-only`
checks the selected environment. The TabICL Git commit in `pyproject.toml`
includes the graph prior missing from its same-version PyPI wheel. TabPFN-3 training uses
`tabpfn==9.1.0`, including the upstream per-estimator regression-target loss correction.
The package version does not change the explicitly selected TabPFN-3 model generation.

The main comparison measures the entire configured credit-prior treatment, including its
encoding, filtering and cardinality differences. It does not identify a target-only effect.
The component study removes one named component from the 50% mixture at a time; it cannot
by itself attribute every difference between the complete treatment and the upstream control.
Library pin checked for this clarification: `de324efcca55be392feb0c01830b3b6ce364587f`.
