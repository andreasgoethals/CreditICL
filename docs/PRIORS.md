# PRIORS — how TabICLv2 makes its synthetic data, and what we change

A tabular foundation model never sees a real table during pretraining; it sees millions of
**synthetic** ones from a generator called the **prior**. The prior decides what "a table" means to
the model, so it decides what the model is good at. This project changes that generator. Here is
what upstream does, what we add, and why every mechanism — and every number in it — comes from the credit-risk
literature rather than from the datasets the models are scored on.

Original in-repo source audit (pin `de324efcca55be392feb0c01830b3b6ce364587f`; later checks record their own pins): paper `tfm-library/papers/2026/02_Qu_et_al._TabICLv2_*.pdf`; code
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

`GraphSCM` creates classification targets with `CategoricalConverter`; `Reg2Cls` and
`MulticlassAssigner` belong to the older `SCMPrior` path. Categorical conversion can produce
imbalance, but an exact frequency of rare binary tasks must be measured, not inferred from
that older path. Our credit mechanism deliberately models a default rate and common shocks.

> **Leave every `use_corrected_*` converter flag at its default.** `Converter._transform` carries a
> known upstream bug (*"this is the wrong version, but it was used in the experiments"*); upstream
> keeps it so the released checkpoints stay reproducible, and "fixing" it makes our arms incomparable
> to the reference model.

## 5. What we change

Three rules govern every addition:

1. **The control is exactly TabICLv2's prior.** `credit_fraction: 0.0` runs the original path with
   none of our code on it — enforced by a test, including that `prior.base` never carries our values.
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


`credit_fraction` **mixes** rather than replaces: at 0.5, each task slot independently has 50% probability of using our prior. A model that
only ever saw credit tables would be a credit model, not a foundation model — which is what the
out-of-domain suites check.

### 5.1 PD — a rare, correlated binary event

`mode: mechanism` is the **Merton/Vasicek one-factor model** (Vasicek 2002), the model behind the
Basel IRB risk-weight formula: asset value = √ρ·Z + √(1−ρ)·ε, default when it falls below Φ⁻¹(PD).
Defaults **cluster** through the systematic factor Z, shared by every borrower of a period — which an
i.i.d. prior cannot express. ε is the table's own SCM latent (so the features stay predictive), of
which `signal_share` is kept and the rest is risk no file records.

| knob | value | source |
|---|---|---|
| `rho_range` | 0.03 – 0.24 | Basel IRB asset correlations, from other retail (3 %) to corporate (24 %) (BCBS 2005) |
| `base_rate_range`, `base_rate_log` | 1 % – 50 %, log-uniform | lower end: the most imbalanced setting of Brown & Mues (2012), who study 30 % down to 1 % defaulters; upper end balanced, so the credit prior is never narrower than the control on class balance; log-uniform gives the imbalanced regime the literature stresses equal weight per factor of two |
| `signal_share_range` | 0.05 – 0.95 | **wide**: discrimination varies across portfolios and no value transfers |
| `cohort` | 1 – 40 periods, Z ~ N(0, 1) | the Vasicek model's standard-normal factor; 40 periods as in Li, Zhang & Zhao (2020) |
| `macro_feature_prob` | 0.5 | macroeconomic variables are explanatory variables in credit models (economic-cycle dependency; Bellotti & Crook 2012, surveyed in Baesens et al. 2026) — half the tables record the period factor as a column |
| `selection` | reject 0 – 50 %, sharpness 0.3 – 1.0 | only approved applicants are observed (support truncation, motivated by selection literature; Banasik & Crook 2007, surveyed in Baesens et al. 2026); acceptance rates are not published, so **wide** |
| `rules` | 0 – 3 cut-offs per table | hard underwriting cut-offs, **from Klein & Hoffart (2026), a position paper with no experiments — the most speculative mechanism, labelled so** |
| `woe_prob` | 0.5 | scorecard-style monotone binning of the risk driver; practitioner convention, no rate published — **uninformative** |
| label noise | **none** | asymmetric label noise (cures booked as non-default) was in the prior until 29-09-2026 with rates from nowhere; removed |

The base-rate range is the **unconditional applicant** PD parameter. A finite cohort,
macroeconomic shocks, acceptance selection and label noise can change its realised default rate.
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

| knob | value | published value (Li, Zhang & Zhao) |
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

The spreads (SD 1 on the logit scale, ranges several times the published effects) are ours and
deliberate: the paper reports one portfolio, and a prior narrower than the spread of real portfolios
would be a prior for that portfolio only. Predictive difficulty is not targeted; the literature's
out-of-sample R² for real LGD — 0.04–0.15 linear, 0.10–0.25 beta regression, 0.20–0.43 tree
ensembles (Loterman et al. 2012, as reported by Marin 2025) — is used only to check the generated
tables are in the right regime.

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
in as recorded — gaps as NaN, features untransformed, the LGD target on [0, 1] — because TabPFN's own
preprocessing does the rest, in training and at prediction alike. The control keeps upstream's
training encoding in both cases.

### 5.4 Sources

In the library (`tfm-library/`, cite by path): Baesens et al. (2026),
`papers/2026/07_Baesens_et_al._Foundation_Models_for_Credit_Risk_Prediction_A_Game_Changer.pdf` — the
survey the imbalance, reject-inference, economic-cycle and LGD-bimodality statements are taken from,
with its own citations; Klein & Hoffart (2026),
`papers/2026/01_Klein_and_Hoffart_Position_Foundation_Models_for_Tabular_Data_within_Systemic_Contexts_Need_Grounding.pdf`.

Read in full for this prior (not in the library — **to add**):

- Li, P., Zhang, X., Zhao, X. (2020). *Modeling Loss Given Default Regressions.* Working paper, Office of
  the Comptroller of the Currency and FDIC; first version 31-05-2017, this version 11-05-2020. Section
  2.1 (the DGP, eqs. 1–6 and the parameter values quoted above).
- Marin, J. (2025). *Loss Given Default Prediction Under Measurement-Induced Mixture Distributions: An
  Information-Theoretic Approach.* Preprint, October 2025 — used only for its literature review
  (Loterman et al. 2012's R² ranges; Altman et al. 2005 on downturn recoveries).

Cited through the sources above, or checked against the publisher's abstract only — **to add and
verify in full** before the paper cites a number from them directly:

- Basel Committee on Banking Supervision (2005). *An Explanatory Note on the Basel II IRB Risk Weight
  Functions.* BIS. (The asset-correlation range: QRRE 0.04 and the other-retail formula confirmed in a
  search result; the corporate 0.12–0.24 and other-retail 0.03–0.16 bounds are the standard IRB
  formulas and should be checked against the note.)
- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk* 15(12).
- Brown, I., Mues, C. (2012). An experimental comparison of classification algorithms for imbalanced
  credit scoring data sets. *Expert Systems with Applications* 39(3). (Imbalance from 70/30 to 99/1.)
- Banasik, J., Crook, J. (2007). Reject inference, augmentation, and sample selection. *European
  Journal of Operational Research* 183(3).
- Bellotti, T., Crook, J. (2012). Loss given default models incorporating macroeconomic variables for
  credit cards. *International Journal of Forecasting* 28(1).
- Ospina, R., Ferrari, S. L. P. (2010). Inflated beta distributions. *Statistical Papers* 51.
- Li, P., Qi, M., Zhang, X., Zhao, X. (2016). Further investigation of parametric loss given default
  modeling. *Journal of Credit Risk* 12(4).
- Yashkir, O., Yashkir, Y. (2013). Loss given default modeling: a comparative analysis. *Journal of
  Risk Model Validation* 7(1).
- Loterman, G., Brown, I., Martens, D., Mues, C., Baesens, B. (2012). Benchmarking regression
  algorithms for loss given default modeling. *International Journal of Forecasting* 28(1).
- Altman, E., Brady, B., Resti, A., Sironi, A. (2005). The link between default and recovery rates.
  *Journal of Business* 78(6).
- Asarnow, E., Edwards, D. (1995); Qi, M., Zhao, X. (2011); Altman, E., Kalotay, E. (2014) — as cited by
  Li, Zhang & Zhao for the bimodal and trimodal shape.

The journal details of the "to verify" list are from memory and search results, not from the papers;
check each before it goes into the manuscript's bibliography.

## 6. Per experiment

Each config owns its prior inline — they are *not* shared, because the three experiments do genuinely
different things. The prior *design* differs as below; the training knobs (LR, steps, optimizer,
L2-SP, freeze strategy) are in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

| | prior | init | steps |
|---|---|---|---|
| **Exp0** | Exp1's prior, `credit_fraction` 0 / 0.5 / 1 and the Exp2 paths — a debug suite, no result | scratch and released | 600 |
| **Exp1** | `credit_fraction` 0 / 0.25 / 0.5 / 1 of the literature-grounded credit prior × 3 seeds = 12 main arms per track; 25% added after the original campaign | scratch | 12,500 |
| **Exp2** | Exp1's prior. TabICLv2 and TabPFN-3 × `credit_fraction` 0 / 0.5 / 1 × 3 seeds = 18 arms per track, with recipes adapted from published continued pretraining | released TabICLv2 and TabPFN-3 | 10,000 |
| **Exp3** | Exp1's prior: the control and the mix Exp1 selects × 2 paired seeds | scratch | 200,000 |

**Exp2 is continued pre-training with a published precedent** — TabPFN-Wide (Kolberg et al. 2026)
extends a model through continued pretraining on a customised synthetic prior and reports it matches
or exceeds the base; its recipe is what Exp2 follows. It sweeps `credit_fraction` all the way to 1.0
because the model *already knows* the original prior, so how much of ours to add is the question, and
Mitra (Zhang et al. 2025) finds mixtures beat single priors — expect an interior optimum, with 1.0
included to measure what forgetting the original prior costs.

**For scale:** TabICLv2's own budget is `500k + 40k + 10k` steps at batch 64 ≈ 35M datasets, ~24.5
GPU-days per model; Exp1 is 2.3 % of that, Exp3 36 %. **Every result here is a statement about priors at a fixed,
small budget, and must be written that way.**

## 7. Open items

**LGD signal strength needs a design decision before the full campaign.** The existing notebook
reports median synthetic ExtraTrees pseudo-R² near 0.05; reported real-data R² of 0.20–0.43
(Loterman et al. 2012, via Marin 2025) is a different statistic on different data. The difference
is not by itself a code failure or a justified calibration target. The earlier real-book median
0.42 also included outcome information (`EXPERIMENTAL_DESIGN.md` §7 item 5). Review the synthetic
diagnostics and primary literature, then freeze the range and any signal-strength ablation without
tuning to the evaluation tables. If the generator changes, repeat the affected Exp0 validation.

**Use the pinned upstream packages**: `python -m pip install -e '.[dev,eval]'`, then
`python -m src.utils.preflight --dependencies-only`. The TabICL Git commit in `pyproject.toml`
includes the graph prior missing from its same-version PyPI wheel. TabPFN-3 training uses
`tabpfn==9.1.0`, including the upstream per-estimator regression-target loss correction.
The package version does not change the explicitly selected TabPFN-3 model generation.

The main comparison measures the entire configured credit-prior treatment, including its
encoding, filtering and cardinality differences. It does not identify a target-only effect.
The component study removes one named component from the 50% mixture at a time; it cannot
by itself attribute every difference between the complete treatment and the upstream control.
Library pin checked for this clarification: `de324efcca55be392feb0c01830b3b6ce364587f`.
