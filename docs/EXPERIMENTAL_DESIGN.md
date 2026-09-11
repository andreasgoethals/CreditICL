# Experimental design

What the three experiments ask, why the design can fail, and the premises it rests on. This
describes what is **built** (the `config/Exp{1,2,3}_*.yaml` sweeps); the prior's mechanics are in
[PRIORS.md](PRIORS.md), every knob's value in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md), and what
each run actually did in [RUNS.md](RUNS.md). Library pin `52dab01`.

## Contents

1. [The claim, and how it fails](#1-the-claim-and-how-it-fails)
2. [Verified premises — and two corrections](#2-verified-premises)
3. [The three experiments](#3-the-three-experiments)
4. [Matched compute, seeds, and the two phases](#4-matched-compute-seeds-and-the-two-phases)
5. [Evaluation](#5-evaluation)
6. [Risks](#6-risks)
7. [Open decisions](#7-open-decisions)

---

## 1. The claim, and how it fails

> Encoding credit-risk-specific structure into a TFM's synthetic pretraining prior improves
> downstream credit-risk performance, and the improvement is **selective** to the sub-task whose
> structure was encoded.

Selectivity is what makes this falsifiable and distinguishes it from O'Prior. An average improvement
over the default prior would be consistent with three explanations we cannot otherwise separate:
(a) we targeted credit; (b) we targeted *something*, and any narrowing helps; (c) our prior is simply
harder. Only selectivity discriminates (a) from (b) and (c).

**The claim is wrong if:**

- the credit prior (`credit_fraction > 0`) does **not** beat the control (`credit_fraction = 0`) on
  the matching track — the intervention does not transfer;
- there is **no selectivity** — the LGD boundary-mass mechanism helps PD as much as LGD, or the PD
  base-rate/Vasicek mechanism helps LGD as much as PD — in which case we shaped a *harder* prior, not
  a *domain* one;
- the gains vanish under **temporal or out-of-domain** evaluation — an i.i.d. artefact of the kind
  Purucker 2026 documents for credit data;
- fine-tuning (Exp2) buys credit gains only by **eroding out-of-domain generality** — measured
  directly, not assumed.

Each is a publishable negative. The design is built to learn which is true, not so that credit wins.

**One honest limitation.** A generic-realism control arm (O'Prior-style: realism *without* credit
targeting) was designed but **not built** — see [§3](#3-the-three-experiments). Without it we cannot
fully separate "credit-specific structure" from "any added realism". The **double dissociation across
the two tracks** ([§5.2](#52-the-double-dissociation)) is the substitute control: it rules out
"narrower/harder" using only our own experiment, with no external control domain.

## 2. Verified premises

Everything here was read from source, not assumed. **Two corrections matter — the original framing
overstated two claims a code-reading reviewer would catch, and the honest versions are load-bearing.**

**2.1 The regression target *is* standard-scaled (verified).** `GraphSCM.__call__` clamps at ±4σ
(`outlier_removing`) then `standard_scaling`; `TabICLRegressor` mirrors this at inference and
inverse-transforms the predicted quantiles. Prior and inference agree.

**2.2 CORRECTION — bounded/bimodal targets are a *frequency* gap, not a structural absence.**
Standard-scaling is affine, therefore shape-preserving: a U-shaped target with two atoms arrives as a
U-shaped target with two atoms on a rescaled axis. Standardisation destroys the **[0,1] support**,
not the bimodality — and atom-producing mechanisms already exist (`outlier_removing` clamps → point
masses; piecewise-constant function families; `rand_kumaraswamy_act` maps to exactly [0,1] with
p≈0.5). The defensible claim: nothing in the prior *targets* boundary mass in real LGD's range, and
nothing aligns the output support with [0,1]. That is weaker than "absent" and, crucially,
**measurable** — the `0.3` prior-visualisation notebook measures it.

**2.3 CORRECTION — TabICL's prior is *not* class-balanced by default.** `_prior_config.py` fixes
`"balanced": False`, so the active path (`MulticlassAssigner`, `mode="rank"`) draws the cut at a
uniformly-random data row → minority rate ≈ Uniform(0,1), so ~10% of binary prior tasks already sit
below 5% minority. So prior-side imbalance is **uncontrolled and unmeasured, not missing** — a better
story: we shape an existing distribution rather than invent a mechanism. (O'Prior is the opposite: it
filters out "collapsed or severely imbalanced" tasks by design.)

**2.4 The head can already represent atoms.** The regressor is a **999-quantile** pinball head, which
represents a point mass exactly (an atom at 0 = quantile levels 0.001…mass all mapping to 0). So
representability is *not* the bottleneck; the real head-side issues are point-prediction extraction
from a bimodal predictive (median/mean land in the sparse interior) and the lack of a [0,1] support
constraint (clipping is a free baseline, not a contribution). This is why a ZOIB/mixture head is held
in reserve rather than built.

**2.5 The predictability filter — characterised.** Three filters, verified in code and against the
TabICLv2 paper: a **statistical** one (ExtraTrees OOB bootstrap test, reject if `pval ≥ 0.05`), a
**triviality** one (reject too-easy), and a **structural** one (reject DAGs where x, y share no
ancestors). Paper rate: "roughly 35% classification and 25% regression datasets are filtered". Two
nuances shape Exp1: it is a *significance* test, not an R² floor (it removes *no-signal* DGPs more
than *low-signal* ones), and the authors show filtering *improves* convergence — so the `banded`
variant contradicts a published result *with evidence behind it*, and must beat that argument.

**2.6 Our data does not match a single premise.** Measured from `data/raw/`: Freddie LGD is genuinely
U-shaped (11.4% at 0, 8.1% at 1); LendingClub LGD is unimodal, left-skewed, peaks at 0.80–0.85; AXA
is fully interior. PD base rates span 6.7% (GMSC) to 40% (myhom). **Consequence:** the prior must be
a *family* over boundary mass and base rate, not a single shape — else it overfits one dataset.

## 3. The three experiments

All three share the same architecture (TabICLv2's, vendored from NanoTabICL — the only public TFM
with regression *and* an open prior generator), the same real datasets, and the same frozen
evaluation ([§5](#5-evaluation)). They differ only in the prior sweep and how training starts.
`credit_fraction` is the master switch throughout: the share of each batch drawn from our
credit-targeted path, the rest from the unmodified TabICL prior; `0.0` is the control.

| | **Exp1 — which prior?** | **Exp2 — fine-tune?** | **Exp3 — run long** |
|---|---|---|---|
| start | from scratch | warm-start released TabICLv2 | from scratch |
| swept knobs | `credit_fraction {0, .5, 1} × filter.mode {tabicl, banded, off} × intensity {mild, aggr} × 3 seeds` | `credit_fraction {0,.25,.5,.75,1} × init.strategy {full, icl_only, head_only} × l2sp_alpha {0, .003} × lr {1e-6, 1e-5}` | `credit_fraction {0, Exp1-winner}` |
| arms/track | 45 (control-dedup) | 60 (1 seed) | small |
| steps | 12,500 | 10,000 | 100,000 |
| optimizer | Muon | AdamW | Muon |
| asks | does the credit prior beat control, and does the cheap `banded` removal help? | does fine-tuning transfer, and at what **out-of-domain** cost? | confirm the winner at length |

**Exp1** screens the prior. Its cheapest sharp result is the **predictability filter**: `banded` keeps
only tasks whose ExtraTrees pseudo-R² sits in credit's low-signal range — a *removal*, so it cannot
be accused of adding capacity, and it contradicts a published convergence claim ([§2.5](#2-verified-premises)).

**Exp2** asks the fine-tuning question, and is where **out-of-domain retention** is measured: `l2sp_alpha`
(pull toward the loaded weights) and `init.strategy` (freeze depth) are the levers that trade credit
gain against forgetting. Its hyperparameters are literature-grounded and genuinely unsettled — see
[CONFIG_REFERENCE](CONFIG_REFERENCE.md) and the memory note on TFM fine-tuning; TabICL specifically is
the architecture that degrades most under naive full fine-tuning, which is *why* freeze depth is swept.

**Exp3** runs the single Exp1-winning prior long, against its control.

**What was designed but not built.** An **O'Prior-style generic-realism arm** (the ideal causal
control) and an **unrealistic-but-complex arm** (TabForestPFN's counter-hypothesis that complexity
beats realism). Both are deferred: O'Prior's realism engine is non-trivial to reimplement faithfully,
and a weak reimplementation would be a strawman that invalidates the headline. Their absence is the
limitation named in [§1](#1-the-claim-and-how-it-fails); the two-track dissociation stands in.

The prior mechanisms each experiment can switch on — PD's Vasicek one-factor defaults, controlled
base rate, underwriting rules, reject-inference selection, high-cardinality categoricals; LGD's
boundary atoms and interior shape; and the shared shift/missingness/noise — are described in
[PRIORS.md](PRIORS.md).

## 4. Matched compute, seeds, and the two phases

**Matched across arms, not across papers.** The comparison is *between arms*, so every arm in an
experiment sees the same optimizer steps and datasets. It is **not** a match to O'Prior's protocol
(the configs follow TabICLv2 stage-1: 1,024 rows/table, 1–100 features), so absolute numbers are not
comparable to theirs — only our arm-to-arm contrasts are. **Two traps:** turning the filter *off*
makes generation cheaper, so wall-clock and credits are not comparable across filter arms (match on
steps/datasets, and report the rejection rate per arm); and TabICLv2's own ablations show a strong
prior×architecture interaction, so a prior change may need retuning — **log every arm's loss curve
and treat divergence as a reportable outcome, never tune it away silently.**

**Seeds.** 3 per arm in Exp1 (Exp2 uses 1 — 60 arms already, seeds come after a winner). Report
mean ± spread; never rank arms on a gap smaller than the seed spread.

**Two phases, and the order is a fact about the data.** Phase 1 (`pretrain_{pd,lgd}.slurm`) trains one
checkpoint per arm; phase 2 (`benchmark.slurm`) scores every checkpoint **plus a shared reference
column** (released TabICLv2, TabPFN-3, CatBoost, linear) at array index *N*. Phase 2 cannot start
until phase 1 has written the checkpoints. The reference sits **inside the same array** so it is
scored by the same code, same day, same context cap, seeds and splits as our arms — every comparison
this project got wrong, it got wrong by scoring the two sides through different paths. Evaluation is
*not* inside the training job for the same reason: an arm would otherwise be benchmarked by whatever
the code looked like the hour it finished.

## 5. Evaluation

Frozen before any prior tuning; the split lives in every config's `eval:` block. `select_on: dev` —
choosing the prior on the holdout would invalidate the experiment.

**5.1 What is compared, and on what.** Every checkpoint against the shared reference, on the real
credit datasets and the out-of-domain suites. The split is fixed:

| track | development (prior choice + all iteration) | holdout (untouched until the end) |
|---|---|---|
| **PD** (14) | gmsc, lendingclub, taiwan_creditcard, german, myhom | vehicle_loan, hackerearth, cobranded, bank_status, thomas, loan_default, home_credit, hmeq, algorithmwatch |
| **LGD** (7) | lgd_lendingclub, base_model, heloc | loss2, axa, base_modelisation, lgd_freddie |

Row/feature caps for the in-context models are applied uniformly and **recorded in every result row**
(a silent subsample makes a model look worse for a reason nothing explains).

**Metrics** (all computed in one run). PD: ROC-AUC, PR-AUC, Brier, log-loss, KS, calibration slope —
**never accuracy alone** (at a 7% base rate "never defaults" already scores 0.93), and ECE with MCE
because the sparse high-score tail is where the defaults are. LGD: pinball, CRPS, interval coverage,
**boundary-mass calibration** (predicted vs observed P(y=0), P(y=1) — the metric that most directly
tests the hypothesis, which no cited paper reports), with R²/RMSE alongside and the decoding rule
stated (it changes the number materially). Success criteria are set to the LGD literature (published
R² ≈ 0.04–0.15 linear, 0.10–0.25 beta, 0.20–0.43 tree ensembles), not classification expectations.

**5.2 The double dissociation — the centrepiece.**

| mechanism enabled | predicted effect on **LGD** | predicted effect on **PD** |
|---|---|---|
| bounded target + boundary mass | **improves** | ~neutral |
| controlled base rate + Vasicek defaults | ~neutral | **improves** |

A crossed dissociation is the strongest evidence the design can produce without an external control
domain: it rules out "narrower" and "harder". *No* dissociation with credit still winning on average
supports only the weaker "harder prior" reading — and must be reported as such.

**5.3 Out-of-domain retention is a first-class axis, not an appendix.** After training on a
credit-targeted prior, score general/out-of-domain benchmarks to measure whether in-domain gains were
bought at out-of-domain cost — Kolberg's Spearman-ρ-vs-base-model instrument (they report 0.9935),
and Hoo 2026 predicts exactly this trade. **If domain-targeting costs general performance, that is
the finding** — Exp2's `2.1`/`2.2` notebooks plot the credit-vs-out-of-domain curve directly.

**5.4 Temporal splits.** Purucker 2026 shows in-context TFMs lose to tuned GBDTs the moment splits
become temporal or grouped; an i.i.d.-only credit result is not credible. Feasibility is
dataset-dependent (Freddie has an origination date; Home Credit has none) — see [§7](#7-open-decisions).

## 6. Risks

| risk | why it is real | mitigation |
|---|---|---|
| our training loop is subtly wrong | it is ours; a bug invalidates all arms equally and invisibly | reproduce the control in the expected range; unit-test the pinball loss; verify a fixed seed reproduces exactly (all in `tests/`) |
| no generic-realism arm | can't fully separate "credit-specific" from "any realism" | the two-track double dissociation ([§5.2](#52-the-double-dissociation)) as substitute control; state the limit |
| prior×architecture interaction | TabICLv2 ablations show divergence on the old prior | log all loss curves; report divergence as an outcome |
| nano→frontier gap | untested at scale (O'Prior's own caveat) | state as a limitation; scope any frontier confirmation separately |
| credit is not distinctive | if the default prior already covers credit's regime the premise is weak | the `0.2`/`0.3` notebooks measure this before credits are spent |
| meta-overfitting | 21 datasets, many iterations | frozen dev/holdout split, `select_on: dev` |
| i.i.d.-only result | Purucker 2026 | temporal splits where the data allows ([§7](#7-open-decisions)) |

## 7. Open decisions

1. **Temporal-split protocol** — which datasets carry a usable date/group column; needs a per-dataset
   audit before the protocol is fixed ([§5.4](#54-temporal-splits)).
2. **The generic-realism arm** — whether to reimplement O'Prior's engine (reading their public repo
   first, to avoid a strawman) or accept the two-track dissociation as the control.
3. **ZOIB/mixture LGD head** — held in reserve behind decoding + support constraints
   ([§2.4](#2-verified-premises)); revisit only if those are insufficient.
4. **Which prior fills Exp2/Exp3's `FILL_FROM_EXP1`** — the Exp1 winner; if Exp1 gives no clean
   winner, run the downstream experiments across several prior settings rather than betting on one.
