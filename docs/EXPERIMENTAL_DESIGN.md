# Experimental design

**Status:** proposal, 2026-08-05. Library pin `21d555a`.
Sections marked **[OPEN]** need a decision before implementation.

---

## 0. The claim, stated so it can fail

> Encoding credit-risk-specific structure into a TFM's synthetic pretraining
> prior improves downstream credit-risk performance **more than** encoding
> generic realism does, and the improvement is **selective** to the sub-task
> whose structure was encoded.

The second clause is what makes this falsifiable and what distinguishes it
from O'Prior. An average improvement over the default prior would be
consistent with three explanations we cannot otherwise separate: (a) we
targeted credit; (b) we targeted *something*, and any narrowing helps;
(c) our prior is simply harder, per TabForestPFN. Only **selectivity**
discriminates (a) from (b) and (c).

**Pre-registered falsifiers.** The claim is wrong if:

- Arm B (general realism) matches or beats arm C (credit-targeted) on credit; or
- arm D (unrealistic-but-complex) beats arm C — realism is not the active
  ingredient; or
- the LGD-specific and PD-specific components help both sub-tasks equally —
  no selectivity, so we measured "harder prior", not "domain prior"; or
- arm C's gains vanish under temporal splits — we measured an i.i.d. artefact
  of exactly the kind Purucker 2026 documents for credit data.

Each is a publishable negative. The design is built so that we learn which
one is true, not so that arm C wins.

---

## 1. Verified premises — and two corrections

Everything here was read from source, not assumed. **The corrections matter:
the original framing of this project overstated two claims in ways a
code-reading reviewer would catch.**

### 1.1 The regression target *is* standard-scaled — verified

`GraphSCM.__call__` in TabICL's `prior/_graph_scm.py`:

```python
if self.regression:
    y = data["y_num"]
    y = outlier_removing(y.float(), threshold=4)   # clamp at ±4σ
    y = standard_scaling(y)                        # zero mean, unit variance
```

`TabICLRegressor` mirrors this at inference with a plain `StandardScaler`,
and `inverse_transform`s the predicted quantiles back. So prior and inference
agree. Confirmed.

### 1.2 CORRECTION — bounded/bimodal targets are a **frequency** gap, not a structural absence

Standard-scaling is **affine, therefore shape-preserving**. A U-shaped
target with two atoms arrives at the model as a U-shaped target with two
atoms on a rescaled axis. Standardisation destroys the **[0,1] support**, not
the bimodality.

Worse for the strong claim, atom-producing mechanisms already exist in the
prior:

- `outlier_removing(threshold=4)` **clamps**, which literally creates point
  masses at the clamp bounds;
- `rand_tree_func` and `rand_discretization_func` are **piecewise constant**,
  so finite-valued (atom-ful) targets occur naturally;
- in NanoTabICL's `prior.py`, `rand_kumaraswamy_act` maps a numerical output
  column to **exactly [0,1]**, applied with probability ≈0.5
  (`return x if randbool() else rand_kumaraswamy_act(x)`).

**The defensible claim:** nothing in the prior *targets* boundary mass in the
range real LGD exhibits, and nothing aligns the model's output support with
[0,1]. That is a claim about **frequency and alignment**, it is weaker than
"structurally absent", and — crucially — it is **measurable**. §3.1 makes
measuring it the first deliverable.

### 1.3 CORRECTION — TabICL's prior is **not** class-balanced by default

`_prior_config.py` sets `"balanced": False` as a fixed default. So
`BalancedBinarize` (median split → exactly 50/50) is *not* the active path.
`MulticlassAssigner` is, and in `mode="rank"` it draws the cut as a
**uniformly random data row**:

```python
boundary_indices = torch.randint(0, T, (self.num_classes - 1,), device=device)
boundaries = input[boundary_indices]
```

For binary targets that makes the minority rate approximately Uniform(0,1),
so on the order of **10% of binary prior tasks already sit below 5%
minority**. `mode="value"` (a N(0,1) cut against a standardised target) gives
a comparable tail. The only guard is `cls_sanity_check`: both classes present
on both sides of the split, with 10 permutation retries.

**Consequence:** on TabICL, prior-side imbalance is **uncontrolled and
unmeasured, not missing**. This is a *better* story — we shape an existing
distribution rather than inventing a mechanism, and the prior's implied
base-rate distribution is itself a reportable finding.

**O'Prior is the opposite** and here the original claim holds exactly: its
§2.4 quality control rejects tasks with *"collapsed or severely imbalanced
support classes"*. Severe imbalance is filtered out of O'Prior by design.

### 1.4 The predictability filter — fully characterised

Three independent filters, all verified in code and against the TabICLv2
paper:

| filter | flag | mechanism |
|---|---|---|
| statistical | `filter_unpredictable_datasets` | `ExtraTreesRegressor(n_estimators=25, bootstrap=True, oob_score=True, max_depth=6)` on the **full** dataset; 200-resample bootstrap test that OOB MSE beats the mean baseline; reject if `pval >= 0.05` |
| triviality | `remove_trivial_datasets` | reject if `RMSE <= trivial_dataset_threshold * dummy_RMSE` (too *easy*) |
| structural | `filter_unpredictable_graphs` | `check_x_y_ancestors_overlap` — reject DAGs where x and y share no ancestors |

Rate, from the paper (not inferred): *"In pretraining stage 1, roughly **35%
classification and 25% regression** datasets are filtered."*

**Two nuances that shape the experiment.** It is a **significance** test, not
an R² floor — a weak-but-real signal at n≈1024 usually *passes*, so it
removes *no-signal* DGPs more than *low-signal* ones. And the authors show
(their Fig. 10) that **filtering improves pretraining convergence**. Our
counter-hypothesis therefore contradicts a published result with evidence
behind it; we must beat the convergence argument, not ignore it.

### 1.5 The head can already represent atoms

TabICLv2's regressor uses a **999-quantile pinball loss**
(`--regression_method quantile --num_quantiles 999`, `max_classes=0`, linear
target embeddings, `layernorm_nobias`).

A 999-quantile grid represents a point mass **exactly**: an atom at 0 with
mass 0.11 is simply quantile levels 0.001…0.11 all mapping to 0. Quantile
heads are among the *better* parametrisations for atoms — strictly better
than a Gaussian head. So **representability is not the bottleneck**, and this
is the main reason the design below demotes the head from a co-equal factor.

The real head-side problems are different, and cheaper to fix:

1. **Point-prediction extraction.** For a U-shaped predictive, the median and
   the mean both land in the sparse interior where almost no mass lives. Any
   RMSE/R² computed from a point prediction is measuring the wrong thing.
2. **No support constraint.** Nothing keeps predicted quantiles inside [0,1].
   Clipping is a free post-hoc baseline, not a contribution.
3. Hoo 2026's convex-hull limitation actually *helps* here: predictions stay
   inside the hull of context targets, which for LGD is already ⊂ [0,1].

### 1.6 Our data does not match the stated premise uniformly

Measured directly from `data/raw/` (2026-08-05). Fraction of mass at
*exactly* 0 and *exactly* 1:

| dataset | n | at 0 | at 1 | shape |
|---|---|---|---|---|
| `0006.lgd_freddie` | 16,002 | **11.4%** | **8.1%** | genuinely U-shaped (15.1% in [0,.05), 10.1% in [.95,1]) |
| `0007.lgd_lendingclub` | 5,627 | 1.5% | 0.3% | **unimodal, left-skewed, peaks at 0.80–0.85** |
| `0003.axa` (recovery rate) | 2,545 | 0.0% | 0.0% | fully interior |

PD base rates: GMSC **6.7%**, Home Credit **8.1%**, HMEQ **20.0%**, Taiwan
**22.1%** (4 of 14 checked).

**Consequences.** "Bimodal with point masses at 0 and 1" is true of Freddie
and false of the other two — so the prior must be a **family over boundary
mass and shape**, not a single shape, or we overfit the prior to one dataset.
And "1–5% positives" is stronger than our data supports; the imbalance
component must be a **distribution over base rates** calibrated to what we
actually have (§3.3).

---

## 2. Venue, architecture, and matched compute

### 2.1 Venue: NanoTabICL architecture + our own training loop

**Decided.** Rationale and the constraint that forces it:

- **NanoTabICL ships no pretraining code.** Its README: *"we currently do not
  provide pre-training code"*, deferring to nanoTabPFN.
- **nanoTabPFN is classification-only**, so it cannot host the LGD arm — and
  LGD is where the novelty is.
- NanoTabICL *does* give us the TabICLv2 architecture with working regression
  (`NanoTabICLv2(max_classes=0, out_dim=999)`), in <170 LOC.
- Happy accident: NanoTabICL's prior **deliberately omits** target
  preprocessing — *"outlier handling, standard scaling … not included as it
  could be done inside the model"* — which puts the exact knob we want to
  manipulate under our control instead of buried in the generator.

So: vendor `model.py`, write `src/train/`, build `src/prior/` as ours.
O'Prior set the precedent that **nano scale is an accepted venue for
controlled prior ablations**; we follow it, and inherit its caveat that
frontier-scale confirmation is untested.

**Known risk:** writing the training loop is the main schedule risk. It is
small (~200 lines) but it is on the critical path, and a subtly wrong loop
invalidates every arm equally and invisibly. Mitigation: §6.

### 2.2 Matched compute — and where it breaks

Follow O'Prior's protocol so our numbers are comparable to theirs:

| | value |
|---|---|
| synthetic datasets per prior | 40,000 |
| rows per table | 512–1,024 |
| features per table | 3–50 |
| batch | 4 tables/step |
| steps/epoch | 1,000 |
| epochs | 10 |
| hyperparameter tuning between conditions | **none** |

**The trap.** Turning the predictability filter *off* makes generation
**cheaper** (no rejection sampling), so wall-clock and credits are **not
comparable across filter arms**. Match on **optimizer steps and datasets
consumed**, never on time or credits, and **report the rejection rate per
arm** so a reader knows how much generation was discarded.

Second trap, from TabICLv2's own ablations: **the architecture diverges when
trained on the old prior** — a strong prior×architecture interaction. A prior
change may need LR/curriculum retuning, which would break the
"no tuning between conditions" claim. **Mitigation:** log training loss
curves for every arm and treat divergence as a *reportable outcome*, not a
bug to tune away. If an arm needs retuning to converge, say so explicitly
rather than quietly tuning it and claiming matched compute.

### 2.3 Seeds

**3 seeds per arm minimum**, 5 preferred. O'Prior reports single runs, which
is a real weakness we can cheaply avoid — and since credits are not the
binding constraint here, seeds are the best available purchase. Report
mean ± spread and never rank arms on differences smaller than the seed
spread.

---

## 3. Phase 1 — the prior (`src/prior/`)

**Design requirement:** arm A must reproduce the TabICLv2 prior *exactly*, so
every component is a gate that can be turned off to recover the baseline.
Components are independently switchable, per O'Prior's ablation discipline,
because that is what makes attribution possible.

```
src/prior/
├── registry.py        name → config resolution; every arm is a named entry
├── base.py            task/episode dataclasses, RNG state handling
├── graph.py           Cauchy DAG sampling            (from NanoTabICL)
├── functions.py       the 8 random function families (from NanoTabICL)
├── converters.py      categorical conversion         (from NanoTabICL)
├── filters.py         the three predictability filters, each parameterised
├── targets/
│   ├── standard.py      outlier_removing + standard_scaling  (arm A default)
│   ├── bounded.py       [0,1] support + sampled boundary mass    (LGD)
│   └── imbalance.py     base-rate distribution for binary        (PD)
├── realism.py         O'Prior-style generic realism             (arm B)
└── complexity.py      deliberately unrealistic complexity       (arm D)
```

### 3.0 Deliverable zero: measure the default prior before changing it

Before any training, sample ~50,000 tasks from arm A and report the
distributions of:

- **regression targets:** boundary mass, modality, support, number of
  distinct values;
- **classification targets:** the implied **base-rate distribution**;
- **predictability:** the ExtraTrees bootstrap p-value and pseudo-R²
  distribution, plus the realised rejection rate;
- **categorical cardinality.**

This is cheap (CPU only, free `interactive` partition), and it converts
§1.2's and §1.3's claims from arguments into measurements. **It is also the
honest go/no-go gate:** if the default prior already covers credit's regime,
the project's premise is weak and we find out in days. Overlay the real
`data/raw/` distributions from §1.6 on the same axes — that figure is the
paper's motivation panel.

### 3.1 LGD component — bounded target with sampled boundary mass

**Per §1.6, a family, not a shape.** Replace arm A's
`outlier_removing → standard_scaling` with a bounded-target transform that
samples:

- a **support** mapping to [0,1] (Kumaraswamy/Beta CDF warp, or rank-Gaussian
  then logistic — `rand_kumaraswamy_act` already exists to build on);
- **boundary mass per atom**, independently for 0 and 1, over roughly
  [0, 0.25], spanning the observed regimes: U-shaped (both atoms large),
  one-sided-inflated, skewed-unimodal-bounded, and interior-only;
- an **interior shape** (Beta with sampled α, β) so the sparse-middle case is
  represented rather than assumed.

The mechanism for producing an atom is **censoring**, which is the honest
generative story for LGD: full recovery and total loss are *censored*
outcomes of a latent recovery process, not draws from a continuous density.
This also matches O'Prior's own listed transform ("censoring-style
transformations"), which strengthens rather than weakens the positioning:
we are exercising a transform they named and never evaluated.

**Whether the model sees the [0,1] scale or a standardised version is itself
a factor**, since §1.1 shows standardisation is unconditional in arm A and
§1.2 shows it is shape-preserving. Test both.

### 3.2 Reporting LGD, given §1.5

Because point predictions from a bimodal predictive are misleading, the
primary metrics are **distributional**:

- **pinball loss / CRPS** over the quantile grid — the loss the model is
  actually trained on;
- **coverage** of central intervals at several nominal levels;
- **boundary-mass calibration:** predicted vs observed P(y=0) and P(y=1) —
  the metric that most directly tests the hypothesis, and which no cited
  paper reports;
- **R² and RMSE reported alongside, not instead**, for comparability with the
  LGD literature, with the decoding rule stated explicitly (median vs mean vs
  mode) because it changes the number materially.

Success criteria are set to the LGD literature, not classification-style
expectations: published R² ≈ **0.04–0.15** linear, **0.10–0.25** beta,
**0.20–0.43** tree ensembles. Baselines: **two-stage logistic + right-tailed
censored beta-mixture**, and **zero-one-inflated beta (ZOIB)** mixtures.

### 3.3 PD component — controlled base-rate distribution

Since §1.3 shows the base-rate distribution already exists and is
approximately uniform, the intervention is to **reshape it** toward the
credit regime (roughly 5–25% per §1.6), and to *measure* what the default
implies. Concretely: replace the uniformly-random cut point with a sampled
target base rate, and relax `cls_sanity_check` only as far as needed.

Also in scope for the PD arm, both flagged by the literature as credit-shaped
and both absent from arm A:

- **high-cardinality categoricals** — Purucker 2026 finds the
  best-GBDT-over-best-TFM margin grows with high-cardinality columns
  (ρ=+0.47);
- **threshold/rule mechanisms** — Klein & Hoffart 2026's charge that a
  statistical prior can only fuzzily approximate a hard `if amount >= 5000`
  rule, which is exactly how underwriting cutoffs and Basel/IFRS9 policy
  shape credit data.

### 3.4 Arm B — general realism (the control that makes the claim causal)

O'Prior's realism engine *without* the credit-specific components: marginal
morphing, feature augmentation, MCAR/MAR/MNAR missingness, generic target
reshaping, plus their shift/shortcut stress. Target their **G2b (SM+SR)**
configuration, which was their best static realism variant.

**This arm is not optional.** Without it, "our prior is better" cannot be
distinguished from "our prior is harder". **[OPEN]** How faithfully to
reimplement: their code is public (`github.com/o-prior/O-prior`) but is *not*
in our library, and I have not read it. Reimplementing from the paper text
risks an unfair strawman — which would be the single easiest way to make this
project's headline result wrong.

### 3.5 Arm D — unrealistic but complex

TabForestPFN found pretraining on deliberately **unrealistic**
forest-generated data gave *better* fine-tuned performance than realistic
priors, and O'Prior independently found mechanism *diversity* beats
observational realism. So "make the prior look like LGD" may well lose to
"make the prior harder". Arm D makes that a **measured hypothesis rather than
an assumption**: maximise decision-boundary complexity (deep tree ensembles,
`rand_prod_func` compositions, extreme activation warping) with no realism
and no domain targeting.

### 3.6 The filter experiment

Three levels, applied to arms A and C at minimum:

1. **as-shipped** — `pval < 0.05` required;
2. **off** — no statistical filter;
3. **banded** — keep only tasks whose ExtraTrees pseudo-R² falls inside
   credit's observed range, *targeting* the low-signal regime rather than
   merely admitting it.

Level 3 is the interesting one and it is a small change in `should_filter`.
Report the rejection rate for each (§2.2). Because this is a **removal**, it
cannot be accused of adding capacity — which is what makes it the cleanest
result available.

---

## 4. Phase 2 — pretraining (`src/train/`)

```
src/train/
├── loop.py         the training loop (ours; nanoTabPFN's train.py as template)
├── checkpoint.py   save/resume incl. optimizer, scheduler, AND prior RNG state
├── schedule.py     LR schedule, optional curriculum
└── budget.py       step/dataset accounting so "matched compute" is enforced, not asserted
```

One checkpoint per (arm × seed), written to `$VSC_SCRATCH` and copied back to
`$VSC_DATA` on completion — scratch is purged at 30 days and is not backed up
(`docs/VSC.md` §4).

**Checkpoint/resume is mandatory, not optional.** The VSC docs contain **no
Slurm requeue recipe** (the string `requeue` does not occur), and the 72 h
walltime ceiling is far below TabICLv2's reference 500,000-step stage 1. The
prior's **RNG state must be checkpointed too**, or a resumed run silently
resamples a different task stream and the arm is no longer the arm.
`docs/VSC.md` §3 has the self-resubmission pattern.

**Fan out one job per arm** (`--array`), since arms are fully independent and
parallelism is the stated preference. `docs/VSC.md` §8 has the array skeleton.

---

## Exp1 runs in two phases, and the order is not a convention

**Phase 2 scores what phase 1 wrote**, so it cannot start earlier. There is no way to shorten
this: a checkpoint has to exist before it can be benchmarked.

| | phase 1 — `pretrain_{lgd,pd}.slurm` | phase 2 — `benchmark.slurm` |
|---|---|---|
| array | `0-74%8` | `0-75%8` |
| each task | trains one arm, 12,500 steps | scores one model on everything |
| produces | one checkpoint on `/lustre1` | result rows on `$VSC_DATA` |
| cost | ~4.4 h (B200) / ~8.5 h (A100) | minutes |
| index 75 | — | the REFERENCE: TabICLv2, CatBoost, linear |

**Why the reference is inside the same array rather than run separately.** It is then scored by
the same code, on the same day, with the same `--max-context-rows`, the same seeds and the same
splits as our 75. Every comparison this project got wrong, it got wrong by scoring the two sides
through different paths — the hand-rolled inference pipeline on 18-08 being the clearest case.

**Why evaluation is not inside the training job**, which it briefly was on 25-08: an arm would
then be benchmarked by whatever the code looked like the hour it happened to finish, against a
reference scored on a different day. The 75 arms finish over ~2 days.

### What phase 2 records, per (model, dataset, seed)

- **Ranking** — ROC-AUC, PR-AUC, Gini, KS, PR-AUC lift, recall at the top 1/5/10/20 %
- **Probability** — Brier, Brier skill score, log loss, **ECE and MCE**, calibration slope,
  intercept and bias
- **Hard labels** — accuracy, balanced accuracy, **F1**, precision, recall, specificity, MCC,
  and the full confusion matrix, at 0.5 **and** at the base rate
- **LGD** — pinball, CRPS, coverage, boundary mass, R², RMSE, MAE
- **Cost** — `fit_seconds` and `predict_seconds`
- **Provenance** — the checkpoint file and step, the arm's `credit_fraction` and run name, the
  inference path, `context_cap` and `n_context_full` (recorded even when the cap does not bind)

ECE and MCE are reported together on purpose. ECE weights bins by occupancy, and in a PD model
the bins that matter — the high-score tail where the defaults are — are the sparsest, so ECE
can look respectable while the worst bin is badly wrong. F1 is reported at the base rate as
well as at 0.5 for the same reason: at a 2 % base rate almost nothing crosses 0.5, and a model
that ranks perfectly can score F1 near zero.

---

## 5. Phase 3 — evaluation (`src/eval/`)

### 5.1 What gets compared

Every pretrained checkpoint against every other, plus:

- **released TabICLv2 v2 checkpoints** already in `checkpoints/` — the
  frontier-scale reference, which our nano-scale models will lose to in
  absolute terms; that is expected and must be stated, since our claim is
  about *prior contrast at matched compute*, not absolute SOTA;
- **classical baselines** — ZOIB and two-stage beta mixtures for LGD;
  logistic regression and tuned GBDT for PD.

### 5.2 The double dissociation — the centrepiece

| component enabled | predicted effect on **LGD** | predicted effect on **PD** |
|---|---|---|
| bounded target + boundary mass (§3.1) | **improves** | ~neutral |
| base-rate distribution (§3.3) | ~neutral | **improves** |
| generic realism (arm B) | improves | improves |
| complexity (arm D) | improves | improves |

**A crossed dissociation in the top two rows is the strongest evidence the
design can produce**, because it rules out "narrower" and "harder" using only
our own experiment — no external control domain needed. Arms B and D are the
competing explanations it must beat. Note the asymmetry in what we learn: a
dissociation supports the claim; *no* dissociation with arm C still winning
on average supports the weaker "harder prior" reading and should be reported
as such.

### 5.3 Calibration is a first-class axis

The synthesis's weakness 3 is that calibration is *"claimed everywhere and
measured almost nowhere"* — TabICLv2, TabDPT, Kolberg, Ye and both
Grinsztajn reports omit ECE/Brier entirely. For PD estimation it is the
metric that matters most.

Report for every arm: **ECE, Brier, log-loss, reliability diagrams** (PD);
**pinball/CRPS, interval coverage, boundary-mass calibration** (LGD). This is
cheap and it is a contribution in itself.

### 5.4 Temporal splits

Purucker 2026 shows TFM in-context learning **loses to tuned GBDTs the moment
splits become temporal or grouped**, and Rubachev documents instability on
Home Credit specifically. An i.i.d.-only result on credit data is not
credible.

**[OPEN]** Feasibility is dataset-dependent: `0006.lgd_freddie` has
`months_since_origination`, but Home Credit has no obvious date column. Needs
a per-dataset audit before the split protocol is fixed.

### 5.5 Forgetting check — a headline, not an appendix

Copy Kolberg's instrument (Spearman ρ against the base model on
original-range tasks; they report ρ=0.9935). For us: after training on a
credit-targeted prior, evaluate on **general** benchmarks to measure whether
in-domain gains were bought at out-of-domain cost.

Hoo 2026's backbone-swap result predicts exactly this trade — real-data
pretraining *narrowed* inductive bias. **If domain-targeting costs general
performance, that is the finding**, and a more interesting one than a win.

### 5.6 Meta-overfitting protection

Mitra selected its prior mixture greedily against a single benchmark family
and the authors concede the meta-overfitting risk. With 21 datasets we are
more exposed, not less.

**Freeze a split before touching the prior:** development datasets for prior
design and all iteration; held-out datasets **never inspected** until the
final run. Record the split, the date, and the commit in
`docs/CHANGELOG.md`.

**[OPEN]** The exact split needs Andreas's ratification — this is a
scientific commitment, not a coding choice. Proposed starting point: hold
out `0006.lgd_freddie` (the only genuinely U-shaped LGD set) **or** develop
on it and hold out the others — the two options trade different risks, and
the choice determines what the paper can claim.

---

## 6. Risks, and what we do about them

| risk | why it is real | mitigation |
|---|---|---|
| our training loop is subtly wrong | it is ours, not upstream; a bug invalidates all arms equally and invisibly | reproduce arm A and check it lands in the expected nano-scale range; unit-test the loss against a hand-computed pinball value; verify a fixed seed reproduces exactly |
| arm B is a strawman | O'Prior's code is public but not in our library and unread; a weak reimplementation would make our headline result wrong | read their code before implementing; document every deviation; **[OPEN]** §3.4 |
| prior×architecture interaction | TabICLv2's own ablations show divergence on the old prior | log all loss curves; report divergence as an outcome, never tune it away silently |
| nano→frontier gap | O'Prior's own untested assumption | state it as a limitation; scope any frontier confirmation separately |
| credit is not distinctive | if arm A already covers credit's regime, the premise is weak | §3.0 measures this in days, before any credits are spent |
| meta-overfitting | 21 datasets, many iterations | §5.6 frozen split, ratified and dated |
| i.i.d.-only result | Purucker 2026 | §5.4 temporal splits |

---

## 7. Open decisions

1. **[OPEN]** §3.4 — how faithfully to reimplement O'Prior's realism engine,
   and whether to read their repository first. *This is the highest-leverage
   open item: arm B is the control the whole claim rests on.*
2. **[OPEN]** §5.6 — the frozen development/held-out dataset split.
3. **[OPEN]** §5.4 — which datasets support temporal splits.
4. **[OPEN]** Which of the 14 PD and 7 LGD datasets are in scope at all;
   §1.6 measured only 4 PD and 3 LGD.
5. **[OPEN]** Whether `arXiv 2605.21742` (inference-time PFN imbalance
   correction) says what the project brief describes — **it is not in the
   library and remains unverified**.
6. **[OPEN]** Whether to add a ZOIB/mixture head at all. §1.5 argues the
   999-quantile head can already represent atoms, so the recommendation is to
   test **decoding and support constraints first** and hold the head redesign
   in reserve. Andreas's original 2×2 put the head as a co-equal factor;
   this is a deliberate deviation and is his to overrule.
