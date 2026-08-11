# Config reference

`config/LGD.yaml` and `config/PD.yaml` hold every setting for their experiment.
This file holds the *reasoning* — why each setting exists and what evidence backs
it — so the configs stay scannable.

The configs used to carry all of this inline and ran to 542 lines, which made them
unreadable. YAML was never the problem; 250 lines of prose in a settings file was.

## How the sweep works

Any setting may be a single value or a list. A list means one run per value, and
all lists are **crossed**: three settings with two values each is 8 runs, times
the number of seeds.

```bash
python scripts/pretrain.py --config config/LGD.yaml --list
```

`# S: [...]` marks a setting left at one value and shows how to open it up. Names
ending in `_range` are literal `[low, high]` intervals sampled from, never swept —
to sweep one, nest it: `boundary_mass_range: [[0.0, 0.1], [0.1, 0.3]]`.

**Open one group at a time.** Crossing is multiplicative, and with every lever
moving you cannot attribute an effect to any of them.

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

## `init.strategy` — how to start

`scratch` (random init), `full` (pretrained, train everything at a low LR),
`icl_only` (freeze the column and row blocks), `head_only` (freeze all blocks).

All four come from what TabICL itself does — v1's final stage froze col+row at
`lr 2e-6`, v2's froze nothing at `lr 2e-5`. **No LoRA**: there is none anywhere in
TabICL, and Tanna 2025 found it unstable on TabPFN. Full write-up in
[`finetuning.md`](finetuning.md).

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

## `eval` — not yet wired up

`dev_datasets` and `holdout_datasets` are deliberately empty. The split must be
**frozen before any prior tuning**, or we meta-overfit the way Mitra warns about.
That is a scientific commitment and needs sign-off, not a default.

---

## A note on the published LGD R² figures

The configs and docs quote "published LGD R² ≈ 0.04–0.15 linear, 0.10–0.25 beta,
0.20–0.43 tree ensembles". **That range came from this project's own brief, not from
a source anyone has verified here.** It is used as a sanity band for our numbers,
and it should be replaced with a citation — or dropped — before it appears in a
paper. Our own measurements so far: CatBoost 0.376 on Freddie and 0.501 on heloc,
both inside the quoted tree band; but `lgd_lendingclub` scores 0.71–0.76, well
outside it, which most likely means a feature derived from the recovery amount is
surviving the recipe.
