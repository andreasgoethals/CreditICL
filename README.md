# CreditICL — encoding domain knowledge into a tabular foundation model's pretraining prior

**Can domain knowledge be deliberately encoded into a tabular foundation
model's synthetic pretraining prior, and does it transfer to downstream
performance on the matching domain?**

The vehicle is **TabICLv2** — the only competitive tabular foundation model
(TFM) whose prior generator *and* pretraining code are public. The testbed
is **credit risk**, in two halves that stress different parts of the prior:

| | task | what makes it hard for a TFM |
|---|---|---|
| **LGD** | Loss Given Default | regression on a **bounded [0,1]** target with mass at the boundaries; sparse interior |
| **PD** | Probability of Default | **imbalanced** binary classification, high-cardinality categoricals, threshold-like business rules, low signal-to-noise |

PhD project, KU Leuven. Literature grounding lives in the pinned
[`tfm-library/`](tfm-library/) submodule (**read-only** — see
[AGENTS.md](AGENTS.md)).

---

## Positioning: what O'Prior already closed, and what it left open

**Read this before writing anything about novelty.** The closest prior work
is **O'Prior** (Bouadi et al. 2026, arXiv
[2605.18971](https://arxiv.org/abs/2605.18971), Lexsi Labs;
[`tfm-library/papers/2026/05_Bouadi_et_al._Shaping_the_Prior_*.pdf`](tfm-library/papers/2026/)).
It already does the **methodological core** of this project: it holds
architecture, optimizer, compute budget and evaluation pipeline fixed and
varies *only* the synthetic task distribution, across nine prior variants
against the TabPFN-v1, TabICL-v1 and TabICL-v2 generators. Its headline
finding is that **structural mechanism diversity is the strongest driver of
transfer**, with observational realism and shift-stress adding
complementary, non-interchangeable gains.

So the general claim *"prior design matters, and here is how to measure it
cleanly"* is **taken**. This project must not be written as if nobody has
studied prior design.

What remains open is the word **domain**:

1. **Domain-targeting is unmeasured.** O'Prior optimises *average* behaviour
   across 52 general classification tasks and reports only cross-dataset
   averages. Whether encoding a *specific* domain's structure transfers to
   that domain is untouched. Their own conclusion invites it: *"Future work
   should … explore domain-specialized realism modules for high-impact
   tabular applications."* And their layer-probing analysis singles out
   **Credit-g** as one of two non-diagnostic datasets — *"no consistent
   depth-wise improvement … its predictive signal is too weak for this
   probing setup to differentiate the priors."* **Credit is where their
   gains vanish.**
2. **Regression is outside their study.** All O'Prior experiments are
   classification (ROC-AUC / accuracy / macro-F1). The regression branch of
   their target-reshaping module is never exercised, so bounded targets with
   boundary mass are unevaluated. The LGD case is genuinely open.
3. **Prior-side class imbalance is unexplored** — and, in O'Prior's case,
   actively excluded: its quality-control step rejects tasks with
   *"collapsed or severely imbalanced support classes"*. Combined with
   Tanna et al. 2026 (*Data Presentation Over Architecture*), which found
   **context construction beats architecture choice** under credit
   imbalance, the prior side is the untried lever.

O'Prior also gives us the **ideal control arm**: domain-targeted prior vs
*generically*-realistic prior vs default. Without that arm, "our prior is
better" is indistinguishable from "our prior is harder".

Two claims in this project's original framing were **weakened after reading
the actual source**, and the honest versions are load-bearing — see
[`docs/experimental_design.md`](docs/experimental_design.md) §"Verified
premises". In short: TabICL's affine target standardisation is
*shape-preserving*, so bounded/bimodal targets are a question of
**frequency and alignment, not absence**; and TabICL's prior is **not**
class-balanced by default, so imbalance there is **uncontrolled, not
missing**.

### Papers to cite pre-emptively

**KnowsTFM** (2606.30258) and the steel *Multitask-Informed Prior*
(2603.22738) sound like prior modification but are **fine-tuning**. Cite
them to pre-empt the reviewer question.

---

## Experimental design in one page

Full version: [`docs/experimental_design.md`](docs/experimental_design.md).

**Three phases.**

1. **Prior** — a configurable synthetic-task generator (`src/prior/`) that
   reproduces the TabICLv2 prior *exactly* as one named variant, then adds
   independently-gated components on top.
2. **Pretrain** — the same TabICLv2 transformer architecture trained on each
   prior variant under a matched budget (`src/train/`), one checkpoint per
   variant.
3. **Evaluate** — compare the resulting checkpoints against each other, and
   against classical baselines, on our LGD and PD datasets (`src/eval/`).

**Prior arms.**

| arm | prior | role |
|---|---|---|
| **A** | TabICLv2 default, unmodified | the control everything is measured against |
| **B** | general realism, O'Prior-style | *is domain-targeting better than generic hardening?* |
| **C** | credit-targeted | the hypothesis |
| **D** | unrealistic-but-complex | TabForestPFN's counter-hypothesis: complexity may beat realism |

**The centrepiece is a double dissociation, not an average win.** Arm C's
components are gated independently, and the two domain-specific ones are
predicted to be *selective*:

- bounded-target-with-boundary-mass → should help **LGD** and **not PD**
- controlled base-rate distribution → should help **PD** and **not LGD**

If that cross-over holds, "domain-targeted" is established against the
reviewer's first objection ("your prior is just narrower/harder") using
only our own experiment. An average improvement alone cannot do that.

**The cheapest sharp result** is the **predictability filter**. TabICLv2
discards datasets a shallow ExtraTrees cannot beat a constant baseline on
(*"roughly 35% classification and 25% regression datasets are filtered"* in
stage 1 — TabICLv2 §Data filtering). Credit is intrinsically low-signal
(AUC 0.70–0.85; published LGD R² ≈ 0.04–0.43). Testing filter
{as-shipped / off / *banded to credit's signal range*} is a one-line change,
is a **removal** rather than an addition (so it cannot be accused of adding
capacity), and directly contradicts a published convergence claim.

**Success criteria are set to the LGD literature, not to
classification-style expectations:** published R² ≈ 0.04–0.15 linear,
0.10–0.25 beta, 0.20–0.43 tree ensembles. Baselines to beat are two-stage
logistic + right-tailed censored beta-mixture, and zero-one-inflated beta
(ZOIB) mixtures.

---

## Repository layout

Follows our research group's project template — **fill it in, do not
restructure** (see [AGENTS.md](AGENTS.md) §2). The original template README
is preserved at [`docs/template_readme.md`](docs/template_readme.md).

```
CreditICL/
├── assets/            non-code files (figures for papers, README images)
├── config/            YAML experiment configuration
│   ├── priors/          one file per prior arm (A/B/C/D + ablations)
│   ├── train/           pretraining budgets and curricula
│   ├── eval/            evaluation protocols, splits, metrics
│   ├── data/            dataset-level settings
│   └── methods/         baseline model hyperparameters
├── checkpoints/       pretrained weights, one per prior arm  [gitignored]
├── data/                                                     [gitignored]
│   ├── raw/lgd/         7 LGD datasets
│   ├── raw/pd/          14 PD datasets
│   └── processed/
├── docs/
│   ├── experimental_design.md   the full design
│   ├── vsc.md                   KU Leuven VSC deployment facts
│   └── changelog.md             dated log of what changed and why
├── lib/               third-party code used unmodified
├── notebooks/         exploration only, never the pipeline
├── res/               results                                [gitignored]
├── scripts/           thin runnables
│   └── slurm/           SLURM job scripts sent to VSC
├── src/               all importable project code
│   ├── prior/           PHASE 1 — synthetic prior generation
│   ├── train/           PHASE 2 — pretraining loop, checkpointing
│   ├── eval/            PHASE 3 — evaluation, metrics, calibration
│   ├── data/            dataset loaders / registry
│   ├── methods/         classical LGD & PD baselines
│   ├── models/          TabICLv2 architecture (from NanoTabICL)
│   └── utils/           config loading, tracker, plotting
└── tfm-library/       PINNED SUBMODULE — READ-ONLY (one exception)
```

`src/` holds reusable code; `scripts/` only wires config + data + method
together. Following the template's `scripts/example_script.py`, runnables
use a **CSV tracker** so a sweep can be resumed and run in parallel across
machines — which maps directly onto SLURM array jobs.

---

## Setup (Windows, PowerShell 5.1)

> **PowerShell 5.1 has no `&&`.** Every command below is on its own line
> deliberately. Do not join them.

### 1. Clone with the submodule

If you already have the repo, just initialise the submodule:

```powershell
git submodule update --init --recursive
```

### 2. Create the virtual environment

Requires **Python 3.11 or 3.12**. See which versions you have:

```powershell
py --list
```

> **On this machine (checked 2026-08-05):** 3.14.0, 3.13.10, 3.12, 3.10.11 —
> **3.11 is not installed**, and bare `python` resolves to **3.14**, which is
> outside the supported range. Use `py -3.12` explicitly, as below.
>
> Why 3.12: VSC documents `Python/3.11.3`, `3.12.3` and `3.13.1` modules, so
> 3.12 gives local↔cluster parity *without* installing a new interpreter. If
> you would rather match VSC's best-established 2023a toolchain exactly,
> install Python 3.11 and use `py -3.11` throughout — both are supported.

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, allow signed local scripts
for your user once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install

```powershell
python -m pip install --upgrade pip
```

Install PyTorch first, matched to your machine. CPU-only is fine for
everything local — all GPU work happens on VSC:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the project, editable, with dev tools:

```powershell
python -m pip install -e ".[dev]"
```

### 4. Verify

```powershell
python -c "import torch, numpy, sklearn, pandas; print(torch.__version__)"
```

```powershell
pytest -q
```

### 5. Auto-activation when you open the project

`.vscode/settings.json` is committed and already points VS Code at
`.venv`. Opening the folder in VS Code will:

- select `.\.venv\Scripts\python.exe` as the interpreter, and
- activate it automatically in every new integrated terminal
  (`python.terminal.activateEnvironment: true`).

VS Code may need one reload to pick up a newly created venv:
**Ctrl+Shift+P → "Developer: Reload Window"**. If the interpreter still
looks wrong, **Ctrl+Shift+P → "Python: Select Interpreter"** → *Enter
interpreter path* → `.\.venv\Scripts\python.exe`.

For a plain (non-VS Code) PowerShell session, activation is one line:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 6. Running on VSC

The local venv is for development and analysis only. Pretraining runs on
KU Leuven VSC via SLURM scripts in `scripts/slurm/`. The cluster needs its
own per-architecture environment — **do not copy `.venv` to VSC**. See
[`docs/vsc.md`](docs/vsc.md) for the environment recipe, GPU/partition
choice, credit costs, the Lustre-vs-GPFS rule, and the checkpoint/resume
gap.

---

## Provenance and honesty conventions

- **Library pin.** Literature claims are written against
  `tfm-library` commit **`21d555a`** (2026-08-05). Record the pin whenever a
  result depends on the literature.
- **Cite code dumps by symbol name, never line number** — the dumps are
  refreshed and line numbers drift.
- **Distinguish evaluated from supported.** A mechanism existing in a
  paper's code is not the same as that paper having measured it. Several of
  this project's premises turned on exactly this distinction.
- **Never write project-specific content into `tfm-library/`.** The one
  permitted file is `tfm-library/PROJECT_SPECIFIC.md`.

## License

MIT — see [LICENSE](LICENSE). Note this covers *our code only*; the
datasets under `data/` carry their own, more restrictive terms.
