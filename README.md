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
[`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md) §"Verified
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

## The four pipelines

Each one lives in `src/`, is driven by a thin runnable in `scripts/`, logs to
`logs/`, and writes its official output to `results/<task>/<pipeline>/`.

| # | pipeline | code | run it with | what it produces |
|---|---|---|---|---|
| 1 | **data** | `src/data/` | `scripts/preprocess.py` | processed datasets in the cache |
| 2 | **prior** | `src/prior/` | `scripts/measure_prior.py` | the synthetic task stream, and reports on it |
| 3 | **training** | `src/train/` | `scripts/pretrain.py` | one checkpoint per prior arm |
| 4 | **eval** | `src/eval/` | `scripts/evaluate.py` | scores on the 21 real credit datasets |

Pipeline 2 is the biggest and the actual research contribution. Pipeline 1 exists
only to feed pipeline 4.

**`logs/` is for information, never for results.** If a file matters, it goes in
`results/`. That split is deliberate: it means anything in `results/` is something
you meant to keep.

### 1. Data

Per-dataset recipes (which columns are IDs, which leak the target, which need log
or clip transforms) are **copied from the sibling TabPFNCredit project**, where
they were developed and validated. Fix bugs in both or they will silently diverge.

The cache is **one parquet file per dataset** plus `meta.json`. Parquet rather than
`.npy` because it keeps column names and which columns are categorical — CatBoost and
TabPFN both need the categorical indices, and an array format cannot carry them. It is
also far smaller (17 MB vs 149 MB on Home Credit) in fewer files, which matters because
project storage has a low inode budget.

`meta.json` is written **last** and is the completeness marker, so an interrupted run
leaves a directory that correctly reads as absent rather than as done. Writes are
atomic, because on the cluster several array tasks can preprocess the same dataset at
once.

The eval pipeline preprocesses whatever is missing on demand, so one command works
from a fresh clone.

### 2. Prior

See [`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md). The mixture lever
`credit_fraction` sets the share of datasets from our credit-targeted path; the
rest come from the unmodified TabICL prior. Defaults sweep `0.0 / 0.1 / 0.2 / 0.3`,
i.e. 70–90% original plus a control arm.

Datasets are **generated once into one folder per variant** and training samples
from them (`src/prior/pool.py`): `lgd__original/` holds the unmodified TabICL prior
and is shared by every arm, `lgd__credit_v1/` holds ours. So a difference between
arms cannot come from the luck of the draw, and generation runs on cheap CPU nodes
instead of making a GPU wait.

Two notebooks, both with all logic in `src/visualize/`:

* **`prior_visualisation.ipynb`** — set `TASK`, run it, and it **discovers whichever
  pools are on the machine** and compares them on shared axes. One notebook for any
  number of variants: adding `credit_v2` needs no edit here, because the useful
  question is never "what does `credit_v1` look like" but "how does it differ from
  `original` and `credit_v2`". With no pools it falls back to generating live.
* **`data_exploration.ipynb`** — the real datasets the prior is aimed at.

To look at cluster-generated pools locally, copy a **sample** rather than the lot — a
full pool is 4.0 GB (LGD) / 5.4 GB (PD) *per variant*, and one shard (~200–270 MB) is
twenty times what the plots use:

```bash
bash scripts/transfer/fetch_prior_sample.sh
```

```bash
python scripts/generate_prior.py --config config/LGD.yaml --status
```

A partial copy is labelled **SAMPLE**, so it can never be mistaken for the pool the
model actually trained on.

### 3. Training

The same TabICLv2 architecture on each prior variant under a matched budget. One
checkpoint per (arm × seed). Adaptation strategy — train from scratch, or
fine-tune with parts frozen — is `init.strategy`; see
[`docs/FINETUNING.md`](docs/FINETUNING.md).

### 4. Eval

Five baselines on all 21 datasets: **ridge/logistic regression** (the floor),
**CatBoost** (the GBDT every credit paper reports), **TabPFN-3**, and
**TabICLv2**. None of them is modified — in particular TabPFN's prior is never
touched; it is a yardstick.

LGD reports pinball/CRPS, interval coverage, **boundary-mass calibration**, and R²
and RMSE alongside. (A rough literature range of R² ≈ 0.04–0.43 across linear, beta
and tree models circulates in this project's notes, but it comes from the project
brief rather than a source I have verified — do not cite it without checking.) PD reports ROC-AUC, PR-AUC, Brier, ECE, log-loss and
recall-at-top-k — **never accuracy alone**, since at a 7% base rate predicting
"never defaults" already scores 0.93.

Row and feature caps are applied to the in-context models (they are built for
about 10k rows, and `algorithmwatch` has 2,986 features) and **always recorded in
the results row**. A silent subsample makes a model look worse for a reason
nothing in the output explains.

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
│   ├── finetuning.md            what fine-tuning TabICL allows, and why
│   ├── vsc.md                   KU Leuven VSC deployment facts
│   └── changelog.md             dated log of what changed and why
├── logs/              timestamped run logs — INFORMATION ONLY, no results
├── notebooks/
│   ├── prior_visualisation.ipynb   compares ALL prior variants found on disk
│   └── data_exploration.ipynb      the real datasets: boundary mass, base rates, leakage
├── results/           OFFICIAL outputs
│   ├── lgd/{data,prior,training,eval}/
│   └── pd/{data,prior,training,eval}/
├── scripts/           every runnable lives here and calls into src/
│   ├── preprocess.py        pipeline 1
│   ├── measure_prior.py     pipeline 2
│   ├── pretrain.py          pipeline 3
│   ├── evaluate.py          pipeline 4
│   ├── smoke_test.py        fast end-to-end check
│   ├── vendor_model.py      extracts the architecture from the pinned library
│   └── slurm/               SLURM jobs sent to VSC
├── src/               all importable project code
│   ├── data/            PIPELINE 1 — recipes, registry, processed cache
│   ├── prior/           PIPELINE 2 — synthetic task generation
│   ├── train/           PIPELINE 3 — pretraining, checkpointing, adaptation
│   ├── eval/            PIPELINE 4 — baselines, metrics, runner
│   ├── models/          TabICLv2 architecture (generated from NanoTabICL)
│   ├── visualize/       plotting for the notebook
│   └── utils/           config, paths, logging, target statistics
└── tfm-library/       PINNED SUBMODULE — READ-ONLY (one exception)
```

`src/` holds reusable code; `scripts/` only wires config + data + method
together. Following the template's `docs/templates/example_script.py`, runnables
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

The environment is named `CreditICL`, not `.venv`, so the VS Code prompt reads
`(CreditICL)` and you can tell it apart from the sibling projects' environments.

```powershell
py -3.12 -m venv CreditICL
```

```powershell
.\CreditICL\Scripts\Activate.ps1
```

⚠️ **If you are replacing an existing `.venv`, close every terminal and editor tab
using it first.** Windows will not delete a DLL that a running process has open, so
`Remove-Item -Recurse -Force .venv` silently leaves a half-deleted skeleton behind —
missing `pyvenv.cfg`, a stub `Scripts\python.exe`, and a `torch/` with two files in
it. That skeleton then fails with `No pyvenv.cfg file` or
`ModuleNotFoundError: No module named 'torch.nn'`, which looks like a broken install
rather than an incomplete delete. If it happens, just delete the leftovers again once
nothing is holding them:

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
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
`CreditICL`. Opening the folder in VS Code will:

- select `.\CreditICL\Scripts\python.exe` as the interpreter,
- activate it automatically in every new integrated terminal
  (`python.terminal.activateEnvironment: true`), and
- run the test suite from the Testing panel (`pytest`, rooted at `tests/`).

VS Code may need one reload to pick up a newly created venv:
**Ctrl+Shift+P → "Developer: Reload Window"**. If the interpreter still
looks wrong, **Ctrl+Shift+P → "Python: Select Interpreter"** → *Enter
interpreter path* → `.\CreditICL\Scripts\python.exe`.

For a plain (non-VS Code) PowerShell session, activation is one line:

```powershell
.\CreditICL\Scripts\Activate.ps1
```

### 6. Where files go, locally and on the cluster

The code detects which it is on by checking for `$VSC_DATA`, so there is nothing
to configure. Two tiers on the cluster:

| | project storage `/lustre1/project/stg_00211/CreditICL/` | `$VSC_DATA/CreditICL/` |
|---|---|---|
| size | ≥1 TB, **not** backed up, low inode budget | 75 GiB, **backed up** |
| holds | processed datasets, generated prior pools, checkpoints, large arrays | the repo, logs, metrics, result CSVs |

Big and regenerable goes to project storage; small and durable goes to `$VSC_DATA`.
Locally, both collapse into the repo. Override the staging root with
`$CREDITICL_STAGING_ROOT`. See [`src/utils/paths.py`](src/utils/paths.py).

### 7. Running the pipelines

```powershell
python scripts/preprocess.py --task both
```

```powershell
python scripts/evaluate.py --task lgd --models linear,catboost
```

`evaluate.py` preprocesses anything missing itself, so the first command is
optional locally. On the cluster run it first, so 48 array tasks do not each
preprocess Home Credit's 307k rows.

**Prior pools.** Training reads pre-generated datasets rather than building them
live. Build them once per variant:

```powershell
python scripts/generate_prior.py --config config/LGD.yaml --variant original --all
```

```powershell
python scripts/generate_prior.py --config config/LGD.yaml --status
```

The first writes `prior_cache/lgd__original/`; `--variant credit_v1` writes ours.
`--status` checks both are complete and hold the **same** count — that equality is
what makes the comparison fair, so it is checked rather than assumed. On the cluster
this is a 20-task CPU array, not `--all`. Training then uses
`--prior-source pool`; `generate` still works and needs no pools.

**TabPFN-3 reads a local checkpoint file, so no token is needed.** Put the weights in
`checkpoints/` (or point `CREDITICL_TABPFN_DIR` at them). Letting the `tabpfn` package
fetch its own weights triggers a licence flow that wants an API key, which on a
compute node with no terminal fails as `OSError: WinError 10038 ... not a socket`.
Passing `model_path` avoids the token, the download, and the need for internet.
Without the file the baseline skips with an explanatory line rather than failing.

### 8. Running on VSC

**The whole pipeline is one command.** It chains all five stages with
`--dependency=afterok`, so you can submit and log out:

```bash
bash scripts/submit_pipeline.sh both
```

preprocess → prior pools (2 arrays × 20 CPU tasks) → verify gate → GPU training
array → evaluation. Each stage gets the hardware it needs; see
[`docs/VSC.md`](docs/VSC.md) for why the pools are 40 parallel CPU tasks and why the
gate exists.

The local venv is for development and analysis only. Pretraining runs on
KU Leuven VSC via SLURM scripts in `scripts/slurm/`. The cluster needs its
own per-architecture environment — **do not copy the local `CreditICL/` venv to VSC**. See
[`docs/VSC.md`](docs/VSC.md) for the environment recipe, GPU/partition
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
