# CreditICL — encoding domain knowledge into a tabular foundation model's pretraining prior

**Can domain knowledge be deliberately encoded into a tabular foundation model's synthetic
pretraining prior, and does it transfer to downstream performance on the matching domain?**

The vehicle is **TabICLv2** — the only competitive tabular foundation model (TFM) whose prior
generator *and* pretraining code are public. The testbed is **credit risk**, in two halves that
stress different parts of the prior:

| | task | what makes it hard for a TFM |
|---|---|---|
| **PD** | Probability of Default | **imbalanced** binary classification, high-cardinality categoricals, threshold-like business rules, low signal-to-noise |
| **LGD** | Loss Given Default | regression on a **bounded [0,1]** target with mass at the boundaries; sparse interior |

PhD project, KU Leuven (Andreas Goethals). Literature grounding lives in the pinned
[`tfm-library/`](tfm-library/) submodule (**read-only** — see [AGENTS.md](AGENTS.md)).

## Contents

1. [What is open, and why (vs O'Prior)](#what-is-open-and-why)
2. [The three experiments](#the-three-experiments)
3. [The four code pipelines](#the-four-code-pipelines)
4. [Repository layout](#repository-layout)
5. [Setup (Windows, PowerShell 5.1)](#setup-windows-powershell-51)
6. [Running it](#running-it)
7. [Conventions](#conventions)
8. [License](#license) · [Based on the repository template](#based-on-the-repository-template)

Deeper reading, each owning one topic: **[EXPERIMENTAL_DESIGN](docs/EXPERIMENTAL_DESIGN.md)** (the
science), **[PRIORS](docs/PRIORS.md)** (how the prior is built), **[CONFIG_REFERENCE](docs/CONFIG_REFERENCE.md)**
(why each knob is what it is), **[VSC](docs/VSC.md)** (running on the cluster).

---

## What is open, and why

The closest prior work is **O'Prior** (Bouadi et al. 2026, arXiv
[2605.18971](https://arxiv.org/abs/2605.18971);
[`tfm-library/papers/2026/05_Bouadi_et_al._Shaping_the_Prior_*.pdf`](tfm-library/papers/2026/)). It
already does the **methodological core** — hold architecture, optimizer, compute and evaluation
fixed, vary *only* the synthetic task distribution — and finds that **structural mechanism diversity
drives transfer**. So "prior design matters, and here is how to measure it cleanly" is *taken*.

What remains open is the word **domain**:

1. **Domain-targeting is unmeasured.** O'Prior optimises *average* behaviour across 52 general
   tasks; whether encoding a *specific* domain's structure transfers to that domain is untouched —
   and their own probing singles out **Credit-g** as a dataset where the signal was too weak to tell
   the priors apart. Credit is where their gains vanish.
2. **Regression is outside their study** — all their experiments are classification, so bounded
   targets with boundary mass (the LGD case) are genuinely open.
3. **Prior-side class imbalance is unexplored**, and in O'Prior actively filtered out.

Two premises in this project's *original* framing were weakened after reading the source, and the
honest versions are load-bearing (see [EXPERIMENTAL_DESIGN §1](docs/EXPERIMENTAL_DESIGN.md)): TabICL's
target standardisation is *shape-preserving*, so bounded/bimodal targets are a question of
**frequency and alignment, not absence**; and TabICL's prior is **not** class-balanced by default,
so imbalance there is **uncontrolled, not missing**.

> **Cite pre-emptively:** KnowsTFM (2606.30258) and the steel *Multitask-Informed Prior* (2603.22738)
> sound like prior modification but are **fine-tuning** — name them to pre-empt the reviewer question.

## The three experiments

Each experiment is one flat config, `config/Exp{1,2,3}_{PD,LGD}.yaml`, carrying an inline `prior:`
block and a `sweep:` block that lists exactly the knobs it crosses. Full rationale in
[EXPERIMENTAL_DESIGN](docs/EXPERIMENTAL_DESIGN.md); every value in
[CONFIG_REFERENCE](docs/CONFIG_REFERENCE.md).

| # | question | how | arms/track |
|---|---|---|---|
| **0** | **Does the revised pipeline run correctly?** | one Slurm job: full tests, CUDA checks, six stop/resume runs and benchmarks | 3 |
| **1** | **Does the prior help at matched compute?** | three mixtures × three seeds, plus five component removals × three seeds; 12,500 steps | 24 |
| **2** | **Does continued pretraining help released models?** | TabICLv2 and TabPFN-3, three mixes × three seeds using upstream or adapted literature recipes, 10,000 steps | 18 |
| **3** | **Does the effect persist with longer training?** | scratch TabICL, Exp1's chosen mix vs control, two seeds, 200,000 steps | 4 |

`credit_fraction` is the master switch: the share of each batch drawn from our credit-targeted path,
the rest from the unmodified TabICL prior. `0.0` is the control every arm is measured against.

Each experiment is **two phases**: phase 1 trains (permanent checkpoints along each arm), phase 2 benchmarks every
checkpoint plus a shared reference column (released TabICLv2, TabPFN-3, CatBoost, linear) through the
*same* code — the ordering is a fact about the data, not a convention.

## The four code pipelines

Each lives in `src/`, is driven by a thin runnable in `scripts/`, and writes under `output CreditICL/`
(one folder per experiment; small files on the backed-up tier, checkpoints on project storage under
the same layout) — the code picks the tier automatically (`output CreditICL/README.md`).

| # | pipeline | code | run it with | produces |
|---|---|---|---|---|
| 1 | data | `src/data/` | `scripts/preprocess.py` | the processed-dataset cache |
| 2 | prior | `src/prior/` | `scripts/generate_prior.py` / `measure_prior.py` | the synthetic task stream + reports |
| 3 | training | `src/train/` | `scripts/pretrain.py` | permanent and resumable checkpoints per arm |
| 4 | eval | `src/eval/` | `scripts/evaluate.py` | scores on the 21 real datasets + out-of-domain suites |

Pipeline 2 is the research contribution; pipeline 1 exists only to feed pipeline 4. A few facts worth
knowing:

- **Data cache:** one parquet file per dataset + a `meta.json` written *last* as the completeness
  marker. Parquet (not `.npy`) so categorical indices survive — CatBoost and TabPFN need them.
  Per-dataset recipes are shared with the sibling TabPFNCredit project; fix bugs in both.
- **Prior pools:** training can read datasets pre-generated once per variant
  (`prior_cache/<task>__original`, `…__credit_v1`) so a GPU never waits on CPU generation and the two
  sides differ only by design, not by draw. `--prior-source generate` builds live instead.
- **Notebooks** — eleven, in three numbered chapter folders read as one story
  ([`notebooks/README.md`](notebooks/README.md)): **`0. General`** (the real data, the PD prior, the
  LGD prior), **`1. Experiment 1`** (which prior — `1.1`/`1.2` training, `1.3`/`1.4` benchmark) and
  **`2. Experiment 2`** (fine-tuning — `2.1`/`2.2` training and out-of-domain retention, `2.3`/`2.4`
  benchmark). Logic lives in `src/visualize/`; every figure is an A4 PDF. Training notebooks average
  **development datasets only**, over **finished** arms — holdout datasets some old arms happened to
  monitor are shown but never averaged — and results notebooks select on development, report on the
  holdout, and show a placeholder until phase 2 has run. Every section is **grounded in the pinned
  `tfm-library`**: `src/visualize/literature.py` holds the citable values (paper-evaluated /
  code-supported / *external*), drawn as reference lines where the axis genuinely matches — teal for a
  library value, amber for external domain knowledge — and printed in each notebook's references block;
  `literature_plots.py` draws the credit-AUC *landscape* in the PD results notebooks (Tanna 2026 on
  Home Credit / Lending Club, Hollmann 2023 on Credit-g), as context under each paper's own protocol.

## Repository layout

Follows the group's project template — **fill it in, do not restructure**; the layout and its rules
are documented once, in [docs/TEMPLATE.md](docs/TEMPLATE.md).

```
CreditICL/
├── config/            flat YAML, one per experiment: Exp{1,2,3}_{PD,LGD}.yaml
├── data/              raw/pd (14) · raw/lgd (7) · processed cache          [gitignored]
├── checkpoints/       released weights; September checkpoints remain here  [gitignored]
├── docs/              the eight files below, each with inbound links
│   ├── EXPERIMENTAL_DESIGN.md   what the three experiments ask, and how they can fail
│   ├── PRIORS.md                how the credit prior is built
│   ├── CONFIG_REFERENCE.md      why each config value is what it is
│   ├── VSC.md                   the cluster: cost, limits, storage, surviving a long sweep
│   ├── RUNS.md                  the full write-up of every cluster run
│   ├── AGENTS_MEMORY.md         one line per run, four per dead end
│   ├── CHANGELOG.md             one chapter per date
│   └── TEMPLATE.md              the layout this project started from
├── notebooks/         0. General · 1. Experiment 1 · 2. Experiment 2 — one story (see its README)
├── output CreditICL/  everything generated: figures/ reference/ experiment_0..3/  [mostly gitignored]
├── scripts/           every runnable; calls into src/ · slurm/ holds the SLURM jobs
├── src/               data/ prior/ train/ eval/ models/ visualize/ utils/
└── tfm-library/       PINNED SUBMODULE — READ-ONLY (one exception)
```

`src/` holds reusable code; `scripts/` only wires config + data + method together. Runnables use a
CSV/manifest tracker so a sweep resumes and parallelises — which maps directly onto SLURM arrays.

## Setup (Windows, PowerShell 5.1)

> PowerShell 5.1 has no `&&`. Each command is on its own line deliberately.

**1. Clone with the submodule** (or initialise it in an existing checkout):

```powershell
git submodule update --init --recursive
```

**2. Create the venv.** Requires **Python 3.11 or 3.12** (VSC ships both, giving local↔cluster
parity). On this machine bare `python` is 3.14 — out of range — so name the interpreter explicitly.
The env is called `CreditICL` (not `.venv`) so it is distinguishable from sibling projects:

```powershell
py -3.12 -m venv CreditICL
.\CreditICL\Scripts\Activate.ps1
```

> Replacing an existing venv? Close every terminal/editor holding it first — Windows leaves a
> half-deleted skeleton otherwise, which then fails as `No pyvenv.cfg` or `No module named
> 'torch.nn'`. If activation is blocked: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`.

**3. Install.** PyTorch first (CPU-only is fine locally — all GPU work is on VSC), then the project:

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

**4. Verify:**

```powershell
python -c "import torch, numpy, sklearn, pandas; print(torch.__version__)"
pytest -q
```

`.vscode/settings.json` is committed and points VS Code at `.\CreditICL\Scripts\python.exe` (auto-
activated in new terminals; `pytest` rooted at `tests/`). One "Developer: Reload Window" may be
needed after first creating the venv.

## Running it

**Locally** (CPU) — `evaluate.py` preprocesses anything missing, so a fresh clone needs no setup:

```powershell
python scripts/evaluate.py --task lgd --models linear,catboost
```

> TabPFN-3 reads a **local** checkpoint (put weights in `checkpoints/` or point
> `CREDITICL_TABPFN_DIR` at them) — letting the package fetch its own triggers a token flow that
> fails on a compute node. Without the file the baseline skips with an explanatory line.

**On VSC** — validate the current code first, in one allocation:

```bash
bash scripts/slurm/update_env.sh
python -m src.utils.preflight && sbatch --clusters=mindwell scripts/slurm/exp0.slurm
# After the job passes:
python -m src.utils.preflight --require-exp0
python -m src.utils.run_experiment 1 --submit
```

The last command includes the main grid and component ablations. Repeat it after training to
submit only missing benchmarks; repeat after an interruption to resume incomplete runs.
The older pool-building launcher is not the corrected campaign entry point.

The local venv is for development and analysis only; **do not copy it to VSC**. The cluster's own
environment, GPU/partition choice, credit costs, storage rule and the checkpoint/resume mechanism are
all in **[docs/VSC.md](docs/VSC.md)**.

## Conventions

- **Library pin.** Literature claims are written against a recorded `tfm-library` commit (currently
  **`de324ef`**, audit 13-10-2026); record the pin whenever a result depends on it (`git submodule status`).
- **Cite code dumps by symbol name, never line number** — the dumps are refreshed and lines drift.
- **Distinguish evaluated from supported.** A mechanism existing in a paper's code is not that paper
  having measured it; several of this project's premises turned on exactly this.
- **`output CreditICL/` is generated; a handful of tracked files in it are the exception**
  (`README.md`, `All_Results.md`, `CAPTIONS.md`, each figure folder's `_figures.json`) — never sweep them
  into a deletion.
- **Never write project content into `tfm-library/`** (the one permitted file is
  `tfm-library/PROJECT_SPECIFIC.md`).

## License

MIT — see [LICENSE](LICENSE). This covers *our code only*; datasets under `data/` carry their own,
more restrictive terms.

---

## Based on the repository template

This repository was created from
[**andreasgoethals/0.-Template**](https://github.com/andreasgoethals/0.-Template).
[`docs/TEMPLATE.md`](docs/TEMPLATE.md) is that template: it explains every folder and file here, and
it is a **starting point, not a contract** — deviating where the work needs it is fine as long as you
say so. Generic rule changes belong at the source above.

*Keep this chapter, at the bottom, in every project that starts from the template. Everything above
it is that project's own.*
