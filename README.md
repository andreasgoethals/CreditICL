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
   the priors apart. That probe does not establish that O'Prior fails on credit tasks; it also reports strong Home Credit results.
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

Each experiment has a task config, `config/Exp{1,2,3}_{PD,LGD}.yaml`, with a `sweep:` block
listing the knobs it crosses. Exp3 inherits Exp2's prior and geometry through `prior_file`.
Full rationale in
[EXPERIMENTAL_DESIGN](docs/EXPERIMENTAL_DESIGN.md); every value in
[CONFIG_REFERENCE](docs/CONFIG_REFERENCE.md).

| # | question | how | arms/track |
|---|---|---|---|
| **0** | **Does the revised pipeline run correctly?** | CPU + GPU Slurm jobs: portable tests, CUDA/save/resume checks, full-table benchmarks, 34 long-context probes and 16 retention-method round trips | 3 base training routes |
| **1** | **Does the prior help at matched compute?** | four mixtures × three seeds, plus five component removals × three seeds; 12,500 steps | 27 |
| **2** | **Does continued pretraining transfer?** | two released backbones × four mixtures × two peak rates × three seeds, 10,000 updates | 48 |
| **3** | **Which constraints preserve generality?** | fixed 50% mixture; no decay, L2-SP, DiagScale, Q/V LoRA; three seeds | 24 new + 6 reused; 18 pilots |

Counts include seeds and both prediction tasks: **Exp1 = (4 mixtures + 5 component removals) × 3 seeds × 2 tasks = 54 runs**;
**Exp2 = 2 backbones × 4 mixtures × 2 rates × 3 seeds × 2 tasks = 96 runs**.
**Exp3 = 48 new main runs + 12 reused Exp2 baselines + 36 development-only pilots**.
The component removals are compared with the main 50% mixture; they are not additional mixture levels.
The 25% Exp1 extension was approved on 26-10-2026 after the initial 48-run campaign. It appends
indices 9–11 per task, preserving existing run names and benchmark tags. Completed runs stay valid;
the extra six runs and their benchmarks remain to be done. Figures sort mixtures numerically.

Exp2's TabICLv2 arm uses the upstream final-stage table distribution: **400–60,000 rows,
log-uniform, 79–81% context**, with the upstream row-dependent synthetic feature ceiling.
TabPFN-3 keeps 1,024-row tasks. The 0%-credit continuation is the original-TabICL-prior control;
released weights are a separate reference. No continuation is guaranteed to avoid forgetting.
Both credit data and 25 native OOD datasets per task are benchmarked with all eligible rows/features
and five folds. The larger TabICL recipe needs new CPU/GPU validation and measured cost before launch.

`credit_fraction` is the master switch: the share of each batch drawn from our credit-targeted path,
the rest from the unmodified TabICL prior. `0.0` is the control every arm is measured against.

Each experiment is **two phases**: phase 1 trains (permanent checkpoints along each arm), phase 2 benchmarks every
checkpoint plus a shared reference column (released TabICLv2, TabPFN-3, CatBoost, linear) through the
*same* code — the ordering is a fact about the data, not a convention.

## The four code pipelines

Each lives in `src/` and writes under `output CreditICL/`. Main experiment entry points live in `scripts/`; diagnostics and maintenance run as Python modules
(one folder per experiment; small files on the backed-up tier, checkpoints on project storage under
the same layout) — the code picks the tier automatically (`output CreditICL/README.md`).

| # | pipeline | code | run it with | produces |
|---|---|---|---|---|
| 1 | data | `src/data/` | `python -m src.data.preprocess` | the processed-dataset cache |
| 2 | prior | `src/prior/` | `python -m src.prior.generate` / `src.prior.measure` | the synthetic task stream + reports |
| 3 | training | `src/train/` | `scripts/pretrain.py` | permanent and resumable checkpoints per arm |
| 4 | eval | `src/eval/` | `scripts/evaluate.py` | scores on the 21 real datasets + out-of-domain suites |

Pipeline 2 is the research contribution; pipeline 1 exists only to feed pipeline 4. A few facts worth
knowing:

- **Data cache:** one parquet file per dataset + a `meta.json` written *last* as the completeness
  marker. Parquet (not `.npy`) so categorical indices survive — CatBoost and TabPFN need them.
  Recipes originated in the sibling TabPFNCredit project. CreditICL's documented corrections
  have dedicated regressions; unchanged columns are still compared with the sibling cache.
- **Prior pools:** training can read datasets pre-generated once per variant
  (`prior_cache/<task>__original`, `…__credit_v1`) so a GPU never waits on CPU generation and the two
  sides differ only by design, not by draw. `--prior-source generate` builds live instead.
- **Notebooks** — twelve, in four numbered chapters ([reading order](notebooks/README.md)):
  benchmark/prior diagnostics; Experiment 1 training and matched-budget effects;
  Experiment 2 adaptation within each backbone; and Experiment 3 retention methods with development-only pilot selection.
  Logic lives in `src/visualize/`. Every figure is a saved A4 vector PDF with a descriptive caption.
  Final panels require complete current-protocol benchmark cells; monitoring and local file absence
  are never presented as final efficacy or live queue state. Exp3 settings are selected on development;
  designated-holdout historical exposure is disclosed. Official notebook PDFs are committed.

## Repository layout

Follows the group's project template — **fill it in, do not restructure**; the layout and its rules
are documented once, in [docs/TEMPLATE.md](docs/TEMPLATE.md).

```
CreditICL/
├── config/            flat YAML, one per experiment: Exp{1,2,3}_{PD,LGD}.yaml
├── data/              raw/pd (14) · raw/lgd (7) · processed cache          [gitignored]
├── checkpoints/       released upstream weights                         [gitignored]
├── docs/              the eight files below, each with inbound links
│   ├── EXPERIMENTAL_DESIGN.md   what the three experiments ask, and how they can fail
│   ├── PRIORS.md                how the credit prior is built
│   ├── CONFIG_REFERENCE.md      why each config value is what it is
│   ├── VSC.md                   the cluster: cost, limits, storage, surviving a long sweep
│   ├── RUNS.md                  the full write-up of every cluster run
│   ├── AGENTS_MEMORY.md         one line per run, four per dead end
│   ├── CHANGELOG.md             one chapter per date
│   └── TEMPLATE.md              the layout this project started from
├── notebooks/         0. General · Experiments 1–3 — one story (see its README)
├── output CreditICL/  everything generated: figures/ reference/ experiment_0..3/  [mostly gitignored]
├── scripts/           four main experiment entry points · slurm/ holds the Slurm jobs
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
> fails on a compute node. Without the file validation fails before production starts.

**On VSC** — validate the current code in a CPU job, then a GPU job:

**While any CreditICL job is queued or running, keep its checkout and environment frozen.**
Do not pull, update packages, reset output or use the setup commands below in that checkout
until it finishes. The revised Exp2 needs fresh validation afterwards; completed Exp1
checkpoints and benchmarks remain evidence for their recorded revision.

```bash
source scripts/slurm/_activate_env.sh &&
python -m src.utils.preflight &&
sbatch --clusters=wice scripts/slurm/exp0_cpu.slurm
# After the CPU job passes:
python -m src.utils.preflight --require-exp0-cpu && sbatch --clusters=mindwell scripts/slurm/exp0.slurm
# After the GPU job passes:
python -m src.utils.preflight --require-exp0
python -m src.utils.run_experiment 1 --submit
```

Use `update_env.sh` or `fetch_ood` only when preparing a missing environment/cache or resolving
a prerequisite failure, before CPU validation. This implementation adds no packages. The current
VSC dependency audit already passed; do not routinely upgrade that environment. The guarded
pull and Exp0-only cleanup sequence is in [VSC](docs/VSC.md).

After Exp1's extension and benchmarks, continue without changing the validated code:

```bash
python -m src.utils.run_experiment 2 --single-cluster --submit
# Repeat after training to submit missing benchmarks; after the required final benchmarks:
python -m src.experiments.selection --stage learning-rate
python -m src.utils.run_experiment 3 --variant pilots --single-cluster --submit
# Repeat the pilot planner after training; after all pilot development benchmarks:
python -m src.experiments.selection --stage adapters
python -m src.utils.run_experiment 3 --single-cluster --submit
# Repeat the main planner after training to submit missing benchmarks.
```

Selection uses complete development folds only. Exp2 retains every mixture/rate. Exp3 freezes
the full-model rate and adapter settings in hashed receipts; no YAML edit or retraining of the
12 reused Exp2 baselines is needed. Its eight-checkpoint figures report credit and native OOD
trajectories separately. See [the registered design](docs/EXPERIMENTAL_DESIGN.md).

The Experiment 1 planner includes its main grid and component ablations. Repeat each experiment's
planner after training to submit only missing benchmarks, or after an interruption to resume incomplete runs.
The older pool-building launchers refuse submission. Experiment 0 uses selected full-row,
full-feature tables and wide-input model probes (2,986 PD / 601 LGD columns); the complete five-fold credit/OOD benchmark runs
in Experiments 1–3. Every eligible column is retained; the synthetic training width is not an
inference limit. The GPU validation job requires the updated CPU job's PASS receipt first.

Freeze code, configuration, dependencies and inputs from the CPU submission through GPU
completion. The GPU script resumes a matching verified campaign, or starts fresh when the code/CPU receipt has changed; it saves every
finished benchmark fold immediately and reuses completed stages. `DEFERRED` (exit 64) means the
allocation ended: resubmit `scripts/slurm/exp0.slurm` without pulling, resetting or rerunning CPU.
`FAILED` requires inspecting the reported error. A two-hour limit per benchmark cell stops an
unacceptably slow probe early; it never reduces rows, columns, folds or ensemble members. Large
TabICL inputs use upstream column-attention caches and bounded row blocks to avoid materializing
the full rows × features × embedding array. Exp0 checks prediction equivalence on CUDA before
running the full-table capacity probes. The 24-hour allocation is a limit, not a completion estimate.

The local venv is for development and analysis only; **do not copy it to VSC**. The cluster's own
environment, GPU/partition choice, credit costs, storage rule and the checkpoint/resume mechanism are
all in **[docs/VSC.md](docs/VSC.md)**.

## Conventions

- **Library pin.** Literature claims are written against a recorded `tfm-library` commit (currently
  **`702d9ab`**, continued-pretraining audit 24-10-2026); record the pin whenever a result depends on it (`git submodule status`).
- **Cite code dumps by symbol name, never line number** — the dumps are refreshed and lines drift.
- **Distinguish evaluated from supported.** A mechanism existing in a paper's code is not that paper
  having measured it; several of this project's premises turned on exactly this.
- **`output CreditICL/` is generated.** `python -m src.utils.clean_run --reset-output --clean`
  clears generated output on both tiers, including trained checkpoints, while preserving inputs,
  released weights and the directory README. Use only for an intentional fresh campaign.
- **Never write project content into `tfm-library/`** (the one permitted file is
  `tfm-library/PROJECT_SPECIFIC.md`).

## License

MIT — see [LICENSE](LICENSE). This covers *our code only*; datasets under `data/` carry their own,
more restrictive terms.

Released model weights retain their upstream licences. In particular, TabPFN-3 uses the
[TabPFN-3 Non-Commercial License](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE),
which also applies to derivatives, including our continued-pretraining checkpoints.

---

## Based on the repository template

This repository was created from
[**andreasgoethals/0.-Template**](https://github.com/andreasgoethals/0.-Template).
[`docs/TEMPLATE.md`](docs/TEMPLATE.md) is that template: it explains every folder and file here, and
it is a **starting point, not a contract** — deviating where the work needs it is fine as long as you
say so. Generic rule changes belong at the source above.

*Keep this chapter, at the bottom, in every project that starts from the template. Everything above
it is that project's own.*
