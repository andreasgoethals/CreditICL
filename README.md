# CreditICL

**How does a credit-targeted synthetic prior affect tabular foundation models?**
CreditICL studies this through controlled pretraining, continued pretraining and constrained
adaptation. It uses TabICLv2 and TabPFN-3 on 14 binary credit benchmarks associated with
probability of default (PD), seven bounded regression benchmarks associated with loss given
default (LGD), and 25 out-of-domain (OOD) datasets per task.

The intervention is a complete prior package: targets, feature exposure, encoding and filtering.
The study measures its effect at fixed budgets; it does not isolate every credit mechanism or
claim that existing priors cannot represent imbalance or endpoint mass. Dataset semantics,
historical benchmark exposure and the absence of temporal validation limit deployment claims.

Research software by Andreas Goethals, KU Leuven. See [the design](docs/EXPERIMENTAL_DESIGN.md)
for hypotheses, evidence and limitations, and [CITATION.cff](CITATION.cff) to cite the software.

## Start here

| Purpose | Document |
|---|---|
| Research questions and comparisons | [Experimental design](docs/EXPERIMENTAL_DESIGN.md) |
| Synthetic generator | [Prior construction](docs/PRIORS.md) |
| Recipes and sweep expansion | [Configuration reference](docs/CONFIG_REFERENCE.md) |
| Local development and saved analyses | [Local setup](#local-setup) · [Notebook guide](notebooks/README.md) |
| VSC validation, submission and recovery | [VSC runbook](docs/VSC.md) |
| Dated results, failures and changes | [Run record](docs/RUNS.md) · [Agent memory](docs/AGENTS_MEMORY.md) · [Changelog](docs/CHANGELOG.md) |
| Generated files and collection | [Output guide](<output CreditICL/README.md>) |

## Experiments

| Experiment | Question and design | Training runs across PD and LGD |
|---|---|---:|
| **0: validation** | CPU tests, data/configuration checks, CUDA correctness, full-table probes and recovery tests | Diagnostic jobs |
| **1: prior usefulness** | Scratch TabICL; four mixtures and five component removals, three seeds, 12,500 updates | **54** |
| **2: continued pretraining** | Two released backbones × four mixtures × two fixed learning rates × two seeds, 1,500 updates | **64** |
| **3: adaptation constraints** | Fixed 50% mixture; six no-decay, L2-SP, DiagScale and Q/V LoRA recipes, two backbones and two seeds, 1,500 updates | **48 new + 16 reused Exp2 baselines** |

Mixtures contain 0%, 25%, 50% or 100% credit tasks in expectation. The remainder uses the public
TabICL prior, including for TabPFN-3: it does **not** reconstruct TabPFN's private prior. Released
weights are a separate reference from the 0% continued-pretraining control.

Exp2/3 retain checkpoints at 125, 250, 500, 1,000 and 1,500 updates. All are benchmarked; the final
checkpoint is the primary endpoint. Both fixed rates and all recipes are reported, without a
learning-rate pilot or best-on-test selection. Comparisons retain all prediction-time-eligible
features and fitted rows, with five common folds. PD threshold tuning uses an inner training split.
Native OOD performance is reported alongside credit performance.

**Campaign snapshot, 9 November 2026.** Exp1 training and benchmarking are complete. All 64 Exp2
training runs and 320 checkpoint benchmarks finished and were verified. Revised Exp0 passed CPU
and GPU validation. Mindwell accepted all 48 Exp3 training tasks. Dated job records are in
[VSC](docs/VSC.md#current-deployment) and [RUNS](docs/RUNS.md#runs).

## Local setup

Use **Python 3.12** for the established local/VSC pairing; supported versions are 3.11–3.12.
Existing environments need no routine upgrade. For a new Windows checkout, run in PowerShell:

~~~powershell
git clone --recurse-submodules https://github.com/andreasgoethals/CreditICL.git
cd CreditICL
py -3.12 -m venv .CreditICL
.\.CreditICL\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,eval]"
~~~

Run each step only after the preceding one succeeds. PowerShell 5.1 does not support `&&`.
The CPU environment is for development and analysis; model inference and training run on VSC.
Datasets are not distributed with the repository: a fresh clone is not a complete benchmark
installation. Do not download weights merely to execute analysis notebooks.

Seven restricted credit datasets appear only under the aliases `PropPD1`–`PropPD2` and
`PropLGD1`–`PropLGD5`. Public benchmarks keep their names. The full benchmark requires the
institution's local dataset configuration. Published exploration figures show source dimensions,
column diagnostics and feature eligibility details only for public datasets.

The Slurm shell tests need Bash. If Git for Windows is installed but Bash is absent from PATH:

~~~powershell
$creditIclGitRoot = Split-Path (Split-Path (Get-Command git).Source -Parent) -Parent
$creditIclBashBin = Join-Path $creditIclGitRoot "bin"
if (Test-Path (Join-Path $creditIclBashBin "bash.exe")) { $env:Path = "$creditIclBashBin;$env:Path" }
~~~

Then run the tests:

~~~powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest -q
~~~

VS Code uses `.CreditICL/Scripts/python.exe`. Local tests do not replace VSC Experiment 0.

## Analyse results

Saved notebooks include figures and section-order printed summaries. To regenerate them from
available analysis files:

~~~powershell
$env:CREDITICL_DRAW_WORKERS = '1'
python -m src.utils.run_notebooks
~~~

Prior notebooks generate seeded CPU diagnostics; results notebooks read scores, logs, resolved
configurations and receipts. Missing results are identified rather than fabricated. See the
[notebook guide](notebooks/README.md) for the storyline and input requirements.

**Weights stay on VSC.** Filter the remote inventory before collection:

~~~powershell
python -m src.utils.result_manifest inventory.json --out analysis.json
~~~

This creates a plan, not a download. Transfer only its listed analysis files and verify their
hashes. Exclude checkpoints, optimizer states, prior pools, runtime images and unfiltered archives.

## Run on VSC

Follow [the VSC runbook](docs/VSC.md): local edit/test → commit/push → safe DATA pull →
CPU/GPU validation → authorized, receipt-backed submission. Freeze code, dependencies and inputs
while active or queued jobs consume them. Validate changed code in a separate DATA checkout
when necessary.

Git and compact records live on DATA; persistent inputs, results and immutable runtime images
live on PROJECT; temporary work uses native scratch. Do not relocate a populated environment,
reset output or repeat a campaign merely to update documentation.

## Repository layout

| Path | Role |
|---|---|
| `config/Exp*.yaml` | Experiment recipes and sweeps |
| `scripts/` and `scripts/slurm/` | Direct experiment entry points and cluster launchers |
| `src/data/`, `src/prior/`, `src/train/`, `src/eval/` | Data, generation, training and evaluation |
| `src/experiments/`, `src/validation/`, `src/utils/` | Fixed designs, validation and maintenance |
| `src/visualize/`, `notebooks/` | Shared plotting logic and thin reports |
| `output CreditICL/` | Generated records and selected committed notebook exports |
| `data/`, `checkpoints/` | Licensed inputs and model assets, outside Git |
| `tfm-library/`, `vsc-library/` | Pinned, read-only literature and infrastructure references |

The flat config names, numbered notebook chapters and `figures/<notebook>` exports are deliberate
retained deviations from the template's experiment-first layout. Their established paths preserve
results and run identities. Inspect pins with `git submodule status`; cite repository dumps by
symbol name rather than unstable line numbers.

## Citation and licences

Use [CITATION.cff](CITATION.cff) for the software and record the exact commit, resolved recipe,
runtime and reference-library pins used in an analysis. No paper DOI or release date is asserted.
Cite the relevant TabICLv2, TabPFN-3 and methodological sources separately.

Original project code is MIT licensed; [LICENSE](LICENSE) distinguishes third-party code,
datasets and model weights. Their upstream terms continue to apply. TabPFN-3 weights and adapted
derivatives use the [upstream TabPFN-3 licence](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE).

## Based on the repository template

Created from [andreasgoethals/0.-Template](https://github.com/andreasgoethals/0.-Template).
[docs/TEMPLATE.md](docs/TEMPLATE.md) is the shared starting policy; project-specific choices and
deviations are documented here and in the VSC runbook. Generic changes belong upstream.
