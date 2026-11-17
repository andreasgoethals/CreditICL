# Agent instructions — CreditICL

## Read before work

Read [README.md](README.md). The maintainer checkout also has a local-only `docs/` directory.
Read its `TEMPLATE.md` and `AGENTS_MEMORY.md` before project work. For VSC operations,
use its `VSC.md` and dated `RUNS.md`. Obtain these from the maintainer if working from a
public clone that lacks them. The local manual covers deployment, access, storage and recovery.

## Protect shared references and research assets

- **`tfm-library/` is a pinned read-only submodule.** Read papers and repository snapshots in
  place. Never edit, move, delete, lint, format or test its contents. The sole exception is its
  gitignored `PROJECT_SPECIFIC.md`, created from `PROJECT_SPECIFIC.template.md`. Corrections
  belong in the standalone library checkout. Use `src.utils.update_tfm_library` for a reviewed
  pin update; record the pin beside evidence that depends on it.
- **`vsc-library/` is entirely read-only**, with no PROJECT_SPECIFIC exception. Read its guides,
  snapshot and Slurm examples; keep project adoption decisions in `docs/VSC.md`.
- Cite papers by path and code dumps by **symbol name**, not unstable snapshot line numbers.
  Read and search the pinned library offline for literature and implementation evidence;
  do not substitute recollection for its primary sources. Distinguish
  evaluated behavior from code support, mechanism capability from measured frequency,
  and a local test from actual VSC acceptance. State unverified claims as such.
- Never commit datasets, released/trained weights, optimizer state or raw private rows.
  Do not force-add ignored assets. Dataset and upstream model licences remain applicable.
- Publish restricted dataset aliases only. Keep source identifiers, mappings, raw columns and
  source dimensions in approved restricted storage, including the ignored `.private-data/`
  configuration. Check staged source, executed notebook text and PDF text with
  `python -m src.utils.check_publication --staged`. Inspect rendered figures too.
- **Weights stay on VSC.** “All output” means analysis files: scores, predictions, logs,
  summaries/configs, receipts and figure metadata. Filter inventories with
  `src.utils.result_manifest` before SFTP and verify the retained hashes. Exclude checkpoints,
  pools, runtime images and unfiltered archives, including hidden backup/transfer copies.
  Download model files only when Andreas explicitly requests them for local use.

## Authorization and deployment

Training spends credits; installs change environments; pushes publish work. Obtain authorization
unless it already exists in the current task. Reuse explicit authorization rather than asking
again. Access by itself does not authorize a new workload, cancellation, migration or cleanup.

The maintainer's local VSC runbook covers authentication, validation, deployment and cleanup.

## Local commands and verification

PowerShell 5.1 does not support `&&`, ternary syntax or `??`. Use separate commands and check
success before the next step. Slurm scripts use Bash on Linux. Test any user-facing PS5.1
instructions in that interpreter rather than assuming PowerShell 7 compatibility.

Use the existing local environment and make Git Bash available to shell tests:

~~~powershell
.\.CreditICL\Scripts\python.exe -m pytest -q
~~~

Run appropriate checks before claiming completion. After notebook or `src/visualize/` changes,
also execute `python -m src.utils.run_notebooks` and inspect the rendered figures. Do not
install dependencies or rerun VSC campaigns merely for prose changes. No CI or hook runs
these checks automatically; disclose remaining failures or environment-specific skips.

## Notebooks and figures

Notebooks contain no `def` or `class`; logic belongs in `src/`. Each notebook ends with a code cell
that prints its text summary in section order. Preserve executed outputs for inspection.

`src/visualize/style.py` owns colours, dimensions and typography. Use `style.figsize` with the
shared A4 widths; do not choose local sizes/colours or rescale figures afterwards.
Save **PDF only** through `FigureSaver` in the notebook's own folder, display each figure inline
and save descriptive paper-ready captions. The saver clears only that notebook's generated files.
Check data, labels, units, overlap and readability in every final figure, not merely execution.

## Durable documentation

Use newest-first DD-MM-YYYY entries:

| File | What to record |
|---|---|
| `docs/CHANGELOG.md` | One concise bullet per substantive change |
| `docs/AGENTS_MEMORY.md` | Compact run/revision/job/outcome records and reusable lessons |
| `docs/RUNS.md` | Full dated configurations, provenance, results, failures and interpretation |

Record every cluster run in both the compact agent-memory index and the full run record.
For a failure costing more than a few minutes, record **Tried / Result / Why / Instead** in
agent memory. Inspect available VSC evidence directly; uploads are a fallback, not the only
source. Preserve historical failures and distinguish dated evidence from current instructions.
Keep README as the entry point, configuration/design rationale in their own docs and generated
notebook summaries/captions under their generators' control. `CLAUDE.md` imports this file.
