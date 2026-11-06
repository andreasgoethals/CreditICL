# Agent instructions — CreditICL

## Read before work

Read [docs/TEMPLATE.md](docs/TEMPLATE.md), [docs/AGENTS_MEMORY.md](docs/AGENTS_MEMORY.md)
and [README.md](README.md). For VSC work, also read [docs/VSC.md](docs/VSC.md) and the relevant
dated [run records](docs/RUNS.md). These prevent duplicate submissions and known failed retries.

The template is a starting policy, not an immutable layout contract. Explain deliberate
deviations in the reply. Compare its copy with the canonical template before syncing;
use the established byte-for-byte document update process and preserve unrelated changes.
Project choices belong in this repository, not in the shared template.

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
- **Weights stay on VSC.** “All output” means analysis files: scores, predictions, logs,
  summaries/configs, receipts and figure metadata. Filter inventories with
  `src.utils.result_manifest` before SFTP and verify the retained hashes. Exclude checkpoints,
  pools, runtime images and unfiltered archives, including hidden backup/transfer copies.
  Download model files only when Andreas explicitly requests them for local use.

## Authorization and deployment

Training spends credits; installs change environments; pushes publish work. Obtain authorization
unless it already exists in the current task. Reuse explicit authorization rather than asking
again. Access by itself does not authorize a new workload, cancellation, migration or cleanup.

Follow **local edit/test → commit/push → safe pull on DATA → Slurm validation**. Never deploy
implementation through untracked script copies or remote code edits. Freeze source, submodules,
dependencies and inputs consumed by active, queued or recoverable jobs. Use a separate actual
DATA checkout and receipt namespace for concurrent validation.

### VSC access

- Andreas manually runs `step ssh login --context VSC` in Windows PowerShell and approves
  KU Leuven Authenticator. Reuse the existing native Windows SSH agent. Do not reinstall SSH,
  initiate authentication, add renewal hooks or schedule authentication reminders.
- If the certificate is missing/expired, ask for manual renewal and wait. A network failure
  does not prove expiry. Keep credentials and authentication URLs out of logs and commits.
- Verify the bounded, noninteractive LF-stdin SSH method in `TEMPLATE.md`, with strict host keys
  and actual stdout/stderr/remote status. Empty output with exit 0 is not success; diagnose a
  broken channel before inspecting results or attempting mutations.
- Run available diagnostics and authorized maintenance yourself. Do not ask Andreas to relay
  command output when access is available. Manual MFA and genuinely unavailable access are
  exceptions. Computational debugging belongs in an authorized Slurm allocation.
- Do not create a scheduled monitor for this campaign. Monitor within the active task when
  requested; monitoring never overrides validation or submission gates.

### Storage and submission

- Git checkout and `.git`: **DATA**. Large persistent assets/results and immutable dependency
  SIF images: **PROJECT**. Temporary working I/O: native **scratch**. Laptop venvs are supported.
  Legacy DATA environments are documented recovery exceptions, not new-installation policy.
- Do not move populated environments. A rebuild/cutover needs recorded dependencies, image
  hashes, binds, quota checks, CPU/GPU acceptance and preserved rollback. Retire originals
  only after recovery and active/queued/resumable consumer checks.
- `CREDITICL_PROJECT_ROOT` or `VSC_PROJECT_ROOT` names a shared allocation root; the resolver
  appends CreditICL. Pass reviewed roots explicitly through job submission. Legacy `STAGING`
  aliases remain compatible. No automatic migration follows a documentation update.
- Use `src.utils.submit_jobs` for receipt-backed admission. Verify the site ceiling and expanded
  live user queue, preserve task IDs when splitting arrays and wait/refill at capacity without
  an artificial running throttle. Never retry uncertain acceptance or cancel other projects.
- Preserve checkpoints, results, receipts and recovery evidence. Cleanup targets must be
  verified obsolete, project-owned and unused; never wipe shared scratch. Persistence on
  PROJECT is not independent backup coverage.
- Report local tests, SSH reads, scheduler outcomes and actual artifact acceptance separately.

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
