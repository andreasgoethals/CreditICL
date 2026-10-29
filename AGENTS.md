# AGENTS.md — CreditICL

Instructions for AI agents working in this repository.

## 0. Before you start

1. [`docs/TEMPLATE.md`](docs/TEMPLATE.md) — the layout and rules this project started from.
   **Follow it**, and when you deviate — which is allowed, it is a starting point rather than a
   contract — **say so in your reply**. Never silently.
2. [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — the cluster runs already done and the dead
   ends already hit. Reading it is not optional: it is how you avoid resubmitting a configuration
   that failed last month, or spending an hour on a known dead end.
3. [`README.md`](README.md) — what this project actually is.

## 1. `tfm-library/` IS READ-ONLY. NO EXCEPTIONS BUT ONE.

A **pinned git submodule** holding the shared TFM literature. This repository does not track its
contents.

It is here so you can answer *"what does the literature say?"* and *"how does the official
implementation do this?"* **by reading and grepping files in this repository** — offline, no web
search, nothing from memory. Use it. A claim you can point at a path for beats a confident
sentence.

**Never create, edit, move, or delete anything inside it** — not a typo fix, not a note, not a
reformat. Anything you write there is lost when the pin moves, or corrupts a resource every other
project shares. **The one exception** is `tfm-library/PROJECT_SPECIFIC.md`, gitignored by the
library for exactly this purpose and created from `PROJECT_SPECIFIC.template.md`. If a library
document is wrong, report it to Andreas rather than patching it — the fix belongs in the
library's own checkout, where it flows down to every consumer. Never lint, format or test it.

Cite papers by path (`tfm-library/papers/<year>/...`, full text under `papers/text/`), and **code
dumps by symbol name, never by line number** — the dumps are re-snapshotted and line numbers drift
by thousands. Record the pin (`git submodule status`) next to any result that depends on it. Bump it with
`python -m src.utils.update_tfm_library`.

## 2. Never commit data or checkpoints

`data/` holds datasets under varying licences; `checkpoints/` holds multi-hundred-MB weights. Both
are gitignored. Do not `git add -f` them, and do not paste raw rows into commits, issues or docs.

## 3. Verify before you assert

This project's value is careful measurement. If a claim cannot be confirmed from the library, the
upstream source, or a primary reference, **say so** rather than filling the gap plausibly.
Distinguish what a paper *evaluated* from what its code merely *supports*; what a mechanism *can*
represent from how *often* it occurs; a library annotation from the primary source it summarises.

## 4. Do not train, install, or push without asking

Cluster runs cost real VSC credits. Installs change a shared environment. Pushes are
Andreas's action unless he has already authorized them in the current task. Reuse explicit
authorization; access alone does not authorize new workloads, installations or migrations.

### VSC access and storage

- Read the current access/storage sections in `docs/TEMPLATE.md` and the project-specific
  adoption state in `docs/VSC.md`. The template is already synchronized as of 29-10-2026;
  compare it byte-for-byte with the canonical template before assuming a newer copy is needed.
- Andreas manually runs `step ssh login --context VSC` in Windows PowerShell and approves
  KU Leuven Authenticator. Reuse the native Windows OpenSSH agent for this Windows account.
  Do not reinstall/bootstrap SSH per project, initiate authentication, add renewal hooks or
  schedule reminders. If the certificate is missing/expired, ask him to run that command.
- For relevant work, inspect VSC directly after verifying the bounded, noninteractive LF-stdin
  SSH channel in TEMPLATE.md. Check actual stdout/stderr and remote status; empty output with
  exit 0 is not success. Keep host-key verification enabled. Diagnose a failed channel before
  reading results or attempting mutations; network failures do not prove certificate expiry.
- The actual checkout and `.git` live on DATA. Freeze code, submodules and runtime versions
  used by active/queued jobs. Large persistent assets and versioned immutable dependency SIF
  images belong on allocated PROJECT storage; temporary working files belong on scratch.
  Local laptop venvs remain supported. Existing DATA venvs are a documented transitional
  exception for recovery, not the policy for new installations.
- For authorized submissions use the receipt-backed admission manager in `src.utils.submit_jobs`.
  Verify the site ceiling and actual expanded user queue; wait for capacity without imposing
  a running-job throttle. Preserve task IDs when splitting arrays. Never retry an uncertain
  acceptance or cancel another project's jobs to create admission space.
- Do not move populated environments. Propose a container rebuild from recorded dependencies,
  image hashes, binds, quotas and CPU/GPU acceptance before cutover; keep originals until no
  active, queued or resumable run needs them. No automatic migration or storage cleanup.
- Use `CREDITICL_PROJECT_ROOT` or `VSC_PROJECT_ROOT` for a shared allocation root (CreditICL
  is appended); legacy STAGING variables remain compatible. Export roots explicitly through
  the job submission path. Keep important results/checkpoints on persistent storage and remove
  only verified, retired, project-owned scratch after checking live jobs and recovery needs.
- Computational debugging belongs in authorized Slurm allocations. Distinguish local tests,
  a verified SSH read, Slurm status and actual CPU/GPU acceptance in every report.

## 5. Windows PowerShell 5.1 — no `&&`

No `&&`, no ternary, no `??`. One command per line, or `;` with `if ($?) { ... }`:

```powershell
python -m venv .venv
if ($?) { .\.venv\Scripts\Activate.ps1 }
```

SLURM job scripts are a separate world — bash on Linux, normal POSIX syntax. Keep the two straight.

## 6. Write both logs

Newest first, dates `DD-MM-YYYY`.

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — every substantive change, one bullet, **as short as
  possible**: what, and why only if it is not obvious. Usually one line; longer only when the change
  genuinely is. The detail belongs in the commit.
- [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — two things. **Every cluster run**, one row in
  the table: config, outcome, headline number. And **every failure** that cost more than a couple of
  minutes, four lines: **Tried**, **Result**, **Why**, **Instead** — even when the eventual fix
  worked, because the dead end is the expensive part.
- [`docs/RUNS.md`](docs/RUNS.md) — the **full write-up** of every cluster run: date, job id,
  hyperparameters, the resolved config, results, bugs, and an interpretation. After each run
  Andreas uploads the output back into a session for debugging; that upload is the only chance to
  capture it, so write the entry then. Use the template in the file.

## 7. Notebooks and figures

- A notebook contains **no `def` and no `class`** — logic goes in `src/` — and its **last code
  cell prints a text summary**, section by section, in the notebook's own section order.
- **Never pick a colour or a size.** `src/visualize/style.py` owns both, so every notebook here
  looks the same. Add a new one there, once, not in the notebook.
- Save through `src/visualize/figures.FigureSaver`: **PDF only**, into that notebook's own
  folder, which it clears before drawing. The notebook displays each figure inline.
- Use `style.figsize(style.WIDTH_FULL)` or `WIDTH_HALF`: every figure is drawn at the width it
  will occupy on an **A4** page, and never rescaled afterwards — rescaling carries the text with it.
- Captions are the **paper's** captions: pure description, ready to paste under the figure.

## 8. Say you are done only when it runs

```powershell
python -m pytest -q
```

And `python -m src.utils.run_notebooks` if you touched a notebook or anything under
`src/visualize/`. Nothing runs these for you — no CI, no hook — so run them before you claim
anything is finished.

## Shared VSC Library

`vsc-library/` is a pinned, entirely read-only infrastructure submodule. Read its
README, source snapshot, guides and Slurm examples; never edit, refresh, lint,
format or test inside it. There is no PROJECT_SPECIFIC exception. Keep this
project's VSC choices and operating instructions in `docs/VSC.md`, outside the
submodule. Shared corrections belong in the standalone VSC Library checkout.
Record the library pin when operational evidence depends on it.
