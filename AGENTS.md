# AGENTS.md — CreditICL

Instructions for AI agents working in this repository. Read this before
touching anything.

## 1. `tfm-library/` IS READ-ONLY. NO EXCEPTIONS BUT ONE.

`tfm-library/` is a **pinned git submodule** — a snapshot of a shared,
public knowledge base ([TFM_Library](https://github.com/andreasgoethals/TFM_Library))
that several projects consume. This repository does **not** track its
contents.

**Never create, edit, move, or delete anything inside `tfm-library/`** —
not to fix a typo, not to add a note, not to "just" reformat. Anything you
write there is either silently lost when the pin moves or silently
corrupts a resource shared with other projects.

**The single exception** is `tfm-library/PROJECT_SPECIFIC.md`. That
filename is gitignored *by the library*, so it lives beside the
literature, survives `git submodule update`, and can never be pushed
upstream. It is the only place project-specific notes about the
literature belong. It is created by copying
`tfm-library/PROJECT_SPECIFIC.template.md`; follow the six entry rules in
that template exactly.

**If a library document is wrong, do not patch it.** Report it to Andreas
so it is fixed in the library's own checkout and flows down to every
consumer.

Read `tfm-library/AGENTS.md` for the full upstream contract.

### Citing the library

- Papers by path: `tfm-library/papers/<year>/<MM>_<Author>_<Title>.pdf`,
  full text at `tfm-library/papers/text/<year>/<same-name>.txt`.
- **Code dumps by symbol name, never by line number.** The dumps are
  refreshed periodically and line numbers drift by thousands.
  `` `TabICL.txt`, `GraphSCM.__call__` `` — yes.
  `` `TabICL.txt:24994` `` — never.
- When a result depends on the literature, record the pinned commit.
  Current pin: **`21d555a`** (2026-08-05).

## 2. Follow the template

`docs/TEMPLATE.md` defines the layout and the rules. **Adhere to it.** The only
reason to deviate is that the user has explicitly told you to, and when you
deviate you must **say so in your reply** — never silently.

In short: `src/` holds all importable logic, `scripts/` holds only runnables
you actually invoke, `config/` holds YAML, `docs/` holds documentation,
`output/` holds everything the code generates, `tests/` mirrors `src/`.

See [README.md](README.md#repository-layout) for what goes where here.

## 3. Never commit data or checkpoints

`data/` holds credit-risk datasets under varying licences and
`checkpoints/` holds multi-hundred-MB weights. Both are gitignored. Do not
add them, do not `git add -f` them, and do not paste raw rows into
commits, issues, or docs.

## 4. Verify before you assert

This project's entire value is careful measurement. If a claim cannot be
confirmed from the library, the upstream source, or a primary reference,
**say so** rather than filling the gap plausibly. Distinguish:

- what a paper *evaluated* from what its code merely *supports*;
- what a mechanism *can represent* from how *often* the prior produces it;
- a library annotation from the primary source it summarises.

Several claims in this project's framing were revised on exactly these
grounds — see [`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md)
§"Verified premises".

## 5. Do not run training, install packages, or push without asking

Pretraining runs cost real VSC credits. Package installs change a shared
environment. Pushes are Andreas's action. Ask first.

## 6. Windows PowerShell 5.1 — no `&&`

Andreas works in **Windows PowerShell 5.1**, which has **no `&&`
operator**, no ternary, and no `??`. Never hand over bash-chained
commands. One command per line, or `;` with `if ($?) { ... }`.

```powershell
python -m venv .venv
if ($?) { .\.venv\Scripts\Activate.ps1 }
```

SLURM job scripts are a separate world — those are bash on Linux and use
normal POSIX syntax. Keep the two straight.

## 7. Log substantive changes

Every substantive change gets a dated entry in
[`docs/CHANGELOG.md`](docs/CHANGELOG.md) — one line each for what and why.
Mirrors the library's own convention.
