# Agents' memory — runs and dead ends

What is worth carrying between sessions: **the cluster runs that have been done**, and **the things
that turned out not to work**. Read it before starting; add to it as you go.

Not the changelog — that records edits to the repository. This records experience: what was run,
what came out, and what is already known to fail.

**Keep it short.** One line per run, four per dead end. Newest first, dates `DD-MM-YYYY`. Never
delete an entry: a run you would otherwise repeat and a dead end you already paid for are both
evidence.

## How a run comes back to you

**After every cluster run, Andreas downloads the output and uploads it into a chat session so an
agent can read and debug it.** Expect to be handed a folder of `.log` files and `.csv` manifests
with no other context, and expect that to be the *only* record of the run — so:

- **Read them properly before theorising.** The `hardware:` line says which GPU it was; the
  `telemetry:` summary at the end of the log says whether the GPU was starved; `grads step` lines
  say whether every block was learning. Most "the model did not learn" questions are answered in
  those three places.
- **Write the run up in [`RUNS.md`](RUNS.md)** using its template, and add the one-line row to the
  table below. The upload is the only chance to capture it.
- **A number that looks too good is a bug until proven otherwise.** Check the config actually
  used (`grid levers:` and `credit_fraction IN USE:`), not the config on disk now.

## Runs

One row per cluster run worth remembering — which is most of them, because *"have we already tried
that configuration?"* is the question this table exists to answer. The **full** write-up of each
one lives in [`RUNS.md`](RUNS.md); this table is the index.

| Date | Run | Outcome | Notes |
|---|---|---|---|
| | | | |

**The table starts empty on purpose, and that is not the same as "nothing has run".** Submission
attempts were made on wICE before 11-08-2026 and their logs were read to fix the bugs recorded
below, but no row was ever written and those logs are no longer on disk, so there is nothing here
that can be trusted. **Add a row for every run from now on**, at submission time rather than
afterwards — a job that dies at the walltime never comes back to write its own entry.

- **Run** — the config or arm, and the commit if it matters.
- **Outcome** — `done` / `walltime` / `OOM` / `crashed` / `diverged` / `cancelled`.
- **Notes** — one line: the headline number, the output path if it is worth finding again, or the
  single thing the run showed. A number here saves re-reading `output/results/`.

## Dead ends

Anything that cost more than a couple of minutes and did not work — including what you eventually
fixed, because the fix is one changelog line and the dead end was the hour.

### 13-08-2026 — `module --force purge`, and pinning a Python module by name
- **Tried:** `setup_venv.sh` opened with `module --force purge` and then
  `module load Python/3.12.3-GCCcore-13.3.0`.
- **Result:** Lmod: *"These module(s) or extension(s) exist but cannot be loaded as
  requested"* — on login-1, while the same module had worked on login-2.
- **Why:** two causes stacked. (1) The `cluster/*` modules on VSC are **sticky** and set up the
  architecture-specific `MODULEPATH`; `--force purge` removes them too, collapsing the tree so
  a module that exists cannot be resolved. (2) Module trees are **per-architecture**
  (`/apps/leuven/rocky9/<arch>/<toolchain>/…`), so a name present on skylake can be absent on
  another login node.
- **Instead:** plain `module purge`, then **discover** — try the preferred pins, else
  `module -t avail Python/3.12 Python/3.11` and take the newest in range. The chosen name is
  written to `.python_module` beside the venv so activation reloads exactly the interpreter the
  venv was built on. The venv is also arch-suffixed (`.venv-$VSC_ARCH_LOCAL`), which the
  project's own `docs/VSC.md` had already required and the first version ignored.
- **Cheap check that would have caught it:** `module -t avail Python` on the node in question,
  before writing any name into a script.

### 12-08-2026 — Three things the first cluster attempt found in ten minutes
- **Tried:** the documented first-run sequence on wICE — `fetch_ood`, then the smoke test.
- **Result:** (1) `fetch_ood` died on the FIRST dataset with `FileNotFoundError` on
  `3.kr-vs-kp.npz.tmp`; (2) the smoke test died at optimizer construction with
  `ModuleNotFoundError: pytorch_optimizer`; (3) a notebook cell that ran locally would have
  crashed with `unhashable type: 'dict'`.
- **Why:** (1) `np.savez_compressed` given a PATH not ending in `.npz` silently **appends** the
  extension, so it wrote `...npz.tmp.npz` and the rename found nothing. (2) VSC runs torch 2.8
  — no `torch.optim.Muon` — and the published `tabicl` wheel does not ship its training
  package, so there was no Muon at all. (3) a `{{...}}` left over from a `.format()` template
  is **valid Python** (a set containing a dict) until it executes.
- **Instead:** write through an open handle; vendor upstream's Muon from the pinned dump
  (`src/train/_muon_vendored.py`) so no pip package is needed and it is the exact optimizer
  that trained the released checkpoints; and `test_every_notebook_cell_compiles` now compiles
  every cell. **The lesson that generalises: none of the three could fail locally.** Local had
  a newer torch, a warm cache and a different notebook on disk. Run the documented sequence on
  the cluster before believing it works.

### 11-08-2026 — Vendoring NanoTabICL instead of TabICL itself
- **Tried:** built the model from `NanoTabICL.txt` (665 lines) because it is small and readable,
  then tried to load the released TabICLv2 checkpoint into it for the warm-start experiment.
- **Result:** **zero of 390 parameter names matched** any of the checkpoint's 347, and our model
  had 44 extra LayerNorm bias tensors. `load_state_dict` loads nothing and does not complain.
- **Why:** NanoTabICL is a *reimplementation*, not the code that wrote the checkpoint. The bias
  gap is the `--norm_type layernorm_nobias` flag from upstream's own stage script, which a
  reimplementation without that switch cannot express.
- **Instead:** `tfm-library/repositories/TabICL.txt` is the **real** repository — 127 files,
  32,794 lines, including `_model/tabicl.py`, the whole `prior/`, **and** `train/_run.py`,
  `train/_muon.py` and `scripts/train_v2_reg_stage{1,2,3}.sh`. Use their model and loop; keep our
  prior as the thing that changes. Cheap check: count matching `state_dict` keys before writing
  any adapter — one line, and it would have ended this in a minute.

### 11-08-2026 — Running anything with the bare `python` on this machine
- **Tried:** `python -m src.utils.run_notebooks` from the repo root.
- **Result:** all 3 notebooks failed with `ModuleNotFoundError: No module named 'torch'`.
- **Why:** bare `python` on this machine is `C:\Python314\python.exe` — **Python 3.14**, outside
  the project's own `>=3.11,<3.13` pin, and with none of the dependencies. The project venv is
  `.CreditICL/`, which was never on PATH.
- **Instead:** always `.CreditICL/Scripts/python.exe`. Check `sys.version` before believing an
  import error is a code bug.

### 11-08-2026 — Every figure drawn at roughly twice the page width
- **Tried:** hard-coded `figsize=(12, 4.6)`, `(13, 3.8)`, `(11.5, 7.4)` in the plot modules, which
  look right on screen.
- **Result:** the A4 text block is **6.30 in**. Every figure would be scaled to ~50 % in the
  document, taking 9pt text to ~4.5pt — below the ~7pt floor for print.
- **Why:** a screen has no fixed width, so nothing pushes back until the figure is on a page.
- **Instead:** sizes come only from `style.figsize` / `grid_figsize` / `row_figsize`.
  `test_no_plot_function_hard_codes_a_figure_size` and `test_every_saved_pdf_fits_the_a4_text_block`
  now measure the PDFs themselves.

### 11-08-2026 — `results_dir()` on the wrong storage tier
- **Tried:** `results_dir()` returning `outputs_dir() / "results"`.
- **Result:** per-row predictions would have gone to `$VSC_DATA` (75 GiB) instead of project
  storage, filling it — and once it is full, every job that writes a log fails too.
- **Why:** `outputs_dir()` is correct for logs, manifests and figures, which are small. `results/`
  is the one part of `output/` that is not.
- **Instead:** `results_dir()` honours the staging root. `test_results_are_the_one_part_of_output_on_project_storage`
  asserts both halves, because moving *everything* to staging is the opposite mistake.

### 11-08-2026 — A test writing into the real `output/` tree
- **Tried:** isolating a figure test with `monkeypatch.setenv("CREDITICL_STAGING_ROOT", tmp_path)`.
- **Result:** two 1-byte fake PDFs left in `output/figures/`, which the A4 size check then read
  as real figures with no MediaBox. The failure surfaced three test files away from its cause.
- **Why:** `figures_dir()` hangs off `outputs_dir()`, which **ignores staging** — only `results/`
  follows it. Setting the staging root alone isolates a quarter of the tree.
- **Instead:** the `isolated_output` fixture sets `VSC_DATA` *and* the staging root. Use it for any
  test that writes.

### Earlier — SLURM dependency chain spanning three clusters
- **Tried:** one submit script chaining wICE → Genius → Mindwell → wICE.
- **Result:** the chain broke; the user's actual error was a Genius stage with **no `--partition`**.
- **Why:** VSC job dependencies **do not cross clusters**, and Genius needs an explicit partition.
- **Instead:** all of Phase 1 stays on wICE. A chain that spans clusters has to be resubmitted by
  hand at the boundary, so do not build one.

### Earlier — `sbatch --parsable` in a dependency expression
- **Tried:** `dep=$(sbatch --parsable job.slurm)`.
- **Result:** malformed dependencies — the value is `jobid;cluster`, not `jobid`.
- **Instead:** `sbatch --parsable "$@" | cut -d';' -f1` (`scripts/slurm/submit_pipeline.sh`).

### Earlier — `head -3` of a traceback in the env check
- **Tried:** one combined import in `_activate_env.sh`, printing `head -3` on failure.
- **Result:** the three lines printed are exactly the boilerplate; the **fourth** line is the
  `ModuleNotFoundError` naming the package. The first real cluster run was undiagnosable.
- **Instead:** test each package separately and name every missing one.

### Earlier — Bounding `atom_prob` by narrowing the sampling ranges
- **Tried:** tightening the lognormal parameter ranges so recovery would not exceed exposure.
- **Result:** `atom_prob=0.0` still produced 57 % boundary mass; 0.8 gave 93 %.
- **Why:** a lognormal has an **unbounded right tail**. Narrowing ranges cannot bound it — the
  tail is always clipped to the boundary, so mass appears at a boundary that was asked for to be
  empty.
- **Instead:** an explicit `interior_only` clamp (`src/prior/targets/mechanisms.py`). Now
  0.0 → 0 %, 0.6 → 58 %, 1.0 → 99 %.

### Earlier — Out-of-domain regression scored against a `[0,1]` clip
- **Tried:** running the OOD regression suite with raw targets.
- **Result:** a standard-normal target scored **R² = 0.34 on a perfectly linear relationship**.
- **Why:** the regression baselines clip to `[0,1]` because LGD is a loss fraction. An
  arbitrary-scale target is destroyed by that clip, for a reason unrelated to generality.
- **Instead:** min-max the target from **train-only** statistics (`src/eval/ood_runner.py`); R² is
  invariant to a consistent affine transform. Test statistics would leak the test distribution.

### Earlier — Collateral recovery capped before costs
- **Tried:** capping recovery at exposure, then subtracting workout costs.
- **Result:** LGD could never reach exactly 0, so the full-recovery atom was unreachable.
- **Why:** economically backwards — costs come out of the proceeds, and only then is the result
  capped at what is owed.
- **Instead:** costs first, cap second.
