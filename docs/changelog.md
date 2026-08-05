# Changelog

Dated log of substantive changes — what changed and why, one line each.
Mirrors the convention in `tfm-library/CHANGELOG.md`. Newest first.

Every entry that depends on the literature should record the `tfm-library`
pin it was written against.

---

## 2026-08-05 (evening) — one config per experiment, logging, tests

### Config layout flattened, by request

`config/` now holds exactly two files — **`LGD.yaml`** and **`PD.yaml`** — and no
subfolders. Each one configures everything about its experiment: prior, mixture
lever, how to start (scratch / fine-tune), model, training, logging, and the
evaluation plan. Removed `config/priors|train|eval/` and the template's leftover
`config/data/ihdp_s_1.yaml` + `config/methods/mlp.yaml` samples (they were from
the template's causal-inference example and recoverable from git history).

The mixture lever now defaults to `[0.0, 0.1, 0.2, 0.3]` — i.e. **70–90% of
datasets from the original prior**, plus `0.0` as the control. Both configs expand
to **48 runs** (`--array=0-47`).

### Logging

`src/utils/logging_setup.py`. Every run writes two files into `logs/`:

- `<run>_<timestamp>.log` — timestamped human-readable lines: environment (host,
  SLURM ids, GPU, torch/CUDA versions), the levers in play, `credit_fraction`,
  where outputs and checkpoints went, the budget, then per-interval progress with
  loss, learning rate, steps/second and an ETA. Flushed on every record, so a job
  killed at the walltime limit keeps the lines explaining why.
- `<run>_<timestamp>.metrics.jsonl` — one JSON object per line, for plotting.

Both land on `$VSC_DATA`, never on staging (low inode budget).

Added a **mid-run prior report** (`log_prior_every`): every N steps it samples a
few datasets and logs the realised boundary mass (LGD) or base rate (PD), the
base/credit split, and the filter's rejection rate. This is cheap insurance — a
config typo that silently switches the credit path off looks exactly like a real
null result in the loss curve.

### Tests

New `tests/` folder, **40 tests passing** without torch installed (the four
torch-dependent modules skip cleanly via a per-module `importorskip`, rather than
a conftest-level skip that would have aborted the whole session).

Coverage: config grid expansion and its `_range` rules; storage-tier resolution;
the LGD target family (bounded, exact boundary mass, monotone in the latent, and
that it really covers U-shaped / one-sided / interior); the PD family (base rate,
signal dilution, asymmetric flipping, rules, selection, missingness); the mixture
lever; the predictability filter in all three modes; noise columns; the pinball
loss against the exact TabICL quantile grid; the freeze strategies; and
checkpoint save/prune/resume.

### Bugs the tests caught immediately

- **`metrics: [pinball, crps, ...]` was being read as a sweep**, which would have
  silently turned one run into six, each computing a single metric. Same for
  `dev_datasets: []`, which raised instead. Fixed by extending the literal-list
  set and giving the empty-list error an actionable message.
- **The documented nested-range escape hatch did not work**: a `*_range` key
  holding a list of ranges was treated as literal instead of swept.
- Two prior bugs were fixed earlier the same day; see the entry below.

### Note on the failed clone

The `pip install -e ".[dev]"` on VSC failed with *"neither setup.py nor
pyproject.toml found"* for a mundane reason: **none of this work had been
committed or pushed**, so the clone only contained the original 45 files. Nothing
wrong with the code.

Also worth noting: the VSC clone checked out the submodule at `4679bde`, while
this working copy is at `21d555a`. Pushing will move the recorded pointer;
`git submodule update --init --recursive` on VSC syncs it.

---

## 2026-08-05 (later) — the pipeline: prior, pretraining, SLURM

Library pin: **`21d555a`**. Nothing installed; nothing run on a GPU.

### The three phases now exist as code

- **`src/prior/`** — the synthetic task generator.
  - `functions.py`, `base.py`, `rng.py` reproduce NanoTabICL's TabICLv2 prior
    (the eight random function families, Cauchy DAG, categorical converters).
    Everything takes an explicit `PriorRNG` instead of global RNG state, so the
    task stream survives checkpoint/resume.
  - `preprocess.py` — TabICL's `outlier_removing` / `standard_scaling`, plus a
    rank transform.
  - `filters.py` — the ExtraTrees predictability filter in three modes:
    `tabicl` (as shipped), `off`, `banded` (target credit's low-signal range).
  - `targets/lgd.py` — the bounded [0,1] target family. Rank-transform, then a
    Kumaraswamy inverse CDF for interior shape, then boundary atoms by
    censoring. **Every credit target is hard-clipped to [0,1].**
  - `targets/pd.py` — base-rate control, signal dilution, asymmetric label
    noise, threshold rules, underwriting selection, MNAR missingness.
  - `noise_features.py` — irrelevant columns on purpose: `pure_noise`,
    `duplicate`, `shuffled`, `constant`.
  - `generator.py` — **the mixture lever**: `credit_fraction` is the probability
    a dataset comes from our path instead of the original prior.
  - `dataset.py` — infinite `IterableDataset` yielding whole batches.
- **`src/train/`** — `loop.py` (pinball / cross-entropy exactly as TabICL's
  `run_micro_batch`), `optim.py` (TabICL's cosine-with-restarts, plus the
  constant schedule v1 stage 3 uses), `checkpoint.py`, `adapt.py`.
- **`src/models/nanotabiclv2.py`** — extracted from the pinned library dump by
  `scripts/vendor_model.py` rather than committed by hand, so provenance is
  unambiguous and it cannot drift from the pin.
- **`src/utils/config.py`** — every lever may be a scalar or a list; lists are
  crossed; one grid point = one SLURM array task.
- **`src/utils/paths.py`** — the two storage tiers (see below).

### Research: what fine-tuning TabICL allows → `docs/finetuning.md`

Read out of TabICL's own code and scripts:

- It ships **three freeze switches** (`freeze_col`, `freeze_row`, `freeze_icl`)
  in `_finetune/base.py`, matching its three stages.
- **v1 stage 3** used `--freeze_col True --freeze_row True`, `lr 2e-6`, constant
  schedule, clip 1.0.
- **v2 stage 3** uses **no freezing** — full training at `lr 2e-5`, cosine,
  clip 1.0. So the v2 authors moved away from freezing.
- **There is no LoRA anywhere in TabICL** (zero matches in the dump), and Tanna
  2025 found LoRA unstable on TabPFN. We do not add it.
- Tanna 2026: full fine-tuning is near-catastrophic for TabICL (TabZilla
  0.873 → 0.567), which is a direct argument for keeping most of the prior
  original.

Implemented as `init.strategy` ∈ {`scratch`, `full`, `icl_only`, `head_only`}.
One refinement over a naive freeze: the target enters the model **twice**
(`y_embed_in` before the column blocks, `y_embed_icl` before the ICL blocks), so
freezing covers the *block stacks* and always leaves both y-embeddings and the
head trainable — otherwise a target-shape change could not be learned.

### Storage now follows CreditPFN

Read from `3. CreditPFN/CreditPFN/docs/VSC_GUIDE.md` and its SLURM scripts:

- **Big files → project staging** `/lustre1/project/stg_00211`: datasets,
  checkpoints, result CSVs.
- **Small durable files → `$VSC_DATA`**: the repo, logs, `metrics.jsonl`,
  resolved configs. 75 GiB and backed up.
- Account **`lp_verbekelab`**; training on Mindwell `gpu_b200`.
- **Correction to `docs/vsc.md`:** there is **no `gpu_*_long` partition on wICE
  or Mindwell** — 72 h is a hard GPU ceiling; only CPU partitions have 7-day
  variants.
- Env activation copies CreditPFN's `_activate_env.sh`, including its guard
  against an active virtualenv silently shadowing conda.

### Bugs found and fixed while tracing the code

- **Config grid:** a hand-curated `NO_EXPAND` list leaked `n_nodes_range` and
  `rule_quantile_range` into the sweep, turning two sampling intervals into
  two-point grids and inflating the PD grid to 6,144 runs. Replaced with a
  suffix rule: any key ending in `_range` is literal data.
- **Ragged batches:** PD's underwriting selection drops rows, and rounding could
  leave a task one row short, which would crash `torch.stack`. Fixed by
  oversampling with a margin *and* filling short tasks by repeating rows —
  not by zero-padding, which would have invented extra class-0 labels and
  shifted the base rate the experiment is trying to control.
- **Paths off-VSC** produced `CreditICL/CreditICL/output`. Fixed.

### Still unverified

- **None of the tensor code has been executed.** torch is not installed here and
  installing was declined, so the prior, the model and the training loop are
  syntax-checked only. `scripts/smoke_test.py` and
  `scripts/slurm/smoke_test.slurm` exist to run this free on the `interactive`
  partition before any real submission.
- Whether the released TabICLv2 checkpoints load into NanoTabICL. `adapt.py`
  refuses rather than silently training a random model.

---

## 2026-08-05 — project scaffolding and design proposal

Library pin: **`21d555a`** (2026-08-05, "readme update").

### Added

- **`AGENTS.md`** — repo contract for AI agents. States the `tfm-library/`
  read-only rule and its single exception (`PROJECT_SPECIFIC.md`) so the next
  agent inherits it, plus the cite-by-symbol-name rule, the
  no-data-in-git rule, and the PowerShell 5.1 / no-`&&` constraint.
- **`README.md`** — replaces the faculty template's own README. Positions the
  research question honestly against O'Prior, summarises the design, and
  documents the layout and setup. The original template README is preserved
  verbatim at `docs/template_readme.md` rather than deleted, so the group's
  folder-structure guidance is not lost.
- **`pyproject.toml`** — TabICL stack (torch/numpy/scipy/scikit-learn/pandas),
  `dev` extra (pytest, ruff), opt-in `tabicl` and `track` extras. TabICL is
  deliberately **not** a default dependency — we modify its prior, so a PyPI
  wheel would be a read-only copy of exactly the code we need to change.

  **Deviation from the brief:** it asked for Python 3.11; the pin is
  `>=3.11,<3.13`. Reason — **3.11 is not installed on the dev machine**
  (`py --list` gives 3.14.0, 3.13.10, 3.12, 3.10.11) and bare `python`
  resolves to **3.14**, outside any supported range. VSC documents three
  Python modules — `Python/3.11.3-GCCcore-12.3.0` (2023a),
  `3.12.3-GCCcore-13.3.0` (2024a), `3.13.1-GCCcore-14.2.0` (2025a) — so
  **3.12 gives local↔cluster parity without installing a new interpreter**,
  while 3.11 remains supported for anyone who wants to match 2023a exactly.
  3.13 is excluded: torch wheel coverage there is newer than is comfortable
  under a load-bearing prior generator. Nothing was installed to reach this
  conclusion.
- **`.gitignore`** — the repo had none, which is why `data/` and
  `checkpoints/` showed as untracked. Both are now ignored: datasets carry
  licences that are not ours to redistribute, and checkpoints are large and
  regenerable.
- **`.vscode/settings.json`, `.vscode/extensions.json`** — auto-selects and
  auto-activates `.venv`; marks `tfm-library/**` read-only in the editor
  (with `PROJECT_SPECIFIC.md` excepted) and excludes it from search and
  file-watching; forces LF line endings on `*.slurm` so `sbatch` does not
  choke on the shebang.
- **`docs/vsc.md`** — distilled KU Leuven VSC deployment facts.
- **`docs/experimental_design.md`** — the proposed design.
- **`docs/changelog.md`** — this file.
- **`tfm-library/PROJECT_SPECIFIC.md`** — filled in from the template
  (gitignored by the library; the only file we may create inside that
  submodule).
- Directory skeleton: `docs/ res/ assets/ lib/ scripts/slurm/`,
  `src/{prior,train,eval}/`, `config/{priors,train,eval}/`, plus `.gitkeep`
  files so gitignored template folders survive a fresh clone.

### Verified against source — two corrections to the project's framing

Both would have been caught by a reviewer who read the code, so they are
recorded here as findings, not edits.

- **Bounded/bimodal targets are a *frequency* gap, not a structural absence.**
  `GraphSCM.__call__` does apply `outlier_removing` then `standard_scaling`
  to regression targets — but standardisation is **affine and therefore
  shape-preserving**, so bimodality and atoms survive it; only the [0,1]
  *support* is destroyed. Atom-producing mechanisms already exist in the prior
  (`outlier_removing` clamps; `rand_tree_func` and
  `rand_discretization_func` are piecewise constant; NanoTabICL's
  `rand_kumaraswamy_act` maps to exactly [0,1] at p≈0.5). The defensible claim
  is about frequency and output-support alignment.
- **TabICL's prior is not class-balanced by default.** `_prior_config.py` sets
  `"balanced": False`, so `MulticlassAssigner` is active, and in `mode="rank"`
  it cuts at a **uniformly random data row** — making the binary minority rate
  roughly Uniform(0,1), so ~10% of binary tasks already sit below 5% minority.
  Prior-side imbalance in TabICL is **uncontrolled and unmeasured, not
  missing**. O'Prior is the opposite and actively rejects *"collapsed or
  severely imbalanced support classes"*.

### Verified as stated

- O'Prior is **classification only** (ROC-AUC / accuracy / macro-F1 on 52
  tasks); its regression target-reshaping branch is never exercised.
- O'Prior's protocol: nanoTabPFN, 40,000 synthetic datasets per prior,
  512–1,024 rows, 3–50 features, batch 4, 1,000 steps/epoch, 10 epochs, no
  tuning between conditions.
- TabICLv2's regressor: 999-quantile pinball loss, `max_classes=0`, linear
  target embeddings, bias-free LayerNorm.
- The predictability filter: `ExtraTreesRegressor(n_estimators=25,
  bootstrap=True, oob_score=True, max_depth=6)`, 200-resample bootstrap,
  reject if `pval >= 0.05`; plus separate triviality and graph-ancestor
  filters. Rate confirmed **from the paper**: *"roughly 35% classification and
  25% regression datasets are filtered"* in stage 1.

### Measured from `data/raw/`

- LGD boundary mass (exactly 0 / exactly 1): `0006.lgd_freddie` **11.4% /
  8.1%** and genuinely U-shaped; `0007.lgd_lendingclub` **1.5% / 0.3%**,
  unimodal left-skewed peaking at 0.80–0.85; `0003.axa` **0% / 0%**, fully
  interior. **The "bimodal with atoms" premise holds for one of three
  datasets**, so the LGD prior must sample a *family* over boundary mass.
- PD base rates: GMSC **6.7%**, Home Credit **8.1%**, HMEQ **20.0%**, Taiwan
  **22.1%** — i.e. 6–22%, not the 1–5% the brief assumed. 4 of 14 checked.

### Notable VSC findings

- **B200 (437.50 credits/GPU-min) is cheaper than H100 (569.444)** and gives
  more CPU cores per GPU (24 vs 16). A100 is 141.667, ~4× cheaper per hour.
  Since TabICL's prior generator is CPU-bound, cores-per-GPU matters as much
  as GPU speed.
- The `interactive` partition is **free** (8 cores, 16 h; a full RTX 5000 Ada
  on Mindwell).
- **`requeue` appears zero times in the VSC documentation**; the only
  documented checkpointing facility is the Torque-era `csub`/BLCR framework.
  With a 72 h walltime ceiling, application-level checkpoint/resume and
  self-resubmission are mandatory.

### Not verified

- **`arXiv 2605.21742`** (inference-time PFN imbalance correction) is **not in
  the library** — zero hits. Nothing about it has been confirmed.
- **O'Prior's implementation** (`github.com/o-prior/O-prior`) is not in the
  library and has not been read. Its paper text names *"bounded
  reparameterizations"* and *"censoring-style transformations"* among its
  regression target transforms, so bounded and atom-ful targets are inside its
  **design space** even though outside its **evaluation**. Reading that code
  is the highest-priority open item, because arm B depends on it.

### Notes

- No errors found in the library to report upstream.
- Nothing was written inside `tfm-library/` except `PROJECT_SPECIFIC.md`.
  Verified afterwards that the submodule's `git status` is **clean**, that
  `PROJECT_SPECIFIC.md` is matched by the library's own `.gitignore` line 10,
  and that its HEAD is unchanged at `21d555a`.
- **Removed the `assets/img/icon_code.png` image tag** from the top of
  `README.md`. The template referenced it but the file was never present in
  this project (`assets/` did not exist), so it rendered as a broken image.
  Restore the tag if you copy the icon over from the template repo.
- Nothing was installed and no training was run.
