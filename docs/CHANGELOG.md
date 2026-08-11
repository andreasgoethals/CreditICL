# Changelog

One chapter per date, `DD-MM-YYYY`, newest first. Terse: what changed, and why if the
reason is not obvious.

---

## 09-08-2026

- **One output root.** Everything generated goes under `output/` — results, figures,
  logs, manifests — locally and on the cluster. Was spread over four top-level folders.
- **One shared `output/figures/CAPTIONS.md`**, grouped per notebook, figures in notebook
  order. Every figure written as PDF (300 dpi, print) and PNG (110 dpi, committable).
- **Captions rewritten as pure description.** No interpretation; a test rejects
  interpretive phrasing.
- **`output/All_Results.md`** collects every notebook's printed summary, alphabetically.
- **Config: a top-level `sweep:` block** holds every multi-value knob, so the experiment
  design is four lines instead of scattered brackets. One short comment per knob.
- **Fixed `atom_prob`.** It claimed to set the share of datasets with boundary atoms but
  leaked: 0 gave 57%, 0.6/0.8 gave 87%/93%. Cause: `lgd_collateral`'s coverage is
  lognormal and its right tail always pushed rows past full recovery, clipping to exactly
  0. Narrowing ranges could not fix an unbounded tail; interior draws now clamp net
  recovery explicitly. Now 0.0→0%, 0.6→58%, 0.8→79%.
- **Fixed: notebooks could not load a config.** Relative paths broke because Jupyter
  starts in `notebooks/`; they now fall back to the repo root.
- **Fixed: `figures/` in `.gitignore` matched at any depth**, so it also ignored
  `output/figures/`. Anchored with a leading slash.
- **Fixed: every figure was saved twice** — `plt.subplots` calls `plt.figure`, so both
  capture hooks fired.
- `docs/` files renamed to CAPITALS. `run_notebooks` moved to `src/utils/`. Transfer
  helpers grouped under `scripts/transfer/`.
- **`docs/TEMPLATE.md`** rewritten as a generic, reusable spec.

## 08-08-2026

- **Fixed the bug that made the first cluster run undiagnosable.** `_activate_env.sh`
  printed `head -3` of the traceback — exactly the three boilerplate lines, cutting off
  the `ModuleNotFoundError` that names the package. Now tests each package separately,
  lists every missing one, and checks `import src`.
- **Fixed: `fetch_ood.py` went silent for minutes** while downloading. Logs every attempt.
- **Shift stress** (`src/prior/shift.py`). O'Prior's ablations report three independent
  contributors: mechanism diversity, realism, and shift-aware stress. We had the first
  two. Three kinds — `cohort`, `covariate`, `prior_prob` — on 30% of our datasets only.
- **Training progress curve** (`src/train/progress.py`). One CSV row per 10,000 synthetic
  datasets with metrics on real and out-of-domain data, so a run is a curve rather than
  one end-of-run number. Never fatal.
- **Run cleanup** (`src/utils/run_artifacts.py`, `scripts/clean_run.py`). Lists by
  default; expensive categories need naming explicitly; raw data and downloaded weights
  are unreachable by construction.
- **Verified:** our ExtraTrees filter matches TabICLv2 Appendix E.14 exactly.

## 07-08-2026

- **Fixed: the SLURM chain could never have run.** It spanned three clusters and Slurm
  dependencies do not cross clusters; the genius stages also had no `--partition`. All of
  Phase 1 now runs on wICE.
- **Fixed: `sbatch --parsable` returns `jobid;cluster`** on a multi-cluster site, so every
  dependency string was malformed.
- **`DRY_RUN=1`** validates the whole chain with `sbatch --test-only` and queues nothing.
- **Added the piece that made the project unmeasurable** (`src/eval/crediticl_baseline.py`):
  nothing could load a checkpoint *we* trained. LGD point predictions use the median, not
  the mean — on a bimodal predictive the mean lands where no loan sits.
- **Out-of-domain evaluation** (`src/eval/ood.py`, `ood_runner.py`): OpenML-CC18 + CTR23,
  credit-like datasets filtered out, suite IDs resolved from the API rather than
  hard-coded.
- **Fixed: OOD regression scores were crushed by a credit assumption.** The LGD baselines
  clip to [0,1]; a standard-normal target scored R²=0.34 on a linear relationship.

## 06-08-2026

- **Muon optimizer**, matching TabICLv2. `torch.optim.Muon` ships in torch ≥ 2.9, so no
  new dependency. Muon takes only 2-D parameters, so `_MuonWithAux` pairs it with AdamW
  for the rest and presents one optimizer.
- **Multi-GPU** (`src/train/distributed.py`). A full-budget checkpoint is 24.5 GPU-days
  against a 72-hour ceiling. The batch is split so the effective budget is unchanged, and
  each rank gets a distinct prior seed — without that, every GPU generates identical data.
- **Credit mechanisms live** (`prior.credit.target.mode: mechanism`). LGD boundary mass
  emerges from collateral coverage and workout cashflows; PD defaults come from the
  Merton/Vasicek one-factor model with Basel's 0.03–0.24 asset correlations.
- **Fixed: the collateral economics were inverted.** Recovery was capped at the exposure
  *before* costs, so an over-collateralised loan still lost the fees and LGD could never
  reach exactly 0 — the atom the mechanism exists to produce.
- **Pre-generated prior pools** (`src/prior/pool.py`): one folder per variant, so every arm
  draws its original share from the same files. `verify_pools` fails the run if counts
  differ.
- **The whole pipeline is one command** (`scripts/submit_pipeline.sh`), five stages chained
  with `--dependency=afterok`.
- **Data pipeline** (`src/data/`), 21 datasets to a parquet cache, recipes copied from
  TabPFNCredit. **Eval pipeline** (`src/eval/`) with ridge/logistic, CatBoost, TabPFN-3
  and TabICLv2.
- **Fixed: the metric bug that inverted the project's motivation.** `(y <= 0).mean()` on a
  standard-scaled target returns ~0.5, which made the original prior look like it had 54%
  mass at zero.
- Two notebooks with all logic in `src/visualize/`, plus a shared figure style.

## 05-08-2026

- Project scaffolding: repo layout, `pyproject.toml`, config expansion with the `_range`
  suffix rule, path resolver for the two VSC storage tiers, logging setup.
- Literature review against the pinned `tfm-library/`. Three claims in the original brief
  were weakened after reading the sources — see `docs/EXPERIMENTAL_DESIGN.md`
  §"Verified premises".
- Prior generator ported from TabICL: Cauchy random DAGs, eight function families,
  categorical converters, ExtraTrees predictability filter.
- LGD and PD target families added, plus the mixture lever `credit_fraction`.
