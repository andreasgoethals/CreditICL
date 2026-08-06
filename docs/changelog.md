# Changelog

One heading per day. Under it, what changed, why, and how — one or two lines each.
Newest first. Library pin recorded when a claim depends on the literature.

---

## 2026-08-06

- **Pre-generated prior pools** (`src/prior/pool.py`, `scripts/generate_prior.py`).
  Datasets are now generated once into one folder per variant — `lgd__original/`,
  `lgd__credit_v1/` — and training samples from them (`prior.pool.source: pool`).
  Why: every arm then draws its original-prior share from *the same files*, so an
  arm-to-arm difference cannot be luck of the draw; generation is CPU-bound, so it
  moves off the GPU onto cheap CPU nodes; and resume becomes exact rather than
  reproducible-from-seed. `verify_pools` fails the run if the pools hold different
  counts, because "matched compute" would otherwise be an assumption.
- **The whole pipeline is one command** (`scripts/submit_pipeline.sh`). Five stages
  chained with `--dependency=afterok`: preprocess → 2×20-task CPU prior arrays →
  verify gate → GPU training array → evaluation. The two variants are submitted as
  separate arrays so 40 tasks are eligible at once, and the gate means a single dead
  shard stops the chain instead of silently shrinking a pool.
- **Shared figure style** (`src/visualize/style.py`). One palette and typography for
  every figure, with `CREDIT`/`ORIGINAL` meaning the same thing everywhere. Figures
  end up in the thesis; styling each notebook separately means redoing them all later.
- **Data exploration notebook** + `src/visualize/data_plots.py`. Measures what the
  prior is *for*: LGD boundary mass spans 1.8% (lendingclub) to 73% (heloc), PD base
  rates 6.7%–40% — all below the original prior's balance point. Includes a
  single-feature leakage screen; on `lgd_lendingclub` the top correlate is
  `months_since_origination` at 0.75, so its high R² is not obvious label leakage.
- **Fixed: `ensure_processed` iterated a bare slug character by character.** Passing
  `"0011.loan_default"` instead of a list preprocessed datasets named `0`, `1`, `.`,
  `l`, `o`… and reported 17 failures for one request. Now accepts the singular form.
- **Fixed: `target_stats` rejected numpy arrays.** The prior path passes tensors, the
  data path passes numpy; requiring the caller to remember which broke the
  exploration notebook on its first run.
- **Fixed: `$CREDITICL_STAGING_ROOT` was ignored off-VSC.** `prior_cache_dir` and
  `processed_write_dir` short-circuited to the repo unless `$VSC_DATA` was set, so
  the override silently did nothing on a laptop — and the pool tests wrote 40,000-file
  directories into the repo. Now honoured everywhere.
- **Data pipeline** (`src/data/`). Turns the 21 raw credit files into a cache.
  Per-dataset recipes copied from the sibling TabPFNCredit project — they encode
  which columns are IDs, leak the target, or need transforms, and that knowledge is
  not recoverable from the CSVs. All 21 preprocess successfully.
- **Processed format is one parquet file per dataset**, not separate arrays. On Home
  Credit that is 17 MB against 149 MB for `X.npy` + `y.npy`, in 2 files instead of
  4 (project storage has a low inode budget), and it keeps dtypes and column names,
  which `.npy` cannot. Reads slower (1.0s vs 0.06s) but a CatBoost fit takes
  minutes, so it does not matter.
- **Eval pipeline** (`src/eval/`). Ridge/logistic, CatBoost, TabPFN-3, TabICLv2 on
  all 21 datasets. LGD reports pinball, CRPS, coverage and boundary-mass
  calibration; PD reports PR-AUC, Brier, ECE, recall-at-top-k, with accuracy shown
  next to the majority baseline so it cannot mislead.
- **TabPFN-3 loads from a local checkpoint.** Letting `tabpfn` fetch its own weights
  triggers a licence flow that wants an API token, and with no terminal to prompt on
  that surfaced as `OSError: WinError 10038 ... not a socket`. `model_path` accepts a
  file, so pointing it at `checkpoints/` avoids the token, the download, and the
  need for internet on a compute node.
- **Row and feature caps for the in-context models**, always recorded in the result
  row. `algorithmwatch` has 2,986 features and Home Credit 307k rows; both are past
  what TabPFN and TabICL accept. Feature choice is by training-set variance —
  crude on purpose, since anything target-aware would leak the target.
- **Groups and subgroups** (`src/prior/grouping.py`), TabICL's correlated
  hyperparameter sampling, now **on by default**. Previously absent because
  NanoTabICL removed it. That was the wrong default: `credit_fraction=0.0` is the
  control arm and the control is supposed to *be* TabICL, so having grouping makes
  it more faithful, not less. `group_size: 1` recovers the old behaviour.
- **`results/{lgd,pd}/{data,prior,training,eval}/`** for official outputs. `logs/`
  holds logs only, and no longer writes files at all when running locally.
- Removed `lib/` and `res/`; `data/processed/` and `results/` replace them.
- Notebooks: `prior_visualisation` and `data_exploration`, with all logic in
  `src/visualize/`.
- LICENSE rewritten to say what the MIT grant does **not** cover — the datasets,
  the vendored code, and TabPFN's licence-gated weights.
- **Corrected an earlier claim of mine.** I said the bimodal LGD premise held for
  one of three datasets. With all seven preprocessed it holds for **four of seven**,
  and `heloc` has **73%** of its mass on the boundaries. Stronger premise than I
  reported. PD base rates run 6.7%–40%, not 6%–22%.
- **First eval result, and it is the motivating one.** On `heloc`, CatBoost reaches
  R² 0.50 while predicting only 4.7% of cases at full recovery when 21.1% actually
  are — a boundary-mass error of 0.68. Respectable R², almost total failure to
  reproduce the structure that defines LGD.
- **Flagged for checking:** `lgd_lendingclub` scores R² 0.71–0.76, far above the
  published LGD range. Smells like a feature derived from the recovery amount
  surviving the recipe.

## 2026-08-05

- **Prior generator** (`src/prior/`). Reproduces NanoTabICL's TabICLv2 prior (eight
  random function families, Cauchy DAG, categorical converters) with an explicit
  RNG object so the task stream survives checkpoint/resume. Adds the credit paths:
  LGD's bounded [0,1] target with censored boundary atoms, PD's base-rate control,
  signal dilution, threshold rules, underwriting selection and MNAR missingness,
  plus deliberately irrelevant columns.
- **The mixture lever** `credit_fraction` — the share of datasets from our path
  rather than the original prior. Defaults sweep 0.0 / 0.1 / 0.2 / 0.3, i.e. 70–90%
  original plus a control arm.
- **Training loop** (`src/train/`), following `tabicl/train/_run.py`. Pinball and
  cross-entropy transcribed exactly, including the `linspace(0, 1, Q+2)[1:-1]`
  quantile grid. Checkpoint, resume and self-resubmission, because the VSC docs
  contain no Slurm requeue facility and GPU walltime caps at 72 h.
- **Fine-tuning research** → `docs/finetuning.md`. TabICL ships three freeze
  switches; its v1 stage 3 froze col+row at lr 2e-6, its v2 stage 3 froze nothing at
  lr 2e-5. No LoRA anywhere in TabICL, and Tanna 2025 found it unstable on TabPFN,
  so we do not add it. Tanna 2026: full fine-tuning drops TabICL from 0.873 to
  0.567 on TabZilla — a direct argument for keeping most of the prior original.
- **Storage split**, following CreditPFN: big files (datasets, checkpoints,
  generated pools) to project storage `/lustre1/project/stg_00211/CreditICL/`; small
  durable files (logs, metrics, result CSVs) and the repo to `$VSC_DATA`. Chosen
  automatically by whether `$VSC_DATA` exists.
- **THE METRIC BUG.** Boundary mass was computed as `(y <= 0).mean()`, which on a
  standard-scaled target just means "below the mean" and returns ~0.5. It made the
  unmodified prior look like it already put 54% of its mass at zero, which would
  have destroyed the project's motivation. Replaced with a scale-invariant
  definition in `src/utils/target_stats.py`.
- **First measured comparison of the two priors.** LGD: the original prior puts its
  target in [0,1] in **0%** of tasks and has an atom in 26%; ours is 100% bounded
  with an atom in 94%, and its boundary mass (0.111 / 0.100) lands near Freddie's
  (0.114 / 0.081). So "structurally absent" was too strong — the original prior does
  make atoms — but the support gap is total. PD: the original prior's base rate is
  centred on **0.500** with only 5% of tasks under 10% positives; ours spans
  0.03–0.23, bracketing the real data.
- **Verified it trains.** LGD pinball 0.130 → 0.072 over 250 steps, monotone. PD
  cross-entropy 0.458 → 0.308. Resume restores step count, weights, optimizer and
  LR exactly. Same seed in two processes gives identical results.
- **Smoke test now uses the GPU, AMP and workers** when available. It previously
  forced CPU with AMP off, so the first GPU step ever taken would have been inside
  the paid array.
- Other bugs found by running things: `filter mode="off"` accepted constant targets;
  the log file handle was never closed; `metrics: [...]` in a config was read as a
  sweep and would have split one run into six; PD's row-dropping could leave a
  ragged batch.
- Scaffolding: `AGENTS.md` (the `tfm-library/` read-only rule), `README.md`,
  `pyproject.toml` (Python 3.11–3.12, matching VSC's modules), `.gitignore` (the
  repo had none), `.gitattributes` (LF endings, or `.slurm` files fail on Linux),
  `.vscode/`, `docs/vsc.md`, `docs/experimental_design.md`, and
  `tfm-library/PROJECT_SPECIFIC.md`.
- **Two corrections to the project's framing**, both from reading the source.
  TabICL's target standardisation is affine and therefore shape-preserving, so
  bounded/bimodal targets are a matter of frequency and support alignment, not
  structural absence. And `_prior_config.py` sets `"balanced": False`, so prior-side
  imbalance is uncontrolled rather than missing.
