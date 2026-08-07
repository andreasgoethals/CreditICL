# Changelog

One heading per day. Under it, what changed, why, and how — one or two lines each.
Newest first. Library pin recorded when a claim depends on the literature.

---

## 2026-08-07

- **Fixed: the SLURM chain could never have run.** It spanned three clusters (wice →
  genius → mindwell → wice) and Slurm job dependencies do not work across clusters. On
  top of that the genius stages had **no `--partition`**, which is the
  `"No partition specified or system default partition"` error seen on the first real
  submit. All of Phase 1 now runs on **wice**: `batch` for CPU, `gpu_a100` for training
  (18 cores/GPU, 72h). Mindwell's B200s stay for the Phase 2 multi-GPU runs, submitted
  separately rather than chained.
- **Fixed: `sbatch --parsable` returns `jobid;cluster` on a multi-cluster site**
  (`61683451;wice`), so every `--dependency=afterok:$jid` was malformed. Now stripped.
- **`DRY_RUN=1 bash scripts/submit_pipeline.sh lgd`** validates the whole chain with
  `sbatch --test-only` and queues nothing. A bad partition otherwise only surfaces after
  some stages are already queued, leaving a half-submitted chain to clean up.
- **Pipeline defaults caught up with the 400,000-dataset budget**: 100 shards per
  variant, not 20.
- **Added the missing piece that made the project unmeasurable**
  (`src/eval/crediticl_baseline.py`): nothing could load a checkpoint *we* trained, so
  the pipeline would have produced 48 sets of weights and zero results. In-context
  inference matching the training episode exactly; the model is rebuilt from the config
  stored inside the checkpoint, and a `load_state_dict` mismatch is fatal rather than a
  warning. LGD point predictions use the **median**, not the mean: on a bimodal
  predictive the mean lands in the empty middle where no loan sits.
- **`atom_prob` is honoured by the mechanism path** and set to `[0.6, 0.8]`, never 1.0 —
  `lgd_lendingclub` has 1.8% boundary mass, so a prior where every dataset has atoms
  cannot represent it. It was previously read only by the marginal path, making it an
  inert entry in the experiment grid. KNOWN LIMITATION: the interior branch still leaks,
  so 0.6/0.8 yields 87%/93% of datasets with atoms rather than 60%/80%.
- **Out-of-domain evaluation** (`src/eval/ood.py`, `ood_runner.py`, `scripts/fetch_ood.py`,
  `scripts/evaluate_ood.py`): OpenML-CC18 + CTR23, credit-like datasets filtered out,
  suite ids resolved from the API and pinned rather than hard-coded. Fetch runs on a
  login node; scoring never touches the network.
- **Fixed: OOD regression scores were crushed by a credit assumption.** The LGD
  baselines clip predictions to [0,1] because LGD is a loss fraction; a standard-normal
  OOD target then scored R²=0.34 on a perfectly linear relationship. Targets are now
  min-maxed using train-only statistics.

## 2026-08-06

- **Muon optimizer, matching TabICLv2** (`src/train/optim.py`). `torch.optim.Muon`
  ships in torch >= 2.9 (verified on 2.13), so no new dependency and no
  reimplementation. Muon only accepts 2-D parameters — it raises on anything else — so
  `_MuonWithAux` pairs it with AdamW for biases/LayerNorm/embeddings and presents one
  `Optimizer` to the loop, scheduler and checkpointer. `muon_lr: 8e-4` is TabICLv2's
  stage-1 rate; Muon wants roughly 8x AdamW's. Now the config default for both tasks.
- **Multi-GPU training** (`src/train/distributed.py`, `scripts/slurm/pretrain_multigpu.slurm`).
  A full-budget checkpoint is 24.5 GPU-days against VSC's 72-hour job ceiling; 4 GPUs
  turns that into 3 chained jobs, 8 into 2. The batch is *split* across ranks so the
  effective batch — and hence the compute budget — is unchanged, and each rank gets a
  distinct prior seed. Without that seed offset every GPU would generate identical
  datasets and an N-GPU run would see 1/N the data diversity while every log line still
  looked right. Launched with `torchrun`, not `srun`, per the VSC docs.
- **Credit mechanisms are now live** (`prior.credit.target.mode: mechanism`, the default
  for both tasks). LGD boundary mass emerges from collateral coverage, workout cashflows
  and portfolio segments; PD defaults come from the Merton/Vasicek one-factor model with
  Basel's 0.03–0.24 asset correlations. Measured span p1=0.013 to p99=0.896, which
  covers all 7 real LGD datasets (1.8% – 73%); the old marginal-shaping path bottomed
  out near 11% and could not reach `lgd_lendingclub`.
- **Training logs now carry progress and a wall-clock ETA**: percent complete, steps/s,
  elapsed, ETA, projected finish time, datasets seen — and a loud warning when the
  projected finish exceeds the *remaining* SLURM walltime, read from `squeue` rather
  than guessed. The rate is measured from the current run only; dividing total steps by
  elapsed time after a resume would over-report it several-fold.
- **Repo cleanup.** `res/` deleted for good — local outputs go to `results/_local/`, so
  there is exactly one place results live. `.ruff_cache/`, `.pytest_cache/` and
  `crediticl.egg-info/` deleted and gitignored. The venv is `.CreditICL/` (dotted);
  `.vscode/settings.json` points at it and hides the caches. `scripts/` now holds only
  runnables — the template's `example_script.py` moved to `docs/templates/`.
- **Fixed: my collateral economics were inverted.** Recovery was capped at the exposure
  *before* costs were subtracted, so an over-collateralised loan still lost the legal
  fees and LGD could never reach exactly 0 — the atom the whole mechanism exists to
  produce was absent in all 60 test draws. Costs now come out of the collateral
  proceeds, then the result is capped.
- **One prior notebook for any number of variants** (`src/visualize/pool_plots.py`).
  `prior_visualisation.ipynb` now *discovers* whichever pools are on the machine and
  compares them all on shared axes; adding `credit_v2` needs no notebook edit. Why not
  one notebook per prior: the useful question is never "what does `credit_v1` look
  like" but "how does it differ from `original` and `credit_v2`", and copies of a
  notebook drift and make the comparison manual. Plots split into **comparison** (all
  variants at once) and **detail** (one `FOCUS` variant, e.g. the 100-histogram grid,
  which cannot be stacked). Pooled episodes are rebuilt into `SyntheticTask`, so every
  existing `prior_plots` function works on them unchanged.
- **Downloading a pool is now cheap** (`scripts/fetch_prior_sample.sh`,
  `scripts/inspect_pools.py`). A full pool measures **4.0 GB (LGD) / 5.4 GB (PD) per
  variant** — ~19 GB for four. The notebooks use a few hundred datasets, so the default
  fetch takes **one shard** (~200–270 MB) per pool. `PoolReader` globs whatever shards
  exist, so a partial copy just works, and `describe_pools` labels it **SAMPLE** so it
  can never be quoted as the pool the model trained on.
- **Fixed: generated notebooks dropped their line endings.** nbformat keeps the
  trailing `\n` on every source line; without it, anything that rejoins the list
  (nbconvert, papermill, jupytext) gets one mashed line and a `SyntaxError`.
- **Fixed: no pools meant a cryptic crash.** An empty variant dict reached matplotlib
  as `subplots(0, n)` → "Number of rows must be a positive integer, not 0".
  `load_variants_or_generate` now falls back to live generation, labelled `(live)`, and
  the plot functions raise with the fix in the message.
- **Fixed: `CreditICL/` (a venv named after the project) was not gitignored** and not
  excluded from ruff — 340 MB would have been committed, and a bare `ruff check` walked
  its site-packages. Neither `.venv/` nor `venv-*/` catches a name like this.
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
