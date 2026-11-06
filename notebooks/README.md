# The notebook story

Local inputs are results, logs and metadata. Model checkpoints and optimizer states remain on
VSC; these notebooks do not need them. Use the analysis-only inventory filter described in the
repository README instead of downloading whole output trees or model-containing archives.

Twelve thin notebooks in four numbered chapters follow the paper's questions.
All computation lives in `src/visualize/`; each notebook displays and saves its
own A4 vector PDFs, records descriptive captions, and ends with a printed summary
in section order. `output CreditICL/All_Results.md` collects those summaries verbatim.

| Chapter | Notebooks | Question |
|---|---|---|
| [0. General](<0. General/>) | 0.1 benchmark tables; 0.2 PD prior; 0.3 LGD prior | What changes in exposure, difficulty and observation, and what remains mismatched? |
| [1. Experiment 1](<1. Experiment 1/>) | 1.1/1.2 training; 1.3/1.4 final results | At equal synthetic-task budgets, do the 25%, 50% and 100% packages improve on 0%? What do component removals change? |
| [2. Experiment 2](<2. Experiment 2/>) | 2.1/2.2 adaptation; 2.3/2.4 final results | With fixed recipes, what is the effect within each backbone, against its trained control and released weights? |
| [3. Experiment 3](<3. Experiment 3/>) | 3.1 retention | Which constrained updates preserve OOD performance while adapting to credit? |

## Evidence boundaries

The binary illustration forces two classes and disables statistical filtering.
A separate small diagnostic uses the production slot planner, native 2–10-class
generic route, and actual validity/filtering rules. Neither is a retained training
corpus or a transfer result. Raw economic LGD masses and standardized endpoint
ties have separate interpretations. Standardization preserves atoms.

Coverage counts describe local files, not the current scheduler. A partial trace
may be stopped; absent output may still be queued. Imported snapshot metadata,
when available, appears beside compact coverage/health tables. The dated cluster record remains in
`docs/RUNS.md`. No training curve is fabricated from that summary.

Final panels use complete current-protocol benchmark loaders: all eligible rows,
all prediction-time-eligible features, and five common folds. Missing evidence
produces one availability notice instead of a gallery of empty plots. The final
credit/OOD frontier is not populated from short-context training monitors.

Experiment 2 retains four mixtures and both peak rates per backbone, with separate plots
for each backbone/rate and full-table checkpoint trajectories. Experiment 3 fixes the 50%
mixture and predeclared full-model/adapter settings, and reuses both fixed-rate Exp2 baselines.
Both experiments use two training seeds and retain 125/250/500/1,000/1,500 updates; paired-retention and checkpoint panels stay
empty until the required evidence exists. A trajectory requires at least two measured checkpoints
within a fixed recipe; a completed final endpoint alone is not a curve. The designated holdout has historical project
exposure; it is **not untouched confirmation**. No plot or notebook chooses a setting.

Paired differences average folds and training seeds within each dataset before
descriptive bootstrap intervals are computed. Seed marks display training spread;
seeds and folds do not inflate the dataset count. Native binary/multiclass OOD
subsets are diagnostics alongside the fixed aggregate. LGD endpoint plots report
aggregate mass matching at tolerance 1e-4 using 99 predictive quantiles, not
conditional calibration.

Checkpoint panels break lines at missing milestones and distinguish treatments by color,
shape and line pattern. Seed ranges are descriptive. Native-regression OOD CRPS is retained
per dataset rather than averaged across different target units; OOD trajectories use R².

## Figure selection

| Question | Main visual evidence | Supporting tables / exclusions |
|---|---|---|
| What changes in the prior? | Accepted class/feature exposure, filtering, controlled mechanisms, raw LGD masses and standardized ties | Binary quantiles and raw marginal distances; omit arbitrary example-task histograms |
| RQ1: does the package transfer? | Paired credit effects, dataset heterogeneity, credit/OOD frontier, conditional removals | Exact absolute/reference scores and native binary/multiclass OOD diagnostics; no duplicate rankings |
| RQ2: what does continuation gain or forget? | Separate backbone effects and dataset maps; frontiers versus both trained controls and released weights | Fixed-order score tables; primary-metric trajectories only when measured; secondary metrics remain numeric |
| RQ3: which constraints retain generality? | Paired recipe effects and primary-metric trajectories when available | Resolved recipe table; no design-only PDF presented as results |
| Did optimization behave as intended? | Synthetic loss and matched development gap (Exp1); backbone/rate-separated loss and monitors (Exp2) | Run coverage, cost and sampled gradient health; no repeated coverage maps |

The current experiment definitions and `CreditICL Paper/0. Notes/main.tex` govern this narrative.
The older ArXiV draft still describes superseded grids; its scratch-Exp3 proposal is not implemented
by these notebooks. No benchmark-driven selection or experiment definition changed in this revision.

Presentation references were inspected as rendered papers at TFM pin
`8f14e5945ca6b068ed0c76218e769d3e672d0d13`: [TabICLv2, Fig. 10](../tfm-library/papers/2026/09_Qu_et_al._TabICLv2_A_better_faster_scalable_and_open_tabular_foundation_model.pdf)
(matched controls across training), [RealTabPFN, Figs. 4–6](../tfm-library/papers/2025/07_Garg_et_al._Real_TabPFN_Improving_Tabular_Foundation_Models_via_Continued_Pre_training_With_Real_World_Data.pdf)
(signed gains and adverse source effects), [Test-Time Compute, Fig. 2](../tfm-library/papers/2026/10_Ning_et_al._Test_Time_Compute_for_Tabular_Foundation_Models_Mechanisms_Gains_and_Limits.pdf)
(per-dataset heterogeneity), and [Shaping the Prior, Fig. 2](../tfm-library/papers/2026/10_Bouadi_et_al._Shaping_the_Prior_How_Synthetic_Task_Distributions_Determine_Tabular_Foundation_Model_Quality.pdf)
(structural diagnostics separated from efficacy). Their visual techniques inform selection; their
results do not establish CreditICL's effects.

## Run and export

```bash
python -m src.utils.run_notebooks
python -m src.utils.run_notebooks --only 1.
python -m src.utils.run_notebooks --only 1.3 1.4
```

The runner discovers notebooks recursively and saves inline outputs into each
notebook. `FigureSaver` clears only that notebook's figures, writes PDFs and
caption metadata, and the runner rebuilds `CAPTIONS.md` and `All_Results.md`.
Official notebook PDFs are committed for direct manuscript reuse. No dataset,
checkpoint, downloaded training log or benchmark table is committed.

The prior notebooks generate fresh seeded diagnostics rather than silently using
old cached pools. These CPU draws are the slowest part of a local rerun. The
figures document their sample sizes and diagnostic row subsamples; those plotting
subsamples do not cap the production benchmark.

For the committed Windows figures, use `CREDITICL_DRAW_WORKERS=1` (PowerShell:
`$env:CREDITICL_DRAW_WORKERS='1'`). Worker count changes the diagnostic random streams;
seed alone does not reproduce these illustrations across different worker counts.
This setting does not change the production training loader or benchmark.
