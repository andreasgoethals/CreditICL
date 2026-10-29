# The notebook story

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
when available, appears beside the run maps. The dated cluster record remains in
`docs/RUNS.md`. No training curve is fabricated from that summary.

Final panels use complete current-protocol benchmark loaders: all eligible rows,
all prediction-time-eligible features, and five common folds. Missing evidence
produces one availability notice instead of a gallery of empty plots. The final
credit/OOD frontier is not populated from short-context training monitors.

Experiment 2 retains four mixtures and both peak rates per backbone, with separate plots
for each backbone/rate and full-table checkpoint trajectories. Experiment 3 fixes the 50%
mixture and predeclared full-model/adapter settings, and reuses the primary-rate Exp2 baseline.
Both experiments retain 125/250/500/1,500 updates; paired-retention and checkpoint panels stay
empty until the required evidence exists. The designated holdout has historical project
exposure; it is **not untouched confirmation**. No plot or notebook chooses a setting.

Paired differences average folds and training seeds within each dataset before
descriptive bootstrap intervals are computed. Seed marks display training spread;
seeds and folds do not inflate the dataset count. Native binary/multiclass OOD
subsets are diagnostics alongside the fixed aggregate. LGD endpoint plots report
aggregate mass matching at tolerance 1e-4 using 99 predictive quantiles, not
conditional calibration.

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
