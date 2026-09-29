# output_CreditICL — everything CreditICL produces

One tree, the same layout everywhere: in this repo, on the cluster under `$VSC_DATA/CreditICL/`
(the small files) and under `/lustre1/project/stg_00211/CreditICL/` (the big ones). The layout is
defined in one place, `src/utils/paths.py`; nothing else builds an output path.

```
output_CreditICL/
├── README.md                this file
├── All_Results.md           every notebook's printed summary, in notebook order
├── CAPTIONS.md              every figure's caption, ready for the paper
├── general/                 not tied to one experiment
│   ├── figures/0.x_*/       chapter 0: the data and the prior
│   ├── logs/                preprocessing, OOD fetch, prior pools, GPU benchmarks
│   ├── data/<track>/        preprocessing summaries
│   └── prior/<track>/       prior-pool status
├── reference/               the released TabICLv2 and TabPFN-3, CatBoost, linear:
│   ├── benchmark/           scored ONCE, on every dataset, and read by every experiment
│   │   ├── pd/  lgd/        results_reference_<track>_<model>.csv, one row per dataset x fold
│   │   ├── ood/             the same models on the out-of-domain suites
│   │   └── receipts/        proof each result is complete and current (src/eval/benchmark_status.py)
│   └── logs/
└── experiment_<N>/          N = 0 debug suite | 1 prior from scratch | 2 continued pretraining | 3 long run
    ├── logs/                one file per SLURM job (training, benchmark, checks)
    ├── runs/<run name>/     one folder per trained arm
    │   ├── config.json      the resolved configuration, exactly as trained
    │   ├── summary.json     completed?, steps, training time over every restart
    │   ├── progress.csv     development-set metrics every 500 steps (real credit + out-of-domain)
    │   ├── telemetry.csv    throughput, GPU and CPU use, memory, time per phase, loss, learning
    │   │                    rate, gradient norms per block
    │   ├── weights.csv      how far each block has moved from its starting weights, and how fast
    │   └── logs/            the trainer's own log and per-step metrics, one pair per (re)start
    ├── benchmark/           the arms' checkpoints scored with the full protocol
    │   ├── pd/  lgd/  ood/  results_exp<N>bench_<track>_a<arm>[_s<step>].csv
    │   └── receipts/
    ├── figures/<N>.x_*/     that experiment's notebooks (chapter N)
    └── report/              experiment_0 only: the checks job's report (checks.md, checks.json)
```

On project storage the same tree holds only what is too big for `$VSC_DATA`:

```
output_CreditICL/
├── experiment_<N>/checkpoints/<run name>/step-<step>.ckpt
└── prior_cache/             pre-generated prior pools (optional)
```

The released weights and the datasets are inputs, not output, and live beside the tree:
`checkpoints/` (TabICLv2 and TabPFN-3 as downloaded) and `data/` (`raw/`, `processed/`, and the
out-of-domain cache `ood/`).

Committed to git: this README, `All_Results.md`, `CAPTIONS.md` and each figure folder's
`_figures.json`. Everything else is regenerated (figures) or downloaded from the cluster (logs,
runs, benchmark results). `python -m src.utils.clean_run` lists what a clean would remove.
