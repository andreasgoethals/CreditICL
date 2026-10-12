# output CreditICL — everything CreditICL produces

One tree, the same layout everywhere. Locally it is this one folder. On the cluster it exists
twice, with the same layout, and you download both halves into this folder:

- `$VSC_DATA/CreditICL/output CreditICL/` — the **small** half: what a job appends to while it runs
  (its log, a run's progress and telemetry). `$VSC_DATA` has 75 GiB for every project together, so
  nothing else goes there.
- `/lustre1/project/stg_00211/CreditICL/output CreditICL/` — the **big** half: benchmark results,
  checkpoints, prior pools. Download it WITHOUT `*/checkpoints/` and `prior_cache/`
  (`docs/VSC.md` §10 has the commands).

The layout is defined in one place, `src/utils/paths.py`; nothing else builds an output path.

```
output CreditICL/
├── README.md                 this file
├── All_Results.md            every notebook's printed summary, in notebook order
├── CAPTIONS.md               every figure's caption, ready for the paper
├── figures/<notebook>/       every notebook's figures (chapters 0, 1 and 2), drawn locally
├── logs/                     jobs that belong to no experiment: preprocessing (with
│                             processed_summary_<track>.csv), the OOD fetch, prior pools, GPU benchmarks
├── reference/                the released TabICLv2 and TabPFN-3, CatBoost, the linear model —
│   ├── logs/                   models whose numbers do not depend on our prior, so they are
│   └── benchmark/              scored ONCE and read by every experiment              [big half]
│       ├── pd/ lgd/            results_reference_<track>_<model>[_ctx1024].csv, one row per dataset x fold
│       ├── ood/                the same models on the out-of-domain suites
│       └── receipts/           proof each result is complete and current (src/eval/benchmark_status.py)
└── experiment_<N>/           N = 0 debug suite | 1 prior from scratch | 2 continued pretraining | 3 long run
    ├── logs/                 one file per SLURM job (training, benchmark, checks)
    ├── runs/<run name>/      one folder per trained arm
    │   ├── config.json       the resolved configuration, exactly as trained
    │   ├── summary.json      completed?, steps, training time over every restart
    │   ├── progress.csv      development-set metrics every 500 steps (real credit + out-of-domain)
    │   ├── telemetry.csv     throughput, GPU and CPU use, memory, time per phase, loss, learning
    │   │                     rate, gradient norms per block
    │   ├── weights.csv       how far each block has moved from its starting weights, and how fast
    │   └── logs/             the trainer's own log and per-step metrics, one pair per (re)start
    ├── benchmark/            [big half] the arms' checkpoints scored with the full protocol
    │   ├── pd/ lgd/ ood/     results_exp<N>bench_<track>_a<arm>[_s<step>|_ctx1024].csv
    │   └── receipts/
    ├── checkpoints/<run>/    [big half, not downloaded] step-<step>.ckpt
    └── report/               experiment_0 only: the checks job's report (checks.md, checks.json)
```

`_s<step>` marks an earlier checkpoint than the final one; `_ctx1024` a result scored with at most
1,024 rows of context (Experiment 1's second evaluation, inside the training distribution).

The released weights and the datasets are inputs, not output, and live beside the tree on project
storage: `checkpoints/` (TabICLv2 and TabPFN-3 as downloaded) and `data/` (`raw/`, `processed/`,
and the out-of-domain cache `ood/`).

Committed to git: this README, `All_Results.md`, `CAPTIONS.md` and each figure folder's
`_figures.json`. Everything else is regenerated (figures) or downloaded from the cluster (logs, runs,
benchmark results). `python -m src.utils.clean_run` lists what a clean would remove.
