# CreditICL output

This tree uses the same relative paths locally and on VSC, but local analysis intentionally
contains **no model weights**. `src/utils/paths.py` chooses the storage tier.

| Location | Contents |
|---|---|
| DATA: `$VSC_DATA/CreditICL/output CreditICL` | Compact job logs, training records, reports and submission receipts |
| PROJECT: `/lustre1/project/stg_00211/CreditICL/output CreditICL` | Benchmark results, retained checkpoints and validation assets |
| Local `output CreditICL` | Verified analysis subset from both tiers and generated notebook reports |

Filter a remote inventory with
`python -m src.utils.result_manifest inventory.json --out analysis.json`. Transfer only its
listed members and verify their sizes/hashes. The command plans collection; it does not transfer
or delete files. Exclude weights, optimizer state, prior pools, runtime images, locks and
unfiltered archives. A request for “all output” does not require a full-tree mirror.

## Layout

~~~text
output CreditICL/
├── README.md
├── All_Results.md                 verbatim final notebook summaries
├── CAPTIONS.md                    descriptive paper captions
├── figures/<notebook>/            PDF exports and caption/summary metadata
├── manifests/                    collection provenance and submission receipts
├── logs/                         shared preparation/diagnostic logs
├── reference/
│   ├── logs/
│   └── benchmark/{pd,lgd,ood,receipts}/
└── experiment_<N>/               0 validation; 1 scratch prior; 2 CPT; 3 retention
    ├── logs/
    ├── runs/<run>/
    │   ├── config.json           resolved training recipe
    │   ├── summary.json          completion, steps and cumulative training time
    │   ├── progress.csv          small-context monitoring
    │   ├── telemetry.csv         sampled resource/gradient/throughput health
    │   ├── weights.csv           parameter displacement, where recorded
    │   └── logs/                 training loss and restart logs
    ├── benchmark/{pd,lgd,ood,receipts}/
    ├── checkpoints/<run>/        VSC only: milestones and recovery state
    ├── report/                   Exp0 CPU/GPU canonical receipts
    └── validation/<attempt>/     Exp0 isolated logs, artifacts and notebook copies
~~~

Shared references are released TabICLv2, TabPFN-3, CatBoost and linear models. Complete compatible
receipts allow reuse across experiments; they are not valid forever regardless of protocol or input
changes. Benchmark files contain dataset/seed/fold/model cells, and receipts bind their hashes
to the request, source, environment and inputs.

Earlier checkpoints have `_s<step>` in their tags; the final endpoint preserves its original tag.
Historical `_ctx1024` outputs used capped contexts and do not populate current full-table panels.
Small-context training monitors are also separate from the final credit/native-OOD benchmark.

Released weights and datasets are inputs beside this output tree on PROJECT:
`checkpoints/`, `data/raw/`, `data/processed/` and `prior_cache/ood/`. The last path is the pinned
OOD cache; it is not `data/ood/`.

## Saved reports and cleanup

Git tracks this guide, `All_Results.md`, `CAPTIONS.md`, official notebook PDFs and `_figures.json`.
Imported scores/logs/receipts and model assets remain ignored. Regenerate summaries and captions
through the notebook runner, not manual edits. Existing `figures/<notebook>` folders intentionally
retain the project's established layout.

`python -m src.utils.clean_run` previews cleanup. Removal needs a reviewed scope and no active
writers/recovery consumers. Full reset options include trained results and are not housekeeping.
Preserve successful Experiment 1/2 baselines and historical failure evidence. See
[the VSC runbook](../docs/VSC.md#cleanup-and-historical-output) for tier and scratch checks.
