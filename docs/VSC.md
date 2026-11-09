# Running CreditICL on KU Leuven VSC

This runbook describes the deployed project. The [template](TEMPLATE.md) owns portable access
and storage policy; the pinned [VSC Library](../vsc-library/README.md) supplies infrastructure
references. Scientific settings belong in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md), and
dated jobs, failures, quotes and migrations belong in [RUNS.md](RUNS.md).

## Contents

1. [Current deployment](#current-deployment)
2. [Access and activation](#access-and-activation)
3. [Validate a revision](#validate-a-revision)
4. [Submit and recover work](#submit-and-recover-work)
5. [Resources and costs](#resources-and-costs)
6. [Storage and collection](#storage-and-collection)
7. [Runtime maintenance](#runtime-maintenance)
8. [Cleanup and historical output](#cleanup-and-historical-output)

## Current deployment

**Verified snapshot: 9 November 2026.** Counts below are dated evidence, not live status.

| Work | Verified state |
|---|---|
| Exp1 | All 54 training runs and 158 benchmark slots complete, including the 25% extension |
| Exp2 training | All 64 runs reached 1,500 updates |
| Exp2 benchmark | All 320 slots completed and verified, covering 56,800 credit/OOD cells |
| Revised CPU Exp0 | wICE 62581713: all 14 stages PASS |
| Revised GPU Exp0 | Mindwell 11774935: 125/125 stages PASS, live preflight accepted |
| Exp3 training | 48 admitted initially; 12 pending TabPFN LGD tasks stopped on 9 November; other 36 preserved |

Production runs use frozen DATA checkout `$VSC_DATA/CreditICL` at **0494e7a**. Its CPU
62516762/GPU 11763662 acceptance remains evidence for that revision. Revised validation uses
the separate DATA checkout `$VSC_DATA/.crediticl-validation-20261106/CreditICL` at **a3891f5**,
with its own receipt namespace. Both use the accepted PROJECT r3 runtime. Preserve both paths
while jobs, receipts or recovery need them. Do not pull the later notebook/documentation changes
into either active checkout.

The five completed Exp2 arrays are 11774787/11774788 (final PD/LGD) and
11775176/11775177/11775178 (earlier PD TabICL, PD TabPFN-3, LGD). Their receipt files under
`output CreditICL/manifests/submissions/` cover every slot exactly once. **Do not resubmit them.**
The [6 November run records](RUNS.md#runs) contain exact ranges, quotes and artifact checks.

Exp3 uses the revised DATA checkout and the same PROJECT image. The durable admission receipt
is `20261109-exp3-all-training.json` in that checkout's `output CreditICL/manifests/submissions/`.
The first training allocations quoted 17,625,600 credits. At admission, the available balance
was 57,946,762, with about 761 GiB free on PROJECT and 30 GiB on DATA.
Recheck quotas and balance before the later benchmark phase.

The 9 November investigation cancelled only `11780839_6-11,18-23`, before training,
pending regression-objective diagnostics. Preserve the stop receipt alongside the original
admission; do not automatically refill those cancelled IDs. Synthetic probes run in separate
DATA checkouts and output namespaces; see the newest [run record](RUNS.md#runs).

<a id="revised-exp3-deployment-gate-05-11-2026"></a>
### Exp3 deployment gate

Require the revised CPU and GPU receipts to pass their source, environment, input and artifact
checks. The older production GPU receipt covers 117 stages and 16 older adaptation routes;
it cannot validate the revised six-recipe grid. The revised gate has 125 stages, including
24 recipe/task/backbone round trips and 34 long-context probes.

Exp2 benchmarking may continue on its accepted checkout while revised validation runs separately.
Once the revised gate passes, production may use that accepted, frozen revision. Unchanged
benchmark output does not itself require another Exp0. A changed source/runtime/input identity does.
Do not retrain Exp2 baselines or duplicate their benchmarks for Exp3. Shared reference receipts
are reused only after their normal input, package and result checks.

## Access and activation

Andreas renews access manually in Windows PowerShell:

~~~powershell
step ssh login --context VSC
~~~

He approves KU Leuven Authenticator. Reuse the existing native Windows OpenSSH agent; do not
bootstrap per project, initiate authentication, add renewal hooks or schedule reminders.
If access has expired, request renewal once and wait for confirmation.

For agent work, follow the bounded LF-stdin probe in
[TEMPLATE.md](TEMPLATE.md#vsc-access-windows-and-agents): native `ssh.exe -T`, `BatchMode`,
strict host keys and no trailing remote command. Verify `stdout`, `stderr` and the expected remote
exit status. Empty output with exit 0 is not evidence of success. A tested fallback captures
the remote payload through Python `subprocess.run` and returns a JSON envelope containing
`stdout`, `stderr` and returncode. Network or transport failures do not establish certificate expiry.

The login endpoint is login.hpc.kuleuven.be. Run metadata/queue checks there; computational
debugging belongs in Slurm. Activate the selected runtime yourself:

~~~bash
cd "$VSC_DATA/CreditICL"
source scripts/slurm/_activate_env.sh
python -c 'import sys; print(sys.version); print(sys.prefix)'
~~~

The PROJECT image exposes /opt/venv. Jobs explicitly source the same launcher.
The old DATA-venv shell hook was removed; **do not install a new hook**. Existing interactive
shells may retain old state, so use a fresh SSH session. Access alone does not authorize new
workloads, installations or migrations.

## Validate a revision

### Safe deployment

Edit and test locally, commit/push, then pull into an **idle DATA checkout**. Never change source,
submodules, dependencies or inputs consumed by active, queued or recoverable jobs. Use a separate
DATA checkout and receipt namespace for concurrent validation; the current split is recorded above.

If tracked notebook renders block an authorized update, preserve only those paths first:

~~~bash
git stash push -m "VSC notebook renders before update" -- \
  notebooks "output CreditICL/All_Results.md" "output CreditICL/CAPTIONS.md" \
  "output CreditICL/figures"
git pull --ff-only
git submodule update --init --recursive
~~~

Run each command only after success of the previous command. Review the stash before restoring
anything; do not reapply obsolete renders automatically. Never stash or delete expensive untracked
results to make a pull succeed. Cluster notebook execution writes copies under the Exp0 attempt,
so it does not modify tracked notebooks. This is an intentional template adaptation.

### CPU then GPU

Preflight checks inputs and dependencies; it is not GPU acceptance. For a new authorized revision,
quote the actual script and submit through the admission manager with a **new stable receipt**.
This example is for an idle normal checkout, not the currently isolated validation campaign:

~~~bash
source scripts/slurm/_activate_env.sh
python -m src.utils.preflight
sam-quote sbatch --clusters=wice scripts/slurm/exp0_cpu.slurm
~~~

After checking the quote, identity and live queue:

~~~bash
python - <<'PY'
from pathlib import Path
import subprocess
from src.utils.submit_jobs import submit_command
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
submit_command([
    "sbatch", "--clusters=wice",
    f"--export=ALL,CREDITICL_EXPECTED_COMMIT={commit},CREDITICL_PROJECT_ROOT=/lustre1/project/stg_00211,CREDITICL_SITE_SUBMIT_LIMIT=500",
    "scripts/slurm/exp0_cpu.slurm",
], Path("output CreditICL/manifests/submissions/REVIEWED-REVISION-cpu.json"))
PY
~~~

Replace `REVIEWED-REVISION` with the recorded campaign identifier; inspect any existing receipt
before reuse. Verify the site's ceiling instead of assuming 500 remains current. After CPU
terminal accounting and all stages/artifacts pass, run preflight with --require-exp0-cpu.
Quote and admit `exp0.slurm` on Mindwell through the same function with a distinct GPU receipt.
Do not queue GPU work before CPU acceptance or retry an uncertain submission.

The current CPU job checks dependencies, configurations, all dataset loads/folds, replay,
portable tests and notebooks. The GPU campaign covers native training/equivalence, forced
stop/resume, full-table five-fold probes, wide inputs (2,986 PD / 601 LGD features),
34 long-context stages and 24 retention-recipe round trips. The 60,000-row adapter probes use
their higher rates. Coverage establishes tested execution paths, not universal absence of bugs
or the scientific effectiveness of a treatment.

CPU reserves 4 hours; GPU reserves 24 hours. Controllers stop before those limits where possible.
Exp0 benchmark/capacity guards never reduce rows, features, folds or ensemble members to pass.
Validation uses selected representative tables; the complete publication matrix belongs to
Experiments 1–3.

### Read the outcome

Receipts live in `output CreditICL/experiment_0/report/`:
`cpu_validation.json` for CPU and `validation.json` for GPU. Attempt logs/artifacts live under
`experiment_0/validation/<attempt>/` on the appropriate tier. GPU acceptance binds the exact
CPU receipt. A rerun of CPU therefore invalidates a previous GPU receipt.

| Recorded outcome | Action |
|---|---|
| PASS | Check current source/input/environment identities and all artifacts |
| RUNNING | Consult `sacct`; this is the last saved controller state, not a live scheduler query |
| DEFERRED / exit 64 | Reconcile terminal accounting, then continue the unchanged script with a new allocation receipt |
| FAILED | Diagnose locally before changing code or retrying |
| Hard kill / interrupted | Preserve evidence and recovery state; reconcile before continuing |

A DEFERRED exit may appear as FAILED in Slurm accounting. Inspect the controller's final message.
Completed stages and benchmark folds are reusable only after their hashes match. Do not pull,
reset output, refresh inputs or rerun CPU between unchanged GPU continuation allocations.

After full completion:

~~~bash
python -m src.utils.preflight --require-exp0
python -m src.validation.verify
~~~

Exp0 reference probes are isolated from publication benchmarks. Local verification of filtered
downloads can report deliberately absent checkpoints; on VSC, preflight verifies those in place.

## Submit and recover work

Inspect before submitting:

~~~bash
squeue --clusters=mindwell --me --array
sacct --clusters=mindwell -j JOB_IDS --format=JobID,State%20,ExitCode,Elapsed,Timelimit
myquota
sam-balance
sacctmgr show qos normal format=Name%20,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30
python -m src.utils.run_experiment 2 --single-cluster --phase benchmark
~~~

The planner checks completion/provenance and live job names; unavailable queue information blocks
admission. `--submit` performs only ready work through `src.utils.submit_jobs`, preserving accepted
IDs and original array indices. Use explicit phases to avoid launching a later stage unintentionally.
For **authorized, genuinely missing** work after validation:

~~~bash
python -m src.utils.run_experiment 2 --single-cluster --phase benchmark --submit
python -m src.utils.run_experiment 3 --single-cluster --phase train --submit
~~~

These are lifecycle examples, not instructions to duplicate the already accepted Exp2 arrays.
Keep `--single-cluster` for Exp2/3: B200 acceptance does not validate the optional historical
A100 split. Quote each generated resource group before admission. Earlier Exp2 benchmark groups
use overrides of 4h for PD TabICL, 3h for PD TabPFN-3 and 45m for LGD; the generic planner's
script defaults do not automatically apply those reviewed overrides.

The observed submission ceiling is 500 pending+running jobs. All projects share this user's
admission capacity. The manager waits/refills at capacity without an artificial running-job
throttle. A definite capacity refusal may retry; an uncertain acknowledgment must be reconciled
against the receipt and scheduler before any retry. Do not cancel another project's jobs.

### Training recovery

The training launcher requests a SIGUSR1 warning 600 seconds before walltime, forwards the signal,
finishes the current update and publishes a checkpoint. A progressing, graceful exit 64 may
requeue the same array task under its restart guard. Cancellations and ordinary crashes do not
trigger that application continuation. `--requeue` also allows site handling of node failure.

Rolling saves bound lost work after an abrupt kill; long-context update duration and publication
time must fit the warning window. The old short-table timing does not establish that bound.
After a failure, inspect logs and publication state before using the planner to resume missing
arms. For one config, `src.utils.sweep_status` reports done/partial/todo; consult the live queue
as well. Completion requires the finished summary, terminal accounting and expected artifacts.

The retired `submit.sh` / `submit_pipeline.sh` refuse the obsolete pool-based launch chain.
No idle deployment controller or scheduled monitor is needed for the current campaign.

## Resources and costs

Quote actual scripts with `sam-quote`; it does not submit jobs. A quote prices the requested limit,
not the eventual bill. Billing and queue delay are different, and a shorter appropriate limit
can backfill sooner. Underestimation can cause interruption and repeated work.

| Workload | Reviewed allocation / limit |
|---|---|
| CPU Exp0 | wICE interactive, 8 CPUs, 48G, 4h; observed quote 0 |
| GPU Exp0 | Mindwell B200, 24 CPUs, 180G, 24h |
| Exp2 PD TabICL benchmark | One B200, 24 CPUs, 180G, 4h |
| Exp2 PD TabPFN-3 benchmark | Same allocation, 3h |
| Exp2 LGD benchmark | Same allocation, 45m |

The verified B200 allocation price was 30,600 credits/hour, including host resources.
Get a fresh quote before new submissions:

~~~bash
sam-quote sbatch --clusters=mindwell --partition=gpu_b200 --account=lp_verbekelab \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=24 --mem=180G \
  --time=01:00:00 --array=0 scripts/slurm/pretrain_pd.slurm
~~~

This site's quote tool requires an actual jobscript; the --wrap-only form failed.
Measured Exp1 training took about 6.4–7.4 GPU-hours per run. Historical Exp1/released-model
benchmark fold timers were roughly 2–3 hours for PD and 6–10 minutes for LGD; these are planning
anchors, not guarantees for every adapted model. Full retained Exp2 coverage is 320 checkpoint
benchmarks, or 56,800 dataset/fold/model cells before shared references and PD calibration refits.
Exp3 adds 240 new checkpoint benchmarks; method-specific training/inference costs still need
measurement. Detailed dated projections and quotes are retained in RUNS.md rather than treated
as current balances or bills.

Use native Slurm clients and a login-shell job script. Explicit cluster and account are required.
Forward required roots/revision through the job export; do not assume arbitrary login exports
survive. Never module --force purge. These launchers run Python directly and retain their tested
24-CPU/180G requests: this is a documented exception to the template's default-memory example,
which addresses the site's `srun` CPU/memory interaction. Do not apply an untested resource rewrite.

## Storage and collection

The authority is `src/utils/paths.py`, with working I/O managed by src/utils/working_storage.py.

| Tier | CreditICL contents |
|---|---|
| DATA: `$VSC_DATA/CreditICL` | Actual checkout and `.git`; compact logs, run records, reports and admission receipts |
| PROJECT: `/lustre1/project/stg_00211/CreditICL` | Inputs, released/trained weights, benchmark results, reusable caches and immutable SIF runtimes |
| Node-local scratch on Mindwell | Staged inputs, prediction offload and a working checkpoint |
| Shared GPFS/Lustre scratch | Explicit temporary project-owned workspaces only; never sole recovery copies |

`CREDITICL_PROJECT_ROOT` or `VSC_PROJECT_ROOT` names the **shared allocation root**; CreditICL
is appended. Legacy `STAGING` aliases remain supported. These overrides change path resolution,
not file placement: changing them is a reviewed migration, not automatic copying.
Keep `CREDITICL_WORK_ROOT` unset for native node-local GPU I/O. There is no automatic shared-GPFS
fallback. Lustre PROJECT access from Mindwell is used for staging and publication, not repeated
training reads/offload.

Each save is published atomically to PROJECT before retiring older working copies. Permanent
milestones and two rolling recovery states remain there. Node-local storage can disappear at job
end; verify publication before cleanup. Check byte **and file-count** quotas before staging/builds
or new campaigns. The reviewed Exp2 reserve was at least 250 GiB/2,500 free PROJECT files and
5 GiB/20,000 free DATA files; additional campaigns need their own capacity estimate.

PROJECT is persistent, but persistence is not an independent backup. Andreas confirmed that
there is no separate trained-checkpoint backup and that checkpoints should remain on PROJECT.
Preserve them there; do not describe the same-filesystem runtime rebuild archive or a local
analysis download as checkpoint backup coverage.

### Local analysis collection

Follow the [output guide](<../output CreditICL/README.md>). Inventory both tiers, filter through
`src.utils.result_manifest`, transfer only listed scores/predictions/logs/configs/receipts/figure
metadata, then verify original sizes and hashes. Model weights, optimizer state, pools, images,
locks and unfiltered archives stay on VSC. A filtered subset is intentionally not a full-tree mirror.

### Restricted dataset configuration

The published checkout uses dataset aliases. To reproduce the full benchmark, provision the
institution-only recipe in the protected DATA checkout at `.private-data/site.py`, or set
`CREDITICL_PRIVATE_DATA` to an approved bound path. Use an authorized copy of the existing site
recipe. It contains the legacy storage keys, input transformations, prediction-time exclusions
and grouped-fold fields. Keep the configuration immutable for the campaign and outside Git.
The loader resolves existing PROJECT cache paths without renaming them. Run fresh Experiment 0
before deploying this checkout; jobs with frozen source and existing receipts use their
original checkout and inputs through completion.

## Runtime maintenance

Dependencies are installed inside immutable PROJECT Apptainer/SIF images, not an unpacked
PROJECT environment. The deployed r3 image is selected by DATA .runtime-build/current.json.
It passed full CPU/GPU runtime acceptance on 31 October; scientific revision acceptance is
separate. The scientific interpreter is /opt/venv/bin/python. Torch is 2.11.0+cu128,
TabPFN 9.1.0, and TabICL uses the Git pin in pyproject.toml. All requirements, extras,
installed versions and `pip check` are audited; package metadata alone is not GPU acceptance.

`_activate_env.sh` constructs reviewed code/input/output/scratch and native Git/Slurm/credit-client
binds, clears inherited container overrides, and never installs packages at job startup.
Code is mounted read-only during computation. A new runtime requires recorded dependencies,
image hashes, offline build inputs and actual CPU/GPU validation before cutover.
Build through `scripts/slurm/build_runtime.slurm` in an authorized allocation; keep temporary
build caches on scratch and final images on PROJECT. Do not move a populated environment.

Legacy DATA `.venv-skylake` (9.77 GiB in the 30 October audit) and the separate Conda CreditICL
environment (5.96 GiB) remain recovery exceptions. The latter has different package versions
and lacks distinct recovery coverage; neither may be removed merely because an image exists.
Retirement also requires no active, queued, resumable or cross-login consumer. The verified
167-member rebuild archive covers the former venv only. Detailed inventories/hashes remain
in the dated run records.

`CREDITICL_LEGACY_RUNTIME=1` and the `--legacy-maintenance` environment scripts are exceptional
recovery paths, not routine setup. Do not reinstall the removed shell autoactivation hook.
Use local laptop venvs for analysis; never copy one to VSC.

## Cleanup and historical output

Preview before considering removal:

~~~bash
python -m src.utils.clean_run
python -m src.utils.scratch_cleanup
~~~

Cleaning requires an idle, scoped project, reviewed targets and preserved recovery/publication
evidence. `--clean` executes the selected plan. Exp0-specific reset options must not remove
production results; a DEFERRED validation should resume, not reset. `--reset-exp2` and the
full `--reset-output` include trained results/checkpoints and are **not routine housekeeping**.
Keep successful Exp1/Exp2 checkpoints needed by future comparisons.

Scratch cleanup considers only positively identified terminal CreditICL jobs. Unknown jobs,
unpublished/corrupt checkpoints and other projects are retained. Never wipe a shared root.
A normal-exit trap alone does not establish successful publication.

The established output root is `output CreditICL`, with the space. Historical layouts remain
readable; `src.utils.rename_output` is an explicit idle-campaign migration tool, not a setup step.
September results retain their own recipes/protocols and do not count as completion of the
current design. Preserve their evidence; do not relaunch old sweeps.

## Reference policy

The local template matched the canonical copy byte-for-byte on 6 November (SHA-256
2c16ea7a906eb4c468f276eeb411a4e757de370a6af5013c205a93718b33fc3a).
VSC reference pin: 914ef4e57e1f153080e2686d03179c223db38c9a. Shared submodules remain
read-only. Retained project deviations are the established output/config layout, isolated
cluster notebook copies and tested explicit resource requests. No documentation edit changes
a live runtime, storage root or campaign.
