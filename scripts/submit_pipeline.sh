#!/bin/bash
# =============================================================================
#  CreditICL — submit the WHOLE pipeline as one dependency chain.
#
#      DRY_RUN=1 bash scripts/submit_pipeline.sh lgd   # validate, queue nothing
#      bash scripts/submit_pipeline.sh lgd
#      bash scripts/submit_pipeline.sh pd
#      bash scripts/submit_pipeline.sh both
#
#  You run this once and then log out. Slurm holds each stage until the previous
#  one finishes successfully (`--dependency=afterok`), so nothing needs babysitting
#  and nothing starts on top of half-written inputs.
#
#  THE CHAIN, and why each stage gets the hardware it does:
#
#    1  preprocess    1 CPU node         21 datasets, pandas — minutes, no GPU
#    2  prior pools   2 x 100 CPU tasks  CPU-bound generation; the two variants are
#                                        independent, so they run side by side
#    3  verify        1 CPU task         the gate: equal, complete pools or stop
#    4  pretrain      GPU array (48)     the only stage that needs a GPU
#    5  evaluate      1 GPU node         baselines + our checkpoints on real data
#
#  EVERYTHING RUNS ON wICE. Slurm dependencies do not cross clusters, so a chain that
#  hopped wice -> genius -> mindwell could never have run.
#
#  The two variants go in as separate array jobs so 200 tasks are eligible at once;
#  stage 3 waits on BOTH via `afterok:jobA:jobB`.
#
#  Array tasks count against your concurrent-job limit. Check it before submitting:
#      sacctmgr show qos normal format=Name%20,MaxSubmitJobsPerUser%15
# =============================================================================

set -euo pipefail

WHICH="${1:-lgd}"
N_SHARDS="${N_SHARDS:-100}"
N_DATASETS="${N_DATASETS:-400000}"
REPO="${CREDITICL_REPO:-$VSC_DATA/CreditICL}"

cd "$REPO"

# DRY_RUN=1 validates every sbatch WITHOUT queueing anything. `--test-only` makes
# Slurm parse the script, check the partition/account/resources and report when the job
# would start — then exit. Worth running first: a bad partition or a malformed
# dependency otherwise only shows up after some stages are already queued, leaving a
# half-submitted chain to clean up.
DRY_RUN="${DRY_RUN:-0}"

# `sbatch --parsable` prints the job id — but on a MULTI-CLUSTER site it prints
# "jobid;cluster" (we saw "61683451;wice"). Feeding that straight into
# --dependency=afterok: produces a malformed dependency, so strip the suffix.
submit() {
    if [[ "$DRY_RUN" == "1" ]]; then
        # --test-only rejects a dependency on a job that does not exist, so drop it:
        # we are checking partitions and resources, not the chain wiring.
        local args=()
        for a in "$@"; do
            [[ "$a" == --dependency=* ]] || args+=("$a")
        done
        sbatch --test-only "${args[@]}" >&2 || echo "  ^^ WOULD FAIL" >&2
        echo "DRYRUN"
        return 0
    fi
    sbatch --parsable "$@" | cut -d';' -f1
}

submit_one_task() {
    local task="$1"
    echo ""
    echo "################  $task  ################"

    # ---- 1. preprocessing (CPU) -------------------------------------------
    local jid_pre
    jid_pre=$(submit --export=ALL,TASK="$task" scripts/slurm/preprocess.slurm)
    echo "  1 preprocess     $jid_pre"

    # ---- 2. prior pools (CPU arrays, one per variant, in parallel) ---------
    # Both depend only on preprocessing, not on each other, so Slurm can start
    # all 40 tasks as soon as nodes free up.
    local jid_orig jid_credit
    jid_orig=$(submit --dependency=afterok:"$jid_pre" \
        --export=ALL,TASK="$task",VARIANT=original,N_SHARDS="$N_SHARDS",N_DATASETS="$N_DATASETS" \
        --array=0-$((N_SHARDS - 1)) scripts/slurm/generate_prior.slurm)
    jid_credit=$(submit --dependency=afterok:"$jid_pre" \
        --export=ALL,TASK="$task",VARIANT=credit_v1,N_SHARDS="$N_SHARDS",N_DATASETS="$N_DATASETS" \
        --array=0-$((N_SHARDS - 1)) scripts/slurm/generate_prior.slurm)
    echo "  2 prior original $jid_orig   ($N_SHARDS tasks)"
    echo "  2 prior credit   $jid_credit   ($N_SHARDS tasks)"

    # ---- 3. the gate ------------------------------------------------------
    # afterok on an ARRAY job waits for every task to succeed, so a single dead
    # shard stops the chain here instead of silently shrinking a pool.
    local jid_verify
    jid_verify=$(submit --dependency=afterok:"$jid_orig":"$jid_credit" \
        --export=ALL,TASK="$task",N_DATASETS="$N_DATASETS" scripts/slurm/verify_prior.slurm)
    echo "  3 verify pools   $jid_verify"

    # ---- 4. pretraining (GPU array) ---------------------------------------
    local jid_train
    jid_train=$(submit --dependency=afterok:"$jid_verify" \
        --export=ALL,TASK="$task" "scripts/slurm/pretrain_${task}.slurm")
    echo "  4 pretrain       $jid_train   (GPU array)"

    # ---- 5. evaluation (GPU) ----------------------------------------------
    # `afterany`, not `afterok`: if 2 of 48 training runs die we still want the
    # numbers for the 46 that finished.
    local jid_eval
    jid_eval=$(submit --dependency=afterany:"$jid_train" \
        --export=ALL,TASK="$task" scripts/slurm/evaluate.slurm)
    echo "  5 evaluate       $jid_eval"

    echo ""
    echo "  chain: $jid_pre -> ($jid_orig,$jid_credit) -> $jid_verify -> $jid_train -> $jid_eval"
}

case "$WHICH" in
    lgd|pd) submit_one_task "$WHICH" ;;
    both)   submit_one_task lgd; submit_one_task pd ;;
    *) echo "usage: bash scripts/submit_pipeline.sh [lgd|pd|both]" >&2; exit 2 ;;
esac

cat <<'EOF'

Done. If you ran with DRY_RUN=1 nothing was queued — re-run without it to submit.

Useful commands:

  squeue --clusters=all -u $USER            # what is queued or running
  squeue --clusters=all -u $USER --start    # when Slurm thinks each will start
  scancel <jobid>                           # cancel one stage (later stages die with it)

Logs land in logs/ inside the repo; official results in results/{lgd,pd}/<pipeline>/.
EOF
