#!/bin/bash
# =============================================================================
#  CreditICL — submit the WHOLE pipeline as one dependency chain.
#
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
#    1  preprocess    1 CPU node        21 datasets, pandas — minutes, no GPU
#    2  prior pools   2 x 20 CPU tasks  CPU-bound generation; the two variants are
#                                       independent, so they run side by side
#    3  verify        1 CPU task        the gate: equal, complete pools or stop
#    4  pretrain      GPU array         the only stage that needs a GPU
#    5  evaluate      1 GPU node        baselines + our checkpoints on real data
#
#  Stages 2's two variants are submitted as separate array jobs so 40 tasks are
#  eligible at once; stage 3 waits on BOTH via `afterok:jobA:jobB`.
#
#  WALL-CLOCK, roughly: stage 2 dominates if run serially (40,000 datasets), which
#  is exactly why it is 40 parallel tasks instead of one loop.
# =============================================================================

set -euo pipefail

WHICH="${1:-lgd}"
N_SHARDS="${N_SHARDS:-20}"
N_DATASETS="${N_DATASETS:-40000}"
REPO="${CREDITICL_REPO:-$VSC_DATA/CreditICL}"

cd "$REPO"

# `sbatch --parsable` prints just the job id, which is what a chain needs.
submit() { sbatch --parsable "$@"; }

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

Submitted. Nothing to do now but wait. Useful commands:

  squeue --clusters=all -u $USER            # what is queued or running
  squeue --clusters=all -u $USER --start    # when Slurm thinks each will start
  scancel <jobid>                           # cancel one stage (later stages die with it)

Logs land in logs/ inside the repo; official results in results/{lgd,pd}/<pipeline>/.
EOF
