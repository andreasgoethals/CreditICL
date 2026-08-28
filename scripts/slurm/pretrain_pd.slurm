#!/bin/bash -l
# =============================================================================
#  CreditICL — Exp1 pretraining array, PD (imbalanced binary classification)
#  Mindwell, partition gpu_b200. One array task = one arm of config/Exp1_PD.yaml.
#
#  Check the grid size BEFORE submitting, and set --array to match:
#      python scripts/pretrain.py --config config/Exp1_PD.yaml --list
#
#  Cluster limits, resilience and the throttle arithmetic: docs/VSC.md
# =============================================================================
#SBATCH --job-name=crediticl-pd
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 24 cores and 194400 MiB is the documented per-GPU ceiling on gpu_b200. Asking for more
# raises a warning; asking for much less starves the prior generator.
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
# MEASURED 4.39 h per arm (job 11523286, end-to-end rung). 12 h is ~2.7x that, which covers
# evaluation, progress scoring and a slow node. Billing is on ACTUAL time so a longer request
# costs nothing extra, but the backfill scheduler starts short jobs sooner — and the job is
# resumable, so underestimating is cheap and overestimating is not.
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
# THREE THINGS END A RUN EARLY on this cluster and none is a bug: the walltime, a node
# failure, and an unannounced maintenance drain. --requeue lets Slurm put the task back in
# the queue for the first two; --signal delivers SIGUSR1 to THIS script 600 s before the
# walltime, which the trap below forwards so the trainer can checkpoint. `maybe_resume`
# picks the checkpoint up. See docs/VSC.md.
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
# 0-74 = the 75 arms of Exp1. %8 caps how many run at once: gpu_b200 has 24 GPUs TOTAL
# shared with every other user, and both MaxSubmitJobsPerUser and MaxTRESPerUser apply.
# Check yours before raising it:
#     sacctmgr show qos normal format=Name,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30
#SBATCH --array=0-74%8

# Jitter the start so concurrent tasks do not hammer the filesystem at once.
sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CREDITICL_OUTPUT_ROOT="${CREDITICL_OUTPUT_ROOT:-${VSC_DATA}/CreditICL}"
mkdir -p "${CREDITICL_OUTPUT_ROOT}/output/logs" "${CREDITICL_OUTPUT_ROOT}/output"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITICL_OUTPUT_ROOT}/output/logs/pretrain_pd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

_log_exit() {
    rc=$?
    echo "=========================================="
    if   [[ "$rc" -eq 0  ]]; then echo "END status=OK - $(date)"
    elif [[ "$rc" -eq 64 ]]; then echo "END status=INCOMPLETE-RESUBMITTED - $(date)"
    else echo "END status=FAILED exit_code=${rc} - $(date)"; fi
    echo "=========================================="
}
trap _log_exit EXIT

cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditICL — PRETRAIN (PD) — Mindwell gpu_b200"
echo "=========================================="
echo "Array job/task : ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Cluster        : ${SLURM_CLUSTER_NAME:-?}  partition=${SLURM_JOB_PARTITION:-?}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Cores / mem    : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MiB"
# FROM SLURM, not from the #SBATCH default. `SBATCH_TIMELIMIT` is not set inside a job,
# so the banner printed "12:00:00" while the actual limit was the 20 minutes passed on
# the command line — the one number a reader checks against the ETA.
LIMIT_S=$(( ${SLURM_JOB_END_TIME:-0} - ${SLURM_JOB_START_TIME:-0} ))
echo "Walltime       : $(printf '%02d:%02d:%02d' $((LIMIT_S/3600)) $((LIMIT_S%3600/60)) $((LIMIT_S%60)))   requeue=yes  signal=USR1@600"
echo "Restart count  : ${SLURM_RESTART_COUNT:-0}   <- >0 means this arm was requeued"
echo "Date           : $(date)"
echo "Log            : ${LOG}"
python -c "import json; from src.utils.paths import describe; print(json.dumps(describe(), indent=2))"
nvidia-smi || true
echo "=========================================="

# Pre-flight: fail in seconds on a broken env rather than after the queue wait.
python -m src.utils.smoke_test --task pd --steps 2 || {
    echo "PREFLIGHT FAILED: PD smoke test. Not starting the real run." >&2
    exit 1
}

# FORWARD THE SIGNAL. Slurm sends SIGUSR1 to this script, not to python, so the trainer
# would never see it. Run python in the background, trap, forward, and `wait` — a foreground
# child would also block the trap until it exited, which is exactly too late.
# WORKERS FROM THE ALLOCATION, not from the config. The prior is generated on the CPU, so
# the right number is however many cores this partition gave us minus one for the main
# process -- 23 on gpu_b200, 17 on gpu_a100, 7 on interactive. A hard-coded 12 wastes half
# a B200 node's cores and oversubscribes a smaller card.
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
echo "Dataloader workers: ${WORKERS} (from SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset})"

python -u scripts/pretrain.py \
    --config config/Exp1_PD.yaml \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --num-workers "${WORKERS}" \
    --prior-source "${PRIOR_SOURCE:-generate}" &
TRAIN_PID=$!
trap 'echo "SIGUSR1 -> forwarding to trainer $TRAIN_PID"; kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1
trap 'echo "SIGTERM -> forwarding to trainer $TRAIN_PID"; kill -TERM "$TRAIN_PID" 2>/dev/null || true' TERM

# Two waits: the first returns as soon as a trapped signal arrives, the second lets the
# trainer finish writing its checkpoint.
set +e
wait "$TRAIN_PID"; RC=$?
if [[ $RC -gt 128 ]]; then wait "$TRAIN_PID"; RC=$?; fi
set -e

# EXIT 64 = saved and stopped early. Resubmit THIS arm only; it resumes from the checkpoint.
if [[ "$RC" -eq 64 ]]; then
    echo "Arm ${SLURM_ARRAY_TASK_ID} incomplete — resubmitting to resume."
    # BACK TO THE SAME PLACE, WITH THE SAME RESOURCES. Carry over cluster/partition AND the
    # cores/mem this task was actually granted -- not the #SBATCH header default, which asks
    # for the gpu_b200 ceiling (24 cores, 180G) that a lower-ceiling partition (wICE gpu_a100:
    # 18 / 126000 MiB) rejects with "Requested node configuration is not available". PD runs on
    # mindwell today so the header would work, but echoing the grant is strictly correct and
    # costs nothing. (See the LGD script: an LGD wICE arm died on exactly this.)
    sbatch --clusters="${SLURM_CLUSTER_NAME:-mindwell}" \
           --partition="${SLURM_JOB_PARTITION:-gpu_b200}" \
           --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
           --mem="${SLURM_MEM_PER_NODE}M" \
           --array="${SLURM_ARRAY_TASK_ID}" \
           "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/pretrain_pd.slurm"
    exit 64
fi
[[ "$RC" -eq 0 ]] || exit "$RC"


# NO EVALUATION HERE. Phase 1 produces checkpoints; phase 2 scores them, and it is a
# SEPARATE array so that all 75 arms and the reference column (released TabICLv2,
# CatBoost, logistic/linear) are benchmarked by the same code, on the same day, with the
# same context cap, seeds and splits. Every comparison this project got wrong, it got
# wrong by scoring the two sides through different paths.
#
#     python -m src.utils.sweep_status --config config/Exp1_<TRACK>.yaml   # 75 done?
#     sbatch --clusters=mindwell --export=ALL,TRACK=<track> \
#            scripts/slurm/benchmark.slurm



mkdir -p "${CREDITICL_OUTPUT_ROOT}/.sentinels"
touch "${CREDITICL_OUTPUT_ROOT}/.sentinels/pretrain_ok_pd_${SLURM_ARRAY_TASK_ID}"
