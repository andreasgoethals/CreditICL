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
# NOT 72:00:00. Billing is on ACTUAL time, so a long request costs nothing extra — but the
# backfill scheduler starts short jobs sooner, and this job is resumable, so a tight request
# is strictly better. Raise it if an arm turns out longer than ~20 h.
#SBATCH --time=24:00:00
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
echo "Walltime       : ${SBATCH_TIMELIMIT:-24:00:00}   requeue=yes  signal=USR1@600"
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
python -u scripts/pretrain.py \
    --config config/Exp1_PD.yaml \
    --index "${SLURM_ARRAY_TASK_ID}" \
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
    sbatch --clusters=mindwell --array="${SLURM_ARRAY_TASK_ID}" \
           "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/pretrain_pd.slurm"
    exit 64
fi
[[ "$RC" -eq 0 ]] || exit "$RC"

mkdir -p "${CREDITICL_OUTPUT_ROOT}/.sentinels"
touch "${CREDITICL_OUTPUT_ROOT}/.sentinels/pretrain_ok_pd_${SLURM_ARRAY_TASK_ID}"
