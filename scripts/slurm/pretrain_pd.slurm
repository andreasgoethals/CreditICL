#!/bin/bash -l
# =============================================================================
#  CreditICL — pretraining array, PD (binary classification, cross-entropy head)
#  TARGET: Mindwell — NVIDIA B200 SXM6, 192 GiB VRAM, 24 cores/GPU
#
#  One array task = one grid point of config/PD.yaml.
#  Check the grid size BEFORE submitting:
#
#      python scripts/pretrain.py --config config/PD.yaml --list
#
#  and set --array to match (default config expands to 48 runs -> 0-47).
#
#  Storage, matching CreditPFN:
#    checkpoints  -> project staging (/lustre1/project/stg_00211), big files
#    metrics/logs -> $VSC_DATA/CreditICL, small + backed up
# =============================================================================

# SINGLE CLUSTER, ON PURPOSE. Slurm job dependencies do not work ACROSS clusters,
# and the pipeline is chained with --dependency=afterok. An earlier version put
# preprocess on wice, the prior arrays on genius and training on mindwell; the chain
# could never have run. Everything in Phase 1 therefore stays on wice, which has both
# `batch` (CPU) and `gpu_a100` (GPU). Mindwell's B200s are for the Phase 2
# multi-GPU confirmation runs, submitted on their own (see pretrain_multigpu.slurm).
#SBATCH --job-name=crediticl-pd
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Propagate the submission env, or a cross-cluster sbatch can hit Slurm's
# login-shell env retrieval ("user env retrieval failed requeued held").
#SBATCH --export=ALL
#SBATCH --array=0-47%12

# Jitter the start so 12 tasks do not hammer the filesystem at once.
sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CREDITICL_OUTPUT_ROOT="${CREDITICL_OUTPUT_ROOT:-${VSC_DATA}/CreditICL}"
mkdir -p "${CREDITICL_OUTPUT_ROOT}/logs" "${CREDITICL_OUTPUT_ROOT}/output"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITICL_OUTPUT_ROOT}/logs/pretrain_pd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

_log_exit() {
    rc=$?
    echo "=========================================="
    if [[ "$rc" -eq 0 ]]; then echo "END status=OK - $(date)"
    else echo "END status=FAILED exit_code=${rc} - $(date)"; fi
    echo "=========================================="
}
trap _log_exit EXIT
trap 'exit 143' TERM INT

cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditICL — PRETRAIN (PD) — Mindwell B200"
echo "=========================================="
echo "Array job/task : ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Cluster        : ${SLURM_CLUSTER_NAME:-?}  partition=${SLURM_JOB_PARTITION:-?}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "Log            : ${LOG}"
python -c "import json; from src.utils.paths import describe; print(json.dumps(describe(), indent=2))"
nvidia-smi || true
echo "=========================================="

# Pre-flight: fail in seconds on a broken env rather than after the queue wait.
python scripts/smoke_test.py --task pd --steps 2 || {
    echo "PREFLIGHT FAILED: PD smoke test. Not starting the real run." >&2
    exit 1
}

# --prior-source pool: read the shards built by stage 2 of the chain. Every arm
# then draws its original-prior share from the SAME files, so a difference between
# arms cannot be luck of the draw. Set PRIOR_SOURCE=generate to build them live.
python -u scripts/pretrain.py \
    --config config/PD.yaml \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --prior-source "${PRIOR_SOURCE:-pool}"

mkdir -p "${CREDITICL_OUTPUT_ROOT}/.sentinels"
touch "${CREDITICL_OUTPUT_ROOT}/.sentinels/pretrain_ok_pd_${SLURM_ARRAY_TASK_ID}"
