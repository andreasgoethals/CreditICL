#!/bin/bash -l
# =============================================================================
#  CreditICL - EXPERIMENT 0: the checks that run before any arm does, on the production GPU.
#
#      sbatch --clusters=mindwell scripts/slurm/exp0.slurm
#
#  GPU allocation after CPU validation: mandatory CUDA checks, isolated model smoke tests,
#  upstream trainer equivalence, six short runs with forced stop/resume, and
#  selected full-row credit/OOD benchmark receipts. Resubmission continues the same
#  source-bound campaign; completed stages/folds are verified and reused.
#  PASS is bound to code, packages, inputs and artifacts; any failure blocks Exp1.
# =============================================================================
#SBATCH --job-name=crediticl-exp0-gpu
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EXP=0

OUTPUT_ROOT="${VSC_DATA}/CreditICL/output CreditICL"   # = src.utils.paths.outputs_dir()
LOGDIR="${OUTPUT_ROOT}/experiment_${EXP}/logs"
mkdir -p "${LOGDIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/exp0_gpu_${TS}_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1

EXP0_PHASE=gpu
source "${VSC_DATA}/CreditICL/scripts/slurm/_exp0_job.sh"

cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditICL - EXPERIMENT 0 - CHECKS"
echo "=========================================="
echo "Job            : ${SLURM_JOB_ID}   node ${SLURMD_NODENAME:-?}"
echo "Cluster / part : ${SLURM_CLUSTER_NAME:-?} / ${SLURM_JOB_PARTITION:-?}"
echo "Cores / mem    : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MiB"
echo "Commit         : $(git rev-parse --short HEAD 2>/dev/null || echo '?')  dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo YES || echo no)"
echo "Date           : $(date)"
echo "Log            : ${LOG}"
nvidia-smi || true
echo "=========================================="

run_exp0_validation --phase gpu --resume
