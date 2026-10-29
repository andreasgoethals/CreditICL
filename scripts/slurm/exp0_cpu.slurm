#!/bin/bash -l
# Experiment 0 CPU: preprocessing, portable tests, prior/data/package checks.
# Submit this job first; the GPU job requires its current PASS receipt.
#SBATCH --job-name=crediticl-exp0-cpu
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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
LOG="${LOGDIR}/exp0_cpu_${TS}_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1

EXP0_PHASE=cpu
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
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "=========================================="

run_exp0_validation --phase cpu
