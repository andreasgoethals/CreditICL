#!/bin/bash -l
# =============================================================================
#  CreditICL — EXPERIMENT 0: the checks that run before any arm does, on the production GPU.
#
#      sbatch --clusters=mindwell scripts/slurm/exp0.slurm
#
#  1. scripts/exp0_checks.py — environment and library versions, both output tiers writable,
#     every credit dataset and the out-of-domain cache readable, the released TabICLv2 and TabPFN-3
#     weights found and loaded (and the file we fine-tune from = the file the reference scores),
#     tasks from every prior path valid and timed, every config against the SLURM array sizes, a
#     few training steps of both trainers on this GPU (speed, memory), one benchmark fold per
#     reference wrapper. Report: output_CreditICL/experiment_0/report/checks.{md,json}.
#  2. scripts/check_equivalence.py — our TabICL trainer against upstream's on the SAME batches
#     from the SAME weights, both tracks, 20 steps on this GPU (bf16: they agree to rounding).
#
#  The rest of Experiment 0 — three tiny arms per track through pretrain_{pd,lgd}.slurm, one of
#  them stopped by SIGUSR1 and resumed, then the benchmark — is submitted separately (docs/VSC.md).
#  EXP stays 0 throughout, so everything lands in output_CreditICL/experiment_0/.
# =============================================================================
#SBATCH --job-name=crediticl-exp0-checks
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EXP=0

OUTPUT_ROOT="${VSC_DATA}/CreditICL/output_CreditICL"   # = src.utils.paths.outputs_dir()
LOGDIR="${OUTPUT_ROOT}/experiment_${EXP}/logs"
mkdir -p "${LOGDIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/exp0_checks_${TS}_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1

_log_exit() {
    rc=$?
    echo "=========================================="
    if [[ "$rc" -eq 0 ]]; then echo "END status=OK - $(date)"
    else echo "END status=FAILED exit_code=${rc} - $(date)"; fi
    echo "=========================================="
}
trap _log_exit EXIT

cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditICL — EXPERIMENT 0 — CHECKS"
echo "=========================================="
echo "Job            : ${SLURM_JOB_ID}   node ${SLURMD_NODENAME:-?}"
echo "Cluster / part : ${SLURM_CLUSTER_NAME:-?} / ${SLURM_JOB_PARTITION:-?}"
echo "Cores / mem    : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MiB"
echo "Commit         : $(git rev-parse --short HEAD 2>/dev/null || echo '?')  dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo YES || echo no)"
echo "Date           : $(date)"
echo "Log            : ${LOG}"
nvidia-smi || true
echo "=========================================="

set +e
python -u scripts/exp0_checks.py
CHECKS_RC=$?

for TRACK in PD LGD; do
    echo
    echo "=== trainer equivalence, ${TRACK}: ours vs upstream, same batches, same weights, 20 steps ==="
    python -u scripts/check_equivalence.py --config "config/Exp1_${TRACK}.yaml" --index 0 --steps 20 --device cuda
    echo "equivalence ${TRACK} exit code: $?"
done
set -e

echo
echo "checks exit code: ${CHECKS_RC} (1 = at least one FAIL; read experiment_0/report/checks.md)"
exit "${CHECKS_RC}"
