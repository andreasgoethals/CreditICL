#!/bin/bash -l
# =============================================================================
#  Run scripts/benchmark_gpu.py on whichever card SLURM gives us.
#
#  Submit it TWICE — the whole point is the comparison:
#      bash scripts/slurm/submit.sh free pd scripts/slurm/benchmark.slurm
#      bash scripts/slurm/submit.sh b200 pd scripts/slurm/benchmark.slurm
#
#  Then diff the two logs row by row. The reading is printed at the bottom of
#  each: raw matmul differing means the card or the wheel; only the end-to-end
#  step differing means the prior generator, and no GPU will fix it.
#
#  Fifteen minutes and one array task, so it is cheap even on a paid partition.
# =============================================================================

#SBATCH --job-name=crediticl-gpubench
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=interactive
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
# ONE task. This is a measurement, not a sweep.
#SBATCH --array=0

set -euo pipefail

REPO="${VSC_DATA}/CreditICL"
cd "$REPO"
source scripts/slurm/_activate_env.sh

# Argument 1 is the config, for symmetry with debug_exp1.slurm — only its task is used.
CONFIG="${1:-config/Exp1_PD.yaml}"
case "$CONFIG" in
    *_PD.yaml) TASK=pd ;;
    *)         TASK=lgd ;;
esac

LOGDIR="${VSC_DATA}/CreditICL/output/logs"
mkdir -p "$LOGDIR"
# The PARTITION is in the filename: the two runs differ only by hardware, so a name that
# does not say which card it was is useless the moment the second one lands.
STDOUT="${LOGDIR}/gpubench_${SLURM_JOB_PARTITION:-unknown}_${SLURM_JOB_ID}.out"
exec > >(tee -a "$STDOUT") 2>&1

echo "=============================================================="
echo " CreditICL GPU BENCHMARK"
echo " job          : ${SLURM_JOB_ID}"
echo " cluster/part : ${SLURM_CLUSTER_NAME:-?} / ${SLURM_JOB_PARTITION:-?}"
echo " node         : $(hostname)"
echo " cores        : ${SLURM_CPUS_PER_TASK:-?}"
echo " task         : ${TASK}"
echo " started      : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo

python scripts/benchmark_gpu.py \
    --task "$TASK" \
    --json "${LOGDIR}/gpubench_${SLURM_JOB_PARTITION:-unknown}_${SLURM_JOB_ID}.json"

echo
echo "compare the two cards with:"
echo "  ls ${LOGDIR}/gpubench_*.json"
echo "  python -m src.utils.compare_gpubench ${LOGDIR}/gpubench_*.json"
echo "finished: $(date -Is)"
