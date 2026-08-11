#!/bin/bash -l
# =============================================================================
#  CreditICL — PIPELINE 4: score the baselines on every real credit dataset.
#
#      sbatch scripts/slurm/evaluate.slurm
#
#  Needs a GPU because TabPFN-3 and TabICLv2 are transformers; CatBoost and the
#  linear models ignore it. wICE gpu_a100 is the cheapest GPU that fits
#  (141.667 credits/GPU-min vs 437.50 for B200 — see docs/VSC.md).
#
#  BEFORE THE FIRST RUN, from a LOGIN node:
#    1) export TABPFN_TOKEN=<your key from https://ux.priorlabs.ai/account>
#    2) python -c "from tabpfn import TabPFNClassifier; TabPFNClassifier(model_path='v3')"
#       to pull the weights into the cache — compute nodes have no internet, so
#       an un-cached TabPFN will fail here no matter what the token says.
#    3) sbatch scripts/slurm/preprocess.slurm   (so this job does not preprocess)
# =============================================================================

#SBATCH --job-name=cicl-eval
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CREDITICL_OUTPUT_ROOT="${CREDITICL_OUTPUT_ROOT:-${VSC_DATA}/CreditICL}"
mkdir -p "${CREDITICL_OUTPUT_ROOT}/output/logs"
LOG="${CREDITICL_OUTPUT_ROOT}/output/logs/slurm_evaluate_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1
echo "slurm wrapper log: ${LOG}"

cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

nvidia-smi || echo "(no GPU visible — the TFM baselines will be slow or fail)"

# TabPFN-3 reads a local checkpoint file, so no token and no download is needed.
# If the file is missing the baseline is skipped and everything else still runs.
if [[ -n "${CREDITICL_TABPFN_DIR:-}" ]]; then
    echo "TabPFN checkpoints: ${CREDITICL_TABPFN_DIR}"
    ls -la "${CREDITICL_TABPFN_DIR}" 2>/dev/null || echo "  (directory not readable)"
else
    echo "CREDITICL_TABPFN_DIR unset — falling back to project staging, then checkpoints/"
fi

python -u scripts/evaluate.py --task "${TASK:-both}" --seeds 0,1,2
