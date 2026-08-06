#!/bin/bash -l
# =============================================================================
#  CreditICL — PIPELINE 1: preprocess the raw credit datasets.
#
#  CPU only and free-ish: this is pure pandas work, no GPU needed. Run it BEFORE
#  submitting an eval array, so 48 tasks do not each try to preprocess Home
#  Credit (307k rows) at the same time.
#
#      sbatch scripts/slurm/preprocess.slurm
#
#  Processed data lands on PROJECT STORAGE
#  (/lustre1/project/stg_00211/CreditICL/data/processed/), never in $VSC_DATA.
# =============================================================================

#SBATCH --job-name=cicl-preprocess
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export CREDITICL_OUTPUT_ROOT="${CREDITICL_OUTPUT_ROOT:-${VSC_DATA}/CreditICL}"
mkdir -p "${CREDITICL_OUTPUT_ROOT}/logs"
LOG="${CREDITICL_OUTPUT_ROOT}/logs/slurm_preprocess_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1
echo "slurm wrapper log: ${LOG}"

cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

# TASK defaults to both, so running this on its own still does everything;
# submit_pipeline.sh sets it per task so chaining both does not preprocess twice.
python -u scripts/preprocess.py --task "${TASK:-both}"
