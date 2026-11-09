#!/bin/bash -l
# Synthetic mechanism diagnostics, never an Experiment 2/3 continuation.
#SBATCH --job-name=crediticl-cpt-probe
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${VSC_DATA}/CreditICL"
test "$(git rev-parse HEAD)" = "${CREDITICL_EXPECTED_COMMIT:?Reviewed commit required}"
test -z "$(git diff --name-only HEAD -- src config scripts tests)"
source scripts/slurm/_activate_env.sh
# The explicit output is PROJECT; all transient trainer files use native scratch.
python -m src.validation.cpt_probe "$@"
