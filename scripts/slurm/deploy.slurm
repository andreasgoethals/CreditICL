#!/bin/bash -l
# One-off supervised deployment requested by Andreas; stops after Exp2 submission.
#SBATCH --job-name=crediticl-deploy
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditICL"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source scripts/slurm/_activate_env.sh
python -m src.utils.deploy --run --record "output CreditICL/manifests/deploy-${SLURM_JOB_ID}.json"
