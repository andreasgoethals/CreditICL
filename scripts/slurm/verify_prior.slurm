#!/bin/bash -l
# =============================================================================
#  CreditICL — the gate between prior generation and training.
#
#  Checks that every pool is complete AND that they all hold the same number of
#  datasets. If they do not, this exits non-zero, and because training is
#  submitted with --dependency=afterok it never starts.
#
#  That matters: "matched compute" is the whole basis of the comparison. If the
#  original pool held 40,000 datasets and ours held 38,400 because two array
#  tasks died, the experiment would still run and still produce numbers — just
#  meaningless ones. Better to stop here.
# =============================================================================

# SINGLE CLUSTER, ON PURPOSE. Slurm job dependencies do not work ACROSS clusters,
# and the pipeline is chained with --dependency=afterok. An earlier version put
# preprocess on wice, the prior arrays on genius and training on mindwell; the chain
# could never have run. Everything in Phase 1 therefore stays on wice, which has both
# `batch` (CPU) and `gpu_a100` (GPU). Mindwell's B200s are for the Phase 2
# multi-GPU confirmation runs, which Exp2 will need and which do not exist yet.
#SBATCH --job-name=crediticl-verify
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

TASK="${TASK:-lgd}"
N_DATASETS="${N_DATASETS:-40000}"

REPO="${CREDITICL_REPO:-$VSC_DATA/CreditICL}"
cd "$REPO"
source scripts/slurm/_activate_env.sh

CFG="config/$(echo "$TASK" | tr '[:lower:]' '[:upper:]').yaml"

echo "verifying $TASK pools hold exactly $N_DATASETS datasets each"
python scripts/generate_prior.py --config "$CFG" --status --n-datasets "$N_DATASETS"
echo "pools verified at $(date) — training may start"
