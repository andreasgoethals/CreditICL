#!/bin/bash -l
# =============================================================================
#  CreditICL — PIPELINE 2: build a pool of synthetic datasets (CPU array job)
#  TARGET: Genius / wICE CPU nodes. NO GPU, on purpose.
#
#  Prior generation is CPU-bound: random DAGs, eight function families, and an
#  ExtraTrees predictability filter per candidate dataset. A GPU would sit idle
#  waiting on scikit-learn. So this runs on cheap CPU nodes, in parallel, and the
#  GPU nodes do nothing but train.
#
#  One array task = one shard. 20 tasks x 2,000 datasets = 40,000 per variant,
#  with no coordination and no locking, because task i only ever writes shard i.
#
#  Submit one variant at a time (TASK and VARIANT come from the environment):
#
#      sbatch --export=ALL,TASK=lgd,VARIANT=original   scripts/slurm/generate_prior.slurm
#      sbatch --export=ALL,TASK=lgd,VARIANT=credit_v1  scripts/slurm/generate_prior.slurm
#
#  or let scripts/submit_pipeline.sh chain the whole thing for you.
# =============================================================================

#SBATCH --job-name=crediticl-prior
#SBATCH --clusters=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --array=0-19

# Stagger the start: 20 tasks opening the same repo at once is a metadata storm
# on Lustre, and Lustre punishes many small concurrent opens far more than reads.
sleep $((RANDOM % 20 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# One thread per BLAS call. The parallelism here is the array, not the maths;
# without this, 8 shards x 8 BLAS threads oversubscribe the node and everything
# slows down together.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TASK="${TASK:-lgd}"
VARIANT="${VARIANT:-original}"
N_SHARDS="${N_SHARDS:-20}"
N_DATASETS="${N_DATASETS:-40000}"

REPO="${CREDITICL_REPO:-$VSC_DATA/CreditICL}"
cd "$REPO"
source scripts/slurm/_activate_env.sh

CFG="config/$(echo "$TASK" | tr '[:lower:]' '[:upper:]').yaml"

echo "=========================================================="
echo " CreditICL prior generation"
echo "   task        : $TASK"
echo "   variant     : $VARIANT"
echo "   shard       : ${SLURM_ARRAY_TASK_ID:-0} of $N_SHARDS"
echo "   pool total  : $N_DATASETS datasets"
echo "   node        : $(hostname)   cores=${SLURM_CPUS_PER_TASK:-?}"
echo "=========================================================="

python scripts/generate_prior.py \
    --config "$CFG" \
    --variant "$VARIANT" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-shards "$N_SHARDS" \
    --n-datasets "$N_DATASETS"

echo "shard ${SLURM_ARRAY_TASK_ID:-0} finished at $(date)"
