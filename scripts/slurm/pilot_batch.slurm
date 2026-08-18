#!/bin/bash -l
# =============================================================================
#  BATCH-SIZE PILOT — settle `batch_size` on the control arm, before Exp1.
#
#      bash scripts/slurm/submit.sh b200 lgd scripts/slurm/pilot_batch.slurm
#      bash scripts/slurm/submit.sh b200 pd  scripts/slurm/pilot_batch.slurm
#
#  THREE runs of the CONTROL arm (`credit_fraction: 0.0`, grid index 0) at three
#  batch sizes, full 12,500 steps, one seed. Nothing else varies.
#
#  WHY A PILOT AND NOT A SWEEP LEVER. Batch size is a nuisance parameter: we need
#  a good value, we are not studying it. Adding it to Exp1's grid would take 75
#  arms to 150 and halve the power of the comparison Exp1 actually asks — and at
#  a fixed learning rate it would be confounded anyway, since batch size and LR
#  are coupled. Three runs settle it for 4% of that cost.
#
#  WHAT TO READ AFTERWARDS. Final `train_loss`, and the progress curve in
#  `output/manifests/*__progress.csv`. Pick the batch size whose curve is lowest
#  AND still falling, then freeze it in every `Exp*.yaml`.
#
#  NOT ON A LOGIN NODE. 12,500 steps is hours of compute; a login node has no GPU
#  (`cuda_available: false`, so `amp=False` too) and is shared with everyone else
#  on the cluster. Always via sbatch.
# =============================================================================

#SBATCH --job-name=crediticl-pilot-batch
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
# One task per batch size.
#SBATCH --array=0-2

set -euo pipefail

REPO="${VSC_DATA}/CreditICL"
cd "$REPO"
source scripts/slurm/_activate_env.sh

CONFIG="${1:-config/Exp1_LGD.yaml}"

# The three candidates. 64 is upstream's; 16 and 256 bracket it by a factor of four
# either way, which is wide enough to show a trend and narrow enough to stay trainable
# at one learning rate.
BATCHES=(16 64 256)
BATCH="${BATCHES[$SLURM_ARRAY_TASK_ID]}"

# Datasets per forward pass — memory only, never the result. The summary at the end of
# the log prints the peak memory and suggests a larger value if there is room.
MICRO="${MICRO:-16}"
[ "$MICRO" -gt "$BATCH" ] && MICRO="$BATCH"

WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"

LOGDIR="${VSC_DATA}/CreditICL/output/logs"
mkdir -p "$LOGDIR"
exec > >(tee -a "${LOGDIR}/pilot_batch_${BATCH}_${SLURM_JOB_ID}.out") 2>&1

echo "=============================================================="
echo " CreditICL BATCH-SIZE PILOT — control arm only"
echo " job        : ${SLURM_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}"
echo " partition  : ${SLURM_JOB_PARTITION:-?}   node $(hostname)"
echo " config     : ${CONFIG}   grid index 0 (the control, credit_fraction=0)"
echo " batch_size : ${BATCH}    micro_batch_size ${MICRO}   workers ${WORKERS}"
echo " started    : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo

# The CONTROL arm, so no credit lever is in play and the only difference between the
# three tasks is the batch size.
python scripts/pretrain.py \
    --config "$CONFIG" \
    --index 0 \
    --batch-size "$BATCH" \
    --micro-batch-size "$MICRO" \
    --num-workers "$WORKERS"

echo
echo "finished: $(date -Is)"
echo "compare the three with:"
echo "  grep -h 'step  *12500' ${LOGDIR}/pilot_batch_*.out"
echo "  ls ${VSC_DATA}/CreditICL/output/manifests/*__progress.csv"
