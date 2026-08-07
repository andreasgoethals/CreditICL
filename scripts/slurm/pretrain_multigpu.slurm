#!/bin/bash -l
# =============================================================================
#  CreditICL — FULL-BUDGET confirmation run, multi-GPU on one node.
#  TARGET: Mindwell gpu_b200 (8x B200 SXM6, 192 GiB each, NVLink, 192 cores/node)
#
#  WHY THIS SCRIPT EXISTS
#
#  A full TabICLv2-budget checkpoint costs 24.5 GPU-days (the paper's own figure).
#  VSC caps a job at 72 hours = 3 days. So:
#
#      1 GPU  -> 24.5 days -> 9 chained jobs
#      4 GPUs ->  6.1 days -> 3 chained jobs
#      8 GPUs ->  3.1 days -> 2 chained jobs   <- what this script does
#
#  The parallelism is over the batch, and nothing couples the datasets in a batch,
#  so splitting it across GPUs is mathematically identical to one big batch. The
#  effective batch — and therefore the compute budget — is unchanged; see
#  src/train/distributed.py.
#
#  USE THIS FOR PHASE 2 ONLY (2 checkpoints: best arm + control). Phase 1's 48-arm
#  sweep at this budget would be 1,056 GPU-days = 2.9 GPU-years, which no amount of
#  credit makes fit in a PhD.
#
#      TASK=lgd sbatch scripts/slurm/pretrain_multigpu.slurm
#
#  Resume is REQUIRED here, not optional: submit the follow-on job with
#      sbatch --dependency=afterany:<jobid> scripts/slurm/pretrain_multigpu.slurm
#  and it picks up from the last checkpoint.
# =============================================================================

#SBATCH --job-name=crediticl-full
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK="${TASK:-lgd}"
NGPU="${NGPU:-4}"
INDEX="${INDEX:-0}"

REPO="${CREDITICL_REPO:-$VSC_DATA/CreditICL}"
export CREDITICL_OUTPUT_ROOT="${CREDITICL_OUTPUT_ROOT:-${VSC_DATA}/CreditICL}"
mkdir -p "${CREDITICL_OUTPUT_ROOT}/logs"
LOG="${CREDITICL_OUTPUT_ROOT}/logs/slurm_full_${TASK}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1
echo "slurm wrapper log: ${LOG}"

cd "$REPO"
source scripts/slurm/_activate_env.sh

CFG="config/$(echo "$TASK" | tr '[:lower:]' '[:upper:]').yaml"

echo "=========================================================="
echo " CreditICL FULL-BUDGET run"
echo "   task    : $TASK   grid index: $INDEX"
echo "   GPUs    : $NGPU   node: $(hostname)"
echo "   cores   : ${SLURM_CPUS_PER_TASK:-?}"
echo "   walltime: ${SBATCH_TIMELIMIT:-72:00:00}"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "(no GPU visible)"

# One BLAS thread per prior worker. The parallelism is the DataLoader workers and
# the GPUs, not the maths; oversubscribing slows every worker at once.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# torchrun, NOT srun: VSC's docs note srun inside a job can conflict with the
# scheduler's task management, while torchrun brings its own rendezvous.
# --standalone keeps the rendezvous on this node, which is all a single-node job needs.
# No --resume flag: pretrain.py always calls trainer.maybe_resume(), which picks up
# the latest checkpoint in the run's directory if one is there. So re-submitting this
# script IS the resume — which is what makes the chained-job pattern work.
torchrun --standalone --nproc_per_node="${NGPU}" \
    -m scripts.pretrain \
    --config "$CFG" \
    --index "$INDEX" \
    --prior-source "${PRIOR_SOURCE:-generate}"

echo "finished at $(date)"
