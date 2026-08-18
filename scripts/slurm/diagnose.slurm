#!/bin/bash -l
# =============================================================================
#  Run the NaN diagnosis ON A GPU, which is where the NaN actually happens.
#
#      bash scripts/slurm/submit.sh free lgd scripts/slurm/diagnose.slurm
#      bash scripts/slurm/submit.sh b200 lgd scripts/slurm/diagnose.slurm
#
#  WHY A JOB AND NOT A LOGIN NODE. Run on the login node, the same checkpoint and
#  the same four datasets came back **completely finite** — including
#  `base_modelisation`, which training reports as 100 % NaN. The only difference in
#  the log was `device=cpu`.
#
#  That points at the attention kernel. PyTorch picks a CUDA backend by itself
#  (flash, mem-efficient, cuDNN or math) and the choice depends on shapes and
#  dtype; CPU only ever has `math`. Upstream treats the choice as load-bearing too
#  — `--use_flash_attn3 False` in stage 1, `True` in stages 2 and 3 — and we do not
#  control it at all.
#
#  So this runs every backend in turn on the real GPU. If `math` is finite and
#  `flash` is not, the cause is named and the fix is one context manager.
#
#  Cheap: one task, one forward pass per dataset per backend, 15 minutes.
#  Run it on `free` first — it costs nothing and the RTX 5000 Ada is a different
#  architecture from the B200, which is itself worth knowing.
# =============================================================================

#SBATCH --job-name=crediticl-diagnose
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=interactive
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --array=0

set -euo pipefail

REPO="${VSC_DATA}/CreditICL"
cd "$REPO"
source scripts/slurm/_activate_env.sh

CONFIG="${1:-config/Exp1_LGD.yaml}"
case "$CONFIG" in
    *_PD.yaml) TASK=pd  ;;
    *)         TASK=lgd ;;
esac

LOGDIR="${VSC_DATA}/CreditICL/output/logs"
mkdir -p "$LOGDIR"
# The PARTITION is in the name: the whole point is comparing GPUs, so a filename that
# does not say which card it was is useless as soon as the second one lands.
STDOUT="${LOGDIR}/diagnose_${SLURM_JOB_PARTITION:-unknown}_${SLURM_JOB_ID}.out"
exec > >(tee -a "$STDOUT") 2>&1

echo "=============================================================="
echo " CreditICL NaN DIAGNOSIS — on a GPU"
echo " job          : ${SLURM_JOB_ID}"
echo " cluster/part : ${SLURM_CLUSTER_NAME:-?} / ${SLURM_JOB_PARTITION:-?}"
echo " node         : $(hostname)"
echo " task         : ${TASK}"
echo " started      : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo

# The three datasets that report all-NaN, then `axa` as a control that does not.
# Every attention backend is tried on each; `--sdpa-backend all` is the default.
DATASETS="${DATASETS:-0005.base_modelisation,0004.base_model,0002.loss2,0003.axa}"
echo "datasets: ${DATASETS}"
echo

python scripts/diagnose_nan.py --task "$TASK" --datasets "$DATASETS"

echo
echo "finished: $(date -Is)"
echo "compare against the CPU run: if math is finite and flash is not, that is the answer."
