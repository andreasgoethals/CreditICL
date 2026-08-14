#!/bin/bash -l
# =============================================================================
#  CreditICL — EXP1 DEBUG RUN. Four arms, short walltime, maximum logging.
#
#  THIS IS NOT THE EXPERIMENT. It is the run that finds out whether the pipeline
#  works on the cluster at all, and what its behaviour looks like: throughput,
#  GPU utilisation, memory headroom, whether the progress curve fires, whether
#  the real and out-of-domain evaluations run, and how big the artefacts get.
#
#  Four arms chosen to cover the code paths that can differ, NOT to answer
#  anything scientific:
#      task 0  credit_fraction 0.0  -> the CONTROL: original prior only
#      task 1  credit_fraction 0.3  mechanism  -> our loss-mechanism path
#      task 2  credit_fraction 0.3  quantile   -> our marginal-shaping path
#      task 3  credit_fraction 0.1  mechanism  -> the mixing path at a low share
#
#  Pick the indices from the grid listing before submitting:
#      python scripts/pretrain.py --config config/Exp1_LGD.yaml --list
#
#  WALLTIME IS 1 HOUR, not 72. A debug run that sits in the queue for a day
#  teaches nothing; a short one backfills into gaps a long one cannot, and the
#  failure modes it finds are the same. 1,500 steps finishes well inside it.
# =============================================================================

#SBATCH --job-name=crediticl-debug-exp1
# DEFAULTS = the FREE target. `scripts/slurm/submit.sh` overrides these on the
# command line for a different partition, and command-line options win over
# `#SBATCH` directives.
#
# NOT gpu_a100. That partition has 16 GPUs — the fewest available to us — and the
# first debug submission sat there behind "Reason: Priority" without ever starting.
# Mindwell's interactive partition costs NO credits, gives a full RTX 5000 Ada
# (32 GiB), allows 16 h, and is not competing with production training runs.
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=interactive
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Propagate the submission env, or a cross-cluster sbatch can hit Slurm's
# login-shell env retrieval ("user env retrieval failed requeued held").
#SBATCH --export=ALL
# Four arms, all at once: the point is to see them side by side.
#SBATCH --array=0-3

set -euo pipefail

REPO="${VSC_DATA}/CreditICL"
cd "$REPO"

# --- environment ------------------------------------------------------------
# Names every missing package rather than printing the first three lines of a
# traceback, which are always the same boilerplate. See docs/AGENTS_MEMORY.md.
source scripts/slurm/_activate_env.sh

# --- which arm is this? -----------------------------------------------------
# TASK -> grid index. Read the mapping from `--list` and edit if the grid changes:
# a hard-coded index that silently points at a different arm is worse than a crash.
# THE CONFIG IS ARGUMENT 1, not an environment variable. `sbatch [options] script args...`
# passes trailing arguments straight to the script, so this needs no `--export` plumbing and
# cannot be silently lost. `CONFIG=x bash submit.sh` set the variable for the CALLING shell,
# not for sbatch's environment, so the job never saw it.
CONFIG="${1:-config/Exp1_LGD.yaml}"
# Resolved from `--list` against config/Exp1_LGD.yaml on 12-08-2026, all at seed 0:
#   0  cf=0.0  mechanism   the control: original prior only
#   3  cf=0.3  mechanism   our loss-mechanism path at the highest mix
#  11  cf=0.3  quantile    our marginal-shaping path
#   1  cf=0.1  mechanism   the mixing path at a low share
# RE-RESOLVE THESE IF THE SWEEP CHANGES. An index that silently points at a different arm
# is worse than a crash, which is why the mapping is written out rather than computed.
ARMS=(0 3 11 1)
INDEX="${ARMS[$SLURM_ARRAY_TASK_ID]}"

LOGDIR="${VSC_DATA}/CreditICL/output/logs"
mkdir -p "$LOGDIR"
STDOUT="${LOGDIR}/debug_exp1_task${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out"

# Everything to one file per arm, so a whole arm can be pasted into a chat.
exec > >(tee -a "$STDOUT") 2>&1

echo "=============================================================="
echo " CreditICL Exp1 DEBUG"
echo " job            : ${SLURM_JOB_ID}  array task ${SLURM_ARRAY_TASK_ID}"
echo " cluster/part   : ${SLURM_CLUSTER_NAME:-?} / ${SLURM_JOB_PARTITION:-?}"
echo " node           : $(hostname)"
echo " config         : ${CONFIG}   grid index ${INDEX}"
echo " started        : $(date -Is)"
echo "=============================================================="

# The GPU this landed on. Two runs on different cards are not comparable, and
# this is the only record of which.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo

# What the arm actually is. Printed BEFORE training, so a job that dies early still says
# which configuration died. `--dry-run` resolves the config and exits without training.
python scripts/pretrain.py --config "$CONFIG" --index "$INDEX" --dry-run || true
echo

# --- the run ----------------------------------------------------------------
# `--max-steps` OVERRIDES the config's 12,500. This is the single most important line in the
# file: a debug run has to FINISH, including the evaluation that follows it, or it proves
# nothing about the half of the pipeline that has never executed on the cluster. 1,500 steps is
# enough to see throughput, GPU utilisation, memory headroom, and three progress-curve points.
#
# Raise it once, to the config's real value, only after this run comes back clean.
DEBUG_STEPS="${DEBUG_STEPS:-1500}"
echo "DEBUG: overriding max_steps to ${DEBUG_STEPS} (config says 12500)"

# WORKERS FROM THE ALLOCATION, not from the config. This script runs on partitions with
# 8 cores (mindwell interactive) through 24 (gpu_b200), and the config's fixed 12 is wrong
# on both: it oversubscribes the small one and wastes half the big one. TabICL generates
# its prior on the CPU, so this is the setting that actually decides throughput.
# One core is left for the main process and the GPU feed.
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"
[ "$WORKERS" -lt 0 ] && WORKERS=0
echo "DEBUG: num_workers=${WORKERS} (from SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset})"

# RESUMING IS AUTOMATIC — there is no flag. `pretrain.py` calls `trainer.maybe_resume()`
# unconditionally, which picks up the latest checkpoint in the run's own directory if one
# exists and starts at step 0 if not. A job killed at the walltime is resumed by resubmitting
# the same command. Passing a `--resume auto` that the parser does not define made argparse
# exit 2 before a single step ran; see docs/AGENTS_MEMORY.md.
#
# `set +e` around the call so a crash does NOT abort the script: STATUS is what the evaluation
# below is gated on, and the artefact summary at the end is most useful precisely when training
# has failed. Under `set -e` the script would die here and print none of it.
set +e
python scripts/pretrain.py \
    --config "$CONFIG" \
    --index "$INDEX" \
    --max-steps "$DEBUG_STEPS" \
    --num-workers "$WORKERS"
STATUS=$?
set -e
echo
echo "training exited with ${STATUS} at $(date -Is)"
if [ "$STATUS" -ne 0 ]; then
    echo "TRAINING FAILED — the evaluation below is SKIPPED. The traceback is above."
fi

# --- evaluate the checkpoint this arm just produced -------------------------
# The debug run is only useful if it proves the WHOLE chain works, and evaluation
# is the half that has never run on the cluster.
if [ "$STATUS" -eq 0 ]; then
    # `crediticl` is the baseline that scores OUR checkpoints — it discovers them under
    # checkpoints/ on project storage. `tabiclv2` is the released model, included so the debug
    # run also proves the comparison TABLE builds, not only that our model scores.
    # Task and out-of-domain KIND follow the config. Hard-coding `lgd` meant a
    # `CONFIG=config/Exp1_PD.yaml` run trained PD and then evaluated LGD — which would have
    # "passed" while measuring the wrong thing entirely.
    case "$CONFIG" in
        *_PD.yaml) TASK=pd;  OOD_KIND=classification ;;
        *)         TASK=lgd; OOD_KIND=regression ;;
    esac
    echo "evaluating as task=${TASK}, out-of-domain kind=${OOD_KIND}"

    # THIS ARM'S OWN CHECKPOINT. `--models crediticl` needs a path: registering the baseline
    # only makes the NAME resolvable, and with no checkpoint every one of its cells failed
    # while the run still printed "25/50 cells OK". Auto-discovery deliberately refuses to
    # choose once several arms exist, which is every run after the first — so the job, which
    # knows exactly which arm it just trained, passes the path itself.
    # The directory comes from `--dry-run`, so it cannot drift from what training used.
    CKPT_DIR="$(python scripts/pretrain.py --config "$CONFIG" --index "$INDEX" --dry-run \
        | sed -n 's/^checkpoints *: *//p')"
    CKPT="$(ls -1 "${CKPT_DIR}"/step-*.ckpt 2>/dev/null | sort -t- -k2 -n | tail -1)"
    if [ -z "$CKPT" ]; then
        echo "WARNING: no checkpoint under ${CKPT_DIR} — 'crediticl' cannot be scored"
        MODELS=tabiclv2
        CKPT_ARG=()
    else
        echo "scoring our checkpoint: ${CKPT}"
        MODELS=crediticl,tabiclv2
        CKPT_ARG=(--checkpoint "$CKPT")
    fi

    echo "=== evaluating on the real credit datasets ==="
    python scripts/evaluate.py \
        --task "$TASK" \
        --models "$MODELS" \
        "${CKPT_ARG[@]}" \
        --seeds 0 \
        --tag "debug_${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}" || \
        echo "WARNING: credit evaluation failed — see the traceback above"

    echo "=== evaluating out-of-domain ==="
    python scripts/evaluate_ood.py \
        --models "$MODELS" \
        "${CKPT_ARG[@]}" \
        --task "$TASK" \
        --seeds 0 \
        --kinds "$OOD_KIND" \
        --tag "debug_${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}" || \
        echo "WARNING: out-of-domain evaluation failed (fetch the cache on a login node first)"
fi

# --- what to collect --------------------------------------------------------
echo
echo "=============================================================="
echo " ARTEFACTS"
echo "=============================================================="
du -sh "${VSC_DATA}/CreditICL/output"/* 2>/dev/null || true
echo
echo "telemetry rows:"
wc -l "${VSC_DATA}/CreditICL/output/manifests/"*telemetry.csv 2>/dev/null || echo "  none"
echo "progress rows:"
wc -l "${VSC_DATA}/CreditICL/output/manifests/"*progress.csv 2>/dev/null || echo "  none"
echo
echo "Collect with:"
echo "  tar czf ~/crediticl_debug_${SLURM_JOB_ID}.tar.gz \\"
echo "      -C \"${VSC_DATA}/CreditICL\" output/logs output/manifests"
echo "finished: $(date -Is)"
