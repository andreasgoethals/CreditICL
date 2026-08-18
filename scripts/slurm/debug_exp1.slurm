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
#  failure modes it finds are the same. 600 steps x batch 64 finishes inside it;
#  1,500 did NOT once batch_size became 64 — job 11518236 died at the wall on step 1,422.
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
# Resolved from `--list` against config/Exp1_LGD.yaml on 17-08-2026, all at seed 0:
#   0  cf=0.0  mechanism   the control: original prior only
#   3  cf=0.3  mechanism   our loss-mechanism path at the highest mix
#   9  cf=0.3  quantile    our marginal-shaping path
#   1  cf=0.1  mechanism   the mixing path at a low share
# RE-RESOLVE THESE IF THE SWEEP CHANGES. An index that silently points at a different arm
# is worse than a crash, which is why the mapping is written out rather than computed.
# The quantile arm moved 11 -> 9 on 17-08-2026 when the sweep dropped 21 duplicate control
# runs (96 -> 75). `test_debug_arms_point_where_the_comments_say` now checks this pairing.
ARMS=(0 3 9 1)
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
# 600, not 1500. The budget is `steps x batch_size`, and batch_size went from 4 to upstream's
# 64 — so 1,500 steps is now 96,000 datasets, not 6,000. Job 11518236 was KILLED at the
# one-hour walltime on step 1,422 of 1,500, losing the evaluation and the final checkpoint
# after 62 minutes of training. 600 steps is 38,400 datasets, ~25 minutes, and still gives
# seven points on the progress curve. A debug run has to FINISH.
DEBUG_STEPS="${DEBUG_STEPS:-600}"
echo "DEBUG: overriding max_steps to ${DEBUG_STEPS} (config says 12500)"

# WORKERS FROM THE ALLOCATION, not from the config. This script runs on partitions with
# 8 cores (mindwell interactive) through 24 (gpu_b200), and the config's fixed 12 is wrong
# on both: it oversubscribes the small one and wastes half the big one. TabICL generates
# its prior on the CPU, so this is the setting that actually decides throughput.
# One core is left for the main process and the GPU feed.
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"
[ "$WORKERS" -lt 0 ] && WORKERS=0
echo "DEBUG: num_workers=${WORKERS} (from SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset})"

# MICRO-BATCH: UPSTREAM'S 4, and the measurement says leave it alone.
#
# It is tempting to raise this on a 183 GB card — peak memory is 6.6 GB — but the B200 data
# says the micro-batch was never the problem. BOTH slow and fast runs used micro_batch 4:
#
#     batch  4, micro 4  ->  1 pass per sync  ->   3.5 % utilisation,  2.2 datasets/s
#     batch 64, micro 4  -> 16 passes per sync -> 88.7 % utilisation, 25.6 datasets/s
#
# What fills a big GPU is the number of CONSECUTIVE passes before a synchronising optimiser
# step, not how many datasets each pass carries. Upstream's `--batch_size 64 --micro_batch_size
# 4` is exactly the good case, and their micro-batch drops to 1 in stages 2 and 3 only because
# the tables grow to 60,000 rows — it tracks sequence length, not hardware.
#
# At 88.7 % there is at most ~1.1x left, so a deviation from upstream would buy almost nothing
# and cost a difference nobody asked for. Override with MICRO= if a run says otherwise; the
# telemetry summary prints the headroom.
MICRO="${MICRO:-4}"
echo "DEBUG: micro_batch_size=${MICRO} on ${SLURM_JOB_PARTITION:-unknown} (memory/speed only)"

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
    --micro-batch-size "$MICRO" \
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

    # ROW CAP, FOR THE DEBUG RUN ONLY. The 14 PD datasets total 2.4 GB and
    # `0014.algorithmwatch` alone is 158,700 x 2,986 float32 = 1.8 GB; loading it, splitting
    # it and imputing it makes several more copies. That is what killed all four arms of job
    # 11517006 with OUT_OF_MEMORY on this 30 GB partition — and it only appeared once
    # preprocessing had succeeded, because before that the evaluation had nothing to load.
    # THE REAL EXP1 MUST NOT SET THIS. A capped evaluation is a plumbing test, not a result;
    # the cap is recorded in every row so the two can never be confused.
    DEBUG_EVAL_ROWS="${DEBUG_EVAL_ROWS:-20000}"
    echo "DEBUG: capping evaluation at ${DEBUG_EVAL_ROWS} rows/dataset (memory, not science)"

    echo "=== evaluating on the real credit datasets ==="
    python scripts/evaluate.py \
        --task "$TASK" \
        --models "$MODELS" \
        "${CKPT_ARG[@]}" \
        --max-rows "$DEBUG_EVAL_ROWS" \
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
