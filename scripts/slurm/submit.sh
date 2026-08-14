#!/bin/bash
# =============================================================================
#  Submit a CreditICL job to the RIGHT partition, without editing any script.
#
#      bash scripts/slurm/submit.sh free   scripts/slurm/debug_exp1.slurm
#      bash scripts/slurm/submit.sh b200   scripts/slurm/debug_exp1.slurm
#      bash scripts/slurm/submit.sh a100   scripts/slurm/debug_exp1.slurm
#
#      bash scripts/slurm/submit.sh --list          # show the inventory and exit
#      DRY_RUN=1 bash scripts/slurm/submit.sh free scripts/slurm/debug_exp1.slurm
#
#  Command-line options override `#SBATCH` directives, which is what lets one job
#  script serve every target: the script carries safe defaults, this chooses the
#  hardware.
#
#  WHY THIS EXISTS. The first debug submission sat in `gpu_a100` behind
#  "Reason: Priority" and never started. That partition has the FEWEST GPUs of
#  any GPU partition available to us (16), while `gpu_b200` has 24 and the
#  interactive partitions are free and barely contended.
#
#  Read out of tfm-library/repositories/VSC Documentation.txt:
#
#   target | cluster  | partition   | GPUs | cores/GPU | mem/GPU  | credits | max t
#   -------|----------|-------------|------|-----------|----------|---------|------
#   free   | mindwell | interactive |  4*  |         8 |    ~32G  | NONE    |  16h
#   b200   | mindwell | gpu_b200    | 24   |        24 |  194400M | 26250/h |  72h
#   a100   | wice     | gpu_a100    | 16   |        18 |  126000M |  8500/h |  72h
#   h100   | wice     | gpu_h100    | 20   |        16 |  187200M | 34167/h |  72h
#   dbg1h  | wice     | gpu_a100_debug | 1 |        18 |  126000M |    -    |   1h
#
#   * 2 interactive nodes x 2 RTX 5000 Ada 32 GiB. A FULL GPU, not a MIG slice
#     (wICE's interactive partition gives a slice instead).
#
#  THE CHOICE THAT MATTERS FOR THIS PROJECT. TabICL generates its prior on the
#  CPU (`--prior_device cpu` in upstream's own stage scripts), so throughput is
#  bounded by CORES PER GPU, not by how fast the GPU is. That ranks the options
#  differently from raw GPU speed:
#
#      b200  24 cores/GPU  <- most cores AND most GPUs. Best on both counts.
#      a100  18 cores/GPU
#      h100  16 cores/GPU  <- fastest GPU, FEWEST cores, and 4x the price
#
#  So `h100` is the worst choice here despite being the fastest chip, and `b200`
#  is the best. `free` is the right default for debugging: it costs nothing and
#  its 8 cores are ample for a 1,500-step check.
# =============================================================================

set -euo pipefail

ACCOUNT="${ACCOUNT:-lp_verbekelab}"

# ONE HOUR by default, not four. Billing is on ACTUAL walltime (the VSC docs are explicit:
# "if this job finishes in 2.5 hours ... the user will be charged" for 150 minutes), so a
# generous limit does not cost credits directly. But the REQUESTED limit is what the scheduler
# backfills against — a short job slots into gaps a long one cannot — and it is what
# `sam-quote` reports, which is the number you actually look at before submitting.
# A 1,500-step debug run does not need four hours. Override with WALLTIME=HH:MM:SS.

usage() {
    sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

[[ "${1:-}" == "--list" || "${1:-}" == "-l" || -z "${1:-}" ]] && usage 0

TARGET="$1"
SCRIPT="${2:-scripts/slurm/debug_exp1.slurm}"
shift 2 2>/dev/null || shift 1

case "${TARGET}" in
    free)
        # FREE, and a full GPU. 16h ceiling. The 8-core cap is the trade: the prior
        # generator gets fewer workers, so a step is slower — but it starts now.
        OPTS=(--clusters=mindwell --partition=interactive
              --gpus-per-node=1 --cpus-per-task=8 --mem=30G --time=${WALLTIME:-01:00:00})
        ;;
    b200)
        # 24 GPUs and 24 cores/GPU: the most of both. The production target.
        OPTS=(--clusters=mindwell --partition=gpu_b200
              --gpus-per-node=1 --cpus-per-task=24 --mem=180G --time=${WALLTIME:-01:00:00})
        ;;
    a100)
        OPTS=(--clusters=wice --partition=gpu_a100
              --gpus-per-node=1 --cpus-per-task=18 --mem=120G --time=${WALLTIME:-01:00:00})
        ;;
    h100)
        OPTS=(--clusters=wice --partition=gpu_h100
              --gpus-per-node=1 --cpus-per-task=16 --mem=180G --time=${WALLTIME:-01:00:00})
        ;;
    dbg1h)
        # One node, one hour, and Slurm allows only ONE queued job here at a time —
        # so an array will not fit. Single-arm sanity checks only.
        OPTS=(--clusters=wice --partition=gpu_a100_debug
              --gpus-per-node=1 --cpus-per-task=18 --mem=120G --time=01:00:00
              --array=0)
        ;;
    *)
        echo "unknown target '${TARGET}'" >&2
        usage 2
        ;;
esac

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: no job script at ${SCRIPT}" >&2
    exit 1
fi

# THE CONFIG, ON SCREEN. `CONFIG` is an env var read inside the job script, so leaving it
# unset silently submits the LGD default — which is how a run intended as PD went out as a
# SECOND LGD job, with nothing in the output to say so until the log appeared.
CONFIG_IN_USE="${CONFIG:-config/Exp1_LGD.yaml (default — set CONFIG= to change)}"
echo "target : ${TARGET}"
echo "script : ${SCRIPT}"
echo "CONFIG : ${CONFIG_IN_USE}"
echo "sbatch : --account=${ACCOUNT} ${OPTS[*]} $*"

# `--export=ALL` in the job script propagates the submitting environment, but only what is
# EXPORTED. `CONFIG=... bash submit.sh` sets it for THIS shell, not for sbatch's environment,
# so it has to be exported here or the job would not see it at all.
[[ -n "${CONFIG:-}" ]] && export CONFIG

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "(DRY_RUN set — nothing submitted)"
    exit 0
fi
# `sam-quote` first, so the credit cost is on screen BEFORE the job is queued. It assumes the
# WORST case — that the job runs to its full time limit — while billing is on actual walltime
# (VSC docs: "if this job finishes in 2.5 hours ... the user will be charged" for 150 minutes).
# So treat the number as a ceiling, not a forecast. Skipped for `free`, where it is always zero.
if [[ "${TARGET}" != "free" ]] && command -v sam-quote >/dev/null 2>&1; then
    echo "--- cost ceiling if it runs the full ${WALLTIME:-01:00:00} (billing is on ACTUAL time) ---"
    sam-quote sbatch --account="${ACCOUNT}" "${OPTS[@]}" "$@" "${SCRIPT}" || true
    echo "-----------------------------------------------------------------------------------"
fi

# Capture the outcome so a QOS rejection can be explained rather than just echoed. Slurm's own
# message ("job submit limit, user's size and/or time limits") does not say WHICH limit was hit,
# and the fix differs per partition.
set +e
OUT="$(sbatch --account="${ACCOUNT}" "${OPTS[@]}" "$@" "${SCRIPT}" 2>&1)"
RC=$?
set -e
echo "${OUT}"

if [[ ${RC} -ne 0 ]] && grep -qi "QOSMaxSubmitJobPerUserLimit\|job submit limit" <<<"${OUT}"; then
    case "${TARGET}" in
        free)  QOS=interactive ;;
        dbg1h) QOS=debug ;;
        *)     QOS=normal ;;
    esac
    cat >&2 <<EOF

  ---------------------------------------------------------------------------
  That is a QUEUE-LENGTH limit, not a resource one. Each partition has a QoS
  capping how many jobs one user may have queued at once, and an ARRAY counts
  as several. The '${TARGET}' target sits on the '${QOS}' QoS.

  See the actual numbers with:
      sacctmgr show qos debug,interactive,long,normal \
          format=Name%20,MaxSubmitJobsPerUser%15,MaxTRESPerUser%30

  Options, cheapest first:
    * wait for the queued array to finish, then resubmit;
    * use a different QoS - 'b200' and 'a100' are on 'normal', a separate
      allowance from 'interactive':
          bash scripts/slurm/submit.sh b200 ${SCRIPT}
    * submit fewer arms at once:
          bash scripts/slurm/submit.sh ${TARGET} ${SCRIPT} --array=0-1
  ---------------------------------------------------------------------------
EOF
fi
exit ${RC}
