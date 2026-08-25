#!/bin/bash
# =============================================================================
#  Submit a CreditICL job to the RIGHT partition, without editing any script.
#
#      bash scripts/slurm/submit.sh <where> <track> [job-script]
#
#      bash scripts/slurm/submit.sh free lgd      # LGD debug, free partition
#      bash scripts/slurm/submit.sh b200 pd       # PD debug, B200
#
#      bash scripts/slurm/submit.sh --list        # show the inventory and exit
#      DRY_RUN=1 bash scripts/slurm/submit.sh free lgd     # print, submit nothing
#
#  TRACK is `lgd` or `pd`, and becomes the job script's first argument. It used to
#  be a `CONFIG=` environment variable, which was wrong twice over: `CONFIG=x bash
#  submit.sh` sets it for the CALLING shell and not for sbatch's environment, so the
#  job never received it — and nothing on screen said which config had been sent, so
#  a run intended as PD went out as a second LGD job unnoticed.
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
TRACK="${2:-lgd}"
SCRIPT="${3:-scripts/slurm/debug_exp1.slurm}"
shift 3 2>/dev/null || shift $#

case "${TRACK}" in
    lgd|LGD) CONFIG="config/Exp1_LGD.yaml" ;;
    pd|PD)   CONFIG="config/Exp1_PD.yaml"  ;;
    config/*|*.yaml) CONFIG="${TRACK}" ;;   # an explicit path still works
    *)
        echo "unknown track '${TRACK}' — expected 'lgd', 'pd', or a config path" >&2
        exit 2
        ;;
esac
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: no config at ${CONFIG}" >&2
    exit 1
fi

case "${TARGET}" in
    free)
        # FREE, and a full GPU. 16h ceiling. The 8-core cap is the trade: the prior
        # generator gets fewer workers, so a step is slower — but it starts now.
        #
        # AN ARRAY RUNS **SERIALLY** HERE. The interactive QoS caps total CPUs per user, and
        # 8 cpus-per-task exhausts it with ONE task: on 14-08-2026 a 4-arm array showed
        # `11516936_0` running and `11516936_[1-3]` pending on `QOSMaxCpuPerUserLimit`. So a
        # 4-arm debug array takes up to 4x the walltime end to end. That is fine for a debug
        # run and free, but do not expect four results in an hour — use `b200` for that.
        OPTS=(--clusters=mindwell --partition=interactive
              --gpus-per-node=1 --cpus-per-task=8 --mem=30G --time=${WALLTIME:-01:00:00})
        ;;
    b200)
        # Charge rate, credits per MINUTE, from the VSC table. Used below to check
        # `sam-quote` still agrees with the published price.
        GPU_RATE=437.50; CORE_RATE=3.03819; CORES=24
        # 24 GPUs and 24 cores/GPU: the most of both. The production target.
        OPTS=(--clusters=mindwell --partition=gpu_b200
              --gpus-per-node=1 --cpus-per-task=24 --mem=180G --time=${WALLTIME:-01:00:00})
        ;;
    a100)
        # Charge rate, credits per MINUTE, from the VSC table. Used below to check
        # `sam-quote` still agrees with the published price.
        GPU_RATE=141.667; CORE_RATE=2.54630; CORES=18
        OPTS=(--clusters=wice --partition=gpu_a100
              --gpus-per-node=1 --cpus-per-task=18 --mem=120G --time=${WALLTIME:-01:00:00})
        ;;
    h100)
        # Charge rate, credits per MINUTE, from the VSC table. Used below to check
        # `sam-quote` still agrees with the published price.
        GPU_RATE=569.444; CORE_RATE=2.54630; CORES=16
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

# EVERY input that changes what the job does, on screen before it is submitted.
echo "where  : ${TARGET}"
echo "track  : ${TRACK}"
echo "config : ${CONFIG}"
echo "script : ${SCRIPT}"
echo "sbatch : --account=${ACCOUNT} ${OPTS[*]} ${SCRIPT} ${CONFIG} $*"

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
    QUOTE=$(sam-quote sbatch --account="${ACCOUNT}" "${OPTS[@]}" "${SCRIPT}" "${CONFIG}" "$@" \
            2>/dev/null | tail -1 | tr -dc '0-9')
    echo "${QUOTE:-<sam-quote returned no number>}"

    # CROSS-CHECK AGAINST THE PUBLISHED RATES. On 25-08-2026 an identical submission quoted
    # 2,325,600 where the same line had quoted 30,600 the day before - 76x, with no change to
    # the partition, the GPU count, the cores or the walltime. It was only caught because
    # someone read the number twice. At 76x, Exp1 goes from ~10 M credits to ~760 M.
    #
    # A quote is the cluster telling you the price. This is us checking we still recognise it.
    if [[ -n "${QUOTE:-}" && -n "${GPU_RATE:-}" ]]; then
        MINUTES=$(awk -F: '{print ($1*60)+$2+($3/60)}' <<< "${WALLTIME:-01:00:00}")
        EXPECT=$(awk -v g="${GPU_RATE}" -v c="${CORE_RATE}" -v n="${CORES}" -v m="${MINUTES}" \
                     'BEGIN{printf "%d", (g + c*n) * m}')
        RATIO=$(awk -v q="${QUOTE}" -v e="${EXPECT}" \
                    'BEGIN{ if (e>0) printf "%.2f", q/e; else print 0 }')
        echo "published rates predict : ${EXPECT}   (quote / expected = ${RATIO}x)"
        if awk -v r="${RATIO}" 'BEGIN{exit !(r > 2.0 || r < 0.5)}'; then
            echo ""
            echo "  *** THE QUOTE DOES NOT MATCH THE PUBLISHED RATES (${RATIO}x) ***"
            echo "  Either the cluster has been repriced, or this job asks for more than it"
            echo "  looks like. DO NOT SUBMIT A SWEEP until you know which. The cheapest way"
            echo "  to find out is what the last small job ACTUALLY cost:"
            echo "      sam-balance"
            echo "      sam-list-usagerecords | tail -5"
            echo ""
        fi
    fi
    echo "-----------------------------------------------------------------------------------"
fi

# Capture the outcome so a QOS rejection can be explained rather than just echoed. Slurm's own
# message ("job submit limit, user's size and/or time limits") does not say WHICH limit was hit,
# and the fix differs per partition.
set +e
OUT="$(sbatch --account="${ACCOUNT}" "${OPTS[@]}" "${SCRIPT}" "${CONFIG}" "$@" 2>&1)"
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
          bash scripts/slurm/submit.sh b200 ${TRACK}
    * submit fewer arms at once:
          bash scripts/slurm/submit.sh ${TARGET} ${TRACK} '' --array=0-1
  ---------------------------------------------------------------------------
EOF
fi
exit ${RC}
