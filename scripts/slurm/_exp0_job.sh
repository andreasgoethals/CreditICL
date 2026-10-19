# Shared Experiment 0 lifecycle. Source after redirecting the allocation log.
# A shell's EXIT status alone is not evidence that validation completed: a
# signal can arrive while bash is waiting and leave the previous status at zero.
EXP0_CHILD_PID=""
EXP0_SIGNAL=""
EXP0_SIGNAL_RC=1
EXP0_FINISHED=0

_exp0_signal() {
    EXP0_SIGNAL="$1"
    EXP0_SIGNAL_RC="$2"
    echo "Experiment 0 interrupted by ${EXP0_SIGNAL}; retaining validation evidence."
    if [[ -n "${EXP0_CHILD_PID}" ]]; then
        kill -TERM "${EXP0_CHILD_PID}" 2>/dev/null || true
    else
        exit "${EXP0_SIGNAL_RC}"
    fi
}

_exp0_exit() {
    local rc=$?
    trap - EXIT
    if [[ -n "${EXP0_SIGNAL}" ]]; then
        rc=${EXP0_SIGNAL_RC}
        echo "END status=INTERRUPTED signal=${EXP0_SIGNAL} exit_code=${rc} - $(date)"
    elif [[ "$rc" -eq 0 && "${EXP0_FINISHED}" -eq 1 ]]; then
        if [[ "${EXP0_PHASE}" == gpu ]]; then
            python -m src.utils.working_storage --cleanup-success || echo "WARNING: scratch retained; inspect cleanup error above"
        fi
        echo "END status=OK - $(date)"
    elif [[ "$rc" -eq 64 ]]; then
        echo "END status=DEFERRED: progress saved; resubmit this same script without changing code."
    else
        if [[ "$rc" -eq 0 ]]; then rc=1; fi
        echo "END status=FAILED exit_code=${rc}; validation did not complete successfully - $(date)"
    fi
    exit "$rc"
}

trap '_exp0_signal TERM 143' TERM
trap '_exp0_signal INT 130' INT
trap '_exp0_signal HUP 129' HUP
trap _exp0_exit EXIT

run_exp0_validation() {
    local rc=0
    python -u scripts/validate_experiment.py "$@" &
    EXP0_CHILD_PID=$!
    # Background + wait lets bash forward a batch-script-only Slurm signal
    # immediately. A trapped signal interrupts wait before the child has ended.
    while true; do
        if wait "${EXP0_CHILD_PID}"; then rc=0; break; else rc=$?; fi
        if ! kill -0 "${EXP0_CHILD_PID}" 2>/dev/null; then break; fi
    done
    EXP0_CHILD_PID=""
    if [[ -n "${EXP0_SIGNAL}" ]]; then return "${EXP0_SIGNAL_RC}"; fi
    if [[ "$rc" -eq 0 ]]; then EXP0_FINISHED=1; fi
    return "$rc"
}
