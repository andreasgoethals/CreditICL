#!/bin/bash -l
# =============================================================================
#  CreditICL — PHASE 2: benchmark every trained checkpoint. SHARED BY ALL THREE EXPERIMENTS.
#
#  PHASE 1 (pretrain_{lgd,pd}.slurm) produces the configured training checkpoints.
#  PHASE 2 is this script. It cannot start until phase 1 has finished, because it scores the
#  checkpoints phase 1 wrote — that ordering is a fact about the data, not a convention.
#
#      python -m src.utils.sweep_status --config config/Exp1_PD.yaml    # must say all done
#      python -m src.eval.benchmark_status --exp 1 --track pd --count   # array size (Exp1: 49)
#      sbatch --clusters=mindwell --export=ALL,EXP=1,TRACK=pd  scripts/slurm/benchmark.slurm
#      sbatch --clusters=mindwell --export=ALL,EXP=1,TRACK=lgd scripts/slurm/benchmark.slurm
#
#  THE PROTOCOL (src/eval/protocol.py, version 3): 5-fold cross-validation on every real credit
#  dataset and every out-of-domain suite; the WHOLE training pool as context, for every model;
#  the PD threshold chosen to maximise F1 on a validation split of the training pool; every
#  metric (ROC-AUC, PR-AUC, Brier, log-loss, ECE, F1, ... / R2, RMSE, CRPS, boundary mass, ...).
#
#  ONE ARRAY TASK = ONE CHECKPOINT OR ONE REFERENCE MODEL (`benchmark_status.slot_for`):
#
#      index = position x n_arms + arm   every saved checkpoint of every arm — position 0 is
#                                        the FINAL step, so the headline table comes first,
#                                        then 2,500, 5,000, ... for the learning curves
#      the last len(REFERENCE_MODELS)    released TabICLv2, TabPFN-3, CatBoost, logistic/linear,
#                                        one per task (TabPFN-3 at full context is the slow one)
#
#  THE REFERENCE COLUMN IS REUSED. Those models do not depend on our prior, so their numbers are
#  identical in Exp1, Exp2 and Exp3; each writes `reference_<track>_<model>` and skips only with a
#  valid receipt for this protocol. Pass FORCE_REFERENCE=1 to rescore deliberately.
#
#  Set TRACK=lgd or TRACK=pd. Logs and results are small and go to $VSC_DATA
#  (output_CreditICL/experiment_<N>/, the reference column to output_CreditICL/reference/); the
#  checkpoints they read are large and live on project storage. See docs/VSC.md.
# =============================================================================
#SBATCH --job-name=crediticl-bench
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
# Full context is the expensive part: `0006.hackerearth` puts ~426,000 training rows in front of
# ~106,000 test rows, five times, and PD scores each fold twice (validation pass, then the refit).
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --requeue
# Exp1: 9 arms x 5 saved checkpoints + 4 reference models = 49 slots, 0-48. For the others run
#     python -m src.eval.benchmark_status --exp N --track T --count
# and pass --array=0-<count-1>%8. Indices past the last slot exit in seconds.
#SBATCH --array=0-48%8

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRACK="${TRACK:-lgd}"
EXP="${EXP:-1}"
# VARIANT=search scores Experiment 2's stage A: development datasets only, no reference slots.
VARIANT="${VARIANT:-}"
CONFIG="config/Exp${EXP}_$(echo "$TRACK" | tr '[:lower:]' '[:upper:]')${VARIANT:+_${VARIANT}}.yaml"
IDX="${SLURM_ARRAY_TASK_ID}"

OUTPUT_ROOT="${VSC_DATA}/CreditICL/output_CreditICL"   # = src.utils.paths.outputs_dir()
LOGDIR="${OUTPUT_ROOT}/experiment_${EXP}/logs"
mkdir -p "${LOGDIR}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/bench_exp${EXP}_${TRACK}_${TS}_j${SLURM_ARRAY_JOB_ID}_a${IDX}.log"
exec > "$LOG" 2>&1

_log_exit() {
    rc=$?
    echo "=========================================="
    if [[ "$rc" -eq 0 ]]; then echo "END status=OK - $(date)"
    else echo "END status=FAILED exit_code=${rc} - $(date)"; fi
    echo "=========================================="
}
trap _log_exit EXIT

cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditICL — PHASE 2 BENCHMARK — experiment ${EXP}, track ${TRACK}"
echo "=========================================="
echo "Array job/task : ${SLURM_ARRAY_JOB_ID}/${IDX}"
echo "Cluster / part : ${SLURM_CLUSTER_NAME:-?} / ${SLURM_JOB_PARTITION:-?}   node ${SLURMD_NODENAME:-?}"
echo "Cores / mem    : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MiB"
echo "Restart count  : ${SLURM_RESTART_COUNT:-0}"
echo "Config         : ${CONFIG}"
echo "Date           : $(date)"
echo "Log            : ${LOG}"
echo "Commit         : $(git rev-parse --short HEAD 2>/dev/null || echo '?')  dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo YES || echo no)"
nvidia-smi || true
echo "=========================================="

# One 5-fold CV (fold-assignment seed 0) — the same folds for every model and every checkpoint.
SEEDS="${SEEDS:-0}"
REFERENCE_MODELS="${REFERENCE_MODELS:-tabiclv2,tabpfn3,catboost,linear}"
STATUS_ARGS=(--exp "$EXP" --track "$TRACK" --index "$IDX" --seeds "$SEEDS"
             --reference-models "$REFERENCE_MODELS" --variant "$VARIANT")
read -r SLOT_KIND TAG ARM STEP MODEL <<< "$(python -m src.eval.benchmark_status "${STATUS_ARGS[@]}" --describe)"
echo "slot ${IDX}: kind=${SLOT_KIND} tag=${TAG} arm=${ARM} step=${STEP} model=${MODEL} | seeds ${SEEDS} | whole training pool as context"

if [[ "$SLOT_KIND" == "none" ]]; then
    echo "index ${IDX} is past the last slot. Nothing to do."
    exit 0
fi

DATASETS=$(python -c "from src.utils.config import load; e=load('${CONFIG}')['eval']; print(','.join(sorted(set(e['dev_datasets'] + e['holdout_datasets']))))")
KIND=classification
if [[ "$TRACK" == "lgd" ]]; then KIND=regression; fi

if [[ "${FORCE_REFERENCE:-0}" != "1" ]] && python -m src.eval.benchmark_status "${STATUS_ARGS[@]}"; then
    echo "Complete credit and OOD benchmark already verified for this slot and protocol."
    exit 0
fi

if [[ "$SLOT_KIND" == "reference" ]]; then
    # ------------------------------------------------------------------ one reference model
    # `linear` is Ridge for LGD and logistic regression for PD — the floor to beat.
    # `tabpfn3` is TabPFN-3, a tabular foundation model whose prior we never touch, so it is the
    # "someone else's prior, properly trained" yardstick.
    echo "=== REFERENCE MODEL ${MODEL} (no checkpoint of ours involved) -> ${TAG} ==="
    python scripts/evaluate.py \
        --task "${TRACK}" \
        --datasets "${DATASETS}" \
        --models "${MODEL}" \
        --seeds "${SEEDS}" \
        --tag "${TAG}"
    python scripts/evaluate_ood.py \
        --models "${MODEL}" \
        --kinds "${KIND}" \
        --seeds "${SEEDS}" \
        --tag "${TAG}"
    python -m src.eval.benchmark_status "${STATUS_ARGS[@]}" --write
    exit 0
fi

# ------------------------------------------------------------- one checkpoint of one arm --
read -r RUN_NAME CKPT_DIR <<< "$(python -c "
from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name
runs = expand_with_seeds(load('${CONFIG}'))
name = run_name(runs[${ARM}])
print(name, paths.run_checkpoints_dir(name))
")"
echo "arm ${ARM}: ${RUN_NAME}, step ${STEP}"
echo "checkpoint dir: ${CKPT_DIR}"

# The slot's own step — a permanent checkpoint (a multiple of save_perm_every) or the final one,
# never a rolling checkpoint picked by a lexical sort.
CKPT="${CKPT_DIR}/step-${STEP}.ckpt"
if [ ! -f "$CKPT" ]; then
    # NOT a failure. This arm has not been trained (that far) yet. Exit 0 so it does not show as
    # FAILED and does not trigger a requeue; `run_experiment` sees no receipt and resubmits
    # phase 2 once phase 1 has produced the checkpoint. Run phase 1 first:
    echo "SKIPPED: no checkpoint under ${CKPT_DIR} for step ${STEP} — arm ${ARM} not trained yet."
    echo "  python -m src.utils.sweep_status --config ${CONFIG}   # what is done"
    exit 0
fi
echo "scoring: ${CKPT}"

# REFUSE A HALF-TRAINED ARM. An arm stopped by SIGUSR1 leaves a checkpoint that loads perfectly
# and is not a result. `summary.json` is the authority, exactly as `sweep_status` uses it.
python - "$CKPT" "${RUN_NAME}" "${CONFIG}" <<'PYCHECK'
import json
import sys
from pathlib import Path

from src.utils import paths
from src.utils.config import load

ckpt, run_name, config = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
summary = paths.run_summary_path(run_name)  # experiment_<N>/runs/<run>/summary.json
if not summary.is_file():
    raise SystemExit(f"FATAL: {summary} is missing — phase 1 never finished this arm.")
data = json.loads(summary.read_text(encoding="utf-8"))
want = (load(Path(config)).get("train") or {}).get("max_steps")
if not data.get("completed"):
    raise SystemExit(
        f"FATAL: arm stopped at step {data.get('steps')} by {data.get('stopped_by_signal')} "
        f"and is NOT complete. Resubmit phase 1 for this index before benchmarking it."
    )
print(f"phase 1 OK: {data.get('steps')}/{want} steps, completed={data.get('completed')}")
PYCHECK

# THE WRAPPER FOLLOWS THE ARCHITECTURE (`benchmark_status.arm_model`): `crediticl` for a TabICL
# checkpoint, `tabpfn3` for a TabPFN-3 one — the same wrapper the released weights are scored with.
echo "=== real credit datasets (model ${MODEL}) ==="
python scripts/evaluate.py \
    --task "${TRACK}" \
    --datasets "${DATASETS}" \
    --models "${MODEL}" \
    --checkpoint "$CKPT" \
    --seeds "${SEEDS}" \
    --tag "${TAG}"

echo "=== out of domain ==="
# A prior that helps credit by destroying generality has to be visible, not inferred.
python scripts/evaluate_ood.py \
    --models "${MODEL}" \
    --kinds "${KIND}" \
    --task "${TRACK}" \
    --checkpoint "$CKPT" \
    --seeds "${SEEDS}" \
    --tag "${TAG}"

python -m src.eval.benchmark_status "${STATUS_ARGS[@]}" --write

echo "arm ${ARM} step ${STEP} benchmarked."
