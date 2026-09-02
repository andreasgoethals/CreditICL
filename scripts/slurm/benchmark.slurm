#!/bin/bash -l
# =============================================================================
#  CreditICL — PHASE 2: benchmark every trained checkpoint. SHARED BY ALL THREE EXPERIMENTS.
#
#  PHASE 1 (pretrain_{lgd,pd}.slurm) produces 75 checkpoints and NOTHING ELSE.
#  PHASE 2 is this script. It cannot start until phase 1 has finished, because it scores the
#  checkpoints phase 1 wrote — that ordering is a fact about the data, not a convention.
#
#      python -m src.utils.sweep_status --config config/Exp1_LGD.yaml   # must say 75 done
#      sbatch --clusters=mindwell --export=ALL,EXP=1,TRACK=lgd scripts/slurm/benchmark.slurm
#      sbatch --clusters=mindwell --export=ALL,EXP=2,TRACK=lgd scripts/slurm/benchmark.slurm
#      sbatch --clusters=mindwell --export=ALL,EXP=3,TRACK=lgd scripts/slurm/benchmark.slurm
#
#  THE REFERENCE COLUMN IS SCORED ONCE, EVER. CatBoost, TabPFN-3, the released TabICLv2 and
#  logistic/linear regression do not depend on our prior, so their numbers are identical in
#  Exp1, Exp2 and Exp3 — and re-running them per experiment would not only waste GPU time, it
#  would introduce three slightly different reference columns to compare against. The last
#  array index writes `reference_<track>` and SKIPS if that already exists; pass
#  `FORCE_REFERENCE=1` to rescore it deliberately.
#
#  ONE ARRAY TASK = ONE MODEL, scored on every real credit dataset and every out-of-domain
#  suite, at every seed:
#
#      tasks 0-44   our 45 checkpoints, one per Exp1 arm
#      task  45     THE REFERENCE COLUMN: released TabICLv2, CatBoost, logistic/linear
#
#  Keeping the reference inside the same array is deliberate. It is scored by the same code,
#  on the same day, with the same `--max-context-rows`, the same seeds and the same splits as
#  the 45. Every previous comparison this project got wrong got it wrong by scoring the two
#  sides through different paths.
#
#  Set TRACK=lgd or TRACK=pd. Logs and results are small and go to $VSC_DATA; the checkpoints
#  they read are large and live on project storage. See docs/VSC.md.
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
# Scoring is minutes per model, not hours — but `0014.algorithmwatch` is 158,700 x 2,986 and
# the 8-member ensemble runs once per dataset per seed. 4 h is generous and schedules well.
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --requeue
# 0-45 fits Exp1 (45 arms + 1 reference). OVERRIDE IT for the others:
#     Exp2: --array=0-60%8      Exp3: --array=0-10%8
# Indices past the reference slot exit in seconds, so an oversized range is safe.
#SBATCH --array=0-45%8

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRACK="${TRACK:-lgd}"
EXP="${EXP:-1}"
CONFIG="config/Exp${EXP}_$(echo "$TRACK" | tr '[:lower:]' '[:upper:]').yaml"
IDX="${SLURM_ARRAY_TASK_ID}"

export CREDITICL_OUTPUT_ROOT="${CREDITICL_OUTPUT_ROOT:-${VSC_DATA}/CreditICL}"
mkdir -p "${CREDITICL_OUTPUT_ROOT}/output/logs"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITICL_OUTPUT_ROOT}/output/logs/bench_exp${EXP}_${TRACK}_${TS}_j${SLURM_ARRAY_JOB_ID}_a${IDX}.log"
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

# CONTEXT CAP, matched to what every model here was trained on (upstream stage 1 is 1,024 rows
# exactly, and so are we). Applied from ONE variable to OUR checkpoints and to the reference
# alike, so it is part of the measurement rather than a handicap on one side, and recorded in
# every row — including on the datasets where it does not bind.
CONTEXT_CAP="${CONTEXT_CAP:-1024}"
SEEDS="${SEEDS:-0,1,2}"
echo "context cap ${CONTEXT_CAP} rows | seeds ${SEEDS} | applies to EVERY model in this array"

N_ARMS=$(python -c "
from src.utils.config import expand_with_seeds, load
print(len(expand_with_seeds(load('${CONFIG}'))))
")
echo "grid has ${N_ARMS} arms; this task is index ${IDX}"

# EXACTLY ONE index is the reference: the first past the grid. Anything beyond it exists only
# because `--array` is a fixed range and the three experiments have different arm counts
# (Exp1 45, Exp2 60, Exp3 10). Those tasks exit immediately rather than all racing to write the
# same reference file.
if [ "${IDX}" -gt "${N_ARMS}" ]; then
    echo "index ${IDX} is past the grid (${N_ARMS} arms) and past the reference slot. Nothing to do."
    exit 0
fi

if [ "${IDX}" -eq "${N_ARMS}" ]; then
    # ------------------------------------------------------------------ the reference column
    echo "=== REFERENCE MODELS (no checkpoint of ours involved) ==="
    # `linear` is Ridge for LGD and logistic regression for PD — the floor to beat.
    # `tabpfn3` is TabPFN-3 via `model_path="v3"`, a tabular foundation model whose prior we
    # never touch, so it is the "someone else's prior, properly trained" yardstick.
    MODELS="${REFERENCE_MODELS:-tabiclv2,tabpfn3,catboost,linear}"
    REF_TAG="reference_${TRACK}"
    REF_FILE="$(python -c "
from src.utils import paths
print(paths.results_dir() / '${TRACK}' / 'eval' / 'results_${REF_TAG}.csv')
")"
    if [ -f "$REF_FILE" ] && [ "${FORCE_REFERENCE:-0}" != "1" ]; then
        echo "reference already scored: ${REF_FILE}"
        echo "It does not depend on our prior, so Exp1/2/3 all compare against THIS file."
        echo "Pass FORCE_REFERENCE=1 to rescore."
        exit 0
    fi
    echo "models: ${MODELS}   ->  ${REF_TAG}"
    python scripts/evaluate.py \
        --task "${TRACK}" \
        --models "${MODELS}" \
        --max-context-rows "${CONTEXT_CAP}" \
        --seeds "${SEEDS}" \
        --tag "${REF_TAG}"
    python scripts/evaluate_ood.py \
        --models "${MODELS}" \
        --seeds "${SEEDS}" \
        --tag "${REF_TAG}"
    exit 0
fi

# ---------------------------------------------------------------------- one of our 45 arms --
read -r RUN_NAME CKPT_DIR <<< "$(python -c "
import sys
from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name
runs = expand_with_seeds(load('${CONFIG}'))
name = run_name(runs[${IDX}])
print(name, paths.checkpoints_dir() / name)
")"
echo "arm ${IDX}: ${RUN_NAME}"
echo "checkpoint dir: ${CKPT_DIR}"

# The LAST permanent checkpoint, which for a completed arm is step-12500. Sorting numerically
# on the step matters: a lexical sort puts step-9500 after step-12500.
# `|| true`: under `set -o pipefail`, `ls` exits 2 when the glob matches nothing and would
# kill the script THERE, before the graceful check below ever runs. That is exactly what
# happened on 25-08 when the benchmark array was submitted before phase 1: 74 tasks each died
# in ~1 s with a bare exit 2 instead of the clean skip intended here.
CKPT="$(ls -1 "${CKPT_DIR}"/step-*.ckpt 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
if [ -z "$CKPT" ]; then
    # NOT a failure. This arm has not been trained yet. Exit 0 so it does not show as FAILED
    # and does not trigger a requeue; `pipeline.py` sees no result file and resubmits phase 2
    # once phase 1 has produced the checkpoint. Run phase 1 first:
    echo "SKIPPED: no checkpoint under ${CKPT_DIR} — arm ${IDX} not trained yet."
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
summary = paths.outputs_dir() / run_name / "summary.json"
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

echo "=== real credit datasets ==="
python scripts/evaluate.py \
    --task "${TRACK}" \
    --models crediticl \
    --checkpoint "$CKPT" \
    --max-context-rows "${CONTEXT_CAP}" \
    --seeds "${SEEDS}" \
    --tag "exp${EXP}bench_${TRACK}_a${IDX}"

echo "=== out of domain ==="
# A prior that helps credit by destroying generality has to be visible, not inferred.
python scripts/evaluate_ood.py \
    --models crediticl \
    --task "${TRACK}" \
    --checkpoint "$CKPT" \
    --seeds "${SEEDS}" \
    --tag "exp${EXP}bench_${TRACK}_a${IDX}"

echo "arm ${IDX} benchmarked."
