#!/bin/bash -l
# =============================================================================
#  CreditICL — free pre-flight check on the `interactive` partition.
#
#  RUN THIS FIRST. The interactive partition costs ZERO credits, so this is the
#  cheapest possible way to find out that the env is broken, the prior crashes,
#  or the vendored model is missing. Only submit the real arrays once this is
#  green.
#
#      sbatch scripts/slurm/smoke_test.slurm
#      # then watch:  tail -f $VSC_DATA/CreditICL/logs/smoke_*.log
#
#  Limits on `interactive` at KU Leuven: 1 node, 8 cores, 16 h, and 1 GPU
#  (a 1/7-A100 MIG slice on wICE, or a full RTX 5000 Ada on Mindwell).
#  This test is CPU-only and takes about a minute, so it fits easily.
# =============================================================================

#SBATCH --job-name=crediticl-smoke
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

export CREDITICL_OUTPUT_ROOT="${CREDITICL_OUTPUT_ROOT:-${VSC_DATA}/CreditICL}"
mkdir -p "${CREDITICL_OUTPUT_ROOT}/logs"
LOG="${CREDITICL_OUTPUT_ROOT}/logs/smoke_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1

echo "Log: ${LOG}"
cd "${VSC_DATA}/CreditICL"
source scripts/slurm/_activate_env.sh

echo ""
echo "=== where things will be written ==="
python -c "import json; from src.utils.paths import describe; print(json.dumps(describe(), indent=2))"

echo ""
echo "=== grid sizes (use these for --array) ==="
python scripts/pretrain.py --config config/LGD.yaml --list | tail -3
python scripts/pretrain.py --config config/PD.yaml  --list | tail -3

echo ""
echo "=== LGD smoke test ==="
python scripts/smoke_test.py --task lgd --steps 3 --report

echo ""
echo "=== PD smoke test ==="
python scripts/smoke_test.py --task pd --steps 3 --report

echo ""
echo "=========================================="
echo "ALL SMOKE TESTS PASSED — safe to submit the real arrays."
echo "=========================================="
