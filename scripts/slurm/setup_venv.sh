#!/bin/bash
# =============================================================================
#  Create THIS project's own venv on the VSC, from pyproject.toml. Run ONCE.
#
#      cd $VSC_DATA/CreditICL
#      bash scripts/slurm/setup_venv.sh
#
#  RUN IT ON A LOGIN NODE. Compute nodes have no outbound internet, so pip
#  cannot reach PyPI there.
#
#  Idempotent: re-running upgrades the packages in place rather than rebuilding.
#  Pass --recreate to start from scratch.
#
#  WHY THIS EXISTS. Packages were being installed into a SIBLING project's
#  environment (TabPFNCredit/tabpfncreditvenv), so "pip install" reported
#  "already satisfied" for things CreditICL had never had, and the two projects
#  could silently disagree about torch. One project, one environment.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# WHERE. $VSC_DATA, never project storage.
#
# A venv with torch is ~5-8 GB across TENS OF THOUSANDS of small files, and
# project storage has a LOW INODE BUDGET — it is sized for few big files. Put a
# venv there and you exhaust inodes long before space. $VSC_DATA is 75 GiB and
# handles the file count fine.
# ---------------------------------------------------------------------------
REPO="${VSC_DATA}/CreditICL"
VENV="${REPO}/.venv"

# The Python module the venv is BUILT ON. It must be loaded whenever the venv is
# used, because `.venv/bin/python` is a thin link to this interpreter — without
# the module its shared library is gone and the venv fails cryptically.
# Kept in one place; `_activate_env.sh` reads the same value.
PYTHON_MODULE="Python/3.12.3-GCCcore-13.3.0"

# torch's CUDA build must match the driver. cu128 is what the working environment
# on this cluster already uses (torch 2.8.0+cu128), so it is the verified choice
# rather than a guess. A CPU-only wheel would train at ~1/50 the speed and would
# not announce itself.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

echo "=============================================================="
echo " CreditICL — VSC environment setup"
echo " repo   : ${REPO}"
echo " venv   : ${VENV}"
echo " module : ${PYTHON_MODULE}"
echo "=============================================================="

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    echo "ERROR: ${REPO}/pyproject.toml not found." >&2
    echo "       Is the repo cloned at \$VSC_DATA/CreditICL? Note this project is" >&2
    echo "       CreditICL, not CreditPFN — they are separate checkouts." >&2
    exit 1
fi
cd "${REPO}"

# --- 1. the interpreter -----------------------------------------------------
module --force purge 2>/dev/null || true
module load "${PYTHON_MODULE}"
echo "[1/5] python module loaded: $(python3 -V)"

# --- 2. the venv ------------------------------------------------------------
if [[ "${RECREATE}" == "1" && -d "${VENV}" ]]; then
    echo "[2/5] --recreate: removing ${VENV}"
    rm -rf "${VENV}"
fi
if [[ -d "${VENV}" ]]; then
    echo "[2/5] venv already exists — will upgrade in place"
else
    # NOT --system-site-packages. Inheriting the module's packages is how you end
    # up with two numpy versions on one path and an ABI error that looks like a
    # code bug. A clean venv costs disk and buys a reproducible environment.
    python3 -m venv "${VENV}"
    echo "[2/5] venv created"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel
echo "      pip: $(python -m pip --version)"

# --- 3. torch FIRST, from the CUDA index ------------------------------------
# Before the project, deliberately. `pip install -e .` would otherwise pull a
# default-index torch (CPU-only or a mismatched CUDA build) to satisfy the
# dependency, and pip would then consider it satisfied.
echo "[3/5] installing torch from ${TORCH_INDEX} ..."
python -m pip install --quiet --index-url "${TORCH_INDEX}" torch
python - <<'PY'
import torch
print(f"      torch {torch.__version__}  cuda_built={torch.version.cuda}")
PY

# --- 4. the project, from pyproject.toml ------------------------------------
# `-e` so an edit to src/ takes effect without reinstalling — the repo IS the
# package. `[dev,eval]` brings pytest/ruff and the baseline + OpenML extras.
echo "[4/5] installing CreditICL from pyproject.toml ..."
python -m pip install --quiet -e ".[dev,eval]"

# --- 5. prove it ------------------------------------------------------------
echo "[5/5] verifying ..."
FAILED=0
for pkg in numpy torch sklearn yaml pandas pyarrow tabicl src; do
    if python -c "import ${pkg}" >/dev/null 2>&1; then
        echo "      ok       ${pkg}"
    else
        echo "      MISSING  ${pkg}" >&2
        FAILED=1
    fi
done

# The architecture check that matters: our model must match the released
# checkpoints, or Exp3 cannot warm-start. Cheap, and it fails here rather than
# after a job has queued.
python - <<'PY' || FAILED=1
from src.models.architecture import build_model, describe, is_available
if not is_available("tabicl"):
    raise SystemExit("      tabicl not importable — the architecture would fall back")
info = describe(build_model("lgd"))
print(f"      ok       model = {info['class']} ({info['total_params']:,} params)")
PY

echo
if [[ "${FAILED}" == "0" ]]; then
    cat <<EOF
==============================================================
 READY.
==============================================================
Activate it in any shell with:

    source ${VENV}/bin/activate

...but you should not have to. Add the auto-activation hook once (see
docs/VSC.md), and every new shell in ${REPO} activates it for you.

Slurm jobs need nothing: scripts/slurm/_activate_env.sh finds this venv
on its own.

Next:
    python -m src.utils.smoke_test --task lgd --steps 3 --report
    python -m src.utils.fetch_ood --n 25
EOF
else
    echo "SETUP INCOMPLETE — see the MISSING lines above." >&2
    exit 1
fi
