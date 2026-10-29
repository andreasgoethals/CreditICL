#!/bin/bash
# =============================================================================
#  Legacy DATA-venv repair only; new cluster runtimes must be immutable SIFs on PROJECT.
#  See docs/VSC.md for the migration/validation gate. This script does not migrate anything.
#
#      cd $VSC_DATA/CreditICL
#      bash scripts/slurm/setup_venv.sh --legacy-maintenance
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
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run setup_venv.sh on a login node, outside any Slurm job." >&2
    exit 1
fi
if [[ "${1:-}" != "--legacy-maintenance" ]]; then
    echo "New VSC dependencies belong in a versioned immutable SIF on PROJECT; see docs/VSC.md section 6." >&2
    echo "No environment was installed or moved. Plan and validate the container migration first." >&2
    echo "For explicitly authorized legacy DATA-venv repair only: setup_venv.sh --legacy-maintenance [--recreate]." >&2
    exit 2
fi
shift
case "${1:-}" in
    ""|--recreate) ;;
    *) echo "Usage: setup_venv.sh --legacy-maintenance [--recreate]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# LEGACY location, preserved for controlled recovery until the SIF cutover.
#
# A venv with torch is ~5-8 GB across TENS OF THOUSANDS of small files, and
# PROJECT has a low file-count budget. The adopted policy packages dependencies
# inside immutable SIF images on PROJECT, not unpacked venv directories there.
# This exceptional repair path keeps the previous DATA environment in place;
# never move a populated venv or treat this as the default new-runtime workflow.
# ---------------------------------------------------------------------------
REPO="${VSC_DATA}/CreditICL"

# ---------------------------------------------------------------------------
# ONE VENV PER MICROARCHITECTURE, suffixed with $VSC_ARCH_LOCAL.
#
# wICE (Icelake / Sapphire Rapids / Zen4) and Mindwell (Granite Rapids / Turin)
# are DIFFERENT microarchitectures, and the module tree itself is arch-specific
# (`/apps/leuven/rocky9/<arch>/<toolchain>/...`). A venv built on one is not
# reliably usable on another, and compiled wheels are the reason. Suffixing is
# the convention VSC's own documentation gives, and it means building on two
# login nodes produces two venvs instead of one corrupted one.
# ---------------------------------------------------------------------------
VENV="${REPO}/.venv-${VSC_ARCH_LOCAL:-generic}"

# Preferred Python modules, best first. DISCOVERED rather than hard-coded: the
# first attempt at this pinned `Python/3.12.3-GCCcore-13.3.0`, which existed on
# the retired Genius Skylake tree but not on every current architecture, and Lmod then reports
# "exist but cannot be loaded as requested" — which reads like a broken module
# rather than the wrong architecture. The project needs >=3.11,<3.13.
PYTHON_MODULE_PREFS=(
    "Python/3.12.3-GCCcore-13.3.0"
    "Python/3.11.3-GCCcore-12.3.0"
)

# The explicit CUDA build is recorded in pyproject.toml.

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

echo "=============================================================="
echo " CreditICL — VSC environment setup"
echo " repo   : ${REPO}"
echo " venv   : ${VENV}"
echo " arch   : ${VSC_ARCH_LOCAL:-unset}   host: $(hostname)"
echo "=============================================================="

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    echo "ERROR: ${REPO}/pyproject.toml not found." >&2
    echo "       Is the repo cloned at \$VSC_DATA/CreditICL? Note this project is" >&2
    echo "       CreditICL, not CreditPFN — they are separate checkouts." >&2
    exit 1
fi
cd "${REPO}"

# --- 1. the interpreter -----------------------------------------------------
# `module purge`, NEVER `--force purge`. On VSC the `cluster/*` modules are
# STICKY and set up the architecture-specific MODULEPATH; force-purging removes
# them too, which collapses the module tree so that even a module that exists
# reports "exist but cannot be loaded as requested". That is exactly how the
# first version of this script failed.
module purge 2>/dev/null || true

_load_python() {
    local candidate
    # 1. The preferred pins, in order.
    for candidate in "${PYTHON_MODULE_PREFS[@]}"; do
        if module load "${candidate}" 2>/dev/null; then
            PYTHON_MODULE="${candidate}"
            return 0
        fi
    done
    # 2. Anything this node actually offers in range. `module -t avail` writes to
    #    stderr and gives one name per line; `sort -V -r` puts the newest first.
    echo "      preferred modules unavailable here — discovering ..." >&2
    for candidate in $(module -t avail Python/3.12 Python/3.11 2>&1 \
                       | grep -E '^Python/3\.1[12]' | sed 's:/$::' | sort -V -r); do
        if module load "${candidate}" 2>/dev/null; then
            PYTHON_MODULE="${candidate}"
            echo "      discovered: ${candidate}" >&2
            return 0
        fi
    done
    return 1
}

if ! _load_python; then
    cat >&2 <<EOF
ERROR: could not load a Python 3.11 or 3.12 module on this node.

  host: $(hostname)   arch: ${VSC_ARCH_LOCAL:-unset}

  See what this node offers, and what a module needs, with:
      module -t avail Python
      module spider Python/3.12.3-GCCcore-13.3.0

  Module trees are per-architecture on VSC, so a module present on one login
  node can be absent on another. If a usable version shows up in the list
  above, add it to PYTHON_MODULE_PREFS at the top of this script.
EOF
    exit 1
fi
echo "[1/5] python module: ${PYTHON_MODULE} -> $(python3 -V 2>&1)"

# The venv is a thin link to THIS interpreter, so record which module built it.
# Without that record, a later shell can activate the venv under a different
# module and get a missing-libpython failure that reads like a corrupt install.
mkdir -p "${VENV%/*}"

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
# The module that built this venv, written beside it. `_activate_env.sh` and the
# ~/.bashrc hook read this instead of hard-coding a module name, so the venv and
# its interpreter can never drift apart.
echo "${PYTHON_MODULE}" > "${VENV}/.python_module"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# setuptools<82: torch 2.11 declares that bound, and a bare `--upgrade setuptools` installs
# 84 and then pip reports a dependency conflict on every subsequent install. Harmless here but
# it is noise in a log that has to stay readable.
python -m pip install --quiet --upgrade pip wheel "setuptools<82"
echo "      pip: $(python -m pip --version)"

# --- 3. torch FIRST, from the CUDA index ------------------------------------
# Before the project, deliberately. `pip install -e .` would otherwise pull a
# default-index torch (CPU-only or a mismatched CUDA build) to satisfy the
# dependency, and pip would then consider it satisfied.
read -r TORCH_VERSION TORCH_INDEX < <(python - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as stream:
    settings = tomllib.load(stream)["tool"]["crediticl"]["vsc"]
print(settings["torch-version"], settings["torch-index"])
PY
)
echo "[3/5] installing torch from ${TORCH_INDEX} ..."
python -m pip install --quiet --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}"
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
python -m src.utils.preflight --dependencies-only || FAILED=1
for pkg in numpy torch sklearn yaml pandas pyarrow tabicl src; do
    if python -c "import ${pkg}" >/dev/null 2>&1; then
        echo "      ok       ${pkg}"
    else
        echo "      MISSING  ${pkg}" >&2
        FAILED=1
    fi
done

# The architecture check that matters: our model must match the released
# checkpoints, or Exp2 cannot warm-start. Cheap, and it fails here rather than
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
    python -m src.validation.smoke --task lgd --steps 3 --report
    python -m src.utils.fetch_ood --n 25
EOF
else
    echo "SETUP INCOMPLETE — see the MISSING lines above." >&2
    exit 1
fi
