#!/bin/bash
# =============================================================================
#  Shared conda-env activator for every CreditICL slurm script.
#
#  Adapted from CreditPFN's scripts/slurm/_activate_env.sh, which was hardened
#  against several real failures on this cluster. Keeping the same shape means
#  the two projects fail the same way and are debugged the same way.
#
#  Usage, from a repo-root cwd, in a script with a `#!/bin/bash -l` shebang:
#
#      source scripts/slurm/_activate_env.sh
#
#  Set CONDA_ENV=<name> before sourcing to use a different env (default CreditICL).
# =============================================================================

CONDA_ENV="${CONDA_ENV:-CreditICL}"

# ---------------------------------------------------------------------------
# An active virtualenv SHADOWS conda and wins silently. `#!/bin/bash -l` sources
# ~/.bashrc; if that auto-activates a venv, its bin/ sits ahead of the conda
# env's on PATH. `conda activate` then reports success while `python` and `pip`
# still resolve to the VENV. CreditPFN hit exactly this. Neutralise it first.
# ---------------------------------------------------------------------------
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "WARNING: a virtualenv is active (${VIRTUAL_ENV}) and would shadow conda." >&2
    echo "         Removing it from PATH for this job." >&2
    PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v "^${VIRTUAL_ENV}/bin$" | paste -sd: -)"
    export PATH
    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT 2>/dev/null || true
fi

_try_source_conda() {
    if [[ -f "$1" ]]; then
        # shellcheck disable=SC1090
        source "$1"
        return 0
    fi
    return 1
}

if [[ -n "${CONDA_EXE:-}" ]] && [[ -x "${CONDA_EXE}" ]]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    _try_source_conda "${VSC_DATA:-}/miniconda3/etc/profile.d/conda.sh"  || \
    _try_source_conda "${VSC_DATA:-}/miniforge3/etc/profile.d/conda.sh"  || \
    _try_source_conda "${VSC_DATA:-}/mambaforge/etc/profile.d/conda.sh"  || \
    _try_source_conda "${HOME}/miniconda3/etc/profile.d/conda.sh"        || \
    _try_source_conda "${HOME}/miniforge3/etc/profile.d/conda.sh"        || \
    _try_source_conda "${HOME}/mambaforge/etc/profile.d/conda.sh"        || \
    _try_source_conda "/apps/leuven/rocky9/sapphirerapids/2024a/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh" || {
        echo "ERROR: could not locate a conda/mamba installation." >&2
        echo "       Create the env once from a Genius login node:" >&2
        echo "         conda create -y -n ${CONDA_ENV} python=3.12" >&2
        echo "         conda activate ${CONDA_ENV}" >&2
        echo "         cd \$VSC_DATA/CreditICL && pip install -e \".[dev]\"" >&2
        exit 1
    }
fi

_prepend_conda_bin() {
    # `conda activate` sets CONDA_PREFIX but does not always win the PATH race
    # (a loaded Python Lmod module can stay ahead of it). Force the env's bin to
    # the front so `python` matches CONDA_PREFIX, then clear bash's path cache.
    [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]] || return 0
    PATH="${CONDA_PREFIX}/bin:${PATH}"
    export PATH
    hash -r 2>/dev/null || true
    return 0
}

_env_is_healthy() {
    # `conda activate` can "succeed" on an empty env, whose bin/ contributes
    # nothing and where `python` silently falls through to /bin/python. So check
    # that the interpreter really lives in the env AND that the deps import.
    local py
    py="$(command -v python || true)"
    if [[ -z "${CONDA_PREFIX:-}" || "${py}" != "${CONDA_PREFIX}"* ]]; then
        echo "  [activate] python (${py:-none}) is NOT inside CONDA_PREFIX (${CONDA_PREFIX:-unset})." >&2
        return 1
    fi
    local err
    if ! err=$(python -c "import numpy, torch, sklearn, yaml" 2>&1); then
        echo "  [activate] dependency import failed in env '${CONDA_DEFAULT_ENV:-?}':" >&2
        echo "${err}" | head -3 | sed 's/^/      /' >&2
        return 1
    fi
    return 0
}

_activated=""
if conda activate "${CONDA_ENV}" 2>/dev/null && _prepend_conda_bin && _env_is_healthy; then
    _activated="${CONDA_ENV}"
else
    echo "WARNING: env '${CONDA_ENV}' is unusable — trying 'base' as a fallback." >&2
    if conda activate base 2>/dev/null && _prepend_conda_bin && _env_is_healthy; then
        _activated="base"
        echo "WARNING: running in the BASE env. Repair the named env when convenient." >&2
    fi
fi

if [[ -z "${_activated}" ]]; then
    echo "ERROR: no usable conda env (tried '${CONDA_ENV}' and 'base')." >&2
    conda env list >&2
    echo "       Repair once from a Genius login node:" >&2
    echo "         conda create -y -n ${CONDA_ENV} python=3.12" >&2
    echo "         conda activate ${CONDA_ENV}" >&2
    echo "         cd \$VSC_DATA/CreditICL" >&2
    echo "         pip install -e \".[dev]\"" >&2
    exit 1
fi

echo "Active conda env: ${CONDA_DEFAULT_ENV:-?} ($(command -v python))"

# The vendored architecture is generated from the pinned library dump and is
# gitignored-adjacent (it IS committed, but a stale pin makes it wrong). Cheap
# to regenerate, so do it every job rather than debugging a stale copy later.
if [[ ! -f src/models/nanotabiclv2.py ]]; then
    echo "Vendoring the model from the pinned tfm-library dump..."
    python scripts/vendor_model.py
fi
