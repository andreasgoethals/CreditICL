#!/bin/bash
# Explicit runtime manifest on DATA; immutable dependency image on PROJECT.
export VSC_DATA=$(realpath "${VSC_DATA:?}")
_repo="${VSC_DATA}/CreditICL"
_manifest="${CREDITICL_RUNTIME_MANIFEST:-${_repo}/.runtime-build/current.json}"
if [[ -z "${CREDITICL_RUNTIME_IMAGE:-}" ]]; then
    [[ -f "$_manifest" ]] || { echo "Missing accepted PROJECT runtime manifest: $_manifest" >&2; return 1; }
    CREDITICL_RUNTIME_IMAGE=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$_manifest")
fi
_assets="${CREDITICL_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}/CreditICL"
case "$CREDITICL_RUNTIME_IMAGE" in "$_assets"/runtimes/*.sif) ;; *) echo 'Runtime must be a PROJECT CreditICL SIF' >&2; return 1;; esac
[[ -f "$CREDITICL_RUNTIME_IMAGE" ]] || return 1
_expected=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$_manifest")
_actual=$(sha256sum "$CREDITICL_RUNTIME_IMAGE")
export CREDITICL_RUNTIME_SHA256=${_actual%% *}
[[ "$CREDITICL_RUNTIME_SHA256" == "$_expected" ]] || { echo 'Runtime image digest changed' >&2; return 1; }
unset CREDITICL_EXEC_IMAGE  # A parent allocation's node-local copy is not portable.
if [[ "${SLURM_CLUSTER_NAME:-}" == mindwell && -n "${SLURM_JOB_ID:-}" ]]; then
    _runtime_work="${VSC_SCRATCH_NODE:?}/CreditICL/runtime-${SLURM_JOB_ID}"
    mkdir -p "$_runtime_work"
    export CREDITICL_EXEC_IMAGE="$_runtime_work/runtime.sif"
    cp "$CREDITICL_RUNTIME_IMAGE" "$CREDITICL_EXEC_IMAGE.partial"
    mv "$CREDITICL_EXEC_IMAGE.partial" "$CREDITICL_EXEC_IMAGE"
    _copied=$(sha256sum "$CREDITICL_EXEC_IMAGE")
    [[ "${_copied%% *}" == "$_expected" ]] || { echo 'Staged runtime digest differs' >&2; return 1; }
fi
if [[ -n "${CREDITICL_EXPECTED_COMMIT:-}" ]]; then
    [[ "$(git -C "$_repo" rev-parse HEAD)" == "$CREDITICL_EXPECTED_COMMIT" ]] || { echo 'Checkout changed since submission' >&2; return 1; }
    git -C "$_repo" diff --quiet HEAD || { echo 'Tracked code changed since submission' >&2; return 1; }
fi
export CREDITICL_RUNTIME_IMAGE
mkdir -p "$_repo/output CreditICL"
# Use the host Python wrapper without changing the legacy environment or shell hooks.
_wrappers="$_repo/.runtime-build/bin"
mkdir -p "$_wrappers"
if [[ ! -L "$_wrappers/python" || "$(readlink "$_wrappers/python")" != "$_repo/scripts/slurm/container_python" ]]; then
    _wrapper_temp="$_wrappers/.python-${SLURM_JOB_ID:-login}-${BASHPID}"
    ln -s "$_repo/scripts/slurm/container_python" "$_wrapper_temp"
    mv -Tf "$_wrapper_temp" "$_wrappers/python"
fi
export PATH="$_wrappers:$PATH"
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME
hash -r
echo "[activate] PROJECT image: $CREDITICL_RUNTIME_IMAGE" >&2
