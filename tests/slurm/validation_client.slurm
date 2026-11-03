#!/bin/bash -l
# Regression checks in a separate DATA checkout while production stays frozen.
#SBATCH --job-name=crediticl-validation-client
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${CREDITICL_TEST_CHECKOUT:?Select the isolated DATA checkout}"
: "${CREDITICL_EXPECTED_COMMIT:?Pin the tested Git commit}"
: "${CREDITICL_RUNTIME_IMAGE:?Select the accepted PROJECT image}"
: "${CREDITICL_RUNTIME_SHA256:?Pin the accepted image digest}"
cd "$CREDITICL_TEST_CHECKOUT"
[[ "$(pwd -P)" == "$(realpath "$VSC_DATA")/"* ]]
[[ "$(git rev-parse HEAD)" == "$CREDITICL_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
[[ "$(sha256sum "$CREDITICL_RUNTIME_IMAGE" | cut -d ' ' -f 1)" == "$CREDITICL_RUNTIME_SHA256" ]]
printf 'Revision: %s\nRuntime: %s\nSHA256: %s\n' \
    "$CREDITICL_EXPECTED_COMMIT" "$CREDITICL_RUNTIME_IMAGE" "$CREDITICL_RUNTIME_SHA256"

work=$(mktemp -d "${VSC_SCRATCH_NODE:-${TMPDIR:?}}/crediticl-client-test.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
mkdir -p .runtime-build/validation-client
set +e
apptainer exec --cleanenv --pwd "$PWD" --bind "$PWD:$PWD:ro" --bind "$work:$work:rw" \
    --env "TMPDIR=$work" --env "PYTHONPATH=$PWD" --env CUDA_VISIBLE_DEVICES= \
    --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
    "$CREDITICL_RUNTIME_IMAGE" /opt/venv/bin/python -m pytest -q -p no:cacheprovider \
    --basetemp "$work/pytest" --junitxml "$work/junit.xml" \
    tests/test_slurm_client.py tests/test_validation.py
status=$?
set -e
if [[ -f "$work/junit.xml" ]]; then
    cp -- "$work/junit.xml" ".runtime-build/validation-client/${SLURM_JOB_ID}.xml"
fi
# Preserve child diagnostics before node-local temporary files are cleared.
while IFS= read -r -d '' logfile; do
    relative=${logfile#"$work/"}
    target=".runtime-build/validation-client/${SLURM_JOB_ID}/$relative"
    mkdir -p -- "$(dirname "$target")"
    cp -- "$logfile" "$target"
done < <(find "$work/pytest" -type f -name '*.log' -print0)
exit "$status"
