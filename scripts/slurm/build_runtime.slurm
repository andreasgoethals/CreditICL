#!/bin/bash -l
# Explicit offline rebuild; never invoked by a training job.
# Args: DATA build folder, scratch working folder, source venv, base SIF, PROJECT image.
#SBATCH --job-name=crediticl-runtime-build
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
set -euo pipefail
[[ $# == 5 ]] || exit 2
build=$1; work=$2; source_env=$3; base=$4; image=$5
[[ ! -e "$image" && "$image" == /lustre1/project/stg_00211/CreditICL/runtimes/*.sif ]] || exit 2
module load Python/3.12.3-GCCcore-13.3.0
export APPTAINER_TMPDIR="${VSC_SCRATCH_NODE:?}/crediticl-build-${SLURM_JOB_ID}"
export APPTAINER_CACHEDIR="$work/apptainer-cache"
mkdir -p "$build" "$work/compat" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$image")"
trap 'rc=$?; printf "exit=%s\n" "$rc" > "$build/status.txt"' EXIT
if [[ ! -f "$work/snapshot/inventory.json" ]]; then
    "$source_env/bin/python" "$build/snapshot.py" "$work/snapshot"
fi
"$source_env/bin/python" "$build/snapshot.py" "$work/snapshot" --verify
find "$source_env/lib/python3.12/site-packages" -type f -perm /111 -printf '%P\n' > "$build/executables.txt"
if [[ "$(realpath "$base")" != "$(realpath "$work/python-base.sif")" ]]; then
    cp "$base" "$work/python-base.sif"
fi
sha256sum "$base" "$work/python-base.sif" > "$build/base.sha256"
for lib in libstdc++.so.6 libgcc_s.so.1 libgomp.so.1; do
    library=$(gcc -print-file-name="$lib")
    [[ -f "$library" ]] || exit 1
    cp -L "$library" "$work/compat/$lib"
done
cat > "$build/image.def" <<DEF
Bootstrap: localimage
From: $work/python-base.sif
%files
    $work/snapshot /opt/artifacts
    $work/compat /opt/compat
    $build/executables.txt /opt/runtime-executables.txt
%post
    export LD_LIBRARY_PATH=/opt/compat
    python -m venv /opt/venv
    /opt/venv/bin/python -m pip install --no-index --no-deps /opt/artifacts/wheels/*.whl
    /opt/venv/bin/python -c 'import json,pathlib; root=pathlib.Path("/opt/venv/lib/python3.12/site-packages"); origins=json.loads(pathlib.Path("/opt/artifacts/origins.json").read_text()); [(root/name/"direct_url.json").write_text(json.dumps(value)) for name,value in origins.items()]'
    /opt/venv/bin/python -c 'import pathlib; root=pathlib.Path("/opt/venv/lib/python3.12/site-packages"); files=pathlib.Path("/opt/runtime-executables.txt").read_text().splitlines(); [p.chmod(p.stat().st_mode | 0o111) for f in files if (p:=root/f).is_file()]'
    cp /opt/artifacts/inventory.json /opt/runtime-inventory.json
    rm -r /opt/artifacts
%environment
    export PATH=/opt/venv/bin:\$PATH
    export LD_LIBRARY_PATH=/opt/compat:\${LD_LIBRARY_PATH:-}
    export PYTHONNOUSERSITE=1
DEF
apptainer build --mksquashfs-args '-processors 8' "$image.partial" "$build/image.def"
mv "$image.partial" "$image"
sha256sum "$image" > "$build/image.sha256"
cp "$work/snapshot/inventory.json" "$work/snapshot/origins.json" "$build/"
apptainer exec --cleanenv "$image" python -m pip check
apptainer exec --cleanenv "$image" python -c 'import torch,tabpfn,tabicl,numpy,scipy,sklearn,catboost; print("RUNTIME_IMPORTS_OK",torch.__version__)'
printf 'IMAGE_BUILT_NOT_ACCEPTED\n' > "$build/built.txt"
