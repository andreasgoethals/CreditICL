#!/bin/bash
# Explicit legacy DATA-venv maintenance, pending validated PROJECT SIF migration.
# Never installs inside a Slurm allocation or switches the runtime automatically.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run update_env.sh on a login node, outside any Slurm job." >&2
    exit 1
fi
if [[ "${1:-}" != "--legacy-maintenance" || "$#" -ne 1 ]]; then
    echo "New VSC dependencies belong in a versioned immutable SIF on PROJECT; see docs/VSC.md section 6." >&2
    echo "No environment was updated. For authorized legacy repair only: update_env.sh --legacy-maintenance." >&2
    exit 2
fi

cd "${VSC_DATA:?VSC_DATA is not set}/CreditICL"
source scripts/slurm/_activate_env.sh
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No project venv is active. Plan the PROJECT SIF migration in docs/VSC.md; no packages installed." >&2
    exit 1
fi
repo_real=$(pwd -P)
venv_real=$(cd "${VIRTUAL_ENV}" && pwd -P)
case "${venv_real}" in
    "${repo_real}"/.venv|"${repo_real}"/.venv-*) ;;
    *) echo "Refusing to install outside this project's .venv: ${venv_real}" >&2; exit 1 ;;
esac

# Keep the explicit CUDA build when resolving the rest of the dependencies.
read -r torch_version torch_index < <(python - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as stream:
    settings = tomllib.load(stream)["tool"]["crediticl"]["vsc"]
print(settings["torch-version"], settings["torch-index"])
PY
)
python -m pip install --index-url "${torch_index}" "torch==${torch_version}"
python -m pip install -e '.[dev,eval]'

# Same-version PyPI and Git builds of TabICL are different. pip may keep an
# installed distribution, so repair the exact source if necessary.
if ! python - <<'PY'
import importlib.metadata as metadata
import json
from src.utils.preflight import TABICL_COMMIT
origin = json.loads(metadata.distribution("tabicl").read_text("direct_url.json") or "{}")
raise SystemExit(0 if origin.get("vcs_info", {}).get("commit_id") == TABICL_COMMIT else 1)
PY
then
    tabicl_commit=$(python -c 'from src.utils.preflight import TABICL_COMMIT; print(TABICL_COMMIT)')
    python -m pip install --force-reinstall --no-deps \
        "tabicl @ git+https://github.com/soda-inria/tabicl.git@${tabicl_commit}"
    python -m pip install -e '.[dev,eval]'
fi

python -m src.utils.preflight --dependencies-only
python - <<'PY'
import json
from src.utils.environment import environment_fingerprint, environment_snapshot
from src.utils.paths import outputs_dir
snapshot = environment_snapshot()
destination = outputs_dir() / "general" / "report" / "environment.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps({"environment": snapshot,
    "fingerprint": environment_fingerprint(snapshot)}, indent=2), encoding="utf-8")
print(f"Verified {len(snapshot['packages'])} installed package records; saved {destination}")
PY
