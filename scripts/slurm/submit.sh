#!/bin/bash
# Retired: this launcher predates the two-phase validation and receipt gates.
set -euo pipefail
cat >&2 <<'EOF'
This legacy launcher is retired; nothing was submitted.
First run Experiment 0:
  sbatch --clusters=wice scripts/slurm/exp0_cpu.slurm
After CPU PASS:
  sbatch --clusters=mindwell scripts/slurm/exp0.slurm
After both PASS, inspect the missing production stages:
  python -m src.utils.run_experiment 1
Add --submit only when ready to launch those stages.
EOF
exit 2
