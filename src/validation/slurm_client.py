"""Exercise site-managed submission clients without creating any jobs."""
from __future__ import annotations

import os
import subprocess

from src.utils.paths import REPO_ROOT


def check() -> None:
    if os.environ.get("CREDITICL_IN_CONTAINER") != "1" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the Slurm client check inside an allocated PROJECT runtime")
    balance = subprocess.run(["sam-balance", "-A", os.environ.get("SLURM_JOB_ACCOUNT", "lp_verbekelab")],
                             capture_output=True, text=True, timeout=90)
    if balance.returncode or not balance.stdout.strip():
        raise RuntimeError(f"Site credit client failed: {balance.stderr}")
    print(f"[PASS] site credit client: {balance.stdout}", flush=True)
    # sbatch dynamically loads serializers not used by squeue. Exercise both
    # target clusters and the actual scripts; --test-only never submits a job.
    for cluster, script in (("wice", "exp0_cpu.slurm"), ("mindwell", "pretrain_pd.slurm")):
        args = ["sbatch", "--test-only", f"--clusters={cluster}", "--array=0",
                str(REPO_ROOT / "scripts/slurm" / script)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=90)
        if result.returncode or any(message in result.stderr for message in (
                "credit balance check is not enforced", "sam-balance failed", "not found")):
            raise RuntimeError(f"Slurm submission client failed: {cluster}: {result.stderr}")
        print(f"[PASS] {cluster} sbatch --test-only: {result.stdout}{result.stderr}", flush=True)


if __name__ == "__main__":
    check()
