"""Exercise site-managed submission clients without creating any jobs."""
from __future__ import annotations

import os
import re
import subprocess
import time

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
        transport_only = False
        while True:
            result = subprocess.run(args, capture_output=True, text=True, timeout=90)
            # Even --test-only applies site admission limits. A definite full
            # queue is transient and cannot validate resources; wait for a real
            # successful dry run. Generic QoS/resource errors still fail below.
            if result.returncode and re.search(
                r'(?:QOS|Assoc).*SubmitJob.*Limit|maximum number of submitted jobs',
                result.stderr, re.I
            ) and not any(message in result.stderr for message in (
                    'credit balance check is not enforced', 'sam-balance failed', 'not found')):
                # A paid GPU allocation must not idle for an unrelated queue.
                # This definite daemon response proves the node client/plugin
                # transport. Resource acceptance is separately bound to the
                # complete, current CPU receipt, never inferred from refusal.
                if os.environ.get('CREDITICL_EXP0_ATTEMPT', '').startswith('gpu_'):
                    from src.validation.campaign import readiness
                    checks = readiness('cpu')
                    if not checks or not all(row.ok for row in checks):
                        raise RuntimeError('Queue is full and current CPU resource validation is missing')
                    transport_only = True
                    break
                print(f'[WAIT] {cluster} dry-run admission is full; retrying in 60 seconds', flush=True)
                time.sleep(60)
                continue
            break
        if transport_only:
            print(f'[PASS] {cluster} node client reached Slurm; admission refused at capacity. '
                  'Resource dry-run acceptance is verified by the current CPU receipt.', flush=True)
            continue
        if result.returncode or any(message in result.stderr for message in (
                "credit balance check is not enforced", "sam-balance failed", "not found")):
            raise RuntimeError(f"Slurm submission client failed: {cluster}: {result.stderr}")
        print(f"[PASS] {cluster} sbatch --test-only: {result.stdout}{result.stderr}", flush=True)


if __name__ == "__main__":
    check()
