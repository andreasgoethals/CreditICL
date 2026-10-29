"""Authorized one-off chain: CPU/GPU acceptance, missing Exp1 scores, Exp2 submission.

This controller stops on failure and never changes code, installs or cleans files.
Run only with --run after the checkout/runtime have been frozen and reviewed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def command(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='', flush=True)
    if result.returncode:
        if args[0] == 'squeue' and 'Invalid job id' in result.stderr:
            return ''  # Confirm terminal state through sacct below.
        raise RuntimeError(f"Command failed ({result.returncode}): {args}")
    return result.stdout


def submit(args: list[str]) -> tuple[str, str]:
    out = command([args[0], '--parsable', *args[1:]])
    line = out.strip().splitlines()[-1]
    job, _, cluster = line.partition(';')
    if not job.isdigit() or cluster not in {'wice', 'mindwell'}:
        raise ValueError(f"Unrecognized submission receipt: {line}")
    return job, cluster


def wait_jobs(jobs: list[tuple[str, str]], *, poll_seconds: int = 60) -> None:
    for job, cluster in jobs:
        while True:
            queue = command(['squeue', f'--clusters={cluster}', '-j', job, '-h', '-o', '%i %T'])
            active = [line for line in queue.splitlines() if re.match(r'^\d', line.strip())]
            if not active:
                account = command(['sacct', f'--clusters={cluster}', '-j', job, '-X', '-n', '-P',
                                   '--format=JobIDRaw,State,ExitCode'])
                rows = [line.split('|') for line in account.splitlines() if line.strip()]
                if rows and all(len(row) >= 3 and row[1] == 'COMPLETED' and row[2] == '0:0' for row in rows):
                    break
                if rows:
                    raise RuntimeError(f"Job {job} did not complete successfully: {account}")
            time.sleep(poll_seconds)


def production_stages(exp: int, kind: str):
    from src.utils.run_experiment import plan
    stages = plan(exp, ['lgd', 'pd'], True, None)
    if exp == 1:
        stages += plan(exp, ['lgd', 'pd'], True, 'ablations')
    return [stage for stage in stages if stage.kind == kind]


def submit_stage_commands(stages) -> list[tuple[str, str]]:
    jobs = []
    for stage in stages:
        if stage.complete:
            continue
        if stage.state != 'ready':
            raise RuntimeError(f"Stage is not ready: {stage.name}: {stage.state}")
        for args in stage.commands:
            jobs.append(submit(args))
    return jobs


def pipeline(record: dict, save) -> None:
    def phase(name: str):
        record['phase'] = name
        save(record)
    phase('exp0_cpu')
    job = submit(['sbatch', '--clusters=wice', 'scripts/slurm/exp0_cpu.slurm'])
    record['cpu_job'] = job; save(record)
    wait_jobs([job])
    command([sys.executable, '-m', 'src.utils.preflight', '--require-exp0-cpu'])
    phase('exp0_gpu')
    job = submit(['sbatch', '--clusters=mindwell', 'scripts/slurm/exp0.slurm'])
    record['gpu_job'] = job; save(record)
    wait_jobs([job])
    command([sys.executable, '-m', 'src.utils.preflight', '--require-exp0'])
    command([sys.executable, '-m', 'src.validation.verify'])
    phase('exp1_benchmark')
    stages = production_stages(1, 'benchmark')
    jobs = submit_stage_commands(stages)
    record['benchmark_jobs'] = jobs; save(record)
    wait_jobs(jobs)
    if not all(stage.complete for stage in production_stages(1, 'benchmark')):
        raise RuntimeError('Exp1 benchmark receipts are incomplete; Exp2 remains blocked')
    phase('exp2_submit')
    stages = production_stages(2, 'train')
    if sum(stage.total for stage in stages) != 64:
        raise RuntimeError('Exp2 is not the reviewed 64-run grid')
    jobs = submit_stage_commands(stages)
    record.update(exp2_jobs=jobs, phase='exp2_submitted', status='COMPLETE')
    save(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action='store_true')
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error('Execution requires explicit --run')
    if not os.environ.get('CREDITICL_EXPECTED_COMMIT'):
        parser.error('Freeze CREDITICL_EXPECTED_COMMIT before starting')
    args.record.parent.mkdir(parents=True, exist_ok=True)
    if args.record.exists():
        raise FileExistsError('Preserve the previous deployment receipt; choose a new record')
    def save(value):
        temp = args.record.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temp.replace(args.record)
    record = dict(status='RUNNING', commit=os.environ['CREDITICL_EXPECTED_COMMIT'])
    try:
        pipeline(record, save)
    except Exception as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        save(record)
        raise


if __name__ == '__main__':
    main()
