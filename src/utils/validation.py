"""CPU and GPU Experiment 0 jobs, and a fail-closed production gate.

Every invocation gets its own output namespace. A killed/failed invocation leaves
RUNNING/FAILED, never a stale PASS from an earlier job. The receipt binds tests,
code, configuration, packages, input data, weights and output artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name

SCHEMA = 2
ATTEMPT_ENV = "CREDITICL_EXP0_ATTEMPT"


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    phase: str = "diagnostics"
    cpu: bool = False
    stop_run: str | None = None
    junit: str | None = None
    report: str | None = None


def plan(phase: str = "gpu") -> list[Stage]:
    """The mandatory manifest; no optional bypasses in the production receipt."""
    from src.eval.benchmark_status import diagnostic_datasets, n_slots, slot_for
    from src.utils.preflight import CHECK_SECTIONS

    py = sys.executable
    report = paths.experiment_dir(0) / "report"
    if phase not in ("cpu", "gpu"):
        raise ValueError(f"Unknown validation phase: {phase}")
    stages = [Stage("preflight", (py, "-m", "src.utils.preflight"))]
    suites = (
        ("unit", ("tests", "-m", "not vsc"), True),
        ("vsc", ("tests/test_vsc.py", "--run-vsc", "-x"), False),
    )
    if phase == "cpu":
        stages.insert(0, Stage("preprocess", (py, "scripts/preprocess.py", "--task", "both")))
    for name, args, cpu in (suites[:1] if phase == "cpu" else suites[1:]):
        junit = str(report / f"{name}.xml")
        stages.append(Stage(name, (py, "-m", "pytest", "-q", *args, f"--junitxml={junit}"),
                            cpu=cpu, junit=junit))
    gpu_sections = {"env", "weights"}
    sections = CHECK_SECTIONS - {"train"} - gpu_sections if phase == "cpu" else gpu_sections
    for section in sorted(sections):
        folder = report / section
        stages.append(Stage(f"check-{section}", (py, "scripts/exp0_checks.py", "--only", section,
                                                 "--out", str(folder)), report=str(folder / "checks.json")))
    if phase == "cpu":
        return stages
    for track in paths.TASKS:
        config = f"config/Exp0_{track.upper()}.yaml"
        runs = expand_with_seeds(load(config))
        for run in runs:
            folder = report / f"train-{track}-{run['arm']}"
            stages.append(Stage(f"smoke-{track}-{run['arm']}",
                                (py, "scripts/exp0_checks.py", "--only", "train", "--track", track,
                                 "--arm", run["arm"], "--out", str(folder)),
                                report=str(folder / "checks.json")))
        stages.append(Stage(f"equivalence-{track}", (py, "scripts/check_equivalence.py", "--config",
                            f"config/Exp1_{track.upper()}.yaml", "--index", "0", "--steps", "20",
                            "--device", "cuda")))
        for index, run in enumerate(runs):
            args = (py, "scripts/pretrain.py", "--config", config, "--index", str(index),
                    "--device", "cuda", "--num-workers",
                    str(max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1)))
            name = run_name(run)
            stages.append(Stage(f"stop-{track}-{index}", args, "training", stop_run=name))
            stages.append(Stage(f"resume-{track}-{index}", args, "training"))
        credit_names, ood_names = diagnostic_datasets(track)
        datasets = ",".join(credit_names)
        for index in range(n_slots(0, track)):
            slot = slot_for(0, track, index)
            common = ("--models", slot.model, "--seeds", "0", "--tag", slot.tag)
            if slot.kind == "arm":
                checkpoint = paths.run_checkpoints_dir(run_name(runs[slot.arm])) / f"step-{slot.step}.ckpt"
                common += ("--checkpoint", str(checkpoint))
            if slot.context is not None:
                common += ("--max-context-rows", str(slot.context))
            # Establish full-table capacity with the released/reference models
            # before spending hours on the six longer diagnostic training runs.
            # These are the same required benchmark cells, executed once.
            scoring_phase = "capacity" if slot.kind == "reference" else "benchmark"
            stages.extend([
                Stage(f"credit-{track}-{index}", (py, "scripts/evaluate.py", "--task", track,
                      "--datasets", datasets, *common), scoring_phase),
                Stage(f"ood-{track}-{index}", (py, "scripts/evaluate_ood.py", "--kinds",
                      "classification" if track == "pd" else "regression", "--datasets", ",".join(ood_names),
                      *common), scoring_phase),
                Stage(f"receipt-{track}-{index}", (py, "-m", "src.eval.benchmark_status", "--exp", "0",
                      "--track", track, "--index", str(index), "--write"), scoring_phase),
            ])
    return sorted(stages, key=lambda s: ("diagnostics", "capacity", "training", "benchmark").index(s.phase))


def receipt_path(phase: str = "gpu") -> Path:
    return paths.outputs_dir() / "experiment_0" / "report" / ("cpu_validation.json" if phase == "cpu" else "validation.json")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


@contextmanager
def attempt_scope(name: str):
    import re

    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError("Invalid validation attempt name")
    old = os.environ.get(ATTEMPT_ENV)
    os.environ[ATTEMPT_ENV] = name
    try:
        paths.experiment_dir(0)  # reject path traversal before writing anything
        yield
    finally:
        if old is None:
            os.environ.pop(ATTEMPT_ENV, None)
        else:
            os.environ[ATTEMPT_ENV] = old


def input_inventory() -> dict:
    """Use the same byte-level data/weight provenance as real benchmark receipts."""
    from src.eval.benchmark_status import n_slots, request, slot_for

    result = {}
    for track in paths.TASKS:
        for index in range(n_slots(1, track)):
            slot = slot_for(1, track, index)
            if slot.kind == "reference" and slot.context is None:
                _, wanted, _ = request(1, track, index)
                result[f"{track}/{slot.model}"] = wanted
    if len(result) != 8:
        raise ValueError("Expected four reference models for each track")
    return result


def check_junit(path: Path, *, mandatory: bool) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {k: sum(int(s.get(k, "0")) for s in suites)
              for k in ("tests", "failures", "errors", "skipped")}
    if not counts["tests"] or counts["failures"] or counts["errors"] or (mandatory and counts["skipped"]):
        raise ValueError(f"Incomplete/failed test coverage: {counts}")
    return counts


def failure_details(stage: Stage, log: Path) -> dict:
    """Keep the cause in the main job log as well as the individual stage log.

    Read only a bounded tail, even when a library has printed millions of lines.
    Missing/malformed diagnostic files must never hide the original exception.
    """
    details = {"log": str(log)}
    try:
        with log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
        details["log_tail"] = "\n".join(tail.splitlines()[-40:])[-8000:]
    except OSError:
        pass
    if stage.junit:
        try:
            root = ET.parse(stage.junit).getroot()
            details["failed_tests"] = [
                f"{case.get('classname', '')}::{case.get('name', '')}"
                for case in root.iter("testcase")
                if case.find("failure") is not None or case.find("error") is not None
            ][:20]
        except (OSError, ET.ParseError):
            pass
    return details


def kill_process_tree(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    elif proc.poll() is None:
        proc.kill()
    proc.wait()


def execute(stage: Stage, log: Path, *, timeout: float) -> dict:
    """Bounded child processes isolate poisoned CUDA contexts and worker leaks."""
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    if stage.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop(ATTEMPT_ENV, None)  # portable tests use their own temporary output fixtures
    started = time.monotonic()
    signalled = False
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(stage.command, cwd=paths.REPO_ROOT, env=env, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
        try:
            while proc.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"{stage.name} exceeded its remaining time budget")
                if stage.stop_run and not signalled:
                    from src.train.checkpoint import latest_checkpoint
                    from src.utils.working_storage import checkpoint_work_dir

                    checkpoint = latest_checkpoint(checkpoint_work_dir(paths.run_checkpoints_dir(stage.stop_run)))
                    if checkpoint is not None:
                        proc.send_signal(signal.SIGUSR1)
                        signalled = True
                time.sleep(1)
            rc = proc.returncode
        finally:
            # Even a crashed parent can leave DataLoader workers behind on the GPU node.
            kill_process_tree(proc)
    expected = 64 if stage.stop_run else 0
    if rc != expected or (stage.stop_run and not signalled):
        raise RuntimeError(f"{stage.name}: exit={rc}, expected={expected}; read {log}")
    detail = {"seconds": round(time.monotonic() - started, 2), "log": str(log)}
    if stage.stop_run:
        summary = json.loads(paths.run_summary_path(stage.stop_run).read_text(encoding="utf-8"))
        if summary.get("completed") or summary.get("stopped_by_signal") != "SIGUSR1" or summary.get("steps", 0) < 1:
            raise ValueError(f"{stage.stop_run}: forced interruption did not save an incomplete run")
        detail["stopped_at"] = summary["steps"]
    if stage.junit:
        detail["tests"] = check_junit(Path(stage.junit), mandatory=(stage.name == "vsc"))
    if stage.report:
        saved = json.loads(Path(stage.report).read_text(encoding="utf-8"))
        if not saved.get("rows") or saved.get("failed") != 0 or any(r["status"] == "FAIL" for r in saved["rows"]):
            raise ValueError(f"Failed or empty check report: {stage.report}")
    return detail


def collect_artifacts(stages: list[Stage], phase: str = "gpu") -> dict[str, str]:
    """Verify actual training completion, resume, monitoring and paired evaluation."""
    import numpy as np
    import pandas as pd

    from src.eval.benchmark_status import check, digest, n_slots, request, slot_for
    from src.eval.benchmark_status import receipt_path as bench_receipt
    from src.train.checkpoint import validate_resume_config, validate_training_fingerprint

    files = []
    for stage in stages:
        files.append(paths.experiment_dir(0) / "logs" / f"{stage.name}.log")
        files.extend(Path(p) for p in (stage.junit, stage.report) if p)
    if phase == "cpu":
        return {str(p.resolve()): digest(p) for p in files}
    for track in paths.TASKS:
        for run in expand_with_seeds(load(f"config/Exp0_{track.upper()}.yaml")):
            name = run_name(run)
            folder = paths.run_dir(name)
            summary = json.loads(paths.run_summary_path(name).read_text(encoding="utf-8"))
            validate_training_fingerprint(summary)
            validate_resume_config(json.loads(paths.run_config_path(name).read_text(encoding="utf-8")), run)
            if (not summary.get("completed") or summary.get("steps") != run["train"]["max_steps"]
                    or not 0 < summary.get("resumed_at", 0) < summary["steps"]
                    or summary.get("datasets_seen") != summary["steps"] * run["train"]["batch_size"]):
                raise ValueError(f"Incomplete training or resume: {name}")
            progress = pd.read_csv(folder / "progress.csv")
            metric = "roc_auc" if track == "pd" else "r2"
            metrics = [c for c in progress if c.startswith(("real__", "ood__")) and c.endswith(f"__{metric}")]
            errors = progress.get("n_errors", pd.Series(0, index=progress.index)).fillna(0)
            if (len(progress) < 2 or not any(c.startswith("real__") for c in metrics)
                    or not any(c.startswith("ood__") for c in metrics) or errors.gt(0).any()
                    or not np.isfinite(progress[metrics].to_numpy(dtype=float)).all()):
                raise ValueError(f"Missing/nonfinite monitoring: {name}")
            weights = pd.read_csv(folder / "weights.csv")
            if len(weights) < 2 or not np.isfinite(weights.drift_all).all() or weights.drift_all.iloc[-1] <= 0:
                raise ValueError(f"Weights did not change: {name}")
            telemetry = pd.read_csv(folder / "telemetry.csv")
            if telemetry.empty or not all(g.step.is_monotonic_increasing for _, g in telemetry.groupby("kind")):
                raise ValueError(f"Missing/nonmonotonic telemetry: {name}")
            files.extend(folder / f for f in ("config.json", "summary.json", "progress.csv", "weights.csv", "telemetry.csv"))
            files.append(paths.run_checkpoints_dir(name) / f"step-{summary['steps']}.ckpt")
        for index in range(n_slots(0, track)):
            if not check(0, track, index):
                raise ValueError(f"Stale/incomplete benchmark: {track}/{index}")
            _, _, results = request(0, track, index)
            files.extend(results.values())
            files.append(bench_receipt(slot_for(0, track, index).tag))
    return {str(p.resolve()): digest(p) for p in files}


def readiness(phase: str = "gpu") -> list:
    from src.eval.benchmark_status import input_digest
    from src.utils.environment import environment_fingerprint
    from src.utils.preflight import Check, source_fingerprint

    try:
        if phase == "gpu":
            cpu = readiness("cpu")
            if not cpu[0].ok:
                raise ValueError(cpu[0].detail)
        saved = json.loads(receipt_path(phase).read_text(encoding="utf-8"))
        if saved.get("schema") != SCHEMA or saved.get("status") != "PASS":
            raise ValueError(f"Latest Experiment 0 status: {saved.get('status', 'missing')}")
        with attempt_scope(saved["attempt"]):
            if saved.get("source_fingerprint") != source_fingerprint():
                raise ValueError("Code/config/tests changed since Experiment 0")
            if saved.get("environment_fingerprint") != environment_fingerprint():
                raise ValueError("Installed packages changed since Experiment 0")
            if phase == "gpu" and saved.get("cpu_receipt") != input_digest(receipt_path("cpu")):
                raise ValueError("CPU validation was rerun after GPU validation")
            expected = [s.name for s in plan(phase)]
            actual = saved.get("stages", [])
            if [s["name"] for s in actual] != expected or any(s["status"] != "PASS" for s in actual):
                raise ValueError("Not every mandatory stage passed")
            if saved.get("inputs") != input_inventory():
                raise ValueError("Input data/released weights changed since Experiment 0")
            artifacts = saved.get("artifacts", {})
            if not artifacts or any(input_digest(Path(p)) != h for p, h in artifacts.items()):
                raise ValueError("Validation artifacts missing or modified")
        return [Check(f"Experiment 0 {phase}", True, f"PASS: {saved['attempt']}; all stages and artifacts current")]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [Check(f"Experiment 0 {phase}", False, f"{exc}. Run the CPU job, then the GPU job, for the current code.")]


def run_validation(*, hours: float = 23.5, phase: str = "gpu") -> int:
    from filelock import FileLock

    from src.utils.environment import environment_fingerprint
    from src.utils.preflight import source_fingerprint

    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit an Experiment 0 Slurm script; validation requires a Linux Slurm allocation")
    pointer = receipt_path(phase)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(receipt_path().with_suffix(".lock")), timeout=0):
        attempt = phase + "_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"_j{os.environ['SLURM_JOB_ID']}_{uuid.uuid4().hex[:8]}"
        with attempt_scope(attempt):
            saved = {"schema": SCHEMA, "phase": phase, "status": "RUNNING", "attempt": attempt, "stages": [],
                     "source_fingerprint": source_fingerprint(), "environment_fingerprint": environment_fingerprint()}
            history = paths.experiment_dir(0) / "report" / "validation.json"

            def publish():
                write_json(history, saved)
                write_json(pointer, saved)

            publish()
            deadline = time.monotonic() + hours * 3600
            try:
                if phase == "gpu":
                    cpu_check = readiness("cpu")[0]
                    if not cpu_check.ok:
                        raise ValueError(f"CPU validation must pass first: {cpu_check.detail}")
                    from src.eval.benchmark_status import digest
                    saved["cpu_receipt"] = digest(receipt_path("cpu"))
                stages = plan(phase)
                if phase == "gpu":
                    saved["inputs"] = input_inventory()
                for stage in stages:
                    if any(s["status"] == "FAIL" for s in saved["stages"]) and stage.phase != "diagnostics":
                        raise RuntimeError("Diagnostics failed; longer training/benchmark stages were not started")
                    print(f"[Experiment 0] START {stage.name}", flush=True)
                    row = {"name": stage.name, "status": "PASS"}
                    log = paths.experiment_dir(0) / "logs" / f"{stage.name}.log"
                    started = time.monotonic()
                    try:
                        row.update(execute(stage, log,
                                           timeout=max(0, deadline - time.monotonic())))
                    except Exception as exc:
                        row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                                   seconds=round(time.monotonic() - started, 2),
                                   **failure_details(stage, log))
                    saved["stages"].append(row)
                    if phase == "cpu" and stage.name == "preflight" and row["status"] == "PASS":
                        saved["inputs"] = input_inventory()
                    publish()
                    print(f"[Experiment 0] { {k: value for k, value in row.items() if k != 'log_tail'} }", flush=True)
                    if row.get("log_tail"):
                        print(f"[Experiment 0] Last lines of {stage.name}:\n{row['log_tail']}", flush=True)
                    if row["status"] == "FAIL" and (stage.name in ("preflight", "preprocess", "vsc") or stage.phase != "diagnostics"):
                        raise RuntimeError(f"Stopping after {stage.name}; dependent work cannot be validated")
                if any(s["status"] != "PASS" for s in saved["stages"]):
                    raise RuntimeError("One or more validation stages failed")
                saved["artifacts"] = collect_artifacts(stages, phase)
                if (saved["inputs"] != input_inventory() or saved["source_fingerprint"] != source_fingerprint()
                        or saved["environment_fingerprint"] != environment_fingerprint()):
                    raise ValueError("Code, environment or inputs changed during validation")
                # Exp0 probes only selected tables; never publish these partial
                # reference results into the production benchmark namespace.
                saved["status"] = "PASS"
            except Exception as exc:
                saved.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            finally:
                publish()
            if saved.get("error"):
                print(f"[Experiment 0] {saved['error']}", flush=True)
            print(f"Experiment 0 {saved['status']}: {pointer}", flush=True)
            return 0 if saved["status"] == "PASS" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true", help="print stages without executing or writing")
    ap.add_argument("--phase", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--hours", type=float, default=None)
    args = ap.parse_args(argv)
    if args.plan:
        for stage in plan(args.phase):
            print(f"{stage.phase:12} {stage.name}")
        return 0
    return run_validation(phase=args.phase, hours=args.hours or (3.8 if args.phase == "cpu" else 23.5))
