"""Per-run logging: one timestamped log file plus a machine-readable metrics file.

Every run writes two things into `logs/`:

* ``<run name>_<timestamp>.log`` — human readable, one timestamped line per
  event, at INFO by default. This is the file you open when something looks off.
* ``<run name>_<timestamp>.metrics.jsonl`` — one JSON object per line. This is the
  file you load into pandas to plot learning curves and compare runs.

Both go to `$VSC_DATA` (small and backed up), never to project staging, which has
a low inode budget and does not want thousands of small files.

Why a JSON-lines file next to the text log, rather than parsing the text later:
parsing your own log format is a job nobody enjoys and it breaks the first time
someone reformats a message. Two files, each good at one thing.

Logs are flushed on every record. A job killed at the walltime limit otherwise
loses the buffered tail, which is exactly the part explaining why it died.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER_NAME = "crediticl"


class _FlushingFileHandler(logging.FileHandler):
    """Flush after every record, so a killed job keeps its last lines."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class MetricsWriter:
    """Append-only JSON-lines sink for numbers.

    `enabled=False` writes nothing but still keeps the records in memory, so a
    local run can inspect `.records` without leaving files behind.
    """

    def __init__(self, path: Path, *, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def write(self, record: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - self._t0, 2),
            **record,
        }
        self.records.append(record)
        if not self.enabled:
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def setup_logging(
    run_name: str,
    log_dir: Path | str,
    *,
    level: str = "INFO",
    console: bool = True,
    to_file: bool | None = None,
) -> tuple[logging.Logger, MetricsWriter, Path]:
    """Create the logger and metrics writer for one run.

    `to_file` defaults to **True on the cluster and False locally**. The log files
    exist so a run on VSC can be debugged from its output alone; running something
    on a laptop should not litter `logs/` with files nobody will read. Set it
    explicitly to override either way.

    Returns (logger, metrics writer, log file path — which may not exist when
    `to_file` is False).
    """
    from src.utils.paths import on_vsc

    if to_file is None:
        to_file = on_vsc()

    log_dir = Path(log_dir)
    if to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Keep the filename usable: a long grid tag would otherwise blow past
    # filesystem limits and be unreadable in a terminal anyway.
    safe = run_name if len(run_name) <= 120 else run_name[:110] + "_etc"
    log_path = log_dir / f"{safe}_{stamp}.log"
    metrics_path = log_dir / f"{safe}_{stamp}.metrics.jsonl"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()  # a resumed run must not double-log
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_file:
        fh = _FlushingFileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger, MetricsWriter(metrics_path, enabled=to_file), log_path


def get_logger() -> logging.Logger:
    """The run logger. Safe to call before `setup_logging` — it just goes nowhere."""
    return logging.getLogger(LOGGER_NAME)


def close_logging() -> None:
    """Close and detach the file handler.

    Not optional housekeeping. An open `FileHandler` keeps the log file locked,
    which on Windows makes the containing directory undeletable — that is how this
    was found, when a temp directory in the smoke test refused to clean up. On
    Linux the same leak is silent but still real: repeated `setup_logging` calls in
    one process would accumulate open file descriptors.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


#: Every SLURM variable worth keeping. A cluster run cannot be watched, so the allocation it
#: actually got — not the one the submit script asked for — has to be in the log or it is lost.
SLURM_VARS = (
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT", "SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS", "SLURMD_NODENAME", "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
    "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS",
    "SLURM_GPUS_PER_NODE", "SLURM_SUBMIT_DIR", "SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME",
)

#: Thread and allocator limits. The numeric libraries read these at import and ignore them
#: afterwards, so what they were AT IMPORT is the only value that matters — and a login-node
#: run already died once with `pthread_create` EAGAIN because nothing recorded them.
THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER", "CREDITICL_STAGING_ROOT",
)


def _git_state() -> dict[str, Any]:
    """The commit this ran from, and the tfm-library pin.

    A result that cannot be traced to a commit is a result nobody can reproduce, and the pin
    matters as much: every claim checked against the library depends on which snapshot. `dirty`
    is the honest part — a run from uncommitted code is not the commit it names.
    """
    import subprocess

    out: dict[str, Any] = {}
    root = Path(__file__).resolve().parents[2]

    def _run(*args: str) -> str | None:
        try:
            r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    out["git_commit"] = _run("git", "rev-parse", "--short", "HEAD")
    status = _run("git", "status", "--porcelain")
    out["git_dirty"] = bool(status) if status is not None else None
    pin = _run("git", "submodule", "status", "tfm-library")
    if pin:
        out["tfm_library_pin"] = pin.split()[0].lstrip("+-U")[:12]
    return {k: v for k, v in out.items() if v is not None}


def _torch_state() -> dict[str, Any]:
    """Everything about the chip and the wheel, because "which GPU" is the first question.

    `exact_arch_shipped` is the one that has actually bitten: a wheel built for sm_90 running on
    an sm_100 card falls back to JIT or to a slower path, and NOTHING says so except this.
    """
    info: dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — worth logging, not fatal
        return {"torch": f"unavailable ({exc})"}

    info["torch"] = torch.__version__
    info["torch_threads"] = torch.get_num_threads()
    info["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        return info
    try:
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        arches = list(torch.cuda.get_arch_list())
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "capability": f"{cap[0]}.{cap[1]}",
            "gpu_memory_gb": round(props.total_memory / 1e9, 1),
            "multi_processors": props.multi_processor_count,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "compiled_for": arches,
            "exact_arch_shipped": f"sm_{cap[0]}{cap[1]}" in arches,
        })
    except Exception as exc:  # noqa: BLE001
        info["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from ..models.backends import sdpa_report

        info["attention"] = sdpa_report()
    except Exception as exc:  # noqa: BLE001
        info["attention"] = f"unavailable ({exc})"
    return info


def _storage_state() -> dict[str, Any]:
    """Where this run will write, and whether it CAN.

    A run that cannot write where it thinks it can reroutes silently to `$VSC_DATA` and only
    fails once the 75 GiB quota is gone — which is exactly what a staging directory left at
    mode 0500 did. Recording writability per root turns that into one line at the top.
    """
    out: dict[str, Any] = {}
    try:
        from . import paths
    except Exception as exc:  # noqa: BLE001
        return {"paths": f"unavailable ({exc})"}
    for name, fn in (
        ("outputs", paths.outputs_dir), ("logs", paths.logs_dir),
        ("manifests", paths.manifests_dir), ("results", paths.results_dir),
        ("checkpoints", paths.checkpoints_dir), ("datasets", paths.datasets_dir),
        ("prior_cache", paths.prior_cache_root), ("staging", paths.staging_root),
    ):
        try:
            path = Path(fn())
            out[name] = {"path": str(path), "exists": path.is_dir(),
                         "writable": os.access(path, os.W_OK) if path.is_dir() else None}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def log_environment(logger: logging.Logger, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record what this run actually ran on — machine, allocation, code, and storage.

    THE LOG IS THE ONLY CHANNEL. Nobody can attach to a compute node, and by the time anything
    is known the job is over; whatever was not written down did not happen. So this errs
    heavily towards writing too much. It is emitted twice: once as an `environment:` JSON line
    that stays machine-readable and diffable between runs, and once as a readable block,
    because both audiences are real.

    Everything is best-effort. A missing `git`, an unreadable path, or a torch without CUDA all
    degrade to one absent key: **a diagnostic that can kill a three-day run is worse than no
    diagnostic.**
    """
    now = datetime.now().astimezone()
    info: dict[str, Any] = {
        "started": now.isoformat(timespec="seconds"),
        "started_epoch": int(now.timestamp()),
        "timezone": str(now.tzinfo),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
    }
    info.update(_git_state())
    for var in (*SLURM_VARS, *THREAD_VARS):
        if os.environ.get(var):
            info[var.lower()] = os.environ[var]
    info.update(_torch_state())
    info["storage"] = _storage_state()
    if extra:
        info.update(extra)

    logger.info("environment: %s", json.dumps(info, default=str))
    _log_environment_readable(logger, info)
    return info


def _log_environment_readable(logger: logging.Logger, info: dict[str, Any]) -> None:
    """The same facts as a block a person can scan. The JSON line above is for diffing."""
    att = info.get("attention") if isinstance(info.get("attention"), dict) else {}
    slurm = " ".join(
        f"{k.replace('slurm_', '').replace('slurmd_', '')}={info[k]}"
        for k in (
            "slurm_job_id", "slurm_array_task_id", "slurm_job_partition", "slurmd_nodename",
            "slurm_cpus_per_task", "slurm_gpus_on_node", "slurm_mem_per_node",
        ) if info.get(k)
    )
    kernels = (
        "OK - exact arch shipped" if info.get("exact_arch_shipped")
        else "*** NO KERNEL FOR THIS ARCH - expect a slow fallback ***"
    )
    rows = [
        "=" * 78,
        " MACHINE — what this run actually got",
        "=" * 78,
        f"  started        : {info.get('started')}  (pid {info.get('pid')})",
        f"  host / platform: {info.get('hostname')}  {info.get('platform')}",
        f"  code           : commit {info.get('git_commit', '?')}"
        f"{'  *** UNCOMMITTED CHANGES ***' if info.get('git_dirty') else ''}"
        f"   tfm-library pin {info.get('tfm_library_pin', '?')}",
        f"  python / torch : {info.get('python')} / {info.get('torch')}",
    ]
    if slurm:
        rows.append(f"  slurm          : {slurm}")
    if info.get("cuda_available"):
        rows += [
            f"  GPU            : {info.get('gpu')} x{info.get('gpu_count')}  "
            f"capability {info.get('capability')}  {info.get('gpu_memory_gb')} GB  "
            f"{info.get('multi_processors')} SMs",
            f"  CUDA / cuDNN   : {info.get('cuda')} / {info.get('cudnn')}",
            f"  wheel kernels  : {kernels}   compiled_for={info.get('compiled_for')}",
        ]
    else:
        rows.append("  GPU            : NONE — running on CPU")
    if att:
        rows.append(f"  attention      : using {att.get('using')}, excluding {att.get('excluded')}")
    rows.append(
        f"  threads        : torch={info.get('torch_threads')} "
        f"OMP={info.get('omp_num_threads', 'unset')} cores={info.get('cpu_count')}"
    )
    for name, st in (info.get("storage") or {}).items():
        if not isinstance(st, dict) or "path" not in st:
            continue
        mark = "ok" if st.get("writable") else ("NOT WRITABLE" if st.get("exists") else "missing")
        rows.append(f"  {name:<14} : [{mark:>12}] {st['path']}")
    rows.append("=" * 78)
    for line in rows:
        logger.info("%s", line)


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("=" * 62)
    logger.info(title)
    logger.info("=" * 62)
