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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER_NAME = "crediticl"


class _FlushingFileHandler(logging.FileHandler):
    """Flush after every record, so a killed job keeps its last lines."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class MetricsWriter:
    """Append-only JSON-lines sink for numbers."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def write(self, record: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - self._t0, 2),
            **record,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def setup_logging(
    run_name: str,
    log_dir: Path | str,
    *,
    level: str = "INFO",
    console: bool = True,
) -> tuple[logging.Logger, MetricsWriter, Path]:
    """Create the logger and metrics writer for one run.

    Returns (logger, metrics writer, log file path).
    """
    log_dir = Path(log_dir)
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

    fh = _FlushingFileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger, MetricsWriter(metrics_path), log_path


def get_logger() -> logging.Logger:
    """The run logger. Safe to call before `setup_logging` — it just goes nowhere."""
    return logging.getLogger(LOGGER_NAME)


def log_environment(logger: logging.Logger, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record what this run actually ran on.

    Worth the twenty lines: when two runs disagree, the first question is always
    whether they ran on the same thing, and six months later nobody remembers.
    """
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }

    for var in (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
        "SLURMD_NODENAME",
        "CUDA_VISIBLE_DEVICES",
    ):
        if os.environ.get(var):
            info[var.lower()] = os.environ[var]

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda"] = torch.version.cuda
    except Exception as exc:  # torch missing or broken — worth logging, not fatal
        info["torch"] = f"unavailable ({exc})"

    if extra:
        info.update(extra)

    logger.info("environment: %s", json.dumps(info, default=str))
    return info


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("=" * 62)
    logger.info(title)
    logger.info("=" * 62)
