"""Handshake before a validation parent deliberately signals a training child."""
from __future__ import annotations

import os
from pathlib import Path

READY_ENV = "CREDITICL_SIGNAL_READY"


def publish() -> None:
    """Called only after the training signal handlers have been installed."""
    value = os.environ.get(READY_ENV)
    if value:
        path = Path(value)
        temporary = path.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(str(os.getpid()), encoding="utf-8")
        os.replace(temporary, path)


def matches(path: Path, pid: int) -> bool:
    try:
        return path.read_text(encoding="utf-8") == str(pid)
    except OSError:
        return False
