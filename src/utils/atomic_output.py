"""Publish complete benchmark tables without exposing a half-written CSV."""
from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4


def replace_complete(temporary: Path, target: Path) -> None:
    """Retry brief Windows sharing locks without removing the previous output."""
    for attempt in range(5):
        try:
            os.replace(temporary, target)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 4:
                raise
            time.sleep(0.05 * 2 ** attempt)


def write_csv(frame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        replace_complete(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
