"""Mindwell's repeated large I/O belongs on GPFS; project storage is permanent.

Inputs are copied once per job. Rolling checkpoints stay on GPFS; permanent,
interrupted and final checkpoints are copied back atomically before completion.
The canonical paths in receipts always name project storage, never scratch.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from pathlib import Path

from filelock import FileLock


def work_root() -> Path | None:
    override = os.environ.get("CREDITICL_WORK_ROOT")
    if override:
        return Path(override)
    if os.environ.get("SLURM_CLUSTER_NAME", "").lower() != "mindwell":
        return None
    scratch = os.environ.get("VSC_SCRATCH_GPFS1") or os.environ.get("VSC_SCRATCH")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not scratch or not re.fullmatch(r"[0-9_]+", job):
        raise RuntimeError("Mindwell requires GPFS scratch and a Slurm job ID; refusing Lustre working I/O")
    return Path(scratch) / "CreditICL" / "jobs" / job


def _key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]


def atomic_copy(source: Path, target: Path) -> None:
    """Keep the previous complete file if a copy fails (e.g. quota exhausted)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.copying")
    try:
        shutil.copy2(source, tmp)
        # Preserve mtime for cache identity, refresh atime for VSC's scratch purge.
        os.utime(tmp, ns=(time.time_ns(), tmp.stat().st_mtime_ns))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def local_input(source: str | Path) -> Path:
    source = Path(source)
    root = work_root()
    if root is None or source.resolve().is_relative_to(root.resolve()):
        return source
    target = root / "inputs" / _key(source) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock", timeout=600):
        stat = source.stat()
        if (not target.is_file() or target.stat().st_size != stat.st_size
                or target.stat().st_mtime_ns != stat.st_mtime_ns):
            atomic_copy(source, target)
    return target


def checkpoint_work_dir(permanent: Path) -> Path:
    root = work_root()
    if root is None:
        return permanent
    folder = root / "checkpoints" / _key(permanent)
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder) + ".lock", timeout=600):
        marker = folder / "destination.txt"
        if not marker.exists():
            # Only the latest state is needed to resume. Historical permanent
            # snapshots remain on project storage and are never removed here.
            from src.train.checkpoint import latest_checkpoint

            previous = latest_checkpoint(permanent)
            if previous is not None:
                atomic_copy(previous, folder / previous.name)
            marker.write_text(str(permanent.resolve()), encoding="utf-8")
    return folder


def publish_checkpoint(folder: Path, step: int, *, permanent_every: int,
                       final: bool, stopped: bool, keep_steps=()) -> None:
    marker = folder / "destination.txt"
    if not marker.is_file() or not (final or stopped or step in keep_steps or (permanent_every > 0 and step % permanent_every == 0)):
        return
    destination = Path(marker.read_text(encoding="utf-8"))
    atomic_copy(folder / f"step-{step}.ckpt", destination / f"step-{step}.ckpt")


def cleanup_successful_job() -> None:
    """Release this job's scratch after verifying its latest states are permanent.

    Failed/cancelled jobs never call this. Explicit work-root overrides are left
    alone: only the fixed scratch/CreditICL/jobs/<current job> layout is removable.
    """
    if os.environ.get("CREDITICL_WORK_ROOT") or os.environ.get("SLURM_CLUSTER_NAME", "").lower() != "mindwell":
        return
    root = work_root()
    if root is None or not root.exists():
        return
    scratch = Path(os.environ.get("VSC_SCRATCH_GPFS1") or os.environ["VSC_SCRATCH"]).resolve()
    base = scratch / "CreditICL" / "jobs"
    if (root.resolve() != base / os.environ["SLURM_JOB_ID"]
            or root.name != os.environ["SLURM_JOB_ID"]):
        raise RuntimeError("Scratch cleanup escaped the current job directory")
    # Validate the entire tree before deleting any file, including Windows tests.
    pending = [root]
    while pending:
        path = pending.pop()
        if (path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
                or not path.resolve().is_relative_to(root.resolve())):
            raise RuntimeError(f"Scratch cleanup refuses linked paths: {path}")
        if path.is_dir():
            pending.extend(path.iterdir())
    from src.train.checkpoint import latest_checkpoint

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").digest()

    for folder in (root / "checkpoints").glob("*"):
        if not folder.is_dir():
            continue
        latest = latest_checkpoint(folder)
        if latest is None:
            continue
        marker = folder / "destination.txt"
        if not marker.is_file():
            raise RuntimeError("Scratch checkpoint has no permanent destination; retained")
        published = Path(marker.read_text(encoding="utf-8")) / latest.name
        if not published.is_file() or digest(latest) != digest(published):
            raise RuntimeError("Latest scratch checkpoint is not safely published; retained")
    shutil.rmtree(root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-success", action="store_true")
    if parser.parse_args().cleanup_success:
        cleanup_successful_job()
