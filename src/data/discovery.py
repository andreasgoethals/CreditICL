"""Find which raw datasets exist, wherever they happen to live.

Repo first, then project storage — so a laptop with the data checked out works
with no configuration, and the same code finds it on the cluster.
"""

from __future__ import annotations

from src.utils.paths import TASKS, raw_task_dirs

RAW_EXTENSIONS = (".csv", ".parquet")


def list_datasets(task: str) -> list[str]:
    """Dataset slugs present on disk for a task, sorted, de-duplicated.

    A slug found in more than one root is returned once; `find_raw_path` decides
    which copy actually gets read (repo wins).
    """
    seen: set[str] = set()
    for directory in raw_task_dirs(task):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.suffix in RAW_EXTENSIONS and entry.is_file():
                seen.add(entry.stem)
    return sorted(seen)


def list_all_datasets() -> dict[str, list[str]]:
    return {task: list_datasets(task) for task in TASKS}


def describe_availability() -> str:
    """A short report for the log, so a run records what it could actually see."""
    lines = []
    for task in TASKS:
        found = list_datasets(task)
        lines.append(f"{task}: {len(found)} raw datasets")
        for directory in raw_task_dirs(task):
            lines.append(f"    searched {directory}  (exists: {directory.is_dir()})")
        if found:
            lines.append(f"    found: {', '.join(found)}")
    return "\n".join(lines)
