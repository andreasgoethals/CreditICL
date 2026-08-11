"""List and clean up everything a previous run produced.

WHY: after a failed or superseded run the tree holds logs, checkpoints, metrics,
manifests and generated prior pools. Some are tiny and some are tens of gigabytes, and
they live on **two different storage tiers**, so "delete the last run" is not one `rm`.
Doing it by hand is how someone eventually deletes `data/raw`.

THE TWO TIERS, and why it matters here:

    project staging ($CREDITICL_STAGING_ROOT / /lustre1/...)   BIG, no backup
        checkpoints/      trained weights, ~100 MB each x 48 runs
        prior_cache/      generated datasets, ~39 GB per variant
        data/processed/   rebuildable in minutes from raw

    personal data ($VSC_DATA)                                   SMALL, backed up
        output/logs/      timestamped run logs
        output/manifests/ training progress CSVs
        output/<run>/     metrics.jsonl, resolved configs

NEVER TOUCHED, at any protection level: `data/raw` (irreplaceable — the datasets
themselves) and `checkpoints/` in the repo (the *downloaded* TabPFN/TabICL weights,
which are not ours and would have to be fetched again). Those are excluded by
construction rather than by a flag, so no combination of arguments can remove them.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.paths import (
    REPO_ROOT,
    checkpoints_dir,
    logs_dir,
    manifests_dir,
    outputs_dir,
    prior_cache_dir,
    results_dir,
)

#: What each category means, in the order a person usually wants to remove them:
#: cheap-to-regenerate first, expensive last.
CATEGORIES = ("logs", "manifests", "metrics", "results", "checkpoints", "prior_pools")

#: Categories that cost real compute to rebuild. Removing these needs an explicit opt-in
#: because "clean up the logs" should never quietly delete 39 GB of generated priors or
#: a week of training.
EXPENSIVE = ("checkpoints", "prior_pools")


@dataclass
class Artifact:
    """One removable thing."""

    category: str
    path: Path
    n_files: int
    bytes: int

    @property
    def mb(self) -> float:
        return self.bytes / 1e6

    def describe(self) -> str:
        size = f"{self.mb:8.1f} MB" if self.mb < 1000 else f"{self.mb / 1000:8.2f} GB"
        return f"  {self.category:<12} {size}  {self.n_files:>6} files  {self.path}"


#: Structure markers, tracked in git so empty result directories survive a clone. They
#: are not run output, so they are neither counted nor deleted — removing them would
#: quietly change what git tracks.
KEEP_FILES = {".gitkeep", ".gitignore"}


def _measure(path: Path) -> tuple[int, int]:
    """(file count, total bytes) under a path, ignoring structure markers."""
    if not path.exists():
        return 0, 0
    if path.is_file():
        return (0, 0) if path.name in KEEP_FILES else (1, path.stat().st_size)
    n = total = 0
    for p in path.rglob("*"):
        if p.is_file() and p.name not in KEEP_FILES:
            n += 1
            # A file vanishing mid-walk is not worth failing over.
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return n, total


def _protected() -> list[Path]:
    """Paths that must never be removed, whatever the arguments say."""
    return [
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "checkpoints",  # DOWNLOADED TabPFN/TabICL weights, not ours
        REPO_ROOT / "src",
        REPO_ROOT / "config",
        REPO_ROOT / "tests",
        REPO_ROOT / "tfm-library",
    ]


def _is_protected(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return True  # cannot resolve it -> refuse to touch it
    for guard in _protected():
        try:
            g = guard.resolve()
        except OSError:
            continue
        if resolved == g or g in resolved.parents or resolved in g.parents:
            return True
    return False


def find_artifacts(tasks: tuple[str, ...] = ("lgd", "pd")) -> list[Artifact]:
    """Everything a previous run left behind, measured but not touched."""
    candidates: list[tuple[str, Path]] = [
        ("logs", logs_dir()),
        ("manifests", manifests_dir()),
        ("checkpoints", checkpoints_dir()),
    ]
    for task in tasks:
        candidates.append(("prior_pools", prior_cache_dir(f"{task}__original")))
        candidates.append(("prior_pools", prior_cache_dir(f"{task}__credit_v1")))
        for pipeline in ("data", "prior", "training", "eval"):
            candidates.append(("results", results_dir(task, pipeline)))

    # Per-run output directories (metrics.jsonl, resolved config) sit directly under the
    # output root alongside logs/ and manifests/, so pick them up separately.
    out = outputs_dir()
    if out.is_dir():
        for child in sorted(out.iterdir()):
            if child.is_dir() and child.name not in ("logs", "manifests"):
                candidates.append(("metrics", child))

    found = []
    for category, path in candidates:
        if _is_protected(path):
            continue
        n, total = _measure(path)
        if n:
            found.append(Artifact(category=category, path=path, n_files=n, bytes=total))
    return found


def summarise(artifacts: list[Artifact]) -> str:
    """A human-readable listing, grouped by category."""
    if not artifacts:
        return "Nothing found — the tree is already clean."
    lines = ["Artifacts from previous runs:", ""]
    by_cat: dict[str, list[Artifact]] = {}
    for a in artifacts:
        by_cat.setdefault(a.category, []).append(a)

    for category in CATEGORIES:
        items = by_cat.get(category)
        if not items:
            continue
        cat_bytes = sum(i.bytes for i in items)
        tag = "  [EXPENSIVE to regenerate]" if category in EXPENSIVE else ""
        lines.append(f"{category.upper()}  ({cat_bytes / 1e6:.1f} MB total){tag}")
        for i in items:
            lines.append(i.describe())
        lines.append("")

    total = sum(a.bytes for a in artifacts)
    lines.append(f"TOTAL: {total / 1e9:.2f} GB across {len(artifacts)} locations")
    lines.append("")
    lines.append("NEVER removed by this tool: data/raw (the datasets) and checkpoints/ in")
    lines.append("the repo (the downloaded TabPFN/TabICL weights).")
    return "\n".join(lines)


def clean(
    categories: tuple[str, ...] = ("logs", "manifests", "metrics", "results"),
    *,
    dry_run: bool = True,
    tasks: tuple[str, ...] = ("lgd", "pd"),
) -> dict[str, Any]:
    """Remove the named categories. **Dry run by default.**

    Defaults to the cheap categories only. `checkpoints` and `prior_pools` must be named
    explicitly, because they are the ones that cost days of compute to rebuild and the
    common intent behind "clean up" is the logs, not the results of a week of training.
    """
    artifacts = [a for a in find_artifacts(tasks) if a.category in categories]
    removed, failed = [], []

    for a in artifacts:
        if _is_protected(a.path):
            continue  # belt and braces; find_artifacts already filtered these
        if dry_run:
            removed.append(str(a.path))
            continue
        try:
            if a.path.is_dir():
                # Remove the CONTENTS, not the directory, so the tracked `.gitkeep`
                # markers and the directory layout survive. `rmtree` on results/lgd/eval
                # would delete a directory git expects to exist.
                for child in a.path.iterdir():
                    if child.name in KEEP_FILES:
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                a.path.unlink()
            removed.append(str(a.path))
        except OSError as exc:
            failed.append(f"{a.path}: {exc}")

    return {
        "dry_run": dry_run,
        "categories": list(categories),
        "removed": removed,
        "failed": failed,
        "freed_bytes": sum(a.bytes for a in artifacts),
        "freed_gb": round(sum(a.bytes for a in artifacts) / 1e9, 3),
    }
