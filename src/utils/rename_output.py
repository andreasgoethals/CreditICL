"""Rename the two output roots without deleting previous results.

    python -m src.utils.rename_output          # show the plan
    python -m src.utils.rename_output --apply  # apply it after jobs have stopped

An updated checkout may already contain tracked files in the destination. Merge
non-conflicting paths; preserve conflicting old files under legacy-output/output/.
The internal layout and recorded configs of historical runs stay unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import paths


def plan_moves(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    """Plan moves on one filesystem, refusing links and occupied archive paths."""
    if source.is_symlink():
        raise ValueError(f"Expected a real output directory: {source}")
    if not source.exists():
        return []
    if not source.is_dir():
        raise ValueError(f"Expected a real output directory: {source}")
    if source.parent.resolve() != destination.parent.resolve():
        raise ValueError("Output rename must stay in the same project directory")
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError(f"Expected a real output directory: {destination}")
    # Do not follow links into checkpoints, datasets, or another storage tier.
    entries = sorted(source.rglob("*"))
    if any(p.is_symlink() for p in entries):
        raise ValueError(f"Symbolic link inside {source}; inspect before renaming")
    if not destination.exists():
        return [(source, destination)]
    moves = []
    planned_targets: set[Path] = set()
    for item in entries:
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        target = destination / relative
        parents = [p for p in target.parents if p != destination and destination in p.parents]
        if target.exists() or target.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in parents):
            target = destination / "legacy-output" / source.name / relative
        parents = [p for p in target.parents if p != destination and destination in p.parents]
        if target in planned_targets or target.exists() or target.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in parents):
            raise FileExistsError(f"Cannot preserve both files at {target}; nothing moved")
        moves.append((item, target))
        planned_targets.add(target)
    return moves


def apply_moves(moves: list[tuple[Path, Path]], source: Path) -> None:
    for old, new in moves:
        if new.exists() or new.is_symlink():
            raise FileExistsError(f"Destination appeared since planning: {new}")
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
    # Only empty directories are removed, never files or nonempty folders.
    if source.exists():
        for folder in sorted((p for p in source.rglob("*") if p.is_dir()),
                             key=lambda p: len(p.parts), reverse=True):
            folder.rmdir()
        source.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        # Resolve both tiers before changing either. They coincide on a laptop.
        destinations = list(dict.fromkeys(p.absolute() for p in
                                         (paths.outputs_dir(), paths.big_outputs_dir())))
        plans = []
        for destination in destinations:
            source = destination.parent / "output"
            moves = plan_moves(source, destination)
            plans.append((source, destination, moves))
            print(f"{source} -> {destination}: {len(moves)} move(s)")
        if args.apply and paths.on_vsc():
            from src.utils.run_experiment import queued_job_names

            active = sorted(n for n in queued_job_names()
                            if n.lower().startswith(("crediticl", "cicl")))
            if active:
                raise RuntimeError("Wait for queued/running CreditICL jobs before renaming: "
                                   + ", ".join(active))
        if args.apply:
            for source, _, moves in plans:
                apply_moves(moves, source)
            print("Output roots renamed. Checkpoints outside those roots were not moved.")
        else:
            print("Preview only. Add --apply once CreditICL jobs have stopped.")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot rename output: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
