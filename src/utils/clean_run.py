"""Wipe what the previous run produced, so the next one starts clean.

    python -m src.utils.clean_run                        list what is there, delete nothing
    python -m src.utils.clean_run --clean                 delete it
    python -m src.utils.clean_run --clean --processed      ...and the data/processed cache too
    python -m src.utils.clean_run --clean --prior-cache    ...and the synthetic prior pools

PROJECT ADDITION: `--prior-cache`. This project's largest artefact is the pre-generated pools
of synthetic datasets, which live outside `output/` (they are far too big for it), so the
template's two roots would leave them behind. Opt-in and listed last, because a pool costs
GPU-hours to regenerate — more than everything else here put together.

Clears the **whole `output/` tree on both storage tiers** — `$VSC_DATA` and project storage — so
one invocation is enough whether you are on a laptop or on the cluster. Off-cluster both tiers
collapse into the repository and it is simply `output/`.

`--processed` additionally clears `data/processed/`, the preprocessing cache. It is separate
because rebuilding that cache can cost far more than re-running the notebooks, so "clean the last
run" should not silently throw it away.

LISTS BY DEFAULT. The two mistakes are not symmetric: a listing you meant as a deletion costs one
more command, and a deletion you meant as a listing costs the run.

NEVER TOUCHES `data/raw/` or `checkpoints/` or `tfm-library/` — the inputs are irreplaceable and
the weights are either downloaded or a training run to reproduce.

NOR `prior_cache/ood/`, even under `--prior-cache`. It is the out-of-domain evaluation cache,
not a prior pool, and **compute nodes have no outbound internet** — so it can only be rebuilt
from a login node with `python -m src.utils.fetch_ood`. See `protected_paths`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import outputs_dir, prior_cache_root, processed_dir, results_dir

#: Tracked so an empty directory survives a clone. Not run output, so never counted or deleted —
#: removing them would leave a fresh clone with nowhere to write.
KEEP = frozenset({".gitkeep", ".gitignore"})


def protected_paths() -> list[Path]:
    """Directories `--prior-cache` must step around.

    `prior_cache/ood/` is the out-of-domain evaluation cache — 50 downloaded datasets — and it
    sits under the prior-cache root only because that is where big things live. It is not a
    prior pool and clearing it is not part of "clean the last run".

    It also cannot be rebuilt where the deletion usually happens: **compute nodes have no
    outbound internet**, so `python -m src.utils.fetch_ood` only works from a login node. A
    sweep that wiped it would find every out-of-domain column empty and would not say why.
    """
    return [prior_cache_root() / "ood"]


def roots(*, processed: bool = False, prior_cache: bool = False) -> list[Path]:
    """Every tree to clear. Two `output/` roots on the cluster, one locally, plus the caches.

    `results_dir()` is listed separately because on the cluster it is the one part of `output/`
    on project storage — clearing only `outputs_dir()` there would leave the largest files behind.
    """
    found = [outputs_dir()]
    results = results_dir()
    if not results.is_relative_to(found[0]):
        found.append(results)
    if processed:
        found.append(processed_dir())
    if prior_cache:
        found.append(prior_cache_root())
    return found


def _is_protected(path: Path, protected: list[Path]) -> bool:
    return any(path == p or p in path.parents for p in protected)


def measure(root: Path) -> tuple[int, int]:
    """(files, bytes) under a root, ignoring the structure markers and protected trees."""
    if not root.is_dir():
        return 0, 0
    prot = protected_paths()
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.name not in KEEP and not _is_protected(p, prot)
    ]
    return len(files), sum(p.stat().st_size for p in files)


def wipe(root: Path) -> int:
    """Delete everything under a root except the structure markers. Returns files removed.

    Two passes, and the order matters: files first, then empty directories bottom-up. That leaves
    exactly the directories holding a tracked `.gitkeep` and removes the per-run ones
    (`figures/<notebook>/`) that do not. An `rmtree` of the subtree would take
    `output/figures/.gitkeep` with it, and the next clone would have nowhere to write.
    """
    if not root.is_dir():
        return 0
    prot = protected_paths()
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.name not in KEEP and not _is_protected(path, prot):
            path.unlink()
            removed += 1
    for path in sorted((p for p in root.rglob("*") if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        if _is_protected(path, prot):
            continue
        if not any(path.iterdir()):
            path.rmdir()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean", action="store_true", help="actually delete; default lists only")
    parser.add_argument("--processed", action="store_true",
                        help="also clear data/processed/, the preprocessing cache")
    parser.add_argument("--prior-cache", action="store_true",
                        help="also clear the pre-generated synthetic prior pools (GPU-hours each)")
    args = parser.parse_args(argv)

    targets = roots(processed=args.processed, prior_cache=args.prior_cache)
    total_files = total_bytes = 0
    print("Output from the previous run:\n")
    for root in targets:
        files, size = measure(root)
        total_files += files
        total_bytes += size
        state = f"{files:>6} files  {size / 1e6:>9.1f} MB" if files else "         empty"
        print(f"  {state}  {root}")

    print(f"\nTOTAL: {total_files} files, {total_bytes / 1e9:.2f} GB")
    print("Never touched: data/raw/, checkpoints/, tfm-library/.")

    if not args.clean:
        if total_files:
            print("\nNothing was deleted. Re-run with --clean to delete.")
        return 0

    print("\nDeleting:")
    for root in targets:
        print(f"  removed {wipe(root):>6} files from {root}")
    print("\nClean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
