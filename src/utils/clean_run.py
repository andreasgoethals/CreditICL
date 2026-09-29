"""Wipe what the previous run produced, so the next one starts clean.

    python -m src.utils.clean_run                          list what is there, delete nothing
    python -m src.utils.clean_run --clean                   delete it
    python -m src.utils.clean_run --clean --checkpoints     ...and OUR trained checkpoints
    python -m src.utils.clean_run --clean --prior-cache     ...and the synthetic prior pools
    python -m src.utils.clean_run --clean --processed       ...and the data/processed cache
    python -m src.utils.clean_run --experiment 0 --clean --checkpoints   only experiment_0

Clears the `output_CreditICL/` tree (`src/utils/paths.py`) on BOTH storage tiers: the small
half on `$VSC_DATA` (logs, run records, benchmark scores, figures) and, when asked, the big
half on project storage (checkpoints, prior pools). Off-cluster both tiers are one folder.
`--experiment N` limits all of it to `experiment_N/`.

LISTS BY DEFAULT. The two mistakes are not symmetric: a listing you meant as a deletion costs one
more command, and a deletion you meant as a listing costs the run.

`--checkpoints` is opt-in, and **without it a rerun resumes from the old checkpoint and trains
nothing** — it exits 0 in two seconds having scored the old weights, which is what happened to
all four arms on 17-08-2026. Opt-in, because a checkpoint is also the only way to debug the model
that produced it.

NEVER TOUCHES, whatever the flags: `data/raw/`, `tfm-library/`, the RELEASED weights in
`paths.pretrained_dir()` (a HuggingFace download, what Exp2 warm-starts from), and the
out-of-domain cache `data/ood/` — **compute nodes have no outbound internet**, so it can only be
rebuilt from a login node with `python -m src.utils.fetch_ood`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import (
    EXPERIMENTS,
    experiment_dir,
    ood_cache_dir,
    outputs_dir,
    pretrained_dir,
    prior_cache_root,
    processed_dir,
)

#: Tracked so an empty directory survives a clone, or the tree's index. Not run output, so never
#: counted or deleted.
KEEP = frozenset({".gitkeep", ".gitignore", "README.md"})


def checkpoint_roots(experiment: int | None = None) -> list[Path]:
    """`experiment_<N>/checkpoints/` on the big tier, for one experiment or all of them."""
    exps = EXPERIMENTS if experiment is None else (int(experiment),)
    return [experiment_dir(n, big=True) / "checkpoints" for n in exps]


def protected_paths(*, checkpoints: bool = False, prior_cache: bool = False) -> list[Path]:
    """What a wipe must step around. The last two only until their flag asks for them.

    Locally the big tier IS the small one, so the checkpoints and pools sit inside the tree a
    plain `--clean` walks; protecting them by path is what keeps them opt-in there too.
    """
    protected = [ood_cache_dir(), pretrained_dir()]
    if not checkpoints:
        protected.extend(checkpoint_roots())
    if not prior_cache:
        protected.append(prior_cache_root())
    return protected


def run_checkpoint_dirs(experiment: int | None = None) -> list[Path]:
    """OUR trained checkpoints: every run folder holding a `.ckpt` under `experiment_<N>/checkpoints/`.

    By structure, not by name: a run name the code did not anticipate is exactly the one a
    rerun would silently resume from.
    """
    return sorted(
        d for root in checkpoint_roots(experiment) if root.is_dir()
        for d in root.iterdir() if d.is_dir() and any(d.rglob("*.ckpt"))
    )


def roots(*, experiment: int | None = None, processed: bool = False, prior_cache: bool = False,
          checkpoints: bool = False) -> list[Path]:
    """Every tree to clear: the small output tree, plus what the flags add from the big one.

    A root nested inside one already listed is dropped (locally every tier collapses into the
    repo, and listing a tree twice double-counts the report); `protected_paths` decides what
    inside it survives.
    """
    found = [outputs_dir() if experiment is None else experiment_dir(experiment)]
    extras: list[Path] = []
    if checkpoints:
        extras.extend(checkpoint_roots(experiment))
    if prior_cache and experiment is None:
        extras.append(prior_cache_root())
    if processed:
        extras.append(processed_dir())
    for extra in extras:
        if not any(extra == f or extra.is_relative_to(f) or f.is_relative_to(extra) for f in found):
            found.append(extra)
    return found


def _is_protected(path: Path, protected: list[Path]) -> bool:
    return any(path == p or p in path.parents for p in protected)


def measure(root: Path, protected: list[Path] | None = None) -> tuple[int, int]:
    """(files, bytes) under a root, ignoring the structure markers and protected trees."""
    if not root.is_dir():
        return 0, 0
    prot = protected_paths() if protected is None else protected
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.name not in KEEP and not _is_protected(p, prot)
    ]
    return len(files), sum(p.stat().st_size for p in files)


def wipe(root: Path, protected: list[Path] | None = None) -> int:
    """Delete everything under a root except the structure markers. Returns files removed.

    Two passes, and the order matters: files first, then empty directories bottom-up. That leaves
    exactly the directories holding a tracked marker and removes the per-run ones that do not.
    """
    if not root.is_dir():
        return 0
    prot = protected_paths() if protected is None else protected
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
    parser.add_argument("--experiment", type=int, choices=EXPERIMENTS, default=None,
                        help="only experiment_<N>/ (both tiers); default the whole tree")
    parser.add_argument("--processed", action="store_true",
                        help="also clear data/processed/, the preprocessing cache")
    parser.add_argument("--prior-cache", action="store_true",
                        help="also clear the pre-generated synthetic prior pools (GPU-hours each)")
    parser.add_argument("--checkpoints", action="store_true",
                        help="also clear OUR trained checkpoints. Without this a rerun RESUMES "
                             "from them and trains nothing. Never touches the released weights.")
    args = parser.parse_args(argv)

    targets = roots(experiment=args.experiment, processed=args.processed,
                    prior_cache=args.prior_cache, checkpoints=args.checkpoints)
    prot = protected_paths(checkpoints=args.checkpoints, prior_cache=args.prior_cache)
    total_files = total_bytes = 0
    print("Output from the previous run:\n")
    for root in targets:
        files, size = measure(root, prot)
        total_files += files
        total_bytes += size
        state = f"{files:>6} files  {size / 1e6:>9.1f} MB" if files else "         empty"
        print(f"  {state}  {root}")

    print(f"\nTOTAL: {total_files} files, {total_bytes / 1e9:.2f} GB")
    kept = ["data/raw/", "data/ood/", "the released weights", "tfm-library/"]
    if not args.checkpoints:
        kept.append("our checkpoints (--checkpoints)")
    if not args.prior_cache:
        kept.append("prior pools (--prior-cache)")
    print("Never touched: " + ", ".join(kept) + ".")

    if not args.clean:
        if total_files:
            print("\nNothing was deleted. Re-run with --clean to delete.")
        return 0

    print("\nDeleting:")
    for root in targets:
        print(f"  removed {wipe(root, prot):>6} files from {root}")
    print("\nClean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
