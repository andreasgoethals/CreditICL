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

NEVER TOUCHES `data/raw/` or `tfm-library/`, nor the RELEASED `*.ckpt` weights at the top of
`checkpoints/` — those are a HuggingFace download and what Exp2 warm-starts from.

`--checkpoints` clears OUR OWN `exp*/` run directories under `checkpoints/`. **Without it a
rerun resumes from the last one and trains nothing** — it exits 0 in two seconds having scored
the old weights, which is what happened to all four arms on 17-08-2026. Opt-in, because a
checkpoint is also the only way to debug the model that produced it.

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


def run_checkpoint_dirs() -> list[Path]:
    """OUR trained checkpoints — the `exp*` run directories under `checkpoints/`.

    Separated from the released TabICLv2 `*.ckpt` files, which sit at the top level of the same
    directory and must never go: they are a HuggingFace download and what Exp2 warm-starts from.

    This exists because of a chain of two fixes. Our checkpoints used to fall back to
    `$VSC_DATA/output/<run>/checkpoints` (staging was mode 0500), where a normal clean removed
    them. Fixing the permission sent them to `checkpoints/` for real — which `clean_run`
    protects — so on 17-08-2026 all four arms found a step-1500 checkpoint from the previous
    run, resumed at `max_steps`, trained nothing, and reported success.
    """
    from src.utils.paths import checkpoints_dir

    base = Path(checkpoints_dir())
    if not base.is_dir():
        return []
    # BY STRUCTURE, NOT BY NAME. This used to match the prefixes `exp1_`, `exp2_`, `exp3_`,
    # which silently skipped everything else the project has since started writing —
    # `debug_exp1_*`, pilot and benchmark runs, and any experiment past Exp3. A missed run
    # directory is not a tidy-up problem: a rerun RESUMES from whatever checkpoint it finds
    # and trains nothing, which is exactly what happened to all four arms on 17-08-2026.
    #
    # So: any SUBDIRECTORY holding a checkpoint is one of ours. The released TabICLv2 weights
    # are safe because they sit at the TOP level of `checkpoints/`, never in a subdirectory.
    return sorted(
        d for d in base.iterdir()
        if d.is_dir() and any(d.rglob("*.ckpt"))
    )


def scratch_outputs_dir() -> Path | None:
    """`$VSC_SCRATCH/CreditICL`, when it is a real third place.

    Scratch is purged by the system eventually, but "eventually" is not "before your next run",
    and a stale file there is as confusing as a stale file anywhere else. Returns None off the
    cluster, where scratch collapses into the repository and `outputs_dir()` already covers it.
    """
    from src.utils.paths import PROJECT_NAME, REPO_ROOT, scratch_root

    root = scratch_root()
    if root == REPO_ROOT:
        return None
    return root / PROJECT_NAME


def roots(*, processed: bool = False, prior_cache: bool = False,
          checkpoints: bool = False) -> list[Path]:
    """Every tree to clear. Two `output/` roots on the cluster, one locally, plus the caches.

    `results_dir()` is listed separately because on the cluster it is the one part of `output/`
    on project storage — clearing only `outputs_dir()` there would leave the largest files behind.
    """
    found = [outputs_dir()]
    for extra in (results_dir(), scratch_outputs_dir()):
        # `is_relative_to` in BOTH directions: on a laptop every root collapses into the repo,
        # and listing the same tree twice double-counts the deletion report.
        if extra is not None and not any(
            extra == f or extra.is_relative_to(f) or f.is_relative_to(extra) for f in found
        ):
            found.append(extra)
    if processed:
        found.append(processed_dir())
    if prior_cache:
        found.append(prior_cache_root())
    if checkpoints:
        found.extend(run_checkpoint_dirs())
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
    parser.add_argument("--checkpoints", action="store_true",
                        help="also clear OUR trained exp*/ checkpoints. Without this a rerun "
                             "RESUMES from them and trains nothing. Never touches the released "
                             "TabICLv2 weights.")
    args = parser.parse_args(argv)

    targets = roots(processed=args.processed, prior_cache=args.prior_cache,
                    checkpoints=args.checkpoints)
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
