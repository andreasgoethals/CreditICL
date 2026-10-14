"""Wipe what the previous run produced, so the next one starts clean.

    python -m src.utils.clean_run                          list what is there, delete nothing
    python -m src.utils.clean_run --stale                  list failed logs and generated figures
    python -m src.utils.clean_run --stale --clean          delete ONLY those rebuildable outputs
    python -m src.utils.clean_run --clean                   delete it
    python -m src.utils.clean_run --clean --checkpoints     ...and OUR trained checkpoints
    python -m src.utils.clean_run --clean --prior-cache     ...and the synthetic prior pools
    python -m src.utils.clean_run --clean --processed       ...and the data/processed cache
    python -m src.utils.clean_run --experiment 0 --clean --checkpoints   only experiment_0
    python -m src.utils.clean_run --reset-output --clean    ALL generated output on both tiers

Clears the `output CreditICL/` tree (`src/utils/paths.py`) on BOTH storage tiers: the small
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
import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from src.utils import paths
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


@dataclass(frozen=True)
class StaleFile:
    path: Path
    root: Path
    reason: str
    size: int
    digest: str


def _linked(path: Path) -> bool:
    # lstat also detects Windows junctions on Python 3.11 (before Path.is_junction).
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _safe_stale_path(path: Path, root: Path) -> bool:
    """Do not follow output links into another project or traverse checkpoint trees."""
    if root.name != paths.OUTPUT_NAME or _linked(root):
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if not relative.parts or {"checkpoints", "prior_cache"}.intersection(relative.parts):
        return False
    if not path.resolve().is_relative_to(root.resolve()):
        return False
    return not any(_linked(p) for p in (path, *path.parents) if p == root or root in p.parents)


def stale_files() -> list[StaleFile]:
    """Conservative cleanup across both tiers; records and trained weights are never candidates.

    A log is failed only if its LAST explicit END marker says so. Running logs, successful
    retries, diagnostics reports, benchmark CSVs/receipts and preprocessing caches survive.
    Figures and their generated summaries are rebuildable, regardless of their timestamp.
    """
    found = []
    seen = set()
    for original in (paths.outputs_dir(), paths.big_outputs_dir()):
        if original.name != paths.OUTPUT_NAME or _linked(original):
            raise RuntimeError(f"Refusing unexpected or linked output root: {original}")
        root = original.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for directory, folders, filenames in os.walk(root, followlinks=False):
            parent = Path(directory)
            folders[:] = [name for name in sorted(folders)
                          if _safe_stale_path(parent / name, root)]
            for name in sorted(filenames):
                path = parent / name
                if not _safe_stale_path(path, root) or not path.is_file():
                    continue
                relative = path.relative_to(root)
                reason = None
                content = None
                if relative.parts[0] == "figures" and (path.suffix == ".pdf" or name == "_figures.json"):
                    reason = "generated figure/index"
                elif parent == root and name in {"All_Results.md", "CAPTIONS.md"}:
                    reason = "generated notebook summary"
                elif path.suffix == ".log" and "logs" in relative.parts[:-1]:
                    content = path.read_bytes()
                    endings = re.findall(rb"^END status=(\S+)", content, flags=re.MULTILINE)
                    if endings and endings[-1] == b"FAILED":
                        reason = "explicitly failed job log"
                if reason:
                    content = path.read_bytes() if content is None else content
                    found.append(StaleFile(path, root, reason, len(content),
                                           hashlib.sha256(content).hexdigest()))
    return sorted(found, key=lambda item: str(item.path))


def _require_idle_cluster() -> None:
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run cleanup on a login node after CreditICL jobs have stopped.")
    if paths.on_vsc():
        from src.utils.run_experiment import queued_job_names

        active = sorted(name for name in queued_job_names()
                        if name.lower().startswith(("crediticl", "cicl")))
        if active:
            raise RuntimeError("CreditICL jobs are still queued/running: " + ", ".join(active))


def delete_stale_files(plan: list[StaleFile]) -> int:
    """Recheck the entire plan before the first deletion; leave empty folders in place."""
    _require_idle_cluster()
    allowed_roots = {p.absolute() for p in (paths.outputs_dir(), paths.big_outputs_dir())}
    resolved_roots = {p.resolve() for p in allowed_roots if not _linked(p)}
    for item in plan:
        if item.root not in resolved_roots or not _safe_stale_path(item.path, item.root):
            raise RuntimeError(f"Output path changed or escaped its root: {item.path}")
        if not item.path.is_file() or hashlib.sha256(item.path.read_bytes()).hexdigest() != item.digest:
            raise RuntimeError(f"Output changed since listing; nothing deleted: {item.path}")
    for item in plan:
        item.path.unlink()
    return len(plan)


def reset_output(*, clean: bool = False) -> tuple[int, int]:
    """Reset BOTH output tiers, including historical output names and flat run checkpoints.

    Inputs live outside these roots. Refuse links/junctions anywhere in a root,
    verify the entire plan before deleting, and never infer sibling project paths.
    """
    roots_seen, files, directories = set(), [], []
    candidates = []
    current = (paths.outputs_dir(), paths.big_outputs_dir())
    if any(p.name != paths.OUTPUT_NAME or _linked(p) for p in current):
        raise RuntimeError(f"Unexpected or linked output roots: {current}")
    projects = {p.parent for p in current}
    for project in sorted(projects):
        for name in (paths.OUTPUT_NAME, "output", "outputCreditICL", "output_CreditICL"):
            candidates.append((project / name, False))
        # Before the output-tree migration, generated runs and released downloads
        # shared checkpoints/. Delete only the recognised RUN subdirectories,
        # never the directory itself or its top-level released model files.
        old = project / "checkpoints"
        if _linked(old):
            raise RuntimeError(f"Linked legacy checkpoint root: {old}")
        if old.is_dir():
            for child in sorted(old.iterdir()):
                if re.match(r"^(?:exp[0-3][a-z]*_|debug_)", child.name):
                    candidates.append((child, True))
    for original, legacy_run in candidates:
        if _linked(original):
            raise RuntimeError(f"Unexpected or linked output root: {original}")
        root = original.resolve()
        if root in roots_seen or not root.exists():
            continue
        project = original.parent.parent if legacy_run else original.parent
        if not root.is_relative_to(project.resolve()) or not root.is_dir():
            raise RuntimeError(f"Output root escaped its project: {original}")
        roots_seen.add(root)
        inputs = [paths.datasets_dir(), paths.ood_cache_dir()]
        if not legacy_run:
            inputs.append(paths.pretrained_dir())
        if os.environ.get("CREDITICL_TABPFN_DIR"):
            weights = Path(os.environ["CREDITICL_TABPFN_DIR"])
            if not legacy_run or weights.resolve() != original.parent.resolve():
                inputs.append(weights)
        if any(p.resolve().is_relative_to(root) or root.is_relative_to(p.resolve()) for p in inputs):
            raise RuntimeError(f"Input data or released weights overlap output root; move inputs before resetting: {root}")
        print(f"{'Legacy run' if legacy_run else 'Output'} reset root: {root}")
        for directory, folders, names in os.walk(root, followlinks=False):
            for name in folders + names:
                item = Path(directory) / name
                if _linked(item) or not item.resolve().is_relative_to(root):
                    raise RuntimeError(f"Linked or escaping output path: {item}")
            directories.extend(Path(directory) / n for n in folders)
            for name in names:
                if name in KEEP:
                    continue
                item = Path(directory) / name
                info = item.stat()
                files.append((item, root, (info.st_size, info.st_mtime_ns, info.st_ino)))
        if legacy_run:
            directories.append(root)
    total = sum(info[0] for _, _, info in files)
    print(f"Full reset: {len(files)} generated files, {total / 1e9:.3f} GB, including trained checkpoints.")
    if clean:
        _require_idle_cluster()
        for item, root, before in files:
            info = item.stat()
            if (any(_linked(p) for p in (item, *item.parents) if p == root or root in p.parents)
                    or not item.resolve().is_relative_to(root)
                    or (info.st_size, info.st_mtime_ns, info.st_ino) != before):
                raise RuntimeError(f"Output changed since planning; reset refused: {item}")
        for item, _, _ in files:
            item.unlink()
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            if _linked(directory) or not any(directory.resolve().is_relative_to(root) for root in roots_seen):
                raise RuntimeError(f"Output directory changed or escaped its root: {directory}")
            if not any(directory.iterdir()):
                try:
                    directory.rmdir()
                except PermissionError:
                    # Explorer can mark ordinary output directories read-only.
                    # On Windows that attribute prevents rmdir even when empty.
                    attributes = getattr(directory.stat(), "st_file_attributes", 0)
                    if os.name != "nt" or not attributes & stat.FILE_ATTRIBUTE_READONLY:
                        raise
                    directory.chmod(stat.S_IWRITE)
                    directory.rmdir()
    return len(files), total


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
        p for p in _safe_tree(root)
        if p.is_file() and p.name not in KEEP and not _is_protected(p, prot)
    ]
    return len(files), sum(p.stat().st_size for p in files)


def _safe_tree(root: Path) -> list[Path]:
    """Validate the whole tree before deletion; never traverse links or junctions."""
    for parent in (root, *root.parents):
        if _linked(parent):
            raise RuntimeError(f"Cleanup refuses a linked path: {parent}")
    base = root.resolve()
    found: list[Path] = []
    pending = [root]
    while pending:
        for child in pending.pop().iterdir():
            if _linked(child) or not child.resolve().is_relative_to(base):
                raise RuntimeError(f"Cleanup refuses a linked/escaping path: {child}")
            found.append(child)
            if child.is_dir():
                pending.append(child)
    return found


def wipe(root: Path, protected: list[Path] | None = None) -> int:
    """Delete everything under a root except the structure markers. Returns files removed.

    Two passes, and the order matters: files first, then empty directories bottom-up. That leaves
    exactly the directories holding a tracked marker and removes the per-run ones that do not.
    """
    if not root.is_dir():
        return 0
    prot = protected_paths() if protected is None else protected
    removed = 0
    tree = _safe_tree(root)
    for path in tree:
        if path.is_file() and path.name not in KEEP and not _is_protected(path, prot):
            path.unlink()
            removed += 1
    for path in sorted((p for p in tree if p.is_dir()),
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
    parser.add_argument("--reset-output", action="store_true",
                        help="fresh campaign: ALL generated output on both tiers, including trained checkpoints and validation attempts")
    parser.add_argument("--stale", action="store_true",
                        help="only failed job logs and generated figures/summaries; preserve all "
                             "checkpoints, successful/unfinished logs, results and caches")
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

    if args.reset_output:
        if args.stale or args.experiment is not None or args.processed or args.prior_cache or args.checkpoints:
            parser.error("--reset-output is a complete output scope; combine only with --clean")
        try:
            reset_output(clean=args.clean)
        except (OSError, RuntimeError) as exc:
            print(f"Reset refused: {exc}")
            return 1
        print("Output reset completed." if args.clean else "Nothing deleted; add --clean to reset.")
        print("Dataset inputs and released reference weights were preserved.")
        return 0

    if args.stale:
        if args.experiment is not None or args.processed or args.prior_cache or args.checkpoints:
            parser.error("--stale cannot be combined with other cleanup scopes")
        try:
            plan = stale_files()
            for item in plan:
                print(f"  {item.reason}: {item.path}")
            print(f"\nTOTAL: {len(plan)} files, {sum(p.size for p in plan) / 1e6:.2f} MB")
            print("Preserved: checkpoints, successful/unfinished logs, run records, diagnostic "
                  "reports, benchmark results/receipts, data and prior pools.")
            if args.clean:
                print(f"Removed {delete_stale_files(plan)} files.")
            else:
                print("Nothing was deleted. Re-run with --stale --clean to delete these files.")
            return 0
        except (OSError, RuntimeError) as exc:
            print(f"Cleanup refused: {exc}")
            return 1

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
    _require_idle_cluster()
    # Validate every root before deleting the first file.
    for root in targets:
        if root.is_dir():
            _safe_tree(root)
    for root in targets:
        print(f"  removed {wipe(root, prot):>6} files from {root}")
    print("\nClean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
