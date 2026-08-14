"""Is every directory this project writes to actually writable? Run it BEFORE submitting.

    python scripts/check_storage.py            # report
    python scripts/check_storage.py --fix      # also try to repair what it safely can

WHY. On 14-08-2026 every arm of two debug arrays printed

    WARNING: cannot write to /lustre1/project/stg_00211/CreditICL/checkpoints/<run>
             ([Errno 13] Permission denied). Falling back to $VSC_DATA.

and carried on. The fallback is deliberate — a finished run on the wrong disk beats a crash at
step 12,000 — but `$VSC_DATA` has a 75 GiB quota and Exp1 is 96 arms, so the fallback that
saves one debug run is what fills the disk during the real one. Notably `data/processed/` and
`output/results/` on the SAME staging root wrote fine, which is the signal: the project
directory is fine and one subdirectory is not.

This checks each directory the way the job does — by actually creating a file — and prints the
exact command to fix whatever is broken. Permissions are not something to guess at from a
laptop, so it reports owner, group and mode for every level rather than inferring.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _describe(path: Path) -> str:
    """Owner, group and mode — the three things that decide whether a write succeeds."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return "does not exist"
    except PermissionError:
        return "cannot stat (no search permission on a parent)"
    mode = stat.filemode(st.st_mode)
    try:
        import grp
        import pwd

        owner = pwd.getpwuid(st.st_uid).pw_name
        group = grp.getgrgid(st.st_gid).gr_name
    except (ImportError, KeyError):  # Windows, or an id with no passwd entry
        owner, group = str(st.st_uid), str(st.st_gid)
    setgid = " SETGID" if st.st_mode & stat.S_ISGID else ""
    return f"{mode} {owner}:{group}{setgid}"


def _can_write(path: Path) -> tuple[bool, str]:
    """Actually write a file. `os.access` lies on Lustre and on group-mapped mounts."""
    if not path.is_dir():
        return False, "not a directory"
    probe = path / f".crediticl_write_probe_{os.getpid()}"
    try:
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
        return True, "writable"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc.strerror or exc}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fix", action="store_true",
                    help="create missing directories and try `chmod u+rwx,g+rwxs` on ours")
    args = ap.parse_args()

    from src.utils.paths import (
        checkpoints_dir,
        logs_dir,
        manifests_dir,
        on_vsc,
        prior_cache_dir,
        processed_dir,
        results_dir,
        staging_root,
    )

    print("=" * 78)
    print(" CreditICL storage check")
    print(f" on VSC: {on_vsc()}")
    print("=" * 78)

    # THE ORDER IS THE DIAGNOSIS. `checkpoints` was the only one that failed on 14-08-2026
    # while `processed` and `results` on the same staging root wrote fine — which is exactly
    # how you know the project directory is healthy and one subdirectory is not.
    targets: list[tuple[str, Path]] = [
        ("staging root", staging_root()),
        ("checkpoints  <- the one that failed", Path(checkpoints_dir())),
        ("processed data", Path(processed_dir())),
        ("results", Path(results_dir("pd", "eval"))),
        # `.parent`: `prior_cache_dir(name)` returns a per-cache SUBdirectory, so passing a
        # made-up name would test a path no run ever uses — and `--fix` would then create it,
        # leaving a stray directory on project storage. Test the cache root itself.
        ("prior cache", Path(prior_cache_dir("probe")).parent),
        ("logs", Path(logs_dir())),
        ("manifests", Path(manifests_dir())),
    ]

    broken: list[Path] = []
    for label, path in targets:
        print(f"\n{label}")
        print(f"  path    {path}")
        # Walk DOWN from the root: a denial three levels up is the actual cause, and
        # reporting only the leaf sends you chmod-ing the wrong directory.
        # The last few levels only: `/` and `/lustre1` are never the problem, and a wall of
        # ancestors buries the line that matters.
        parts = ([*list(path.parents)[::-1], path])[-5:]
        for parent in parts:
            marker = "->" if parent == path else "  "
            print(f"  {marker} {str(parent):<58} {_describe(parent)}")

        if not path.exists():
            # A directory that does not exist yet is NOT a problem — the pipeline creates it
            # on first use. What matters is whether the nearest existing ancestor will let it.
            # Reporting "missing" as "cannot be written" would send you chmod-ing a path that
            # was always going to work.
            ancestor = next((p for p in path.parents if p.exists()), path.anchor and Path("/"))
            ok_parent, why_parent = _can_write(Path(ancestor))
            if ok_parent:
                print(f"  MISSING — fine, {ancestor} is writable and it will be created")
                continue
            print(f"  MISSING, and its parent {ancestor} refuses: {why_parent}")
            broken.append(Path(ancestor))
            continue

        ok, why = _can_write(path)
        print(f"  WRITE   {'OK' if ok else 'FAILED — ' + why}")
        if not ok:
            broken.append(path)
            if args.fix:
                try:
                    # setgid so anything created here keeps the project group, which is what
                    # lets a second person in the group write to it later.
                    path.chmod(path.stat().st_mode | 0o2770)
                    ok2, why2 = _can_write(path)
                    print(f"  after chmod: {'OK' if ok2 else 'still failing — ' + why2}")
                    if ok2:
                        broken.remove(path)
                except OSError as exc:
                    print(f"  chmod refused ({exc}) — you are probably not the owner")

    print("\n" + "=" * 78)
    if not broken:
        print("  Everything this project writes to is writable. Submit away.")
        return 0

    print(f"  {len(broken)} directory/ies cannot be written:")
    for p in broken:
        print(f"    {p}")
    print()
    print("  WHAT TO DO, in order:")
    print()
    print("  1. If you own it, take it back:")
    for p in broken:
        print(f"       chmod u+rwx,g+rwxs {p}")
    print()
    print("  2. If it exists but someone else owns it, it cannot be repaired by you.")
    print("     Move it aside and make your own (nothing is deleted):")
    for p in broken:
        print(f"       mv {p} {p}.foreign && mkdir -p {p}")
    print()
    print("  3. If the PARENT is the one refusing, that is the level to fix — the walk")
    print("     above shows owner and mode for every level, so read upward from the leaf.")
    print()
    print("  4. New files should inherit the project group. Put this in ~/.bashrc:")
    print("       umask 002")
    print()
    print("  Then re-run this script. Do not submit a 96-arm sweep until it is clean:")
    print("  the fallback to $VSC_DATA works, and $VSC_DATA is 75 GiB.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
