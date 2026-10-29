"""Preview retired CreditICL scratch; preserve every unpublished checkpoint.

python -m src.utils.scratch_cleanup [--clean]
Only configured scratch/CreditICL/jobs directories are considered. No persistent
output, dataset, released weight or other project's directory is deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path

from src.utils import clean_run, paths


def retired_jobs(ids: set[str]) -> set[str]:
    """Accounting must positively identify this project's terminated jobs."""
    found = set()
    if not ids:
        return found
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    for cluster in ("mindwell", "wice"):
        result = subprocess.run(
            ["sacct", f"--clusters={cluster}", "-X", "--noheader", "--parsable2",
             "--jobs=" + ",".join(sorted(ids)), "--format=JobIDRaw,JobName%128,State%32"],
            capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f"Cannot read {cluster} accounting; no scratch deleted")
        for line in result.stdout.splitlines():
            fields = line.split("|")
            if (len(fields) >= 3 and fields[0] in ids
                    and re.fullmatch(r"(?:crediticl|cicl)(?:[-_].*)?", fields[1])
                    and fields[2].split() and fields[2].split()[0] in terminal):
                found.add(fields[0])
    return found


def digest(path: Path) -> bytes:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def clean_published(*, clean=False) -> tuple[int, int]:
    clean_run._require_idle_cluster()
    roots = {Path(os.environ[k]).resolve() for k in ("VSC_SCRATCH_GPFS1", "VSC_SCRATCH_LUSTRE1")
             if os.environ.get(k)}
    if not roots:
        raise RuntimeError("Set the explicit GPFS/Lustre scratch variables; no path guessed")
    persistent = paths.big_outputs_dir().resolve()
    files, directories, anchors, proofs = [], [], set(), []
    for root in sorted(roots):
        base = root / paths.PROJECT_NAME / "jobs"
        if not base.exists():
            continue
        clean_run._safe_tree(base)  # Validate all paths before reading a marker.
        jobs = [p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"[0-9_]+", p.name)]
        retired = retired_jobs({p.name for p in jobs})
        for job in jobs:
            if job.name not in retired:
                print(f"Retained without terminal ownership evidence: {job}")
                continue
            candidates = []
            for child in job.iterdir():
                if child.name == "inputs" or child.name.startswith("tabicl-offload-"):
                    candidates.append(child)
                elif child.name == "checkpoints" and child.is_dir():
                    for folder in child.iterdir():
                        if not folder.is_dir():
                            continue
                        marker = folder / "destination.txt"
                        if not marker.is_file():
                            continue
                        destination = Path(marker.read_text(encoding="utf-8").strip()).resolve()
                        if not destination.is_relative_to(persistent):
                            print(f"Retained unknown checkpoint destination: {folder}")
                            continue
                        copies = list(folder.glob("step-*.ckpt"))
                        verified = []
                        for copy in copies:
                            published = destination / copy.name
                            if not published.is_file() or clean_run._linked(published):
                                break
                            checksum = digest(copy)
                            if checksum != digest(published):
                                break
                            verified.append((published, checksum))
                        else:
                            # Preserve partial/corrupt writers for diagnosis, too.
                            if all(p.name == "destination.txt" or p in copies for p in folder.iterdir()):
                                candidates.append(folder)
                                proofs.extend(verified)
                                continue
                        print(f"Retained checkpoint without matching persistent copy: {folder}")
            for candidate in candidates:
                tree = [candidate, *clean_run._safe_tree(candidate)] if candidate.is_dir() else [candidate]
                for item in tree:
                    if item.is_dir():
                        directories.append(item)
                    else:
                        info = item.stat()
                        files.append((item, root, (info.st_size, info.st_mtime_ns, info.st_ino)))
                anchors.add(root)
                print(f"Disposable scratch: {candidate}")
    total = sum(info[0] for _, _, info in files)
    print(f"Verified scratch cleanup: {len(files)} files, {total / 1e9:.3f} GB")
    if clean:
        # Recheck publication proofs before any deletion, in addition to the
        # common queue, containment and mutation checks on the full plan.
        if any(not p.is_file() or digest(p) != h for p, h in proofs):
            raise RuntimeError("Persistent copy changed during planning; nothing deleted")
        clean_run._apply_reset_plan(files, directories, anchors)
    return len(files), total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)
    try:
        clean_published(clean=args.clean)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Scratch cleanup refused: {exc}")
        return 1
    print("Cleanup completed." if args.clean else "Nothing deleted; add --clean to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
