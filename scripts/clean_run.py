"""List — and optionally remove — everything a previous run produced.

    python scripts/clean_run.py                    # LIST only, removes nothing
    python scripts/clean_run.py --clean            # logs, manifests, metrics, results
    python scripts/clean_run.py --clean --all      # ALSO checkpoints and prior pools
    python scripts/clean_run.py --clean --only checkpoints

Listing is the default and removal always needs `--clean`, because the expensive
categories here represent days of compute.

`data/raw` (the datasets) and the repo's `checkpoints/` (downloaded TabPFN/TabICL
weights) are never removed, whatever you pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_artifacts import CATEGORIES, clean, find_artifacts, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help="actually delete (default: list only)")
    ap.add_argument("--all", action="store_true", help="include checkpoints and prior pools")
    ap.add_argument("--only", default=None, help=f"comma-separated subset of {list(CATEGORIES)}")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    from src.utils.paths import describe

    print("storage:")
    for k, v in describe().items():
        print(f"  {k:16} {v}")
    print()
    print(summarise(find_artifacts()))

    if not args.clean:
        print("\nListing only. Add --clean to remove.")
        return 0

    if args.only:
        cats = tuple(c.strip() for c in args.only.split(",") if c.strip())
        unknown = [c for c in cats if c not in CATEGORIES]
        if unknown:
            print(f"\nunknown categories: {unknown}; known: {list(CATEGORIES)}", file=sys.stderr)
            return 2
    elif args.all:
        cats = CATEGORIES
    else:
        cats = ("logs", "manifests", "metrics", "results")

    preview = clean(cats, dry_run=True)
    if not preview["removed"]:
        print(f"\nNothing to remove in {list(cats)}.")
        return 0

    print(f"\nWOULD REMOVE {len(preview['removed'])} locations "
          f"({preview['freed_gb']} GB) in categories {list(cats)}:")
    for p in preview["removed"]:
        print(f"  {p}")

    if not args.yes:
        # A prompt, because --clean --all deletes a week of training. `--yes` exists for
        # scripted use where the caller has already decided.
        try:
            answer = input("\nType 'yes' to delete these: ").strip().lower()
        except EOFError:
            print("no terminal to confirm on; re-run with --yes", file=sys.stderr)
            return 1
        if answer != "yes":
            print("aborted, nothing removed.")
            return 1

    result = clean(cats, dry_run=False)
    print(f"\nremoved {len(result['removed'])} locations, freed {result['freed_gb']} GB")
    for f in result["failed"]:
        print(f"  FAILED {f}", file=sys.stderr)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
