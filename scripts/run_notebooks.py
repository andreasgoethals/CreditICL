"""Run every notebook in parallel and export the figures as PDFs.

    python scripts/run_notebooks.py                 # all of them
    python scripts/run_notebooks.py --only data_exploration
    python scripts/run_notebooks.py --workers 3

Each notebook's figures go to `output/figures/<notebook>/` as PDFs, alongside a `captions.md`
with a paper-ready caption per figure. The folder is WIPED first, so only the current
run's figures survive — stale PDFs mixed with fresh ones is how a paper ends up with a
figure that no longer matches the code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_notebooks import NOTEBOOKS, run_all, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated notebook names")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes")
    args = ap.parse_args()

    notebooks = tuple(n.strip() for n in args.only.split(",")) if args.only else NOTEBOOKS
    unknown = [n for n in notebooks if n not in NOTEBOOKS]
    if unknown:
        print(f"unknown notebooks: {unknown}; known: {list(NOTEBOOKS)}", file=sys.stderr)
        return 2

    print(f"running {len(notebooks)} notebook(s) in parallel: {', '.join(notebooks)}")
    results = run_all(notebooks, max_workers=args.workers)
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
