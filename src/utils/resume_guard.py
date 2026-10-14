"""Permit automatic continuation only after a walltime signal and real progress."""
from __future__ import annotations

import argparse
import json


def require_progress(summary: dict, segment: int, limit: int = 32) -> None:
    if segment < 0 or limit < 1 or segment >= limit:
        raise ValueError(f"Automatic continuation limit reached ({segment}/{limit}); inspect the run")
    if (summary.get("completed") or summary.get("stopped_by_signal") != "SIGUSR1"
            or int(summary.get("steps", 0)) <= int(summary.get("resumed_at", 0))):
        raise ValueError("No automatic continuation: cancellation, completed run, or no progress")


def main() -> None:
    from src.utils.config import expand_with_seeds, load, run_name
    from src.utils.paths import run_summary_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("index", type=int)
    parser.add_argument("segment", type=int)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    run = expand_with_seeds(load(args.config))[args.index]
    require_progress(json.loads(run_summary_path(run_name(run)).read_text(encoding="utf-8")),
                     args.segment, args.limit)


if __name__ == "__main__":
    main()
