"""PIPELINE 1 — preprocess the raw credit datasets into the cache.

    python scripts/preprocess.py --task lgd
    python scripts/preprocess.py --task both --force
    python scripts/preprocess.py --task pd --datasets 0008.german,0013.hmeq

You rarely need to run this by hand: the eval pipeline preprocesses whatever is
missing. Run it when you want the cache built up front (for example on a login
node before submitting an array, so 48 tasks do not all preprocess Home Credit at
once).

Writes a per-dataset summary to results/<task>/data/. Logs go to logs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.discovery import describe_availability  # noqa: E402
from src.data.pipeline import cache_report, ensure_processed  # noqa: E402
from src.utils.logging_setup import log_environment, log_section, setup_logging  # noqa: E402
from src.utils.paths import describe, logs_dir, results_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("pd", "lgd", "both"), default="both")
    ap.add_argument("--datasets", default=None, help="comma-separated slugs; default is all found")
    ap.add_argument("--force", action="store_true", help="re-preprocess even if cached")
    args = ap.parse_args()

    tasks = ["pd", "lgd"] if args.task == "both" else [args.task]
    log, _, log_path = setup_logging("preprocess", logs_dir(), level="INFO", console=True)
    log_section(log, "CreditICL — DATA PIPELINE")
    log_environment(log, {"pipeline": "data"})
    log.info("storage:\n%s", "\n".join(f"    {k} = {v}" for k, v in describe().items()))
    log.info("raw data availability:\n%s", describe_availability())
    log.info("log file -> %s", log_path)

    failures = 0
    for task in tasks:
        datasets = args.datasets.split(",") if args.datasets else None
        result = ensure_processed(task, datasets, force=args.force)
        failures += sum(v is None for v in result.values())

        out = results_dir(task, "data")
        out.mkdir(parents=True, exist_ok=True)
        report = cache_report(task)
        path = out / "processed_summary.csv"
        report.to_csv(path, index=False)
        log.info("[data] wrote %s (%d rows)", path, len(report))

    if failures:
        log.error("%d dataset(s) failed — see the tracebacks above", failures)
        return 1
    log.info("all datasets ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
