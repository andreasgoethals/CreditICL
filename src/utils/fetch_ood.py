"""Download the out-of-domain benchmark suites. RUN THIS ON A VSC LOGIN NODE.

    python scripts/fetch_ood.py            # fetch 10 classification + 10 regression
    python scripts/fetch_ood.py --status   # what is already cached
    python scripts/fetch_ood.py --n 20     # more per task

WHY A LOGIN NODE: VSC compute nodes have **no outbound internet**, so a training or
evaluation job cannot download anything. This caches the tables to project storage once;
everything afterwards reads the cache and never imports `openml`.

WHAT IT FETCHES: OpenML-CC18 (classification — the suite O'Prior evaluated on) and
OpenML-CTR23 (regression). Dataset ids are resolved from the suites through the API and
pinned into `ood_manifest.json`, never hard-coded. Anything credit-like is dropped, since
a credit dataset is not out-of-domain for this project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging_setup import log_environment, log_section, setup_logging  # noqa: E402
from src.utils.paths import logs_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="datasets per task type")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--max-rows", type=int, default=50_000)
    ap.add_argument("--max-features", type=int, default=500)
    args = ap.parse_args()

    log, _, log_path = setup_logging("fetch_ood", logs_dir(), console=True)
    log_section(log, "CreditICL — OUT-OF-DOMAIN BENCHMARK FETCH")

    from src.eval.ood import fetch_ood_datasets, ood_status

    if args.status:
        status = ood_status()
        print(json.dumps(status, indent=2))
        return 0 if status["complete"] else 1

    log_environment(log, {"pipeline": "ood_fetch", "n_per_task": args.n})
    log.info(
        "This needs internet. If it fails with a connection error you are on a COMPUTE "
        "node — run it from a login node instead."
    )

    kept = fetch_ood_datasets(
        n_per_task=args.n, force=args.force,
        max_rows=args.max_rows, max_features=args.max_features,
    )

    status = ood_status()
    log.info("cached %d datasets: %s", len(kept), status["by_kind"])
    for kind, names in status["names"].items():
        log.info("  %-15s %s", kind, ", ".join(sorted(names)))
    if not status["complete"]:
        log.warning(
            "Fewer than %d datasets for at least one task type. Usable, but say so when "
            "reporting — a thin out-of-domain set is weak evidence either way.",
            args.n,
        )
    log.info("log file -> %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
