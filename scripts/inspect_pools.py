"""What prior pools does this machine have, and are they complete?

    python scripts/inspect_pools.py
    python scripts/inspect_pools.py --task lgd

Run it after downloading pools from the cluster. It answers the two questions that
matter before you plot anything:

* **which variants are here** — so you know what the notebook will discover;
* **complete pool or sample** — a partial download is fine for looking at, but it
  must never be mistaken for the pool that was trained on.

Prints, and nothing else. Reports are the notebook's job; this is a status check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["lgd", "pd", "both"], default="both")
    args = ap.parse_args()

    from src.utils.paths import describe, prior_cache_dir
    from src.visualize.pool_plots import describe_pools, discover_pools

    root = prior_cache_dir("x").parent
    print(f"pool root: {root}")
    print(f"  (exists: {root.is_dir()})")
    where = describe()
    print(f"running on VSC: {where['on_vsc']}  (staging: {where['staging_root']})\n")

    tasks = ["lgd", "pd"] if args.task == "both" else [args.task]
    any_found = False
    for task in tasks:
        variants = discover_pools(task)
        print(f"=== {task.upper()} ===")
        if not variants:
            print("  no pools found\n")
            continue
        any_found = True
        frame = describe_pools(task, variants)
        print(frame.drop(columns=["path"]).to_string(index=False))
        samples = frame[frame["state"] == "SAMPLE"]["variant"].tolist()
        if samples:
            print(f"\n  NOTE: {', '.join(samples)} are partial downloads (a SAMPLE).")
            print("  Fine for the notebooks. Not the full pool the model trained on.")
        print()

    if not any_found:
        print("Nothing to inspect. Either generate pools:")
        print("    python scripts/generate_prior.py --config config/LGD.yaml --variant original --all")
        print("or copy them from the cluster:")
        print("    bash scripts/fetch_prior_sample.sh")
        return 1

    print("The notebooks discover these automatically — open")
    print("notebooks/prior_visualisation.ipynb.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
