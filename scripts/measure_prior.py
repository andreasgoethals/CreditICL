"""Measure what a prior actually produces, before training anything on it.

This is the honesty check. Several claims about "what the original prior does and
does not contain" are easy to assert and easy to get wrong — this project already
had to walk two of them back after reading the code. So measure instead.

    python scripts/measure_prior.py --config config/LGD.yaml --n 500
    python scripts/measure_prior.py --config config/PD.yaml  --n 500 \
        --credit-fraction 0.0   # measure the ORIGINAL prior on its own

What it reports, per prior:

* **LGD** — how much mass sits at exactly 0 and exactly 1, how often either atom
  appears at all, the target's range, and how many distinct values it takes.
* **PD** — the base-rate distribution: mean, percentiles, and what fraction of
  tasks land under 5% and under 10% minority.
* **both** — the ExtraTrees pseudo-R² distribution (how predictable the tasks
  are), the filter's rejection rate, feature counts and categorical cardinality.

Set `--credit-fraction 0.0` to characterise the unmodified TabICL prior. That
number is the baseline every claim in the write-up should be measured against.

CPU only, so it runs free on the `interactive` partition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.utils.config import expand_with_seeds, load_yaml  # noqa: E402


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=500, help="how many tasks to sample")
    ap.add_argument("--index", type=int, default=0, help="which grid point to measure")
    ap.add_argument("--credit-fraction", type=float, default=None, help="override the mixture lever")
    ap.add_argument("--out", default=None, help="write the JSON report here as well")
    args = ap.parse_args()

    cfg = expand_with_seeds(load_yaml(args.config))[args.index]
    task = cfg["task"]
    if args.credit_fraction is not None:
        cfg["prior"]["credit_fraction"] = args.credit_fraction

    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG

    gen = TaskGenerator(cfg["prior"], task, PriorRNG(cfg.get("seed", 0)))

    p0s: list[float] = []
    p1s: list[float] = []
    rates: list[float] = []
    n_features: list[int] = []
    n_distinct: list[int] = []
    y_min, y_max = float("inf"), float("-inf")
    sources = {"base": 0, "credit": 0}

    for i in range(args.n):
        t = gen.sample()
        sources[t.source] = sources.get(t.source, 0) + 1
        n_features.append(t.n_features)
        y = t.y
        y_min = min(y_min, float(y.min()))
        y_max = max(y_max, float(y.max()))
        n_distinct.append(int(len(y.unique())))

        if task == "lgd":
            p0s.append(float((y <= 0.0).float().mean()))
            p1s.append(float((y >= 1.0).float().mean()))
        else:
            rates.append(float(y.mean()))

        if (i + 1) % 100 == 0:
            print(f"  sampled {i + 1}/{args.n}", flush=True, file=sys.stderr)

    report: dict = {
        "config": args.config,
        "task": task,
        "credit_fraction": cfg["prior"]["credit_fraction"],
        "n_sampled": args.n,
        "sources": sources,
        "features": {"mean": round(float(np.mean(n_features)), 1), "min": min(n_features), "max": max(n_features)},
        "target_range": [round(y_min, 4), round(y_max, 4)],
        "distinct_target_values": {
            "median": int(np.median(n_distinct)),
            "p10": int(pct(n_distinct, 10)),
            "p90": int(pct(n_distinct, 90)),
        },
        "filter": gen.filter_summary(),
    }

    if task == "lgd":
        both = [a + b for a, b in zip(p0s, p1s)]
        report["boundary_mass"] = {
            "at_0_mean": round(float(np.mean(p0s)), 4),
            "at_1_mean": round(float(np.mean(p1s)), 4),
            "at_0_p90": round(pct(p0s, 90), 4),
            "at_1_p90": round(pct(p1s, 90), 4),
            "any_atom_fraction": round(float(np.mean([b > 0 for b in both])), 4),
            "total_mass_mean": round(float(np.mean(both)), 4),
        }
        # The comparison that matters: real LGD, measured 2026-08-05.
        report["real_data_reference"] = {
            "0006.lgd_freddie": {"at_0": 0.114, "at_1": 0.081, "shape": "U-shaped"},
            "0007.lgd_lendingclub": {"at_0": 0.015, "at_1": 0.003, "shape": "unimodal, left-skewed"},
            "0003.axa_recovery": {"at_0": 0.0, "at_1": 0.0, "shape": "fully interior"},
        }
    else:
        report["base_rate"] = {
            "mean": round(float(np.mean(rates)), 4),
            "p10": round(pct(rates, 10), 4),
            "p50": round(pct(rates, 50), 4),
            "p90": round(pct(rates, 90), 4),
            "frac_under_05": round(float(np.mean([r < 0.05 for r in rates])), 4),
            "frac_under_10": round(float(np.mean([r < 0.10 for r in rates])), 4),
        }
        report["real_data_reference"] = {
            "gmsc": 0.067,
            "home_credit": 0.081,
            "hmeq": 0.200,
            "taiwan_creditcard": 0.221,
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
