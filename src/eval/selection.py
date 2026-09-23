"""Development-only, seed-aggregated configuration ranking for the next experiments."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import expand_with_seeds, load, run_name

ROOT = Path(__file__).resolve().parents[2]


def development_ranking(df: pd.DataFrame, track: str, exp: str = "exp1",
                        metric: str | None = None) -> pd.DataFrame:
    """Equal dataset weight, then equal evaluation/training seed weight.

    Incomplete configurations are excluded, never rewarded for missing hard datasets.
    Holdouts and OOD rows cannot influence the selection statistic.
    """
    cfg = load(ROOT / f"config/Exp{int(exp.removeprefix('exp'))}_{track.upper()}.yaml",
               allow_placeholders=True)
    if cfg["eval"].get("select_on") != "dev":
        raise ValueError("Prior selection requires eval.select_on: dev")
    dev = cfg["eval"]["dev_datasets"]
    metric = metric or ("roc_auc" if track == "pd" else "r2")
    columns = ["configuration", "mean", "std", "training_seeds", "datasets", "metric"]
    if not {"dataset", "model", "seed", "status", metric}.issubset(df):
        return pd.DataFrame(columns=columns)
    part = df[df["dataset"].isin(dev)].copy()
    identity = part.get("info_run_name", pd.Series("", index=part.index)).fillna("")
    part["identity"] = identity.where(part["model"].eq("crediticl"), part["model"])
    part["configuration"] = part["identity"].str.replace(r"__s\d+$", "", regex=True)
    groups: dict[str, list[str]] = {}
    for run in expand_with_seeds(cfg):
        name = run_name(run)
        groups.setdefault(re.sub(r"__s\d+$", "", name), []).append(name)
    for model in part.loc[~part["model"].eq("crediticl"), "model"].unique():
        groups[model] = [model]
    rows = []
    for configuration, names in groups.items():
        cells = part[part["configuration"].eq(configuration)]
        expected = {(name, ds, seed) for name in names for ds in dev for seed in (0, 1, 2)}
        keys = list(cells[["identity", "dataset", "seed"]].itertuples(index=False, name=None))
        if (len(keys) != len(set(keys)) or set(keys) != expected
                or not cells["status"].eq("ok").all()
                or not np.isfinite(cells[metric].astype(float)).all()):
            continue
        per_training_seed = cells.groupby("identity")[metric].mean()
        rows.append({"configuration": configuration, "mean": per_training_seed.mean(),
                     "std": per_training_seed.std(ddof=1) if len(names) > 1 else 0.0,
                     "training_seeds": len(names), "datasets": len(dev), "metric": metric})
    return pd.DataFrame(rows, columns=columns).sort_values("mean", ascending=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=("pd", "lgd"), required=True)
    args = ap.parse_args(argv)
    from src.eval.benchmark_status import complete
    from src.visualize.results_plots import load_results

    runs = expand_with_seeds(load(ROOT / f"config/Exp1_{args.track.upper()}.yaml"))
    missing = [i for i in range(len(runs) + 1) if not complete(1, args.track, i)]
    if missing:
        print(f"Selection blocked: incomplete benchmark indices {missing}")
        return 1
    ranking = development_ranking(load_results(args.track), args.track)
    print(ranking.to_string(index=False))
    print("Choose the prior from development results; review OOD retention and uncertainty before filling Exp2/3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
