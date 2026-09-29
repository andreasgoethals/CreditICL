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

    The evaluation cells a configuration must have depend on the protocol that wrote the rows
    (`src/eval/protocol.py`): protocol 3 — one per development dataset and cross-validation fold;
    protocol 2 — one per development dataset and evaluation seed 0, 1, 2.
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
    # Arms of a grid the config no longer describes (the 45-arm Exp1 benchmark, read as protocol 2
    # after the redesign) are ranked with the training seeds the data holds for them.
    ours = part[part["model"].eq("crediticl")]
    for configuration, identities in ours.groupby("configuration")["identity"]:
        groups.setdefault(configuration, sorted(set(identities)))
    for model in part.loc[~part["model"].eq("crediticl"), "model"].unique():
        groups[model] = [model]
    rows = []
    by_fold = "fold" in part.columns and part["fold"].notna().any()
    if by_fold:
        from src.eval.protocol import N_FOLDS

        cell_cols = ["identity", "dataset", "seed", "fold"]
        eval_cells = [(0, f) for f in range(N_FOLDS)]
    else:
        cell_cols = ["identity", "dataset", "seed"]
        eval_cells = [(seed,) for seed in (0, 1, 2)]
    for configuration, names in groups.items():
        cells = part[part["configuration"].eq(configuration)]
        expected = {(name, ds, *cell) for name in names for ds in dev for cell in eval_cells}
        keys = list(cells[cell_cols].astype({c: int for c in cell_cols[2:]}).itertuples(index=False, name=None))
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
    from src.eval.benchmark_status import complete, n_slots, slot_for
    from src.visualize.results_plots import load_results

    # Selection reads the FINAL checkpoints and the reference models; earlier checkpoints are
    # the learning curve, not a candidate.
    needed = [i for i in range(n_slots(1, args.track))
              if (slot := slot_for(1, args.track, i)).kind == "reference" or slot.final]
    missing = [i for i in needed if not complete(1, args.track, i)]
    if missing:
        print(f"Selection blocked: incomplete benchmark indices {missing}")
        return 1
    ranking = development_ranking(load_results(args.track), args.track)
    print(ranking.to_string(index=False))
    print("Choose the prior from development results; review OOD retention and uncertainty before filling Exp2/3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
