"""Descriptive Exp1 development ranking; Exp3 selection is in src.experiments.selection."""
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
    if df is None or not {"dataset", "model", "seed", "status", metric}.issubset(df):
        return pd.DataFrame(columns=columns)
    part = df[df["dataset"].isin(dev)].copy()
    identity = part.get("info_run_name", pd.Series("", index=part.index)).fillna("")
    part["identity"] = identity.where(identity.ne(""), part["model"])
    part["configuration"] = part["identity"].str.replace(r"__s\d+$", "", regex=True)
    groups: dict[str, list[str]] = {}
    for run in expand_with_seeds(cfg):
        name = run_name(run)
        groups.setdefault(re.sub(r"__s\d+$", "", name), []).append(name)
    # Arms of a grid the config no longer describes (the 45-arm Exp1 benchmark, read as protocol 2
    # after the redesign) are ranked with the training seeds the data holds for them.
    ours = part[identity.ne("")]
    for configuration, identities in ours.groupby("configuration")["identity"]:
        groups.setdefault(configuration, sorted(set(identities)))
    for model in part.loc[identity.eq(""), "model"].unique():
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
    from src.eval.metrics import higher_is_better

    higher = higher_is_better(metric)
    ranking = pd.DataFrame(rows, columns=columns).sort_values("mean", ascending=not higher)
    if not ranking.empty:
        candidates = ranking.configuration.str.startswith(f"{exp}_{track}__")
        ranking["within_seed_spread_of_best"] = False
        if candidates.any():
            best = ranking.loc[candidates].iloc[0]
            distance = (best["mean"] - ranking["mean"]) * (1 if higher else -1)
            ranking["within_seed_spread_of_best"] = candidates & (
                distance <= np.maximum(float(best["std"]), ranking["std"]))
    return ranking


def exp3_candidate(ranking: pd.DataFrame, track: str) -> dict | None:
    """Describe the best nonzero Exp1 mixture on the predeclared primary metric.

    Historical function name retained for callers; it no longer configures Exp3.

    The 0% arm remains the control even when it ranks first. Exact numerical ties
    between credit mixtures prefer the smaller credit fraction; seed spread is
    descriptive and does not change the ranking. References and historical arms
    cannot become candidates. All currently configured mixtures must be complete.
    """
    required = {"configuration", "mean", "metric"}
    if ranking is None or ranking.empty or not required.issubset(ranking):
        return None
    cfg = load(ROOT / f"config/Exp1_{track.upper()}.yaml")
    mixtures = {re.sub(r"__s\d+$", "", run_name(run)): float(run["prior"]["credit_fraction"])
                for run in expand_with_seeds(cfg)}
    part = ranking[ranking.configuration.isin(mixtures)].copy()
    metric = "roc_auc" if track == "pd" else "r2"
    if (set(part.configuration) != set(mixtures) or part.configuration.duplicated().any()
            or not part.metric.eq(metric).all() or not np.isfinite(part["mean"]).all()):
        return None
    part["credit_fraction"] = part.configuration.map(mixtures)
    controls = part[part.credit_fraction.eq(0)]
    candidates = part[part.credit_fraction.gt(0)].sort_values(
        ["mean", "credit_fraction"], ascending=[False, True], kind="stable")
    if len(controls) != 1 or candidates.empty:
        return None
    best = candidates.iloc[0]
    control_mean = float(controls.iloc[0]["mean"])
    return {"configuration": best.configuration, "credit_fraction": float(best.credit_fraction),
            "metric": metric, "mean": float(best["mean"]), "control_mean": control_mean,
            "difference_from_control": float(best["mean"]) - control_mean}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=("pd", "lgd"), required=True)
    args = ap.parse_args(argv)
    from src.eval.benchmark_status import complete, n_slots, slot_for
    from src.eval.results import load_results
    from src.eval.protocol import PROTOCOL

    # Selection reads the FINAL checkpoints and the reference models; earlier checkpoints are
    # the learning curve, not a candidate.
    needed = [i for i in range(n_slots(1, args.track))
              if (slot := slot_for(1, args.track, i)).kind == "reference" or slot.final]
    missing = [i for i in needed if not complete(1, args.track, i)]
    if missing:
        print(f"Selection blocked: incomplete benchmark indices {missing}")
        return 1
    ranking = development_ranking(load_results(args.track, protocol=PROTOCOL), args.track)
    choice = exp3_candidate(ranking, args.track)
    if choice is None:
        print(f"Selection blocked: no complete current-protocol ({PROTOCOL}) development ranking "
              "for every configured Exp1 mixture.")
        return 1
    print(ranking.to_string(index=False))
    print(f"Descriptive Exp1 ranking: credit_fraction={choice['credit_fraction']:g}; development "
          f"{choice['metric']}={choice['mean']:.6f}, 0% control={choice['control_mean']:.6f}, "
          f"difference={choice['difference_from_control']:+.6f}.")
    if choice["difference_from_control"] <= 0:
        print("No credit mixture beats the control on the development mean. "
              "This is an observed development ranking; "
              "it is not an established improvement.")
    print("The candidate is the best nonzero mixture; exact score ties prefer the smaller credit share. "
          "Seed spread describes stability, not a significance test. OOD retention is reported "
          "descriptively and must not re-rank this development-only choice. Exp2 uses all four "
          "mixtures at both predeclared learning rates per backbone/task. Exp3 fixes 50%, "
          "the primary-rate baseline and all method settings before observing scores. "
          "This ranking changes neither experiment. No config was changed and no job was submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
