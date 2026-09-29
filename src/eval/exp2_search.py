"""Experiment 2, stage A: which continued-pretraining recipe each model gets.

    python -m src.eval.exp2_search --track pd      # the table and the values for FILL_FROM_SEARCH
    python -m src.eval.exp2_search --track lgd

Reads the stage-A benchmark (`config/Exp2_<TRACK>_search.yaml`, scored with `VARIANT=search`:
DEVELOPMENT datasets only, every saved checkpoint) and the reference column's released weights,
and ranks every (arm, checkpoint) of each model by the development metric — ROC-AUC for PD, R²
for LGD, the metrics `selection.development_ranking` uses — with equal weight per dataset.

THE FORGETTING GUARD. A recipe that gains on credit by losing its generality is not a better
recipe for a foundation model. A candidate is admissible only if its mean out-of-domain metric is
no more than `--ood-tolerance` below the released weights' (default 0.01 ROC-AUC / 0.02 R²); the
best admissible candidate wins. If none is admissible the least-forgetting one is reported, and
flagged — the decision is then a person's, not this script's.

Nothing here reads a holdout dataset: the search configs list none, and the reference rows are
filtered to the development datasets.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name

ROOT = Path(__file__).resolve().parents[2]
METRIC = {"pd": "roc_auc", "lgd": "r2"}
OOD_TOLERANCE = {"pd": 0.01, "lgd": 0.02}
RELEASED = {"tabicl": "tabiclv2", "tabpfn3": "tabpfn3"}


def _per_dataset(df: pd.DataFrame, metric: str) -> pd.Series:
    """Fold-mean per dataset, then the mean over datasets: equal weight per dataset."""
    ok = df[df.get("status", pd.Series("ok", index=df.index)).eq("ok")]
    if ok.empty or metric not in ok:
        return pd.Series(dtype=float)
    return ok.groupby("dataset")[metric].mean()


def _read(files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in files if f.is_file()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def search_table(track: str) -> pd.DataFrame:
    """One row per (arm, checkpoint step) of the stage-A grid, with the released model beside it."""
    cfg = load(ROOT / "config" / f"Exp2_{track.upper()}_search.yaml")
    dev = list(cfg["eval"]["dev_datasets"])
    metric = METRIC[track]
    runs = expand_with_seeds(cfg)
    out_credit = paths.benchmark_dir(2, track)
    out_ood = paths.benchmark_dir(2, "ood")
    ref_credit = paths.benchmark_dir(paths.REFERENCE, track)
    ref_ood = paths.benchmark_dir(paths.REFERENCE, "ood")

    released: dict[str, tuple[float, float, int]] = {}
    for arch, ref_model in RELEASED.items():
        c = _read([ref_credit / f"results_reference_{track}_{ref_model}.csv"])
        o = _read([ref_ood / f"ood_results_reference_{track}_{ref_model}.csv"])
        c = c[c["dataset"].isin(dev)] if len(c) else c
        per = _per_dataset(c, metric)
        released[arch] = (float(per.mean()) if len(per) else np.nan,
                          float(_per_dataset(o, metric).mean()) if len(o) else np.nan, len(per))

    rows = []
    for i, run in enumerate(runs):
        arch = str(run.get("architecture", "tabicl"))
        pattern = re.compile(rf"results_exp2searchbench_{track}_a{i}(?:_s\d+)?\.csv$")
        for f in sorted(out_credit.glob(f"results_exp2searchbench_{track}_a{i}*.csv")):
            if not pattern.match(f.name):
                continue
            c = _read([f])
            if c.empty or "info_checkpoint_step" not in c:
                continue
            step = int(pd.to_numeric(c["info_checkpoint_step"], errors="coerce").dropna().iloc[0])
            per = _per_dataset(c[c["dataset"].isin(dev)], metric)
            o = _read([out_ood / f.name.replace("results_", "ood_results_", 1)])
            if len(o) and "status" in o:
                o = o[o["status"] != "skipped"]
            rel_dev, rel_ood, _ = released[arch]
            dev_mean = float(per.mean()) if len(per) else np.nan
            ood_mean = float(_per_dataset(o, metric).mean()) if len(o) else np.nan
            rows.append({
                "model": arch, "arm": run.get("arm"), "run_name": run_name(run), "step": step,
                "optimizer": run["train"]["optimizer"], "lr": float(run["train"]["lr"]),
                "warmup_proportion": float(run["train"]["warmup_proportion"]),
                "dev": dev_mean, "dev_datasets": int(len(per)), "complete": len(per) == len(dev),
                "ood": ood_mean, "released_dev": rel_dev, "released_ood": rel_ood,
                "dev_gain": dev_mean - rel_dev, "ood_change": ood_mean - rel_ood,
            })
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame, track: str, ood_tolerance: float | None = None) -> dict[str, dict]:
    """The recipe per model: the best admissible development score (see the module docstring)."""
    tol = OOD_TOLERANCE[track] if ood_tolerance is None else float(ood_tolerance)
    picks: dict[str, dict] = {}
    if table.empty:
        return picks
    for model, part in table[table["complete"]].groupby("model"):
        admissible = part[~(part["ood_change"] < -tol)]  # a missing OOD score does not disqualify
        if len(admissible):
            best = admissible.sort_values(["dev", "step"], ascending=[False, True]).iloc[0]
            flag = ""
        else:
            best = part.sort_values("ood_change", ascending=False).iloc[0]
            flag = (f"NO candidate kept out-of-domain within {tol} of the released weights; this is "
                    f"the least-forgetting one. Decide by hand.")
        picks[model] = {**best.to_dict(), "ood_tolerance": tol, "flag": flag}
    return picks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=("pd", "lgd"))
    ap.add_argument("--ood-tolerance", type=float, default=None)
    args = ap.parse_args(argv)
    table = search_table(args.track)
    if table.empty:
        print(f"No stage-A benchmark results under {paths.benchmark_dir(2, args.track)} yet.")
        return 1
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.sort_values(["model", "dev"], ascending=[True, False]).to_string(index=False))
    print()
    for model, pick in choose(table, args.track, args.ood_tolerance).items():
        print(f"{model}: {pick['arm']} at step {pick['step']}  dev {METRIC[args.track]} {pick['dev']:.4f} "
              f"({pick['dev_gain']:+.4f} vs released)  OOD {pick['ood']:.4f} ({pick['ood_change']:+.4f})")
        if pick["flag"]:
            print(f"  WARNING: {pick['flag']}")
    print("\nFill config/Exp2_%s.yaml:" % args.track.upper())
    picks = choose(table, args.track, args.ood_tolerance)
    if "tabicl" in picks:
        p = picks["tabicl"]
        print(f"  arms.tabicl: train.optimizer: {p['optimizer']}, train.lr: {p['lr']:g}, "
              f"train.warmup_proportion: {p['warmup_proportion']:g}")
    if "tabpfn3" in picks:
        print(f"  arms.tabpfn3: train.lr: {picks['tabpfn3']['lr']:g}")
    steps = {m: int(p["step"]) for m, p in picks.items()}
    if len(set(steps.values())) == 1:
        print(f"  eval.headline_step: {next(iter(steps.values()))}")
    else:
        print(f"  eval.headline_step: the two models chose different lengths {steps} — one headline "
              f"step per model is needed; report both and decide before looking at the holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
