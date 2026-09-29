"""Did Experiment 0 work? Read the downloaded `output_CreditICL/experiment_0/` and say so.

    python -m src.utils.exp0_verify              # the output tree in this repo
    python -m src.utils.exp0_verify --root D:/x  # a downloaded copy elsewhere (the folder holding output_CreditICL)

Every item is PASS, WARN or FAIL with the evidence, and the last section turns the measured
speeds into what Experiment 1 and Experiment 2 will cost — the reason Experiment 0 runs the real
protocol rather than a toy one. Reads only; the big tier (checkpoints) is not downloaded and not
needed: the run summaries and the benchmark receipts are the evidence that the checkpoints
existed and loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
ROOT = Path(__file__).resolve().parents[2]


class Verdicts:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def add(self, status: str, what: str, evidence: str = "") -> None:
        self.items.append((status, what, evidence))
        print(f"[{status}] {what}" + (f" — {evidence}" if evidence else ""))


def _runs(track: str) -> list[dict]:
    from src.utils.config import expand_with_seeds, load

    return expand_with_seeds(load(ROOT / "config" / f"Exp0_{track.upper()}.yaml"))


def verify(out_root: Path) -> Verdicts:
    from src.eval.benchmark_status import slot_for

    v = Verdicts()
    exp0 = out_root / "experiment_0"
    report = exp0 / "report" / "checks.json"
    if report.is_file():
        rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
        bad = [r for r in rows if r["status"] == FAIL]
        warn = [r for r in rows if r["status"] == WARN]
        v.add(FAIL if bad else PASS, "checks job",
              f"{len(rows)} checks, {len(bad)} FAIL, {len(warn)} WARN"
              + ("".join(f"\n      FAIL {r['check']}: {r['name']} — {str(r['detail'])[:300]}" for r in bad)))
        for r in rows:
            if r["check"] == "train":
                v.add(PASS if r["status"] == PASS else FAIL, f"checks: {r['name']} steps", str(r["detail"])[:300])
    else:
        v.add(FAIL, "checks job", f"no {report} — was scripts/slurm/exp0.slurm run and downloaded?")

    logs = sorted((exp0 / "logs").glob("*.log")) if (exp0 / "logs").is_dir() else []
    text = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in logs}
    failed = [n for n, t in text.items() if "END status=FAILED" in t]
    v.add(FAIL if failed else (PASS if text else FAIL), "job logs",
          f"{len(text)} logs; FAILED: {', '.join(failed) or 'none'}")
    resubmitted = [n for n, t in text.items() if "END status=INCOMPLETE-RESUBMITTED" in t]
    v.add(PASS if resubmitted else FAIL, "forced stop: a job saved on SIGUSR1 and resubmitted itself",
          ", ".join(resubmitted) or "no log ends INCOMPLETE-RESUBMITTED — was the scratch arm submitted with the short walltime?")

    speeds: dict[tuple[str, str], float] = {}
    for track in ("pd", "lgd"):
        for run in _runs(track):
            name, arm = run["_run_name"], run["arm"]
            d = exp0 / "runs" / name
            summary = d / "summary.json"
            if not summary.is_file():
                v.add(FAIL, f"{track} {arm}: trained", f"no {summary}")
                continue
            s = json.loads(summary.read_text(encoding="utf-8"))
            ok = s.get("completed") and s.get("steps") == run["train"]["max_steps"]
            v.add(PASS if ok else FAIL, f"{track} {arm}: trained to the end",
                  f"steps {s.get('steps')}/{run['train']['max_steps']}, {s.get('train_seconds_total')} s of training")
            if arm == "scratch_tabicl":
                v.add(PASS if s.get("resumed_at") else FAIL, f"{track} {arm}: resumed from a checkpoint",
                      f"resumed_at={s.get('resumed_at')}")
            tel = d / "telemetry.csv"
            if tel.is_file():
                t = pd.read_csv(tel)
                hw = t[t.get("kind", pd.Series(dtype=str)).eq("hardware")] if "kind" in t else t
                steps = t["step"].tolist()
                util = hw.get("gpu0_utilization_gpu")
                dup = len(steps) != len(set(zip(t["step"], t.get("kind", pd.Series([""] * len(t))))))
                v.add(WARN if dup else PASS, f"{track} {arm}: telemetry",
                      f"{len(t)} rows, GPU util median {float(util.median()) if util is not None and util.notna().any() else 'n/a'}"
                      + (" — DUPLICATE steps (a resume re-wrote rows)" if dup else ""))
                if "steps_per_s" in hw and hw["steps_per_s"].notna().any():
                    speeds[(track, arm)] = float(hw["steps_per_s"].tail(max(1, len(hw) // 2)).median())
            else:
                v.add(FAIL, f"{track} {arm}: telemetry", f"no {tel}")
            w = d / "weights.csv"
            if w.is_file():
                wd = pd.read_csv(w)
                v.add(PASS if len(wd) >= 2 and wd["drift_all"].iloc[-1] > 0 else FAIL, f"{track} {arm}: weight drift",
                      f"{len(wd)} rows, final relative drift {wd['drift_rel_all'].iloc[-1]:.3g}")
            else:
                v.add(FAIL, f"{track} {arm}: weight drift", f"no {w}")
            prog = d / "progress.csv"
            if prog.is_file():
                pr = pd.read_csv(prog)
                real = [c for c in pr.columns if c.startswith("real__")]
                ood = [c for c in pr.columns if c.startswith("ood__")]
                v.add(PASS if len(pr) >= 2 and real else FAIL, f"{track} {arm}: progress curve",
                      f"{len(pr)} points, {len(real)} real-data columns, {len(ood)} out-of-domain columns")
            else:
                v.add(FAIL, f"{track} {arm}: progress curve", f"no {prog}")

    costs: dict[str, list[float]] = {}
    for track in ("pd", "lgd"):
        from src.eval.benchmark_status import n_slots

        for index in range(n_slots(0, track)):
            slot = slot_for(0, track, index)
            owner = "reference" if slot.kind == "reference" else "experiment_0"
            receipt = out_root / owner / "benchmark" / "receipts" / f"{slot.tag}.json"
            result = out_root / owner / "benchmark" / track / f"results_{slot.tag}.csv"
            label = f"{track} benchmark slot {index} ({slot.kind} {slot.model})"
            if receipt.is_file():
                hours = pd.read_csv(result)["total_seconds"].sum() / 3600 if result.is_file() else float("nan")
                v.add(PASS, label, f"receipt written; {hours:.2f} h of scoring")
                costs.setdefault(f"{track} {slot.kind} {slot.model}", []).append(hours)
            else:
                v.add(FAIL, label, f"no receipt {receipt.name} (did the slot finish? its log says why)")

    print("\n--- WHAT THE REAL EXPERIMENTS WILL COST (from these measurements) " + "-" * 20)
    for (track, arm), sps in sorted(speeds.items()):
        steps = {"scratch_tabicl": 12_500, "finetune_tabicl": 10_000, "finetune_tabpfn3": 10_000}[arm]
        print(f"  {track} {arm:<17} {sps:.3f} steps/s -> {steps / sps / 3600:5.1f} h per arm of {steps:,} steps")
    for key, hours in sorted(costs.items()):
        print(f"  {key:<34} {sum(hours) / len(hours):5.2f} h per benchmark slot")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="the folder that holds output_CreditICL/")
    args = ap.parse_args(argv)
    from src.utils.paths import OUTPUT_NAME, outputs_dir

    out_root = Path(args.root) / OUTPUT_NAME if args.root else outputs_dir()
    v = verify(out_root)
    n_fail = sum(s == FAIL for s, _, _ in v.items)
    print(f"\n{len(v.items)} items, {n_fail} FAIL.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
