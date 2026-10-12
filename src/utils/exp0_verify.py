"""Did Experiment 0 work? Read the downloaded `output CreditICL/experiment_0/` and say so.

    python -m src.utils.exp0_verify              # the output tree in this repo
    python -m src.utils.exp0_verify --root D:/x  # a downloaded copy elsewhere (the folder holding output CreditICL)

Every item is PASS, WARN or FAIL with the evidence, and the last section turns the measured
speeds and sizes into what Experiments 1-3 will cost — GPU-hours, project storage, and whether
a benchmark slot fits its walltime — the reason Experiment 0 runs the real protocol rather than a
toy one. Reads only. Download both tiers into one `output CreditICL/` (docs/VSC.md §10): the small
one ($VSC_DATA: logs, run records, the checks report) and the big one WITHOUT its checkpoints
(project storage: the benchmark CSVs and receipts).
"""
from __future__ import annotations

import argparse
import json
import re
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


def _verify_receipt(receipt: Path, track: str, slot) -> float:
    """Verify a downloaded paired benchmark without requiring checkpoint binaries."""
    from src.eval.benchmark_status import digest, validate_cells
    from src.eval.protocol import PROTOCOL

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    request = payload["request"]
    if request["protocol"] != PROTOCOL or request["models"] != [slot.model]:
        raise ValueError("receipt belongs to a different protocol or model")
    if request.get("context_cap") != slot.context:
        raise ValueError("receipt has a different context budget")
    if slot.kind == "arm" and request.get("step") != slot.step:
        raise ValueError("receipt belongs to a different checkpoint step")
    root = receipt.parent.parent
    files = {"credit": root / track / f"results_{slot.tag}.csv",
             "ood": root / "ood" / f"ood_results_{slot.tag}.csv"}
    seconds = 0.0
    for domain, path in files.items():
        if digest(path) != payload["files"][domain]:
            raise ValueError(f"{domain} CSV hash differs from its receipt")
        rows = pd.read_csv(path)
        if domain == "ood" and "status" in rows:
            rows = rows[rows.status != "skipped"]
        validate_cells(rows, request[domain], request["models"], request["seeds"],
                       request["n_folds"], "roc_auc" if track == "pd" else "r2", slot.context)
        seconds += float(rows["total_seconds"].sum())
    return seconds / 3600


def verify(out_root: Path, *, big_root: Path | None = None) -> Verdicts:
    from src.utils.preflight import CHECK_SECTIONS, source_fingerprint, training_fingerprint
    from src.eval.benchmark_status import slot_for

    v = Verdicts()
    exp0 = out_root / "experiment_0"
    big_root = out_root if big_root is None else big_root
    report = exp0 / "report" / "checks.json"
    if report.is_file():
        saved_report = json.loads(report.read_text(encoding="utf-8"))
        rows = saved_report["rows"]
        current = (CHECK_SECTIONS <= {r["check"] for r in rows}
                   and saved_report.get("source_fingerprint") == source_fingerprint())
        v.add(PASS if current else FAIL, "complete checks report for current code",
              "all check sections and matching source fingerprint required")
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
    v.add(WARN if failed else (PASS if text else FAIL), "historical job logs",
          f"{len(text)} logs; FAILED: {', '.join(failed) or 'none'}")
    check_logs = [p for p in logs if p.name.startswith("exp0_checks_")]
    if check_logs:
        latest = max(check_logs, key=lambda p: p.name)
        latest_text = text[latest.name]
        ok = "END status=OK" in latest_text and all(
            f"equivalence {track} exit code: 0" in latest_text for track in ("PD", "LGD"))
        v.add(PASS if ok else FAIL, "latest checks job including trainer equivalence", latest.name)
    else:
        v.add(FAIL, "latest checks job including trainer equivalence", "missing exp0_checks log")
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
            from src.train.checkpoint import validate_resume_config

            try:
                saved_config = json.loads((d / "config.json").read_text(encoding="utf-8"))
                if not {"task", "prior", "train"} <= saved_config.keys():
                    raise ValueError("Incomplete saved training configuration")
                validate_resume_config(saved_config, run)
                v.add(PASS, f"{track} {arm}: training recipe", "matches current configuration")
            except (OSError, ValueError, TypeError) as exc:
                v.add(FAIL, f"{track} {arm}: training recipe", str(exc))
            ok = s.get("completed") and s.get("steps") == run["train"]["max_steps"]
            ok = ok and s.get("training_fingerprint") == training_fingerprint()
            v.add(PASS if ok else FAIL, f"{track} {arm}: trained to the end",
                  f"steps {s.get('steps')}/{run['train']['max_steps']}, {s.get('train_seconds_total')} s of training")
            if arm == "scratch_tabicl":
                v.add(PASS if s.get("resumed_at") else FAIL, f"{track} {arm}: resumed from a checkpoint",
                      f"resumed_at={s.get('resumed_at')}")
            tel = d / "telemetry.csv"
            if tel.is_file():
                t = pd.read_csv(tel)
                if arm == "scratch_tabicl":
                    # A resumed arm's telemetry must continue where it stopped: every kind's steps
                    # increase, and the last one is the final step.
                    kinds = t["kind"] if "kind" in t else pd.Series(["all"] * len(t))
                    rising = all(g["step"].is_monotonic_increasing for _, g in t.groupby(kinds))
                    reached = int(t["step"].max()) >= int(run["train"]["max_steps"]) - int(
                        run["train"].get("log_every", 1) or 1)
                    v.add(PASS if rising and reached else FAIL, f"{track} {arm}: telemetry continues across the resume",
                          f"steps rising per kind: {rising}; last step {int(t['step'].max())} of {run['train']['max_steps']}")
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
    slot_hours: list[float] = []
    for track in ("pd", "lgd"):
        from src.eval.benchmark_status import n_slots

        for index in range(n_slots(0, track)):
            slot = slot_for(0, track, index)
            owner = "reference" if slot.kind == "reference" else "experiment_0"
            receipt = big_root / owner / "benchmark" / "receipts" / f"{slot.tag}.json"
            result = big_root / owner / "benchmark" / track / f"results_{slot.tag}.csv"
            label = f"{track} benchmark slot {index} ({slot.kind} {slot.model})"
            if receipt.is_file():
                try:
                    hours = _verify_receipt(receipt, track, slot)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    v.add(FAIL, label, str(exc))
                    continue
                v.add(PASS, label, f"paired CSV hashes and cells verified; {hours:.2f} h of scoring")
                kind = f"{slot.kind} {slot.model}" + (f" ctx{slot.context}" if slot.context else "")
                costs.setdefault(f"{track} {kind}", []).append(hours)
                slot_hours.append(hours)
            else:
                v.add(FAIL, label, f"no receipt {receipt.name} (did the slot finish? its log says why)")

    walltime = _walltime_hours(ROOT / "scripts" / "slurm" / "benchmark.slurm")
    if slot_hours and walltime:
        worst = max(slot_hours)
        v.add(PASS if worst < 0.8 * walltime else WARN, "benchmark slots fit their walltime",
              f"longest slot {worst:.2f} h of a {walltime:.0f} h limit (a requeue re-runs a slot from the start)")

    print("\n--- WHAT THE REAL EXPERIMENTS WILL COST (from these measurements) " + "-" * 20)
    # (experiment, arm kind in Exp0, steps, arms over both tracks, kept checkpoints per arm)
    plan = [("Exp1", "scratch_tabicl", 12_500, 9, 5 + 2), ("Exp2", "finetune_tabicl", 10_000, 9, 4 + 2),
            ("Exp2", "finetune_tabpfn3", 10_000, 9, 4 + 2), ("Exp3", "scratch_tabicl", 200_000, 4, 8 + 2)]
    sizes = _checkpoint_mb(exp0 / "report" / "checks.json")
    totals: dict[str, list[float]] = {}
    for exp, arm, steps, arms_per_track, kept in plan:
        for track in ("pd", "lgd"):
            sps = speeds.get((track, arm))
            if not sps:
                continue
            hours = steps / sps / 3600
            gb = arms_per_track * kept * (sizes.get((track, arm)) or float("nan")) / 1000
            totals.setdefault(exp, [0.0, 0.0])
            totals[exp][0] += arms_per_track * hours
            totals[exp][1] += gb if gb == gb else 0.0
            print(f"  {exp} {track} {arm:<17} {sps:.3f} steps/s -> {hours:6.1f} h per arm of {steps:,} steps, "
                  f"x{arms_per_track} arms; checkpoints ~{gb:.1f} GB")
    for exp, (hours, gb) in totals.items():
        print(f"  {exp} total: ~{hours:,.0f} GPU-h of training, ~{gb:.0f} GB of checkpoints on project storage")
    for key, hours in sorted(costs.items()):
        print(f"  {key:<40} {sum(hours) / len(hours):5.2f} h per benchmark slot")
    return v


def _walltime_hours(script: Path) -> float | None:
    m = re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", script.read_text(encoding="utf-8")) if script.is_file() else None
    return int(m.group(1)) + int(m.group(2)) / 60 if m else None


def _checkpoint_mb(report: Path) -> dict[tuple[str, str], float]:
    """The checks job's measured checkpoint sizes, per (track, arm)."""
    if not report.is_file():
        return {}
    out = {}
    for r in json.loads(report.read_text(encoding="utf-8")).get("rows", []):
        if r.get("check") == "train" and isinstance(r.get("detail"), dict) and r["detail"].get("checkpoint_mb"):
            track, arm = r["name"].split(" ", 1)
            out[(track, arm)] = float(r["detail"]["checkpoint_mb"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="the folder that holds output CreditICL/")
    args = ap.parse_args(argv)
    from src.utils.paths import OUTPUT_NAME, outputs_dir, big_outputs_dir

    out_root = Path(args.root) / OUTPUT_NAME if args.root else outputs_dir()
    v = verify(out_root, big_root=out_root if args.root else big_outputs_dir())
    n_fail = sum(s == FAIL for s, _, _ in v.items)
    print(f"\n{len(v.items)} items, {n_fail} FAIL.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
