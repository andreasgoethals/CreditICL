"""Did Experiment 0 work? Read the downloaded `output CreditICL/experiment_0/` and say so.

    python -m src.validation.verify              # the output tree in this repo
    python -m src.validation.verify --root D:/x  # a downloaded copy elsewhere (the folder holding output CreditICL)

Every item is PASS, WARN or FAIL with the evidence, and the last section turns the measured
speeds and sizes into estimates where the measured workload is comparable — GPU-hours, project storage, and whether
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
                       request["n_folds"], "roc_auc" if track == "pd" else "r2", slot.context,
                       row_counts=request.get(f"{domain}_rows"))
        seconds += float(rows["total_seconds"].sum())
    return seconds / 3600


def verify(out_root: Path, *, big_root: Path | None = None) -> Verdicts:
    modern = out_root / "experiment_0" / "report" / "validation.json"
    if modern.is_file() or modern.with_name("cpu_validation.json").is_file():
        return verify_validation(out_root, big_root=big_root)
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

    print("\n--- TRAINING COST ESTIMATES (only comparable short-route workloads) " + "-" * 20)
    _print_legacy_training_costs(speeds, _checkpoint_mb(exp0 / "report" / "checks.json"))
    for key, hours in sorted(costs.items()):
        print(f"  {key:<40} {sum(hours) / len(hours):5.2f} h per benchmark slot")
    return v


def _long_context_cost_notice(label: str) -> None:
    print(f"  {label}: GPU-hours unknown; the 1,024-row Exp0 route does not measure the "
          "400–60,000-row workload. Measure representative long-context throughput first; "
          "a single capacity-test update is not a reliable campaign estimate.")


def _print_legacy_training_costs(speeds: dict, sizes: dict) -> None:
    """Keep historical route estimates from becoming a false long-context campaign ETA."""
    # (experiment, arm kind in Exp0, steps, arms per track, kept checkpoints per arm)
    plan = [("Exp1", "scratch_tabicl", 12_500, 12, 5 + 2), ("Exp2", "finetune_tabicl", 2_500, 12, 5 + 2),
            ("Exp2", "finetune_tabpfn3", 2_500, 12, 5 + 2)]
    totals: dict[str, list[float]] = {}
    for exp, arm, steps, arms_per_track, kept in plan:
        for track in ("pd", "lgd"):
            if exp == "Exp2" and arm == "finetune_tabicl":
                _long_context_cost_notice(f"{exp} {track} {arm}")
                continue
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
        label = "TabPFN subtotal only" if exp == "Exp2" else "total"
        print(f"  {exp} {label}: ~{hours:,.0f} GPU-h of training, ~{gb:.0f} GB of checkpoints on project storage")
    print("  Exp2 total: unknown until representative TabICL long-context throughput is measured.")
    print("  Exp3 total: unknown; retention treatments replaced the former long scratch experiment.")


def _print_training_costs(exp: int, variant: str, track: str, runs: list[dict], summaries: dict) -> None:
    """Extrapolate compatible short routes, with an explicit gap for long-context TabICL."""
    if exp == 3:
        print(f"  Exp3{variant} {track}: {len(runs)} new runs; GPU-hours unknown. "
              "Retention-method throughput must be measured separately; scratch timings do not apply.")
        return
    if exp == 2:
        groups = [("TabICL", [r for r in runs if r["architecture"] != "tabpfn3"]),
                  ("TabPFN", [r for r in runs if r["architecture"] == "tabpfn3"])]
    else:
        groups = [("", runs)]
    unknown_long_context = False
    for backbone, group in groups:
        if not group:
            continue
        label = f"Exp{exp}{variant} {track}" + (f" {backbone}" if backbone else "")
        if exp == 2 and backbone == "TabICL":
            _long_context_cost_notice(label)
            unknown_long_context = True
            continue
        total, measured = 0.0, 0
        for run in group:
            arm = "finetune_tabpfn3" if exp == 2 else "scratch_tabicl"
            summary = summaries.get(arm, {})
            if summary.get("completed") and summary.get("steps", 0) > 0:
                total += summary["train_seconds_total"] / summary["steps"] * run["train"]["max_steps"] / 3600
                measured += 1
        if measured == len(group):
            qualifier = "; subtotal only" if exp == 2 else ""
            print(f"  {label}: {len(group)} runs, about {total:.1f} GPU-h (short-run extrapolation{qualifier})")
    if unknown_long_context:
        print(f"  Exp2 {track} total: unknown; the TabPFN subtotal excludes TabICL long-context training.")


def verify_validation(out_root: Path, *, big_root: Path | None = None) -> Verdicts:
    """Inspect downloaded CPU/GPU receipts without pretending to be the VSC environment."""
    from src.eval.benchmark_status import digest
    from src.utils import paths
    from src.utils.config import expand_with_seeds, load
    from src.utils.preflight import source_fingerprint
    from src.validation.campaign import SCHEMA, attempt_scope, plan

    v = Verdicts()
    receipt = out_root / "experiment_0" / "report" / "validation.json"
    cpu_file = receipt.with_name("cpu_validation.json")
    cpu_saved = None
    try:
        cpu_saved = json.loads(cpu_file.read_text(encoding="utf-8"))
        with attempt_scope(cpu_saved["attempt"]):
            cpu_expected = [s.name for s in plan("cpu")]
        cpu_rows = cpu_saved.get("stages", [])
        cpu_ok = (cpu_saved.get("schema") == SCHEMA and cpu_saved.get("status") == "PASS"
                  and [r["name"] for r in cpu_rows] == cpu_expected
                  and all(r["status"] == "PASS" for r in cpu_rows)
                  and bool(cpu_saved.get("artifacts")))
        v.add(PASS if cpu_ok else FAIL, "CPU Experiment 0", str(cpu_saved.get("status")))
        v.add(PASS if cpu_saved.get("source_fingerprint") == source_fingerprint() else FAIL,
              "CPU current code/config/tests")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        v.add(FAIL, "CPU Experiment 0", str(exc))
    try:
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        with attempt_scope(saved["attempt"]):
            expected = [s.name for s in plan()]
        rows = saved.get("stages", [])
        complete = (saved.get("schema") == SCHEMA and saved.get("status") == "PASS"
                    and [r["name"] for r in rows] == expected and all(r["status"] == "PASS" for r in rows))
        v.add(PASS if complete else FAIL, "GPU Experiment 0",
              f"Recorded {saved.get('status')}: {saved.get('attempt')}; {len(rows)}/{len(expected)} stages recorded")
        if saved.get("status") == "RUNNING":
            v.add(WARN, "GPU scheduler state unknown",
                  "RUNNING is the last saved report, not a live scheduler check. Check sacct for this attempt; "
                  "an END status=OK shell footer cannot replace a complete PASS receipt.")
        if saved.get("status") == "DEFERRED":
            v.add(WARN, "GPU continuation required", f"{len(rows)}/{len(expected)} stages saved; "
                  "resubmit the same GPU script without changing code, inputs, packages or CPU receipt")
        v.add(PASS if cpu_saved and saved.get("cpu_receipt") == digest(cpu_file) else FAIL,
              "GPU used this CPU validation")
        v.add(PASS if cpu_saved and all(saved.get(k) == cpu_saved.get(k)
                                       for k in ("inputs", "environment_fingerprint")) else FAIL,
              "CPU and GPU used the same inputs and packages")
        for row in rows:
            v.add(PASS if row["status"] == "PASS" else FAIL, row["name"],
                  row.get("error", f"{row.get('seconds', '?')} seconds"))
        if saved.get("error"):
            v.add(WARN if saved.get("status") == "DEFERRED" else FAIL, "validation stopped", saved["error"])
        v.add(PASS if saved.get("source_fingerprint") == source_fingerprint() else FAIL,
              "current code/config/tests", "source fingerprint comparison; package compatibility is checked on VSC")
        roots = [out_root] + ([big_root] if big_root else [])
        missing_weights = 0
        artifacts = {**(cpu_saved or {}).get("artifacts", {}), **saved.get("artifacts", {})}
        for remote, expected_hash in artifacts.items():
            relative = remote.replace("\\", "/").split(paths.OUTPUT_NAME + "/", 1)[-1]
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe artifact path: {remote}")
            candidates = [r / rel for r in roots]
            local = next((p for p in candidates if p.is_file()), None)
            if local is None and rel.suffix == ".ckpt":
                missing_weights += 1
                continue
            if local is None or digest(local) != expected_hash:
                v.add(FAIL, "artifact missing or changed", relative)
        if complete and not saved.get("artifacts"):
            v.add(FAIL, "evidence", "PASS without artifact hashes")
        if missing_weights:
            v.add(WARN, "checkpoints omitted from download", f"{missing_weights}; VSC preflight must verify them in place")

        attempt = out_root / "experiment_0" / "validation" / saved["attempt"]
        print("\nTraining cost projections (GPU-hours, not queue time or a billing statement):")
        for track in ("pd", "lgd"):
            summaries = {}
            for run in expand_with_seeds(load(f"config/Exp0_{track.upper()}.yaml")):
                file = attempt / "runs" / run["_run_name"] / "summary.json"
                if file.is_file():
                    summaries[run["arm"]] = json.loads(file.read_text(encoding="utf-8"))
            for exp, variant in ((1, ""), (1, "_ablations"), (2, ""), (3, "")):
                runs = expand_with_seeds(load(f"config/Exp{exp}_{track.upper()}{variant}.yaml", allow_placeholders=True))
                _print_training_costs(exp, variant, track, runs, summaries)
        return v
    except (OSError, ValueError, KeyError, TypeError) as exc:
        v.add(FAIL, "validation receipt", str(exc))
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
