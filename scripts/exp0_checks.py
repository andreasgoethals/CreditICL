"""Experiment 0 — every check that can run before a real arm does, ON THE CLUSTER.

    python scripts/exp0_checks.py                  # everything (a GPU job: scripts/slurm/exp0.slurm)
    python scripts/exp0_checks.py --only env paths # a subset
    python scripts/exp0_checks.py --quick          # skip the slow ones (training steps, benchmark fold)

Each check is isolated — one failing never stops the others — and reports PASS, WARN or FAIL
with the numbers behind it. The report goes to `output_CreditICL/experiment_0/report/`
(`checks.json`, `checks.md`); the exit code is 1 if anything FAILED. Nothing here trains an arm,
writes a checkpoint an experiment reads, or touches a benchmark receipt.

THE CHECKS, in order:

    env        interpreter, torch/CUDA/GPU, library versions (tabicl, tabpfn, catboost, ...), git
    paths      every resolved location; the output tiers WRITABLE, the inputs READABLE
    data       every configured credit dataset processed and loadable (rows, columns, gaps, target)
    ood        the out-of-domain cache: manifest and enough datasets of each kind
    weights    the released TabICLv2 / TabPFN-3 files: found, loaded into the architectures we
               train, the TabICL wrapper loads offline, and the file we fine-tune FROM is the
               file the reference column scores
    prior      tasks from every prior path (control, credit, both encodings): valid, and how fast
    configs    every config expands; grid sizes and benchmark slots against the SLURM headers
    train      a few real training steps of both trainers on the GPU: speed and memory
    benchmark  one fold of one dataset through every reference wrapper, on the GPU
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, check: str, name: str, status: str, detail: Any = None) -> None:
        self.rows.append({"check": check, "name": name, "status": status, "detail": detail})
        print(f"[{status}] {check:<9} {name}" + (f"  {detail}" if detail not in (None, "", {}) else ""),
              flush=True)

    @property
    def failed(self) -> int:
        return sum(r["status"] == FAIL for r in self.rows)


def _version(module: str) -> str:
    try:
        mod = __import__(module)
        return str(getattr(mod, "__version__", "?"))
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__}: {exc})"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


# -- the checks ------------------------------------------------------------------------------


def check_env(rep: Report) -> None:
    import torch

    rep.add("env", "python", PASS, f"{sys.version.split()[0]} on {platform.platform()} ({platform.node()})")
    versions = {m: _version(m) for m in ("torch", "numpy", "pandas", "sklearn", "scipy", "tabicl",
                                         "tabpfn", "catboost", "psutil", "yaml")}
    missing = [m for m, v in versions.items() if v.startswith("MISSING") and m not in ("psutil",)]
    rep.add("env", "libraries", FAIL if missing else PASS, versions)
    dirty = bool(_git("status", "--porcelain"))
    rep.add("env", "git", WARN if dirty else PASS,
            {"commit": _git("rev-parse", "--short", "HEAD"), "dirty": dirty})
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rep.add("env", "gpu", PASS, {"name": props.name, "memory_gb": round(props.total_memory / 1024 ** 3, 1),
                                     "capability": f"{props.major}.{props.minor}",
                                     "count": torch.cuda.device_count(), "cuda": torch.version.cuda})
    else:
        rep.add("env", "gpu", FAIL, "no CUDA device — the experiments train on a GPU")
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in (
        "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CLUSTER_NAME", "SLURM_CPUS_PER_TASK", "SLURMD_NODENAME")}
    rep.add("env", "slurm", PASS if slurm else WARN, slurm or "not inside a SLURM job")
    from src.train.telemetry import gpu_stats, host_stats

    rep.add("env", "telemetry probes", PASS if gpu_stats() else WARN,
            {"nvidia-smi": bool(gpu_stats()), "psutil": bool(host_stats())})


def check_paths(rep: Report) -> None:
    from src.utils import paths

    rep.add("paths", "resolved", PASS, paths.describe())
    for label, root in (("small output tier", paths.outputs_dir()), ("big output tier", paths.big_outputs_dir())):
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / f".exp0_probe_{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            rep.add("paths", f"{label} writable", PASS, str(root))
        except OSError as exc:
            rep.add("paths", f"{label} writable", FAIL, f"{root}: {exc}")
    for label, root in (("released weights", paths.pretrained_dir()), ("datasets", paths.datasets_dir()),
                        ("ood cache", paths.ood_cache_dir())):
        ok = root.is_dir() and os.access(root, os.R_OK)
        rep.add("paths", f"{label} readable", PASS if ok else FAIL, str(root))


def check_data(rep: Report) -> None:
    from src.data.pipeline import load_processed
    from src.utils.config import load
    from src.utils.paths import find_processed_dir

    for track in ("pd", "lgd"):
        cfg = load(ROOT / "config" / f"Exp1_{track.upper()}.yaml")
        names = list(cfg["eval"]["dev_datasets"]) + list(cfg["eval"]["holdout_datasets"])
        for name in names:
            if find_processed_dir(track, name) is None:
                rep.add("data", f"{track}/{name}", FAIL, "not processed (python scripts/preprocess.py)")
                continue
            try:
                ds = load_processed(track, name)
                X, y = np.asarray(ds.X, dtype=float), np.asarray(ds.y, dtype=float)
                info = {"rows": int(X.shape[0]), "columns": int(X.shape[1]),
                        "missing_share": round(float(np.mean(~np.isfinite(X))), 4),
                        "categorical": len(ds.cat_indices)}
                if track == "pd":
                    info["minority_share"] = round(float(min(np.mean(y >= 0.5), np.mean(y < 0.5))), 4)
                else:
                    info["target_range"] = [round(float(np.nanmin(y)), 4), round(float(np.nanmax(y)), 4)]
                status = PASS
                if X.shape[1] > 500:
                    info["note"] = "wider than 500: the benchmark keeps the 500 top-variance columns"
                if not np.isfinite(y).all():
                    status, info["note"] = WARN, f"{int((~np.isfinite(y)).sum())} non-finite targets"
                rep.add("data", f"{track}/{name}", status, info)
            except Exception as exc:  # noqa: BLE001
                rep.add("data", f"{track}/{name}", FAIL, f"{type(exc).__name__}: {exc}")


def check_ood(rep: Report) -> None:
    from src.eval.ood import N_PER_TASK, ood_status

    st = ood_status()
    ok = st["exists"] and st["complete"]
    rep.add("ood", "cache", PASS if ok else FAIL,
            {"root": st["root"], "by_kind": st["by_kind"], "needed_per_kind": N_PER_TASK})


def check_weights(rep: Report) -> None:
    import torch

    from src.eval.baselines import find_local_tabpfn_checkpoint
    from src.models.architecture import build_model
    from src.train.adapt import load_pretrained
    from src.utils.paths import find_pretrained

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    for task, name in (("pd", "tabicl-classifier-v2-20260212.ckpt"), ("lgd", "tabicl-regressor-v2-20260212.ckpt")):
        try:
            path = find_pretrained(f"checkpoints/{name}")
            model = build_model(task, architecture="tabicl")
            report = load_pretrained(model, path)
            rep.add("weights", f"TabICLv2 {task} file -> our architecture", PASS, {"path": str(path), **report})
        except Exception as exc:  # noqa: BLE001
            rep.add("weights", f"TabICLv2 {task} file -> our architecture", FAIL, f"{type(exc).__name__}: {exc}")
            continue
        try:
            from tabicl import TabICLClassifier, TabICLRegressor

            y = (X[:, 0] > 0).astype(int) if task == "pd" else np.clip(0.5 + 0.2 * X[:, 0], 0, 1)
            # The wrapper's OWN resolution, forbidden to download: what a compute node can do.
            est = (TabICLClassifier if task == "pd" else TabICLRegressor)(
                device=device, random_state=0, allow_auto_download=False)
            est.fit(X[:200], y[:200])
            est.predict(X[200:])
            ref = {k: v for k, v in est.model_.state_dict().items()}
            ours = torch.load(path, map_location="cpu", weights_only=False)
            ours = ours.get("state_dict", ours)
            shared = [k for k in ref if k in ours]
            diff = max((float((ref[k].float().cpu() - ours[k].float()).abs().max()) for k in shared), default=float("nan"))
            same = len(shared) == len(ref) and diff == 0.0
            rep.add("weights", f"TabICLv2 {task}: wrapper's own copy offline = our file",
                    PASS if same else FAIL, {"tensors_compared": len(shared), "of": len(ref), "max_abs_diff": diff})
        except Exception as exc:  # noqa: BLE001
            # Not fatal for the benchmark, which points the wrapper at our file; still worth knowing.
            rep.add("weights", f"TabICLv2 {task}: wrapper's own copy offline", WARN,
                    f"{type(exc).__name__}: {exc} (the benchmark uses our local file instead)")
        try:
            from src.eval.baselines import build

            y = (X[:, 0] > 0).astype(float) if task == "pd" else np.clip(0.5 + 0.2 * X[:, 0], 0, 1)
            m = build("tabiclv2", task, seed=0)
            m.fit(X[:200], y[:200], [])
            m.predict(X[200:])
            rep.add("weights", f"TabICLv2 {task}: the reference wrapper", PASS, dict(m.report.extra))
        except Exception as exc:  # noqa: BLE001
            rep.add("weights", f"TabICLv2 {task}: the reference wrapper", FAIL, f"{type(exc).__name__}: {exc}")
    for task, which in (("pd", "classifier"), ("lgd", "regressor")):
        path = find_local_tabpfn_checkpoint(which)
        if path is None:
            rep.add("weights", f"TabPFN-3 {which}", FAIL, "no local checkpoint")
            continue
        try:
            from src.train.tabpfn_trainer import build_estimator

            est = build_estimator(task, path, n_estimators=2, device=device, seed=0)
            n = sum(p.numel() for p in est.models_[0].parameters())
            rep.add("weights", f"TabPFN-3 {which} -> training estimator", PASS, {"path": str(path), "parameters": n})
        except Exception as exc:  # noqa: BLE001
            rep.add("weights", f"TabPFN-3 {which} -> training estimator", FAIL, f"{type(exc).__name__}: {exc}")


def check_prior(rep: Report, n_tasks: int = 24) -> None:
    import copy

    import torch

    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.utils.config import expand_with_seeds, load

    for track in ("pd", "lgd"):
        base = expand_with_seeds(load(ROOT / "config" / f"Exp1_{track.upper()}.yaml"))[0]["prior"]
        for fraction in (0.0, 1.0):
            for encoding in ("tabicl", "raw"):
                if fraction == 0.0 and encoding == "raw":
                    continue  # the control does not depend on the encoding
                prior = copy.deepcopy(base)
                prior.update(credit_fraction=fraction, encoding=encoding)
                label = f"{track} credit_fraction={fraction:g} encoding={encoding}"
                try:
                    gen = TaskGenerator(prior, track, PriorRNG(0))
                    started = time.time()
                    stats: dict[str, list[float]] = {"minority": [], "at0": [], "at1": [], "nan": [], "d": []}
                    for i in range(n_tasks):
                        task = gen.sample()
                        Xd = task.X[:, : int(task.meta["d"])]
                        if torch.isinf(Xd).any() or (encoding == "tabicl" and torch.isnan(Xd).any()):
                            raise ValueError(f"task {i}: non-finite features")
                        stats["nan"].append(float(torch.isnan(Xd).float().mean()) if Xd.numel() else 0.0)
                        stats["d"].append(int(task.meta["d"]))
                        if track == "pd" and fraction > 0:
                            m = float(task.y.float().mean())
                            stats["minority"].append(min(m, 1 - m))
                        if track == "lgd" and fraction > 0:
                            stats["at0"].append(float(task.meta.get("frac_at_0", np.nan)))
                            stats["at1"].append(float(task.meta.get("frac_at_1", np.nan)))
                    rate = n_tasks / (time.time() - started)
                    summary = {"tasks_per_s_one_process": round(rate, 2),
                               "median_columns": float(np.median(stats["d"])),
                               "missing_share": round(float(np.mean(stats["nan"])), 4)}
                    for key in ("minority", "at0", "at1"):
                        if stats[key]:
                            summary[f"{key}_median"] = round(float(np.nanmedian(stats[key])), 4)
                    rep.add("prior", label, PASS, summary)
                except Exception as exc:  # noqa: BLE001
                    rep.add("prior", label, FAIL, f"{type(exc).__name__}: {exc}")


def check_configs(rep: Report) -> None:
    import re

    from src.eval.benchmark_status import n_slots
    from src.utils.config import expand_with_seeds, load

    header = {}
    for script in ("pretrain_pd", "pretrain_lgd", "benchmark"):
        text = (ROOT / "scripts" / "slurm" / f"{script}.slurm").read_text(encoding="utf-8")
        m = re.search(r"#SBATCH --array=0-(\d+)", text)
        header[script] = int(m.group(1)) + 1 if m else None
    for exp, variant in ((0, None), (1, None), (2, "search"), (2, None), (3, None)):
        for track in ("pd", "lgd"):
            name = f"Exp{exp}_{track.upper()}{'_' + variant if variant else ''}"
            path = ROOT / "config" / f"{name}.yaml"
            if not path.is_file():
                rep.add("configs", name, WARN, "no such config")
                continue
            try:
                runs = expand_with_seeds(load(path, allow_placeholders=True))
                slots = n_slots(exp, track, variant=variant)
                detail = {"arms": len(runs), "benchmark_slots": slots}
                status = PASS
                if exp == 1:
                    detail["pretrain_header_array"] = header[f"pretrain_{track}"]
                    detail["benchmark_header_array"] = header["benchmark"]
                    if header[f"pretrain_{track}"] != len(runs) or header["benchmark"] != slots:
                        status = WARN
                rep.add("configs", name, status, detail)
            except Exception as exc:  # noqa: BLE001
                rep.add("configs", name, FAIL, f"{type(exc).__name__}: {exc}")


def check_train(rep: Report, steps: int = 10) -> None:
    import copy
    import tempfile

    import torch

    from src.utils.config import expand_with_seeds, load

    workers = max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1)
    for track in ("pd", "lgd"):
        runs = expand_with_seeds(load(ROOT / "config" / f"Exp0_{track.upper()}.yaml"))
        for run in runs:
            label = f"{track} {run['arm']}"
            cfg = copy.deepcopy(run)
            cfg["train"].update(max_steps=steps, num_workers=workers, save_temp_every=0, save_perm_every=0,
                                log_every=max(1, steps // 2))
            cfg["progress"]["every_datasets"] = 0
            cfg["logging"].update(to_file=False, console=False, log_hardware_every=max(1, steps // 2))
            tmp = Path(tempfile.mkdtemp(prefix="exp0_train_"))
            try:
                if cfg["architecture"] == "tabpfn3":
                    from src.train.tabpfn_trainer import TabPFNTrainer as Trainer
                else:
                    from src.train.loop import Trainer
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                trainer = Trainer(cfg, tmp, ckpt_dir=tmp / "ckpt")
                trainer.maybe_resume()
                started = time.time()
                summary = trainer.train()
                seconds = time.time() - started
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
                rep.add("train", label, PASS if summary.get("completed") else FAIL,
                        {"steps": summary.get("steps"), "seconds": round(seconds, 1),
                         "steps_per_s_incl_startup": round(steps / seconds, 3), "peak_gpu_gb": round(peak, 2),
                         "batch_size": cfg["train"]["batch_size"], "workers": workers})
            except Exception as exc:  # noqa: BLE001
                rep.add("train", label, FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
            finally:
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)


def check_benchmark(rep: Report) -> None:
    from src.eval.benchmark_status import REFERENCE_MODELS
    from src.eval.protocol import cv_folds
    from src.eval.runner import load_arrays, score_fold

    for track, dataset in (("pd", "0008.german"), ("lgd", "0004.base_model")):
        try:
            X, y, cats, _ = load_arrays(track, dataset)
            tr, te = cv_folds(y, track)[0]
        except Exception as exc:  # noqa: BLE001
            rep.add("benchmark", f"{track}/{dataset}", FAIL, f"cannot load: {exc}")
            continue
        for model in REFERENCE_MODELS.split(","):
            try:
                started = time.time()
                out = score_fold(track, model, X[tr], y[tr], X[te], y[te], cats, seed=0, fold=0)
                key = "roc_auc" if track == "pd" else "r2"
                rep.add("benchmark", f"{track}/{dataset} {model}", PASS,
                        {key: round(float(out[key]), 4), "seconds": round(time.time() - started, 1)})
            except Exception as exc:  # noqa: BLE001
                rep.add("benchmark", f"{track}/{dataset} {model}", FAIL, f"{type(exc).__name__}: {exc}")


CHECKS: dict[str, Callable[[Report], None]] = {
    "env": check_env, "paths": check_paths, "data": check_data, "ood": check_ood,
    "weights": check_weights, "prior": check_prior, "configs": check_configs,
    "train": check_train, "benchmark": check_benchmark,
}
SLOW = ("train", "benchmark")


def write(rep: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "checks.json"
    js.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "failed": rep.failed,
                              "rows": rep.rows}, indent=2, default=str), encoding="utf-8")
    lines = ["# Experiment 0 — checks", "", f"Written {time.strftime('%d-%m-%Y %H:%M')}; "
             f"{sum(r['status'] == PASS for r in rep.rows)} PASS, {sum(r['status'] == WARN for r in rep.rows)} WARN, "
             f"{rep.failed} FAIL.", "", "| check | name | status | detail |", "|---|---|---|---|"]
    for r in rep.rows:
        detail = json.dumps(r["detail"], default=str) if isinstance(r["detail"], (dict, list)) else str(r["detail"] or "")
        lines.append(f"| {r['check']} | {r['name']} | {r['status']} | {detail.replace('|', '/')[:400]} |")
    md = out_dir / "checks.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return js, md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(CHECKS), default=None)
    ap.add_argument("--quick", action="store_true", help="skip the slow checks (train, benchmark)")
    ap.add_argument("--out", default=None, help="report folder; default experiment_0/report/")
    args = ap.parse_args(argv)
    from src.utils.paths import experiment_dir

    rep = Report()
    for name, fn in CHECKS.items():
        if args.only and name not in args.only:
            continue
        if args.quick and name in SLOW:
            continue
        started = time.time()
        try:
            fn(rep)
        except Exception as exc:  # noqa: BLE001 — a crashing check is a failed check
            rep.add(name, "crashed", FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
        print(f"          ({name}: {time.time() - started:.1f}s)", flush=True)
    js, md = write(rep, Path(args.out) if args.out else experiment_dir(0) / "report")
    print(f"\n{rep.failed} FAILED. Report: {md}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
