"""Experiment 0 — every check that can run before a real arm does, ON THE CLUSTER.

    python scripts/exp0_checks.py                  # everything (a GPU job: scripts/slurm/exp0.slurm)
    python scripts/exp0_checks.py --only env paths # a subset
    python scripts/exp0_checks.py --quick          # skip the slow ones (training steps, benchmark fold)

Each check is isolated — one failing never stops the others — and reports PASS, WARN or FAIL
with the numbers behind it. The report goes to `output CreditICL/experiment_0/report/`
(`checks.json`, `checks.md`); the exit code is 1 if anything FAILED. Nothing here trains an arm,
writes a checkpoint an experiment reads, or touches a benchmark receipt.

THE CHECKS, in order:

    env           interpreter, torch/CUDA/GPU, library versions (tabicl, tabpfn, catboost, ...), git
    paths         every resolved location; the output tiers WRITABLE, the inputs READABLE
    storage       free space on both tiers, what each tree already holds, the VSC quota (`myquota`)
    data          every configured credit dataset processed and loadable (rows, columns, gaps, target)
    ood           the out-of-domain cache: manifest and enough datasets of each kind
    weights       the released TabICLv2 / TabPFN-3 files: found, loaded into the architectures we
                  train, the TabICL wrapper loads offline, and the file we fine-tune FROM is the
                  file the reference column scores
    prior         tasks from every prior path (control, credit, both encodings): valid, and how fast
    determinism   the task stream is the same with 0 or 2 workers, in a FRESH process (the hash seed
                  reaches every job), and arms with one seed share their base tasks — what makes
                  the arms comparable
    configs       every config expands; grid sizes and benchmark slots against the SLURM headers
    orchestration the SLURM tools the jobs call; `run_experiment` plans each experiment (dry run)
    train         a few real training steps of every training path on the GPU (speed, memory), then
                  the saved checkpoint through the BENCHMARK'S wrapper: loaded weights = trained ones
    benchmark     every dataset loads and supports full-row five-fold splitting on CPU;
                  selected whole-table GPU predictions are separate validation stages
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
    from src.utils.preflight import dependency_checks

    for result in dependency_checks():
        rep.add("env", result.name + " required version", PASS if result.ok else FAIL, result.detail)
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


def _tree(root: Path) -> tuple[int, int]:
    """(bytes, files) under `root`."""
    total = files = 0
    for f in root.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
                files += 1
            except OSError:
                pass
    return total, files


def check_storage(rep: Report) -> None:
    """`$VSC_DATA` is 75 GiB for every project together; project storage has a low inode budget.
    What each output tier already holds, and how much room is left, before the big runs add to it
    (`exp0_verify` projects what Experiments 1-3 will need from the measured checkpoint size)."""
    import shutil

    from src.utils import paths

    for label, root in (("small tier (logs, run records)", paths.outputs_dir()),
                        ("big tier (benchmarks, checkpoints)", paths.big_outputs_dir())):
        try:
            root.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(root)
            size, files = _tree(root)
            rep.add("storage", label, PASS, {"root": str(root), "tree_gb": round(size / 1e9, 3), "tree_files": files,
                                              "filesystem_free_gb": round(usage.free / 1e9, 1)})
        except OSError as exc:
            rep.add("storage", label, FAIL, f"{root}: {exc}")
    tool = shutil.which("myquota")
    if tool:
        try:
            out = subprocess.run([tool], capture_output=True, text=True, timeout=120).stdout
            rep.add("storage", "myquota", PASS, out.strip()[-1500:])
        except Exception as exc:  # noqa: BLE001
            rep.add("storage", "myquota", WARN, f"{type(exc).__name__}: {exc}")
    else:
        rep.add("storage", "myquota", WARN, "not on PATH — quota not read (not a VSC node?)")


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
                info["feature_budget"] = "all prediction-time eligible columns; no width cap"
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


def _batch_digest(prior: dict, track: str, *, workers: int, n_batches: int = 2, batch_size: int = 8) -> str:
    import hashlib

    from src.prior.dataset import build_loader

    loader = build_loader(prior, track, batch_size, seed=0, num_workers=workers)
    h = hashlib.sha256()
    batches = iter(loader)
    for _ in range(n_batches):
        for tensor in next(batches):
            h.update(tensor.numpy().tobytes())
    return h.hexdigest()[:16]


_FRESH_PROCESS = """
import copy, sys
sys.path.insert(0, {root!r})
from scripts.exp0_checks import _batch_digest
from src.utils.config import expand_with_seeds, load
prior = copy.deepcopy(expand_with_seeds(load({config!r}))[0]["prior"])
prior["credit_fraction"] = 0.5
print(_batch_digest(prior, {track!r}, workers=0))
"""


def check_determinism(rep: Report) -> None:
    """Arms are compared on COMMON RANDOM NUMBERS: the same seed, the same base tasks, in every job.
    That holds only if the stream ignores the worker count, repeats in a fresh interpreter (upstream
    orders feature groups by Python's string hash: `PYTHONHASHSEED` must reach every job and every
    worker) and gives the control's tasks to a mixed arm's control slots."""
    import copy

    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.prior.stream import plan_batch
    from src.utils.config import expand_with_seeds, load

    rep.add("determinism", "PYTHONHASHSEED", PASS if os.environ.get("PYTHONHASHSEED") == "0" else WARN,
            os.environ.get("PYTHONHASHSEED", "unset — the job scripts export 0"))
    for track in ("pd", "lgd"):
        config = str(ROOT / "config" / f"Exp1_{track.upper()}.yaml")
        prior = copy.deepcopy(expand_with_seeds(load(config))[0]["prior"])
        prior["credit_fraction"] = 0.5
        try:
            one, two = _batch_digest(prior, track, workers=0), _batch_digest(prior, track, workers=2)
            rep.add("determinism", f"{track}: 0 workers = 2 workers", PASS if one == two else FAIL,
                    {"0": one, "2": two})
            code = _FRESH_PROCESS.format(root=str(ROOT), config=config, track=track)
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, cwd=ROOT)
            fresh = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"failed: {r.stderr[-300:]}"
            rep.add("determinism", f"{track}: this process = a fresh one", PASS if fresh == one else FAIL,
                    {"here": one, "fresh": fresh})
            control = copy.deepcopy(prior)
            control["credit_fraction"] = 0.0
            g0 = TaskGenerator(control, track, PriorRNG(0), stream_seed=0)
            g5 = TaskGenerator(prior, track, PriorRNG(0), stream_seed=0)
            same = compared = 0
            for b in range(2):
                p0 = plan_batch(0, b, control, 8, regression=track == "lgd", credit_fraction=0.0)
                p5 = plan_batch(0, b, prior, 8, regression=track == "lgd", credit_fraction=0.5)
                for s0, s5 in zip(p0.slots, p5.slots):
                    if s5.use_credit:
                        continue
                    t0, t5 = g0.sample_slot(s0), g5.sample_slot(s5)
                    compared += 1
                    same += int(t0.X.shape == t5.X.shape and bool((t0.X == t5.X).all()) and bool((t0.y == t5.y).all()))
            rep.add("determinism", f"{track}: a 50 % arm's control slots hold the control's tasks",
                    PASS if compared and same == compared else FAIL, {"control_slots": compared, "identical": same})
        except Exception as exc:  # noqa: BLE001
            rep.add("determinism", track, FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}")


def check_orchestration(rep: Report) -> None:
    """The jobs resubmit themselves and `run_experiment` reads the queue: the SLURM tools must be
    there, and the planner must plan every experiment — Exp3 refusing until Exp1 chose its mix."""
    import shutil

    from src.utils.run_experiment import unconfigured_tracks

    tools = {t: bool(shutil.which(t)) for t in ("sbatch", "squeue", "sacct", "scontrol")}
    rep.add("orchestration", "SLURM tools on PATH", PASS if all(tools.values()) else WARN, tools)
    for exp in (1, 2, 3):
        try:
            want = int(bool(unconfigured_tracks(exp, ["pd", "lgd"])))
            r = subprocess.run([sys.executable, "-m", "src.utils.run_experiment", str(exp)], cwd=ROOT,
                               capture_output=True, text=True, timeout=900)
            ok = r.returncode == want and (not want or "NOT CONFIGURED YET" in r.stdout)
            rep.add("orchestration", f"run_experiment {exp} (plan only)", PASS if ok else FAIL,
                    {"exit": r.returncode, "expected": want, "tail": r.stdout.strip()[-700:]})
        except Exception as exc:  # noqa: BLE001
            rep.add("orchestration", f"run_experiment {exp} (plan only)", FAIL, f"{type(exc).__name__}: {exc}")


def check_configs(rep: Report) -> None:
    import re

    from src.eval.benchmark_status import n_slots
    from src.utils.config import expand_with_seeds, load

    header = {}
    for script in ("pretrain_pd", "pretrain_lgd", "benchmark"):
        text = (ROOT / "scripts" / "slurm" / f"{script}.slurm").read_text(encoding="utf-8")
        m = re.search(r"#SBATCH --array=0-(\d+)", text)
        header[script] = int(m.group(1)) + 1 if m else None
    for exp, variant in ((0, None), (1, None), (1, "ablations"), (2, None), (3, None)):
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
                if exp == 1 and variant is None:
                    detail["pretrain_header_array"] = header[f"pretrain_{track}"]
                    detail["benchmark_header_array"] = header["benchmark"]
                    if header[f"pretrain_{track}"] != len(runs) or header["benchmark"] != slots:
                        status = WARN
                rep.add("configs", name, status, detail)
            except Exception as exc:  # noqa: BLE001
                rep.add("configs", name, FAIL, f"{type(exc).__name__}: {exc}")


def check_train(rep: Report, steps: int = 10, *, track_only=None, arm_only=None) -> None:
    import copy
    import tempfile

    import torch

    from src.utils.config import expand_with_seeds, load

    workers = max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1)
    for track in ("pd", "lgd"):
        if track_only and track != track_only:
            continue
        runs = expand_with_seeds(load(ROOT / "config" / f"Exp0_{track.upper()}.yaml"))
        for run in runs:
            if arm_only and run["arm"] != arm_only:
                continue
            label = f"{track} {run['arm']}"
            cfg = copy.deepcopy(run)
            cfg["train"].update(max_steps=steps, num_workers=workers, save_temp_every=0, save_perm_every=steps,
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
                ckpt = tmp / "ckpt" / f"step-{steps}.ckpt"
                rep.add("train", label, PASS if summary.get("completed") else FAIL,
                        {"steps": summary.get("steps"), "seconds": round(seconds, 1),
                         "steps_per_s_incl_startup": round(steps / seconds, 3), "peak_gpu_gb": round(peak, 2),
                         "batch_size": cfg["train"]["batch_size"], "workers": workers,
                         "checkpoint_mb": round(ckpt.stat().st_size / 1e6, 1) if ckpt.is_file() else None})
                _checkpoint_round_trip(rep, track, run, trainer, ckpt)
            except Exception as exc:  # noqa: BLE001
                rep.add("train", label, FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
            finally:
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)


def _checkpoint_round_trip(rep: Report, track: str, run: dict, trainer: Any, ckpt: Path) -> None:
    """The file a run writes, scored the way the benchmark scores it: through the arm's own wrapper
    (`benchmark_status.arm_model`), and the weights that wrapper loaded must BE the trained ones."""
    import torch

    from src.eval.baselines import build
    from src.eval.benchmark_status import arm_model
    from src.eval.runner import load_arrays

    label = f"{track} {run['arm']}: checkpoint -> benchmark wrapper"
    if not ckpt.is_file():
        rep.add("train", label, FAIL, f"no {ckpt.name} written")
        return
    try:
        model_name = arm_model(run)
        if model_name == "crediticl":
            from src.eval.crediticl_baseline import register

            register()
            m = build("crediticl", track, seed=0, checkpoint=str(ckpt))
        else:
            m = build("tabpfn3", track, seed=0, model_path=str(ckpt), n_estimators=1)
        X, y, cats, _ = load_arrays(track, "0008.german" if track == "pd" else "0004.base_model", max_rows=400)
        m.fit(X[:300], y[:300], cats)
        pred = np.asarray(m.predict(X[300:]), dtype=float)
        est = m._model
        net = est.model_ if model_name == "crediticl" else est.models_[0]
        live, loaded = trainer.model.state_dict(), net.state_dict()
        shared = [k for k in live if k in loaded and live[k].shape == loaded[k].shape]
        diff = max((float((live[k].detach().float().cpu() - loaded[k].detach().float().cpu()).abs().max())
                    for k in shared), default=float("nan"))
        ok = bool(np.isfinite(pred).all()) and len(shared) == len(live) and diff == 0.0
        rep.add("train", label, PASS if ok else FAIL,
                {"wrapper": model_name, "tensors_compared": len(shared), "of": len(live), "max_abs_diff": diff,
                 "predictions_finite": bool(np.isfinite(pred).all())})
    except Exception as exc:  # noqa: BLE001
        rep.add("train", label, FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}")
    finally:
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def check_benchmark(rep: Report) -> None:
    """CPU validation of every table and every fold; real GPU scoring is a later stage."""
    from src.data.discovery import list_datasets
    from src.eval.protocol import cv_folds
    from src.eval.runner import load_arrays
    from src.eval.ood import list_ood_datasets
    from src.eval.ood_runner import _prepare, OODEvalConfig

    def check(label, task, X, y, groups=None):
        if X.shape[0] != len(y) or not np.isfinite(y).all() or np.isinf(X).any():
            raise ValueError("Misaligned rows or non-finite targets/infinite features")
        seen = np.zeros(len(y), dtype=np.uint8)
        folds = cv_folds(y, task, groups=groups)
        for train, test in folds:
            if len(train) + len(test) != len(y) or np.intersect1d(train, test).size:
                raise ValueError("Invalid train/test partition")
            if groups is not None and np.intersect1d(groups[train], groups[test]).size:
                raise ValueError("An entity appears on both sides of a fold")
            if task == "pd" and (len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2):
                raise ValueError("A classification fold is missing a class")
            seen[test] += 1
        if len(folds) != 5 or not np.all(seen == 1):
            raise ValueError("Five-fold CV did not test every row exactly once")
        rep.add("benchmark", label, PASS, {"rows": len(y), "folds": 5, "grouped": groups is not None})

    for track in ("pd", "lgd"):
        for dataset in list_datasets(track):
            try:
                X, y, _, info = load_arrays(track, dataset)
                check(f"{track}/{dataset}", track, X, y, info.pop("_cv_groups", None))
            except Exception as exc:
                rep.add("benchmark", f"{track}/{dataset}", FAIL, f"{type(exc).__name__}: {exc}")
    for entry in list_ood_datasets():
        try:
            X, y, _, _ = _prepare(entry, 0, OODEvalConfig())
            check(f"ood/{entry.name}", "pd" if entry.kind == "classification" else "lgd", X, y)
        except Exception as exc:
            rep.add("benchmark", f"ood/{entry.name}", FAIL, f"{type(exc).__name__}: {exc}")


CHECKS: dict[str, Callable[[Report], None]] = {
    "env": check_env, "paths": check_paths, "storage": check_storage, "data": check_data, "ood": check_ood,
    "weights": check_weights, "prior": check_prior, "determinism": check_determinism,
    "configs": check_configs, "orchestration": check_orchestration,
    "train": check_train, "benchmark": check_benchmark,
}
SLOW = ("determinism", "train", "benchmark")


def write(rep: Report, out_dir: Path) -> tuple[Path, Path]:
    from src.utils.preflight import source_fingerprint
    from src.utils.environment import environment_fingerprint, environment_snapshot

    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "checks.json"
    environment = environment_snapshot()
    js.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "failed": rep.failed,
                              "rows": rep.rows, "source_fingerprint": source_fingerprint(),
                              "environment": environment,
                              "environment_fingerprint": environment_fingerprint(environment)},
                             indent=2, default=str), encoding="utf-8")
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
    ap.add_argument("--track", choices=("pd", "lgd"), help="isolate one training track")
    ap.add_argument("--arm", choices=("scratch_tabicl", "finetune_tabicl", "finetune_tabpfn3"),
                    help="isolate one training route in its own CUDA process")
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
            if name == "train":
                fn(rep, track_only=args.track, arm_only=args.arm)
            else:
                fn(rep)
        except Exception as exc:  # noqa: BLE001 — a crashing check is a failed check
            rep.add(name, "crashed", FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
        print(f"          ({name}: {time.time() - started:.1f}s)", flush=True)
    js, md = write(rep, Path(args.out) if args.out else experiment_dir(0) / "report")
    print(f"\n{rep.failed} FAILED. Report: {md}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
