"""Read-only checks before requesting a GPU: python -m src.utils.preflight.

No training, model loading, downloading or dataset preprocessing. Safe on a login
node. A git pull does not update an installed package or a processed data cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from dataclasses import dataclass
from pathlib import Path

TABPFN_VERSION = "9.1.0"
TABICL_COMMIT = "8f1aa2098c894ab91dba15209cf2002ba4be6c6c"
CHECK_SECTIONS = frozenset({"env", "paths", "storage", "data", "ood", "weights", "prior", "determinism",
                            "configs", "orchestration", "train", "benchmark"})


def source_fingerprint() -> str:
    """Relevant executable code/configs; documentation edits do not invalidate a check."""
    from src.utils.paths import REPO_ROOT

    files = [REPO_ROOT / "pyproject.toml"]
    for directory in ("src/data", "src/eval", "src/experiments", "src/models", "src/prior", "src/train", "src/utils", "src/validation", "src/visualize",
                      "config", "scripts", "tests"):
        files.extend(p for p in (REPO_ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".slurm", ".sh"))
    files.extend((REPO_ROOT / "notebooks").rglob("*.ipynb"))
    return _fingerprint(files)


def training_fingerprint() -> str:
    """Version the training implementation separately from benchmark/config edits."""
    return training_fingerprint_for(training_implementation_fingerprint())


def training_implementation_fingerprint() -> str:
    """Training source only; dependency versions are added by training_fingerprint_for."""
    from src.utils.paths import REPO_ROOT

    files = [REPO_ROOT / "pyproject.toml"]
    for directory in ("src/models", "src/prior", "src/train"):
        files.extend((REPO_ROOT / directory).rglob("*.py"))
    return _fingerprint(files)


def training_fingerprint_for(implementation: str) -> str:
    """Bind a reviewed source digest to the CURRENT training dependency versions.

    This also lets completed-run compatibility reconstruct a historical digest
    without accepting a change to the packages that produced those weights.
    Checkpoint resume always calls training_fingerprint(), for current source.
    """
    # Resume is an update-rule contract, including the installed upstream code.
    versions = {}
    for name in ("torch", "tabicl", "tabpfn", "numpy", "scipy", "scikit-learn"):
        try:
            dist = metadata.distribution(name)
            origin = json.loads(dist.read_text("direct_url.json") or "{}")
            versions[name] = {"version": dist.version,
                              "commit": origin.get("vcs_info", {}).get("commit_id")}
        except metadata.PackageNotFoundError:
            versions[name] = None
    return hashlib.sha256((implementation + json.dumps(versions, sort_keys=True)).encode()).hexdigest()


def _fingerprint(files: list[Path]) -> str:
    from src.utils.paths import REPO_ROOT

    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        # Git converts line endings between Windows and Linux.
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            # Executing notebooks changes outputs/metadata, never the tested cell sources.
            nb = json.loads(text)
            text = json.dumps([(c["cell_type"], "".join(c["source"])) for c in nb["cells"]], ensure_ascii=False)
        digest.update(text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def dependency_checks() -> list[Check]:
    """Check package metadata without importing torch or initializing a GPU."""
    rows = []
    for name in ("tabpfn", "tabicl", "catboost"):
        try:
            dist = metadata.distribution(name)
            if name == "tabpfn":
                ok = dist.version == TABPFN_VERSION
                detail = f"installed {dist.version}; required {TABPFN_VERSION}"
            elif name == "tabicl":
                origin = json.loads(dist.read_text("direct_url.json") or "{}")
                commit = origin.get("vcs_info", {}).get("commit_id")
                ok = commit == TABICL_COMMIT
                detail = f"installed {dist.version}, commit {commit}; required {TABICL_COMMIT}"
            else:
                ok, detail = True, f"installed {dist.version}"
            rows.append(Check(name, ok, detail))
        except (metadata.PackageNotFoundError, OSError, ValueError) as exc:
            rows.append(Check(name, False, str(exc)))
    from src.utils.environment import pip_consistency, project_metadata, requirement_audit
    from src.utils.paths import on_vsc

    try:
        versions, issues = requirement_audit()
        rows.append(Check("all project requirements", not issues,
                          "; ".join(issues) if issues else f"{len(versions)} dependency versions checked, including extras"))
    except (OSError, ValueError) as exc:
        rows.append(Check("all project requirements", False, str(exc)))
    ok, detail = pip_consistency()
    rows.append(Check("pip check (whole environment)", ok, detail))
    if on_vsc():
        wanted = project_metadata()["tool"]["crediticl"]["vsc"]["torch-version"]
        try:
            installed = metadata.version("torch")
            rows.append(Check("VSC CUDA torch", installed == wanted,
                              f"installed {installed}; required {wanted}"))
        except metadata.PackageNotFoundError as exc:
            rows.append(Check("VSC CUDA torch", False, str(exc)))
    return rows


def require_tabpfn_version() -> None:
    """Fail before initializing a training estimator on incompatible internals."""
    try:
        installed = metadata.version("tabpfn")
    except metadata.PackageNotFoundError:
        installed = "not installed"
    if installed != TABPFN_VERSION:
        raise RuntimeError(f"TabPFN training requires tabpfn=={TABPFN_VERSION}; found {installed}. "
                           "A git pull does not update the virtual environment. On the login node, "
                           "activate the project environment and run: python -m pip install -e '.[dev,eval]'")


def check_cache(path: Path | None, version: int) -> tuple[bool, str]:
    if path is None:
        return False, "missing processed cache"
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        actual = meta.get("cache_version")
        if actual != version:
            return False, f"cache_version={actual}; required {version}: {path}"
        parquet = path / "data.parquet"
        if not parquet.is_file() or parquet.stat().st_size == 0:
            return False, f"missing/empty data.parquet: {path}"
        return True, f"version {actual}, {meta.get('n_rows', '?')} rows: {path}"
    except (OSError, ValueError) as exc:
        return False, f"{path}: {exc}"


def run_checks(*, dependencies_only: bool = False) -> list[Check]:
    rows = dependency_checks()
    if dependencies_only:
        return rows
    from src.data.pipeline import CACHE_VERSION
    from src.eval.baselines import TABICL_RELEASED, find_local_tabpfn_checkpoint
    from src.eval.ood import KINDS, N_PER_TASK, list_ood_datasets, ood_root
    from src.utils import paths
    from src.utils.config import load

    for task in ("pd", "lgd"):
        cfg = load(paths.REPO_ROOT / "config" / f"Exp1_{task.upper()}.yaml")
        for name in dict.fromkeys(cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"]):
            ok, detail = check_cache(paths.find_processed_dir(task, name), CACHE_VERSION)
            rows.append(Check(f"data {task}/{name}", ok, detail))
        try:
            p = paths.find_pretrained(TABICL_RELEASED[task])
            rows.append(Check(f"TabICLv2 {task} weights", p.is_file() and p.stat().st_size > 0, str(p)))
        except FileNotFoundError as exc:
            rows.append(Check(f"TabICLv2 {task} weights", False, str(exc)))
        p = find_local_tabpfn_checkpoint("classifier" if task == "pd" else "regressor")
        rows.append(Check(f"TabPFN-3 {task} weights", p is not None and p.stat().st_size > 0,
                          str(p) if p is not None else "missing local released weights"))
    try:
        entries = list_ood_datasets()
        root = ood_root()
        for kind in KINDS:
            subset = [e for e in entries if e.kind == kind]
            missing = [e.slug for e in subset if not (root / f"{e.slug}.npz").is_file()]
            rows.append(Check(f"OOD {kind}", len(subset) >= N_PER_TASK and not missing,
                              f"{len(subset)} datasets, {len(missing)} missing files: {root}"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        rows.append(Check("OOD cache", False, str(exc)))
    return rows


def experiment_zero_checks(*, include_runs: bool = True) -> list[Check]:
    """The paid experiment launcher requires completed validation for this code."""
    from src.validation.campaign import readiness, receipt_path

    if include_runs or receipt_path().is_file():
        return readiness()
    from src.utils import paths
    from src.utils.environment import environment_changes, environment_fingerprint, environment_snapshot

    rows = []
    root = paths.experiment_dir(0)
    try:
        report = json.loads((root / "report" / "checks.json").read_text(encoding="utf-8"))
        checks = report["rows"]
        ok = (CHECK_SECTIONS <= {r["check"] for r in checks} and not any(r["status"] == "FAIL" for r in checks)
              and report.get("source_fingerprint") == source_fingerprint())
        rows.append(Check("Exp0 full checks for current code", ok,
                          "complete, successful, matching report required"))
        environment = environment_snapshot()
        matching = report.get("environment_fingerprint") == environment_fingerprint(environment)
        changes = environment_changes(report.get("environment", {}), environment)
        rows.append(Check("Exp0 validated package environment", matching,
                          "matches successful GPU checks" if matching else
                          "Installed packages changed or no validated inventory exists: " + "; ".join(changes)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        rows.append(Check("Exp0 full checks for current code", False, str(exc)))
    logs = list((root / "logs").glob("exp0_checks_*.log"))
    latest = max(logs, key=lambda p: p.name) if logs else None
    log = latest.read_text(encoding="utf-8", errors="replace") if latest else ""
    ok = "END status=OK" in log and all(f"equivalence {t} exit code: 0" in log for t in ("PD", "LGD"))
    rows.append(Check("Exp0 latest GPU job and both equivalence checks", ok, str(latest)))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies-only", action="store_true")
    parser.add_argument("--require-exp0", action="store_true", help="also gate the paid experiment arrays on Exp0")
    parser.add_argument("--require-exp0-checks", action="store_true",
                        help="require the checks job before running Exp0's six training arms")
    parser.add_argument("--require-exp0-cpu", action="store_true", help="require current successful CPU validation")
    args = parser.parse_args(argv)
    rows = run_checks(dependencies_only=args.dependencies_only)
    if args.require_exp0_cpu:
        from src.validation.campaign import readiness
        rows.extend(readiness("cpu"))
    if args.require_exp0 or args.require_exp0_checks:
        rows.extend(experiment_zero_checks(include_runs=args.require_exp0))
    for row in rows:
        print(f"[{'PASS' if row.ok else 'FAIL'}] {row.name}: {row.detail}")
    failed = [r for r in rows if not r.ok]
    if failed:
        print(f"\nSTOP: {len(failed)} prerequisite(s) failed. Do not launch Experiments 1-3. "
              "Follow the failed checks above before submitting or continuing validation.")
        if any(r.name in ("tabpfn", "tabicl", "catboost", "all project requirements",
                          "pip check (whole environment)", "VSC CUDA torch") for r in failed):
            print("Review the PROJECT SIF migration in docs/VSC.md before changing packages. "
                  "For explicitly authorized legacy repair while all consumers are idle: "
                  "bash scripts/slurm/update_env.sh --legacy-maintenance")
        if any(r.name.startswith("data ") for r in failed):
            print("Rebuild stale data on a CPU node: sbatch --clusters=wice scripts/slurm/preprocess.slurm")
        if any(r.name.startswith("OOD") for r in failed):
            print("Inspect the existing OOD cache: python -m src.utils.fetch_ood --status. "
                  "Fetch only if missing, on the login node: python -m src.utils.fetch_ood")
        if any(r.name.startswith(("Exp0", "Experiment 0")) for r in failed):
            print("Finish Experiment 0 with the current code before launching Experiments 1-3.")
        return 1
    print("\nPrerequisites passed. This does not replace Experiment 0's GPU validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
