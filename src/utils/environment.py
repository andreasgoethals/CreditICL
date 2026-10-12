"""Inspect dependency metadata without importing models or contacting package servers."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


def project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def requirement_audit(project: dict | None = None) -> tuple[dict[str, str], list[str]]:
    """Check the declared runtime/dev/eval dependencies, including nested extras.

    pip check separately audits the whole installed environment, but does not
    activate extras such as TabICL's pretraining dependencies. Check those here.
    """
    project = project_metadata()["project"] if project is None else project
    issues = []
    python = platform.python_version()
    if python not in SpecifierSet(project.get("requires-python", "")):
        issues.append(f"Python {python}; required {project['requires-python']}")
    roots = list(project.get("dependencies", []))
    for extra in ("dev", "eval"):
        roots.extend(project.get("optional-dependencies", {}).get(extra, []))
    queue = [(project.get("name", "crediticl"), raw) for raw in roots]
    versions, expanded, checked = {}, set(), set()
    while queue:
        parent, raw = queue.pop(0)
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        if str(req) in checked:
            continue
        checked.add(str(req))
        name = canonicalize_name(req.name)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            issues.append(f"{parent} requires {req}; {name} is missing")
            continue
        versions[name] = dist.version
        if dist.version not in req.specifier:
            issues.append(f"{parent} requires {req}; installed {dist.version}")
        if req.url and req.url.startswith("git+"):
            origin = json.loads(dist.read_text("direct_url.json") or "{}")
            wanted_url, separator, revision = req.url.removeprefix("git+").rpartition("@")
            actual = origin.get("vcs_info", {}).get("commit_id")
            if not separator or actual != revision or origin.get("url") != wanted_url:
                issues.append(f"{name} source differs: required {req.url}, installed commit {actual}")
        key = name, tuple(sorted(req.extras))
        if key in expanded:
            continue
        expanded.add(key)
        package_meta = getattr(dist, "metadata", {})
        python_spec = package_meta.get("Requires-Python", "")
        if python not in SpecifierSet(python_spec):
            issues.append(f"{name} {dist.version} requires Python {python_spec}; running {python}")
        for child_raw in getattr(dist, "requires", None) or []:
            child = Requirement(child_raw)
            if child.marker and not any(child.marker.evaluate({"extra": extra})
                                        for extra in {"", *req.extras}):
                continue
            # The marker has already been evaluated in its parent's extra context.
            child.marker = None
            queue.append((name, str(child)))
    return dict(sorted(versions.items())), sorted(set(issues))


def pip_consistency() -> tuple[bool, str]:
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                capture_output=True, text=True, timeout=60)
        return result.returncode == 0, (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def environment_snapshot() -> dict:
    """Record every installed distribution, without credentials or machine paths."""
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        packages[canonicalize_name(name)] = {
            "version": dist.version,
            "commit": origin.get("vcs_info", {}).get("commit_id"),
        }
    return {"python": platform.python_version(), "platform": sys.platform,
            "packages": dict(sorted(packages.items()))}


def environment_fingerprint(snapshot: dict | None = None) -> str:
    snapshot = environment_snapshot() if snapshot is None else snapshot
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def environment_changes(expected: dict, actual: dict) -> list[str]:
    changes = [f"{key}: {expected.get(key)} -> {actual.get(key)}" for key in ("python", "platform")
               if expected.get(key) != actual.get(key)]
    before, after = expected.get("packages", {}), actual.get("packages", {})
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) != after.get(name):
            changes.append(f"{name}: {before.get(name)} -> {after.get(name)}")
    return changes
