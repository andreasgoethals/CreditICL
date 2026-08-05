"""Where files go on VSC — two storage tiers, one resolver.

Same split CreditPFN uses, for the same reasons:

| tier | path | what goes here | backed up | quota |
|---|---|---|---|---|
| **project staging** | `/lustre1/project/stg_00211` (`$VSC_PROJECT_LUSTRE1/stg_00211`) | the **big files**: datasets, trained checkpoints, result CSVs | no | large (>=1 TB), low inode budget |
| **personal data** | `$VSC_DATA` | the repo, plus **small durable outputs**: logs, metrics.jsonl, manifests, figures | **yes** | 75 GiB, tight |
| scratch | `$VSC_SCRATCH` | working scratch only | no | 500 GiB, **purged after 30 days of no access** |

Rules of thumb:

* Datasets and checkpoints are the largest artefacts, so they go to staging.
  `$VSC_DATA` at 75 GiB cannot hold them, and scratch is purged.
* Staging has a **low inode budget** — few big files, not thousands of small
  ones. So per-step metrics go to `$VSC_DATA`, not staging.
* Staging is on **Lustre**, so it is reachable from Genius and wICE but Mindwell
  jobs must use their own GPFS scratch for heavy I/O. Read a checkpoint from
  staging once at job start; do not stream from it.

Staging is resolved in this order, so it can be overridden without editing code:
``$CREDITICL_STAGING_ROOT`` -> ``$CREDITPFN_STAGING_ROOT`` (shared lab default)
-> ``$VSC_PROJECT_LUSTRE1/stg_00211`` -> the literal ``/lustre1/project/stg_00211``.

`resolve_writable` exists because staging permissions have failed mid-run before
(CreditPFN hit this on 2026-07-03 and lost a run's checkpoints). It probes with a
real write and falls back to `$VSC_DATA` with a loud warning, because a completed
run in the wrong place beats a crashed one.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_NAME = "CreditICL"
STAGING_FALLBACK = "/lustre1/project/stg_00211"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def on_vsc() -> bool:
    """True when running on a VSC node (the env vars are set by the site)."""
    return bool(os.environ.get("VSC_DATA"))


def staging_root() -> Path:
    """Project staging root — the big-file tier.

    Off-VSC (a laptop) there is no staging, so everything collapses into the repo
    and the tier distinction becomes a no-op. That keeps the same code path
    working locally without pretending `/lustre1/...` exists.
    """
    for var in ("CREDITICL_STAGING_ROOT", "CREDITPFN_STAGING_ROOT", "TABPFN_STAGING_ROOT"):
        p = _env_path(var)
        if p:
            return p
    lustre = _env_path("VSC_PROJECT_LUSTRE1")
    if lustre:
        return lustre / "stg_00211"
    if on_vsc():
        return Path(STAGING_FALLBACK)
    return REPO_ROOT


def data_root() -> Path:
    """Personal data root — the small, backed-up tier."""
    p = _env_path("VSC_DATA")
    return p if p else REPO_ROOT


def scratch_root() -> Path:
    p = _env_path("VSC_SCRATCH")
    return p if p else REPO_ROOT


def _under(root: Path, *parts: str) -> Path:
    """Join under `root`, inserting the project name only when `root` is shared.

    On VSC, `$VSC_DATA` and staging are shared across projects, so paths need a
    `CreditICL/` component. The repo root already *is* the project, so adding it
    there would give `CreditICL/CreditICL/output`.
    """
    if root == REPO_ROOT:
        return root.joinpath(*parts)
    return root.joinpath(PROJECT_NAME, *parts)


# -- the four things this project writes -------------------------------------


def datasets_dir() -> Path:
    """Raw + processed credit datasets. BIG -> staging."""
    return _under(staging_root(), "data")


def checkpoints_dir() -> Path:
    """Pretrained weights, one dir per run. BIG -> staging."""
    return _under(staging_root(), "checkpoints")


def outputs_dir() -> Path:
    """Metrics, logs, manifests, resolved configs. SMALL -> $VSC_DATA."""
    return _under(data_root(), "output") if on_vsc() else REPO_ROOT / "res"


def logs_dir() -> Path:
    return _under(data_root(), "logs")


def repo_dir() -> Path:
    """Where the code is cloned on VSC: $VSC_DATA/CreditICL (so it is backed up)."""
    return _under(data_root())


def resolve_writable(preferred: Path, fallback: Path | None = None) -> Path:
    """Return `preferred` if we can actually write there, else `fallback`.

    Probes with a real file create+delete. `mkdir` alone is not enough — a
    directory can exist and still be unwritable by this user, which is exactly
    the failure mode this guards against.
    """
    fallback = fallback or (data_root() / PROJECT_NAME / "fallback")
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return preferred
    except OSError as exc:
        print(
            f"WARNING: cannot write to {preferred} ({exc}).\n"
            f"         Falling back to {fallback}. Copy results to staging afterwards, "
            f"or $VSC_DATA will fill up (75 GiB quota).",
            flush=True,
        )
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def touch_tree(path: Path) -> None:
    """Update access times so `$VSC_SCRATCH`'s 30-day purge does not eat the files.

    `mv` and timestamp-preserving `rsync` do NOT count as an access, so freshly
    staged data can be deleted almost immediately. Copy, then call this.
    """
    for p in path.rglob("*"):
        if p.is_file():
            p.touch()


def describe() -> dict[str, str]:
    """For logging at job start, so a run records where it actually wrote."""
    return {
        "on_vsc": str(on_vsc()),
        "staging_root": str(staging_root()),
        "data_root": str(data_root()),
        "datasets_dir": str(datasets_dir()),
        "checkpoints_dir": str(checkpoints_dir()),
        "outputs_dir": str(outputs_dir()),
    }
