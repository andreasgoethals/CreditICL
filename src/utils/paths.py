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


def staging_override() -> Path | None:
    """An explicitly requested staging root, or None.

    Checked separately from `staging_root` because the big-file directories below
    otherwise short-circuit to the repo whenever we are off-VSC — which would make
    the override silently do nothing on a laptop. It has to work off-VSC: the tests
    point it at a temp dir, and it is the supported way to put pools on an external
    drive instead of inside the repo.
    """
    for var in ("CREDITICL_STAGING_ROOT", "CREDITPFN_STAGING_ROOT", "TABPFN_STAGING_ROOT"):
        p = _env_path(var)
        if p:
            return p
    return None


def staging_root() -> Path:
    """Project staging root — the big-file tier.

    Off-VSC (a laptop) there is no staging, so everything collapses into the repo
    and the tier distinction becomes a no-op. That keeps the same code path
    working locally without pretending `/lustre1/...` exists.
    """
    override = staging_override()
    if override:
        return override
    lustre = _env_path("VSC_PROJECT_LUSTRE1")
    if lustre:
        return lustre / "stg_00211"
    if on_vsc():
        return Path(STAGING_FALLBACK)
    return REPO_ROOT


def _use_staging() -> bool:
    """Should big files go to the staging tier rather than into the repo?

    True on VSC, and true anywhere the override is set.
    """
    return on_vsc() or staging_override() is not None


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
    """THE single root for everything the code produces. SMALL -> $VSC_DATA on VSC.

    Every generated artefact lives under here — results, figures, logs, manifests — so
    "what did this run produce?" and "what can I safely delete?" both have one answer.
    Locally it is `output/` in the repo; on the cluster `$VSC_DATA/CreditICL/output/`,
    which is the backed-up tier. Only genuinely large files (checkpoints, generated
    prior pools, processed datasets) go elsewhere, to project storage.
    """
    if on_vsc():
        return _under(data_root(), "output")
    return REPO_ROOT / "output"


def logs_dir() -> Path:
    """Timestamped run logs. SMALL -> inside the output tree on $VSC_DATA.

    Under `output/logs/`, not a separate top-level `logs/`. Everything a run writes that
    is small and durable now lives in one place, so "what did this run produce?" has a
    single answer and the cleanup helper has a single tree to walk.
    """
    return outputs_dir() / "logs"


def manifests_dir() -> Path:
    """Per-run CSV manifests — the training progress curves."""
    return outputs_dir() / "manifests"


def figures_dir(notebook: str | None = None) -> Path:
    """Generated figures. One folder per notebook, plus a shared CAPTIONS.md."""
    root = outputs_dir() / "figures"
    return root / notebook if notebook else root


def all_results_path() -> Path:
    """Every notebook's text summary, concatenated in notebook order."""
    return outputs_dir() / "All_Results.md"


def captions_path() -> Path:
    """The single shared captions file for every generated figure."""
    return figures_dir() / "CAPTIONS.md"


def repo_dir() -> Path:
    """Where the code is cloned on VSC: $VSC_DATA/CreditICL (so it is backed up)."""
    return _under(data_root())


# ---------------------------------------------------------------------------
# Raw and processed datasets
#
# Search order is always REPO FIRST, then project storage. That way a laptop with
# the datasets checked out locally works with no configuration, and on the
# cluster the same code finds them on staging. Writing goes the other way round:
# on the cluster, freshly processed data is written to staging so it never fills
# $VSC_DATA's 75 GiB.
# ---------------------------------------------------------------------------

TASKS = ("pd", "lgd")


def _check_task(task: str) -> str:
    task = task.lower()
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return task


def data_roots() -> list[Path]:
    """Every root that may hold a `raw/` or `processed/` tree, repo first."""
    roots = [REPO_ROOT / "data"]
    if _use_staging():
        staged = _under(staging_root(), "data")
        if staged not in roots:
            roots.append(staged)
    return roots


def raw_task_dirs(task: str) -> list[Path]:
    """All candidate `raw/<task>` directories, in search order."""
    task = _check_task(task)
    return [r / "raw" / task for r in data_roots()]


def processed_task_dirs(task: str) -> list[Path]:
    task = _check_task(task)
    return [r / "processed" / task for r in data_roots()]


RAW_EXTENSIONS = (".csv", ".parquet")


def raw_file_for(stem: Path, ext: str) -> Path:
    """`stem` + `ext`, APPENDED — never `Path.with_suffix`.

    Every dataset slug looks like `NNNN.name` ("0001.gmsc"), so pathlib reads
    `.gmsc` as the suffix and `with_suffix(".csv")` would REPLACE it, giving
    "0001.csv". In TabPFNCredit that silently made every lookup fail. Appending is
    the only correct operation here.
    """
    return stem.parent / (stem.name + ext)


def find_raw_path(task: str, dataset: str) -> Path | None:
    """Return the raw dataset path **stem** (no extension), or None.

    Returning a stem rather than the file is TabPFNCredit's contract, and
    `dataset_preprocessing.py` — copied from there — depends on it: it appends
    `.csv` / `.parquet` itself. Changing this to return the full filename gives
    `0006.lgd_freddie.csv.csv`. Use `find_raw_file` when you want the real file.
    """
    task = _check_task(task)
    for d in raw_task_dirs(task):
        stem = d / dataset
        if any(raw_file_for(stem, ext).is_file() for ext in RAW_EXTENSIONS):
            return stem
    return None


def find_raw_file(task: str, dataset: str) -> Path | None:
    """The actual raw file on disk, extension included. For our own code."""
    stem = find_raw_path(task, dataset)
    if stem is None:
        return None
    for ext in RAW_EXTENSIONS:
        candidate = raw_file_for(stem, ext)
        if candidate.is_file():
            return candidate
    return None


def find_processed_dir(task: str, dataset: str) -> Path | None:
    """Return an existing processed directory for this dataset, or None.

    A directory only counts as processed once its `meta.json` exists. That file is
    written LAST, so a run interrupted halfway through leaves an incomplete
    directory that is correctly treated as absent rather than silently reused.
    """
    task = _check_task(task)
    for d in processed_task_dirs(task):
        candidate = d / dataset
        if (candidate / "meta.json").is_file():
            return candidate
    return None


def processed_write_dir(task: str, dataset: str) -> Path:
    """Where to WRITE freshly processed data.

    On the cluster this is project storage, so processed datasets never eat into
    the backed-up 75 GiB of `$VSC_DATA`. Locally it is the repo's own
    `data/processed/`.
    """
    task = _check_task(task)
    if _use_staging():
        return _under(staging_root(), "data", "processed", task, dataset)
    return REPO_ROOT / "data" / "processed" / task / dataset


def prior_cache_dir(name: str) -> Path:
    """Where a pre-generated pool of synthetic datasets lives. BIG -> staging.

    Pools are the largest thing this project writes (40,000 datasets per variant),
    so `$CREDITICL_STAGING_ROOT` is honoured off-VSC too — otherwise they land
    inside the repo, which is exactly where they must not be.
    """
    if _use_staging():
        return _under(staging_root(), "prior_cache", name)
    return REPO_ROOT / "prior_cache" / name


# ---------------------------------------------------------------------------
# Results
#
# Layout: results/<task>/<pipeline>/ — task first, then pipeline.
# `logs/` is for information only and never holds results; that separation is
# deliberate so a results file is always something you meant to keep.
# ---------------------------------------------------------------------------

PIPELINES = ("data", "prior", "training", "eval")


#: Extra results namespaces that are not modelling tasks. `ood` holds the
#: out-of-domain (non-credit) scores, which must never be mixed into the credit results
#: — the two answer different questions and a mean across both is meaningless.
RESULT_NAMESPACES = TASKS + ("ood",)


def results_dir(task: str | None = None, pipeline: str | None = None) -> Path:
    """results/ , results/<task>/ or results/<task>/<pipeline>/.

    Accepts `ood` alongside the real tasks, so out-of-domain scores get their own tree
    rather than being written next to the credit numbers.
    """
    root = outputs_dir() / "results"
    if task is None:
        return root
    task = task.lower()
    if task not in RESULT_NAMESPACES:
        raise ValueError(f"results namespace must be one of {RESULT_NAMESPACES}, got {task!r}")
    if pipeline is None:
        return root / task
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline must be one of {PIPELINES}, got {pipeline!r}")
    return root / task / pipeline


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
