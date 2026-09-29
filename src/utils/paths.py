"""Where files go on VSC — two storage tiers, one resolver.

Same split CreditPFN uses, for the same reasons:

| tier | path | what goes here | backed up | quota |
|---|---|---|---|---|
| **project staging** | `/lustre1/project/stg_00211` (`$VSC_PROJECT_LUSTRE1/stg_00211`) | the **big files**: datasets, released weights, and the big half of `output_CreditICL/` (our checkpoints, prior pools) | no | large (>=1 TB), low inode budget |
| **personal data** | `$VSC_DATA` | the repo, plus the small half of `output_CreditICL/`: logs, run records, benchmark CSVs, figures | **yes** | 75 GiB, tight |
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
import re
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


#: Checked in order; the first one set wins. The two lab-wide names are accepted so a shell
#: already configured for a sibling project works here unchanged. A module constant rather
#: than a literal inside the function because the test fixture has to set the same variable
#: this reads — two copies of that name would drift.
STAGING_ENV_VARS = ("CREDITICL_STAGING_ROOT", "CREDITPFN_STAGING_ROOT", "TABPFN_STAGING_ROOT")


def staging_override() -> Path | None:
    """An explicitly requested staging root, or None.

    Checked separately from `staging_root` because the big-file directories below
    otherwise short-circuit to the repo whenever we are off-VSC — which would make
    the override silently do nothing on a laptop. It has to work off-VSC: the tests
    point it at a temp dir, and it is the supported way to put pools on an external
    drive instead of inside the repo.
    """
    for var in STAGING_ENV_VARS:
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


# -- the output tree -----------------------------------------------------------
#
#   output_CreditICL/                      SMALL tier: the repo locally, $VSC_DATA/CreditICL on VSC
#       README.md                          what lives where (committed)
#       All_Results.md, CAPTIONS.md        every notebook's text summary / figure captions
#       general/                           chapter 0 figures; logs of jobs that belong to no
#                                          experiment (preprocess, OOD fetch, prior pools)
#       reference/                         the models whose numbers do not depend on our prior
#           benchmark/{pd,lgd,ood,receipts}  (released TabICLv2 / TabPFN-3, CatBoost, linear):
#           logs/                            scored ONCE, read by every experiment
#       experiment_<N>/                    N = 0 (the cluster debug suite), 1, 2, 3
#           logs/                          one file per SLURM job (training, benchmark, checks)
#           runs/<run name>/               one folder per trained arm: config.json, summary.json,
#                                          progress.csv, telemetry.csv, weights.csv, logs/
#           benchmark/{pd,lgd,ood,receipts}  one CSV per scored checkpoint, one receipt each
#           figures/<notebook>/            the PDFs of that experiment's notebooks
#
#   output_CreditICL/                      BIG tier: <staging>/CreditICL on VSC, the SAME tree
#       experiment_<N>/checkpoints/<run name>/step-*.ckpt
#       prior_cache/                       pre-generated prior pools (optional)
#
# Locally both tiers are the same folder. The released weights (TabICLv2, TabPFN-3) are not
# output — they are a download — and stay in `pretrained_dir()`.

OUTPUT_NAME = "output_CreditICL"
EXPERIMENTS = (0, 1, 2, 3)
GENERAL = "general"
REFERENCE = "reference"
#: The four places a benchmark writes, per owner. `ood` never mixes with the credit results:
#: a mean across both answers no question.
BENCHMARK_PARTS = ("pd", "lgd", "ood", "receipts")
#: What one run folder holds, by kind. Fixed names, so a reader needs no run-name globbing.
RUN_FILES = {
    "config": "config.json",
    "summary": "summary.json",
    "progress": "progress.csv",
    "telemetry": "telemetry.csv",
    "weights": "weights.csv",
}


def datasets_dir() -> Path:
    """Raw + processed credit datasets. BIG -> staging."""
    return _under(staging_root(), "data")


def pretrained_dir(*parts: str) -> Path:
    """The RELEASED weights (TabICLv2, TabPFN-3), a HuggingFace download. BIG -> staging.

    Not output: never cleaned, never written by a run. Our own checkpoints go to
    `run_checkpoints_dir`, inside the experiment they belong to.
    """
    return _under(staging_root(), "checkpoints", *parts)


def find_pretrained(path: str | Path) -> Path:
    """The released-weights file a config names, wherever this machine keeps it.

    Configs name the weights repo-relative (`checkpoints/tabicl-classifier-v2-20260212.ckpt`), which
    is where a laptop has them; the cluster keeps them in `pretrained_dir()` on project storage. An
    absolute path is taken as given. Raises rather than falling back to anything: fine-tuning from
    the wrong file would be invisible in every number afterwards.
    """
    p = Path(path)
    candidates = [p] if p.is_absolute() else [REPO_ROOT / p, pretrained_dir() / p.name, pretrained_dir() / p]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"released weights {str(path)!r} not found; looked in: "
                            + ", ".join(str(c) for c in candidates))


def outputs_dir() -> Path:
    """THE root of everything the code produces that is small. `$VSC_DATA` on VSC.

    Named `output_CreditICL` so a downloaded copy says which project it came from. Locally it
    sits in the repo; on the cluster under `$VSC_DATA/CreditICL/`, the backed-up tier.
    """
    if on_vsc():
        return _under(data_root(), OUTPUT_NAME)
    return REPO_ROOT / OUTPUT_NAME


def big_outputs_dir() -> Path:
    """The same tree on project storage, for the files too big for `$VSC_DATA`.

    Only checkpoints and prior pools live here. Off-VSC, without a staging override, it IS
    `outputs_dir()`: one folder, same layout.
    """
    if _use_staging():
        return _under(staging_root(), OUTPUT_NAME)
    return outputs_dir()


def _check_experiment(exp: int | str) -> int:
    n = int(exp)
    if n not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {exp!r}")
    return n


def experiment_dir(exp: int | str, *, big: bool = False) -> Path:
    """`output_CreditICL/experiment_<N>/` on the small tier, or its mirror on the big one."""
    root = big_outputs_dir() if big else outputs_dir()
    return root / f"experiment_{_check_experiment(exp)}"


def general_dir() -> Path:
    """Chapter 0 and the jobs that belong to no experiment."""
    return outputs_dir() / GENERAL


def reference_dir() -> Path:
    """The reference column: models whose numbers do not depend on our prior."""
    return outputs_dir() / REFERENCE


_EXPERIMENT_NAME = re.compile(r"^exp(\d+)[a-z]*_")  # exp1_, exp1bench_, exp2searchbench_


def experiment_of(name: str) -> int:
    """The experiment a run name or benchmark tag belongs to.

    `exp1_pd__...__s0` and `exp1bench_pd_a3` are Exp1, `exp2searchbench_pd_a0` Exp2 (its search);
    `debug_...` and `exp0_...` are the
    debug suite, Exp0. Anything else RAISES: a checkpoint filed under the wrong experiment
    is scored against the wrong grid, and a guess would hide that.
    """
    if name.startswith("debug_"):
        return 0
    m = _EXPERIMENT_NAME.match(name)
    if not m:
        raise ValueError(f"cannot tell which experiment {name!r} belongs to (want exp<N>_...)")
    return _check_experiment(m.group(1))


def owner_of(tag: str) -> int | str:
    """Who owns a benchmark tag: an experiment number, `reference`, or `general` (ad hoc)."""
    if tag.startswith(f"{REFERENCE}_"):
        return REFERENCE
    try:
        return experiment_of(tag)
    except ValueError:
        return GENERAL


def owner_dir(owner: int | str) -> Path:
    """The folder of an experiment number, or of `general` / `reference`."""
    if owner in (GENERAL, REFERENCE):
        return outputs_dir() / str(owner)
    return experiment_dir(owner)


def logs_dir(owner: int | str = GENERAL) -> Path:
    """Job logs of one experiment (or of `reference` / `general`). SMALL."""
    return owner_dir(owner) / "logs"


def runs_dir(exp: int | str) -> Path:
    """Every trained arm of one experiment, one folder each."""
    return experiment_dir(exp) / "runs"


def run_dir(run_name: str) -> Path:
    """One trained arm's folder: its config, summary, curves and training logs."""
    return runs_dir(experiment_of(run_name)) / run_name


def run_file(run_name: str, kind: str) -> Path:
    """`run_dir(run_name) / RUN_FILES[kind]` — `config`, `summary`, `progress`, ..."""
    if kind not in RUN_FILES:
        raise ValueError(f"run file kind must be one of {sorted(RUN_FILES)}, got {kind!r}")
    return run_dir(run_name) / RUN_FILES[kind]


def run_summary_path(run_name: str) -> Path:
    """A finished arm's summary. The authority on whether the arm completed."""
    return run_file(run_name, "summary")


def run_config_path(run_name: str) -> Path:
    """A run's resolved config, exactly as trained."""
    return run_file(run_name, "config")


def run_checkpoints_dir(run_name: str) -> Path:
    """OUR checkpoints of one arm. BIG: project storage, inside the arm's experiment."""
    return experiment_dir(experiment_of(run_name), big=True) / "checkpoints" / run_name


def benchmark_dir(owner: int | str, part: str | None = None) -> Path:
    """`<owner>/benchmark/` or one of its parts: `pd`, `lgd`, `ood`, `receipts`.

    The part is checked: a typo would silently create a new folder and the scores would go
    missing instead of raising.
    """
    root = owner_dir(owner) / "benchmark"
    if part is None:
        return root
    if part not in BENCHMARK_PARTS:
        raise ValueError(f"benchmark part must be one of {BENCHMARK_PARTS}, got {part!r}")
    return root / part


def benchmark_dir_for(tag: str, part: str) -> Path:
    """Where the benchmark files of one tag go — the tag names its owner."""
    return benchmark_dir(owner_of(tag), part)


def chapter_of(notebook: str) -> int | None:
    """`1.3_pd_results` -> 1. None for a name without a chapter number."""
    m = re.match(r"^(\d+)\.", notebook)
    return int(m.group(1)) if m else None


def figures_dir(notebook: str) -> Path:
    """One notebook's figures, inside the chapter it belongs to.

    Chapter 0 (and a name without a chapter) goes to `general/figures/`; chapter N to
    `experiment_N/figures/`.
    """
    chapter = chapter_of(notebook)
    owner = GENERAL if not chapter else chapter
    return owner_dir(owner) / "figures" / notebook


def all_figure_dirs() -> list[Path]:
    """Every per-notebook figure folder that exists, in notebook order."""
    roots = [general_dir()] + [experiment_dir(n) for n in EXPERIMENTS]
    found = [d for r in roots if (r / "figures").is_dir()
             for d in (r / "figures").iterdir() if d.is_dir()]
    return sorted(found, key=lambda d: d.name)


def all_results_path() -> Path:
    """Every notebook's text summary, concatenated in notebook order."""
    return outputs_dir() / "All_Results.md"


def captions_path() -> Path:
    """The single shared captions file for every generated figure."""
    return outputs_dir() / "CAPTIONS.md"


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


def raw_dir(*parts: str) -> Path:
    """`data/raw/` — never modified, never committed, never deleted by the cleaner.

    The plain root. The task-aware search helpers below (`find_raw_path` and friends) are
    what this project actually calls; this exists because it is the template's name for
    the concept and because `clean_run` and any new code should not have to know about
    `<task>/<dataset>` layout to name the folder they must not touch.
    """
    if _use_staging():
        return _under(staging_root(), "data", "raw", *parts)
    return REPO_ROOT.joinpath("data", "raw", *parts)


def processed_dir(*parts: str) -> Path:
    """`data/processed/` — a cache, rebuildable from `raw/`, so the first thing to delete."""
    if _use_staging():
        return _under(staging_root(), "data", "processed", *parts)
    return REPO_ROOT.joinpath("data", "processed", *parts)


def config_path(name: str) -> Path:
    """`config/<name>.yaml`. Always in the repo — configs are code, not data.

    Takes the name with or without the extension, so a caller can pass either what the
    user typed on the command line or a bare experiment name.
    """
    stem = name[:-5] if name.endswith(".yaml") else name
    return REPO_ROOT / "config" / f"{stem}.yaml"


def notebooks_dir() -> Path:
    """Where the notebooks live. `run_notebooks` discovers them here rather than from a
    hard-coded list, so a notebook someone adds is covered automatically."""
    return REPO_ROOT / "notebooks"


def library_dir() -> Path:
    """The read-only literature submodule. READ from it; never write inside it."""
    return REPO_ROOT / "tfm-library"


def ensure(path: Path) -> Path:
    """`mkdir -p` the directory and return the path unchanged.

    Given a file path (anything with a suffix) it creates the *parent*, so a caller can
    write `ensure(run_file(name, "summary")).write_text(...)` in one line.
    """
    target = path.parent if path.suffix else path
    target.mkdir(parents=True, exist_ok=True)
    return path


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


def prior_cache_root() -> Path:
    """The parent of every pre-generated prior pool. BIG -> the big output tree.

    Separate from `prior_cache_dir` so the cleaner can name the whole tree without
    inventing a pool name.
    """
    return big_outputs_dir() / "prior_cache"


def prior_cache_dir(name: str) -> Path:
    """Where one pre-generated pool of synthetic datasets lives. BIG.

    Pools are the largest thing this project can write (40,000 datasets per variant), so
    `$CREDITICL_STAGING_ROOT` is honoured off-VSC too — otherwise they land inside the repo.
    """
    return prior_cache_root() / name


def ood_cache_dir() -> Path:
    """The downloaded out-of-domain (OpenML) tables. INPUT data, beside the credit datasets.

    Never cleaned: compute nodes have no outbound internet, so it can only be rebuilt from a
    login node with `python -m src.utils.fetch_ood`.
    """
    return datasets_dir() / "ood"


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
        "pretrained_dir": str(pretrained_dir()),
        "outputs_dir": str(outputs_dir()),
        "big_outputs_dir": str(big_outputs_dir()),
    }
