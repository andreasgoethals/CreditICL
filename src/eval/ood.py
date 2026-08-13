"""Out-of-domain evaluation: does a credit-tailored prior cost us elsewhere?

THE QUESTION THIS ANSWERS

We deliberately bend the pretraining prior toward credit risk. The obvious risk is that
we buy credit performance by *losing* general performance — a prior that only knows
about collateral and default thresholds might make a worse general-purpose model. If
that happened and we only ever measured credit datasets, we would never see it, and the
contribution would be much weaker than it looked.

So: score every checkpoint on tasks that have nothing to do with credit, and report the
change. Three outcomes, all publishable:

* **no degradation** — the credit gain is free, which is the strongest result;
* **mild degradation** — a specialisation trade-off, which is the expected and still
  interesting one, and quantifies the price;
* **severe degradation** — the prior is destructive rather than specialising, which is
  a negative result worth reporting honestly.

WHICH BENCHMARKS, AND WHY THESE

Several suites, because a mean over one suite is one suite's opinion:

* **OpenML-CC18** — the suite **O'Prior itself evaluated on** (arXiv 2605.18971), the
  closest prior work, so our control stays directly comparable with theirs.
* **TabArena** and **TALENT** — the two benchmarks **TabICLv2 itself reports on**, so our
  numbers can be put beside the model we started from. TabArena carries BOTH task kinds.
* **OpenML-CTR23** — the curated regression suite, needed because the LGD arm is a
  regression model. Its published alias 404s on the live API, so its numeric study id is
  listed as well.

**THE TASK DECIDES ITS OWN KIND, NOT THE SUITE.** Quotas are per kind (`KINDS`) and filled
across every suite. The first version bucketed suites by kind, which silently coded
`diamonds`' price into thousands of classes — see `task_kind`.

A suite that will not resolve is **skipped with a warning**, not fatal: these aliases are
not guaranteed to exist as OpenML studies and a login node's API access is not guaranteed
either, so a partially fetched cache beats losing the whole download.

**DATASET IDS ARE NEVER HARD-CODED HERE.** They are resolved from the suite through the
OpenML API at fetch time and then pinned into `ood_manifest.json`. Writing IDs from
memory is how you end up silently evaluating on the wrong tables — the suite *names* are
the verifiable thing, the IDs are not.

NO INTERNET ON COMPUTE NODES. Fetching must run on a **login node**
(`python -m src.utils.fetch_ood`), which caches to project storage. `load_ood_dataset`
never reaches the network; it raises with the fix if the cache is missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger
from src.utils.paths import prior_cache_dir

#: Suites resolved by NAME through the OpenML API — never by hard-coded DATASET id.
#: A FLAT list, and the task's OWN type decides which bucket it lands in.
#:
#: THIS USED TO BE `{kind: [suites]}` AND IT CORRUPTED DATA. TabArena contains regression
#: AND classification tasks; listing it under "classification" meant every task it returned was
#: labelled classification, so `diamonds` (price), `houses`, `airfoil_self_noise` and
#: `miami_housing` had their CONTINUOUS targets run through `.astype("category").cat.codes` —
#: thousands of arbitrary integer classes. The cache looked full and was meaningless.
#:
#: Suite membership says nothing about task type. Only the task does, so only the task is asked.
SUITES = (
    # O'Prior evaluated on CC18, so keeping it keeps our control comparable with the closest
    # prior work. Classification only.
    "OpenML-CC18",
    # TabArena and TALENT are what TabICLv2 itself reports on. TabArena carries both kinds.
    "tabarena-v0.1",
    # Regression. `OpenML-CTR23` is the published alias and it 404s on the live API
    # ("Study does not exist"), so the numeric study id is listed too — a suite id is a
    # verifiable identifier, unlike a dataset id written from memory.
    "OpenML-CTR23",
    "353",
    # Names seen in the literature; skipped with a warning when they do not resolve.
    "talent-classification",
    "talent-regression",
)

#: Both kinds are needed: LGD is regression and PD is classification, so a cache with only
#: one kind leaves half the project with no out-of-domain evidence at all.
KINDS = ("classification", "regression")

#: Per SUITE, not per task. The out-of-domain average is the number that catches a prior which
#: buys credit performance by destroying generality, so it wants to be a real average — and
#: 25 x 3 suites is ~75 per task, against the 10 it used to be.
#:
#: The ceiling is inference cost, not principle: this runs at every `progress.every_datasets`
#: during training as well as in the final pass, so it must stay cheap enough that the
#: diagnostic does not dominate the run it is diagnosing. `progress.n_ood` samples a handful
#: mid-training; the full set is used for the end-of-run report.
N_PER_SUITE = 25

#: Kept as the old name so nothing that imported it breaks; it now means "per suite".
N_PER_TASK = N_PER_SUITE

#: Anything matching these is NOT out-of-domain and must be dropped. CC18 contains
#: `credit-g` and `credit-approval`, and O'Prior specifically singles out Credit-g as a
#: dataset where its gains vanish — including it here would quietly contaminate the
#: control with the very domain we are specialising toward.
CREDIT_PATTERNS = [
    r"credit", r"default", r"loan", r"lending", r"bank", r"mortgage",
    r"insurance", r"risk", r"financ",
    # Credit scoring under a name that does not say so.
    r"gmsc", r"hmeq", r"heloc", r"fico", r"scorecard", r"delinquen", r"repay",
    r"borrower", r"debt", r"arrears", r"collater", r"good.?customer", r"churn",
]


def our_dataset_names() -> frozenset[str]:
    """The names of the datasets THIS PROJECT evaluates on, lower-cased.

    DERIVED, not hand-listed. A keyword list missed `heloc`, `loss2`, `axa`, `base_model`,
    `base_modelisation` and `hackerearth` — six of our own tables would have passed as
    out-of-domain, and `heloc` actually did on the first real fetch. Reading the names off disk
    means adding a dataset to `data/raw/` cannot silently reopen the hole.

    Returns an empty set when the data is not on this machine, in which case the keyword
    patterns are all the protection there is — so the fetch logs that.
    """
    names: set[str] = set()
    try:
        from src.data.discovery import list_datasets
    except Exception:  # noqa: BLE001
        return frozenset()
    for task in ("lgd", "pd"):
        try:
            for slug in list_datasets(task):
                # `0001.heloc` -> `heloc`
                names.add(slug.split(".", 1)[-1].strip().lower())
        except Exception:  # noqa: BLE001
            continue
    return frozenset(n for n in names if n)

#: Bump when the on-disk cache layout changes OR when a cache written by an older version is
#: WRONG rather than merely old. `load_manifest` treats a version mismatch as an empty cache,
#: so bumping is how a bad fetch is retired.
#:
#: 2: the first real fetch (13-08-2026) cached 50 datasets with `kind` taken from the suite
#:    list instead of the task, so continuous targets (diamonds, houses, airfoil_self_noise,
#:    miami_housing) were coded into thousands of arbitrary classes — and `heloc`, one of our
#:    own LGD datasets, was cached as out-of-domain. Neither is repairable in place.
OOD_VERSION = 2


def is_credit_like(name: str) -> bool:
    """True if a dataset must NOT be treated as out-of-domain.

    Two independent reasons to exclude, and both are needed:

    1. **It is one of ours.** Matched against `our_dataset_names()`, read off `data/raw/`. A
       dataset we select priors on cannot also be the evidence that generality survived.
    2. **It looks like credit.** Keyword patterns, for OpenML tables that are credit-adjacent
       without being ours.

    Deliberately over-inclusive: dropping a borderline dataset costs one out-of-domain task,
    whereas keeping a credit-adjacent one silently weakens the entire out-of-domain claim.
    """
    low = name.strip().lower()
    if low in our_dataset_names():
        return True
    return any(re.search(p, low) for p in CREDIT_PATTERNS)


def task_kind(task: Any) -> str | None:
    """`"classification"`, `"regression"`, or None — asked of the TASK, never inferred.

    THIS IS THE FIX FOR A DATA-CORRUPTION BUG. Suites were listed under a kind, so every task
    from a suite inherited it. TabArena carries both kinds, so listing it under
    "classification" meant `diamonds` (price), `houses`, `airfoil_self_noise` and
    `miami_housing` had their continuous targets pushed through
    `.astype("category").cat.codes` — thousands of arbitrary integer classes. The cache filled
    up and the numbers meant nothing.

    OpenML exposes the type as a `TaskType` enum on newer clients and as a plain string on
    older ones, so both are handled rather than assuming a version.
    """
    raw = getattr(task, "task_type", None)
    text = str(getattr(raw, "name", raw) or "").lower()
    if not text:
        # `task_type_id` is the numeric fallback: 1 = supervised classification,
        # 2 = supervised regression, per OpenML's own task-type table.
        type_id = getattr(task, "task_type_id", None)
        numeric = getattr(type_id, "value", type_id)
        return {1: "classification", 2: "regression"}.get(numeric)
    if "classification" in text:
        return "classification"
    if "regression" in text:
        return "regression"
    return None


def pd_to_float(series: Any) -> np.ndarray | None:
    """A regression target as float32, or None if it simply is not numeric.

    A categorical target reaching the regression branch means the task type and the data
    disagree; returning None skips the dataset rather than coercing a label into a number.
    """
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(values).any():
        return None
    return values


def ood_root() -> Path:
    """Where the cached OOD tables live. BIG-ish -> the same tier as prior pools."""
    return prior_cache_dir("ood").parent / "ood"


def manifest_path() -> Path:
    return ood_root() / "ood_manifest.json"


@dataclass
class OODDataset:
    """One cached out-of-domain table."""

    name: str
    openml_id: int
    kind: str  # "classification" | "regression"
    n_rows: int
    n_features: int
    n_classes: int | None = None

    @property
    def slug(self) -> str:
        # OpenML names contain spaces, dots and slashes; a filesystem-safe slug keeps
        # the id so two datasets with the same name can never collide.
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", self.name).strip("_").lower()
        return f"{self.openml_id}.{safe}"


def load_manifest() -> dict[str, Any]:
    """The pinned dataset list, or an empty manifest."""
    p = manifest_path()
    if not p.is_file():
        return {"version": OOD_VERSION, "datasets": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("version") != OOD_VERSION:
        return {"version": OOD_VERSION, "datasets": []}
    return data


def list_ood_datasets(kind: str | None = None) -> list[OODDataset]:
    """The cached OOD datasets, optionally filtered to one kind."""
    entries = [OODDataset(**d) for d in load_manifest()["datasets"]]
    return [d for d in entries if kind is None or d.kind == kind]


def fetch_ood_datasets(
    n_per_task: int = N_PER_TASK,
    *,
    force: bool = False,
    max_rows: int = 50_000,
    max_features: int = 500,
) -> list[OODDataset]:
    """Download and cache the OOD suites. **Run on a login node only.**

    Resolves each suite through the OpenML API, drops anything credit-like or too
    large, takes the first `n_per_task` by task id (deterministic, not random, so the
    selection is reproducible without a seed), and writes one `.npz` per dataset plus a
    manifest.

    Size caps exist because the in-context models have practical row and feature
    ceilings; a 500k-row OOD table would be capped at evaluation time anyway and would
    cost a long download for data we then throw away.
    """
    log = get_logger()
    try:
        import openml
    except ImportError as exc:
        raise ImportError(
            "out-of-domain evaluation needs the OpenML client:\n"
            "    pip install -e '.[eval]'\n"
            "It is only needed to FETCH the datasets, on a login node. Scoring reads "
            "the local cache and does not import openml."
        ) from exc

    root = ood_root()
    root.mkdir(parents=True, exist_ok=True)
    existing = {d.openml_id: d for d in list_ood_datasets()} if not force else {}
    kept: list[OODDataset] = []

    unavailable: list[str] = []
    # Quota PER KIND, filled across every suite. Both kinds have to be covered or half the
    # project has no out-of-domain evidence, and no single suite covers both reliably.
    chosen_by_kind: dict[str, int] = dict.fromkeys(KINDS, 0)
    for suite_name in SUITES:
        if all(chosen_by_kind[k] >= n_per_task for k in KINDS):
            log.info("[ood] every kind is full — stopping before %s", suite_name)
            break
        log.info("[ood] resolving suite %s (have %s)", suite_name, dict(chosen_by_kind))
        try:
            suite = openml.study.get_suite(suite_name)
        except Exception as exc:  # noqa: BLE001
                # A suite alias that OpenML does not host must not cost the whole download.
                # TabArena and TALENT are named in TabICLv2's paper but are not guaranteed to
                # exist as OpenML studies under these aliases, and a login node's API access is
                # not guaranteed either. Skip loudly and keep the suites that do resolve.
            log.warning("[ood] suite %s UNAVAILABLE (%s: %s) — skipping",
                        suite_name, type(exc).__name__, exc)
            unavailable.append(suite_name)
            continue
        task_ids = list(suite.tasks or [])
        log.info("[ood] %s advertises %d tasks", suite_name, len(task_ids))

        added_here = 0
        for task_id in sorted(task_ids):
            if all(chosen_by_kind[k] >= n_per_task for k in KINDS):
                break
            # Log EVERY attempt, not just successes. The first cluster run logged
            # "advertises 72 tasks" and then went silent for minutes while downloading,
            # which is indistinguishable from a hang. Downloads are the slow part, so
            # the log has to show it is making progress.
            log.info("[ood] %s [%s] checking task %s ...", suite_name,
                     " ".join(f"{k[:3]}={chosen_by_kind[k]}/{n_per_task}" for k in KINDS),
                     task_id)
            try:
                task = openml.tasks.get_task(task_id, download_data=False)
                ds = openml.datasets.get_dataset(task.dataset_id, download_data=False)
            except Exception as exc:  # noqa: BLE001 — one bad task must not stop the sweep
                log.warning("[ood] task %s unavailable: %s", task_id, exc)
                continue

            # THE KIND COMES FROM THE TASK. Deriving it from which suite list the name sat in
            # silently turned `diamonds`' price into thousands of arbitrary class codes.
            kind = task_kind(task)
            if kind is None:
                log.info("[ood] SKIP %s — task type %r is neither classification nor "
                         "regression", ds.name, getattr(task, "task_type", "?"))
                continue
            if chosen_by_kind[kind] >= n_per_task:
                log.info("[ood] SKIP %s — %s quota already full", ds.name, kind)
                continue

            if is_credit_like(ds.name):
                log.info("[ood] SKIP %s — credit-like, so not out-of-domain", ds.name)
                continue

            if ds.dataset_id in existing:
                kept.append(existing[ds.dataset_id])
                chosen_by_kind[existing[ds.dataset_id].kind] += 1
                continue

            log.info("[ood]   downloading %s (id=%s, %s) ...", ds.name, ds.dataset_id, kind)
            try:
                X_df, y_s, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
            except Exception as exc:  # noqa: BLE001
                log.warning("[ood] %s failed to download: %s", ds.name, exc)
                continue

            if X_df is None or y_s is None or len(X_df) == 0:
                continue
            if X_df.shape[1] > max_features:
                log.info("[ood] SKIP %s — %d features exceeds the cap", ds.name, X_df.shape[1])
                continue
            if len(X_df) > max_rows:
                # Subsample rather than skip: a large table is still a valid OOD task,
                # and the in-context models cap rows anyway.
                X_df = X_df.iloc[:max_rows]
                y_s = y_s.iloc[:max_rows]

            # Category codes for object/categorical columns, so the cache is numeric and
            # `.npz` can hold it. Which columns were categorical is recorded separately.
            cat_idx = [
                i for i, c in enumerate(X_df.columns)
                if str(X_df[c].dtype) in ("object", "category", "bool")
            ]
            X = X_df.copy()
            for c in X.columns:
                if str(X[c].dtype) in ("object", "category", "bool"):
                    X[c] = X[c].astype("category").cat.codes
            X_arr = X.to_numpy(dtype=np.float32)

            if kind == "classification":
                y_arr = y_s.astype("category").cat.codes.to_numpy().astype(np.int64)
                n_classes = int(len(np.unique(y_arr)))
                if n_classes < 2:
                    continue
                # A "classification" target with hundreds of levels is a mislabelled
                # regression task, and coding it would produce meaningless classes. This is
                # the belt to the braces of asking the task its type.
                if n_classes > 100:
                    log.info("[ood] SKIP %s — %d classes; that is a regression target",
                             ds.name, n_classes)
                    continue
            else:
                y_arr = pd_to_float(y_s)
                if y_arr is None:
                    log.info("[ood] SKIP %s — regression target is not numeric", ds.name)
                    continue
                n_classes = None
                if not np.isfinite(y_arr).all():
                    keep = np.isfinite(y_arr)
                    X_arr, y_arr = X_arr[keep], y_arr[keep]
                    if len(y_arr) < 50:
                        continue

            entry = OODDataset(
                name=ds.name, openml_id=int(ds.dataset_id), kind=kind,
                n_rows=int(X_arr.shape[0]), n_features=int(X_arr.shape[1]),
                n_classes=n_classes,
            )
            out = root / f"{entry.slug}.npz"
            tmp = out.with_suffix(".npz.tmp")
            # WRITE THROUGH AN OPEN HANDLE, not a path. Given a path whose name does not
            # end in `.npz`, `np.savez_compressed` silently APPENDS the extension — so
            # `3.kr-vs-kp.npz.tmp` became `3.kr-vs-kp.npz.tmp.npz` and the rename below
            # then failed with FileNotFoundError on the very first dataset.
            with tmp.open("wb") as fh:
                np.savez_compressed(
                    fh, X=X_arr, y=y_arr, cat_indices=np.asarray(cat_idx, dtype=np.int64)
                )
            # Rename LAST, so a download killed halfway leaves a `.tmp` that no reader
            # picks up rather than a truncated `.npz` that looks complete.
            tmp.replace(out)
            kept.append(entry)
            chosen_by_kind[kind] += 1
            added_here += 1
            log.info("[ood] cached %-28s %6d x %-4d (%s)",
                     entry.name, entry.n_rows, entry.n_features, kind)

        log.info("[ood] %s: +%d datasets (totals %s)", suite_name, added_here,
                 dict(chosen_by_kind))

    # Manifest LAST, as the completeness marker — same contract as the data pipeline.
    payload = {"version": OOD_VERSION, "suites": SUITES, "datasets": [asdict(d) for d in kept]}
    tmp = manifest_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_path())
    log.info("[ood] wrote %s with %d datasets", manifest_path(), len(kept))
    return kept


def load_ood_dataset(entry: OODDataset) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load one cached OOD table. Never touches the network."""
    path = ood_root() / f"{entry.slug}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Fetch the out-of-domain suites first, ON A LOGIN NODE "
            f"(compute nodes have no internet):\n    python -m src.utils.fetch_ood"
        )
    with np.load(path) as z:
        return z["X"], z["y"], [int(i) for i in z["cat_indices"]]


def ood_status() -> dict[str, Any]:
    """What is cached, for a log line or a status check."""
    entries = list_ood_datasets()
    by_kind: dict[str, list[str]] = {}
    for d in entries:
        by_kind.setdefault(d.kind, []).append(d.name)
    return {
        "root": str(ood_root()),
        "manifest": str(manifest_path()),
        "exists": manifest_path().is_file(),
        "n_datasets": len(entries),
        "by_kind": {k: len(v) for k, v in by_kind.items()},
        "names": by_kind,
        "suites": SUITES,
        "complete": all(len(by_kind.get(k, [])) >= N_PER_TASK for k in SUITES),
    }
