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

* **OpenML-CC18** for classification. It is the suite **O'Prior itself evaluated on**
  (arXiv 2605.18971), so our out-of-domain numbers sit directly alongside the closest
  prior work rather than on a benchmark of our own choosing.
* **OpenML-CTR23** for regression. The curated OpenML regression suite, and the LGD arm
  is a regression model, so a classification-only benchmark would leave half the project
  unmeasured.

TabICLv2 reports on TabArena and TALENT instead. Those are bigger and better, but they
are also harder to obtain and slower to run; CC18/CTR23 is the cheaper choice that keeps
comparability with O'Prior. Stated so nobody later assumes we matched TabICLv2's suite.

**DATASET IDS ARE NEVER HARD-CODED HERE.** They are resolved from the suite through the
OpenML API at fetch time and then pinned into `ood_manifest.json`. Writing IDs from
memory is how you end up silently evaluating on the wrong tables — the suite *names* are
the verifiable thing, the IDs are not.

NO INTERNET ON COMPUTE NODES. Fetching must run on a **login node**
(`python scripts/fetch_ood.py`), which caches to project storage. `load_ood_dataset`
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

#: The two OpenML suites. Names, not IDs — see the module docstring.
SUITES = {
    "classification": "OpenML-CC18",
    "regression": "OpenML-CTR23",
}

#: How many of each to use. The user asked for 10 and 10; enough to see a trend without
#: making the eval pipeline longer than the training it evaluates.
N_PER_TASK = 10

#: Anything matching these is NOT out-of-domain and must be dropped. CC18 contains
#: `credit-g` and `credit-approval`, and O'Prior specifically singles out Credit-g as a
#: dataset where its gains vanish — including it here would quietly contaminate the
#: control with the very domain we are specialising toward.
CREDIT_PATTERNS = [
    r"credit", r"default", r"loan", r"lending", r"bank", r"mortgage",
    r"insurance", r"risk", r"financ", r"gmsc", r"hmeq",
]

#: Bump when the on-disk cache layout changes.
OOD_VERSION = 1


def is_credit_like(name: str) -> bool:
    """True if a dataset name suggests credit or lending.

    Deliberately over-inclusive: dropping a borderline dataset costs one of ten
    out-of-domain tasks, whereas keeping a credit-adjacent one silently weakens the
    entire out-of-domain claim.
    """
    low = name.lower()
    return any(re.search(p, low) for p in CREDIT_PATTERNS)


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

    for kind, suite_name in SUITES.items():
        log.info("[ood] resolving suite %s (%s)", suite_name, kind)
        suite = openml.study.get_suite(suite_name)
        task_ids = list(suite.tasks or [])
        log.info("[ood] %s advertises %d tasks", suite_name, len(task_ids))

        chosen = 0
        for task_id in sorted(task_ids):
            if chosen >= n_per_task:
                break
            try:
                task = openml.tasks.get_task(task_id, download_data=False)
                ds = openml.datasets.get_dataset(task.dataset_id, download_data=False)
            except Exception as exc:  # noqa: BLE001 — one bad task must not stop the sweep
                log.warning("[ood] task %s unavailable: %s", task_id, exc)
                continue

            if is_credit_like(ds.name):
                log.info("[ood] SKIP %s — credit-like, so not out-of-domain", ds.name)
                continue

            if ds.dataset_id in existing:
                kept.append(existing[ds.dataset_id])
                chosen += 1
                continue

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
            else:
                y_arr = y_s.to_numpy(dtype=np.float32)
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
            np.savez_compressed(tmp, X=X_arr, y=y_arr, cat_indices=np.asarray(cat_idx, dtype=np.int64))
            tmp.replace(out)
            kept.append(entry)
            chosen += 1
            log.info("[ood] cached %-28s %6d x %-4d (%s)",
                     entry.name, entry.n_rows, entry.n_features, kind)

        log.info("[ood] %s: %d/%d datasets cached", suite_name, chosen, n_per_task)

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
            f"(compute nodes have no internet):\n    python scripts/fetch_ood.py"
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
