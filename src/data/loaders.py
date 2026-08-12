"""Loading and preprocessing — the entry point.

THE TEMPLATE'S NAME FOR THIS CONCEPT, and in this project the concept is big enough to have
been split in two, so this file is the door rather than the implementation:

* :mod:`src.data.discovery` — *what* datasets exist, on this machine, for a task.
* :mod:`src.data.pipeline` — turning one raw dataset into a cached, model-ready frame.

Importing from here is the supported way in; the two modules behind it are free to be
reorganised. It also means the template's `src/data/loaders.py` resolves to something real
instead of an empty stub sitting next to the code that actually does the work.

Both of the template's asks hold:

1. **No path is ever built here.** Everything comes from :mod:`src.utils.paths`, so one call
   works on a laptop and on both cluster tiers.
2. **`data/raw/` is read-only**, and a cache counts as valid only once its marker exists — the
   marker is written LAST (`pipeline._atomic_write_text`), so a run killed halfway leaves a
   cache correctly treated as absent. A half-written cache that looks complete is a wrong
   result nobody investigates.
"""

from __future__ import annotations

from src.data.discovery import describe_availability, list_all_datasets, list_datasets
from src.data.pipeline import (
    DatasetMeta,
    ProcessedDataset,
    cache_report,
    ensure_processed,
    load_frame,
    load_processed,
    preprocess_one,
)

__all__ = [
    # discovery — what is available
    "list_datasets",
    "list_all_datasets",
    "describe_availability",
    # preprocessing — raw -> cache
    "preprocess_one",
    "ensure_processed",
    "cache_report",
    # loading — cache -> arrays
    "load_processed",
    "load_frame",
    "ProcessedDataset",
    "DatasetMeta",
]
