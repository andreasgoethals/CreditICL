"""Load the institution's restricted dataset recipes outside version control."""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PRIVATE_DATASETS = {
    "pd": ("PropPD1", "PropPD2"),
    "lgd": ("PropLGD1", "PropLGD2", "PropLGD3", "PropLGD4", "PropLGD5"),
}
PRIVATE_KEYS = frozenset(key for keys in PRIVATE_DATASETS.values() for key in keys)
_DEFAULT = Path(__file__).resolve().parents[2] / ".private-data" / "site.py"


def site_path() -> Path:
    return Path(os.environ.get("CREDITICL_PRIVATE_DATA", str(_DEFAULT))).resolve()


@lru_cache(maxsize=2)
def _load_site(path: Path):
    spec = spec_from_file_location("_crediticl_private_site", path)
    if spec is None or spec.loader is None:
        raise ValueError("The restricted dataset configuration must be a Python module")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = module.LEGACY_SLUGS
    if set(keys) != PRIVATE_KEYS or len(set(keys.values())) != len(keys) or any(
        re.fullmatch(r"\d{4}\.[A-Za-z0-9_-]+", str(x)) is None for x in keys.values()
    ):
        raise ValueError("The restricted dataset configuration has invalid identifiers")
    module.__source_sha256__ = hashlib.sha256(path.read_bytes()).hexdigest()
    return module


def site(*, required: bool = False):
    path = site_path()
    if path.is_file():
        return _load_site(path)
    if required or "CREDITICL_PRIVATE_DATA" in os.environ:
        raise FileNotFoundError("Restricted inputs require a local dataset configuration")
    return None


def site_digest() -> str | None:
    module = site()
    return module.__source_sha256__ if module else None


def legacy_alias(dataset: str) -> str | None:
    """Resolve historic keys only with the trusted local mapping."""
    module = site()
    if module is None:
        return None
    token = dataset.casefold()
    for alias, slug in module.LEGACY_SLUGS.items():
        if token in (slug.casefold(), slug.replace(".", "_").casefold(),
                     slug.partition(".")[2].casefold()):
            return alias
    return None


def storage_keys(task: str, dataset: str) -> tuple[str, ...]:
    """Keep existing raw data and processed caches at their original paths."""
    if dataset not in PRIVATE_DATASETS.get(task, ()):
        return (dataset,)
    module = site()
    return (dataset, module.LEGACY_SLUGS[dataset]) if module else (dataset,)


def publication_text(text: str) -> str:
    """Convert legacy result/monitoring identifiers without touching input files."""
    module = site()
    if module is None:
        return text
    for alias, slug in module.LEGACY_SLUGS.items():
        tokens = (slug, slug.replace(".", "_"), slug.partition(".")[2])
        expression = "|".join(map(re.escape, tokens))
        text = re.sub(rf"(?<![a-z0-9])(?:{expression})(?![a-z0-9])", alias, text, flags=re.I)
    return text
