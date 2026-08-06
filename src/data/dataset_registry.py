"""Single source of truth for how datasets are NAMED and ORDERED in outputs.

Why this module exists
---------------------
Proprietary datasets must appear in the paper under anonymised names
(``PropPD1``, ``PropLGD1``, ...), and every dataset is renumbered so that
**public datasets come first in alphabetical order, followed by the
proprietary ones**. That renumbering is a *display-layer* concern only:

* On-disk slugs, directory names, result files, cached predictions and figure
  output paths are **unchanged** (existing ``\\includegraphics`` paths in the
  LaTeX keep working). The slug stays the join key everywhere.
* Only the text a reader sees -- axis tick labels, legends, row/column
  headers, titles, table cells -- goes through this registry.

Public API
----------
* :func:`display_name`  -- slug -> reader-facing label (e.g. ``"PropPD1"``).
* :func:`paper_id`      -- slug -> new paper ID (e.g. ``"PD13"``).
* :func:`is_proprietary`/:func:`source_of` -- provenance flags.
* :func:`sort_key`      -- ``(is_proprietary, display_name)``; THE ordering.
* :func:`sort_datasets` -- sort any iterable of slugs into paper order.
* :func:`display_names` -- map a list of slugs to labels in one call.
* :func:`validate_registry` -- fail-loudly consistency checks.
* :func:`format_mapping_table` -- printable old -> new ID table.

Rules for callers
-----------------
No display name may be hard-coded anywhere else. Any plotting/table code that
used to sort datasets implicitly (by filename, insertion order, or numeric slug
prefix -- all of which give the OLD ordering) must sort with
:func:`sort_key` / :func:`sort_datasets` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
#  Entry type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset's identity across the on-disk, old-paper and new-paper worlds.

    Attributes:
        slug: The on-disk dataset directory name, e.g. ``"0009.bank_status"``.
            This is the join key for results/figures and is NEVER renamed.
        task: ``"pd"`` or ``"lgd"``.
        old_id: The paper ID used before the renumbering (e.g. ``"PD9"``).
        new_id: The paper ID used from now on (e.g. ``"PD13"``).
        display_name: The reader-facing label. Anonymised for proprietary data.
        proprietary: True for data that may not be named in the paper.
    """

    slug: str
    task: str
    old_id: str
    new_id: str
    display_name: str
    proprietary: bool

    @property
    def source(self) -> str:
        """``"proprietary"`` or ``"public"`` -- the registry's source string."""
        return "proprietary" if self.proprietary else "public"


# ---------------------------------------------------------------------------
#  The authoritative mapping
# ---------------------------------------------------------------------------
# Ordered by NEW id. Slugs use the on-disk form (zero-padded number, DOT,
# snake_case) -- verified against data/processed/ and results/experiment1/.
# Public datasets keep their real names; proprietary ones are anonymised.

_ENTRIES: Tuple[DatasetEntry, ...] = (
    # ---- PD: public, alphabetical by display name ----
    DatasetEntry("0007.cobranded",          "pd", "PD7",  "PD1",  "Cobranded",          False),
    DatasetEntry("0008.german",             "pd", "PD8",  "PD2",  "German Credit",      False),
    DatasetEntry("0001.gmsc",               "pd", "PD1",  "PD3",  "GMSC",               False),
    DatasetEntry("0006.hackerearth",        "pd", "PD6",  "PD4",  "Hackerearth",        False),
    DatasetEntry("0013.hmeq",               "pd", "PD13", "PD5",  "HMEQ",               False),
    DatasetEntry("0012.home_credit",        "pd", "PD12", "PD6",  "Home Credit",        False),
    DatasetEntry("0004.lendingclub",        "pd", "PD4",  "PD7",  "LendingClub (PD)",   False),
    DatasetEntry("0014.algorithmwatch",     "pd", "PD14", "PD8",  "MicroFinance",       False),
    DatasetEntry("0005.myhom",              "pd", "PD5",  "PD9",  "Myhom",              False),
    DatasetEntry("0002.taiwan_creditcard",  "pd", "PD2",  "PD10", "Taiwan Credit Card", False),
    DatasetEntry("0010.thomas",             "pd", "PD10", "PD11", "Thomas",             False),
    DatasetEntry("0003.vehicle_loan",       "pd", "PD3",  "PD12", "Vehicle Loan",       False),
    # ---- PD: proprietary, anonymised ----
    DatasetEntry("0009.bank_status",        "pd", "PD9",  "PD13", "PropPD1",            True),
    DatasetEntry("0011.loan_default",       "pd", "PD11", "PD14", "PropPD2",            True),
    # ---- LGD: public, alphabetical by display name ----
    DatasetEntry("0006.lgd_freddie",        "lgd", "LGD6", "LGD1", "Freddie Mac (LGD)", False),
    DatasetEntry("0007.lgd_lendingclub",    "lgd", "LGD7", "LGD2", "LendingClub (LGD)", False),
    # ---- LGD: proprietary, anonymised ----
    DatasetEntry("0001.heloc",              "lgd", "LGD1", "LGD3", "PropLGD1",          True),
    DatasetEntry("0002.loss2",              "lgd", "LGD2", "LGD4", "PropLGD2",          True),
    DatasetEntry("0003.axa",                "lgd", "LGD3", "LGD5", "PropLGD3",          True),
    DatasetEntry("0004.base_model",         "lgd", "LGD4", "LGD6", "PropLGD4",          True),
    DatasetEntry("0005.base_modelisation",  "lgd", "LGD5", "LGD7", "PropLGD5",          True),
)

REGISTRY: Dict[str, DatasetEntry] = {e.slug: e for e in _ENTRIES}

# Slug aliases: callers may hold an underscore-separated variant
# (``0009_bank_status``) or a bare name (``bank_status``, ``gmsc``) -- e.g. a
# figure filename stem or a hand-typed notebook argument. Normalise those to
# the canonical dotted slug so lookups never silently miss.
_ALIASES: Dict[str, str] = {}
for _e in _ENTRIES:
    _num, _, _name = _e.slug.partition(".")
    for _alias in (
        _e.slug.replace(".", "_"), _name, f"{_num}.{_name}",
        _e.new_id, _e.new_id.lower(),
        # The display name itself, so a NEUTRAL figure filename built from it
        # (e.g. ``corr_proplgd1.pdf``, ``pd_row_limit_proppd1_auc.pdf``) still
        # resolves back to this entry -- the caption generator parses the
        # dataset token out of the filename stem.
        _e.display_name, _e.display_name.lower(),
        _e.display_name.lower().replace(" ", "_"),
    ):
        _ALIASES.setdefault(_alias, _e.slug)


# ---------------------------------------------------------------------------
#  Lookup helpers
# ---------------------------------------------------------------------------


def canonical_slug(dataset: object) -> Optional[str]:
    """Return the canonical dotted slug for ``dataset``, or ``None`` if unknown.

    Accepts the canonical slug, the underscore variant, the bare dataset name
    or a new paper ID, so filename stems and hand-typed notebook arguments
    resolve to the same entry.
    """
    key = str(dataset).strip()
    if key in REGISTRY:
        return key
    return _ALIASES.get(key)


def get(dataset: object) -> Optional[DatasetEntry]:
    """Return the :class:`DatasetEntry` for ``dataset``, or ``None`` if unknown."""
    slug = canonical_slug(dataset)
    return REGISTRY[slug] if slug else None


def display_name(dataset: object) -> str:
    """Reader-facing label for ``dataset`` (anonymised when proprietary).

    Unknown datasets fall back to a readable form of the slug (digits and
    separators stripped) rather than raising, so a newly added dataset shows
    something sensible in a figure instead of crashing the plot -- add it to
    the registry to give it a real name.
    """
    entry = get(dataset)
    if entry is not None:
        return entry.display_name
    raw = str(dataset)
    import re

    return re.sub(r"^\d+[._-]", "", raw).replace("_", " ")


def paper_id(dataset: object) -> str:
    """New paper ID (e.g. ``"PD13"``); falls back to :func:`display_name`."""
    entry = get(dataset)
    return entry.new_id if entry is not None else display_name(dataset)


def is_proprietary(dataset: object) -> bool:
    """True iff ``dataset`` may not be named in the paper (unknown -> False)."""
    entry = get(dataset)
    return bool(entry and entry.proprietary)


def source_of(dataset: object) -> str:
    """``"public"`` / ``"proprietary"`` (unknown -> ``"unknown"``)."""
    entry = get(dataset)
    return entry.source if entry is not None else "unknown"


# ---------------------------------------------------------------------------
#  Ordering -- the ONE ordering rule for every output
# ---------------------------------------------------------------------------


def sort_key(dataset: object) -> Tuple[int, str]:
    """Paper order: public first (alphabetical), then proprietary.

    Use as ``sorted(datasets, key=sort_key)`` or
    ``df.sort_values("dataset", key=lambda s: s.map(sort_key))``. Sorting by
    slug or old ID gives the OLD order and must not be used for output.
    """
    entry = get(dataset)
    if entry is None:
        # Unknown datasets sort last, alphabetically, so they are visible.
        return (2, display_name(dataset).lower())
    return (1 if entry.proprietary else 0, entry.display_name.lower())


def sort_datasets(datasets: Iterable[object]) -> List[str]:
    """Return ``datasets`` (as given) sorted into paper order."""
    return sorted((str(d) for d in datasets), key=sort_key)


def display_names(datasets: Sequence[object]) -> List[str]:
    """Map a sequence of slugs to display labels (order preserved)."""
    return [display_name(d) for d in datasets]


def datasets_for_task(task: str, *, in_paper_order: bool = True) -> List[str]:
    """Every registered slug for ``task``, in paper order by default."""
    slugs = [e.slug for e in _ENTRIES if e.task == task.lower()]
    return sort_datasets(slugs) if in_paper_order else slugs


def entries_for_task(task: str) -> List[DatasetEntry]:
    """Registry entries for ``task`` in paper order."""
    return [REGISTRY[s] for s in datasets_for_task(task)]


# ---------------------------------------------------------------------------
#  Validation + reporting
# ---------------------------------------------------------------------------


def validate_registry(known_slugs: Optional[Dict[str, Iterable[str]]] = None) -> None:
    """Fail loudly on any registry inconsistency.

    Checks (per the renaming spec):
      1. Exactly 14 PD + 7 LGD entries; new IDs are ``PD1..PD14`` /
         ``LGD1..LGD7`` with no gaps or duplicates (old IDs unique too).
      2. Slugs unique; display names unique.
      3. Sorting by ``(is_proprietary, display_name)`` reproduces the new-ID
         order exactly.
      4. When ``known_slugs`` is given (``{"pd": [...], "lgd": [...]}``, e.g.
         from the results/processed directory), every on-disk slug appears
         exactly once in the registry and vice versa.

    Raises:
        AssertionError: with a message naming the exact discrepancy.
    """
    for task, n_expected, prefix in (("pd", 14, "PD"), ("lgd", 7, "LGD")):
        entries = [e for e in _ENTRIES if e.task == task]
        assert len(entries) == n_expected, (
            f"registry: expected {n_expected} {task.upper()} entries, got {len(entries)}"
        )
        new_ids = [e.new_id for e in entries]
        expected_ids = [f"{prefix}{i}" for i in range(1, n_expected + 1)]
        assert sorted(new_ids, key=lambda s: int(s[len(prefix):])) == expected_ids, (
            f"registry: {task.upper()} new IDs must be {prefix}1..{prefix}{n_expected} "
            f"with no gaps/duplicates; got {sorted(new_ids)}"
        )
        old_ids = [e.old_id for e in entries]
        assert len(set(old_ids)) == len(old_ids), (
            f"registry: duplicate old IDs in {task.upper()}: {old_ids}"
        )
        # Rule 3: the sort rule must reproduce the new-ID numbering exactly.
        by_rule = sort_datasets(e.slug for e in entries)
        by_new_id = [
            e.slug for e in sorted(entries, key=lambda e: int(e.new_id[len(prefix):]))
        ]
        assert by_rule == by_new_id, (
            f"registry: sort_key order != new-ID order for {task.upper()}.\n"
            f"  by (proprietary, display_name): {[REGISTRY[s].display_name for s in by_rule]}\n"
            f"  by new ID                     : {[REGISTRY[s].display_name for s in by_new_id]}"
        )

    slugs = [e.slug for e in _ENTRIES]
    assert len(set(slugs)) == len(slugs), "registry: duplicate slugs"
    names = [e.display_name for e in _ENTRIES]
    assert len(set(names)) == len(names), f"registry: duplicate display names: {names}"

    if known_slugs:
        for task, found in known_slugs.items():
            found_set = {str(s) for s in found}
            reg_set = {e.slug for e in _ENTRIES if e.task == task.lower()}
            missing = sorted(found_set - reg_set)
            extra = sorted(reg_set - found_set)
            assert not missing, (
                f"registry: {task} slug(s) on disk but not in the registry: {missing}"
            )
            assert not extra, (
                f"registry: {task} slug(s) in the registry but not on disk: {extra}"
            )


def format_mapping_table() -> str:
    """Printable old -> new ID table, in paper order, for eyeball verification."""
    lines = [
        f"{'slug':24s} {'old':>5s} -> {'new':<5s} {'display name':20s} source",
        "-" * 72,
    ]
    for task in ("pd", "lgd"):
        for e in entries_for_task(task):
            lines.append(
                f"{e.slug:24s} {e.old_id:>5s} -> {e.new_id:<5s} "
                f"{e.display_name:20s} {e.source}"
            )
        lines.append("")
    return "\n".join(lines).rstrip("\n")


__all__ = [
    "DatasetEntry",
    "REGISTRY",
    "canonical_slug",
    "get",
    "display_name",
    "paper_id",
    "is_proprietary",
    "source_of",
    "sort_key",
    "sort_datasets",
    "display_names",
    "datasets_for_task",
    "entries_for_task",
    "validate_registry",
    "format_mapping_table",
]


if __name__ == "__main__":  # pragma: no cover -- manual check
    from src.data.discovery import list_datasets

    validate_registry({t: list_datasets(t) for t in ("pd", "lgd")})
    print(format_mapping_table())
    print("\n[OK] registry validates against the datasets on disk.")
