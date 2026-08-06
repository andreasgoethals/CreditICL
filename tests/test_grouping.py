"""Three-level sampling: group -> subgroup -> dataset.

TabICL's prior does not draw each dataset independently. It draws a *group* that
shares meta-distributions, then subgroups that fix concrete values, then individual
datasets. This is on by default here (`group_size: 4`), for two reasons:

* `credit_fraction=0.0` is the control arm, and the control should BE TabICL — so
  the more faithfully it copies upstream, the more the comparison means.
* a group sharing hyperparameters *is* a small domain, which is the very thing the
  research question is about.

So it needs to be correct, and it needs to be genuinely switchable — a lever nobody
tests is a lever that quietly does nothing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

from src.prior.grouping import GroupedSampler
from src.prior.rng import PriorRNG


def make(group_size=4, subgroup_size=2, seed=0, **extra):
    cfg = {
        "grouping": {"group_size": group_size, "subgroup_size": subgroup_size},
        "n_rows_range": [256, 1024],
        "n_features_range": [4, 40],
        "max_features": 64,
        **extra,
    }
    return GroupedSampler(cfg, PriorRNG(seed))


def test_group_size_one_disables_grouping():
    """The documented way to turn it off. It must return nothing at all, so the
    generator falls back to independent draws rather than silently reusing a stale
    shared dict."""
    s = make(group_size=1)
    assert s.enabled is False
    assert all(s.next_dataset() == {} for _ in range(10))


def test_datasets_in_a_subgroup_share_their_shape():
    """The point of the mechanism. Consecutive datasets inside one subgroup get
    identical shape; that is what makes a group a coherent little domain."""
    s = make(group_size=4, subgroup_size=2)
    draws = [s.next_dataset() for _ in range(4)]
    assert draws[0] == draws[1], "first subgroup must be internally identical"
    assert draws[2] == draws[3], "second subgroup must be internally identical"


def test_subgroups_within_a_group_differ():
    """If every subgroup came out the same, the middle level would be pointless."""
    s = make(group_size=8, subgroup_size=2)
    draws = [s.next_dataset() for _ in range(8)]
    subgroup_values = [draws[i] for i in range(0, 8, 2)]
    assert len({tuple(sorted(d.items())) for d in subgroup_values}) > 1


def test_groups_differ_from_each_other():
    s = make(group_size=2, subgroup_size=2)
    first = s.next_dataset()
    for _ in range(3):
        s.next_dataset()
    later = s.next_dataset()
    assert first != later, "a new group must redraw its region"


def test_group_counters_advance_exactly():
    """Off-by-one here would make groups the wrong size — a silent change to how
    correlated the training stream is."""
    s = make(group_size=4, subgroup_size=2)
    for _ in range(4):
        s.next_dataset()
    assert s.state.group_index == 1
    s.next_dataset()
    assert s.state.group_index == 2, "the 5th dataset starts a new group"


def test_subgroup_never_outlives_its_group():
    """subgroup_size larger than group_size must clamp, not bleed the shared values
    into the next group — which would silently merge two groups into one."""
    s = make(group_size=3, subgroup_size=10)
    draws = [s.next_dataset() for _ in range(3)]
    assert all(d == draws[0] for d in draws)
    assert s.state.group_index == 1
    nxt = s.next_dataset()
    assert s.state.group_index == 2
    assert nxt != draws[0]


def test_shared_values_stay_inside_the_configured_ranges():
    """A group picks a REGION of each range; the region must not escape the range."""
    s = make(group_size=4, subgroup_size=1)
    for _ in range(200):
        d = s.next_dataset()
        assert 256 <= d["n_rows"] <= 1024
        assert 4 <= d["n_features"] <= 40
        assert 0.0 <= d["signal_strength"] <= 1.0


def test_max_features_caps_the_range():
    """max_features is the model's real limit; a group must not ask for more."""
    s = make(group_size=4, subgroup_size=1, n_features_range=[4, 500], max_features=20)
    for _ in range(100):
        assert s.next_dataset()["n_features"] <= 20


def test_callers_cannot_mutate_the_shared_state():
    """next_dataset returns a copy. Without that, a generator writing into the dict
    it was handed would corrupt every later dataset in the same subgroup."""
    s = make(group_size=4, subgroup_size=2)
    first = s.next_dataset()
    first["n_rows"] = -999
    second = s.next_dataset()
    assert second["n_rows"] != -999


def test_same_seed_gives_the_same_group_structure():
    a = [make(seed=7).next_dataset() for _ in range(1)]
    b = [make(seed=7).next_dataset() for _ in range(1)]
    assert a == b


def test_different_seeds_give_different_structure():
    assert make(seed=1).next_dataset() != make(seed=2).next_dataset()


def test_describe_reports_what_a_log_needs():
    """Runs are debugged from logs alone, so the log must say whether grouping was on."""
    s = make(group_size=4, subgroup_size=2)
    s.next_dataset()
    d = s.describe()
    assert d["grouping_enabled"] is True
    assert d["group_size"] == 4 and d["subgroup_size"] == 2


def test_degenerate_config_does_not_crash():
    """A one-wide range has no room for a region. Guard against the div-by-zero and
    empty-randint that a naive span calculation would hit."""
    s = make(group_size=4, subgroup_size=2, n_rows_range=[100, 100], n_features_range=[5, 5])
    for _ in range(20):
        d = s.next_dataset()
        assert d["n_rows"] == 100 and d["n_features"] == 5


def test_grouping_is_on_by_default_in_both_configs():
    """The control arm should be as faithful to TabICL as we can make it, and TabICL
    groups. If someone turns this off in a config, this test should say so loudly.
    """
    from src.utils.config import load_yaml

    for path in ("config/LGD.yaml", "config/PD.yaml"):
        cfg = load_yaml(path)
        gs = cfg["prior"]["grouping"]["group_size"]
        # The lever is sweepable, so it may be a list; either way the default (first
        # value) must be > 1.
        first = gs[0] if isinstance(gs, list) else gs
        assert first > 1, f"{path}: grouping is disabled by default"
