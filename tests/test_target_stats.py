"""The boundary-mass metric.

This module exists because of a real bug that would have wrecked the project's
motivation. The naive metric `(y <= 0).mean()` reports about 0.5 on a
standard-scaled target, because there "y <= 0" just means "below the mean". It
made the unmodified TabICL prior look like it already put 54% of its mass at zero.

The regression test for that is `test_standard_scaled_target_is_not_reported_as
_having_boundary_mass`. Do not delete it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.utils.target_stats import summarise, target_stats


def test_continuous_target_has_almost_no_boundary_mass():
    g = torch.Generator().manual_seed(0)
    y = torch.rand(1000, generator=g)
    s = target_stats(y)
    assert s["frac_at_min"] == pytest.approx(0.001, abs=0.002)
    assert s["frac_at_max"] == pytest.approx(0.001, abs=0.002)
    assert not s["has_atom_at_min"] and not s["has_atom_at_max"]


def test_atoms_are_detected():
    y = torch.cat([torch.zeros(200), torch.rand(600), torch.ones(200)])
    s = target_stats(y)
    assert s["frac_at_min"] == pytest.approx(0.2, abs=0.01)
    assert s["frac_at_max"] == pytest.approx(0.2, abs=0.01)
    assert s["has_atom_at_min"] and s["has_atom_at_max"]
    assert s["boundary_mass"] == pytest.approx(0.4, abs=0.02)


def test_standard_scaled_target_is_not_reported_as_having_boundary_mass():
    """THE regression test. A standard-scaled continuous target has no atoms, and
    the metric must say so — even though half its values are below zero."""
    g = torch.Generator().manual_seed(0)
    y = torch.randn(1000, generator=g)  # mean 0, so ~50% of values are <= 0
    s = target_stats(y)
    assert s["boundary_mass"] < 0.01, "a standard-scaled continuous target has no atoms"
    assert s["in_unit_interval"] is False
    assert s["unit_mass_at_0"] is None, "must refuse to report a [0,1] metric off [0,1]"
    assert s["unit_mass_at_1"] is None


def test_metric_is_invariant_to_rescaling():
    """The point of the design: an affine map keeps the shape, so it must keep the
    measured boundary mass. This is what lets us compare a [0,1] target against a
    standard-scaled one at all."""
    y = torch.cat([torch.zeros(300), torch.rand(400), torch.ones(300)])
    scaled = (y - y.mean()) / y.std()
    a, b = target_stats(y), target_stats(scaled)
    assert a["frac_at_min"] == pytest.approx(b["frac_at_min"])
    assert a["frac_at_max"] == pytest.approx(b["frac_at_max"])
    assert a["boundary_mass"] == pytest.approx(b["boundary_mass"])


def test_unit_metrics_reported_only_on_unit_targets():
    y = torch.cat([torch.zeros(100), torch.rand(800), torch.ones(100)])
    s = target_stats(y)
    assert s["in_unit_interval"] is True
    assert s["unit_mass_at_0"] == pytest.approx(0.1, abs=0.01)
    assert s["unit_mass_at_1"] == pytest.approx(0.1, abs=0.01)


def test_one_sided_atom():
    y = torch.cat([torch.zeros(250), torch.rand(750)])
    s = target_stats(y)
    assert s["has_atom_at_min"] and not s["has_atom_at_max"]


def test_constant_target():
    s = target_stats(torch.zeros(100))
    assert s["frac_at_min"] == 1.0 and s["n_distinct"] == 1


def test_empty_target_does_not_crash():
    assert target_stats(torch.zeros(0))["n"] == 0


def test_distinct_fraction_separates_discrete_from_continuous():
    g = torch.Generator().manual_seed(0)
    continuous = target_stats(torch.rand(500, generator=g))
    discrete = target_stats(torch.randint(0, 5, (500,), generator=g).float())
    assert continuous["distinct_fraction"] > 0.9
    assert discrete["distinct_fraction"] < 0.05


def test_summarise_aggregates():
    tasks = [
        target_stats(torch.cat([torch.zeros(100), torch.rand(800), torch.ones(100)])),
        target_stats(torch.rand(1000)),
    ]
    out = summarise(tasks)
    assert out["tasks"] == 2
    assert 0.0 <= out["tasks_with_any_atom"] <= 1.0
    assert out["tasks_with_both_atoms"] == pytest.approx(0.5)
    assert out["tasks_in_unit_interval"] == pytest.approx(1.0)


def test_summarise_handles_mixed_scaling():
    """Half the tasks bounded, half standard-scaled — the aggregate must not
    silently average a None into the unit metrics."""
    tasks = [
        target_stats(torch.cat([torch.zeros(100), torch.rand(900)])),
        target_stats(torch.randn(1000)),
    ]
    out = summarise(tasks)
    assert out["tasks_in_unit_interval"] == pytest.approx(0.5)
    assert "unit_mass_at_0_mean" in out  # computed from the bounded half only


def test_summarise_on_empty_list():
    assert summarise([]) == {}
