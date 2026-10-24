"""Preserve the meaning and separability of the paper's diagnostic figures."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.visualize import style
from src.visualize.exp1_plots import plot_standardised_endpoint_ties
from src.visualize.mechanism_plots import informative_missingness, missingness_by_feature_group


def test_endpoint_ties_keep_generators_separate_on_the_same_scale():
    task = SimpleNamespace(y=torch.tensor([0., 0., .5, 1.]), meta={"train_size": 3})
    shifted = SimpleNamespace(y=task.y*5+7, meta=task.meta)
    fig = plot_standardised_endpoint_ties({"original": [task], "credit": [shifted]})
    try:
        assert len(fig.axes) == 2
        left, right = fig.axes
        np.testing.assert_allclose(left.collections[0].get_offsets(), [[.5, .25]])
        np.testing.assert_allclose(right.collections[0].get_offsets(), [[.5, .25]])
        assert left.get_xlim() == right.get_xlim()
        assert left.get_ylim() == right.get_ylim()
        assert "Original" in left.get_title(loc="left")
        assert "Credit" in right.get_title(loc="left")
        assert not np.array_equal(left.collections[0].get_facecolors(), right.collections[0].get_facecolors())
    finally:
        plt.close(fig)


def test_missingness_groups_account_for_every_row_and_show_feature_dependence():
    frame = missingness_by_feature_group(n_rows=4000, seed=0)
    for _, rows in frame.groupby("coupling"):
        assert rows.n_rows.sum() == 4000
        assert rows.n_rows.max()-rows.n_rows.min() <= 1
        assert rows.group.tolist() == list(range(1, 7))
        assert rows.missing_share.between(0, 1).all()
        assert np.average(rows.missing_share, weights=rows.n_rows) == pytest.approx(.2, abs=.04)
    # This controlled seed is explanatory, not a claim every random direction
    # increases the probability with this particular feature.
    independent = frame[frame.coupling.eq(0)].missing_share.to_numpy()
    dependent = frame[frame.coupling.eq(1)].missing_share.to_numpy()
    assert dependent[-1]-dependent[0] > .1
    assert abs(independent[-1]-independent[0]) < .05


def test_missingness_plot_explains_groups_and_redundantly_encodes_conditions():
    fig = informative_missingness("pd")
    try:
        ax = fig.axes[0]
        assert "synthetic feature" in ax.get_xlabel()
        assert "Missing feature" in ax.get_ylabel()
        assert ax.lines[0].get_marker() != ax.lines[1].get_marker()
        assert ax.lines[0].get_linestyle() != ax.lines[1].get_linestyle()
        assert ax.lines[0].get_color() != ax.lines[1].get_color()
        assert [t.get_text() for t in ax.get_xticklabels()][::5] == ["Lowest", "Highest"]
    finally:
        plt.close(fig)


def test_main_mixtures_are_distinct_without_relying_only_on_colour():
    assert len({style.credit_fraction_colour(x) for x in (0, .5, 1)}) == 3
    assert len({style.credit_fraction_line(x) for x in (0, .5, 1)}) == 3
    assert len({style.credit_fraction_marker(x) for x in (0, .5, 1)}) == 3
