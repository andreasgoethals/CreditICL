"""Development contrasts must compare the same seed, recipe and observed update."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualize import training_plots as tp


def _frame(steps, values):
    return pd.DataFrame({"step": steps, "real__dev__roc_auc": values,
                         "real__holdout__roc_auc": np.full(len(steps), .99)})


def test_control_gap_pairs_seeds_and_exact_updates_without_interpolation():
    names = "exp1_pd__prior-credit_fraction={}__s{}"
    runs = {
        names.format(0, 0): _frame([0, 20, 40], [.1, .2, .3]),
        names.format("0p5", 0): _frame([0, 10, 40], [.3, .9, .4]),
        names.format(0, 1): _frame([0, 20, 40], [.7, .7, .7]),
        names.format("0p5", 1): _frame([0, 20, 40], [.8, .9, .8]),
        # An unpaired high-performing seed must not enter the comparison.
        names.format("0p5", 2): _frame([0, 20, 40], [1., 1., 1.]),
    }
    paired = tp._paired_control_gaps(runs, "roc_auc", ["dev"])
    gap = next(iter(paired.values()))
    assert list(gap.columns) == [0, 1]
    assert list(gap.index) == [0, 20, 40]
    assert np.isnan(gap.loc[20, 0])
    np.testing.assert_allclose(gap.loc[0], [.2, .1])
    np.testing.assert_allclose(gap.loc[40], [.1, .1])


def test_control_gap_never_pairs_different_training_recipes():
    runs = {
        "exp1_pd__prior-credit_fraction=0__filter-mode=off__s0": _frame([0, 1], [.1, .2]),
        "exp1_pd__prior-credit_fraction=1__filter-mode=banded__s0": _frame([0, 1], [.8, .9]),
    }
    assert not tp._paired_control_gaps(runs, "roc_auc", ["dev"])


def test_control_gap_mean_and_band_describe_paired_seed_differences(monkeypatch):
    names = "exp1_pd__prior-credit_fraction={}__s{}"
    runs = {
        names.format(0, 0): _frame([0, 1], [.1, .1]),
        names.format(1, 0): _frame([0, 1], [.3, .3]),
        names.format(0, 1): _frame([0, 1], [.8, .8]),
        names.format(1, 1): _frame([0, 1], [.9, np.nan]),
    }
    monkeypatch.setattr(tp, "_view", lambda *args: (runs, runs, ["dev"], ["holdout"]))
    fig = tp.credit_vs_control_over_training("pd")
    try:
        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(line.get_ydata()[0], .15)
        assert np.isnan(line.get_ydata()[1])
        vertices = fig.axes[0].collections[0].get_paths()[0].vertices
        expected_sd = np.std([.2, .1], ddof=1)
        np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()],
                                   [.15 - expected_sd, .15 + expected_sd])
        assert "2 paired seeds" in line.get_label()
        assert "0% control" in fig.axes[0].get_ylabel()
    finally:
        plt.close(fig)
