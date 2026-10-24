"""The compact source inventory must preserve dataset identity and measured units."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualize import data_plots, style


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def dataset(values, categorical):
    X = np.asarray(values, dtype=float)
    return SimpleNamespace(X=X, n_features=X.shape[1], cat_indices=categorical)


def test_inventory_keeps_a_shared_order_and_distinct_denominators():
    books = {
        "lgd": {"0001.example": dataset([[1, np.nan], [1, np.nan], [1, 2]], [0])},
        "pd": {"0002.other": dataset([[np.nan, 1, 2, 3]], [0, 2, 3]),
               "0001.example": dataset([[1, 2]], [])},
    }
    table = data_plots.table_inventory(books)
    assert table[["task", "dataset"]].values.tolist() == [
        ["pd", "0001.example"], ["pd", "0002.other"], ["lgd", "0001.example"]]
    assert table["categorical_share"].tolist() == [0, .75, .5]
    assert table["missing_share"].tolist() == pytest.approx([0, .25, 1 / 3])
    fig = data_plots.plot_table_inventory(books)
    assert fig.get_figwidth() == pytest.approx(style.WIDTH_FULL)
    assert fig.get_figheight() <= style.MAX_HEIGHT
    assert fig.axes[0].get_yticks().tolist() == fig.axes[1].get_yticks().tolist()
    assert [label.get_text() for label in fig.axes[0].get_yticklabels()] == [
        "PD · example", "PD · other", "LGD · example"]
    for axis, expected in zip(fig.axes, ([0, .75, .5], [0, .25, 1 / 3])):
        actual = [collection.get_offsets()[0, 0] for collection in axis.collections]
        assert actual == pytest.approx(expected)
        assert all(tick <= 1 for tick in axis.get_xticks())


def test_missingness_zero_and_small_positive_values_stay_distinguishable():
    X = np.ones((2000, 1))
    X[0, 0] = np.nan
    books = {"pd": {"0001.clean": dataset([[1]], []),
                     "0002.sparse": dataset(X, [])}}
    fig = data_plots.plot_table_inventory(books)
    labels = [text.get_text() for text in fig.axes[1].texts]
    assert labels == ["0.0%", "<0.1%"]
    assert data_plots.table_inventory(books).iloc[1]["missing_share"] == pytest.approx(.0005)


def test_empty_source_inventory_does_not_draw_zero_data():
    fig = data_plots.plot_table_inventory({"pd": {}, "lgd": {}})
    assert not any(ax.collections or ax.lines for ax in fig.axes)
    assert "No processed source tables" in fig.axes[0].texts[0].get_text()


def test_correlation_inventory_preserves_task_identity_without_threshold_flags():
    measured = pd.DataFrame([
        {"task": "pd", "dataset": "0001.shared", "feature": "a", "|corr with target|": .99,
         "suspicious": True},
        {"task": "pd", "dataset": "0001.shared", "feature": "b", "|corr with target|": .5,
         "suspicious": False},
        {"task": "lgd", "dataset": "0001.shared", "feature": "c", "|corr with target|": .2,
         "suspicious": False},
    ])
    table = data_plots.correlation_inventory(measured)
    assert len(table) == 2
    assert set(table["feature"]) == {"a", "c"}
    assert "suspicious" not in table
    assert set(table["|corr with target|"]) == {.99, .2}
    assert not data_plots.correlation_inventory(pd.DataFrame()).columns.empty
