"""The notebooks' logic.

Notebooks are the least-tested part of most projects: nothing runs them in CI, so a
figure silently breaks and nobody notices until they open it months later. Since all
the logic lives in `src/visualize/`, it can be tested like anything else — these
tests assert the figures BUILD and carry the right structure, not what they look
like. Pixel comparison would fail on every matplotlib upgrade for no benefit.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")  # no display on a cluster node, and none wanted in a test

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test here makes figures; matplotlib warns after 20 stay open."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def tasks():
    from src.visualize.prior_plots import sample_tasks

    drawn, _ = sample_tasks("config/LGD.yaml", n=8, credit_fraction=1.0, seed=0)
    return drawn


# -- style -------------------------------------------------------------------


def test_style_is_idempotent():
    """A notebook re-runs its setup cell constantly; that must not accumulate state."""
    style.use_style()
    first = dict(matplotlib.rcParams)
    style.use_style()
    assert dict(matplotlib.rcParams) == first


def test_colours_are_distinct_and_meaningful():
    """CREDIT vs ORIGINAL is the comparison the whole project makes. If those two
    were ever set to the same value every figure would silently lose its point."""
    assert style.CREDIT != style.ORIGINAL != style.REAL
    assert style.source_color("credit") == style.CREDIT
    assert style.source_color("base") == style.ORIGINAL
    assert style.source_color("anything else") == style.ORIGINAL


def test_saved_figures_are_opaque():
    """A transparent PNG dropped into a dark slide turns all the text invisible."""
    style.use_style()
    assert matplotlib.rcParams["savefig.facecolor"] == "white"


def test_series_palette_has_enough_colours():
    """21 datasets, but plots distinguish at most a handful; 8 is the working limit."""
    assert len(style.SERIES) >= 8
    assert len(set(style.SERIES)) == len(style.SERIES), "no duplicate colours"


def test_savefig_creates_missing_directories(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = style.savefig(fig, str(tmp_path / "deep" / "nested" / "f.png"))
    assert (tmp_path / "deep" / "nested" / "f.png").is_file(), out


def test_palette_swatch_builds():
    assert style.show_palette() is not None


# -- prior plots -------------------------------------------------------------


def test_sample_tasks_honours_the_credit_fraction_override():
    """The notebook's whole comparison rests on being able to force each side."""
    from src.visualize.prior_plots import sample_tasks

    base, info_b = sample_tasks("config/LGD.yaml", n=6, credit_fraction=0.0, seed=0)
    ours, info_o = sample_tasks("config/LGD.yaml", n=6, credit_fraction=1.0, seed=0)
    assert info_b["sources"]["credit"] == 0
    assert info_o["sources"]["base"] == 0
    assert all(t.source == "credit" for t in ours)
    assert len(base) == len(ours) == 6


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("plot_target_grid", {"n_show": 8, "ncols": 4}),
        ("plot_boundary_mass", {}),
        ("plot_table_shapes", {}),
        ("plot_feature_relationships", {"n_show": 3}),
        ("plot_correlation_spectrum", {}),
        ("plot_feature_target_relation", {"n_show": 4}),
    ],
)
def test_prior_figures_build(tasks, name, kwargs):
    import src.visualize.prior_plots as pp

    fig = getattr(pp, name)(tasks, **kwargs)
    assert fig is not None and len(fig.axes) >= 1


def test_boundary_mass_plot_takes_a_real_reference(tasks):
    """The stars are the real datasets; the overlay must not crash on them."""
    from src.visualize.prior_plots import plot_boundary_mass

    fig = plot_boundary_mass(tasks, real_reference={"heloc": (0.21, 0.52)})
    assert fig is not None


def test_target_grid_handles_fewer_tasks_than_cells(tasks):
    """Asking for 100 panels with 8 tasks must blank the rest, not raise."""
    from src.visualize.prior_plots import plot_target_grid

    fig = plot_target_grid(tasks, n_show=100, ncols=10)
    assert fig is not None


def test_compare_priors_returns_a_summary_that_shows_the_difference():
    """This is the motivating figure, and its summary is what goes in a results file.

    The assertion is the project's premise: the original prior puts essentially
    nothing in [0,1] with atoms, ours does.
    """
    from src.visualize.prior_plots import compare_priors

    fig, summary = compare_priors("config/LGD.yaml", n=8, seed=0)
    assert fig is not None
    assert set(summary) >= {"original", "ours"}
    assert summary["ours"]["tasks_in_unit_interval"] > summary["original"]["tasks_in_unit_interval"]


# -- data plots --------------------------------------------------------------


def test_summary_table_columns_differ_by_task():
    """LGD is about boundary mass, PD about base rate. Showing base_rate for LGD
    (or vice versa) was the shape of a real earlier bug."""
    from src.visualize.data_plots import summary_table

    class FakeDS:
        def __init__(self, y):
            self.y = np.asarray(y, dtype=np.float32)
            self.X = np.zeros((len(y), 3), dtype=np.float32)
            self.n_rows, self.n_features = len(y), 3
            self.cat_indices = [2]
            self.feature_names = ["a", "b", "c"]

    lgd = summary_table({"0001.x": FakeDS([0.0, 0.0, 0.5, 1.0])}, "lgd")
    assert "boundary mass" in lgd.columns and "base rate" not in lgd.columns
    assert lgd["boundary mass"].iloc[0] == pytest.approx(0.75)

    pdt = summary_table({"0002.y": FakeDS([0.0, 0.0, 0.0, 1.0])}, "pd")
    assert "base rate" in pdt.columns and "boundary mass" not in pdt.columns
    assert pdt["base rate"].iloc[0] == pytest.approx(0.25)
    assert pdt["imbalance 1:n"].iloc[0] == pytest.approx(3.0)


def test_target_stats_accepts_numpy_and_torch():
    """The data path hands numpy, the prior path hands torch. Requiring the caller to
    remember which broke the exploration notebook on its first run."""
    import torch

    from src.utils.target_stats import target_stats

    y = [0.0, 0.0, 0.5, 1.0]
    a = target_stats(np.asarray(y, dtype=np.float32))
    b = target_stats(torch.tensor(y))
    assert a["frac_at_min"] == b["frac_at_min"] == pytest.approx(0.5)
    assert a["frac_at_max"] == b["frac_at_max"] == pytest.approx(0.25)


def test_leakage_check_flags_a_planted_copy_of_the_target():
    """The screen has to actually fire, or it is decoration."""
    from src.visualize.data_plots import leakage_check

    rng = np.random.default_rng(0)
    y = rng.random(300).astype(np.float32)

    class DS:
        def __init__(self):
            noise = rng.normal(size=300).astype(np.float32)
            self.X = np.column_stack([y, noise]).astype(np.float32)  # column 0 IS y
            self.y = y
            self.feature_names = ["leaky", "noise"]

    flags = leakage_check({"0001.x": DS()}, "lgd", top_k=2)
    top = flags.iloc[0]
    assert top["feature"] == "leaky"
    assert bool(top["suspicious"]) is True


@pytest.mark.slow
def test_real_data_figures_build():
    """The exploration notebook end to end, if the raw data is present."""
    from src.utils.paths import find_raw_path
    from src.visualize import data_plots

    if find_raw_path("lgd", "0003.axa") is None:
        pytest.skip("raw data not present")
    datasets = data_plots.load_all("lgd", verbose=False)
    if not datasets:
        pytest.skip("no LGD datasets could be loaded")
    assert data_plots.plot_lgd_targets(datasets) is not None
    assert data_plots.plot_boundary_mass_ranking(datasets) is not None
    assert data_plots.plot_shapes({"lgd": datasets}) is not None
    assert data_plots.plot_feature_correlations(datasets, n_show=2) is not None


# -- notebooks themselves ----------------------------------------------------


# Notebook structure is covered by tests/test_summaries.py, which knows about the
# per-task split (prior_visualisation_lgd / _pd) and additionally checks that each one
# ENDS with a printed text summary. The old test here hard-coded the single combined
# notebook name and broke when it was split.
