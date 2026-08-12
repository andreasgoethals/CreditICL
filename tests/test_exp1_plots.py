"""The Exp1 figures — the ones that answer "which prior is best?".

These exist because of the failure mode they replaced: a figure can build, look professional,
and answer nothing. So the assertions are about the QUANTITY each figure encodes, not about the
drawing. Pixel comparison would only break on matplotlib upgrades.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.visualize import exp1_plots as e1

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


class FakeTask:
    """A synthetic dataset, shaped like `SyntheticTask`."""

    def __init__(self, y, n_features=6, mechanism=None, seed=0):
        rng = np.random.default_rng(seed)
        y = np.asarray(y, dtype=np.float32)
        self.y = torch.tensor(y)
        # One informative feature, so a predictability score is not pure noise.
        X = rng.normal(size=(len(y), n_features)).astype(np.float32)
        X[:, 0] = y + rng.normal(0, 0.1, len(y))
        self.X = torch.tensor(X)
        self.n_rows, self.n_features = len(y), n_features
        self.source = "credit"
        self.mechanism = mechanism


class FakeReal:
    def __init__(self, y, seed=0):
        rng = np.random.default_rng(seed)
        self.y = np.asarray(y, dtype=np.float32)
        X = rng.normal(size=(len(y), 5)).astype(np.float32)
        X[:, 0] = self.y + rng.normal(0, 0.2, len(y))
        self.X = X
        self.n_rows, self.n_features = len(y), 5


def _lgd_y(n=300, atom=0.2, seed=0):
    """A bounded target with atoms of a chosen size at both ends."""
    rng = np.random.default_rng(seed)
    y = rng.random(n)
    k = int(atom * n)
    y[:k] = 0.0
    y[k:2 * k] = 1.0
    return y


def _variants():
    return {
        "original": [FakeTask(_lgd_y(atom=0.0, seed=i), seed=i) for i in range(6)],
        "credit": [FakeTask(_lgd_y(atom=0.25, seed=i), seed=i) for i in range(6)],
    }


def _real():
    return {"0001.a": FakeReal(_lgd_y(atom=0.25, seed=99)),
            "0002.b": FakeReal(_lgd_y(atom=0.05, seed=98))}


# -- the distance metric, which the whole ranking rests on ---------------------


def test_distance_is_zero_for_identical_and_one_for_disjoint():
    a = np.concatenate([np.zeros(50), np.full(50, 0.1)])
    b = np.full(100, 0.9)
    assert e1.distribution_distance(a, a) == pytest.approx(0.0)
    assert e1.distribution_distance(a, b) == pytest.approx(1.0)


def test_distance_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    a, b = rng.random(400), rng.beta(2, 5, 400)
    d = e1.distribution_distance(a, b)
    assert 0.0 <= d <= 1.0
    assert d == pytest.approx(e1.distribution_distance(b, a))


def test_distance_notices_a_different_atom_size():
    """Why total variation and not a KS statistic: LGD's whole story is point masses, and a
    CDF-based metric treats a spike as a step and understates the difference."""
    rng = np.random.default_rng(0)
    small = np.concatenate([np.zeros(10), rng.random(390)])
    large = np.concatenate([np.zeros(200), rng.random(200)])
    assert e1.distribution_distance(small, large) > 0.2


def test_the_prior_closer_to_real_data_scores_lower():
    """The ranking must actually rank. Inverted, the money figure would be backwards."""
    real = _lgd_y(atom=0.25, seed=7)
    variants = _variants()
    close = np.concatenate([t.y.numpy() for t in variants["credit"]])
    far = np.concatenate([t.y.numpy() for t in variants["original"]])
    assert e1.distribution_distance(close, real) < e1.distribution_distance(far, real)


# -- each figure encodes what it claims to ------------------------------------


def test_realism_ranking_orders_variants_best_first():
    fig = e1.plot_prior_realism_ranking(_variants(), _real(), task="lgd")
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    # The y axis is inverted, so the first label is the closest to real data.
    assert labels[0] == "credit", f"expected the closer prior first, got {labels}"


def test_realism_ranking_survives_having_no_real_data():
    """A machine without the datasets must still run the notebook."""
    assert e1.plot_prior_realism_ranking(_variants(), None, task="lgd") is not None


def test_mechanism_decomposition_makes_one_panel_per_mechanism():
    tasks = (
        [FakeTask(_lgd_y(atom=0.4, seed=i), mechanism="collateral", seed=i) for i in range(3)]
        + [FakeTask(_lgd_y(atom=0.1, seed=i), mechanism="workout", seed=i) for i in range(3)]
    )
    assert len(e1.plot_mechanism_decomposition(tasks).axes) == 2


def test_mechanism_decomposition_handles_unlabelled_tasks():
    """A `quantile`-mode arm carries no mechanism labels; the figure must degrade, not raise."""
    fig = e1.plot_mechanism_decomposition([FakeTask(_lgd_y(), seed=i) for i in range(4)])
    assert len(fig.axes) == 1


def test_default_clustering_marks_the_independence_line():
    rng = np.random.default_rng(0)
    variants = {
        "original": [FakeTask((rng.random(400) < 0.1).astype(float), seed=i) for i in range(4)],
        # Correlated: whole blocks default together, which is what a factor model produces.
        "credit": [FakeTask(np.repeat((rng.random(20) < 0.3).astype(float), 20), seed=i)
                   for i in range(4)],
    }
    ax = e1.plot_default_clustering(variants, None).axes[0]
    ys = [line.get_ydata()[0] for line in ax.get_lines() if len(line.get_ydata())]
    assert any(abs(y - 1.0) < 1e-9 for y in ys), "the independence reference must be drawn"


def test_clustered_defaults_score_above_independent_ones():
    """The figure's whole claim. If a correlated prior did not score higher than an independent
    one, the Vasicek mechanism is not doing what we say it does."""
    rng = np.random.default_rng(1)
    indep = (rng.random(600) < 0.2).astype(float)
    waves = np.repeat((rng.random(12) < 0.2).astype(float), 50)

    def ratio(y):
        rates = np.array([b.mean() for b in np.array_split(y, 12)])
        base, per = y.mean(), len(y) // 12
        return np.std(rates) / max(np.sqrt(base * (1 - base) / per), 1e-12)

    assert ratio(waves) > ratio(indep)


def test_difficulty_calibration_clips_a_wild_outlier():
    """A real dataset scoring R^2 = -4.8 must not flatten the axis into uselessness — which is
    what min-max on the real scores did before."""
    fig = e1.plot_difficulty_calibration(
        _variants(), {"a": 0.3, "b": 0.25, "c": -4.8}, task="lgd", max_datasets=4
    )
    bottom, _ = fig.axes[0].get_ylim()
    assert bottom >= -1.0, f"axis floor {bottom} lets one outlier dominate"


def test_side_by_side_tables_separates_the_target_column():
    fig = e1.plot_side_by_side_tables(FakeTask(_lgd_y()), FakeReal(_lgd_y(seed=3)), n_cols=5)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels[-1] == "y", "the target must be labelled and last"
    assert len(fig.axes) == 2, "one panel for synthetic, one for real"


def test_side_by_side_tables_works_without_real_data():
    assert len(e1.plot_side_by_side_tables(FakeTask(_lgd_y()), None).axes) == 1


def test_boundary_mass_sources_puts_each_atom_on_its_own_axis():
    fig = e1.plot_boundary_mass_sources(_variants(), _real())
    ax = fig.axes[0]
    assert "0" in ax.get_xlabel() and "1" in ax.get_ylabel()
    assert len(fig.axes) == 2


# -- the uninformative figures are gone ---------------------------------------


@pytest.mark.parametrize("gone", ["plot_target_grid", "plot_feature_relationships"])
def test_the_uninformative_figures_were_removed(gone):
    """`plot_target_grid` drew 100 thumbnails nobody can compare; `plot_feature_relationships`
    showed only that random graphs produce random correlations."""
    from src.visualize import prior_plots

    assert not hasattr(prior_plots, gone), f"{gone} is still there"


def test_the_pd_notebook_does_not_draw_a_target_histogram():
    """A PD target histogram is a bar at 0 and a bar at 1 — the default rate, drawn as a
    picture. The base-rate figure says it properly, and clustering says the part that matters."""
    nb = json.loads((ROOT / "notebooks" / "prior_visualisation_pd.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "plot_target_shapes_by_variant" not in code
    assert "plot_target_comparison" in code or "plot_base_rate_by_variant" in code
    assert "plot_default_clustering" in code, "PD's clustering figure must be present"


def test_both_notebooks_visualise_exp1_because_that_is_the_sweep():
    """Exp2 runs one prior and Exp3 sweeps a mixture; the 32-prior sweep is Exp1, so that is
    the config these figures describe. Exp2/Exp3 configs would also refuse to load."""
    for track in ("lgd", "pd"):
        nb = json.loads(
            (ROOT / "notebooks" / f"prior_visualisation_{track}.ipynb").read_text(encoding="utf-8")
        )
        code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert f"config/Exp1_{track.upper()}.yaml" in code
