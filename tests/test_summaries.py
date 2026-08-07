"""The text summaries every notebook ends with.

Repo-wide rule: a notebook's last cell prints text, not a figure, so the whole picture
survives a copy-paste into a message. These tests protect the part that makes that
worth doing — that the summary carries **numbers**, and that it does not overstate.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import numpy as np
import torch

from src.prior.base import SyntheticTask
from src.visualize import summaries


def _tasks(values: list[list[float]]) -> list[SyntheticTask]:
    return [
        SyntheticTask(X=torch.randn(len(v), 4), y=torch.tensor(v, dtype=torch.float32))
        for v in values
    ]


@pytest.fixture
def lgd_loaded():
    """Two arms: a continuous control, and ours with real atoms.

    Row counts must be realistic (400, not 4). `frac_at_min` is the share of rows on
    the minimum VALUE, so on a 4-row target it is 25% by construction and even a
    perfectly continuous control would read as having heavy atoms.
    """
    rng = np.random.default_rng(0)
    continuous = [list(rng.uniform(0.2, 0.8, 400)) for _ in range(4)]
    with_atoms = [[0.0] * 100 + list(rng.uniform(0.2, 0.8, 200)) + [1.0] * 100 for _ in range(4)]
    return {"original": _tasks(continuous), "credit_v1": _tasks(with_atoms)}


def test_lgd_summary_reports_the_boundary_numbers(lgd_loaded):
    out = summaries.prior_summary(lgd_loaded, "lgd")
    assert "boundary mass" in out
    assert "in [0,1]" in out
    assert "base rate" not in out, "PD's metric must not appear in an LGD summary"
    assert "original" in out and "credit_v1" in out


def test_lgd_summary_separates_the_two_arms(lgd_loaded):
    """The whole point of the figure, in text: ours has atoms, the control does not."""
    out = summaries.prior_summary(lgd_loaded, "lgd")
    original_block = out.split("original:")[1].split("credit_v1:")[0]
    credit_block = out.split("credit_v1:")[1]
    assert "any atoms      0.0%" in original_block
    assert "any atoms      100.0%" in credit_block


def test_pd_summary_reports_base_rate_not_boundary_mass():
    loaded = {"v": _tasks([[0.0] * 9 + [1.0]])}
    out = summaries.prior_summary(loaded, "pd")
    assert "base rate" in out
    assert "boundary mass" not in out
    assert "Vasicek" in out, "the PD summary should name the mechanism in use"


def test_summary_warns_that_range_coverage_is_weak(lgd_loaded):
    """A range can span a real value while placing almost no mass near it. The
    original prior 'spans' 6 of 7 real datasets on the strength of outlier draws while
    its median boundary mass is 0.3%. Reporting only coverage would overstate both
    arms, so the summary must say which column to read.
    """
    ref = {"heloc": (0.211, 0.519), "lendingclub": (0.015, 0.003)}
    out = summaries.prior_summary(lgd_loaded, "lgd", reference=ref)
    assert "spans" in out and "within 5pp" in out
    assert "weaker statement than it looks" in out


def test_summary_names_the_data_source(lgd_loaded):
    """A figure made from a live draw must never be mistaken for one made from the
    pool the model actually trained on."""
    out = summaries.prior_summary(lgd_loaded, "lgd", source="live")
    assert "live" in out
    assert "pool" in out, "the summary should explain what the other source means"


def test_summary_carries_a_caveat_about_transfer(lgd_loaded):
    """The prior looking closer is not the same as it transferring. Pasted into a
    message without that line, the summary would read as a result."""
    out = summaries.prior_summary(lgd_loaded, "lgd")
    assert "not downstream performance" in out


def test_summary_is_plain_text(lgd_loaded):
    out = summaries.prior_summary(lgd_loaded, "lgd")
    assert isinstance(out, str) and "\n" in out
    assert "<" not in out and "matplotlib" not in out


def test_summary_survives_an_empty_variant():
    """A pool that failed to load must not crash the last cell of the notebook."""
    out = summaries.prior_summary({"a": [], "b": _tasks([[0.0, 1.0]])}, "lgd")
    assert "PRIOR SUMMARY" in out


# -- data summary ------------------------------------------------------------


class _DS:
    def __init__(self, y, n_features=4, cat=(3,), missing=0.0):
        self.y = np.asarray(y, dtype=np.float32)
        self.X = np.zeros((len(y), n_features), dtype=np.float32)
        if missing:
            self.X[0, 0] = np.nan
        self.n_rows, self.n_features = len(y), n_features
        self.cat_indices = list(cat)
        self.feature_names = [f"f{i}" for i in range(n_features)]


def test_data_summary_covers_both_tasks():
    both = {
        "lgd": {"0001.a": _DS([0.0, 0.0, 0.5, 1.0])},
        "pd": {"0002.b": _DS([0.0, 0.0, 0.0, 1.0])},
    }
    out = summaries.data_summary(both)
    assert "LGD" in out and "PD" in out
    assert "boundary mass" in out and "base rate" in out
    assert "2 datasets loaded" in out


def test_data_summary_states_the_odds_for_pd():
    """A base rate as odds is easier to feel than a percentage."""
    out = summaries.data_summary({"pd": {"0002.b": _DS([0.0] * 9 + [1.0])}})
    assert "1 default per 9 non-defaults" in out


def test_data_summary_flags_a_suspicious_leak():
    import pandas as pd

    leak = pd.DataFrame(
        [{"dataset": "0007.x", "feature": "leaky", "|corr with target|": 0.99, "suspicious": True}]
    )
    out = summaries.data_summary({"lgd": {"0001.a": _DS([0.0, 1.0])}}, leakage=leak)
    assert "INSPECT" in out and "leaky" in out


def test_data_summary_says_so_when_nothing_is_flagged():
    import pandas as pd

    leak = pd.DataFrame(
        [{"dataset": "0007.x", "feature": "ok", "|corr with target|": 0.4, "suspicious": False}]
    )
    out = summaries.data_summary({"lgd": {"0001.a": _DS([0.0, 1.0])}}, leakage=leak)
    assert "Nothing above 0.9" in out


def test_data_summary_warns_against_tuning_to_missingness():
    """Most of these datasets were imputed upstream, so their missingness measures the
    pipeline that produced them, not the domain."""
    out = summaries.data_summary({"lgd": {"0001.a": _DS([0.0, 1.0])}})
    assert "pre-imputed" in out or "upstream" in out


# -- the notebooks themselves ------------------------------------------------


@pytest.mark.parametrize(
    "name", ["prior_visualisation_lgd", "prior_visualisation_pd", "data_exploration"]
)
def test_notebook_exists_ends_with_a_text_summary_and_holds_no_logic(name):
    """One notebook per task, each ending in text. The no-logic rule is what makes the
    plots testable at all."""
    import json
    import pathlib

    path = pathlib.Path("notebooks") / f"{name}.ipynb"
    assert path.is_file(), f"{name} is missing"
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4

    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for c in code:
        body = "".join(c["source"])
        assert "def " not in body, f"{name}: logic belongs in src/visualize/"
        assert len(c["source"]) == 1 or c["source"][0].endswith("\n"), "nbformat line endings"

    last = "".join(code[-1]["source"])
    assert "summaries." in last and "print(" in last, (
        f"{name} must END with a printed text summary — the repo rule is that a "
        f"notebook's final output is copy-pasteable text, not a figure"
    )


def test_the_old_combined_notebook_is_gone():
    """Superseded by one notebook per task; leaving it would let the two drift."""
    import pathlib

    assert not (pathlib.Path("notebooks") / "prior_visualisation.ipynb").exists()


def test_data_notebook_does_not_show_the_prior_palette():
    """`show_palette` explains the prior colours (ours vs TabICL's). The data notebook
    is about the datasets we EVALUATE on, where those colours mean nothing."""
    import json
    import pathlib

    nb = json.loads(
        (pathlib.Path("notebooks") / "data_exploration.ipynb").read_text(encoding="utf-8")
    )
    body = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "show_palette" not in body
