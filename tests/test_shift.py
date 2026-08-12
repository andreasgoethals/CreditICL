"""Shift stress — the third of O'Prior's three independent contributors.

Every other part of the prior draws the context and the predicted rows from the same
population. Real credit scoring never does: a scorecard is built on past loans and
applied to next quarter's applications.

Two properties matter most here, and both are ways this could silently be useless:

* the **control arm must be untouched** — `credit_fraction=0` has to stay exactly
  TabICL's prior, or the comparison measures two changes at once;
* the **feature-to-target relationship must survive** — shifting the inputs is a hard
  but solvable problem; breaking the function would make the task unlearnable and the
  model would simply learn to ignore its context.
"""

from __future__ import annotations

import copy
from collections import Counter

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.prior.rng import PriorRNG
from src.prior.shift import SHIFT_KINDS, apply_shift

CFG = {"shift_prob": 1.0, "prior_prob_range": [0.15, 0.45]}


@pytest.fixture
def table():
    """400 rows where y depends on X[:, 0], so a broken relationship is detectable."""
    rng = PriorRNG(0)
    X = rng.randn(400, 5)
    y = (X[:, 0] * 0.8 + rng.randn(400) * 0.2).sigmoid()
    return X, y


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def test_disabled_by_default(table):
    X, y = table
    Xo, yo, meta = apply_shift(PriorRNG(1), X, y, {})
    assert meta["shift"] == "none"
    assert torch.equal(Xo, X) and torch.equal(yo, y)


def test_probability_is_honoured(table):
    X, y = table
    rng = PriorRNG(2)
    kinds = Counter(
        apply_shift(rng, X, y, {**CFG, "shift_prob": 0.3})[2]["shift"] for _ in range(400)
    )
    shifted = sum(v for k, v in kinds.items() if k != "none")
    assert 0.2 < shifted / 400 < 0.4, kinds


def test_every_kind_is_reachable(table):
    """A dispatcher that quietly always picked one kind would undercut the design."""
    X, y = table
    rng = PriorRNG(3)
    seen = {apply_shift(rng, X, y, CFG)[2]["shift"] for _ in range(200)}
    assert set(SHIFT_KINDS) <= seen, f"only saw {seen}"


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_rows_are_rearranged_never_invented(table, kind):
    """Only the ORDER changes. If rows were duplicated or dropped, the table's own
    marginal would change and the shift would be confounded with a resample."""
    X, y = table
    weights = {k: (1.0 if k == kind else 0.0) for k in SHIFT_KINDS}
    Xo, yo, meta = apply_shift(PriorRNG(4), X, y, {**CFG, "kind_weights": weights})
    if meta["shift"] == "none":
        pytest.skip("this draw declined to shift")
    assert Xo.shape == X.shape and yo.shape == y.shape
    assert torch.allclose(yo.sort().values, y.sort().values), "y multiset changed"


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_the_feature_target_relationship_survives(table, kind):
    """THE test. Shifting inputs is solvable; breaking the function is not, and a model
    trained on unlearnable tasks learns to ignore its context entirely."""
    X, y = table
    weights = {k: (1.0 if k == kind else 0.0) for k in SHIFT_KINDS}
    Xo, yo, meta = apply_shift(PriorRNG(5), X, y, {**CFG, "kind_weights": weights})
    if meta["shift"] == "none":
        pytest.skip("this draw declined to shift")
    before = abs(_spearman(X[:, 0], y))
    after = abs(_spearman(Xo[:, 0], yo))
    assert after > 0.9 * before, f"{kind}: relationship weakened {before:.3f} -> {after:.3f}"


def test_covariate_shift_makes_the_query_range_unseen(table):
    """The point of sorting rather than adding an offset: the query must fall OUTSIDE
    the feature range the context showed, so it needs real extrapolation."""
    X, y = table
    weights = {"covariate": 1.0, "cohort": 0.0, "prior_prob": 0.0}
    for seed in range(10):
        Xo, yo, meta = apply_shift(PriorRNG(seed), X, y, {**CFG, "kind_weights": weights})
        if meta["shift"] != "covariate":
            continue
        col, cut = meta["shift_feature"], meta["shift_cut"]
        ctx, qry = Xo[:cut, col], Xo[cut:, col]
        assert ctx.mean() != qry.mean()
        # One side's whole range sits beyond the other's midpoint.
        assert ctx.max() <= qry.max() or ctx.min() >= qry.min()
        return
    pytest.skip("no covariate draw in 10 tries")


def test_prior_prob_shift_moves_the_base_rate(table):
    """The dangerous shift: a model anchoring on the context's rate is miscalibrated."""
    X, y = table
    weights = {"prior_prob": 1.0, "cohort": 0.0, "covariate": 0.0}
    for seed in range(10):
        _, _, meta = apply_shift(PriorRNG(seed), X, y, {**CFG, "kind_weights": weights})
        if meta["shift"] != "prior_prob":
            continue
        assert abs(meta["context_high_rate"] - meta["query_high_rate"]) > 0.1
        return
    pytest.skip("no prior_prob draw in 10 tries")


def test_tiny_tables_decline_gracefully():
    """Too few rows to split meaningfully must return the table unchanged, not crash."""
    rng = PriorRNG(6)
    X, y = rng.randn(10, 3), rng.randn(10)
    Xo, yo, meta = apply_shift(rng, X, y, CFG)
    assert Xo.shape == X.shape and yo.shape == y.shape
    assert torch.isfinite(Xo).all()


def test_constant_feature_declines_covariate_shift():
    """Sorting on a constant column would produce an arbitrary order, not a shift."""
    rng = PriorRNG(7)
    X = torch.ones(200, 3)
    y = rng.randn(200)
    _, _, meta = apply_shift(rng, X, y, {**CFG, "kind_weights": {"covariate": 1.0}})
    assert meta["shift"] == "none"


def test_reproducible(table):
    X, y = table
    a = apply_shift(PriorRNG(8), X, y, CFG)
    b = apply_shift(PriorRNG(8), X, y, CFG)
    assert torch.equal(a[0], b[0]) and a[2] == b[2]


# -- integration: the control arm must not move ------------------------------


@pytest.mark.parametrize("task", ["lgd", "pd"])
def test_control_arm_is_never_shifted(task):
    """credit_fraction=0 must stay EXACTLY TabICL's prior. If shift leaked into the
    control, every arm would differ from TabICL by two changes instead of one and the
    comparison would answer a different question than the one we asked."""
    from src.prior.generator import TaskGenerator
    from src.utils.config import expand_with_seeds, load

    cfg = expand_with_seeds(load(f"config/Exp1_{task.upper()}.yaml"))[0]["prior"]
    control = copy.deepcopy(cfg)
    control["credit_fraction"] = 0.0
    gen = TaskGenerator(control, task, PriorRNG(0))
    kinds = {gen.sample().meta.get("shift", "none") for _ in range(40)}
    assert kinds == {"none"}, f"shift leaked into the control arm: {kinds}"


@pytest.mark.parametrize("task", ["lgd", "pd"])
def test_credit_arm_gets_shifted_datasets(task):
    from src.prior.generator import TaskGenerator
    from src.utils.config import expand_with_seeds, load

    cfg = expand_with_seeds(load(f"config/Exp1_{task.upper()}.yaml"))[0]["prior"]
    ours = copy.deepcopy(cfg)
    ours["credit_fraction"] = 1.0
    gen = TaskGenerator(ours, task, PriorRNG(0))
    kinds = Counter(gen.sample().meta.get("shift", "none") for _ in range(120))
    shifted = sum(v for k, v in kinds.items() if k != "none")
    assert shifted > 10, f"shift barely fires: {kinds}"
    assert len({k for k in kinds if k != "none"}) >= 2, f"only one kind appears: {kinds}"


def test_shift_is_configurable_from_the_yaml():
    """A lever unreachable from config is dead code."""
    from src.utils.config import load

    for path in ("config/Exp1_LGD.yaml", "config/Exp1_PD.yaml"):
        shift = load(path)["prior"]["credit"].get("shift")
        assert shift is not None, f"{path}: no shift block under prior.credit"
        assert shift["shift_prob"] > 0, f"{path}: shift stress is disabled"
