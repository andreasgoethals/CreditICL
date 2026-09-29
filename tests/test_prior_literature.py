"""The literature-grounded credit prior (29-09-2026): every number from the credit-risk literature
or deliberately wide, none from the evaluation datasets (docs/PRIORS.md, section 5).

These pin the published values where the config claims them, the mechanisms that carry them
(the ZOIB LGD process, the Vasicek PD process with its period factor), and the raw encoding the
TabPFN arms train on.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _prior(track: str, **overrides) -> dict:
    from src.utils.config import expand_with_seeds, load

    prior = copy.deepcopy(expand_with_seeds(load(ROOT / "config" / f"Exp1_{track.upper()}.yaml"))[0]["prior"])
    prior.update(n_rows_range=[256, 256], n_features_range=[4, 10], max_features=16,
                 n_nodes_range=[2, 6], max_filter_attempts=4, credit_fraction=1.0)
    prior.update(overrides)
    return prior


def _tasks(track: str, n: int = 12, **overrides):
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG

    gen = TaskGenerator(_prior(track, **overrides), track, PriorRNG(3))
    return [gen.sample() for _ in range(n)]


# -- the configs say what the literature says ---------------------------------------------


@pytest.mark.parametrize("track", ["PD", "LGD"])
def test_the_credit_prior_is_not_calibrated_to_the_evaluation_data(track):
    """The words 'calibrated' and 'real:' (a real-data statistic) must not reappear next to a
    prior knob: that is exactly the leakage the redesign removed."""
    text = (ROOT / "config" / f"Exp1_{track}.yaml").read_text(encoding="utf-8")
    credit = text[text.index("\n  credit:"):text.index("\n  filter:")]
    assert "calibrat" not in credit.lower()
    assert "(real:" not in credit and "real data" not in credit.lower()


def test_pd_uses_the_basel_correlations_and_the_brown_mues_imbalance_range():
    from src.utils.config import load

    t = load(ROOT / "config" / "Exp1_PD.yaml")["prior"]["credit"]["target"]
    mech = t["mechanism"]
    assert t["mode"] == "mechanism"
    assert mech["rho_range"] == [0.03, 0.24], "the Basel IRB range, other retail to corporate"
    assert mech["base_rate_range"] == [0.01, 0.50] and mech["base_rate_log"] is True
    assert mech["cohort"]["cohort_sd_range"] == [1.0, 1.0], "the Vasicek factor is standard normal"
    # no label noise: the rates had no source
    assert "flip_pos_to_neg_range" not in t and "flip_neg_to_pos" not in t
    assert t["selection"]["selection_drop_range"] == [0.0, 0.5]


def test_lgd_is_the_zoib_process_centred_on_the_published_values():
    from src.utils.config import load

    t = load(ROOT / "config" / "Exp1_LGD.yaml")["prior"]["credit"]["target"]
    z = t["zoib"]
    assert t["mode"] == "zoib"
    assert z["alpha0_mean_sd"][0] == -0.54 and z["beta0_mean_sd"][0] == -1.46 and z["gamma0_mean_sd"][0] == 0.0
    lo, hi = z["phi_range"]
    assert lo <= 5 <= hi, "the published precision must lie inside the range"
    for key, published in (("loading_p0_range", 0.6), ("loading_p1_range", 0.15), ("loading_mu_range", 0.15),
                           ("macro_p0_range", -0.09), ("macro_p1_range", 0.11), ("macro_mu_range", 0.01)):
        lo, hi = z[key]
        assert lo <= published <= hi, f"{key} {z[key]} must contain the published {published}"


@pytest.mark.parametrize("track", ["PD", "LGD"])
def test_both_filters_judge_only_tabicls_own_tables(track):
    from src.utils.config import load

    assert load(ROOT / "config" / f"Exp1_{track}.yaml")["prior"]["filter"]["apply_to"] == "base"


def test_a_mean_sd_pair_is_data_not_a_sweep():
    """`alpha0: [-0.54, 1.0]` would have been read as a two-point sweep and multiplied the grid."""
    from src.utils.config import expand_with_seeds, is_literal_list, load

    assert is_literal_list("alpha0_mean_sd", [-0.54, 1.0])
    assert not is_literal_list("alpha0", [-0.54, 1.0])
    assert len(expand_with_seeds(load(ROOT / "config" / "Exp1_LGD.yaml"))) == 9


# -- the ZOIB process ------------------------------------------------------------------------


def _zoib(seed=0, n=2000, cfg=None, X=None):
    from src.prior.rng import PriorRNG
    from src.prior.targets.lgd import apply_lgd_zoib

    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 6, generator=g) if X is None else X
    latent = X[:, 0] + 0.3 * torch.randn(n, generator=g)
    return apply_lgd_zoib(PriorRNG(seed), X, latent, cfg or {}, 16)


def test_zoib_targets_are_losses_with_atoms_at_both_ends():
    for seed in range(8):
        _, y, meta = _zoib(seed)
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
        inner = y[(y > 0) & (y < 1)]
        assert len(inner) > 0, "the interior beta must be there"
        assert float((inner <= 0).sum() + (inner >= 1).sum()) == 0
        assert meta["frac_at_0"] == pytest.approx(float((y == 0).float().mean()), abs=1e-4)
        assert meta["frac_at_1"] == pytest.approx(float((y == 1).float().mean()), abs=1e-4)


def test_zoib_at_the_published_values_reproduces_their_shape():
    """Li, Zhang & Zhao's portfolio: intercepts -0.54 / -1.46, phi 5, weak covariates, no spread —
    about a quarter of the losses at 0, a fifth at 1, the rest interior (their Figure 1)."""
    fixed = {"alpha0_mean_sd": [-0.54, 0.0], "beta0_mean_sd": [-1.46, 0.0], "gamma0_mean_sd": [0.0, 0.0],
             "phi_range": [5.0, 5.0], "loading_p0_range": [0.6, 0.6], "loading_p1_range": [0.15, 0.15],
             "loading_mu_range": [0.15, 0.15], "macro_p0_range": [0.0, 0.0], "macro_p1_range": [0.0, 0.0],
             "macro_mu_range": [0.0, 0.0], "cohort": {"min_cohorts": 1, "max_cohorts": 1}}
    _, y, _ = _zoib(0, n=20000, cfg=fixed)
    p0, p1 = float((y == 0).float().mean()), float((y == 1).float().mean())
    # e^-0.54 / (1 + e^-0.54 + e^-1.46) = 0.31, e^-1.46 / (...) = 0.13 at the mean covariate
    assert 0.25 < p0 < 0.40 and 0.08 < p1 < 0.20, (p0, p1)


def test_a_downturn_means_fewer_full_recoveries_and_more_total_losses():
    """The published signs (Altman et al. 2005): with only the period factor moving, the tables
    of bad periods carry more ones and fewer zeros."""
    cfg = {"alpha0_mean_sd": [-0.5, 0.0], "beta0_mean_sd": [-1.5, 0.0], "loading_p0_range": [0.0, 0.0],
           "loading_p1_range": [0.0, 0.0], "loading_mu_range": [0.0, 0.0], "macro_p0_range": [-1.0, -1.0],
           "macro_p1_range": [1.0, 1.0], "macro_mu_range": [0.5, 0.5], "macro_feature_prob": 1.0,
           "cohort": {"min_cohorts": 20, "max_cohorts": 20}}
    X, y, meta = _zoib(1, n=20000, cfg=cfg)
    assert meta["macro_feature"] is True and X.shape[1] == 7
    macro = X[:, -1]
    bad, good = macro > 0.5, macro < -0.5
    assert float((y[bad] == 1).float().mean()) > float((y[good] == 1).float().mean())
    assert float((y[bad] == 0).float().mean()) < float((y[good] == 0).float().mean())


def test_zoib_recovery_driver_raises_full_recoveries():
    """A better recovery driver (the latent) raises P(0) and lowers P(1): the published signs."""
    cfg = {"loading_p0_range": [1.5, 1.5], "loading_p1_range": [1.5, 1.5], "component_overlap_range": [1.0, 1.0],
           "signal_share_range": [1.0, 1.0], "cohort": {"min_cohorts": 1, "max_cohorts": 1}}
    g = torch.Generator().manual_seed(5)
    X = torch.randn(20000, 4, generator=g)
    from src.prior.rng import PriorRNG
    from src.prior.targets.lgd import apply_lgd_zoib

    _, y, _ = apply_lgd_zoib(PriorRNG(2), X, X[:, 0], cfg, 16)
    high, low = X[:, 0] > 1, X[:, 0] < -1
    assert float((y[high] == 0).float().mean()) > float((y[low] == 0).float().mean())
    assert float((y[high] == 1).float().mean()) < float((y[low] == 1).float().mean())


def test_zoib_is_deterministic_in_its_seed():
    a, b, c = _zoib(4), _zoib(4), _zoib(5)
    assert torch.equal(a[1], b[1]) and not torch.equal(a[1], c[1])


def test_zoib_never_grows_a_full_table():
    from src.prior.rng import PriorRNG
    from src.prior.targets.lgd import apply_lgd_zoib

    X = torch.randn(500, 16)
    for seed in range(6):
        Xo, _, _ = apply_lgd_zoib(PriorRNG(seed), X, X[:, 0], {"macro_feature_prob": 1.0,
                                                              "cohort": {"min_cohorts": 5, "max_cohorts": 5}}, 16)
        assert Xo.shape[1] == 16, "no column may be added past max_features"


def test_the_generator_routes_zoib_through_the_features():
    for task in _tasks("lgd", 10):
        assert task.meta["mode"] == "zoib"
        assert torch.isfinite(task.y).all()


# -- the PD process ---------------------------------------------------------------------------


def test_the_default_rate_is_log_uniform_over_the_literature_range():
    """Log-uniform on [1 %, 50 %]: the median default rate near 7 %, not the 25 % of a uniform."""
    from src.prior.rng import PriorRNG
    from src.prior.targets.mechanisms import pd_vasicek

    rates = []
    for seed in range(400):
        _, meta = pd_vasicek(PriorRNG(seed), torch.randn(64),
                             {"base_rate_range": [0.01, 0.5], "base_rate_log": True, "rho_range": [0.03, 0.03]})
        rates.append(meta["target_base_rate"])
    rates = np.array(rates)
    assert rates.min() >= 0.01 and rates.max() <= 0.5
    assert 0.05 < np.median(rates) < 0.10, np.median(rates)


def test_selection_and_rules_are_drawn_per_table():
    from src.prior.rng import PriorRNG
    from src.prior.targets.pd import apply_threshold_rules, apply_underwriting_selection

    drops, rules = set(), set()
    X = torch.randn(400, 5)
    for seed in range(30):
        _, _, meta = apply_underwriting_selection(PriorRNG(seed), X, X[:, 0],
                                                  {"selection_drop_range": [0.0, 0.5],
                                                   "selection_sharpness_range": [0.3, 1.0]})
        drops.add(round(meta.get("selection_drop", 0.0), 3))
        _, rmeta = apply_threshold_rules(PriorRNG(seed), X, X[:, 0], {"n_rules_range": [0, 3]})
        rules.add(rmeta["n_rules"])
    assert len(drops) > 10, "the share rejected must vary between tables"
    assert rules == {0, 1, 2, 3}, "some books have no cut-offs, some three"


def test_the_period_factor_becomes_a_column_in_some_pd_tables():
    tasks = _tasks("pd", 24)
    flagged = [t for t in tasks if t.meta.get("macro_feature")]
    assert 0 < len(flagged) < len(tasks), "macro_feature_prob 0.5: some tables, not all"
    assert all(float(t.y.float().mean()) not in (0.0, 1.0) for t in tasks)


def test_pd_minority_shares_cover_the_imbalanced_regime():
    tasks = _tasks("pd", 30)
    shares = [min(float(t.y.mean()), 1 - float(t.y.mean())) for t in tasks]
    assert min(shares) < 0.05, "the 1-5 % regime must be present"
    assert max(shares) > 0.15


# -- the raw encoding (TabPFN) -----------------------------------------------------------------


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_raw_tables_keep_their_gaps_and_their_scale(track):
    raw = _tasks(track, 16, encoding="raw")
    tab = _tasks(track, 16, encoding="tabicl")
    raw_nan = [float(torch.isnan(t.X[:, : int(t.meta["d"])]).float().mean()) for t in raw]
    tab_nan = [float(torch.isnan(t.X).float().mean()) for t in tab]
    assert max(raw_nan) > 0, "a gap stays NaN for TabPFN"
    assert max(tab_nan) == 0, "TabICL's encoding fills every gap"
    assert all(not torch.isinf(t.X).any() for t in raw)
    assert all(t.meta.get("encoding") == "raw" and "prediction_view" not in t.meta for t in raw)
    if track == "lgd":
        assert all(float(t.y.min()) >= 0 and float(t.y.max()) <= 1 for t in raw), "raw LGD stays on [0, 1]"
        assert any(float(t.y.min()) < 0 for t in tab), "TabICL's LGD is standardised on the context"


def test_the_encoding_changes_the_encoding_and_nothing_else():
    """Same seed, same draws: the two encodings of a table hold the same labels, so the TabICL and
    TabPFN arms see the same synthetic tasks (common random numbers)."""
    raw = _tasks("pd", 8, encoding="raw")
    tab = _tasks("pd", 8, encoding="tabicl")
    for a, b in zip(raw, tab):
        assert torch.equal(a.y, b.y)


def test_constant_columns_are_judged_on_observed_values():
    from src.prior.generator import delete_constant_columns

    X = torch.tensor([[1.0, float("nan"), 2.0], [1.0, float("nan"), 3.0], [1.0, 4.0, float("nan")]])
    out, d = delete_constant_columns(X, 3)
    assert d == 1, "column 0 is constant, column 1 has one observed value, column 2 varies"


def test_an_unknown_encoding_is_refused():
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG

    with pytest.raises(ValueError, match="prior.encoding"):
        TaskGenerator(_prior("pd", encoding="tabpfn"), "pd", PriorRNG(0))


# -- two bugs the literature ranges exposed (29-09-2026) --------------------------------------


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_a_recorded_period_factor_never_widens_the_table(track):
    """The period column takes a noise column's place. When it widened the table past the slot's
    width, `delete_constant_columns` (which scans only that many) zeroed a real column."""
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG

    gen = TaskGenerator(_prior(track), track, PriorRNG(11))
    recorded = 0
    for i in range(30):
        width = 3 + i % 7
        task = gen.sample((256, width))
        assert int(task.meta["d"]) <= width
        recorded += bool(task.meta.get("macro_feature"))
    assert recorded, "some tables should record the period factor"


def test_a_repaired_table_is_encoded_on_its_final_context():
    """At a 1 % default rate the query often holds no default and the rows are permuted to fix
    it. That now happens BEFORE the encoding, so the context is centred as prediction centres it."""
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG

    prior = _prior("pd")
    prior["credit"]["target"]["mechanism"]["base_rate_range"] = [0.01, 0.02]
    prior["credit"]["shift"]["shift_prob"] = 0.0
    gen = TaskGenerator(prior, "pd", PriorRNG(2))
    repaired = 0
    for _ in range(40):
        task = gen.sample((256, 6))
        if not task.meta.get("class_repair"):
            continue
        repaired += 1
        d, ts = int(task.meta["d"]), int(task.meta["train_size"])
        ctx = task.X[:ts, :d].double()
        assert float(ctx.mean(0).abs().max()) < 0.15, "encoded on rows that are not the context"
        y = task.y
        assert set(y[:ts].tolist()) == set(y[ts:].tolist()) == {0.0, 1.0}
    assert repaired, "rare defaults should need the repair in some of these tables"
