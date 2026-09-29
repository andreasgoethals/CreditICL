"""A credit table reaches the model the same way in training as a real table does at prediction.

The model is TRAINED on what our prior emits and asked to PREDICT on real tables pushed through
upstream's `TabICLClassifier` / `TabICLRegressor`. So the prior's output must already be in the
form upstream's prediction preprocessing produces. The strongest check is a round trip: treat a
generated credit table as if it were a real dataset, run it through upstream's own prediction-time
preprocessing, and compare the result with the tensor the model saw in training.

* FEATURES — the credit path ends in upstream's own `PreprocessingPipeline("none")` fitted on the
  context rows (`prediction_view`), so running that pipeline again must leave the table unchanged
  but for the few values its soft clip already compressed (the clip is not idempotent). Until
  28-09-2026 credit tables were standardised on the WHOLE table, and a table with a shift came
  out up to 4 SD away from what prediction would have shown the model.
* LGD TARGET — `TabICLRegressor.fit` standardises y with a StandardScaler fitted on the context;
  the training target must already be exactly that, with the boundary atoms preserved as two
  atoms (Exp1 trained on raw [0, 1] instead, and the 100 % credit arms collapsed).
* PD LABELS — prediction rotates class labels across ensemble members, so training must show both
  label identities (upstream's `permute_labels`).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

pytest.importorskip("tabicl")

from src.prior.generator import TaskGenerator  # noqa: E402
from src.prior.rng import PriorRNG  # noqa: E402


def _credit(cfg: dict, task: str) -> TaskGenerator:
    prior = copy.deepcopy(cfg["prior"])
    prior["credit_fraction"] = 1.0
    prior["n_rows_range"] = [512, 512]
    return TaskGenerator(prior, task, PriorRNG(0))


@pytest.mark.parametrize("which", ["pd", "lgd"])
def test_a_credit_table_reaches_the_model_as_upstreams_prediction_path_encodes_it(
        pd_cfg, lgd_cfg, which, monkeypatch):
    """The model's input is upstream's `UniqueFeatureFilter` + `PreprocessingPipeline("none")`,
    fitted on the context rows, applied ONCE to the table — and nothing after it changes it."""
    from tabicl._sklearn.preprocessing import PreprocessingPipeline, UniqueFeatureFilter

    import src.prior.generator as generator

    seen: list[tuple[torch.Tensor, int, int]] = []
    real_view = generator.prediction_view

    def recording_view(X, width, train_size):
        seen.append((X.clone(), int(width), int(train_size)))
        return real_view(X, width, train_size)

    monkeypatch.setattr(generator, "prediction_view", recording_view)
    gen = _credit(pd_cfg if which == "pd" else lgd_cfg, which)
    for _ in range(6):
        seen.clear()
        task = gen.sample(shape=(512, 8))
        before, width, ts = seen[-1]  # the candidate that was kept is the last one encoded
        raw = before[:, :width].numpy().astype(np.float64)
        keep = UniqueFeatureFilter().fit(raw[:ts])
        cols = keep.transform(raw)
        expected = PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0).fit(cols[:ts]).transform(cols)
        d = int(task.meta["d"])
        assert d == cols.shape[1]
        np.testing.assert_allclose(task.X[:, :d].numpy(), expected, atol=1e-5)
        assert not task.X[:, d:].any(), "the padding stays zero"


@pytest.mark.parametrize("which", ["pd", "lgd"])
def test_a_shifted_credit_table_is_standardised_on_its_context_as_prediction_does(pd_cfg, lgd_cfg, which):
    """The statistics that scale a column come from the context rows alone, so a shifted query
    shows up as a query off-centre — which is what a real shifted table looks like to the model."""
    prior = copy.deepcopy((pd_cfg if which == "pd" else lgd_cfg)["prior"])
    prior["credit_fraction"] = 1.0
    prior["credit"]["shift"].update(shift_prob=1.0, kind_weights={"covariate": 1.0})
    gen = TaskGenerator(prior, which, PriorRNG(5))
    checked = 0
    for _ in range(8):
        task = gen.sample(shape=(512, 8))
        if task.meta.get("shift") != "covariate":
            continue
        d, ts = int(task.meta["d"]), int(task.meta["train_size"])
        ctx = task.X[:ts, :d].double()
        # Centred on the context (the soft clip moves a clipped column's mean a little). With
        # whole-table statistics the sorted column's context sat about 0.8 SD off centre.
        assert ctx.mean(0).abs().max() < 0.15
        # Unit SD on the context, less wherever the soft clip compressed a column (a rare binary
        # level at +10 SD is pulled in to about +2.3 — at prediction too); never more.
        assert float(ctx.std(0, unbiased=False).max()) < 1.0 + 1e-3
        checked += 1
    assert checked, "the covariate shift should apply to most of these tables"


def test_the_lgd_target_is_already_what_the_regressor_rescales_it_to_with_its_atoms(lgd_cfg):
    from sklearn.preprocessing import StandardScaler

    gen = _credit(lgd_cfg, "lgd")
    seen = 0
    for _ in range(12):
        task = gen.sample(shape=(512, 8))
        y, ts = task.y.numpy().astype(np.float64), int(task.meta["train_size"])
        rescaled = StandardScaler().fit(y[:ts, None]).transform(y[:, None]).ravel()
        # `TabICLRegressor.fit` standardises y on the context: on the training target, the identity.
        assert np.abs(rescaled - y).max() < 1e-4
        values, counts = np.unique(np.round(y, 6), return_counts=True)
        atoms = values[counts >= 0.02 * len(y)]
        if len(atoms) >= 1:
            seen += 1
            # The atoms at LGD 0 and 1 are still exact, repeated values after both rescalings.
            r_values, r_counts = np.unique(np.round(rescaled, 6), return_counts=True)
            assert (r_counts >= 0.02 * len(y)).sum() == len(atoms)
    assert seen, "the calibrated LGD prior should put atoms in most tables"


def test_credit_pd_tables_show_both_label_identities_as_prediction_does(pd_cfg):
    gen = _credit(pd_cfg, "pd")
    minority_is_one = []
    for _ in range(20):
        y = gen.sample(shape=(256, 6)).y
        minority_is_one.append(float(y.mean()) < 0.5)
    assert any(minority_is_one) and not all(minority_is_one)


def test_missing_values_reach_training_as_prediction_imputes_them(pd_cfg):
    """No NaN and no indicator column: a gap is the mean-imputed value, as upstream's wrapper
    imputes a real table (`SimpleImputer(strategy='mean')`, no `add_indicator`)."""
    prior = copy.deepcopy(pd_cfg["prior"])
    prior["credit_fraction"] = 1.0
    prior["credit"]["missingness"].update(missing_prob=1.0, missing_col_fraction_range=[0.5, 0.5])
    task = TaskGenerator(prior, "pd", PriorRNG(3)).sample(shape=(512, 10))
    assert torch.isfinite(task.X).all()
    assert task.meta.get("missing_indicators", 0) == 0 and task.meta.get("missing_cols", 0) > 0
    d = int(task.meta["d"])
    assert d <= 10, "gaps must not widen the table"


def test_without_a_shift_the_context_query_split_is_random_like_a_cv_fold(pd_cfg):
    """The PD mechanism stacks vintages in contiguous row blocks. Unless a cohort shift is drawn,
    the rows must be reshuffled — otherwise every context is the early vintages and every query
    the late ones, a shift in every table (the bug fixed 28-09-2026)."""
    prior = copy.deepcopy(pd_cfg["prior"])
    prior["credit_fraction"] = 1.0
    prior["credit"]["shift"]["shift_prob"] = 0.0
    mech = prior["credit"]["target"]["mechanism"]
    mech["cohort"] = {"min_cohorts": 12, "max_cohorts": 12, "cohort_sd_range": [1.5, 1.5]}
    gen = TaskGenerator(prior, "pd", PriorRNG(11))
    gaps = []
    for _ in range(16):
        task = gen.sample(shape=(1024, 8))
        y, ts = task.y, int(task.meta["train_size"])
        gaps.append(abs(float(y[:ts].mean()) - float(y[ts:].mean())))
    # Random halves differ by sampling noise (~0.03 here); vintage-ordered halves by far more.
    assert float(np.mean(gaps)) < 0.05


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_the_training_monitor_scores_the_live_weights_through_upstreams_wrapper(task, tmp_path):
    """The progress curve puts the IN-TRAINING model inside `TabICLClassifier` /
    `TabICLRegressor`, so it measures exactly what the benchmark measures, at a smaller size."""
    from src.models.architecture import build_model
    from src.train.progress import ProgressConfig, ProgressTracker

    small = dict(embed_dim=32, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=2, col_nhead=2,
                 row_nhead=2, icl_nhead=2, n_cls_rows=16)
    if task == "lgd":
        small["num_quantiles"] = 999
    model = build_model(task, **small)
    model.train()
    tracker = ProgressTracker(ProgressConfig(every_datasets=1, context_rows=128, max_test_rows=64),
                              task, "unit", tmp_path)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan  # the wrapper imputes, as on real data
    signal = np.nan_to_num(X[:, 0])
    y = ((signal > 0.5).astype(np.float32) if task == "pd"
         else np.clip(0.4 + 0.2 * signal, 0, 1).astype(np.float32))
    first = tracker._score(model, "unit/toy", X, y, regression=task == "lgd")
    again = tracker._score(model, "unit/toy", X, y, regression=task == "lgd")
    headline = "roc_auc" if task == "pd" else "crps"
    assert np.isfinite(first[headline])
    assert first.keys() == again.keys()
    for key in first:  # a fixed split: the same rows, the same numbers, every time
        assert first[key] == pytest.approx(again[key], nan_ok=True), key
