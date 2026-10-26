"""Behavioural regressions for the independently checked October review findings."""
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def test_tied_vasicek_scores_preserve_the_requested_applicant_rate():
    from src.prior.targets.mechanisms import pd_vasicek
    from src.prior.rng import PriorRNG

    latent = torch.arange(90_000).remainder(3).float()
    cfg = {"rho_range": [0., 0.], "base_rate_range": [.05, .05],
           "signal_share_range": [1., 1.], "macro_scale": 0.}
    y, meta = pd_vasicek(PriorRNG(71), latent, cfg)
    assert float(y.mean()) == pytest.approx(.05, abs=.004)
    assert meta["target_base_rate"] == .05


def test_missingness_has_no_outcome_input_and_no_independent_label_votes():
    from src.prior.targets.pd import apply_informative_missingness
    from src.prior.rng import PriorRNG

    assert "y" not in inspect.signature(apply_informative_missingness).parameters
    X = torch.randn(1000, 30, generator=torch.Generator().manual_seed(7))
    cfg = {"missing_col_fraction": 1., "missing_rate_range": [.2, .2],
           "missing_risk_coupling": 1., "missing_indicators": False}
    missing, _ = apply_informative_missingness(PriorRNG(8), X, cfg, 30, fill="nan")
    mask = missing.isnan()
    assert .15 < float(mask.float().mean()) < .25
    assert torch.equal(mask, mask[:, :1].expand_as(mask))


def test_component_random_streams_do_not_shift_after_another_component_draws():
    from src.prior.rng import PriorRNG
    rng = PriorRNG(19)
    expected = rng.fork("missing").rand(40)
    rng.fork("marginals").rand(10_000)
    rng.rand(5_000)
    torch.testing.assert_close(rng.fork("missing").rand(40), expected, rtol=0, atol=0)


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_tail_ablation_preserves_targets_row_order_noise_and_other_random_draws(track):
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.prior.stream import plan_batch
    from src.utils.config import expand_with_seeds, load

    cfg = expand_with_seeds(load(f"config/Exp1_{track.upper()}.yaml"))[1]["prior"]
    cfg = copy.deepcopy(cfg)
    cfg.update(n_rows_range=[128, 128], n_features_range=[8, 8], max_features=8)
    cfg["filter"]["mode"] = "off"
    cfg["credit"]["shift"]["shift_prob"] = 0.
    cfg["credit"]["missingness"]["missing_prob"] = 0.
    cfg["credit"]["marginals"]["col_fraction_range"] = [1., 1.]
    other = copy.deepcopy(cfg)
    other["credit"]["marginals"]["col_fraction_range"] = [0., 0.]
    plan = plan_batch(0, 0, cfg, 4, regression=track == "lgd", credit_fraction=1.)
    a, b = [TaskGenerator(c, track, PriorRNG(0), stream_seed=0) for c in (cfg, other)]
    for slot in plan.slots:
        first, second = a.sample_slot(slot), b.sample_slot(slot)
        torch.testing.assert_close(first.y, second.y, rtol=0, atol=0)
        assert first.meta["labels_swapped"] == second.meta["labels_swapped"] if "labels_swapped" in first.meta else True


def test_future_gdp_feature_is_excluded_with_source_prefix_and_categorical_copy():
    from src.eval.feature_policy import prediction_features
    names = ["C_PY1_evol_pib", "C_PY1_evol_pib__cat", "PY1_future", "C_PY0_evol_pib"]
    X, cats, report = prediction_features("lgd", "0005.base_modelisation", np.ones((12, 4)), [1], names)
    assert X.shape == (12, 1) and cats == [] and report["n_features_excluded"] == 3


def test_freddie_missing_codes_do_not_become_extreme_risk_values(monkeypatch):
    from src.data import dataset_preprocessing as recipes
    raw = pd.DataFrame({"lgd": [.1, .2, .3], "dti": [999., 40., 60.], "fico": [9999., 710., 650.]})
    monkeypatch.setattr(recipes, "_load_dataset_file", lambda _: raw.copy())
    frame, _, _, _ = recipes.preprocess_dataset_specific("lgd", "0006.lgd_freddie", raw_dir=Path("unused"))
    assert len(frame) == 3 and np.isnan(frame.dti.iloc[0]) and np.isnan(frame.fico.iloc[0])
    assert frame.dti.iloc[1] == 40 and frame.fico.iloc[1] == 710


def test_extreme_finite_features_remain_distinct_and_safe_for_float32_reductions():
    from src.data.dataset_preprocessing import _sanitize_numeric_values
    values = np.array([-1e41, -1e30, -1e20, 0, 1e20, 1e30, 1e41])
    out = _sanitize_numeric_values(pd.DataFrame({"x": values}), ["x"]).x.to_numpy(np.float32)
    assert np.all(np.diff(out) > 0)
    assert np.isfinite(np.tile(out, 100_000).std())


def test_reference_winner_cannot_hide_a_tie_between_prior_mixtures():
    from src.eval.selection import development_ranking
    from src.utils.config import expand_with_seeds, load, run_name
    cfg = load("config/Exp1_PD.yaml")
    rows = []
    for run in expand_with_seeds(cfg):
        share = run["prior"]["credit_fraction"]
        score = .6 if share == 0 else .7 + run["seed"] * .01
        for ds in cfg["eval"]["dev_datasets"]:
            for fold in range(5):
                rows.append(dict(dataset=ds, model="crediticl", info_run_name=run_name(run), seed=0,
                                 fold=fold, status="ok", roc_auc=score, log_loss=1-score))
    for ds in cfg["eval"]["dev_datasets"]:
        for fold in range(5):
            rows.append(dict(dataset=ds, model="catboost", seed=0, fold=fold, status="ok", roc_auc=.99, log_loss=.01))
    df = pd.DataFrame(rows)
    for metric in ("roc_auc", "log_loss"):
        rank = development_ranking(df, "pd", metric=metric)
        assert rank.iloc[0].configuration == "catboost"
        assert not rank.iloc[0].within_seed_spread_of_best
        tied = rank[rank.within_seed_spread_of_best]
        assert len(tied) == 3 and set(tied.training_seeds) == {3}


def test_learning_curves_use_common_steps_and_never_pool_architectures(monkeypatch):
    from src.visualize import results_plots as plots
    rows = []
    for model, score in (("tabicl", .9), ("tabpfn3", .1)):
        for seed in (0, 1):
            for step in (100, 200, 300):
                if model == "tabicl" and seed == 1 and step == 300:
                    continue
                rows.append(dict(dataset="d", model="crediticl", checkpoint_step=step,
                                 info_run_name=f"exp2_pd__arm={model}__prior-credit_fraction=0p5__s{seed}", roc_auc=score))
    monkeypatch.setattr(plots, "load_checkpoint_results", lambda *a: pd.DataFrame(rows))
    curve = plots._checkpoint_means("pd", "exp2", "roc_auc", None)
    assert set(curve.checkpoint_step) == {100, 200}
    assert set(curve["mean"]) == {.9, .1} and curve.configuration.nunique() == 2


def test_reference_models_are_not_averaged_with_each_other():
    from src.visualize.results_plots import _means_per_group
    df = pd.DataFrame([dict(dataset="d", model=m, roc_auc=s) for m, s in (("catboost", .9), ("linear", .5))])
    table, groups = _means_per_group(df, "roc_auc")
    assert len(groups) == 2 and set(table.iloc[0]) == {.9, .5}


def test_unbounded_control_target_cannot_create_lgd_boundary_atoms():
    from src.visualize.exp1_plots import _unit_target
    with pytest.raises(ValueError, match="unbounded"):
        _unit_target(np.array([-3., -3., 0., 2., 2.]))


def test_summary_refuses_success_only_means():
    from src.eval.runner import summarise
    rows = [dict(model="linear", dataset="d", seed=0, fold=f, status="ok", roc_auc=.8) for f in range(5)]
    rows[-1]["status"] = "failed"
    with pytest.raises(ValueError, match="successful"):
        summarise(pd.DataFrame(rows), "pd")


def test_notebook_outputs_do_not_change_validation_fingerprint_but_source_does(tmp_path, monkeypatch):
    from src.utils import paths, preflight
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "notebooks").mkdir()
    p = tmp_path / "notebooks/probe.ipynb"
    nb = {"cells": [{"cell_type": "code", "source": ["print(1)"], "outputs": []}]}
    p.write_text(json.dumps(nb)); before = preflight.source_fingerprint()
    nb["cells"][0]["outputs"] = [{"text": "1"}]
    p.write_text(json.dumps(nb)); assert preflight.source_fingerprint() == before
    nb["cells"][0]["source"] = ["print(2)"]
    p.write_text(json.dumps(nb)); assert preflight.source_fingerprint() != before


def test_only_main_experiment_entry_points_live_in_scripts():
    root = Path(__file__).resolve().parents[1]
    assert {p.name for p in (root / "scripts").glob("*.py")} == {
        "pretrain.py", "evaluate.py", "evaluate_ood.py", "validate_experiment.py"}
    from src.validation.campaign import plan
    assert any(s.name == "notebooks" for s in plan("cpu"))


def test_tabicl_telemetry_records_the_gradient_before_clipping(pd_cfg, tmp_path, monkeypatch):
    from src.train.loop import Trainer
    from src.prior.dataset import PriorBatchDataset
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.train.telemetry import grad_norms

    cfg = copy.deepcopy(pd_cfg)
    cfg["train"]["gradient_clipping"] = 1e-5
    trainer = Trainer(cfg, tmp_path / "run", device="cpu", ckpt_dir=tmp_path / "ck",
                      log_dir=tmp_path / "log")
    trainer.grad_clip = 1e-5
    recorded, clipped = [], []
    monkeypatch.setattr(trainer.telemetry, "due_grad", lambda _: True)
    monkeypatch.setattr(trainer.telemetry, "record", lambda row: recorded.append(row))
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip(parameters, *args, **kwargs):
        before = grad_norms(trainer.model)["grad_global"]
        result = original_clip(parameters, *args, **kwargs)
        clipped.append((before, grad_norms(trainer.model)["grad_global"]))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    ds = PriorBatchDataset(cfg["prior"], "pd", cfg["train"]["batch_size"], seed=0)
    gen = TaskGenerator(cfg["prior"], "pd", PriorRNG(0), stream_seed=0)
    try:
        trainer.train_step(ds.batch(gen, 0))
        assert recorded[0]["grad_global"] == pytest.approx(clipped[0][0])
        assert recorded[0]["grad_global"] > 1e-5 and clipped[0][1] <= 1.01e-5
    finally:
        trainer.close()


def test_summary_follows_notebook_section_order(monkeypatch):
    from src.visualize import results_plots as plots
    monkeypatch.setattr(plots, "load_results", lambda *args, **kwargs: None)
    text = plots.results_summary("pd")
    headings = ["A.", "B.", "B1.", "B2.", "B3.", "B4.", "C."]
    assert [text.index(h) for h in headings] == sorted(text.index(h) for h in headings)


def test_dataset_means_do_not_pool_exp2_architectures():
    from src.visualize.results_plots import _means_per_group
    rows = [dict(dataset="d", model="crediticl", roc_auc=s,
                 info_run_name=f"exp2_pd__arm={m}__prior-credit_fraction=0p5__s0")
            for m, s in (("tabicl", .8), ("tabpfn3", .4))]
    table, _ = _means_per_group(pd.DataFrame(rows), "roc_auc")
    assert table.shape == (1, 2) and set(table.iloc[0]) == {.8, .4}
