"""Scientific meaning and data gates of the paper-facing notebook figures."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from src.eval.protocol import PROTOCOL
from src.visualize import campaign, exposure, paper_plots as paper


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_credit_scope_preserves_exp1_and_labels_exp2_tuning_population(monkeypatch):
    monkeypatch.setattr(paper, "load", lambda *a, **kw: {
        "eval": {"dev_datasets": ["dev"], "holdout_datasets": ["old_holdout"]}})
    frame = pd.DataFrame({"dataset": ["dev", "old_holdout", "unrelated"]})
    assert list(paper._role(frame, "pd", "exp1").dataset) == ["old_holdout"]
    assert list(paper._role(frame, "pd", "exp2").dataset) == ["dev", "old_holdout"]
    assert list(paper._role(frame, "pd", "exp3").dataset) == ["dev", "old_holdout"]


def _cells(name, score, datasets=("a", "b", "c"), seeds=(0, 1, 2)):
    trained = name.startswith("exp")
    return [dict(dataset=dataset, model="crediticl" if trained else name,
                 info_run_name=f"{name}__s{seed}" if trained else "", seed=0, fold=fold,
                 status="ok", protocol=PROTOCOL, roc_auc=score+.005*index+.001*seed,
                 log_loss=1-score, r2=score, crps=1-score,
                 boundary_mass_abs_err=(1-score)/2, n_classes=2 if index == 0 else 4,
                 true_mass_at_0=.1+index*.1, true_mass_at_1=.05+index*.05,
                 pred_mass_at_0=.1+index*.1+.01*seed,
                 pred_mass_at_1=.05+index*.05+.005*seed)
            for index, dataset in enumerate(datasets)
            for seed in (seeds if trained else (0,)) for fold in range(5)]


@pytest.fixture
def final_tables():
    return pd.DataFrame(_cells("exp1_pd__prior-credit_fraction=0", .7)
                        + _cells("exp1_pd__prior-credit_fraction=0p5", .75)
                        + _cells("exp1_pd__prior-credit_fraction=1", .72)
                        + _cells("tabiclv2", .8))


def test_seed_marks_do_not_treat_folds_as_training_seeds(final_tables):
    values = paper.seed_differences(final_tables, "exp1_pd__prior-credit_fraction=0p5",
                                    "exp1_pd__prior-credit_fraction=0", "roc_auc")
    assert len(values) == 3
    assert [seed for seed, _ in values] == [0, 1, 2]
    assert [value for _, value in values] == pytest.approx([.05]*3)
    broken = final_tables[~((final_tables.dataset == "b") & final_tables.info_run_name.eq(
        "exp1_pd__prior-credit_fraction=0p5__s1"))]
    with pytest.raises(ValueError, match="same paired datasets"):
        paper.seed_differences(broken, "exp1_pd__prior-credit_fraction=0p5",
                               "exp1_pd__prior-credit_fraction=0", "roc_auc")


def test_missing_final_cells_do_not_turn_monitoring_into_efficacy(monkeypatch):
    monkeypatch.setattr(paper, "final_credit", lambda *a, **k: None)
    monkeypatch.setattr(paper.analysis, "load_ood_results", lambda *a: None)
    for fig in (paper.paired_scores("pd"), paper.retention_frontier("pd"), paper.endpoint_mass()):
        assert len(fig.axes) == 1 and not fig.axes[0].has_data()
        assert any("required" in t.get_text() or "requires" in t.get_text()
                   for t in fig.axes[0].texts)


def test_final_credit_explicitly_requires_current_protocol(monkeypatch, final_tables):
    requested = []
    def load(track, **kwargs):
        requested.append(kwargs.get("protocol"))
        return None
    monkeypatch.setattr(paper, "load_results", load)
    assert paper.final_credit("pd") is None and not paper.final_ready("pd")
    assert requested == [PROTOCOL, PROTOCOL]
    mixed = final_tables.copy()
    mixed.loc[mixed.index[0], "protocol"] = PROTOCOL-1
    monkeypatch.setattr(paper.analysis, "load_ablation_results", lambda *a: mixed)
    assert paper._components("pd") is None


def test_final_scores_keep_metric_direction_and_dataset_unit(monkeypatch, final_tables):
    monkeypatch.setattr(paper, "final_credit", lambda *a: final_tables)
    fig = paper.paired_scores("pd")
    assert "higher" in fig.axes[0].get_xlabel()
    assert "lower" in fig.axes[1].get_xlabel()
    assert "n=3 datasets" in fig._suptitle.get_text()
    assert len(fig.axes[0].collections) == 8  # 2 contrasts × (mean + 3 seed marks)
    # Failed/incomplete folds may never be made into a persuasive figure.
    monkeypatch.setattr(paper, "final_credit", lambda *a: final_tables.iloc[1:])
    with pytest.raises(ValueError, match="Incomplete"):
        paper.paired_scores("pd")


def test_native_ood_subsets_preserve_multiclass_and_are_not_selection(monkeypatch, final_tables):
    binary = paper._subset(final_tables, "binary")
    multi = paper._subset(final_tables, "multiclass")
    assert set(binary.dataset) == {"a"}
    assert set(multi.dataset) == {"b", "c"}
    assert set(multi.n_classes) == {4}
    with pytest.raises(ValueError, match="class counts"):
        paper._subset(final_tables.drop(columns="n_classes"), "multiclass")
    monkeypatch.setattr(paper, "final_credit", lambda *a: final_tables)
    monkeypatch.setattr(paper.analysis, "load_ood_results", lambda *a: final_tables)
    fig = paper.retention_frontier("pd", subset="multiclass")
    assert "n=2 OOD" in " ".join(fig.axes[0].get_title(loc=loc) for loc in ("left", "center", "right"))
    assert "Credit" in fig.axes[0].get_xlabel() and "OOD" in fig.axes[0].get_ylabel()


def test_released_comparators_remain_separate_within_backbones(monkeypatch):
    table = pd.DataFrame(_cells("exp2_pd__arm=tabicl__prior-credit_fraction=0p5", .75)
                         + _cells("exp2_pd__arm=tabpfn3__prior-credit_fraction=0p5", .65)
                         + _cells("tabiclv2", .8) + _cells("tabpfn3", .5))
    effect = paper._table(table, "roc_auc", "pd", reference=True).set_index("control")
    assert effect.loc["tabiclv2", "mean_difference"] == pytest.approx(-.049)
    assert effect.loc["tabpfn3", "mean_difference"] == pytest.approx(.151)
    assert "TabPFN-3" in paper.label("exp2_pd__arm=tabpfn3__prior-credit_fraction=0p5")


def test_two_rate_frontier_uses_unambiguous_shared_marker_legend(monkeypatch):
    rows = []
    for model in ("tabicl", "tabpfn3"):
        for rate in (0, 1):
            for fraction in (0, .25, .5, 1):
                tag = str(fraction).replace(".", "p")
                name = f"exp2_pd__arm={model}__lr_index={rate}__prior-credit_fraction={tag}"
                rows += _cells(name, .65+.03*fraction+.01*rate)
    table = pd.DataFrame(rows)
    monkeypatch.setattr(paper, "final_credit", lambda *a: table)
    monkeypatch.setattr(paper.analysis, "load_ood_results", lambda *a: table)
    fig = paper.retention_frontier("pd", "exp2")
    fig.canvas.draw()
    labels = [text.get_text() for text in fig.legends[0].texts]
    assert len(labels) == 5
    assert labels[-2:] == ["Higher peak LR", "Lower peak LR"]
    assert "2e-05 / 1e-05" in fig.axes[0].get_title(loc="left")
    assert "1e-05 / 1e-06" in fig.axes[1].get_title(loc="left")


def test_endpoint_panel_omits_point_only_models(monkeypatch, final_tables):
    extra = pd.DataFrame(_cells("linear", .5))
    extra[["pred_mass_at_0", "pred_mass_at_1"]] = np.nan
    monkeypatch.setattr(paper, "final_credit", lambda *a: pd.concat([final_tables, extra]))
    fig = paper.endpoint_mass()
    labels = [t.get_text() for t in fig.legends[0].texts]
    assert "linear" not in labels
    assert len(fig.axes) == 2
    assert "Observed portfolio mass" == fig.axes[0].get_xlabel()


def test_raw_credit_endpoints_cannot_include_standardised_targets():
    from src.visualize.exp1_plots import plot_boundary_mass_sources, plot_standardised_endpoint_ties
    raw = SimpleNamespace(X=torch.zeros(4, 1), y=torch.tensor([0., 0., .5, 1.]), meta={"train_size": 2})
    generic = SimpleNamespace(y=torch.tensor([-3., -1., 1., 3.]), meta={"train_size": 2})
    fig = plot_boundary_mass_sources({"original": [generic], "credit": [raw]})
    assert len(fig.axes) == 1 and len(fig.axes[0].collections) == 1
    with pytest.raises(ValueError, match="raw"):
        plot_boundary_mass_sources({"credit": [generic]})
    missing = copy.deepcopy(raw); missing.meta = {}
    with pytest.raises(ValueError, match="context split"):
        plot_standardised_endpoint_ties({"credit": [missing]})


def test_production_observer_does_not_change_task_or_acceptance():
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.prior.stream import plan_batch
    from src.utils.config import load
    cfg = copy.deepcopy(load("config/Exp1_PD.yaml")["prior"])
    cfg.update(n_rows_range=[128, 128], n_features_range=[4, 8], max_features=8,
               credit_fraction=.5)
    plans = plan_batch(1729, 0, cfg, 4, regression=False, credit_fraction=.5)
    original = TaskGenerator(cfg, "pd", PriorRNG(1729))
    observed = exposure._ObservedGenerator(cfg, "pd", PriorRNG(1729))
    for plan in plans.slots:
        a, b = original.sample_slot(plan), observed.sample_slot(plan)
        assert torch.equal(a.X, b.X) and torch.equal(a.y, b.y)
        assert a.meta == b.meta
    assert original.filter_summary() == observed.filter_summary()


def test_accepted_stream_keeps_native_slot_class_counts():
    from src.utils.config import load
    cfg = copy.deepcopy(load("config/Exp1_PD.yaml"))
    cfg["prior"].update(n_rows_range=[128, 128], n_features_range=[4, 8], max_features=8)
    frame = exposure.accepted_stream(cfg, n=8, seed=1729)
    assert frame.groupby("mixture").accepted.sum().tolist() == [8, 8, 8]
    assert frame.loc[frame.accepted & frame.mixture.eq(0), "classes"].gt(2).any()
    assert frame.loc[frame.accepted & frame.mixture.eq(1), "classes"].eq(2).all()
    assert not frame.loc[frame.source.eq("credit"), "statistically_judged"].any()
    assert frame.loc[frame.accepted, "valid"].all()


def test_local_coverage_distinguishes_partial_absent_complete(tmp_path, monkeypatch):
    from src.utils import paths
    monkeypatch.setattr(paths, "runs_dir", lambda *a: tmp_path)
    expected = campaign.coverage("pd")
    a, b = [tmp_path / name for name in expected.run[:2]]
    a.mkdir(); b.mkdir()
    (a / "summary.json").write_text(json.dumps({"completed": True, "steps": 12500}))
    (b / "progress.csv").write_text("step,train_loss\n100,0.2\n")
    actual = campaign.coverage("pd")
    assert actual.state.value_counts().to_dict() == {"absent": 10, "completed": 1, "partial": 1}
    assert len(campaign.coverage("pd", components=True)) == 15
    assert "not a live scheduler" in campaign.provenance("pd")


def test_retention_design_does_not_present_hypothetical_outcomes():
    from src.visualize.continuation import design_figure
    fig = design_figure(3)
    assert not any(ax.has_data() for ax in fig.axes)
    text = ' '.join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert '48 additional main runs' in text and '36 development-only pilots' in text
    assert 'not results' in fig._suptitle.get_text()


def test_unfiltered_difficulty_has_no_dormant_banded_overlay_or_clipped_scores():
    from src.visualize.mechanism_plots import plot_predictability
    data = pd.DataFrame({"source": ["original prior", "credit prior"],
                         "pseudo_r2": [-.35, .4], "pvalue": [.5, .01]})
    fig = plot_predictability(data, "config/Exp1_PD.yaml")
    ax = fig.axes[0]
    assert ax.get_xlim()[0] < -.35
    assert len(ax.patches) == 0


def test_real_pd_strip_uses_same_minority_definition_as_synthetic():
    from src.visualize.pool_plots import plot_base_rate_by_variant
    task = SimpleNamespace(y=torch.tensor([0., 0., 0., 1.]))
    fig = plot_base_rate_by_variant({"credit": [task]}, {"book": .8})
    assert fig.axes[1].collections[0].get_offsets()[0, 0] == pytest.approx(.2)


def test_feature_inventory_counts_policy_without_certifying_timing():
    from src.visualize.data_plots import feature_policy_inventory
    ds = SimpleNamespace(feature_names=["age", "recoveries", "loan_amnt"], cat_indices=[], groups=None)
    counts, excluded = feature_policy_inventory({"pd": {"0006.hackerearth": ds}})
    assert counts.retained.iloc[0] + counts.excluded.iloc[0] == 3
    assert counts.excluded.iloc[0] >= 1
    assert "not independently certified" in counts.timing_status.iloc[0]
    assert "recoveries" in set(excluded.source_column)


def test_exp2_endpoint_panels_do_not_pool_backbones(monkeypatch):
    table = pd.DataFrame(_cells("exp2_lgd__arm=tabicl__prior-credit_fraction=0p5", .7)
                         + _cells("exp2_lgd__arm=tabpfn3__prior-credit_fraction=0p5", .6)
                         + _cells("tabiclv2", .75) + _cells("tabpfn3", .65))
    monkeypatch.setattr(paper, "final_credit", lambda *a: table)
    fig = paper.endpoint_mass("exp2", backbone="tabpfn3")
    labels = [t.get_text() for t in fig.legends[0].texts]
    assert all("tabicl" not in label.casefold() for label in labels)
    assert any("TabPFN-3" in label for label in labels)


def test_dense_paired_scorecard_ticks_remain_separate(monkeypatch):
    table = pd.DataFrame(_cells("exp2_lgd__arm=tabicl__prior-credit_fraction=0", .7)
                         + _cells("exp2_lgd__arm=tabicl__prior-credit_fraction=0p5", .70025)
                         + _cells("exp2_lgd__arm=tabpfn3__prior-credit_fraction=0", .7)
                         + _cells("exp2_lgd__arm=tabpfn3__prior-credit_fraction=1", .7005))
    monkeypatch.setattr(paper, "final_credit", lambda *a: table)
    fig = paper.distributional_scorecard("exp2")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        boxes = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels() if tick.get_visible()]
        assert not any(a.overlaps(b) for a, b in zip(boxes, boxes[1:]))


def test_controlled_zoib_response_respects_probability_and_driver_direction():
    from src.visualize.mechanism_plots import zoib_response
    fig = zoib_response("config/Exp1_LGD.yaml")
    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert len(ax.lines) == 3
        assert all(np.isfinite(line.get_ydata()).all() for line in ax.lines)
        assert all(((line.get_ydata() >= 0) & (line.get_ydata() <= 1)).all() for line in ax.lines)
    assert all(line.get_ydata()[0] < line.get_ydata()[-1] for line in fig.axes[0].lines)
    assert all(line.get_ydata()[0] > line.get_ydata()[-1] for line in fig.axes[1].lines)


def test_controlled_correlation_probe_really_disables_selection_and_label_noise(monkeypatch):
    from src.prior.rng import PriorRNG
    from src.prior.targets.pd import apply_label_noise, max_selection_drop
    from src.visualize import mechanism_plots as mechanisms

    def capture(task, prior, n, seed, overrides=None):
        resolved = copy.deepcopy(prior)
        for key, value in overrides.items():
            mechanisms._set(resolved, key, value)
        target = resolved["credit"]["target"]
        assert max_selection_drop(target["selection"]) == 0
        assert target["rules"]["n_rules_range"] == [0, 0]
        assert resolved["credit"]["shift"]["shift_prob"] == 0
        y = torch.tensor([0., 1.]*100)
        assert torch.equal(apply_label_noise(PriorRNG(7), y, target)[0], y)
        return [SimpleNamespace(y=torch.tensor([0.]*(90-i)+[1.]*(10+i))) for i in range(4)]

    monkeypatch.setattr(mechanisms, "_generate", capture)
    assert len(mechanisms.correlated_defaults("config/Exp1_PD.yaml", n=4).axes) == 2


def test_lgd_economic_mass_does_not_count_interior_minimum_as_an_atom():
    from src.visualize.data_plots import plot_boundary_mass_ranking
    book = SimpleNamespace(y=np.array([.2]*50+[.8]*50))
    fig = plot_boundary_mass_ranking({"0001.interior": book})
    assert all(bar.get_width() == 0 for bar in fig.axes[0].patches)


def test_single_raw_credit_distance_panel_names_each_real_portfolio():
    from src.visualize.exp1_plots import plot_prior_realism_ranking
    task = SimpleNamespace(y=torch.tensor([0., .5, 1.]), source="credit")
    base = SimpleNamespace(y=torch.tensor([-1., 0., 1.]), source="base")
    books = {"0001.a": SimpleNamespace(y=np.array([0., .5, 1.])),
             "0002.b": SimpleNamespace(y=np.array([.5]*3))}
    fig = plot_prior_realism_ranking({"original": [base], "credit": [task]}, books)
    assert {t.get_text() for t in fig.axes[0].get_yticklabels()} == {"a", "b"}


def test_clustered_dataset_names_do_not_intersect():
    from matplotlib.text import Text
    from src.visualize.data_plots import plot_shapes
    points = [(9578, 12), (16002, 20), (5627, 13), (5960, 8), (7000, 10)]
    names = ["lendingclub", "lgd_freddie", "lgd_lendingclub", "hmeq", "myhom"]
    books = {name: SimpleNamespace(n_rows=x, n_features=y) for name, (x, y) in zip(names, points)}
    fig = plot_shapes({"pd": books})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(t, renderer) for t in fig.axes[0].texts]
    assert len(boxes) == len(names)
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:])
