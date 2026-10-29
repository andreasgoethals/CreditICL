"""Exercise fixed retention recipes and adapter routes, including incomplete evidence."""
import copy

import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("scale", [None, 8.])
def test_tabpfn_anchor_gradient_is_added_once_after_unscale(scale):
    from test_tabpfn_safety import tiny_trainer
    trainer = tiny_trainer(scale=scale)
    trainer.grad_clip = 100.
    trainer.l2sp_alpha = .3
    trainer._l2sp_ref = {"weight": torch.tensor([[.5]])}
    gradients = []
    trainer.optimizer.register_step_pre_hook(
        lambda opt, args, kwargs: gradients.append(trainer.model.weight.grad.clone()))
    # Two task losses are averaged. The anchor is independent of task count/AMP scale.
    stats = trainer.train_step([{"item": 1}, {"item": 2}])
    torch.testing.assert_close(gradients[0], torch.tensor([[2.15]]))
    assert stats["l2sp_penalty"] == pytest.approx(.3 / 2 * .5**2)
    assert torch.equal(trainer._l2sp_ref["weight"], torch.tensor([[.5]]))


def test_anchor_also_updates_a_trainable_parameter_unused_by_this_table():
    from test_tabpfn_safety import tiny_trainer
    from src.train.loop import Trainer
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.bias.fill_(1.)
    trainer = object.__new__(Trainer)
    trainer.model, trainer.l2sp_alpha = model, .3
    trainer._l2sp_ref = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.bias.grad = None
    trainer._apply_l2sp()
    torch.testing.assert_close(model.bias.grad, torch.tensor([.3]))
    tabpfn = tiny_trainer()
    tabpfn.model.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
    tabpfn.optimizer.add_param_group({"params": [tabpfn.model.unused]})
    tabpfn.l2sp_alpha = .3
    tabpfn._l2sp_ref = {n: torch.zeros_like(p) for n, p in tabpfn.model.named_parameters()}
    gradients = []
    tabpfn.optimizer.register_step_pre_hook(lambda *a: gradients.append(tabpfn.model.unused.grad.clone()))
    tabpfn.grad_clip = 100.
    tabpfn.train_step([{"item": 1}])
    torch.testing.assert_close(gradients[0], torch.tensor([.3]))


def test_diagnostic_stop_keeps_the_registered_schedule_horizon():
    from src.train.optim import build_scheduler
    from src.experiments.design import MAIN_STEPS
    from src.utils.config import expand_with_seeds, load
    main = next(r for r in expand_with_seeds(load("config/Exp3_PD.yaml")) if r["method"] == "lora")
    diagnostic = copy.deepcopy(main)
    diagnostic["train"]["max_steps"] = 5
    opt_a = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=main["train"]["lr"])
    opt_b = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=main["train"]["lr"])
    a = build_scheduler(opt_a, diagnostic["train"], diagnostic["train"]["schedule_total_steps"])
    b = build_scheduler(opt_b, main["train"], MAIN_STEPS)
    for _ in range(6):
        assert opt_a.param_groups[0]["lr"] == opt_b.param_groups[0]["lr"]
        opt_a.step(); opt_b.step(); a.step(); b.step()


def test_gpu_gate_has_all_sixteen_method_routes():
    from src.validation.campaign import plan
    stages = [s for s in plan("gpu") if s.name.startswith("adaptation-")]
    expected = {f"adaptation-{t}-{b}-{m}" for t in ("pd", "lgd")
                for b in ("tabicl", "tabpfn3") for m in ("no_decay", "l2sp", "diagscale", "lora")}
    assert {s.name for s in stages} == expected
    assert all(s.report and s.max_seconds == 1800 for s in stages)


def test_checkpoint_plot_reader_withholds_partial_or_wrong_cells(tmp_path):
    from src.experiments.results import read_cells, complete_scores
    from src.eval.protocol import PROTOCOL
    rows = [dict(dataset="d", model="crediticl", seed=0, fold=f, info_run_name="run", status="ok",
                 protocol=PROTOCOL, info_checkpoint_step=125, roc_auc=.7) for f in range(5)]
    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert read_cells(path, ["d"], identity="run", step=125) is not None
    with pytest.warns(UserWarning, match="Withheld"):
        assert read_cells(path, ["d"], identity="different", step=125) is None
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.warns(UserWarning, match="Withheld"):
        assert read_cells(path, ["d"], identity="run", step=125) is None
    frame = pd.concat([pd.DataFrame(rows).assign(training_seed=s, architecture="tabicl", lr_index=lr,
                mixture=.5, checkpoint_step=125, roc_auc=.6 + lr*.1) for s in (0, 1, 2) for lr in (0, 1)])
    scores = complete_scores(frame, "roc_auc", experiment=2)
    assert list(scores.sort_values("lr_index")["mean"]) == pytest.approx([.6, .7])
    assert complete_scores(frame[frame.training_seed.ne(2)], "roc_auc", experiment=2).empty


@pytest.fixture
def shared_reference_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.experiments import results
    from src.eval import ood
    from src.eval.protocol import PROTOCOL

    monkeypatch.setattr(results.paths, "benchmark_dir", lambda owner, track: tmp_path / str(owner) / track)
    monkeypatch.setattr(results, "load", lambda *a, **k: {
        "eval": {"dev_datasets": ["development"], "holdout_datasets": ["held-out"]}})
    monkeypatch.setattr(ood, "list_ood_datasets", lambda *a: [SimpleNamespace(name=n) for n in ("ood-a", "ood-b")])

    def write(track, backbone, domain):
        model = "tabiclv2" if backbone == "tabicl" else "tabpfn3"
        folder = tmp_path / "reference" / (track if domain == "credit" else "ood")
        folder.mkdir(parents=True, exist_ok=True)
        prefix = "results" if domain == "credit" else "ood_results"
        path = folder / f"{prefix}_reference_{track}_{model}.csv"
        names = ("development", "held-out") if domain == "credit" else ("ood-a", "ood-b")
        frame = pd.DataFrame([
            dict(dataset=name, model=model, seed=0, fold=fold, status="ok", protocol=PROTOCOL,
                 roc_auc=.5 + dataset_index * .2 + fold * .01,
                 context_cap=None, info_row_cap=None, subsampled=False, n_train=80, n_train_used=80,
                 prep_features_in=8, prep_features_used=8, n_features_used=8, pred_nonfinite_frac=0.)
            for dataset_index, name in enumerate(names) for fold in range(5)])
        frame.to_csv(path, index=False)
        return path, frame

    return write


@pytest.mark.parametrize("track", ["pd", "lgd"])
@pytest.mark.parametrize("backbone", ["tabicl", "tabpfn3"])
@pytest.mark.parametrize("domain", ["credit", "ood"])
def test_released_score_uses_its_own_complete_reference_before_any_cpt_results(
        shared_reference_files, track, backbone, domain):
    from src.experiments.results import released_score
    path, _ = shared_reference_files(track, backbone, domain)
    # Only a shared reference exists: no final or early CPT benchmark files.
    assert released_score(track, backbone, domain, "roc_auc") == pytest.approx(.62)
    path.unlink()
    assert released_score(track, backbone, domain, "roc_auc") is None


@pytest.mark.parametrize("domain", ["credit", "ood"])
@pytest.mark.parametrize("fault", ["missing_fold", "missing_dataset", "duplicate", "wrong_identity",
                                  "old_protocol", "failed_cell", "missing_status", "missing_protocol"])
def test_released_score_withholds_incomplete_or_invalid_reference(shared_reference_files, domain, fault):
    from src.experiments.results import released_score
    path, frame = shared_reference_files("pd", "tabicl", domain)
    if fault == "missing_fold":
        frame = frame.iloc[:-1]
    elif fault == "missing_dataset":
        frame = frame.iloc[:5]
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif fault == "wrong_identity":
        frame["model"] = "tabpfn3"
    elif fault == "old_protocol":
        frame["protocol"] = "old"
    elif fault == "failed_cell":
        frame.loc[0, "status"] = "failed"
    else:
        frame = frame.drop(columns=fault.removeprefix("missing_"))
    frame.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="Withheld"):
        assert released_score("pd", "tabicl", domain, "roc_auc") is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "invalid"])
def test_released_score_withholds_nonfinite_or_nonnumeric_metrics(shared_reference_files, value):
    from src.experiments.results import released_score
    path, frame = shared_reference_files("lgd", "tabpfn3", "credit")
    frame["roc_auc"] = frame["roc_auc"].astype(object)
    frame.loc[0, "roc_auc"] = value
    frame.to_csv(path, index=False)
    assert released_score("lgd", "tabpfn3", "credit", "roc_auc") is None


@pytest.mark.parametrize("domain", ["credit", "ood"])
@pytest.mark.parametrize("column,value", [
    ("context_cap", 1024), ("context_cap", 0), ("info_context_cap", 1024),
    ("info_row_cap", 10000), ("subsampled", True), ("n_train_used", 79),
    ("n_train_used", float("nan")), ("prep_features_used", 7), ("n_features_used", 7),
    ("n_features_used", float("nan")), ("pred_nonfinite_frac", .01)])
def test_released_score_cannot_show_capped_or_incomplete_tables_as_full_tables(
        shared_reference_files, domain, column, value):
    from src.experiments.results import released_score
    path, frame = shared_reference_files("pd", "tabicl", domain)
    frame[column] = value
    frame.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="Withheld"):
        assert released_score("pd", "tabicl", domain, "roc_auc") is None


def test_retired_selection_preserves_history_and_cannot_influence_launch(tmp_path, monkeypatch):
    from src.experiments import selection
    from src.utils import paths
    from src.utils.config import expand_with_seeds, load
    monkeypatch.setattr(paths, "experiment_dir", lambda e: tmp_path / f"experiment_{e}")
    receipt = selection.selection_path("cpt")
    receipt.parent.mkdir(parents=True)
    evidence = '{"choices":{"pd/tabicl":{"full_lr":999}}}'
    receipt.write_text(evidence)
    for operation in (lambda: selection.freeze("learning-rate"),
                      lambda: selection.load_selection("cpt", optional=True, for_analysis=True)):
        with pytest.raises(ValueError, match="retired"):
            operation()
    assert selection.main(["--stage", "learning-rate"]) == 2
    assert receipt.read_text() == evidence
    runs = expand_with_seeds(load("config/Exp2_PD.yaml"))
    assert {r["train"]["lr"] for r in runs if r["architecture"] == "tabicl"} == {2e-5, 1e-5}
