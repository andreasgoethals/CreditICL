"""Exercise scientific selection and adapter routes, including incomplete evidence."""
import copy

import numpy as np
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


def test_pilot_uses_the_prefix_of_the_main_learning_rate_schedule():
    from src.train.optim import build_scheduler
    from src.experiments.design import apply_design
    from src.utils.config import expand_with_seeds, load
    base = expand_with_seeds(load("config/Exp2_PD.yaml", allow_placeholders=True))[0]
    pilot = copy.deepcopy(base)
    pilot.update(design="retention_pilot", method="lora", candidate=1)
    apply_design(pilot, {"choices": {"pd/tabicl": {"full_lr": 2e-5}}})
    pilot["train"]["max_steps"] = 500
    opt_a = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=pilot["train"]["lr"])
    opt_b = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=pilot["train"]["lr"])
    a = build_scheduler(opt_a, pilot["train"], pilot["train"]["schedule_total_steps"])
    b = build_scheduler(opt_b, pilot["train"], 2500)
    for step in range(501):
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


def test_two_stage_selection_is_frozen_complete_and_invalidated_by_environment(tmp_path, monkeypatch, isolated_output):
    from src.experiments import selection as selection
    from src.eval import benchmark_status as bs
    from src.utils import preflight, environment
    from src.validation import campaign
    from src.utils.config import expand_with_seeds, load
    monkeypatch.setattr(preflight, "source_fingerprint", lambda: "tested-source")
    monkeypatch.setattr(environment, "environment_fingerprint", lambda: "tested-environment")
    monkeypatch.setattr(campaign, "input_inventory", lambda: {"data": "tested-data"})
    monkeypatch.setattr(bs, "complete", lambda *a, **kw: True)
    records = {}
    monkeypatch.setattr(bs, "request", lambda e, t, i, variant=None: records[e, t, i])
    monkeypatch.setattr(bs, "receipt_path", lambda tag: tmp_path / (tag + ".json"))

    def prepare(exp, variant=None):
        for track in ("pd", "lgd"):
            cfg = load(bs.config_path(exp, track, variant))
            for index, run in enumerate(expand_with_seeds(cfg)):
                tag = f"e{exp}-{track}-{index}"
                metric = "roc_auc" if track == "pd" else "r2"
                # The lower full-model LR wins; the middle adapter candidate wins.
                score = .6 + .01 * (run.get("lr_index", 0) if exp == 2 else int(run["candidate"] == 1))
                rows = [dict(dataset=d, seed=0, fold=f, status="ok", info_run_name=run["_run_name"],
                             **{metric: score if d in cfg["eval"]["dev_datasets"] else -1000.})
                        for d in cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"] for f in range(5)]
                csv = tmp_path / (tag + ".csv")
                pd.DataFrame(rows).to_csv(csv, index=False)
                (tmp_path / (tag + ".json")).write_text('{"valid": true}')
                records[exp, track, index] = (tag, {}, {"credit": csv})

    prepare(2, "pilots")
    first = selection.freeze("learning-rate")
    assert first == selection.freeze("learning-rate")  # idempotent, not a new search
    frozen = selection.load_selection("retention_pilot")
    assert len(frozen["baselines"]) == 12
    assert frozen["choices"]["pd/tabicl"]["full_lr"] == 1e-5
    assert frozen["choices"]["pd/tabpfn3"]["full_lr"] == 1e-6
    prepare(3, "pilots")
    selection.freeze("adapters")
    frozen = selection.load_selection("retention")
    assert len(frozen["candidates"]) == 36
    assert all(c[m] == 1 for c in frozen["choices"].values() for m in ("lora", "diagscale", "l2sp"))
    assert all(not r["design_unresolved"] for r in expand_with_seeds(load("config/Exp3_PD.yaml")))
    monkeypatch.setattr(environment, "environment_fingerprint", lambda: "different-environment")
    with pytest.raises(ValueError, match="environment"):
        load("config/Exp3_PD.yaml")
    # A downloaded receipt can be described on another machine, never used to launch there.
    assert selection.load_selection("retention", for_analysis=True)["choices"] == frozen["choices"]
