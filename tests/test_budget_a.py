"""Selection leakage, schedule prefixes and the approved smaller campaign."""
import copy

import pytest
import torch

from src.experiments.design import apply_design
from src.train.optim import build_scheduler
from src.utils.config import expand_with_seeds, load


@pytest.mark.parametrize("task", ["PD", "LGD"])
def test_pilot_datasets_and_seeds_are_separate_from_main(task, isolated_output):
    cfg = load(f"config/Exp2_{task}.yaml", allow_placeholders=True)
    pilot_cfg = load(f"config/Exp2_{task}_pilots.yaml")
    main, pilots = expand_with_seeds(cfg), expand_with_seeds(pilot_cfg)
    assert len(main) == 24 and len(pilots) == 8
    assert all(r["design_unresolved"] for r in main)
    with pytest.raises(ValueError, match="selection is not frozen"):
        load(f"config/Exp2_{task}.yaml")
    assert pilot_cfg["eval"]["dev_datasets"] == cfg["eval"]["dev_datasets"]
    assert not pilot_cfg["eval"]["holdout_datasets"]
    assert not pilot_cfg["eval"]["include_ood"]
    assert {r["seed"] for r in pilots}.isdisjoint(r["seed"] for r in main)
    assert {r["prior"]["credit_fraction"] for r in pilots} == {0., .5}
    assert all(r["train"]["keep_steps"] == [500] and r["train"]["max_steps"] == 500
               and r["progress"]["every_datasets"] == 0 for r in pilots)
    assert all(r["train"]["l2sp_alpha"] == 0. for r in main)


@pytest.mark.parametrize("model,rate", [("tabicl", 1e-5), ("tabpfn3", 1e-6)])
def test_main_and_lr_pilot_have_identical_first_500_schedule_updates(model, rate):
    pilots = expand_with_seeds(load("config/Exp2_PD_pilots.yaml"))
    pilot = next(r for r in pilots if r["architecture"] == model and r["lr_index"] == 1)
    main = copy.deepcopy(pilot)
    main.update(design="cpt", lr_index=0)
    main["train"]["max_steps"] = 2500
    apply_design(main, {"choices": {f"pd/{model}": {"full_lr": rate}}})
    optimizers = [torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=rate) for _ in range(2)]
    schedulers = [build_scheduler(o, r["train"], r["train"]["schedule_total_steps"])
                  for o, r in zip(optimizers, (pilot, main))]
    for _ in range(501):
        assert optimizers[0].param_groups[0]["lr"] == optimizers[1].param_groups[0]["lr"]
        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step()
            scheduler.step()


def test_same_frozen_rate_applies_to_every_mixture_and_retention_baseline():
    from src.experiments.design import RATES
    cfg = load("config/Exp2_PD.yaml", allow_placeholders=True)
    receipt = {"choices": {f"pd/{m}": {"full_lr": rates[1]} for m, rates in RATES.items()}}
    for run in expand_with_seeds(cfg):
        apply_design(run, receipt)
        assert not run["design_unresolved"]
        assert run["train"]["lr"] == RATES[run["architecture"]][1]
        no_decay = copy.deepcopy(run)
        no_decay.update(design="retention", method="no_decay")
        apply_design(no_decay, receipt)
        assert no_decay["train"]["lr"] == run["train"]["lr"]
        assert no_decay["train"]["schedule_total_steps"] == run["train"]["schedule_total_steps"] == 2500


def test_checkpoint_plot_draws_measured_milestones_and_step_zero(monkeypatch):
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.visualize import continuation as plots

    frame = pd.DataFrame([
        dict(architecture="tabicl", lr_index=0, mixture=fraction,
             checkpoint_step=step, training_seed=seed, dataset="held-out",
             roc_auc=.6 + step / 100000 + fraction / 100)
        for fraction in (0, .25, .5, 1) for step in (125, 250, 500, 1000, 2500)
        for seed in (0, 1, 2)])
    monkeypatch.setattr(plots.results, "benchmark_frames", lambda *a, **k: frame)
    monkeypatch.setattr(plots.results, "released_score", lambda *a: .6)
    fig = plots.checkpoint_curves("pd", "tabicl")
    try:
        for ax in fig.axes:
            assert len(ax.lines) == 5  # Four mixtures plus released reference.
            assert list(ax.lines[0].get_xdata()) == [125, 250, 500, 1000, 2500]
            assert ax.lines[-1].get_ydata() == [.6, .6]
    finally:
        plt.close(fig)
