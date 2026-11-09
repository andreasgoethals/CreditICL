"""Mechanism probes must preserve controls and compare objectives honestly."""
from types import SimpleNamespace

import pytest
import torch

from src.validation.cpt_probe import OBJECTIVES, bar_objective, probe_config, regression_objective


@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("backbone", ["tabicl", "tabpfn3"])
def test_probe_changes_only_diagnostic_scope_and_requested_rate(task, backbone):
    from src.utils.config import expand_with_seeds, load
    cfg = probe_config(backbone, task, 5e-5)
    production = next(r for r in expand_with_seeds(load(f"config/Exp2_{task.upper()}.yaml", allow_placeholders=True))
                      if r["architecture"] == backbone and r["seed"] == 0
                      and r["prior"]["credit_fraction"] == .5 and r["lr_index"] == 0)
    assert cfg["experiment"] == "cpt_mechanism_probe" and "design" not in cfg
    assert cfg["train"]["max_steps"] == 32 and cfg["train"]["schedule_total_steps"] == 1500
    assert cfg["train"]["batch_size"] == production["train"]["batch_size"]
    for key in ("optimizer", "weight_decay", "gradient_clipping", "amp", "warmup_proportion"):
        assert cfg["train"][key] == production["train"][key]
    for key in ("train_frac_range", "n_features_range", "encoding", "credit_fraction"):
        assert cfg["prior"][key] == production["prior"][key]
    assert production["train"]["max_steps"] == 1500 and production["train"]["lr"] != 5e-5


@pytest.mark.parametrize("target", [0., 1., 1e25, 1e69])
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_objectives_have_finite_loss_and_gradients(target, objective):
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    logits = torch.zeros(1, 4, 8, dtype=torch.float64, requires_grad=True)
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 9, dtype=torch.float64))
    loss = bar_objective(logits, torch.full((1, 4), target, dtype=torch.float64), bar, objective)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    if objective != "native":
        assert logits.grad.norm() < 10


@pytest.mark.parametrize("target", [0., 2., 1e25])
def test_native_diagnostic_matches_production_loss_and_gradient(target):
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    from src.train.tabpfn_trainer import REGRESSION_LOSS_WEIGHTS, regression_loss
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 9))
    batch = SimpleNamespace(X_query=None, y_query=torch.full((1, 3), target, dtype=torch.float64),
                            znorm_space_bardist=bar, configs=[[SimpleNamespace(target_transform=None)]])
    a = torch.randn(3, 1, 8, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    x = regression_loss(SimpleNamespace(forward=lambda _: (None, [a], None)), batch, 1, REGRESSION_LOSS_WEIGHTS)
    y = regression_objective(SimpleNamespace(forward=lambda _: (None, [b], None)), batch, 1,
                             REGRESSION_LOSS_WEIGHTS, objective="native")
    x.backward()
    y.backward()
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_huber_does_not_zero_out_large_error_gradient_like_loss_clamping():
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 9, dtype=torch.float64))
    a = torch.zeros(1, 1, 8, dtype=torch.float64, requires_grad=True)
    target = torch.full((1, 1), 1e25, dtype=torch.float64)
    robust = bar_objective(a, target, bar, "crps_huber") - bar_objective(a, target, bar, "crps")
    robust.backward()
    assert a.grad.norm() > 0 and torch.isfinite(a.grad).all()


def test_invalid_objective_fails_before_forward():
    with pytest.raises(ValueError):
        bar_objective(None, None, None, "clip_whatever")
