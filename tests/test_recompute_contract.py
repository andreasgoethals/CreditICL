"""Activation recomputation changes memory use, not losses, gradients or RNG."""

import pytest
import torch
import torch.nn.functional as F

from src.models.architecture import build_model
from src.models.backends import sdpa_context
from src.train.loop import pinball_loss


@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_upstream_activation_recompute_preserves_training_contract(pd_cfg, lgd_cfg, task, dropout):
    cfg = pd_cfg if task == "pd" else lgd_cfg
    kwargs = dict(cfg["model"], dropout=dropout, num_quantiles=16)
    normal = build_model(task, recompute=False, **kwargs).train()
    recomputed = build_model(task, recompute=True, **kwargs).train()
    recomputed.load_state_dict(normal.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(1, 40, 5, generator=generator)
    y = (torch.randn(1, 40, generator=generator) if task == "lgd"
         else torch.randint(0, 10, (1, 40), generator=generator).float())
    state = torch.get_rng_state().clone()
    losses, gradients, after = [], [], []
    for model in (normal, recomputed):
        torch.set_rng_state(state)
        with sdpa_context():
            prediction = model(x, y[:, :30])
            loss = (pinball_loss(prediction, y[:, 30:], 16) if task == "lgd"
                    else F.cross_entropy(prediction.flatten(end_dim=-2), y[:, 30:].long().flatten()))
            loss.backward()
        losses.append(loss.detach())
        gradients.append({key: None if p.grad is None else p.grad.clone()
                          for key, p in model.named_parameters()})
        after.append(torch.get_rng_state().clone())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert gradients[0].keys() == gradients[1].keys()
    assert any(g is not None and bool(g.abs().sum() > 0) for g in gradients[0].values())
    for key, left in gradients[0].items():
        right = gradients[1][key]
        if left is None:
            assert right is None, key
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=key)
    torch.testing.assert_close(after[0], after[1], rtol=0, atol=0)
