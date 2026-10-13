"""Regression tests for the two GPU failures in VSC job 11681071.

The scaler tests use PyTorch's real CPU GradScaler and injected overflow, so
the optimizer/scheduler contract is tested even on a machine without CUDA.
The CUDA loss and full-model cases also run in Experiment 0's VSC tests.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from src.train.tabpfn_trainer import TabPFNTrainer, regression_loss


def tiny_trainer(*, scale=None, device='cpu'):
    trainer = object.__new__(TabPFNTrainer)
    trainer.model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        trainer.model.weight.fill_(1.)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.01, weight_decay=0.)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=.9)
    trainer.scaler = torch.amp.GradScaler(device, init_scale=scale) if scale else None
    trainer.device = device
    trainer.step = trainer.datasets_seen = trainer.skipped_tasks = trainer.amp_retries = 0
    trainer.grad_clip = 1.
    trainer.log = logging.getLogger('test_amp')
    trainer.telemetry = SimpleNamespace(due_grad=lambda _: False)
    trainer._task_loss = lambda _: trainer.model.weight.square().sum()
    return trainer


def test_overflow_retries_identical_batch_and_commits_one_update():
    t = tiny_trainer(scale=8.)
    baseline = tiny_trainer()
    seen = []
    draws = []

    def loss(task):
        seen.append(task['item'])
        draws.append(torch.rand(3))
        return t.model.weight.square().sum()

    t._task_loss = loss
    # Each task contributes half the update: scaled gradient > 2 overflows.
    t.model.weight.register_hook(lambda g: torch.full_like(g, float('inf')) if g.abs().max() > 2 else g)
    batch = [{'item': 'first'}, {'item': 'second'}]
    stats = t.train_step(batch)
    baseline.train_step(batch)
    assert stats['amp_retries'] == 2
    assert t.amp_retries == 2 and t.scaler.get_scale() == 2.
    assert seen == ['first', 'second'] * 3
    for offset in (2, 4):
        torch.testing.assert_close(draws[0], draws[offset], rtol=0, atol=0)
        torch.testing.assert_close(draws[1], draws[offset+1], rtol=0, atol=0)
    torch.testing.assert_close(t.model.weight, baseline.model.weight, rtol=0, atol=0)
    assert t.datasets_seen == 2 and t.skipped_tasks == 0
    assert t.scheduler.state_dict() == baseline.scheduler.state_dict()
    assert int(t.optimizer.state[t.model.weight]['step']) == 1
    assert t.model.weight.grad is None


@pytest.mark.parametrize('scale', [None, 8.])
def test_persistent_nonfinite_gradients_cannot_update_or_advance(scale):
    t = tiny_trainer(scale=scale)
    before = t.model.weight.detach().clone()
    schedule = t.scheduler.state_dict().copy()
    t.model.weight.register_hook(lambda g: torch.full_like(g, float('nan')))
    with pytest.raises(FloatingPointError, match='non-finite gradients'):
        t.train_step([{'item': 1}])
    torch.testing.assert_close(t.model.weight, before, rtol=0, atol=0)
    assert t.scheduler.state_dict() == schedule
    assert not t.optimizer.state and t.datasets_seen == 0
    assert t.model.weight.grad is None
    assert t.amp_retries == (16 if scale else 0)


def test_nonfinite_forward_after_one_task_discards_partial_gradients():
    t = tiny_trainer(scale=8.)
    t._task_loss = lambda p: t.model.weight.sum() * (float('nan') if p['item'] == 2 else 1.)
    with pytest.raises(FloatingPointError, match='non-finite loss'):
        t.train_step([{'item': 1}, {'item': 2}])
    assert t.model.weight.grad is None and not t.optimizer.state
    assert t.datasets_seen == 0 and t.scheduler.last_epoch == 0


@pytest.mark.parametrize('batch,match', [([], 'No usable'), ([None], 'No usable'),
                                      ([{'item': 1}, None], 'Incomplete'),
                                      ([{'item': 1}, {'error': 'bad input'}], 'preparation failed')])
def test_failed_or_partial_preparation_never_counts_a_training_update(batch, match):
    t = tiny_trainer()
    with pytest.raises(RuntimeError, match=match):
        t.train_step(batch)
    assert not t.optimizer.state and t.datasets_seen == 0 and t.skipped_tasks == 0


def assert_regression_loss_devices(device):
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 5))
    batch = SimpleNamespace(X_query=None, y_query=torch.tensor([[0., 1., 2.]]),
                            znorm_space_bardist=bar,
                            configs=[[SimpleNamespace(target_transform=None)]] * 2)
    logits = [torch.randn(3, 1, 4, device=device, requires_grad=True) for _ in range(2)]
    est = SimpleNamespace(forward=lambda _: (None, logits, None))
    loss = regression_loss(est, batch, 2, dict(ce=0., crps=1., crls=0., mse=1., mae=0.))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in logits)
    assert bar.borders.device.type == 'cpu', 'worker-prepared objects must stay on CPU'
    assert batch.y_query.device.type == 'cpu'
    return loss


def test_regression_loss_has_finite_gradients_without_mutating_prepared_batch():
    assert_regression_loss_devices('cpu')
