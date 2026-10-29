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
    trainer.amp = bool(scale)
    trainer.est = SimpleNamespace(use_autocast_=bool(scale))
    trainer.step = trainer.datasets_seen = trainer.skipped_tasks = trainer.amp_retries = 0
    trainer.full_precision_retries = 0
    trainer.l2sp_alpha = 0.
    trainer._l2sp_ref = {}
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
    assert t.full_precision_retries == (1 if scale else 0)
    assert t.amp == bool(scale) and t.est.use_autocast_ == bool(scale)


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


def assert_extreme_regression_replay(device, target_magnitude=3.390504907822858e18):
    """A finite MSE with the target magnitude observed in synthetic batch 217.

    The model output is explicitly cast to half to exercise the float16 gradient
    boundary on CPU as well as CUDA. Full precision must replay the entire batch,
    retaining the original objective, targets, RNG draws and one optimizer step.
    """
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    trainer = tiny_trainer(scale=128., device=device)
    baseline = tiny_trainer(device=device)
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 5))
    target = torch.tensor([[0., target_magnitude, 1.]], dtype=torch.float64)
    batch = SimpleNamespace(X_query=None, y_query=target, znorm_space_bardist=bar,
                            configs=[[SimpleNamespace(target_transform=None)]])
    seen, draws = [], []

    def loss(t, task):
        if t is trainer:
            seen.append(task['item'])
            draws.append(torch.rand(2, device=device).cpu())
        output = t.model(torch.ones(3, 1, device=device))
        output = output.to(torch.float16 if t.est.use_autocast_ else torch.float32)
        logits = torch.cat([output, -output, output * 0, output * .5], dim=-1).unsqueeze(1)
        estimator = SimpleNamespace(forward=lambda _: (None, [logits], None))
        return regression_loss(estimator, batch, 1, dict(ce=0., crps=1., crls=0., mse=1., mae=0.))

    trainer._task_loss = lambda p: loss(trainer, p)
    baseline._task_loss = lambda p: loss(baseline, p)
    tasks = [{'item': 'first'}, {'item': 'second'}]
    original_target = target.clone()
    stats = trainer.train_step(tasks)
    baseline.train_step(tasks)
    assert stats['amp_retries'] == 16 and stats['full_precision_retry'] == 1
    assert trainer.full_precision_retries == 1 and trainer.datasets_seen == 2
    assert trainer.scaler.get_scale() == 128., 'retain the calibrated scale for subsequent AMP batches'
    assert trainer.skipped_tasks == 0 and trainer.scheduler.last_epoch == 1
    assert int(trainer.optimizer.state[trainer.model.weight]['step']) == 1
    assert trainer.amp and trainer.est.use_autocast_
    assert seen == ['first', 'second'] * 18  # initial + 16 backoffs + one float32 replay
    for i in range(2, len(draws), 2):
        torch.testing.assert_close(draws[i], draws[0], rtol=0, atol=0)
        torch.testing.assert_close(draws[i + 1], draws[1], rtol=0, atol=0)
    torch.testing.assert_close(target, original_target, rtol=0, atol=0)
    torch.testing.assert_close(trainer.model.weight, baseline.model.weight, rtol=1e-6, atol=1e-7)
    assert trainer.model.weight.grad is None
    # Restored AMP and scaler state must also work on the following ordinary batch.
    trainer._task_loss = lambda _: trainer.model.weight.square().sum()
    ordinary = trainer.train_step(tasks)
    assert ordinary['full_precision_retry'] == 0 and trainer.full_precision_retries == 1
    assert trainer.datasets_seen == 4 and trainer.scheduler.last_epoch == 2


def test_extreme_regression_target_replays_full_precision_without_changing_the_objective():
    assert_extreme_regression_replay('cpu')


def test_reported_batch_685_target_range_keeps_one_complete_update():
    assert_extreme_regression_replay('cpu', 1.0339849692701e25)


def test_beyond_float32_target_range_keeps_one_complete_update():
    assert_extreme_regression_replay('cpu', 1e40)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('regularizer', [0., .3])
def test_wide_batch_rescaling_preserves_the_true_clipped_gradient(reverse, regularizer):
    trainer = tiny_trainer()
    trainer.model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.copy_(torch.tensor([[1., -1.]]))
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.25)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
    trainer.l2sp_alpha = regularizer
    trainer._l2sp_ref = {'weight': torch.zeros_like(trainer.model.weight)}
    tasks = [{'item': (torch.tensor([1e5, 1.]), -1e35)},
             {'item': (torch.tensor([1., -1.]), -1e40)}]
    if reverse:
        tasks.reverse()
    before = trainer.model.weight.detach().double().clone()
    expected = torch.zeros_like(before)
    for task in tasks:
        x, y = task['item']
        x = x.double().reshape(1, -1)
        expected += 2 * ((before * x).sum() - y) * x / len(tasks)
    expected += regularizer * before
    expected /= torch.linalg.vector_norm(expected) + 1e-6
    trainer._task_loss = lambda task: (trainer.model(task['item'][0]).double() - task['item'][1]).square().mean()
    stats = trainer.train_step(tasks)
    assert stats['backward_divisor'] > 1.
    torch.testing.assert_close(trainer.model.weight.double(), before - .25 * expected, rtol=1e-6, atol=1e-7)
    assert trainer.datasets_seen == 2 and trainer.scheduler.last_epoch == 1


def test_wide_backward_restores_unclipped_gradients_and_regularizer():
    trainer = tiny_trainer()
    trainer.grad_clip = 0.
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.01)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
    trainer.l2sp_alpha = .25
    trainer._l2sp_ref = {'weight': torch.zeros_like(trainer.model.weight)}
    # A huge constant tests normalization without overwhelming the regularizer.
    trainer._task_loss = lambda _: trainer.model.weight.double().square().sum() + 1e60
    stats = trainer.train_step([{'item': 1}])
    assert stats['backward_divisor'] > 1.
    torch.testing.assert_close(trainer.model.weight, torch.tensor([[1 - .01 * 2.25]]))


@pytest.mark.parametrize('max_norm', [1., 0.])
def test_full_precision_gradient_norm_does_not_overflow(max_norm):
    from src.train.tabpfn_trainer import _clip_full_precision_grad_norm

    params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))]
    params[0].grad = torch.tensor([1e30, -1e30])
    params[1].grad = torch.tensor([1e30])
    before = [p.grad.clone() for p in params]
    _clip_full_precision_grad_norm(params, max_norm)
    actual = torch.cat([p.grad for p in params]).double()
    assert torch.isfinite(actual).all()
    if max_norm > 0:
        expected = torch.cat(before).double()
        expected /= torch.linalg.vector_norm(expected)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    else:
        torch.testing.assert_close(actual, torch.cat(before).double(), rtol=0, atol=0)


def test_full_precision_scope_disables_inner_autocast_and_restores_on_error():
    trainer = tiny_trainer(scale=8.)
    with pytest.raises(RuntimeError, match='probe'):
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            with trainer._full_precision():
                assert not trainer.amp and not trainer.est.use_autocast_
                assert not torch.is_autocast_enabled('cpu')
                raise RuntimeError('probe')
    assert trainer.amp and trainer.est.use_autocast_
