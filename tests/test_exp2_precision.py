"""Exercise nested autocast rather than trusting an AMP flag in a run card."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def assert_task_precision(monkeypatch, device, amp):
    from src.train import tabpfn_trainer as module
    from tabpfn.finetuning import data_util
    from tabpfn.utils import get_autocast_context

    trainer = object.__new__(module.TabPFNTrainer)
    trainer.device, trainer.amp = device, amp
    trainer.regression, trainer.n_estimators = False, 2
    trainer.performance = None
    fits, observed = [], []
    trainer.est = SimpleNamespace(use_autocast_=not amp)
    trainer.est.fit_from_preprocessed = lambda *a, **k: fits.append(trainer.est.use_autocast_)
    batch = SimpleNamespace(X_context=None, y_context=None, cat_indices=None, configs=None)
    monkeypatch.setattr(data_util, 'meta_dataset_collator', lambda _: batch)

    def loss(est, batch, members):
        # The real batched executor uses this same context helper internally.
        with get_autocast_context(torch.device(device), enabled=est.use_autocast_):
            x = torch.ones(4, 4, device=device, requires_grad=True)
            out = x @ x
            observed.append(out.dtype)
        return out.float().sum()

    monkeypatch.setattr(module, 'classification_loss', loss)
    # An enclosing autocast must not override an explicitly disabled trainer.
    outer = torch.autocast(device, dtype=torch.bfloat16) if not amp else nullcontext()
    with outer:
        value = trainer._task_loss({'item': object(), 'n_classes': 2})
    value.backward()
    assert fits == [amp]
    assert observed == [torch.float16 if amp else torch.float32]
    assert torch.isfinite(value)


def test_disabled_amp_disables_nested_executor_and_ambient_autocast(monkeypatch):
    assert_task_precision(monkeypatch, 'cpu', False)


def test_four_mixtures_have_distinct_redundant_plot_encodings():
    from src.visualize import style
    fractions = (0., .25, .5, 1.)
    for encode in (style.credit_fraction_colour, style.credit_fraction_marker, style.credit_fraction_line):
        assert len({encode(f) for f in fractions}) == 4


@pytest.mark.parametrize('track', ['pd', 'lgd'])
def test_quarter_mixture_is_nested_and_control_tasks_remain_paired(track):
    from src.prior.stream import plan_batch
    from src.utils.config import expand_with_seeds, load

    runs = expand_with_seeds(load(f'config/Exp2_{track.upper()}.yaml', allow_placeholders=True))
    for arm in ('tabicl', 'tabpfn3'):
        cfg = next(c for c in runs if c['arm'] == arm and c['seed'] == 0
                   and c['prior']['credit_fraction'] == .25)
        size = cfg['train']['batch_size']
        for batch_index in (0, 1, 10001):
            kwargs = dict(regression=track == 'lgd')
            quarter = plan_batch(0, batch_index, cfg['prior'], size, credit_fraction=.25, **kwargs)
            control = plan_batch(0, batch_index, cfg['prior'], size, credit_fraction=0., **kwargs)
            half = plan_batch(0, batch_index, cfg['prior'], size, credit_fraction=.5, **kwargs)
            # Mixtures are per-slot Bernoulli probabilities, not fixed quotas.
            assert all(not a.use_credit or b.use_credit for a, b in zip(quarter.slots, half.slots))
            assert quarter.seq_lens == control.seq_lens
            assert quarter.train_sizes == control.train_sizes
            for a, b in zip(quarter.slots, control.slots):
                if not a.use_credit:
                    assert a == b
