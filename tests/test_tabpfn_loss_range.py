"""The native transformed-target loss must survive production batch 685."""
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.train.tabpfn_trainer import REGRESSION_LOSS_WEIGHTS, regression_loss


class TailTransform:
    def __init__(self, value):
        self.value = value

    def transform(self, values):
        return np.full(values.shape, self.value, dtype=values.dtype)


@pytest.mark.parametrize("value,queries", [(1.0339849692701e25, 3), (1e40, 3), (1e18, 512)])
def test_transform_and_native_loss_widen_before_float32_overflow(value, queries):
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss

    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 5))
    query = torch.arange(queries, dtype=torch.float32)[None]
    batch = SimpleNamespace(X_query=None, y_query=query, znorm_space_bardist=bar,
                            configs=[[SimpleNamespace(target_transform=TailTransform(value))]])
    logits = torch.zeros(queries, 1, 4, requires_grad=True)
    estimator = SimpleNamespace(forward=lambda _: (None, [logits], None))
    before = query.clone()
    loss = regression_loss(estimator, batch, 1, REGRESSION_LOSS_WEIGHTS)
    assert loss.dtype == torch.float64 and torch.isfinite(loss)
    expected = _compute_regression_loss(
        logits_BQL=logits.permute(1, 0, 2).double(),
        targets_BQ=torch.full((1, queries), value, dtype=torch.float64),
        bardist_loss_fn=bar.double(), ce_loss_weight=0., crps_loss_weight=1., mse_loss_weight=1.)
    torch.testing.assert_close(loss, expected)
    if queries == 512:
        # Even finite individual squared errors can overflow the native mean's
        # FP32 accumulation. The precision guard must include reduction length.
        errors = torch.full((1, queries), value, dtype=torch.float32).square()
        assert torch.isfinite(errors).all() and not torch.isfinite(errors.mean())
    if value < 1e38:
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
    torch.testing.assert_close(query, before, rtol=0, atol=0)


def test_nonfinite_float64_transform_fails_closed():
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    batch = SimpleNamespace(X_query=None, y_query=torch.ones(1, 2),
                            znorm_space_bardist=FullSupportBarDistribution(torch.linspace(-3, 3, 5)),
                            configs=[[SimpleNamespace(target_transform=TailTransform(float('inf')))]])
    logits = torch.zeros(2, 1, 4, requires_grad=True)
    with pytest.raises(FloatingPointError, match="non-finite float64 target"):
        regression_loss(SimpleNamespace(forward=lambda _: (None, [logits], None)),
                        batch, 1, REGRESSION_LOSS_WEIGHTS)
    assert logits.grad is None


def _regenerate_production_batch_685():
    """Regenerate the failing shared base task with native preprocessing."""
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG, one_thread
    from src.prior.stream import plan_batch
    from src.train.tabpfn_trainer import TabPFNTaskPreparer, resolve_tabpfn_weights
    from src.utils.config import expand_with_seeds, load

    run = next(r for r in expand_with_seeds(load('config/Exp2_LGD.yaml'))
               if r['architecture'] == 'tabpfn3' and r['seed'] == 0
               and r['prior']['credit_fraction'] == 0 and r['lr_index'] == 0)
    slots = [plan_batch(0, 685, run['prior'], 16, regression=True, credit_fraction=f).slots[15]
             for f in (0., .25, .5)]
    assert slots[0] == slots[1] == slots[2] and not slots[0].use_credit
    slot = slots[0]
    with one_thread():
        task = TaskGenerator(run['prior'], 'lgd', PriorRNG(0), stream_seed=0).sample_slot(slot)
        item = TabPFNTaskPreparer('lgd', resolve_tabpfn_weights('lgd'), 2, 0).prepare(
            task.X[:, :max(1, task.meta['d'])].numpy().astype(np.float64),
            task.y.numpy().astype(np.float64), slot.train_size,
            zlib.crc32(b'0:685:15') & 0x7fffffff)
    assert torch.isfinite(task.y).all()
    return run, item


def production_batch_685():
    """Isolate upstream's hash-sensitive set ordering, exactly as pretrain.py does."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = ("import runpy,sys,torch; "
            "ns=runpy.run_path('tests/test_tabpfn_loss_range.py'); "
            "torch.save(ns['_regenerate_production_batch_685'](), sys.argv[1])")
    environment = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='crediticl-production-case-') as directory:
        path = Path(directory) / 'prepared.pt'
        result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=root, env=environment,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr[-5000:]
        return torch.load(path, map_location='cpu', weights_only=False)


def assert_production_batch_685_loss(device):
    from tabpfn.finetuning.data_util import meta_dataset_collator

    _, item = production_batch_685()
    batch = meta_dataset_collator([item['item']])
    logits = [torch.zeros(batch.y_query.shape[-1], 1, batch.znorm_space_bardist.num_bars,
                          device=device, requires_grad=True) for _ in range(2)]
    loss = regression_loss(SimpleNamespace(forward=lambda _: (None, logits, None)),
                           batch, 2, REGRESSION_LOSS_WEIGHTS)
    assert loss.dtype == torch.float64 and torch.isfinite(loss)
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in logits)
    assert batch.y_query.device.type == 'cpu'


def test_production_batch_685_native_preprocessing_and_loss():
    assert_production_batch_685_loss('cpu')


def assert_production_batch_685_update(directory, device):
    """Run the reported task through released weights and a real optimizer update."""
    import copy
    import math

    from src.train.tabpfn_trainer import TabPFNTrainer

    run, item = production_batch_685()
    cfg = copy.deepcopy(run)
    cfg['train'].update(batch_size=1, num_workers=0)
    cfg['train']['warmup_proportion'] = 0.
    cfg['progress'].update(every_datasets=0, n_ood=0)
    cfg['logging'].update(console=False, to_file=False, log_hardware_every=0,
                          log_grad_every=0, log_weights_every=0)
    trainer = TabPFNTrainer(cfg, directory / 'records', device=device,
                           ckpt_dir=directory / 'checkpoints')
    try:
        trainer.maybe_resume()
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        stats = trainer.train_step([item])
        assert math.isfinite(stats['loss']) and trainer.datasets_seen == 1
        assert trainer.scheduler.last_epoch == 1 and trainer.skipped_tasks == 0
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
        assert any(not torch.equal(parameter.detach(), before[name])
                   for name, parameter in trainer.model.named_parameters())
    finally:
        trainer.close()


@pytest.mark.slow
def test_production_batch_685_updates_released_weights(tmp_path):
    from src.prior.rng import one_thread
    with one_thread():
        assert_production_batch_685_update(tmp_path, 'cpu')
