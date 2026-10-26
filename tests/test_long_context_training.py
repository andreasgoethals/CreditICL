"""Variable-length training must preserve per-task weighting and resume exactly."""
import copy

import pytest
import torch

from src.train.loop import Trainer


def _batch():
    g = torch.Generator().manual_seed(41)
    xs = [torch.randn(n, 3, generator=g) for n in (40, 64, 40)]
    ys = [torch.randn(n, generator=g) for n in (40, 64, 40)]
    return xs, ys, torch.tensor([3, 3, 3]), torch.tensor([40, 64, 40]), torch.tensor([25, 50, 25])


def _cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['prior']['grouping'] = {'group_size': 1, 'subgroup_size': 1}
    cfg['prior']['batch_layout'] = 'ragged'
    cfg['train'].update(batch_size=3, micro_batch_size=1, activation_checkpointing=True,
                        warmup_proportion=0.0)
    return cfg


def test_ragged_update_matches_dense_padded_update(tmp_path, lgd_cfg):
    cfg = _cfg(lgd_cfg)
    batch = _batch()
    xs, ys, *metadata = batch
    dense_x, dense_y = torch.zeros(3, 64, 3), torch.zeros(3, 64)
    for i, (x, y) in enumerate(zip(xs, ys)):
        dense_x[i, :len(x)] = x
        dense_y[i, :len(y)] = y
    with Trainer(cfg, tmp_path/'ragged', device='cpu') as a:
        initial = {k: v.clone() for k, v in a.model.state_dict().items()}
        assert a.optimizer.param_groups[0]['lr'] > 0
        stats_a = a.train_step(batch)
        state = {k: v.clone() for k, v in a.model.state_dict().items()}
        assert any(not torch.equal(state[k], v) for k, v in initial.items())
        assert a.datasets_seen == 3
    with Trainer(cfg, tmp_path/'dense', device='cpu') as b:
        stats_b = b.train_step((dense_x, dense_y, *metadata))
        assert stats_a['loss'] == pytest.approx(stats_b['loss'], abs=1e-7)
        for key, value in b.model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize('track', ['pd', 'lgd'])
def test_ragged_trainer_resume_matches_uninterrupted_updates(
        tmp_path, monkeypatch, pd_cfg, lgd_cfg, track):
    """Replay the real prior, recomputed dropout, optimizer and scheduler together."""
    from src.utils import signal_ready

    # Exercise the trainer's stop/save/resume path without changing pytest's
    # process signal handlers or writing a cluster readiness sentinel.
    monkeypatch.setattr(Trainer, '_install_stop_signals', lambda self: None)
    monkeypatch.setattr(signal_ready, 'publish', lambda: None)
    cfg = _cfg(pd_cfg if track == 'pd' else lgd_cfg)
    cfg['train']['max_steps'] = 3
    cfg['model']['dropout'] = 0.2
    cfg['prior']['credit_fraction'] = 0.5
    whole_dir, resumed_dir = tmp_path/'whole', tmp_path/'resumed'

    with Trainer(cfg, whole_dir, device='cpu') as whole:
        initial = {k: v.clone() for k, v in whole.model.state_dict().items()}
        whole.maybe_resume()
        complete = whole.train()
        expected = copy.deepcopy(whole.model.state_dict())
        expected_optimizer = copy.deepcopy(whole.optimizer.state_dict())
        expected_scheduler = copy.deepcopy(whole.scheduler.state_dict())
        expected_rng = torch.get_rng_state().clone()
    assert complete['completed'] and complete['datasets_seen'] == 9
    assert any(not torch.equal(expected[k], v) for k, v in initial.items())

    with Trainer(cfg, resumed_dir, device='cpu') as first:
        first.maybe_resume()
        first._stop_requested = 'SIGUSR1'
        stopped = first.train()
    assert not stopped['completed'] and stopped['steps'] == 1
    assert stopped['datasets_seen'] == 3
    assert (resumed_dir/'checkpoints'/'step-1.ckpt').is_file()

    torch.rand(23)  # Loading the checkpoint must restore the model RNG stream.
    with Trainer(cfg, resumed_dir, device='cpu') as resumed:
        resumed.maybe_resume()
        assert resumed.step == 1 and resumed.datasets_seen == 3
        completed = resumed.train()
        assert completed['completed'] and completed['resumed_at'] == 1
        assert completed['steps'] == 3 and completed['datasets_seen'] == 9
        torch.testing.assert_close(resumed.model.state_dict(), expected, rtol=0, atol=0)
        torch.testing.assert_close(resumed.optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)
        torch.testing.assert_close(resumed.scheduler.state_dict(), expected_scheduler, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


@pytest.mark.parametrize('field,bad', [(2, 4), (3, 39), (4, 0), (4, 40)])
def test_ragged_trainer_rejects_invalid_metadata_before_updating(tmp_path, lgd_cfg, field, bad):
    batch = _batch()
    batch[field][0] = bad
    with Trainer(_cfg(lgd_cfg), tmp_path/'run', device='cpu') as trainer:
        initial = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        with pytest.raises(ValueError, match='shape/split metadata'):
            trainer.train_step(batch)
        assert trainer.datasets_seen == 0
        assert not trainer.optimizer.state
        torch.testing.assert_close(trainer.model.state_dict(), initial, rtol=0, atol=0)


def test_long_context_loader_has_bounded_prefetch(tmp_path, lgd_cfg):
    cfg = _cfg(lgd_cfg)
    cfg['train'].update(num_workers=23, max_workers=4, prefetch_factor=1)
    with Trainer(cfg, tmp_path/'run', device='cpu') as trainer:
        assert trainer.num_workers == 4
        assert trainer.prefetch_factor == 1
        assert trainer.model.creditcl_model_config['recompute'] is True


def test_ragged_groups_cannot_be_combined_into_one_microbatch(tmp_path, lgd_cfg):
    cfg = _cfg(lgd_cfg)
    cfg['train']['micro_batch_size'] = 2
    with pytest.raises(ValueError, match='exceeds'):
        Trainer(cfg, tmp_path/'run', device='cpu')


def test_efficient_attention_policy_excludes_quadratic_fallback():
    from src.models.backends import sdpa_context

    with sdpa_context(require_efficient=True):
        assert not torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.cudnn_sdp_enabled()
    with pytest.raises(ValueError, match='override'):
        sdpa_context(only='MATH', require_efficient=True)


def test_efficient_attention_requires_backend_support(monkeypatch):
    from src.models import backends

    monkeypatch.setattr(backends, '_backends', lambda: {})
    with pytest.raises(RuntimeError, match='efficient CUDA'):
        backends.sdpa_context(require_efficient=True)


@pytest.mark.parametrize('track', ['PD', 'LGD'])
def test_exp2_geometry_and_controls_are_matched_within_backbone(track):
    from src.utils.config import expand_with_seeds, load

    runs = expand_with_seeds(load(f'config/Exp2_{track}.yaml'))
    assert len(runs) == 24
    for architecture in ('tabicl', 'tabpfn3'):
        selected = [r for r in runs if r['architecture'] == architecture]
        assert {r['prior']['credit_fraction'] for r in selected} == {0., .25, .5, 1.}
        prior = [dict(r['prior']) for r in selected]
        for p in prior:
            p.pop('credit_fraction')
        assert all(p == prior[0] for p in prior)
        assert all(r['train'] == selected[0]['train'] for r in selected)
        p, t = selected[0]['prior'], selected[0]['train']
        if architecture == 'tabicl':
            assert p['n_rows_range'] == [400, 60000]
            assert p['n_rows_sampling'] == 'log_uniform'
            assert p['train_frac_range'] == [.79, .81]
            assert p['batch_layout'] == 'ragged'
            assert t['micro_batch_size'] == p['grouping']['group_size'] == 1
            assert t['require_efficient_attention'] and t['activation_checkpointing']
        else:
            assert p['n_rows_range'] == [1024, 1024]
            assert p['batch_layout'] == 'dense'
