"""Quota failures must preserve previous checkpoints and report storage access clearly."""
import errno
from types import SimpleNamespace

import pytest
import torch


def test_provenance_hash_never_stages_an_input_copy(tmp_path, monkeypatch):
    from src.eval.benchmark_status import digest, input_digest

    path = tmp_path / 'validation-checkpoint.ckpt'
    path.write_bytes(b'validated bytes')
    monkeypatch.setattr('src.utils.working_storage.local_input',
                        lambda *_: pytest.fail('hashing copied a validation artifact'))
    assert input_digest(path) == digest(path)


def test_tabicl_failed_save_removes_partial_temp_and_keeps_previous(tmp_path, monkeypatch):
    from src.train.checkpoint import save_checkpoint

    old = tmp_path / 'step-125.ckpt'
    old.write_bytes(b'previous complete checkpoint')
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    def fail(payload, path):
        path.write_bytes(b'partial')
        raise OSError(errno.EDQUOT, 'Disk quota exceeded')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='quota'):
        save_checkpoint(tmp_path, step=125, model=model, optimizer=optimizer,
                        scheduler=SimpleNamespace(state_dict=lambda: {}), scaler=None, config={})
    assert old.read_bytes() == b'previous complete checkpoint'
    assert not list(tmp_path.glob('*.tmp'))


def test_tabpfn_failed_save_removes_partial_temp_and_keeps_previous(tmp_path, monkeypatch):
    from src.train.tabpfn_trainer import TabPFNTrainer

    trainer = object.__new__(TabPFNTrainer)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.est = SimpleNamespace()
    trainer.run_name, trainer.step, trainer.cfg = 'exp2_lgd', 125, {}
    old = tmp_path / 'step-125.ckpt'
    old.write_bytes(b'previous complete checkpoint')
    def fail(estimator, path, **kwargs):
        path.write_bytes(b'partial')
        raise OSError(errno.EDQUOT, 'Disk quota exceeded')
    monkeypatch.setattr('tabpfn.model_loading.save_tabpfn_model', fail)
    with pytest.raises(OSError, match='quota'):
        trainer.save_weights(old)
    assert old.read_bytes() == b'previous complete checkpoint'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('number', [errno.EDQUOT, errno.ENOSPC, errno.EACCES])
def test_quota_during_validation_is_not_reported_as_a_failed_experiment(
        isolated_output, monkeypatch, number):
    from test_validation import fake_campaign

    from src.validation import campaign

    fake_campaign(monkeypatch, isolated_output)
    assert campaign.run_validation() == 0
    def fail(_):
        raise OSError(number, 'storage inaccessible')
    monkeypatch.setattr('src.eval.benchmark_status.input_digest', fail)
    row = campaign.readiness('gpu')[0]
    assert not row.ok and row.name == 'Validation storage access'
    assert 'Restore storage capacity/access' in row.detail
    assert 'Run the CPU job' not in row.detail
