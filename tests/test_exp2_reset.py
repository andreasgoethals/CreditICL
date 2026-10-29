"""An Exp2 reset must reach both tiers and its scratch without touching other work."""
from types import SimpleNamespace

import pytest

from src.utils import clean_run, paths


def put(path, text='evidence'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def exp2_storage(isolated_output, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr('src.utils.run_experiment.queued_job_names', lambda: set())
    scratch = isolated_output / 'gpfs'
    monkeypatch.setenv('VSC_SCRATCH_GPFS1', str(scratch))
    monkeypatch.delenv('VSC_SCRATCH_LUSTRE1', raising=False)
    monkeypatch.setattr(clean_run, '_exp2_accounting_jobs', lambda ids: {'22'} & ids)
    roots = [paths.outputs_dir(), paths.big_outputs_dir()]
    removed, kept = [], []
    for root in roots:
        removed.extend(put(root / 'experiment_2' / p) for p in
                       ['logs/pretrain.log', 'runs/exp2_lgd__s0/summary.json',
                        'checkpoints/exp2_lgd__s0/step-125.ckpt'])
        removed.append(put(root / 'locks/exp2_lgd.lock'))
        for exp in [0, 1, 3]:
            kept.append(put(root / f'experiment_{exp}/checkpoints/keep.ckpt'))
        kept.extend([put(root / 'reference/benchmark/keep.csv'),
                     put(root / 'locks/exp1_pd.lock')])
        removed.append(put(root.parent / 'slurm-crediticl-exp2-lgd-22.out'))
        removed.append(put(root.parent / 'slurm-crediticl-exp2pilots-pd-bench-25.err'))
        removed.append(put(root / 'locks/exp2pilotsbench_pd_a0.lock'))
        kept.append(put(root / 'locks/exp20_pd.lock'))
        removed.append(put(root.parent / 'checkpoints/exp2_old/step-1.ckpt'))
    kept.extend([put(paths.datasets_dir() / 'raw/keep.csv'),
                 put(paths.pretrained_dir() / 'released.ckpt'),
                 put(paths.ood_cache_dir() / 'keep.npz')])
    base = scratch / 'CreditICL/jobs'
    # A destination marker identifies a training job; accounting identifies a
    # preflight-only job, even though it wrote no training marker.
    removed.append(put(base / '21/checkpoints/run/destination.txt',
                       str(roots[1] / 'experiment_2/checkpoints/run')))
    removed.append(put(base / '21/checkpoints/run/step-250.ckpt.tmp'))
    removed.append(put(base / '22/inputs/copy/model.ckpt'))
    kept.append(put(base / '11/checkpoints/run/destination.txt',
                    str(roots[1] / 'experiment_1/checkpoints/run')))
    kept.append(put(base / '11/checkpoints/run/step-12500.ckpt'))
    kept.append(put(base / '33/inputs/unidentified-copy'))
    kept.append(put(scratch / 'JEVPFN/jobs/99/inputs/keep'))
    return removed, kept, base


def test_exp2_reset_previews_then_erases_only_its_output(exp2_storage):
    removed, kept, base = exp2_storage
    count, size = clean_run.reset_exp2()
    assert count == len(removed) and size == sum(p.stat().st_size for p in removed)
    assert all(p.exists() for p in removed + kept)
    clean_run.reset_exp2(clean=True)
    assert all(not p.exists() for p in removed)
    assert all(p.exists() for p in kept)
    assert not (base / '21').exists() and not (base / '22').exists()
    assert clean_run.reset_exp2() == (0, 0)


def test_exp2_reset_refuses_active_crediticl_jobs(exp2_storage, monkeypatch):
    removed, kept, _ = exp2_storage
    monkeypatch.setattr('src.utils.run_experiment.queued_job_names', lambda: {'crediticl-exp2-pd'})
    with pytest.raises(RuntimeError, match='queued/running'):
        clean_run.reset_exp2(clean=True)
    assert all(p.exists() for p in removed + kept)


def test_exp2_reset_refuses_modified_output_before_first_deletion(exp2_storage, monkeypatch):
    removed, kept, _ = exp2_storage
    monkeypatch.setattr(clean_run, '_require_idle_cluster', lambda: removed[-1].write_text('changed'))
    with pytest.raises(RuntimeError, match='changed since planning'):
        clean_run.reset_exp2(clean=True)
    assert all(p.exists() for p in removed + kept)


def test_exp2_reset_rejects_mixed_experiment_scratch(exp2_storage):
    removed, kept, base = exp2_storage
    put(base / '22/checkpoints/foreign/destination.txt', str(paths.big_outputs_dir() / 'experiment_0/checkpoints/x'))
    with pytest.raises(RuntimeError, match='outside Experiment 2'):
        clean_run.reset_exp2(clean=True)
    assert all(p.exists() for p in removed + kept)


def test_exp2_reset_refuses_output_link_before_any_delete(exp2_storage, isolated_output):
    from test_clean_run import _directory_alias

    removed, kept, _ = exp2_storage
    outside = isolated_output / 'outside'
    sentinel = put(outside / 'keep')
    _directory_alias(paths.big_outputs_dir() / 'experiment_2/escape', outside)
    with pytest.raises(RuntimeError, match='link|escaping'):
        clean_run.reset_exp2(clean=True)
    assert sentinel.exists() and all(p.exists() for p in removed + kept)


def test_accounting_classification_rejects_other_names_and_ids(monkeypatch):
    import subprocess

    calls = []
    def accounting(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=(
            '21|crediticl-exp2-lgd|\n22|crediticl-exp1-pd|\n'
            '23|crediticl-exp20-pd|\n24|jevpfn-exp2|\n25|crediticl-exp2pilots-pd|\n999|crediticl-exp2-pd|\n'))
    monkeypatch.setattr(subprocess, 'run', accounting)
    assert clean_run._exp2_accounting_jobs({'21', '22', '23', '24', '25'}) == {'21', '25'}
    assert len(calls) == 2


def test_local_exp2_reset_is_scoped_and_idempotent(tmp_path, monkeypatch):
    root = tmp_path / 'output CreditICL'
    monkeypatch.setattr(paths, 'outputs_dir', lambda: root)
    monkeypatch.setattr(paths, 'big_outputs_dir', lambda: root)
    monkeypatch.setattr(paths, 'on_vsc', lambda: False)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    victim = put(root / 'experiment_2/logs/run.log')
    checkpoint = put(root / 'experiment_1/checkpoints/keep.ckpt')
    assert clean_run.main(['--reset-exp2']) == 0 and victim.exists()
    assert clean_run.main(['--reset-exp2', '--clean']) == 0
    assert not victim.exists() and checkpoint.exists()


def test_reset_exp2_cli_cannot_be_combined_with_broader_reset():
    with pytest.raises(SystemExit):
        clean_run.main(['--reset-exp2', '--reset-output', '--clean'])
