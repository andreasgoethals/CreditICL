"""Paid training is submitted only after both validation and all Exp1 receipts pass."""
from types import SimpleNamespace

import pytest

from src.utils import deploy as d


@pytest.mark.parametrize('failure', ['cpu', 'gpu', 'benchmark', None])
def test_deployment_gates_exp2_on_actual_acceptance_and_complete_scores(monkeypatch, failure):
    commands, submitted, phases = [], [], []
    def command(args):
        commands.append(args)
        if failure == 'cpu' and '--require-exp0-cpu' in args:
            raise RuntimeError('CPU failed')
        if failure == 'gpu' and '--require-exp0' in args:
            raise RuntimeError('GPU failed')
        return ''
    monkeypatch.setattr(d, 'command', command)
    monkeypatch.setattr(d, 'submit', lambda args, receipt: [('123', 'wice')])
    monkeypatch.setattr(d, 'wait_jobs', lambda jobs: None)
    counts = {1: 0}
    def stages(exp, kind):
        if exp == 1:
            counts[1] += 1
            return [SimpleNamespace(complete=counts[1] > 1 and failure != 'benchmark', total=60)]
        return [SimpleNamespace(complete=False, total=32) for _ in range(2)]
    monkeypatch.setattr(d, 'production_stages', stages)
    def submit_stages(stages, receipt_root):
        submitted.append(sum(s.total for s in stages))
        return [('456', 'mindwell')]
    monkeypatch.setattr(d, 'submit_stage_commands', submit_stages)
    record = {}
    if failure:
        with pytest.raises(RuntimeError):
            d.pipeline(record, lambda r: phases.append(r['phase']))
        assert 64 not in submitted
    else:
        d.pipeline(record, lambda r: phases.append(r['phase']))
        assert submitted == [60, 64]
        assert record['status'] == 'COMPLETE' and record['phase'] == 'exp2_submitted'


def test_cluster_label_does_not_keep_a_finished_job_waiting(monkeypatch):
    monkeypatch.setattr(d, 'command', lambda args: 'CLUSTER: wice\n' if args[0] == 'squeue'
                        else '123|COMPLETED|0:0\n')
    monkeypatch.setattr(d.time, 'sleep', lambda _: pytest.fail('finished job must not wait'))
    d.wait_jobs([('123', 'wice')])


def test_controller_waits_for_every_admitted_array_chunk(monkeypatch, tmp_path):
    calls = []
    def submit(args, receipt):
        calls.append((args, receipt))
        return [('123', 'mindwell'), ('124', 'mindwell')]
    monkeypatch.setattr(d, 'submit', submit)
    stage = SimpleNamespace(complete=False, state='ready', name='exp1-pd-benchmark',
                            commands=[['sbatch', '--array=9-11', 'job.slurm']])
    assert d.submit_stage_commands([stage], tmp_path) == [('123', 'mindwell'), ('124', 'mindwell')]
    assert calls[0][1] == tmp_path / 'exp1-pd-benchmark-0.json'
