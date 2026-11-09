"""Local analysis needs evidence, never transferred model weights or runtime state."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.utils.result_manifest import analysis_manifest, main


def member(path, size=12):
    return dict(path=path, bytes=size, sha256='a' * 64, tier='project')


def test_analysis_plan_keeps_scores_logs_and_provenance_without_weights_or_archives():
    wanted = ['experiment_1/benchmark/pd/results.csv', 'experiment_1/runs/arm/metrics.jsonl',
              'experiment_0/validation/run/logs/unit.log', 'reference/benchmark/receipts/a.json',
              'figures/plot/chart.pdf', 'All_Results.md', 'logs/preprocess.log']
    excluded = ['experiment_1/checkpoints/arm/step-12500.ckpt',
                'experiment_0/validation/run/checkpoints/model.json',
                'experiment_1/weights.safetensors', 'experiment_1/optimizer.pt',
                'experiment_1/state.pth', 'experiment_1/runtime.bin', 'experiment_1/full-output.zip',
                'experiment_1/export.tar.gz', 'experiment_1/receipt.lock', 'data/processed/data.csv']
    original = dict(members=[member(p) for p in wanted + excluded])
    before = copy.deepcopy(original)
    filtered = analysis_manifest(original)
    assert [m['path'] for m in filtered['members']] == wanted
    assert {m['path'] for m in filtered['excluded']} == set(excluded)
    assert filtered['bytes'] == 12 * len(wanted)
    assert original == before


@pytest.mark.parametrize('path', ['/tmp/report.csv', '../report.csv', 'experiment_1/../report.csv',
    'C:/report.csv', 'experiment_1\\report.csv', 'experiment_1//report.csv',
    'experiment_1/result.csv:stream', 'experiment_1/x\nget x.csv', 'experiment_1/CON/report.csv',
    'experiment_1/run. /report.csv', 'experiment_1/"quote.csv',
    'experiment_1/x\treport.csv', 'experiment_1/x\x01report.csv'])
def test_download_plan_rejects_unsafe_windows_or_sftp_destinations(path):
    with pytest.raises(ValueError, match='Unsafe'):
        analysis_manifest(dict(members=[member(path)]))


def test_conflicting_tiers_or_windows_case_collisions_cannot_overwrite_local_results():
    a = member('experiment_1/results.csv')
    assert analysis_manifest(dict(members=[a, dict(a, tier='data')]))['files'] == 1
    for other in (dict(a, bytes=13), dict(a, path='experiment_1/RESULTS.csv')):
        with pytest.raises(ValueError, match='Conflicting'):
            analysis_manifest(dict(members=[a, other]))


def test_cli_preserves_original_inventory_and_source_hash(tmp_path):
    source, output = tmp_path / 'inventory.json', tmp_path / 'analysis.json'
    source.write_text(json.dumps(dict(members=[member('experiment_1/results.csv')])), encoding='utf-8')
    before = source.read_bytes()
    assert main([str(source), '--out', str(output)]) == 0
    assert source.read_bytes() == before
    assert json.loads(output.read_text())['source_inventory_sha256'] == hashlib.sha256(before).hexdigest()
    with pytest.raises(SystemExit):
        main([str(source), '--out', str(source)])
    assert source.read_bytes() == before


def test_reviewed_results_docstring_change_preserves_receipts_but_not_changed_scores():
    from src.utils.benchmark_compat import compatible_requests
    old = dict(code={'src/eval/results.py': 'b1b4efe3b98c2c40829c81a22a7295617762437369ce06c8608dca4cdb943667'},
               protocol=9, checkpoint_sha256='original', scoring='original')
    new = copy.deepcopy(old)
    new['code']['src/eval/results.py'] = '119269160fa306757f1be012e0382b19962fac7357dbb3e71d3443cd31db8210'
    assert compatible_requests(old, new)
    assert not compatible_requests(old, dict(new, scoring='different'))
    assert not compatible_requests(old, dict(new, checkpoint_sha256='different'))


def test_storage_probe_never_overwrites_an_existing_probe(tmp_path):
    import os
    from src.utils.check_storage import _can_write
    existing = tmp_path / f'.crediticl_write_probe_{os.getpid()}'
    existing.write_text('preserve')
    assert _can_write(tmp_path) == (True, 'writable')
    assert existing.read_text() == 'preserve'
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize('writable', [True, False])
def test_storage_advice_does_not_authorize_data_fallback_or_foreign_directory_moves(tmp_path, monkeypatch, capsys, writable):
    from src.utils import check_storage, paths
    for name in ('big_outputs_dir', 'ood_cache_dir', 'outputs_dir', 'pretrained_dir',
                 'prior_cache_root', 'processed_dir', 'staging_root'):
        monkeypatch.setattr(paths, name, lambda: tmp_path)
    monkeypatch.setattr(check_storage.sys, 'argv', ['check_storage', '--fix'])
    monkeypatch.setattr(check_storage.os, 'getuid', lambda: -1, raising=False)
    monkeypatch.setattr(check_storage, '_can_write', lambda p: (writable, 'fixture'))
    monkeypatch.setattr(Path, 'chmod', lambda *a: pytest.fail('No verified ownership'))
    assert check_storage.main() == int(not writable)
    out = capsys.readouterr().out
    assert 'Submit away' not in out and '.foreign' not in out and 'fallback to $VSC_DATA works' not in out
    assert ('Experiment 0' if writable else 'no fallback to DATA') in out
