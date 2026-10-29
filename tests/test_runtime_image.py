"""The rebuild archives package bytes, regenerates scripts and retains source evidence."""
import email
import json
import zipfile
from pathlib import Path

import pytest

from src.utils import runtime_image as image


def test_universal_wheel_selects_python3_instead_of_its_first_python2_tag():
    assert image.compatible_tag('Tag: py2-none-any\nTag: py3-none-any\n') == 'py3-none-any'


def test_snapshot_preserves_payload_and_origins_without_transplanting_scripts(tmp_path, monkeypatch):
    site = tmp_path / 'env/lib/python3.12/site-packages'
    info = site / 'example-1.0.dist-info'
    info.mkdir(parents=True)
    (site / 'example.py').write_text('VALUE = 3\n')
    (info / 'METADATA').write_text('Name: example\nVersion: 1.0\n')
    (info / 'WHEEL').write_text('Wheel-Version: 1.0\nTag: py3-none-any\n')
    (info / 'direct_url.json').write_text(json.dumps({'vcs_info': {'commit_id': 'abc'}}))
    binary = tmp_path / 'env/bin/example'
    binary.parent.mkdir()
    binary.write_bytes(b'\x7fELFstandalone-tool')
    (binary.parent / 'old-python-script').write_text('#!/obsolete/venv/python\n')
    monkeypatch.setattr(image.sys, 'prefix', str(tmp_path / 'env'))
    class Distribution:
        metadata = email.message_from_string('Name: example\nVersion: 1.0\n')
        version = '1.0'
        files = [Path('example.py'), Path('example-1.0.dist-info/METADATA'),
                 Path('example-1.0.dist-info/WHEEL'), Path('example-1.0.dist-info/direct_url.json'),
                 Path('../../../bin/example'), Path('../../../bin/old-python-script'),
                 Path('__pycache__/example.pyc')]
        def locate_file(self, path):
            return site / path
        def read_text(self, name):
            path = info / name
            return path.read_text() if path.exists() else None
    monkeypatch.setattr(image.metadata, 'distributions', lambda: [Distribution()])
    result = image.snapshot(tmp_path / 'snapshot')
    wheel = tmp_path / 'snapshot/wheels/example-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read('example.py') == (site / 'example.py').read_bytes()
        assert len(archive.namelist()) == 5
        tool = 'example-1.0.data/scripts/example'
        assert archive.read(tool) == binary.read_bytes()
        assert archive.getinfo(tool).external_attr >> 16 == 0o100755
        assert not any('old-python-script' in name for name in archive.namelist())
        assert all('..' not in Path(n).parts for n in archive.namelist())
        assert b'example.py,sha256=' in archive.read('example-1.0.dist-info/RECORD')
    assert result['packages'][0]['sha256']
    origins = json.loads((tmp_path / 'snapshot/origins.json').read_text())
    assert origins['example-1.0.dist-info']['vcs_info']['commit_id'] == 'abc'
    with pytest.raises(FileExistsError):
        image.snapshot(tmp_path / 'snapshot')
