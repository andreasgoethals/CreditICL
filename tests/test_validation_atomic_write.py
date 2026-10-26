"""A transient Windows file lock must not corrupt or falsely pass validation."""
import json
from pathlib import Path

import pytest

from src.validation import campaign


@pytest.mark.parametrize("failures,winerror,expected_calls", [(2, 5, 3), (99, 32, 5), (99, None, 1)])
def test_receipt_publish_retries_only_bounded_windows_lock_errors(
        tmp_path, monkeypatch, failures, winerror, expected_calls):
    target = tmp_path / "validation.json"
    target.write_text(json.dumps({"status": "RUNNING"}))
    original_replace = Path.replace
    calls, sleeps = [], []

    def locked_replace(source, destination):
        calls.append(source)
        if len(calls) <= failures:
            assert json.loads(target.read_text()) == {"status": "RUNNING"}
            error = PermissionError("file locked")
            if winerror is not None:
                error.winerror = winerror
            raise error
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(campaign.time, "sleep", sleeps.append)
    if failures < expected_calls:
        campaign.write_json(target, {"status": "PASS"})
        assert json.loads(target.read_text()) == {"status": "PASS"}
        assert not target.with_suffix(".json.tmp").exists()
    else:
        with pytest.raises(PermissionError):
            campaign.write_json(target, {"status": "PASS"})
        assert json.loads(target.read_text()) == {"status": "RUNNING"}
    assert len(calls) == expected_calls
    assert len(sleeps) == expected_calls - 1
