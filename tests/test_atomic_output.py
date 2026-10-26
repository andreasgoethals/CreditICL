"""Transient sharing locks cannot destroy complete benchmark evidence."""
import json

import pandas as pd
import pytest

from src.utils import atomic_output


@pytest.mark.parametrize("format", ["json", "csv"])
@pytest.mark.parametrize("failure", ["transient", "persistent", "other"])
def test_publication_retries_only_bounded_windows_locks(tmp_path, monkeypatch, format, failure):
    from src.eval.progress import atomic_json

    target = tmp_path / ("result." + format)
    target.write_text("previous complete result")
    replace = atomic_output.os.replace
    attempts = []

    def locked(source, destination):
        attempts.append(1)
        assert target.read_text() == "previous complete result"
        if failure != "transient" or len(attempts) < 3:
            error = PermissionError("fixture sharing lock")
            if failure != "other":
                error.winerror = 5
            raise error
        replace(source, destination)

    monkeypatch.setattr(atomic_output.os, "replace", locked)
    monkeypatch.setattr(atomic_output.time, "sleep", lambda _: None)
    def publish():
        if format == "json":
            atomic_json(target, {"new": 3})
        else:
            atomic_output.write_csv(pd.DataFrame({"new": [3]}), target)

    if failure == "transient":
        publish()
        assert len(attempts) == 3
        actual = json.loads(target.read_text())["new"] if format == "json" else pd.read_csv(target).new.iloc[0]
        assert actual == 3
    else:
        with pytest.raises(PermissionError):
            publish()
        assert len(attempts) == (5 if failure == "persistent" else 1)
        assert target.read_text() == "previous complete result"
    assert not list(tmp_path.glob("*.tmp"))
