"""Appending campaign arms must neither discard valid scores nor hide real changes."""
import copy
import json

import pytest

from src.eval import benchmark_status as bench
from src.utils.benchmark_compat import REVIEWED_TRANSITIONS, compatible_requests
from src.utils.config import expand_with_seeds, load


def requests():
    run = expand_with_seeds(load("config/Exp1_PD.yaml"))[0]
    wanted = {
        "code": {path: bench.digest(bench.ROOT / path) for path in REVIEWED_TRANSITIONS},
        "config": run, "step": 12500, "checkpoint_sha256": "same-weights",
        "protocol": 9, "n_folds": 5, "context_cap": None,
        "environment_fingerprint": "same-environment", "credit_inputs": {"dataset": "same-data"},
    }
    wanted["code"]["src/eval/runner.py"] = "same-evaluator"
    recorded = copy.deepcopy(wanted)
    recorded["config"]["_grid"].update(total=3, array_total=9)
    for path, (old, approved) in REVIEWED_TRANSITIONS.items():
        assert wanted["code"][path] in approved, "This transition must be retired/reviewed after code changes"
        recorded["code"][path] = sorted(old)[0]
    return recorded, wanted


def test_only_the_reviewed_extension_preserves_request_identity():
    recorded, wanted = requests()
    untouched = copy.deepcopy(recorded)
    assert compatible_requests(recorded, wanted)
    assert recorded == untouched
    assert not compatible_requests(None, wanted)
    assert not compatible_requests(wanted, recorded), "The provenance transition is directional"


@pytest.mark.parametrize("path,value", [
    (("checkpoint_sha256",), "different-weights"),
    (("environment_fingerprint",), "changed-package"),
    (("credit_inputs", "dataset"), "changed-data"),
    (("protocol",), 10), (("n_folds",), 3), (("context_cap",), 1024),
    (("step",), 2500), (("config", "seed"), 1),
    (("config", "prior", "credit_fraction"), .25),
    (("config", "train", "lr"), .1),
    (("config", "_grid", "array_index"), 9),
    (("code", "src/eval/runner.py"), "changed-scoring"),
    (("code", "src/eval/selection.py"), "unknown-edit"),
    (("code", "scripts/slurm/benchmark.slurm"), "changed-evaluation-command"),
    (("code", "src/eval/benchmark_status.py"), "changed-receipt-check"),
])
def test_scientific_or_unreviewed_code_changes_still_invalidate_receipts(path, value):
    recorded, wanted = requests()
    node = wanted
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert not compatible_requests(recorded, wanted)


def test_new_evaluation_sources_are_not_ignored():
    recorded, wanted = requests()
    wanted["code"]["src/eval/new_scoring.py"] = "new"
    assert not compatible_requests(recorded, wanted)


def test_receipt_checks_result_bytes_even_for_a_compatible_grid_extension(tmp_path, monkeypatch):
    recorded, wanted = requests()
    result = tmp_path / "result.csv"
    result.write_text("original scores")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"request": recorded, "files": {"credit": bench.digest(result)}}))
    monkeypatch.setattr(bench, "request", lambda *a, **k: ("arm", wanted, {"credit": result}))
    monkeypatch.setattr(bench, "receipt_path", lambda tag: receipt)
    assert bench.check(1, "pd", 0)
    result.write_text("changed scores")
    assert not bench.check(1, "pd", 0)
