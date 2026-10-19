"""Crash recovery must preserve exactly the same full-table five-fold experiment."""
import time

import numpy as np
import pandas as pd
import pytest

from src.eval.progress import CellJournal, DEADLINE_ENV, EvaluationDeferred


def cell(fold=0, status="ok"):
    return dict(dataset="table", model="linear", seed=0, fold=fold, status=status)


def test_committed_fold_survives_kill_before_csv_publication(tmp_path, monkeypatch):
    import src.eval.progress as progress

    path = tmp_path / "result.csv"
    with CellJournal(path, "frozen") as journal:
        def interrupted(*_):
            raise KeyboardInterrupt
        monkeypatch.setattr(progress, "write_csv", interrupted)
        with pytest.raises(KeyboardInterrupt):
            journal.save(cell())
    monkeypatch.undo()
    with CellJournal(path, "frozen") as journal:
        assert journal.get(cell()) == cell()
        assert len(pd.read_csv(path)) == 1


def test_only_successful_cells_are_reused(tmp_path):
    with CellJournal(tmp_path / "result.csv", "frozen") as journal:
        journal.save(cell(0, "failed"))
        journal.save(cell(1))
        assert journal.get(cell(0)) is None
        assert journal.get(cell(1)) == cell(1)
        journal.save(cell(0))
        assert len(journal.rows) == 2


@pytest.mark.parametrize("change", ["recipe", "malformed", "legacy"])
def test_stale_or_malformed_scores_cannot_be_reused(tmp_path, change):
    path = tmp_path / "result.csv"
    with CellJournal(path, "original") as journal:
        journal.save(cell())
    if change == "malformed":
        journal.path.write_text('{"schema":1,"identity":"original","cells":{"bad":{}}}')
    if change == "legacy":
        journal.path.unlink()
    with pytest.raises((ValueError, KeyError)):
        with CellJournal(path, "changed" if change == "recipe" else "original"):
            pass


def test_insufficient_walltime_defers_without_marking_a_cell_failed(tmp_path, monkeypatch):
    with CellJournal(tmp_path / "result.csv", "frozen") as journal:
        journal.save({**cell(), "total_seconds": 1000})
        monkeypatch.setenv(DEADLINE_ENV, str(time.time() + 500))
        with pytest.raises(EvaluationDeferred):
            journal.begin(cell(1))
        assert journal.get(cell(1)) is None
        assert len(journal.rows) == 1


@pytest.mark.parametrize("domain", ["credit", "ood"])
def test_interrupted_five_folds_resume_only_missing_cells_with_all_rows_and_features(tmp_path, monkeypatch, domain):
    from src.eval import runner, ood_runner
    from src.eval.ood import OODDataset

    X = np.random.default_rng(7).normal(size=(57, 129)).astype(np.float32)
    y = np.arange(57) % 2
    monkeypatch.setattr(runner, "load_arrays", lambda *a, **k: (X, y, [], {"n_features": 129}))
    monkeypatch.setattr(ood_runner, "_prepare", lambda *a, **k: (X, y, [], {}))
    seen, interrupted = [], False

    def score(task, model, Xtr, ytr, Xte, yte, cats, *, fold, stages, **kwargs):
        nonlocal interrupted
        if len(seen) == 2 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        assert Xtr.shape[1] == Xte.shape[1] == 129
        assert len(Xtr) + len(Xte) == len(X)
        assert len(stages["final"].X_fit) == len(Xtr)
        seen.append(fold)
        return dict(_headline="finite", roc_auc=.5, n_train_used=len(Xtr), n_features_used=129)

    monkeypatch.setattr(runner if domain == "credit" else ood_runner, "score_fold", score)
    def evaluate(journal):
        if domain == "credit":
            return runner.evaluate_dataset("pd", "table", ["linear"], 0, journal=journal)
        return ood_runner.evaluate_ood_dataset(OODDataset("table", 7, "classification", 57, 129),
                                               ["linear"], 0, ood_runner.OODEvalConfig(), journal=journal)
    path = tmp_path / "results.csv"
    with CellJournal(path, "frozen") as journal:
        with pytest.raises(KeyboardInterrupt):
            evaluate(journal)
    assert seen == [0, 1]
    assert len(pd.read_csv(path)) == 2
    with CellJournal(path, "frozen") as journal:
        rows = evaluate(journal)
    assert seen == list(range(5))
    assert len(rows) == len(pd.read_csv(path)) == 5
    with CellJournal(path, "frozen") as journal:
        assert evaluate(journal) == rows
    assert seen == list(range(5))


def test_journal_lock_prevents_two_writers(tmp_path):
    from filelock import Timeout
    path = tmp_path / "results.csv"
    with CellJournal(path, "frozen"):
        with pytest.raises(Timeout):
            with CellJournal(path, "frozen"):
                pass


@pytest.mark.parametrize("change", ["data", "weights", "code", "packages", "settings"])
def test_credit_progress_identity_covers_every_input(tmp_path, monkeypatch, change):
    from src.eval import baselines, progress
    from src.eval.runner import EvalConfig
    from src.utils import paths, environment, preflight

    data = tmp_path / "table"
    data.mkdir()
    parquet = data / "data.parquet"
    parquet.write_bytes(b"first table")
    weights = tmp_path / "released.ckpt"
    weights.write_bytes(b"first weights")
    monkeypatch.setattr("src.data.pipeline.ensure_processed", lambda *a: None)
    monkeypatch.setattr(paths, "find_processed_dir", lambda *a: data)
    monkeypatch.setattr(baselines, "find_local_tabpfn_checkpoint", lambda *a: weights)
    monkeypatch.setattr(preflight, "source_fingerprint", lambda: "source")
    monkeypatch.setattr(environment, "environment_fingerprint", lambda: "packages")
    cfg = EvalConfig(task="pd", models=["tabpfn3"], datasets=["table"])
    before = progress.identity(cfg, "credit")
    assert progress.identity(cfg, "credit") == before
    if change == "data":
        parquet.write_bytes(b"changed table bytes")
    elif change == "weights":
        weights.write_bytes(b"changed checkpoint bytes")
    elif change == "code":
        monkeypatch.setattr(preflight, "source_fingerprint", lambda: "new source")
    elif change == "packages":
        monkeypatch.setattr(environment, "environment_fingerprint", lambda: "new packages")
    else:
        cfg.seeds = [1]
    assert progress.identity(cfg, "credit") != before


def test_ood_identity_binds_manifest_cache_and_explicit_checkpoint(tmp_path, monkeypatch):
    from src.eval import ood, progress
    from src.eval.ood_runner import OODEvalConfig

    manifest, cache, weight = [tmp_path / f for f in ("manifest.json", "cache.npz", "model.ckpt")]
    for path in (manifest, cache, weight):
        path.write_bytes(b"initial")
    entry = ood.OODDataset("table", 7, "classification", 57, 129)
    monkeypatch.setattr(ood, "list_ood_datasets", lambda: [entry])
    monkeypatch.setattr(ood, "manifest_path", lambda: manifest)
    monkeypatch.setattr(ood, "cache_path", lambda e: cache)
    cfg = OODEvalConfig(models=["crediticl"], kinds=["classification"],
                        model_kwargs={"crediticl": {"checkpoint": str(weight)}})
    before = progress.identity(cfg, "ood")
    for path in (manifest, cache, weight):
        path.write_bytes(b"changed bytes")
        after = progress.identity(cfg, "ood")
        assert after != before
        before = after


@pytest.mark.parametrize("domain", ["credit", "ood"])
def test_cli_publishes_all_folds_and_second_invocation_does_no_model_work(isolated_output, monkeypatch, domain):
    import sys
    from scripts import evaluate, evaluate_ood
    from src.eval import ood, ood_runner, runner
    from src.utils import paths
    from src.utils.logging_setup import close_logging

    X = np.random.default_rng(12).normal(size=(70, 8)).astype(np.float32)
    y = np.arange(70) % 2
    monkeypatch.setattr("src.eval.progress.identity", lambda *a: "frozen-cli-inputs")
    monkeypatch.setattr(runner, "ensure_processed", lambda *a: {"table": isolated_output})
    monkeypatch.setattr(runner, "load_arrays", lambda *a, **k: (X, y, [], {"n_features": 8}))
    monkeypatch.setattr(ood_runner, "_prepare", lambda *a, **k: (X, y, [], {}))
    monkeypatch.setattr(ood_runner, "list_ood_datasets", lambda *a: [ood.OODDataset("table", 7, "classification", 70, 8)])
    monkeypatch.setattr(ood_runner, "ood_status", lambda: dict(manifest="fixture", exists=True,
                                                             complete=True, n_datasets=1, by_kind={"classification": 1}))
    script = evaluate if domain == "credit" else evaluate_ood
    args = ["--task", "pd"] if domain == "credit" else ["--kinds", "classification"]
    monkeypatch.setattr(sys, "argv", ["evaluate", *args, "--models", "linear", "--datasets", "table", "--tag", "probe"])
    result = paths.benchmark_dir_for("probe", "pd" if domain == "credit" else "ood") / (
        "results_probe.csv" if domain == "credit" else "ood_results_probe.csv")
    try:
        assert script.main() == 0
        first = pd.read_csv(result)
        first_bytes = result.read_bytes()
        assert len(first) == 5 and first.status.eq("ok").all()
        monkeypatch.setattr(runner if domain == "credit" else ood_runner, "score_fold",
                            lambda *a, **k: pytest.fail("repeated a completed prediction"))
        assert script.main() == 0
        pd.testing.assert_frame_equal(pd.read_csv(result), first)
        assert result.read_bytes() == first_bytes
    finally:
        close_logging()
