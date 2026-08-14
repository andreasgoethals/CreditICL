

def test_one_failing_dataset_does_not_discard_the_others(monkeypatch, tmp_path):
    """A single `try` around both loops meant a NaN target in the SECOND real dataset threw
    away the remaining real datasets AND all eight out-of-domain suites — the row held one
    dataset and an error string, and it looked like the progress curve barely worked."""
    import numpy as np

    from src.train import progress as P

    calls = []

    class FakeDS:
        def __init__(self, n):
            self.X = np.zeros((n, 3), np.float32)
            self.y = np.zeros(n, np.float32)

    tracker = P.ProgressTracker.__new__(P.ProgressTracker)

    def fake_score(model, X, y, rng):
        calls.append(len(X))
        if len(calls) == 2:
            raise ValueError("Input contains NaN.")
        return {"roc_auc": 0.7}

    monkeypatch.setattr(tracker, "_score", fake_score, raising=False)
    monkeypatch.setattr(
        tracker, "_real_datasets",
        lambda: [(f"000{i}.d{i}", FakeDS(50)) for i in range(4)], raising=False,
    )
    monkeypatch.setattr(tracker, "_ood_datasets", lambda: [], raising=False)
    rows = []
    monkeypatch.setattr(tracker, "_append", rows.append, raising=False)
    fake_cfg = type("C", (), {"seed": 0, "every_datasets": 5000,
                              "context_rows": 32, "max_test_rows": 64})()
    monkeypatch.setattr(tracker, "cfg", fake_cfg, raising=False)
    monkeypatch.setattr(tracker, "task", "pd", raising=False)
    monkeypatch.setattr(tracker, "run_name", "test_run", raising=False)

    class M:
        def parameters(self):
            return iter(())
        def train(self, *a): pass
        def eval(self): pass
        @property
        def training(self): return False

    tracker.record(M(), step=10, datasets_seen=40, train_loss=0.5, elapsed_s=1.0)

    assert len(rows) == 1
    row = rows[0]
    # THE POINT: the 2nd dataset failed, and 1, 3 and 4 still produced numbers
    assert len(calls) == 4, "every dataset must be attempted"
    scored = [k for k in row if k.startswith("real__") and k.endswith("roc_auc")]
    assert len(scored) == 3, f"expected 3 surviving datasets, got {scored}"
    assert row["n_errors"] == 1
    assert "Input contains NaN" in row["error"]
    assert "d1" in row["error"], "the error must name WHICH dataset failed"
