"""Run cleanup and the training progress curve.

The cleanup tests are mostly about what must NOT be deleted. A helper that removes the
wrong thing is worse than no helper: `data/raw` is irreplaceable, and the repo's
`checkpoints/` holds downloaded TabPFN/TabICL weights that are not ours to regenerate.
Those are excluded by construction, so no combination of arguments can reach them.
"""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import numpy as np

from src.utils import run_artifacts as ra


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake output tree, with BOTH storage tiers redirected into `tmp_path`.

    `VSC_DATA` has to be set, not deleted. Without it `paths.on_vsc()` is false, so
    `outputs_dir()` — and therefore `logs_dir()` and `manifests_dir()` — resolves to the REAL
    repository `output/`. These tests then write files there and the cleanup tests below
    delete from it: the earlier version of this fixture destroyed a notebook's committed
    figures and `CAPTIONS.md` on every run, silently, three test files away from the cause.
    """
    import importlib

    monkeypatch.setenv("VSC_DATA", str(tmp_path / "vsc_data"))
    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path / "staging"))
    import src.utils.paths as paths

    importlib.reload(paths)
    importlib.reload(ra)

    # Guard rather than trust: if a future edit re-breaks the redirect, fail here instead of
    # deleting real output.
    assert tmp_path in paths.outputs_dir().parents, (
        f"output tree not isolated: {paths.outputs_dir()} is outside {tmp_path}"
    )

    (paths.logs_dir()).mkdir(parents=True, exist_ok=True)
    (paths.logs_dir() / "run.log").write_text("x" * 500, encoding="utf-8")
    (paths.manifests_dir()).mkdir(parents=True, exist_ok=True)
    (paths.manifests_dir() / "a__progress.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    ck = paths.checkpoints_dir() / "some_run"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "step-100.ckpt").write_bytes(b"0" * 4096)
    pool = paths.prior_cache_dir("lgd__original")
    pool.mkdir(parents=True, exist_ok=True)
    (pool / "shard_00000.pt").write_bytes(b"0" * 8192)

    yield ra, paths
    importlib.reload(paths)
    importlib.reload(ra)


def test_finds_every_category(tree):
    mod, _ = tree
    cats = {a.category for a in mod.find_artifacts()}
    assert {"logs", "manifests", "checkpoints", "prior_pools"} <= cats


def test_raw_data_is_never_removable(tree):
    """The one thing that cannot be regenerated."""
    mod, paths = tree
    raw = paths.REPO_ROOT / "data" / "raw"
    assert mod._is_protected(raw)
    assert mod._is_protected(raw / "lgd")
    assert not any(str(raw) in a.path.parts for a in mod.find_artifacts())


def test_downloaded_model_weights_are_never_removable(tree):
    """`checkpoints/` in the repo holds TabPFN/TabICL weights we downloaded, not ours."""
    mod, paths = tree
    assert mod._is_protected(paths.REPO_ROOT / "checkpoints")


def test_source_and_config_are_protected(tree):
    mod, paths = tree
    for name in ("src", "config", "tests", "tfm-library"):
        assert mod._is_protected(paths.REPO_ROOT / name), name


def test_dry_run_is_the_default_and_removes_nothing(tree):
    mod, paths = tree
    result = mod.clean(("logs",))
    assert result["dry_run"] is True
    assert result["removed"], "it should still SAY what it would remove"
    assert (paths.logs_dir() / "run.log").exists(), "dry run must not delete"


def test_expensive_categories_need_an_explicit_opt_in(tree):
    """'clean up the logs' must never take out 39 GB of generated priors."""
    mod, paths = tree
    mod.clean(dry_run=False)  # the defaults
    assert not (paths.logs_dir() / "run.log").exists(), "logs should be gone"
    assert (paths.checkpoints_dir() / "some_run" / "step-100.ckpt").exists()
    assert (paths.prior_cache_dir("lgd__original") / "shard_00000.pt").exists()


def test_expensive_categories_are_removed_when_named(tree):
    mod, paths = tree
    mod.clean(("checkpoints", "prior_pools"), dry_run=False)
    assert not (paths.checkpoints_dir() / "some_run" / "step-100.ckpt").exists()
    assert not (paths.prior_cache_dir("lgd__original") / "shard_00000.pt").exists()


def test_gitkeep_markers_survive(tree):
    """They are tracked in git so empty result directories exist after a clone.
    Deleting them would quietly change what git tracks."""
    mod, paths = tree
    d = paths.logs_dir()
    (d / ".gitkeep").write_text("", encoding="utf-8")
    (d / "junk.log").write_text("x", encoding="utf-8")
    mod.clean(("logs",), dry_run=False)
    assert (d / ".gitkeep").exists()
    assert not (d / "junk.log").exists()


def test_a_directory_holding_only_gitkeep_is_not_listed(tree):
    mod, paths = tree
    empty = paths.logs_dir()
    for f in empty.iterdir():
        f.unlink()
    (empty / ".gitkeep").write_text("", encoding="utf-8")
    assert not [a for a in mod.find_artifacts() if a.category == "logs"]


def test_summary_flags_the_expensive_ones(tree):
    mod, _ = tree
    text = mod.summarise(mod.find_artifacts())
    assert "EXPENSIVE" in text
    assert "NEVER removed" in text


def test_summary_of_nothing_is_readable(tree):
    mod, _ = tree
    assert "already clean" in mod.summarise([])


# -- the progress curve ------------------------------------------------------


def test_progress_disabled_by_default():
    from src.train.progress import ProgressConfig, ProgressTracker

    t = ProgressTracker(ProgressConfig(every_datasets=0), "lgd", "r", ".")
    assert not t.enabled
    assert not t.due(1_000_000)


def test_progress_fires_on_dataset_count_not_steps(tmp_path):
    """It measures per synthetic dataset, so it stays comparable between runs with
    different batch sizes."""
    from src.train.progress import ProgressConfig, ProgressTracker

    t = ProgressTracker(ProgressConfig(every_datasets=100), "lgd", "r", tmp_path)
    assert not t.due(99)
    assert t.due(100)


def test_progress_writes_a_widening_csv(tmp_path):
    """Columns can appear late — the OOD cache may be populated mid-run. Ragged rows
    would silently misalign in pandas, so the file is rewritten with the union."""
    from src.train.progress import ProgressConfig, ProgressTracker

    t = ProgressTracker(ProgressConfig(every_datasets=10), "lgd", "r", tmp_path)
    t._append({"step": 1, "a": 1.0})
    t._append({"step": 2, "a": 2.0, "b": 9.0})
    rows = list(csv.DictReader(t.path.open()))
    assert len(rows) == 2
    assert set(rows[0]) == {"step", "a", "b"}
    assert rows[0]["b"] == "" and rows[1]["b"] == "9.0"


def test_progress_never_kills_training(tmp_path):
    """A diagnostic that can end a 3-day run is worse than no diagnostic."""
    from src.train.progress import ProgressConfig, ProgressTracker

    t = ProgressTracker(ProgressConfig(every_datasets=10, n_ood=0), "lgd", "r", tmp_path)

    class Exploding:
        training = True

        def eval(self):
            raise RuntimeError("boom")

        def train(self):
            pass

    with pytest.raises(RuntimeError):
        Exploding().eval()  # the failure is real...

    t._cached_real = []
    t._cached_ood = []
    row = t.record(_DummyModel(), step=1, datasets_seen=10, train_loss=0.5, elapsed_s=1.0)
    assert "progress_eval_seconds" in row


class _DummyModel:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def parameters(self):
        import torch

        return iter([torch.zeros(1)])


def test_progress_restores_training_mode(tmp_path):
    """Leaving the model in eval() would silently disable dropout for the rest of the
    run — a corruption that produces plausible numbers."""
    from src.train.progress import ProgressConfig, ProgressTracker

    t = ProgressTracker(ProgressConfig(every_datasets=10, n_ood=0), "lgd", "r", tmp_path)
    t._cached_real = []
    t._cached_ood = []
    m = _DummyModel()
    t.record(m, step=1, datasets_seen=10, train_loss=0.5, elapsed_s=1.0)
    assert m.training is True


def test_progress_schedules_the_next_measurement(tmp_path):
    from src.train.progress import ProgressConfig, ProgressTracker

    t = ProgressTracker(ProgressConfig(every_datasets=100, n_ood=0), "lgd", "r", tmp_path)
    t._cached_real = []
    t._cached_ood = []
    t.record(_DummyModel(), step=1, datasets_seen=100, train_loss=0.5, elapsed_s=1.0)
    assert not t.due(150)
    assert t.due(200)


def test_progress_config_is_read_from_the_yaml():
    """The lever has to be reachable from config, or it is dead code."""
    from src.utils.config import expand_with_seeds, load_yaml

    for path in ("config/LGD.yaml", "config/PD.yaml"):
        cfg = expand_with_seeds(load_yaml(path))[0]
        assert cfg["progress"]["every_datasets"] > 0, f"{path}: progress curve is off"


@pytest.mark.slow
def test_progress_scores_a_real_dataset(tmp_path):
    """End to end on real data, if it is present."""
    from src.train.progress import ProgressConfig, ProgressTracker
    from src.utils.paths import find_raw_path

    if find_raw_path("lgd", "0003.axa") is None:
        pytest.skip("raw data not present")

    from src.models.nanotabiclv2 import NanoTabICLv2

    model = NanoTabICLv2(max_classes=0, out_dim=16, embed_dim=32, col_num_blocks=1,
                         row_num_blocks=1, icl_num_blocks=1, col_nhead=2, row_nhead=2,
                         icl_nhead=2, n_cls_rows=8)
    t = ProgressTracker(
        ProgressConfig(every_datasets=10, n_datasets=1, n_ood=0, context_rows=64,
                       max_test_rows=100),
        "lgd", "r", tmp_path,
    )
    row = t.record(model, step=1, datasets_seen=10, train_loss=0.5, elapsed_s=1.0)
    assert any(k.startswith("real__") for k in row), f"no real metrics in {list(row)}"
    assert np.isfinite([v for k, v in row.items() if k.startswith("real__")]).any()
