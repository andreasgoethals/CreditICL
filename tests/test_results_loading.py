"""Which checkpoint a results notebook reports (30-09-2026).

Every saved checkpoint is benchmarked; the notebooks report the FINAL one unless a config names
another in `eval.headline_step`. Until 30-09-2026 that key was written and never read, so a chosen
length would silently not reach the tables.
"""

from __future__ import annotations

import pandas as pd


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_the_final_checkpoint_is_reported_by_default_and_a_headline_step_when_named(tmp_path, monkeypatch):
    from src.eval.protocol import PROTOCOL
    from src.utils import paths
    from src.visualize import results_plots as rp

    monkeypatch.setattr(paths, "benchmark_dir", lambda owner, part=None: tmp_path / str(owner) / str(part))
    base = {"dataset": "d", "seed": 0, "fold": 0, "status": "ok", "protocol": PROTOCOL}
    out = tmp_path / "2" / "pd"
    _write(out / "results_exp2bench_pd_a0.csv", [{**base, "model": "arm-0", "roc_auc": 0.70}])
    _write(out / "results_exp2bench_pd_a0_s5000.csv", [{**base, "model": "arm-0", "roc_auc": 0.60,
                                                         "info_checkpoint_step": 5000}])
    cfg = tmp_path / "Exp2_PD.yaml"
    cfg.write_text("train: {max_steps: 10000}\neval: {}\n", encoding="utf-8")
    monkeypatch.setattr(paths, "config_path", lambda name: cfg)
    assert rp.headline_step("pd", "exp2") is None
    assert rp.load_results("pd", "exp2")["roc_auc"].tolist() == [0.70]

    cfg.write_text("train: {max_steps: 10000}\neval: {headline_step: 5000}\n", encoding="utf-8")
    assert rp.headline_step("pd", "exp2") == 5000
    assert rp.load_results("pd", "exp2")["roc_auc"].tolist() == [0.60]

    cfg.write_text("train: {max_steps: 10000}\neval: {headline_step: FILL_FROM_EXP1}\n", encoding="utf-8")
    assert rp.headline_step("pd", "exp2") is None, "an unfilled placeholder means the final checkpoint"


def test_no_config_names_a_headline_step_today():
    """Both Exp2 recipes fix 10,000 steps; nothing selects a checkpoint."""
    from src.visualize import results_plots as rp

    for exp in ("exp1", "exp2", "exp3"):
        for track in ("pd", "lgd"):
            assert rp.headline_step(track, exp) is None


def test_the_capped_context_results_are_read_apart_from_the_full_context_ones(tmp_path, monkeypatch):
    from src.eval.protocol import PROTOCOL
    from src.utils import paths
    from src.visualize import results_plots as rp

    monkeypatch.setattr(paths, "benchmark_dir", lambda owner, part=None: tmp_path / str(owner) / str(part))
    base = {"dataset": "d", "seed": 0, "fold": 0, "status": "ok", "protocol": PROTOCOL}
    _write(tmp_path / "1" / "pd" / "results_exp1bench_pd_a0.csv", [{**base, "model": "arm-0", "roc_auc": 0.7}])
    _write(tmp_path / "1" / "pd" / "results_exp1bench_pd_a0_ctx1024.csv",
           [{**base, "model": "arm-0", "roc_auc": 0.5, "context_cap": 1024, "info_checkpoint_step": 12500}])
    ref = tmp_path / str(paths.REFERENCE) / "pd"
    _write(ref / "results_reference_pd_tabiclv2.csv", [{**base, "model": "tabiclv2", "roc_auc": 0.8}])
    _write(ref / "results_reference_pd_tabiclv2_ctx1024.csv",
           [{**base, "model": "tabiclv2", "roc_auc": 0.6, "context_cap": 1024}])
    assert sorted(rp.load_results("pd", "exp1")["roc_auc"].tolist()) == [0.7, 0.8]
    assert sorted(rp.load_results("pd", "exp1", context=1024)["roc_auc"].tolist()) == [0.5, 0.6]
    assert rp.load_results("pd", "exp1", context=2048) is None
    assert rp.load_checkpoint_results("pd", "exp1") is None, "the capped final checkpoint is not a curve point"
