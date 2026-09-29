"""Experiment 2 (continued pretraining of TabICLv2 and TabPFN-3) and Experiment 0 (the cluster
debug suite): the grid, the benchmark bookkeeping, the stage-A choice, the wide-table rule, the
TabPFN-3 trainer, and the tools that read Experiment 0 back."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# -- arm presets --------------------------------------------------------------------------------


def _cfg(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


BASE = """experiment: exp2_pd
task: pd
architecture: tabicl
sweep:
  arm: [a, b]
  seeds: [0]
arms:
  a: {architecture: tabicl, train.lr: 1.0e-5}
  b: {architecture: tabpfn3, train.lr: 3.0e-5, prior.encoding: raw}
prior: {encoding: tabicl, credit_fraction: 0.0}
train: {lr: 1.0, optimizer: adamw}
init: {strategy: full}
"""


def test_an_arm_preset_writes_its_overrides_into_the_run(tmp_path):
    from src.utils.config import expand_with_seeds, load

    runs = expand_with_seeds(load(_cfg(tmp_path, BASE)))
    by_arm = {r["arm"]: r for r in runs}
    assert by_arm["a"]["architecture"] == "tabicl" and by_arm["a"]["train"]["lr"] == 1e-5
    assert by_arm["b"]["architecture"] == "tabpfn3" and by_arm["b"]["prior"]["encoding"] == "raw"
    assert by_arm["b"]["_grid"]["arm_overrides"]["train.lr"] == 3e-5
    assert "arms" not in by_arm["a"], "the preset table is not part of a run"
    assert by_arm["a"]["_run_name"] != by_arm["b"]["_run_name"]


def test_expanding_twice_gives_the_same_runs(tmp_path):
    """`expand_grid` pops `arms`; it must not pop it from the caller's config."""
    from src.utils.config import expand_with_seeds, load

    cfg = load(_cfg(tmp_path, BASE))
    first = [r["_run_name"] for r in expand_with_seeds(cfg)]
    assert [r["_run_name"] for r in expand_with_seeds(cfg)] == first


@pytest.mark.parametrize("bad, match", [
    (BASE.replace("  b: {architecture: tabpfn3", "  c: {architecture: tabpfn3"), "not defined"),
    (BASE.replace("prior.encoding: raw", "prior.nonexistent: raw"), "no such key"),
])
def test_a_broken_preset_is_refused(tmp_path, bad, match):
    from src.utils.config import expand_with_seeds, load

    with pytest.raises(ValueError, match=match):
        expand_with_seeds(load(_cfg(tmp_path, bad)))


def test_a_search_placeholder_keeps_a_config_a_template(tmp_path):
    from src.utils.config import find_placeholders, load

    text = BASE.replace("train.lr: 1.0e-5", "train.lr: FILL_FROM_SEARCH")
    with pytest.raises(ValueError, match="still a template"):
        load(_cfg(tmp_path, text))
    assert find_placeholders(load(_cfg(tmp_path, text), allow_placeholders=True)) == ["arms.a.train.lr"]


# -- the benchmark for Experiment 2 ---------------------------------------------------------------


def test_each_arm_is_scored_by_the_wrapper_of_its_architecture():
    from src.eval.benchmark_status import slot_for

    models = {slot_for(2, "pd", i, variant="search").model for i in range(8)}
    assert models == {"crediticl", "tabpfn3"}


def test_the_search_benchmark_has_no_reference_slots_and_its_own_tags():
    """Development-only reference results would overwrite the shared reference column."""
    from src.eval.benchmark_status import n_slots, slot_for
    from src.utils.paths import owner_of

    total = n_slots(2, "pd", variant="search")
    assert total == 8 * 4, "8 arms x 4 saved checkpoints (2,500 ... 10,000), nothing else"
    kinds = {slot_for(2, "pd", i, variant="search").kind for i in range(total)}
    assert kinds == {"arm"}
    tag = slot_for(2, "pd", 0, variant="search").tag
    assert tag == "exp2searchbench_pd_a0" and owner_of(tag) == 2
    assert slot_for(2, "pd", total, variant="search").kind == "none"


def test_the_main_benchmark_scores_both_models_and_the_reference_column():
    from src.eval.benchmark_status import n_slots, slot_for

    total = n_slots(2, "lgd")
    assert total == 18 * 4 + 4
    assert slot_for(2, "lgd", total - 1).tag == "reference_lgd_linear"


# -- the stage-A choice -----------------------------------------------------------------------------


def _table(rows):
    base = {"complete": True, "released_dev": 0.70, "released_ood": 0.80, "warmup_proportion": 0.1,
            "optimizer": "adamw"}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["dev_gain"] = df["dev"] - df["released_dev"]
    df["ood_change"] = df["ood"] - df["released_ood"]
    return df


def test_the_best_development_score_wins_among_those_that_did_not_forget():
    from src.eval.exp2_search import choose

    t = _table([
        {"model": "tabicl", "arm": "x", "lr": 3e-5, "step": 10000, "dev": 0.75, "ood": 0.70},  # forgot
        {"model": "tabicl", "arm": "y", "lr": 1e-5, "step": 7500, "dev": 0.74, "ood": 0.795},
        {"model": "tabicl", "arm": "z", "lr": 3e-6, "step": 10000, "dev": 0.72, "ood": 0.80},
    ])
    pick = choose(t, "pd")["tabicl"]
    assert pick["arm"] == "y" and pick["step"] == 7500 and not pick["flag"]


def test_when_every_candidate_forgot_the_least_forgetting_is_flagged():
    from src.eval.exp2_search import choose

    t = _table([
        {"model": "tabpfn3", "arm": "x", "lr": 3e-5, "step": 10000, "dev": 0.75, "ood": 0.60},
        {"model": "tabpfn3", "arm": "y", "lr": 1e-5, "step": 10000, "dev": 0.74, "ood": 0.70},
    ])
    pick = choose(t, "pd")["tabpfn3"]
    assert pick["arm"] == "y" and "Decide by hand" in pick["flag"]


def test_an_incomplete_candidate_cannot_win():
    from src.eval.exp2_search import choose

    t = _table([
        {"model": "tabicl", "arm": "x", "lr": 3e-5, "step": 10000, "dev": 0.99, "ood": 0.80, "complete": False},
        {"model": "tabicl", "arm": "y", "lr": 1e-5, "step": 10000, "dev": 0.71, "ood": 0.80},
    ])
    assert choose(t, "pd")["tabicl"]["arm"] == "y"


# -- the same columns for every model ----------------------------------------------------------------


def test_wide_tables_keep_their_highest_variance_training_columns():
    from src.eval.protocol import MAX_FEATURES, apply_columns, select_columns

    rng = np.random.default_rng(0)
    base = rng.normal(size=40)
    X = np.outer((base - base.mean()) / base.std(), np.linspace(0.1, 6.0, 600))  # spread rises with the index
    keep = select_columns(X)
    assert len(keep) == MAX_FEATURES and list(keep) == sorted(keep)
    assert set(keep) == set(range(100, 600)), "the 500 widest-spread columns"
    Xa, Xb, cats = apply_columns(keep, [0, 150, 599], X, X[:5])
    assert Xa.shape == (40, 500) and Xb.shape == (5, 500)
    assert cats == [50, 499], "categorical indices follow their columns; dropped ones go"
    assert select_columns(X[:, :500]) is None


def test_the_choice_never_looks_at_the_target():
    from inspect import signature

    from src.eval.protocol import select_columns

    assert list(signature(select_columns).parameters) == ["X_train", "max_features"]


def test_every_model_gets_the_same_columns_in_a_fold(monkeypatch):
    from src.eval import runner

    seen: dict[str, tuple[int, ...]] = {}

    def fake_score(task, model_name, X_tr, y_tr, X_te, y_te, cats, **kw):
        seen[model_name] = (X_tr.shape[1], X_te.shape[1])
        return {"roc_auc": 0.5, "_headline": "x"}

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 700))
    y = (rng.random(60) > 0.5).astype(float)
    monkeypatch.setattr(runner, "load_arrays", lambda *a, **k: (X, y, [], {"n_rows": 60, "n_features": 700}))
    monkeypatch.setattr(runner, "score_fold", fake_score)
    rows = runner.evaluate_dataset("pd", "fake", ["catboost", "tabpfn3", "linear"], seed=0, n_folds=2)
    assert set(seen.values()) == {(500, 500)}, seen
    assert all(r["features_selected_from"] == 700 and r["features_used"] == 500 for r in rows)


# -- the released weights, wherever they live ---------------------------------------------------------


def test_released_weights_are_found_in_the_repo_or_on_project_storage(tmp_path, monkeypatch):
    from src.utils import paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(paths, "pretrained_dir", lambda *a: tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    (tmp_path / "staging" / "w.ckpt").write_bytes(b"x")
    assert paths.find_pretrained("checkpoints/w.ckpt") == tmp_path / "staging" / "w.ckpt"
    with pytest.raises(FileNotFoundError, match="not found"):
        paths.find_pretrained("checkpoints/missing.ckpt")


# -- the TabPFN-3 trainer ----------------------------------------------------------------------------


def _tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401

        from src.eval.baselines import find_local_tabpfn_checkpoint

        return find_local_tabpfn_checkpoint("classifier") is not None
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _tabpfn_available(), reason="tabpfn or its released weights not here")
def test_tabpfn_continued_pretraining_trains_saves_resumes_and_is_scored(tmp_path):
    """Two steps of TabPFN-3 on the CPU, a resume to three, and the checkpoint scored by the
    benchmark's own TabPFN wrapper, which names the arm and step it came from."""
    import torch

    from src.eval.baselines import build, tabpfn_checkpoint_identity
    from src.train.tabpfn_trainer import TabPFNTrainer
    from src.utils.config import expand_with_seeds, load

    run = copy.deepcopy(next(r for r in expand_with_seeds(load(ROOT / "config" / "Exp2_PD_search.yaml"))
                             if r["architecture"] == "tabpfn3"))
    run["prior"].update(n_rows_range=[96, 96], n_features_range=[3, 5], max_features=8,
                        n_nodes_range=[2, 4], max_filter_attempts=3, credit_fraction=0.5)
    run["train"].update(max_steps=2, batch_size=2, num_workers=0, log_every=1, save_temp_every=0,
                        save_perm_every=2, amp=False)
    run["logging"] = {"level": "WARNING", "console": False, "to_file": False, "log_prior_every": 0,
                      "log_hardware_every": 1, "log_grad_every": 1, "log_weights_every": 1}
    run["progress"]["every_datasets"] = 0
    trainer = TabPFNTrainer(run, tmp_path / "run", device="cpu", ckpt_dir=tmp_path / "ckpt")
    before = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}
    trainer.maybe_resume()
    summary = trainer.train()
    assert summary["completed"] and summary["steps"] == 2
    moved = sum(not torch.equal(v.detach(), before[k]) for k, v in trainer.model.named_parameters())
    assert moved > 0, "the weights must move"

    run["train"]["max_steps"] = 3
    again = TabPFNTrainer(run, tmp_path / "run", device="cpu", ckpt_dir=tmp_path / "ckpt")
    again.maybe_resume()
    assert again.step == 2
    assert again.train()["resumed_at"] == 2

    weights = pd.read_csv(tmp_path / "run" / "weights.csv")
    assert weights["step"].tolist() == [0, 1, 2, 3]
    ck = tmp_path / "ckpt" / "step-2.ckpt"
    assert tabpfn_checkpoint_identity(ck) == {"run_name": run["_run_name"], "checkpoint_step": 2}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(float)
    m = build("tabpfn3", "pd", seed=0, model_path=str(ck), n_estimators=1)
    m.fit(X[:90], y[:90], [])
    p = m.predict(X[90:])
    assert p.shape == (30,) and np.isfinite(p).all()
    assert m.report.extra["run_name"] == run["_run_name"] and m.report.extra["checkpoint_step"] == 2


def test_a_tabpfn_arm_refuses_tabicls_encoding():
    from src.train.tabpfn_trainer import TabPFNTrainer
    from src.utils.config import expand_with_seeds, load

    run = copy.deepcopy(next(r for r in expand_with_seeds(load(ROOT / "config" / "Exp2_PD_search.yaml"))
                             if r["architecture"] == "tabpfn3"))
    run["prior"]["encoding"] = "tabicl"
    run["logging"] = {"console": False, "to_file": False}
    with pytest.raises(ValueError, match="encoding: raw"):
        TabPFNTrainer(run, ROOT / "output_CreditICL" / "_never_written", device="cpu")


def test_the_preparer_skips_a_task_whose_query_holds_an_unseen_class():
    from src.train.tabpfn_trainer import TabPFNTaskPreparer

    prep = TabPFNTaskPreparer("pd", "unused.ckpt", 2, 0)
    X = np.zeros((10, 2))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    assert prep.prepare(X, y, train_size=5, seed=0) is None, "class 1 only in the query"


def test_pretrain_dispatches_on_the_architecture():
    text = (ROOT / "scripts" / "pretrain.py").read_text(encoding="utf-8")
    assert 'str(run.get("architecture", "tabicl")) == "tabpfn3"' in text
    assert "TabPFNTrainer as Trainer" in text


# -- Experiment 0's tools ----------------------------------------------------------------------------


def test_the_checks_script_writes_its_report(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "report"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "exp0_checks.py"), "--only", "configs",
                        "--out", str(out)], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads((out / "checks.json").read_text(encoding="utf-8"))
    names = {row["name"] for row in report["rows"]}
    assert {"Exp0_PD", "Exp1_LGD", "Exp2_PD_search", "Exp2_LGD"} <= names
    assert (out / "checks.md").read_text(encoding="utf-8").startswith("# Experiment 0")


def _fake_exp0_tree(root: Path, *, resumed: bool = True) -> None:
    from src.eval.benchmark_status import n_slots, slot_for
    from src.utils.config import expand_with_seeds, load

    out = root / "output_CreditICL"
    exp0 = out / "experiment_0"
    (exp0 / "report").mkdir(parents=True)
    (exp0 / "report" / "checks.json").write_text(json.dumps({"rows": [
        {"check": "env", "name": "gpu", "status": "PASS", "detail": {}}]}), encoding="utf-8")
    (exp0 / "logs").mkdir()
    (exp0 / "logs" / "pretrain_pd_1.log").write_text(
        "...\nEND status=INCOMPLETE-RESUBMITTED - now\n" if resumed else "END status=OK\n", encoding="utf-8")
    (exp0 / "logs" / "pretrain_pd_2.log").write_text("END status=OK\n", encoding="utf-8")
    for track in ("pd", "lgd"):
        for run in expand_with_seeds(load(ROOT / "config" / f"Exp0_{track.upper()}.yaml")):
            d = exp0 / "runs" / run["_run_name"]
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({
                "completed": True, "steps": run["train"]["max_steps"], "train_seconds_total": 600,
                "resumed_at": 240 if run["arm"] == "scratch_tabicl" and resumed else None}), encoding="utf-8")
            pd.DataFrame({"step": [20, 40, 50], "kind": ["hardware", "hardware", "grad"],
                          "gpu0_utilization_gpu": [90, 92, None], "steps_per_s": [0.8, 0.9, None]}
                         ).to_csv(d / "telemetry.csv", index=False)
            pd.DataFrame({"step": [0, 50], "drift_all": [0.0, 0.1], "drift_rel_all": [0.0, 0.001]}
                         ).to_csv(d / "weights.csv", index=False)
            pd.DataFrame({"step": [0, 200], "real__german__roc_auc": [0.6, 0.7],
                          "ood__x__roc_auc": [0.8, 0.8]}).to_csv(d / "progress.csv", index=False)
        for index in range(n_slots(0, track)):
            slot = slot_for(0, track, index)
            owner = out / ("reference" if slot.kind == "reference" else "experiment_0") / "benchmark"
            (owner / "receipts").mkdir(parents=True, exist_ok=True)
            (owner / track).mkdir(parents=True, exist_ok=True)
            (owner / "receipts" / f"{slot.tag}.json").write_text("{}", encoding="utf-8")
            pd.DataFrame({"total_seconds": [3600.0, 1800.0]}).to_csv(owner / track / f"results_{slot.tag}.csv",
                                                                     index=False)


def test_the_verifier_passes_a_complete_experiment_0(tmp_path, capsys):
    from src.utils.exp0_verify import FAIL, verify

    _fake_exp0_tree(tmp_path)
    v = verify(tmp_path / "output_CreditICL")
    fails = [item for item in v.items if item[0] == FAIL]
    assert not fails, fails
    assert "h per arm" in capsys.readouterr().out, "the cost extrapolation is printed"


def test_the_verifier_fails_when_the_forced_stop_never_happened(tmp_path):
    from src.utils.exp0_verify import FAIL, verify

    _fake_exp0_tree(tmp_path, resumed=False)
    v = verify(tmp_path / "output_CreditICL")
    failed = {what for status, what, _ in v.items if status == FAIL}
    assert any("forced stop" in w for w in failed)
    assert any("resumed from a checkpoint" in w for w in failed)
