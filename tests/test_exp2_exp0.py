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


def test_a_placeholder_keeps_a_config_a_template(tmp_path):
    from src.utils.config import find_placeholders, load

    text = BASE.replace("train.lr: 1.0e-5", "train.lr: FILL_FROM_EXP1")
    with pytest.raises(ValueError, match="still a template"):
        load(_cfg(tmp_path, text))
    assert find_placeholders(load(_cfg(tmp_path, text), allow_placeholders=True)) == ["arms.a.train.lr"]


# -- the benchmark for Experiment 2 ---------------------------------------------------------------


def test_each_arm_is_scored_by_the_wrapper_of_its_architecture():
    from src.eval.benchmark_status import slot_for

    models = {slot_for(2, "pd", i).model for i in range(18)}
    assert models == {"crediticl", "tabpfn3"}


def test_experiment_1_uses_full_tables_only():
    from src.eval.benchmark_status import n_slots, slot_for
    total = n_slots(1, "pd")
    assert total == 9 * 5 + 4
    assert all(slot_for(1, "pd", i).context is None for i in range(total))
    assert slot_for(1, "pd", total).kind == "none"


def test_the_main_benchmark_scores_both_models_and_the_reference_column():
    from src.eval.benchmark_status import n_slots, slot_for

    total = n_slots(2, "lgd")
    assert total == 18 * 4 + 4
    assert slot_for(2, "lgd", total - 1).tag == "reference_lgd_linear"


# -- the same columns for every model ----------------------------------------------------------------


def test_wide_tables_keep_the_columns_most_associated_with_the_target():
    """The cap keeps the columns with the strongest rank association with the target — whatever
    their unit: raw variance would keep the huge-unit noise first."""
    from src.eval.protocol import rank_columns
    MAX_FEATURES = 100  # explicit legacy feature-budget ablation, never the benchmark default

    rng = np.random.default_rng(0)
    n, d = 2000, 300
    y = (rng.random(n) < 0.3).astype(float)
    signal = np.linspace(0.0, 1.0, d)  # association rises with the index ...
    scale = np.logspace(6, -3, d)  # ... while the unit shrinks
    X = (y[:, None] * signal + rng.normal(size=(n, d))) * scale
    keep = np.sort(rank_columns(X, y, [])[:MAX_FEATURES])
    assert (keep >= d - MAX_FEATURES - 20).mean() > 0.9, "mostly the most associated columns"
    by_variance = np.sort(rank_columns(X, y, [], method="variance")[:MAX_FEATURES])
    assert (by_variance < MAX_FEATURES).all(), "variance keeps the biggest units instead"


def test_the_column_score_ignores_units_and_monotone_transforms():
    from src.eval.protocol import rank_columns

    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5))
    y = (X[:, 2] + rng.normal(size=500) > 0).astype(float)
    order = rank_columns(X, y, [])
    Z = X.copy()
    Z[:, 2] = np.exp(Z[:, 2]) * 1e6  # monotone, and a huge unit
    assert list(rank_columns(Z, y, [])) == list(order)
    assert order[0] == 2


def test_the_fold_preprocessing_learns_from_the_fitted_rows_only():
    """Changing the rows being predicted changes nothing that was learned: the columns kept, the
    rows removed and the clipping bounds all come from the fitted rows."""
    from src.eval.protocol import fit_fold_preprocessing

    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 150))
    X[:, 7] = 1.0  # the full-feature protocol also retains near-constant columns
    y = (X[:, 0] + rng.normal(size=400) > 0).astype(float)
    prep, Xf, yf = fit_fold_preprocessing(X, y, [])
    again, _, _ = fit_fold_preprocessing(X, y, [])
    assert 7 in prep.keep and len(prep.keep) == 150 and Xf.shape == (len(yf), 150)
    assert np.array_equal(prep.keep, again.keep)
    test = rng.normal(size=(50, 150)) * 100  # wild prediction rows
    out = prep.transform(test)
    assert out.shape == (50, 150)
    kept = np.isfinite(prep.lower)
    assert (out[:, kept] <= prep.upper[kept] + 1e-6).all() and (out[:, kept] >= prep.lower[kept] - 1e-6).all()


def test_every_model_gets_the_same_data_in_a_fold(monkeypatch):
    """One preprocessing per fold, shared by every model: the same columns, the same rows."""
    from src.eval import runner

    seen: dict[str, tuple] = {}

    def fake_score(task, model_name, X_tr, y_tr, X_te, y_te, cats, **kw):
        final = kw["stages"]["final"]
        seen[model_name] = (final.X_fit.shape, final.X_pred.shape, id(final))
        return {"roc_auc": 0.5, "_headline": "x", **{f"prep_{k}": v for k, v in final.report.items()}}

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 700))
    y = (rng.random(200) > 0.5).astype(float)
    monkeypatch.setattr(runner, "load_arrays", lambda *a, **k: (X, y, [], {"n_rows": 200, "n_features": 700}))
    monkeypatch.setattr(runner, "score_fold", fake_score)
    rows = runner.evaluate_dataset("pd", "fake", ["catboost", "tabpfn3", "linear"], seed=0, n_folds=2)
    assert len({v for v in seen.values()}) == 1, seen
    assert all(r["prep_features_in"] == 700 and r["prep_features_used"] == 700 for r in rows)


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
@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_tabpfn_continued_pretraining_trains_saves_resumes_and_is_scored(tmp_path, track):
    """Two steps of TabPFN-3 on the CPU, a resume to three, and the checkpoint scored by the
    benchmark's own TabPFN wrapper, which names the arm and step it came from."""
    import torch

    from src.eval.baselines import build, tabpfn_checkpoint_identity
    from src.train.tabpfn_trainer import TabPFNTrainer
    from src.utils.config import expand_with_seeds, load

    run = copy.deepcopy(next(r for r in expand_with_seeds(load(ROOT / "config" / f"Exp2_{track.upper()}.yaml"))
                             if r["architecture"] == "tabpfn3"))
    run["prior"].update(n_rows_range=[96, 96], n_features_range=[3, 5], max_features=8,
                        n_nodes_range=[2, 4], max_filter_attempts=3, credit_fraction=0.5)
    # This integration test exercises training/checkpoint/inference, not statistical
    # rejection sampling on tiny 96-row tables. Filter acceptance is tested separately.
    run["prior"]["filter"]["mode"] = "off"
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
    y = (X[:, 0] > 0).astype(float) if track == "pd" else 1.0 / (1.0 + np.exp(-X[:, 0]))
    m = build("tabpfn3", track, seed=0, model_path=str(ck), n_estimators=1)
    m.fit(X[:90], y[:90], [])
    p = m.predict(X[90:])
    assert p.shape == (30,) and np.isfinite(p).all()
    assert m.report.extra["run_name"] == run["_run_name"] and m.report.extra["checkpoint_step"] == 2


def test_a_tabpfn_arm_refuses_tabicls_encoding(tmp_path):
    from src.train.tabpfn_trainer import TabPFNTrainer
    from src.utils.config import expand_with_seeds, load

    run = copy.deepcopy(next(r for r in expand_with_seeds(load(ROOT / "config" / "Exp2_PD.yaml"))
                             if r["architecture"] == "tabpfn3"))
    run["prior"]["encoding"] = "tabicl"
    run["logging"] = {"console": False, "to_file": False}
    # A temporary folder: until 30-09-2026 this pointed at the real output tree and left an empty
    # `_never_written` folder behind in it.
    with pytest.raises(ValueError, match="encoding: raw"):
        TabPFNTrainer(run, tmp_path / "never_written", device="cpu")
    assert not (tmp_path / "never_written").exists()


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
    r = subprocess.run([sys.executable, str(ROOT / "src/validation/checks.py"), "--only", "configs",
                        "--out", str(out)], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads((out / "checks.json").read_text(encoding="utf-8"))
    names = {row["name"] for row in report["rows"]}
    assert {"Exp0_PD", "Exp1_LGD", "Exp2_PD", "Exp2_LGD", "Exp3_PD"} <= names
    assert (out / "checks.md").read_text(encoding="utf-8").startswith("# Experiment 0")


def _fake_exp0_tree(root: Path, *, resumed: bool = True) -> None:
    from src.eval.benchmark_status import n_slots, slot_for
    from src.utils.config import expand_with_seeds, load
    from src.utils.preflight import CHECK_SECTIONS, source_fingerprint, training_fingerprint

    out = root / "output CreditICL"
    exp0 = out / "experiment_0"
    (exp0 / "report").mkdir(parents=True)
    (exp0 / "report" / "checks.json").write_text(json.dumps({
        "source_fingerprint": source_fingerprint(), "rows": [
            {"check": name, "name": "fixture", "status": "PASS", "detail": {}}
            for name in sorted(CHECK_SECTIONS)]}), encoding="utf-8")
    (exp0 / "logs").mkdir()
    (exp0 / "logs" / "exp0_checks_2.log").write_text(
        "equivalence PD exit code: 0\nequivalence LGD exit code: 0\nEND status=OK\n", encoding="utf-8")
    (exp0 / "logs" / "pretrain_pd_1.log").write_text(
        "...\nEND status=INCOMPLETE-RESUBMITTED - now\n" if resumed else "END status=OK\n", encoding="utf-8")
    (exp0 / "logs" / "pretrain_pd_2.log").write_text("END status=OK\n", encoding="utf-8")
    for track in ("pd", "lgd"):
        for run in expand_with_seeds(load(ROOT / "config" / f"Exp0_{track.upper()}.yaml")):
            d = exp0 / "runs" / run["_run_name"]
            d.mkdir(parents=True)
            (d / "config.json").write_text(json.dumps(run), encoding="utf-8")
            (d / "summary.json").write_text(json.dumps({
                "training_fingerprint": training_fingerprint(),
                "completed": True, "steps": run["train"]["max_steps"], "train_seconds_total": 600,
                "resumed_at": 240 if run["arm"] == "scratch_tabicl" and resumed else None}), encoding="utf-8")
            pd.DataFrame({"step": [300, 600, 600], "kind": ["hardware", "hardware", "grad"],
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
            from src.eval.benchmark_status import digest
            from src.eval.protocol import PROTOCOL

            (owner / "ood").mkdir(parents=True, exist_ok=True)
            files = {"credit": owner / track / f"results_{slot.tag}.csv",
                     "ood": owner / "ood" / f"ood_results_{slot.tag}.csv"}
            for domain, path in files.items():
                pd.DataFrame([dict(dataset=domain, model=slot.model, seed=0, fold=f, status="ok",
                                   protocol=PROTOCOL, context_cap=slot.context, r2=.3, roc_auc=.7,
                                   n_train=2000, n_train_used=min(2000, slot.context or 2000), n_test=500,
                                   prep_features_in=216, prep_features_used=216, n_features_used=216,
                                   total_seconds=900) for f in range(5)]).to_csv(path, index=False)
            request = dict(protocol=PROTOCOL, models=[slot.model], seeds=[0], n_folds=5,
                           credit=["credit"], ood=["ood"], context_cap=slot.context, step=slot.step)
            (owner / "receipts" / f"{slot.tag}.json").write_text(json.dumps({"request": request,
                "files": {d: digest(p) for d, p in files.items()}}), encoding="utf-8")


def test_the_verifier_passes_a_complete_experiment_0(tmp_path, capsys):
    from src.validation.verify import FAIL, verify

    _fake_exp0_tree(tmp_path)
    v = verify(tmp_path / "output CreditICL")
    fails = [item for item in v.items if item[0] == FAIL]
    assert not fails, fails
    assert "h per arm" in capsys.readouterr().out, "the cost extrapolation is printed"


def test_the_verifier_fails_when_the_forced_stop_never_happened(tmp_path):
    from src.validation.verify import FAIL, verify

    _fake_exp0_tree(tmp_path, resumed=False)
    v = verify(tmp_path / "output CreditICL")
    failed = {what for status, what, _ in v.items if status == FAIL}
    assert any("forced stop" in w for w in failed)
    assert any("resumed from a checkpoint" in w for w in failed)


def test_verifier_cannot_accept_a_receipt_with_a_missing_ood_result(tmp_path):
    from src.validation.verify import FAIL, verify

    _fake_exp0_tree(tmp_path)
    root = tmp_path / "output CreditICL"
    next((root / "experiment_0" / "benchmark" / "ood").glob("*.csv")).unlink()
    v = verify(root)
    assert any(s == FAIL and "benchmark slot" in what for s, what, _ in v.items)


def test_verifier_rejects_completed_training_with_a_changed_recipe(tmp_path):
    from src.validation.verify import FAIL, verify

    _fake_exp0_tree(tmp_path)
    root = tmp_path / "output CreditICL"
    path = next((root / "experiment_0" / "runs").glob("*/config.json"))
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["prior"]["credit_fraction"] = .75
    path.write_text(json.dumps(cfg), encoding="utf-8")
    verdict = verify(root)
    assert any(s == FAIL and "training recipe" in what for s, what, _ in verdict.items)
