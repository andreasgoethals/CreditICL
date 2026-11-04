"""Revised retention grid, prior campaign compatibility and rate-safe analysis."""
import hashlib
import json

import pandas as pd
import pytest
import torch

from src.experiments.design import RETENTION_RECIPES, RETENTION_REVISION, apply_design, retention_baseline
from src.utils.config import load, expand_with_seeds, effective_fingerprint, run_name


@pytest.mark.parametrize("experiment,task,digest", [
    (1, "PD", "26a37fe421a7338791b3e2d417c86301b7721ff14c7686e6bbda8908d7e5f302"),
    (1, "LGD", "3c06f72ec6bebc327216c45af58db9a2841810046c0d6c5b9f0d4900af72ccd8"),
    (2, "PD", "a20d00fee3b7d7b9578736cf46f3717d8d57d34aa207fb305f591bb25ded625b"),
    (2, "LGD", "bdae6e01cbd1229670c3deb772adf60560c92d66930f0a2d9a36206f3b5ec05a"),
])
def test_existing_campaign_names_array_ids_and_effective_configs_are_unchanged(experiment, task, digest):
    # Captured from main 5e1f815 BEFORE the Exp3 revision, not generated from new settings.
    runs = expand_with_seeds(load(f"config/Exp{experiment}_{task}.yaml"))
    actual = [(r["_run_name"], r["_grid"]["array_index"], effective_fingerprint(r)) for r in runs]
    assert hashlib.sha256(json.dumps(actual).encode()).hexdigest() == digest


@pytest.mark.parametrize("task", ["PD", "LGD"])
def test_unique_recipes_and_both_reused_baselines_have_separate_identities(task):
    runs = expand_with_seeds(load(f"config/Exp3_{task}.yaml"))
    baselines = [r for r in expand_with_seeds(load(f"config/Exp2_{task}.yaml")) if retention_baseline(r)]
    assert len(runs) == 24 and len(baselines) == 8
    assert len({run_name(r) for r in runs + baselines}) == 32
    assert len({effective_fingerprint(r) for r in runs}) == 24
    assert {r["lr_index"] for r in baselines} == {0, 1}
    for r in runs:
        assert r["design_revision"] == RETENTION_REVISION
        old = dict(r, design_revision="fixed-rates-1500-two-seeds-v2")
        assert run_name(old) != run_name(r)
        with pytest.raises(ValueError, match="revision"):
            apply_design(old)


@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("backbone", ["tabicl", "tabpfn3"])
@pytest.mark.parametrize("method", list(RETENTION_RECIPES))
def test_exp0_each_recipe_uses_production_rate_decay_precision_and_anchor(task, backbone, method):
    from src.validation.adaptation import diagnostic_config
    actual = diagnostic_config(task, backbone, method)
    expected = next(r for r in expand_with_seeds(load(f"config/Exp3_{task.upper()}.yaml"))
                    if r["architecture"] == backbone and r["method"] == method and r["seed"] == 0)
    for k in ("lr", "optimizer", "beta1", "beta2", "weight_decay", "amp", "l2sp_alpha", "adaptation", "gradient_clipping"):
        assert actual["train"][k] == expected["train"][k]
    assert actual["design_revision"] == expected["design_revision"]
    assert actual["prior"]["train_frac_range"] == expected["prior"]["train_frac_range"]
    assert actual["init"] == expected["init"]


@pytest.mark.parametrize("backbone", ["tabicl", "tabpfn3"])
@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("method", ["diagscale_low", "diagscale_high", "lora_low", "lora_high"])
def test_actual_adapter_rates_update_only_adapters_and_keep_identity_at_start(backbone, task, method):
    from test_adapters import tiny_model, forward
    from src.models.adapters import inject
    from src.train.optim import build_optimizer
    torch.manual_seed(42)
    model = tiny_model(backbone, task).eval()
    initial = forward(model, backbone, task).detach()
    run = next(r for r in expand_with_seeds(load(f"config/Exp3_{task.upper()}.yaml"))
               if r["architecture"] == backbone and r["method"] == method)
    cfg = run["train"]
    inject(model, cfg)
    torch.testing.assert_close(forward(model, backbone, task), initial, atol=0, rtol=0)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = build_optimizer(model, cfg)
    model.train()
    forward(model, backbone, task).square().mean().backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["gradient_clipping"])
    optimizer.step()
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before)
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_exp3_readers_reuse_eight_exp2_files_and_never_relabel_old_exp3(monkeypatch, tmp_path):
    from src.experiments import results
    from src.eval.protocol import PROTOCOL
    monkeypatch.setattr(results.paths, "benchmark_dir", lambda exp, task: tmp_path / str(exp) / task)
    cfg = load("config/Exp2_PD.yaml")
    names = cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"]
    folder = tmp_path / "2" / "pd"
    folder.mkdir(parents=True)
    for r in expand_with_seeds(cfg):
        if not retention_baseline(r):
            continue
        pd.DataFrame([dict(dataset=d, seed=0, fold=f, model="crediticl", info_run_name=run_name(r),
                           status="ok", protocol=PROTOCOL, info_checkpoint_step=1500,
                           roc_auc=.6 + r["lr_index"] * .1) for d in names for f in range(5)]).to_csv(
            folder / f"results_exp2bench_pd_a{r['_grid']['array_index']}.csv", index=False)
    retired = expand_with_seeds(load("config/Exp3_PD.yaml"))[0]
    retired["design_revision"] = "fixed-rates-1500-two-seeds-v2"
    stale = tmp_path / "3" / "pd"
    stale.mkdir(parents=True)
    pd.DataFrame([dict(dataset=d, seed=0, fold=f, model="crediticl", info_run_name=run_name(retired),
                       status="ok", protocol=PROTOCOL, info_checkpoint_step=1500, roc_auc=.99)
                  for d in names for f in range(5)]).to_csv(stale / "results_exp3bench_pd_a0.csv", index=False)
    with pytest.warns(UserWarning, match="wrong identity"):
        frame = results.benchmark_frames("pd", 3, final_only=True)
    assert set(frame.treatment) == {"full_wd", "full_wd_lower"}
    scores = results.complete_scores(frame, "roc_auc", experiment=3)
    assert scores[scores.treatment.eq("full_wd")]["mean"].tolist() == pytest.approx([.6, .6])
    assert scores[scores.treatment.eq("full_wd_lower")]["mean"].tolist() == pytest.approx([.7, .7])
    assert results.paired_retention(frame[frame.architecture.eq("tabicl")], "roc_auc").empty


def test_anchor_contrast_is_not_confounded_with_decay_and_never_pools_rates():
    from test_continuation_plots import curve_cells
    from src.experiments.results import paired_retention
    frame = curve_cells(3).query("architecture == 'tabicl' and checkpoint_step == 1500").copy()
    frame.loc[frame.treatment.eq("no_decay"), "roc_auc"] += .02
    frame.loc[frame.treatment.eq("l2sp"), "roc_auc"] += .05
    frame.loc[frame.treatment.eq("full_wd_lower"), "roc_auc"] += .01
    primary = paired_retention(frame, "roc_auc").set_index("configuration")
    lower = paired_retention(frame, "roc_auc", baseline="full_wd_lower").set_index("configuration")
    assert primary.loc["exp3_l2sp", "mean_difference"] == pytest.approx(.03)
    assert primary.loc["exp3_l2sp", "control"] == "exp3_no_decay"
    assert lower.loc["exp3_l2sp", "mean_difference"] == pytest.approx(.04)
    for m in RETENTION_RECIPES:
        assert paired_retention(frame[~((frame.treatment == m) & (frame.training_seed == 1))], "roc_auc").empty
    with pytest.raises(ValueError, match="one backbone"):
        paired_retention(curve_cells(3), "roc_auc")


def test_planner_schedules_only_48_new_runs_and_240_new_checkpoint_slots(monkeypatch, tmp_path):
    from src.utils import paths, run_experiment
    from src.eval.benchmark_status import n_slots
    monkeypatch.setattr(paths, "outputs_dir", lambda: tmp_path / "output")
    monkeypatch.setattr(run_experiment, "queued_job_names", set)
    stages = run_experiment.plan(3, ["pd", "lgd"], single_cluster=True)
    train = [s for s in stages if s.kind == "train"]
    assert len(train) == 2 and sum(s.total for s in train) == 48
    assert all(s.done == 0 and len(s.commands) == 1 for s in train)
    assert all("--array=0-23" in s.commands[0] for s in train)
    assert all(s.blocked_by for s in stages if s.kind == "benchmark")
    # Four shared reference slots per task are not new CPT baseline training.
    assert sum(n_slots(3, t) for t in ("pd", "lgd")) == 240 + 8


@pytest.mark.parametrize("task", ["PD", "LGD"])
def test_message_only_receipt_transition_keeps_exp2_and_rejects_scientific_changes(task):
    from src.utils.benchmark_compat import compatible_requests
    from src.eval.benchmark_status import digest
    from pathlib import Path
    import copy
    new_hash = digest(Path("src/eval/selection.py"))
    old_hash = "192beb936e80b767f4750169e3561d7c29db15414ecb1c127d9d7edc820c6215"
    for run in expand_with_seeds(load(f"config/Exp2_{task}.yaml")):
        old = dict(config=run, code={"src/eval/selection.py": old_hash}, step=1500,
                   checkpoint_sha256="same", protocol=9, n_folds=5, credit_inputs="same", files="same")
        new = copy.deepcopy(old)
        new["code"]["src/eval/selection.py"] = new_hash
        assert compatible_requests(old, new)
        for key in ("checkpoint_sha256", "protocol", "n_folds", "credit_inputs", "files"):
            mutated = copy.deepcopy(new); mutated[key] = "changed"
            assert not compatible_requests(old, mutated)
        new["config"]["train"]["lr"] *= 2
        assert not compatible_requests(old, new)
