"""Fixed scientific budgets, paired treatment baselines and safe continuation."""
import copy

import pytest

from src.utils.config import expand_with_seeds, load
from src.experiments.design import MAIN_STEPS, MILESTONES, RATES, FIXED_RETENTION


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_exp2_crosses_each_recipe_without_changing_geometry(task):
    runs = expand_with_seeds(load(f"config/Exp2_{task.upper()}.yaml", allow_placeholders=True))
    assert len(runs) == len({r["_run_name"] for r in runs}) == 48
    assert {(r["architecture"], r["train"]["lr"], r["prior"]["credit_fraction"], r["seed"])
            for r in runs} == {(model, rate, mix, seed) for model, rates in RATES.items()
                              for rate in rates for mix in (0, .25, .5, 1) for seed in range(3)}
    for r in runs:
        t, p = r["train"], r["prior"]
        assert t["keep_steps"] == list(MILESTONES) and t["max_steps"] == MAIN_STEPS
        assert t["weight_decay"] == .01
        assert t["batch_size"] == (64 if r["architecture"] == "tabicl" else 16)
        assert p["n_rows_range"] == ([400, 60000] if r["architecture"] == "tabicl" else [1024, 1024])


@pytest.mark.parametrize("task", ["PD", "LGD"])
def test_exp3_is_fixed_with_exactly_six_reusable_baselines_per_task(task, isolated_output):
    main = expand_with_seeds(load(f"config/Exp3_{task}.yaml"))
    exp2 = expand_with_seeds(load(f"config/Exp2_{task}.yaml"))
    baselines = [r for r in exp2 if r["lr_index"] == 0 and r["prior"]["credit_fraction"] == .5]
    assert len(main) == 24 and len(baselines) == 6
    assert all(not r["design_unresolved"] for r in main)
    for r in main:
        baseline = next(b for b in baselines if (b["architecture"], b["seed"]) == (r["architecture"], r["seed"]))
        assert r["init"]["strategy"] == "full" and r["prior"]["credit_fraction"] == .5
        assert r["lr_index"] == 0 and r["prior"] == baseline["prior"]
        for key in ("max_steps", "batch_size", "micro_batch_size", "amp", "schedule_total_steps"):
            assert r["train"][key] == baseline["train"][key]
        assert r["train"]["schedule_total_steps"] == MAIN_STEPS
        if r["method"] in {"lora", "diagscale"}:
            assert r["train"]["optimizer"] == "adamw" and r["train"]["weight_decay"] == .01
            assert r["train"]["lr"] == FIXED_RETENTION[f"{r['method']}_lr"]
        else:
            assert r["train"]["weight_decay"] == 0
            assert r["train"]["lr"] == baseline["train"]["lr"]
        assert r["train"]["l2sp_alpha"] == (.003 if r["method"] == "l2sp" else 0.)
        assert baseline["train"]["weight_decay"] == .01


def test_retention_rate_cannot_be_changed_after_observing_exp2():
    from src.experiments.design import apply_design
    run = expand_with_seeds(load("config/Exp3_PD.yaml"))[0]
    run["lr_index"] = 1
    with pytest.raises(ValueError, match="primary"):
        apply_design(run)


def test_early_checkpoints_are_published_and_never_pruned(tmp_path):
    from src.train.checkpoint import prune_checkpoints
    from src.utils.working_storage import publish_checkpoint
    from src.eval.protocol import checkpoint_steps
    folder, destination = tmp_path / "scratch", tmp_path / "project"
    folder.mkdir()
    (folder / "destination.txt").write_text(str(destination))
    for step in (125, 250, 500, 750, 1000, 1250, 1500, 1750):
        (folder / f"step-{step}.ckpt").write_bytes(str(step).encode())
        publish_checkpoint(folder, step, permanent_every=2500, keep_steps=MILESTONES,
                           final=False, stopped=False)
    prune_checkpoints(folder, save_perm_every=2500, max_temp=2, keep_steps=MILESTONES)
    assert all((destination / f"step-{s}.ckpt").exists()
               for s in (125, 250, 500, 1000))
    assert not (folder / "step-750.ckpt").exists()
    cfg = expand_with_seeds(load("config/Exp2_PD.yaml", allow_placeholders=True))[0]
    assert checkpoint_steps(cfg) == [MAIN_STEPS, 125, 250, 500]


def test_stale_gpu_campaign_starts_fresh_but_corrupt_matching_campaign_does_not(monkeypatch):
    from src.validation import campaign as c
    from src.utils import preflight, environment
    from src.eval import benchmark_status as b
    monkeypatch.setattr(preflight, "source_fingerprint", lambda: "new")
    monkeypatch.setattr(environment, "environment_fingerprint", lambda: "packages")
    monkeypatch.setattr(c, "input_inventory", lambda: {"data": "one"})
    monkeypatch.setattr(b, "input_digest", lambda p: "cpu")
    saved = dict(schema=c.SCHEMA, phase="gpu", source_fingerprint="new",
                 environment_fingerprint="packages", inputs={"data": "one"}, cpu_receipt="cpu",
                 status="FAILED", stages=[])
    assert c.continuation_is_current(saved, "gpu")
    with pytest.raises(ValueError, match="failed"):
        c.verify_continuation(saved, "gpu")
    for key in ("source_fingerprint", "environment_fingerprint", "cpu_receipt", "inputs"):
        stale = copy.deepcopy(saved)
        stale[key] = "old"
        assert not c.continuation_is_current(stale, "gpu")
