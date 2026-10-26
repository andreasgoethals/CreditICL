"""Scientific budgets, development-only choice and safe campaign continuation."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.utils.config import expand_with_seeds, load
from src.experiments.design import MILESTONES, RATES


@pytest.mark.parametrize("task", ["pd", "lgd"])
def test_exp2_crosses_each_recipe_without_changing_geometry(task):
    runs = expand_with_seeds(load(f"config/Exp2_{task.upper()}.yaml"))
    assert len(runs) == len({r["_run_name"] for r in runs}) == 48
    assert {(r["architecture"], r["train"]["lr"], r["prior"]["credit_fraction"], r["seed"])
            for r in runs} == {(model, rate, mix, seed) for model, rates in RATES.items()
                              for rate in rates for mix in (0, .25, .5, 1) for seed in range(3)}
    for r in runs:
        t, p = r["train"], r["prior"]
        assert t["keep_steps"] == list(MILESTONES) and t["max_steps"] == 10000
        assert t["weight_decay"] == .01
        assert t["batch_size"] == (64 if r["architecture"] == "tabicl" else 16)
        assert p["n_rows_range"] == ([400, 60000] if r["architecture"] == "tabicl" else [1024, 1024])


@pytest.mark.parametrize("task", ["PD", "LGD"])
def test_exp3_is_bounded_and_requires_frozen_selection(task, isolated_output):
    with pytest.raises(ValueError, match="selection is not frozen"):
        load(f"config/Exp3_{task}.yaml")
    main = expand_with_seeds(load(f"config/Exp3_{task}.yaml", allow_placeholders=True))
    pilots = expand_with_seeds(load(f"config/Exp3_{task}_pilots.yaml", allow_placeholders=True))
    assert len(main) == 24 and len(pilots) == 18
    assert all(r["design_unresolved"] for r in main + pilots)
    for r in main + pilots:
        assert r["init"]["strategy"] == "full" and r["prior"]["credit_fraction"] == .5
        assert r["train"]["schedule_total_steps"] == 10000
        if r["method"] in {"lora", "diagscale"}:
            assert r["train"]["optimizer"] == "adamw" and r["train"]["weight_decay"] == .01
        else:
            assert r["train"]["weight_decay"] == 0
    assert all(r["train"]["max_steps"] == 2500 and r["seed"] == 991 for r in pilots)
    assert all(not r["eval"]["holdout_datasets"] and not r["eval"]["include_ood"]
               and r["progress"]["every_datasets"] == 0 for r in pilots)


def test_selection_ignores_holdout_and_ood_and_rejects_missing_folds():
    from src.experiments.selection import development_score, choose
    df = pd.DataFrame([dict(dataset=d, fold=f, seed=0, info_run_name="run", status="ok", roc_auc=.7)
                       for d in ("dev1", "dev2", "holdout", "ood") for f in range(5)])
    assert development_score(df, ["dev1", "dev2"], "run", "roc_auc") == pytest.approx(.7)
    df.loc[df.dataset.isin(["holdout", "ood"]), "roc_auc"] = np.nan
    assert development_score(df, ["dev1", "dev2"], "run", "roc_auc") == pytest.approx(.7)
    with pytest.raises(ValueError, match="Incomplete"):
        development_score(df.iloc[1:], ["dev1", "dev2"], "run", "roc_auc")
    assert choose([dict(score=.7, value=.01), dict(score=.7, value=.001)])["value"] == .001


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
    assert all((folder / f"step-{s}.ckpt").exists() and (destination / f"step-{s}.ckpt").exists()
               for s in (125, 250, 500, 1000))
    assert not (folder / "step-750.ckpt").exists()
    cfg = expand_with_seeds(load("config/Exp2_PD.yaml"))[0]
    assert checkpoint_steps(cfg) == [10000, 125, 250, 500, 1000, 2500, 5000, 7500]


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
