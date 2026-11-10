"""Preserve valid pre-repair CPT prefixes without granting general provenance bypasses."""
import copy

import pytest

from src.train.checkpoint import validate_training_fingerprint
from src.utils import completed_runs as compat
from src.utils import preflight
from src.utils.config import expand_with_seeds, load

RUNS = [r for track in ("PD", "LGD") for r in expand_with_seeds(load(f"config/Exp2_{track}.yaml"))]


def checkpoint(run, step=250):
    return {"step": step, "crediticl_config": copy.deepcopy(run),
            "training_fingerprint": preflight.training_fingerprint_for(compat.CONTEXT_FIX_PREDECESSOR),
            "extra": {"datasets_seen": step * run["train"]["batch_size"], "skipped_tasks": 0}}


@pytest.mark.parametrize("run", RUNS, ids=lambda r: r["_run_name"])
def test_all_64_exact_cpt_recipes_keep_completed_results_and_frozen_prefixes(run, monkeypatch):
    run = copy.deepcopy(run)
    retired = run["architecture"] == "tabpfn3" and run["task"] == "lgd"
    run["train"].pop("regression_target_policy", None)
    state = checkpoint(run)
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match="training code differs"):
        validate_training_fingerprint(state, run)
    summary = {"steps": 1500, "completed": True,
               "datasets_seen": 1500 * run["train"]["batch_size"], "skipped_tasks": 0,
               "training_fingerprint": state["training_fingerprint"]}
    if retired:
        with pytest.raises(ValueError, match="training code differs"):
            compat.validate_completed_training(summary, run, copy.deepcopy(run))
    else:
        compat.validate_completed_training(summary, run, copy.deepcopy(run))
    summary["training_fingerprint"] = preflight.training_fingerprint_for(compat.REVIEWED_IMPLEMENTATION)
    if retired:
        with pytest.raises(ValueError, match="training code differs"):
            compat.validate_completed_training(summary, run, copy.deepcopy(run))
    else:
        compat.validate_completed_training(summary, run, copy.deepcopy(run))
    summary.update(steps=250, datasets_seen=250 * run["train"]["batch_size"],
                   completed=False, stopped_by_signal="SIGUSR1")
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(summary, run, copy.deepcopy(run))
    monkeypatch.setattr(preflight, "training_implementation_fingerprint", lambda: compat.REVIEWED_IMPLEMENTATION)
    validate_training_fingerprint(state, run)
    assert state == original
    summary["training_fingerprint"] = state["training_fingerprint"]
    compat.validate_completed_training(summary, run, copy.deepcopy(run))


@pytest.mark.parametrize("change", ["source", "packages", "seed", "recipe", "saved_recipe",
                                    "budget", "skipped", "unknown_history", "missing_config",
                                    "boolean_step", "past_horizon"])
def test_repair_permission_rejects_unreviewed_changes(monkeypatch, change):
    monkeypatch.setattr(preflight, "training_implementation_fingerprint", lambda: compat.REVIEWED_IMPLEMENTATION)
    run = copy.deepcopy(RUNS[11])
    state = checkpoint(run)
    if change == "source":
        monkeypatch.setattr(preflight, "training_implementation_fingerprint", lambda: "future")
    elif change == "packages":
        before = preflight.training_fingerprint_for
        monkeypatch.setattr(preflight, "training_fingerprint_for", lambda v: before(v) + "changed")
    elif change == "seed":
        run["seed"] = 3
    elif change == "recipe":
        run["train"]["lr"] = state["crediticl_config"]["train"]["lr"] = 0.1
    elif change == "saved_recipe":
        state["crediticl_config"]["prior"]["credit_fraction"] = .25
    elif change == "budget":
        state["extra"]["datasets_seen"] -= 1
    elif change == "skipped":
        state["extra"]["skipped_tasks"] = 1
    elif change == "unknown_history":
        state["training_fingerprint"] = "unknown"
    elif change == "missing_config":
        state.pop("crediticl_config")
    elif change == "boolean_step":
        state["step"] = True
    elif change == "past_horizon":
        state["step"] = 1501
    with pytest.raises(ValueError, match="training code differs"):
        validate_training_fingerprint(state, run)


def test_resume_requires_the_recipe_even_for_a_known_predecessor():
    with pytest.raises(ValueError, match="training code differs"):
        validate_training_fingerprint(checkpoint(RUNS[11]))
