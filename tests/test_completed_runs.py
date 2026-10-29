"""Completed weights remain usable after reviewed Exp2-only changes; resume does not."""
import copy

import pytest

from src.utils import completed_runs as compat, preflight
from src.utils.config import expand_with_seeds, load


def original_runs():
    return [run for track in ("PD", "LGD") for variant in ("", "_ablations")
            for run in expand_with_seeds(load(f"config/Exp1_{track}{variant}.yaml"))
            if variant or run["prior"]["credit_fraction"] != .25]


def historical_summary():
    return {"steps": 12500, "completed": True, "datasets_seen": 800000,
            "training_fingerprint": preflight.training_fingerprint_for(compat.HISTORICAL_IMPLEMENTATION)}


@pytest.mark.parametrize("run", original_runs(), ids=lambda r: r["_run_name"])
def test_all_48_reviewed_runs_remain_complete_without_rewriting_provenance(run):
    summary = historical_summary()
    before = copy.deepcopy(summary)
    assert summary["training_fingerprint"] != preflight.training_fingerprint()
    compat.validate_completed_training(summary, run, copy.deepcopy(run))
    assert summary == before


@pytest.mark.parametrize("field,value", [
    ("completed", False), ("steps", 12250), ("steps", 15000),
    ("datasets_seen", 799999), ("training_fingerprint", "unreviewed-history"),
    ("training_fingerprint", None),
])
def test_incomplete_or_unknown_history_is_still_rejected(field, value):
    run = original_runs()[0]
    summary = historical_summary()
    summary[field] = value
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(summary, run, copy.deepcopy(run))


@pytest.mark.parametrize("key,value", [("seed", 4), ("seed", True), ("experiment", "exp2_pd")])
def test_other_campaigns_and_seeds_do_not_inherit_permission(key, value):
    run = copy.deepcopy(original_runs()[0])
    run[key] = value
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(historical_summary(), run, copy.deepcopy(run))


@pytest.mark.parametrize("field,value", [("lr", .1), ("batch_size", 16),
                                         ("amp", True), ("max_steps", 15000)])
def test_a_changed_recipe_is_rejected_even_if_both_config_copies_change(field, value):
    run = copy.deepcopy(original_runs()[0])
    run["train"][field] = value
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(historical_summary(), run, copy.deepcopy(run))


def test_missing_saved_recipe_and_new_quarter_runs_cannot_be_grandfathered():
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(historical_summary(), original_runs()[0], None)
    quarter = expand_with_seeds(load("config/Exp1_PD.yaml"))[9]
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(historical_summary(), quarter, copy.deepcopy(quarter))


def test_future_training_source_change_invalidates_the_exception(monkeypatch):
    summary = historical_summary()
    monkeypatch.setattr(preflight, "training_implementation_fingerprint", lambda: "future-source")
    run = original_runs()[0]
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(summary, run, copy.deepcopy(run))


def test_dependency_change_invalidates_historical_provenance(monkeypatch):
    summary = historical_summary()
    original = preflight.metadata.distribution

    class ChangedDistribution:
        version = "999.0"

        def read_text(self, name):
            return None

    monkeypatch.setattr(preflight.metadata, "distribution", lambda name:
                        ChangedDistribution() if name == "numpy" else original(name))
    run = original_runs()[0]
    with pytest.raises(ValueError, match="training code differs"):
        compat.validate_completed_training(summary, run, copy.deepcopy(run))


def test_historical_completion_never_authorizes_checkpoint_resume():
    from src.train.checkpoint import validate_training_fingerprint

    summary = historical_summary()
    run = original_runs()[0]
    compat.validate_completed_training(summary, run, copy.deepcopy(run))
    with pytest.raises(ValueError, match="training code differs"):
        validate_training_fingerprint(summary)


def test_current_training_fingerprint_is_unchanged_by_helper_extraction():
    assert preflight.training_implementation_fingerprint() == compat.REVIEWED_IMPLEMENTATION
    assert preflight.training_fingerprint() == preflight.training_fingerprint_for(compat.REVIEWED_IMPLEMENTATION)


@pytest.mark.parametrize("track", ["PD", "LGD"])
def test_previous_revision_preserves_all_27_recipes_and_only_their_exact_resume(track):
    from src.train.checkpoint import validate_training_fingerprint
    runs = [r for variant in ("", "_ablations")
            for r in expand_with_seeds(load(f"config/Exp1_{track}{variant}.yaml"))]
    assert len(runs) == 27
    for run in runs:
        fingerprint = preflight.training_fingerprint_for(compat.PREVIOUS_IMPLEMENTATION)
        completed = dict(steps=12500, completed=True, datasets_seen=800000,
                         training_fingerprint=fingerprint)
        compat.validate_completed_training(completed, run, copy.deepcopy(run))
        checkpoint = dict(step=250, extra={"datasets_seen": 16000},
                          crediticl_config=copy.deepcopy(run), training_fingerprint=fingerprint)
        validate_training_fingerprint(checkpoint, current_config=run)
        bad = copy.deepcopy(checkpoint)
        bad["extra"]["datasets_seen"] -= 1
        with pytest.raises(ValueError, match="training code differs"):
            validate_training_fingerprint(bad, current_config=run)
        bad = copy.deepcopy(checkpoint)
        bad["crediticl_config"]["seed"] = 99
        with pytest.raises(ValueError, match="training code differs"):
            validate_training_fingerprint(bad, current_config=run)


@pytest.mark.parametrize("track", ['PD', 'LGD'])
def test_oct26_exp1_results_survive_tabpfn_only_numerical_fix(track):
    from src.train.checkpoint import validate_training_fingerprint

    for variant in ['', '_ablations']:
        for run in expand_with_seeds(load(f'config/Exp1_{track}{variant}.yaml')):
            fingerprint = preflight.training_fingerprint_for(compat.OCT26_IMPLEMENTATION)
            summary = dict(steps=12500, completed=True, datasets_seen=800000,
                           training_fingerprint=fingerprint)
            compat.validate_completed_training(summary, run, copy.deepcopy(run))
            # Exp2 has a changed numerical path and must not inherit Exp1's exception.
            changed = copy.deepcopy(run)
            changed['experiment'] = f'exp2_{track.lower()}'
            with pytest.raises(ValueError, match='training code differs'):
                validate_training_fingerprint(summary, current_config=changed)
