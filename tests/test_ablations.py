"""The component study changes one specified mechanism, at a fixed budget/mix."""
import copy
from collections import Counter

import pytest
import torch

from src.utils.config import expand_with_seeds, load


def leaves(obj, prefix=""):
    result = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(leaves(value, path))
        else:
            result[path] = value
    return result


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_five_component_tests_share_the_main_training_and_evaluation_budget(track):
    main = expand_with_seeds(load(f"config/Exp1_{track.upper()}.yaml"))
    runs = expand_with_seeds(load(f"config/Exp1_{track.upper()}_ablations.yaml"))
    assert len(main) == 9 and len(runs) == 15
    assert Counter(r["arm"] for r in runs) == dict.fromkeys(
        ["no_macro_target", "no_informative_missingness", "no_shift", "no_tail_warp", "base_cardinality"], 3)
    permitted = {
        "no_macro_target": ({"credit.target.mechanism.rho_range"} if track == "pd" else
                            {f"credit.target.zoib.macro_{x}_range" for x in ("p0", "p1", "mu")}),
        "no_informative_missingness": {"credit.missingness.missing_target_coupling_range"},
        "no_shift": {"credit.shift.shift_prob"},
        "no_tail_warp": {"credit.marginals.col_fraction_range"},
        "base_cardinality": {"credit.max_cat_size"},
    }
    for run in runs:
        reference = next(r for r in main if r["seed"] == run["seed"] and r["prior"]["credit_fraction"] == .5)
        assert run["train"] == {**reference["train"], "save_perm_every": 12500}
        assert run["eval"] == {**reference["eval"], "include_reference_slots": False}
        left, right = leaves(reference["prior"]), leaves(run["prior"])
        assert {k for k in left.keys() | right.keys() if left.get(k) != right.get(k)} == permitted[run["arm"]]
        assert run["_run_name"] != reference["_run_name"]
    from src.eval.benchmark_status import n_slots, slot_for

    assert n_slots(1, track, variant="ablations") == 30
    assert all(slot_for(1, track, i, variant="ablations").kind == "arm" for i in range(30))


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_every_component_variant_generates_valid_tasks_and_preserves_control_slots(track):
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.prior.stream import plan_batch

    main = next(r for r in expand_with_seeds(load(f"config/Exp1_{track.upper()}.yaml"))
                if r["seed"] == 0 and r["prior"]["credit_fraction"] == .5)
    prior = copy.deepcopy(main["prior"])
    prior.update(n_rows_range=[128, 128], n_features_range=[4, 8], max_features=8)
    prior["filter"]["mode"] = "off"
    plan = plan_batch(0, 0, prior, 8, regression=(track == "lgd"), credit_fraction=.5)
    reference = TaskGenerator(prior, track, PriorRNG(0), stream_seed=0)
    controls = {s.slot: reference.sample_slot(s) for s in plan.slots if not s.use_credit}
    assert controls
    for run in expand_with_seeds(load(f"config/Exp1_{track.upper()}_ablations.yaml")):
        if run["seed"] != 0:
            continue
        modified = copy.deepcopy(prior)
        modified["credit"] = run["prior"]["credit"]
        gen = TaskGenerator(modified, track, PriorRNG(0), stream_seed=0)
        for slot in plan.slots:
            task = gen.sample_slot(slot)
            assert torch.isfinite(task.X).all() and torch.isfinite(task.y).all()
            if slot.slot in controls:
                torch.testing.assert_close(task.X, controls[slot.slot].X, rtol=0, atol=0)
                torch.testing.assert_close(task.y, controls[slot.slot].y, rtol=0, atol=0)
