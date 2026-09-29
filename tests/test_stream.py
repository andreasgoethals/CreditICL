"""The task stream and the batch format — the guarantees Experiment 1's comparisons rest on.

Each test names what it protects. The first three are the paired design: arms that differ only
in `credit_fraction` must see IDENTICAL base tasks, in the same order, however many workers
build them and whether or not the run was resumed. The rest pin upstream's batch rules and the
train/prediction consistency of what reaches the model.
"""

from __future__ import annotations

import copy

import pytest
import torch

from src.prior.dataset import PriorBatchDataset
from src.prior.generator import TaskGenerator, delete_constant_columns, split_noise_budget
from src.prior.rng import PriorRNG
from src.prior.stream import plan_batch, slot_seed


def _arm(cfg: dict, cf: float, seed: int = 3, batch_size: int = 8):
    prior = copy.deepcopy(cfg["prior"])
    prior["credit_fraction"] = cf
    ds = PriorBatchDataset(prior, cfg["task"], batch_size, seed=seed)
    gen = TaskGenerator(prior, cfg["task"], PriorRNG(seed), stream_seed=seed)
    return ds, gen, prior


@pytest.mark.parametrize("which", ["pd", "lgd"])
def test_base_slots_are_identical_across_credit_fractions(pd_cfg, lgd_cfg, which):
    """Common random numbers: a slot that holds a base task in two arms holds THE SAME task."""
    cfg = pd_cfg if which == "pd" else lgd_cfg
    ds0, gen0, _ = _arm(cfg, 0.0)
    ds5, gen5, prior5 = _arm(cfg, 0.5)
    for b in range(2):
        X0, y0, d0, s0, t0 = ds0.batch(gen0, b)
        X5, y5, d5, s5, t5 = ds5.batch(gen5, b)
        assert torch.equal(t0, t5) and torch.equal(s0, s5), "splits and row counts are shared"
        plan = plan_batch(3, b, prior5, 8, regression=which == "lgd", credit_fraction=0.5)
        base = [s.slot for s in plan.slots if not s.use_credit]
        assert base, "the test needs at least one base slot"
        for j in base:
            assert torch.equal(X0[j], X5[j]) and torch.equal(y0[j], y5[j]) and d0[j] == d5[j]


def test_credit_slots_are_nested_across_fractions(pd_cfg):
    """The coin is shared, so the 25 % arm's credit slots are a subset of the 50 % arm's."""
    p = pd_cfg["prior"]
    for b in range(5):
        c25 = {s.slot for s in plan_batch(1, b, p, 64, regression=False, credit_fraction=0.25).slots if s.use_credit}
        c50 = {s.slot for s in plan_batch(1, b, p, 64, regression=False, credit_fraction=0.5).slots if s.use_credit}
        assert c25 <= c50


def test_a_batch_does_not_depend_on_the_worker_that_builds_it(pd_cfg):
    ds, gen, prior = _arm(pd_cfg, 0.5)
    other = TaskGenerator(prior, "pd", PriorRNG(3, worker_id=5), stream_seed=3)
    for a, b in zip(ds.batch(gen, 4), ds.batch(other, 4)):
        assert torch.equal(a, b)


def test_iteration_strides_workers_and_resumes_at_the_right_batch(pd_cfg):
    """Worker w of W yields batches start+w, start+w+W, ...; a resumed loader starts at `step`."""
    ds, gen, prior = _arm(pd_cfg, 0.5)
    resumed = PriorBatchDataset(prior, "pd", 8, seed=3, start_batch=2)
    first = next(iter(resumed))
    for a, b in zip(first, ds.batch(gen, 2)):
        assert torch.equal(a, b)


def test_groups_share_their_split_and_subgroups_their_feature_count(pd_cfg):
    """Upstream's `GraphPrior.get_batch`: per group, one row count and split; per subgroup,
    one feature count; per dataset, its own number of classes (2..max_classes)."""
    p = copy.deepcopy(pd_cfg["prior"])
    p["max_classes"] = 10
    plan = plan_batch(0, 0, p, 16, regression=False, credit_fraction=0.0)
    for g in range(4):
        slots = [s for s in plan.slots if s.group == g]
        assert len({s.train_size for s in slots}) == 1 and len({s.seq_len for s in slots}) == 1
        assert len({s.num_features for s in slots}) == 1   # subgroup == group by default
    classes = {s.num_classes for s in plan.slots}
    assert classes <= set(range(2, 11)) and len(classes) > 1


def test_different_groups_get_different_splits(pd_cfg):
    splits = {s.train_size for b in range(3)
              for s in plan_batch(0, b, pd_cfg["prior"], 16, regression=False, credit_fraction=0.0).slots}
    assert len(splits) > 3, "one split per GROUP, not one per batch"


def test_slot_seeds_never_collide_across_kinds():
    seeds = {slot_seed(0, b, j, k) for b in range(4) for j in range(16) for k in ("coin", "base", "credit", "classes")}
    assert len(seeds) == 4 * 16 * 4


def test_constant_columns_are_deleted_and_the_table_compacted():
    """Upstream's `delete_unique_features`: constants out, real columns first, zero padding."""
    X = torch.randn(32, 6)
    X[:, 1] = 3.0
    X[:, 4] = -1.0
    out, d = delete_constant_columns(X, width=6)
    assert d == 4
    assert torch.equal(out[:, :4], X[:, [0, 2, 3, 5]]) and (out[:, 4:] == 0).all()


def test_noise_columns_live_inside_the_feature_budget():
    """Appending noise made credit tables wider than their neighbours, so the width alone told
    the model which prior a table came from."""
    for n in (1, 2, 7, 20, 100):
        info, noise = split_noise_budget(n, 0.3)
        assert info + noise == n and info >= 1


def test_credit_lgd_targets_are_standardised_like_a_real_target_at_prediction(lgd_cfg):
    """`TabICLRegressor.fit` standardises y (`y_scaler_`); Exp1 trained on raw [0, 1]."""
    prior = copy.deepcopy(lgd_cfg["prior"])
    prior["credit_fraction"] = 1.0
    gen = TaskGenerator(prior, "lgd", PriorRNG(0))
    for _ in range(3):
        task = gen.sample(shape=(256, 8))
        assert task.meta["target_scaling"] == "standard"
        # On the CONTEXT targets, as `y_scaler_` is fitted: a shifted query may sit off centre.
        ctx = task.y[: int(task.meta["train_size"])].double()
        assert abs(float(ctx.mean())) < 1e-4 and abs(float(ctx.std(unbiased=False)) - 1.0) < 1e-4


def test_indicator_columns_are_refused(pd_cfg):
    prior = copy.deepcopy(pd_cfg["prior"])
    prior["credit"]["missingness"]["missing_indicators"] = True
    with pytest.raises(ValueError, match="missing_indicators must be false"):
        TaskGenerator(prior, "pd", PriorRNG(0))


def test_credit_pd_labels_have_random_identity(pd_cfg):
    """Upstream permutes class labels on every table; a credit prior that always writes
    `1 = default` would teach an asymmetry the prediction ensemble's class rotation never has."""
    prior = copy.deepcopy(pd_cfg["prior"])
    prior["credit_fraction"] = 1.0
    gen = TaskGenerator(prior, "pd", PriorRNG(0))
    swapped = [bool(gen.sample(shape=(128, 6)).meta.get("labels_swapped")) for _ in range(16)]
    assert any(swapped) and not all(swapped)


def test_mechanism_mode_applies_label_noise_and_selection(pd_cfg):
    """Until 28-09-2026 the mechanism branch returned before both, although the config set them."""
    from src.prior.targets.pd import apply_pd_target

    cfg = {"mode": "mechanism", "flip_pos_to_neg": 0.1, "flip_neg_to_pos": 0.01,
           "selection": {"selection_drop": 0.2, "selection_sharpness": 0.7},
           "mechanism": {"rho_range": [0.05, 0.1], "base_rate_range": [0.1, 0.2]}}
    X = torch.randn(512, 5)
    X2, y, meta = apply_pd_target(PriorRNG(0), X, X[:, 0] + 0.1 * torch.randn(512), cfg, 16)
    assert meta["flip_pos_to_neg"] == pytest.approx(0.1)
    assert meta["selection_drop"] == pytest.approx(0.2) and X2.shape[0] < 512


def test_the_signal_share_makes_defaults_less_predictable():
    """At signal share 1 the Vasicek default is a deterministic threshold on the features
    within a cohort; below 1, part of the borrower's risk is not in the file."""
    from sklearn.metrics import roc_auc_score

    from src.prior.targets.mechanisms import pd_vasicek

    latent = torch.randn(4000)
    aucs = {}
    for share in (1.0, 0.2):
        y, meta = pd_vasicek(PriorRNG(1), latent, {"rho_range": [0.05, 0.05], "base_rate_range": [0.2, 0.2],
                                                    "signal_share_range": [share, share]})
        aucs[share] = roc_auc_score(y.numpy(), -latent.numpy())
        assert meta["signal_share"] == pytest.approx(share)
    assert aucs[1.0] > 0.95 > aucs[0.2] > 0.6


def test_every_experiment_uses_upstreams_group_and_subgroup_sizes():
    """Upstream v2: groups of 4 (`--batch_size_per_gp 4`), subgroup = group, micro-batch = group."""
    from src.utils.config import load

    for exp in ("Exp1", "Exp2", "Exp3"):
        for track in ("PD", "LGD"):
            cfg = load(f"config/{exp}_{track}.yaml", allow_placeholders=True)
            g = cfg["prior"]["grouping"]
            assert g["group_size"] == 4 and g.get("subgroup_size", 4) == 4, f"{exp}_{track}"
            micro = int(cfg["train"]["micro_batch_size"])
            # A micro-batch is one group (or a divisor of one): the trainer refuses anything
            # else, which is how Exp2's `micro_batch_size: 16` would have died at startup.
            assert micro <= g["group_size"] and g["group_size"] % micro == 0, f"{exp}_{track}"
