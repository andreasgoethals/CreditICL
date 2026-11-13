"""Mechanism probes must preserve controls and compare objectives honestly."""
from types import SimpleNamespace

import pytest
import torch

from src.validation.cpt_probe import OBJECTIVES, bar_objective, probe_config, regression_objective


@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("backbone", ["tabicl", "tabpfn3"])
def test_probe_changes_only_diagnostic_scope_and_requested_rate(task, backbone):
    from src.utils.config import expand_with_seeds, load
    cfg = probe_config(backbone, task, 5e-5)
    production = next(r for r in expand_with_seeds(load(f"config/Exp2_{task.upper()}.yaml", allow_placeholders=True))
                      if r["architecture"] == backbone and r["seed"] == 0
                      and r["prior"]["credit_fraction"] == .5 and r["lr_index"] == 0)
    assert cfg["experiment"] == "cpt_mechanism_probe" and "design" not in cfg
    assert cfg["train"]["max_steps"] == 32 and cfg["train"]["schedule_total_steps"] == 1500
    assert cfg["train"]["batch_size"] == production["train"]["batch_size"]
    for key in ("optimizer", "weight_decay", "gradient_clipping", "amp", "warmup_proportion"):
        assert cfg["train"][key] == production["train"][key]
    for key in ("train_frac_range", "n_features_range", "encoding", "credit_fraction"):
        assert cfg["prior"][key] == production["prior"][key]
    assert production["train"]["max_steps"] == 1500 and production["train"]["lr"] != 5e-5


@pytest.mark.parametrize("target", [0., 1., 1e25, 1e69])
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_objectives_have_finite_loss_and_gradients(target, objective):
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    logits = torch.zeros(1, 4, 8, dtype=torch.float64, requires_grad=True)
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 9, dtype=torch.float64))
    loss = bar_objective(logits, torch.full((1, 4), target, dtype=torch.float64), bar, objective)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    if objective != "native":
        assert logits.grad.norm() < 10


@pytest.mark.parametrize("target", [0., 2., 1e25])
def test_native_diagnostic_matches_production_loss_and_gradient(target):
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    from src.train.tabpfn_trainer import REGRESSION_LOSS_WEIGHTS, regression_loss
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 9))
    batch = SimpleNamespace(X_query=None, y_query=torch.full((1, 3), target, dtype=torch.float64),
                            znorm_space_bardist=bar, configs=[[SimpleNamespace(target_transform=None)]])
    a = torch.randn(3, 1, 8, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    x = regression_loss(SimpleNamespace(forward=lambda _: (None, [a], None)), batch, 1, REGRESSION_LOSS_WEIGHTS)
    y = regression_objective(SimpleNamespace(forward=lambda _: (None, [b], None)), batch, 1,
                             REGRESSION_LOSS_WEIGHTS, objective="native")
    x.backward()
    y.backward()
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_huber_does_not_zero_out_large_error_gradient_like_loss_clamping():
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, 9, dtype=torch.float64))
    a = torch.zeros(1, 1, 8, dtype=torch.float64, requires_grad=True)
    target = torch.full((1, 1), 1e25, dtype=torch.float64)
    robust = bar_objective(a, target, bar, "crps_huber") - bar_objective(a, target, bar, "crps")
    robust.backward()
    assert a.grad.norm() > 0 and torch.isfinite(a.grad).all()


def test_invalid_objective_fails_before_forward():
    with pytest.raises(ValueError):
        bar_objective(None, None, None, "clip_whatever")


def test_precision_control_changes_only_amp_and_preserves_production_default():
    native = probe_config("tabpfn3", "lgd", 1e-5, steps=1)
    fp32 = probe_config("tabpfn3", "lgd", 1e-5, steps=1, force_fp32=True)
    assert native["train"]["amp"] is True and fp32["train"]["amp"] is False
    fp32["train"]["amp"] = True
    assert fp32 == native
    with pytest.raises(ValueError, match="TabPFN only"):
        probe_config("tabicl", "lgd", 1e-5, force_fp32=True)


@pytest.mark.parametrize("mode", ["tail", "trajectory"])
def test_precision_flag_rejects_modes_without_a_precision_intervention(mode):
    from src.validation.cpt_probe import main

    with pytest.raises(SystemExit) as error:
        main(["--mode", mode, "--full-precision", "--out", "unused.json"])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["tail", "stress"])
@pytest.mark.parametrize("backbone,task", [("tabicl", "lgd"), ("tabpfn3", "pd")])
def test_tail_intervention_rejects_mislabeled_model_or_task_before_loading_weights(mode, backbone, task):
    from src.validation.cpt_probe import main

    with pytest.raises(SystemExit) as error:
        main(["--mode", mode, "--backbone", backbone, "--task", task, "--out", "unused.json"])
    assert error.value.code == 2


def test_standardization_probe_is_explicit_and_keeps_historical_default():
    native = probe_config("tabpfn3", "lgd", 1e-5, seed=1)
    standardize = probe_config("tabpfn3", "lgd", 1e-5, seed=1, target_policy="standardize")
    assert native["train"]["regression_target_policy"] == "native"
    assert standardize["train"]["regression_target_policy"] == "standardize"
    assert standardize["seed"] == 1
    standardize["train"]["regression_target_policy"] = "native"
    assert standardize == native
    with pytest.raises(ValueError, match="Unknown diagnostic target policy"):
        probe_config("tabpfn3", "lgd", 1e-5, target_policy="clip")
    with pytest.raises(ValueError, match="requires TabPFN regression"):
        probe_config("tabicl", "lgd", 1e-5, target_policy="standardize")


@pytest.mark.parametrize("mode", ["tail", "trajectory"])
@pytest.mark.parametrize("option", [["--target-policy", "standardize"], ["--seed", "1"], ["--batch-id", "269"]])
def test_stress_options_are_not_silently_ignored_by_other_modes(mode, option):
    from src.validation.cpt_probe import main
    with pytest.raises(SystemExit) as error:
        main(["--mode", mode, "--out", "unused.json", *option])
    assert error.value.code == 2


@pytest.mark.parametrize("option", [["--seed", "2"], ["--batch-id", "-1"], ["--target-policy", "clip"]])
def test_invalid_stress_options_fail_before_loading_weights(option):
    from src.validation.cpt_probe import main
    with pytest.raises(SystemExit) as error:
        main(["--mode", "stress", "--out", "unused.json", *option])
    assert error.value.code == 2


def test_parameter_drift_reports_actual_displacement_and_preserves_reference():
    from src.validation.cpt_probe import _relative_parameter_drift
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3., 4.]]))
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    assert _relative_parameter_drift(model, initial) == 0.
    with torch.no_grad():
        model.weight[0, 0] += .5
    assert _relative_parameter_drift(model, initial) == pytest.approx(.1)
    assert initial["weight"][0, 0] == 3.
    with pytest.raises(ValueError, match="names changed"):
        _relative_parameter_drift(model, {})
    with pytest.raises(ValueError, match="shape changed"):
        _relative_parameter_drift(model, {"weight": torch.ones(3)})
    with pytest.raises(ValueError, match="nonzero initial"):
        _relative_parameter_drift(model, {"weight": torch.zeros_like(model.weight)})


def test_stress_metadata_exposes_constant_context_without_modifying_targets():
    from src.validation.cpt_probe import _stress_batch_metadata
    y = torch.tensor([[0., 0., 1., -2.]])
    batch = (torch.arange(8).reshape(1, 4, 2), y, torch.tensor([2]), torch.tensor([4]), torch.tensor([2]))
    prepared = [{"item": SimpleNamespace(y_query=torch.tensor([1e8, -2e8]),
                                         configs=[SimpleNamespace(target_transform=None)])}]
    result = _stress_batch_metadata(batch, prepared)
    case = result["tasks"][0]
    assert case["raw_context_std"] == 0.
    assert case["raw_query_min"] == -2. and case["raw_query_max"] == 1.
    assert case["context_zscore_query_abs_max"] == 2e8
    assert case["target_transforms"] == ["NoneType"]
    assert len(result["batch_sha256"]) == 64
    torch.testing.assert_close(y, torch.tensor([[0., 0., 1., -2.]]))
    changed = list(batch)
    changed[4] = torch.tensor([1])
    assert _stress_batch_metadata(changed, prepared)["batch_sha256"] != result["batch_sha256"]
    assert _stress_batch_metadata(batch, [None])["tasks"][0]["preparation_status"] == "SKIPPED"
    assert _stress_batch_metadata(batch, [{"error": "preparation failed"}])["tasks"][0]["preparation_status"] == "ERROR"
    with pytest.raises(ValueError, match="task count"):
        _stress_batch_metadata(batch, [])


@pytest.mark.parametrize("seed,batch_id,target_policy", [(0, 685, "native"), (1, 269, "standardize"),
                                                        (1, 1491, "standardize"), (0, 1247, "standardize")])
def test_stress_variants_share_seeded_batch_initial_weights_and_report_drift(
        monkeypatch, tmp_path, seed, batch_id, target_policy):
    import src.prior.dataset as dataset_module
    import src.prior.generator as generator_module
    import src.train.tabpfn_trainer as trainer_module
    import src.validation.cpt_probe as probe

    events = []
    raw = (torch.zeros(1, 4, 2), torch.tensor([[0., 0., 1., -2.]]),
           torch.tensor([2]), torch.tensor([4]), torch.tensor([2]))

    class FakeDataset:
        def __init__(self, prior, task, batch_size, *, seed):
            events.append(("dataset", seed, batch_size, prior["credit_fraction"]))

        def batch(self, generator, batch_id):
            events.append(("batch", batch_id))
            return raw

    class FakePreparer:
        def __init__(self, task, model_path, n_estimators, seed, *, target_policy):
            events.append(("preparer", seed, target_policy))

        def __call__(self, batch, batch_id):
            assert batch is raw
            events.append(("prepare", batch_id))
            return [{"item": SimpleNamespace(y_query=torch.tensor([1e8, -2e8]),
                                              configs=[SimpleNamespace(target_transform=None)])}]

    class FakeTrainer:
        def __init__(self, cfg, directory, device):
            assert cfg["seed"] == seed and cfg["train"]["regression_target_policy"] == target_policy
            assert cfg["train"]["warmup_proportion"] == 0.
            self.model = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.model.weight.fill_(1.)
            self.weights_path = "unused-no-checkpoint-opened"
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-5}])
            self.amp = cfg["train"]["amp"]

        def maybe_resume(self):
            pass

        def train_step(self, prepared):
            assert torch.equal(self.model.weight, torch.ones_like(self.model.weight))
            assert prepared[0]["item"].y_query[1] == -2e8
            events.append(("update", trainer_module.regression_loss.keywords["objective"]))
            with torch.no_grad():
                self.model.weight.add_(.125)
            return {"loss": 42.}

        def close(self):
            events.append(("close",))

    def generator(prior, task, rng, *, stream_seed):
        events.append(("generator", stream_seed))
        return object()

    monkeypatch.setattr(dataset_module, "PriorBatchDataset", FakeDataset)
    monkeypatch.setattr(generator_module, "TaskGenerator", generator)
    monkeypatch.setattr(trainer_module, "TabPFNTaskPreparer", FakePreparer)
    monkeypatch.setattr(trainer_module, "TabPFNTrainer", FakeTrainer)
    monkeypatch.setattr(probe, "_validation_batches", lambda *args: [])
    monkeypatch.setattr(probe, "_measure", lambda *args: [])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    report = {}
    probe.run_stress_update(tmp_path, report, lambda: None, seed=seed, batch_id=batch_id,
                            target_policy=target_policy)
    assert report["seed"] == seed and report["batch_id"] == batch_id
    assert report["target_policy"] == target_policy and report["batch_indexing"] == "zero-based"
    assert "not a replay" in report["intervention"]
    assert [v["objective"] for v in report["variants"]] == list(OBJECTIVES)
    assert len({v["batch"]["batch_sha256"] for v in report["variants"]}) == 1
    for variant in report["variants"]:
        assert variant["relative_parameter_drift_before"] == 0.
        assert variant["relative_parameter_drift_after"] == pytest.approx(.125)
    assert events.count(("dataset", seed, 16, 0.)) == 4
    assert events.count(("generator", seed)) == 4
    assert events.count(("preparer", seed, target_policy)) == 4
    assert events.count(("batch", batch_id)) == 4 and events.count(("prepare", batch_id)) == 4
    assert [e[1] for e in events if e[0] == "update"] == list(OBJECTIVES)
    assert events.count(("close",)) == 4


@pytest.mark.parametrize("options,expected", [([], {"force_fp32": False, "target_policy": "native", "seed": 0, "batch_id": 685}),
    (["--target-policy", "standardize", "--seed", "1", "--batch-id", "269", "--full-precision"],
     {"force_fp32": True, "target_policy": "standardize", "seed": 1, "batch_id": 269})])
def test_stress_cli_passes_controls_to_diagnostic_without_loading_weights(monkeypatch, tmp_path, options, expected):
    import json

    import src.utils.preflight as preflight
    import src.validation.cpt_probe as probe
    monkeypatch.setenv("SLURM_JOB_ID", "mock-only-no-vsc")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "mock-only-no-gpu")
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    monkeypatch.setattr(preflight, "source_fingerprint", lambda: "test-source")
    monkeypatch.setattr(preflight, "training_fingerprint", lambda: "test-training")
    received = []
    monkeypatch.setattr(probe, "run_stress_update", lambda directory, report, save, **kwargs: received.append(kwargs))
    output = tmp_path / "diagnostic.json"
    assert probe.main(["--mode", "stress", "--out", str(output), *options]) == 0
    assert received == [expected]
    assert json.loads(output.read_text())["status"] == "PASS"
