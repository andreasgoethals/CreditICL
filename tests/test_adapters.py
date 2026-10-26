"""Real native layers: identity, frozen weights, packed QKV, export and gradients."""
import copy

import pytest
import torch

from src.models.adapters import QVLowRank, QueryScale, inject, checkpoint_fields, restore_inference


def test_weight_drift_uses_effective_lora_weights_not_factor_norms(tmp_path):
    from src.train.telemetry import WeightTracker
    from src.models.adapters import measured_parameters
    model = tiny_model("tabicl", "pd")
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    inject(model, {"adaptation": "lora"})
    tracker = WeightTracker(tmp_path, every=1)
    tracker.start(model)
    assert set(dict(measured_parameters(model))) == set(initial)
    assert tracker.sample(model, step=0, datasets_seen=0)["drift_all"] == 0.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, QVLowRank):
                module.b.fill_(.01)
    measured = dict(measured_parameters(model))
    expected = sum(float((measured[n] - p).square().sum()) for n, p in initial.items()) ** .5
    assert tracker.sample(model, step=1, datasets_seen=64)["drift_all"] == pytest.approx(expected)


def tiny_model(backbone, task):
    if backbone == "tabicl":
        from src.models.architecture import build_model
        return build_model(task, embed_dim=32, col_num_blocks=1, row_num_blocks=1,
                           icl_num_blocks=1, col_nhead=2, row_nhead=2, icl_nhead=2, col_num_inds=8,
                           **({"num_quantiles": 9} if task == "lgd" else {}))
    from tabpfn.architectures.tabpfn_v3 import TabPFNV3, TabPFNV3Config
    cfg = TabPFNV3Config(max_num_classes=10 if task == "pd" else 0, num_buckets=9, embed_dim=16, dist_embed_num_blocks=1, dist_embed_num_heads=2,
                        dist_embed_num_inducing_points=4, feat_agg_num_blocks=1,
                        feat_agg_num_heads=2, feat_agg_num_cls_tokens=2, nlayers=1, icl_num_heads=2,
                        icl_num_kv_heads=1, decoder_head_dim=8, decoder_num_heads=2)
    model = TabPFNV3(config=cfg, task_type="regression" if task == "lgd" else "multiclass", n_out=9)
    # Native random initialisation zeroes residual output projections. Released
    # checkpoints do not: exercise an active pretrained-like attention path.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("out_projection.weight"):
                parameter.normal_(std=.02)
    return model


def forward(model, backbone, task):
    generator = torch.Generator().manual_seed(71)
    x = torch.randn(1, 16, 4, generator=generator)
    y = (torch.arange(10) % 2).float()[None]
    if task == "lgd":
        y = torch.linspace(-1, 1, 10)[None]
    if backbone == "tabicl":
        return model(x, y)
    from tabpfn.architectures.interface import PerformanceOptions
    return model(x.transpose(0, 1), y.transpose(0, 1),
                 performance_options=PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False))


@pytest.mark.parametrize("backbone", ["tabicl", "tabpfn3"])
@pytest.mark.parametrize("task", ["pd", "lgd"])
@pytest.mark.parametrize("method", ["lora", "diagscale"])
def test_native_adapter_identity_backward_frozen_and_checkpoint(backbone, task, method):
    torch.manual_seed(31)
    model = tiny_model(backbone, task).eval()
    original = copy.deepcopy(model)
    with torch.no_grad():
        expected = forward(model, backbone, task)
    rng = torch.get_rng_state().clone()
    report = inject(model, {"adaptation": method}, seed=7)
    assert torch.equal(rng, torch.get_rng_state())
    assert 0 < report["trainable"] < report["total"]
    with torch.no_grad():
        torch.testing.assert_close(forward(model, backbone, task), expected, atol=1e-6, rtol=1e-5)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimized = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optimized, lr=.01)
    model.train()
    forward(model, backbone, task).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in optimized)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name]), name
    model.eval()
    payload = checkpoint_fields(model)
    restored = copy.deepcopy(original)
    restored.load_state_dict(payload["state_dict"], strict=True)
    restore_inference(restored, payload)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(forward(restored, backbone, task), forward(model, backbone, task),
                                   atol=2e-6, rtol=2e-5)
    resumed = copy.deepcopy(original)
    inject(resumed, {"adaptation": method}, seed=7)
    resumed.load_state_dict(payload["crediticl_training_state"], strict=True)
    for a, b in zip(resumed.state_dict().values(), model.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_packed_lora_never_changes_keys():
    adapter = QVLowRank(36, 12, 4, 4, packed=True)
    with torch.no_grad():
        adapter.b.fill_(.3)
    weights = torch.randn(36, 12)
    changed = adapter(weights)
    assert torch.equal(changed[12:24], weights[12:24])
    assert not torch.equal(changed[:12], weights[:12])
    assert not torch.equal(changed[24:], weights[24:])


@pytest.mark.parametrize("layout,shape", [("BHSD", (2, 3, 7, 4)), ("BSHD", (2, 7, 3, 4))])
def test_diagscale_is_after_native_query_scaling(layout, shape):
    class Native(torch.nn.Module):
        def forward(self, q, n):
            return q.square() * n
    module = QueryScale(Native(), 3, 4, layout=layout)
    with torch.no_grad():
        module.delta.fill_(.2)
    q = torch.randn(shape)
    torch.testing.assert_close(module(q, 17), q.square() * 17 * 1.2)
