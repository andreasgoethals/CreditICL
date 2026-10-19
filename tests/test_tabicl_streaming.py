"""Bounded working memory must never become a smaller attention context."""
import pytest
import torch
from tabicl._model.tabicl import TabICL
from tabicl._model.inference_config import InferenceConfig

from src.eval import tabicl_streaming as stream


def small_model(*, classes=2, grouping="same", target=True, affine=False):
    torch.manual_seed(41)
    return TabICL(max_classes=classes, num_quantiles=7, embed_dim=16, col_num_blocks=2,
                  col_nhead=2, col_num_inds=4, col_affine=affine, col_feature_group=grouping,
                  col_target_aware=target, row_num_blocks=1, row_nhead=2, row_num_cls=2,
                  icl_num_blocks=1, icl_nhead=2).eval()


def inputs(classes=2):
    torch.manual_seed(63)
    X = torch.randn(1, 37, 11)
    y = (torch.arange(23) % classes).float().reshape(1, -1) if classes else torch.randn(1, 23)
    X[:, :, 0] = -100  # full-column sentinel
    X[:, :6, 3] = -100  # a block sentinel is NOT a skipped full column
    return X, y


@pytest.mark.parametrize("classes,grouping,target,affine", [
    (2, "same", True, False), (2, "valid", True, False),
    (2, False, False, False), (0, "same", True, False), (0, False, False, True),
])
def test_streamed_layers_match_upstream_with_all_context_and_boundary_blocks(classes, grouping, target, affine):
    model = small_model(classes=classes, grouping=grouping, target=target, affine=affine)
    X, y = inputs(classes)
    config = InferenceConfig()
    for cfg in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):
        cfg.update(dict(device="cpu", use_amp=False))
    with torch.inference_mode():
        expected = model(X, y, inference_config=config)
        stream.install_streaming(model)
        model._crediticl_stream_bytes = 0
        model._crediticl_work_bytes = 3500  # column and row blocks, crossing the 23-row boundary
        actual = model(X, y, inference_config=config)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("affine", [False, True])
def test_many_class_streaming_averages_radix_hidden_states_before_row_attention(affine):
    # Upstream affine projection requires scalar features, not 3-wide groups.
    model = small_model(classes=3, affine=affine, grouping=False if affine else "same")
    X, y = inputs(11)
    config = InferenceConfig()
    for cfg in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):
        cfg.update(dict(device="cpu", use_amp=False))
    with torch.inference_mode():
        expected = model(X, y, inference_config=config)
        stream.install_streaming(model)
        model._crediticl_stream_bytes = 0
        model._crediticl_work_bytes = 3500
        actual = model(X, y, inference_config=config)
    assert actual.shape[-1] == 11
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("fail_stage", ["context", "rows"])
def test_allocator_retry_reduces_only_work_blocks_not_data(monkeypatch, fail_stage):
    model = small_model()
    X, y = inputs()
    config = InferenceConfig()
    for cfg in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):
        cfg.update(dict(device="cpu", use_amp=False))
    with torch.inference_mode():
        expected = model(X, y, inference_config=config)
    contexts, row_blocks, failed = [], [], False
    real_context, real_rows = stream.column_context, stream.cached_embeddings

    def context(col, values, labels):
        nonlocal failed
        assert values.shape[-2] == X.shape[1] and labels.shape[-1] == y.shape[-1]
        if fail_stage == "context" and not failed:
            failed = True
            raise torch.OutOfMemoryError("test allocation")
        contexts.append(values.shape[1])
        return real_context(col, values, labels)

    def rows(col, values, labels, cache, masks):
        nonlocal failed
        assert values.shape[1] == X.shape[-1]
        if fail_stage == "rows" and not failed:
            failed = True
            raise torch.OutOfMemoryError("test allocation")
        row_blocks.append(values.shape[-2])
        return real_rows(col, values, labels, cache, masks)

    monkeypatch.setattr(stream, "column_context", context)
    monkeypatch.setattr(stream, "cached_embeddings", rows)
    stream.install_streaming(model)
    model._crediticl_stream_bytes = 0
    model._crediticl_work_bytes = 10_000
    with torch.inference_mode():
        actual = model(X, y, inference_config=config)
    assert failed and sum(contexts) == X.shape[-1] and sum(row_blocks) == X.shape[1]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
