"""Full-context TabICL inference without the rows x features x embedding tensor.

Use upstream ISAB key/value caches to retain the inducing-point context computed
from ALL training rows. Recompute row embeddings in bounded blocks, immediately
reduce them with upstream row attention, then give ALL row representations to
upstream ICL. No rows, features, ensemble members or attention contexts are cut.
The arithmetic lives in the pinned upstream layers, not a reimplementation.
"""
from __future__ import annotations

import time
from types import MethodType

import torch
from torch.nn import functional as F
from tabicl._model.inference_config import InferenceConfig
from tabicl._model.kv_cache import KVCache, KVCacheEntry

from src.utils.logging_setup import get_logger

LOG = get_logger()
STREAM_BYTES = 8 * 2**30
WORK_BYTES = 512 * 2**20


def features(col, X):
    # CLS embeddings are overwritten by RowInteraction. Do not compute/cache
    # these padding columns; prepend them immediately before row attention.
    return col.feature_grouping(X).transpose(1, 2) if col.feature_group else X.transpose(1, 2).unsqueeze(-1)


def encoded(col, values, labels):
    src = col.in_linear(values)
    if col.target_aware and labels.shape[-1]:
        y = labels.unsqueeze(1).expand(-1, values.shape[1], -1)
        y_emb = col.y_encoder(y.float()) if col.max_classes > 0 else col.y_encoder(y.unsqueeze(-1))
        src[..., :labels.shape[-1], :] = src[..., :labels.shape[-1], :] + y_emb
    return src


def column_context(col, values, labels):
    """Upstream cache construction, retaining the full-table skip decisions.

    A column constant at the upstream skip sentinel in ONE row block must not
    become a skipped column if the full table was not skipped. Record the masks
    before each layer, instead of recomputing them on individual row blocks.
    """
    src, cache, masks = encoded(col, values, labels), KVCache(), []
    for i, block in enumerate(col.tf_col.blocks):
        masks.append((src == block.skip_value).all(dim=(-2, -1)))
        src = block.forward_with_cache(src, cache, i, train_size=labels.shape[-1], store_cache=True)
    return cache, masks


def cached_embeddings(col, values, labels, cache, masks, *, affine=True):
    src = encoded(col, values, labels)
    for i, block in enumerate(col.tf_col.blocks):
        if i in cache.kv:
            src = block.induced_attention_with_cache(src, cache, i, use_cache=True, store_cache=False)
            src[masks[i]] = block.skip_value
        else:
            if not bool(masks[i].all()):
                raise RuntimeError("Incomplete streaming column context")
            src = torch.full_like(src, block.skip_value)
    if affine and col.affine:
        return values * col.ln_w(col.out_w(src)) + col.ln_b(col.out_b(src))
    return src


def full_context(col, values, y, *, work_bytes):
    """Bound column work while keeping all training rows in each attention cache."""
    n_columns, n_rows = values.shape[1:3]
    width = col.inference_mgr.out_dim
    # Bound temporary activations and keep upstream's adaptive row-attention
    # batching. Any allocator OOM halves the working block, never the dataset.
    columns = max(1, min(n_columns, work_bytes // max(1, n_rows * width * 4)))
    cache, masks = KVCache(), [torch.ones((1, n_columns), dtype=torch.bool, device=values.device)
                              for _ in col.tf_col.blocks]
    start, began, last_log = 0, time.monotonic(), 0.0
    while start < n_columns:
        end = min(n_columns, start + columns)
        try:
            part, part_masks = col.inference_mgr._run_forward(
                column_context, dict(col=col, values=values[:, start:end], labels=y))
        except torch.OutOfMemoryError:
            if columns == 1:
                raise
            columns = max(1, columns // 2)
            torch.cuda.empty_cache()
            LOG.warning("TabICL streaming context retry: column block=%d", columns)
            continue
        for i, entry in part.kv.items():
            if i not in cache.kv:
                cache.kv[i] = KVCacheEntry(
                    key=entry.key.new_zeros((1, n_columns, *entry.key.shape[2:])),
                    value=entry.value.new_zeros((1, n_columns, *entry.value.shape[2:])))
            cache.kv[i].key[:, start:end] = entry.key
            cache.kv[i].value[:, start:end] = entry.value
        for full, partial in zip(masks, part_masks):
            full[:, start:end] = partial
        del part, part_masks
        start = end
        now = time.monotonic()
        if now - last_log >= 30 or start == n_columns:
            LOG.info("TabICL streaming context: columns %d/%d, all %d training rows, %.1fs",
                     start, n_columns, y.shape[-1], now - began)
            last_log = now
    return cache, masks


def streaming_representations(model, X, y, config, *, work_bytes=WORK_BYTES):
    col, row = model.col_embedder, model.row_interactor
    if X.shape[0] != 1 or col.reserve_cls_tokens != row.num_cls:
        raise ValueError("Streaming inference requires one member and matching CLS tokens")
    labels = [y]
    if col.target_aware and col.max_classes > 0 and int(y.max()) >= col.max_classes:
        if not col.mixed_radix_ensemble:
            raise ValueError("Upstream mixed_radix_ensemble must be enabled for many-class tasks")
        bases = col._compute_mixed_radix_bases(int(y.max()) + 1)
        labels = [col._extract_mixed_radix_digit(y, i, bases) for i in range(len(bases))]
    col.inference_mgr.configure(**config.COL_CONFIG)
    values = features(col, X)
    n_columns, n_rows = values.shape[1:3]
    width = col.inference_mgr.out_dim
    contexts = [full_context(col, values, digit, work_bytes=work_bytes) for digit in labels]
    del values
    rows = max(1, min(n_rows, work_bytes // max(1, (n_columns + row.num_cls) * width * 4)))
    representations, start, began, last_log = None, 0, time.monotonic(), 0.0
    while start < n_rows:
        end = min(n_rows, start + rows)

        def block_forward():
            def embed():
                values = features(col, X[:, start:end])
                if len(labels) == 1:
                    cache, masks = contexts[0]
                    return cached_embeddings(col, values, y[:, start:min(end, y.shape[-1])], cache, masks)
                # Match ColEmbedding._compute_embeddings: average hidden states
                # over ALL radix digits before the affine projection or row attention.
                hidden = None
                for digit, (cache, masks) in zip(labels, contexts):
                    item = cached_embeddings(col, values, digit[:, start:min(end, digit.shape[-1])],
                                             cache, masks, affine=False)
                    hidden = item if hidden is None else hidden + item
                hidden = hidden / len(labels)
                if col.affine:
                    return values * col.ln_w(col.out_w(hidden)) + col.ln_b(col.out_b(hidden))
                return hidden
            emb = col.inference_mgr._run_forward(embed, {})
            # The manager's original full buffer has the input dtype (float32),
            # including when column attention computed in AMP. Preserve it.
            emb = F.pad(emb.to(X.dtype).transpose(1, 2), (0, 0, row.num_cls, 0))
            return row(emb, mgr_config=config.ROW_CONFIG)

        try:
            result = block_forward()
        except torch.OutOfMemoryError:
            if rows == 1:
                raise
            rows = max(1, rows // 2)
            torch.cuda.empty_cache()
            LOG.warning("TabICL streaming rows retry: block=%d", rows)
            continue
        if representations is None:
            representations = result.new_empty((1, n_rows, result.shape[-1]))
        representations[:, start:end] = result
        del result
        start = end
        now = time.monotonic()
        if now - last_log >= 30 or start == n_rows:
            LOG.info("TabICL streaming rows: %d/%d, %d features retained, %.1fs",
                     start, n_rows, X.shape[-1], now - began)
            last_log = now
    return representations


def _inference(model, X, y_train, feature_shuffles=None, embed_with_test=False,
               return_logits=True, softmax_temperature=.9, inference_config=None):
    estimate = X.shape[0] * X.shape[1] * (X.shape[2] + model.row_interactor.num_cls) * model.embed_dim * 4
    threshold = model._crediticl_stream_bytes
    # A zero threshold enables small CPU/CUDA equivalence tests of the real path.
    if estimate <= threshold or (X.device.type == "cpu" and threshold != 0):
        return model._crediticl_original_inference(
            X, y_train, feature_shuffles=feature_shuffles, embed_with_test=embed_with_test,
            return_logits=return_logits, softmax_temperature=softmax_temperature, inference_config=inference_config)
    if embed_with_test:
        raise ValueError("Streaming requires the benchmark's embed_with_test=False")
    config = inference_config or InferenceConfig()
    LOG.info("TabICL streaming START: rows=%d context=%d features=%d; avoiding %.2f GiB embedding",
             X.shape[1], y_train.shape[1], X.shape[2], estimate / 2**30)
    began = time.monotonic()
    representations = streaming_representations(model, X, y_train, config,
                                               work_bytes=model._crediticl_work_bytes)
    LOG.info("TabICL row representations ready in %.1fs; full-context ICL START", time.monotonic() - began)
    result = model.icl_predictor(representations, y_train=y_train, return_logits=return_logits,
                                 softmax_temperature=softmax_temperature, mgr_config=config.ICL_CONFIG)
    LOG.info("TabICL streaming COMPLETE in %.1fs", time.monotonic() - began)
    return result


def install_streaming(model):
    if hasattr(model, "_crediticl_original_inference"):
        return
    model._crediticl_original_inference = model._inference_forward
    model._crediticl_stream_bytes = STREAM_BYTES
    model._crediticl_work_bytes = WORK_BYTES
    model._inference_forward = MethodType(_inference, model)
