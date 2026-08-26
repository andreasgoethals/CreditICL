"""How to adapt the model: train from scratch, or start from pretrained weights
and train some or all of it.

All of this is grounded in what TabICL itself does. See docs/CONFIG_REFERENCE.md (`init.strategy`) for
the full write-up; the short version:

* TabICL ships three freeze switches in `_finetune/base.py`
  (`_frozen_submodules` / `_apply_freezing`): `freeze_col`, `freeze_row`,
  `freeze_icl`, matching its three stages. They set `requires_grad = False`
  **and** keep those parts in `eval()` mode, because `model.train()` is recursive
  and would otherwise switch dropout back on inside a frozen block.
* TabICL **v1** stage 3 used `--freeze_col True --freeze_row True` (train the ICL
  stack only) at `lr 2e-6`, constant schedule, grad clip 1.0.
* TabICL **v2** stage 3 uses **no freezing** — full training at `lr 2e-5`,
  cosine schedule, grad clip 1.0, with `--only_load_model True`. So the v2
  authors moved away from freezing.
* There is **no LoRA anywhere in TabICL** (zero matches in the whole repo dump),
  and Tanna 2025 found LoRA unstable on TabPFN. We do not add it.

ONE REFINEMENT OVER A NAIVE FREEZE. Our change is to the *target* distribution,
not the feature distribution. In NanoTabICLv2 the target enters twice:
`y_embed_in` (added before the column blocks) and `y_embed_icl` (added before the
ICL blocks). A naive "freeze the column stage" would freeze `y_embed_in` too and
stop the model adapting to the new target shape in exactly the place it needs to.
So freezing here covers the *blocks* and leaves every target-side parameter —
both y-embeddings, the output norm and the output MLP — trainable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Parameters that stay trainable under every freeze setting: the target
# embeddings and the output head. See the docstring for why.
ALWAYS_TRAINABLE = ("y_embed_in", "y_embed_icl", "out_ln", "out_mlp", "row_ln", "row_cls_tokens")

STRATEGIES = ("scratch", "full", "icl_only", "head_only")


def describe_strategies() -> str:
    return (
        "scratch    train every weight from random init on the mixed prior (default)\n"
        "full       start from pretrained weights, train everything at a low LR\n"
        "           (TabICL v2 stage 3: lr 2e-5, cosine, grad clip 1.0)\n"
        "icl_only   start from pretrained, freeze the col + row blocks, train the\n"
        "           ICL stack, both y-embeddings and the head\n"
        "           (TabICL v1 stage 3: lr 2e-6, constant, grad clip 1.0)\n"
        "head_only  freeze col + row + icl blocks, train only y-embeddings and the\n"
        "           head. Cheapest; a floor, not a serious candidate.\n"
    )


#: The three stacks, under every name they go by. Upstream `TabICL` calls them
#: `col_embedder` / `row_interactor` / `icl_predictor`; the vendored NanoTabICL fallback calls
#: them `col_blocks` / `row_blocks` / `icl_blocks`. Resolved by lookup rather than hard-coded,
#: because hard-coding one set made `icl_only` raise AttributeError on the other — which is
#: exactly the arm Exp2 needs, and it would have failed only after the job had queued.
_STACK_NAMES = {
    "col": ("col_embedder", "col_blocks"),
    "row": ("row_interactor", "row_blocks"),
    "icl": ("icl_predictor", "icl_blocks"),
}


def _stack(model: nn.Module, which: str) -> nn.Module:
    """One of the three stacks, whatever the architecture calls it."""
    for name in _STACK_NAMES[which]:
        found = getattr(model, name, None)
        if found is not None:
            return found
    raise AttributeError(
        f"cannot find the {which!r} stack on {type(model).__name__}. Tried "
        f"{_STACK_NAMES[which]}. Freezing needs to know which submodule is which, so add "
        f"this architecture's name to _STACK_NAMES rather than guessing."
    )


def _frozen_block_lists(model: nn.Module, strategy: str) -> list[nn.Module]:
    """Which block stacks to freeze, mirroring TabICL's `_frozen_submodules`."""
    if strategy in ("scratch", "full"):
        return []
    if strategy == "icl_only":
        return [_stack(model, "col"), _stack(model, "row")]
    if strategy == "head_only":
        return [_stack(model, "col"), _stack(model, "row"), _stack(model, "icl")]
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def apply_freezing(model: nn.Module, strategy: str) -> dict[str, Any]:
    """Set `requires_grad` per the strategy. Returns a trainable-parameter report."""
    for p in model.parameters():
        p.requires_grad = True

    frozen = _frozen_block_lists(model, strategy)
    for sub in frozen:
        for p in sub.parameters():
            p.requires_grad = False

    # Re-enable the target-side parameters even if they sit inside a frozen stack.
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ALWAYS_TRAINABLE):
            p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "strategy": strategy,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": round(trainable / max(total, 1), 4),
        "frozen_stacks": len(frozen),
    }


def set_training_mode(model: nn.Module, training: bool, strategy: str) -> None:
    """Toggle train/eval, then snap frozen stacks back to eval.

    `nn.Module.train()` is recursive, so calling it would re-enable dropout inside
    frozen blocks. TabICL handles this the same way in `_set_training_mode`.
    """
    model.train(training)
    if training:
        for sub in _frozen_block_lists(model, strategy):
            sub.eval()


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """What to hand the optimizer. TabICL filters the same way."""
    return [p for p in model.parameters() if p.requires_grad]


def load_pretrained(model: nn.Module, ckpt_path: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Load pretrained weights into the model built by `src.models.architecture`.

    THE COMPATIBILITY QUESTION IS SETTLED, in the direction that matters. With
    `architecture: tabicl` — every real config — the released TabICLv2 checkpoints load
    **exactly**: 347/347 tensors for the regressor (with `bias_free_ln=True`) and 391/391
    for the classifier. Warm-starting Exp2 works.

    It is settled the other way too: with `architecture: nanotabicl`, **zero** of its 390
    names match the released checkpoint's 347, because the reimplementation renames every
    module (`col_blocks` / `row_blocks` / `icl_blocks` against upstream's `col_embedder` /
    `row_interactor` / `icl_predictor`). That is why Nano is a smoke-test fallback only and
    must never produce a result.

    This function therefore reports exactly what matched and what did not, and
    **raises** when almost nothing matched, rather than quietly training a
    randomly-initialised model and reporting it as fine-tuned. That failure mode
    would be invisible in the loss curve and would silently invalidate the whole
    comparison.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError(f"{ckpt_path}: could not find a state dict in the checkpoint")

    # Strip common wrappers (DDP, torch.compile).
    cleaned = {}
    for k, v in state.items():
        k = k.removeprefix("module.").removeprefix("_orig_mod.")
        cleaned[k] = v

    own = model.state_dict()
    matched, shape_mismatch = {}, []
    for k, v in cleaned.items():
        if k in own:
            if own[k].shape == v.shape:
                matched[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(own[k].shape)))

    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_tensors": len(cleaned),
        "model_tensors": len(own),
        "matched": len(matched),
        "shape_mismatched": len(shape_mismatch),
        "matched_fraction": round(len(matched) / max(len(own), 1), 4),
    }

    if report["matched_fraction"] < 0.5:
        raise RuntimeError(
            "Refusing to continue: only "
            f"{report['matched']}/{len(own)} tensors matched between {ckpt_path} and "
            "the model in memory.\n"
            "With architecture='tabicl' the released checkpoints match exactly (347/347 "
            "regressor, 391/391 classifier), so a near-zero match almost always means the "
            "model was built as 'nanotabicl', whose module names differ throughout.\n"
            "Options: (a) use init.strategy='scratch', which needs no checkpoint; "
            "(b) write an explicit key mapping and verify a forward pass reproduces "
            "the full model's output; (c) pretrain your own base checkpoint with this "
            "same code and fine-tune from that — the cleanest route, since it keeps "
            "architecture identical by construction.\n"
            f"Details: shape mismatches = {shape_mismatch[:5]}"
        )

    missing = model.load_state_dict(matched, strict=False)
    report["missing_keys"] = len(missing.missing_keys)
    report["unexpected_keys"] = len(missing.unexpected_keys)
    if strict and (missing.missing_keys or shape_mismatch):
        raise RuntimeError(f"strict load failed: {report}")
    return report


def recommended_hparams(strategy: str) -> dict[str, float | str]:
    """The LR / schedule / clipping each strategy was actually used with upstream.

    Not defaults that get applied silently — call this to fill a config, so the
    numbers stay visible and overridable.
    """
    if strategy == "scratch":
        # TabICLv2 stage 1.
        return {"lr": 8e-4, "scheduler": "cosine_with_restarts", "gradient_clipping": 10.0}
    if strategy == "full":
        # TabICLv2 stage 3.
        return {"lr": 2e-5, "scheduler": "cosine_with_restarts", "gradient_clipping": 1.0}
    if strategy in ("icl_only", "head_only"):
        # TabICL v1 stage 3.
        return {"lr": 2e-6, "scheduler": "constant", "gradient_clipping": 1.0}
    raise ValueError(f"unknown strategy {strategy!r}")
