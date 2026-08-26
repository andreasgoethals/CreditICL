"""THE model. One architecture, TabICLv2's own, used by all three experiments.

WHY THIS FILE EXISTS AT ALL

Every experiment here trains the *same* network. The whole project is a statement about the
**prior**, so the architecture has to be a constant — and not merely a constant we chose, but
demonstrably the same network TabICLv2 published. Anything else and a difference in results
could always be the architecture rather than the prior.

So the model comes from the **upstream `tabicl` package**, the code that produced the released
checkpoints. Three things follow, and all three matter:

1. **"Same architecture as TabICLv2" becomes a fact**, not a claim about a reimplementation
   that nobody can verify by reading.
2. **Exp2 can warm-start.** A checkpoint only loads into the code that saved it — parameter
   names are how the numbers find their slots.
3. **Their training loop and their Muon come with it**, so ours can be checked against theirs.

WHAT WAS HERE BEFORE, AND WHY IT IS GONE

`nanotabiclv2.py` was vendored from NanoTabICL: a 665-line minimal *reimplementation* by the
same lab, chosen because it is one self-contained file that needs nothing installed. That
convenience cost the project Exp2 outright — **zero of its 390 parameter names matched the
released checkpoint's 347** — and it made architectural identity unverifiable, because you
cannot confirm a 665-line rewrite equals a 4,000-line model by reading it. It is kept only as
an explicitly-selected fallback for environments where `tabicl` is not installed, and it must
never be used for a result.

    architecture: tabicl   # every config says this
"""

from __future__ import annotations

from typing import Any

#: What `architecture:` may be set to in a config.
KNOWN = ("tabicl", "nanotabicl")

#: The one that produces results. Anything else is for a smoke test on a bare machine.
DEFAULT = "tabicl"

_INSTALL_HINT = (
    "The upstream TabICL package is not installed, and it is now a required dependency —\n"
    "it is the architecture every experiment trains and the only code the released\n"
    "checkpoints load into.\n\n"
    "    pip install -e \".[dev,eval]\"\n\n"
    "or, on its own:\n\n"
    "    pip install \"tabicl>=2.0\"\n\n"
    "Set `architecture: nanotabicl` to fall back to the vendored reimplementation, but ONLY\n"
    "for a smoke test: its parameter names do not match the released checkpoints, so Exp2\n"
    "cannot warm-start and no result from it is comparable to TabICLv2."
)


def is_available(name: str = DEFAULT) -> bool:
    """Whether this architecture can be built here, without importing it."""
    import importlib.util

    if name == "nanotabicl":
        return True
    return importlib.util.find_spec("tabicl") is not None


def build_model(task: str, *, architecture: str = DEFAULT, **overrides: Any) -> Any:
    """The model for `task` ("lgd" or "pd"), built from `architecture`.

    The two tasks differ ONLY in the head, exactly as upstream's own stage scripts do:
    classification passes `--max_classes 10`, regression passes `--regression_method quantile
    --num_quantiles 999` and `--norm_type layernorm_nobias`. Everything before the head — the
    column encoder, the row encoder, the ICL stack — is identical, which is why one function
    covers both.

    `overrides` are for tests that need a tiny model. A real run passes none: the architecture
    is TabICLv2's defaults and is never swept.
    """
    if architecture not in KNOWN:
        raise ValueError(f"unknown architecture {architecture!r}; known: {KNOWN}")

    if architecture == "nanotabicl":
        from src.models.nanotabiclv2 import NanoTabICLv2

        extra = _translate(overrides, NanoTabICLv2)
        # `max_classes` and `out_dim` are set from the task below, so they must not also arrive
        # through `extra` — the trainer passes one of them in and it would collide.
        for key in ("max_classes", "out_dim"):
            extra.pop(key, None)
        if task == "lgd":
            n_q = int(overrides.get("num_quantiles", overrides.get("out_dim", 999)))
            return NanoTabICLv2(max_classes=0, out_dim=n_q, **extra)
        n_cls = int(overrides.get("max_classes", 10))
        return NanoTabICLv2(max_classes=n_cls, out_dim=n_cls, **extra)

    if not is_available("tabicl"):
        raise ModuleNotFoundError(_INSTALL_HINT)

    from tabicl._model.tabicl import TabICL

    # Upstream's own defaults, from scripts/train_v2_{clf,reg}_stage1.sh in the pinned dump.
    # Written out rather than relied upon, because a silent upstream default change would move
    # the architecture underneath a running experiment.
    kwargs: dict[str, Any] = dict(
        embed_dim=128,
        col_num_blocks=3, col_nhead=8, col_num_inds=128,
        col_affine=False, col_feature_group="same", col_feature_group_size=3,
        col_target_aware=True, col_ssmax="qassmax-mlp-elementwise",
        row_num_blocks=3, row_nhead=8, row_num_cls=4,
        row_rope_base=100000, row_rope_interleaved=False,
        icl_num_blocks=12, icl_nhead=8, icl_ssmax="qassmax-mlp-elementwise",
        ff_factor=2, norm_first=True,
    )
    if task == "lgd":
        # Regression: `max_classes=0` switches the target embedding to a linear layer and the
        # head to the quantile grid, and `bias_free_ln=True` is upstream's
        # `--norm_type layernorm_nobias` — the flag that accounts for the 44-tensor difference
        # against a model built without it.
        kwargs.update(max_classes=0, num_quantiles=999, bias_free_ln=True)
    else:
        # Classification keeps LayerNorm biases (the default), per Table A.1.
        kwargs.update(max_classes=10, bias_free_ln=False)
    kwargs.update(_translate(overrides, TabICL))
    model = TabICL(**kwargs)
    # RECORDED ON THE MODEL so a checkpoint can carry the exact architecture kwargs.
    # `TabICLRegressor(model_path=...)` / `TabICLClassifier(model_path=...)` read a checkpoint
    # whose `config` is upstream's `model_config_` — the kwargs to `TabICL`, not a project YAML.
    # Storing them here is what lets our checkpoints be loaded by upstream's own wrapper, and
    # that matters for fairness: the released model is scored through that wrapper, so it gets
    # upstream's preprocessing AND its 8-member ensemble while ours got a hand-rolled single
    # pass. Same weights-only comparison requires the same pipeline.
    model.creditcl_model_config = dict(kwargs)
    return model


#: NanoTabICL's names for things upstream calls something else. Only needed because the tiny
#: test/smoke models were written against the fallback; a real config sets none of these.
_ALIASES = {
    "n_cls_rows": "col_num_inds",
    "n_cls_cols": "row_num_cls",
    "feature_group_size": "col_feature_group_size",
    "out_dim": "num_quantiles",
}


def _translate(overrides: dict[str, Any], cls: Any) -> dict[str, Any]:
    """Rename known aliases, then DROP anything the target class does not accept.

    Dropping rather than raising, because these overrides only ever shrink a model for a test:
    a name that does not exist upstream means "this knob is not separately configurable there",
    which is fine — the model is still small enough. Raising would make the whole test suite
    depend on which architecture happens to be installed.

    A real run passes no overrides at all, so nothing here can silently change an experiment.
    """
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        # Alias only when the target does NOT accept the original name — the two architectures
        # each accept one side of every alias pair, so renaming unconditionally would break
        # whichever one already had the right name.
        name = key if key in accepted else _ALIASES.get(key, key)
        if name in accepted:
            out[name] = value
    return out


def describe(model: Any) -> dict[str, Any]:
    """Parameter counts and the class actually used. Logged at the start of every run.

    `class` is the point: it records which architecture produced a checkpoint, so a result can
    never be silently attributed to the wrong one.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "total_params": total,
        "trainable_params": trainable,
        "n_tensors": len(list(model.state_dict())),
    }
