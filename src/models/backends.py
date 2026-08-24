"""Which attention kernel PyTorch is allowed to use, and a report saying what happened.

WHY THIS EXISTS. On 20-08-2026 the GPU benchmark ran the real model at the real batch size
(64) with AMP on a B200 and died inside cuDNN:

    RuntimeError: Expected mha_graph.execute(handle, variant_pack, workspace_ptr.get())
                  .is_good() to be true

That is `torch/csrc/cudnn/MHA.cpp` — the **fused cuDNN multi-head-attention graph**, one of the
four backends `scaled_dot_product_attention` picks between. PyTorch chooses silently, per call,
from the shapes and dtype, so the same model can take a different kernel at a different batch
size and only then fail. Nothing in this project had ever pinned the choice.

Upstream evidently treats the choice as load-bearing too: `--use_flash_attn3 False` in stage 1,
`True` in stages 2-3.

WHAT WE DO. Exclude cuDNN, keep flash / mem-efficient / math, and **write down which backends
were available and which were excluded**, because a cluster run cannot be watched and a kernel
choice that changes silently is exactly the kind of thing the log has to carry.
"""

from __future__ import annotations

import contextlib
from typing import Any

#: cuDNN's fused MHA graph. Excluded by default — it is the one that raised on the B200 at
#: batch 64 under AMP, and the alternatives are neither slower nor less accurate here.
EXCLUDED = ("CUDNN_ATTENTION",)

#: Preference order for the rest. Flash is fastest where it applies, mem-efficient is the
#: general fallback, math always works and is the reference implementation.
PREFERRED = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH")


def _backends() -> dict[str, Any]:
    """Every SDPA backend this torch build knows, by name. Empty if torch is too old."""
    try:
        from torch.nn.attention import SDPBackend
    except ImportError:  # pragma: no cover — torch < 2.3
        return {}
    return {n: getattr(SDPBackend, n) for n in (*PREFERRED, *EXCLUDED) if hasattr(SDPBackend, n)}


def sdpa_report() -> dict[str, Any]:
    """What is available, what we will use, and what we are deliberately leaving out.

    Meant to be logged verbatim at the start of every run. `available` is what the build
    shipped; `using` is what the kernels are restricted to; `excluded` is the deliberate part,
    with the reason, so a future reader does not have to guess whether it was a choice.
    """
    have = _backends()
    return {
        "available": sorted(have),
        "using": [n for n in PREFERRED if n in have],
        "excluded": [n for n in EXCLUDED if n in have],
        "why_excluded": (
            "cuDNN fused MHA raised `mha_graph.execute(...).is_good()` on a B200 at batch 64 "
            "under AMP, 20-08-2026 (job 11521108)"
        ),
    }


def sdpa_context(only: str | None = None):
    """Restrict attention to the backends we trust — or to exactly one, for diagnosis.

    A no-op context when torch is too old or nothing matched, so this can wrap a forward pass
    unconditionally: **a diagnostic that can kill a three-day run is worse than no diagnostic.**
    """
    have = _backends()
    if not have:
        return contextlib.nullcontext()
    if only is not None:
        if only not in have:
            return contextlib.nullcontext()
        chosen = [have[only]]
    else:
        chosen = [have[n] for n in PREFERRED if n in have]
    if not chosen:
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel

    return sdpa_kernel(chosen)
