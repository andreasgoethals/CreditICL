"""Which attention kernel PyTorch is allowed to use, and a report saying what happened.

WHY THIS EXISTS. On 20-08-2026 the GPU benchmark ran the model at batch 64 in ONE forward pass
on a B200 under AMP and died inside cuDNN:

    RuntimeError: Expected mha_graph.execute(handle, variant_pack, workspace_ptr.get())
                  .is_good() to be true

That is `torch/csrc/cudnn/MHA.cpp` — the **fused cuDNN multi-head-attention graph**, one of the
four backends `scaled_dot_product_attention` picks between. PyTorch chooses silently, per call,
from the shapes and dtype, so the same model can take a different kernel at a different batch
size and only then fail.

**CORRECTED 24-08-2026: that failure was SHAPE-DEPENDENT, and training never meets that shape.**
Job 11523286 timed all four backends at the shape training actually runs — micro-batch 4,
[4, 1024, 50] — and cuDNN worked: 24.11 ms in bf16. It fails only at the 16x-larger pass the
benchmark was wrongly building. So this was never a threat to the 75-arm sweep, and the earlier
claim that it would have been is withdrawn.

WHY IT STAYS EXCLUDED ANYWAY, on the honest reason rather than the dramatic one:

    flash      bf16 23.01 ms   <- fastest, and what we use
    efficient  bf16 23.23 ms
    cuDNN      bf16 24.11 ms   <- 4.8 % slower, and the one with a known failure mode
    math       bf16 46.56 ms   <- 2x slower; the fallback is real but costly

Excluding cuDNN costs nothing measurable and removes a kernel that has already been seen to
raise at a nearby shape. Upstream treats the choice as load-bearing too: `--use_flash_attn3
False` in stage 1, `True` in stages 2-3.

WHAT WE DO. Pin flash / mem-efficient / math, and **write down which backends were available
and which were excluded**, because a cluster run cannot be watched and a kernel choice that
changes silently is exactly the kind of thing the log has to carry.
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
            "cuDNN fused MHA is 4.8% slower than flash at our shape (24.11 vs 23.01 ms bf16, "
            "job 11523286) and raised `mha_graph.execute(...).is_good()` at a 16x larger pass "
            "on the same card (job 11521108). Not a threat to training, which never builds "
            "that shape - excluded because it buys nothing."
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
