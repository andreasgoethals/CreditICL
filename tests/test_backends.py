"""The attention backend is pinned, because PyTorch picking it silently cost a run."""

from __future__ import annotations

from src.models.backends import EXCLUDED, PREFERRED, sdpa_context, sdpa_report


def test_cudnn_is_excluded_and_the_reason_is_recorded():
    """cuDNN's fused MHA graph raised on a B200 at batch 64 under AMP (job 11521108).

    The exclusion matters less than the RECORD of it: six months from now the question is
    whether the missing backend was a decision or an accident, and only the log can say.
    """
    rep = sdpa_report()
    assert "CUDNN_ATTENTION" in EXCLUDED
    assert "CUDNN_ATTENTION" not in rep["using"]
    assert rep["excluded"] == ["CUDNN_ATTENTION"]
    assert "11521108" in rep["why_excluded"], "the reason must name the run that proved it"
    assert rep["using"], "excluding everything would leave no kernel at all"


def test_math_is_always_kept_so_there_is_always_a_viable_kernel():
    """`No viable backend for scaled_dot_product_attention` is what happens otherwise."""
    assert PREFERRED[-1] == "MATH"
    assert "MATH" in sdpa_report()["using"]


def test_the_context_never_raises_however_it_is_called():
    """It wraps every forward pass in training. A diagnostic that can kill a three-day run is
    worse than no diagnostic, so an unknown name degrades to a no-op rather than an error."""
    for arg in (None, "MATH", "CUDNN_ATTENTION", "NOT_A_BACKEND"):
        with sdpa_context(only=arg):
            pass


def test_training_pins_the_kernel_around_its_forward_pass():
    """The wrap is the whole point; an import with no call site would pass every other test."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    after = src.split("with sdpa_context():", 1)
    assert len(after) == 2, "training never enters the pinned context"
    assert "self._loss_for(" in after[1][:200], "the forward pass must be INSIDE the context"
