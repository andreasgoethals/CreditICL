"""One architecture for all three experiments, and it is TabICLv2's own.

The test that matters here is the LAST one: our model's parameter names match the released
checkpoint's EXACTLY. That is the difference between Exp2 being possible and not, and it is the
thing that was silently false while the model was vendored from NanoTabICL — a 665-line
reimplementation whose 390 names matched zero of the checkpoint's 347.
"""

from __future__ import annotations

import glob
import os

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.models.architecture import DEFAULT, KNOWN, build_model, describe, is_available

REAL = pytest.mark.skipif(not is_available("tabicl"),
                          reason="upstream tabicl not installed — pip install 'tabicl>=2.0'")


def test_the_default_architecture_is_the_upstream_one():
    assert DEFAULT == "tabicl"
    assert "tabicl" in KNOWN


@REAL
@pytest.mark.parametrize("task", ["lgd", "pd"])
def test_the_model_comes_from_the_upstream_package(task):
    """Not a reimplementation. `describe` records the class so a checkpoint can never be
    silently attributed to the wrong architecture."""
    info = describe(build_model(task))
    assert info["class"].startswith("tabicl."), info["class"]


@REAL
def test_the_regressor_has_the_parameter_count_from_the_paper():
    """28.6M in Table A.1; 28,544,991 exactly, with bias-free layer norms."""
    total = describe(build_model("lgd"))["total_params"]
    assert total == 28_544_991, total


@REAL
def test_the_two_tasks_differ_only_in_the_head_and_the_norms():
    """Upstream's own stage scripts differ in exactly this: `--max_classes 10` versus
    `--regression_method quantile --num_quantiles 999 --norm_type layernorm_nobias`."""
    lgd, pd_ = build_model("lgd"), build_model("pd")
    # The regressor's bias-free norms are the whole 44-tensor difference.
    assert len(pd_.state_dict()) - len(lgd.state_dict()) == 44


@REAL
@pytest.mark.parametrize("task,pattern", [("lgd", "regressor"), ("pd", "classifier")])
def test_our_model_matches_the_released_checkpoint_exactly(task, pattern):
    """THE TEST THIS FILE EXISTS FOR. Exp2 warm-starts from these weights, and a checkpoint only
    loads into the code that saved it. A name mismatch loads NOTHING and raises nothing — the
    model still runs and outputs partly random numbers — so this must be checked by name, not by
    whether `load_state_dict` threw."""
    found = glob.glob(f"checkpoints/*tabicl*{pattern}*.ckpt")
    if not found:
        pytest.skip(f"no released {pattern} checkpoint in checkpoints/")
    obj = torch.load(found[0], map_location="cpu", weights_only=False)
    state = obj
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    theirs = {k for k, v in state.items() if hasattr(v, "shape")}
    ours = set(build_model(task).state_dict())
    assert ours == theirs, (
        f"{os.path.basename(found[0])}: {len(ours & theirs)} of {len(theirs)} names match. "
        f"Only ours: {sorted(ours - theirs)[:5]}. Only theirs: {sorted(theirs - ours)[:5]}"
    )


@REAL
@pytest.mark.parametrize("task,pattern", [("lgd", "regressor"), ("pd", "classifier")])
def test_the_released_checkpoint_loads_strictly(task, pattern):
    """The end-to-end version of the test above: `strict=True` must succeed, because Exp2 sets
    `strict_load: true` precisely so a mismatch is a crash rather than a silent no-op."""
    found = glob.glob(f"checkpoints/*tabicl*{pattern}*.ckpt")
    if not found:
        pytest.skip(f"no released {pattern} checkpoint in checkpoints/")
    obj = torch.load(found[0], map_location="cpu", weights_only=False)
    state = obj
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    model = build_model(task)
    model.load_state_dict({k: v for k, v in state.items() if hasattr(v, "shape")}, strict=True)


def test_test_sized_overrides_do_not_have_to_know_the_architecture():
    """The tiny models in `conftest._shrink` were written against the fallback's parameter
    names. Translating and dropping unknown ones keeps the suite runnable whichever
    architecture is installed — and a real config passes no overrides at all."""
    model = build_model("lgd", architecture="nanotabicl", embed_dim=32, n_cls_rows=8,
                        icl_num_blocks=1, a_knob_that_does_not_exist=123)
    assert model is not None


@REAL
def test_freezing_finds_the_stacks_on_the_upstream_model():
    """`icl_only` is an Exp2 arm. Upstream names the stacks `col_embedder`/`row_interactor`/
    `icl_predictor` and the fallback names them `col_blocks`/`row_blocks`/`icl_blocks`; hard-
    coding either set made this raise AttributeError only after a job had queued."""
    from src.train.adapt import apply_freezing

    report = apply_freezing(build_model("lgd"), "icl_only")
    assert 0 < report["trainable_params"] < report["total_params"]
    assert report["frozen_stacks"] == 2
