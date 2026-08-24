"""L2-SP — the regulariser that stops continued pre-training forgetting what it started from.

    Omega(w) = (alpha / 2) * || w - w0 ||^2

Introduced for transfer learning by Li et al. 2018; used for continued pre-training of
TabPFNv2 by Real-TabPFN (Garg et al. 2025) at alpha = 0.003, which is the value Exp3 sweeps
against 0.0. Exp3 is the only experiment that uses it: Exp1 and Exp2 train from scratch, where
there is no starting point to be pulled toward.

Two things here are easy to get wrong and neither shows up in a loss curve:

* applying the penalty inside the micro-batch loop, which multiplies alpha by `n_micro`;
* applying it to AMP-scaled gradients, which multiplies alpha by the loss scaler.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from src.train.loop import Trainer

ROOT = Path(__file__).resolve().parents[1]


def _trainer(cfg, tmp_path, **train_overrides):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train", {}).update(train_overrides)
    return Trainer(
        cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=tmp_path / "logs"
    )


def test_l2sp_needs_a_starting_point(lgd_cfg, tmp_path):
    """`scratch` + L2-SP is a contradiction: a random init is not a starting point worth
    staying near. Silently ignoring it would make an Exp3 arm look like it ran with the
    regulariser when it did not."""
    with pytest.raises(ValueError, match="STARTING POINT"):
        _trainer(lgd_cfg, tmp_path, l2sp_alpha=0.003)


def test_a_negative_alpha_is_rejected(lgd_cfg, tmp_path):
    """A negative alpha would push the weights AWAY from where they started."""
    with pytest.raises(ValueError, match="must be >= 0"):
        _trainer(lgd_cfg, tmp_path, l2sp_alpha=-1.0)


def test_off_by_default_and_costs_nothing(lgd_cfg, tmp_path):
    t = _trainer(lgd_cfg, tmp_path)
    assert t.l2sp_alpha == 0.0
    assert t._l2sp_ref == {}, "no reference copy should be held when the penalty is off"
    assert t._apply_l2sp() == 0.0


def test_the_gradient_added_is_exactly_alpha_times_the_drift(lgd_cfg, tmp_path, monkeypatch):
    """d/dw of (alpha/2)||w - w0||^2 is alpha * (w - w0). Written straight onto `.grad`, so
    this checks the arithmetic rather than trusting autograd on a term we never build."""
    # Patch the name AS `loop` sees it: `from .adapt import load_pretrained` binds a second
    # reference, so patching `adapt.load_pretrained` would leave the trainer's copy intact.
    from src.train import loop as loop_mod

    monkeypatch.setattr(loop_mod, "load_pretrained", lambda *a, **k: {"strategy": "full"})
    cfg = copy.deepcopy(lgd_cfg)
    cfg["init"] = {"strategy": "full", "pretrained_path": str(tmp_path / "fake.ckpt")}
    alpha = 0.003
    t = _trainer(cfg, tmp_path, l2sp_alpha=alpha)

    assert t._l2sp_ref, "the reference must be captured for a warm start"
    # Move every weight by a known amount and give it a zero gradient to add onto.
    for name, param in t.model.named_parameters():
        if name in t._l2sp_ref:
            with torch.no_grad():
                param.add_(0.5)
            param.grad = torch.zeros_like(param)

    penalty = t._apply_l2sp()

    n = 0
    for name, param in t.model.named_parameters():
        if name not in t._l2sp_ref:
            continue
        expected = alpha * (param.detach() - t._l2sp_ref[name])
        assert torch.allclose(param.grad, expected, atol=1e-9)
        n += param.numel()
    # Omega = (alpha/2) * sum(0.5^2) over every element that moved.
    assert penalty == pytest.approx(0.5 * alpha * n * 0.25, rel=1e-5)


def test_the_penalty_is_applied_once_per_step_not_once_per_pass():
    """A parameter penalty is not per-example. Inside the micro-batch loop it would be added
    `n_micro` times and the effective alpha would depend silently on the micro-batch size —
    16x on Exp1's settings."""
    src = (ROOT / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    body = src[src.index("n_micro = math.ceil("):src.index("self._phase[\"fwd_bwd\"] +=")]
    assert "_apply_l2sp" not in body, "L2-SP must not be called inside the micro-batch loop"
    assert "self._apply_l2sp()" in src


def test_the_penalty_lands_on_unscaled_gradients_and_before_the_clip():
    """After `unscale_`, because an unscaled penalty on AMP-scaled gradients would make the
    effective alpha depend on the loss scaler's current guess. Before `clip_grad_norm_`,
    because L2-SP is part of the objective, so the norm being clipped is the whole gradient."""
    src = (ROOT / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    unscale = src.index("self.scaler.unscale_(self.optimizer)")
    apply_ = src.index("self._apply_l2sp()")
    clip = src.index("torch.nn.utils.clip_grad_norm_")
    assert unscale < apply_ < clip


def test_exp3_sweeps_it_and_the_others_do_not():
    """It is a continued-pretraining tool. Exp1 and Exp2 start from random weights."""
    import yaml

    for path in sorted((ROOT / "config").glob("Exp*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        swept = (raw.get("sweep") or {}).get("train.l2sp_alpha")
        body = (raw.get("train") or {}).get("l2sp_alpha")
        if path.name.startswith("Exp3"):
            assert swept == [0.0, 0.003], f"{path.name}: expected off vs Real-TabPFN's value"
            assert body is None, "one knob, one home"
        else:
            assert swept is None and not body, f"{path.name} must not use L2-SP"
