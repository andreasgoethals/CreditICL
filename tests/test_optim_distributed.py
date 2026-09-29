"""Muon, and the multi-GPU plumbing.

Both exist to match TabICLv2: Muon is the optimizer it uses, and DDP is what makes
its 24.5-GPU-day budget fit inside VSC's 72-hour job ceiling.

The distributed tests deliberately do NOT spawn processes. What matters is the
*arithmetic* — batch splitting and per-rank seeding — because getting either wrong
produces a run that looks perfectly healthy in the logs while silently training on
1/N the data, or on N times the compute budget. Those are checked directly.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.train import distributed as dist
from src.train.optim import build_optimizer


@pytest.fixture
def small_model():
    return torch.nn.Sequential(
        torch.nn.Linear(16, 16),
        torch.nn.LayerNorm(16),
        torch.nn.Linear(16, 4),
    )


# -- Muon --------------------------------------------------------------------


def _muon(model, **cfg):
    return build_optimizer(model, {"optimizer": "muon", "lr": 8e-4, **cfg})


def test_muon_is_built_exactly_as_upstream_builds_it(small_model):
    """Upstream's `Trainer.configure_optimizer`: ONE group, EVERY parameter, `use_muon=True`,
    momentum = beta1, Moonlight RMS matching 0.2, Nesterov, 5 Newton-Schulz steps. Until
    28-09-2026 we used torch's Muon on the matrices only (Keller scaling, momentum 0.95) and an
    AdamW at 3e-4 on the rest - steps 4.8x smaller on the weights than TabICLv2's."""
    from src.train._muon_vendored import Muon

    opt = _muon(small_model, beta1=0.9, beta2=0.95, weight_decay=0.01)
    assert isinstance(opt, Muon)
    assert len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert g["use_muon"] is True, "without the flag the vendored class silently runs AdamW"
    assert len(g["params"]) == len(list(small_model.parameters())), "biases and norms too"
    assert g["lr"] == pytest.approx(8e-4)
    assert g["momentum"] == pytest.approx(0.9)
    assert g["matched_adamw_rms"] == pytest.approx(0.2)
    assert g["nesterov"] is True and g["ns_steps"] == 5
    assert g["adamw_betas"] == (0.9, 0.95)
    assert g["use_cautious_wd"] is False, "the released checkpoints ran without it"


def test_a_muon_step_is_the_orthogonalised_update_with_moonlight_scaling():
    """Recompute one step by hand with upstream's own functions and compare: the change in a
    2-D weight AND in a 1-D bias must be `-lr * 0.2 * sqrt(max(A, B)) * NS5(g + m*g)` after
    decoupled weight decay. This is the test that would have caught the 4.8x."""
    import math

    from src.train._muon_vendored import zeropower_via_newtonschulz5

    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8)
    lr, wd, m = 8e-4, 0.01, 0.9
    opt = _muon(model, lr=lr, weight_decay=wd, beta1=m)
    model(torch.randn(32, 16)).pow(2).sum().backward()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    for name, p in model.named_parameters():
        g = grads[name]
        ns_in = (g + m * g).reshape(len(g), -1)          # first step: buffer == g, Nesterov
        update = zeropower_via_newtonschulz5(ns_in, steps=5).view(g.shape)
        scale = 0.2 * math.sqrt(max(ns_in.shape))
        expected = before[name] * (1 - lr * wd) - lr * scale * update
        assert torch.allclose(p.detach(), expected, atol=1e-7), name


def test_the_old_second_rate_is_refused(small_model):
    with pytest.raises(ValueError, match="muon_lr no longer exists"):
        build_optimizer(small_model, {"optimizer": "muon", "lr": 8e-4, "muon_lr": 8e-4})


def test_muon_and_adamw_both_build_and_step(small_model):
    for name in ("adamw", "muon"):
        model = copy.deepcopy(small_model)
        opt = build_optimizer(model, {"optimizer": name, "lr": 1e-3})
        before = model[0].weight.detach().clone()
        model(torch.randn(8, 16)).sum().backward()
        opt.step()
        assert not torch.equal(before, model[0].weight), f"{name} did not update weights"


def test_the_scheduler_decays_the_single_muon_rate(small_model):
    from src.train.optim import build_scheduler

    opt = _muon(small_model)
    sched = build_scheduler(opt, {"scheduler": "cosine_with_restarts", "warmup_proportion": 0.0,
                                  "lr": 8e-4}, 10)
    start = opt.param_groups[0]["lr"]
    for _ in range(9):
        sched.step()
    assert opt.param_groups[0]["lr"] < start


def test_muon_state_dict_round_trips(small_model):
    opt = _muon(small_model)
    small_model(torch.randn(4, 16)).sum().backward()
    opt.step()
    state = opt.state_dict()
    fresh = _muon(copy.deepcopy(small_model))
    fresh.load_state_dict(state)  # must not raise


def test_resuming_across_a_changed_optimizer_is_refused(tmp_path, small_model):
    """Silently resetting optimizer state mid-run would corrupt a comparison. Checked at the
    checkpoint layer, which records the optimizer's class."""
    from torch.optim.lr_scheduler import LambdaLR

    from src.train.checkpoint import load_checkpoint, save_checkpoint

    adamw = build_optimizer(small_model, {"optimizer": "adamw", "lr": 1e-3})
    path = save_checkpoint(tmp_path, step=1, model=small_model, optimizer=adamw,
                           scheduler=LambdaLR(adamw, lambda s: 1.0), scaler=None, config={})
    muon = _muon(small_model)
    with pytest.raises(ValueError, match="different optimizer"):
        load_checkpoint(path, model=small_model, optimizer=muon)


def test_unknown_optimizer_is_rejected(small_model):
    with pytest.raises(ValueError, match="not implemented"):
        build_optimizer(small_model, {"optimizer": "lion"})


def test_frozen_model_gives_a_clear_error():
    model = torch.nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(model, {"optimizer": "adamw"})


# -- distributed arithmetic --------------------------------------------------


def test_single_process_is_the_default(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info = dist.detect()
    assert not info.enabled and info.is_main and info.world_size == 1


def test_torchrun_env_is_detected(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    info = dist.detect()
    assert info.enabled and info.world_size == 4 and info.rank == 2
    assert not info.is_main, "only rank 0 may write logs and checkpoints"


def test_world_size_one_is_treated_as_single_process(monkeypatch):
    """torchrun --nproc_per_node=1 sets the vars but there is nothing to coordinate."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not dist.detect().enabled


def test_batch_is_split_so_the_effective_budget_is_unchanged():
    """THE compute-budget invariant. If each rank took the full batch, a 4-GPU run
    would burn 4x the datasets and 'matched compute' would be false."""
    info = dist.DistInfo(rank=0, local_rank=0, world_size=4)
    assert dist.local_batch_size(64, info) == 16
    assert dist.local_batch_size(64, info) * info.world_size == 64


def test_indivisible_batch_is_refused_not_rounded():
    """Rounding would silently change the budget in one direction or the other."""
    info = dist.DistInfo(world_size=4)
    with pytest.raises(ValueError, match="not divisible"):
        dist.local_batch_size(10, info)


def test_single_process_batch_is_untouched():
    assert dist.local_batch_size(4, dist.DistInfo()) == 4


def test_each_rank_gets_a_different_prior_seed():
    """Without this every GPU generates identical datasets, so an N-GPU run sees the
    same batch N times at 1/N the real diversity — and every log line still looks
    correct. This is the quietest possible way to ruin a run.
    """
    seeds = {dist.rank_seed(7, dist.DistInfo(rank=r, world_size=4)) for r in range(4)}
    assert len(seeds) == 4, "ranks must not share a prior seed"


def test_rank_seed_is_stable_for_a_single_process():
    """A single-GPU run must keep the exact seed it always had, so existing results
    stay reproducible after this feature was added."""
    assert dist.rank_seed(7, dist.DistInfo()) == 7


def test_unwrap_returns_the_inner_model(small_model):
    """A DDP state_dict prefixes every key with `module.` and will not load into a
    plain model at evaluation time."""
    assert dist.unwrap(small_model) is small_model

    class FakeDDP:
        def __init__(self, m):
            self.module = m

    assert dist.unwrap(FakeDDP(small_model)) is small_model


def test_reduce_mean_is_identity_without_distribution():
    assert dist.reduce_mean(1.5, dist.DistInfo(), "cpu") == 1.5


def test_wrap_model_is_a_noop_without_distribution(small_model):
    assert dist.wrap_model(small_model, dist.DistInfo()) is small_model


def test_setup_picks_a_device_without_distribution():
    assert dist.setup(dist.DistInfo()) in ("cpu", "cuda")


def test_describe_reports_what_a_log_needs():
    d = dist.DistInfo(rank=1, local_rank=1, world_size=2).describe()
    assert d == {"distributed": True, "rank": 1, "local_rank": 1, "world_size": 2}


# -- the training loop honours all of it -------------------------------------


def test_trainer_uses_the_split_batch_and_rank_seed(lgd_cfg, tmp_path, monkeypatch):
    """End to end: the values the loop actually passes to the loader."""
    from src.train.loop import Trainer

    cfg = copy.deepcopy(lgd_cfg)
    cfg["train"]["batch_size"] = 4
    trainer = Trainer(cfg, tmp_path / "o", device="cpu", ckpt_dir=tmp_path / "c", log_dir=tmp_path / "l")
    assert trainer.local_batch_size == 4
    assert trainer.prior_seed == trainer.seed
    assert not trainer.dist.enabled


def test_trainer_trains_with_muon(lgd_cfg, tmp_path):
    from src.train.loop import Trainer

    cfg = copy.deepcopy(lgd_cfg)
    cfg["train"]["optimizer"] = "muon"
    trainer = Trainer(cfg, tmp_path / "o", device="cpu", ckpt_dir=tmp_path / "c", log_dir=tmp_path / "l")
    summary = trainer.train()
    assert summary["steps"] == cfg["train"]["max_steps"]


def test_walltime_probe_is_silent_off_slurm(monkeypatch):
    """The overrun warning must never be able to crash or hang a run."""
    from src.train.loop import _slurm_seconds_left

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert _slurm_seconds_left() == 0.0


def test_hms_formats_readably():
    from src.train.loop import _hms

    assert _hms(0) == "0s"
    assert _hms(45) == "45s"
    assert _hms(605) == "10m 05s"
    assert _hms(11_072) == "3h 04m"
    assert _hms(-5) == "0s", "a negative ETA must not print nonsense"


# -- the vendored optimizer --------------------------------------------------


def test_the_vendored_muon_is_upstreams_and_says_so():
    """Vendored VERBATIM, so it is the optimizer that produced the released checkpoints rather
    than a second implementation that ought to agree. A hand-written Muon would be subtly wrong
    in a way that degrades every arm equally and invisibly."""
    import pathlib

    from src.train import _muon_vendored

    text = pathlib.Path(_muon_vendored.__file__).read_text(encoding="utf-8")
    assert "DO NOT EDIT" in text
    assert "tfm-library" in text, "the provenance must be recorded in the file"
    # The details that are easy to get wrong, and which we therefore did not write.
    assert "zeropower_via_newtonschulz5" in text
    assert "adjust_lr_wd_for_muon" in text


def test_torch_muon_is_never_used_even_when_it_exists():
    """REVERSED 28-09-2026. The old rule - "prefer torch's maintained Muon" - is how Exp1 ended
    up with torch's defaults (Keller step scaling, momentum 0.95) instead of upstream's
    (Moonlight scaling, momentum 0.9): 4.8x smaller weight updates. The vendored class, built as
    upstream builds it, is the only Muon."""
    from src.train._muon_vendored import Muon

    model = torch.nn.Linear(4, 4)
    opt = build_optimizer(model, {"optimizer": "muon", "lr": 8e-4})
    assert type(opt) is Muon
    if hasattr(torch.optim, "Muon"):
        assert not isinstance(opt, torch.optim.Muon)
