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


def test_torch_ships_muon():
    """We depend on the built-in rather than reimplementing. If a torch upgrade ever
    removed it, the fallback path in `_resolve_muon` should be exercised knowingly,
    not discovered mid-run."""
    assert hasattr(torch.optim, "Muon"), "torch.optim.Muon absent — install .[muon]"


def test_muon_only_takes_matrices():
    """The reason `_MuonWithAux` exists at all. Documented here so nobody 'simplifies'
    the pairing away."""
    m = torch.nn.LayerNorm(8)
    with pytest.raises(ValueError, match="2D"):
        torch.optim.Muon(list(m.parameters()), lr=1e-3)


def test_muon_splits_matrices_from_the_rest(small_model):
    from src.train.optim import _split_muon_params

    matrices, others = _split_muon_params(small_model)
    assert all(p.ndim == 2 for p in matrices), "Muon group must be matrices only"
    assert others, "biases and LayerNorm weights must go to the auxiliary optimizer"
    assert all(p.ndim == 1 for p in others)
    total = sum(p.numel() for p in small_model.parameters())
    assert sum(p.numel() for p in matrices + others) == total, "no parameter dropped"


def test_embeddings_are_not_given_to_muon():
    """An Embedding weight is 2-D but is a lookup table, not a linear map, so
    orthogonalising it is meaningless."""
    from src.train.optim import _split_muon_params

    model = torch.nn.Sequential(torch.nn.Embedding(10, 8), torch.nn.Linear(8, 8))
    matrices, others = _split_muon_params(model)
    emb = model[0].weight
    assert not any(p is emb for p in matrices)
    assert any(p is emb for p in others)


def test_muon_and_adamw_both_build_and_step(small_model):
    for name in ("adamw", "muon"):
        model = copy.deepcopy(small_model)
        opt = build_optimizer(model, {"optimizer": name, "lr": 1e-3, "muon_lr": 8e-4})
        before = model[0].weight.detach().clone()
        model(torch.randn(8, 16)).sum().backward()
        opt.step()
        assert not torch.equal(before, model[0].weight), f"{name} did not update weights"


def test_muon_keeps_a_higher_lr_than_its_aux_half(small_model):
    """TabICLv2 uses 8e-4 for Muon vs 1e-4 for AdamW. Collapsing them to one rate
    would waste the run, so the two groups must keep different rates."""
    opt = build_optimizer(small_model, {"optimizer": "muon", "lr": 3e-4, "muon_lr": 8e-4})
    rates = sorted(g["lr"] for g in opt.param_groups)
    assert rates == [pytest.approx(3e-4), pytest.approx(8e-4)]


def test_scheduler_reaches_both_muon_groups(small_model):
    """`LambdaLR` walks `param_groups`. If the facade did not expose both optimizers'
    groups, the cosine schedule would decay Muon and leave AdamW at its initial rate.
    """
    from src.train.optim import build_scheduler

    opt = build_optimizer(small_model, {"optimizer": "muon", "lr": 3e-4, "muon_lr": 8e-4})
    sched = build_scheduler(opt, {"scheduler": "cosine_with_restarts", "warmup_proportion": 0.0}, 10)
    start = [g["lr"] for g in opt.param_groups]
    for _ in range(9):
        sched.step()
    end = [g["lr"] for g in opt.param_groups]
    assert len(start) == len(end) == 2
    assert all(e < s for s, e in zip(start, end)), "both groups must decay"


def test_muon_state_dict_round_trips(small_model):
    opt = build_optimizer(small_model, {"optimizer": "muon"})
    small_model(torch.randn(4, 16)).sum().backward()
    opt.step()
    state = opt.state_dict()
    fresh = build_optimizer(copy.deepcopy(small_model), {"optimizer": "muon"})
    fresh.load_state_dict(state)  # must not raise


def test_resuming_across_a_changed_optimizer_is_refused(small_model):
    """Silently resetting optimizer state mid-run would corrupt a comparison."""
    muon = build_optimizer(small_model, {"optimizer": "muon"})
    with pytest.raises(ValueError, match="different optimizer"):
        muon.load_state_dict({"kind": "adamw", "state": {}})


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


# -- Muon must resolve on the cluster, not only on a new torch -----------------


def test_muon_resolves_without_torch_or_any_pip_package(monkeypatch):
    """REGRESSION, from the first cluster smoke test. VSC runs torch 2.8 — no
    `torch.optim.Muon` — and the published `tabicl` wheel does not ship the training package,
    so every run died at optimizer construction with ImportError. Upstream's own Muon is now
    vendored from the pinned dump as the last resort."""
    import sys

    import torch

    from src.train import optim

    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    monkeypatch.setitem(sys.modules, "muon", None)
    monkeypatch.setitem(sys.modules, "pytorch_optimizer", None)
    cls = optim._resolve_muon()
    assert cls.__module__.endswith("_muon_vendored"), cls.__module__


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


def test_torch_muon_still_wins_when_it_exists():
    """The vendored copy is a FALLBACK. A maintained implementation should be preferred where
    one is available, so a newer torch on the cluster silently upgrades us."""
    import torch

    from src.train import optim

    if not hasattr(torch.optim, "Muon"):
        pytest.skip("this torch has no built-in Muon")
    assert optim._resolve_muon() is torch.optim.Muon
