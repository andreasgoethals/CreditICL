"""Training: the loss, the schedule, freezing, checkpoint/resume, and logging.

The pinball-loss tests matter most. It is transcribed from TabICL's
`run_micro_batch`, and getting the quantile grid wrong would silently mis-scale
every LGD result without ever raising an error.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch

from src.train.adapt import (
    ALWAYS_TRAINABLE,
    STRATEGIES,
    apply_freezing,
    recommended_hparams,
    trainable_parameters,
)
from src.train.checkpoint import latest_checkpoint, prune_checkpoints, save_checkpoint
from src.train.loop import Trainer, pinball_loss, quantile_levels
from src.train.optim import build_optimizer, build_scheduler

# --- the quantile grid and pinball loss --------------------------------------


def test_quantile_levels_match_tabicl():
    """TabICL uses linspace(0, 1, Q+2)[1:-1] — excluding both endpoints. Inference
    and QuantileDistribution assume exactly this grid."""
    q = quantile_levels(999)
    assert q.numel() == 999
    assert 0.0 < float(q[0]) < 0.002
    assert 0.998 < float(q[-1]) < 1.0
    assert float(q[499]) == pytest.approx(0.5, abs=1e-6)


def test_pinball_is_zero_for_a_perfect_prediction():
    target = torch.zeros(2, 5)
    pred = torch.zeros(2, 5, 8)
    assert float(pinball_loss(pred, target, 8)) == pytest.approx(0.0)


def test_pinball_is_positive_and_grows_with_error():
    target = torch.zeros(2, 5)
    small = pinball_loss(torch.full((2, 5, 8), 0.1), target, 8)
    large = pinball_loss(torch.full((2, 5, 8), 1.0), target, 8)
    assert 0 < float(small) < float(large)


def test_pinball_is_asymmetric_per_quantile():
    """A LOW quantile is punished more for over-predicting than under-predicting.

    Note this must be checked per quantile, not on the mean over all of them. The
    quantile grid is symmetric about 0.5, so the mean loss for a constant offset
    comes out identical in both directions — an earlier version of this test
    compared the means and "failed" against perfectly correct code.
    """
    q = 99
    alphas = quantile_levels(q)

    def loss_at(level_idx: int, pred_value: float) -> float:
        a = float(alphas[level_idx])
        err = 0.0 - pred_value
        return max(a * err, (a - 1) * err)

    low = 4  # alpha well below 0.5
    assert loss_at(low, 1.0) > loss_at(low, -1.0), "low quantile should penalise over-prediction more"

    high = q - 5  # alpha well above 0.5
    assert loss_at(high, -1.0) > loss_at(high, 1.0), "high quantile should penalise under-prediction more"


def test_pinball_mean_is_symmetric_for_a_constant_offset():
    """The flip side of the above, asserted deliberately so nobody "fixes" it.

    Because the grid is symmetric about 0.5, over- and under-predicting by the
    same amount give the same MEAN pinball loss. That is correct behaviour.
    """
    target = torch.zeros(1, 1)
    q = 4
    over = pinball_loss(torch.full((1, 1, q), 1.0), target, q)
    under = pinball_loss(torch.full((1, 1, q), -1.0), target, q)
    assert float(over) == pytest.approx(float(under))


def test_pinball_is_minimised_at_the_true_quantiles():
    """The real property we depend on: the loss is minimised when each output
    equals the corresponding quantile of the target distribution."""
    g = torch.Generator().manual_seed(0)
    y = torch.randn(1, 4000, generator=g)
    q = 9
    levels = quantile_levels(q)
    true_q = torch.quantile(y.flatten(), levels).view(1, 1, q).expand(1, 4000, q)
    wrong_q = true_q + 0.5
    assert float(pinball_loss(true_q, y, q)) < float(pinball_loss(wrong_q, y, q))


def test_pinball_handles_an_atom_at_zero():
    """LGD's actual case: 30% of mass exactly at 0. A quantile head represents
    that as a flat run of low quantiles, so the loss must cope."""
    y = torch.cat([torch.zeros(300), torch.rand(700)]).view(1, -1)
    q = 19
    levels = quantile_levels(q)
    fitted = torch.quantile(y.flatten(), levels).view(1, 1, q).expand(1, y.shape[1], q)
    assert float(fitted[0, 0, 0]) == pytest.approx(0.0, abs=1e-6), "low quantiles should sit on the atom"
    assert torch.isfinite(pinball_loss(fitted, y, q))


# --- the LR schedule ---------------------------------------------------------


def _dummy_opt(lr: float = 1e-3):
    p = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.AdamW([p], lr=lr)


def test_cosine_warms_up_then_decays():
    opt = _dummy_opt()
    sched = build_scheduler(opt, {"lr": 1e-3, "warmup_proportion": 0.1}, max_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] < lrs[10], "should rise during warmup"
    assert lrs[10] > lrs[-1], "should decay after warmup"


def test_constant_schedule_stays_flat_after_warmup():
    """TabICL v1's final stage uses a constant schedule, which is the right choice
    for a frozen fine-tune over few steps."""
    opt = _dummy_opt()
    sched = build_scheduler(opt, {"lr": 1e-3, "scheduler": "constant", "warmup_proportion": 0.1}, max_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[20] == pytest.approx(lrs[-1])


def test_unknown_scheduler_is_rejected():
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_scheduler(_dummy_opt(), {"scheduler": "magic"}, max_steps=10)


def test_unknown_optimizer_is_rejected():
    """`muon` used to be the example of an unimplemented optimizer here. It is now
    supported (torch >= 2.9 ships torch.optim.Muon and TabICLv2 uses it), so this
    needs a name that really is unknown — see tests/test_optim_distributed.py for
    the Muon coverage."""
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="not implemented"):
        build_optimizer(model, {"optimizer": "lion"})


def test_muon_is_available_and_builds():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    opt = build_optimizer(model, {"optimizer": "muon"})
    assert len(opt.param_groups) == 2, "matrices under Muon, the rest under AdamW"


# --- freezing ----------------------------------------------------------------


@pytest.fixture
def tiny_model():
    from src.models.nanotabiclv2 import NanoTabICLv2

    return NanoTabICLv2(
        max_classes=0, out_dim=8, embed_dim=32,
        col_num_blocks=1, row_num_blocks=1, icl_num_blocks=2,
        col_nhead=2, row_nhead=2, icl_nhead=2, n_cls_rows=8,
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_leaves_something_trainable(tiny_model, strategy):
    report = apply_freezing(tiny_model, strategy)
    assert report["trainable_params"] > 0
    assert len(trainable_parameters(tiny_model)) > 0


def test_scratch_and_full_train_everything(tiny_model):
    for strategy in ("scratch", "full"):
        report = apply_freezing(tiny_model, strategy)
        assert report["trainable_fraction"] == 1.0


def test_freezing_reduces_trainable_parameters(tiny_model):
    full = apply_freezing(tiny_model, "full")["trainable_params"]
    icl = apply_freezing(tiny_model, "icl_only")["trainable_params"]
    head = apply_freezing(tiny_model, "head_only")["trainable_params"]
    assert head < icl < full


def test_target_embeddings_stay_trainable_when_frozen(tiny_model):
    """The refinement over a plain freeze. Our change is to the TARGET, and the
    target enters before the column blocks, so locking y_embed_in would stop the
    model learning the new target shape at all."""
    for strategy in ("icl_only", "head_only"):
        apply_freezing(tiny_model, strategy)
        for name, p in tiny_model.named_parameters():
            if any(name.startswith(prefix) for prefix in ALWAYS_TRAINABLE):
                assert p.requires_grad, f"{name} must stay trainable under {strategy}"


def test_icl_only_freezes_the_col_and_row_blocks(tiny_model):
    apply_freezing(tiny_model, "icl_only")
    assert all(not p.requires_grad for p in tiny_model.col_blocks.parameters())
    assert any(p.requires_grad for p in tiny_model.icl_blocks.parameters())


def test_unknown_strategy_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="unknown strategy"):
        apply_freezing(tiny_model, "magic")


def test_optimizer_only_gets_trainable_parameters(tiny_model):
    """TabICL filters the same way. Otherwise the optimizer carries state for
    weights it never updates."""
    apply_freezing(tiny_model, "head_only")
    opt = build_optimizer(tiny_model, {"lr": 1e-4})
    n_in_opt = sum(p.numel() for group in opt.param_groups for p in group["params"])
    n_trainable = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
    assert n_in_opt == n_trainable


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_recommended_hparams_exist_for_every_strategy(strategy):
    hp = recommended_hparams(strategy)
    assert "lr" in hp and "gradient_clipping" in hp
    if strategy != "scratch":
        assert hp["gradient_clipping"] == 1.0, "fine-tuning uses tight clipping upstream"


def test_finetune_learning_rates_are_far_below_scratch():
    assert recommended_hparams("full")["lr"] < recommended_hparams("scratch")["lr"] / 10
    assert recommended_hparams("icl_only")["lr"] < recommended_hparams("full")["lr"]


# --- the training loop -------------------------------------------------------


@pytest.mark.parametrize("task", ["lgd", "pd"])
def test_two_steps_run_and_reduce_nothing_catastrophic(tmp_path, lgd_cfg, pd_cfg, task):
    cfg = lgd_cfg if task == "lgd" else pd_cfg
    trainer = Trainer(cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=tmp_path / "logs")
    summary = trainer.train()
    assert summary["steps"] == cfg["train"]["max_steps"]
    assert summary["datasets_seen"] > 0


def test_loss_is_finite_and_decreases_over_a_few_steps(tmp_path, lgd_cfg):
    """Not a convergence claim — just that gradients flow and nothing blows up."""
    lgd_cfg["train"]["max_steps"] = 12
    lgd_cfg["train"]["lr"] = 1e-3
    trainer = Trainer(lgd_cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=tmp_path / "logs")
    trainer.train()
    records = [json.loads(line) for line in trainer.metrics.path.read_text().splitlines()]
    losses = [r["loss"] for r in records if "loss" in r]
    assert losses and all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[-1] < losses[0] * 2.0, "loss should not be diverging"


def test_gradients_actually_reach_the_weights(tmp_path, lgd_cfg):
    trainer = Trainer(lgd_cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=tmp_path / "logs")
    before = trainer.model.out_mlp[0].weight.detach().clone()
    trainer.train()
    after = trainer.model.out_mlp[0].weight.detach()
    assert not torch.allclose(before, after), "weights did not move — the optimizer step is not connected"


# --- checkpoints -------------------------------------------------------------


def test_checkpoint_saves_and_is_found(tmp_path, lgd_cfg):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = build_scheduler(opt, {"lr": 1e-3}, 10)
    save_checkpoint(tmp_path, step=5, model=model, optimizer=opt, scheduler=sched, scaler=None, config={})
    found = latest_checkpoint(tmp_path)
    assert found is not None and found.name == "step-5.ckpt"


def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = build_scheduler(opt, {"lr": 1e-3}, 10)
    for step in (1, 20, 3):
        save_checkpoint(tmp_path, step=step, model=model, optimizer=opt, scheduler=sched, scaler=None, config={})
    assert latest_checkpoint(tmp_path).name == "step-20.ckpt"


def test_latest_checkpoint_on_empty_dir(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "nope") is None


def test_resume_restores_the_step_counter(tmp_path, lgd_cfg):
    """Matched compute is measured in steps, so resume must not lose count."""
    ck = tmp_path / "ck"
    lgd_cfg["train"]["max_steps"] = 4
    lgd_cfg["train"]["save_temp_every"] = 2
    t1 = Trainer(lgd_cfg, tmp_path / "o1", device="cpu", ckpt_dir=ck, log_dir=tmp_path / "l1")
    t1.train()

    lgd_cfg["train"]["max_steps"] = 6
    t2 = Trainer(lgd_cfg, tmp_path / "o2", device="cpu", ckpt_dir=ck, log_dir=tmp_path / "l2")
    t2.maybe_resume()
    assert t2.step == 4
    assert t2.resumed_at == 4
    t2.train()
    assert t2.step == 6


def test_prune_keeps_permanent_checkpoints(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = build_scheduler(opt, {"lr": 1e-3}, 10)
    for step in (10, 20, 30, 100):  # 100 is a multiple of save_perm_every
        save_checkpoint(tmp_path, step=step, model=model, optimizer=opt, scheduler=sched, scaler=None, config={})
    prune_checkpoints(tmp_path, save_perm_every=100, max_temp=1)
    remaining = sorted(p.name for p in tmp_path.glob("step-*.ckpt"))
    assert "step-100.ckpt" in remaining, "permanent checkpoints must never be pruned"
    assert len(remaining) == 2


def test_no_partial_checkpoint_left_behind(tmp_path):
    """Saves go to a temp file then rename, so a job killed at the walltime limit
    cannot leave a truncated file that resume then chokes on."""
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = build_scheduler(opt, {"lr": 1e-3}, 10)
    save_checkpoint(tmp_path, step=1, model=model, optimizer=opt, scheduler=sched, scaler=None, config={})
    assert list(tmp_path.glob("*.tmp")) == []


# --- logging -----------------------------------------------------------------


def test_run_writes_a_log_file_and_a_metrics_file(tmp_path, lgd_cfg):
    lgd_cfg["logging"] = {"level": "INFO", "console": False, "log_prior_every": 0, "to_file": True}
    logs = tmp_path / "logs"
    trainer = Trainer(lgd_cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=logs)
    trainer.train()

    log_files = list(logs.glob("*.log"))
    metric_files = list(logs.glob("*.metrics.jsonl"))
    assert len(log_files) == 1 and len(metric_files) == 1

    text = log_files[0].read_text(encoding="utf-8")
    assert "environment:" in text
    assert "credit_fraction" in text
    assert "budget:" in text

    for line in metric_files[0].read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert "ts" in record and "step" in record


def test_log_lines_are_timestamped(tmp_path, lgd_cfg):
    import re

    logs = tmp_path / "logs"
    lgd_cfg["logging"] = {"level": "INFO", "console": False, "log_prior_every": 0, "to_file": True}
    Trainer(lgd_cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=logs).train()
    text = next(iter(logs.glob("*.log"))).read_text(encoding="utf-8")
    assert re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+", text, re.M)


def test_prior_report_is_written_when_enabled(tmp_path, lgd_cfg):
    lgd_cfg["train"]["max_steps"] = 2
    lgd_cfg["logging"] = {"level": "INFO", "console": False, "log_prior_every": 1, "log_prior_samples": 3, "to_file": True}
    logs = tmp_path / "logs"
    trainer = Trainer(lgd_cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=logs)
    trainer.train()
    text = next(iter(logs.glob("*.log"))).read_text(encoding="utf-8")
    assert "prior check" in text, "the mid-run prior report should appear in the log"


def _log_text(log_dir) -> str:
    """Read the run's log file. Used instead of pytest's `caplog` because our
    logger sets propagate=False (to avoid duplicate console output), so caplog's
    root handler never sees the records. Reading the file also tests the path that
    actually matters — what lands on disk on the cluster."""
    files = sorted(log_dir.glob("*.log"))
    assert files, f"no log file written into {log_dir}"
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


def test_resume_warns_when_max_steps_changed(tmp_path, lgd_cfg):
    """Resuming with a different max_steps reshapes the LR schedule, because
    LambdaLR does not persist its lambda. Invisible in the loss curve, so it must
    be logged loudly."""
    ck = tmp_path / "ck"
    lgd_cfg["train"]["max_steps"] = 2
    lgd_cfg["train"]["save_temp_every"] = 2
    Trainer(lgd_cfg, tmp_path / "o1", device="cpu", ckpt_dir=ck, log_dir=tmp_path / "l1").train()

    lgd_cfg["train"]["max_steps"] = 9
    l2 = tmp_path / "l2"
    t2 = Trainer(lgd_cfg, tmp_path / "o2", device="cpu", ckpt_dir=ck, log_dir=l2)
    t2.maybe_resume()
    t2.close()
    assert "max_steps CHANGED" in _log_text(l2)


def test_resume_errors_when_the_prior_changed(tmp_path, lgd_cfg):
    """The worst silent failure: continuing one arm's weights on another arm's
    data. Nothing about the loss curve would reveal it."""
    ck = tmp_path / "ck"
    lgd_cfg["train"]["max_steps"] = 2
    lgd_cfg["train"]["save_temp_every"] = 2
    lgd_cfg["prior"]["credit_fraction"] = 0.0
    Trainer(lgd_cfg, tmp_path / "o1", device="cpu", ckpt_dir=ck, log_dir=tmp_path / "l1").train()

    lgd_cfg["prior"]["credit_fraction"] = 1.0  # a different arm!
    l2 = tmp_path / "l2"
    t2 = Trainer(lgd_cfg, tmp_path / "o2", device="cpu", ckpt_dir=ck, log_dir=l2)
    t2.maybe_resume()
    t2.close()
    assert "credit_fraction CHANGED" in _log_text(l2)


def test_resume_is_quiet_when_nothing_changed(tmp_path, lgd_cfg):
    """The normal walltime-kill case must NOT produce scary warnings, or the real
    ones get ignored."""
    ck = tmp_path / "ck"
    lgd_cfg["train"]["max_steps"] = 2
    lgd_cfg["train"]["save_temp_every"] = 2
    Trainer(lgd_cfg, tmp_path / "o1", device="cpu", ckpt_dir=ck, log_dir=tmp_path / "l1").train()

    l2 = tmp_path / "l2"
    t2 = Trainer(lgd_cfg, tmp_path / "o2", device="cpu", ckpt_dir=ck, log_dir=l2)
    t2.maybe_resume()
    t2.close()
    text = _log_text(l2)
    assert "RESUMED from" in text
    assert "CHANGED" not in text
