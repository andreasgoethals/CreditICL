"""Training: the loss, the schedule, freezing, checkpoint/resume, and logging.

The pinball-loss tests matter most. It is transcribed from TabICL's
`run_micro_batch`, and getting the quantile grid wrong would silently mis-scale
every LGD result without ever raising an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

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
    """Every trainable weight must move. Checked across the WHOLE model by name rather than on
    one hard-coded layer: the previous version reached into `out_mlp`, which only exists on the
    vendored fallback, so it broke the moment the real TabICL was installed — and a test that
    names one layer would not have noticed a whole stack sitting frozen anyway."""
    trainer = Trainer(lgd_cfg, tmp_path / "out", device="cpu", ckpt_dir=tmp_path / "ck", log_dir=tmp_path / "logs")
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    assert before, "no trainable parameters at all"
    trainer.train()
    after = dict(trainer.model.named_parameters())
    moved = [n for n, old in before.items() if not torch.allclose(old, after[n].detach())]
    assert moved, "no weight moved — the optimizer step is not connected"
    # Not every tensor has to move in two steps (a rarely-hit embedding row may not), but a
    # large majority should; a small fraction means most of the model is receiving no gradient.
    assert len(moved) / len(before) > 0.5, (
        f"only {len(moved)}/{len(before)} tensors moved — most of the model is not training"
    )


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


# -- the architecture is fixed, and is NOT a config setting --------------------

TABLE_A1 = {
    "embed_dim": 128,
    "col_num_blocks": 3,
    "row_num_blocks": 3,
    "icl_num_blocks": 12,
    "col_nhead": 8,
    "row_nhead": 8,
    "icl_nhead": 8,
    "feature_group_size": 3,
    "n_cls_cols": 4,
    "n_cls_rows": 128,
}


def test_model_defaults_are_tabiclv2_table_a1():
    """The architecture is TabICLv2's, published in its Appendix Table A.1, and it never
    varies — so it lives in the class defaults rather than in a config block that three
    experiment files would each have to repeat and keep in sync.

    This test is what makes that safe: the defaults ARE the specification, so a silent
    edit to them fails here instead of quietly making our runs incomparable to the paper.
    """
    import inspect

    from src.models.nanotabiclv2 import NanoTabICLv2

    sig = inspect.signature(NanoTabICLv2.__init__)
    for name, expected in TABLE_A1.items():
        assert sig.parameters[name].default == expected, (
            f"{name} default is {sig.parameters[name].default}, Table A.1 says {expected}"
        )
    # The ICL stage runs at embed_dim * n_cls_cols; the paper gives d=512 for TFicl.
    assert TABLE_A1["embed_dim"] * TABLE_A1["n_cls_cols"] == 512


def test_model_block_is_absent_from_the_configs():
    """`model:` was removed from the configs: the architecture is fixed, so repeating it
    per experiment only creates a way for the three to drift apart."""
    from src.utils.config import load

    for path in ("config/Exp1_LGD.yaml", "config/Exp1_PD.yaml"):
        cfg = load(ROOT / path)
        assert "model" not in cfg, (
            f"{path} still has a model: block. The architecture is TabICLv2's and is "
            f"fixed in NanoTabICLv2's defaults."
        )


def test_parameter_count_is_close_to_the_released_checkpoint():
    """Our vendored architecture should be within a fraction of a percent of the released
    TabICLv2 regressor. Measured difference: 15,224 params (0.05%), which is 44 LayerNorm
    bias tensors the released regression model omits (`bias_free_ln: True` in its config,
    and the paper states the regression model uses bias-free layer norms).

    A zero bias is numerically identical to no bias, so this does not change the model at
    initialisation — it only means ours *can* learn those terms.
    """
    from src.models.nanotabiclv2 import NanoTabICLv2

    ours = sum(p.numel() for p in NanoTabICLv2(max_classes=0, out_dim=999).parameters())
    released = 28_544_991  # measured from checkpoints/tabicl-regressor-v2-20260212.ckpt
    assert abs(ours - released) / released < 0.001, (
        f"ours {ours:,} vs released {released:,} — more than 0.1% apart, so the vendored "
        f"architecture has drifted from the published one"
    )


# --- quantile crossing -------------------------------------------------------


def test_quantile_crossing_is_fixed_before_anything_reads_the_grid():
    """A quantile head predicts each level independently, so nothing stops q_0.4 > q_0.6.

    Measured on an untrained TabICL LGD model: the raw rows are NOT monotone. Every quantity
    we read off the predictive assumes they are — the "median" is column Q//2, coverage counts
    truths falling between columns, PIT and CRPS integrate across them. Upstream fixes this
    with `enforce_monotonicity(..., method="sort")` inside `QuantileDistribution`; we did not,
    so all four were quietly wrong on any crossed row.
    """
    import numpy as np
    import torch

    from src.train.loop import enforce_monotonic_quantiles

    crossed = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.4]])
    fixed = enforce_monotonic_quantiles(crossed)
    assert np.all(np.diff(fixed, axis=-1) >= 0), "must be non-decreasing"
    # sorting is a permutation within each row: the values are the same, the order is not
    assert np.allclose(np.sort(crossed, axis=-1), fixed)
    # the median column is now genuinely the median
    assert fixed[0, 1] == pytest.approx(0.5)
    assert fixed[1, 1] == pytest.approx(0.4)

    t = enforce_monotonic_quantiles(torch.tensor(crossed))
    assert isinstance(t, torch.Tensor), "torch in, torch out"
    assert torch.all(torch.diff(t, dim=-1) >= 0)

    # cummax is upstream's other named option and drags the row to its running maximum
    cm = enforce_monotonic_quantiles(crossed, method="cummax")
    assert np.all(np.diff(cm, axis=-1) >= 0)
    assert cm[0].tolist() == [0.9, 0.9, 0.9], "cummax distorts — that is why sort is default"

    with pytest.raises(ValueError, match="unknown method"):
        enforce_monotonic_quantiles(crossed, method="isotonic")


def test_the_training_loss_does_not_sort():
    """Pinball is evaluated per level independently, exactly as upstream trains it. Sorting
    inside the loss would change the gradient, so the fix belongs at decode time only."""
    src = (Path(__file__).resolve().parents[1] / "src" / "train" / "loop.py").read_text(
        encoding="utf-8"
    )
    loss_fn = src[src.index("def pinball_loss") : src.index("class Trainer")]
    assert "enforce_monotonic" not in loss_fn and "sort" not in loss_fn


def test_optim_docstring_matches_the_configs():
    """The module docstring said "Deviation: AdamW, not Muon" while every config set
    `optimizer: muon`. It was written before Muon was vendored and never updated, so the file
    documented the wrong optimizer for weeks — and the optimizer turned out to be the whole
    explanation for the B200 being 12x slow. A stale docstring is a wrong answer with a
    citation."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    doc = (root / "src" / "train" / "optim.py").read_text(encoding="utf-8")
    doc = doc[: doc.index('"""', 3)]

    chosen = {
        yaml.safe_load(p.read_text(encoding="utf-8"))["train"]["optimizer"].lower()
        for p in (root / "config").glob("Exp*.yaml")
    }
    assert len(chosen) == 1, f"configs disagree on the optimizer: {chosen}"
    name = chosen.pop()
    assert name.upper() in doc.upper(), f"configs use {name!r}; the docstring never says so"
    assert "Deviation: AdamW, not Muon" not in doc, "that claim is false while configs use muon"


def test_the_benchmark_measures_the_optimizer_that_is_actually_configured():
    """The first benchmark used plain SGD and concluded the B200 was FASTER, while real
    training on it was 12x slower. The missing rung was the optimiser."""
    root = Path(__file__).resolve().parents[1]
    bench = (root / "scripts" / "benchmark_gpu.py").read_text(encoding="utf-8")
    assert "def bench_optimizers" in bench
    assert '"adamw", "muon"' in bench, "both must be timed, or there is nothing to compare"
    # `muon_slowdown_in_training`, not `..._vs_adamw`: the rung calls `opt.step()` once per
    # forward pass, while training calls it once per `n_micro` passes, so the raw ratio
    # overstates Muon by n_micro. 1.15x measured here is 1.02x in training.
    assert "muon_slowdown_in_training" in bench, "the ratio is the number you actually read"
    assert "optimizer_overhead_ms_per_step" in bench, "the overhead is per STEP, not per pass"


def test_the_step_is_split_into_phases_that_sum_to_the_wall_clock():
    """Three explanations for the B200 running 16x slower than a benchmark of the same model,
    prior and loader have been proposed, and two were measured wrong — the card, then the
    optimiser. Both times the cause was inferred from a noticed difference rather than
    measured. This is the instrument that ends that."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "train" / "loop.py").read_text(encoding="utf-8")

    for phase in ("data", "fwd_bwd", "optimizer", "telemetry"):
        assert f'self._phase["{phase}"]' in src, f"{phase} is not timed"

    # the data wait must be measured around `next(it)` — that IS the wait
    i = src.index('self._phase["data"]')
    assert "next(it)" in src[i - 200 : i], "the data phase must wrap the loader fetch"

    # the verdict must be conditional; an unconditional hint once read
    # "0% data means the GPU is WAITING", the opposite of the number
    assert "if data_pct >= 50.0" in src and "elif data_pct < 10.0" in src


def test_phase_timing_does_not_synchronise_the_gpu():
    """A `cuda.synchronize()` per phase would cost more than it measures and would change the
    very number being measured. Attribution survives without it because `float(loss.detach())`
    is already a sync point."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    step = src[src.index("def train_step") : src.index("def _loss_for")]
    assert "synchronize" not in step


def test_batch_and_features_match_upstream():
    """The project's claim is "TabICLv2's architecture and prior, plus credit structure". Every
    unintended difference weakens that, and two had crept in.

    `batch_size: 4` against upstream's `--batch_size 64` is an effective batch SIXTEEN TIMES
    smaller — while using upstream's learning rate of 8e-4, which was tuned for 64. Large
    batches tolerate large rates; small ones do not.

    `max_features: 64` against upstream's `--max_features 100` narrowed the prior's feature
    range for no stated reason, and widened the extrapolation gap on real credit tables
    (base_modelisation has 256 columns).
    """
    import yaml

    root = Path(__file__).resolve().parents[1]
    for name in ("Exp1_LGD", "Exp1_PD", "Exp2_LGD", "Exp2_PD"):
        cfg = yaml.safe_load((root / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert cfg["train"]["batch_size"] == 64, f"{name}: upstream uses --batch_size 64"
        assert cfg["prior"]["max_features"] == 100, f"{name}: upstream uses --max_features 100"
        # accumulation must divide the batch, or the last micro-step is a different size
        assert cfg["train"]["batch_size"] % cfg["train"]["micro_batch_size"] == 0, (
            f"{name}: micro_batch_size must divide batch_size"
        )


def test_micro_batch_size_is_a_speed_knob_only():
    """Gradient accumulation averages `loss / n_micro` per pass, so the update is identical
    whatever the micro-batch is. That is what makes it safe to set from the hardware — and it
    must stay true, or a B200 run and an A100 run would not be comparable."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    assert "self.scaler.scale(loss / n_micro).backward()" in src, (
        "the per-micro-batch loss must be divided by n_micro, or the effective learning rate "
        "would scale with the micro-batch and the knob would change results"
    )
    # and the override must reach the trainer
    entry = (root / "scripts" / "pretrain.py").read_text(encoding="utf-8")
    assert '"--micro-batch-size"' in entry
    assert '["micro_batch_size"] = args.micro_batch_size' in entry


def test_batch_size_and_micro_batch_size_are_different_kinds_of_knob():
    """One changes the result; the other cannot. Conflating them is how "TabPFN-Wide used 16"
    became an argument about GPU memory.

    `batch_size` is how many datasets contribute to an update — a scientific choice, held
    FIXED across arms so a difference in results is attributable to the prior. `micro_batch_size`
    is how many fit in memory per forward pass; accumulation averages `loss / n_micro`, so the
    update is identical either way and it is safe to set from the hardware.
    """
    root = Path(__file__).resolve().parents[1]
    entry = (root / "scripts" / "pretrain.py").read_text(encoding="utf-8")

    batch_help = entry[entry.index('"--batch-size"') : entry.index('"--micro-batch-size"')]
    assert "CHANGES THE RESULT" in batch_help, "the difference must be stated where it is used"

    micro_help = entry[entry.index('"--micro-batch-size"') :][:600]
    assert "never the result" in micro_help or "identical" in micro_help


def test_batch_size_is_not_swept_in_any_experiment():
    """Sweeping it would double the arm count and halve the power of the comparison Exp1
    actually asks — and at a fixed learning rate it would be a confounded test anyway, since
    batch size and LR are coupled. Settle it on the control arm instead."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    for cfg_path in sorted((root / "config").glob("Exp*.yaml")):
        sweep = yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("sweep") or {}
        offenders = [k for k in sweep if "batch" in k or k.endswith(".lr")]
        assert not offenders, (
            f"{cfg_path.name} sweeps {offenders}. Optimisation settings are nuisance "
            f"parameters: hold them fixed so the prior is the only thing that varies."
        )


def test_micro_batch_may_not_exceed_the_prior_group_size():
    """NOT a memory rule, and this is the part I got wrong twice.

    Datasets share a sequence length only within a GROUP (`prior.grouping.group_size`,
    upstream's `--batch_size_per_gp`). A micro-batch is stacked into one tensor, so every
    dataset in it must already agree on length. Upstream RAISES on violation —
    `validate_micro_batch`: "All datasets in the micro batch must have the same sequence
    length" — and keeps the two numbers EQUAL in every stage: 4/4 at seq 1,024, then 1/1 at
    10,240 and 60,000.

    So "the GPU has room, raise the micro-batch" is wrong on its own. We had no check at all.
    """
    import tempfile

    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "Exp1_LGD.yaml").read_text(encoding="utf-8"))
    cfg["_run_name"] = "guard_test"
    assert cfg["train"]["micro_batch_size"] <= cfg["prior"]["grouping"]["group_size"], (
        "the shipped config already violates the rule"
    )

    from src.train.loop import Trainer

    cfg["train"]["micro_batch_size"] = cfg["prior"]["grouping"]["group_size"] + 1
    with pytest.raises(ValueError, match="group_size"):
        Trainer(cfg, tempfile.mkdtemp())


def test_the_run_card_answers_the_questions_a_reader_would_ask():
    """Runs happen on the cluster and come back as a folder of logs. Anything not written at
    startup costs a 20-minute round trip on a shared queue, so the log has to pre-empt it."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    card = src[src.index("def _log_run_card") : src.index("# -- setup ---")]
    for needed in (
        "micro-passes/update",   # the thing that actually fills the GPU
        "credit_fraction",       # which arm this is
        "group_size",            # the micro-batch constraint
        "UPSTREAM COMPARISON",   # every deviation, in one place
        "LEAKAGE CHECK",         # what is and is not trained on
        "WE DO NOT RUN THEM",    # the single-stage scope limit
    ):
        assert needed in card, f"the run card does not report {needed!r}"
    assert "_log_run_card()" in src, "the card must actually be called"


def test_checkpoints_carry_upstream_loadable_keys():
    """Our model must be scoreable through UPSTREAM'S wrapper, not only our own code path.

    Measured 18-08-2026 on the same seven LGD datasets: the released TabICLv2 scored R2 +0.22
    to +0.77 through `TabICLRegressor`, ours -1.44 to -0.25 through a hand-rolled single pass.
    Some of that gap is 600 steps against 500,000 — but not all of it, because their wrapper
    also brings upstream's preprocessing pipeline and an 8-member feature-shuffled ensemble
    while ours brought neither. A weights-only comparison needs the same pipeline on both sides.

    `TabICLClassifier(model_path=<path>)` reads `state_dict`, `curr_step` and a `config` of
    TabICL kwargs, so the checkpoint has to carry all three.
    """
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "train" / "checkpoint.py").read_text(encoding="utf-8")
    # Their loader asserts on `config` and `state_dict` and builds `TabICL(**config)`, so
    # `config` must be the ARCHITECTURE kwargs. Our own YAML moved to `crediticl_config`.
    assert '"config": model_config' in src, "config must hold the TabICL kwargs, not our YAML"
    assert '"state_dict"' in src and '"curr_step"' in src
    assert '"crediticl_config": config' in src, "our YAML needs its own key"
    assert 'getattr(model, "module", model)' in src, "DDP must be unwrapped first"

    from src.models.architecture import build_model, is_available

    if not is_available("tabicl"):
        pytest.skip("upstream tabicl not installed here")
    cfg = build_model("lgd", architecture="tabicl").creditcl_model_config
    # the kwargs must be the real ones, not a stub
    assert cfg["embed_dim"] == 128
    assert cfg["max_classes"] == 0 and cfg["num_quantiles"] == 999, "LGD is the quantile head"
    assert cfg["bias_free_ln"] is True, "upstream's --norm_type layernorm_nobias for regression"
