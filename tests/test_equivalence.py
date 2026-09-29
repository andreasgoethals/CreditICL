"""Our trainer IS upstream's trainer: same batches, same starting weights, bit-identical steps.

`scripts/check_equivalence.py` runs our `Trainer.train_step` beside upstream's `Trainer.run_batch`
(pinned dump, `src/tabicl/train/_run.py`) with upstream's own optimizer and scheduler. On CPU in
float32 the two perform the same operations in the same order, so every loss and every parameter
must agree EXACTLY. When this was first run (28-09-2026) it found three differences at once:
the classification loss sliced to the classes present (upstream's softmax runs over all 10),
the rotary-embedding frequencies were trained (upstream keeps them fixed), and the warm-up length
was rounded down. Each is now fixed; this test keeps it that way.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("tabicl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _upstream_trainer_available() -> bool:
    import check_equivalence

    check_equivalence._stub_optional_imports()
    try:
        from tabicl.train._run import Trainer  # noqa: F401
    except Exception:  # noqa: BLE001 — the published wheel may lack the training package
        return False
    return True


@pytest.mark.parametrize("track", ["PD", "LGD"])
def test_our_training_steps_are_upstreams_bit_for_bit(track):
    if not _upstream_trainer_available():
        pytest.skip("upstream's training package is not importable here")
    import check_equivalence as ce
    import torch

    from src.prior.dataset import PriorBatchDataset
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.train.loop import Trainer
    from src.utils.config import expand_with_seeds, load

    cfg = copy.deepcopy(expand_with_seeds(load(ROOT / f"config/Exp1_{track}.yaml"))[0])
    cfg["prior"]["credit_fraction"] = 0.0
    cfg = ce._tiny(cfg)
    cfg["seed"] = 0
    task = cfg["task"]
    with tempfile.TemporaryDirectory() as tmp:
        ours = Trainer(cfg, Path(tmp) / "o", device="cpu", ckpt_dir=Path(tmp) / "c",
                       log_dir=Path(tmp) / "l", manifest_dir=Path(tmp) / "m")
        up = ce.upstream_trainer(cfg, dict(ours.model.creditcl_model_config),
                                 copy.deepcopy(ours.model.state_dict()), "cpu", task == "lgd")
        ds = PriorBatchDataset(cfg["prior"], task, int(cfg["train"]["batch_size"]), seed=0)
        gen = TaskGenerator(cfg["prior"], task, PriorRNG(0), stream_seed=0)
        key = "pinball" if task == "lgd" else "ce"
        for step in range(4):
            batch = ds.batch(gen, step)
            mine = ours.train_step(batch)
            theirs = up.run_batch([t.clone() for t in batch])
            assert mine["loss"] == pytest.approx(theirs[key], abs=1e-6), step
            assert ours.scheduler.get_last_lr()[0] == pytest.approx(up.scheduler.get_last_lr()[0], rel=1e-9)
            for (name, a), b in zip(ours.model.named_parameters(), up.model.parameters()):
                assert torch.allclose(a.detach(), b.detach(), atol=1e-6), (step, name)
        ours.close()


def test_the_architectures_fixed_parameters_stay_fixed():
    """TabICL builds its RoPE frequencies with `requires_grad=False`; no strategy may train them."""
    from src.models.architecture import build_model
    from src.train.adapt import STRATEGIES, apply_freezing

    model = build_model("pd", embed_dim=32, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1,
                        col_nhead=2, row_nhead=2, icl_nhead=2, col_num_inds=8)
    fixed = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert fixed, "expected TabICL to build at least one fixed parameter"
    for strategy in STRATEGIES:
        apply_freezing(model, strategy)
        assert not any(p.requires_grad for n, p in model.named_parameters() if n in fixed), strategy
