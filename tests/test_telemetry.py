"""Hardware and gradient telemetry.

The point of these is that telemetry must be **impossible to make fatal**. A diagnostic that
can end a three-day cluster run is worse than no diagnostic, so every probe is tested for the
degraded case — no CUDA, no `nvidia-smi`, no `psutil`, an unwritable directory — and required
to return something rather than raise.

The other half is the timing of the gradient probe: it has to read gradients between
`backward()` and `zero_grad()`. Called one line too late, every norm reads 0.0, which looks
exactly like a model that is not learning.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")

import torch
import torch.nn as nn

from src.train.telemetry import (
    Telemetry,
    _classify,
    _describe_environment,
    gpu_stats,
    grad_norms,
    host_stats,
    torch_memory,
    weight_norms,
)


class _Model(nn.Module):
    """Named to hit each reported block, so `_classify` is exercised end to end."""

    def __init__(self) -> None:
        super().__init__()
        self.x_embed = nn.Linear(4, 8)      # -> col
        self.tf_row = nn.Linear(8, 8)       # -> row
        self.icl_block = nn.Linear(8, 8)    # -> icl
        self.head = nn.Linear(8, 1)         # -> head

    def forward(self, x):
        return self.head(self.icl_block(self.tf_row(self.x_embed(x))))


def _stepped_model() -> _Model:
    """A model with real gradients on every parameter."""
    model = _Model()
    loss = model(torch.randn(16, 4)).square().mean()
    loss.backward()
    return model


# -- nothing may raise --------------------------------------------------------


@pytest.mark.parametrize("probe", [gpu_stats, host_stats, torch_memory, _describe_environment])
def test_every_probe_returns_a_dict_even_when_the_tool_is_missing(probe):
    """On this dev machine there is no CUDA and `nvidia-smi` may be absent. Each probe must
    degrade to an empty dict, never to an exception in the middle of a training step."""
    assert isinstance(probe(), dict)


def test_probes_do_not_raise_when_the_binary_is_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert gpu_stats() == {}


def test_nvidia_smi_failure_is_swallowed(monkeypatch):
    """A timeout or non-zero exit must not end the run."""
    import subprocess

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr("subprocess.run", boom)
    assert gpu_stats() == {}


def test_nvidia_smi_output_is_parsed_per_gpu(monkeypatch):
    """Two GPUs must give two prefixed sets, and `[N/A]` must be skipped rather than crash."""

    monkeypatch.setattr("shutil.which", lambda _: "nvidia-smi")

    class Done:
        stdout = "97, 42, 12000, 40960, 61, [N/A], 1410\n88, 40, 11000, 40960, 59, 250.5, 1400\n"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: Done())
    stats = gpu_stats()
    assert stats["gpu0_utilization_gpu"] == 97.0
    assert stats["gpu1_utilization_gpu"] == 88.0
    assert stats["gpu0_memory_total"] == 40960.0
    assert "gpu0_power_draw" not in stats, "[N/A] must be skipped, not parsed"
    assert stats["gpu1_power_draw"] == 250.5


def test_an_unwritable_output_directory_does_not_raise(tmp_path):
    """The CSV is a convenience. Losing it must not lose the run."""
    t = Telemetry("run", tmp_path / "nested", hardware_every=1)
    t.record({"step": 1, "a": 1.0})
    assert t.path.is_file()
    # Point it somewhere impossible and check `record` still returns.
    t.path = tmp_path / "\0bad" / "x.csv"
    t.record({"step": 2, "a": 2.0})


# -- gradient and weight norms -----------------------------------------------


def test_grad_norms_separate_the_architecture_blocks():
    row = grad_norms(_stepped_model())
    for block in ("col", "row", "icl", "head"):
        assert row[f"grad_{block}"] > 0, f"{block} received no gradient"
    assert row["grad_global"] > 0


def test_grad_norms_are_zero_before_backward_which_is_why_timing_matters():
    """Documents the failure this probe is easy to get wrong: read the gradients after
    `zero_grad(set_to_none=True)` and every block reports nothing at all."""
    model = _Model()
    assert grad_norms(model) == {"grad_global": 0.0}, "no grads yet -> only the global zero"

    model = _stepped_model()
    assert grad_norms(model)["grad_global"] > 0

    model.zero_grad(set_to_none=True)
    assert grad_norms(model) == {"grad_global": 0.0}, (
        "after zero_grad the probe sees nothing — it must run before the optimizer step"
    )


def test_the_global_norm_is_the_quadrature_sum_of_the_blocks():
    """If this drifts, the per-block split is measuring something other than the whole."""
    row = grad_norms(_stepped_model())
    blocks = [v for k, v in row.items() if k.startswith("grad_") and k != "grad_global"]
    combined = sum(v ** 2 for v in blocks) ** 0.5
    assert combined == pytest.approx(row["grad_global"], rel=1e-5)


def test_weight_norms_cover_the_same_blocks():
    grads, weights = grad_norms(_stepped_model()), weight_norms(_Model())
    assert {k.replace("grad_", "") for k in grads if k.startswith("grad_") and k != "grad_global"} \
        <= {k.replace("weight_", "") for k in weights}


def test_gradient_to_weight_ratio_is_reported(tmp_path):
    """The gradient alone is not interpretable — 0.01 is tiny against weights of 0.1 and huge
    against 1e-5. The ratio is the number worth watching."""
    t = Telemetry("run", tmp_path, grad_every=1)
    row = t.sample_grads(_stepped_model())
    for block in ("col", "row", "icl", "head"):
        assert row[f"gw_ratio_{block}"] > 0
        assert row[f"gw_ratio_{block}"] == pytest.approx(
            row[f"grad_{block}"] / row[f"weight_{block}"], rel=1e-6
        )


def test_a_frozen_block_is_visible():
    """The failure this exists to catch: one stack receiving no signal while the loss curve
    looks unremarkable."""
    model = _Model()
    for p in model.icl_block.parameters():
        p.requires_grad_(False)
    model(torch.randn(8, 4)).square().mean().backward()
    row = grad_norms(model)
    assert "grad_icl" not in row, "a frozen block must not report a gradient"
    assert row["grad_col"] > 0


@pytest.mark.parametrize(
    "name,expected",
    [
        ("col_embedder.tf_col.blocks.0.weight", "col"),
        ("x_embed.weight", "col"),
        ("tf_row.0.attn.weight", "row"),
        ("icl_blocks.3.mlp.weight", "icl"),
        ("head.out_proj.weight", "head"),
        ("y_embed_in.weight", "head"),
        ("some_final_norm.weight", "other"),
    ],
)
def test_parameter_names_map_to_the_right_block(name, expected):
    assert _classify(name) == expected


# -- cadence and the CSV ------------------------------------------------------


def test_zero_disables_each_cadence_independently(tmp_path):
    off = Telemetry("r", tmp_path, hardware_every=0, grad_every=0)
    assert not off.enabled and not off.due_hardware(100) and not off.due_grad(250)

    hw_only = Telemetry("r", tmp_path, hardware_every=10, grad_every=0)
    assert hw_only.enabled
    assert hw_only.due_hardware(10) and not hw_only.due_grad(10)


def test_the_csv_is_rewritten_with_the_union_of_columns(tmp_path):
    """Hardware rows and gradient rows carry different keys. A ragged CSV misaligns silently
    in pandas, which is the worst way to lose a diagnostic."""
    import csv

    t = Telemetry("r", tmp_path, hardware_every=1, grad_every=1)
    t.record({"step": 1, "kind": "hardware", "gpu0_utilization_gpu": 90.0})
    t.record({"step": 1, "kind": "grad", "grad_global": 0.5})
    rows = list(csv.DictReader(t.path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert set(rows[0]) == {"step", "kind", "gpu0_utilization_gpu", "grad_global"}
    assert rows[0]["grad_global"] == "" and rows[1]["grad_global"] == "0.5"


def test_summary_warns_when_the_run_is_starved(tmp_path):
    """A compute-bound run and a data-starved run look identical from the outside — same
    wall-clock, same loss curve — and the fix is opposite. This warning is the whole point of
    sampling utilisation at all."""
    t = Telemetry("r", tmp_path, hardware_every=1)
    for step in range(5):
        t.record({"step": step, "gpu0_utilization_gpu": 22.0, "steps_per_s": 0.5})
    text = t.summary()
    assert "STARVED" in text
    assert "num_workers" in text, "the warning must say what to change"


def test_summary_stays_quiet_when_the_gpu_is_busy(tmp_path):
    t = Telemetry("r", tmp_path, hardware_every=1)
    for step in range(5):
        t.record({"step": step, "gpu0_utilization_gpu": 95.0, "steps_per_s": 2.0})
    assert "STARVED" not in t.summary()


def test_summary_of_nothing_is_readable(tmp_path):
    assert "nothing sampled" in Telemetry("r", tmp_path).summary()


def test_environment_records_what_makes_a_run_reproducible(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345678")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    env = _describe_environment()
    assert env["slurm_job_id"] == "12345678"
    assert env["slurm_array_task_id"] == "7", "the array index identifies WHICH arm ran"
    assert "torch" in env and "python" in env and "hostname" in env


# -- it is actually wired into the loop --------------------------------------


def test_the_trainer_reads_the_cadences_from_the_logging_block():
    """They live under `logging:` rather than `train:` because switching them off must not
    change a result. A test, because putting them under `train:` would be an easy mistake and
    would make two runs incomparable for a reason nobody would look for."""
    import inspect

    from src.train import loop

    source = inspect.getsource(loop)
    assert 'log_hardware_every' in source and 'log_grad_every' in source
    assert 'cfg.get("logging", {})' in source
    # and the gradient probe must sit before the optimizer step
    before = source.index("self.telemetry.due_grad")
    after = source.index("self.scaler.step(self.optimizer)")
    assert before < after, "gradients must be sampled BEFORE the optimizer step"


def test_every_experiment_config_asks_for_telemetry():
    """Logging that is off by default is logging nobody has. Every run should produce these."""
    from src.utils.config import load

    for exp in ("Exp1", "Exp2", "Exp3"):
        for track in ("LGD", "PD"):
            cfg = load(f"config/{exp}_{track}.yaml", allow_placeholders=True)
            log_cfg = cfg["logging"]
            assert log_cfg["log_hardware_every"] > 0, f"{exp}_{track}"
            assert log_cfg["log_grad_every"] > 0, f"{exp}_{track}"
