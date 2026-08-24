"""The GPU benchmark exists to answer one question, so it must not lie about the answer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAST = {"gpu": "Fast", "capability": "8.9", "torch": "2.11.0+cu128",
        "compiled_for": ["sm_89"], "has_kernels_for_this_card": True}
SLOW = {"gpu": "Slow", "capability": "10.0", "torch": "2.11.0+cu128",
        "compiled_for": ["sm_89"], "has_kernels_for_this_card": False}


def _write(tmp_path: Path, name: str, info: dict, matmul: float, e2e: float,
           ceiling: float) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({
        "device_info": info,
        "matmul": {"matmul_4096_float32_tflops": matmul},
        "model": {"max_steps_per_s_if_data_were_free": ceiling},
        "prior": {"datasets_per_s_one_worker": 8.7},
        "end_to_end": {"steps_per_s": e2e},
    }), encoding="utf-8")
    return p


def _run(*paths: Path) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "src.utils.compare_gpubench", *map(str, paths)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_a_slow_card_is_named_and_the_missing_kernels_flagged(tmp_path):
    """The B200 case: raw compute already differs, so it is the hardware."""
    a = _write(tmp_path, "a.json", FAST, matmul=40.0, e2e=6.9, ceiling=16.7)
    b = _write(tmp_path, "b.json", SLOW, matmul=3.1, e2e=0.51, ceiling=1.27)
    out = _run(a, b)
    assert "12.9x apart" in out and "Fast ahead" in out
    assert "NO KERNELS FOR THIS CARD" in out, "a JIT fallback is the first thing to suspect"


def test_matching_compute_with_a_slow_end_to_end_points_at_the_prior(tmp_path):
    """The opposite diagnosis, which needs the opposite fix: more cores, not a bigger GPU."""
    a = _write(tmp_path, "a.json", FAST, matmul=40.0, e2e=6.9, ceiling=16.7)
    b = _write(tmp_path, "b.json", dict(FAST, gpu="Same"), matmul=40.0, e2e=0.5, ceiling=16.7)
    out = _run(a, b)
    assert "NO KERNELS" not in out
    lines = [ln for ln in out.splitlines() if "x apart" in ln]
    assert len(lines) == 1, f"only the end-to-end row should differ, got: {lines}"
    assert "END TO END" in out


def test_one_file_still_prints_its_numbers(tmp_path):
    """A comparison needs two, but one column answers "did the benchmark even work?" —
    which is the next question when a job has just finished."""
    a = _write(tmp_path, "a.json", FAST, matmul=40.0, e2e=6.9, ceiling=16.7)
    r = subprocess.run(
        [sys.executable, "-m", "src.utils.compare_gpubench", str(a)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode == 2, "one file cannot answer a comparison question"
    assert "Only ONE result" in r.stderr
    assert "40.0" in r.stdout, "the single card's numbers must still be shown"


def test_an_unexpanded_glob_says_the_files_do_not_exist_yet(tmp_path):
    """The shell passes a pattern through literally when nothing matches, so the argument
    count is 1. Reporting that as "need at least two JSON files" sent the reader looking for
    a second card when in fact NEITHER had finished."""
    r = subprocess.run(
        [sys.executable, "-m", "src.utils.compare_gpubench",
         str(tmp_path / "gpubench_*.json")],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode == 2
    assert "No benchmark results matched" in r.stderr
    assert "still queued or running" in r.stderr
    assert "squeue" in r.stderr, "tell the reader how to check"
    assert "need at least two" not in r.stderr, "the misleading message must be gone"


def test_bench_shape_and_batch_come_from_the_config():
    """A benchmark that cannot see the setting you changed will answer confidently anyway.

    On 20-08-2026 a B200 job was submitted purely to measure what matching upstream's stage-1
    prior shape costs. Every GPU row came back at `1024, 40, 768` and batch 4, because those
    were hardcoded in four places and `--batch-size` still defaulted to 4 from before
    `train.batch_size` became 64. The run measured the old shape at a batch the experiments do
    not use, and its verdict divided a batch-4 rate by a batch-1 ceiling.
    """
    import importlib.util

    import yaml

    spec = importlib.util.spec_from_file_location(
        "benchmark_gpu", ROOT / "scripts" / "benchmark_gpu.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    for task, name in (("lgd", "LGD"), ("pd", "PD")):
        cfg = yaml.safe_load((ROOT / "config" / f"Exp1_{name}.yaml").read_text(encoding="utf-8"))
        rows, feats, train_size = bench.bench_shape(task)
        assert rows == max(cfg["prior"]["n_rows_range"])
        lo, hi = cfg["prior"]["n_features_range"]
        assert feats == round((lo + hi) / 2), "the MEAN width, so this is the mean step"
        lo, hi = cfg["prior"]["train_frac_range"]
        assert train_size == round(rows * (lo + hi) / 2)
        assert bench._train_cfg(task)["batch_size"] == cfg["train"]["batch_size"]


def test_no_hardcoded_dataset_shape_is_left_in_the_benchmark():
    """`1024, 40, 768` appeared four times. It must come from `bench_shape` or the next config
    change is measured against the last one again."""
    src = (ROOT / "scripts" / "benchmark_gpu.py").read_text(encoding="utf-8")
    # The ASSIGNMENT, not the string — `bench_shape`'s docstring names the old triple on purpose.
    assert "train_size = 1024, 40, 768" not in src
    # >= 4: the four original rungs, plus any new one — the point is that none is hardcoded.
    assert src.count("rows, feats, train_size = bench_shape(task)") >= 4


def test_micro_plan_matches_what_a_training_step_actually_does():
    """A step is `ceil(batch/micro)` passes of `micro`, not one pass of `batch`.

    Benchmarking one pass of 64 (job 11521108) OOMed at 176 GiB of a 178 GiB card, reported a
    1,617 ms "step", and made Muon look free at 1.01x because its fixed ~18 ms of Newton-Schulz
    was amortised over a pass sixteen times too large.
    """
    import importlib.util
    import math

    import yaml

    spec = importlib.util.spec_from_file_location(
        "benchmark_gpu", ROOT / "scripts" / "benchmark_gpu.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    for task, name in (("lgd", "LGD"), ("pd", "PD")):
        cfg = yaml.safe_load((ROOT / "config" / f"Exp1_{name}.yaml").read_text(encoding="utf-8"))
        batch, micro_cfg = cfg["train"]["batch_size"], cfg["train"]["micro_batch_size"]
        micro, n_micro = bench.micro_plan(task)
        assert micro == micro_cfg
        assert n_micro == math.ceil(batch / micro_cfg)
        assert micro * n_micro >= batch, "the passes must cover the batch"


def test_no_gpu_rung_builds_a_full_batch_tensor():
    """Every `torch.randn` in a timing rung must be sized by `micro`, never by `batch_size` —
    that is the difference between measuring a training step and OOMing the card."""
    src = (ROOT / "scripts" / "benchmark_gpu.py").read_text(encoding="utf-8")
    assert "torch.randn(batch_size," not in src
    assert "torch.rand(batch_size," not in src
