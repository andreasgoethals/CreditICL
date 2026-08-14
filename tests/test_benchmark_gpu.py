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


def test_two_files_are_required(tmp_path):
    a = _write(tmp_path, "a.json", FAST, matmul=40.0, e2e=6.9, ceiling=16.7)
    r = subprocess.run(
        [sys.executable, "-m", "src.utils.compare_gpubench", str(a)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode == 2, "one file cannot answer a comparison question"
