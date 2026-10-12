"""The draw pool's worker processes end with the process that started them.

On 29-09-2026 a notebook run finished in four minutes and then looked hung for an hour: the runner
stops each kernel outright (`shutdown_kernel(now=True)`), and on Windows that leaves the kernel's
spawned draw workers running — sixteen of them, holding the runner's output pipe open. The workers
now follow their parent, and the pool is ended at exit and before a kernel is stopped.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

REPO = Path(__file__).resolve().parents[1]

PARENT = f"""
import sys, time
sys.path.insert(0, {str(REPO)!r})
from src.visualize import draw

if __name__ == "__main__":
    draw.parallel_map(abs, list(range(-50, 50)), workers=2)
    print(" ".join(str(p.pid) for p in draw._POOL._processes.values()), flush=True)
    if sys.argv[1] == "hang":
        time.sleep(600)
"""


def _workers_left(tmp_path: Path, mode: str) -> tuple[int, int]:
    script = tmp_path / "parent.py"
    script.write_text(PARENT, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), mode], stdout=subprocess.PIPE, text=True)
    pids = [int(x) for x in proc.stdout.readline().split()]
    if mode == "hang":
        proc.kill()  # what `shutdown_kernel(now=True)` does to a kernel
    proc.wait(timeout=120)
    deadline = time.time() + 20
    while time.time() < deadline and any(psutil.pid_exists(p) for p in pids):
        time.sleep(0.5)
    return len(pids), sum(psutil.pid_exists(p) for p in pids)


@pytest.mark.parametrize("mode", ["normal", "hang"])
def test_workers_end_with_their_parent(tmp_path, mode):
    started, left = _workers_left(tmp_path, mode)
    assert started == 2, "the draw did not use the pool"
    assert left == 0, f"{left} draw workers outlived their parent ({mode} exit)"


def test_the_notebook_runner_ends_the_pool_before_stopping_a_kernel():
    from src.utils import run_notebooks

    assert "shutdown_pool()" in run_notebooks._END_POOL
    source = Path(run_notebooks.__file__).read_text(encoding="utf-8")
    assert source.index("_END_POOL, time.time()") < source.index("km.shutdown_kernel(now=True)")
