"""Put two `benchmark_gpu.py` JSON files side by side and say which layer differs.

    python -m src.utils.compare_gpubench output/logs/gpubench_*.json

The comparison is the measurement. One card's numbers alone cannot say whether 0.5 steps/s is
bad — only that the same work ran 12x faster elsewhere can, which is why the benchmark is
designed to be run twice and read as a diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: (json path, label, higher-is-better). The ladder from raw compute to the real thing.
ROWS: tuple[tuple[str, str, bool], ...] = (
    ("matmul.matmul_4096_float32_tflops", "matmul fp32 (TFLOP/s)", True),
    ("matmul.matmul_4096_bfloat16_tflops", "matmul bf16 (TFLOP/s)", True),
    ("attention.attention_1024_ms", "attention 1024 (ms)", False),
    ("model.forward_ms", "model forward (ms)", False),
    ("model.forward_backward_ms", "model fwd+bwd (ms)", False),
    ("model.max_steps_per_s_if_data_were_free", "GPU ceiling (steps/s)", True),
    ("optimizers.adamw_step_ms", "AdamW step (ms)", False),
    ("optimizers.muon_step_ms", "MUON step (ms)", False),
    ("optimizers.muon_slowdown_vs_adamw", "Muon / AdamW", False),
    ("prior.datasets_per_s_one_worker", "prior, 1 worker (ds/s)", True),
    ("end_to_end.steps_per_s", "END TO END (steps/s)", True),
)


def _dig(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def main(argv: list[str]) -> int:
    raw = argv[1:]

    # AN UNEXPANDED GLOB IS NOT A FILE. When no file matches, the shell passes the pattern
    # through literally, so the argument count is 1 and the honest message is "nothing
    # matched yet" — not "give me two files", which is what it used to say while the jobs
    # were still queued.
    unexpanded = [a for a in raw if any(c in a for c in "*?[") and not Path(a).exists()]
    if unexpanded:
        print("No benchmark results matched:", file=sys.stderr)
        for a in unexpanded:
            print(f"    {a}", file=sys.stderr)
        print(
            "\nThe pattern came back unexpanded, which means the files do not exist yet.\n"
            "The jobs are probably still queued or running. Check with:\n"
            "    squeue --clusters=mindwell -u $USER\n"
            "    sacct --clusters=mindwell -u $USER --starttime=today \\\n"
            "        --format=JobID%18,JobName%22,State,ExitCode,Elapsed\n"
            "and if a job has finished but wrote no JSON, read its log:\n"
            "    ls -t $VSC_DATA/CreditICL/output/logs/gpubench_*.out | head -2",
            file=sys.stderr,
        )
        return 2

    paths = [Path(p) for p in raw]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"missing: {p}", file=sys.stderr)
        return 1

    runs = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        info = data.get("device_info") or {}
        runs.append((info.get("gpu", p.stem), data, info))

    if len(runs) < 2:
        # One card still gets a table. The comparison is the point, but a single column
        # answers "did the benchmark actually work?", which is the next question anyway.
        print("Only ONE result so far — this cannot answer the comparison question.\n"
              "Printing it alone so you can see the benchmark ran; submit the other card:\n"
              "    bash scripts/slurm/submit.sh b200 pd scripts/slurm/benchmark.slurm\n",
              file=sys.stderr)

    width = max(len(r[0]) for r in runs) + 2
    print("=" * (30 + width * len(runs)))
    print(f"{'measurement':<30}" + "".join(f"{r[0]:>{width}}" for r in runs))
    print("-" * (30 + width * len(runs)))

    for key, label, higher_better in ROWS:
        vals = [_dig(d, key) for _, d, _ in runs]
        if all(v is None for v in vals):
            continue
        cells = "".join(f"{v if v is not None else '—':>{width}}" for v in vals)
        print(f"{label:<30}{cells}")
        nums = [v for v in vals if isinstance(v, (int, float))]
        if len(nums) >= 2 and min(nums) > 0:
            ratio = max(nums) / min(nums)
            if ratio >= 1.5:
                best = max(nums) if higher_better else min(nums)
                which = runs[vals.index(best)][0]
                print(f"{'':<30}  -> {ratio:.1f}x apart, {which} ahead")

    print("-" * (30 + width * len(runs)))
    for name, _, info in runs:
        kernels = info.get("has_kernels_for_this_card")
        # ASCII: this line is the punchline, and it must survive a console that cannot
        # print an em dash.
        note = "" if kernels is not False else "  *** NO KERNELS FOR THIS CARD -> JIT fallback"
        print(f"  {name}: capability {info.get('capability')}, torch {info.get('torch')}, "
              f"compiled_for={info.get('compiled_for')}{note}")

    print()
    if len(runs) >= 2:
        print("  matmul/attention differ      -> the CARD or the wheel. Change hardware.")
        print("  those match, model differs   -> a kernel this model needs is missing there.")
        print("  all match, END TO END differs-> the GPU was never the problem; the prior is,")
        print("                                  and only more CPU cores will help.")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
