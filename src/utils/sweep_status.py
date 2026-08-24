"""Which arms of a sweep are done, which are half-done, and what to resubmit.

    python -m src.utils.sweep_status --config config/Exp1_LGD.yaml
    python -m src.utils.sweep_status --config config/Exp1_LGD.yaml --array
    python -m src.utils.sweep_status --config config/Exp1_LGD.yaml --resubmit

WHY THIS EXISTS
---------------

A 75-arm array does not finish in one sitting, and on this cluster it will be interrupted.
Three things do it, and only one of them gives the job a chance to react:

    the 72 h walltime      -> SIGUSR1 600 s early; the trainer checkpoints and exits 64
    a node failure         -> --requeue puts the task back
    a cluster-wide drain   -> **every PENDING task is cancelled and nothing is signalled**

The third is the one nothing in the job script can help with. The arms that were running get
their signal and resume; the arms still queued simply vanish. Resubmitting the whole array
afterwards would redo the finished ones — 75 arms of work to recover a handful.

So this reads the output tree and asks, per arm, *what actually happened*:

    done      `summary.json` says `completed: true`
    partial   a checkpoint exists, but no completed summary -> RESUME, do not restart
    todo      nothing on disk

and prints the Slurm array specification covering everything that is not `done`. Resubmitting
that spec is safe: `maybe_resume` picks up each partial arm's checkpoint, and a `todo` arm
starts at step 0.

**`summary.json` is the authority, not the checkpoint**, because a checkpoint exists from step
250 onwards and says nothing about whether the run finished. `completed` is written by
`Trainer.train` and is false when a signal ended the run early.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: What an arm can be. Ordered worst-to-best so a sort puts the work first.
TODO, PARTIAL, DONE = "todo", "partial", "done"


@dataclass
class ArmState:
    """One array index, and everything on disk about it."""

    index: int
    name: str
    state: str
    steps: int | None = None
    max_steps: int | None = None
    stopped_by: str | None = None
    checkpoint: str | None = None

    @property
    def progress(self) -> str:
        if self.steps is None or not self.max_steps:
            return ""
        return f"{self.steps:,}/{self.max_steps:,} ({100 * self.steps / self.max_steps:.0f}%)"


def _latest_checkpoint_name(ckpt_dir: Path) -> str | None:
    if not ckpt_dir.is_dir():
        return None
    found = sorted(ckpt_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return found[-1].name if found else None


def arm_states(config: Path, out_root: Path | None = None) -> list[ArmState]:
    """Inspect every arm of a config's expanded grid.

    Deliberately reads the FILESYSTEM and not Slurm: `sacct` forgets, a drained cluster loses
    its queue, and the output tree is the only record that survives both.
    """
    from src.utils import paths
    from src.utils.config import expand_with_seeds, load, run_name

    runs = expand_with_seeds(load(config, allow_placeholders=True))
    root = Path(out_root) if out_root else paths.outputs_dir()

    states: list[ArmState] = []
    for index, run in enumerate(runs):
        name = run_name(run)
        out_dir = root / name
        summary = out_dir / "summary.json"
        ckpt = _latest_checkpoint_name(paths.checkpoints_dir() / name) or _latest_checkpoint_name(
            out_dir / "checkpoints"
        )

        if summary.is_file():
            try:
                data: dict[str, Any] = json.loads(summary.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            # `completed` was added on 24-08-2026. An older summary that reached max_steps is
            # still done; treating a missing key as "not done" would redo finished work.
            steps = data.get("steps")
            done = data.get("completed")
            if done is None:
                done = bool(steps) and steps >= (run.get("train") or {}).get("max_steps", 0)
            states.append(
                ArmState(
                    index, name, DONE if done else PARTIAL, steps,
                    (run.get("train") or {}).get("max_steps"),
                    data.get("stopped_by_signal"), ckpt,
                )
            )
            continue

        states.append(
            ArmState(index, name, PARTIAL if ckpt else TODO, None,
                     (run.get("train") or {}).get("max_steps"), None, ckpt)
        )
    return states


def array_spec(states: list[ArmState]) -> str:
    """Slurm `--array` for every arm that is not done, collapsed into ranges.

    `0-4,9,12-74` rather than 71 comma-separated numbers: Slurm accepts both, but a spec you
    can read is a spec you can check before submitting 300 GPU-hours.
    """
    todo = sorted(s.index for s in states if s.state != DONE)
    if not todo:
        return ""
    parts, start, prev = [], todo[0], todo[0]
    for i in todo[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def report(states: list[ArmState], verbose: bool = False) -> str:
    counts = {s: sum(1 for a in states if a.state == s) for s in (DONE, PARTIAL, TODO)}
    lines = [
        "=" * 78,
        f" SWEEP STATUS — {len(states)} arms",
        "=" * 78,
        f"  done     {counts[DONE]:>4}",
        f"  partial  {counts[PARTIAL]:>4}   <- resumable; a checkpoint exists",
        f"  todo     {counts[TODO]:>4}",
        "",
    ]
    shown = [a for a in states if a.state != DONE or verbose]
    if shown:
        lines.append(f"  {'idx':>4}  {'state':<8} {'progress':<18} arm")
        for a in shown[: 200 if verbose else 40]:
            note = f"  (stopped by {a.stopped_by})" if a.stopped_by else ""
            lines.append(f"  {a.index:>4}  {a.state:<8} {a.progress:<18} {a.name[:70]}{note}")
        if len(shown) > (200 if verbose else 40):
            lines.append(f"  ... and {len(shown) - 40} more (use --verbose)")
        lines.append("")

    spec = array_spec(states)
    if spec:
        lines += [
            "  RESUBMIT ONLY WHAT IS LEFT:",
            f"    sbatch --clusters=mindwell --array={spec}%8 scripts/slurm/pretrain_<track>.slurm",
            "",
            "  Safe to run repeatedly: a `partial` arm resumes from its checkpoint and a",
            "  `todo` arm starts at step 0. Nothing already `done` is touched.",
        ]
    else:
        lines.append("  EVERY ARM IS DONE. Nothing to resubmit.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="the sweep config, e.g. config/Exp1_LGD.yaml")
    ap.add_argument("--out-root", default=None, help="override the output root")
    ap.add_argument("--array", action="store_true",
                    help="print ONLY the array spec, for scripting")
    ap.add_argument("--verbose", action="store_true", help="list finished arms too")
    ap.add_argument("--resubmit", action="store_true",
                    help="actually sbatch the remaining arms (asks nothing; check --array first)")
    ap.add_argument("--script", default=None, help="job script for --resubmit")
    ap.add_argument("--throttle", type=int, default=8, help="the %%N cap for --resubmit")
    args = ap.parse_args(argv)

    states = arm_states(Path(args.config), args.out_root)
    spec = array_spec(states)

    if args.array:
        print(spec)
        return 0

    print(report(states, args.verbose))

    if not args.resubmit:
        return 0
    if not spec:
        return 0
    script = args.script or (
        "scripts/slurm/pretrain_lgd.slurm" if "LGD" in Path(args.config).name
        else "scripts/slurm/pretrain_pd.slurm"
    )
    cmd = ["sbatch", "--clusters=mindwell", f"--array={spec}%{args.throttle}", script]
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
