"""Run ONE experiment end to end: `python -m src.utils.run_experiment <1|2|3>`.

    python -m src.utils.run_experiment 1            # Exp1: what is done, what is next
    python -m src.utils.run_experiment 1 --submit   # Exp1: submit whatever is ready
    python -m src.utils.run_experiment 2 --submit   # Exp2 (only after Exp1's winner is filled in)

WHICH EXPERIMENT, ALWAYS EXPLICIT. The experiment number is a REQUIRED argument, not a default,
and it is printed at the top of every report — because "did I just submit Exp1 or Exp3?" is not
a question anyone should have to answer from memory. One invocation drives exactly one
experiment's two phases (train -> benchmark) for both tracks.

WHY THIS EXISTS
---------------

This cluster stops. Not occasionally — routinely: unannounced maintenance drains, node
failures, the 72 h walltime, and running out of credits until the balance is topped up. A plan
that consists of "run these five sbatch commands in the right order on the right days" is a
plan that will be executed wrongly, because nobody remembers which of the five already
happened after a week's interruption.

So this replaces the commands. It reads the OUTPUT TREE — not `sacct`, which forgets, and not
the queue, which is emptied by a drain — decides what is finished, and submits only the rest.
**Running it repeatedly is the intended usage.** After any interruption, whatever its cause,
the recovery procedure is to run it again.

THE ORDER IS A FACT, NOT A PREFERENCE

    train(exp, track)  ->  benchmark(exp, track)

Phase 2 scores the checkpoints phase 1 writes, so it cannot start earlier and this refuses to
start it. `benchmark` appears as `blocked` until every arm of its own experiment and track is
`done`.

WHAT IT WILL NOT DO

Submit something that is already queued or running. A drain empties the queue, but a slow day
does not, and re-running this while 8 arms are pending must not add 8 more. It matches on the
Slurm job name, so `squeue` is consulted for that and nothing else.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Where each track's training array goes by default. LGD is split across two clusters ONLY
#: because they have separate queues and separate contention; there is nothing about LGD that
#: needs wICE. Set `--single-cluster` to put everything on Mindwell, which is simpler to reason
#: about and slower by roughly the ratio of the two GPUs.
SPLIT_LGD = [
    ("mindwell", "gpu_b200", 24, "180G", "0-39"),
    ("wice", "gpu_a100", 18, "120G", "40-74"),
]


@dataclass
class Stage:
    """One submittable unit of work, and everything needed to decide whether to submit it."""

    name: str
    exp: int
    track: str
    kind: str  # "train" | "benchmark"
    total: int = 0
    done: int = 0
    queued: bool = False
    blocked_by: str | None = None
    commands: list[list[str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done >= self.total

    @property
    def state(self) -> str:
        if self.complete:
            return "done"
        if self.queued:
            return "running"
        if self.blocked_by:
            return "blocked"
        return "ready"


def queued_job_names() -> set[str]:
    """Job names currently pending or running, so a re-run does not double-submit.

    Best-effort: no `squeue` (a laptop, or a login node with a broken module) returns an empty
    set, which makes this fall back to "submit it" rather than "silently skip it". Of the two
    failure modes, a duplicate job you can cancel beats work that never starts.
    """
    names: set[str] = set()
    for clusters in (["--clusters=all"], []):
        try:
            out = subprocess.run(
                ["squeue", "--me", "--noheader", "--format=%j", *clusters],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            names |= {line.strip() for line in out.stdout.splitlines() if line.strip()}
            break
    return names


def _train_stage(exp: int, track: str, queued: set[str], single_cluster: bool) -> Stage:
    from src.utils.sweep_status import DONE, arm_states

    cfg = ROOT / "config" / f"Exp{exp}_{track.upper()}.yaml"
    states = arm_states(cfg)
    job_name = f"crediticl-{track}"
    script = f"scripts/slurm/pretrain_{track}.slurm"

    stage = Stage(
        name=f"exp{exp}-{track}-train", exp=exp, track=track, kind="train",
        total=len(states), done=sum(1 for s in states if s.state == DONE),
        queued=job_name in queued,
    )
    if stage.complete or stage.queued:
        return stage

    # ONLY WHAT IS LEFT. After a drain this is the difference between resubmitting 6 arms and
    # resubmitting 75; a `partial` arm resumes from its checkpoint rather than restarting.
    from src.utils.sweep_status import array_spec

    spec = array_spec(states)
    if track == "lgd" and not single_cluster and stage.done == 0:
        for cluster, part, cores, mem, rng in SPLIT_LGD:
            stage.commands.append([
                "sbatch", f"--clusters={cluster}", f"--partition={part}",
                f"--cpus-per-task={cores}", f"--mem={mem}", f"--array={rng}%8", script,
            ])
    else:
        # A partial sweep goes back to one cluster: the surviving indices are scattered and
        # splitting a scattered spec across two partitions gains nothing but confusion.
        stage.commands.append(
            ["sbatch", "--clusters=mindwell", f"--array={spec}%8", script]
        )
    return stage


def _benchmark_stage(exp: int, track: str, queued: set[str], train: Stage) -> Stage:
    from src.utils import paths
    from src.utils.config import expand_with_seeds, load

    cfg = ROOT / "config" / f"Exp{exp}_{track.upper()}.yaml"
    n_arms = len(expand_with_seeds(load(cfg, allow_placeholders=True)))
    eval_dir = paths.results_dir() / track / "eval"

    # One results file per arm, plus the shared reference column.
    wanted = [f"results_exp{exp}bench_{track}_a{i}.csv" for i in range(n_arms)]
    wanted.append(f"results_reference_{track}.csv")
    done = sum(1 for name in wanted if (eval_dir / name).is_file())

    stage = Stage(
        name=f"exp{exp}-{track}-benchmark", exp=exp, track=track, kind="benchmark",
        total=len(wanted), done=done, queued="crediticl-bench" in queued,
        blocked_by=None if train.complete else train.name,
    )
    if stage.complete or stage.queued or stage.blocked_by:
        return stage
    stage.commands.append([
        "sbatch", "--clusters=mindwell", f"--export=ALL,EXP={exp},TRACK={track}",
        f"--array=0-{n_arms}%8", "scripts/slurm/benchmark.slurm",
    ])
    return stage


def unconfigured_tracks(exp: int, tracks: list[str]) -> list[str]:
    """Tracks whose config still holds `FILL_FROM_EXP1`. Empty for a runnable experiment.

    Exp2 and Exp3 ship as templates: the prior mix and (for Exp2) the winning arm are blank
    until Exp1 has picked a winner. Submitting one anyway would run for hours and measure the
    wrong thing, so `run_experiment 2 --submit` must refuse until the holes are filled.
    """
    from src.utils.config import find_placeholders, load_yaml

    blocked = []
    for track in tracks:
        cfg = load_yaml(ROOT / "config" / f"Exp{exp}_{track.upper()}.yaml")
        if find_placeholders(cfg):
            blocked.append(track)
    return blocked


def plan(exp: int, tracks: list[str], single_cluster: bool = False) -> list[Stage]:
    """Every stage of ONE experiment, in dependency order, state read off the filesystem."""
    queued = queued_job_names()
    stages: list[Stage] = []
    for track in tracks:
        train = _train_stage(exp, track, queued, single_cluster)
        stages.append(train)
        stages.append(_benchmark_stage(exp, track, queued, train))
    return stages


def render(stages: list[Stage], exp: int, blocked_tracks: list[str] | None = None) -> str:
    mark = {"done": "[x]", "running": "[~]", "ready": "[ ]", "blocked": "[-]"}
    lines = [
        "=" * 78,
        f" CREDITICL — EXPERIMENT {exp} — read from the output tree, not from memory",
        "=" * 78,
    ]
    if blocked_tracks:
        lines += [
            f"  EXPERIMENT {exp} IS NOT CONFIGURED YET: {', '.join(blocked_tracks)} still hold",
            "  FILL_FROM_EXP1. Finish Exp1, choose the winning prior, and fill it into",
            f"  config/Exp{exp}_*.yaml before running this. Nothing will be submitted.",
            "=" * 78,
        ]
        return "\n".join(lines)
    for s in stages:
        bar = f"{s.done}/{s.total}"
        note = f"  waiting on {s.blocked_by}" if s.blocked_by else ""
        lines.append(f"  {mark[s.state]} {s.name:26s} {s.state:8s} {bar:>8s}{note}")
    lines.append("")

    ready = [s for s in stages if s.state == "ready"]
    if not ready:
        if all(s.complete for s in stages):
            lines.append("  EVERYTHING IS DONE.")
        else:
            lines.append("  Nothing to submit: work is either running or waiting on a")
            lines.append("  stage above it. Run this again after it finishes.")
    else:
        lines.append("  READY TO SUBMIT:")
        for s in ready:
            for cmd in s.commands:
                lines.append("    " + " ".join(cmd))
        lines.append("")
        lines.append("  Add --submit to run these. Safe to repeat: a stage already queued or")
        lines.append("  already finished is skipped, and a partial sweep resubmits only its")
        lines.append("  missing arms.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("exp", type=int, choices=(1, 2, 3),
                    help="WHICH EXPERIMENT to run (1, 2, or 3). Required, never a default.")
    ap.add_argument("--track", nargs="*", default=["lgd", "pd"], choices=["lgd", "pd"])
    ap.add_argument("--submit", action="store_true", help="actually sbatch what is ready")
    ap.add_argument("--single-cluster", action="store_true",
                    help="keep everything on Mindwell instead of splitting LGD across two")
    args = ap.parse_args(argv)

    blocked = unconfigured_tracks(args.exp, args.track)
    if blocked:
        # Refuse a template. Print the plan header with the reason and stop — no filesystem
        # scan, no submission.
        print(render([], args.exp, blocked_tracks=blocked))
        return 1

    stages = plan(args.exp, args.track, args.single_cluster)
    print(render(stages, args.exp))
    if not args.submit:
        return 0

    rc = 0
    for stage in stages:
        if stage.state != "ready":
            continue
        for cmd in stage.commands:
            print(f"\n$ {' '.join(cmd)}")
            rc |= subprocess.call(cmd, cwd=ROOT)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
