"""What the hardware and the gradients were doing, sampled during training.

WHY THIS EXISTS. A cluster run cannot be watched. By the time anything is known about it, the
job is over and all that survives is what it wrote down. Two questions come up after almost
every run, and neither can be answered afterwards unless it was logged at the time:

* **"Was the GPU actually busy?"** A run that takes three days because it is compute-bound and
  a run that takes three days because the CPU cannot generate synthetic datasets fast enough
  look *identical* from the outside — same wall-clock, same loss curve. The fix is opposite in
  each case (more GPUs vs more `num_workers`), so guessing wastes another three days.
* **"Was the whole model learning?"** A per-block gradient norm shows whether the column
  encoder, the row encoder and the ICL blocks are all receiving signal. A block whose gradient
  is orders of magnitude below the rest is effectively frozen, and the loss curve will not say
  so — it will just be slightly worse than it should be, forever.

EVERYTHING HERE IS BEST-EFFORT AND NEVER FATAL. A diagnostic that can kill a three-day run is
worse than no diagnostic, so every probe is wrapped and a failure degrades to a missing column.
`nvidia-smi` in particular is absent on a login node and on the Windows dev machine.

Output goes to the run's `.log` (human-readable) and to `output/manifests/<run>__telemetry.csv`
(one row per sample, for plotting afterwards).
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Queried in one `nvidia-smi` call. Order matters — it is the CSV column order too.
_SMI_FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
)

#: Blocks worth separating in a gradient report. TabICLv2 is three stacked transformers plus a
#: head, and "which stage is learning" is the question, so grouping by these prefixes is more
#: informative than either one global norm or 390 per-tensor norms.
_BLOCK_PREFIXES = (
    ("col", ("col_", "x_embed", "tf_col", "col_embedder")),
    ("row", ("row_", "tf_row", "row_interaction")),
    ("icl", ("icl_", "tf_icl", "learning")),
    ("head", ("head", "out_", "y_embed", "quantile")),
)


def _classify(name: str) -> str:
    """Which reported block a parameter belongs to. `other` is not a bug — it catches
    embeddings and norms that sit outside the three stacks, and seeing it move is useful."""
    lowered = name.lower()
    for block, prefixes in _BLOCK_PREFIXES:
        if any(p in lowered for p in prefixes):
            return block
    return "other"


def gpu_stats() -> dict[str, float]:
    """One `nvidia-smi` sample per visible GPU, flattened. `{}` when unavailable.

    Uses the CLI rather than `torch.cuda` because utilisation and power are not exposed by
    torch at all, and utilisation is the one number that distinguishes a compute-bound run
    from a starved one.
    """
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("nvidia-smi unavailable: %s", exc)
        return {}

    stats: dict[str, float] = {}
    for gpu_index, line in enumerate(row for row in out.splitlines() if row.strip()):
        for field, raw in zip(_SMI_FIELDS, line.split(",")):
            key = f"gpu{gpu_index}_{field.replace('.', '_')}"
            try:
                stats[key] = float(raw.strip())
            except ValueError:
                # "[N/A]" is normal for power on some cards; a missing column beats a crash.
                continue
    return stats


def torch_memory() -> dict[str, float]:
    """Allocator high-water marks, in GB. These are what an OOM is actually about.

    `max_memory_allocated` is the number to size a batch against; `max_memory_reserved` shows
    how much the caching allocator is holding, and a large gap between them means
    fragmentation — which is why an OOM can appear at step 40,000 after 39,999 identical steps.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        gb = 1024 ** 3
        return {
            "mem_allocated_gb": torch.cuda.memory_allocated() / gb,
            "mem_reserved_gb": torch.cuda.memory_reserved() / gb,
            "mem_max_allocated_gb": torch.cuda.max_memory_allocated() / gb,
            "mem_max_reserved_gb": torch.cuda.max_memory_reserved() / gb,
        }
    except Exception as exc:  # noqa: BLE001 — never fatal
        log.debug("torch memory probe failed: %s", exc)
        return {}


def host_stats() -> dict[str, float]:
    """CPU and RAM. Optional: `psutil` is not a dependency, so this is empty without it.

    Worth having because the prior is generated on the CPU. A run pinned at 100 % CPU with an
    idle GPU is starved, and that is a `num_workers` problem, not a model problem.
    """
    try:
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": psutil.virtual_memory().used / 1024 ** 3,
            "ram_percent": psutil.virtual_memory().percent,
        }
    except Exception:  # noqa: BLE001 — psutil is optional by design
        return {}


def grad_norms(model: Any) -> dict[str, float]:
    """L2 gradient norm per architecture block, plus the global norm.

    Call AFTER `backward()` and BEFORE the optimizer step — afterwards the gradients may have
    been zeroed and the numbers would all be 0.0, which looks like a dead model rather than a
    mistimed probe.
    """
    try:
        import torch

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        total = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            value = float(torch.linalg.vector_norm(param.grad.detach()) ** 2)
            block = _classify(name)
            sums[block] = sums.get(block, 0.0) + value
            counts[block] = counts.get(block, 0) + 1
            total += value

        out = {f"grad_{block}": sums[block] ** 0.5 for block in sums}
        out.update({f"nparams_with_grad_{block}": float(counts[block]) for block in counts})
        out["grad_global"] = total ** 0.5
        return out
    except Exception as exc:  # noqa: BLE001 — never fatal
        log.debug("gradient probe failed: %s", exc)
        return {}


def weight_norms(model: Any) -> dict[str, float]:
    """L2 weight norm per block. Paired with the gradient norms on purpose.

    The gradient alone is not interpretable: a norm of 0.01 is tiny on a block whose weights
    are 0.1 and enormous on one whose weights are 1e-5. The RATIO is the thing to watch, and it
    is also the cheapest early sign of divergence — weights growing while the loss looks flat.
    """
    try:
        import torch

        sums: dict[str, float] = {}
        for name, param in model.named_parameters():
            block = _classify(name)
            sums[block] = sums.get(block, 0.0) + float(torch.linalg.vector_norm(param.detach()) ** 2)
        return {f"weight_{block}": v ** 0.5 for block, v in sums.items()}
    except Exception as exc:  # noqa: BLE001
        log.debug("weight probe failed: %s", exc)
        return {}


class Telemetry:
    """Samples the hardware and the model on a step cadence, and writes one CSV.

    Two independent cadences, because the two probes cost very different amounts: an
    `nvidia-smi` call is a subprocess (milliseconds, but not free), while walking every
    parameter's gradient is proportional to the model size. `0` disables either.
    """

    def __init__(
        self,
        run_name: str,
        out_dir: str | Path,
        *,
        hardware_every: int = 100,
        grad_every: int = 250,
        micro_batch_size: int | None = None,
    ) -> None:
        #: Only so the summary can suggest a bigger one when memory is barely touched.
        #: Gradient accumulation makes the micro-batch mathematically invisible, so raising it
        #: is free speed — but nothing was telling anyone there was room.
        self.micro_batch_size = int(micro_batch_size) if micro_batch_size else None
        self.hardware_every = int(hardware_every or 0)
        self.grad_every = int(grad_every or 0)
        self.path = Path(out_dir) / f"{run_name}__telemetry.csv"
        self._fields: list[str] = []
        self._rows: list[dict[str, Any]] = []
        self._started = time.time()
        #: Recorded once and reported in the summary. Answers "what was this actually run on?",
        #: which no amount of later inspection can recover.
        self.environment = _describe_environment()

    @property
    def enabled(self) -> bool:
        return bool(self.hardware_every or self.grad_every)

    def due_hardware(self, step: int) -> bool:
        return bool(self.hardware_every) and step % self.hardware_every == 0

    def due_grad(self, step: int) -> bool:
        return bool(self.grad_every) and step % self.grad_every == 0

    def sample_grads(self, model: Any) -> dict[str, float]:
        """Gradient and weight norms. Must be called between backward() and the step."""
        row = grad_norms(model)
        row.update(weight_norms(model))
        for block in ("col", "row", "icl", "head", "other"):
            g, w = row.get(f"grad_{block}"), row.get(f"weight_{block}")
            if g is not None and w:
                # The interpretable quantity: how big a relative change this step wants to make.
                row[f"gw_ratio_{block}"] = g / w
        return row

    def sample_hardware(self, *, step: int, datasets_seen: int, steps_per_s: float) -> dict[str, Any]:
        row: dict[str, Any] = {
            "step": step,
            "datasets_seen": datasets_seen,
            "elapsed_s": round(time.time() - self._started, 2),
            "steps_per_s": round(steps_per_s, 4),
            "datasets_per_s": round(steps_per_s * max(datasets_seen / max(step, 1), 0.0), 4),
        }
        row.update(gpu_stats())
        row.update(torch_memory())
        row.update(host_stats())
        return row

    def record(self, row: dict[str, Any]) -> None:
        """Append a row and rewrite the CSV with the union of all columns seen.

        Rewritten rather than appended because columns can appear late — gradient rows and
        hardware rows carry different keys, and a ragged CSV misaligns silently in pandas,
        which is the worst way to lose a diagnostic.
        """
        self._rows.append(row)
        new = [k for k in row if k not in self._fields]
        self._fields.extend(new)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fields)
                writer.writeheader()
                writer.writerows(self._rows)
        except Exception as exc:  # noqa: BLE001
            # Deliberately not just OSError. A malformed path raises ValueError, a value the
            # csv module cannot serialise raises TypeError, and a full filesystem raises OSError
            # — and every one of them would otherwise end a three-day run to protect a
            # diagnostic file. The in-memory rows survive, so `summary()` still works.
            log.debug("could not write telemetry to %s: %s", self.path, exc)

    def summary(self) -> str:
        """What the hardware did across the whole run, for the end of the log."""
        if not self._rows:
            return "telemetry: nothing sampled."
        lines = [f"telemetry: {len(self._rows)} samples -> {self.path}"]
        for key, label in (
            ("gpu0_utilization_gpu", "GPU utilisation %"),
            ("mem_max_allocated_gb", "peak allocated GB"),
            ("steps_per_s", "steps/s"),
            ("cpu_percent", "CPU %"),
        ):
            values = [r[key] for r in self._rows if isinstance(r.get(key), (int, float))]
            if values:
                lines.append(
                    f"  {label:<22} min {min(values):8.2f}  mean "
                    f"{sum(values) / len(values):8.2f}  max {max(values):8.2f}"
                )
        util = [r["gpu0_utilization_gpu"] for r in self._rows
                if isinstance(r.get("gpu0_utilization_gpu"), (int, float))]
        if util and sum(util) / len(util) < 70:
            lines.append(
                f"  WARNING mean GPU utilisation {sum(util) / len(util):.0f}% — the run is "
                f"probably STARVED, not compute-bound. Raise train.num_workers before "
                f"asking for more GPUs."
            )
        return "\n".join(lines)


def _describe_environment() -> dict[str, str]:
    """Host, GPU, library versions and SLURM identity. Recorded once per run.

    This is what makes a result reproducible six months later, and none of it can be
    reconstructed from the numbers alone.
    """
    import os
    import platform
    import sys

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
                "SLURM_NNODES", "SLURM_GPUS_ON_NODE", "VSC_INSTITUTE_CLUSTER"):
        if os.environ.get(var):
            info[var.lower()] = os.environ[var]
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "cpu"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = str(torch.cuda.device_count())
            info["gpu_capability"] = ".".join(str(c) for c in torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        pass
    return info
