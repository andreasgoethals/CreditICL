"""Why is this GPU slow? Separate "the chip is slow" from "the data pipeline stalls".

    python scripts/benchmark_gpu.py                    # everything, ~2 minutes
    python scripts/benchmark_gpu.py --skip-prior       # no CPU prior work, GPU only
    python scripts/benchmark_gpu.py --steps 30 --json out.json

WHY THIS EXISTS. On 14-08-2026 the same PD configuration ran at 0.5 steps/s on a B200 and
6.9 steps/s on a free RTX 5000 Ada — 12x slower on hardware costing 26,250 credits an hour,
at 3 % GPU utilisation. Utilisation that low means the GPU is *waiting*, but it does not say
what for, and the two candidate explanations need opposite fixes:

    the chip is slow here    -> stop using it; a wrong-architecture wheel or a JIT fallback
    the data pipeline stalls -> the GPU is fine and the prior generator is the problem

So this measures the layers separately, on whatever card it is run on:

    1. raw matmul            pure compute, no model, no data. TFLOP/s.
    2. attention             the shape TabICL actually runs, still no data.
    3. model forward         the real network, synthetic tensors already on the GPU.
    4. model fwd+bwd         adds the backward pass and the optimiser step.
    5. prior generation      CPU only, one worker, no GPU at all. datasets/s.
    6. end-to-end            the real DataLoader feeding real steps.

Read it as a ladder. If 1-4 are healthy and 6 is slow, the GPU is fine and the prior is
starving it. If 1 is already slow, the card or the wheel is wrong and no amount of
`num_workers` will help.

Run it on BOTH cards and compare the same row:

    bash scripts/slurm/submit.sh free  lgd scripts/slurm/benchmark.slurm
    bash scripts/slurm/submit.sh b200  lgd scripts/slurm/benchmark.slurm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, n: int, warmup: int, torch: Any) -> float:
    """Median seconds per call, warmed up and synchronised.

    Median, not mean: a single scheduler hiccup or a lazy JIT compile in the first timed
    call would otherwise dominate, and on a machine we suspect of a JIT fallback that is
    exactly the number we must not report.
    """
    for _ in range(warmup):
        fn()
    _sync(torch)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        _sync(torch)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def describe_device(torch: Any) -> dict[str, Any]:
    """Everything that could plausibly explain a slow card, recorded before any timing."""
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(0)
    cap = f"{props.major}.{props.minor}"
    info.update(
        gpu=props.name,
        capability=cap,
        memory_gb=round(props.total_memory / 1024**3, 1),
        multi_processors=props.multi_processor_count,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        # If the wheel carries no kernels for this card, CUDA JIT-compiles PTX — legal,
        # silent and slow.
        #
        # MINOR VERSIONS ARE BINARY-COMPATIBLE WITHIN A MAJOR ARCHITECTURE. An exact-string
        # test called `sm_89` (RTX 5000 Ada) unsupported against a list of
        # sm_75/80/86/90/100/120, and printed a scary "NO KERNELS" banner for the card that
        # turned out to be *fine* — while the B200, which the same test passed, was the slow
        # one. A false alarm on a diagnostic is worse than no diagnostic: it sends you at the
        # wrong suspect. `sm_86` cubins run on `sm_89`, so the test is on the MAJOR version.
        compiled_for=torch.cuda.get_arch_list(),
        has_kernels_for_this_card=any(
            a.startswith(f"sm_{props.major}") for a in torch.cuda.get_arch_list()
        ),
        exact_arch_shipped=f"sm_{props.major}{props.minor}" in torch.cuda.get_arch_list(),
    )
    return info


def bench_matmul(torch: Any, device: str, n: int, warmup: int) -> dict[str, float]:
    """Pure compute. Nothing here touches our code, so a bad number is the machine."""
    out = {}
    for size, dtype in ((4096, torch.float32), (4096, torch.bfloat16)):
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        # Bound as defaults, not captured: a late-binding closure over the loop variables
        # would time the LAST dtype under every label — the classic version of this bug, and
        # here it would silently report fp32 numbers as bf16.
        sec = _time(lambda x=a, w=b: x @ w, n, warmup, torch)
        # one matmul is 2*N^3 floating-point operations
        out[f"matmul_{size}_{str(dtype).split('.')[-1]}_tflops"] = round(
            2 * size**3 / sec / 1e12, 2
        )
    return out


def bench_attention(torch: Any, device: str, n: int, warmup: int) -> dict[str, float]:
    """Scaled dot-product attention at TabICL's own shape (128-dim, 8 heads, 1024 rows)."""
    import torch.nn.functional as F

    b, h, s, d = 4, 8, 1024, 16
    q, k, v = (torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16) for _ in range(3))
    sec = _time(lambda: F.scaled_dot_product_attention(q, k, v), n, warmup, torch)
    return {"attention_1024_ms": round(sec * 1000, 3)}


def bench_model(
    torch: Any, device: str, task: str, n: int, warmup: int, batch_size: int
) -> dict[str, Any]:
    """The real network on synthetic tensors ALREADY ON THE GPU — no data pipeline at all.

    This is the pivotal row. If it matches across two cards but the end-to-end step does not,
    the model is fine and the prior generator is the bottleneck.
    """
    from src.models.architecture import build_model, is_available

    if not is_available("tabicl"):
        return {"error": "upstream tabicl not installed"}

    model = build_model(task, architecture="tabicl").to(device)
    # AT THE REAL BATCH SIZE. It used to be 1, and the verdict below then divided a batch-N
    # end-to-end rate by a batch-1 ceiling — which reads as starvation whatever the truth is.
    rows, feats, train_size = bench_shape(task)
    X = torch.randn(batch_size, rows, feats, device=device)
    y = (
        torch.rand(batch_size, train_size, device=device)
        if task == "lgd"
        else (torch.rand(batch_size, train_size, device=device) > 0.7).float()
    )

    model.eval()
    with torch.no_grad():
        fwd = _time(lambda: model(X, y), n, warmup, torch)

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-6)

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(X, y)
        out.float().square().mean().backward()
        opt.step()

    both = _time(step, max(3, n // 2), warmup, torch)
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "rows": rows,
        "features": feats,
        "forward_ms": round(fwd * 1000, 2),
        "forward_backward_ms": round(both * 1000, 2),
        "max_steps_per_s_if_data_were_free": round(1.0 / both, 2),
    }


def bench_shape(task: str) -> tuple[int, int, int]:
    """The dataset shape TRAINING actually uses, read from this task's Exp1 config.

    WHY THIS IS NOT HARDCODED ANY MORE. It was `1024, 40, 768` in four places, and on
    20-08-2026 that silently wasted a benchmark. The run was submitted to measure what
    matching upstream's stage-1 prior shape costs — rows [512, 1024] -> exactly 1,024 and
    features [3, 50] -> [1, 100] — and every GPU row came back at the OLD shape, because the
    only sections reading the config were the two CPU/loader ones. A benchmark that cannot see
    the change you submitted it to measure is worse than no benchmark: it answers confidently.
    """
    prior = _prior_cfg(task)
    rows = int(max(prior.get("n_rows_range", [512, 1024])))
    flo, fhi = prior.get("n_features_range", [3, 50])
    feats = int(round((int(flo) + int(fhi)) / 2))      # the MEAN width, so this is the mean step
    lo, hi = prior.get("train_frac_range", [0.3, 0.9])
    return rows, feats, int(round(rows * (float(lo) + float(hi)) / 2))


def _prior_cfg(task: str) -> dict[str, Any]:
    """The prior block from this task's real Exp1 config, with the sweep grid removed."""
    import yaml

    cfg_path = ROOT / "config" / f"Exp1_{'LGD' if task == 'lgd' else 'PD'}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    prior_cfg = dict(cfg.get("prior") or {})
    prior_cfg.pop("grid", None)
    return prior_cfg


def _train_cfg(task: str) -> dict[str, Any]:
    """The `train` block from this task's real Exp1 config."""
    import yaml

    cfg_path = ROOT / "config" / f"Exp1_{'LGD' if task == 'lgd' else 'PD'}.yaml"
    return dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("train") or {})


def bench_prior(task: str, n_batches: int, batch_size: int) -> dict[str, Any]:
    """CPU only, ONE worker, no GPU. How fast can a single process make datasets?

    Multiply by `num_workers` for a ceiling on `datasets_per_s`. If that ceiling sits below
    what training achieved, the prior is the bottleneck and a faster GPU changes nothing.
    """
    from src.prior.dataset import build_loader

    try:
        loader = build_loader(_prior_cfg(task), task, batch_size, seed=0, num_workers=0)
        it = iter(loader)
        next(it)  # discard the first: it carries all the one-off setup
        t0 = time.perf_counter()
        for _ in range(n_batches):
            next(it)
        sec = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 — a benchmark must not fail the diagnosis
        return {"error": f"{type(exc).__name__}: {exc}"}
    made = n_batches * batch_size
    return {
        "datasets": made,
        "seconds": round(sec, 2),
        "datasets_per_s_one_worker": round(made / sec, 2) if sec else None,
        "s_per_dataset_one_worker": round(sec / made, 4) if made else None,
    }


def bench_amp(torch: Any, device: str, task: str, n: int, batch_size: int) -> dict[str, Any]:
    """Forward+backward AT THE REAL BATCH SIZE, with and without AMP.

    THE LAST DIFFERENCE between this benchmark and real training. `bench_model` runs batch 1 in
    fp32 and reported 43.7 ms on the B200; the per-phase timing added to the trainer shows real
    training spending **98.4 % of a 1.8 s step inside forward+backward** on that same card — 41x
    more, with data at 0.0 % and the optimiser at 1.4 %. Batch 4 accounts for 4x of it. The
    remainder is either AMP or nothing, and `train.amp` is `True` with `bfloat16` autocast plus
    a GradScaler.

    Measured, not assumed: three explanations for the B200 gap have been asserted and all three
    were wrong.
    """
    from src.models.architecture import build_model, is_available

    if not is_available("tabicl"):
        return {"error": "upstream tabicl not installed"}

    rows, feats, train_size = bench_shape(task)
    X = torch.randn(batch_size, rows, feats, device=device)
    y = (
        torch.rand(batch_size, train_size, device=device)
        if task == "lgd"
        else (torch.rand(batch_size, train_size, device=device) > 0.7).float()
    )
    out: dict[str, Any] = {"batch_size": batch_size}
    for label, amp in (("fp32", False), ("amp_bf16", True)):
        try:
            model = build_model(task, architecture="tabicl").to(device)
            model.train()
            opt = torch.optim.SGD(model.parameters(), lr=1e-6)
            scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp and device.startswith("cuda")
                else nullcontext()
            )

            def step(m=model, o=opt, s=scaler, c=ctx):
                o.zero_grad(set_to_none=True)
                with c:
                    loss = m(X, y).float().square().mean()
                s.scale(loss).backward()
                s.step(o)
                s.update()

            sec = _time(step, max(3, n // 4), 2, torch)
            out[f"{label}_step_ms"] = round(sec * 1000, 2)
        except Exception as exc:  # noqa: BLE001
            out[f"{label}_error"] = f"{type(exc).__name__}: {exc}"
    a, b = out.get("fp32_step_ms"), out.get("amp_bf16_step_ms")
    if a and b:
        out["amp_slowdown_vs_fp32"] = round(b / a, 2)
    return out


def profile_step(torch: Any, device: str, task: str, batch_size: int, amp: bool) -> dict[str, Any]:
    """Name the kernels. `torch.profiler` on a handful of real training steps, top ops by CUDA
    time, run once with AMP and once without.

    THE LAST RESORT, and the right one. Four explanations for the B200 running training ~15x
    slower than a benchmark of the same model, prior, loader and batch have now been asserted
    — the card, the prior, Muon, the wheel — and every one was wrong, because each was inferred
    from a difference someone noticed rather than measured. A profile does not have opinions:
    it says which kernel consumed the time, and if one op dominates on one card and not the
    other, that is the answer with no inference step in between.
    """
    from torch.profiler import ProfilerActivity, profile

    from src.models.architecture import build_model, is_available

    if not is_available("tabicl"):
        return {"error": "upstream tabicl not installed"}
    if not device.startswith("cuda"):
        return {"skipped": "profiling is only meaningful on CUDA"}

    rows, feats, train_size = bench_shape(task)
    X = torch.randn(batch_size, rows, feats, device=device)
    y = (
        torch.rand(batch_size, train_size, device=device)
        if task == "lgd"
        else (torch.rand(batch_size, train_size, device=device) > 0.7).float()
    )
    model = build_model(task, architecture="tabicl").to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-6)
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else nullcontext()
    )

    def step():
        opt.zero_grad(set_to_none=True)
        with ctx:
            loss = model(X, y).float().square().mean()
        loss.backward()
        opt.step()

    for _ in range(3):          # warm up: the first steps compile and allocate
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            step()
        torch.cuda.synchronize()

    rank = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    top = []
    for ev in sorted(prof.key_averages(), key=lambda e: -e.cuda_time_total)[:12]:
        top.append({
            "op": ev.key[:60],
            "cuda_ms": round(ev.cuda_time_total / 1000.0, 2),
            "calls": ev.count,
        })
    total = sum(e.cuda_time_total for e in prof.key_averages()) / 1000.0
    return {"amp": amp, "total_cuda_ms_over_5_steps": round(total, 1), "top_ops": top,
            "table": rank}


def bench_optimizers(
    torch: Any, device: str, task: str, n_steps: int, batch_size: int
) -> dict[str, Any]:
    """Time one optimiser step per optimiser, on tensors already on the GPU.

    THE RUNG THAT WAS MISSING, and it was the answer. The first benchmark used plain SGD and
    reported the B200 as 1.8x FASTER end to end than the free RTX 5000 Ada — while real
    training on the same B200 ran at 0.53 steps/s against the free card's 6.6. The only
    difference was the optimiser: `config/Exp1_*.yaml` sets `optimizer: muon`.

    Muon orthogonalises each weight matrix with a Newton-Schulz iteration — a chain of small
    matmuls, per matrix, per step. That is latency-bound rather than throughput-bound, so a
    card with enormous peak FLOPs can still lose badly on it, and no amount of `num_workers`
    or a faster GPU will help. This row separates it from everything else.
    """
    from src.models.architecture import build_model, is_available
    from src.train.optim import build_optimizer

    if not is_available("tabicl"):
        return {"error": "upstream tabicl not installed"}

    rows, feats, train_size = bench_shape(task)
    X = torch.randn(batch_size, rows, feats, device=device)
    y = (
        torch.rand(batch_size, train_size, device=device)
        if task == "lgd"
        else (torch.rand(batch_size, train_size, device=device) > 0.7).float()
    )

    out: dict[str, Any] = {"batch_size": batch_size}
    for name in ("adamw", "muon"):
        try:
            model = build_model(task, architecture="tabicl").to(device)
            model.train()
            opt = build_optimizer(model, {"optimizer": name, "lr": 1e-4, "muon_lr": 8e-4})

            def step(m=model, o=opt):
                o.zero_grad(set_to_none=True)
                m(X, y).float().square().mean().backward()
                o.step()

            sec = _time(step, max(3, n_steps // 4), 2, torch)
            out[f"{name}_step_ms"] = round(sec * 1000, 2)
            out[f"{name}_steps_per_s"] = round(1.0 / sec, 2)
        except Exception as exc:  # noqa: BLE001 — one optimiser must not sink the report
            out[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    a, m = out.get("adamw_step_ms"), out.get("muon_step_ms")
    if a and m:
        out["muon_slowdown_vs_adamw"] = round(m / a, 2)
    return out


def bench_end_to_end(
    torch: Any, device: str, task: str, n_steps: int, workers: int, batch_size: int
) -> dict[str, Any]:
    """The real thing: real DataLoader, real workers, real optimiser steps.

    This is the number the training log reports. Comparing it against
    `max_steps_per_s_if_data_were_free` is the whole point of the script — if the ceiling is
    high and this is low, the GPU spent the run waiting.
    """
    from src.models.architecture import build_model, is_available
    from src.prior.dataset import build_loader

    if not is_available("tabicl"):
        return {"error": "upstream tabicl not installed"}
    try:
        model = build_model(task, architecture="tabicl").to(device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-6)
        loader = build_loader(_prior_cfg(task), task, batch_size, seed=0, num_workers=workers)

        it = iter(loader)
        next(it)  # worker start-up is not part of the steady state
        _sync(torch)
        t0 = time.perf_counter()
        done = 0
        for _ in range(n_steps):
            X, y, train_size = next(it)
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(X, y[:, :train_size])
            out.float().square().mean().backward()
            opt.step()
            done += 1
        _sync(torch)
        sec = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "workers": workers,
        "steps": done,
        "seconds": round(sec, 2),
        "steps_per_s": round(done / sec, 3) if sec else None,
        "datasets_per_s": round(done * batch_size / sec, 2) if sec else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=("lgd", "pd"), default="pd")
    ap.add_argument("--steps", type=int, default=20, help="timed repetitions per measurement")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--prior-batches", type=int, default=10)
    # DEFAULTS TO THE CONFIG, not to 4. The default WAS 4, from when `train.batch_size` was 4,
    # and it stayed 4 after the config moved to upstream's 64 — so every benchmark since has
    # measured a batch the experiments do not use. Batch 4 is the exact regime the 17-08 run
    # showed puts a B200 at 3.5 % utilisation, so it reports starvation by construction.
    ap.add_argument("--batch-size", type=int, default=None,
                    help="datasets per step; default: train.batch_size from this task's config")
    ap.add_argument("--workers", type=int, default=None,
                    help="for the end-to-end row; defaults to $SLURM_CPUS_PER_TASK - 1")
    ap.add_argument("--skip-prior", action="store_true", help="GPU rows only")
    ap.add_argument("--skip-model", action="store_true", help="raw compute rows only")
    ap.add_argument("--skip-end-to-end", action="store_true")
    ap.add_argument(
        "--profile", action="store_true",
        help="run torch.profiler on real steps and print the top ops by CUDA time, with and "
             "without AMP. Slow, and the definitive answer when a timing gap has no obvious "
             "cause: it names the kernel instead of inferring it.",
    )
    ap.add_argument("--json", default=None, help="also write the results here")
    args = ap.parse_args()

    if args.batch_size is None:
        args.batch_size = int(_train_cfg(args.task).get("batch_size", 64))

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict[str, Any] = {"device_info": describe_device(torch), "task": args.task}

    print("=" * 74)
    print(" GPU DIAGNOSTIC — is the chip slow, or is it waiting for data?")
    print("=" * 74)
    for k, v in results["device_info"].items():
        print(f"  {k:28s} {v}")
    if results["device_info"].get("has_kernels_for_this_card") is False:
        print("\n  *** THIS WHEEL HAS NO KERNELS FOR THIS CARD. CUDA will JIT-compile PTX,")
        print("      which is silent and slow. This alone can explain everything below. ***")
    print()

    print("--- 1/2. raw compute (no model, no data) " + "-" * 32)
    results["matmul"] = bench_matmul(torch, device, args.steps, args.warmup)
    results["attention"] = bench_attention(torch, device, args.steps, args.warmup)
    for k, v in {**results["matmul"], **results["attention"]}.items():
        print(f"  {k:44s} {v}")
    print()

    if not args.skip_model:
        print("--- 3/4. the real model, tensors already on the GPU " + "-" * 22)
        results["model"] = bench_model(
            torch, device, args.task, args.steps, args.warmup, args.batch_size
        )
        for k, v in results["model"].items():
            print(f"  {k:44s} {v}")
        print()

    if not args.skip_prior:
        print("--- 5. prior generation (CPU, ONE worker, no GPU) " + "-" * 24)
        results["prior"] = bench_prior(args.task, args.prior_batches, args.batch_size)
        for k, v in results["prior"].items():
            print(f"  {k:44s} {v}")
        print()

    if not args.skip_model:
        print(f"--- 4a. AMP at the REAL batch size ({args.batch_size}) " + "-" * 26)
        results["amp"] = bench_amp(torch, device, args.task, args.steps, args.batch_size)
        for k, v in results["amp"].items():
            print(f"  {k:44s} {v}")
        print()

    if not args.skip_model:
        print("--- 4b. OPTIMISER STEP — the row that explains the B200 " + "-" * 18)
        results["optimizers"] = bench_optimizers(
            torch, device, args.task, args.steps, args.batch_size
        )
        for k, v in results["optimizers"].items():
            print(f"  {k:44s} {v}")
        print()

    if not args.skip_end_to_end:
        import os

        workers = args.workers
        if workers is None:
            workers = max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1)
        print(f"--- 6. end to end, {workers} workers (what the training log reports) " + "-" * 8)
        results["end_to_end"] = bench_end_to_end(
            torch, device, args.task, max(10, args.steps), workers, args.batch_size
        )
        for k, v in results["end_to_end"].items():
            print(f"  {k:44s} {v}")
        print()

    if args.profile:
        for amp in (False, True):
            label = "AMP bf16" if amp else "fp32"
            print(f"--- PROFILE, {label}, batch {args.batch_size} " + "-" * 30)
            res = profile_step(torch, device, args.task, args.batch_size, amp)
            results[f"profile_{'amp' if amp else 'fp32'}"] = {
                k: v for k, v in res.items() if k != "table"   # the table is for the log only
            }
            if "table" in res:
                print(res["table"])
                print(f"  TOTAL CUDA over 5 steps: {res['total_cuda_ms_over_5_steps']} ms")
            else:
                print(f"  {res}")
            print()

    # -- the reading -----------------------------------------------------------
    print("=" * 74)
    model = results.get("model") or {}
    prior = results.get("prior") or {}
    ceiling = model.get("max_steps_per_s_if_data_were_free")
    if ceiling:
        print(f"  GPU ceiling with free data      : {ceiling} steps/s")
    if prior.get("datasets_per_s_one_worker"):
        per_worker = prior["datasets_per_s_one_worker"]
        print(f"  Prior, one worker               : {per_worker} datasets/s")
        for w in (7, 23):
            print(f"    x{w:2d} workers -> {per_worker * w / args.batch_size:6.2f} steps/s "
                  f"({args.batch_size} datasets/step)")
    # The REALISTIC ceiling, not the SGD/fp32 one. Training runs AMP and Muon, and Muon adds a
    # Newton-Schulz chain per weight matrix per step that no batch size amortises away. Comparing
    # end-to-end against the plain fwd+bwd number overstates the headroom and reads as
    # starvation; on 20-08-2026 it reported "20% of the GPU, STARVED, more cores" while its own
    # prior extrapolation two lines above said the workers could supply 13x what was measured.
    amp_ms = (results.get("amp") or {}).get("amp_bf16_step_ms")
    opt = results.get("optimizers") or {}
    muon_overhead = (
        opt["muon_step_ms"] - opt["adamw_step_ms"]
        if opt.get("muon_step_ms") and opt.get("adamw_step_ms") else 0.0
    )
    realistic = 1000.0 / (amp_ms + muon_overhead) if amp_ms else None
    if realistic:
        print(f"  GPU ceiling, AMP + Muon         : {realistic:.2f} steps/s "
              f"({amp_ms:.1f} ms step + {muon_overhead:.1f} ms Muon)")

    e2e = (results.get("end_to_end") or {}).get("steps_per_s")
    if e2e:
        print(f"  Measured end to end             : {e2e} steps/s")
        ref = realistic or ceiling
        label = "AMP + Muon" if realistic else "plain fwd+bwd"
        if ref:
            pct = 100.0 * e2e / ref
            print(f"  -> reaching {pct:.0f}% of the {label} ceiling at batch {args.batch_size}")
            supply = (
                prior["datasets_per_s_one_worker"] * (results.get("end_to_end") or {}).get(
                    "workers", 0) / args.batch_size
                if prior.get("datasets_per_s_one_worker") else None
            )
            if pct >= 60:
                print("     COMPUTE-BOUND: the GPU is the limit. A faster card would help;")
                print("     more cores would not.")
            elif supply and supply < ref:
                print("     STARVED BY THE PRIOR: the workers cannot supply the GPU. More")
                print("     cores, or a cheaper prior — not a bigger GPU.")
            else:
                print("     NEITHER CEILING EXPLAINS THIS. The workers can supply "
                      f"{supply:.1f} steps/s and the GPU can take {ref:.1f}, yet the measured")
                print("     rate is below both. Suspect DataLoader overhead, IPC of the padded")
                print("     (batch, rows, max_features) tensors, or core contention. Run with")
                print("     --profile before believing any story about it.")
    print()
    print("  HOW TO READ IT: run this on both cards and compare row by row.")
    print("    matmul/attention differ    -> the CARD or the wheel. Change hardware.")
    print("    those match, model differs -> a kernel this model needs is missing here.")
    print("    all match, training differs-> the GPU was never the problem; it is the")
    print("                                  prior, and only more cores will help.")
    print(f"  Shape measured: {bench_shape(args.task)} (rows, features, train_size), "
          f"batch {args.batch_size}, from config/Exp1_*.yaml.")
    print("=" * 74)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
