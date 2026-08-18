"""Find WHERE a trained checkpoint starts producing non-finite values.

    python scripts/diagnose_nan.py
    python scripts/diagnose_nan.py --datasets 0005.base_modelisation
    python scripts/diagnose_nan.py --task pd --checkpoint <path>

WHY THIS EXISTS, AND WHY IT RUNS ON THE CLUSTER

Our LGD checkpoints return **100% NaN predictions on 4 of 7 real datasets** while `axa` scores
normally. Three explanations are already ruled out by measurement:

  * not the data         — the imputed input contains no non-finite value
  * not the architecture — an UNTRAINED model returns 0% NaN on the same four datasets
  * not divergence       — weight norms are flat across training (col 61.4->61.7, icl 174->175)

So it comes from the trained weights meeting a particular input, and the only way to see that
is to run the model and watch. A checkpoint is ~229 MB, which is awkward to move between
machines; a log is not. This walks the network with forward hooks and prints the FIRST module
whose output goes non-finite, plus the activation magnitudes leading up to it.

Read the output top-down. The first `-> NON-FINITE` line names the culprit, and the `absmax`
column on the lines above it says whether the run-up was an overflow (magnitudes climbing) or a
division by something that reached zero (magnitudes collapsing).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

# THREAD LIMITS, SET BEFORE TORCH IS IMPORTED — the numeric libraries read these at import and
# ignore later changes.
#
# A LOGIN NODE CAPS THREADS PER USER. Unbounded, torch opens one OMP thread per core on a
# machine with dozens of cores shared by everyone logged in, and the second dataset died with
# `std::system_error ... Resource temporarily unavailable` — that is `pthread_create` returning
# EAGAIN, not anything about the model. Four threads is ample: this does ONE forward pass per
# dataset and is not trying to be fast.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("CREDITICL_DIAG_THREADS", "4"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stats(t: Any) -> str:
    """Magnitude summary of one tensor, or a short note when it is not one."""
    import torch

    if not isinstance(t, torch.Tensor):
        return f"(not a tensor: {type(t).__name__})"
    f = t.detach().float()
    finite = torch.isfinite(f)
    n_bad = int((~finite).sum())
    if not bool(finite.any()):
        return f"shape={tuple(t.shape)} ALL {n_bad} values non-finite"
    vals = f[finite]
    tail = f"  NON-FINITE {n_bad}/{f.numel()}" if n_bad else ""
    return (
        f"shape={str(tuple(t.shape)):<20} absmax={float(vals.abs().max()):10.3e} "
        f"absmean={float(vals.abs().mean()):10.3e} std={float(vals.std()):10.3e}{tail}"
    )


def sdpa_context(backend: str, log):
    """A context forcing one attention kernel, or a no-op for `auto`.

    PyTorch chooses a CUDA backend by itself — flash, mem-efficient, cuDNN or math — and the
    choice depends on shapes and dtype. CPU only ever has `math`. That is the leading suspect
    for a model that is finite on CPU and all-NaN on CUDA with identical weights and inputs,
    and upstream evidently regards the choice as load-bearing: `--use_flash_attn3 False` in
    stage 1, `True` in stages 2-3.
    """
    import contextlib


    if backend == "auto":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover — older torch
        log(f"    (torch too old for sdpa_kernel; ignoring --sdpa-backend {backend})")
        return contextlib.nullcontext()
    names = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", SDPBackend.MATH),
    }
    return sdpa_kernel(names[backend])


def walk(model: Any, x: Any, y: Any, log) -> None:
    """Forward pass with a hook on every leaf module, reporting the first bad output."""
    import torch

    seen: list[tuple[str, str, bool]] = []

    def make_hook(name: str):
        def fn(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                tensors = [out]
            elif isinstance(out, (tuple, list)):
                tensors = [o for o in out if isinstance(o, torch.Tensor)]
            else:
                tensors = []
            bad = any(not bool(torch.isfinite(t).all()) for t in tensors)
            text = "; ".join(_stats(t) for t in tensors) or "(no tensor output)"
            seen.append((name, text, bad))

        return fn

    # Leaves only: a container's output just repeats its last child's, which doubles the
    # report and makes the first-bad index point at the wrong place.
    handles = [
        m.register_forward_hook(make_hook(n))
        for n, m in model.named_modules()
        if n and not list(m.children())
    ]
    try:
        with torch.no_grad():
            out = model(x, y)
    finally:
        for h in handles:
            h.remove()

    first_bad = next((i for i, (_, _, bad) in enumerate(seen) if bad), None)
    if first_bad is None:
        log(f"  every module finite. FINAL OUTPUT: {_stats(out)}")
        return

    log(f"  {len(seen)} modules ran; FIRST non-finite output is #{first_bad}")
    log("  the ten modules leading up to it:")
    for name, stat, bad in seen[max(0, first_bad - 9) : first_bad + 1]:
        marker = "-> NON-FINITE" if bad else "   ok        "
        log(f"    {marker} {name:<50} {stat}")
    log(f"  FINAL OUTPUT: {_stats(out)}")


def report_weights(model: Any, log) -> None:
    """A non-finite parameter means training diverged, and nothing about the input matters."""
    import torch

    log("")
    log("-- WEIGHTS " + "-" * 66)
    ranked = []
    for name, p in model.named_parameters():
        f = p.detach().float()
        n_bad = int((~torch.isfinite(f)).sum())
        mx = float(f.abs().max()) if bool(torch.isfinite(f).any()) else float("inf")
        ranked.append((mx, name, n_bad))
        if n_bad:
            log(f"  NON-FINITE PARAMETER {name}: {n_bad}/{f.numel()}")
    ranked.sort(reverse=True)
    log("  largest-magnitude parameters (a runaway weight shows up here):")
    for mx, name, n_bad in ranked[:8]:
        log(f"    absmax={mx:10.3e}  {name}{'  NON-FINITE' if n_bad else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=("lgd", "pd"), default="lgd")
    ap.add_argument("--checkpoint", default=None, help="default: newest matching --task")
    ap.add_argument("--datasets", default=None, help="comma-separated slugs; default all")
    ap.add_argument("--context-rows", type=int, default=512, help="as the evaluation uses")
    ap.add_argument("--max-test-rows", type=int, default=512)
    ap.add_argument(
        "--device", default=None,
        help="cpu | cuda. Default: cuda when available. THIS MATTERS — the same weights and "
             "the same data were finite on CPU and all-NaN in training on CUDA, which is what "
             "--sdpa-backend exists to pin down.",
    )
    ap.add_argument(
        "--sdpa-backend", default="all",
        choices=("all", "auto", "math", "flash", "efficient", "cudnn"),
        help="which scaled-dot-product-attention kernel to force. `all` tries each in turn and "
             "is the point of the tool on a GPU: PyTorch picks a CUDA backend on its own, CPU "
             "only ever has `math`, and upstream disables FlashAttention in stage 1 "
             "(`--use_flash_attn3 False`) while enabling it in stages 2-3 — so they treat the "
             "choice as load-bearing too.",
    )
    ap.add_argument(
        "--threads", type=int, default=int(os.environ.get("CREDITICL_DIAG_THREADS", "4")),
        help="torch CPU threads. Kept small because a LOGIN NODE limits threads per user and "
             "an unbounded torch dies with 'Resource temporarily unavailable'.",
    )
    args = ap.parse_args()

    import numpy as np
    import torch

    torch.set_num_threads(max(1, args.threads))
    # Already initialised is harmless, and not worth failing a diagnostic over.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)

    from src.data.discovery import list_datasets
    from src.data.pipeline import load_processed
    from src.eval.crediticl_baseline import find_our_checkpoints, load_our_checkpoint
    from src.utils.logging_setup import log_section, setup_logging
    from src.utils.paths import logs_dir

    log_obj, _, log_path = setup_logging(
        f"diagnose_nan_{args.task}", logs_dir(), console=True
    )

    def log(message: str) -> None:
        log_obj.info("%s", message)

    log_section(log_obj, f"NaN DIAGNOSIS - {args.task}")

    ckpt = args.checkpoint
    if ckpt is None:
        found = [p for p in find_our_checkpoints() if f"_{args.task}__" in str(p)]
        if not found:
            log(f"no checkpoint for task={args.task}. Pass --checkpoint explicitly.")
            return 1
        ckpt = max(found, key=lambda p: p.stat().st_mtime)
        log(f"using the newest checkpoint: {ckpt}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_our_checkpoint(ckpt, device)
    device = next(model.parameters()).device
    log(
        f"checkpoint step={meta.get('step')} architecture={meta.get('architecture')} "
        f"credit_fraction={meta.get('credit_fraction')} device={device}"
    )

    report_weights(model, log)

    slugs = (
        args.datasets.split(",")
        if args.datasets
        else [str(e) for e in list_datasets(args.task)]
    )
    rng = np.random.default_rng(0)
    for slug in slugs:
        log("")
        log("=" * 78)
        try:
            ds = load_processed(args.task, slug)
        except Exception as exc:  # noqa: BLE001 — report and keep going
            log(f"{slug}: cannot load ({type(exc).__name__}: {exc})")
            continue

        X = np.asarray(ds.X, np.float32)
        y = np.asarray(ds.y, np.float32)
        n = len(X)
        ctx_n = min(args.context_rows, max(8, n // 2))
        perm = rng.permutation(n)
        ci, ti = perm[:ctx_n], perm[ctx_n : ctx_n + args.max_test_rows]
        Xc, yc, Xt = X[ci], y[ci], X[ti]
        # Exactly what the progress scorer does: impute from the CONTEXT only.
        med = np.nan_to_num(np.nanmedian(Xc, axis=0))
        Xc = np.where(np.isfinite(Xc), Xc, med)
        Xt = np.where(np.isfinite(Xt), Xt, med)
        x_in = np.concatenate([Xc, Xt])

        log(f"{slug}: rows={n} features={X.shape[1]} context={ctx_n} test={len(ti)}")
        log(
            f"  INPUT  absmax={np.abs(x_in).max():.3e} "
            f"non-finite={int((~np.isfinite(x_in)).sum())} "
            f"zero-variance-cols={int((x_in.std(axis=0) < 1e-12).sum())}"
        )
        log(
            f"  TARGET absmax={np.abs(yc).max():.3e} "
            f"non-finite={int((~np.isfinite(yc)).sum())} "
            f"unique={len(np.unique(yc))}"
        )

        import gc

        gc.collect()
        xt = torch.from_numpy(x_in).unsqueeze(0).to(device)
        yt = torch.from_numpy(yc).unsqueeze(0).to(device)
        backends = (
            ["math", "flash", "efficient", "cudnn"]
            if args.sdpa_backend == "all" and str(device).startswith("cuda")
            else ["math"] if args.sdpa_backend == "all" else [args.sdpa_backend]
        )
        for backend in backends:
            log(f"  --- attention backend: {backend} ---")
            try:
                with sdpa_context(backend, log):
                    walk(model, xt, yt, log)
            except Exception as exc:  # noqa: BLE001 — an unavailable kernel is informative
                log(f"    unavailable here: {type(exc).__name__}: {exc}")

    log("")
    log(f"log file -> {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
