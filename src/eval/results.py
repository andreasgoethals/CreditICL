"""Load complete, protocol-consistent benchmark tables for analysis and selection."""
from __future__ import annotations
import re
from typing import Any
import numpy as np
import pandas as pd
from src.utils import paths
_CTX_FILE = re.compile(r"_ctx\d+\.csv$")

def _read_frames(files: list[Any], protocol: int | None, context: int | None = None) -> pd.DataFrame | None:
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # Full-context rows carry no cap; capped ones only when asked for, and only that cap.
    cap = pd.to_numeric(df["context_cap"], errors="coerce") if "context_cap" in df \
        else pd.Series(np.nan, index=df.index)
    df = df[cap.isna()] if context is None else df[cap.eq(int(context))]
    # Rows written before the protocol column existed are protocol 2 (one split per seed,
    # context capped at 1,024 rows).
    version = pd.to_numeric(df["protocol"], errors="coerce").fillna(2) if "protocol" in df \
        else pd.Series(2, index=df.index)
    want = int(version.max()) if protocol is None and len(df) else protocol
    df = df[version.eq(want)].assign(protocol=want)
    if "status" in df and not df["status"].eq("ok").all():
        import warnings
        warnings.warn("Benchmark contains failed or skipped cells; comparisons are withheld", stacklevel=2)
        return None
    if want is not None and want >= 6 and len(df):
        from src.eval.comparisons import require_balanced_cells
        try:
            require_balanced_cells(df)
        except ValueError as exc:
            import warnings
            warnings.warn(f"Incomplete benchmark; comparisons are withheld: {exc}", stacklevel=2)
            return None
    return df if len(df) else None


def _raw_config(track: str, exp: str) -> dict:
    """Read optional evaluation overrides without requiring development-selection receipts."""
    import yaml

    path = paths.config_path(f"{exp.capitalize()}_{track.upper()}")
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def context_caps(track: str, exp: str = "exp1") -> list[int]:
    """The capped-context evaluations the experiment's config asks for (`eval.context_caps`)."""
    return [int(c) for c in ((_raw_config(track, exp).get("eval") or {}).get("context_caps") or [])]


def headline_step(track: str, exp: str = "exp1") -> int | None:
    """The checkpoint an experiment REPORTS: its config's `eval.headline_step`, or None for the
    final one (no such key, a `FILL_FROM_*` placeholder not filled yet, or the last step).

    No config sets one today (both Exp2 recipes fix 10,000 steps); every saved checkpoint is
    scored, so a reported earlier one would be a choice of file, not a new run.
    """
    cfg = _raw_config(track, exp)
    step = (cfg.get("eval") or {}).get("headline_step")
    if step is None or (isinstance(step, str) and not step.strip().isdigit()):
        return None
    step = int(step)
    return None if step >= int((cfg.get("train") or {}).get("max_steps", 0) or 0) else step


def load_results(track: str, exp: str = "exp1", protocol: int | None = None,
                 context: int | None = None) -> pd.DataFrame | None:
    """Per-(dataset, model, seed, fold) results of the REPORTED checkpoints for `exp` on `track`,
    or `None` if none exist: the final checkpoint, or the one `headline_step` names.

    Concatenates the experiment's arm files (`results_<exp>bench_<track>_a<i>[_s<step>].csv`) with
    the shared reference (`results_reference_<track>_<model>.csv`). Other experiments and other
    checkpoints are excluded, and so is every protocol but one: the newest present when `protocol`
    is None (the `protocol` column says which), else the one asked for. `context` reads the
    capped-context evaluation instead (`..._ctx<cap>.csv`): every arm's FINAL checkpoint and the
    reference models that take a context, each with its context capped at `context` rows.
    """
    out = paths.benchmark_dir(paths.experiment_of(f"{exp}_"), track)
    ref = paths.benchmark_dir(paths.REFERENCE, track)
    if not out.exists() and not ref.exists():
        return None
    # The reported checkpoint's arm files, whatever grid wrote them: the protocol filter below
    # keeps one benchmark. The final checkpoint has no `_s<step>` suffix; the capped slots score
    # the final checkpoint only.
    if context is None:
        step = headline_step(track, exp)
        tail = f"_s{step}" if step is not None else ""
    else:
        tail = f"_ctx{int(context)}"
    wanted = re.compile(rf"results_{re.escape(exp)}bench_{re.escape(track)}_a\d+{tail}\.csv$")
    files = sorted(p for p in out.glob(f"results_{exp}bench_{track}_a*.csv") if wanted.match(p.name))
    files += sorted(p for p in ref.glob(f"results_reference_{track}_*.csv")
                    if (p.name.endswith(f"{tail}.csv") if context is not None
                        else not _CTX_FILE.search(p.name)))
    result = _read_frames(files, protocol, context)
    if result is not None and result.protocol.min() >= 6:
        from src.eval.benchmark_status import REFERENCE_MODELS
        from src.eval.comparisons import identities, require_balanced_cells
        from src.utils.config import expand_with_seeds, load, run_name
        cfg = load(paths.config_path(f"{exp.capitalize()}_{track.upper()}"), allow_placeholders=True)
        evaluation = cfg["eval"]
        try:
            require_balanced_cells(result, evaluation["dev_datasets"] + evaluation["holdout_datasets"])
        except ValueError as exc:
            import warnings
            warnings.warn(f"Incomplete benchmark; comparisons are withheld: {exc}", stacklevel=2)
            return None
        expected = {run_name(run) for run in expand_with_seeds(cfg)}
        expected.update(REFERENCE_MODELS.split(","))
        if set(identities(result)) != expected:
            import warnings
            warnings.warn("Benchmark still lacks complete model coverage; final comparisons are withheld", stacklevel=2)
            return None
    return result


def load_checkpoint_results(track: str, exp: str = "exp1", protocol: int | None = None) -> pd.DataFrame | None:
    """Every saved checkpoint of every arm — the final one and the `_s<step>` ones — with the
    step in `checkpoint_step`: the learning curve on real data, one point per saved checkpoint.
    One protocol only, as `load_results`."""
    out = paths.benchmark_dir(paths.experiment_of(f"{exp}_"), track)
    if not out.exists():
        return None
    files = sorted(p for p in out.glob(f"results_{exp}bench_{track}_a*.csv") if not _CTX_FILE.search(p.name))
    df = _read_frames(files, protocol)
    if df is None or "info_checkpoint_step" not in df:
        return None
    df = df.assign(checkpoint_step=pd.to_numeric(df["info_checkpoint_step"], errors="coerce"))
    df = df.dropna(subset=["checkpoint_step"])
    if len(df) and df.protocol.min() >= 6:
        from src.eval.comparisons import identities, require_balanced_cells
        from src.utils.config import expand_with_seeds, load, run_name
        cfg = load(paths.config_path(f"{exp.capitalize()}_{track.upper()}"), allow_placeholders=True)
        expected = {run_name(run) for run in expand_with_seeds(cfg)}
        try:
            require_balanced_cells(df, cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"])
            if set(identities(df)) != expected:
                raise ValueError("missing configuration or training seed")
        except ValueError as exc:
            import warnings
            warnings.warn(f"Incomplete checkpoint curves; comparisons are withheld: {exc}", stacklevel=2)
            return None
    return df


