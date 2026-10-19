"""Final OOD retention and component-ablation tables, separate from training monitors."""
from __future__ import annotations

import re
import warnings
import pandas as pd

from src.eval.comparisons import identities, paired_ablation_effects, paired_prior_effects, paired_reference_effects, require_balanced_cells
from src.eval.protocol import PROTOCOL
from src.eval.results import _read_frames, load_results
from src.utils import paths
from src.utils.config import expand_with_seeds, load, run_name


def _complete(df, expected_models, datasets):
    if df is None:
        return None
    try:
        require_balanced_cells(df, datasets)
        if set(identities(df)) != set(expected_models):
            raise ValueError("missing model or training seed")
    except ValueError as exc:
        warnings.warn(f"Final analysis withheld: {exc}", stacklevel=2)
        return None
    return df


def load_ood_results(track: str, exp: str = "exp1") -> pd.DataFrame | None:
    """Final checkpoints and references on the original native OOD tasks."""
    from src.eval.benchmark_status import REFERENCE_MODELS
    from src.eval.ood import list_ood_datasets

    out = paths.benchmark_dir(int(exp.removeprefix("exp")), "ood")
    ref = paths.benchmark_dir(paths.REFERENCE, "ood")
    pattern = re.compile(rf"ood_results_{exp}bench_{track}_a\d+\.csv$")
    files = sorted(p for p in out.glob(f"ood_results_{exp}bench_{track}_a*.csv") if pattern.fullmatch(p.name))
    files += sorted(ref.glob(f"ood_results_reference_{track}_*.csv"))
    df = _read_frames(files, PROTOCOL)
    cfg = load(paths.config_path(f"{exp.capitalize()}_{track.upper()}"), allow_placeholders=True)
    expected = {run_name(r) for r in expand_with_seeds(cfg)} | set(REFERENCE_MODELS.split(","))
    datasets = [d.name for d in list_ood_datasets("classification" if track == "pd" else "regression")]
    if not datasets:
        return None
    return _complete(df, expected, datasets)


def load_ablation_results(track: str) -> pd.DataFrame | None:
    """All five component removals, plus main 50% runs, for seed-paired comparisons."""
    main = load_results(track)
    if main is None:
        return None
    out = paths.benchmark_dir(1, track)
    pattern = re.compile(rf"results_exp1ablationsbench_{track}_a\d+\.csv$")
    files = sorted(p for p in out.glob(f"results_exp1ablationsbench_{track}_a*.csv") if pattern.fullmatch(p.name))
    df = _read_frames(files, PROTOCOL)
    cfg = load(paths.config_path(f"Exp1_{track.upper()}_ablations"))
    df = _complete(df, {run_name(r) for r in expand_with_seeds(cfg)},
                   cfg["eval"]["dev_datasets"] + cfg["eval"]["holdout_datasets"])
    if df is None:
        return None
    main = main[identities(main).str.match(rf"exp1_{track}__prior-credit_fraction=0p5__s\d+$")]
    return pd.concat([main, df], ignore_index=True)


def effects(track: str, exp: str = "exp1", *, ablations: bool = False,
            metric: str | None = None, reference: bool = False) -> pd.DataFrame:
    metric = metric or ("log_loss" if track == "pd" else "r2")
    df = load_ablation_results(track) if ablations else load_ood_results(track, exp)
    if df is None:
        return pd.DataFrame()
    if ablations:
        cfg = load(paths.config_path(f"Exp1_{track.upper()}"))
        df = df[df.dataset.isin(cfg["eval"]["holdout_datasets"])]
        return paired_ablation_effects(df, metric, track)
    return (paired_reference_effects if reference else paired_prior_effects)(df, metric)


def summary(track: str, exp: str = "exp1", *, ablations: bool = False) -> str:
    title = "COMPONENT REMOVALS vs full 50% prior" if ablations else "FINAL OOD RETENTION vs matched 0% control"
    lines = [title, "Differences are treatment minus control; lower is better for loss/error metrics."]
    metrics = ("roc_auc", "log_loss") if track == "pd" else (("r2", "crps", "boundary_mass_abs_err") if ablations else ("r2", "crps"))
    for metric in metrics:
        table = effects(track, exp, ablations=ablations, metric=metric)
        lines.append(f"{metric}: " + ("not yet completely benchmarked" if table.empty else "\n" + table.to_string(index=False)))
        if not ablations:
            table = effects(track, exp, metric=metric, reference=True)
            lines.append("Against each architecture's released weights: " +
                         ("not yet completely benchmarked" if table.empty else "\n" + table.to_string(index=False)))
    return "\n".join(lines)
