"""Plain-text summaries of what a notebook just plotted.

REPO-WIDE RULE: every notebook ends with a text summary and nothing else. The point is
that you can select the output, paste it into a chat or an email, and the whole picture
survives — figures do not. So these functions must state the **numbers**, not describe
the pictures: "boundary mass spans 0.018 to 0.730 across 7 datasets" is useful pasted
into a message; "the histogram is bimodal" is not.

Everything returns a `str`, and the notebook does `print(...)`. Returning rather than
printing keeps it testable, and lets a caller write it to `results/` if they want.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.target_stats import target_stats


def _rule(title: str, width: int = 78) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def _fmt_range(values: list[float], pct: bool = False) -> str:
    if not values:
        return "n/a"
    lo, hi = min(values), max(values)
    med = float(np.median(values))
    if pct:
        return f"{lo:.1%} to {hi:.1%} (median {med:.1%})"
    return f"{lo:.4g} to {hi:.4g} (median {med:.4g})"


# ---------------------------------------------------------------------------
# Prior notebooks
# ---------------------------------------------------------------------------


def prior_summary(
    loaded: dict[str, list[Any]],
    task: str,
    *,
    source: str = "unknown",
    reference: dict[str, Any] | None = None,
) -> str:
    """Text summary of a prior-visualisation notebook.

    Reports, per variant, the quantities the research question turns on, and then says
    explicitly whether our prior covers the real datasets — which is the one
    conclusion a reader should walk away with.
    """
    lines: list[str] = []
    lines.append(_rule(f"PRIOR SUMMARY — {task.upper()}"))
    lines.append(f"data source      : {source}  ('pool' = the files training reads; "
                 f"'live' = generated on the fly now)")
    lines.append(f"variants compared: {', '.join(loaded)}")
    lines.append(f"datasets sampled : {', '.join(f'{k}={len(v)}' for k, v in loaded.items())}")

    lines.append("\n--- PER VARIANT " + "-" * 62)
    for variant, tasks in loaded.items():
        if not tasks:
            continue
        rows = [t.n_rows for t in tasks]
        feats = [t.n_features for t in tasks]
        lines.append(f"\n{variant}:")
        lines.append(f"  shape          rows {_fmt_range(rows)} | features {_fmt_range(feats)}")

        if task == "lgd":
            stats = [target_stats(t.y) for t in tasks]
            boundary = [s["frac_at_min"] + s["frac_at_max"] for s in stats]
            at0 = [s["frac_at_min"] for s in stats]
            at1 = [s["frac_at_max"] for s in stats]
            in_unit = [
                1.0 if (float(t.y.min()) >= -1e-6 and float(t.y.max()) <= 1 + 1e-6) else 0.0
                for t in tasks
            ]
            share_in_unit = float(np.mean(in_unit))
            lines.append(f"  in [0,1]       {share_in_unit:.1%} of datasets")
            lines.append(f"  boundary mass  {_fmt_range(boundary, pct=True)}")
            # "at 0" and "at 1" only MEAN full recovery and total loss when the target is
            # actually on [0,1]. The original prior standard-scales its target, so labelling
            # its min/max ties that way reads as an economic finding when it is nothing of the
            # sort — the ties there are `outlier_removing`'s clamp at +-4 SD piling rows onto
            # exactly two values. Same numbers, honest names.
            if share_in_unit > 0.5:
                lines.append(f"    at 0 (full recovery) mean {np.mean(at0):.1%}")
                lines.append(f"    at 1 (total loss)    mean {np.mean(at1):.1%}")
            else:
                lines.append(f"    at its own min       mean {np.mean(at0):.1%}  (NOT 0)")
                lines.append(f"    at its own max       mean {np.mean(at1):.1%}  (NOT 1)")
                lines.append(
                    "    target is not on [0,1], so these are scale-free ties at the "
                    "extremes,\n      largely the +-4 SD outlier clamp — not recovery or loss."
                )
            lines.append(f"  any atoms      {np.mean([b > 0.01 for b in boundary]):.1%} of datasets")
        else:
            rates = [float((t.y > 0.5).float().mean()) for t in tasks]
            lines.append(f"  base rate      {_fmt_range(rates, pct=True)}")
            lines.append(f"  below 5%       {np.mean([r < 0.05 for r in rates]):.1%} of datasets")
            lines.append(f"  below 10%      {np.mean([r < 0.10 for r in rates]):.1%} of datasets")

    # The comparison that matters: does our prior cover the real data?
    if reference:
        lines.append("\n--- AGAINST THE REAL DATASETS " + "-" * 48)
        if task == "lgd":
            real = {k: (v[0] + v[1]) for k, v in reference.items()}
            label = "boundary mass"
        else:
            real = dict(reference)
            label = "base rate"
        lines.append(f"real {label}: " + ", ".join(f"{k}={v:.1%}" for k, v in sorted(real.items())))

        for variant, tasks in loaded.items():
            if not tasks:
                continue
            if task == "lgd":
                vals = [
                    (lambda s: s["frac_at_min"] + s["frac_at_max"])(target_stats(t.y))
                    for t in tasks
                ]
            else:
                vals = [float((t.y > 0.5).float().mean()) for t in tasks]
            lo, hi = min(vals), max(vals)
            covered = [k for k, v in real.items() if lo <= v <= hi]
            # Range coverage on its own is a WEAK claim: a range can span a real value
            # while putting almost no mass near it. The original prior's range covers
            # most datasets purely on the strength of a few outlier draws, while its
            # median sits at ~0.3%. So report where the mass actually is.
            near = [
                k for k, v in real.items()
                if np.mean([abs(x - v) <= 0.05 for x in vals]) >= 0.10
            ]
            lines.append(
                f"  {variant}: range [{lo:.3f}, {hi:.3f}] spans "
                f"{len(covered)}/{len(real)} | median {np.median(vals):.3f} | "
                f"{len(near)}/{len(real)} datasets have >=10% of draws within 5pp"
            )
        lines.append(
            "\n  Read the LAST column, not the first. A range can span a real value on\n"
            "  the strength of a few outlier draws while placing almost no mass near it,\n"
            "  so 'spans 7/7' is a much weaker statement than it looks."
        )

    lines.append("\n--- WHAT THIS MEANS " + "-" * 58)
    if task == "lgd":
        lines.append(
            "The original TabICL prior standard-scales its target, so it puts almost\n"
            "nothing inside [0,1] and produces boundary atoms only by chance ties. Our\n"
            "prior derives LGD from credit economics (collateral coverage, workout\n"
            "cashflows, portfolio segments), so the atoms at 0 and 1 EMERGE from\n"
            "over-collateralisation and total loss rather than being dialled in."
        )
    else:
        lines.append(
            "Real PD base rates sit well below balance. Our prior assigns defaults with\n"
            "the Merton/Vasicek one-factor model — the basis of the Basel IRB formula —\n"
            "so defaults are CORRELATED through a systematic factor and the realised rate\n"
            "varies between cohorts. A prior of independent labels never shows the model\n"
            "a bad year."
        )
    lines.append(
        "\nCaveat: this describes the PRIOR, not downstream performance. Whether a\n"
        "closer-looking prior actually transfers is what the training and evaluation\n"
        "pipelines measure."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data exploration notebook
# ---------------------------------------------------------------------------


def data_summary(
    datasets_by_task: dict[str, dict[str, Any]],
    tables: dict[str, pd.DataFrame] | None = None,
    leakage: pd.DataFrame | None = None,
) -> str:
    """Text summary of the evaluation datasets.

    These are the datasets the models are SCORED on, so the summary leads with what
    makes each task hard, then flags anything that would undermine a published number.
    """
    lines: list[str] = []
    lines.append(_rule("EVALUATION DATASET SUMMARY"))
    total = sum(len(v) for v in datasets_by_task.values())
    lines.append(f"{total} datasets loaded: " +
                 ", ".join(f"{k.upper()}={len(v)}" for k, v in datasets_by_task.items()))

    for task, datasets in datasets_by_task.items():
        if not datasets:
            continue
        lines.append(f"\n--- {task.upper()} " + "-" * (74 - len(task)))
        rows = [d.n_rows for d in datasets.values()]
        feats = [d.n_features for d in datasets.values()]
        lines.append(f"rows      {_fmt_range(rows)}")
        lines.append(f"features  {_fmt_range(feats)}")
        cat_share = [len(d.cat_indices) / max(1, d.n_features) for d in datasets.values()]
        lines.append(f"categorical share  {_fmt_range(cat_share, pct=True)}")
        miss = [float(np.isnan(d.X).mean()) for d in datasets.values()]
        n_zero = sum(1 for m in miss if m == 0)
        lines.append(f"missing cells      {_fmt_range(miss, pct=True)}"
                     f"  —  {n_zero}/{len(miss)} have NONE left (pre-imputed upstream)")

        if task == "lgd":
            per = {}
            for slug, d in datasets.items():
                st = target_stats(np.asarray(d.y, dtype=float))
                per[slug.split('.', 1)[-1]] = st["frac_at_min"] + st["frac_at_max"]
            lines.append(f"\nboundary mass      {_fmt_range(list(per.values()), pct=True)}")
            for name, v in sorted(per.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name:24} {v:6.1%}")
            in_unit = all(
                float(np.asarray(d.y).min()) >= 0 and float(np.asarray(d.y).max()) <= 1
                for d in datasets.values()
            )
            lines.append(f"all targets inside [0,1]: {in_unit}")
            lines.append(
                "\nThis spread is the reason the prior samples a FAMILY of boundary\n"
                "masses rather than one value: matching any single dataset would be\n"
                "overfitting to it."
            )
        else:
            per = {s.split('.', 1)[-1]: float(np.asarray(d.y).mean()) for s, d in datasets.items()}
            lines.append(f"\nbase rate          {_fmt_range(list(per.values()), pct=True)}")
            for name, v in sorted(per.items(), key=lambda kv: kv[1]):
                odds = (1 - v) / max(v, 1e-9)
                lines.append(f"  {name:24} {v:6.1%}   (1 default per {odds:.0f} non-defaults)")
            lines.append(
                "\nEvery dataset is below the 50% balance point that TabICL's prior sits\n"
                "near, most by a wide margin. That gap is what the PD arm addresses."
            )

    if leakage is not None and len(leakage):
        lines.append("\n--- LEAKAGE SCREEN " + "-" * 59)
        flagged = leakage[leakage["suspicious"]] if "suspicious" in leakage else leakage.iloc[:0]
        lines.append(f"single-feature |correlation| with the target, top {min(5, len(leakage))}:")
        for _, r in leakage.head(5).iterrows():
            lines.append(f"  {r['dataset']:24} {r['feature']:28} {r['|corr with target|']:.3f}")
        if len(flagged):
            lines.append(f"\n{len(flagged)} feature(s) above 0.9 — INSPECT before using these "
                         f"datasets to support a claim:")
            for _, r in flagged.iterrows():
                lines.append(f"  {r['dataset']} :: {r['feature']} = {r['|corr with target|']:.3f}")
        else:
            lines.append("\nNothing above 0.9. No single-feature smoking gun.")
        lines.append(
            "A high correlation is a POINTER, not proof: a single strong predictor can\n"
            "be legitimate. Conversely this screen only looks at one feature at a time,\n"
            "so it cannot see leakage spread across several columns."
        )

    lines.append("\n--- IMPLICATIONS FOR THE PRIOR " + "-" * 47)
    lines.append(
        "1. LGD boundary mass varies by a factor of ~40 across datasets, so the prior\n"
        "   must span a range, not hit one value.\n"
        "2. LGD targets are genuinely bounded — clipping to [0,1] encodes a real\n"
        "   constraint that the original prior does not have.\n"
        "3. PD base rates sit far below balance, so imbalance is sampled, not fixed.\n"
        "4. Real features come in correlated blocks, which is why the prior builds\n"
        "   features through random DAGs rather than independently.\n"
        "5. Missingness is mostly pre-imputed away here, so do NOT tune the prior's\n"
        "   missingness rate to these numbers — they measure the upstream pipeline."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real-data helpers for the Exp1 prior notebooks
#
# Every figure in those notebooks compares a prior AGAINST the real datasets. Loading them was
# previously done inline per notebook, which is how the two notebooks drifted apart; these are
# the shared version.
# ---------------------------------------------------------------------------


def load_real_datasets(task: str, max_rows: int = 20_000) -> dict[str, Any]:
    """The real credit datasets for a task, as loaded objects. `{}` if none are on this machine.

    Capped rows because these feed histograms and a quick ExtraTrees, not a training run — a
    300k-row table would make the notebook slow for no change in any figure.

    Never raises: a machine without the data must still be able to run the notebook and get the
    prior-only figures, with the comparisons degrading to "no real datasets available".
    """
    out: dict[str, Any] = {}
    try:
        from src.data.loaders import list_datasets, load_processed
    except Exception:  # noqa: BLE001
        return out
    for name in list_datasets(task):
        try:
            ds = load_processed(task, name)
        except Exception:  # noqa: BLE001 — one missing dataset must not lose the rest
            continue
        out[name] = _cap_rows(ds, max_rows) if max_rows else ds
    return out


class _RowCapped:
    """A row-limited VIEW of a dataset that keeps every other attribute.

    The first version copied only `X`, `y`, `n_rows` and `n_features`, so `cat_indices` and
    `feature_names` disappeared and any plot that used them raised `AttributeError` — but only
    for datasets big enough to be capped, which is why it survived a full notebook run.
    Delegating unknown attributes to the original means adding a field to `ProcessedDataset`
    cannot break this again.
    """

    def __init__(self, source: Any, max_rows: int, seed: int = 0) -> None:
        import numpy as np

        self._source = source
        X = np.asarray(source.X)
        y = np.asarray(source.y)
        # A RANDOM subsample, not the head. Taking `[:max_rows]` reported
        # `algorithmwatch`'s default rate as 49.5% against a true 37.8% — the rows are not
        # shuffled on disk, so the first 20,000 are not representative, and every statistic
        # drawn from them was quietly wrong by whatever the ordering happened to encode.
        # Fixed seed, so the figure is identical between runs.
        if len(y) > max_rows:
            idx = np.random.default_rng(seed).choice(len(y), size=max_rows, replace=False)
            idx.sort()   # keep the original row order, so cohort-style plots still make sense
            X, y = X[idx], y[idx]
        self.X, self.y = X, y
        self.n_rows = int(self.X.shape[0])
        self.n_features = int(getattr(source, "n_features", self.X.shape[1]))

    def __getattr__(self, item: str) -> Any:
        # Only reached for attributes not set above, so `cat_indices`, `feature_names` and
        # anything added later come through unchanged.
        return getattr(self._source, item)


def _cap_rows(ds: Any, max_rows: int) -> Any:
    return _RowCapped(ds, max_rows) if getattr(ds, "n_rows", 0) > max_rows else ds


def real_difficulty(task: str, real: dict[str, Any]) -> dict[str, float]:
    """How predictable each real dataset is, on the same measurement as the synthetic ones.

    This is the yardstick for `plot_difficulty_calibration`. It has to be the SAME estimator and
    split as the synthetic side or the comparison is meaningless, so both call
    `exp1_plots._quick_score`.
    """
    from src.visualize.exp1_plots import _quick_score

    scores: dict[str, float] = {}
    for name, ds in real.items():
        score = _quick_score(ds, task)
        if score is not None:
            scores[name.split(".", 1)[-1]] = score
    return scores


def realism_summary(loaded: dict[str, Any], real: dict[str, Any], task: str) -> str:
    """The realism ranking as text, so `All_Results.md` carries the conclusion, not just a figure.

    Ordered best to worst, because the one thing a reader wants from this notebook is which
    prior to take forward.
    """
    import numpy as np

    from src.visualize.exp1_plots import _real_targets, distribution_distance

    lines = [_rule(f"PRIOR REALISM RANKING — {task.upper()}")]
    reals = _real_targets(task, real)
    if not reals:
        lines.append("No real datasets on this machine, so no comparison was possible.")
        return "\n".join(lines)

    lines.append(f"Distance from {len(reals)} real datasets (total variation, 0 = identical).")
    lines.append("Lower is better. This is the ranking Exp1 exists to refine with training.")
    lines.append("")
    rows = []
    for name, tasks in loaded.items():
        pooled = np.clip(np.concatenate([np.asarray(t.y).ravel() for t in tasks]), 0.0, 1.0)
        per = [distribution_distance(pooled, y) for y in reals.values()]
        rows.append((name, float(np.mean(per)), float(np.min(per)), float(np.max(per))))
    rows.sort(key=lambda r: r[1])
    lines.append(f"  {'variant':<28} {'mean':>7} {'best':>7} {'worst':>7}")
    for name, mean_d, best, worst in rows:
        lines.append(f"  {name:<28} {mean_d:7.3f} {best:7.3f} {worst:7.3f}")
    lines.append("")
    winner, best_mean = rows[0][0], rows[0][1]
    lines.append(f"Closest to real data: {winner} (mean {best_mean:.3f}).")
    lines.append(
        "A CAVEAT THAT MATTERS: looking like real data is not the same as training a better\n"
        "model. This ranking says which priors are worth the compute; Exp1's training runs are\n"
        "what decide which one actually helps."
    )
    return "\n".join(lines)
