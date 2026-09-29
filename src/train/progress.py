"""A learning curve on REAL data, written during pretraining.

WHAT THIS PRODUCES: one CSV per run, one row every `eval_every_datasets` synthetic
datasets, holding the model's score on the real credit datasets **and** on the
out-of-domain datasets at that moment.

    datasets_seen, step, train_loss, lgd_pinball_heloc, ..., ood_clf_roc_auc, ...

WHY IT IS WORTH THE GPU TIME

Without it, a 400,000-dataset run is a single number at the end: better or worse. You
cannot tell the difference between a prior that helps from the start, one that only pays
off late, and one that helped early and then over-specialised. Those imply completely
different conclusions, and the last one is invisible to an end-of-run score.

It also answers the practical question directly: **has this run converged, or is the
budget too small?** If the curve is still climbing at 400,000 datasets, the honest
finding is "we are compute-limited", not "prior X beats prior Y".

And it is the early-warning system for the out-of-domain question. If general
performance falls while credit performance rises, that trade-off shows up here, mid-run,
rather than after every arm has finished.

HOW IT MEASURES (protocol 3, 28-09-2026). Exactly the way the benchmark does, minus the size:
the CURRENT weights are put inside upstream's own `TabICLClassifier` / `TabICLRegressor` — the
same preprocessing, 8-member ensemble and decoding that score the final checkpoints — and each
dataset is scored on ONE FIXED split (same rows at every step, in every arm), so a curve moves
only because the weights did. Every metric the benchmark reports without a tuned threshold is
recorded: for PD ROC-AUC, PR-AUC, Brier, log-loss, ECE, KS, calibration slope and the hard-label
family at the base-rate threshold; for LGD R², RMSE, MAE, CRPS, pinball, coverage and the
boundary masses. The first row is written at step 0, before any training, so every arm's curve
starts from the same untrained model.

Until 28-09-2026 the monitor ran its own forward pass instead: no outlier clipping, no ensemble,
and — once the LGD prior standardised its target — raw [0, 1] context labels the model no longer
trained on. Its numbers were a different measurement from the benchmark's.

COST CONTROL. A fixed subsample per dataset (`context_rows` context, `max_test_rows` scored),
the development datasets only, a handful of out-of-domain suites. This is a *trend*, not the
final measurement — `scripts/evaluate.py` produces the numbers that go in the paper.

NEVER FATAL. Any failure inside the hook is caught and recorded as a row with an
`error` column. A diagnostic that can kill a 3-day training run is worse than no
diagnostic.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger


@dataclass
class ProgressConfig:
    """How often to measure, and how cheaply."""

    #: Datasets between measurements. 0 disables the hook entirely.
    every_datasets: int = 100_000
    #: Real credit datasets to score. Empty = auto-pick the smallest few, which keeps
    #: the hook fast; the full set is what the eval pipeline is for.
    datasets: list[str] = field(default_factory=list)
    #: When provided by an experiment, monitoring may only use its development split.
    allowed_datasets: list[str] | None = None
    n_datasets: int = 4
    #: Out-of-domain datasets to score, if the cache is present.
    n_ood: int = 4
    #: Context rows for in-context scoring: a fixed, stratified subsample per dataset. The
    #: benchmark gives the model the whole training pool; the monitor keeps the cost of a
    #: measurement to seconds.
    context_rows: int = 4096
    #: Cap on test rows scored per dataset.
    max_test_rows: int = 2_048
    seed: int = 0

#: Bumped when the measurement changes; a CSV of an older protocol is set aside, never appended to.
PROGRESS_PROTOCOL = 3


class ProgressTracker:
    """Scores the in-training model on real data and appends a CSV row."""

    def __init__(self, cfg: ProgressConfig, task: str, run_name: str, out_dir: Path):
        self.cfg = cfg
        self.task = task
        self.run_name = run_name
        self.path = Path(out_dir) / "progress.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            with self.path.open(encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh), [])
            protocol = None
            if "progress_protocol" in header:
                with self.path.open(encoding="utf-8", newline="") as fh:
                    first = next(csv.DictReader(fh), {}) or {}
                protocol = str(first.get("progress_protocol", "")).strip()
            if protocol != str(PROGRESS_PROTOCOL):
                # Old curves used other splits, contexts and scoring paths. Keep them as
                # evidence, never append the new protocol to the same plotted series.
                self.path.rename(self.path.with_name(f"progress_legacy_{time.time_ns()}.csv"))
        self._fieldnames: list[str] | None = None
        self._next_at = cfg.every_datasets
        self._cached_real: list[tuple[str, Any]] | None = None
        self._cached_ood: list[Any] | None = None
        self._splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def resume_from(self, datasets_seen: int) -> None:
        """After a restart, measure next at the next multiple of the interval — not at once."""
        interval = self.cfg.every_datasets
        if interval > 0:
            self._next_at = (int(datasets_seen) // interval + 1) * interval

    @property
    def enabled(self) -> bool:
        return self.cfg.every_datasets > 0

    def due(self, datasets_seen: int) -> bool:
        return self.enabled and datasets_seen >= self._next_at

    # -- data selection, done once ------------------------------------------
    def _real_datasets(self) -> list[tuple[str, Any]]:
        if self._cached_real is not None:
            return self._cached_real
        log = get_logger()
        from src.data.discovery import list_datasets
        from src.data.pipeline import load_processed

        slugs = self.cfg.datasets or list_datasets(self.task)
        if self.cfg.allowed_datasets is not None:
            slugs = [s for s in slugs if s in self.cfg.allowed_datasets]
        loaded: list[tuple[str, Any]] = []
        for slug in slugs:
            try:
                loaded.append((slug, load_processed(self.task, slug)))
            except Exception as exc:  # noqa: BLE001 — skip, never fail the run
                log.warning("[progress] %s unavailable: %s", slug, exc)
        if not self.cfg.datasets:
            # Smallest first: the hook runs many times, so cheap datasets keep it cheap.
            loaded.sort(key=lambda kv: kv[1].n_rows)
            loaded = loaded[: self.cfg.n_datasets]
        self._cached_real = loaded
        log.info("[progress] tracking %d real datasets: %s",
                 len(loaded), ", ".join(s for s, _ in loaded))
        return loaded

    def _ood_datasets(self) -> list[Any]:
        if self._cached_ood is not None:
            return self._cached_ood
        log = get_logger()
        try:
            from src.eval.ood import list_ood_datasets

            kind = "regression" if self.task == "lgd" else "classification"
            entries = list_ood_datasets(kind)[: self.cfg.n_ood]
        except Exception as exc:  # noqa: BLE001
            log.warning("[progress] out-of-domain cache unavailable: %s", exc)
            entries = []
        self._cached_ood = entries
        if entries:
            log.info("[progress] tracking %d out-of-domain datasets: %s",
                     len(entries), ", ".join(e.name for e in entries))
        else:
            log.info("[progress] no out-of-domain cache — run src/utils/fetch_ood.py "
                     "on a login node to add those columns")
        return entries

    # -- the measurement -----------------------------------------------------
    def _split(self, key: str, y: np.ndarray, classification: bool) -> tuple[np.ndarray, np.ndarray]:
        """`(context_idx, test_idx)` for one dataset: drawn once, from a seed of the dataset's
        NAME, so it is the same rows at every step and in every arm. Stratified for a class
        target, so the context always holds both classes."""
        if key in self._splits:
            return self._splits[key]
        import zlib

        from sklearn.model_selection import train_test_split

        n = len(y)
        ctx_n = min(self.cfg.context_rows, max(8, n // 2))
        test_n = min(self.cfg.max_test_rows, n - ctx_n)
        seed = int(zlib.crc32(f"{self.cfg.seed}:{key}".encode()) % (2**31))
        index = np.arange(n)
        strat = (y >= 0.5).astype(int) if classification else None
        if strat is not None and np.bincount(strat, minlength=2).min() < 4:
            strat = None
        ctx, rest = train_test_split(index, train_size=ctx_n, random_state=seed, stratify=strat)
        if len(rest) > test_n:
            rest_strat = strat[rest] if strat is not None else None
            rest, _ = train_test_split(rest, train_size=test_n, random_state=seed + 1, stratify=rest_strat)
        self._splits[key] = (np.sort(ctx), np.sort(rest))
        return self._splits[key]

    def _wrapper(self, model, regression: bool):
        """Upstream's sklearn wrapper, holding the IN-TRAINING weights instead of a file: the same
        preprocessing, ensemble and decoding the benchmark scores the saved checkpoints with."""
        from tabicl import TabICLClassifier, TabICLRegressor

        device = next(model.parameters()).device
        cls = TabICLRegressor if regression else TabICLClassifier
        est = cls(device=device, random_state=self.cfg.seed, allow_auto_download=False)

        def _use_training_weights() -> None:
            est.model_ = model
            est.model_path_ = None
            est.model_config_ = {}
            est.model_.eval()

        est._load_model = _use_training_weights
        return est

    def _fit_predict(self, model, Xc: np.ndarray, yc: np.ndarray, Xt: np.ndarray, *,
                     regression: bool) -> dict[str, np.ndarray]:
        """The CURRENT weights on one table: `{"point", "quantiles"}` for a continuous target,
        `{"prob"}` (P(class 1)) for a binary one. Upstream's TabICL wrapper; `TabPFNProgressTracker`
        overrides this for TabPFN-3 and leaves everything else — data, split, metrics — shared."""
        import torch

        est = self._wrapper(model, regression)
        with torch.no_grad():
            est.fit(Xc, yc)
            if regression:
                from src.eval.protocol import QUANTILE_LEVELS

                out = est.predict(Xt, output_type=["mean", "quantiles"],
                                  alphas=[float(a) for a in QUANTILE_LEVELS])
                return {"point": np.asarray(out["mean"]), "quantiles": np.asarray(out["quantiles"])}
            prob = np.asarray(est.predict_proba(Xt), dtype=float)
            # Column of class 1: the wrapper's LabelEncoder sorts the labels.
            return {"prob": prob[:, list(est.classes_).index(1)]}

    def _score(self, model, key: str, X: np.ndarray, y: np.ndarray, *, regression: bool,
               scale_target: bool = False, bounded_target: bool = True) -> dict[str, float]:
        """Score one table with the CURRENT weights, in context. Pure numpy in, floats out."""
        # A NON-FINITE TARGET CANNOT BE SCORED, AND IS NOT THE SAME AS A BROKEN MODEL.
        # sklearn raises the same "Input contains NaN." for a NaN label and a NaN prediction,
        # which is why the 14-08-2026 run said only that and nothing about which dataset or
        # which side. Features are imputed by the wrapper; targets cannot be — an invented label
        # is a fabricated measurement — so those rows are dropped and counted instead.
        finite = np.isfinite(y)
        dropped = int((~finite).sum())
        X, y = X[finite], y[finite]
        if len(y) < 32:
            return {"skipped_nonfinite_target": 1.0} if dropped else {}
        ctx_idx, test_idx = self._split(key, y, classification=not regression)
        Xc, yc, Xt, yt = X[ctx_idx], y[ctx_idx], X[test_idx], y[test_idx]
        if len(yt) < 8:
            return {}
        if scale_target:
            lo, hi = float(yc.min()), float(yc.max())
            if hi - lo < 1e-12:
                raise ValueError("Constant OOD context target")
            yc, yt = (yc - lo) / (hi - lo), (yt - lo) / (hi - lo)

        if not regression and len(np.unique(yt)) < 2:
            return {}
        pred = self._fit_predict(model, Xc, yc, Xt, regression=regression)
        if regression:
            from src.eval.metrics import lgd_metrics
            from src.eval.protocol import QUANTILE_LEVELS

            point = np.clip(np.asarray(pred["point"], dtype=float), 0.0, 1.0)
            q = np.clip(np.asarray(pred["quantiles"], dtype=float), 0.0, 1.0)
            if not (np.isfinite(point).all() and np.isfinite(q).all()):
                raise ValueError("Non-finite predictions")
            m = lgd_metrics(yt, point, quantiles=q, levels=QUANTILE_LEVELS, decoding="mean")
            if not bounded_target:
                from src.eval.runner import BOUNDARY_KEYS

                m = {k: v for k, v in m.items() if not k.startswith(BOUNDARY_KEYS)}
        else:
            from src.eval.metrics import pd_metrics

            prob = np.asarray(pred["prob"], dtype=float)
            if not np.isfinite(prob).all():
                return {"pred_nonfinite_frac": round(float(np.mean(~np.isfinite(prob))), 6),
                        "n_dropped_nonfinite_target": float(dropped)}
            m = pd_metrics(yt, prob)
        out_m = {k: float(v) for k, v in m.items()
                 if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)}
        if dropped:
            out_m["n_dropped_nonfinite_target"] = float(dropped)
        return out_m

    def record(self, model, *, step: int, datasets_seen: int, train_loss: float,
               elapsed_s: float) -> dict[str, Any]:
        """Measure and append one row. Never raises."""
        log = get_logger()
        started = time.time()
        was_training = model.training
        model.eval()

        row: dict[str, Any] = {
            "run_name": self.run_name,
            "task": self.task,
            "step": step,
            "datasets_seen": datasets_seen,
            "train_loss": round(float(train_loss), 6),
            "elapsed_s": round(elapsed_s, 1),
            "progress_protocol": PROGRESS_PROTOCOL,
        }

        # ONE DATASET MUST NOT COST ALL THE OTHERS. A single `try` around both loops meant
        # that on 14-08-2026 a NaN target in the SECOND real dataset discarded the remaining
        # two AND all eight out-of-domain suites: the row held one dataset and an `error`
        # string, and the run looked like the progress curve barely worked. Each dataset is
        # now isolated, so a failure costs its own columns and nothing else.
        errors: list[str] = []
        try:
            for slug, ds in self._real_datasets():
                short = slug.split(".", 1)[-1]
                try:
                    for k, v in self._score(model, f"real/{slug}", np.asarray(ds.X, np.float32),
                                            np.asarray(ds.y, np.float32),
                                            regression=self.task == "lgd").items():
                        row[f"real__{short}__{k}"] = round(v, 6)
                except Exception as exc:  # noqa: BLE001 — one dataset, one failure
                    errors.append(f"real/{short}: {type(exc).__name__}: {exc}")

            from src.eval.ood import load_ood_dataset

            for entry in self._ood_datasets():
                try:
                    Xo, yo, _ = load_ood_dataset(entry)
                    if self.task == "pd" and len(np.unique(yo)) > 2:
                        yo = (yo == np.bincount(yo.astype(int)).argmax()).astype(np.float32)
                    for k, v in self._score(model, f"ood/{entry.name}", np.asarray(Xo, np.float32),
                                            np.asarray(yo, np.float32), regression=self.task == "lgd",
                                            scale_target=self.task == "lgd",
                                            bounded_target=False).items():
                        row[f"ood__{entry.name}__{k}"] = round(v, 6)
                except Exception as exc:  # noqa: BLE001 — a missing OOD file is not fatal
                    errors.append(f"ood/{entry.name}: {type(exc).__name__}: {exc}")

        except Exception as exc:  # noqa: BLE001 — a diagnostic must never kill training
            errors.append(f"{type(exc).__name__}: {exc}")

        if errors:
            # Truncated in the CSV cell, but every one goes to the log.
            row["error"] = " | ".join(errors[:4])
            row["n_errors"] = len(errors)
            for e in errors:
                log.warning("[progress] skipped (training continues): %s", e)

        row["progress_eval_seconds"] = round(time.time() - started, 2)
        try:
            self._append(row)
        except OSError as exc:
            log.warning("[progress] could not save diagnostic: %s", exc)
        finally:
            if was_training:
                model.train()

        self.resume_from(datasets_seen)
        headline = {k: v for k, v in row.items() if k.startswith(("real__", "ood__"))}
        log.info(
            "[progress] datasets=%s step=%d took %.1fs | %s",
            f"{datasets_seen:,}", step, row["progress_eval_seconds"],
            ", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in list(headline.items())[:6]) or "no metrics",
        )
        return row

    def _append(self, row: dict[str, Any]) -> None:
        """Append one CSV row, widening the header if new columns appear.

        Columns can appear late — the out-of-domain cache may be populated after a run
        starts, and a dataset can fail one round and succeed the next. Rewriting the file
        with the union of columns keeps the CSV readable by pandas instead of producing
        ragged rows that silently misalign.
        """
        new_file = not self.path.exists()
        if not new_file and self._fieldnames is None:
            with self.path.open("r", encoding="utf-8", newline="") as fh:
                self._fieldnames = next(csv.reader(fh), [])
        if new_file:
            self._fieldnames = list(row)
        missing = [k for k in row if k not in self._fieldnames]
        if missing and not new_file:
            import csv as _csv

            with self.path.open("r", encoding="utf-8", newline="") as fh:
                existing = list(_csv.DictReader(fh))
            self._fieldnames = self._fieldnames + missing
            with self.path.open("w", encoding="utf-8", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=self._fieldnames)
                w.writeheader()
                w.writerows(existing)
            new_file = False
        elif missing:
            self._fieldnames = self._fieldnames + missing

        with self.path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
