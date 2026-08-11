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

COST CONTROL. Evaluation is capped hard: a handful of datasets, a small context, one
seed. This is a *trend*, not the final measurement — `scripts/evaluate.py` produces the
numbers that go in the paper. A progress hook that noticeably slowed training would be
a bad trade, so the defaults are deliberately cheap and everything is configurable.

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
    n_datasets: int = 4
    #: Out-of-domain datasets to score, if the cache is present.
    n_ood: int = 4
    #: Context rows for in-context scoring. Smaller than evaluation's 1024 on purpose.
    context_rows: int = 512
    #: Cap on test rows scored per dataset.
    max_test_rows: int = 2_000
    seed: int = 0


class ProgressTracker:
    """Scores the in-training model on real data and appends a CSV row."""

    def __init__(self, cfg: ProgressConfig, task: str, run_name: str, out_dir: Path):
        self.cfg = cfg
        self.task = task
        self.run_name = run_name
        self.path = Path(out_dir) / f"{run_name}__progress.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        self._next_at = cfg.every_datasets
        self._cached_real: list[tuple[str, Any]] | None = None
        self._cached_ood: list[Any] | None = None

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
            log.info("[progress] no out-of-domain cache — run scripts/fetch_ood.py "
                     "on a login node to add those columns")
        return entries

    # -- the measurement -----------------------------------------------------
    def _score(self, model, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        """Score one table with the CURRENT weights, in context. Pure numpy in, floats out."""
        import torch

        n = len(X)
        ctx_n = min(self.cfg.context_rows, max(8, n // 2))
        perm = rng.permutation(n)
        ctx_idx, test_idx = perm[:ctx_n], perm[ctx_n : ctx_n + self.cfg.max_test_rows]
        if len(test_idx) < 8:
            return {}

        Xc, yc = X[ctx_idx], y[ctx_idx]
        Xt, yt = X[test_idx], y[test_idx]
        # Impute from the CONTEXT only; using test statistics would leak.
        med = np.nan_to_num(np.nanmedian(Xc, axis=0))
        Xc = np.where(np.isfinite(Xc), Xc, med)
        Xt = np.where(np.isfinite(Xt), Xt, med)

        device = next(model.parameters()).device
        x = torch.from_numpy(np.concatenate([Xc, Xt]).astype(np.float32)).unsqueeze(0).to(device)
        yy = torch.from_numpy(yc.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, yy)
        q = out[0].float().cpu().numpy()

        if self.task == "lgd":
            from src.eval.metrics import lgd_metrics
            from src.train.loop import quantile_levels

            levels = quantile_levels(q.shape[1]).numpy()
            point = np.clip(q[:, q.shape[1] // 2], 0.0, 1.0)
            m = lgd_metrics(yt, point, quantiles=np.clip(q, 0.0, 1.0), levels=levels)
            return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}

        from scipy.special import softmax
        from sklearn.metrics import average_precision_score, roc_auc_score

        prob = softmax(q, axis=-1)[:, 1]
        if len(np.unique(yt)) < 2:
            return {}
        return {
            "roc_auc": float(roc_auc_score(yt, prob)),
            "pr_auc": float(average_precision_score(yt, prob)),
        }

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
        }
        rng = np.random.default_rng(self.cfg.seed)

        try:
            for slug, ds in self._real_datasets():
                short = slug.split(".", 1)[-1]
                for k, v in self._score(model, np.asarray(ds.X, np.float32),
                                        np.asarray(ds.y, np.float32), rng).items():
                    row[f"real__{short}__{k}"] = round(v, 6)

            from src.eval.ood import load_ood_dataset

            for entry in self._ood_datasets():
                try:
                    Xo, yo, _ = load_ood_dataset(entry)
                except Exception:  # noqa: BLE001 — a missing OOD file is not fatal
                    continue
                if self.task == "lgd":
                    # OOD regression targets have arbitrary scales; the model predicts in
                    # [0,1], so min-max the target the same way the OOD runner does.
                    lo, hi = float(np.min(yo)), float(np.max(yo))
                    if hi - lo < 1e-12:
                        continue
                    yo = ((yo - lo) / (hi - lo)).astype(np.float32)
                elif len(np.unique(yo)) > 2:
                    yo = (yo == np.bincount(yo.astype(int)).argmax()).astype(np.float32)
                for k, v in self._score(model, np.asarray(Xo, np.float32),
                                        np.asarray(yo, np.float32), rng).items():
                    row[f"ood__{entry.name}__{k}"] = round(v, 6)

        except Exception as exc:  # noqa: BLE001 — a diagnostic must never kill training
            row["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("[progress] measurement failed (training continues): %s", row["error"])

        row["progress_eval_seconds"] = round(time.time() - started, 2)
        self._append(row)
        if was_training:
            model.train()

        self._next_at = datasets_seen + self.cfg.every_datasets
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
        if new_file or self._fieldnames is None:
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
