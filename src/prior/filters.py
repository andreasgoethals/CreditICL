"""Predictability filtering — TabICLv2's `should_filter`, plus a banded mode.

Upstream (verified in `TabICL.txt`, `should_filter` in `prior/_genload.py`, and
TabICLv2 Appendix E.14): fit an ExtraTreesRegressor with `n_estimators=25`,
`bootstrap=True`, `oob_score=True`, `max_depth=6` on the **full** dataset; test
whether the out-of-bag MSE beats predicting the mean, on at least 95% of 200
bootstrap resamples; reject if not. The paper reports that this discards *"roughly
35% classification and 25% regression datasets"* in stage 1, and that it
**improves pretraining convergence** (their Fig. 10).

Three modes:

``tabicl``  as shipped — reject when the bootstrap p-value is >= 0.05.
``off``     no statistical filter at all.
``banded``  keep only tasks whose ExtraTrees pseudo-R^2 falls inside a target
            band, i.e. *target* the low-signal regime rather than merely admit
            it. Credit is intrinsically low-signal (AUC 0.70-0.85; published LGD
            R^2 ~0.04-0.43), so a prior that discards weak-signal DGPs may leave
            the model miscalibrated for the regime credit actually occupies.

`banded` is the cheapest sharp experiment in the project: it is a **removal**
rather than an addition, so it cannot be accused of adding capacity. Note it
contradicts a published convergence result, so the rejection statistics matter —
`FilterStats` is what lets us report them rather than assert them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor


@dataclass
class FilterStats:
    """Rejection bookkeeping. Reported per arm so no cap is silent."""

    attempts: int = 0
    accepted: int = 0
    rejected_unpredictable: int = 0
    rejected_trivial: int = 0
    rejected_band: int = 0
    pseudo_r2: list[float] = field(default_factory=list)
    pvalues: list[float] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        return 0.0 if self.attempts == 0 else 1.0 - self.accepted / self.attempts

    def summary(self) -> dict[str, float]:
        r2 = np.asarray(self.pseudo_r2, dtype=float)
        out = {
            "attempts": float(self.attempts),
            "accepted": float(self.accepted),
            "rejection_rate": self.rejection_rate,
            "rejected_unpredictable": float(self.rejected_unpredictable),
            "rejected_trivial": float(self.rejected_trivial),
            "rejected_band": float(self.rejected_band),
        }
        if r2.size:
            out.update(
                {
                    "pseudo_r2_mean": float(r2.mean()),
                    "pseudo_r2_p10": float(np.percentile(r2, 10)),
                    "pseudo_r2_p50": float(np.percentile(r2, 50)),
                    "pseudo_r2_p90": float(np.percentile(r2, 90)),
                }
            )
        return out


def predictability(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    is_classif: bool,
    n_estimators: int = 25,
    n_boot: int = 200,
    seed: int = 1,
) -> tuple[float, float]:
    """Return (bootstrap p-value, ExtraTrees pseudo-R^2).

    Mirrors upstream exactly, including `random_state=1` — upstream notes that
    `random_state=0` fails to give OOB scores for all samples at some
    133 <= n <= 257, so this is not an arbitrary choice.
    """
    if is_classif:
        y_int = y.to(torch.long)
        n_classes = int(y_int.max().item() + 1)
        Y = F.one_hot(y_int, num_classes=max(n_classes, 2)).to(torch.float32)
        if Y.shape[1] == 2:
            Y = Y[:, :1]  # drop one dimension for binary, to match upstream and be faster
    else:
        Y = y.reshape(-1, 1).to(torch.float32)

    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    et = ExtraTreesRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        random_state=seed,
        max_depth=6,
    ).fit(X_np, np.squeeze(Y_np, axis=-1) if Y_np.shape[-1] == 1 else Y_np)

    Yhat = et.oob_prediction_
    if Yhat.ndim == 1:
        Yhat = Yhat[:, None]
    mask = ~np.isnan(Yhat).any(axis=1)
    if mask.sum() < 8:  # too few valid OOB rows to say anything
        return 1.0, 0.0
    Yv, Yhatv = Y_np[mask], Yhat[mask]

    baseline = Y_np.mean(axis=0, keepdims=True)
    imp = ((Yv - baseline) ** 2).sum(axis=1) - ((Yv - Yhatv) ** 2).sum(axis=1)

    baseline_mse = float(((Yv - baseline) ** 2).sum(axis=1).mean())
    et_mse = float(((Yv - Yhatv) ** 2).sum(axis=1).mean())

    rng = np.random.default_rng(0)  # fixed, as upstream — this is a test statistic, not a sample
    n = imp.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))

    pseudo_r2 = 0.0 if baseline_mse <= 0 else 1.0 - et_mse / baseline_mse
    return pval, pseudo_r2


class PredictabilityFilter:
    """Accept/reject a candidate task under one of the three modes."""

    def __init__(
        self,
        mode: str = "tabicl",
        *,
        quantile_band: tuple[float, float] = (0.02, 0.45),
        remove_trivial: bool = False,
        trivial_threshold: float = 0.1,
        pvalue_threshold: float = 0.05,
        n_boot: int = 200,
    ):
        if mode not in {"tabicl", "off", "banded"}:
            raise ValueError(f"unknown filter mode {mode!r}; expected 'tabicl', 'off' or 'banded'")
        self.mode = mode
        self.band = tuple(quantile_band)
        self.remove_trivial = remove_trivial
        # RMSE ratio in upstream terms; a dataset is "trivial" if ExtraTrees RMSE
        # is below threshold * dummy RMSE, i.e. pseudo-R^2 above 1 - threshold^2.
        self.trivial_r2 = 1.0 - trivial_threshold**2
        self.pvalue_threshold = pvalue_threshold
        self.n_boot = n_boot
        self.stats = FilterStats()

    def accept(self, X: torch.Tensor, y: torch.Tensor, *, is_classif: bool) -> bool:
        self.stats.attempts += 1

        if self.mode == "off" and not self.remove_trivial:
            self.stats.accepted += 1
            return True

        # Degenerate targets are rejected under every mode: a constant target
        # carries no gradient signal and would just waste steps.
        if is_classif:
            if len(torch.unique(y)) < 2:
                self.stats.rejected_unpredictable += 1
                return False
        elif float(y.std()) < 1e-8:
            self.stats.rejected_unpredictable += 1
            return False

        pval, r2 = predictability(X, y, is_classif=is_classif, n_boot=self.n_boot)
        self.stats.pvalues.append(pval)
        self.stats.pseudo_r2.append(r2)

        if self.remove_trivial and r2 >= self.trivial_r2:
            self.stats.rejected_trivial += 1
            return False

        if self.mode == "tabicl" and pval >= self.pvalue_threshold:
            self.stats.rejected_unpredictable += 1
            return False

        if self.mode == "banded":
            lo, hi = self.band
            if not (lo <= r2 <= hi):
                self.stats.rejected_band += 1
                return False

        self.stats.accepted += 1
        return True
