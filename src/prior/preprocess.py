"""Target and feature preprocessing, faithful to TabICL.

`outlier_removing` and `standard_scaling` are transcribed from TabICL's
`prior/_reg2cls.py`. They matter more than they look:

* `GraphSCM.__call__` applies `outlier_removing(threshold=4)` then
  `standard_scaling` to every regression target. That is arm A's behaviour and
  is what a bounded-target arm replaces.
* `standard_scaling` is **affine**, therefore *shape-preserving*: a U-shaped
  target with two atoms survives it as a U-shaped target with two atoms on a
  rescaled axis. Only the [0,1] *support* is destroyed. This is why the project
  claims a **frequency and support-alignment** gap rather than a structural
  absence — see docs/EXPERIMENTAL_DESIGN.md §1.2.
* `outlier_removing` **clamps**, which literally creates point masses at the
  clamp bounds. Upstream therefore already manufactures atoms, just tiny ones
  far out in the tails.

NanoTabICL deliberately omits both ("could be done inside the model"), which is
what puts this knob under our control instead of buried in the generator.
"""

from __future__ import annotations

import torch


def standard_scaling(x: torch.Tensor, clip_value: float = 100.0) -> torch.Tensor:
    """Zero mean, unit variance per column, clipped. TabICL's `standard_scaling`."""
    mean = torch.nanmean(x, dim=0)
    std = _nanstd(x, dim=0).clip(min=1e-6)
    return torch.clip((x - mean) / std, min=-clip_value, max=clip_value)


def outlier_removing(x: torch.Tensor, threshold: float = 4.0) -> torch.Tensor:
    """Two-pass clamp at +/- `threshold` sigma. TabICL's `outlier_removing`."""
    mean = torch.nanmean(x, dim=0)
    std = _nanstd(x, dim=0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    mask = (lower <= x) & (x <= upper) & ~torch.isnan(x)
    masked = torch.where(mask, x, torch.nan)
    masked_mean = torch.nanmean(masked, dim=0)
    masked_std = _nanstd(masked, dim=0)

    # Columns left with <=1 valid value give NaN moments; fall back to pass one.
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std).clip(min=1e-6)

    cut_off = masked_std * threshold
    lower = torch.nan_to_num(masked_mean - cut_off, nan=-torch.inf)
    upper = torch.nan_to_num(masked_mean + cut_off, nan=torch.inf)
    return x.clamp(min=lower, max=upper)


def _nanstd(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """NaN-ignoring std with ddof=1 when there is more than one row."""
    n_valid = (~torch.isnan(x)).sum(dim=dim)
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    sq = torch.where(torch.isnan(x), torch.zeros_like(x), (x - mean) ** 2).sum(dim=dim)
    ddof = 1 if x.shape[dim] > 1 else 0
    denom = (n_valid - ddof).clamp(min=1).to(x.dtype)
    return torch.sqrt(sq / denom)


def process_features(X: torch.Tensor) -> torch.Tensor:
    """Arm A's feature path: clamp outliers, then standardise."""
    return standard_scaling(outlier_removing(X, threshold=4.0))


def to_ranks(x: torch.Tensor) -> torch.Tensor:
    """Map a 1-D tensor to (0, 1) by rank. Strictly monotone, so the predictive
    relationship with the features is preserved exactly (Spearman rho = 1).

    This is what lets a target-shaping transform change only the *marginal*
    without changing how much signal the features carry — the property that
    makes the LGD arm a clean intervention rather than a confounded one.
    """
    n = x.numel()
    if n <= 1:
        return torch.full_like(x, 0.5)
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.long)
    ranks[order] = torch.arange(n)
    return (ranks.to(x.dtype) + 0.5) / n
