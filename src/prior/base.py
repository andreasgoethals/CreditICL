"""The base TabICLv2 prior: Cauchy DAG, node functions, categorical converters.

Transcribed from NanoTabICL's `prior.py` (`rand_dataset_plain`,
`rand_cauchy_graph`, `rand_node_func`, `rand_converter`, `rand_cat_sizes`).

Two credit-relevant knobs are exposed here rather than in `targets/`, because
they change how *features* are generated and so must sit inside the graph
evaluation:

* ``max_cat_size`` — upstream caps categorical cardinality at 100. Credit files
  routinely carry columns well past that (US state, MSA, seller/servicer name,
  occupation code), and Purucker 2026 finds the best-GBDT-over-best-TFM margin
  grows with high-cardinality columns (rho=+0.47). So the cap is a lever.
* ``category_frequency`` — upstream category assignment is roughly balanced
  (nearest-centre or softmax over random points). Real credit categoricals are
  severely unbalanced: a handful of states hold most of the book, and the long
  tail is where the risk concentrates. ``"power_law"`` reproduces that.

Both default to upstream behaviour, so arm A is recovered exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .functions import (
    l2_normalize,
    rand_func,
    rand_kumaraswamy_act,
    rand_multi_func,
    rand_points,
    rand_weights,
    standardize,
)
from .rng import PriorRNG


@dataclass
class SyntheticTask:
    """One in-context episode: features, target, and provenance."""

    X: torch.Tensor  # (n_rows, n_features) float32
    y: torch.Tensor  # (n_rows,) float32 — class index or regression value
    source: str = "base"  # which sub-prior produced it, for logging
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


# ---------------------------------------------------------------------------
# Random DAG
# ---------------------------------------------------------------------------


def rand_cauchy_graph(rng: PriorRNG, n_nodes: int) -> list[list[int]]:
    """Return the parent index list for each node. Upper-triangular, so acyclic."""
    output_importances = rng.cauchy(n_nodes)
    input_importances = rng.cauchy(n_nodes)
    logits = rng.standard_cauchy() + output_importances[None, :] + input_importances[:, None]
    adjacency = rng.rand_like(logits) <= torch.sigmoid(logits)
    return [[i_in for i_in in range(i_out) if adjacency[i_in, i_out]] for i_out in range(n_nodes)]


# ---------------------------------------------------------------------------
# Categorical sizes and converters
# ---------------------------------------------------------------------------


def rand_cat_sizes(
    rng: PriorRNG,
    n_features: int,
    max_cat_size: int = 100,
    cat_fraction_range: tuple[float, float] = (-0.5, 1.2),
) -> list[int]:
    """Per-column cardinality: 0 for numerical, >=2 for categorical.

    The asymmetric `(-0.5, 1.2)` default is upstream: clipping to [0, 1] puts
    real point mass on "all numerical" and "all categorical".
    """
    lo, hi = cat_fraction_range
    cat_fraction = float(torch.clamp(torch.tensor(rng.uniform(lo, hi)), 0.0, 1.0))
    n_cat = round(n_features * cat_fraction)
    cat_size_limit = rng.logint(2, max_cat_size + 1)
    return [0] * (n_features - n_cat) + [rng.logint(2, cat_size_limit + 1) for _ in range(n_cat)]


def _power_law_assign(rng: PriorRNG, x: torch.Tensor, cat_size: int, alpha: float) -> torch.Tensor:
    """Assign categories with power-law frequencies, monotone in a projection of x.

    Rank rows along a random projection, then cut the ranking at the cumulative
    boundaries of a Zipf(alpha) frequency vector. Monotone in the projection, so
    the categorical still carries signal; unbalanced, as real credit columns are.
    """
    proj = x @ rng.randn(x.shape[1], 1)
    order = torch.argsort(proj.squeeze(-1))
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order))

    freqs = 1.0 / torch.arange(1, cat_size + 1, dtype=torch.float) ** alpha
    freqs = freqs[rng.randperm(cat_size)]  # so the big category is not always index 0
    bounds = torch.cumsum(freqs / freqs.sum(), dim=0) * len(order)
    return torch.bucketize(ranks.float(), bounds, right=False).clamp_(max=cat_size - 1)[:, None]


def rand_converter(
    rng: PriorRNG,
    x: torch.Tensor,
    cat_size: int,
    *,
    category_frequency: str = "balanced",
    power_law_alpha_range: tuple[float, float] = (0.6, 1.6),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (value propagated through the graph, extracted column)."""
    if cat_size <= 0:  # numerical
        return x, (x if rng.boolean() else rand_kumaraswamy_act(rng, x))

    if category_frequency == "power_law" and x.shape[0] > cat_size:
        alpha = rng.uniform(*power_law_alpha_range)
        x_idxs = _power_law_assign(rng, x, cat_size, alpha)
        x_disc = rand_points(rng, cat_size, cat_size)[x_idxs.squeeze(-1)]
        mode = rng.choice(["neigh_id", "neigh_disc", "neigh_int"])
    else:
        mode = rng.choice(
            ["neigh_id", "neigh_disc", "neigh_func", "neigh_int", "softmax_id", "softmax_disc", "softmax_int"]
        )
        if mode.startswith("softmax"):
            x = rng.lognum(0.1, 10) * standardize(x) + torch.log(rand_weights(rng, 1, x.shape[1]) + 1e-4)
            x[~torch.isfinite(x)] = 0.0  # guard torch.multinomial
            x_idxs = rng.multinomial(torch.softmax(x, dim=-1), num_samples=1)
            x_disc = rand_points(rng, cat_size, cat_size)[x_idxs.squeeze(-1)]
        else:  # neighbour-based
            centers = x[rng.randperm(len(x))[:cat_size]]
            x_idxs = torch.cdist(x, centers, p=rng.lognum(0.5, 4.0)).argmin(dim=-1)[:, None]
            x_disc = centers[x_idxs.squeeze(-1)]

    if mode.endswith("_disc"):
        x = x_disc
    elif mode.endswith("_func"):
        x = rand_func(rng, x_disc, cat_size, only_cheap=True)
    elif mode.endswith("_int"):
        x = x_idxs.float()

    return x, x_idxs


# ---------------------------------------------------------------------------
# Node evaluation
# ---------------------------------------------------------------------------


def rand_node_func(
    rng: PriorRNG,
    cat_sizes: dict[str, int],
    xs: list[torch.Tensor],
    n_samples: int,
    *,
    category_frequency: str = "balanced",
    power_law_alpha_range: tuple[float, float] = (0.6, 1.6),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    n_features = sum(max(csz, 1) for csz in cat_sizes.values()) + rng.logint(1, 32)
    x = rand_points(rng, n_samples, n_features) if len(xs) == 0 else rand_multi_func(rng, xs, n_features)
    weights = rand_weights(rng, 1, x.shape[1])
    x = l2_normalize(standardize(x) * (weights / weights.square().mean().sqrt()))

    out_features: dict[str, torch.Tensor] = {}
    start_idx = 0
    for name, cat_size in cat_sizes.items():
        end_idx = start_idx + max(cat_size, 1)
        x[:, start_idx:end_idx], out_features[name] = rand_converter(
            rng,
            x[:, start_idx:end_idx],
            cat_size,
            category_frequency=category_frequency,
            power_law_alpha_range=power_law_alpha_range,
        )
        start_idx = end_idx

    return x * rng.lognum(0.1, 10.0), out_features


def rand_dataset_plain(
    rng: PriorRNG,
    x_cat_sizes: list[int],
    y_cat_sizes: list[int],
    n_samples: int,
    *,
    n_nodes_range: tuple[int, int] = (2, 33),
    category_frequency: str = "balanced",
    power_law_alpha_range: tuple[float, float] = (0.6, 1.6),
) -> dict[str, torch.Tensor]:
    """Sample one raw dataset from the DAG. Columns keyed 'x_<i>' and 'y_<i>'."""
    n_nodes = rng.logint(*n_nodes_range)
    graph = rand_cauchy_graph(rng, n_nodes)
    node_cat_sizes: list[dict[str, int]] = [{} for _ in range(n_nodes)]

    for feature_group, cat_sizes in [("x", x_cat_sizes), ("y", y_cat_sizes)]:
        feature_nodes = rng.np.permutation(n_nodes)[: rng.randint(1, n_nodes + 1)]
        feature_node_idxs = rng.np.choice(feature_nodes, replace=True, size=len(cat_sizes))
        for idx, (node_idx, cat_size) in enumerate(zip(feature_node_idxs, cat_sizes)):
            node_cat_sizes[int(node_idx)][f"{feature_group}_{idx}"] = cat_size

    node_values: list[torch.Tensor | None] = [None] * n_nodes
    columns: dict[str, torch.Tensor] = {}
    for node_idx in range(n_nodes):
        parent_values = [node_values[parent] for parent in graph[node_idx]]
        node_values[node_idx], out_features = rand_node_func(
            rng,
            node_cat_sizes[node_idx],
            parent_values,
            n_samples,
            category_frequency=category_frequency,
            power_law_alpha_range=power_law_alpha_range,
        )
        columns.update(out_features)

    for col in columns.values():
        col[~torch.isfinite(col)] = 0  # upstream: fill nan/inf with zero rather than discarding
    return columns


def assemble_xy(columns: dict[str, torch.Tensor], n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the 'x_<i>' columns into X and return the raw latent 'y_0'."""
    X = torch.cat([columns[f"x_{i}"].float() for i in range(n_features)], dim=-1)
    y_latent = columns["y_0"].float().reshape(-1)
    return X, y_latent
