"""Extra feature columns that carry no signal about the target.

Real credit files are full of columns that turn out to be useless: fields kept
for operational reasons, near-duplicates of other fields, IDs, codes nobody uses.
A model that has only ever seen tables where most columns matter will over-weight
noise at prediction time.

The base prior does produce *some* of this — a DAG node that is not an ancestor of
the target node is effectively an irrelevant column — but two things limit it:

1. `filter_unpredictable_graphs` (`check_x_y_ancestors_overlap`) rejects graphs
   where x and y share **no** ancestors, so the fully-disconnected case is
   deliberately removed;
2. how many irrelevant columns you get is a side effect of random graph geometry,
   not something you can dial.

So this module adds them explicitly and countably. Four kinds, matching what
actually shows up in credit data:

``pure_noise``   an independent draw — a column with no relationship to anything
``duplicate``    a near-copy of an existing column plus noise (redundant fields)
``shuffled``     an existing column with its rows permuted: same marginal
                 distribution as a real feature, zero signal. This is the
                 sneakiest kind, because column-wise statistics cannot tell it
                 from a useful column — only the relationship to y can.
``constant``     an almost-constant column (a flag that is nearly always the same)

`shuffled` is the one worth caring about: it keeps the marginal and destroys only
the dependence, so it directly tests whether the model is using relationships
rather than marginals.
"""

from __future__ import annotations

import torch

from .rng import PriorRNG

KINDS = ("pure_noise", "duplicate", "shuffled", "constant")


def add_noise_features(
    rng: PriorRNG,
    X: torch.Tensor,
    cfg: dict,
    max_features: int,
) -> tuple[torch.Tensor, dict]:
    """Append irrelevant columns to X.

    `fraction` is relative to the current width: 0.3 on a 20-column table adds 6.
    Capped at `max_features` so batches stay a predictable width.
    """
    fraction = float(cfg.get("fraction", 0.0))
    if fraction <= 0.0 or X.shape[0] == 0:
        return X, {"noise_features": 0}

    n_add = int(round(fraction * X.shape[1]))
    n_add = min(n_add, max_features - X.shape[1])
    if n_add <= 0:
        return X, {"noise_features": 0}

    weights = cfg.get(
        "kind_weights",
        {"pure_noise": 1.0, "duplicate": 1.0, "shuffled": 1.0, "constant": 0.5},
    )
    kinds = [k for k in KINDS if float(weights.get(k, 0.0)) > 0]
    if not kinds:
        return X, {"noise_features": 0}
    probs = [float(weights.get(k, 0.0)) for k in kinds]

    n_rows = X.shape[0]
    new_cols, used = [], {k: 0 for k in KINDS}

    for _ in range(n_add):
        kind = rng.weighted_choice(kinds, probs)
        src = X[:, rng.randint(0, X.shape[1])]

        if kind == "pure_noise":
            col = rng.randn(n_rows)
        elif kind == "duplicate":
            # Redundant field: highly correlated with a real column, so it looks
            # informative on its own but adds nothing.
            col = src + rng.uniform(0.05, 0.5) * rng.randn_like(src)
        elif kind == "shuffled":
            col = src[rng.randperm(n_rows)]
        else:  # constant
            col = torch.full((n_rows,), rng.uniform(-1.0, 1.0))
            n_flip = max(1, int(0.01 * n_rows))
            col[rng.randperm(n_rows)[:n_flip]] += rng.randn(n_flip).squeeze()

        new_cols.append(col.reshape(-1))
        used[kind] += 1

    X = torch.cat([X, torch.stack(new_cols, dim=-1)], dim=-1)

    # Shuffle columns so the useless ones are not always on the right, which the
    # model could otherwise learn as a positional shortcut.
    X = X[:, rng.randperm(X.shape[1])]

    return X, {"noise_features": n_add, **{f"noise_{k}": v for k, v in used.items() if v}}
