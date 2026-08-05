"""Random function families, activations, matrices and point clouds.

A faithful transcription of NanoTabICL's `prior.py` (the simplified TabICLv2
prior), with two mechanical changes and no behavioural ones:

* every function takes a `PriorRNG` instead of using global RNG state, so the
  task stream survives checkpoint/resume (see `rng.py`);
* `randn_like` / `rand_like` are routed through the generator, which the torch
  built-ins cannot do.

**This module is the baseline and must not be "improved".** Arm A is defined as
"what this file does"; if it drifts from upstream, every comparison in the
project loses its control. Credit-specific behaviour belongs in `targets/`.

Reference: `tfm-library/repositories/NanoTabICL.txt`, `prior.py`. Symbol names
are kept identical to upstream so the two can be diffed by eye.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .rng import PriorRNG

# ---------------------------------------------------------------------------
# Elementwise helpers
# ---------------------------------------------------------------------------


def standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True, correction=0) + 1e-4)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.square().sum(dim=-1).mean().sqrt() + 1e-8)


def row_normalize(matrix: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    return matrix / (eps + matrix.norm(dim=-1, keepdim=True))


def rand_rescale(rng: PriorRNG, x: torch.Tensor) -> torch.Tensor:
    # Random datapoints as shifts, so activations like ReLU are not always zero.
    idx = rng.torch_randint(x.shape[0], (x.shape[1],))
    bias = -x[idx, torch.arange(x.shape[1])][None, :]
    return rng.lognum(1e-0, 1e1) * (x + bias)


def rand_kumaraswamy_act(rng: PriorRNG, x: torch.Tensor) -> torch.Tensor:
    """Monotone CDF warp onto [0, 1].

    Worth flagging: this is the one place upstream already produces an exactly
    bounded column, and it is applied to numerical outputs with p=0.5. It is both
    the evidence that bounded targets are not *absent* from the prior, and the
    natural primitive our LGD target family builds on.
    """
    a, b = (rng.lognum(0.2, 5) for _ in range(2))
    lo, hi = x.min(dim=0).values, x.max(dim=0).values
    x = torch.clamp((x - lo) / (hi - lo + 1e-30), 0.0, 1.0)
    return 1.0 - (1.0 - x**a) ** b


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

# Kept in upstream order: TabICLv1 set, then TabPFNv2 additions, then v2's.
ACTS = [
    lambda x: x, torch.tanh, F.leaky_relu, F.elu, F.selu, F.silu, F.relu, F.softplus, F.relu6, F.hardtanh, torch.sign,
    torch.exp, torch.sin, torch.square, torch.abs,
    lambda x: (x >= 0.0).float(), lambda x: torch.exp(-(x**2)), lambda x: (torch.abs(x) <= 1.0).float(),
    lambda x: torch.log(torch.clamp(torch.abs(x), min=1e-6)),
    F.sigmoid, torch.round, lambda x: x - torch.floor(x),
    lambda x: torch.argsort(torch.argsort(x, dim=-1), dim=-1).float(),
    F.logsigmoid, lambda x: F.softmax(x, dim=-1),
    lambda x: (x == torch.max(x, dim=-1, keepdim=True).values).float(),
    lambda x: torch.argsort(x, dim=-1).float(),
]


def rand_power_relu_act(rng: PriorRNG, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** rng.lognum(0.1, 10.0)


def rand_power_act(rng: PriorRNG, x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (x.abs() ** rng.lognum(0.1, 10.0))


def rand_plain_act(rng: PriorRNG, x: torch.Tensor) -> torch.Tensor:
    # Upstream weights the plain-activation list 4:1:1 against the two power acts.
    which = rng.randint(0, 6)
    if which < 4:
        return rng.choice(ACTS)(x)
    return rand_power_relu_act(rng, x) if which == 4 else rand_power_act(rng, x)


def rand_act(rng: PriorRNG, x: torch.Tensor) -> torch.Tensor:
    return standardize(rand_plain_act(rng, rand_rescale(rng, standardize(x))))


# ---------------------------------------------------------------------------
# Random weights and matrices
# ---------------------------------------------------------------------------


def rand_weights(rng: PriorRNG, n_batch: int, n: int) -> torch.Tensor:
    decay_rate = torch.as_tensor(
        np.exp(rng.np.uniform(np.log(0.1 / np.log(1 + n)), np.log(6), size=n_batch))
    ).float()
    base_weights = torch.linspace(1.0, n, n)
    log_weights = -decay_rate[:, None] * torch.log(base_weights)
    std_scale = torch.as_tensor(np.exp(rng.np.uniform(np.log(1e-4), np.log(10), size=n_batch))).float()
    logits = log_weights + std_scale[:, None] * rng.randn(n_batch, n)
    logits = torch.stack([logits[i, rng.randperm(n)] for i in range(n_batch)], dim=0)
    return np.sqrt(n) * row_normalize(torch.softmax(logits, dim=-1))


def rand_weights_matrix(rng: PriorRNG, n_batch: int, n: int, m: int) -> torch.Tensor:
    matrix = rand_weights(rng, n_batch * n, m).reshape(n_batch, n, m)
    return row_normalize(matrix * rng.randn_like(matrix), eps=1e-6)  # Gaussian -> allow negative weights


def rand_singular_values_matrix(rng: PriorRNG, n_batch: int, n: int, m: int) -> torch.Tensor:
    k = min(n, m)
    U, D, V = rng.randn(n_batch, n, k), rand_weights(rng, n_batch, k), rng.randn(n_batch, k, m)
    return (U * D[:, None, :]) @ V  # SVD-like; Gaussian is cheaper than orthogonal, still rotation-invariant


def rand_kernel_matrix(rng: PriorRNG, n_batch: int, n: int, m: int) -> torch.Tensor:
    # Laplace kernel in d=3 (arbitrary choice, as upstream).
    points = rand_gauss_mixture_points(rng, n_batch * (n + m), 3).reshape(n_batch, n + m, 3)
    dists = rng.lognum(0.1, 10.0) * torch.cdist(points[:, :n], points[:, n:])
    return torch.exp(-dists) * torch.sign(rng.randn(n_batch, n, m))


def rand_activation_matrix(rng: PriorRNG, n_batch: int, n: int, m: int) -> torch.Tensor:
    base = rand_matrix(rng, n_batch, n, m, no_act=True).reshape(n_batch, n * m)
    matrix = rand_plain_act(rng, base).reshape(n_batch, n, m)
    return matrix + 1e-3 * rng.randn_like(matrix)


def rand_matrix(rng: PriorRNG, n_batch: int, n: int, m: int, no_act: bool = False) -> torch.Tensor:
    no_act_types = [
        lambda r, b, i, j: r.randn(b, i, j),
        rand_weights_matrix,
        rand_singular_values_matrix,
        rand_kernel_matrix,
    ]
    types = no_act_types if no_act else no_act_types + [rand_activation_matrix]
    matrix = rng.choice(types)(rng, n_batch, n, m)
    return row_normalize(matrix + 1e-6 * rng.randn(n_batch, n, m), eps=1e-6)


# ---------------------------------------------------------------------------
# Random point clouds
# ---------------------------------------------------------------------------


def rand_unif_points(rng: PriorRNG, n_batch: int, n: int) -> torch.Tensor:
    return 2 * rng.rand(n_batch, n) - 1.0


def rand_cov_points(rng: PriorRNG, n_batch: int, n: int) -> torch.Tensor:
    base = rng.choice([rand_unif_points, lambda r, b, i: r.randn(b, i)])(rng, n_batch, n)
    return base @ (rng.randn(n, n) * rand_weights(rng, 1, n)).t()


def rand_circle_points(rng: PriorRNG, n_batch: int, n: int) -> torch.Tensor:
    # Radial density ∝ r^{n-1}, so radial CDF is F(r)=r^n and inverse CDF r=u^{1/n}.
    return (rng.rand(n_batch, 1) ** (1 / n)) * row_normalize(rng.randn(n_batch, n))


def rand_gauss_mixture_points(rng: PriorRNG, n_batch: int, n: int) -> torch.Tensor:
    n_centers = rng.logint(1, 16)
    center_idxs = rng.multinomial(rand_weights(rng, 1, n_centers).squeeze(0), num_samples=n_batch)
    matrices = rng.randn(n_centers, n, n) * rand_weights(rng, n_centers, n)[:, None, :]
    matrices = matrices * torch.exp(rng.randn(1) + rng.randn(1) * rng.randn(n_centers, 1, 1))
    return rng.randn(n_centers, n)[center_idxs] + (matrices[center_idxs] @ rng.randn(n_batch, n, 1)).squeeze(-1)


def rand_points(rng: PriorRNG, n_batch: int, n: int) -> torch.Tensor:
    # rand_gauss_mixture_points is excluded here upstream because of RAM cost.
    base = rng.choice([rand_cov_points, lambda r, b, i: r.randn(b, i), rand_unif_points, rand_circle_points])
    return rand_func(rng, base(rng, n_batch, n), n)


# ---------------------------------------------------------------------------
# Random functions — the eight families
# ---------------------------------------------------------------------------


def rand_lin_func(rng: PriorRNG, x: torch.Tensor, d_out: int) -> torch.Tensor:
    return x @ (rand_matrix(rng, 1, d_out, x.shape[1])[0].t())


def rand_quad_func(rng: PriorRNG, x: torch.Tensor, d_out: int) -> torch.Tensor:
    idxs = rng.randperm(x.shape[1])[:20] if x.shape[1] > 20 else torch.arange(x.shape[1])
    tensor_3d = rand_matrix(rng, d_out, len(idxs) + 1, len(idxs) + 1)
    x = torch.cat([x[:, idxs], torch.ones(x.shape[0], 1)], dim=-1)  # constant + linear terms
    return torch.einsum("oij,bi,bj->bo", tensor_3d, x, x)


def rand_mlp_func(rng: PriorRNG, x: torch.Tensor, d_out: int) -> torch.Tensor:
    hidden_width = rng.logint(1, 128)
    x = x if rng.boolean() else rand_act(rng, x)
    for _ in range(rng.logint(1, 4) - 1):
        x = rand_act(rng, rand_lin_func(rng, x, hidden_width))
    x = rand_lin_func(rng, x, d_out)
    return x if rng.boolean() else rand_act(rng, x)


def rand_tree_func(rng: PriorRNG, x: torch.Tensor, d_out: int) -> torch.Tensor:
    n_trees = rng.logint(1, 128)
    depth = rng.randint(1, 8)
    feature_imp = torch.clamp(x.std(dim=0, correction=0), 1e-8)
    feature_imp[~torch.isfinite(feature_imp)] = 1e-8
    split_dims = rng.multinomial(feature_imp, n_trees * depth)
    split_points = x[rng.torch_randint(x.shape[0], (n_trees * depth,)), split_dims]
    split_sides = (x[:, split_dims] > split_points).reshape(x.shape[0], n_trees, depth)
    leaf_idxs = torch.einsum("btd,d->bt", split_sides.long(), 2 ** torch.arange(depth, dtype=torch.long))
    tree_idxs = torch.arange(n_trees, dtype=torch.long).expand(x.shape[0], n_trees)
    leaf_values = rng.randn(n_trees, 2**depth, d_out)  # Gaussian leaves -> avoid recursion
    return leaf_values[tree_idxs, leaf_idxs].mean(dim=1)


def rand_discretization_func(rng: PriorRNG, x: torch.Tensor, d_out: int) -> torch.Tensor:
    n_centers = x.shape[0] if x.shape[0] <= 2 else rng.logint(2, min(x.shape[0], 256))
    centers = x[rng.randperm(len(x))[:n_centers]]
    targets = rand_lin_func(rng, centers, d_out)
    dists = torch.cdist(x, centers, p=rng.lognum(0.5, 4.0))
    return targets[dists.argmin(dim=-1)]


def rand_gp_func(rng: PriorRNG, x: torch.Tensor, d_out: int, n_freqs: int = 256) -> torch.Tensor:
    a = rng.lognum(2.0, 20.0)  # global decay rate a > 1

    if rng.boolean():  # standard kernel
        input_tfm = rng.randn(x.shape[1], x.shape[1]) * rand_weights(rng, 1, x.shape[1]).t()
        u = torch.clamp(rng.rand(n_freqs), 1e-6, 1 - 1e-6)
        invcdf = torch.pow(u, 1 / (1 - a)) - 1.0
        freqs = rng.randn(x.shape[1], n_freqs)
        freqs *= invcdf[None, :] / freqs.norm(dim=0, keepdim=True)
        freqs = rng.lognum(0.5, 10.0) * input_tfm @ freqs
    else:  # product kernel
        u = torch.clamp(rng.rand(x.shape[1], n_freqs), 1e-6, 1 - 1e-6)
        freqs = torch.pow(u, 1 / (1 - a)) - 1.0

    bias = 2 * np.pi * rng.rand(1, n_freqs)
    weights = rng.randn(n_freqs, d_out) / np.sqrt(n_freqs)
    return torch.cos(x @ freqs + bias) @ weights


def rand_em_func(rng: PriorRNG, x: torch.Tensor, d_out: int) -> torch.Tensor:
    n_ind = rng.logint(2, max(16, 2 * d_out) + 1)
    x_ind = x[rng.torch_randint(x.shape[0], (n_ind,))] + rng.randn(n_ind, x.shape[1])
    stds = torch.exp(rng.rand(1) * rng.randn(1, n_ind))
    consts = -torch.log(2 * torch.pi * stds**2) * (x.shape[-1] / 2)
    dists = torch.cdist(x, x_ind, p=rng.lognum(1.0, 4.0))
    logits = consts - torch.clamp(dists / stds, min=0.0) ** rng.uniform(1.0, 2.0)
    return rand_lin_func(rng, torch.softmax(logits, dim=-1), d_out)


def rand_prod_func(rng: PriorRNG, x: torch.Tensor, d_out: int) -> torch.Tensor:
    return rand_func(rng, x, d_out, only_cheap=True) * rand_func(rng, x, d_out, only_cheap=True)


CHEAP_FUNCS = [rand_lin_func, rand_quad_func, rand_gp_func, rand_tree_func, rand_discretization_func]
ALL_FUNCS = CHEAP_FUNCS + [rand_mlp_func, rand_em_func, rand_prod_func]


def rand_func(rng: PriorRNG, x: torch.Tensor, d_out: int, only_cheap: bool = False) -> torch.Tensor:
    return rng.choice(CHEAP_FUNCS if only_cheap else ALL_FUNCS)(rng, x, d_out)


def rand_multi_func(rng: PriorRNG, xs: list[torch.Tensor], d_out: int) -> torch.Tensor:
    if rng.boolean():
        return rand_func(rng, torch.cat(xs, dim=-1), d_out)  # concatenate before the function
    out_cat = torch.stack([rand_func(rng, x, d_out) for x in xs], dim=0)
    agg = rng.choice([torch.sum, torch.prod, torch.max, torch.logsumexp])(out_cat, dim=0)
    return agg.values if isinstance(agg, tuple) else agg  # torch.max returns a namedtuple
