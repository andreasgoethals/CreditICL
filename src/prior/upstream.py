"""The REAL TabICLv2 prior, imported from `tabicl`, not transcribed.

WHY THIS FILE EXISTS
--------------------

`src/prior/base.py` opens with *"Transcribed from NanoTabICL's `prior.py`"*. NanoTabICL is a
reimplementation. So until 26-08-2026 the arm this project calls "the original TabICLv2 prior"
was a transcription of a reimplementation, and one divergence had already been found by hand
(`max_cat_size` 100 against upstream's 200). That is the same mistake the model class and the
inference wrapper each made once, and each time the fix was the same: **stop reimplementing and
import the real thing.**

WHY IT WAS NOT DONE SOONER — and this is worth writing down, because it looks like negligence

`pip install tabicl==2.1.1` gives you a package **without** `_graph_scm.py` and **without** the
whole `graph_lib/` subpackage — 29 files. So does `git+https://github.com/soda-inria/tabicl@v2.1.1`.
Only the **default branch** ships them, at the same version string `2.1.1`. Verified by
installing all three and listing the tree. `graph_scm` is what upstream's own stage-1 script
selects (`--prior_type graph_scm`), so the prior TabICLv2 was actually trained on is absent
from every released artefact of TabICLv2.

    pip install "tabicl[pretrain] @ git+https://github.com/soda-inria/tabicl.git"

`[pretrain]` pulls `xgboost`, which `tabicl.prior._tree_scm` imports at module scope, so the
whole `tabicl.prior` package fails to import without it.

WHAT THIS MODULE GUARANTEES
---------------------------

* the **control arm** (`credit_fraction: 0.0`) runs `GraphSCM` **verbatim** — upstream's class,
  upstream's call, no code of ours in the path;
* the **credit arms** are built on the same upstream sample, using upstream's own `Context`,
  `DatasetProperties`, `RandomDataset`, `sample_categorical_sizes`, `outlier_removing` and
  `standard_scaling`, so the base half of a mixed prior is the same distribution as the control.

That second point is the one that matters. Switching only the control would have made the
control and the credit arms differ in TWO ways — the credit structure and the base
implementation — which is worse than having them both wrong in the same way.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from typing import Any

import torch

#: Upstream stage 1: `--graph_noise False --allow_act_warping False --min_n_nodes 2
#: --max_n_nodes 32 --cauchy_dag_offset 0.0`. Everything else stays at `PriorConfig`'s default,
#: which is what upstream leaves it at.
STAGE1_CONFIG: dict[str, Any] = {
    "add_gaussian_noise": False,
    "allow_act_warping": False,
    "min_n_nodes": 2,
    "max_n_nodes": 32,
    "cauchy_dag_offset": 0.0,
}

#: `sample_categorical_sizes(self.num_features, context, max_cat_size=200)` in
#: `_graph_scm.py`. Not 100 — this project believed 100 for weeks.
UPSTREAM_MAX_CAT_SIZE = 200


@lru_cache(maxsize=1)
def availability() -> tuple[bool, str]:
    """`(available, why_not)`. Cached: the import cost is real and the answer cannot change."""
    try:
        import tabicl.prior._graph_scm  # noqa: F401
        from tabicl.prior.graph_lib._config import PriorConfig  # noqa: F401
    except ImportError as exc:
        return False, (
            f"{exc}. `graph_scm` ships ONLY on the default branch — neither PyPI nor the "
            f"v2.1.1 tag has it. Install with:\n"
            f'    pip install "tabicl[pretrain] @ git+https://github.com/soda-inria/tabicl.git"'
        )
    return True, ""


def is_available() -> bool:
    return availability()[0]


def require() -> None:
    """Raise with the install command rather than silently falling back to a transcription."""
    ok, why = availability()
    if not ok:
        raise ImportError(f"upstream TabICLv2 prior unavailable: {why}")


@contextlib.contextmanager
def seeded(seed: int):
    """Make upstream's sampler reproducible under OUR seed, without disturbing the process.

    `GraphSCM` and everything under `graph_lib` draw from torch's GLOBAL generator — they take
    no `generator=` argument. Our `PriorRNG` is a separate, explicitly-seeded stream, so the
    moment the base prior became upstream's, the same `PriorRNG` seed stopped reproducing the
    same dataset and the base path stopped being isolated from anything else in the process
    that touches global randomness.

    TWO generators, not one. `graph_lib` also calls `np.random.standard_cauchy`, `randint`,
    `choice`, `uniform` and `permutation` — six of its modules import numpy — so seeding torch
    alone leaves the sample non-reproducible. Measured: same torch seed, different data.

    Both global states are snapshotted and restored, so nothing outside this block is
    disturbed: `fork_rng` for torch, an explicit get/set for numpy.
    """
    import numpy as np

    # The LEGACY global API on purpose, and the lint rule is wrong here: `graph_lib` calls
    # `np.random.standard_cauchy` / `randint` / `choice` / `uniform` / `permutation`, which read
    # the legacy global state. Seeding a `np.random.Generator` would leave them untouched.
    np_state = np.random.get_state()  # noqa: NPY002
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))  # noqa: NPY002
            yield
    finally:
        np.random.set_state(np_state)  # noqa: NPY002


def prior_config(overrides: dict[str, Any] | None = None) -> Any:
    """Upstream's `PriorConfig`, set to its stage-1 values."""
    require()
    from tabicl.prior.graph_lib._config import PriorConfig

    return PriorConfig(**{**STAGE1_CONFIG, **(overrides or {})})


def sample_control(
    *,
    regression: bool,
    seq_len: int,
    num_features: int,
    max_features: int,
    num_classes: int = 2,
    device: str = "cpu",
    config: Any | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The control arm. `GraphSCM` verbatim — no code of ours between the config and the data.

    Returns exactly what upstream returns: X already outlier-removed, standard-scaled,
    feature-permuted and zero-padded to `max_features`, and y already scaled (regression) or
    label-permuted (classification).
    """
    require()
    from tabicl.prior._graph_scm import GraphSCM

    with seeded(seed) if seed is not None else contextlib.nullcontext():
        return _call_graph_scm(
            GraphSCM, regression, seq_len, num_features, max_features, num_classes,
            config if config is not None else prior_config(), device,
        )


def _call_graph_scm(GraphSCM, regression, seq_len, num_features, max_features, num_classes,
                    config, device):  # noqa: N803
    scm = GraphSCM(
        regression=regression,
        seq_len=seq_len,
        num_features=num_features,
        max_features=max_features,
        num_classes=num_classes,
        config=config,
        device=device,
    )
    return scm()


def sample_base_latent(
    *,
    seq_len: int,
    num_features: int,
    device: str = "cpu",
    config: Any | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The credit arms' starting point: `(X, y_latent)`, UNPADDED and UNPERMUTED.

    Same upstream objects as `GraphSCM.__call__`, stopping one step earlier. Two deliberate
    differences from `sample_control`, both forced by what happens next:

    * **unpadded** — the credit path appends noise columns and was-missing indicators, and
      padding to `max_features` first would leave nowhere to put them;
    * **unpermuted** — the permutation is applied once at the end, after those columns exist,
      so a junk column is as likely to land anywhere as a real one.

    `y` is `y_num`, the CONTINUOUS latent, because every credit mechanism (collateral,
    workout, the Vasicek factor model) needs a risk score rather than a label. That is the
    same tensor `GraphSCM` uses for its regression target, put through upstream's own
    `outlier_removing` and `standard_scaling`.
    """
    require()
    with seeded(seed) if seed is not None else contextlib.nullcontext():
        return _sample_base_latent(seq_len, num_features, device, config)


def _sample_base_latent(seq_len, num_features, device, config):
    from tabicl.prior._reg2cls import outlier_removing, standard_scaling
    from tabicl.prior.graph_lib._base import Context, DatasetProperties
    from tabicl.prior.graph_lib._dataset import RandomDataset
    from tabicl.prior.graph_lib._properties import sample_categorical_sizes

    context = Context(config=config if config is not None else prior_config(), device=device)
    properties = DatasetProperties(
        n_train=seq_len,
        n_test=0,
        cat_sizes={
            "x": sample_categorical_sizes(
                num_features, context, max_cat_size=UPSTREAM_MAX_CAT_SIZE
            ),
            "y": [0],  # numerical target: we want the latent, not a class index
        },
    )
    data = RandomDataset(context).sample(properties).get_concat_tensors()

    parts = [t for t in (data.get("x_cat"), data.get("x_num")) if t is not None]
    if not parts:
        raise ValueError("upstream prior returned no features")
    X = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    X = standard_scaling(outlier_removing(X.float(), threshold=4))

    y = standard_scaling(outlier_removing(data["y_num"].float(), threshold=4)).view(-1)
    return X, y
