"""Task generator: the original TabICL prior, our credit prior, and the mixture.

The central lever, per the design brief: ``credit_fraction`` is the probability
that a given synthetic dataset comes from the credit-targeted path rather than
the original one. ``0.0`` recovers arm A exactly; ``1.0`` is a pure credit prior;
anything between is a mixture. Sweeping it (``credit_fraction: [0.0, 0.25, 0.5,
1.0]``) is how we measure whether domain-targeting helps monotonically, saturates,
or needs the general prior's diversity to remain useful.

A NOTE ON WHAT ARM A ACTUALLY DOES FOR CLASSIFICATION — this corrects a claim
made earlier in this project's own notes. Two different code paths exist upstream:

* the **v1** path (`_reg2cls.py`, `MulticlassAssigner`) cuts the latent at a
  *uniformly random data row*, making the binary minority rate roughly
  Uniform(0,1);
* the **v2 / graph_scm** path — which NanoTabICL and therefore this project use —
  takes the class label straight from the *categorical converter*
  (`rand_converter` with `cat_size=n_classes`), i.e. nearest-centre or softmax
  assignment over random points.

So arm A's base-rate distribution here is a property of the converter geometry,
**not** of a random-row cut, and it is an empirical question rather than something
to be asserted. `scripts/measure_prior.py` answers it. Do not repeat the
random-row claim about the v2 path.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import SyntheticTask, assemble_xy, rand_cat_sizes, rand_dataset_plain
from .filters import PredictabilityFilter
from .grouping import GroupedSampler
from .noise_features import add_noise_features
from .preprocess import process_features, standard_scaling
from .rng import PriorRNG
from .targets.lgd import apply_lgd_target
from .targets.pd import apply_pd_target


class TaskGenerator:
    """Samples in-context episodes from a configured prior.

    Parameters come from the ``prior`` block of a run config; see
    `config/priors/lgd.yaml` and `config/priors/pd.yaml`.
    """

    def __init__(self, cfg: dict[str, Any], task: str, rng: PriorRNG):
        self.cfg = cfg
        self.task = task  # "lgd" (regression) or "pd" (binary classification)
        self.rng = rng
        self.regression = task == "lgd"

        self.credit_fraction = float(cfg.get("credit_fraction", 0.0))
        if not 0.0 <= self.credit_fraction <= 1.0:
            raise ValueError(f"credit_fraction must be in [0, 1], got {self.credit_fraction}")

        self.n_rows_range = tuple(cfg.get("n_rows_range", [512, 1024]))
        self.n_features_range = tuple(cfg.get("n_features_range", [3, 50]))
        self.max_features = int(cfg.get("max_features", 64))
        self.n_nodes_range = tuple(cfg.get("n_nodes_range", [2, 33]))
        self.n_classes = int(cfg.get("n_classes", 2))

        base = cfg.get("base", {})
        self.base_max_cat_size = int(base.get("max_cat_size", 100))
        self.base_category_frequency = base.get("category_frequency", "balanced")

        credit = cfg.get("credit", {})
        self.credit_max_cat_size = int(credit.get("max_cat_size", self.base_max_cat_size))
        self.credit_category_frequency = credit.get("category_frequency", self.base_category_frequency)
        self.credit_target_cfg = credit.get("target", {})
        self.credit_noise_cfg = credit.get("noise_features", {})

        # TabICL's correlated hyperparameter sampling: datasets inside a group
        # share a shape and a difficulty, so a batch contains relatives rather
        # than strangers. ON by default — see src/prior/grouping.py for why.
        self.groups = GroupedSampler(cfg, rng)

        fcfg = cfg.get("filter", {})
        self.filter = PredictabilityFilter(
            mode=fcfg.get("mode", "tabicl"),
            quantile_band=tuple(fcfg.get("quantile_band", [0.02, 0.45])),
            remove_trivial=bool(fcfg.get("remove_trivial", False)),
            trivial_threshold=float(fcfg.get("trivial_threshold", 0.1)),
        )
        self.max_attempts = int(cfg.get("max_filter_attempts", 40))

    # -- sizes ---------------------------------------------------------------
    def group_summary(self) -> dict:
        return self.groups.describe()

    def sample_shape(self) -> tuple[int, int]:
        """Draw (n_rows, n_features) for one dataset — or for a whole batch.

        Batches share a shape because the model consumes a dense
        (batch, rows, cols) tensor. TabICL solves this the same way, via
        `batch_size_per_gp`: datasets within a group share characteristics.
        """
        shared = self.groups.next_dataset()
        if shared:
            return int(shared["n_rows"]), int(shared["n_features"])
        n_rows = self.rng.randint(self.n_rows_range[0], self.n_rows_range[1] + 1)
        n_features = self.rng.randint(self.n_features_range[0], min(self.n_features_range[1], self.max_features) + 1)
        return n_rows, n_features

    # -- one candidate -------------------------------------------------------
    def _sample_candidate(self, shape: tuple[int, int] | None = None) -> SyntheticTask:
        use_credit = self.rng.boolean(self.credit_fraction)
        source = "credit" if use_credit else "base"

        n_rows, n_features = shape if shape is not None else self.sample_shape()
        max_cat = self.credit_max_cat_size if use_credit else self.base_max_cat_size
        cat_freq = self.credit_category_frequency if use_credit else self.base_category_frequency

        # Underwriting selection removes rows, so oversample to land on n_rows.
        # The +8 margin is not cosmetic: without it, rounding in the selection
        # step can leave n_rows-1 rows, and a batch of tasks with different row
        # counts cannot be stacked into a dense tensor.
        n_gen = n_rows
        if use_credit and self.task == "pd":
            drop = float(self.credit_target_cfg.get("selection", {}).get("selection_drop", 0.0))
            if drop > 0:
                n_gen = int(round(n_rows / max(1.0 - min(drop, 0.5), 0.5))) + 8

        x_cat_sizes = rand_cat_sizes(self.rng, n_features, max_cat_size=max_cat)
        y_cat_sizes = [0 if self.regression else self.n_classes]

        columns = rand_dataset_plain(
            self.rng,
            x_cat_sizes,
            y_cat_sizes,
            n_gen,
            n_nodes_range=self.n_nodes_range,
            category_frequency=cat_freq,
        )
        X, y_latent = assemble_xy(columns, n_features)
        meta: dict[str, Any] = {"source": source, "n_rows_requested": n_rows}

        if self.regression:
            if use_credit:
                y, tmeta = apply_lgd_target(self.rng, y_latent, self.credit_target_cfg)
            else:
                # Arm A: the converter may already have Kumaraswamy-warped this
                # column; standard scaling is what `GraphSCM.__call__` applies.
                y = standard_scaling(y_latent.unsqueeze(-1)).squeeze(-1)
                tmeta = {"target": "base_regression", "target_scaling": "standard"}
            X = process_features(X)
        else:
            if use_credit:
                # Feed the raw graph latent, not the converter's class index, so
                # base rate and rules operate on a continuous risk score.
                X = process_features(X)
                X, y, tmeta = apply_pd_target(self.rng, X, y_latent, self.credit_target_cfg, self.max_features)
            else:
                # Arm A: the label IS the converter output for the y node.
                y = columns["y_0"].float().reshape(-1)
                perm = self.rng.randperm(self.n_classes)
                y = perm[y.long().clamp(0, self.n_classes - 1)].float()
                X = process_features(X)
                tmeta = {"target": "base_classification"}

        meta.update(tmeta)

        # Irrelevant columns, on the credit path only. Real credit files carry
        # plenty of columns that turn out to be useless, and the base prior only
        # produces them as a side effect of random graph geometry — it cannot
        # dial the count. Added AFTER the target so a shuffled column is
        # guaranteed to carry no signal about y.
        if use_credit and self.credit_noise_cfg:
            X, nmeta = add_noise_features(self.rng, X, self.credit_noise_cfg, self.max_features)
            meta.update(nmeta)

        # Trim to the requested row count after any selection step.
        if X.shape[0] > n_rows:
            X, y = X[:n_rows], y[:n_rows]

        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return SyntheticTask(X=X.float(), y=y.float(), source=source, meta=meta)

    # -- public --------------------------------------------------------------
    def sample(self, shape: tuple[int, int] | None = None) -> SyntheticTask:
        """Rejection-sample until a task passes the predictability filter.

        Falls back to the last candidate after `max_attempts` rather than looping
        forever. Fallbacks are counted in `meta` so a run that hits the cap often
        is visible in the logs instead of silently biasing the task stream.
        """
        last: SyntheticTask | None = None
        for attempt in range(self.max_attempts):
            task = self._sample_candidate(shape)
            last = task
            if self.filter.accept(task.X, task.y, is_classif=not self.regression):
                task.meta["filter_attempts"] = attempt + 1
                return task
        assert last is not None
        last.meta["filter_attempts"] = self.max_attempts
        last.meta["filter_fallback"] = True
        return last

    def filter_summary(self) -> dict[str, float]:
        return self.filter.stats.summary()
