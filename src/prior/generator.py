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

from . import upstream
from .base import SyntheticTask, assemble_xy, rand_cat_sizes, rand_dataset_plain
from .filters import PredictabilityFilter
from .grouping import GroupedSampler
from .noise_features import add_noise_features
from .preprocess import process_features, standard_scaling
from .rng import PriorRNG
from .shift import apply_shift
from .targets.lgd import apply_lgd_target
from .targets.pd import apply_informative_missingness, apply_pd_target


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
        # Shift stress lives under the credit path, so credit_fraction=0 is untouched.
        self.shift_cfg = (cfg.get("credit", {}) or {}).get("shift", {}) or {}
        # The midpoint of the train fraction range, used to place the shift boundary
        # where the split is most likely to land. The exact split is chosen later by
        # dataset.py, so this is an approximation on purpose.
        lo, hi = cfg.get("train_frac_range", [0.3, 0.9])
        self.train_frac_mid = float((lo + hi) / 2)
        if not 0.0 <= self.credit_fraction <= 1.0:
            raise ValueError(f"credit_fraction must be in [0, 1], got {self.credit_fraction}")

        self.n_rows_range = tuple(cfg.get("n_rows_range", [512, 1024]))
        self.n_features_range = tuple(cfg.get("n_features_range", [3, 50]))
        self.max_features = int(cfg.get("max_features", 64))
        self.n_nodes_range = tuple(cfg.get("n_nodes_range", [2, 33]))
        self.n_classes = int(cfg.get("n_classes", 2))

        base = cfg.get("base", {})
        # `upstream` = `tabicl.prior.GraphSCM`, the real thing. `transcribed` = the
        # NanoTabICL transcription in `base.py`, kept only so a machine without the default
        # branch of `tabicl` installed can still run the tests. Nothing chooses it silently:
        # `upstream` raises with the install command if the package is short.
        self.base_impl = str(base.get("implementation", "upstream")).lower()
        if self.base_impl not in {"upstream", "transcribed"}:
            raise ValueError(
                f"prior.base.implementation must be 'upstream' or 'transcribed', "
                f"got {self.base_impl!r}"
            )
        if self.base_impl == "upstream":
            upstream.require()
        self.base_max_cat_size = int(base.get("max_cat_size", upstream.UPSTREAM_MAX_CAT_SIZE))
        self.base_category_frequency = base.get("category_frequency", "balanced")

        credit = cfg.get("credit", {})
        self.credit_max_cat_size = int(credit.get("max_cat_size", self.base_max_cat_size))
        self.credit_category_frequency = credit.get("category_frequency", self.base_category_frequency)
        self.credit_target_cfg = credit.get("target", {})
        self.credit_noise_cfg = credit.get("noise_features", {})
        self.credit_missing_cfg = credit.get("missingness", {})

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
    def _sample_candidate(
        self, shape: tuple[int, int] | None = None, use_credit: bool | None = None
    ) -> SyntheticTask:
        # `use_credit` is decided ONCE per slot by `sample()` and held across every rejection
        # attempt. Re-drawing it per attempt (the behaviour until 02-09-2026) over-represented
        # credit: credit datasets are structured and pass the predictability filter more often
        # than base ones, which can draw unlearnable noise and get re-rolled — so a nominal
        # `credit_fraction=0.3` realised ~0.45. Fixing the type and re-sampling only the DATA
        # within it makes the realised fraction equal the nominal knob.
        if use_credit is None:
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

        meta: dict[str, Any] = {"source": source, "n_rows_requested": n_rows}

        if self.base_impl == "upstream":
            # THE CONTROL ARM IS UPSTREAM'S CLASS, CALLED. No code of ours sits between the
            # config and the data, which is the only way "the same prior as TabICLv2" can be
            # a fact rather than a claim about a transcription.
            if not use_credit:
                X, y = upstream.sample_control(
                    regression=self.regression,
                    seq_len=n_gen,
                    num_features=n_features,
                    max_features=self.max_features,
                    num_classes=self.n_classes,
                    # SEEDED FROM OUR STREAM. Upstream draws from torch's GLOBAL generator, so
                    # without this the same `PriorRNG` seed stops reproducing the same dataset
                    # and the base path stops being isolated from anything else in the process.
                    seed=self.rng.randint(0, 2**31 - 1),
                )
                meta["base_impl"] = "upstream.GraphSCM"
                meta["target"] = "base_regression" if self.regression else "base_classification"
                if X.shape[0] > n_rows:
                    X, y = X[:n_rows], y[:n_rows]
                return self._finish(X, y, meta)

            # The credit arms start from the SAME upstream sample, unpadded so the credit
            # columns have somewhere to go. Switching only the control would have made the two
            # differ in two ways at once — the credit structure AND the base implementation.
            X, y_latent = upstream.sample_base_latent(
                seq_len=n_gen, num_features=n_features,
                seed=self.rng.randint(0, 2**31 - 1),
            )
            meta["base_impl"] = "upstream.graph_lib"
        else:
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
            meta["base_impl"] = "transcribed"

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

        # INFORMATIVE MISSINGNESS, both tracks, credit path only. After the target so the
        # missingness can depend on it — a thin credit file is itself a risk signal, which is
        # what `missing_target_coupling` encodes — and before the noise columns so junk
        # columns are not themselves punched full of holes.
        #
        # It lived inside `apply_pd_target` until 25-08-2026, which meant LGD never got it:
        # `apply_lgd_target` receives only the latent, not X. The LGD configs carried a
        # `credit.missingness` block that nothing read.
        if use_credit and self.credit_missing_cfg:
            X, mmeta = apply_informative_missingness(
                self.rng, X, y, self.credit_missing_cfg, self.max_features
            )
            meta.update(mmeta)

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

        # PERMUTE THEN PAD TO max_features, exactly as `GraphSCM.__call__` ends. The control
        # arm already got this inside upstream's class; the credit arms need it here or a
        # mixed batch would hold 100-wide control datasets beside 34-wide credit ones, and
        # the widths themselves would tell the model which prior a dataset came from.
        # Permutation comes FIRST so a noise column or a was-missing flag is as likely to land
        # anywhere as a real feature.
        if self.base_impl == "upstream":
            X = X[..., torch.randperm(X.shape[1])]
            if X.shape[1] < self.max_features:
                X = torch.nn.functional.pad(X, (0, self.max_features - X.shape[1]), value=0.0)
            elif X.shape[1] > self.max_features:
                X = X[..., : self.max_features]

        # Shift stress: arrange the rows so the context/query split falls across a
        # distribution change. O'Prior's ablations find shift-aware stress contributes
        # INDEPENDENTLY of mechanism diversity and realism, and it is the most
        # credit-relevant of the three — a scorecard is always applied to a later,
        # different population than the one it was built on. Applied to OUR datasets
        # only, so the control arm stays exactly TabICL's prior.
        if use_credit and self.shift_cfg:
            X, y, smeta = apply_shift(self.rng, X, y, self.shift_cfg, self.train_frac_mid)
            meta.update(smeta)

        return self._finish(X, y, meta)

    def _finish(self, X: Any, y: Any, meta: dict[str, Any]) -> SyntheticTask:
        """The last three lines every path shares. Extracted so the control arm — which
        returns straight out of `upstream.sample_control` — cannot drift from the credit
        arms on NaN handling or dtype."""
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return SyntheticTask(X=X.float(), y=y.float(), source=meta["source"], meta=meta)

    # -- public --------------------------------------------------------------
    def sample(self, shape: tuple[int, int] | None = None) -> SyntheticTask:
        """Rejection-sample until a task passes the predictability filter.

        Falls back to the last candidate after `max_attempts` rather than looping
        forever. Fallbacks are counted in `meta` so a run that hits the cap often
        is visible in the logs instead of silently biasing the task stream.
        """
        # Decide credit-vs-base ONCE for this slot, before the rejection loop, so the filter
        # cannot skew the realised credit fraction above the nominal knob (see _sample_candidate).
        use_credit = self.rng.boolean(self.credit_fraction)
        last: SyntheticTask | None = None
        for attempt in range(self.max_attempts):
            task = self._sample_candidate(shape, use_credit=use_credit)
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
