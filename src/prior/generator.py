"""Task generator: the original TabICL prior, our credit prior, and the mixture.

``credit_fraction`` is the probability that a slot of the task stream holds a dataset from the
credit path rather than TabICLv2's own prior (`src/prior/stream.py` decides which slots). ``0.0``
is the control — upstream's `GraphSCM`, called — and ``1.0`` a pure credit prior.

EVERY DATASET MUST REACH THE MODEL THE WAY UPSTREAM'S DO, AND LOOK LIKE A REAL TABLE DOES AT
PREDICTION TIME. The model is trained on what this module returns and later asked to predict on
real data pushed through `TabICLClassifier` / `TabICLRegressor`. Both ends are upstream's:

* TRAINING (upstream `GraphSCM.__call__`, `GraphPrior.generate_dataset`): features outlier-clipped
  at 4 SD and standardised, permuted, zero-padded; a regression target outlier-clipped and
  standardised; class labels randomly permuted; constant columns deleted and the table compacted
  (`delete_unique_features`); a classification table kept only if the context and the query hold
  the same set of classes (`cls_sanity_check`); unpredictable tables rejected (`should_filter`).
* PREDICTION (upstream `TabICLClassifier.fit`/`predict_proba`): categories as integer codes,
  missing numbers mean-imputed WITHOUT indicator columns, constant columns dropped, every
  ensemble member standardising and 4-SD-clipping the features on the context rows, class labels
  rotated between members, a regression target standardised on the context (`y_scaler_`).

So the credit path ends in the same place as the control — standardised features, no indicator
columns, labels of random identity, a standardised LGD target, no constant columns, the same
sanity check and filter — and it gets there by upstream's PREDICTION code: its last step is
`prediction_view`, upstream's own `UniqueFeatureFilter` + `PreprocessingPipeline("none")` fitted
on the context rows, and an LGD target is standardised on the context as `y_scaler_` does. A
credit table therefore reaches the model in training exactly as a real table of the same values
would at prediction (`tests/test_train_predict_consistency.py`). The control stays upstream's
training encoding, untouched; without a shift the two differ by sampling noise only. What the
credit path adds is credit STRUCTURE, never a different encoding. The details, and what changed
on 28-09-2026, are in each step below.
"""

from __future__ import annotations

from typing import Any

import torch

from . import upstream
from .base import SyntheticTask, assemble_xy, rand_cat_sizes, rand_dataset_plain
from .filters import PredictabilityFilter
from .noise_features import add_noise_features
from .preprocess import process_features, standard_scaling
from .rng import PriorRNG
from .shift import apply_shift
from .stream import SlotPlan, max_classes_of, slot_seed
from .targets.lgd import apply_lgd_target, apply_lgd_zoib
from .targets.pd import apply_informative_missingness, apply_pd_target, max_selection_drop


def delete_constant_columns(X: torch.Tensor, width: int) -> tuple[torch.Tensor, int]:
    """Upstream's `Prior.delete_unique_features` for one table: drop columns with a single value
    among the first `width`, compact the rest to the left, zero-pad back to `X.shape[1]`.

    A constant column carries nothing, upstream's training never shows one, and upstream's
    prediction drops it (`UniqueFeatureFilter`). Returns `(X, d)` with `d` real columns."""
    n_cols = X.shape[1]
    width = min(int(width), n_cols)
    # Among the OBSERVED values: under the raw encoding a missing cell is NaN, and `unique` counts
    # every NaN as a value of its own, which would keep an all-missing column.
    keep = [j for j in range(width) if torch.unique(X[:, j][~torch.isnan(X[:, j])]).numel() > 1]
    out = torch.zeros_like(X)
    if keep:
        out[:, : len(keep)] = X[:, keep]
    return out, len(keep)


def same_classes_on_both_sides(y: torch.Tensor, train_size: int, min_classes: int = 2) -> bool:
    """Upstream's `is_valid_split`: the context and the query hold the same set of classes, and
    at least `min_classes` of them. A class seen only in the query cannot be predicted from the
    context, and a class seen only in the context teaches nothing about it."""
    if train_size <= 0 or train_size >= y.shape[0]:
        return False
    tr = set(torch.unique(y[:train_size]).tolist())
    te = set(torch.unique(y[train_size:]).tolist())
    return tr == te and len(tr) >= min_classes


def prediction_view(X: torch.Tensor, width: int, train_size: int) -> tuple[torch.Tensor, int]:
    """A table as upstream's prediction path hands a real one to the model, returned as `(X, d)`.

    `TabICLClassifier` / `TabICLRegressor` fit everything on the CONTEXT rows and apply it to
    every row: `UniqueFeatureFilter` drops a column constant in the context, then each ensemble
    member's `PreprocessingPipeline` standardises on the context (`CustomStandardScaler`) and
    soft-clips beyond 4 context SDs (`OutlierRemover`, a log-compressed clip, not a clamp). This
    calls upstream's own classes, with the `"none"` normalisation the training prior matches (the
    other half of the ensemble adds a Yeo-Johnson step no training table ever gets, upstream's
    included).

    Upstream's own training standardises on the WHOLE table instead. Without a shift the two
    views differ by sampling noise; with one they do not: whole-table statistics describe the
    context by numbers that include the shifted query rows, which no real table can do."""
    from tabicl._sklearn.preprocessing import PreprocessingPipeline

    n_cols = X.shape[1]
    width, ts = min(int(width), n_cols), int(train_size)
    keep = [j for j in range(width) if torch.unique(X[:ts, j]).numel() > 1]
    out = torch.zeros_like(X)
    if keep:
        cols = X[:, keep].double().numpy()
        pipe = PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0).fit(cols[:ts])
        out[:, : len(keep)] = torch.from_numpy(pipe.transform(cols)).to(X.dtype)
    return out, len(keep)


def standardise_on_context(y: torch.Tensor, train_size: int) -> torch.Tensor:
    """`TabICLRegressor`'s `y_scaler_`: a scikit-learn `StandardScaler` (population SD, and a
    zero SD treated as 1) fitted on the context targets, applied to all of them."""
    ctx = y[: int(train_size)].double()
    mean, sd = ctx.mean(), ctx.std(unbiased=False)
    sd = torch.where(sd > 10 * torch.finfo(torch.float64).eps, sd, torch.ones_like(sd))
    return ((y.double() - mean) / sd).to(y.dtype)


def warp_marginals(rng: PriorRNG, X: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """Give a share of the continuous columns a long tail, then clip and standardise again.

    `x -> expm1(s * z) / s` on the standardised column `z` is monotone, so the ranks — every
    split a tree could make, and the column's dependence on the target — are unchanged; only
    the marginal changes, towards the log-normal shape of money amounts (a share of warps is
    mirrored, for left tails). The re-processing is upstream's own (`outlier_removing` at 4 SD,
    then `standard_scaling`), which is what GraphSCM does to every feature it returns.

    `col_fraction_range`: share of continuous columns warped, drawn per table.
    `strength_range`: the skew parameter `s`, drawn per column.
    """
    lo, hi = cfg.get("col_fraction_range", [0.0, 0.0])
    frac = float(rng.uniform(float(lo), float(hi))) if float(hi) > float(lo) else float(hi)
    if frac <= 0 or X.shape[1] == 0:
        return X, {"warped_cols": 0}
    s_lo, s_hi = (float(v) for v in cfg.get("strength_range", [0.5, 1.5]))
    left_prob = float(cfg.get("left_tail_prob", 0.2))
    cont = [j for j in range(X.shape[1]) if torch.unique(X[:, j]).numel() > 20]
    k = int(round(frac * len(cont)))
    if k == 0:
        return X, {"warped_cols": 0}
    X = X.clone()
    for idx in rng.randperm(len(cont))[:k].tolist():
        j = cont[idx]
        z = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-8)
        s = rng.uniform(s_lo, s_hi)
        sign = -1.0 if rng.boolean(left_prob) else 1.0
        X[:, j] = sign * torch.expm1(s * z.clamp(-6.0, 6.0)) / s
    return process_features(X), {"warped_cols": k}


def split_noise_budget(num_features: int, fraction: float) -> tuple[int, int]:
    """`(informative, noise)` columns summing to `num_features`.

    `fraction` keeps its old meaning — noise columns per informative column — but the TOTAL is
    now the slot's feature count, shared with every other table in its subgroup. Appending noise
    on top made credit tables wider than their neighbours, and the width itself told the model
    which prior a table came from."""
    if fraction <= 0 or num_features <= 1:
        return num_features, 0
    informative = max(1, int(round(num_features / (1.0 + fraction))))
    return informative, num_features - informative


class TaskGenerator:
    """Samples in-context episodes from a configured prior.

    `sample_slot(plan)` is what training uses: one slot of the common task stream, fully
    determined by `plan` and `stream_seed`. `sample()` draws a stand-alone task for plots and
    reports, with the same code path.
    """

    def __init__(self, cfg: dict[str, Any], task: str, rng: PriorRNG, stream_seed: int | None = None):
        self.cfg = cfg
        self.task = task  # "lgd" (regression) or "pd" (binary classification)
        self.rng = rng
        self.regression = task == "lgd"
        self.stream_seed = int(rng.seed if stream_seed is None else stream_seed)
        self._standalone = 0  # counter for `sample()`, so repeated calls differ

        self.credit_fraction = float(cfg.get("credit_fraction", 0.0))
        if not 0.0 <= self.credit_fraction <= 1.0:
            raise ValueError(f"credit_fraction must be in [0, 1], got {self.credit_fraction}")
        # Shift stress lives under the credit path, so credit_fraction=0 is untouched.
        self.shift_cfg = (cfg.get("credit", {}) or {}).get("shift", {}) or {}
        lo, hi = cfg.get("train_frac_range", [0.3, 0.9])
        self.train_frac_range = (float(lo), float(hi))

        self.n_rows_range = tuple(cfg.get("n_rows_range", [1024, 1024]))
        self.n_features_range = tuple(cfg.get("n_features_range", [1, 100]))
        self.max_features = int(cfg.get("max_features", 100))
        self.n_nodes_range = tuple(cfg.get("n_nodes_range", [2, 33]))
        # Upstream draws 2..max_classes classes per dataset (`--max_classes 10`); a credit PD
        # table is always binary. `max_classes: 2` recovers a binary-only control.
        self.max_classes = max_classes_of(cfg)

        base = cfg.get("base", {})
        # `upstream` = `tabicl.prior.GraphSCM`, the real thing. `transcribed` = the NanoTabICL
        # transcription in `base.py`, kept only so a machine without the default branch of
        # `tabicl` installed can still run the tests. Nothing chooses it silently.
        self.base_impl = str(base.get("implementation", "upstream")).lower()
        if self.base_impl not in {"upstream", "transcribed"}:
            raise ValueError(
                f"prior.base.implementation must be 'upstream' or 'transcribed', got {self.base_impl!r}"
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
        self.credit_missing_cfg = dict(credit.get("missingness", {}) or {})
        if self.credit_missing_cfg.get("missing_indicators", False):
            # Real tables reach the model through upstream's wrapper, which mean-imputes and adds
            # NO indicator column; a prior that adds them trains on a column that never exists at
            # prediction time. The signal must live in the imputed value itself.
            raise ValueError(
                "prior.credit.missingness.missing_indicators must be false: TabICL's prediction "
                "path mean-imputes without indicator columns, so training on indicators would "
                "teach the model a column it never sees on real data."
            )
        self.credit_marginals_cfg = dict(credit.get("marginals", {}) or {})
        # WHICH MODEL'S PREDICTION PATH a credit table imitates. `tabicl` (default): upstream
        # TabICL's — mean-imputed gaps, `prediction_view`, an LGD target standardised on the
        # context. `raw`: the table as a real dataset is handed to TabPFN, which does its own
        # preprocessing — gaps stay NaN, features and the [0, 1] LGD target are left as they are.
        # The control tables are upstream's either way.
        self.encoding = str(cfg.get("encoding", "tabicl"))
        if self.encoding not in ("tabicl", "raw"):
            raise ValueError(f"prior.encoding must be 'tabicl' or 'raw', got {self.encoding!r}")
        self.lgd_scaling = str(self.credit_target_cfg.get("target_scaling", "standard"))
        if self.lgd_scaling not in ("standard", "none"):
            raise ValueError(f"unknown target_scaling {self.lgd_scaling!r}; expected 'standard' or 'none'")

        fcfg = cfg.get("filter", {})
        # WHICH TABLES THE FILTER JUDGES. `all` or `base`: only TabICL's own tables, exactly as
        # TabICLv2 filters them. The credit prior is specified from the literature (docs/PRIORS.md),
        # and re-selecting its tables by learnability would change that specification — so the
        # experiments use `base`.
        self.filter_scope = str(fcfg.get("apply_to", "all"))
        if self.filter_scope not in ("all", "base"):
            raise ValueError(f"prior.filter.apply_to must be 'all' or 'base', got {self.filter_scope!r}")
        self.filter = PredictabilityFilter(
            mode=fcfg.get("mode", "tabicl"),
            quantile_band=tuple(fcfg.get("quantile_band", [0.02, 0.45])),
            remove_trivial=bool(fcfg.get("remove_trivial", False)),
            trivial_threshold=float(fcfg.get("trivial_threshold", 0.1)),
        )
        self.max_attempts = int(cfg.get("max_filter_attempts", 40))
        self.max_invalid_attempts = int(cfg.get("max_invalid_attempts", 100))
        if min(self.max_attempts, self.max_invalid_attempts) < 1:
            raise ValueError("prior attempt limits must be positive")
        self.invalid_candidates = 0
        self.sanity_rejections = 0
        self.filter_fallbacks = 0

    # -- summaries -----------------------------------------------------------
    def group_summary(self) -> dict:
        g = self.cfg.get("grouping", {}) or {}
        return {"grouping_enabled": int(g.get("group_size", 4)) > 1,
                "group_size": int(g.get("group_size", 4)),
                "subgroup_size": int(g.get("subgroup_size", g.get("group_size", 4)))}

    def filter_summary(self) -> dict[str, float]:
        return {**self.filter.stats.summary(), "invalid_candidates": self.invalid_candidates,
                "sanity_rejections": self.sanity_rejections, "filter_fallbacks": self.filter_fallbacks}

    # -- one slot of the stream ------------------------------------------------
    def sample_slot(self, plan: SlotPlan) -> SyntheticTask:
        """Rejection-sample the slot until its table passes validity, the class check and the
        filter. Attempt `k` uses seed `(stream, batch, slot, kind, k)`, so a rejection never
        shifts any other slot's draws, and the slot's kind (credit or base) never changes."""
        kind = "credit" if plan.use_credit else "base"
        last: SyntheticTask | None = None
        attempts = invalid = 0
        k = 0
        while attempts < self.max_attempts and invalid < self.max_invalid_attempts:
            seed = slot_seed(self.stream_seed, plan.batch, plan.slot, kind, k)
            k += 1
            rng = PriorRNG(seed)
            task = self._candidate(plan, rng, seed)
            if task is None or not self._valid(task, plan):
                invalid += 1
                self.invalid_candidates += 1
                continue
            attempts += 1
            last = task
            d = int(task.meta["d"])
            judged = not (plan.use_credit and self.filter_scope == "base")
            if not judged or self.filter.accept(task.X[:, :d], task.y, is_classif=not self.regression):
                task.meta.update(filter_attempts=attempts, invalid_attempts=invalid)
                return task
        if last is None:
            raise RuntimeError(f"prior produced {invalid} invalid tasks without a valid candidate "
                               f"(task={self.task}, source={kind})")
        last.meta.update(filter_attempts=attempts, invalid_attempts=invalid, filter_fallback=True)
        self.filter_fallbacks += 1
        return last

    def _valid(self, task: SyntheticTask, plan: SlotPlan) -> bool:
        """Hard validity, then upstream's class check (which may repair the split by permuting
        rows, as upstream does, unless the rows were deliberately ordered by a shift)."""
        X, y = task.X, task.y
        # Under the raw encoding a gap is NaN by design; anything infinite is still invalid.
        x_ok = (not torch.isinf(X).any()) if self.encoding == "raw" else bool(torch.isfinite(X).all())
        if (X.ndim != 2 or y.ndim != 1 or len(y) != len(X) or len(y) < 4
                or not torch.isfinite(y).all() or not x_ok
                or torch.unique(y).numel() < 2 or int(task.meta.get("d", 0)) < 1):
            return False
        if self.regression:
            return True
        # GraphSCM returns y=-100 on invalid graphs; labels must be 0..n_classes-1 integers.
        n_cls = int(plan.num_classes or 2)
        if not bool(((y >= 0) & (y < n_cls) & (y == y.round())).all()):
            return False
        ts = int(plan.train_size)
        if same_classes_on_both_sides(y, ts):
            return True
        if task.meta.get("shift", "none") != "none":
            self.sanity_rejections += 1
            return False  # the row order IS the shift; permuting it would undo it
        # Upstream's repair: up to 10 random row permutations.
        rng = PriorRNG(slot_seed(self.stream_seed, plan.batch, plan.slot, "classes", 99))
        for _ in range(10):
            perm = rng.randperm(len(y))
            if same_classes_on_both_sides(y[perm], ts):
                task.X, task.y = X[perm], y[perm]
                return True
        self.sanity_rejections += 1
        return False

    # -- one candidate ---------------------------------------------------------
    def _candidate(self, plan: SlotPlan, rng: PriorRNG, seed: int) -> SyntheticTask | None:
        meta: dict[str, Any] = {"source": "credit" if plan.use_credit else "base",
                                "n_rows_requested": plan.seq_len, "num_features": plan.num_features}
        if plan.use_credit:
            X, y = self._credit_candidate(plan, rng, seed, meta)
        else:
            X, y = self._base_candidate(plan, rng, seed, meta)
        if X is None:
            return None
        keep_nan = self.encoding == "raw" and plan.use_credit
        X = torch.nan_to_num(X.float(), nan=float("nan") if keep_nan else 0.0, posinf=0.0, neginf=0.0)
        X, d = delete_constant_columns(X, plan.num_features)
        meta["d"] = d
        return SyntheticTask(X=X, y=y.float(), source=meta["source"], meta=meta)

    def _base_candidate(self, plan: SlotPlan, rng: PriorRNG, seed: int, meta: dict) -> tuple:
        """The control: upstream's `GraphSCM`, called with the slot's shape and class count.
        It returns X clipped, standardised, permuted and padded to `max_features`, and y
        standardised (regression) or label-permuted (classification) — nothing of ours in
        between."""
        if self.base_impl == "upstream":
            X, y = upstream.sample_control(
                regression=self.regression, seq_len=plan.seq_len, num_features=plan.num_features,
                max_features=self.max_features, num_classes=int(plan.num_classes or 2), seed=seed,
            )
            meta["base_impl"] = "upstream.GraphSCM"
        else:
            n_cls = int(plan.num_classes or 2)
            cols = rand_dataset_plain(rng, rand_cat_sizes(rng, plan.num_features, max_cat_size=self.base_max_cat_size),
                                      [0 if self.regression else n_cls], plan.seq_len,
                                      n_nodes_range=self.n_nodes_range,
                                      category_frequency=self.base_category_frequency)
            X, y_latent = assemble_xy(cols, plan.num_features)
            X = process_features(X)
            if self.regression:
                y = standard_scaling(y_latent.unsqueeze(-1)).squeeze(-1)
            else:
                y = cols["y_0"].float().reshape(-1)
                y = rng.randperm(n_cls)[y.long().clamp(0, n_cls - 1)].float()
            X = self._permute_and_pad(rng, X)
            meta["base_impl"] = "transcribed"
        meta["target"] = "base_regression" if self.regression else "base_classification"
        if not self.regression:
            meta["num_classes"] = int(plan.num_classes or 2)
        return X[: plan.seq_len], y[: plan.seq_len]

    def _credit_candidate(self, plan: SlotPlan, rng: PriorRNG, seed: int, meta: dict) -> tuple:
        n_rows = plan.seq_len
        informative, n_noise = split_noise_budget(
            plan.num_features, float(self.credit_noise_cfg.get("fraction", 0.0)) if self.credit_noise_cfg else 0.0
        )
        # Underwriting selection removes rows, so oversample (+8 absorbs rounding).
        n_gen = n_rows
        if not self.regression:
            drop = max_selection_drop(self.credit_target_cfg.get("selection", {}) or {})
            if drop > 0:
                n_gen = int(round(n_rows / max(1.0 - min(drop, 0.5), 0.5))) + 8

        # The credit arms start from the SAME upstream graph sampler as the control, unpadded
        # and unpermuted so the credit columns have somewhere to go.
        if self.base_impl == "upstream":
            X, y_latent = upstream.sample_base_latent(seq_len=n_gen, num_features=informative, seed=seed)
            meta["base_impl"] = "upstream.graph_lib"
        else:
            cols = rand_dataset_plain(rng, rand_cat_sizes(rng, informative, max_cat_size=self.credit_max_cat_size),
                                      [0], n_gen, n_nodes_range=self.n_nodes_range,
                                      category_frequency=self.credit_category_frequency)
            X, y_latent = assemble_xy(cols, informative)
            meta["base_impl"] = "transcribed"
        if not (torch.isfinite(X).all() and torch.isfinite(y_latent).all()):
            # A NaN graph (about 1 draw in 60): upstream marks such a dataset invalid
            # (`GraphSCM.__call__` sets y to -100) and it is redrawn. Here it used to be zeroed
            # by `nan_to_num` and kept, rows of zeros with made-up labels.
            return None, None

        if self.regression:
            # The raw [0, 1] target first: missingness below couples to it on that scale, and
            # the standardisation that matches prediction is applied last.
            tcfg = {**self.credit_target_cfg, "target_scaling": "none"}
            if str(tcfg.get("mode", "quantile")) == "zoib":
                # The literature's LGD process reads the features (its three predictors each get
                # a direction in feature space), so they are processed first.
                X = process_features(X)
                X, y, tmeta = apply_lgd_zoib(rng, X, y_latent, tcfg.get("zoib", {}) or {}, self.max_features)
            else:
                y, tmeta = apply_lgd_target(rng, y_latent, tcfg)
                X = process_features(X)
        else:
            X = process_features(X)
            X, y, tmeta = apply_pd_target(rng, X, y_latent, self.credit_target_cfg, self.max_features)
        meta.update(tmeta)

        # THE SLOT'S WIDTH IS FIXED. A target that ADDS a column (the period factor, recorded as a
        # macroeconomic variable) takes it from the noise budget, so the table keeps the feature
        # count its group shares, as upstream's tables do. Without this the table came out one
        # column wider than `plan.num_features`, and `delete_constant_columns` — which scans only
        # that many — zeroed its last real column (found by the train/predict round-trip test).
        added = int(X.shape[1]) - informative
        if added > 0:
            take = min(added, n_noise)
            n_noise -= take
            over = added - take
            if over > 0:
                # No noise column left to give up: drop the added column(s) — the period factor
                # then stays an unrecorded driver, as in the tables that never record it.
                X = X[:, : X.shape[1] - over]
                meta["macro_feature"] = False
                meta["target_columns_dropped"] = over

        # Long-tailed marginals: amounts, balances and incomes are right-skewed, and real credit
        # columns are far more skewed than the graph prior's (median |skew| 1.5 against 0.5).
        if self.credit_marginals_cfg:
            X, wmeta = warp_marginals(rng, X, self.credit_marginals_cfg)
            meta.update(wmeta)

        # Irrelevant columns, inside the slot's feature budget (see `split_noise_budget`).
        if n_noise > 0:
            X, nmeta = add_noise_features(rng, X, self.credit_noise_cfg, self.max_features, n_add=n_noise)
            meta.update(nmeta)

        X, y = X[:n_rows], y[:n_rows]
        if X.shape[0] < n_rows:
            return None, None  # selection left too few rows; resample the slot
        width = int(X.shape[1])  # the table's real columns; zero padding follows them

        # PERMUTE THEN PAD, exactly as `GraphSCM.__call__` ends, so a noise column is as likely
        # to land anywhere as a real feature.
        X = self._permute_and_pad(rng, X)

        # Shift stress: order the rows so the context/query split — known exactly here — falls
        # across a change of population.
        if self.shift_cfg:
            X, y, smeta = apply_shift(rng, X, y, self.shift_cfg, plan.train_size / n_rows)
            meta.update(smeta)
        # Otherwise the rows go in a random order, so the split is random: upstream's rows are
        # i.i.d. and a cross-validation fold is a random split. The PD mechanism stacks vintages
        # in contiguous row blocks (`sample_cohort_factor`); until 28-09-2026 nothing reshuffled
        # them, so every table's context was its early vintages and its query the late ones — a
        # cohort shift in every table, not in the `shift_prob` share the config asks for.
        if meta.get("shift", "none") == "none":
            order = rng.randperm(X.shape[0])
            X, y = X[order], y[order]

        # UPSTREAM'S CLASS CHECK, BEFORE ANYTHING IS FITTED ON THE CONTEXT. A table whose context
        # and query hold different classes is repaired by permuting its rows (`_valid`, as
        # `GraphSCM` does). For a credit table that has to happen here: the gaps below are filled
        # and the table encoded with CONTEXT statistics, and a permutation afterwards left that
        # encoding fitted on rows that were no longer the context. Rare defaults make it common —
        # at a 1 % default rate a 512-row table's query often holds no default at all.
        if not self.regression and not same_classes_on_both_sides(y, plan.train_size):
            if meta.get("shift", "none") != "none":
                return None, None  # the row order IS the shift; permuting would undo it
            for _ in range(10):
                perm = rng.randperm(len(y))
                if same_classes_on_both_sides(y[perm], plan.train_size):
                    X, y = X[perm], y[perm]
                    meta["class_repair"] = True
                    break
            else:
                return None, None

        # Informative missingness, in a `missing_prob` share of tables (half the real PD tables
        # have no gaps at all): whether a value is missing depends on the target (a thin credit
        # file is itself a risk signal). Applied once the row order is final, because a gap is
        # filled as upstream's wrapper fills a real one: with the mean of the observed CONTEXT
        # values (`SimpleImputer` fitted in `fit`), and no indicator column.
        if self.credit_missing_cfg and rng.boolean(float(self.credit_missing_cfg.get("missing_prob", 1.0))):
            filled, mmeta = apply_informative_missingness(
                rng, X[:, :width], y, self.credit_missing_cfg, self.max_features, fit_rows=plan.train_size,
                fill="nan" if self.encoding == "raw" else "mean")
            X = X.clone()
            X[:, :width] = filled
            meta.update(mmeta)

        if self.encoding == "raw":
            # TabPFN preprocesses a real table itself (imputation, per-member transforms, target
            # z-scoring on the context); the table goes in as recorded.
            meta["encoding"] = "raw"
            if not self.regression and rng.boolean(0.5):
                y = 1.0 - y
                meta["labels_swapped"] = True
            if not self.regression:
                meta["num_classes"] = 2
            return X, y

        # The table as upstream's prediction path hands a real one to the model — the last step,
        # so nothing later can undo it.
        X, _ = prediction_view(X, width, plan.train_size)
        meta["prediction_view"] = True

        if self.regression:
            if self.lgd_scaling == "standard":
                # What `TabICLRegressor.fit` does to a real LGD target: a `StandardScaler` fitted
                # on the CONTEXT targets (`y_scaler_`), applied to all of them. Affine, so the
                # atoms at 0 and 1 survive as two atoms, at positions the model reads from the
                # context labels. NO outlier clipping: with a rare atom (1 % of rows at 1) the atom
                # sits more than 4 SD out, clipping would erase it, and prediction never clips y.
                y = standardise_on_context(y, plan.train_size)
            meta["target_scaling"] = self.lgd_scaling
        else:
            # Random label identity, as upstream's `permute_labels` gives every control table and
            # as the prediction ensemble's class rotation gives every real one: a model that only
            # ever saw "1 = default" in credit tables would learn an asymmetry.
            if rng.boolean(0.5):
                y = 1.0 - y
                meta["labels_swapped"] = True
            meta["num_classes"] = 2
        return X, y

    def _permute_and_pad(self, rng: PriorRNG, X: torch.Tensor) -> torch.Tensor:
        X = X[..., rng.randperm(X.shape[1])]
        if X.shape[1] < self.max_features:
            X = torch.nn.functional.pad(X, (0, self.max_features - X.shape[1]), value=0.0)
        return X[..., : self.max_features]

    # -- stand-alone draws (plots, reports) --------------------------------------
    def sample_shape(self) -> tuple[int, int]:
        """One `(n_rows, n_features)` draw from the configured ranges."""
        lo, hi = (int(v) for v in self.n_rows_range)
        n_rows = hi if lo >= hi else self.rng.randint(lo, hi)
        f_lo, f_hi = (int(v) for v in self.n_features_range)
        n_features = int(round(self.rng.uniform(f_lo, max(f_lo, min(f_hi, self.max_features)))))
        return n_rows, max(1, n_features)

    def sample(self, shape: tuple[int, int] | None = None, *, num_classes: int | None = None) -> SyntheticTask:
        """A stand-alone task through the training code path. `num_classes` pins the class
        count of a base classification table (plots of the control's binary tables)."""
        n_rows, n_features = shape if shape is not None else self.sample_shape()
        use_credit = self.rng.boolean(self.credit_fraction)
        if self.regression:
            n_cls = None
        elif use_credit:
            n_cls = 2
        else:
            n_cls = int(num_classes) if num_classes else self.rng.randint(2, self.max_classes + 1)
        lo, hi = self.train_frac_range
        train_size = max(2, min(n_rows - 2, int(n_rows * self.rng.uniform(lo, hi))))
        self._standalone += 1
        plan = SlotPlan(batch=self._standalone, slot=0, group=0, seq_len=n_rows,
                        train_size=train_size, num_features=n_features, num_classes=n_cls,
                        use_credit=use_credit)
        task = self.sample_slot(plan)
        task.meta["train_size"] = train_size
        return task
