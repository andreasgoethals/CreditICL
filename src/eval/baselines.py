"""The models we compare against, behind one interface.

Every baseline is wrapped so the runner can treat them identically: `fit(X, y)`
then `predict(X)`, plus `predict_quantiles(X)` for LGD where the model supports it.

NONE OF THESE IS MODIFIED. In particular **TabPFN's prior is never touched** —
this project alters TabICL's prior, and TabPFN-3 is here purely as a yardstick.
Any baseline that cannot be imported reports itself unavailable and is skipped
with a clear log line, so a missing package costs one column of results rather
than the whole run.

The five:

* ``linear`` / ``logistic``  — scikit-learn, with imputation and scaling. The floor.
  A result that cannot beat these is not a result.
* ``catboost``               — the GBDT every credit-risk paper reports, and the
  model that beats TFMs on large and high-cardinality data (Purucker 2026).
* ``tabpfn3``                — TabPFN-3 via `model_path="v3"`. Open weights.
* ``tabiclv2``               — the released TabICLv2 checkpoints. This is the
  model whose *prior* we modify, so its stock performance is the number our
  retrained versions have to be read against.

EVERY MODEL GETS THE WHOLE TRAINING POOL (evaluation protocol 3, `src/eval/protocol.py`). Until
28-09-2026 the foundation models were given at most 10,000 training rows (and the benchmark
capped their context at 1,024); TabICLv2 and TabPFN-3 both handle hundreds of thousands, so
neither cap is needed and neither is applied. Every model runs at its library's DEFAULT
settings — CatBoost's 1,000 iterations with its automatic learning rate, TabPFN-3's and
TabICLv2's own ensembles — so no column of the table is tuned and none is handicapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

# Row cap for the in-context models: `None` = the whole training pool, which is the protocol.
# Kept as a hook for debug runs only; a cap is recorded in every row it touches.
TFM_MAX_TRAIN_ROWS: int | None = None
# Column cap for the foundation models (top variance, training rows only). Since 29-09-2026 the
# benchmark runners select `protocol.MAX_FEATURES` columns for EVERY model before any fit, so on
# that path this is a no-op; it remains the guard for a caller that hands a wider table directly.
TFM_MAX_FEATURES = 500

#: The released TabICLv2 files: the ones Exp2 fine-tunes from and the reference column scores.
TABICL_RELEASED = {"pd": "tabicl-classifier-v2-20260212.ckpt", "lgd": "tabicl-regressor-v2-20260212.ckpt"}


def find_local_tabpfn_checkpoint(which: str) -> Path | None:
    """Locate a TabPFN checkpoint we already have on disk.

    This is the whole answer to TabPFN's licence gate. `model_path` accepts a
    **file path**, so pointing it at a checkpoint we already hold bypasses the
    browser auth flow, the API token, and the download entirely — which also means
    it works on a compute node with no internet.

    `which` is "classifier" or "regressor". Searched, in order:
      1. $CREDITICL_TABPFN_DIR
      2. project storage: <staging>/CreditICL/checkpoints/
      3. the repo's own checkpoints/
    """
    import os

    from src.utils.paths import REPO_ROOT as _repo
    from src.utils.paths import pretrained_dir

    roots = []
    if os.environ.get("CREDITICL_TABPFN_DIR"):
        roots.append(Path(os.environ["CREDITICL_TABPFN_DIR"]))
    roots.append(pretrained_dir())
    roots.append(_repo / "checkpoints")

    for root in roots:
        if not root.is_dir():
            continue
        matches = sorted(root.glob(f"tabpfn*{which}*.ckpt"))
        if matches:
            return matches[0].resolve()
    return None


_IDENTITY_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def tabpfn_checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """`{run_name, checkpoint_step}` of a TabPFN checkpoint our trainer wrote, `{}` for a released
    one. Cached per file version: the benchmark fits every fold from the same file."""
    import torch

    p = Path(path)
    key = (str(p.resolve()), p.stat().st_mtime_ns)
    if key not in _IDENTITY_CACHE:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        ident = {}
        if payload.get("crediticl_run_name") is not None:
            ident = {"run_name": payload["crediticl_run_name"],
                     "checkpoint_step": int(payload.get("crediticl_step", -1))}
        _IDENTITY_CACHE[key] = ident
    return dict(_IDENTITY_CACHE[key])


@dataclass
class FitReport:
    """What actually happened during fit. Ends up in the results row."""

    model: str
    available: bool = True
    error: str | None = None
    n_train_used: int = 0
    n_train_available: int = 0
    subsampled: bool = False
    n_features: int = 0
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class Baseline:
    """Common interface. Subclasses implement `_fit` and `_predict`."""

    name = "base"
    needs_dense = True  # cannot handle NaN; the wrapper imputes

    def __init__(self, task: str, seed: int = 0, **kwargs: Any):
        self.task = task
        self.seed = seed
        self.kwargs = kwargs
        self.report = FitReport(model=self.name)
        self._fitted = False

    # -- availability --------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        return True, None

    # -- subclass hooks ------------------------------------------------------
    def _fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
        raise NotImplementedError

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(quantiles (n, L), levels (L,)) if the model gives a distribution."""
        return None

    # -- public --------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int] | None = None) -> FitReport:
        import time

        log = get_logger()
        cat_indices = cat_indices or []
        self.report.n_train_available = int(X.shape[0])
        self.report.n_features = int(X.shape[1])

        Xf, yf = self._maybe_subsample(X, y)
        self.report.n_train_used = int(Xf.shape[0])

        t0 = time.time()
        self._fit(Xf, yf, cat_indices)
        self.report.fit_seconds = round(time.time() - t0, 2)
        self._fitted = True
        log.info(
            "[eval]     %-10s fit on %d/%d rows x %d features%s in %.1fs",
            self.name,
            self.report.n_train_used,
            self.report.n_train_available,
            self.report.n_features,
            " (SUBSAMPLED)" if self.report.subsampled else "",
            self.report.fit_seconds,
        )
        return self.report

    def predict(self, X: np.ndarray) -> np.ndarray:
        import time

        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict before fit")
        t0 = time.time()
        out = self._predict(X)
        self.report.predict_seconds = round(time.time() - t0, 2)
        return out

    # -- helpers -------------------------------------------------------------
    def _maybe_subsample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X, y

    def _impute(self, X: np.ndarray, stats: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Median-impute NaNs. Statistics come from TRAIN only, never test."""
        X = np.asarray(X, dtype=np.float64)
        if stats is None:
            stats = np.nanmedian(X, axis=0)
            stats = np.where(np.isfinite(stats), stats, 0.0)
        out = np.where(np.isnan(X), stats, X)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), stats


class _TFMBaseline(Baseline):
    """Shared row and feature capping for the in-context models.

    Both caps are needed on our data and both are recorded, never silent:

    * ROWS — TabPFN and TabICL are built for roughly 10k context rows. Home Credit
      has 307k, hackerearth 532k, vehicle_loan 233k.
    * FEATURES — `0014.algorithmwatch` has **2,986** features and
      `0011.loan_default` has 759, well past what these models accept.

    Feature selection is by variance on the training split only. That is a crude
    criterion, and deliberately so: anything target-aware (mutual information,
    model-based importance) would leak the target into feature choice and inflate
    the score. A weak-but-honest filter beats a strong-but-leaky one.
    """

    def _select_features(self, X: np.ndarray) -> np.ndarray | None:
        """Column indices to keep, or None to keep all."""
        n_features = X.shape[1]
        if n_features <= TFM_MAX_FEATURES:
            return None
        var = np.nanvar(X, axis=0)
        var = np.where(np.isfinite(var), var, -np.inf)
        keep = np.sort(np.argsort(-var)[:TFM_MAX_FEATURES])
        self.report.extra["features_capped_from"] = n_features
        self.report.extra["feature_selection"] = "top-variance (train only)"
        return keep

    def fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int] | None = None) -> FitReport:
        self._keep = self._select_features(X)
        if self._keep is not None:
            keep_set = set(self._keep.tolist())
            remap = {old: new for new, old in enumerate(self._keep.tolist())}
            cat_indices = [remap[c] for c in (cat_indices or []) if c in keep_set]
            X = X[:, self._keep]
        return super().fit(X, y, cat_indices)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_keep", None) is not None:
            X = X[:, self._keep]
        return super().predict(X)

    def _maybe_subsample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        if TFM_MAX_TRAIN_ROWS is None or n <= TFM_MAX_TRAIN_ROWS:
            return X, y
        rng = np.random.default_rng(self.seed)
        # Stratified for classification, so a 7%-positive dataset does not lose
        # its positives to an unlucky uniform draw.
        if self.task == "pd" and 0 < y.sum() < n:
            pos = np.flatnonzero(y >= 0.5)
            neg = np.flatnonzero(y < 0.5)
            n_pos = max(1, int(round(TFM_MAX_TRAIN_ROWS * len(pos) / n)))
            n_neg = TFM_MAX_TRAIN_ROWS - n_pos
            idx = np.concatenate(
                [
                    rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
                    rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
                ]
            )
            rng.shuffle(idx)
        else:
            idx = rng.choice(n, size=TFM_MAX_TRAIN_ROWS, replace=False)
        self.report.subsampled = True
        self.report.extra["subsample_strategy"] = "stratified" if self.task == "pd" else "uniform"
        return X[idx], y[idx]


# ---------------------------------------------------------------------------
# scikit-learn floors
# ---------------------------------------------------------------------------


class LinearBaseline(Baseline):
    """Ridge for LGD, logistic regression for PD. The floor to beat."""

    name = "linear"

    def _fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        Xi, self._impute_stats = self._impute(X)
        self._scaler = StandardScaler().fit(Xi)
        Xs = self._scaler.transform(Xi)

        if self.task == "pd":
            self.name = "logistic"
            self.report.model = "logistic"
            self._model = LogisticRegression(max_iter=2000, random_state=self.seed).fit(Xs, (y >= 0.5).astype(int))
        else:
            self._model = Ridge(alpha=1.0, random_state=self.seed).fit(Xs, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        Xi, _ = self._impute(X, self._impute_stats)
        Xs = self._scaler.transform(Xi)
        if self.task == "pd":
            return self._model.predict_proba(Xs)[:, 1]
        # Clip to [0,1]: LGD is a fraction, and an unclipped linear model
        # routinely predicts outside it. Free, and reported via pred_out_of_unit
        # before clipping so the raw behaviour stays visible.
        return np.clip(self._model.predict(Xs), 0.0, 1.0)


class CatBoostBaseline(Baseline):
    """The GBDT every credit paper reports, and the one TFMs lose to at scale."""

    name = "catboost"

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        try:
            import catboost  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        return True, None

    def _fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
        from catboost import CatBoostClassifier, CatBoostRegressor

        # CatBoost handles NaN natively, so no imputation — one of the reasons it
        # is strong on credit data, where missingness is informative.
        # LIBRARY DEFAULTS (1,000 iterations, depth 6, learning rate chosen from the data size),
        # as every other model here runs at its defaults. Until 28-09-2026: 500 iterations at a
        # fixed 0.05, a choice nobody could have defended in a paper.
        common = dict(random_seed=self.seed, verbose=False, allow_writing_files=False, thread_count=-1)
        for key in ("iterations", "learning_rate", "depth"):
            if key in self.kwargs:
                common[key] = self.kwargs[key]
        if self.task == "pd":
            self._model = CatBoostClassifier(**common)
            self._model.fit(X, (y >= 0.5).astype(int))
        else:
            self._model = CatBoostRegressor(**common)
            self._model.fit(X, y)
        self.report.extra["iterations"] = int(self._model.tree_count_)
        self.report.extra["learning_rate"] = float(self._model.learning_rate_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "pd":
            return self._model.predict_proba(X)[:, 1]
        return np.clip(self._model.predict(X), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Tabular foundation models — inference only, priors untouched
# ---------------------------------------------------------------------------


class TabPFNBaseline(_TFMBaseline):
    """TabPFN-3, inference only. Its prior is NEVER modified — reference point.

    WEIGHTS ARE LOADED FROM A LOCAL FILE, and that is deliberate. Asking `tabpfn`
    to fetch its own weights triggers a licence flow: it opens a browser at
    ux.priorlabs.ai and waits for an API token. With no terminal to prompt on — a
    SLURM job, a piped shell — that surfaces as a baffling
    `OSError: [WinError 10038] ... not a socket`.

    Since `model_path` accepts a path, pointing it at a checkpoint already on disk
    sidesteps all of it: no token, no download, and it works on a compute node with
    no internet. Drop the two files in `checkpoints/` (or on project storage) and
    this baseline just runs.

    Falls back to `model_path="v3"` — i.e. let tabpfn resolve it — only when no
    local file is found, and reports clearly if that would need a token.
    """

    name = "tabpfn3"

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        import os

        try:
            import tabpfn  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

        have_clf = find_local_tabpfn_checkpoint("classifier") is not None
        have_reg = find_local_tabpfn_checkpoint("regressor") is not None
        if have_clf or have_reg:
            return True, None

        if not os.environ.get("TABPFN_TOKEN"):
            # Say this now, plainly, rather than letting the interactive auth
            # prompt blow up mid-run with a socket error that names nothing.
            return False, (
                "no local TabPFN checkpoint found and TABPFN_TOKEN is not set. "
                "Easiest fix: put tabpfn-v3-classifier-*.ckpt and "
                "tabpfn-v3-regressor-*.ckpt in checkpoints/ (or on project storage, "
                "or point $CREDITICL_TABPFN_DIR at them). Otherwise get a token from "
                "https://ux.priorlabs.ai/account and let tabpfn download them — but "
                "that needs internet, which compute nodes do not have."
            )
        return True, None

    def _fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
        import torch
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        which = "classifier" if self.task == "pd" else "regressor"
        local = find_local_tabpfn_checkpoint(which)
        # A local file if we have one (no token, works offline); otherwise let
        # tabpfn resolve "v3" itself, which needs a token and internet. An explicit `model_path`
        # is one of OUR fine-tuned checkpoints (Experiment 2), which names its arm and step.
        given = self.kwargs.get("model_path")
        version = str(given or (str(local) if local else "v3"))
        self.report.extra["weights_source"] = ("given checkpoint" if given else
                                               "local file" if local else "tabpfn download")
        if given:
            self.report.extra.update(tabpfn_checkpoint_identity(given))
        common = dict(
            model_path=version,
            device=device,
            random_state=self.seed,
            # Our largest tables exceed the documented envelope; without this the wrapper
            # refuses them. Nothing is subsampled: the protocol gives every model the whole pool.
            ignore_pretraining_limits=True,
            categorical_features_indices=cat_indices or None,
        )
        # The library's own ensemble size (`"auto"`) unless one is asked for. Until 28-09-2026: 4.
        if "n_estimators" in self.kwargs:
            common["n_estimators"] = int(self.kwargs["n_estimators"])
        self.report.extra["model_path"] = version
        self.report.extra["device"] = device

        if self.task == "pd":
            self._model = TabPFNClassifier(**common)
            self._model.fit(X, (y >= 0.5).astype(int))
        else:
            self._model = TabPFNRegressor(**common)
            self._model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "pd":
            return self._model.predict_proba(X)[:, 1]
        return np.clip(self._model.predict(X), 0.0, 1.0)

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """TabPFN-3's predictive distribution at `protocol.QUANTILE_LEVELS`, clipped to [0, 1]."""
        if self.task == "pd":
            return None
        from src.eval.protocol import QUANTILE_LEVELS

        if getattr(self, "_keep", None) is not None:
            X = X[:, self._keep]
        q = self._model.predict(X, output_type="quantiles", quantiles=[float(a) for a in QUANTILE_LEVELS])
        return np.clip(np.stack([np.asarray(c) for c in q], axis=1), 0.0, 1.0), QUANTILE_LEVELS


class TabICLBaseline(_TFMBaseline):
    """The released TabICLv2 checkpoints.

    This is the model whose *prior* the project modifies, so its stock score is
    the number our retrained versions must be read against. Prior untouched here.
    """

    name = "tabiclv2"

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        try:
            import tabicl  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        return True, None

    #: Cap on context rows handed to the wrapper. `None` = give it everything, which is the
    #: protocol since 28-09-2026 (`src/eval/protocol.py`): the whole training pool is the context,
    #: for our checkpoints and the released model alike.
    #:
    #: Protocol 2 capped it at 1,024 rows, because we train on tables of at most 1,024 rows
    #: (upstream stage 1) and `heloc`'s training split is 46x that. That made the comparison
    #: about a context size nobody deploys; the length mismatch is now reported as a property of
    #: our models (trained on stage 1 only) rather than removed from the measurement. The hook
    #: stays for debug runs and is recorded in every row.
    max_context_rows: int | None = None

    def _cap_context(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Subsample the context to `max_context_rows`, seeded, stratified for PD.

        Stratified because a 3 %-positive dataset cut uniformly to 1,024 rows can arrive with
        almost no defaults, and then the model has nothing to learn the positive class from —
        Tanna et al. 2026 measure balanced context construction at 3-4 AUC points on credit
        data, which is larger than most model-family differences.
        """
        cap = self.max_context_rows
        # RECORD IT EVEN WHEN IT DOES NOT BIND. Writing these columns only on the datasets
        # that were actually trimmed makes an absent value ambiguous six months later: "no cap
        # was set" and "the cap was set and the table was smaller" are different facts and
        # only one of them is a caveat on the number.
        self.report.extra["context_cap"] = cap
        self.report.extra["n_context_full"] = int(len(X))
        if cap is None or len(X) <= cap:
            return X, y
        rng = np.random.default_rng(self.seed)
        if self.task == "pd":
            pos = np.flatnonzero(y >= 0.5)
            neg = np.flatnonzero(y < 0.5)
            per = cap // 2
            take_pos = rng.choice(pos, size=min(per, len(pos)), replace=False)
            take_neg = rng.choice(neg, size=min(cap - len(take_pos), len(neg)), replace=False)
            keep = np.concatenate([take_pos, take_neg])
            rng.shuffle(keep)
        else:
            keep = rng.choice(len(X), size=cap, replace=False)
        return X[keep], y[keep]

    def _wrapper_kwargs(self) -> dict[str, Any]:
        """Extra arguments for upstream's wrapper: the RELEASED weights, from our own copy.

        The local file (`TABICL_RELEASED`, found by `paths.find_pretrained`) is the file Exp2
        fine-tunes from — tensor-for-tensor the wrapper's own download (Experiment 0 checks it) —
        and a compute node has no internet to download anything, so the wrapper is pointed at it
        and forbidden to fetch. Without a local copy the wrapper resolves its own, as before.

        The one hook `CreditICLBaseline` overrides, returning `model_path=<our checkpoint>` and
        nothing else, so our model and the released model go through the same preprocessing, the
        same ensemble and the same decoding — the only difference between the two columns of a
        results table is the weights, which is the only difference the experiment is about.
        """
        from src.utils.paths import find_pretrained

        try:
            path = find_pretrained(f"checkpoints/{TABICL_RELEASED[self.task]}")
        except FileNotFoundError:
            self.report.extra["weights_source"] = "tabicl download"
            return {}
        self.report.extra["weights_source"] = str(path)
        return {"model_path": str(path), "allow_auto_download": False}

    def _fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.report.extra["device"] = device
        extra = self._wrapper_kwargs()
        X, y = self._cap_context(X, y)

        if self.task == "pd":
            from tabicl import TabICLClassifier

            self._model = TabICLClassifier(device=device, random_state=self.seed, **extra)
            self._model.fit(X, (y >= 0.5).astype(int))
        else:
            from tabicl import TabICLRegressor

            self._model = TabICLRegressor(device=device, random_state=self.seed, **extra)
            self._model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "pd":
            return self._model.predict_proba(X)[:, 1]
        return np.clip(self._model.predict(X), 0.0, 1.0)

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """TabICL's predictive distribution (`TabICLRegressor.predict(output_type="quantiles")`)
        at `protocol.QUANTILE_LEVELS`, clipped to [0, 1]. Released weights and ours alike."""
        if self.task == "pd":
            return None
        from src.eval.protocol import QUANTILE_LEVELS

        if getattr(self, "_keep", None) is not None:
            X = X[:, self._keep]
        q = self._model.predict(X, output_type="quantiles", alphas=[float(a) for a in QUANTILE_LEVELS])
        return np.clip(np.asarray(q, dtype=float), 0.0, 1.0), QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES: dict[str, type[Baseline]] = {
    "linear": LinearBaseline,
    "catboost": CatBoostBaseline,
    "tabpfn3": TabPFNBaseline,
    "tabiclv2": TabICLBaseline,
}

DEFAULT_BASELINES = ("linear", "catboost", "tabpfn3", "tabiclv2")


def build(name: str, task: str, seed: int = 0, **kwargs: Any) -> Baseline:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    return BASELINES[name](task=task, seed=seed, **kwargs)


def availability_report() -> dict[str, tuple[bool, str | None]]:
    """Which baselines can actually run. Logged at the start of every eval run."""
    return {name: cls.is_available() for name, cls in BASELINES.items()}
