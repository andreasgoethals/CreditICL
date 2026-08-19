"""Score OUR OWN pretrained checkpoints — the thing the whole project produces.

WHY THIS IS A SEPARATE MODULE AND WHY IT MATTERS

`baselines.py` scores four *external* models: ridge/logistic, CatBoost, TabPFN-3 and the
**released** TabICLv2. None of them can load a checkpoint we trained. Without this file
the project trains 48 models and has no way to measure any of them — there is no result,
only weights.

HOW INFERENCE WORKS HERE (and why it is not `model.fit`)

TabICL-family models do **in-context learning**: there is no fitting step. You hand the
model a single tensor containing the labelled context rows *and* the rows you want
predicted, plus the labels for the context only, and it returns predictions for every
row. `fit()` therefore just stores the context; `predict()` does the real work.

    forward(x, y) where x is (1, n_context + n_query, n_features)
                        y is (1, n_context)

That is the same contract the training loop uses, which is deliberate: if evaluation
built the episode differently from training, the model would be scored on a task it was
never trained for and every number would be quietly wrong.

CONTEXT SIZE IS A REAL LEVER, NOT AN IMPLEMENTATION DETAIL. Tanna et al. 2026 found that
on credit data the *context construction* explains more variance in AUC than the choice
of model family, with balanced sampling worth 3-4 AUC points over uniform. So the context
is capped and sampled explicitly here, and every choice is recorded in the result row —
never left implicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.eval.baselines import TabICLBaseline
from src.utils.logging_setup import get_logger

#: Rows of context to give the model. TabICLv2 trains on 512-1024-row tables, so a
#: context far outside that range is out-of-distribution for it regardless of how much
#: memory we have.
DEFAULT_CONTEXT_ROWS = 1024


def load_our_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild one of our checkpoints into the architecture that produced it.

    THIS GOES THROUGH `src.models.architecture.build_model` — the same entry point the trainer
    uses — and reads `architecture:` from the config stored *inside* the checkpoint. It used to
    hard-code `NanoTabICLv2`, which was correct only while training also used Nano. After
    training moved to upstream `TabICL`, every single parameter name mismatched and every
    `crediticl` cell died with "does not match the architecture in its own config"
    (`missing=['row_cls_tokens', 'x_embed.weight', …]` — Nano's names —
    `unexpected=['col_embedder.in_linear.weight', …]` — upstream's).

    The architecture comes from the checkpoint and never from the current YAML: editing
    `config/Exp1_LGD.yaml` after training would otherwise produce a shape mismatch, or worse a
    silent mis-load.
    """
    from src.models.architecture import DEFAULT, build_model

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=device, weights_only=False)

    # `config` is upstream's slot for the TabICL kwargs; OUR resolved YAML lives in
    # `crediticl_config`. The fallback reads checkpoints written before 18-08-2026,
    # whose `config` was the YAML.
    cfg = payload.get("crediticl_config") or payload.get("config") or {}
    mcfg = dict(cfg.get("model") or {})
    task = cfg.get("task", "lgd")
    regression = task == "lgd"
    num_quantiles = int((cfg.get("train") or {}).get("num_quantiles", 999))
    n_classes = int((cfg.get("prior") or {}).get("n_classes", 2))
    architecture = cfg.get("architecture", DEFAULT)

    # Mirrors Trainer._build_model exactly. Any divergence here is a mis-load waiting to
    # happen, so the two must be read together.
    if regression:
        mcfg.setdefault("num_quantiles", num_quantiles)
    else:
        mcfg.setdefault("max_classes", n_classes)
    model = build_model(task, architecture=architecture, **mcfg)
    state = payload.get("model") or payload.get("state_dict")
    if state is None:
        raise KeyError(f"{path} has no model weights (keys: {sorted(payload)})")
    # DDP saves with a `module.` prefix. We strip it in the trainer, but a checkpoint
    # from an older run or an external source may still carry it.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {path.name} does not match architecture={architecture!r}, which is "
            f"what its own config records.\n"
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}\n"
            f"Loading it non-strictly would score a partly-random model as if it were "
            f"trained, so this is fatal rather than a warning."
        )
    model.to(device).eval()

    meta = {
        "checkpoint": str(path),
        "step": payload.get("step"),
        "task": task,
        "architecture": architecture,
        # How many of the head's `max_classes` logits are real for this task. PD is binary,
        # so 2 of the architecture's 10 — see `predict_proba`.
        "n_classes": n_classes,
        "regression": regression,
        "num_quantiles": num_quantiles,
        "run_name": cfg.get("_run_name"),
        "credit_fraction": (cfg.get("prior") or {}).get("credit_fraction"),
    }
    return model, meta


def standardise_from_context(
    x_context: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Standard-scale features using the CONTEXT rows' mean and std. Upstream's step 1.

    THIS IS WHY OUR LGD MODEL RETURNED ALL-NaN ON HALF THE REAL DATASETS, and it is not subtle
    once measured. `col_embedder.in_linear` — the very FIRST layer — went non-finite with an
    output `absmax` of exactly **6.550e+04**, which is `float16`'s largest finite value. The
    inputs feeding it were raw:

        axa                absmax 1.98    -> finite
        heloc              absmax 820     -> finite
        loss2              absmax 2.4e6   -> NaN
        base_modelisation  absmax 8.1e7   -> NaN
        base_model         absmax 9.6e8   -> NaN

    Upstream never does this. `PreprocessingPipeline.fit` starts with
    `CustomStandardScaler().fit_transform(X)` before any normalisation, and every prediction
    goes through `preprocessors_[...].transform(X)`. We fed the network unscaled currency
    amounts and loan balances and it overflowed, exactly as it should have.

    FITTED ON THE CONTEXT ONLY. The query rows are what we are predicting, so their mean and
    variance are not ours to look at — using them would be leakage of precisely the kind this
    project has to be able to rule out. That also matches the median imputation next to it,
    which is already context-only.

    Constant columns get std 1.0 rather than 0: dividing by a zero std is the other way to
    manufacture a NaN, and a column with no variation carries no information anyway.
    """
    mean = np.nanmean(x_context, axis=0, keepdims=True)
    std = np.nanstd(x_context, axis=0, keepdims=True)
    mean = np.nan_to_num(mean)
    std = np.nan_to_num(std)
    std[std < 1e-12] = 1.0
    return tuple(((a - mean) / std).astype(np.float32) for a in (x_context, *others))


class CreditICLBaseline(TabICLBaseline):
    """One of OUR pretrained checkpoints, scored through UPSTREAM'S OWN WRAPPER.

    THE POINT OF SUBCLASSING `TabICLBaseline`: the released TabICLv2 and our checkpoints then
    travel *the same code path*, and the only difference between them is the weights — which is
    the only difference the experiment is about. Everything else comes for free and stays in
    step automatically: upstream's `PreprocessingPipeline` (standard scaling, then power or
    quantile normalisation), its 8-member feature-shuffled ensemble, its context construction,
    its label encoding, its quantile decoding.

    THIS REPLACED A HAND-ROLLED INFERENCE PATH, and the difference was not cosmetic. Measured
    18-08-2026 on the same seven LGD datasets in one run:

        released TabICLv2, through TabICLRegressor   R2  +0.224 .. +0.770
        ours, through our own single forward pass    R2  -1.437 .. -0.246

    Some of that is 600 steps against 500,000. But our path also had no preprocessing (which
    overflowed `col_embedder.in_linear` into all-NaN on half the datasets), no ensemble, and its
    own context rules — so the number was never a weights-only comparison, and there was no way
    to say how much of the gap was the prior and how much was our plumbing. Only one pipeline
    can be the measured one.

    All this needs is `model_path`, which upstream already supports:

        assert "config" in checkpoint          # kwargs to TabICL
        assert "state_dict" in checkpoint
        self.model_ = TabICL(**checkpoint["config"])

    `src/train/checkpoint.py` writes exactly that schema, so our checkpoints load into their
    wrapper unchanged.
    """

    name = "crediticl"

    def __init__(
        self,
        task: str,
        seed: int = 0,
        checkpoint: str | Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(task=task, seed=seed, **kwargs)
        if checkpoint is None:
            raise ValueError(
                "CreditICLBaseline needs checkpoint=<path to one of our .ckpt files>. "
                "Without it there is nothing to score."
            )
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"no checkpoint at {self.checkpoint}")

    def _wrapper_kwargs(self) -> dict[str, Any]:
        """Point upstream's wrapper at OUR weights instead of the released ones.

        `allow_auto_download=False` on purpose: if the checkpoint were unreadable, the default
        would quietly fetch the RELEASED model from Hugging Face and score that instead —
        reporting the baseline's numbers under our own name, which is the worst failure this
        file could have. Better a hard error.
        """
        return {
            "model_path": str(self.checkpoint),
            "allow_auto_download": False,
        }

    def _fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
        super()._fit(X, y, cat_indices)
        # Recorded in every result row, so a number can always be traced to the arm that
        # produced it without opening the checkpoint.
        meta = checkpoint_metadata(self.checkpoint)
        self.report.extra.update(
            {
                "checkpoint": self.checkpoint.name,
                "checkpoint_step": meta.get("step"),
                "credit_fraction": meta.get("credit_fraction"),
                "run_name": meta.get("run_name"),
                "inference": "upstream TabICL wrapper (same as tabiclv2)",
            }
        )


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Which arm produced this checkpoint. Reads the header, never builds the model.

    `load_our_checkpoint` builds a network to verify the weights fit, which is right for the NaN
    diagnostic and pure waste when all that is wanted is a run name for a results row.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = payload.get("crediticl_config") or payload.get("config") or {}
    return {
        "step": payload.get("curr_step", payload.get("step")),
        "task": cfg.get("task"),
        "run_name": cfg.get("_run_name"),
        "credit_fraction": (cfg.get("prior") or {}).get("credit_fraction"),
    }


def register() -> None:
    """Add `crediticl` to the baseline registry.

    Called explicitly rather than at import time so `baselines.py` has no dependency on
    the training code, and a broken checkpoint cannot stop the external baselines from
    running.
    """
    from src.eval import baselines

    baselines.BASELINES["crediticl"] = CreditICLBaseline
    get_logger().info("[eval] registered the 'crediticl' baseline (our own checkpoints)")


def register_or_warn(log: Any = None) -> bool:
    """`register()`, but never fatal. Call this from the evaluation ENTRY POINTS.

    Registration is explicit rather than automatic (see `register`), which means it is also
    easy to forget — and it was: no production caller existed, so `--models crediticl` failed
    with an unknown-baseline error inside a SLURM job whose exit status still said success.
    `tests/test_slurm_scripts.py` now checks that every model name the job scripts ask for is
    actually resolvable.

    Returns True if ours is now available. A failure is logged and swallowed, because the
    external baselines are still worth having when our own checkpoints cannot be loaded.
    """
    warn = (log.warning if log is not None else get_logger().warning)
    try:
        register()
        return True
    except Exception as exc:  # noqa: BLE001 — never let ours take the others down
        warn("could not register the 'crediticl' baseline; OUR checkpoints will be SKIPPED: %s", exc)
        return False


def resolve_our_checkpoint(
    explicit: str | Path | None,
    task: str,
    log: Any = None,
    root: str | Path | None = None,
) -> Path | None:
    """Which of our checkpoints should `--models crediticl` score?

    Registering the baseline is not enough to evaluate anything: it also needs a checkpoint,
    and on 14-08-2026 no caller passed one, so every `crediticl` cell in both evaluations
    failed with "needs checkpoint=<path>" while `tabiclv2` scored normally. The run looked
    half-successful (25/50 cells OK) and contained nothing about OUR model.

    An explicit path wins. Otherwise pick the single checkpoint belonging to `task`, and
    REFUSE to choose when there are several — scoring an arbitrary arm of a 96-run sweep
    would produce a number that looks like a result.
    """
    warn = (log.warning if log is not None else get_logger().warning)
    info = (log.info if log is not None else get_logger().info)

    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            warn("checkpoint %s does not exist — 'crediticl' cannot be scored", path)
            return None
        info("[eval] crediticl checkpoint: %s", path)
        return path

    # Run directories are named `exp1_<task>__…`, so the task is in the path.
    candidates = [p for p in find_our_checkpoints(root) if f"_{task}__" in str(p)]
    if not candidates:
        warn(
            "no checkpoint for task=%r found under the checkpoints tree, so 'crediticl' "
            "will be SKIPPED. Pass --checkpoint <path to a step-*.ckpt>.", task,
        )
        return None
    if len(candidates) > 1:
        warn(
            "found %d checkpoints for task=%r and will NOT guess which one is meant. "
            "Pass --checkpoint explicitly. Candidates:\n  %s",
            len(candidates), task, "\n  ".join(str(p) for p in candidates),
        )
        return None
    info("[eval] crediticl checkpoint (auto-discovered): %s", candidates[0])
    return candidates[0]


def find_our_checkpoints(root: str | Path | None = None) -> list[Path]:
    """Every `step-*.ckpt` under the checkpoints tree, newest step per run directory.

    Returns one checkpoint per run — the highest step — because scoring every
    intermediate checkpoint of 48 runs is rarely what anyone means.
    """
    from src.utils.paths import checkpoints_dir

    base = Path(root) if root is not None else checkpoints_dir()
    if not base.is_dir():
        return []
    best: dict[Path, tuple[int, Path]] = {}
    for p in base.rglob("step-*.ckpt"):
        try:
            step = int(p.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        cur = best.get(p.parent)
        if cur is None or step > cur[0]:
            best[p.parent] = (step, p)
    return [p for _, p in sorted(best.values(), key=lambda t: str(t[1]))]
