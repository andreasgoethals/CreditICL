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

from src.eval.baselines import Baseline
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

    cfg = payload.get("config") or {}
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


class CreditICLBaseline(Baseline):
    """One of our pretrained checkpoints, scored in-context.

    Construct with `checkpoint=<path>`. The task comes from the checkpoint's own config
    and must match the dataset being scored — a regression checkpoint cannot score a
    classification dataset, and mixing them would produce numbers that look plausible.
    """

    name = "crediticl"

    def __init__(
        self,
        task: str,
        seed: int = 0,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        context_rows: int = DEFAULT_CONTEXT_ROWS,
        balanced_context: bool = True,
        **kwargs: Any,
    ):
        super().__init__(task=task, seed=seed, **kwargs)
        if checkpoint is None:
            raise ValueError(
                "CreditICLBaseline needs checkpoint=<path to one of our .ckpt files>. "
                "Without it there is nothing to score."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_rows = int(context_rows)
        self.balanced_context = bool(balanced_context)
        self.model, self.meta = load_our_checkpoint(checkpoint, self.device)

        if self.meta["regression"] != (task == "lgd"):
            raise ValueError(
                f"checkpoint is for task={self.meta['task']!r} but it is being asked to "
                f"score task={task!r}. A regression head cannot produce class "
                f"probabilities, and the reverse gives meaningless quantiles."
            )
        self._Xc: np.ndarray | None = None
        self._yc: np.ndarray | None = None

    # -- context selection ---------------------------------------------------
    def _select_context(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pick the context rows. Records nothing here — the caller logs the report.

        For classification, `balanced_context` samples the two classes evenly. That is
        the single highest-leverage choice in this file: on imbalanced credit data a
        uniform context can contain almost no defaults, and the model has nothing to
        learn the positive class from.
        """
        rng = np.random.default_rng(self.seed)
        n = len(X)
        if n <= self.context_rows:
            return X, y

        if self.task == "pd" and self.balanced_context:
            pos = np.flatnonzero(y > 0.5)
            neg = np.flatnonzero(y <= 0.5)
            per = self.context_rows // 2
            take_pos = rng.choice(pos, size=min(per, len(pos)), replace=False) if len(pos) else pos
            remaining = self.context_rows - len(take_pos)
            take_neg = rng.choice(neg, size=min(remaining, len(neg)), replace=False)
            idx = np.concatenate([take_pos, take_neg])
            rng.shuffle(idx)
        else:
            idx = rng.choice(n, size=self.context_rows, replace=False)
        return X[idx], y[idx]

    def _fit(self, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
        """No fitting — just store the context. ICL has no training step.

        Returns None and writes to `self.report.extra`, matching the base class: the
        wrapper in `Baseline.fit` owns the FitReport and times the call itself.
        """
        self._Xc, self._yc = self._select_context(
            np.asarray(X, np.float32), np.asarray(y, np.float32)
        )
        # Everything about how the context was built lands in the results row. Tanna et
        # al. 2026 found context construction explains more AUC variance than model
        # choice, so leaving it implicit would make the numbers uninterpretable.
        self.report.extra.update(
            {
                "context_rows": int(len(self._Xc)),
                "context_from_rows": int(len(X)),
                "balanced_context": bool(self.balanced_context and self.task == "pd"),
                **{f"ckpt_{k}": v for k, v in self.meta.items()},
            }
        )
        if self.task == "pd":
            self.report.extra["context_base_rate"] = round(float((self._yc > 0.5).mean()), 4)

    @torch.no_grad()
    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._Xc is None or self._yc is None:
            raise RuntimeError("fit() must run before predict() — the context is set there")

        Xq = np.asarray(X, np.float32)
        # Both halves must have the same width. A query table with more columns than the
        # context would silently misalign features, so this is an error, not a trim.
        if Xq.shape[1] != self._Xc.shape[1]:
            raise ValueError(
                f"query has {Xq.shape[1]} features but the context has {self._Xc.shape[1]}"
            )

        # NaNs: the model standardises internally and a NaN would propagate to every
        # output. Impute from the CONTEXT only — using query statistics would leak.
        ctx_median = np.nan_to_num(np.nanmedian(self._Xc, axis=0))
        Xc = np.where(np.isfinite(self._Xc), self._Xc, ctx_median)
        Xq = np.where(np.isfinite(Xq), Xq, ctx_median)

        preds: list[np.ndarray] = []
        # Chunk the queries so a large test set cannot exhaust memory: every chunk is a
        # fresh episode with the SAME context, which is exactly what ICL expects.
        chunk = max(1, min(2048, self.context_rows))
        for start in range(0, len(Xq), chunk):
            block = Xq[start : start + chunk]
            x = torch.from_numpy(np.concatenate([Xc, block], axis=0)).unsqueeze(0).to(self.device)
            y = torch.from_numpy(self._yc).unsqueeze(0).to(self.device)
            out = self.model(x, y)
            # NanoTabICLv2 returns predictions for the QUERY rows only — verified:
            # x=(1,100,d), y=(1,70) gives out=(1,30,...). Asserted rather than
            # defensively sliced, so a future architecture change fails loudly instead
            # of silently scoring the wrong rows.
            if out.shape[1] != len(block):
                raise RuntimeError(
                    f"expected {len(block)} query predictions, got {out.shape[1]}. The "
                    f"model's output convention changed; fix this slice before trusting "
                    f"any number from it."
                )
            q = out[0]

            if self.meta["regression"]:
                # 999 quantiles -> a point prediction. The MEDIAN, not the mean: for a
                # bimodal LGD predictive the mean lands in the empty middle where no
                # loan actually sits, which is the failure mode this project is about.
                # Sorted first — a quantile head predicts each level independently, so
                # column Q//2 is the median only once crossing is fixed.
                from src.train.loop import enforce_monotonic_quantiles

                q = enforce_monotonic_quantiles(q)
                point = q[:, q.shape[1] // 2]
                preds.append(point.float().cpu().numpy())
            else:
                # Defensive slice, matching upstream's `_compute_batch_loss`. MEASURED: with
                # `tabicl` this is a no-op, because `forward` already returns exactly the
                # classes present in the context (a 10-wide head with binary y returns 2
                # columns). Kept so that a change in that convention becomes a shape error
                # here rather than a softmax quietly spreading P(default) over classes the
                # data never contained.
                n_classes = min(self.meta["n_classes"], q.shape[-1])
                prob = torch.softmax(q[..., :n_classes].float(), dim=-1)[:, 1]
                preds.append(prob.cpu().numpy())

        out_arr = np.concatenate(preds)
        if self.meta["regression"]:
            # LGD is a loss fraction. Clipping here matches what the credit baselines do.
            out_arr = np.clip(out_arr, 0.0, 1.0)
        return out_arr

    @torch.no_grad()
    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The full predictive distribution, for pinball/CRPS and boundary-mass checks.

        This is the reason a quantile head is worth having: a point prediction cannot
        say "11% chance of exactly zero loss", and that is the quantity the project
        claims to improve.
        """
        if not self.meta["regression"] or self._Xc is None:
            return None
        from src.train.loop import quantile_levels

        ctx_median = np.nan_to_num(np.nanmedian(self._Xc, axis=0))
        Xc = np.where(np.isfinite(self._Xc), self._Xc, ctx_median)
        Xq = np.where(np.isfinite(np.asarray(X, np.float32)), np.asarray(X, np.float32), ctx_median)

        rows = []
        chunk = max(1, min(2048, self.context_rows))
        for start in range(0, len(Xq), chunk):
            block = Xq[start : start + chunk]
            x = torch.from_numpy(np.concatenate([Xc, block], axis=0)).unsqueeze(0).to(self.device)
            y = torch.from_numpy(self._yc).unsqueeze(0).to(self.device)
            out = self.model(x, y)
            if out.shape[1] != len(block):
                raise RuntimeError(f"expected {len(block)} query rows, got {out.shape[1]}")
            rows.append(out[0].float().cpu().numpy())

        from src.train.loop import enforce_monotonic_quantiles

        # Sort BEFORE clipping: clipping cannot introduce crossing, and sorting a clipped row
        # is the same row, but doing it in this order keeps the two operations independent.
        # Pinball, CRPS, coverage and PIT all read this grid as ordered.
        quants = np.clip(enforce_monotonic_quantiles(np.concatenate(rows, axis=0)), 0.0, 1.0)
        levels = quantile_levels(int(self.meta["num_quantiles"])).numpy()
        return quants, levels


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
