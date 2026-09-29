"""The learning curve on real data for the TabPFN-3 arms of Experiment 2.

Same data, same fixed context/test rows, same metrics as `ProgressTracker` — only the model call
differs. The current weights are written to a checkpoint in TabPFN's own format and scored
through `src.eval.baselines.TabPFNBaseline`, the wrapper the benchmark scores every TabPFN
checkpoint with: TabPFN's own preprocessing and ensemble, the library's default ensemble size.
A curve point is therefore the benchmark's measurement on a smaller context, not a proxy for it.
The weights file lives beside the arm's checkpoints (the big tier), is not named `step-*`, and is
removed after each point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .progress import ProgressConfig, ProgressTracker


class TabPFNProgressTracker(ProgressTracker):
    def __init__(self, cfg: ProgressConfig, task: str, run_name: str, out_dir: Path, *, trainer: Any):
        super().__init__(cfg, task, run_name, out_dir)
        self.trainer = trainer
        self._weights: Path | None = None

    def record(self, model, **kwargs) -> dict[str, Any]:
        path = Path(self.trainer.ckpt_dir) / "monitor-weights.ckpt"
        try:
            self._weights = self.trainer.save_weights(path)
            return super().record(model, **kwargs)
        finally:
            self._weights = None
            path.unlink(missing_ok=True)

    def _fit_predict(self, model, Xc: np.ndarray, yc: np.ndarray, Xt: np.ndarray, *,
                     regression: bool) -> dict[str, np.ndarray]:
        from src.eval.baselines import build

        if self._weights is None:
            raise RuntimeError("no weights file: _fit_predict runs inside record()")
        m = build("tabpfn3", "lgd" if regression else "pd", seed=self.cfg.seed,
                  model_path=str(self._weights))
        m.fit(Xc, yc, [])
        if regression:
            q = m.predict_quantiles(Xt)
            if q is None:
                raise RuntimeError("TabPFN regressor returned no quantiles")
            return {"point": m.predict(Xt), "quantiles": q[0]}
        return {"prob": m.predict(Xt)}
