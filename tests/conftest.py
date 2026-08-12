"""Shared fixtures.

Every test here needs a config small enough to run in about a second.

Note on the torch requirement: it is declared per *module* with
`pytest.importorskip("torch")`, NOT here. A skip raised at conftest import time
aborts the entire session, so the config and paths tests — which need no torch —
would silently stop running on a machine that only has the light dependencies.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(autouse=True)
def _headless_matplotlib() -> None:
    """Force the Agg backend. A compute node has no display, and the default backend either
    fails there or opens a window that blocks the run."""
    import matplotlib

    matplotlib.use("Agg", force=True)


@pytest.fixture
def isolated_output(tmp_path, monkeypatch) -> Path:
    """Point every resolved path at `tmp_path` for the duration of one test.

    Setting `VSC_DATA` *and* the staging override is how the cluster's two-tier layout gets
    exercised off-cluster: `paths.on_vsc()` becomes true, so the test covers the branch that
    otherwise only ever runs in production.

    Use this for anything that writes. Setting only the staging root is not enough — figures,
    logs and manifests hang off `outputs_dir()`, which ignores staging and stays in the repo,
    so a test that set only staging wrote into the real `output/` tree and left files behind.
    """
    import importlib

    from src.utils import paths

    monkeypatch.setenv("VSC_DATA", str(tmp_path / "vsc_data"))
    monkeypatch.setenv(paths.STAGING_ENV_VARS[0], str(tmp_path / "staging"))
    importlib.reload(paths)
    # Fail loudly rather than write into the real tree. Both tiers, because they resolve
    # through different branches and only `results/` follows staging.
    for resolved in (paths.outputs_dir(), paths.results_dir(), paths.logs_dir()):
        assert tmp_path in resolved.parents, f"not isolated: {resolved} is outside {tmp_path}"
    yield tmp_path
    importlib.reload(paths)


@pytest.fixture
def lgd_cfg() -> dict:
    """A tiny LGD config: real code paths, ~100 rows, a handful of features."""
    from src.utils.config import expand_with_seeds, load

    cfg = expand_with_seeds(load(ROOT / "config" / "Exp1_LGD.yaml"))[0]
    return _shrink(cfg)


@pytest.fixture
def pd_cfg() -> dict:
    from src.utils.config import expand_with_seeds, load

    cfg = expand_with_seeds(load(ROOT / "config" / "Exp1_PD.yaml"))[0]
    return _shrink(cfg)


def _shrink(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    p = cfg["prior"]
    p["n_rows_range"] = [96, 128]
    p["n_features_range"] = [4, 8]
    p["max_features"] = 16
    p["n_nodes_range"] = [2, 5]
    p["max_filter_attempts"] = 6
    # `model:` and `architecture:` are NOT set this way in any real config — the architecture
    # is TabICLv2's own, from the upstream `tabicl` package, and is never swept. Tests use the
    # vendored fallback at a fraction of the size purely so the suite runs in seconds on a
    # machine with nothing installed. NO RESULT may come from this path: its parameter names
    # do not match the released checkpoints. `test_every_experiment_names_the_same_architecture`
    # is what keeps the real configs honest.
    # Prefer the REAL architecture, shrunk, so the suite exercises the model the experiments
    # actually train. Falls back only when `tabicl` is absent, which keeps a bare clone
    # runnable. Unknown override names are dropped per architecture by
    # `src.models.architecture._translate`, so one dict works for both.
    from src.models.architecture import DEFAULT, is_available

    cfg["architecture"] = DEFAULT if is_available(DEFAULT) else "nanotabicl"
    m = cfg.setdefault("model", {})
    m.update(
        {
            "embed_dim": 32,
            "col_num_blocks": 1,
            "row_num_blocks": 1,
            "icl_num_blocks": 2,
            "col_nhead": 2,
            "row_nhead": 2,
            "icl_nhead": 2,
            "n_cls_rows": 16,
        }
    )
    t = cfg["train"]
    t.update(
        {
            "max_steps": 2,
            "batch_size": 2,
            "micro_batch_size": 2,
            "num_quantiles": 16,
            "amp": False,
            "num_workers": 0,
            "log_every": 1,
            "save_temp_every": 0,
            "save_perm_every": 0,
        }
    )
    # to_file=True because several tests assert on the log file's CONTENT — that
    # is the artefact the cluster actually produces. Locally, real runs default to
    # no file at all (see src/utils/logging_setup.setup_logging).
    cfg["logging"] = {"level": "INFO", "console": False, "log_prior_every": 0, "to_file": True}
    return cfg


@pytest.fixture
def rng():
    from src.prior.rng import PriorRNG

    return PriorRNG(0)
