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


@pytest.fixture
def lgd_cfg() -> dict:
    """A tiny LGD config: real code paths, ~100 rows, a handful of features."""
    from src.utils.config import expand_with_seeds, load_yaml

    cfg = expand_with_seeds(load_yaml(ROOT / "config" / "LGD.yaml"))[0]
    return _shrink(cfg)


@pytest.fixture
def pd_cfg() -> dict:
    from src.utils.config import expand_with_seeds, load_yaml

    cfg = expand_with_seeds(load_yaml(ROOT / "config" / "PD.yaml"))[0]
    return _shrink(cfg)


def _shrink(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    p = cfg["prior"]
    p["n_rows_range"] = [96, 128]
    p["n_features_range"] = [4, 8]
    p["max_features"] = 16
    p["n_nodes_range"] = [2, 5]
    p["max_filter_attempts"] = 6
    # `model:` is absent from the real configs — the architecture is TabICLv2's and is
    # fixed in NanoTabICLv2's defaults. Tests override it to a tiny model purely for
    # speed, which is the only legitimate reason to set it.
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
