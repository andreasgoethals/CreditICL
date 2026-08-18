"""The NaN diagnostic is the only way to see inside a checkpoint that stays on the cluster.

A checkpoint is ~229 MB; a log is not. So the tool has to run there and report here, which
means its output is the entire deliverable — if it points at the wrong module, or says
"everything finite" on a broken model, the whole round trip is wasted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch not installed")

ROOT = Path(__file__).resolve().parents[1]


def test_it_names_the_module_where_the_first_nan_appears(tmp_path, capsys):
    """Verified against a PLANTED failure: a NaN-valued weight in ICL block 5 must be reported
    at block 5, not at the output where it eventually surfaces."""
    import torch

    from src.models.architecture import build_model, is_available

    if not is_available("tabicl"):
        pytest.skip("upstream tabicl not installed here")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_nan", ROOT / "scripts" / "diagnose_nan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(0)
    model = build_model("lgd", architecture="tabicl").eval()
    target = "icl_predictor.tf_icl.blocks.5.attn.out_proj.weight"
    with torch.no_grad():
        dict(model.named_parameters())[target].fill_(float("nan"))

    lines: list[str] = []
    mod.walk(model, torch.randn(1, 96, 6), torch.rand(1, 64), lines.append)
    text = "\n".join(lines)

    assert "FIRST non-finite output is" in text, "a broken model must not read as healthy"
    bad = [ln for ln in lines if "-> NON-FINITE" in ln]
    assert bad, "the culprit line is missing"
    assert "blocks.5" in bad[0], f"pointed at the wrong module: {bad[0]}"
    # the run-up matters as much as the culprit: it says overflow vs collapse
    assert sum("absmax=" in ln for ln in lines) >= 5, "magnitudes leading up are missing"


def test_a_healthy_model_reports_healthy():
    """The opposite error — crying wolf — would send the reader chasing nothing."""
    import importlib.util

    import torch

    from src.models.architecture import build_model, is_available

    if not is_available("tabicl"):
        pytest.skip("upstream tabicl not installed here")

    spec = importlib.util.spec_from_file_location(
        "diagnose_nan", ROOT / "scripts" / "diagnose_nan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(0)
    lines: list[str] = []
    mod.walk(build_model("lgd", architecture="tabicl").eval(),
             torch.randn(1, 96, 6), torch.rand(1, 64), lines.append)
    assert any("every module finite" in ln for ln in lines)
