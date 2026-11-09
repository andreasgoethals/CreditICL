import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import private_data
from src.data.dataset_registry import display_name, is_proprietary
from src.eval.ood import is_credit_like
from src.utils.check_publication import _decode, findings
from src.visualize.data_plots import correlation_pages, summary_table


@pytest.fixture
def restricted_site(tmp_path, monkeypatch):
    path = tmp_path / "site.py"
    slugs = {alias: f"{index:04}.restricted_source_{index}"
             for index, alias in enumerate(sorted(private_data.PRIVATE_KEYS), 1)}
    path.write_text(f"LEGACY_SLUGS = {slugs!r}\n", encoding="utf-8")
    monkeypatch.setenv("CREDITICL_PRIVATE_DATA", str(path))
    return slugs


def test_legacy_storage_and_ood_exclusion(restricted_site):
    legacy = restricted_site["PropLGD5"]
    assert display_name(legacy) == "PropLGD5"
    assert is_proprietary(legacy)
    assert private_data.storage_keys("lgd", "PropLGD5") == ("PropLGD5", legacy)
    assert is_credit_like(legacy.split(".", 1)[1])
    assert display_name("0001.gmsc") == "GMSC"


def test_publication_dimensions_and_correlations_exclude_restricted_source(restricted_site):
    legacy = restricted_site["PropPD1"]
    sample = SimpleNamespace(y=np.array([0, 1]), X=np.ones((2, 2)), n_rows=2,
                             n_features=2, cat_indices=[1])
    datasets = {legacy: sample, "0001.gmsc": sample}
    table = summary_table(datasets, "pd").set_index("dataset")
    assert table.loc["PropPD1", ["rows", "features", "categorical", "n positive"]].isna().all()
    assert table.loc["GMSC", "rows"] == 2
    assert correlation_pages(datasets, per_page=1) == 1


def test_publication_check_reads_saved_outputs_and_does_not_report_secrets(restricted_site):
    legacy = restricted_site["PropLGD5"]
    notebook = {"cells": [{"source": ["PropLGD5"], "outputs": [
        {"text": [f"{legacy} raw_private_amount", "PropLGD5: 12,345 rows"]},
        {"data": {"image/png": "raw_private_amount"}},
    ]}]}
    content = _decode("saved.ipynb", json.dumps(notebook).encode())
    assert [kind for _, kind in findings(content, ["raw_private_amount"], [12345])] == [
        "source identifier", "restricted column", "restricted row count",
    ]
    assert not list(findings("raw_private_amount.public_component", ["raw_private_amount"], []))


def test_explicit_missing_configuration_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITICL_PRIVATE_DATA", str(tmp_path / "missing.py"))
    with pytest.raises(FileNotFoundError, match="local dataset configuration"):
        private_data.site()
