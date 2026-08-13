"""Out-of-domain evaluation: did the credit-tailored prior break everything else?

No test here touches the network. Fetching is a login-node operation; what matters for
correctness is the **filtering, caching and reporting**, all of which work off a local
cache that these tests build by hand.

The single most important property: a credit dataset must never end up in the
out-of-domain set. OpenML-CC18 contains `credit-g` and `credit-approval`, and O'Prior
specifically names Credit-g as a dataset where its gains vanish. Letting either through
would quietly turn the "we did not hurt general performance" claim into a comparison
against the very domain we specialise toward.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from src.eval import ood


@pytest.fixture
def ood_cache(tmp_path, monkeypatch):
    """A hand-built OOD cache: 2 classification + 2 regression datasets."""
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as paths

    importlib.reload(paths)
    importlib.reload(ood)

    root = ood.ood_root()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    entries = []

    for i, (name, kind) in enumerate(
        [("wine", "classification"), ("segment", "classification"),
         ("abalone", "regression"), ("cpu_act", "regression")]
    ):
        n, d = 400, 6
        X = rng.normal(size=(n, d)).astype(np.float32)
        signal = X[:, 0] + 0.5 * X[:, 1]
        if kind == "classification":
            y = (signal > 0).astype(np.int64)
            n_classes = 2
        else:
            y = signal.astype(np.float32)
            n_classes = None
        e = ood.OODDataset(name=name, openml_id=1000 + i, kind=kind,
                           n_rows=n, n_features=d, n_classes=n_classes)
        np.savez_compressed(root / f"{e.slug}.npz", X=X, y=y,
                            cat_indices=np.asarray([], dtype=np.int64))
        entries.append(e)

    from dataclasses import asdict

    ood.manifest_path().write_text(
        json.dumps({"version": ood.OOD_VERSION, "suites": ood.SUITES,
                    "datasets": [asdict(e) for e in entries]}),
        encoding="utf-8",
    )
    yield ood
    importlib.reload(paths)
    importlib.reload(ood)


# -- the credit filter, the one that must not fail --------------------------


@pytest.mark.parametrize(
    "name",
    ["credit-g", "credit-approval", "GermanCredit", "default-of-credit-card-clients",
     "loan_default", "bank-marketing", "home_mortgage", "LendingClub", "hmeq",
     "insurance_claims", "risk_factors", "financial_distress"],
)
def test_credit_like_datasets_are_excluded(name):
    """These are NOT out-of-domain. Deliberately over-inclusive: dropping a borderline
    dataset costs one of ten tasks; keeping one silently weakens the whole claim."""
    assert ood.is_credit_like(name), f"{name} should have been filtered out"


@pytest.mark.parametrize(
    "name", ["wine-quality", "segment", "abalone", "phoneme", "mfeat-pixel", "cpu_act",
             "diamonds", "car", "dna", "PhishingWebsites"]
)
def test_genuinely_unrelated_datasets_are_kept(name):
    assert not ood.is_credit_like(name), f"{name} was filtered out but is fine"


def test_the_suites_are_named_not_hardcoded_ids():
    """Suite NAMES are verifiable; dataset ids written from memory are not. The ids must
    come from the API at fetch time and be pinned into the manifest."""
    for kind, names in ood.SUITES.items():
        assert isinstance(names, list) and names, kind
        for name in names:
            assert isinstance(name, str) and not name.isdigit(), f"{name} looks like an id"
    text = pathlib.Path(ood.__file__).read_text(encoding="utf-8")
    assert "NEVER hard-coded" in text or "never hard-coded" in text.lower()


def test_both_task_kinds_are_covered_by_several_suites():
    """A mean over one suite is one suite's opinion, and the out-of-domain average is the
    number that would catch a prior which buys credit performance by destroying generality.
    Both kinds are needed because LGD is regression and PD is classification."""
    assert set(ood.SUITES) == {"classification", "regression"}
    for kind, names in ood.SUITES.items():
        assert len(names) >= 2, f"{kind} rests on a single suite: {names}"


def test_the_suites_tabiclv2_reports_on_are_included():
    """TabICLv2 evaluates on TabArena and TALENT. Including them is what lets our numbers be
    put beside the model we started from; O'Prior's CC18 stays so the control also stays
    comparable with the closest prior work."""
    flat = " ".join(n.lower() for names in ood.SUITES.values() for n in names)
    for expected in ("cc18", "ctr23", "tabarena", "talent"):
        assert expected in flat, f"{expected} is missing from SUITES"


def test_the_out_of_domain_sample_is_large_enough_to_average():
    """10 datasets was one suite's worth and too noisy to detect a real regression."""
    assert ood.N_PER_SUITE >= 20
    total = sum(len(v) for v in ood.SUITES.values()) * ood.N_PER_SUITE
    assert total >= 100, f"only {total} out-of-domain datasets across both kinds"


def test_an_unresolvable_suite_does_not_abort_the_fetch():
    """TabArena and TALENT are not guaranteed to exist as OpenML studies under these aliases,
    and a login node's API access is not guaranteed either. A partial cache is useful; losing
    the whole download to one bad alias is not."""
    text = pathlib.Path(ood.__file__).read_text(encoding="utf-8")
    block = text[text.index("def fetch_ood_datasets"):]
    assert "UNAVAILABLE" in block and "continue" in block, (
        "the fetch loop must skip a suite it cannot resolve rather than raising"
    )


# -- cache contract ---------------------------------------------------------


def test_status_is_empty_without_a_cache(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path / "nothing"))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as paths

    importlib.reload(paths)
    importlib.reload(ood)
    st = ood.ood_status()
    assert st["exists"] is False and st["n_datasets"] == 0 and st["complete"] is False


def test_status_reports_the_cache(ood_cache):
    st = ood_cache.ood_status()
    assert st["exists"] and st["n_datasets"] == 4
    assert st["by_kind"] == {"classification": 2, "regression": 2}
    assert st["complete"] is False, "4 < N_PER_TASK, so it must NOT read as complete"


def test_loading_a_missing_dataset_names_the_fix(ood_cache):
    """Compute nodes have no internet, so the error must say where to run the fetch."""
    ghost = ood_cache.OODDataset(name="ghost", openml_id=9999, kind="regression",
                                 n_rows=1, n_features=1)
    with pytest.raises(FileNotFoundError, match="src.utils.fetch_ood"):
        ood_cache.load_ood_dataset(ghost)
    with pytest.raises(FileNotFoundError, match="LOGIN NODE|login node"):
        ood_cache.load_ood_dataset(ghost)


def test_slug_is_filesystem_safe_and_unique():
    """OpenML names contain spaces, dots and slashes; two datasets can share a name."""
    a = ood.OODDataset(name="wall-robot navigation/v2", openml_id=1, kind="classification",
                       n_rows=1, n_features=1)
    b = ood.OODDataset(name="wall-robot navigation/v2", openml_id=2, kind="classification",
                       n_rows=1, n_features=1)
    for ch in " /\\.:":
        assert ch not in a.slug.replace(".", "", 1), f"{ch!r} leaked into the slug"
    assert a.slug != b.slug, "the id must disambiguate identical names"


def test_stale_cache_version_is_ignored(ood_cache):
    """A cache from an older layout must not be silently mixed in."""
    p = ood_cache.manifest_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    data["version"] = 999
    p.write_text(json.dumps(data), encoding="utf-8")
    assert ood_cache.list_ood_datasets() == []


def test_round_trip(ood_cache):
    entry = ood_cache.list_ood_datasets("regression")[0]
    X, y, cat = ood_cache.load_ood_dataset(entry)
    assert X.shape == (entry.n_rows, entry.n_features)
    assert len(y) == entry.n_rows and cat == []


def test_fetch_without_openml_names_the_install(ood_cache, monkeypatch):
    """openml is an optional extra; the error must say how to get it."""
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "openml":
            raise ImportError("no openml")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"\[eval\]"):
        ood_cache.fetch_ood_datasets()


# -- the runner -------------------------------------------------------------


def test_ood_eval_scores_both_kinds(ood_cache):
    from src.eval.ood_runner import OODEvalConfig, run_ood

    df = run_ood(OODEvalConfig(models=["linear"], seeds=[0]))
    assert len(df) == 4
    assert (df["status"] == "ok").all(), df.get("error").tolist()
    clf = df[df["kind"] == "classification"]
    reg = df[df["kind"] == "regression"]
    assert clf["roc_auc"].notna().all() and clf["roc_auc"].min() > 0.5, "signal is learnable"
    assert reg["r2"].notna().all() and reg["r2"].min() > 0.5


def test_classification_and_regression_are_never_pooled(ood_cache):
    """A mean of ROC-AUC and R^2 is not a number."""
    from src.eval.ood_runner import OODEvalConfig, run_ood, summarise_ood

    summary = summarise_ood(run_ood(OODEvalConfig(models=["linear"], seeds=[0])))
    assert set(summary["kind"]) == {"classification", "regression"}
    assert set(summary["metric"]) == {"roc_auc", "r2"}
    assert len(summary) == 2, "one row per (kind, model), never a single pooled row"


def test_text_summary_reports_deltas_against_a_reference(ood_cache):
    """The absolute AUC on CC18 is uninteresting; the delta against the control arm is
    the entire point of running this."""
    from src.eval.ood_runner import OODEvalConfig, ood_text_summary, run_ood

    df = run_ood(OODEvalConfig(models=["linear"], seeds=[0]))
    text = ood_text_summary(df, reference_model="linear")
    assert "delta vs linear: +0.0000" in text
    assert "HOW TO READ THIS" in text
    assert "nothing to do with credit" in text


def test_text_summary_survives_all_failures(ood_cache):
    import pandas as pd

    from src.eval.ood_runner import ood_text_summary

    df = pd.DataFrame([{"status": "failed", "kind": "classification", "dataset": "x",
                        "model": "linear", "error": "boom"}])
    text = ood_text_summary(df)
    assert "no successful cells" in text


def test_a_failing_cell_becomes_a_row_not_an_exception(ood_cache):
    """One bad dataset must not cost the other nineteen."""
    from src.eval.ood_runner import OODEvalConfig, evaluate_one_ood

    ghost = ood_cache.OODDataset(name="ghost", openml_id=9999, kind="regression",
                                 n_rows=1, n_features=1)
    row = evaluate_one_ood(ghost, "linear", 0, OODEvalConfig())
    assert row["status"] == "failed" and "FileNotFoundError" in row["error"]


def test_run_ood_without_a_cache_names_the_fix(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path / "empty"))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as paths

    importlib.reload(paths)
    importlib.reload(ood)
    import src.eval.ood_runner as runner

    importlib.reload(runner)
    with pytest.raises(FileNotFoundError, match="src.utils.fetch_ood"):
        runner.run_ood(runner.OODEvalConfig(models=["linear"]))


def test_ood_results_go_to_their_own_tree():
    """Out-of-domain numbers must never be written next to the credit results."""
    from src.utils.paths import results_dir

    assert results_dir("ood", "eval").parts[-2:] == ("ood", "eval")
    with pytest.raises(ValueError, match="results namespace"):
        results_dir("not_a_namespace", "eval")


def test_the_cache_is_written_atomically_and_with_the_right_name(tmp_path):
    """REGRESSION, from the first cluster fetch. `np.savez_compressed` given a PATH whose name
    does not end in `.npz` silently APPENDS the extension, so writing to `x.npz.tmp` produced
    `x.npz.tmp.npz` and the rename then raised FileNotFoundError on the very first dataset.
    Passing an open handle suppresses the renaming."""
    import numpy as np

    out = tmp_path / "3.kr-vs-kp.npz"
    tmp = out.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, X=np.zeros((2, 2)), y=np.zeros(2))
    assert tmp.is_file(), "the handle form must write exactly the name it was given"
    assert not (tmp_path / "3.kr-vs-kp.npz.tmp.npz").exists()
    tmp.replace(out)
    assert out.is_file() and np.load(out)["X"].shape == (2, 2)


def test_the_fetcher_does_not_pass_a_path_to_savez():
    """The bug is invisible in review unless you know numpy renames, so pin the fix in place."""
    import pathlib

    text = pathlib.Path(ood.__file__).read_text(encoding="utf-8")
    block = text[text.index("def fetch_ood_datasets"):]
    assert "with tmp.open(\"wb\") as fh:" in block
    assert "np.savez_compressed(tmp" not in block, (
        "passing the Path back means numpy appends .npz and the rename fails"
    )
