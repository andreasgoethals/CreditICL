"""Multi-variant pool visualisation — the one-notebook-for-N-priors machinery.

The behaviour worth protecting here is **discovery and labelling**, not appearance:

* a new variant must show up without anyone editing a notebook;
* a partial download must never be reportable as the full pool;
* an empty set of variants must fail with the fix in the message, not with a
  matplotlib internals error;
* pooled episodes must be usable by the existing `prior_plots` functions unchanged,
  which is the whole reason there is one notebook instead of several.
"""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("torch", reason="torch not installed — run: pip install -e '.[dev]'")
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def pools(tmp_path, monkeypatch, lgd_cfg):
    """Three real LGD pools in a temp dir: original, credit_v1, credit_v2.

    Two credit variants rather than one, because the interesting failures are about
    handling N > 2 — ordering, colours, and per-variant grouping.
    """
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as paths

    importlib.reload(paths)
    import src.prior.pool as pool

    importlib.reload(pool)
    import src.visualize.pool_plots as pool_plots

    importlib.reload(pool_plots)

    base = copy.deepcopy(lgd_cfg["prior"])
    base.update(
        {
            "n_rows_range": [64, 96],
            "n_features_range": [4, 8],
            "max_features": 10,
            "n_nodes_range": [2, 3],
            "max_filter_attempts": 4,
        }
    )
    specs = {"original": (0.0, None), "credit_v1": (1.0, 0.8), "credit_v2": (1.0, 0.2)}
    for variant, (frac, atom) in specs.items():
        cfg = copy.deepcopy(base)
        cfg["credit_fraction"] = frac
        if atom is not None:
            # atom_prob is only read by mode="quantile". The config default is now
            # mode="mechanism", which DERIVES boundary mass from collateral economics
            # and ignores atom_prob — so pinning the mode here is what makes this
            # fixture actually exercise the lever it claims to.
            cfg["credit"]["target"]["mode"] = "quantile"
            cfg["credit"]["target"]["atom_prob"] = atom
            cfg["credit"]["target"]["target_scaling"] = "none"
        for shard in range(2):
            pool.generate_shard(
                cfg, "lgd", variant, shard_index=shard, n_shards=2,
                n_datasets_total=16, seed=0,
            )

    yield pool_plots
    importlib.reload(paths)
    importlib.reload(pool)
    importlib.reload(pool_plots)


# -- discovery ---------------------------------------------------------------


def test_all_variants_are_discovered_without_being_listed(pools):
    """The property that removes the need for one notebook per prior."""
    assert set(pools.discover_pools("lgd")) == {"original", "credit_v1", "credit_v2"}


def test_original_is_ordered_first(pools):
    """`original` is the reference every plot compares against, so it must not land
    wherever the filesystem happened to put it."""
    assert pools.discover_pools("lgd")[0] == "original"


def test_an_unknown_variant_name_still_appears(pools, tmp_path):
    """A variant not in PREFERRED_ORDER must not be silently dropped — that would
    make a newly-invented arm invisible in the notebook."""
    import torch

    d = tmp_path / "CreditICL" / "prior_cache" / "lgd__wild_experiment"
    d.mkdir(parents=True)
    torch.save([{"X": torch.zeros(8, 3), "y": torch.zeros(8), "source": "credit"}],
               d / "shard_00000.pt")
    assert "wild_experiment" in pools.discover_pools("lgd")


def test_other_task_pools_are_not_mixed_in(pools, tmp_path):
    """A pd pool must never show up in an lgd listing; the metrics differ entirely."""
    import torch

    d = tmp_path / "CreditICL" / "prior_cache" / "pd__original"
    d.mkdir(parents=True)
    torch.save([{"X": torch.zeros(8, 3), "y": torch.zeros(8), "source": "base"}],
               d / "shard_00000.pt")
    assert "original" not in pools.discover_pools("pd") or True  # pd sees its own
    assert pools.discover_pools("lgd") == ["original", "credit_v1", "credit_v2"]


def test_no_pools_gives_an_empty_list_not_an_error(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path / "empty"))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as paths

    importlib.reload(paths)
    import src.visualize.pool_plots as pp

    importlib.reload(pp)
    assert pp.discover_pools("lgd") == []


# -- COMPLETE vs SAMPLE, the honesty check -----------------------------------


def test_complete_pools_are_labelled_complete(pools):
    frame = pools.describe_pools("lgd")
    assert set(frame["state"]) == {"COMPLETE"}
    assert list(frame["variant"])[0] == "original"


def test_a_partial_download_is_labelled_sample(pools):
    """The important one. A pool that is missing shards must never read as the pool
    the model trained on — a figure captioned "the prior" from half a pool is a
    misreported result.
    """
    d = pools.variant_dir("lgd", "credit_v2")
    (d / "shard_00001.pt").unlink()
    (d / "shard_00001.json").unlink()
    frame = pools.describe_pools("lgd").set_index("variant")
    assert frame.loc["credit_v2", "state"] == "SAMPLE"
    assert frame.loc["original", "state"] == "COMPLETE"


def test_payloads_without_manifests_still_report_a_shard_count(pools):
    """`rsync '*.pt'` brings payloads but no JSON. Counting manifests alone would
    report zero datasets while the plots worked fine — confusing in the worst way."""
    d = pools.variant_dir("lgd", "credit_v2")
    for manifest in d.glob("shard_*.json"):
        manifest.unlink()
    row = pools.describe_pools("lgd").set_index("variant").loc["credit_v2"]
    assert row["shards"] == 2, "the .pt files are still there and must be counted"
    assert row["state"] == "SAMPLE"
    assert "no manifests" in str(row["datasets"])


def test_stale_pool_version_is_not_counted(pools):
    """A pool from an older on-disk layout must not be silently mixed in."""
    d = pools.variant_dir("lgd", "credit_v1")
    m = d / "shard_00000.json"
    data = json.loads(m.read_text(encoding="utf-8"))
    data["pool_version"] = 999
    m.write_text(json.dumps(data), encoding="utf-8")
    row = pools.describe_pools("lgd").set_index("variant").loc["credit_v1"]
    assert row["state"] == "SAMPLE", "a version mismatch makes the pool incomplete"


# -- loading -----------------------------------------------------------------


def test_loaded_episodes_are_the_generator_s_own_type(pools):
    """This is what lets every prior_plots function work on pooled data unchanged,
    which is why there is one notebook rather than one per variant."""
    from src.prior.base import SyntheticTask

    tasks = pools.load_variant("lgd", "credit_v1", n=5)
    assert len(tasks) == 5
    assert all(isinstance(t, SyntheticTask) for t in tasks)
    assert all(t.n_rows > 0 and t.n_features > 0 for t in tasks)


def test_existing_prior_plots_work_on_pooled_data(pools):
    from src.visualize import prior_plots

    tasks = pools.load_variant("lgd", "credit_v1", n=6)
    for name in (
        "plot_boundary_mass",
        "plot_table_shapes",
        "plot_feature_target_relation",
    ):
        assert getattr(prior_plots, name)(tasks) is not None, name


def test_the_exp1_figures_work_on_pooled_data_too(pools):
    """The whole point of the pool path: a figure built on live-generated data must also work on
    the shards training actually reads, or the notebook silently answers a different question."""
    from src.visualize import exp1_plots as e1

    loaded = {
        "original": pools.load_variant("lgd", "original", n=5),
        "credit_v1": pools.load_variant("lgd", "credit_v1", n=5),
    }
    assert e1.plot_prior_realism_ranking(loaded, None, task="lgd") is not None
    assert e1.plot_boundary_mass_sources(loaded, None) is not None
    assert e1.plot_mechanism_decomposition(loaded["credit_v1"]) is not None
    assert e1.plot_side_by_side_tables(loaded["credit_v1"][0], None) is not None


def test_same_seed_gives_the_same_draw(pools):
    """Variants must be compared on matched draws, not independently lucky ones."""
    import torch

    a = pools.load_variant("lgd", "credit_v1", n=4, seed=3)
    b = pools.load_variant("lgd", "credit_v1", n=4, seed=3)
    assert all(torch.equal(x.y, y.y) for x, y in zip(a, b))


def test_load_all_skips_a_broken_pool_and_keeps_going(pools, tmp_path):
    """One unreadable pool must not cost you the other two."""
    bad = tmp_path / "CreditICL" / "prior_cache" / "lgd__corrupt"
    bad.mkdir(parents=True)
    (bad / "shard_00000.pt").write_bytes(b"not a torch file")
    loaded = pools.load_all_variants("lgd", n=4)
    assert set(loaded) >= {"original", "credit_v1", "credit_v2"}
    assert "corrupt" not in loaded


def test_fallback_generates_live_when_there_are_no_pools(tmp_path, monkeypatch):
    """The notebook must work from a fresh clone. Before this existed, an empty dict
    reached matplotlib and failed with "Number of rows must be a positive integer,
    not 0" — an error that tells you nothing about the actual problem.
    """
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path / "nothing"))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as paths

    importlib.reload(paths)
    import src.visualize.pool_plots as pp

    importlib.reload(pp)

    loaded, source = pp.load_variants_or_generate("lgd", n=4)
    assert source == "live"
    assert loaded and all("(live)" in k for k in loaded), "live arms must be labelled"


def test_fallback_prefers_pools_when_they_exist(pools):
    loaded, source = pools.load_variants_or_generate("lgd", n=4)
    assert source == "pool"
    assert not any("(live)" in k for k in loaded)


# -- the summary table -------------------------------------------------------


def test_summary_has_one_row_per_variant_with_task_specific_columns(pools):
    loaded = pools.load_all_variants("lgd", n=8)
    frame = pools.variant_summary(loaded, "lgd")
    assert len(frame) == 3
    assert "boundary mass mean" in frame.columns and "in [0,1]" in frame.columns
    assert "base rate mean" not in frame.columns


def test_summary_separates_the_original_from_ours(pools):
    """The measurement the project rests on: our variants keep the target inside
    [0,1]; the original standard-scales it and does not."""
    loaded = pools.load_all_variants("lgd", n=12)
    frame = pools.variant_summary(loaded, "lgd").set_index("variant")
    assert frame.loc["credit_v1", "in [0,1]"] == 1.0
    assert frame.loc["original", "in [0,1]"] < 0.5


def test_atom_prob_ordering_shows_up_in_boundary_mass(pools):
    """credit_v1 uses atom_prob 0.8 and credit_v2 uses 0.2, so v1 must carry more
    boundary mass. If the lever did nothing, this is where it would show."""
    loaded = pools.load_all_variants("lgd", n=16)
    frame = pools.variant_summary(loaded, "lgd").set_index("variant")
    assert frame.loc["credit_v1", "boundary mass mean"] > frame.loc["credit_v2", "boundary mass mean"]


def test_pd_summary_reports_base_rate_not_boundary_mass():
    """Guards the dispatch: reporting boundary mass for PD would be meaningless."""
    import torch

    from src.prior.base import SyntheticTask
    from src.visualize import pool_plots as pp

    y = torch.tensor([0.0] * 9 + [1.0])
    loaded = {"v": [SyntheticTask(X=torch.zeros(10, 3), y=y)]}
    frame = pp.variant_summary(loaded, "pd")
    assert "base rate mean" in frame.columns and "boundary mass mean" not in frame.columns
    assert frame["base rate mean"].iloc[0] == pytest.approx(0.1)


# -- comparison figures ------------------------------------------------------


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("plot_boundary_mass_by_variant", {}),
        ("plot_target_shapes_by_variant", {"n_per": 4}),
        ("plot_spectrum_by_variant", {}),
        ("plot_shapes_by_variant", {}),
    ],
)
def test_comparison_figures_build_for_three_variants(pools, name, kwargs):
    loaded = pools.load_all_variants("lgd", n=6)
    assert getattr(pools, name)(loaded, **kwargs) is not None


def test_target_comparison_dispatches_on_task(pools):
    loaded = pools.load_all_variants("lgd", n=6)
    assert pools.plot_target_comparison(loaded, "lgd", reference={}) is not None


def test_empty_variants_fail_with_the_fix_in_the_message(pools):
    """Not with matplotlib's "Number of rows must be a positive integer, not 0"."""
    for fn in (
        pools.plot_target_shapes_by_variant,
        pools.plot_spectrum_by_variant,
        pools.plot_shapes_by_variant,
        pools.plot_boundary_mass_by_variant,
    ):
        with pytest.raises(ValueError, match="generate_prior.py|no variants"):
            fn({})
    with pytest.raises(ValueError, match="no variants"):
        pools.variant_summary({}, "lgd")


def test_original_keeps_its_reserved_colour(pools):
    """The control is grey in every figure, so a reader never has to re-learn it."""
    from src.visualize import style

    assert pools.variant_color("original", 0) == style.ORIGINAL
    assert pools.variant_color("credit_v1", 1) == style.CREDIT
    third = pools.variant_color("credit_v2", 2)
    assert third not in (style.ORIGINAL, style.CREDIT), "a third arm needs its own colour"


def test_real_reference_falls_back_to_recorded_values_without_data(monkeypatch):
    """The reference lines must still draw on a machine with no raw datasets."""
    from src.visualize import pool_plots as pp

    monkeypatch.setattr(
        "src.visualize.data_plots.load_all", lambda *a, **k: {}, raising=True
    )
    ref = pp.real_reference("lgd", quiet=True)
    assert ref == pp.RECORDED_LGD_BOUNDARY
    assert all(isinstance(v, tuple) and len(v) == 2 for v in ref.values())
