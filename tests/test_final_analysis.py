"""Final benchmark coverage and paired comparisons, independent of monitoring files."""
import pandas as pd
import pytest

from src.eval import analysis
from src.eval.comparisons import paired_ablation_effects, paired_reference_effects
from src.eval.protocol import PROTOCOL


def cells(name, score, datasets=("a", "b", "c"), seeds=(0, 1, 2)):
    trained = name.startswith("exp")
    return [dict(dataset=d, model="crediticl" if trained else name,
                 info_run_name=f"{name}__s{s}" if trained else "", seed=0, fold=f,
                 status="ok", protocol=PROTOCOL, roc_auc=score, log_loss=1-score)
            for d in datasets for s in (seeds if trained else (0,)) for f in range(5)]


def test_ablation_compares_every_removal_to_seed_matched_full_prior():
    full = "exp1_pd__prior-credit_fraction=0p5"
    names = ["exp1ablations_pd__arm=" + a for a in ("no_macro_target", "no_tail_warp")]
    df = pd.DataFrame(cells(full, .8) + cells(names[0], .7) + cells(names[1], .6))
    out = paired_ablation_effects(df, "log_loss", "pd", repetitions=20)
    assert set(out.control) == {full} and set(out.configuration) == set(names)
    assert sorted(out.mean_difference) == pytest.approx([.1, .2])
    assert set(out.datasets) == {3}
    broken = df[~df.info_run_name.eq(names[0] + "__s2")]
    with pytest.raises(ValueError, match="Unmatched training seeds"):
        paired_ablation_effects(broken, "log_loss", "pd")
    with pytest.raises(ValueError, match="Missing comparison control"):
        paired_ablation_effects(df[~df.info_run_name.str.startswith(full)], "log_loss", "pd")


def test_reference_gap_uses_the_matching_architecture_without_seed_pseudoreplication():
    df = pd.DataFrame(cells("exp2_pd__arm=tabicl__prior-credit_fraction=0p5", .8)
                      + cells("exp2_pd__arm=tabpfn3__prior-credit_fraction=0p5", .6)
                      + cells("tabiclv2", .9) + cells("tabpfn3", .4))
    out = paired_reference_effects(df, "roc_auc", repetitions=20).set_index("control")
    assert out.loc["tabiclv2", "mean_difference"] == pytest.approx(-.1)
    assert out.loc["tabpfn3", "mean_difference"] == pytest.approx(.2)
    assert set(out.datasets) == {3}, "folds and training seeds must not increase sample size"


def test_final_ood_loader_reads_final_native_protocol_files_and_requires_all_models(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.utils.config import expand_with_seeds, load, run_name
    from src.eval.benchmark_status import REFERENCE_MODELS
    from src.eval import ood
    from src.utils import paths

    refs, results = tmp_path / "references", tmp_path / "results"
    refs.mkdir(); results.mkdir()
    monkeypatch.setattr(paths, "benchmark_dir", lambda exp, task: refs if exp == paths.REFERENCE else results)
    monkeypatch.setattr(ood, "list_ood_datasets", lambda kind: [SimpleNamespace(name=d) for d in ("a", "b", "c")])
    cfg = load("config/Exp1_PD.yaml")
    for i, run in enumerate(expand_with_seeds(cfg)):
        name = run_name(run)
        frame = pd.DataFrame(cells(name.rsplit("__s", 1)[0], .7, seeds=(run["seed"],)))
        frame.to_csv(results / f"ood_results_exp1bench_pd_a{i}.csv", index=False)
    for ref in REFERENCE_MODELS.split(","):
        pd.DataFrame(cells(ref, .8)).to_csv(refs / f"ood_results_reference_pd_{ref}.csv", index=False)
    # A malformed monitoring file must not enter the final-checkpoint analysis.
    (results / "ood_results_exp1bench_pd_a0_s100.csv").write_text("unrelated monitor")
    result = analysis.load_ood_results("pd")
    assert result is not None and len(result) == (12 + 4) * 3 * 5
    (results / "ood_results_exp1bench_pd_a8.csv").unlink()
    with pytest.warns(UserWarning, match="missing model"):
        assert analysis.load_ood_results("pd") is None


def test_incomplete_final_analysis_draws_placeholder_and_reports_missing_results(monkeypatch):
    from src.visualize import comparison_plots
    monkeypatch.setattr(analysis, "load_ood_results", lambda *a: None)
    assert "not yet completely benchmarked" in analysis.summary("pd")
    fig = comparison_plots.paired_effects("pd")
    assert "Complete final" in fig.axes[0].texts[0].get_text()


def test_point_reference_without_crps_does_not_block_distributional_comparisons():
    from src.eval.comparisons import paired_prior_effects
    df = pd.DataFrame(cells("exp1_lgd__prior-credit_fraction=0", .7)
                      + cells("exp1_lgd__prior-credit_fraction=0p5", .8)
                      + cells("linear", .4))
    df["crps"] = df.log_loss
    df.loc[df.model.eq("linear"), "crps"] = float("nan")
    result = paired_prior_effects(df, "crps", repetitions=20)
    assert result.iloc[0].mean_difference == pytest.approx(-.1)
    df.loc[df.info_run_name.str.contains("0p5"), "crps"] = float("nan")
    with pytest.raises(ValueError, match="Finite scores"):
        paired_prior_effects(df, "crps", repetitions=20)
