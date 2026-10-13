"""Publication summaries must use matched complete cells, not lucky test seeds."""
import numpy as np
import pandas as pd
import pytest

from src.eval.comparisons import holm_adjust, paired_prior_effects, require_balanced_cells


def scores():
    rows = []
    for dataset, delta in zip(("a", "b", "c", "d"), (.01, .03, -.02, .02)):
        for share in ("0", "0p5", "1"):
            for seed in (0, 1, 2):
                for fold in range(5):
                    rows.append({"dataset": dataset, "model": "crediticl", "seed": 0,
                                 "fold": fold, "status": "ok", "roc_auc": .7 + seed * .001
                                 + (delta if share != "0" else 0), "protocol": 6,
                                 "info_run_name": f"exp1_pd__prior-credit_fraction={share}__s{seed}"})
    return pd.DataFrame(rows)


def test_dataset_is_the_uncertainty_unit_and_seed_means_are_used():
    result = paired_prior_effects(scores(), "roc_auc")
    assert len(result) == 2 and set(result.datasets) == {4}
    np.testing.assert_allclose(result.mean_difference, .01)
    assert (result.ci_low < result.mean_difference).all()
    assert (result.ci_high > result.mean_difference).all()
    assert (result.p_holm >= result.p_value).all()
    pd.testing.assert_frame_equal(result, paired_prior_effects(scores(), "roc_auc"))


@pytest.mark.parametrize("damage", ["missing_fold", "duplicate", "failure", "missing_dataset"])
def test_incomplete_pairs_never_produce_a_success_only_average(damage):
    df = scores()
    if damage == "missing_fold":
        df = df.iloc[1:]
    elif damage == "duplicate":
        df = pd.concat([df, df.iloc[:1]])
    elif damage == "failure":
        df.loc[0, "status"] = "failed"
    else:
        df = df[df.dataset != "a"]
    with pytest.raises(ValueError):
        require_balanced_cells(df, ["a", "b", "c", "d"])


def test_missing_training_seed_is_not_a_paired_comparison():
    df = scores()
    df = df[~df.info_run_name.eq("exp1_pd__prior-credit_fraction=0p5__s2")]
    with pytest.raises(ValueError, match="training seeds"):
        paired_prior_effects(df, "roc_auc")


def test_holm_adjustment_restores_original_order():
    np.testing.assert_allclose(holm_adjust([.03, .01, .04]), [.06, .03, .06])
    assert holm_adjust([]) == []


def test_per_dataset_plot_does_not_choose_the_best_training_seed():
    from src.visualize.results_plots import _best_per_group
    df = scores()
    df.loc[df.info_run_name.str.endswith("__s2"), "roc_auc"] += .2
    matrix, _ = _best_per_group(df, "roc_auc")
    expected = df[df.dataset.eq("a") & df.info_run_name.str.contains("credit_fraction=0p5")].roc_auc.mean()
    assert matrix.loc["a", "50 % credit"] == pytest.approx(expected)


def test_failure_in_old_protocol_does_not_contaminate_current_report(tmp_path):
    from src.visualize.results_plots import _read_frames
    df = scores()
    old = df.iloc[:1].copy().assign(protocol=5, status="failed")
    file = tmp_path / "results.csv"
    pd.concat([df, old]).to_csv(file, index=False)
    assert len(_read_frames([file], protocol=6)) == len(df)
