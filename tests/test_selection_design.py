"""Protect the distinction between fixed Exp2 comparisons and Exp3 selection."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.eval import selection
from src.eval.protocol import N_FOLDS, PROTOCOL
from src.utils.config import expand_with_seeds, load, run_name


def _development_cells(track="pd", scores=None):
    cfg = load(selection.ROOT / f"config/Exp1_{track.upper()}.yaml")
    scores = scores or {0.0: .8, .25: .65, .5: .7, 1.0: .6}
    metric = "roc_auc" if track == "pd" else "r2"
    return pd.DataFrame([
        {"dataset": dataset, "model": "crediticl", "info_run_name": run_name(run),
         "seed": 0, "fold": fold, "status": "ok", "protocol": PROTOCOL,
         metric: scores[run["prior"]["credit_fraction"]]}
        for run in expand_with_seeds(cfg)
        for dataset in cfg["eval"]["dev_datasets"] for fold in range(N_FOLDS)
    ])


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_exp3_candidate_remains_nonzero_when_control_wins(track):
    ranking = selection.development_ranking(_development_cells(track), track)
    choice = selection.exp3_candidate(ranking, track)
    assert choice["credit_fraction"] == .5
    assert choice["difference_from_control"] == pytest.approx(-.1)
    assert ranking.iloc[0].configuration.endswith("credit_fraction=0")


def test_exp3_exact_tie_has_deterministic_rule_not_seed_spread_rule():
    cells = _development_cells(scores={0.: .6, .25: .65, .5: .7, 1.: .7})
    ranking = selection.development_ranking(cells, "pd")
    ranking.loc[ranking.configuration.str.endswith("=1"), "std"] = 100.
    assert selection.exp3_candidate(ranking.iloc[::-1], "pd")["credit_fraction"] == .5


def test_quarter_mixture_can_win_only_after_its_development_cells_are_complete():
    cells = _development_cells(scores={0.: .6, .25: .8, .5: .7, 1.: .7})
    choice = selection.exp3_candidate(selection.development_ranking(cells, "pd"), "pd")
    assert choice["credit_fraction"] == .25
    incomplete = cells[~cells.info_run_name.str.contains("credit_fraction=0p25")]
    assert selection.exp3_candidate(selection.development_ranking(incomplete, "pd"), "pd") is None


def test_exp3_choice_ignores_holdout_ood_reference_and_historical_scores():
    cells = _development_cells()
    extra = cells.iloc[:5].copy()
    extra["dataset"] = "holdout_or_ood"
    extra["roc_auc"] = 1.
    ranking = selection.development_ranking(pd.concat([cells, extra], ignore_index=True), "pd")
    original = selection.exp3_candidate(ranking, "pd")
    misleading = ranking.iloc[[0]].copy()
    misleading["configuration"] = "released_reference"
    misleading["mean"] = 1.
    historical = misleading.assign(configuration="exp1_pd__old-grid__prior-credit_fraction=1")
    assert selection.exp3_candidate(pd.concat([ranking, misleading, historical]), "pd") == original


@pytest.mark.parametrize("mutation", ["missing_fold", "missing_control", "wrong_metric", "duplicate"])
def test_exp3_choice_refuses_incomplete_or_incompatible_rankings(mutation):
    cells = _development_cells()
    if mutation == "missing_fold":
        cells = cells.iloc[1:]
    ranking = selection.development_ranking(cells, "pd")
    if mutation == "missing_control":
        ranking = ranking[~ranking.configuration.str.endswith("credit_fraction=0")]
    elif mutation == "wrong_metric":
        ranking["metric"] = "log_loss"
    elif mutation == "duplicate":
        ranking = pd.concat([ranking, ranking.iloc[[0]]])
    assert selection.exp3_candidate(ranking, "pd") is None


def _allow_current_receipts(monkeypatch):
    from src.eval import benchmark_status
    monkeypatch.setattr(benchmark_status, "n_slots", lambda *args: 1)
    monkeypatch.setattr(benchmark_status, "slot_for",
                        lambda *args: SimpleNamespace(kind="reference", final=True))
    monkeypatch.setattr(benchmark_status, "complete", lambda *args: True)


def test_cli_filters_historical_and_newer_unrecognised_protocols(tmp_path, monkeypatch, capsys):
    from src.eval import results
    _allow_current_receipts(monkeypatch)
    current = _development_cells()
    other = _development_cells(scores={0.: .1, .25: .15, .5: .2, 1.: .99})
    mixed = pd.concat([current, other.assign(protocol=PROTOCOL - 1),
                       other.assign(protocol=PROTOCOL + 1)], ignore_index=True)
    file = tmp_path / "results.csv"
    mixed.to_csv(file, index=False)

    def load_file(track, *, protocol):
        assert track == "pd" and protocol == PROTOCOL
        return results._read_frames([file], protocol)

    monkeypatch.setattr(results, "load_results", load_file)
    assert selection.main(["--track", "pd"]) == 0
    text = capsys.readouterr().out
    assert "credit_fraction=0.5;" in text
    assert "No credit mixture beats the control" in text
    assert "must not re-rank" in text and "Exp2 uses all four" in text


def test_cli_refuses_no_current_ranking_even_with_receipts(monkeypatch, capsys):
    from src.eval import results
    _allow_current_receipts(monkeypatch)
    monkeypatch.setattr(results, "load_results", lambda *args, **kwargs: None)
    assert selection.main(["--track", "pd"]) == 1
    assert "Selection blocked" in capsys.readouterr().out


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_exp2_grid_does_not_depend_on_exp1_winner(track):
    from src.utils.run_experiment import unconfigured_tracks
    cfg = load(selection.ROOT / f"config/Exp2_{track.upper()}.yaml")
    runs = expand_with_seeds(cfg)
    assert unconfigured_tracks(2, [track]) == []
    assert len(runs) == 24
    for architecture in ("tabicl", "tabpfn3"):
        group = [r for r in runs if r["architecture"] == architecture]
        assert len(group) == 12
        assert {(r["prior"]["credit_fraction"], r["seed"]) for r in group} == {
            (fraction, seed) for fraction in (0., .25, .5, 1.) for seed in (0, 1, 2)}
        assert all(r["init"]["strategy"] == "full" and r["init"]["pretrained_path"] for r in group)
