"""Behavioural regressions for the independently verified 21 October audit."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("dataset,names", [
    ("0005.base_modelisation", ["an", "EAD_BALOISE_REELLE", "Perim_poids", "ponderation_11"]),
    ("0002.loss2", ["_ELAO", "_Accrued_int"]),
])
def test_known_outcome_aliases_and_categorical_copies_cannot_bypass_policy(dataset, names):
    from src.eval.feature_policy import prediction_features

    columns = ["at_default", *names, *[name + "__cat" for name in names], "safe_category"]
    X = np.arange(7 * len(columns)).reshape(7, -1)
    view, cats, report = prediction_features("lgd", dataset, X, list(range(1, len(columns))), columns)
    np.testing.assert_array_equal(view, X[:, [0, -1]])
    assert cats == [1] and report["n_features_excluded"] == 2 * len(names)


@pytest.mark.parametrize("track,classes", [("pd", 3), ("lgd", None)])
def test_control_tensors_equal_direct_upstream_graph_scm(track, classes):
    from tabicl.prior._graph_scm import GraphSCM
    from src.prior import upstream
    from src.prior.generator import TaskGenerator
    from src.prior.rng import PriorRNG
    from src.prior.stream import SlotPlan

    plan = SlotPlan(batch=0, slot=0, group=0, seq_len=128, train_size=72,
                    num_features=4, num_classes=classes, use_credit=False)
    generator = TaskGenerator({"credit_fraction": 0., "max_features": 12,
                               "filter": {"mode": "off"}}, track, PriorRNG(0))
    actual = generator._base_candidate(plan, PriorRNG(813), 813, {})
    with upstream.seeded(813):
        expected = GraphSCM(regression=track == "lgd", seq_len=128, num_features=4,
                            max_features=12, num_classes=classes or 2,
                            config=upstream.prior_config(), device="cpu")()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def test_covariate_dependent_missingness_is_not_null_by_construction():
    from src.prior.targets.pd import apply_informative_missingness
    from src.prior.rng import PriorRNG

    X = torch.linspace(-3, 3, 8192).reshape(-1, 1)
    cfg = dict(missing_col_fraction=1., missing_rate_range=[.5, .5],
               missing_risk_coupling=6., missing_indicators=False)
    coupled, _ = apply_informative_missingness(PriorRNG(14), X, cfg, 1, fill="nan")
    mcar, _ = apply_informative_missingness(PriorRNG(14), X,
                                          {**cfg, "missing_risk_coupling": 0.}, 1, fill="nan")
    # Dependence on X is the implemented MAR contract. It promises no particular
    # AUC for a target generated from a different, random direction in X.
    difference = lambda A: abs(float(A[:2048].isnan().float().mean()
                                     - A[-2048:].isnan().float().mean()))
    assert difference(coupled) > .15
    assert difference(mcar) < .05


def test_macro_target_ablation_does_not_change_macro_feature_coin():
    from src.prior.targets.lgd import apply_lgd_zoib
    from src.prior.rng import PriorRNG

    X = torch.randn(96, 3, generator=torch.Generator().manual_seed(10))
    latent = X[:, 0]
    cfg = {"cohort": {"min_cohorts": 3, "max_cohorts": 3}, "macro_feature_prob": .5}
    no_macro = {**cfg, **dict.fromkeys(("macro_p0_range", "macro_p1_range", "macro_mu_range"), [0., 0.])}
    decisions = []
    for seed in range(32):
        a, _, ma = apply_lgd_zoib(PriorRNG(seed), X, latent, cfg, 4)
        b, _, mb = apply_lgd_zoib(PriorRNG(seed), X, latent, no_macro, 4)
        assert ma.get("macro_feature", False) == mb.get("macro_feature", False)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        decisions.append(a.shape[1] == 4)
    assert any(decisions) and not all(decisions), "exercise both sides of the feature coin"


def test_classification_measurement_does_not_average_multiclass_label_codes():
    from src.prior.measure import classification_rates

    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    assert classification_rates(labels) == (None, .2, 3)
    assert classification_rates(np.choose(labels, [2, 0, 1])) == (None, .2, 3)
    assert classification_rates([0, 0, 0, 1]) == (.25, .25, 2)
    assert classification_rates([1, 1, 1, 0]) == (.75, .25, 2)


def test_data_plot_does_not_crop_or_mislabel_a_repayment_oriented_target():
    import matplotlib.pyplot as plt
    from src.visualize import data_plots
    data = {"0006.hackerearth": SimpleNamespace(y=np.array([1] * 7 + [0] * 3), n_rows=10)}
    fig = data_plots.plot_pd_base_rates(data)
    assert fig.axes[0].get_xlim()[1] >= .7
    assert "positive-label" in fig.axes[0].get_xlabel()
    plt.close(fig)


@pytest.mark.parametrize("metric", ["ks", "mcc_tuned", "roc_auc"])
def test_statistical_and_plot_directions_agree_for_rewards(metric):
    from src.eval.metrics import higher_is_better
    from src.visualize import results_plots, training_plots
    assert higher_is_better(metric)
    assert results_plots.HIGHER_IS_BETTER[metric] and training_plots.HIGHER_IS_BETTER[metric]


def test_errors_are_minimized_but_calibration_slope_has_a_target():
    from src.eval.metrics import higher_is_better
    from src.visualize import style
    assert not higher_is_better("boundary_mass_abs_err")
    assert style.metric_goal("calibration_slope") == 1.
    with pytest.raises(ValueError, match="no declared monotone"):
        higher_is_better("calibration_slope")


@pytest.mark.parametrize("truth", [.0, .25, .5, 1.])
def test_crps_matches_uniform_distribution_closed_form(truth):
    from src.eval.metrics import crps_from_quantiles
    levels = (np.arange(10_000) + .5) / 10_000
    score = crps_from_quantiles(np.array([truth]), levels.reshape(1, -1), levels)
    assert score == pytest.approx(truth ** 2 - truth + 1 / 3, abs=1e-8)


def test_ranking_draws_all_architecture_mixture_and_reference_markers(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from src.eval import selection
    from src.visualize import results_plots as rp

    names = [f"exp2_pd__arm={a}__prior-credit_fraction={c}"
             for a in ("tabicl", "tabpfn3") for c in ("0", "0p5", "1")]
    names += ["tabiclv2", "tabpfn3", "catboost", "linear"]
    ranking = pd.DataFrame({"configuration": names, "mean": np.linspace(.5, .8, 10), "std": .01})
    monkeypatch.setattr(rp, "load_results", lambda *a: pd.DataFrame({"roc_auc": [.7]}))
    monkeypatch.setattr(selection, "development_ranking", lambda *a: ranking)
    fig = rp.overall_ranking("pd", "exp2")
    assert sum(len(p.get_offsets()) for p in fig.axes[0].collections if isinstance(p, PathCollection)) == 10
    plt.close(fig)


def test_checkpoint_reference_line_excludes_continued_tabpfn_arms(monkeypatch):
    import matplotlib.pyplot as plt
    from src.visualize import results_plots as rp

    configuration = "exp2_pd__arm=tabpfn3__prior-credit_fraction=0p5"
    curve = pd.DataFrame({"configuration": [configuration] * 2, "share": [.5, .5],
                          "checkpoint_step": [10, 20], "mean": [.5, .6], "low": [.5, .6], "high": [.5, .6]})
    results = pd.DataFrame({"dataset": ["d"] * 3, "model": ["tabpfn3", "tabpfn3", "tabiclv2"],
                            "info_run_name": [configuration + "__s0", "", ""], "roc_auc": [.6, .8, .9]})
    monkeypatch.setattr(rp, "_checkpoint_means", lambda *a: curve)
    monkeypatch.setattr(rp, "load_results", lambda *a: results)
    monkeypatch.setattr(rp, "_restrict", lambda df, *a: df)
    fig = rp.checkpoint_curve("pd", "exp2")
    horizontal = [float(line.get_ydata()[0]) for line in fig.axes[0].lines
                  if len(set(line.get_ydata())) == 1]
    assert sorted(horizontal) == [.8, .9]
    plt.close(fig)


def test_training_comparison_keeps_architectures_and_mixtures_separate(monkeypatch):
    import matplotlib.pyplot as plt
    from src.visualize import training_plots as tp

    curves = {}
    for family, base in (("tabicl", .2), ("tabpfn3", .8)):
        for share in (0, .5, 1):
            for seed in (0, 1):
                name = f"exp2_pd__arm={family}__prior-credit_fraction={str(share).replace('.', 'p')}__s{seed}"
                curves[name] = pd.DataFrame({"step": [0, 1, 2],
                    "real__dev__roc_auc": np.full(3, base + share * .01)})
    monkeypatch.setattr(tp, "_view", lambda *a: (curves, curves, ["dev"], []))
    fig = tp.credit_vs_control_over_training("pd", "exp2")
    differences = [line for line in fig.axes[0].lines if "paired seeds" in line.get_label()]
    assert len(differences) == 4
    np.testing.assert_allclose(sorted(line.get_ydata()[0] for line in differences),
                               [.005, .005, .01, .01])
    assert all("2 paired seeds" in line.get_label() for line in differences)
    assert list(fig.axes[0].lines[-1].get_ydata()) == [0, 0]
    plt.close(fig)


def test_portable_junit_accepts_only_named_expected_skips(tmp_path):
    from src.validation.campaign import check_junit
    path = tmp_path / "junit.xml"
    def xml(module, name, reason):
        return (f'<testsuite tests="2" failures="0" errors="0" skipped="1">'
                f'<testcase classname="{module}" name="{name}"><skipped message="{reason}"/></testcase>'
                '<testcase classname="tests.test_anything" name="passed"/></testsuite>')
    path.write_text(xml("tests.test_data_identity", "test_every_dataset_matches_tabpfncredit[NOTSET]",
                        "the TabPFNCredit project is not on this machine"))
    assert check_junit(path, mandatory=False)["skipped"] == 1
    path.write_text(xml("tests.test_equivalence", "test_recipe", "required dependency unavailable"))
    with pytest.raises(ValueError, match="unexpected skipped"):
        check_junit(path, mandatory=False)


def test_stop_stage_waits_for_current_child_signal_handlers_even_with_old_checkpoint(tmp_path, monkeypatch):
    from src.validation import campaign as v
    from src.train import checkpoint
    from src.utils import working_storage
    # Exercise the Unix parent/child handshake on Windows as well; no real
    # signal is sent by this fake process.
    monkeypatch.setattr(v.signal, "SIGUSR1", 10, raising=False)

    saved = tmp_path / "step-100.ckpt"
    saved.touch()
    log = tmp_path / "stop.log"
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(dict(completed=False, stopped_by_signal="SIGUSR1", steps=101)))
    sent = []
    proc = SimpleNamespace(pid=12345, returncode=None)
    proc.poll = lambda: proc.returncode
    def send(sig):
        sent.append(sig)
        proc.returncode = 64
    proc.send_signal = send
    monkeypatch.setattr(v.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(v, "kill_process_tree", lambda *a: None)
    monkeypatch.setattr(checkpoint, "latest_checkpoint", lambda *a: saved)
    monkeypatch.setattr(working_storage, "checkpoint_work_dir", lambda p: p)
    monkeypatch.setattr(v.paths, "run_summary_path", lambda *a: summary)
    def first_tick(seconds):
        if not log.with_suffix(".ready").exists():
            assert not sent, "old checkpoint is not proof the new child installed handlers"
            log.with_suffix(".ready").write_text(str(proc.pid))
    monkeypatch.setattr(v.time, "sleep", first_tick)
    result = v.execute(v.Stage("stop", ("unused",), phase="training", stop_run="exp0_pd__s0"), log, timeout=300)
    assert result["stopped_at"] == 101 and len(sent) == 1
    assert not log.with_suffix(".ready").exists()


def test_signal_readiness_belongs_to_the_current_process(tmp_path, monkeypatch):
    from src.utils import signal_ready
    path = tmp_path / "ready"
    monkeypatch.setenv(signal_ready.READY_ENV, str(path))
    assert not signal_ready.matches(path, os.getpid())
    signal_ready.publish()
    assert signal_ready.matches(path, os.getpid())
    assert not signal_ready.matches(path, os.getpid() + 1)


@pytest.mark.parametrize("start,end", [
    ("SLOT_DESCRIPTION=$(", "# A capped-context"),
    ("RUN_DESCRIPTION=$(", 'echo "arm ${ARM}'),
])
def test_benchmark_descriptor_failure_is_a_nonzero_job_exit(start, end):
    import shutil
    import subprocess
    from pathlib import Path

    bash = shutil.which("bash")
    assert bash, "Experiment 0 needs Bash to validate its actual Slurm error propagation"
    source = Path("scripts/slurm/benchmark.slurm").read_text(encoding="utf-8")
    segment = source[source.index(start):source.index(end, source.index(start))]
    code = ('set -euo pipefail\npython() { return 17; }\n'
            'STATUS_ARGS=()\nCONFIG=unused\nARM=0\n' + segment + '\necho INCORRECT_SUCCESS\n')
    result = subprocess.run([bash, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 17 and "INCORRECT_SUCCESS" not in result.stdout


def test_fresh_process_determinism_rejects_a_child_that_printed_then_failed(monkeypatch):
    from src.validation import checks
    from src.prior import generator, stream

    monkeypatch.setattr(checks, "_batch_digest", lambda *a, **k: "same-digest")
    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="same-digest\n", stderr="crash after printing", returncode=1))
    monkeypatch.setattr(generator, "TaskGenerator", lambda *a, **k: None)
    monkeypatch.setattr(stream, "plan_batch", lambda *a, **k: SimpleNamespace(slots=[]))
    report = checks.Report()
    checks.check_determinism(report)
    rows = [r for r in report.rows if "a fresh one" in r["name"]]
    assert len(rows) == 2 and all(r["status"] == checks.FAIL for r in rows)


def test_current_notebook_descriptions_do_not_reintroduce_retired_protocol():
    from pathlib import Path
    markdown = []
    for path in Path("notebooks").rglob("*.ipynb"):
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] == "markdown":
                markdown.append("".join(cell["source"]))
    text = "\n".join(markdown)
    for obsolete in ("500 highest-variance", "Evaluation protocol 3", "untouched holdout",
                     "FILL_FROM_SEARCH", "search chose on development", "gap narrower than the bands"):
        assert obsolete not in text
