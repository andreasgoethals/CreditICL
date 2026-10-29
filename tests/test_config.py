"""Config loading and grid expansion.

These tests exist because this code already shipped one real bug: a hand-curated
list of "do not sweep" keys leaked `n_nodes_range` and `rule_quantile_range` into
the grid, turning two sampling intervals into two-point sweeps and inflating the
PD grid to 6,144 runs. The `_range` suffix rule replaced it, and
`test_range_keys_are_never_swept` is the regression test.

Deliberately no torch import here, so these run even on a bare environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.design import MAIN_STEPS  # noqa: E402
from src.utils.config import (  # noqa: E402
    PLACEHOLDER,
    expand_grid,
    expand_with_seeds,
    find_placeholders,
    is_literal_list,
    load,
    load_yaml,
    sweep_axes,
)

#: One file per experiment per track. Exp0 is the cluster debug suite; Exp2 runs at each model's
#: backbone-specific continuation recipes with two rates; Exp3 uses fixed method settings.
EXPERIMENTS = [f"config/{e}_{t}.yaml" for e in ("Exp0", "Exp1", "Exp2", "Exp3") for t in ("LGD", "PD")]
#: The runnable ones: nothing left to fill in.
CONFIGS = list(EXPERIMENTS)


def _load(path):
    """Templates included, so structural checks cover all six files."""
    return load(ROOT / path, allow_placeholders=True)


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_config_loads(path):
    cfg = _load(path)
    # No "model": the architecture is fixed in NanoTabICLv2's defaults, not configured.
    for key in ("experiment", "task", "seeds", "prior", "train", "init"):
        assert key in cfg, f"{path} is missing the '{key}' block"


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_task_matches_filename(path):
    cfg = _load(path)
    expected = "lgd" if "LGD" in path else "pd"
    assert cfg["task"] == expected
    exp, track = cfg["experiment"].split("_")[:2]
    assert track == expected
    assert exp == Path(path).stem.split("_")[0].lower()


# -- the six-file layout, and prior_file: ------------------------------------



# -- placeholders --------------------------------------------------------------


@pytest.mark.parametrize("path", CONFIGS)
def test_configs_contain_no_unfilled_recipe_placeholders(path):
    """Predeclared recipes need no selection receipts or manual editing."""
    cfg = load(ROOT / path, allow_placeholders=True)
    assert find_placeholders(cfg) == []


@pytest.mark.parametrize("path", [p for p in CONFIGS if "Exp3" in p])
def test_exp3_is_resolved_without_any_scores_or_selection_receipt(path):
    runs = expand_with_seeds(load(ROOT / path))
    assert runs and all(not r["design_unresolved"] for r in runs)


def test_exp2_refuses_a_silent_partial_load():
    """A name mismatch that loads nothing still runs and still outputs numbers — they are
    just partly random. `strict_load` is what turns that into a crash, and the trainer reads it."""
    for name in ("Exp2_LGD", "Exp2_PD"):
        cfg = _load(f"config/{name}.yaml")
        assert cfg["init"]["strict_load"] is True
        assert cfg["init"]["pretrained_path"], "a warm start needs a checkpoint"
    loop = (ROOT / "src" / "train" / "loop.py").read_text(encoding="utf-8")
    assert 'strict=bool(icfg.get("strict_load", False))' in loop


# -- the screening budget ------------------------------------------------------


def test_exp1_budget_is_preserved_and_exp3_is_continuation():
    for track in ("LGD", "PD"):
        assert _load(f"config/Exp1_{track}.yaml")["train"]["max_steps"] == 12500
        cfg = _load(f"config/Exp3_{track}.yaml")
        assert cfg["train"]["max_steps"] == MAIN_STEPS
        assert cfg["init"]["strategy"] == "full"


def test_exp3_has_two_paired_seeds_and_five_kept_milestones():
    from src.experiments.design import MILESTONES
    for track in ("LGD", "PD"):
        cfg = _load(f"config/Exp3_{track}.yaml")
        assert cfg["seeds"] == [0, 1]
        assert cfg["train"]["max_steps"] == MAIN_STEPS
        assert tuple(cfg["train"]["keep_steps"]) == MILESTONES


@pytest.mark.parametrize("path", [p for p in EXPERIMENTS if "_search" not in p])
def test_dev_and_holdout_are_disjoint_and_non_empty(path):
    """A dataset in both splits leaks the holdout into prior selection, which is the one
    mistake that cannot be fixed after the fact."""
    ev = _load(path)["eval"]
    dev, hold = set(ev["dev_datasets"]), set(ev["holdout_datasets"])
    assert dev and hold, f"{path}: both splits must be populated before any run"
    assert not (dev & hold), f"{path}: {dev & hold} is in both splits"
    assert ev["select_on"] == "dev", "selecting on holdout invalidates the experiment"


@pytest.mark.parametrize("path", [p for p in EXPERIMENTS if "_search" not in p])
def test_the_split_covers_every_dataset_on_disk(path):
    """A dataset in neither split is silently never evaluated."""
    from src.data.discovery import list_datasets

    cfg = _load(path)
    available = set(list_datasets(cfg["task"]))
    if not available:
        pytest.skip("datasets not present on this machine")
    named = set(cfg["eval"]["dev_datasets"]) | set(cfg["eval"]["holdout_datasets"])
    assert named == available, f"{path}: unassigned {available - named}, unknown {named - available}"


def test_single_value_is_not_swept():
    cfg = {"a": 1, "b": "x", "c": {"d": True}}
    assert sweep_axes(cfg) == []
    assert len(expand_grid(cfg)) == 1


def test_list_becomes_a_sweep():
    cfg = {"a": [1, 2, 3]}
    assert len(expand_grid(cfg)) == 3


def test_lists_are_crossed():
    """Five settings with two values each must give 32 runs, as the docs claim."""
    cfg = {f"k{i}": [0, 1] for i in range(5)}
    assert len(expand_grid(cfg)) == 32


def test_nested_paths_are_addressed_by_dots():
    cfg = {"prior": {"credit": {"x": [1, 2]}}}
    axes = sweep_axes(cfg)
    assert axes == [("prior.credit.x", [1, 2])]
    runs = expand_grid(cfg)
    assert [r["prior"]["credit"]["x"] for r in runs] == [1, 2]


def test_range_keys_are_never_swept():
    """Regression test for the bug that inflated the PD grid to 6,144 runs."""
    cfg = {
        "n_rows_range": [512, 1024],
        "rule_quantile_range": [0.1, 0.9],
        "some_range": [1, 2],
        "quantile_band": [0.0, 0.5],
        "real_sweep": [1, 2],
    }
    axes = dict(sweep_axes(cfg))
    assert "real_sweep" in axes
    for literal in ("n_rows_range", "rule_quantile_range", "some_range", "quantile_band"):
        assert literal not in axes, f"{literal} must be literal data, not a sweep"
    assert len(expand_grid(cfg)) == 2


def test_is_literal_list_uses_a_suffix_rule():
    assert is_literal_list("anything_range")
    assert is_literal_list("quantile_band")
    assert is_literal_list("seeds")
    assert not is_literal_list("credit_fraction")


def test_nested_range_can_be_swept():
    """The documented escape hatch: a list of ranges IS a sweep."""
    cfg = {"boundary_mass_range": [[0.0, 0.1], [0.1, 0.3]]}
    runs = expand_grid(cfg)
    assert len(runs) == 2
    assert runs[0]["boundary_mass_range"] == [0.0, 0.1]


def test_empty_sweep_list_is_an_error():
    with pytest.raises(ValueError):
        expand_grid({"a": []})


@pytest.mark.parametrize("path", CONFIGS)
def test_grid_is_deterministic(path):
    """A resubmitted array task must land on the same config. Non-negotiable."""
    cfg = _load(path)
    a = [r["_run_name"] for r in expand_with_seeds(cfg)]
    b = [r["_run_name"] for r in expand_with_seeds(_load(path))]
    assert a == b


@pytest.mark.parametrize("path", CONFIGS)
def test_run_names_are_unique(path):
    """Two runs sharing a name would overwrite each other's checkpoints."""
    names = [r["_run_name"] for r in expand_with_seeds(_load(path))]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("path", CONFIGS)
def test_seeds_are_crossed_outermost_before_campaign_extensions(path):
    """New campaigns cover levers before seeds; explicit append-only extensions
    preserve completed indices (covered by test_exp1_quarter_extension)."""
    cfg = _load(path)
    cfg.pop("array_append", None)
    runs = expand_with_seeds(cfg)
    n_seeds = len(cfg["seeds"])
    per_seed = len(runs) // n_seeds
    assert {r["seed"] for r in runs[:per_seed]} == {runs[0]["seed"]}


@pytest.mark.parametrize("path", CONFIGS)
def test_grid_stays_a_sane_size(path):
    """A guard against re-committing an accidental combinatorial explosion."""
    runs = expand_with_seeds(_load(path))
    assert len(runs) <= 200, (
        f"{path} expands to {len(runs)} runs. That is almost certainly an accident — "
        "open one lever group at a time."
    )


@pytest.mark.parametrize("path", CONFIGS)
def test_credit_fraction_is_a_probability(path):
    cfg = _load(path)
    values = cfg["prior"]["credit_fraction"]
    for v in values if isinstance(values, list) else [values]:
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("path", [p for p in CONFIGS if "Exp0" not in p and "Exp3" not in p])
def test_control_arm_is_present(path):
    """credit_fraction = 0 is the baseline everything is measured against. If it
    is missing there is nothing to compare to."""
    values = _load(path)["prior"]["credit_fraction"]
    values = values if isinstance(values, list) else [values]
    assert 0.0 in values, "the control arm (credit_fraction 0.0) must be in the sweep"


@pytest.mark.parametrize("path", CONFIGS)
def test_init_strategy_is_known(path):
    cfg = _load(path)
    strategies = cfg["init"]["strategy"]
    for s in strategies if isinstance(strategies, list) else [strategies]:
        assert s in ("scratch", "full", "icl_only", "head_only")


@pytest.mark.parametrize("path", CONFIGS)
def test_pretrained_path_required_when_not_scratch(path):
    """Catch the config mistake, not the crash six minutes into a queued job."""
    cfg = _load(path)
    strategies = cfg["init"]["strategy"]
    strategies = strategies if isinstance(strategies, list) else [strategies]
    if any(s != "scratch" for s in strategies):
        assert cfg["init"].get("pretrained_path"), (
            "init.strategy includes a non-scratch option but pretrained_path is null"
        )


@pytest.mark.parametrize("path", CONFIGS)
def test_no_model_block_in_the_configs(path):
    """The architecture is TabICLv2's, is identical for LGD and PD, and never varies — so
    it lives in NanoTabICLv2's defaults, not in a config block each file would have to
    repeat and keep in sync. See test_train.py for the check that those defaults match
    the paper's Table A.1."""
    assert "model" not in load_yaml(ROOT / path), (
        f"{path} has a model: block; the architecture is fixed in code"
    )




# -- each experiment owns its own prior ---------------------------------------


def test_exp3_inherits_exp2_prior_but_fixes_the_mixture():
    for track in ("LGD", "PD"):
        raw = load_yaml(ROOT / f"config/Exp3_{track}.yaml")
        assert raw["prior_file"] == f"Exp2_{track}.yaml"
        cfg = _load(f"config/Exp3_{track}.yaml")
        assert cfg["prior"]["credit_fraction"] == .5
        assert cfg["prior"]["credit"] == _load(f"config/Exp2_{track}.yaml")["prior"]["credit"]


def test_exp1_sweeps_priors_and_never_the_architecture():
    """The claim the whole project rests on: the architecture is fixed and only the prior
    varies. An architecture knob appearing in a sweep would silently confound the two."""
    for track in ("LGD", "PD"):
        axes = dict(sweep_axes(_load(f"config/Exp1_{track}.yaml")))
        prior_axes = [k for k in axes if k.startswith("prior.")]
        # Since 28-09-2026 the one lever is how much of the calibrated credit prior is in the mix;
        # the control (0.0) must be one of its values.
        assert prior_axes == ["prior.credit_fraction"], f"{track}: Exp1 must sweep the mix, got {prior_axes}"
        assert 0.0 in axes["prior.credit_fraction"] and max(axes["prior.credit_fraction"]) > 0
        banned = ("model", "embed_dim", "num_blocks", "nhead", "architecture")
        for key in axes:
            assert not any(b in key for b in banned), f"{track}: {key} is an architecture knob"


def test_exp1_defines_many_priors_and_exp3_exactly_one():
    """The counts that make the two-phase design what it is."""
    for track in ("LGD", "PD"):
        axes = [(k, v) for k, v in sweep_axes(_load(f"config/Exp1_{track}.yaml")) if k != "seeds"]
        n_priors = 1
        for _, values in axes:
            n_priors *= len(values)
        # Four fractions; the 25% extension preserves the original three arms' ordering.
        assert n_priors == 4, f"{track}: expected 4 Exp1 priors, got {n_priors}"

        exp3 = [(k, v) for k, v in sweep_axes(_load(f"config/Exp3_{track}.yaml")) if k != "seeds"]
        assert dict(exp3) == {"arm": ["tabicl", "tabpfn3"],
                              "method": ["no_decay", "l2sp", "diagscale", "lora"]}
        assert _load(f"config/Exp3_{track}.yaml")["prior"]["credit_fraction"] == .5


def test_exp2_and_exp3_pin_every_knob_exp1_swept():
    """The bug this catches, which shipped once: the winner placeholders were never inserted, so
    Exp3 would have silently trained on whatever default the prior happened to carry — measuring
    an arm nobody chose."""
    for track in ("LGD", "PD"):
        swept = {k for k, _ in sweep_axes(_load(f"config/Exp1_{track}.yaml"))
                 if k not in ("seeds", "prior.credit_fraction")}
        for exp in ("Exp2", "Exp3"):
            holes = set(find_placeholders(_load(f"config/{exp}_{track}.yaml")))
            missing = swept - holes
            assert not missing, f"{exp}_{track} does not pin Exp1's choice for: {sorted(missing)}"


def test_the_control_arm_prior_never_carries_a_placeholder():
    """`prior.base` IS the unmodified TabICL prior and defines the control arm. A placeholder
    there would let Exp1's winner change what we measure against — which happened: a regex put
    PD's `category_frequency` under `base` instead of `credit`."""
    for path in EXPERIMENTS:
        base = _load(path)["prior"]["base"]
        assert PLACEHOLDER not in str(base), f"{path}: control-arm prior was modified: {base}"
        assert base["category_frequency"] == "balanced"
        # 200, NOT 100. `_graph_scm.py` calls
        # `sample_categorical_sizes(self.num_features, context, max_cat_size=200)`.
        # This test asserted 100 until 26-08-2026 and the configs were set to match the
        # test, so the control arm was NARROWER than TabICLv2 and the credit arm's 500 was
        # a smaller step than documented. A test pins a wrong number as firmly as a right
        # one; this one quotes the upstream call site instead of a remembered value.
        assert base["max_cat_size"] == 200, "upstream graph_scm's value"


# -- Exp2 is continued pre-training, not pretraining ---------------------------


def test_exp2_has_two_predeclared_rates_per_backbone():
    from src.experiments.design import RATES
    for track in ("LGD", "PD"):
        runs = expand_with_seeds(load(ROOT / f"config/Exp2_{track}.yaml", allow_placeholders=True))
        assert {r["architecture"] for r in runs} == {"tabicl", "tabpfn3"}
        assert {r["lr_index"] for r in runs} == {0, 1}
        for r in runs:
            t = r["train"]
            assert t["max_steps"] == MAIN_STEPS
            assert t["lr"] == RATES[r["architecture"]][r["lr_index"]]
            assert t["gradient_clipping"] == 1.0
            assert t["weight_decay"] == .01
            if r["architecture"] == "tabicl":
                assert (t["optimizer"], t["warmup_proportion"], t["batch_size"]) == ("muon", .01, 64)
            else:
                assert (t["optimizer"], t["warmup_proportion"], t["batch_size"]) == ("adamw", .1, 16)
                assert r["prior"]["encoding"] == "raw"


def test_exp2_sweeps_the_mixture_because_that_is_its_question():
    """How much of our prior to add IS the experiment. `1.0` must be included so 'forgetting the
    original prior' is measurable, `0.0` so 'continued pretraining alone' is the control — for
    both released models."""
    for track in ("LGD", "PD"):
        axes = dict(sweep_axes(_load(f"config/Exp2_{track}.yaml")))
        mixture = axes["prior.credit_fraction"]
        assert mixture == [0., .25, .5, 1.]
        assert 0.0 in mixture and 1.0 in mixture, f"{track}: mixture must span 0..1, got {mixture}"
        assert sorted(axes["arm"]) == ["tabicl", "tabpfn3"]


def test_exp3_matches_exp2_task_budget_within_backbone():
    for track in ("LGD", "PD"):
        runs2 = expand_with_seeds(_load(f"config/Exp2_{track}.yaml"))
        runs3 = expand_with_seeds(_load(f"config/Exp3_{track}.yaml"))
        for r in runs3:
            base = next(b for b in runs2 if b["architecture"] == r["architecture"])
            for key in ("max_steps", "batch_size", "micro_batch_size"):
                assert r["train"][key] == base["train"][key]


def test_every_experiment_names_the_same_architecture():
    """TabICLv2's own architecture everywhere; Experiment 2 alone adds TabPFN-3, and only through
    its arms (the top-level value is TabICL's in every file). Two experiments on different
    architectures cannot be compared, and the paper's claim is about the prior."""
    from src.utils.config import expand_with_seeds

    named = {_load(p)["architecture"] for p in EXPERIMENTS}
    assert named == {"tabicl"}, f"experiments disagree on the architecture: {named}"
    for p in CONFIGS:
        archs = {r["architecture"] for r in expand_with_seeds(_load(p))}
        allowed = {"tabicl", "tabpfn3"} if ("Exp2" in p or "Exp3" in p or "Exp0" in p) else {"tabicl"}
        assert archs <= allowed, f"{p}: {archs}"


def test_config_folder_holds_exactly_the_experiments():
    """Four experiments x two tracks, each self-contained. Anything else here is a file nobody
    named, which is how a stale config gets submitted."""
    found = sorted(p.name for p in (ROOT / "config").iterdir())
    expected = [Path(p).name for p in EXPERIMENTS] + ["Exp1_PD_ablations.yaml", "Exp1_LGD_ablations.yaml"]
    assert found == sorted(expected), f"unexpected: {found}"


def test_exp2_and_exp3_differ_only_where_they_should():
    """They are not identical — Exp2 is continued pre-training, so its budget, learning rate and
    mixture differ by design. What must still match is the EVALUATION SPLIT, or the two cannot be
    compared at all."""
    for track in ("LGD", "PD"):
        two, three = _load(f"config/Exp2_{track}.yaml"), _load(f"config/Exp3_{track}.yaml")
        for key in ("dev_datasets", "holdout_datasets", "select_on"):
            assert two["eval"][key] == three["eval"][key], f"{track}: eval.{key} must be identical"
        assert two["task"] == three["task"]
        assert three["init"]["strategy"] == "full"
        assert two["init"]["strategy"] != "scratch"
        assert two["init"]["pretrained_path"], "a warm start needs a checkpoint"


#: TabICLv2 stage 1, read off `scripts/train_v2_reg_stage1.sh` in the pinned tfm-library.
#: `sample_seq_len` opens with `if min_seq_len is None: return max_seq_len`, and stage 1 passes
#: only `--max_seq_len 1024` — so 1,024 rows exactly, not a range.
UPSTREAM_STAGE1 = {
    "n_rows_range": [1024, 1024],       # --max_seq_len 1024, no --min_seq_len
    "n_features_range": [1, 100],       # --min_features 1 --max_features 100
    "max_features": 100,                # --max_features 100
    "n_nodes_range": [2, 33],           # --min_n_nodes 2 --max_n_nodes 32, exclusive upper
    "train_frac_range": [0.3, 0.9],     # --min_train_size 0.3 --max_train_size 0.9
}


@pytest.mark.parametrize("path", EXPERIMENTS)
@pytest.mark.parametrize("key", sorted(UPSTREAM_STAGE1))
def test_prior_shape_matches_upstream_stage_one(path, key):
    """The prior's SHAPE is not ours to choose — only its CONTENT is.

    The whole experiment rests on one sentence: the only difference from TabICLv2 is the
    prior's credit structure. A narrower row or feature range is a second, unintended
    difference, and it silently makes every arm cheaper — which is exactly why it survived two
    rounds of "make everything the same". Both of these had drifted ([512, 1024] rows and
    [3, 50] features against upstream's 1,024 and [1, 100]) while `max_features: 100` padded
    the tensors to 100 regardless, so the top half of the feature axis was never trained.
    """
    assert _load(path)["prior"][key] == UPSTREAM_STAGE1[key], (
        f"{path}: prior.{key} must match TabICLv2 stage 1 "
        f"({UPSTREAM_STAGE1[key]}); see scripts/train_v2_reg_stage1.sh in tfm-library"
    )


@pytest.mark.parametrize("path", CONFIGS)
def test_micro_batch_cannot_exceed_the_group_size(path):
    """`Trainer.validate_micro_batch` RAISES when datasets in one micro-batch disagree on their
    sequence length or their train/test split, and both are drawn per group. So this is a hard
    upstream constraint, not a memory heuristic — a bigger GPU does not buy a bigger
    micro-batch."""
    cfg = _load(path)
    assert cfg["train"]["micro_batch_size"] <= cfg["prior"]["grouping"]["group_size"]
    assert cfg["train"]["batch_size"] % cfg["train"]["micro_batch_size"] == 0


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_a_swept_knob_has_exactly_one_home(path):
    """A swept path must NOT also carry a literal in the config body.

    `apply_sweep_block` writes the sweep list over whatever is below it, so such a literal is
    dead text that reads like a setting. `prior.credit_fraction: 0.2` sat under `prior:` for
    months, commented "SWEPT above; this value is only a fallback" — there is no fallback,
    because nothing reads a config without expanding the grid first.
    """
    import yaml

    from src.utils.config import _MISSING, _get_path

    raw = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    for swept in (raw.get("sweep") or {}):
        if swept == "seeds":
            continue
        assert _get_path(raw, swept) is _MISSING, (
            f"{path}: `{swept}` is swept AND has a literal in the body. Delete the literal."
        )


def test_the_duplicate_is_rejected_rather_than_silently_overwritten():
    from src.utils.config import apply_sweep_block

    with pytest.raises(ValueError, match="ALSO has the literal value"):
        apply_sweep_block(
            {"sweep": {"prior.credit_fraction": [0.0, 0.1]}, "prior": {"credit_fraction": 0.2}}
        )


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_every_config_expands_including_the_templates(path):
    """`--list` on Exp2/Exp3 used to raise: `float('FILL_FROM_EXP1')`. A template is exactly
    what someone inspects before filling it in, so it has to survive being read."""
    from src.utils.config import expand_with_seeds

    assert len(expand_with_seeds(_load(path))) > 0


@pytest.mark.parametrize("track", ["LGD", "PD"])
def test_datasets_that_share_loans_sit_on_one_side_of_the_split(track):
    """Audit 30-09-2026: `hackerearth` holds 5,696 of `lendingclub`'s loans, and `base_modelisation`
    shares defaults with `base_model`. A pair split across dev and holdout would let the choice made
    on dev be tested on the same loans; each pair sits in dev, in every experiment."""
    pair = ("0004.lendingclub", "0006.hackerearth") if track == "PD" else ("0004.base_model", "0005.base_modelisation")
    for exp in ("Exp0", "Exp1", "Exp2", "Exp3"):
        ev = _load(f"config/{exp}_{track}.yaml")["eval"]
        assert set(pair) <= set(ev["dev_datasets"]), f"{exp}_{track}: {pair} must both be development data"
        assert not set(pair) & set(ev["holdout_datasets"])


@pytest.mark.parametrize("track", ["LGD", "PD"])
def test_campaign_only_scores_full_contexts(track):
    """User decision: every experiment evaluates all eligible rows, with no capped companion."""
    for exp in ("Exp0", "Exp1", "Exp2", "Exp3"):
        assert not _load(f"config/{exp}_{track}.yaml")["eval"].get("context_caps")


@pytest.mark.parametrize("track", ["LGD", "PD"])
def test_every_experiment_preserves_experiment_1s_prior_mechanisms(track):
    """Continuation changes declared geometry, never the credit intervention itself.

    Compare resolved arms, not only YAML bodies: an arm override must not hide a
    changed mechanism. Exact backbone geometry and mixture matching are checked
    in test_long_context_training.
    """
    import copy
    from src.utils.config import expand_with_seeds

    def strip(prior):
        prior = copy.deepcopy(prior)
        for key in ("credit_fraction", "encoding", "n_rows_range", "n_rows_sampling",
                    "train_frac_range", "grouping", "batch_layout"):
            prior.pop(key, None)
        return prior

    exp1 = strip(_load(f"config/Exp1_{track}.yaml")["prior"])
    for name in (f"Exp0_{track}", f"Exp2_{track}", f"Exp3_{track}"):
        for arm in expand_with_seeds(_load(f"config/{name}.yaml")):
            assert strip(arm["prior"]) == exp1, f"{name} drifted from Exp1's prior mechanisms"
