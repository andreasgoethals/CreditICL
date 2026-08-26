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

#: One file per experiment per track. Exp1 screens the prior grid; Exp2 and Exp3 build on
#: its winner, so they stay templates until Exp1 has finished.
EXPERIMENTS = [f"config/{e}_{t}.yaml" for e in ("Exp1", "Exp2", "Exp3") for t in ("LGD", "PD")]
#: The runnable ones. Exp2/Exp3 hold `FILL_FROM_EXP1` and must not expand.
CONFIGS = ["config/Exp1_LGD.yaml", "config/Exp1_PD.yaml"]


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
    assert cfg["experiment"].endswith(expected)
    assert cfg["experiment"].startswith(Path(path).stem.split("_")[0].lower())


# -- the six-file layout, and prior_file: ------------------------------------



# -- placeholders --------------------------------------------------------------


@pytest.mark.parametrize("path", CONFIGS)
def test_exp1_is_runnable_now(path):
    """Exp1 depends on nothing, so it must load without the escape hatch."""
    cfg = load(ROOT / path)
    assert find_placeholders(cfg) == []


@pytest.mark.parametrize("path", [p for p in EXPERIMENTS if "Exp1" not in p])
def test_exp2_and_exp3_refuse_to_expand_until_exp1_has_run(path):
    """A config that runs with a placeholder burns GPU-hours measuring nothing. The
    refusal is the whole safeguard, so it gets a test."""
    with pytest.raises(ValueError, match="still a template"):
        load(ROOT / path)
    holes = find_placeholders(_load(path))
    assert holes, f"{path} claims to be a template but has no {PLACEHOLDER}"



def test_exp2_refuses_a_silent_partial_load():
    """A name mismatch that loads nothing still runs and still outputs numbers — they are
    just partly random. `strict_load` is what turns that into a crash."""
    for track in ("LGD", "PD"):
        cfg = _load(f"config/Exp2_{track}.yaml")
        assert cfg["init"]["strict_load"] is True
        assert cfg["init"]["pretrained_path"], "a warm start needs a checkpoint"


# -- the screening budget ------------------------------------------------------


def test_exp1_is_cheaper_per_arm_than_exp3():
    """Exp1 exists to make the grid affordable. If its budget ever matched Exp3's, the
    two-phase design would have quietly become one very expensive phase."""
    for track in ("LGD", "PD"):
        one = _load(f"config/Exp1_{track}.yaml")["train"]["max_steps"]
        two = _load(f"config/Exp3_{track}.yaml")["train"]["max_steps"]
        assert one < two, f"{track}: screening budget {one} is not below the full {two}"


def test_exp3_reports_more_seeds_than_the_screen():
    """Ranking arms tolerates 3 seeds; a headline interval does not."""
    for track in ("LGD", "PD"):
        assert len(_load(f"config/Exp1_{track}.yaml")["seeds"]) == 3
        assert len(_load(f"config/Exp3_{track}.yaml")["seeds"]) >= 5


# -- the frozen evaluation split ----------------------------------------------


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_dev_and_holdout_are_disjoint_and_non_empty(path):
    """A dataset in both splits leaks the holdout into prior selection, which is the one
    mistake that cannot be fixed after the fact."""
    ev = _load(path)["eval"]
    dev, hold = set(ev["dev_datasets"]), set(ev["holdout_datasets"])
    assert dev and hold, f"{path}: both splits must be populated before any run"
    assert not (dev & hold), f"{path}: {dev & hold} is in both splits"
    assert ev["select_on"] == "dev", "selecting on holdout invalidates the experiment"


@pytest.mark.parametrize("path", EXPERIMENTS)
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
def test_seeds_are_crossed_outermost(path):
    """A cut-short array should cover the lever grid at one seed, not one lever
    at every seed. That only holds if seed is the outer loop."""
    runs = expand_with_seeds(_load(path))
    n_seeds = len(_load(path)["seeds"])
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


@pytest.mark.parametrize("path", CONFIGS)
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


def test_no_experiment_shares_a_prior_file():
    """Exp1 sweeps 32 priors, Exp3 runs the ONE that won, Exp2 sweeps the mixture. A shared
    `prior_file:` encoded the false claim that all three use the same prior, so each config is
    now self-contained."""
    for path in EXPERIMENTS:
        assert "prior_file" not in load_yaml(ROOT / path), (
            f"{path} names a shared prior file; each experiment must hold its own"
        )
        assert "prior" in load_yaml(ROOT / path), f"{path} has no inline prior block"


def test_exp1_sweeps_priors_and_never_the_architecture():
    """The claim the whole project rests on: the architecture is fixed and only the prior
    varies. An architecture knob appearing in a sweep would silently confound the two."""
    for track in ("LGD", "PD"):
        axes = dict(sweep_axes(_load(f"config/Exp1_{track}.yaml")))
        prior_axes = [k for k in axes if k.startswith("prior.")]
        assert len(prior_axes) >= 3, f"{track}: Exp1 must sweep the prior, got {prior_axes}"
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
        assert n_priors == 32, f"{track}: expected 32 Exp1 priors, got {n_priors}"

        exp3 = [(k, v) for k, v in sweep_axes(_load(f"config/Exp3_{track}.yaml")) if k != "seeds"]
        assert exp3 == [("prior.credit_fraction", [0.0, PLACEHOLDER])], (
            f"{track}: Exp3 must be exactly control-vs-winner, got {exp3}"
        )


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


def test_exp2_uses_a_continued_pretraining_learning_rate():
    """TabPFN-Wide (Kolberg et al. 2026) continued-pretrains at 1e-5. Applying pretraining's
    8e-4 to already-trained weights destroys them in a few hundred steps, and the loss curve
    would look like a bad prior rather than a bad learning rate."""
    for track in ("LGD", "PD"):
        two = _load(f"config/Exp2_{track}.yaml")["train"]
        three = _load(f"config/Exp3_{track}.yaml")["train"]
        # A LIST since 24-08-2026: Exp2 sweeps the rate, because the published continued-
        # pretraining rates disagree by two orders of magnitude (Real-TabPFN 3e-7,
        # TabPFN-Wide 1e-5, TabICLv2 stage 3 2e-5). Every value swept must still be far below
        # pretraining's, or the sweep is testing "does destroying the weights help".
        rates = two["lr"] if isinstance(two["lr"], list) else [two["lr"]]
        assert max(rates) <= three["lr"] / 10, (
            f"{track}: Exp2 rates {rates} are not far below Exp3's {three['lr']}"
        )
        assert two["muon_lr"] <= three["muon_lr"] / 10
        assert two["warmup_proportion"] > three["warmup_proportion"], "warm start needs longer warmup"
        assert two["gradient_clipping"] < three["gradient_clipping"]


def test_exp2_sweeps_the_mixture_because_that_is_its_question():
    """The model already knows the original prior, so how much of ours to add IS the experiment.
    `1.0` must be included so 'forgetting the original prior' is measurable, and `0.0` so
    'continued pretraining alone' is a control."""
    for track in ("LGD", "PD"):
        axes = dict(sweep_axes(_load(f"config/Exp2_{track}.yaml")))
        mixture = axes["prior.credit_fraction"]
        assert 0.0 in mixture and 1.0 in mixture, f"{track}: mixture must span 0..1, got {mixture}"
        assert len(mixture) >= 4, "an interior optimum needs interior points"
        assert "init.strategy" in axes, "full vs parameter-efficient must be measured, not assumed"


def test_exp2_is_cheaper_than_exp3():
    """The point of Exp2 is that it is cheap. If its budget ever matched Exp3's there would be
    no claim left to make."""
    for track in ("LGD", "PD"):
        two = _load(f"config/Exp2_{track}.yaml")["train"]
        three = _load(f"config/Exp3_{track}.yaml")["train"]
        assert two["max_steps"] * two["batch_size"] < three["max_steps"] * three["batch_size"]


# -- one architecture, everywhere ---------------------------------------------


def test_every_experiment_names_the_same_architecture():
    """One architecture for all three, and it is TabICLv2's own. Two experiments on different
    architectures cannot be compared, and the paper's claim is about the prior."""
    named = {_load(p)["architecture"] for p in EXPERIMENTS}
    assert named == {"tabicl"}, f"experiments disagree on the architecture: {named}"


def test_config_folder_holds_exactly_the_six_experiments():
    """Three experiments x two tracks, each self-contained. Anything else here is a file nobody
    named, which is how a stale config gets submitted."""
    found = sorted(p.name for p in (ROOT / "config").iterdir())
    assert found == sorted(Path(p).name for p in EXPERIMENTS), f"unexpected: {found}"


def test_exp2_and_exp3_differ_only_where_they_should():
    """They are not identical any more — Exp2 is continued pre-training, so its budget, learning
    rate and mixture all differ by design. What must still match is the EVALUATION, or the two
    cannot be compared at all."""
    for track in ("LGD", "PD"):
        two, three = _load(f"config/Exp2_{track}.yaml"), _load(f"config/Exp3_{track}.yaml")
        assert two["eval"] == three["eval"], f"{track}: evaluation must be identical"
        assert two["task"] == three["task"]
        assert two["architecture"] == three["architecture"]
        assert three["init"]["strategy"] == "scratch"
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
