"""Predeclared CPT and retention recipes, with no benchmark-driven selection.

Experiment 1 does not use this module. Both CPT learning rates are reported;
Experiment 3 reuses both 50%-mixture CPT rates, regardless of their scores.
"""
from __future__ import annotations

RATES = {"tabicl": (2e-5, 1e-5), "tabpfn3": (1e-5, 1e-6)}
DESIGN_REVISION = "fixed-rates-1500-two-seeds-v2"
RETENTION_REVISION = "retention-six-recipes-1500-v3"
SEEDS = (0, 1)
MAIN_STEPS = 1500
MILESTONES = (125, 250, 500, 1000, MAIN_STEPS)
# Six complete recipes, not a Cartesian product that duplicates full-update arms.
# Lower adapter rates follow the reviewed CreditPFN archive sensitivity. These
# are research choices for synthetic CPT, not claimed upstream CPT optima.
RETENTION_RECIPES = {
    "no_decay": ("full", None, 0.),
    "l2sp": ("full", None, 0.),
    "diagscale_low": ("diagscale", 1e-4, .01),
    "diagscale_high": ("diagscale", 1e-3, .01),
    "lora_low": ("lora", 3e-6, 0.),
    "lora_high": ("lora", 1e-5, 0.),
}
FIXED_RETENTION = {"l2sp_alpha": .003, "lora_rank": 16, "lora_alpha": 16}


def retention_baseline(run: dict) -> bool:
    """Both registered CPT rates at 50%; selection never reads outcomes."""
    return (run.get("design") == "cpt" and run.get("design_revision") == DESIGN_REVISION
            and run.get("lr_index") in (0, 1) and run["prior"]["credit_fraction"] == .5)


def apply_design(run: dict, selection: dict | None = None) -> dict:
    """Resolve the registered grid without reading any scores or receipts."""
    design = run.get("design")
    if not design:
        return run
    if design in {"cpt_pilot", "retention_pilot"}:
        raise ValueError("Pilot selection is retired; use the fixed Exp2/Exp3 configurations")
    if design not in {"cpt", "retention"}:
        raise ValueError(f"Unknown experimental design: {design}")
    if selection is not None or any(k in run for k in ("_selection", "selection_provenance", "candidate")):
        raise ValueError("Selection receipts/candidates cannot change a predeclared experiment")
    expected_revision = DESIGN_REVISION if design == "cpt" else RETENTION_REVISION
    revision = run.get("design_revision", expected_revision)
    if revision != expected_revision:
        raise ValueError(f"Unsupported experiment design revision: {revision}")
    model = run["architecture"]
    if model not in RATES:
        raise ValueError(f"Unknown CPT backbone: {model}")
    index = run.get("lr_index")
    if type(index) is not int or not 0 <= index < len(RATES[model]):
        raise ValueError(f"lr_index must be an integer in [0, {len(RATES[model]) - 1}]")
    if design == "retention" and index != 0:
        raise ValueError("Retention uses the predeclared primary CPT rate (lr_index=0)")
    run.update(design_revision=expected_revision, design_unresolved=False)
    train = run["train"]
    if model == "tabpfn3" and run["task"] == "lgd":
        train["regression_target_policy"] = "standardize"
    train.update(lr=RATES[model][index], schedule_total_steps=MAIN_STEPS)
    if design == "cpt":
        return run
    method = run["method"]
    if method not in RETENTION_RECIPES:
        raise ValueError(f"Unknown retention treatment: {method}")
    adaptation, rate, decay = RETENTION_RECIPES[method]
    train.update(adaptation=adaptation, l2sp_alpha=0.0, weight_decay=decay)
    if method == "l2sp":
        train["l2sp_alpha"] = FIXED_RETENTION["l2sp_alpha"]
    elif adaptation in {"diagscale", "lora"}:
        train.update(optimizer="adamw", lr=rate,
                     beta1=.9, beta2=.999, warmup_proportion=.1,
                     lora_rank=FIXED_RETENTION["lora_rank"], lora_alpha=FIXED_RETENTION["lora_alpha"])
    return run
