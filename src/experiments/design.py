"""Predeclared CPT and retention recipes, with no benchmark-driven selection.

Experiment 1 does not use this module. Both CPT learning rates are reported;
Experiment 3 always compares against rate index zero, regardless of its score.
"""
from __future__ import annotations

RATES = {"tabicl": (2e-5, 1e-5), "tabpfn3": (1e-5, 1e-6)}
DESIGN_REVISION = "fixed-rates-1500-two-seeds-v2"
SEEDS = (0, 1)
MAIN_STEPS = 1500
MILESTONES = (125, 250, 500, 1000, MAIN_STEPS)
# External numerical anchors, not hyperparameters selected on this benchmark.
# Real-TabPFN: L2-SP .003; Ning et al.: DiagScale .03. Q/V-only LoRA at
# 1e-4/rank16 is a fixed engineering choice, not Ning's all-linear LoRA recipe.
FIXED_RETENTION = {"l2sp_alpha": .003, "diagscale_lr": .03,
                   "lora_lr": 1e-4, "lora_rank": 16, "lora_alpha": 16}


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
    revision = run.get("design_revision", DESIGN_REVISION)
    if revision != DESIGN_REVISION:
        raise ValueError(f"Unsupported experiment design revision: {revision}")
    model = run["architecture"]
    if model not in RATES:
        raise ValueError(f"Unknown CPT backbone: {model}")
    index = run.get("lr_index")
    if type(index) is not int or not 0 <= index < len(RATES[model]):
        raise ValueError(f"lr_index must be an integer in [0, {len(RATES[model]) - 1}]")
    if design == "retention" and index != 0:
        raise ValueError("Retention uses the predeclared primary CPT rate (lr_index=0)")
    run.update(design_revision=DESIGN_REVISION, design_unresolved=False)
    train = run["train"]
    train.update(lr=RATES[model][index], schedule_total_steps=MAIN_STEPS)
    if design == "cpt":
        return run
    method = run["method"]
    if method not in {"no_decay", "l2sp", "diagscale", "lora"}:
        raise ValueError(f"Unknown retention treatment: {method}")
    train.update(adaptation="full", l2sp_alpha=0.0, weight_decay=0.0)
    if method == "l2sp":
        train["l2sp_alpha"] = FIXED_RETENTION["l2sp_alpha"]
    elif method in {"diagscale", "lora"}:
        train.update(adaptation=method, optimizer="adamw", lr=FIXED_RETENTION[f"{method}_lr"],
                     beta1=.9, beta2=.999, weight_decay=.01, warmup_proportion=.1,
                     lora_rank=FIXED_RETENTION["lora_rank"], lora_alpha=FIXED_RETENTION["lora_alpha"])
    return run
