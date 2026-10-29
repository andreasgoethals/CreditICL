"""Resolved recipes for the CPT and retention experiments.

Experiment 1 does not call this module. Candidate indices are stable array
identities; the actual numeric settings are saved in each resolved run config.
"""
from __future__ import annotations

RATES = {"tabicl": (2e-5, 1e-5), "tabpfn3": (1e-5, 1e-6)}
CANDIDATES = {"l2sp": (3e-4, 3e-3, 3e-2),
              "diagscale": (1e-3, 1e-2, 3e-2), "lora": (1e-5, 1e-4, 1e-3)}
MAIN_STEPS = 2500
PILOT_STEPS = 500
MILESTONES = (125, 250, 500, 1000, MAIN_STEPS)


def apply_design(run: dict, selection: dict | None = None) -> dict:
    design = run.get("design")
    if not design:
        return run
    model, task = run["architecture"], run["task"]
    train = run["train"]
    if design == "cpt_pilot":
        train["lr"] = RATES[model][int(run["lr_index"])]
        train["schedule_total_steps"] = MAIN_STEPS
        run["progress"].update(every_datasets=0, n_ood=0)
        return run
    if design not in {"cpt", "retention_pilot", "retention"}:
        raise ValueError(f"Unknown experimental design: {design}")
    run["design_unresolved"] = selection is None
    selected = (selection or {}).get("choices", {}).get(f"{task}/{model}", {})
    train["lr"] = float(selected.get("full_lr", RATES[model][0]))
    train["schedule_total_steps"] = MAIN_STEPS
    if design == "cpt":
        return run
    if design == "retention_pilot":
        run["progress"].update(every_datasets=0, n_ood=0)
    method = run["method"]
    if method not in {"no_decay", *CANDIDATES}:
        raise ValueError(f"Unknown retention treatment: {method}")
    train.update(adaptation="full", l2sp_alpha=0.0, weight_decay=0.0,
                 schedule_total_steps=MAIN_STEPS)
    if method == "no_decay":
        return run
    candidate = (int(run["candidate"]) if design == "retention_pilot"
                 else int(selected.get(method, 0)))
    value = CANDIDATES[method][candidate]
    if method == "l2sp":
        train["l2sp_alpha"] = value
    else:
        train.update(adaptation=method, optimizer="adamw", lr=value,
                     beta1=0.9, beta2=0.999, weight_decay=0.01,
                     warmup_proportion=0.1, lora_rank=16, lora_alpha=16)
    return run
