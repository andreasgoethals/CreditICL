"""Recognise the completed Exp1 campaign after the isolated Exp2 additions.

Historical source has completion/evaluation permission only. The separate
19a76cc/26-October-to-current review also permits unchanged Exp1 checkpoint resume. The a073d7d-to-a18efa8 source diff preserves Exp1's fixed 1024-row
dense stream, loss and optimiser. It adds opt-in long-context/ragged training,
TabPFN precision handling, validation and telemetry. Exact source and recipe
digests below constrain that review; unknown changes still fail closed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

HISTORICAL_IMPLEMENTATION = "abb00376738fe71ca1d5590eab5299fa9ae796e8515be6ba559d462495418155"
PREVIOUS_IMPLEMENTATION = "5b9cf0c29f222ac9555c5712976f3deb0f1f4d95ed7eedc75a5087316cf82b9d"
OCT26_IMPLEMENTATION = "ea055886d91ddd805ffe378d5d2722c3d3c0bcc07f909d5c30489905ba58e2e0"
REVIEWED_IMPLEMENTATION = "01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4"

# PD/LGD, original main mixtures and five component removals. Seed is checked
# separately; 25% was completed under OCT26_IMPLEMENTATION. Digests include
# all prior/model/init settings and training settings except operational knobs.
REVIEWED_RECIPES: frozenset[str] = frozenset({
    "018eb8e311c6a3e4629ce385b62bc6b555585d655f854cdc541bc33cde8b4f21",
    "18113f75078981ecf0f4335342c954f9d6e1bd87681eb3ab7232bbd08ce84e9b",
    "3abb0b11e29a981f749050d79a6ca2ea95b41b2d4eef8e5d693ec3950a1e7435",
    "4c740e29f07ba64f92eeeb906826e65abd1896755a4d7c0630059850f278afbc",
    "93826cf1d0b31b178ca1d43e73d65c2a50662e9b32c08b8e9c0cf4a9a1c8df32",
    "9cd7e5de228460fc479c1c822d944fab6737f6daeaaae3b234bc42a83cfc1310",
    "9e9f30b8f096da5b854c88557a914477a64a652cac9f66eabea702f573b3d83a",
    "a74c67682cce0dd3914e779e2519e7f859c7b07d5c8ab454bc905bb8f910c5b2",
    "ae57f18e4c009644468d8f396ac748580d7f6d9bd39817131c6bd2ed27de0224",
    "b20422eb691b42d88130d063c45b0b0c7c1bf78318a49adc090f9adcdfadf0a0",
    "b31b1584ba55bda975791fd91f2773b090b192b252740214e9b242afff04aada",
    "cf8438a968657c3227fa26804a290387468c153f00d51bf38e80d0de12d6a2c1",
    "d862efcc32ee8d3de1b9e04bc640eb2f90bfdd94aa94d5333e9bcfce2b8dedc1",
    "dfa5d01758ec02730b141af593c2c25f93f306942399b7d420085f40b4a702f3",
    "eb6c7a21c3a0c2806baf0f7b13b534c9c2947f87103f20fd0fbd63ae375d5703",
    "fd4394e9981e454bcf244f8a55f13d7212b3c54c908a5714541edd030adb79a7",
})
OPERATIONAL_TRAIN_KEYS = frozenset({
    "num_workers", "max_workers", "prefetch_factor", "log_every", "save_temp_every",
    "save_perm_every", "max_temp_checkpoints",
})

# Added below from the unchanged 19a76cc quarter-mixture recipes. Historical
# original-grid provenance does not acquire permission for these later arms.
QUARTER_RECIPES: frozenset[str] = frozenset(['2cf510e27f2d40406e5edc6654e1199471299dfbf8ce7fe54466c690a486a0a0', '604a9b66640997e88edd3d613dab38faa632e0e2833a97e48a9d2c9fdc40d93d'])


def recipe_digest(cfg: dict[str, Any]) -> str:
    recipe = {key: cfg.get(key) for key in ("experiment", "task", "architecture", "model", "init", "prior")}
    recipe["train"] = {key: value for key, value in cfg.get("train", {}).items()
                       if key not in OPERATIONAL_TRAIN_KEYS}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def validate_completed_training(summary: dict[str, Any], current: dict[str, Any],
                                saved: dict[str, Any] | None) -> None:
    """Allow only an explicitly reviewed, completed Exp1 run with unchanged packages."""
    from src.train.checkpoint import validate_resume_config, validate_training_fingerprint
    from src.utils import preflight

    if summary.get("training_fingerprint") == preflight.training_fingerprint():
        return
    if (saved is not None and summary.get("completed") is True
            and summary.get("steps") == current.get("train", {}).get("max_steps") == 12500
            and summary.get("datasets_seen") == 800000
            and type(current.get("seed")) is int and current["seed"] in (0, 1, 2)
            and recipe_digest(current) in REVIEWED_RECIPES | QUARTER_RECIPES
            and recipe_digest(saved) == recipe_digest(current)
            and saved.get("seed") == current["seed"]
            and preflight.training_implementation_fingerprint() == REVIEWED_IMPLEMENTATION
            and (summary.get("training_fingerprint") in {
                    preflight.training_fingerprint_for(PREVIOUS_IMPLEMENTATION),
                    preflight.training_fingerprint_for(OCT26_IMPLEMENTATION)}
                 or (recipe_digest(current) in REVIEWED_RECIPES and summary.get("training_fingerprint")
                     == preflight.training_fingerprint_for(HISTORICAL_IMPLEMENTATION)))):
        validate_resume_config(saved, current)
        return
    # Retain the existing actionable error, including for missing provenance.
    validate_training_fingerprint(summary)


def reviewed_exp1_resume(payload: dict, current: dict) -> bool:
    """Only the reviewed unchanged Exp1 update rule may continue across this patch.

    Original campaign fingerprints remain completion/evaluation-only. A change
    to source, packages, seed, recipe or task accounting closes this exception.
    """
    from src.utils import preflight
    saved = payload.get("crediticl_config")
    step = payload.get("step")
    return bool(isinstance(saved, dict) and type(step) is int and 0 <= step <= 12500
        and type(current.get("seed")) is int and current["seed"] in (0, 1, 2)
        and (payload.get("extra") or {}).get("datasets_seen") == 64 * step
        and recipe_digest(current) in REVIEWED_RECIPES | QUARTER_RECIPES
        and recipe_digest(saved) == recipe_digest(current)
        and saved.get("seed") == current["seed"]
        and preflight.training_implementation_fingerprint() == REVIEWED_IMPLEMENTATION
        and payload.get("training_fingerprint") in {
            preflight.training_fingerprint_for(PREVIOUS_IMPLEMENTATION),
            preflight.training_fingerprint_for(OCT26_IMPLEMENTATION)})
