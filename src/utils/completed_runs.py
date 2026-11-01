"""Recognise reviewed Exp1 results and valid CPT prefixes across isolated repairs.

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
CONTEXT_FIX_PREDECESSOR = "01a5f056044cc261b83b3973956d55211c64b4e81ee58d285712e852109007e4"
REVIEWED_IMPLEMENTATION = "5300c294e4078273fa635f18b82714ebfdb17169046f25035e356aa2392a8e75"

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


# Exact unchanged 64-run/two-seed CPT grid; not permission for arbitrary future recipes.
CONTEXT_FIX_CPT_RECIPES: frozenset[str] = frozenset(['04a479735e582c7bda4304c238b4a9e6db27b1173e058dc8168290f03f5d4435',
 '09d1bc324bfe0e12c21cad0fcc4bcac9a2811c4bcdccd1307fdd416e99bf518e',
 '129281a008c3efb7a6a187bef15d3c51a4cec744cf2d235d524ee6da4f9514e8',
 '16cc43032676c0d5ea6dc44ab262755b707d0d21c54ec678c8dda3c21f339865',
 '232b976a81da27c2cd93dc2113545e26be013474e1ac8d02acd6aa182ff56d1a',
 '2c7c48e8222c71eb419078e245e306509c6cebead94ef04f3fe6dc03f220a6f0',
 '2dd109ae8e70ff86a5a6d77051f0ff9e3eb088e37d48865b4a7986bb319259ed',
 '2f35eb052af39fb1faf5f0cdf9e4af98d91159d0691313f3a88d0054d78519c0',
 '315b30debc8d7e73b129a1abadacf87182c1fabe2b2d6542cc78b4a0b9a4c3c0',
 '34b8a881a90b8f9650cd8dbd4b059669f027e28d898f7abfb53804b9b97e268e',
 '3d9a5be8d63d51f92f0578a7d75d4e47208e8292cde590c4fb10bd569e560f3c',
 '472020e06dde3e4ec81c2c1b105e44118ed14dbd667cf305dacd9a85cc3c8a0f',
 '48d3d49789be245d7736c2f70a5d5cc5497dc3cc991dec2e6627706cb875b075',
 '504612c9ea26f60987d05a6db253527b8381d62869fea16538b74762a1cc0436',
 '55cc27ac074fe02208d935ee7645e61e62218735d9e965e8129e6bb7c3d3daf1',
 '6bcc686b26252081a3d59deb2977cfc57f871781ac8030ca2314afe2961f9ca4',
 '6c93caa209508385afd440e73f3feade330430d4bf6b6340ead8dbe94814b7bf',
 '702537bb6d60c98a96a09f76bd92e32bb3146e7d44546f501432c88742ea3038',
 '73379447ae9d9ac59bb0cb763783895a8749f430757130f7a1697cd4f38365ab',
 '797d011d1c811b5c38ddd3177c1c1dfe3750b7a737c14f1c2d18e92635b1a996',
 '972fda6d82d9a06aa514c75c003405e77efd5bfdc8eb140921c14fc12bd00f5a',
 '9974b62eafc8e52998491c93721310492ba5c8de3cf0fc5b16ce6bdec3f4e57a',
 'a3f170f0de213875dfdaa4e13bf0d240b3eb96dd9f28e2b17d8256a10f83a9db',
 'a81bc5621da7828f03e10d0eed66457b54844e19f2e9a6901738aef7a6329291',
 'b770c3be029ce9029c9a4578c5db5e23ed7fd5337aab3c509c3c135572c6cfdb',
 'b8e9bbff58d51ae29debbb05b4d9e4c95c2643338e370270ed93da4c38833564',
 'c352a7a2adceba1c9ee981ed5db1a3e1c694eda51a6d612c4408a32cd5ba47f6',
 'e22c6b77c90ea4026220f4ddf22f20ef6541163ad30cbc0ff9dbd96184c0e354',
 'e7504fbdddf9d2d0678a0753b1a93b5bc01b2db1c2239fb235533920d7b618f8',
 'f374129a4a0196c45575279f0b9458b2345181b3db1e41237be8b1c5d70dc9b8',
 'f3b974cc42ae2991ab845f5c9476ca566a06fd0437dc20eb83f55b3e0c53af22',
 'f94585d77ba564f9f530a2702872803e5c059bad72ab54b84517eedfc60d27aa'])


def recipe_digest(cfg: dict[str, Any]) -> str:
    recipe = {key: cfg.get(key) for key in ("experiment", "task", "architecture", "model", "init", "prior")}
    recipe["train"] = {key: value for key, value in cfg.get("train", {}).items()
                       if key not in OPERATIONAL_TRAIN_KEYS}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def validate_completed_training(summary: dict[str, Any], current: dict[str, Any],
                                saved: dict[str, Any] | None) -> None:
    """Recognise only reviewed recipes and packages, retaining their original provenance."""
    from src.train.checkpoint import validate_resume_config, validate_training_fingerprint
    from src.utils import preflight

    if summary.get("training_fingerprint") == preflight.training_fingerprint():
        return
    if reviewed_context_fix_state(summary, current, saved, checkpoint=False):
        validate_resume_config(saved, current)
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
                    preflight.training_fingerprint_for(CONTEXT_FIX_PREDECESSOR),
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
            preflight.training_fingerprint_for(CONTEXT_FIX_PREDECESSOR),
            preflight.training_fingerprint_for(PREVIOUS_IMPLEMENTATION),
            preflight.training_fingerprint_for(OCT26_IMPLEMENTATION)})


def reviewed_context_fix_state(state: dict, current: dict, saved: dict | None,
                               *, checkpoint: bool) -> bool:
    """Only the exact pre-fix CPT recipes and valid prefixes may cross this repair.

    The old native preprocessor could not accept a constant context, so a saved
    completed update never contains a newly rejected task. Encoded TabICL has no
    changed branch. Keep original fingerprints; bind the exception to exact old/new
    implementations, unchanged packages/recipes/seed and full table accounting.
    """
    from src.utils import preflight

    if (preflight.training_implementation_fingerprint() != REVIEWED_IMPLEMENTATION
            or state.get("training_fingerprint") != preflight.training_fingerprint_for(CONTEXT_FIX_PREDECESSOR)
            or not isinstance(saved, dict)
            or type(current.get("seed")) is not int or current["seed"] not in (0, 1)
            or saved.get("seed") != current["seed"]
            or recipe_digest(current) not in CONTEXT_FIX_CPT_RECIPES
            or recipe_digest(saved) != recipe_digest(current)):
        return False
    step = state.get("step") if checkpoint else state.get("steps")
    extra = state.get("extra", {}) if checkpoint else state
    if (type(step) is not int or not 0 <= step <= 1500 or not isinstance(extra, dict)
            or type(extra.get("datasets_seen")) is not int
            or extra.get("datasets_seen") != step * current["train"]["batch_size"]
            or type(extra.get("skipped_tasks", 0)) is not int
            or extra.get("skipped_tasks", 0) != 0):
        return False
    if checkpoint:
        return True
    return bool((state.get("completed") is True and step == 1500)
                or (state.get("completed") is False and step < 1500
                    and state.get("stopped_by_signal") in ("SIGUSR1", "SIGTERM")))
