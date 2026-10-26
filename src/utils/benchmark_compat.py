"""A narrow receipt transition for the approved Exp1 quarter-mixture extension.

Do not turn this into a general ignore-code option. The reviewed transition only
changes selection messages, the launcher's default array range, and the receipt
comparison call. Evaluation, inputs, packages, checkpoints and result digests
must still agree. A later edit to any of those three files fails closed again.
"""
from __future__ import annotations

import copy


# Filled with SHA-256 of the exact reviewed before/after files (LF and CRLF).
# The old revision is 50b3d7e; this is provenance, not a Git/network dependency.
REVIEWED_TRANSITIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    'src/eval/benchmark_status.py': (
        frozenset(['007019fbd1ad4f53cf5be434ececfb34280db431fabc30f2801e7afb1f6856ed', '89ea0d838855bc6ded19d131a5d3ee68018bc4588435abf981b541377c4bde20']),
        frozenset(['095b1b144eeea4f28870b51422db6a2a99921b01cf009cc8edf2c1f76960b7a8', 'ad2477a1a2dea9021766004748f77760516878d406bc4a06f5acc3fd59b08cef']),
    ),
    'src/eval/selection.py': (
        frozenset(['1d50a171d2c787bff9176b4909dffb09e39342398e8a71d180d34ab22f64a19a', '52d495780ae9a9b7a23cb78c16324377a36aca59dc8b50cf76545a504b3ea816']),
        frozenset(['24523f3fcbaa8cf6a930bdeefac220aa66019677bc640dc57dfdea9dc1cffbd4', 'adfcf8fc8ee6a25b3cd8b721fa0701112b001a9cb4ea1a6c172240dfce9d5b5c']),
    ),
    'scripts/slurm/benchmark.slurm': (
        frozenset(['a760ef1aa7fbf8aaed984b2a50e3918f3b3d3d05d69f52b88c5e21a83465611f', 'ca0395369d515cb48ec8f92eed2f7786223e6d27deefe706e3e64c2c4a895904']),
        frozenset(['457022c7f0a8f883d58cedc42d083862195e5dc3c50be8cce196fbf2af3f0082', '565908fe79049393f9022d188e08884b915ac9126b911cb1d06c6be9c0a1ae7a']),
    ),
}


def compatible_requests(recorded: object, wanted: dict) -> bool:
    """Allow changed grid counts only when every scientific input is identical."""
    if recorded == wanted:
        return True
    if not isinstance(recorded, dict):
        return False
    before, after = copy.deepcopy(recorded), copy.deepcopy(wanted)
    old_code, new_code = before.get("code"), after.get("code")
    if not isinstance(old_code, dict) or not isinstance(new_code, dict):
        return False
    if old_code.keys() != new_code.keys():
        return False
    for path, new_hash in new_code.items():
        old_hash = old_code[path]
        if old_hash == new_hash:
            continue
        transition = REVIEWED_TRANSITIONS.get(path)
        if not transition or old_hash not in transition[0] or new_hash not in transition[1]:
            return False
        old_code[path] = new_hash
    for request in (before, after):
        config = request.get("config")
        if isinstance(config, dict) and isinstance(config.get("_grid"), dict):
            # Keep the arm identity, assignments, step, seed and every recipe
            # setting; only the number of OTHER runs in the campaign may differ.
            config["_grid"].pop("total", None)
            config["_grid"].pop("array_total", None)
    return before == after
