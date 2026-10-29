"""A narrow receipt transition for the approved Exp1 quarter-mixture extension.

Do not turn this into a general ignore-code option. Reviewed transitions add
selection messages, array sizes, provenance checks and opt-in adapter/checkpoint
routes. Experiment 1 retains its native full-model prediction path and cadence.
Evaluation inputs, packages, checkpoints and result digests must still agree.
Any unreviewed scientific-code change fails closed again.
"""
from __future__ import annotations

import copy


# Filled with SHA-256 of the exact reviewed before/after files (LF and CRLF).
# Reviewed old revisions: a073d7d, 50b3d7e and a18efa8. No Git/network dependency.
REVIEWED_TRANSITIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    'src/eval/benchmark_status.py': (
        frozenset(['007019fbd1ad4f53cf5be434ececfb34280db431fabc30f2801e7afb1f6856ed', '095b1b144eeea4f28870b51422db6a2a99921b01cf009cc8edf2c1f76960b7a8', '3e00a8cf280276213ebd24d27f1ee6e321d75175abd98780d69982563a57cb23', '89ea0d838855bc6ded19d131a5d3ee68018bc4588435abf981b541377c4bde20', 'ad2477a1a2dea9021766004748f77760516878d406bc4a06f5acc3fd59b08cef', 'b84e09b9de10c20021c6d95340384322a9601716aa9663163d6cd000532e2c53']),
        frozenset(['2dbfd1b8f4b7a21e17ae98901d1731f25110bd227df792d0bf14b81e06878433', '879860ad9fee464d37976a37ffc5fdec303d01086845872e8f677d23bbf6d145']),
    ),
    'src/eval/selection.py': (
        frozenset(['1d50a171d2c787bff9176b4909dffb09e39342398e8a71d180d34ab22f64a19a', '24523f3fcbaa8cf6a930bdeefac220aa66019677bc640dc57dfdea9dc1cffbd4', '4d6376184777ba73899f106ff575cc5a7c6a3f92f797d4cf24fab9360649aaab', '52d495780ae9a9b7a23cb78c16324377a36aca59dc8b50cf76545a504b3ea816', 'adfcf8fc8ee6a25b3cd8b721fa0701112b001a9cb4ea1a6c172240dfce9d5b5c', 'f8f2ae5ca8b1780d375da43d920144173df5c9f16219ef62610bd742abbf591a']),
        frozenset(['873e3d9fe878d9815f16611a6d3e11c1794798c61bd877932d17e3283e23541e', 'f9ba50446c660a549c3b0aa0fc8d68f0fdb5e82906a7bc875db6ee2811e735e3']),
    ),
    'scripts/slurm/benchmark.slurm': (
        frozenset(['457022c7f0a8f883d58cedc42d083862195e5dc3c50be8cce196fbf2af3f0082', '565908fe79049393f9022d188e08884b915ac9126b911cb1d06c6be9c0a1ae7a', 'a760ef1aa7fbf8aaed984b2a50e3918f3b3d3d05d69f52b88c5e21a83465611f', 'ca0395369d515cb48ec8f92eed2f7786223e6d27deefe706e3e64c2c4a895904']),
        frozenset(['ebca55e7a3b6a7a70e93834ffb6a50c2826248801251c12f464ca96b4162b489', 'eccc69e3956231b27f3bf145ffa1dfe658135de435fb1961f2b337dd80f6596a']),
    ),
    'src/eval/baselines.py': (
        frozenset(['5827560184405383d41f51e86b62e74a6fccb8cf281b449fc936dc20d4fb87bb', 'a303d09a78106a195b2c8a724e67cf61b6f5e527810728161c47669a2da924f5']),
        frozenset(['4420b7d03babd6296c84d469c9997123d07b056b6831f70d8f82df10f40bca0e', 'd8e46d9cc50ac845b3870fa10c1b22866707fb33bed265c05eaac37b4b9c7d79']),
    ),
    'src/eval/crediticl_baseline.py': (
        frozenset(['8b86f80bd7e3f046fb6f796d4495f8c8568b07c4722bb9ab429640ed3c280731', '8eec485d8a1d80e4775ffbd0f168afc5a9536097c75eb8c12c377e13741bcb22']),
        frozenset(['71735ef53bafeee0deb7a5089ba5744d9052ff8a7ff1fe603c44d15622dfdd6b', '85b2d15afdbb7f50e3f0ea2c452591c40d703c5f9ba110c7cf77f46b0d8bb930']),
    ),
    'src/eval/protocol.py': (
        frozenset(['475c27c16dc72879776a45e8b37496562541b38ae98f6199c1faa226e00342c5', 'f379f4dd429496cd07b9d0b474e5f1be6a2afa8c8375c72cb16858501a0d17ec']),
        frozenset(['115a32e7d1c8fd00ffa34ceedd7ad24a96fe2a92fea77588c31e5ec8a30db0c0', 'acf6ecb6e4a7c76e14c83337c5313dabd71482462954f5db26e58680a21d2eba']),
    ),
    'src/eval/progress.py': (
        frozenset(['940cf6869408ea99939f2dec8e42847dba3e7f8f63b2695fdd9bf674f547e3af', 'c0d7190abc027b8cf3f9b81a80ae9ddd1b6449ad165b3b6bcc5673b78fb8a67c']),
        frozenset(['c01b0e634fedc707dee01b6ef791c956ec2a8a42f8d28e993832f486a89642d2', 'df84cde59c6b0c150b6731361b77cff099894e2b4548cf4cb5c836943d59b789']),
    ),
    'src/eval/results.py': (
        frozenset(['0367aebe55452b0f440f24b9b14acbd8291236c23edff536063b526fdc64c0e5', '4049724a245c259cfd42e0edc714f83fdf95cedca876d4807c4c0ec22e073d06']),
        frozenset(['0c137b553290b2f7c3db0f8d6a6f7041a8c44d58633039cf41e2541d34c52b1c', 'b1b4efe3b98c2c40829c81a22a7295617762437369ce06c8608dca4cdb943667']),
    ),
}

# The 29-10 provenance fix hashes the same input bytes directly rather than
# staging a copy only to hash it. It changes neither prediction nor scoring.
_before, _after = REVIEWED_TRANSITIONS['src/eval/benchmark_status.py']
REVIEWED_TRANSITIONS['src/eval/benchmark_status.py'] = (
    _before | _after,
    _after | frozenset({'bb4f97a3230f52a3985ff1dc8296d53945f028d7ffd9ef7d0737c56e14fa876a',
                        '71d165032f62dfbbe12da8de4a657fb6d8f79c8880f7bc7b9d023b2b7ee5cce3'}),
)


# Budget A changes only the selection CLI explanation; Exp1 predictions are unchanged.
_before, _after = REVIEWED_TRANSITIONS["src/eval/selection.py"]
REVIEWED_TRANSITIONS["src/eval/selection.py"] = (
    _before | _after, _after | frozenset(['a664729176dc4b31266c8e69b6088eece8bf0ffc4c4b9b94b6aa1abc33f2e40d', 'c326e779c94f20278755694659c73945ecc55f5a50abd4ba7b259de8ccaa4d80']),
)


# Fixed-rate design changes only the explanatory CLI text, never Exp1 scoring.
_before, _after = REVIEWED_TRANSITIONS["src/eval/selection.py"]
REVIEWED_TRANSITIONS["src/eval/selection.py"] = (
    _before | _after, _after | frozenset(['192beb936e80b767f4750169e3561d7c29db15414ecb1c127d9d7edc820c6215', '5ed96d55b76068bec09eba48dca800e11f6cfde84f652e3f4a8801a8b621e83b']),
)


# Workspace description only; the prediction implementation is identical.
REVIEWED_TRANSITIONS["src/eval/tabicl_memory.py"] = (
    frozenset(['2dd1ac97ebbc849424cbc1ce6f742eba6918b6bfef557c4dda1098bc00867d35', '9ef0f00013debd76bb53f8df221ed11a6425e8b034a02e576b3bb3c5cced614e']), frozenset(['094f50b18d0ecb949ab45e2f4161fb33ea1355357956b38a137275b68b594cbb', 'b43848335996c286d0357c8397f7433f52255674d587586c7df8f538eb2d4aca']),
)


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
