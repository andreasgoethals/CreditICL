# Full retraining or fine-tuning? What TabICL actually allows

Written 2026-08-05. Library pin `21d555a`.

**Short answer.** Four options are real, and one is not:

| option | what it does | where it comes from | recommendation |
|---|---|---|---|
| `scratch` | train every weight from random init | TabICLv2 stage 1 | **start here** |
| `full` | pretrained weights, train everything at a low LR | TabICLv2 stage 3 | best fine-tune option |
| `icl_only` | freeze the col + row blocks, train the ICL stack + y-embeddings + head | TabICL **v1** stage 3 | good second arm |
| `head_only` | freeze all three block stacks | our own floor | reference only |
| ~~LoRA~~ | low-rank adapters | — | **do not** — see §4 |

All four are implemented in [`src/train/adapt.py`](../src/train/adapt.py) and
selected by `init.strategy` in the configs.

---

## 1. What the architecture allows

TabICL has three stages in sequence, and the model file makes the split obvious:

```
TF_col   column embedding      InducedTransformerBlock x 3   "what is in this column"
TF_row   row attention         TransformerBlock x 3          "what is in this row"
TF_icl   in-context learning   TransformerBlock x 12         "what does y look like given the context"
```

The target enters **twice**: `y_embed_in` is added before the column blocks, and
`y_embed_icl` is added again before the ICL blocks. That detail matters for us and
is covered in §3.

TabICL ships freezing support for exactly these three stages. From
`_finetune/base.py`:

```python
def _frozen_submodules(self, model):
    out = []
    if self.freeze_col: out.append(model.col_embedder)
    if self.freeze_row: out.append(model.row_interactor)
    if self.freeze_icl: out.append(model.icl_predictor)
    return out
```

Two implementation details worth copying, both of which we did:

* `_apply_freezing` sets `requires_grad = False`, and `_set_training_mode`
  separately snaps frozen parts back to `eval()`. This is not redundant:
  `nn.Module.train()` is recursive, so calling it would switch dropout back on
  inside a frozen block.
* the optimizer only ever receives
  `[p for p in self.model_.parameters() if p.requires_grad]`.

## 2. What the TabICL authors actually did

This is the most useful evidence available, because it is the people who built the
model choosing a recipe rather than us guessing. Read straight off their training
scripts.

**TabICL v1, stage 3** (`scripts/train_stage3.sh`) — a partial fine-tune:

```
--freeze_col True          # column embedder frozen
--freeze_row True          # row encoder frozen
--lr 2e-6                  # very low
--scheduler constant       # no decay
--gradient_clipping 1.0    # down from 10.0 in stage 1
--only_load_model True     # weights only, not optimizer state
--max_steps 50 --batch_size 512
```

**TabICL v2, stage 3** (`scripts/train_v2_{clf,reg}_stage3.sh`) — **no freezing**:

```
--lr 2e-5
--scheduler cosine_with_restarts
--gradient_clipping 1.0
--only_load_model True
--max_steps 10000
```

So **the v2 authors moved away from freezing.** With the v2 architecture they
train everything at a low LR in the final stage. That is a real signal, and it is
why `full` rather than `icl_only` is the recommended fine-tune arm.

Note the pattern in both: the final stage is always a **low LR with tight
gradient clipping** (1.0 instead of stage 1's 10.0). Whatever you freeze, do not
fine-tune at stage-1 learning rates.

## 3. The one place we improve on a naive freeze

Our change is to the **target** distribution, not the feature distribution. So the
parts that most need to move are the target-side parts.

A naive "freeze the column stage" would also freeze `y_embed_in`, which is where
the target first enters the model — exactly the parameter that needs to adapt to a
bounded [0,1] target with mass at the boundaries. That would be
self-defeating.

So in [`src/train/adapt.py`](../src/train/adapt.py), freezing covers the **block
stacks** and always leaves these trainable, whatever the strategy:

```python
ALWAYS_TRAINABLE = ("y_embed_in", "y_embed_icl", "out_ln", "out_mlp",
                    "row_ln", "row_cls_tokens")
```

`icl_only` therefore means: keep the learned table representation, retrain
everything that touches the target.

## 4. Why not LoRA

Three reasons, in order of weight:

1. **TabICL has no LoRA.** Searching the entire repository dump for "lora" returns
   **zero matches**. There is nothing upstream to copy or validate against.
2. **Tanna 2025 found LoRA unstable on TabPFN** — batched-inference constraints
   force an automatic fallback to full fine-tuning. Rubachev 2025 independently
   found that for TabPFN v2, full fine-tuning matches LoRA and every other
   parameter-efficient variant on accuracy while converging fastest. So even where
   LoRA works, it buys nothing.
3. **It would be a third confound.** We are already varying the prior and the
   adaptation strategy. Adding an unvalidated adapter means a bad result could be
   the prior, the strategy, or the adapter, and we could not tell which.

If you later want a parameter-efficient option, the better-evidenced one is
BETA-style input adapters (Liu & Ye 2025): a small learnable input encoder,
0.6 MB of trainable parameters, weights otherwise frozen. But it is v1-only and
classification-only in the paper, so it is future work, not a tonight decision.

## 5. The warning that matters most

**Fine-tuning TabICL is documented as dangerous.** Tanna 2026 (*Exploring
Fine-Tuning for Tabular Foundation Models*) reports that full supervised
fine-tuning is **near-catastrophic for TabICL**: TabZilla accuracy drops from
**0.873 to 0.567**. TabPFN survives the same treatment essentially intact. This is
a strong architecture-by-adaptation interaction, and TabICL is on the bad side of
it.

Two honest caveats on that result: it is a four-page paper with **one
hyperparameter setting per strategy**, so "TabICL collapses" may partly be
under-tuning rather than a property of the architecture; and the authors are
comparing against their own models. But it points the same way as the v2 authors'
own choice of a very low LR, so the two independent signals agree: **be gentle**.

This is also a direct argument for the mixture lever. Keeping 70–90% of datasets
from the original prior means the model keeps seeing the distribution it was built
for, which is the natural defence against exactly the collapse Tanna documents.

## 6. The practical blocker

**Loading the released TabICLv2 checkpoints into NanoTabICL is untested and may
not work.** The released weights come from the full implementation, whose module
names differ (`col_embedder` / `row_interactor` / `icl_predictor` versus
NanoTabICL's `col_blocks` / `row_blocks` / `icl_blocks`). NanoTabICL's own README
points you at the main repository for pretrained weights, and notes a RoPE change
that "permutes the neurons", so a mismatch is plausible.

`load_pretrained` in `src/train/adapt.py` handles this by **refusing to run** when
fewer than half the tensors match, rather than quietly training a randomly
initialised model and reporting it as fine-tuned. That failure would be invisible
in the loss curve and would silently invalidate every comparison.

Three ways forward, best first:

1. **Pretrain your own base with `scratch`, then fine-tune from that.** The
   architecture is identical by construction, so there is no mapping problem at
   all. This is why `scratch` is the default and the recommended first run.
2. Write an explicit key mapping from the full checkpoint to NanoTabICL, and
   verify a forward pass reproduces the full model's output on the same input.
   Do not trust a mapping that has not been checked numerically.
3. Use the full `tabicl` package instead of NanoTabICL for the fine-tuning arms
   (`pip install -e ".[tabicl]"`), which loads its own checkpoints natively. Costs
   us the small, readable codebase.

## 7. Recommended sequence

1. **`scratch`, `credit_fraction` swept over 0.0 / 0.1 / 0.2 / 0.3.** This
   answers the actual research question — does adding our datasets to the prior
   help — with nothing else moving. Run this first.
2. **`full` from the best `scratch` checkpoint**, at `lr 2e-5`, cosine, clip 1.0.
   Answers "is it cheaper to fine-tune than to retrain?"
3. **`icl_only` from the same base**, at `lr 2e-6`, constant, clip 1.0. Answers
   "does keeping the table representation fixed protect against collapse?"

Steps 2 and 3 need step 1's checkpoint, so they cannot run tonight. That is fine —
step 1 is the one that answers the research question.

`recommended_hparams()` in `adapt.py` returns the LR / schedule / clipping each
strategy was actually used with upstream, so those numbers stay visible in the
config rather than being applied silently.
