# The notebooks, as one story

Eleven notebooks in three chapters, read in order. Each chapter is a folder, and each notebook's number
starts with its chapter's, so the file list, `output/All_Results.md` and `output/figures/CAPTIONS.md`
all read in the same order as the story. Every notebook opens with a line linking the one before and the
one after.

| chapter | notebook | the question it answers |
|---|---|---|
| **[0. General](<0. General/>)** — the problem and the prior | [0.1 · the real credit data](<0. General/0.1_data_exploration.ipynb>) | What must a credit prior reproduce? How rare is default, where does the loss mass sit, can the evaluation be trusted? |
| | [0.2 · the PD prior](<0. General/0.2_prior_visualisation_pd.ipynb>) | What does our PD prior generate, how does it differ from TabICL's own, and does it look like real credit? |
| | [0.3 · the LGD prior](<0. General/0.3_prior_visualisation_lgd.ipynb>) | The same for LGD: where do the boundary atoms come from, and do they land where real books are? |
| **[1. Experiment 1](<1. Experiment 1/>)** — which prior? | [1.1 · PD training](<1. Experiment 1/1.1_pd_training.ipynb>) | Did every PD arm train soundly, and what does development monitoring say about each lever? |
| | [1.2 · LGD training](<1. Experiment 1/1.2_lgd_training.ipynb>) | The same for LGD. |
| | [1.3 · PD results](<1. Experiment 1/1.3_pd_results.ipynb>) | Which prior does the development split select, and how does it do on the holdout? |
| | [1.4 · LGD results](<1. Experiment 1/1.4_lgd_results.ipynb>) | The same for LGD. |
| **[2. Experiment 2](<2. Experiment 2/>)** — fine-tuning the released model | [2.1 · PD fine-tuning](<2. Experiment 2/2.1_pd_finetuning.ipynb>) | Did every fine-tuning arm train soundly, and what does specialising on credit cost out of domain? |
| | [2.2 · LGD fine-tuning](<2. Experiment 2/2.2_lgd_finetuning.ipynb>) | The same for LGD. |
| | [2.3 · PD results](<2. Experiment 2/2.3_pd_results.ipynb>) | Which fine-tuning recipe does development select, and how does it do on the holdout? |
| | [2.4 · LGD results](<2. Experiment 2/2.4_lgd_results.ipynb>) | The same for LGD. |

## Every notebook tells its part the same way

A reader who has learned one notebook can read the rest. Each opens with what it answers and how it fits
the story, then a colour key (except 0.1 — the prior colours mean nothing for real data), then parts:

- **The General notebooks** go *what the data looks like* → *where the difference comes from* (0.2 and
  0.3 take the prior apart one credit mechanism at a time) → *whether the task is learnable and sane*.
- **The training notebooks** go **A · Is the run sound?** (the sweep map, what each arm was scored on,
  what each cost, whether every arm learned) → **B · What does development monitoring say?** (lever by
  lever, then within each credit fraction, then against seed noise) → **C · Where does it hold?** (every
  dataset, out of domain, every metric) → **D · Every arm, in detail**.
- **The results notebooks** go **A · Which prior does development select?** → **B · How does it do on
  the holdout?** → **C · Where does it hold?** → **D · Where the field sits**.

Every notebook ends by printing its findings in the same order, which is what `output/All_Results.md`
collects.

## The one rule the whole story rests on

A prior is **chosen on the development split** and **reported on the holdout**, which stays untouched
until the end (`docs/EXPERIMENTAL_DESIGN.md` §5). The training notebooks only ever average development
datasets: arms monitored before the development-only protocol (23-09-2026) also scored some holdout
datasets, which the notebooks *show* — labelled, in their own panels — but never average. Each training
notebook's figure A2 draws exactly which scores count.

The cross-task question the design calls its centrepiece — the PD/LGD double dissociation (§5.2) — needs
the benchmark scores of both tasks, and is read from 1.3 and 1.4 together once phase 2 has run.

## Running them

```
python -m src.utils.run_notebooks                       # all eleven, in parallel
python -m src.utils.run_notebooks --only 1.1_pd_training
```

Notebooks are found by recursing into the chapter folders; a notebook is named by its stem alone, so
stems must be unique across folders (the runner refuses duplicates). A notebook opened interactively
from its own folder walks up to the repository root before importing anything. The two prior notebooks
generate their priors live and are slow — 0.2 can take over an hour — so give a full run room:
`--timeout 7200`.
