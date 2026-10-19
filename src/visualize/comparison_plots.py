"""Paired final OOD and ablation effects, with datasets as the uncertainty units."""
import matplotlib.pyplot as plt
import numpy as np

from src.eval.analysis import effects
from src.visualize import style
from src.visualize.results_plots import _placeholder


def paired_effects(track, exp="exp1", *, ablations=False, metric=None):
    metric = metric or ("log_loss" if track == "pd" else "r2")
    table = effects(track, exp, ablations=ablations, metric=metric)
    style.apply()
    if table.empty:
        return _placeholder("Complete final benchmark results required for paired comparisons")
    fig, ax = plt.subplots(figsize=style.row_figsize(len(table)))
    y = np.arange(len(table))
    for i, row in enumerate(table.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color=style.CREDIT)
    ax.scatter(table.mean_difference, y, color=style.CREDIT)
    ax.axvline(0, color=style.REFERENCE)
    ax.set_yticks(y, [name.split("__", 1)[-1] for name in table.configuration])
    ax.set_xlabel(f"{style.metric_label(metric)}: treatment minus control (paired datasets)")
    return fig
