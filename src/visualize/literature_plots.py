"""Figures that visualise the LITERATURE itself, not our runs.

`literature.py` holds the numbers and draws single reference lines onto our figures; this module
draws the standalone figures whose subject *is* the published landscape — where the field sits, so a
reader can place our results beside it. The distinction these figures exist to keep honest: every
value here is a paper's, measured under the paper's protocol, and is drawn on its own axis and
labelled as such — never mixed into a panel of our own measurements, where a different protocol would
read as a head-to-head score. The notebooks call these and hold no logic of their own.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import literature as lit
from src.visualize import style

#: Colour by how the number was established: an evaluated result in REFERENCE teal, a value the paper
#: only *describes* (a dataset property, a competition ceiling it cites) in EXTERNAL amber, so the two
#: never read as the same kind of claim.
_TAG_COLOUR = {"paper-evaluated": style.REFERENCE, "paper+code": style.REFERENCE,
               "code+paper": style.REFERENCE, "code-supported": style.MUTED}


def _colour(tag: str) -> str:
    return _TAG_COLOUR.get(tag, style.EXTERNAL)


def credit_benchmark_landscape(keys: tuple[str, ...] | None = None):
    """The literature's reported ROC-AUC on real credit datasets, as a horizontal landscape.

    One row per published number (Tanna 2026 on Home Credit / Lending Club, Hollmann 2023 on
    Credit-g), sorted by value, each bar ending at the reported AUC with the source named on the
    axis. Evaluated results and merely-described ceilings take different colours. This is the field,
    on its own protocol — the caption says so — not a scoreboard our arms appear on.
    """
    keys = keys or lit.CREDIT_BENCHMARKS
    refs = [(k, lit.REFS[k]) for k in keys if lit.REFS[k].value is not None]
    refs.sort(key=lambda kv: kv[1].value)
    style.apply()
    fig, ax = plt.subplots(figsize=style.row_figsize(len(refs), per_row=0.30, base=1.3))
    y = np.arange(len(refs))
    values = [r.value for _, r in refs]
    colours = [_colour(r.tag) for _, r in refs]
    bars = ax.barh(y, values, color=colours, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r.label for _, r in refs], fontsize=7)
    style.bar_value_labels(ax, bars, fmt="{:.3f}", horizontal=True)
    # A random classifier scores 0.5; anchoring there gives the reader the scale of the spread.
    ax.axvline(0.5, color=style.MUTED, ls=":", lw=1.0, zorder=1)
    ax.annotate("0.5 = random", (0.5, len(refs) - 0.4), fontsize=6.5, color=style.MUTED,
                xytext=(3, 0), textcoords="offset points", va="top")
    lo = min(0.5, min(values)) - 0.02
    ax.set_xlim(lo, max(values) + 0.03)
    ax.set_xlabel("reported ROC-AUC (each paper's own protocol)")
    ax.legend(handles=style.legend_patches({"evaluated": style.REFERENCE,
                                            "described / ceiling": style.EXTERNAL}),
              loc="lower right")
    style.title(ax, "Credit-domain AUCs in the literature")
    fig.suptitle("Where the field sits on real credit data")
    return fig


def summary(keys: tuple[str, ...] | None = None) -> str:
    """A text digest of the literature landscape, for a notebook's printed summary."""
    keys = keys or lit.CREDIT_BENCHMARKS
    refs = [(k, lit.REFS[k]) for k in keys if lit.REFS[k].value is not None]
    refs.sort(key=lambda kv: kv[1].value, reverse=True)
    lines = [f"CREDIT-DOMAIN LITERATURE LANDSCAPE (tfm-library pin {lit.PIN})",
             "  reported ROC-AUC on real credit datasets, each under its own protocol:"]
    for _, r in refs:
        lines.append(f"    {r.value:.3f}  {r.label}   [{r.tag}]")
    lines.append("  these are the field's numbers, not a like-for-like target — the protocol differs.")
    return "\n".join(lines)
