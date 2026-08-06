"""One visual language for every figure in the project.

Two reasons this is a module rather than a few lines at the top of each notebook:

1. **Figures end up in the thesis.** A plot styled per-notebook drifts — different
   fonts, different blues, different grid weights — and then twenty figures have to
   be redone by hand at writing time.
2. **The colours carry meaning.** `CREDIT` and `ORIGINAL` mean the same thing in
   every plot: our prior versus TabICL's. If those two colours were chosen ad hoc
   per figure, the reader would have to re-learn the legend each time.

Defaults are set for **screen reading in a notebook** (legible sizes, light grid).
`use_style(context="paper")` switches to tighter print settings.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# -- the meaning-carrying colours --------------------------------------------
#: Our credit-targeted prior. A saturated blue: it is the subject of the figure.
CREDIT = "#2b6cb0"
#: The unmodified TabICL prior — the control. Deliberately grey and recessive.
ORIGINAL = "#94a3b8"
#: Real credit data, when overlaid as a reference. Warm, so it reads as "measured".
REAL = "#c2410c"
#: For "this is wrong / out of range" annotations.
WARN = "#b91c1c"
#: Neutral ink for text, axes and annotations.
INK = "#1e293b"
MUTED = "#64748b"
GRID = "#e2e8f0"

#: Ordered palette for when several things must be distinguished (e.g. 7 datasets).
#: Colour-blind-safe ordering (Okabe-Ito), so a reader with deuteranopia can still
#: tell the series apart.
SERIES = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#8c564b", "#7f7f7f",
]

#: Semantic map for the two tasks, used by the exploration notebook.
TASK_COLOR = {"lgd": "#2b6cb0", "pd": "#7c3aed"}


def use_style(context: str = "notebook") -> None:
    """Apply the project style to every subsequent figure.

    Call once at the top of a notebook. Idempotent, so re-running a cell is safe.
    """
    scale = {"notebook": 1.0, "paper": 0.85}.get(context, 1.0)
    base = 11 * scale

    mpl.rcParams.update(
        {
            # Type. DejaVu Sans ships with matplotlib, so figures look identical on
            # the laptop and on the cluster — no missing-font fallbacks.
            "font.family": "DejaVu Sans",
            "font.size": base,
            "axes.titlesize": base * 1.05,
            "axes.labelsize": base * 0.95,
            "xtick.labelsize": base * 0.85,
            "ytick.labelsize": base * 0.85,
            "legend.fontsize": base * 0.85,
            "figure.titlesize": base * 1.35,
            # Titles left-aligned and bold: the eye finds them without hunting.
            "axes.titlelocation": "left",
            "axes.titleweight": "semibold",
            "axes.titlepad": 10,
            # Ink. Only the left and bottom spines; the other two carry no data.
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # A grid you can read through, behind the data.
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            # Figure. White, not transparent: a transparent PNG pasted into a dark
            # slide turns all the black text invisible.
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
            # Data marks.
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "patch.linewidth": 0.6,
            "patch.edgecolor": "white",
            "hist.bins": 40,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def source_color(source: str) -> str:
    """Colour for a task, by which prior produced it."""
    return CREDIT if source == "credit" else ORIGINAL


def title(ax: Any, headline: str, subtitle: str | None = None) -> None:
    """Headline plus an optional quieter line under it.

    Most of these figures need a sentence of interpretation ("higher is worse"),
    and a subtitle keeps that next to the data instead of in prose far away.
    """
    ax.set_title(headline)
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] * 0.82, color=MUTED, va="bottom",
        )


def figure_note(fig: Any, text: str) -> None:
    """A caption under the whole figure — what to look for, in words."""
    fig.supxlabel(text, fontsize=mpl.rcParams["font.size"] * 0.82, color=MUTED)


def legend_patches(labels: dict[str, str]) -> list[Any]:
    """Proxy handles for {label: colour}, for plots drawn with bare `hist`/`bar`."""
    from matplotlib.patches import Patch

    return [Patch(facecolor=c, label=lbl, edgecolor="white") for lbl, c in labels.items()]


def annotate_value(ax: Any, x: float, y: float, text: str, *, color: str = INK) -> None:
    """Put a number on the mark it belongs to. Saves the reader squinting at ticks."""
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(0, 6),
        ha="center", fontsize=mpl.rcParams["font.size"] * 0.8, color=color,
    )


def savefig(fig: Any, path: str) -> str:
    """Save where figures belong, making the directory if needed."""
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    return str(p)


def show_palette() -> Any:
    """A swatch of the palette, so the notebook documents its own colour meanings."""
    use_style()
    entries = [
        ("our prior (credit)", CREDIT),
        ("original TabICL prior", ORIGINAL),
        ("real credit data", REAL),
        ("problem / out of range", WARN),
    ]
    fig, ax = plt.subplots(figsize=(7, 1.1))
    for i, (label, colour) in enumerate(entries):
        ax.add_patch(plt.Rectangle((i, 0), 0.85, 1, color=colour))
        ax.text(i + 0.425, -0.28, label, ha="center", va="top", fontsize=9, color=MUTED)
    ax.set_xlim(-0.1, len(entries))
    ax.set_ylim(-1.1, 1)
    ax.axis("off")
    ax.set_title("What the colours mean")
    return fig
