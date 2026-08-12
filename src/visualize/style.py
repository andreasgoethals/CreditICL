"""THE shared style. One place, every notebook in THIS project.

Two halves, and the split is deliberate.

**What the template fixes** is what follows from the output medium: every figure ends up in a
paper printed on A4, so it is drawn at the width it will occupy on the page and its text is
sized to be readable there. Same in every project, so it is not ours to choose.

**What this project fills in** is the look — and here the colours carry meaning. `CREDIT` and
`ORIGINAL` mean the same thing in every figure: our credit-targeted prior versus TabICL's
unmodified one. Chosen once, so a reader who has understood one figure can read the next
without going back to the legend.

Call `apply()` once at the top of every notebook. A notebook never picks a colour or a size
itself; if it needs a new one it is added here, and every figure gains it together.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# FIGURE SIZES — A4, and nothing else.
#
# A4 is 210 x 297 mm. With 25 mm margins that leaves a 160 x 247 mm text block, and the numbers
# below are that block in inches.
#
# DRAW AT FINAL WIDTH, and never rescale a figure in the document. Rescaling carries the text with
# it: 9pt squeezed to 70% arrives as 6.3pt, under the ~7pt floor where small print stops being
# legible on paper. So the point sizes in `_RC` are the point sizes ON THE PRINTED PAGE.
# ---------------------------------------------------------------------------

WIDTH_FULL = 6.30   # 160 mm — the full A4 text width
WIDTH_HALF = 3.05   # two side by side, with a ~5 mm gutter
WIDTH_THIRD = 1.95  # three side by side. Label sparingly at this width.

#: The A4 text block is 247 mm tall, but a figure taking all of it leaves no room for its caption
#: and pushes every surrounding paragraph onto another page. Half a page is the practical ceiling,
#: and `figsize` clamps to it rather than letting a tall panel grid silently overflow.
MAX_HEIGHT = 4.80   # 122 mm

GOLDEN = 0.618      # height = width * GOLDEN, unless the data wants otherwise


def figsize(width: float = WIDTH_FULL, ratio: float = GOLDEN) -> tuple[float, float]:
    """(width, height) in inches, clamped to `MAX_HEIGHT`. `ratio` is height/width.

    Pass `WIDTH_FULL`, `WIDTH_HALF` or `WIDTH_THIRD` — never a number of your own, because the
    whole point is that the figure arrives on the page at exactly the width it was drawn at.
    """
    return (width, min(width * ratio, MAX_HEIGHT))


def grid_figsize(ncols: int, nrows: int, *, panel_ratio: float = 0.78) -> tuple[float, float]:
    """Size for an `nrows` x `ncols` panel grid, always `WIDTH_FULL` wide.

    Panel grids are where hard-coded sizes creep back in: it is tempting to write
    `figsize=(3.4 * ncols, 2.7 * nrows)`, which is correct on screen and 13 inches wide for
    four columns — twice the page. The width is fixed here and only the HEIGHT scales with the
    row count, then gets clamped like everything else. Panels get narrower as you add columns,
    which is what actually happens on paper.
    """
    height = (WIDTH_FULL / max(ncols, 1)) * panel_ratio * max(nrows, 1)
    return (WIDTH_FULL, min(height, MAX_HEIGHT))


def row_figsize(n_rows: int, *, per_row: float = 0.16, base: float = 1.0) -> tuple[float, float]:
    """Size for a horizontal-bar / table-like figure with one line per item.

    Height grows with the number of rows (a 20-dataset bar chart needs more than a 5-dataset
    one) but the width never does, and the total is clamped to half a page.
    """
    return (WIDTH_FULL, min(base + per_row * max(n_rows, 1), MAX_HEIGHT))


#: Below this, a panel is too small to carry a readable axis label at 8pt. Ten histograms
#: across A4 gives 0.63 in each, which is a thumbnail, not a figure.
MIN_PANEL_WIDTH = 1.15


def max_cols(min_width: float = MIN_PANEL_WIDTH) -> int:
    """How many panels fit across A4 before they stop being legible."""
    return max(1, int(WIDTH_FULL // min_width))


def paginate(items: list[Any], per_page: int | None = None,
             *, min_width: float = MIN_PANEL_WIDTH) -> list[list[Any]]:
    """Split items into page-sized chunks, the way a paper does it.

    THE PROBLEM THIS SOLVES. `grid_figsize` guarantees a grid never exceeds the page width,
    which is necessary but not sufficient: ten histograms across 160 mm are 0.63 in each, so
    the figure is page-correct and unreadable. Making it *taller* does not help — the
    constraint is horizontal.

    So the answer is the one a journal uses: more than one figure. Ten panels become
    "Figure 3 (page 1 of 2)" and "(page 2 of 2)", each with panels wide enough to read.

    Returns a list of pages; a single page means no split was needed, and callers can then
    skip the page suffix entirely.
    """
    if not items:
        return []
    per_page = per_page or max_cols(min_width)
    return [items[i:i + per_page] for i in range(0, len(items), per_page)]


def page_suffix(page: int, n_pages: int) -> str:
    """`" (page 2 of 3)"`, or `""` when there is only one page.

    Goes in the figure's suptitle AND should be repeated in the caption, because a reader who
    meets page 2 first needs to know that page 1 exists.
    """
    return f" (page {page} of {n_pages})" if n_pages > 1 else ""


# ---------------------------------------------------------------------------
# What A4 output requires. Everything here is about the figure being correct on paper.
# ---------------------------------------------------------------------------

#: Most preferred first. DejaVu Sans is LAST and is the one that matters: it ships with matplotlib,
#: so it is the only face guaranteed present both locally and on a compute node. A missing face
#: makes matplotlib fall back silently, which changes text metrics — moving every label and making
#: a cluster-drawn figure differ from the local one for no visible reason.
_FONT_STACK = ["Source Sans 3", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]

_RC = {
    # A4 full text width by default, so a figure saved without thinking about it is already the
    # right size for the page.
    "figure.figsize": figsize(),
    "figure.dpi": 110,

    # constrained_layout, and NOT savefig.bbox="tight". Tight-bbox crops to the drawn content, so
    # two figures declared at the same width come out at different widths and the paper's font
    # sizes stop matching between them. constrained_layout fits the content INSIDE the declared
    # size instead. `None` is matplotlib's spelling for "use the declared figure size".
    "figure.constrained_layout.use": True,
    "savefig.bbox": None,
    "savefig.facecolor": "white",
    "savefig.transparent": False,

    # TrueType, not the default Type 3: Type 3 is rejected by several journal submission systems
    # and cannot be searched or copied out of the PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Point sizes ON THE PRINTED A4 PAGE, since the figure is drawn at final width. 9pt sits just
    # under a paper's own 10-11pt, which reads as "part of the document" rather than shrunken;
    # 7pt is the floor below which small print stops being legible on paper.
    "font.family": "sans-serif",
    "font.sans-serif": _FONT_STACK,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
}

# ---------------------------------------------------------------------------
# THIS PROJECT'S OWN LOOK.
#
# The colours carry meaning and the meaning does not change between figures. Everything else
# here is about getting out of the data's way: two spines instead of four, a grid you read
# through rather than at, and ticks short enough not to read as stray marks.
# ---------------------------------------------------------------------------

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

#: Ordered palette for when several things must be distinguished (e.g. 7 datasets). Okabe-Ito,
#: which is colour-blind-safe AND separates in greyscale by lightness — a paper gets photocopied.
SERIES = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#8c564b", "#7f7f7f",
]

#: Semantic map for the two tasks. LGD shares CREDIT's blue; PD gets a violet so a figure
#: showing both is readable without a legend.
TASK_COLOR = {"lgd": "#2b6cb0", "pd": "#7c3aed"}

_PROJECT_RC: dict = {
    # Minus signs as a proper typographic minus rather than a hyphen.
    "axes.unicode_minus": True,
    # Titles left-aligned and bold: the eye finds them without hunting.
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "figure.titleweight": "bold",
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
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "grid.alpha": 0.9,
    "axes.axisbelow": True,
    # Ticks pulled in and shortened; with only two spines the long default ticks read as
    # stray marks.
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    # White, not transparent: a transparent figure pasted into a dark slide turns all the
    # black text invisible.
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # Breathing room between panels. The default packs subplots so tightly that a two-line
    # title on one panel touches the axis labels of the one above.
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.w_pad": 0.04,
    "figure.constrained_layout.hspace": 0.06,
    "figure.constrained_layout.wspace": 0.05,
    # Data marks, thinned for print: at A4 width a 1.8pt line is heavy.
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.5,
    "patch.edgecolor": "white",
    "hist.bins": 40,
    "legend.frameon": False,
    "legend.borderaxespad": 0.4,
    "legend.handlelength": 1.4,
    "legend.columnspacing": 1.0,
    "axes.prop_cycle": mpl.cycler(color=SERIES),
}


def apply() -> None:
    """Install the style. Call once, at the top of every notebook and plotting script.

    Idempotent, so re-running a cell is safe.
    """
    mpl.rcParams.update(_RC)
    mpl.rcParams.update(_PROJECT_RC)


def source_color(source: str) -> str:
    """Colour for a dataset, by which prior produced it."""
    return CREDIT if source == "credit" else ORIGINAL


#: Characters per inch for the title face, measured empirically at 10pt DejaVu Sans. Used only
#: to decide where to wrap, so it does not need to be exact — one character either way changes
#: nothing, and a real text-extent measurement needs a renderer that does not exist yet when
#: the title is being set.
_TITLE_CHARS_PER_INCH = 12.5


def _wrap_to_axes(ax: Any, text: str, scale: float = 1.0) -> str:
    """Wrap `text` to the width of `ax`, in characters.

    Left-aligned titles do not get wrapped by matplotlib and do not participate in horizontal
    layout, so on a 1x3 grid a long title simply runs across its neighbour's panel. This is
    the fix: wrap before setting, using the axes' actual width on the figure.
    """
    import textwrap

    try:
        width_in = ax.get_window_extent().width / ax.figure.dpi
    except Exception:  # noqa: BLE001 — no renderer yet; fall back to the declared size
        width_in = ax.figure.get_size_inches()[0] / max(len(ax.figure.axes), 1)
    size = mpl.rcParams["axes.titlesize"] * scale
    chars = max(12, int(width_in * _TITLE_CHARS_PER_INCH * (10.0 / max(size, 1e-6))))
    return "\n".join(textwrap.wrap(text, chars)) or text


def title(ax: Any, headline: str, subtitle: str | None = None) -> None:
    """Headline plus an optional quieter line under it, wrapped to the panel.

    Implemented as a **`set_title`**, not a separate `ax.text`. An `ax.text` at `y > 1` in axes
    coordinates is invisible to matplotlib's layout engine, so it overlapped the panel above it
    in every dense grid and ran into the figure suptitle. A real title is measured and laid out.

    WRAPPING IS NOT COSMETIC. `axes.titlelocation` is `left`, and matplotlib neither wraps a
    title nor counts its width when laying panels out — so two panels side by side on A4 had
    their titles run into each other. Now each is wrapped to its own panel's width.
    """
    scale = 0.92 if subtitle else 1.0
    lines = [_wrap_to_axes(ax, headline, scale)]
    if subtitle:
        lines.append(_wrap_to_axes(ax, subtitle, scale))
    ax.set_title("\n".join(lines), linespacing=1.3)
    if subtitle:
        # The two parts cannot take different sizes on one Text object, so the whole title
        # drops slightly and the subtitle reads as a continuation. Keeping one object is what
        # makes the layout correct, which matters more than the two-tone look.
        ax.title.set_fontsize(mpl.rcParams["axes.titlesize"] * scale)


def figure_note(fig: Any, text: str) -> None:
    """A line under the whole figure — what to look for, in words."""
    fig.supxlabel(text, fontsize=mpl.rcParams["font.size"] * 0.85, color=MUTED)


def legend_patches(labels: dict[str, str]) -> list[Any]:
    """Proxy handles for {label: colour}, for plots drawn with bare `hist`/`bar`."""
    from matplotlib.patches import Patch

    return [Patch(facecolor=c, label=lbl, edgecolor="white") for lbl, c in labels.items()]


def annotate_value(ax: Any, x: float, y: float, text: str, *, color: str = INK) -> None:
    """Put a number on the mark it belongs to. Saves the reader squinting at ticks."""
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(0, 5),
        ha="center", fontsize=mpl.rcParams["font.size"] * 0.8, color=color,
    )


def show_palette() -> Any:
    """A swatch of the palette, so a notebook can document its own colour meanings."""
    apply()
    entries = [
        ("our prior (credit)", CREDIT),
        ("original TabICL prior", ORIGINAL),
        ("real credit data", REAL),
        ("problem / out of range", WARN),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 0.95))
    for i, (label, colour) in enumerate(entries):
        ax.add_patch(plt.Rectangle((i, 0), 0.85, 1, color=colour))
        ax.text(i + 0.425, -0.28, label, ha="center", va="top",
                fontsize=mpl.rcParams["xtick.labelsize"], color=MUTED)
    ax.set_xlim(-0.1, len(entries))
    ax.set_ylim(-1.1, 1)
    ax.axis("off")
    ax.set_title("What the colours mean")
    return fig
