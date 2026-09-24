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
#: MARKERS for real datasets drawn ON TOP of coloured data — stars, reference points.
#: Deliberately NOT `REAL`: orange stars over orange dots are invisible, which is exactly what
#: happened in the LGD boundary-mass figure. Magenta appears nowhere else in the palette, so a
#: star can never be mistaken for a data point whatever colour the series behind it takes.
STAR = "#e6007e"
#: The two credit-prior INTENSITIES swept in Exp1 — mild retail vs aggressive corporate default
#: correlation (PD), light vs heavy boundary atoms (LGD). Two shades of CREDIT's blue, so they read
#: as "the same prior, turned up": mild lighter, aggressive deeper.
CREDIT_MILD = "#7ba7d7"
CREDIT_STRONG = "#173a63"

#: Context (the rows the model conditions on) vs query (the rows it must predict), for the shift
#: and reject-inference figures. Context is the recessive grey of the given book; query is a
#: violet that means nothing else in these figures. It used to be `REAL`'s orange, which made a
#: synthetic query read as "measured on real data" to anyone who had learned the palette.
CONTEXT = "#94a3b8"
QUERY = "#6d28d9"
#: External baseline MODELS in the benchmark (CatBoost, TabPFN, the released TabICLv2, a linear
#: model) — neither our prior (blue) nor the control (grey), and not real data (orange) either,
#: which is what they were drawn in before. Olive appears nowhere else in the palette.
BASELINE = "#4d7c0f"

#: A published value from the literature, overlaid on our own measurement as a reference — a Basel
#: asset correlation, a TabICLv2 filtering rate, an LGD R² band. Teal, so it reads as "the paper
#: said" and is never mistaken for our data (blue), the control (grey) or the real data (orange).
REFERENCE = "#0d9488"
#: The two LGD boundary atoms, given opposed meanings a reader can feel: mass at 0 is FULL RECOVERY
#: (a good outcome, cool green) and mass at 1 is TOTAL LOSS (a bad outcome, deep rose). Distinct
#: from REAL/WARN so a figure showing atoms and a real-data overlay stays legible.
ATOM_LO = "#0f766e"
ATOM_HI = "#9d174d"

#: A value from OUTSIDE the tfm-library — credit-domain knowledge the library does not contain (a
#: Basel asset correlation, the Merton/Vasicek default model). Amber, deliberately NOT the REFERENCE
#: teal, so a reference line says at a glance whether the number is library-grounded (teal) or domain
#: knowledge cited from beyond it (amber). The honesty distinction the whole project rests on, made
#: visible on the axis rather than left to the caption.
EXTERNAL = "#a16207"

#: For "this is wrong / out of range" annotations.
WARN = "#b91c1c"
#: Neutral ink for text, axes and annotations.
INK = "#1e293b"
MUTED = "#64748b"
GRID = "#e2e8f0"

#: Named colormaps, both colour-blind-safe and monotone in lightness so they survive a greyscale
#: photocopy — `CMAP_SEQ` for one-directional intensity (a correlation magnitude, a density),
#: `CMAP_DIV` for a signed quantity around a meaningful zero (an effect, a difference from control).
CMAP_SEQ = "cividis"
CMAP_DIV = "RdBu_r"

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


def variant_colour(variant: str, index: int = 0) -> str:
    """Colour for a prior VARIANT by what it is, never by where it sits in a dict.

    `original…` is the grey control and `credit…` our blue prior, whatever the suffix — a pool
    called `credit_v1`, a live draw called `credit`. Anything else (a second credit variant from a
    pool) takes the next series colour. Picking by position is what drew the control orange and
    our prior green in the prior notebooks: the lookup matched only the exact names `original` and
    `credit_v1`, and live draws were called `original (live)` / `credit (live)`.
    """
    name = str(variant).lower()
    if name.startswith("original"):
        return ORIGINAL
    if name in ("credit", "credit_v1") or name.startswith(("credit (", "credit prior")):
        return CREDIT
    return SERIES[(index + 1) % len(SERIES)]


def variant_label(variant: str) -> str:
    """How a prior variant reads in a legend or tick: `original prior`, `credit prior`, or the pool
    name for anything else. Where the tasks came from (a live draw or a pool) is reported once,
    in the notebook's printed source line — not repeated in every label."""
    name = str(variant).replace(" (live)", "")
    return {"original": "original prior", "credit": "credit prior",
            "credit_v1": "credit prior"}.get(name, name.replace("_", " "))


def credit_fraction_marker(fraction: float) -> str:
    """A marker per credit fraction, so cf 0.5 and cf 1 separate by SHAPE as well as by shade —
    two blues of similar weight are hard to tell apart in a small scatter."""
    f = float(fraction)
    return "o" if f <= 0 else ("^" if f >= 1 else "s")


#: Ordered styles for a lever's values when the lever is not the credit fraction (filter mode,
#: freeze strategy, L2-SP, learning rate): ink shades plus line styles, so the values separate in
#: greyscale and borrow no colour that means something else in the palette.
LEVEL_STYLES = [("#1e293b", "-"), ("#475569", "--"), ("#7b8799", ":"), ("#334155", "-.")]


def credit_fraction_colour(fraction: float) -> str:
    """Colour for a credit fraction: ORIGINAL grey at 0 (the control), CREDIT blue at 1, and a
    straight blend between. The fraction is "how much of our prior", so the colour says it
    continuously — and the blend is monotone in lightness, so cf 0 / 0.5 / 1 still separate in a
    greyscale photocopy. Deliberately NOT CREDIT_MILD/CREDIT_STRONG: those mean prior *intensity*,
    and a figure with intensity on its x-axis must not also use them for the fraction."""
    import matplotlib.colors as mcolors

    f = min(max(float(fraction), 0.0), 1.0)
    lo, hi = mcolors.to_rgb(ORIGINAL), mcolors.to_rgb(CREDIT)
    return mcolors.to_hex(tuple(a + (b - a) * f for a, b in zip(lo, hi)))


#: How a metric reads on an axis, in a heading or a legend — the paper's spelling, not the CSV
#: column's. `roc_auc` in a figure is a variable name; "ROC-AUC" is a quantity.
METRIC_LABEL = {
    "roc_auc": "ROC-AUC", "auc": "ROC-AUC", "pr_auc": "PR-AUC", "ap": "PR-AUC",
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "crps": "CRPS", "pinball": "pinball loss",
    "brier": "Brier score", "logloss": "log loss", "ks": "KS statistic", "bias": "bias",
    "calibration_slope": "calibration slope", "spearman": "Spearman ρ", "kendall": "Kendall τ",
    "boundary_mass_abs_err": "boundary-mass error", "boundary_mass_err_0": "mass error at 0",
    "boundary_mass_err_1": "mass error at 1", "coverage_50": "50% interval coverage",
    "coverage_80": "80% interval coverage", "coverage_90": "90% interval coverage",
    "pit_mean": "PIT mean", "pit_uniformity_error": "PIT uniformity error",
    "mae_boundary": "MAE at the boundaries", "mae_interior": "MAE in the interior",
    "pred_mass_at_0": "predicted mass at 0", "pred_mass_at_1": "predicted mass at 1",
    "true_mass_at_0": "true mass at 0", "true_mass_at_1": "true mass at 1",
    "pred_out_of_unit": "share outside [0, 1]", "pred_nonfinite_frac": "non-finite share",
    "nan_predictions": "NaN predictions", "pred_min": "smallest prediction",
    "pred_max": "largest prediction",
}


def metric_label(metric: str) -> str:
    """The display name of a metric; an unknown one falls back to its words, never its code."""
    return METRIC_LABEL.get(metric, metric.replace("_", " "))


#: What "better" means for each metric: `"max"`, `"min"`, or the ideal VALUE for a metric that is
#: right at a target rather than at an extreme — a calibration slope of 1, a bias of 0, a 90%
#: interval that covers 90%. Treating every unknown metric as higher-is-better drew "↑" over
#: errors, biases and coverages alike; an unknown metric now has no goal rather than a wrong one.
METRIC_GOAL: dict[str, Any] = {
    "roc_auc": "max", "auc": "max", "pr_auc": "max", "ap": "max", "r2": "max", "ks": "max",
    "spearman": "max", "kendall": "max",
    "rmse": "min", "mae": "min", "pinball": "min", "crps": "min", "brier": "min",
    "logloss": "min", "ece": "min", "boundary_mass_abs_err": "min", "pit_uniformity_error": "min",
    "mae_boundary": "min", "mae_interior": "min",
    "calibration_slope": 1.0, "bias": 0.0, "pit_mean": 0.5, "coverage_50": 0.5,
    "coverage_80": 0.8, "coverage_90": 0.9, "boundary_mass_err_0": 0.0, "boundary_mass_err_1": 0.0,
}


def metric_goal(metric: str) -> Any:
    """`"max"`, `"min"`, an ideal value, or `None` when the metric has no agreed direction."""
    return METRIC_GOAL.get(metric)


def goal_mark(metric: str) -> str:
    """The direction as it reads after a metric's name: `↑`, `↓`, `→ 1`, or nothing."""
    goal = metric_goal(metric)
    if goal == "max":
        return "↑"
    if goal == "min":
        return "↓"
    if isinstance(goal, (int, float)):
        return f"→ {goal:g}"
    return ""


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
    """A line under the whole figure — what to look for, in words.

    WRAPPED to the figure width. `supxlabel` does not wrap and is centred, so a note longer
    than the page ran off BOTH edges and lost its first and last words — the failure is
    symmetric, which makes it easy to miss when skimming.
    """
    import textwrap

    width_in = fig.get_size_inches()[0]
    size = mpl.rcParams["font.size"] * 0.85
    # Same characters-per-inch estimate as the title wrapper, at the note's smaller size.
    chars = max(40, int(width_in * _TITLE_CHARS_PER_INCH * (10.0 / max(size, 1e-6))))
    fig.supxlabel("\n".join(textwrap.wrap(text, chars)), fontsize=size, color=MUTED)


def legend_patches(labels: dict[str, str]) -> list[Any]:
    """Proxy handles for {label: colour}, for plots drawn with bare `hist`/`bar`."""
    from matplotlib.patches import Patch

    return [Patch(facecolor=c, label=lbl, edgecolor="white") for lbl, c in labels.items()]


def legend_below(target: Any, handles: list[Any] | None = None, *, ncol: int | None = None,
                 **kwargs: Any) -> Any:
    """A legend OUTSIDE the data, under the plot — never on top of a point, a bar or a line.

    A legend placed inside the axes (`loc="best"`, `"upper right"`) lands on the data sooner or
    later, because where the data falls changes with every run: the cost figure's legend sat on the
    `tabicl` arms, the lever figure's on the cf = 1 whiskers, the shift figure's on a histogram
    bar. Below the axes nothing is ever drawn, and `constrained_layout` makes room for it.

    `target` is an Axes (legend under that panel) or a Figure (one legend under the whole figure).
    """
    from matplotlib.figure import Figure

    if handles is None:
        axes = target.axes if isinstance(target, Figure) else [target]
        handles, seen = [], set()
        for ax in axes:
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl not in seen and not lbl.startswith("_"):
                    handles.append(h)
                    seen.add(lbl)
    if not handles:
        return None
    ncol = ncol or min(len(handles), 4)
    opts = dict(ncol=ncol, frameon=False, fontsize=mpl.rcParams["legend.fontsize"],
                handlelength=1.6, columnspacing=1.2)
    opts.update(kwargs)
    if isinstance(target, Figure):
        return target.legend(handles=handles, loc="outside lower center", **opts)
    # Offset in POINTS below the axes, clearing the tick labels and the x label: an offset in axes
    # fractions (-0.2) sits on the x label of a short panel and floats far below a tall one.
    from matplotlib.transforms import ScaledTranslation

    has_ticks = bool(target.get_xticklabels()) and target.xaxis.get_visible()
    drop = 6.0 + (14.0 if has_ticks else 0.0) + (13.0 if target.get_xlabel() else 0.0)
    shift = ScaledTranslation(0, -drop / 72.0, target.figure.dpi_scale_trans)
    return target.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                         bbox_transform=target.transAxes + shift, borderaxespad=0.0, **opts)


def place_labels(ax: Any, xs: Any, ys: Any, labels: list[str], *, fontsize: float | None = None,
                 color: str = MUTED, pad_points: float = 5.0) -> None:
    """Name every point of a scatter without two names landing on each other.

    Greedy: each label tries eight positions around its point, in order, and takes the first
    whose box overlaps neither an earlier label nor another point. Twenty-one dataset names placed
    at one fixed offset collided (`lgd_lendingclub` on `hmeq`, `lgd_freddie` under a marker).
    Needs the axes limits to be final, so call it last.
    """
    import numpy as np
    from matplotlib.transforms import Bbox

    fontsize = fontsize or mpl.rcParams["font.size"] * 0.8
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([np.asarray(xs, float), np.asarray(ys, float)]))
    point_boxes = [Bbox.from_extents(x - 4, y - 4, x + 4, y + 4) for x, y in pts]
    placed: list[Any] = []
    offsets = [(1, 0.6), (1, -0.6), (-1, 0.6), (-1, -0.6), (0, 1.3), (0, -1.3), (1.6, 0), (-1.6, 0)]
    for (x, y), text in zip(pts, labels):
        best = None
        for dx, dy in offsets:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
            t = ax.annotate(text, ax.transData.inverted().transform((x, y)),
                            xytext=(dx * pad_points, dy * pad_points), textcoords="offset points",
                            ha=ha, va=va, fontsize=fontsize, color=color)
            box = t.get_window_extent(renderer).expanded(1.05, 1.1)
            clash = any(box.overlaps(b) for b in placed) or any(
                box.overlaps(b) for i, b in enumerate(point_boxes) if not b.contains(x, y))
            if not clash:
                best = (t, box)
                break
            t.remove()
        if best is None:  # nowhere free: keep the first position rather than drop the name
            dx, dy = offsets[0]
            t = ax.annotate(text, ax.transData.inverted().transform((x, y)),
                            xytext=(dx * pad_points, dy * pad_points), textcoords="offset points",
                            ha="left", va="bottom", fontsize=fontsize, color=color)
            best = (t, t.get_window_extent(renderer))
        placed.append(best[1])


def annotate_value(ax: Any, x: float, y: float, text: str, *, color: str = INK) -> None:
    """Put a number on the mark it belongs to. Saves the reader squinting at ticks."""
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(0, 5),
        ha="center", fontsize=mpl.rcParams["font.size"] * 0.8, color=color,
    )


def reference_line(ax: Any, value: float, label: str | None = None, *, orient: str = "v",
                   color: str = REFERENCE, top: bool = True, inline: bool = True) -> None:
    """A published value from the literature, drawn as a labelled dashed line to compare against.

    This is how a figure grounds itself: our measured distribution in the data colours, the value a
    paper reports drawn over it in `REFERENCE` teal with a short label (a number, not a sentence —
    the interpretation belongs in the caption). `orient="v"` draws a vertical line at `value` on the
    x-axis, `"h"` a horizontal one on the y-axis.

    `inline=False` puts the label in the LEGEND instead of beside the line. Use it wherever the
    data reaches the top of the axes: a rotated label there is written across the bars.
    """
    if not inline:
        draw = ax.axvline if orient == "v" else ax.axhline
        draw(value, color=color, ls=(0, (4, 2)), lw=1.1, zorder=2.5, label=label)
        return
    if orient == "v":
        ax.axvline(value, color=color, ls=(0, (4, 2)), lw=1.1, zorder=2.5)
        if label:
            lo, hi = ax.get_ylim()
            y = hi - (hi - lo) * 0.04 if top else lo + (hi - lo) * 0.04
            ax.annotate(label, (value, y), rotation=90, va="top" if top else "bottom", ha="right",
                        fontsize=mpl.rcParams["font.size"] * 0.72, color=color)
    else:
        ax.axhline(value, color=color, ls=(0, (4, 2)), lw=1.1, zorder=2.5)
        if label:
            lo, hi = ax.get_xlim()
            x = hi - (hi - lo) * 0.01
            ax.annotate(label, (x, value), va="bottom", ha="right",
                        fontsize=mpl.rcParams["font.size"] * 0.72, color=color)


def callout(ax: Any, xy: tuple[float, float], text: str, *, xytext: tuple[float, float] = (16, 14),
            color: str = INK) -> None:
    """A short label with a thin leader to the point it describes — for the one mark worth naming."""
    ax.annotate(
        text, xy=xy, xytext=xytext, textcoords="offset points",
        fontsize=mpl.rcParams["font.size"] * 0.78, color=color,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                        connectionstyle="arc3,rad=0.2"),
    )


def bar_value_labels(ax: Any, bars: Any, fmt: str = "{:.2f}", *, color: str = INK,
                     horizontal: bool = False) -> None:
    """Write each bar's value at its end, so the reader never has to trace back to an axis tick."""
    size = mpl.rcParams["font.size"] * 0.72
    for bar in bars:
        if horizontal:
            w = bar.get_width()
            ax.annotate(fmt.format(w), (w, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", va="center", ha="left",
                        fontsize=size, color=color)
        else:
            h = bar.get_height()
            ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=size, color=color)


def show_palette() -> Any:
    """A swatch of the palette, so a notebook can document its own colour meanings."""
    apply()
    entries = [
        ("our prior\n(credit)", CREDIT),
        ("original\nTabICL prior", ORIGINAL),
        ("real credit\ndata", REAL),
        ("real-data\nmarker", STAR),
        ("library\n(tfm-library)", REFERENCE),
        ("external\n(domain)", EXTERNAL),
        ("full recovery\n(atom at 0)", ATOM_LO),
        ("total loss\n(atom at 1)", ATOM_HI),
        ("out of\nrange", WARN),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 1.05))
    for i, (label, colour) in enumerate(entries):
        ax.add_patch(plt.Rectangle((i, 0), 0.85, 1, color=colour))
        ax.text(i + 0.425, -0.28, label, ha="center", va="top",
                fontsize=mpl.rcParams["xtick.labelsize"] * 0.85, color=MUTED)
    ax.set_xlim(-0.1, len(entries))
    ax.set_ylim(-1.4, 1)
    ax.axis("off")
    ax.set_title("What the colours mean")
    return fig
