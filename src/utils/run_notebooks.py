"""Execute every notebook in parallel and collect its figures and text summary.

WHAT IT PRODUCES, all under `output/`:

    output/figures/<notebook>/fig01_....pdf     vector, print quality
    output/figures/<notebook>/fig01_....png     raster, small enough to commit
    output/figures/CAPTIONS.md                  ONE file, ordered by notebook
    output/All_Results.md                       every notebook's text summary

Each notebook's figure folder is **wiped** before it runs, so only the current run's
figures survive. Stale PDFs mixed with fresh ones is how a paper ends up with a figure
that no longer matches the code that made it.

Notebooks run in separate processes, in parallel. Separate processes rather than threads
because matplotlib's state is global — two notebooks in one interpreter would capture
each other's figures.

CAPTIONS ARE PURE DESCRIPTION. They say what is plotted, on what axes, from how much
data. No interpretation, no conclusions, no "this shows that" — a journal caption
describes the figure so it can be read on its own, and the argument belongs in the body
text. Caption text lives in `FIGURE_CAPTIONS`, keyed by the function that drew it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import REPO_ROOT, all_results_path, captions_path, figures_dir

#: Notebooks to run, alphabetical — which is also the order they appear in the shared
#: CAPTIONS.md and All_Results.md.
NOTEBOOKS = (
    "data_exploration",
    "prior_visualisation_lgd",
    "prior_visualisation_pd",
)

#: DPI for the committed PNGs. Large enough to read on screen, small enough that a
#: couple of dozen of them do not bloat the repository.
PNG_DPI = 110
#: DPI for the PDFs. Vector already, but raster elements (heatmaps, scatter) need this.
PDF_DPI = 300


@dataclass
class NotebookResult:
    name: str
    ok: bool
    seconds: float
    n_figures: int
    error: str = ""


#: Journal-style captions: what is plotted, nothing more. Keyed by a substring of the
#: drawing function's name.
FIGURE_CAPTIONS: dict[str, str] = {
    "show_palette": (
        "Colour key. Grey denotes the unmodified TabICL prior, blue the credit-targeted "
        "prior, orange values measured from the real datasets, and red out-of-range or "
        "flagged values."
    ),
    "lgd_targets": (
        "Histograms of the Loss Given Default target for each of the seven LGD "
        "datasets, ordered by sample size. Forty bins per panel. Dashed vertical lines "
        "mark the minimum and maximum observed values where more than 1% of "
        "observations fall exactly on them. Panel subtitles give the combined share of "
        "observations at the two boundaries."
    ),
    "boundary_mass_ranking": (
        "Share of observations lying exactly at a boundary of the observed target range, "
        "per LGD dataset, ordered by total. Bars are split into mass at the minimum "
        "(blue) and at the maximum (orange). Percentages give the total per dataset."
    ),
    "pd_base_rates": (
        "Default rate per PD dataset, ordered by rate, on a logarithmic horizontal axis. "
        "The dashed vertical line marks a 50% rate. Percentages give the rate per "
        "dataset."
    ),
    "shapes": (
        "Number of rows against number of features for all 21 evaluation datasets, both "
        "axes logarithmic. Colour denotes task; each point is labelled with its dataset "
        "name."
    ),
    "type_mix": (
        "Share of columns that are categorical, per dataset, ordered by share. Colour "
        "denotes task."
    ),
    "missingness": (
        "Share of cells that are missing, per dataset, ordered by share, measured after "
        "preprocessing. Colour denotes task."
    ),
    "feature_correlations": (
        "Pearson correlation matrices between features for the six largest datasets of "
        "the task, computed on the first 5,000 rows with constant columns removed. "
        "Colour scale spans -1 to 1. Panel subtitles give the number of columns "
        "retained and the mean absolute off-diagonal correlation."
    ),
    "boundary_mass_by_variant": (
        "Left: distribution of total boundary mass per synthetic dataset, one step "
        "histogram per prior variant, 30 bins. Dotted vertical lines mark values "
        "measured from the real datasets. Right: mass at the low boundary against mass "
        "at the high boundary, one point per synthetic dataset; stars mark the real "
        "datasets. Legend gives the mean per variant."
    ),
    "base_rate_by_variant": (
        "Distribution of the positive-class rate per synthetic dataset, one step "
        "histogram per prior variant, 30 bins. The dashed vertical line marks a 50% "
        "rate; dotted lines mark rates measured from the real datasets. Legend gives "
        "the mean per variant."
    ),
    "target_shapes_by_variant": (
        "Histograms of the target for ten synthetic datasets per prior variant, one "
        "variant per row, 25 bins per panel. Rows use the same random draw, so panels "
        "in the same column are directly comparable."
    ),
    "spectrum_by_variant": (
        "Eigenvalue spectra of the feature correlation matrix for up to 40 synthetic "
        "datasets per prior variant, normalised by the largest eigenvalue and plotted "
        "against normalised eigenvalue rank. Faint lines are individual datasets; bold "
        "lines are the per-variant median."
    ),
    "shapes_by_variant": (
        "Left: distribution of rows per synthetic dataset. Right: distribution of "
        "features per synthetic dataset. One step histogram per prior variant, 20 bins."
    ),
    "target_grid": (
        "Histograms of the target for 100 synthetic datasets from one prior variant, "
        "30 bins per panel. Axes are suppressed."
    ),
    "boundary_mass": (
        "Left: mass at the low boundary against mass at the high boundary, one point "
        "per synthetic dataset, with real datasets marked as stars. Right: distribution "
        "of the total boundary mass, 25 bins, with the mean marked by a dashed line."
    ),
    "table_shapes": (
        "Distributions of rows per dataset, features per dataset, and the ratio of "
        "distinct target values to rows, across sampled synthetic datasets. 25 bins "
        "each; dashed lines mark the medians given in the panel titles."
    ),
    "feature_relationships": (
        "Pearson correlation matrices between features for individual synthetic "
        "datasets, one panel each. Colour scale spans -1 to 1. Panel subtitles give the "
        "number of features and the mean absolute off-diagonal correlation."
    ),
    "correlation_spectrum": (
        "Eigenvalue spectra of the feature correlation matrix for up to 60 synthetic "
        "datasets, normalised by the largest eigenvalue and plotted against normalised "
        "eigenvalue rank. Faint lines are individual datasets; the bold line is the "
        "median."
    ),
    "feature_target_relation": (
        "Target against the most strongly correlated feature, one panel per synthetic "
        "dataset. Dotted horizontal lines mark the minimum and maximum target values. "
        "Panel subtitles give the feature index and its Pearson correlation with the "
        "target."
    ),
}


#: Functions that dispatch to another plotter and never draw anything themselves, so
#: the figure they produce is captioned by the function they delegate to.
DISPATCHERS = ("plot_target_comparison",)


def caption_for(figure_name: str, index: int) -> str:
    """Caption for a figure, matched on the name the notebook gave it."""
    for key, text in FIGURE_CAPTIONS.items():
        if key in figure_name:
            return text
    return (
        f"No caption registered for {figure_name!r} — add one to FIGURE_CAPTIONS in "
        f"src/utils/run_notebooks.py."
    )


def _prelude() -> str:
    """Injected at the top of each notebook: headless backend, capture printed output.

    Figures are NOT captured here any more. The notebooks save their own via
    `src.visualize.figures.FigureSaver`, so an interactive Jupyter run produces exactly
    the same files as this runner. Monkey-patching matplotlib only ever worked in the
    runner, which meant Run All in Jupyter drew figures and saved nothing.
    """
    return """
import matplotlib
matplotlib.use("Agg")
import io as _io
from contextlib import redirect_stdout as _redirect

# Everything the notebook prints is captured, so All_Results.md can be assembled
# without the notebook knowing about it.
_TEXT = _io.StringIO()
"""


def _build_script(nb_path: Path, text_path: Path) -> str:
    """Flatten a notebook's code cells into one script with the capture prelude.

    A plain script rather than a Jupyter kernel: nothing extra to install, and a
    traceback points at readable line numbers instead of a cell index.
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts = [_prelude(), "\nwith _redirect(_TEXT):\n"]
    for i, cell in enumerate(nb["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        src = re.sub(r"^\s*%.*$", "", "".join(cell["source"]), flags=re.M)
        # Indented under the stdout redirect, so printed output reaches All_Results.md.
        body = "\n".join(f"    {line}" for line in src.split("\n"))
        parts.append(f"\n    # ---- cell {i} ----\n{body}\n")
    # The notebook has already written its own figures and manifest. All this needs to do
    # is persist what it printed, so All_Results.md can be assembled.
    parts.append(
        f'\nimport pathlib as _pl\n'
        f'_pl.Path(r"{text_path}").write_text(_TEXT.getvalue(), encoding="utf-8")\n'
    )
    return "".join(parts)


def run_one(name: str, timeout: int = 1800) -> NotebookResult:
    """Execute one notebook in a fresh process. Never raises."""
    import time

    started = time.time()
    # The figure folder is NOT wiped here. The notebook's own setup cell does it, so the
    # behaviour is identical whether it runs here or interactively in Jupyter — and only
    # ever that one notebook's folder.
    out_dir = figures_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    nb_path = REPO_ROOT / "notebooks" / f"{name}.ipynb"
    if not nb_path.is_file():
        return NotebookResult(name, False, 0.0, 0, f"{nb_path} not found")

    text_path = out_dir / "_stdout.txt"
    # NOT inside out_dir: the notebook clears its own figure folder as its first act, and
    # on Windows a directory cannot be modified while it holds the running script.
    tmp = Path(tempfile.gettempdir()) / f"crediticl_{name}_run.py"
    tmp.write_text(_build_script(nb_path, text_path), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp)], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return NotebookResult(name, False, time.time() - started, 0, f"timed out after {timeout}s")
    finally:
        tmp.unlink(missing_ok=True)

    n_figs = len(list(out_dir.glob("*.pdf")))
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        return NotebookResult(name, False, time.time() - started, n_figs, tail)
    return NotebookResult(name, True, time.time() - started, n_figs)


def _read_capture(name: str) -> dict:
    """What a notebook produced: its figure manifest, and whatever it printed."""
    from src.visualize.figures import read_manifest

    text_path = figures_dir(name) / "_stdout.txt"
    return {
        "figures": read_manifest(name),
        "text": text_path.read_text(encoding="utf-8") if text_path.is_file() else "",
    }


def write_captions(notebooks: tuple[str, ...]) -> Path:
    """One shared CAPTIONS.md, grouped by notebook, figures in notebook order."""
    lines = [
        "# Figure captions",
        "",
        "Generated by `python scripts/run_notebooks.py`. Grouped by notebook, in the",
        "order the figures appear in each notebook. Caption text is maintained in",
        "`FIGURE_CAPTIONS` in `src/utils/run_notebooks.py` — edits here are overwritten.",
        "",
    ]
    for name in notebooks:
        cap = _read_capture(name)
        lines.append(f"## {name}")
        lines.append("")
        if not cap["figures"]:
            lines.append("_No figures produced._")
            lines.append("")
            continue
        for e in cap["figures"]:
            lines.append(f"**{e['stem']}**")
            lines.append("")
            lines.append(caption_for(e["name"], e["index"]))
            lines.append("")
    path = captions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_all_results(notebooks: tuple[str, ...]) -> Path:
    """One file holding every notebook's printed output, in notebook order."""
    lines = [
        "# All results",
        "",
        "Every notebook's printed text summary, concatenated in notebook order.",
        "Generated by `python scripts/run_notebooks.py`.",
        "",
    ]
    for name in notebooks:
        text = _read_capture(name)["text"].strip()
        lines.append("---")
        lines.append("")
        lines.append(f"# {name}")
        lines.append("")
        lines.append("```")
        lines.append(text or "(no output captured)")
        lines.append("```")
        lines.append("")
    path = all_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cleanup_intermediate(notebooks: tuple[str, ...]) -> None:
    """Remove the runner's scratch files once folded into the two summaries.

    `_figures.json` is KEPT: it is the notebook's own record of what it drew and in what
    order, and CAPTIONS.md must be rebuildable after an interactive run without
    re-executing anything.
    """
    for name in notebooks:
        (figures_dir(name) / "_stdout.txt").unlink(missing_ok=True)


def run_all(notebooks: tuple[str, ...] = NOTEBOOKS, max_workers: int | None = None) -> list[NotebookResult]:
    """Run every notebook in parallel, then write CAPTIONS.md and All_Results.md."""
    workers = max_workers or min(len(notebooks), 4)
    results: list[NotebookResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, nb): nb for nb in notebooks}
        for fut in as_completed(futures):
            results.append(fut.result())

    write_captions(notebooks)
    write_all_results(notebooks)
    _cleanup_intermediate(notebooks)
    return sorted(results, key=lambda r: notebooks.index(r.name))


def summarise(results: list[NotebookResult]) -> str:
    lines = ["", "=" * 74, "NOTEBOOK RUN SUMMARY", "=" * 74]
    for r in results:
        lines.append(
            f"  {'OK    ' if r.ok else 'FAILED'} {r.name:<28} "
            f"{r.seconds:6.1f}s  {r.n_figures:2d} figures"
        )
        if not r.ok:
            lines.extend(f"           {line}" for line in r.error.splitlines())
    ok = sum(1 for r in results if r.ok)
    lines += [
        "",
        f"{ok}/{len(results)} notebooks OK, {sum(r.n_figures for r in results)} figures",
        f"  figures  -> {figures_dir()}",
        f"  captions -> {captions_path()}",
        f"  summaries-> {all_results_path()}",
    ]
    return "\n".join(lines)
