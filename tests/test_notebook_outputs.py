"""The output contract: one root, one captions file, one results file.

These enforce the template's rules about generated output, because each one has already
been broken once: figures landed in three different places, captions were written per
folder instead of once, and a `figures/` gitignore rule silently swallowed
`output/figures/` as well.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.utils import paths  # noqa: E402
from src.utils.run_notebooks import (  # noqa: E402
    FIGURE_CAPTIONS,
    NOTEBOOKS,
    caption_for,
)

# -- one output root ---------------------------------------------------------


def test_everything_generated_lives_under_output():
    out = paths.outputs_dir()
    for path in (
        paths.results_dir(),
        paths.figures_dir(),
        paths.logs_dir(),
        paths.manifests_dir(),
        paths.all_results_path(),
    ):
        assert out in path.parents or path.parent == out, f"{path} is outside {out}"


def test_no_superseded_output_directories_at_the_repo_root():
    """`res/`, `results/`, `figures/` and `logs/` were all output roots at some point.
    Any of them reappearing means something is writing outside `output/`."""
    for stale in ("res", "results", "figures", "logs"):
        assert not (ROOT / stale).exists(), (
            f"{stale}/ is back at the repo root — everything generated belongs under "
            f"output/. Check src/utils/paths.py."
        )


def test_captions_and_summary_are_committed_but_pdfs_are_not():
    """The PNG carries the picture and is small enough to commit; the PDF is
    regenerated. A `figures/` rule without a leading slash previously matched
    `output/figures/` too and ignored everything."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "output/**/*.pdf" in gitignore
    for anchored in ("/figures/", "/results/", "/logs/", "/res/"):
        assert anchored in gitignore, (
            f"{anchored} must be anchored with a leading slash, or it also matches "
            f"output{anchored}"
        )


# -- captions ----------------------------------------------------------------


def test_captions_are_description_not_interpretation():
    """A journal caption describes the figure. The argument belongs in the body text, so
    interpretive phrasing in a caption is a defect."""
    banned = [
        "this shows", "demonstrates", "confirms", "proves", "the reason",
        "which is why", "motivating", "worth noting", "note that", "importantly",
        "suggests", "indicates that", "we can see",
    ]
    for key, text in FIGURE_CAPTIONS.items():
        low = text.lower()
        for phrase in banned:
            assert phrase not in low, f"caption {key!r} interprets rather than describes: {phrase!r}"


def test_captions_say_what_is_plotted():
    """Every caption should name the quantity or the plot type; a caption that only
    editorialises tells a reader nothing about the axes."""
    concrete = re.compile(
        r"histogram|distribution|against|per |axis|axes|bins|colour|matri|spectra|"
        r"share|rate|number of|scale|key",
        re.I,
    )
    for key, text in FIGURE_CAPTIONS.items():
        assert concrete.search(text), f"caption {key!r} does not say what is plotted"


def test_unknown_figure_gets_an_actionable_placeholder():
    text = caption_for("plot_something_new", 3)
    assert "FIGURE_CAPTIONS" in text and "run_notebooks" in text


def test_every_plot_function_has_a_caption():
    """A figure without a caption reaches CAPTIONS.md as a placeholder, which is easy to
    miss. Check every public plotting function is covered."""
    import src.visualize.data_plots as dp
    import src.visualize.pool_plots as pp
    import src.visualize.prior_plots as prp
    from src.utils.run_notebooks import DISPATCHERS

    uncovered = []
    for module in (dp, pp, prp):
        for name in dir(module):
            if not name.startswith("plot_") or name in DISPATCHERS:
                continue
            if not any(key in name for key in FIGURE_CAPTIONS):
                uncovered.append(f"{module.__name__}.{name}")
    assert not uncovered, f"no caption registered for: {uncovered}"


# -- notebook rules ----------------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_exists_and_holds_no_logic(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for cell in code:
        assert "def " not in "".join(cell["source"]), (
            f"{name}: logic belongs in src/, not the notebook"
        )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_ends_with_a_printed_text_summary(name):
    """The template rule: the final output is copy-pasteable text, not a figure."""
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    last = "".join(code[-1]["source"])
    assert "print(" in last and "summar" in last.lower(), (
        f"{name} must end by printing a text summary"
    )


def test_notebooks_are_listed_alphabetically():
    """CAPTIONS.md and All_Results.md are ordered by this tuple, and the template says
    alphabetical."""
    assert list(NOTEBOOKS) == sorted(NOTEBOOKS)


def test_every_notebook_on_disk_is_registered():
    """An unregistered notebook is never run and never contributes a figure."""
    on_disk = {p.stem for p in (ROOT / "notebooks").glob("*.ipynb")}
    assert on_disk == set(NOTEBOOKS), f"registered {set(NOTEBOOKS)}, on disk {on_disk}"


# -- the runner ---------------------------------------------------------------


def test_a_notebook_clears_only_its_own_figures(tmp_path, monkeypatch):
    """Running one notebook must never delete another's work, and the clearing must
    happen BEFORE anything is drawn."""
    import importlib

    monkeypatch.setenv("CREDITICL_STAGING_ROOT", str(tmp_path))
    monkeypatch.delenv("VSC_DATA", raising=False)
    import src.utils.paths as p

    importlib.reload(p)
    import src.visualize.figures as figs

    importlib.reload(figs)

    other = p.figures_dir("other_notebook")
    other.mkdir(parents=True, exist_ok=True)
    (other / "fig01_keep.pdf").write_bytes(b"x")

    mine = p.figures_dir("mine")
    mine.mkdir(parents=True, exist_ok=True)
    (mine / "fig01_stale.pdf").write_bytes(b"x")
    (mine / "fig01_stale.png").write_bytes(b"x")

    figs.for_notebook("mine")  # constructing it is what clears

    assert not (mine / "fig01_stale.pdf").exists(), "own stale figures must be removed"
    assert (other / "fig01_keep.pdf").exists(), "another notebook's figures were deleted"
    importlib.reload(p)
    importlib.reload(figs)


def test_both_pdf_and_png_are_written():
    """Saving is the NOTEBOOK's job now, not the runner's — that is what makes an
    interactive Jupyter run produce the same files as the batch runner."""
    from src.visualize import figures

    text = Path(figures.__file__).read_text(encoding="utf-8")
    assert 'format="pdf"' in text and 'format="png"' in text
    assert figures.PDF_DPI > figures.PNG_DPI, (
        "the PDF is for print and must be the higher-resolution one"
    )


def test_notebooks_save_their_own_figures():
    """The defect this replaced: capture lived only in the runner, so Run All in Jupyter
    drew figures and saved nothing."""
    for name in NOTEBOOKS:
        nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        body = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert "figures.for_notebook(" in body, f"{name} does not clear its own folder"
        assert "FIGS.save(" in body, f"{name} does not save its figures"


def test_runner_lives_in_utils_not_visualize():
    """It orchestrates processes and files; it draws nothing."""
    assert (ROOT / "src" / "utils" / "run_notebooks.py").is_file()
    assert not (ROOT / "src" / "visualize" / "run_notebooks.py").exists()


def test_scripts_folder_holds_only_runnables():
    """Anything importable belongs in src/. A module in scripts/ cannot be imported by
    the package and cannot be tested."""
    offenders = []
    for p in (ROOT / "scripts").glob("*.py"):
        if "if __name__" not in p.read_text(encoding="utf-8"):
            offenders.append(p.name)
    assert not offenders, f"not runnable, move to src/: {offenders}"
