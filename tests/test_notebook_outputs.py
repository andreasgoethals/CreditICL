"""The output contract: one root, one captions file, one results file, PDF only.

These enforce the template's rules about generated output, because each one has already
been broken once: figures landed in three different places, captions were written per
folder instead of once, a `figures/` gitignore rule silently swallowed `output/figures/`
as well, and every figure was drawn about twice as wide as an A4 text block.

Captions are read out of the notebook source rather than from a central registry, which
is where they now live — passed to `FigureSaver.save(..., caption=...)` so a caption sits
next to the figure it describes and cannot go stale when that figure is renamed. Reading
the `.ipynb` means these checks run without executing anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.utils import paths  # noqa: E402
from src.utils.run_notebooks import discover  # noqa: E402
from src.visualize import figures, style  # noqa: E402

NOTEBOOKS = discover()

#: `caption=("a" "b")` implicit concatenation across lines, or `caption="one line"`.
_CAPTION_CALL = re.compile(r'caption=\(?\s*((?:"[^"]*"\s*)+)\)?', re.S)
_SAVE_CALL = re.compile(r"FIGS\.save\(")


def _code(name: str) -> str:
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _captions(name: str) -> list[str]:
    """Every caption string in one notebook, un-concatenated."""
    found = []
    for match in _CAPTION_CALL.finditer(_code(name)):
        found.append("".join(re.findall(r'"([^"]*)"', match.group(1))))
    return found


def _all_captions() -> dict[str, str]:
    return {f"{nb}[{i}]": c for nb in NOTEBOOKS for i, c in enumerate(_captions(nb))}


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


def test_results_are_the_one_part_of_output_on_project_storage():
    """Per-row predictions across every arm and seed reach gigabytes, and `$VSC_DATA` is
    75 GiB. This shipped wrong once: `results_dir()` returned `outputs_dir()/results`, so
    the largest files would have gone to the small tier and filled it."""
    import importlib

    import src.utils.paths as p

    original = dict(__import__("os").environ)
    try:
        __import__("os").environ["CREDITICL_STAGING_ROOT"] = str(ROOT / "_probe_staging")
        importlib.reload(p)
        assert "_probe_staging" in p.results_dir().as_posix(), (
            "results/ must move to project storage when staging is configured"
        )
        assert "_probe_staging" not in p.logs_dir().as_posix(), (
            "logs/ are small and must stay on the browsable, backed-up tier"
        )
    finally:
        __import__("os").environ.clear()
        __import__("os").environ.update(original)
        importlib.reload(p)


def test_captions_and_summary_are_committed_but_pdfs_are_not():
    """A `figures/` rule without a leading slash previously matched `output/figures/` too
    and ignored everything under it."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "output/**/*.pdf" in gitignore
    for anchored in ("/figures/", "/results/", "/logs/", "/res/"):
        assert anchored in gitignore, (
            f"{anchored} must be anchored with a leading slash, or it also matches "
            f"output{anchored}"
        )


# -- figures are PDF, and sized for the page ---------------------------------


def test_figures_are_pdf_only():
    """The notebook displays each figure inline, so there is no second raster copy on disk
    to go stale. We wrote PNGs alongside for a while; the template is explicit that the
    inline render is the only raster there should be."""
    text = Path(figures.__file__).read_text(encoding="utf-8")
    assert 'format="pdf"' in text
    assert "png" not in text.lower(), "figures.py must not write a raster copy"
    assert figures._OWNED == ("*.pdf", "_figures.json", "_stdout.txt")


def test_no_png_is_produced_under_output():
    assert not list(paths.outputs_dir().rglob("*.png")), (
        "a PNG appeared under output/ — figures are PDF only"
    )


def test_every_saved_pdf_fits_the_a4_text_block():
    """The failure this catches: figures were drawn 11-13 inches wide against a 6.30 inch
    text block, so the document would scale them to ~50% and take 9pt text to 4.5pt —
    under the ~7pt floor for print. Skips when nothing has been run yet."""
    pdfs = sorted(paths.figures_dir().rglob("*.pdf"))
    if not pdfs:
        pytest.skip("no figures on disk; run python -m src.utils.run_notebooks")
    box = re.compile(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for pdf in pdfs:
        match = box.search(pdf.read_bytes()[:4000])
        assert match, f"{pdf.name}: no MediaBox"
        x0, y0, x1, y1 = (float(v) for v in match.groups())
        width, height = (x1 - x0) / 72, (y1 - y0) / 72
        assert width <= style.WIDTH_FULL + 0.05, (
            f"{pdf.name} is {width:.2f} in wide; the A4 text block is {style.WIDTH_FULL} in"
        )
        assert height <= style.MAX_HEIGHT + 0.05, (
            f"{pdf.name} is {height:.2f} in tall; the ceiling is {style.MAX_HEIGHT} in"
        )


def test_no_plot_function_hard_codes_a_figure_size():
    """A literal `figsize=(12, 4.6)` is correct on screen and twice the page in print. Sizes
    come from `style.figsize`/`grid_figsize`/`row_figsize` so the width can only be A4."""
    offenders = []
    for module in sorted((ROOT / "src" / "visualize").glob("*_plots.py")):
        for n, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"figsize=\(\s*[\d.]", line):
                offenders.append(f"{module.name}:{n}")
    assert not offenders, f"hard-coded figure sizes: {offenders}"


def test_a4_widths_are_the_template_values():
    """160 mm text block on A4. If these drift, every figure in the paper is the wrong size."""
    assert style.WIDTH_FULL == 6.30
    assert style.MAX_HEIGHT == 4.80
    assert style.figsize(style.WIDTH_FULL, 2.0)[1] == style.MAX_HEIGHT, "height must clamp"


# -- captions ----------------------------------------------------------------


def test_every_save_call_passes_a_caption():
    """A figure with no caption reaches CAPTIONS.md as a placeholder, which is easy to miss.
    The counts must match: one caption per saved figure."""
    for name in NOTEBOOKS:
        code = _code(name)
        n_saves = len(_SAVE_CALL.findall(code))
        n_captions = len(_captions(name))
        assert n_saves, f"{name} saves no figures"
        assert n_saves == n_captions, (
            f"{name}: {n_saves} save calls but {n_captions} captions"
        )


def test_captions_are_description_not_interpretation():
    """A journal caption describes the figure. The argument belongs in the body text, so
    interpretive phrasing in a caption is a defect."""
    banned = [
        "this shows", "demonstrates", "confirms", "proves", "the reason",
        "which is why", "motivating", "worth noting", "note that", "importantly",
        "suggests", "indicates that", "we can see",
    ]
    for key, text in _all_captions().items():
        low = text.lower()
        for phrase in banned:
            assert phrase not in low, f"caption {key} interprets rather than describes: {phrase!r}"


def test_captions_say_what_is_plotted():
    """Every caption should name the quantity or the plot type; a caption that only
    editorialises tells a reader nothing about the axes."""
    concrete = re.compile(
        r"histogram|distribution|against|per |axis|axes|bins|colour|matri|spectra|"
        r"share|rate|number of|scale|key",
        re.I,
    )
    for key, text in _all_captions().items():
        assert concrete.search(text), f"caption {key} does not say what is plotted"


def test_captions_are_a_full_sentence():
    """A two-word caption passes the checks above and is still useless in a manuscript."""
    for key, text in _all_captions().items():
        assert len(text) >= 40, f"caption {key} is too short to be the paper's caption: {text!r}"


# -- notebook rules ----------------------------------------------------------


#: A DEFINITION at the start of a line — not the substring. Matching `"class "` anywhere
#: flagged the caption phrase "positive-class rate", which is prose, not logic.
_DEFINITION = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w", re.M)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_exists_and_holds_no_logic(name):
    """A function defined in a notebook cannot be imported or tested, so it gets copied
    into the next notebook and the copies diverge."""
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for cell in code:
        found = _DEFINITION.findall("".join(cell["source"]))
        assert not found, f"{name}: logic belongs in src/, not the notebook ({found})"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_ends_with_a_printed_text_summary(name):
    """The template rule: the final output is copy-pasteable text, not a figure."""
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    last = "".join(code[-1]["source"])
    assert "print(" in last and "summar" in last.lower(), (
        f"{name} must end by printing a text summary"
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_applies_the_shared_style_and_picks_no_colour_itself(name):
    """One style per project, defined in style.py. A hex literal in a notebook is a colour
    chosen locally, which is how twenty figures end up with three different blues."""
    code = _code(name)
    assert "style.apply()" in code, f"{name} must call style.apply()"
    hexes = re.findall(r'"#[0-9a-fA-F]{6}"', code)
    assert not hexes, f"{name} picks its own colours: {hexes} — add them to style.py"


def test_notebooks_are_discovered_not_listed():
    """A hard-coded tuple silently stops covering a notebook someone added."""
    import inspect

    from src.utils import run_notebooks

    source = inspect.getsource(run_notebooks.discover)
    assert "glob" in source and "sorted" in source
    on_disk = tuple(sorted(p.stem for p in (ROOT / "notebooks").glob("*.ipynb")))
    assert on_disk == NOTEBOOKS


def test_notebooks_save_their_own_figures():
    """The defect this replaced: capture lived only in the runner, so Run All in Jupyter
    drew figures and saved nothing."""
    for name in NOTEBOOKS:
        code = _code(name)
        assert "figures.FigureSaver(" in code, f"{name} does not clear its own folder"
        assert "FIGS.save(" in code, f"{name} does not save its figures"


# -- the runner ---------------------------------------------------------------


def test_a_notebook_clears_only_its_own_figures(isolated_output):
    """Running one notebook must never delete another's work, and the clearing must
    happen BEFORE anything is drawn.

    Uses `isolated_output`, not a bare staging override: `figures_dir()` hangs off
    `outputs_dir()`, which ignores staging, so the earlier version of this test wrote its
    two fake PDFs into the REAL output tree and left them there — where the A4 size check
    then found a 1-byte file with no MediaBox.
    """
    import importlib

    import src.utils.paths as p

    importlib.reload(p)
    import src.visualize.figures as figs

    importlib.reload(figs)

    other = p.figures_dir("other_notebook")
    other.mkdir(parents=True, exist_ok=True)
    (other / "01_keep.pdf").write_bytes(b"x")

    mine = p.figures_dir("mine")
    mine.mkdir(parents=True, exist_ok=True)
    (mine / "01_stale.pdf").write_bytes(b"x")

    figs.FigureSaver("mine")  # constructing it is what clears

    assert not (mine / "01_stale.pdf").exists(), "own stale figures must be removed"
    assert (other / "01_keep.pdf").exists(), "another notebook's figures were deleted"
    importlib.reload(p)
    importlib.reload(figs)


def test_a_figure_name_cannot_escape_its_folder(tmp_path):
    """`..` in a figure name would put a generated file outside output/, the one rule the
    whole layout rests on."""
    from src.visualize import figures as figs

    # `_slug` turns a path separator into an underscore, so the guard should never fire in
    # practice. Assert both: the slug is the protection, the guard is the backstop.
    assert "/" not in figs._slug("../../etc/passwd")
    assert "\\" not in figs._slug(r"..\..\windows")
    with pytest.raises(ValueError):
        figs._guard(tmp_path / "elsewhere" / "x.pdf", tmp_path / "folder")


def test_runner_lives_in_utils_not_visualize():
    """It orchestrates processes and files; it draws nothing."""
    assert (ROOT / "src" / "utils" / "run_notebooks.py").is_file()
    assert not (ROOT / "src" / "visualize" / "run_notebooks.py").exists()


def test_scripts_folder_holds_only_experiments():
    """`scripts/` is the experiments — the things submitted to the cluster. A utility
    (cleanup, the notebook runner, the submodule pin) belongs in `src/utils/`, invoked with
    `python -m`, or it buries the two files that matter among eight that do not."""
    offenders = []
    for path in (ROOT / "scripts").glob("*.py"):
        if "if __name__" not in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"not runnable, move to src/: {offenders}"

    utilities = {"clean_run.py", "run_notebooks.py", "vendor_model.py", "update_tfm_library.py"}
    present = {p.name for p in (ROOT / "scripts").glob("*.py")} & utilities
    assert not present, f"utilities belong in src/utils/, not scripts/: {sorted(present)}"


def test_every_notebook_cell_compiles():
    """REGRESSION. A `{{...}}` left over from a `.format()` template shipped in the LGD
    notebook, reached the cluster, and only failed at RUN time with `unhashable type: 'dict'`
    — because `{{k: v}}` is valid Python (a set containing a dict) right up until it executes.

    Compiling every cell is cheap and would have caught it before the commit. Magics are
    stripped, exactly as `run_notebooks` strips them.
    """
    for name in NOTEBOOKS:
        nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
            source = "".join(cell["source"])
            clean = "\n".join(
                "" if line.lstrip().startswith(("%", "!")) else line
                for line in source.split("\n")
            )
            try:
                compile(clean, f"{name}#cell{i}", "exec")
            except SyntaxError as exc:
                raise AssertionError(f"{name} cell {i} does not compile: {exc}") from exc


def test_no_notebook_carries_doubled_braces():
    """The specific shape of the bug above. `{{` only ever appears in a notebook because a
    `.format()` template was written out without being formatted, and it is invisible in review
    because the cell still parses."""
    for name in NOTEBOOKS:
        code = _code(name)
        assert "{{" not in code and "}}" not in code, (
            f"{name} contains doubled braces — a .format() template leaked into the notebook"
        )


def test_the_shell_hook_is_a_real_file_not_a_paste_from_the_docs():
    """A 25-line function pasted into `~/.bashrc` over SSH is error-prone AND stops receiving
    fixes. The hook is a file in the repo with a one-command installer, so `git pull` updates it
    and `~/.bashrc` carries a single line."""
    hook = ROOT / "scripts" / "slurm" / "shell_hook.sh"
    assert hook.is_file(), "scripts/slurm/shell_hook.sh is missing"
    text = hook.read_text(encoding="utf-8")
    for flag in ("--install", "--uninstall", "--status"):
        assert flag in text, f"{flag} is not handled"
    # It must be safe to SOURCE as well as run: the installer body has to be guarded.
    assert 'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then' in text, (
        "the installer must not run when the file is sourced"
    )
    # And it must stand down another project's venv, which is what actually went wrong.
    assert "deactivate" in text
    doc = (ROOT / "docs" / "VSC.md").read_text(encoding="utf-8")
    assert "shell_hook.sh --install" in doc, "the docs must point at the installer"


# -- a figure is the plot; the caption explains it -----------------------------


def test_no_figure_carries_prose():
    """THE RULE FOR A PAPER: the figure carries data, axis labels and at most a short heading.
    Interpretation goes in the caption and the body text, where it can be edited without
    re-rendering. `style.figure_note` was the mechanism for prose-in-figure and every call is
    gone; keeping it out is what this test is for."""
    offenders = []
    for module in sorted((ROOT / "src" / "visualize").glob("*.py")):
        if module.name == "style.py":
            continue
        for n, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            if "style.figure_note(" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{module.name}:{n}")
    assert not offenders, (
        f"prose in a figure: {offenders}. Put it in the caption, or in a markdown cell above "
        f"the figure in the notebook."
    )


def test_no_figure_heading_is_a_sentence():
    """A heading is a label, not a claim. "Real PD datasets are imbalanced; the original prior
    is not" wrapped to three bold lines and ate a third of the figure."""
    import re as _re

    def visible_length(literal: str) -> int:
        """Length of what a READER sees. An f-string's `{...}` expressions are source, not
        text — `f"Target shapes{style.page_suffix(page, n_pages)}"` renders to 27 characters
        while its source is 46, and measuring the source failed a heading that was fine."""
        return len(_re.sub(r"\{[^}]*\}", "", literal.strip('f').strip('"')))

    long_headings = []
    for module in sorted((ROOT / "src" / "visualize").glob("*_plots.py")):
        text = module.read_text(encoding="utf-8")
        patterns = (r"style\.title\([^,]+,\s*(f?\"[^\"]{0,200}\")", r"suptitle\(\s*(f?\"[^\"]{0,200}\")")
        for pattern in patterns:
            for call in _re.finditer(pattern, text):
                if visible_length(call.group(1)) > 42:
                    long_headings.append(f"{module.name}: {call.group(1)!r}")
    assert not long_headings, f"headings too long for a figure: {long_headings}"


def test_every_dataset_is_shown_not_just_the_largest():
    """`plot_feature_correlations` took the 6 largest and dropped the rest, so a figure claiming
    to describe "real credit data" showed 6 of 14 datasets. It paginates now."""
    from src.visualize import data_plots

    assert hasattr(data_plots, "correlation_pages")
    fake = {f"{i:04d}.d{i}": type("D", (), {"n_rows": 100 - i, "n_features": 4})() for i in range(14)}
    assert data_plots.correlation_pages(fake, per_page=6) == 3


def test_reference_markers_are_not_the_same_colour_as_the_data():
    """Orange stars over orange dots are invisible, which is what the LGD boundary-mass figure
    did. `style.STAR` appears nowhere else in the palette."""
    assert style.STAR != style.REAL
    assert style.STAR not in style.SERIES
    assert style.STAR not in (style.CREDIT, style.ORIGINAL, style.WARN)


def test_the_row_cap_samples_randomly_not_from_the_head():
    """It reported `algorithmwatch`'s default rate as 49.5% against a true 37.8%: the rows are
    not shuffled on disk, so the first 20,000 encode whatever the file's ordering encodes."""
    import inspect

    from src.visualize import summaries

    source = inspect.getsource(summaries._RowCapped)
    assert "default_rng" in source and "choice" in source, "the cap must subsample randomly"
    assert "idx.sort()" in source, "row order must be preserved for cohort-style plots"


# -- cluster targets -----------------------------------------------------------


def test_no_job_script_hard_codes_the_most_contended_partition():
    """`gpu_a100` has 16 GPUs — the fewest available to us — and the first debug submission sat
    there behind "Reason: Priority" without starting. Defaults point at the free partition;
    `submit.sh` chooses anything else on the command line, which overrides `#SBATCH`."""
    debug = (ROOT / "scripts" / "slurm" / "debug_exp1.slurm").read_text(encoding="utf-8")
    active = [ln for ln in debug.splitlines() if ln.startswith("#SBATCH --partition")]
    assert active == ["#SBATCH --partition=interactive"], active


def test_the_submitter_knows_every_target_and_its_core_budget():
    """Cores per GPU is the real constraint — TabICL generates its prior on the CPU — and the
    VSC limits differ per partition (8 / 18 / 16 / 24). Requesting more than the limit earns a
    warning and requesting fewer wastes the allocation."""
    text = (ROOT / "scripts" / "slurm" / "submit.sh").read_text(encoding="utf-8")
    for target in ("free", "b200", "a100", "h100", "dbg1h"):
        assert f"{target})" in text, f"submit.sh does not handle {target}"
    # The documented per-GPU core limits, from the VSC docs.
    for partition, cores in (("interactive", 8), ("gpu_b200", 24), ("gpu_a100", 18),
                             ("gpu_h100", 16)):
        block = text[text.index(f"--partition={partition}"):]
        assert f"--cpus-per-task={cores}" in block[:200], (
            f"{partition} should request {cores} cores per GPU"
        )


def test_the_job_takes_its_worker_count_from_the_allocation():
    """One script runs on 8 cores and on 24. A fixed `num_workers` oversubscribes the small
    allocation and leaves half the big one idle."""
    debug = (ROOT / "scripts" / "slurm" / "debug_exp1.slurm").read_text(encoding="utf-8")
    assert "SLURM_CPUS_PER_TASK" in debug
    assert "--num-workers" in debug


def test_the_debug_job_evaluates_the_task_it_trained():
    """It hard-coded `--task lgd`, so a `CONFIG=config/Exp1_PD.yaml` run would have trained PD
    and evaluated LGD — passing, while measuring the wrong thing."""
    debug = (ROOT / "scripts" / "slurm" / "debug_exp1.slurm").read_text(encoding="utf-8")
    assert "--task lgd" not in debug
    assert '--task "$TASK"' in debug
    assert "*_PD.yaml) TASK=pd" in debug


def test_the_job_config_is_an_argument_not_an_environment_variable():
    """`CONFIG=x bash submit.sh` sets the variable for the CALLING shell, not for sbatch's
    environment — so the job never received it, and a run intended as PD went out as a second
    LGD job with nothing on screen to say so. `sbatch [opts] script args...` passes trailing
    arguments straight to the script, which needs no plumbing and cannot be lost."""
    job = (ROOT / "scripts" / "slurm" / "debug_exp1.slurm").read_text(encoding="utf-8")
    assert 'CONFIG="${1:-config/Exp1_LGD.yaml}"' in job, "the config must be argument 1"
    assert 'CONFIG="${CONFIG:-' not in job, "the env-var form is what broke"

    sub = (ROOT / "scripts" / "slurm" / "submit.sh").read_text(encoding="utf-8")
    assert 'TRACK="${2:-lgd}"' in sub, "submit.sh must take the track as an argument"
    # and every input that changes what runs must be printed before submitting
    for label in ('echo "where', 'echo "track', 'echo "config', 'echo "script'):
        assert label in sub, f"{label} is not reported before submission"


def test_changelog_is_one_chapter_per_date():
    """`CHANGELOG.md` says "One chapter per date, DD-MM-YYYY, newest first" in its own header.

    It had drifted to 27 chapters over 12 dates, with invented time-of-day labels —
    `(evening)`, `(late night)`, `(night, later)` — several of which were simply wrong about
    the time of day. A file that does not obey the rule written at the top of it is not a
    reliable record.
    """
    import re
    from datetime import datetime

    text = (ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.*)$", text, re.M)
    assert headings, "no chapters found — has the format changed?"

    dates = []
    for h in headings:
        m = re.match(r"^(\d{2}-\d{2}-\d{4})\b", h)
        assert m, f"chapter heading must start with a DD-MM-YYYY date: {h!r}"
        assert not re.match(r"^\d{2}-\d{2}-\d{4}\s*\(", h), (
            f"no time-of-day label on a chapter heading — one chapter per DATE: {h!r}"
        )
        dates.append(datetime.strptime(m.group(1), "%d-%m-%Y"))

    dupes = {d for d in dates if dates.count(d) > 1}
    assert not dupes, (
        f"{len(dupes)} date(s) have more than one chapter: "
        f"{sorted(d.strftime('%d-%m-%Y') for d in dupes)}. Merge them."
    )
    assert dates == sorted(dates, reverse=True), "chapters must be newest first"
