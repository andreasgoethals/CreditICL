"""Saving figures from a notebook, so an interactive run produces the same files.

THE PROBLEM THIS SOLVES

Figure capture used to live entirely in `scripts/run_notebooks.py`, which monkey-patches
matplotlib while executing a notebook as a script. That works for the batch runner and
**not at all** in Jupyter: opening a notebook and pressing Run All produced figures on
screen and nothing on disk. Since Jupyter is how the notebooks are actually read and
iterated on, that was the wrong way round.

So the notebook saves its own figures. `FigureSaver` works identically in Jupyter and
under the runner, which also means the runner no longer has to guess which function drew
what — the notebook names each figure explicitly.

TWO GUARANTEES:

* **A notebook deletes only its OWN figures**, never another notebook's. The folders are
  per notebook and `wipe()` touches exactly one.
* **Deletion happens BEFORE anything is drawn**, in the setup cell, so a figure removed
  from a notebook does not linger from the previous run.

Every figure is written twice: PDF at high DPI for the paper, PNG at lower DPI so it is
small enough to commit and to render inline.
"""

from __future__ import annotations

import json
from typing import Any

from src.utils.paths import figures_dir

#: PDF for print. Vector already, but raster elements (heatmaps, dense scatter) need this.
PDF_DPI = 300
#: PNG for the repository and for inline display. Readable, and small enough to commit.
PNG_DPI = 110


class FigureSaver:
    """Names, numbers and saves a notebook's figures.

    Usage in a notebook's setup cell::

        FIGS = figures.for_notebook("data_exploration")

    then, per plot::

        FIGS.save(data_plots.plot_lgd_targets(lgd), "lgd_targets")

    `save` returns the figure, so Jupyter still displays it inline.
    """

    def __init__(self, notebook: str, wipe: bool = True):
        self.notebook = notebook
        self.dir = figures_dir(notebook)
        self.dir.mkdir(parents=True, exist_ok=True)
        if wipe:
            # Remove the FILES this notebook produces, not the directory. Deleting the
            # directory failed on Windows when the batch runner had a file open inside it,
            # and removing only known extensions means an unrelated file can never be
            # caught. Never touches another notebook's folder.
            for pattern in ("*.pdf", "*.png", "_figures.json"):
                for stale in self.dir.glob(pattern):
                    stale.unlink(missing_ok=True)
        self.entries: list[dict[str, Any]] = []

    def save(self, fig: Any, name: str) -> Any:
        """Write one figure as PDF and PNG. Returns the figure for inline display."""
        if fig is None:
            raise ValueError(
                f"{self.notebook}/{name}: got None instead of a figure. The plotting "
                f"functions all return a Figure — check you are not passing the result "
                f"of something that only draws."
            )
        index = len(self.entries) + 1
        stem = f"fig{index:02d}_{name}"
        fig.savefig(self.dir / f"{stem}.pdf", format="pdf", dpi=PDF_DPI, bbox_inches="tight")
        fig.savefig(self.dir / f"{stem}.png", format="png", dpi=PNG_DPI, bbox_inches="tight")
        self.entries.append({"index": index, "stem": stem, "name": name})
        self._write_manifest()
        return fig

    def _write_manifest(self) -> None:
        """Record order and names, so CAPTIONS.md can be rebuilt without re-running.

        Written after every save rather than once at the end: a notebook abandoned
        half-way should still leave a usable record of what it produced.
        """
        (self.dir / "_figures.json").write_text(
            json.dumps({"notebook": self.notebook, "figures": self.entries}, indent=1),
            encoding="utf-8",
        )

    def summary(self) -> str:
        """One line per figure, for the notebook's closing text summary."""
        if not self.entries:
            return "No figures saved."
        lines = [f"{len(self.entries)} figures written to {self.dir}:"]
        lines += [f"  {e['stem']}.pdf / .png" for e in self.entries]
        return "\n".join(lines)


def for_notebook(notebook: str, wipe: bool = True) -> FigureSaver:
    """Start (and by default clear) a notebook's figure folder."""
    return FigureSaver(notebook, wipe=wipe)


def read_manifest(notebook: str) -> list[dict[str, Any]]:
    """The figures a notebook produced, in order. Empty if it has not run."""
    path = figures_dir(notebook) / "_figures.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("figures", [])
