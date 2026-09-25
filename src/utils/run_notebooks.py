"""Run every notebook in parallel, then rebuild the two summary documents.

    python -m src.utils.run_notebooks                     every notebook in notebooks/
    python -m src.utils.run_notebooks --only 1.3 1.4      just these: a stem, or its start ("1." = chapter 1)
    python -m src.utils.run_notebooks --summaries-only    rebuild the two .md files only

It prints a line as each notebook finishes, and every 30 s names the ones not finished yet.
A partial run (`--only`) keeps every other notebook's block in `All_Results.md`.

    output/figures/<notebook>/*.pdf   written by the notebooks themselves
    output/figures/CAPTIONS.md        ONE file, all notebooks, notebook order
    output/All_Results.md             every notebook's printed summary, alphabetical

SEPARATE PROCESSES, NOT THREADS: matplotlib's figure registry is global, so two notebooks in
one interpreter would capture each other's figures — silently, giving plausible figures
attributed to the wrong notebook.

A REAL JUPYTER KERNEL, AND THE OUTPUTS ARE SAVED INTO THE NOTEBOOK: each notebook runs cell by
cell in a fresh IPython kernel, exactly as *Run All* does, and every output — printed text,
inline figures, errors — is written back into its `.ipynb`, so opening a notebook (or viewing it on
GitHub) shows the run. Until 25-09-2026 the runner executed a flattened copy instead, which left the
notebooks empty (a deviation from the template, made for that reason). `jupyter_client`, which
comes with `ipykernel`, drives the kernel; nothing else is needed.

THE RUNNER DOES NOT SAVE FIGURES; each notebook does, through `FigureSaver`, so an interactive
*Run All* produces exactly the same PDFs. The runner adds parallelism and the two documents.

NOTEBOOKS ARE DISCOVERED, NOT LISTED, alphabetically — which is also the order in both summary
documents. A hard-coded list silently stops covering a notebook someone added.

NOTEBOOKS LIVE IN NUMBERED CHAPTER FOLDERS (`0. General/`, `1. Experiment 1/`, ...), so discovery
recurses. A notebook is still identified by its file STEM alone (`1.1_pd_training`): figures,
`--only` and both summary documents are keyed by it, which is why stems must be unique across
folders and why each carries its chapter number. That keeps `output/figures/<notebook>/` exactly as
the template lays it out, and makes alphabetical order the reading order. (A project deviation from
the template's flat `notebooks/`; the recursion itself is generic and worth upstreaming.)
"""

from __future__ import annotations

import json
import queue
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import (
    REPO_ROOT,
    all_results_path,
    captions_path,
    figures_dir,
    notebooks_dir,
)

#: Per-notebook wall-clock limit. A notebook summarises a finished computation; one needing
#: longer is doing work that belongs in a script.
DEFAULT_TIMEOUT = 1800

#: Captured stdout, parked between execution and assembly, then removed. `_figures.json` is
#: KEPT: CAPTIONS.md must be rebuildable from disk without re-executing anything.
STDOUT_FILE = "_stdout.txt"

#: How often a run with notebooks still going says which ones, so a quiet terminal is never
#: mistaken for a hung one.
HEARTBEAT_SECONDS = 30


@dataclass
class NotebookResult:
    name: str
    ok: bool
    seconds: float
    n_figures: int
    error: str = ""


def _notebook_files() -> list[Path]:
    """Every notebook under `notebooks/`, at any depth, skipping Jupyter's checkpoint copies.

    Raises on a duplicated stem: two notebooks called `x` in different folders would write into the
    same `output/figures/x/` and each would clear the other's figures on construction.
    """
    files = [p for p in notebooks_dir().rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts]
    stems = [p.stem for p in files]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        raise ValueError(
            f"notebook names must be unique across folders — figures and summaries are keyed "
            f"by name alone: {dupes}"
        )
    return files


def discover(names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Notebook stems, alphabetical. `names` overrides discovery for a partial rerun."""
    if names:
        return tuple(names)
    return tuple(sorted(p.stem for p in _notebook_files()))


def select(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """The notebooks each pattern names: a whole stem, or its start (`1.` is chapter 1, `1.3` is
    `1.3_pd_results`). A pattern matching nothing is kept as typed, so the run reports it."""
    stems = discover()
    chosen: list[str] = []
    for pattern in patterns:
        for stem in [s for s in stems if s.startswith(pattern)] or [pattern]:
            if stem not in chosen:
                chosen.append(stem)
    return tuple(chosen)


def notebook_path(name: str) -> Path | None:
    """Where notebook `name` lives — in whichever chapter folder — or `None` if it does not."""
    return next((p for p in _notebook_files() if p.stem == name), None)


def chapter_of(name: str) -> str | None:
    """The chapter folder a notebook sits in (`"1. Experiment 1"`), or `None` at the top level."""
    path = notebook_path(name)
    if path is None or path.parent == notebooks_dir():
        return None
    return path.parent.relative_to(notebooks_dir()).as_posix()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


#: Terminal colour codes in kernel tracebacks — kept in the notebook, stripped from the report.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _run_cell(km, kc, source: str, deadline: float, timeout: int) -> tuple[list[dict], str]:
    """Execute one cell and collect its outputs as nbformat dicts. Returns (outputs, error)."""
    msg_id = kc.execute(source, store_history=True, allow_stdin=False)
    outputs: list[dict] = []
    error = ""
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            km.interrupt_kernel()
            return outputs, f"timed out after {timeout}s"
        try:
            msg = kc.get_iopub_msg(timeout=min(remaining, 5))
        except queue.Empty:
            if not km.is_alive():
                return outputs, "the kernel died"
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind, content = msg["msg_type"], msg["content"]
        if kind == "status" and content.get("execution_state") == "idle":
            return outputs, error
        if kind == "stream":
            last = outputs[-1] if outputs else {}
            if last.get("output_type") == "stream" and last.get("name") == content["name"]:
                last["text"] += content["text"]
            else:
                outputs.append({"output_type": "stream", "name": content["name"],
                                "text": content["text"]})
        elif kind in ("display_data", "execute_result"):
            out = {"output_type": kind, "data": content["data"],
                   "metadata": content.get("metadata", {})}
            if kind == "execute_result":
                out["execution_count"] = content["execution_count"]
            outputs.append(out)
        elif kind == "error":
            outputs.append({"output_type": "error", "ename": content["ename"],
                            "evalue": content["evalue"], "traceback": content["traceback"]})
            error = _ANSI.sub("", "\n".join(content["traceback"]))
        elif kind == "clear_output":
            outputs.clear()


def _execute(nb_path: Path, timeout: int) -> tuple[dict, str, str]:
    """Run a notebook's code cells in a fresh IPython kernel, as *Run All* does.

    Returns (the notebook with its outputs, everything it printed to stdout, an error or "").
    It stops at the first failing cell, like *Run All*; the cells after it are left empty, so the
    saved notebook shows where it broke.
    """
    from jupyter_client.manager import start_new_kernel

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    # `--log-level=ERROR` hides the kernel's own start-up warning that it talks over TCP without
    # encryption — printed once per kernel and harmless here: the ports are bound to localhost and
    # every message is signed with a per-kernel key, as in any Jupyter on Windows. Kernel errors
    # still print; a cell's own warnings go into the notebook, not the terminal.
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(REPO_ROOT), startup_timeout=120,
                              extra_arguments=["--log-level=ERROR"])
    printed: list[str] = []
    error = ""
    deadline = time.time() + timeout
    try:
        count = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            cell["outputs"], cell["execution_count"] = [], None
            source = "".join(cell.get("source", []))
            if error or not source.strip():
                continue
            count += 1
            cell["execution_count"] = count
            cell["outputs"], error = _run_cell(km, kc, source, deadline, timeout)
            printed += [o["text"] for o in cell["outputs"]
                        if o["output_type"] == "stream" and o["name"] == "stdout"]
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    return nb, "".join(printed), error


def run_one(name: str, timeout: int = DEFAULT_TIMEOUT) -> NotebookResult:
    """Execute one notebook in a fresh kernel and save its outputs into it. Never raises."""
    started = time.time()
    nb_path = notebook_path(name)
    if nb_path is None:
        return NotebookResult(name, False, 0.0, 0, f"{name}.ipynb not found under {notebooks_dir()}")

    out_dir = figures_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        nb, printed, error = _execute(nb_path, timeout)
    except Exception as exc:  # the kernel would not start — report it, as every failure is
        return NotebookResult(name, False, time.time() - started, 0, f"{type(exc).__name__}: {exc}")

    # Saved even when a cell failed: the notebook then shows the error where it happened.
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_figs = len(list(out_dir.glob("*.pdf")))
    if error:
        # Only the tail: a full traceback from twelve notebooks buries the one that matters.
        tail = "\n".join(error.strip().splitlines()[-12:])
        return NotebookResult(name, False, time.time() - started, n_figs, tail)
    (out_dir / STDOUT_FILE).write_text(printed, encoding="utf-8")
    return NotebookResult(name, True, time.time() - started, n_figs)


# ---------------------------------------------------------------------------
# The two summary documents
# ---------------------------------------------------------------------------


def _captured_text(name: str) -> str:
    path = figures_dir(name) / STDOUT_FILE
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def previous_blocks() -> dict[str, str]:
    """Each notebook's block in the current `All_Results.md`, verbatim.

    A partial run keeps these for the notebooks it did not execute. It used to rebuild the file
    from the executed notebooks alone, so `--only 1.3_pd_results` silently deleted the other ten
    summaries — and `--summaries-only` all eleven, because the captured text is cleaned up after
    every run.
    """
    path = all_results_path()
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    return {m.group(1): m.group(2)
            for m in re.finditer(r"^## (\S+)\n\n```\n(.*?)\n```\n", text, flags=re.S | re.M)}


def write_captions(notebooks: tuple[str, ...]) -> Path:
    """ONE CAPTIONS.md for the project, grouped per notebook, in notebook order.

    Built from each `_figures.json`, so it regenerates from disk after an interactive run. A
    figure with no caption gets a loud placeholder rather than being skipped — a gap should be
    visible in the document meant to contain it.
    """
    from src.visualize.figures import read_manifest

    lines = [
        "# Figure captions",
        "",
        "Generated by `python -m src.utils.run_notebooks`. Grouped by notebook, figures in",
        "the order that notebook drew them. Caption text is passed to",
        "`FigureSaver.save(..., caption=...)` in the notebook — edits here are overwritten.",
        "",
        "These are the paper's captions: paste one straight under its figure. Pure description",
        "— what is plotted, on what axes, from how much data. No interpretation.",
        "",
        "Figures are PDFs, drawn at the width they will occupy on an A4 page; never rescale one",
        "in the document, because that rescales its text with it.",
        "",
    ]
    chapter = None
    for name in notebooks:
        entries = read_manifest(name)
        # A chapter divider when the folder changes, so the file reads in the notebooks' order
        # of chapters. Notebook blocks stay `##` — that is the documented per-notebook contract.
        here = chapter_of(name)
        if here and here != chapter:
            lines += [f"# {here}", ""]
            chapter = here
        lines += [f"## {name}", ""]
        if not entries:
            lines += ["_No figures produced._", ""]
            continue
        for e in entries:
            lines.append(f"**{e['stem']}** — `{e['name']}`")
            lines.append("")
            lines.append(e["caption"] or "> MISSING CAPTION. Add one at the `save()` call.")
            lines.append("")
    path = captions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_all_results(notebooks: tuple[str, ...], keep: dict[str, str] | None = None) -> Path:
    """Every notebook's printed summary, concatenated. The shape is fixed:

    one block per notebook, **sorted alphabetically by notebook name**; each block is that
    notebook's printed summary **verbatim**, not a rewrite; and that summary follows the
    notebook's own section order, so the file and the notebook read the same way round.

    Verbatim matters: the moment this file paraphrases, the two disagree and the notebook wins —
    but this file is the one anybody actually reads.

    `keep` supplies the block of a notebook that has no freshly captured text — one a partial
    run did not execute (`previous_blocks`).
    """
    names = tuple(sorted(notebooks))
    lines = [
        "# All results",
        "",
        "Every notebook's printed summary, verbatim, one block per notebook in alphabetical",
        "order. Each block follows that notebook's own section order.",
        "Generated by `python -m src.utils.run_notebooks`.",
        "",
    ]
    chapter = None
    for name in names:
        text = _captured_text(name).strip() or (keep or {}).get(name, "").strip()
        here = chapter_of(name)
        if here and here != chapter:
            lines += ["---", "", f"# {here}", ""]
            chapter = here
        lines += ["---", "", f"## {name}", "", "```", text or "(no output captured)", "```", ""]
    path = all_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cleanup(notebooks: tuple[str, ...]) -> None:
    """Drop the captured-stdout scratch files once folded into `All_Results.md`."""
    for name in notebooks:
        (figures_dir(name) / STDOUT_FILE).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def run_all(
    notebooks: tuple[str, ...] | None = None,
    max_workers: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[NotebookResult]:
    """Run every notebook in parallel, then rebuild both summary documents.

    Rebuilt even when a notebook failed, from whatever the successful ones wrote: a
    half-updated summary beats a stale one, and the failure is reported separately. Both
    documents always cover EVERY notebook; one not run this time keeps its previous block.
    """
    names = discover(notebooks)
    if not names:
        return []
    # Capped at 4: notebooks are numpy-heavy and each already uses several threads, so more
    # workers than this trades parallelism for cache thrashing.
    workers = max_workers or min(len(names), 4)
    everything = discover()
    previous = previous_blocks()

    results: list[NotebookResult] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, name, timeout): name for name in names}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
            for fut in sorted(done, key=lambda f: futures[f]):
                r = fut.result()
                results.append(r)
                _say(f"  [{len(results):>2}/{len(names)}] {'OK    ' if r.ok else 'FAILED'} "
                     f"{r.name:<32} {r.seconds:6.1f}s  {r.n_figures:2d} figures")
            if pending and not done:
                _say(f"  ... {_clock(time.time() - started)} in, not finished yet: "
                     + ", ".join(sorted(futures[f] for f in pending)))

    write_captions(everything)
    write_all_results(everything, keep={n: t for n, t in previous.items() if n not in names})
    _cleanup(names)
    return sorted(results, key=lambda r: names.index(r.name))


def _say(line: str) -> None:
    """Progress goes out immediately: a buffered line is as good as none on a long run."""
    print(line, flush=True)


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def summarise(results: list[NotebookResult]) -> str:
    if not results:
        return "No notebooks found in notebooks/."
    lines = ["", "=" * 74, "NOTEBOOK RUN SUMMARY", "=" * 74]
    for r in results:
        lines.append(
            f"  {'OK    ' if r.ok else 'FAILED'} {r.name:<32} "
            f"{r.seconds:6.1f}s  {r.n_figures:2d} figures"
        )
        if not r.ok:
            lines += [f"           {line}" for line in r.error.splitlines()]
    ok = sum(1 for r in results if r.ok)
    lines += [
        "",
        f"{ok}/{len(results)} notebooks OK, {sum(r.n_figures for r in results)} figures",
        f"  figures   -> {figures_dir()}",
        f"  captions  -> {captions_path()}",
        f"  summaries -> {all_results_path()}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point. `--summaries-only` exists because both documents are built from what the notebooks
# left on disk (`_figures.json`), so after an interactive Jupyter session they can be regenerated
# without executing anything.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="STEM",
                        help="notebooks to run: a stem or its start, e.g. 1.3 or 1.")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per notebook")
    parser.add_argument("--summaries-only", action="store_true",
                        help="rebuild both documents from disk, run nothing")
    args = parser.parse_args(argv)

    names = select(tuple(args.only)) if args.only else discover()
    if not names:
        print("No notebooks found in notebooks/.")
        return 0

    if args.summaries_only:
        everything = discover()
        print(f"Rebuilding summaries from disk for all {len(everything)} notebooks")
        print(f"  captions  -> {write_captions(everything)}")
        print(f"  summaries -> {write_all_results(everything, keep=previous_blocks())}")
        return 0

    workers = args.workers or min(len(names), 4)
    _say(f"Running {len(names)} notebook(s), {workers} at a time: {', '.join(names)}")
    _say("  (a line per notebook as it finishes; 0.2 and 0.3 take longest — they generate "
         "the prior's tasks, in parallel worker processes)")
    results = run_all(names, max_workers=args.workers, timeout=args.timeout)
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
