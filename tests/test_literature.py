"""The literature module — the single source of the tfm-library grounding.

These guard the two things that would quietly corrupt the notebooks' honesty: a reference drawn as a
figure line must actually have a number, and a fact the library does NOT contain (Merton/Vasicek,
Basel) must stay tagged `external` so it can never be presented as library-grounded.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.visualize import literature as lit

_VALID_TAGS = {"paper-evaluated", "code-supported", "code+paper", "paper-described", "paper+code",
               "editorial", "editorial+paper", "external", "paper (position, no experiments)"}


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_pin_is_recorded():
    assert lit.PIN and lit.PIN != "unknown"


def test_every_ref_has_a_source_and_a_known_tag():
    for key, ref in lit.REFS.items():
        assert ref.label and ref.source, key
        assert ref.tag in _VALID_TAGS, f"{key}: unknown tag {ref.tag!r}"


def test_credit_domain_models_are_flagged_external_not_library_grounded():
    """The library contains no Merton/Vasicek or Basel source — the research confirmed a whole-library
    search returns nothing — so these must never read as paper-evaluated."""
    for key in ("merton_vasicek", "basel_retail", "basel_corp"):
        ref = lit.REFS[key]
        assert ref.tag == "external", key
        assert "EXTERNAL" in ref.source, key


def test_library_grounded_refs_cite_a_path_or_symbol():
    """A non-external ref must point at something in tfm-library (a papers/ path or a code dump)."""
    for key, ref in lit.REFS.items():
        if ref.tag == "external":
            continue
        assert ("papers/" in ref.source or ".txt" in ref.source or ".md" in ref.source), \
            f"{key}: {ref.source!r} does not look like a tfm-library citation"


def test_line_draws_a_numeric_ref_and_refuses_a_wordy_one():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    lit.line(ax, "filter_rate_clf")            # has value 0.35
    assert ax.lines  # drew something
    with pytest.raises(ValueError):
        lit.line(ax, "oprior_headline")        # a finding, no number


def test_cite_and_references_md_format_and_dedup():
    c = lit.cite("oprior_ticlv2")
    assert "0.791" in c and "paper-evaluated" in c and "Bouadi" in c
    md = lit.references_md(["filter_rate_clf", "filter_rate_reg", "merton_vasicek"])
    assert lit.PIN in md
    # filter_rate_clf and _reg share a source stem but differ; external line present and tagged
    assert "[external]" in md and "papers/2026/02_Qu_TabICLv2" in md


def test_line_colours_by_provenance_and_marks_external():
    """The honesty signal is the COLOUR, not the caption: a library value draws in REFERENCE teal,
    an external one in EXTERNAL amber and always carries an '(external)' mark, and the caller cannot
    override the colour to make domain knowledge look library-grounded."""
    import matplotlib.pyplot as plt

    from src.visualize import style

    fig, ax = plt.subplots()
    lit.line(ax, "tanna_tabicl_hc")                 # paper-evaluated, 0.771
    lit.line(ax, "basel_corp", label="Basel 0.24")  # external, 0.24
    colours = {tuple(_rgba(line.get_color())) for line in ax.lines}
    assert _rgba(style.REFERENCE) in colours and _rgba(style.EXTERNAL) in colours
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "(external)" in texts and "Basel 0.24" in texts
    # even if a caller tries to force the colour, provenance wins
    fig2, ax2 = plt.subplots()
    lit.line(ax2, "basel_corp", color=style.REFERENCE)
    assert _rgba(style.EXTERNAL) in {tuple(_rgba(line.get_color())) for line in ax2.lines}


def test_credit_benchmarks_and_dataset_refs_are_consistent():
    """Every key the landscape figure and the per-dataset map name must resolve to a ref, and every
    benchmark must carry a number, or a figure built from them breaks at run time in a notebook."""
    for key in lit.CREDIT_BENCHMARKS:
        assert key in lit.REFS, key
        assert lit.REFS[key].value is not None, key
    for ds, keys in lit.DATASET_REFS.items():
        for key in keys:
            assert key in lit.REFS, f"{ds}: {key}"


def test_credit_benchmark_landscape_builds_and_summarises():
    from src.visualize import literature_plots as lp

    fig = lp.credit_benchmark_landscape()
    assert any(ax.patches for ax in fig.axes), "the landscape drew no bars"
    text = lp.summary()
    assert lit.PIN in text and "0.786" in text and "protocol" in text.lower()


def _rgba(colour):
    import matplotlib.colors as mcolors

    return tuple(round(v, 4) for v in mcolors.to_rgba(colour))
