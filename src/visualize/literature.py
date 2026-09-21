"""The literature, as data — one place, cited once, drawn everywhere.

Every published number this project compares itself against lives here with its **exact source** in
the pinned `tfm-library/` submodule (papers by path, code by symbol — never a line number, the dumps
drift) and an honesty **tag**. A figure overlays one as a reference line (`literature.line(ax, key)`);
a notebook's references cell prints the sources (`literature.references_md(...)`). Nothing is quoted
from memory: if a fact is not in the library, it is tagged `external` and says so, so a reader never
mistakes domain knowledge for a library-grounded result.

Library pin: `52dab01` (`git submodule status`). Tags:
  paper-evaluated — the paper *measured* it        code-supported — the code does it, unevaluated
  editorial       — the library's own synthesis    external — NOT in the library (domain knowledge)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The tfm-library commit these citations were read against. Record it beside any result that uses
#: them; bump with `python -m src.utils.update_tfm_library`.
PIN = "52dab01"


@dataclass(frozen=True)
class Ref:
    """One citable fact: a short label, its source, an honesty tag, and — where it is a number a
    figure can draw — a value."""

    label: str
    source: str
    tag: str
    value: float | None = None
    note: str = ""


#: Keyed by short slug. Grouped by the notebook family that leans on them.
REFS: dict[str, Ref] = {
    # -- the prior generator (graph_scm), for 0.2 / 0.3 -----------------------
    "filter_rate_clf": Ref("TabICLv2 filters ~35% (classif.)", "papers/2026/02_Qu_TabICLv2 §Data filtering, Fig. 10",
                           "paper-evaluated", 0.35, "stage-1 rejection rate of the predictability filter"),
    "filter_rate_reg": Ref("TabICLv2 filters ~25% (regr.)", "papers/2026/02_Qu_TabICLv2 §Data filtering",
                           "paper-evaluated", 0.25, "stage-1 rejection rate, regression"),
    "quantiles": Ref("999-quantile pinball head", "papers/2026/02_Qu_TabICLv2 §I; NanoTabICLv2(out_dim=999)",
                     "paper+code", 999, "represents a point mass exactly — atoms are representable"),
    "max_cat_size": Ref("graph_scm cap 200 categories", "TabICL.txt GraphSCM(max_cat_size=200)",
                        "code-supported", 200, "NanoTabICL uses 100"),
    "outlier_clamp": Ref("outlier clamp at 4σ", "TabICL.txt outlier_removing(threshold=4)",
                         "code-supported", None, "clamping is what manufactures the original prior's accidental atoms"),
    "kumaraswamy": Ref("Kumaraswamy warp a,b∈LogNum(0.2,5)", "TabICL.txt rand_kumaraswamy_act; §E.6",
                       "code+paper", None, "the [0,1] shape family already in their prior — the same we use for LGD"),
    "imbalance_source": Ref("prior imbalance via softmax b=log(w)", "TabICL.txt rand_converter; §E.6",
                            "paper-described", None, "graph_scm makes categorical y natively; the softmax bias sets imbalance"),

    # -- TabICLv2 training budget, for 1.1 / 1.2 ------------------------------
    "batch": Ref("batch 64 datasets/step", "papers/2026/02_Qu_TabICLv2 §4.1", "paper-evaluated", 64),
    "stage1_lr": Ref("stage-1 max LR 8e-4", "papers/2026/02_Qu_TabICLv2 §4.1", "paper-evaluated", 8e-4),
    "stage1_steps": Ref("stage 1 = 500k steps", "papers/2026/02_Qu_TabICLv2 §4.1", "paper-evaluated", 500000),
    "optimizer": Ref("Muon (per-param 0.2·√max(n,m))", "papers/2026/02_Qu_TabICLv2 §4.1", "paper-evaluated"),
    "pretrain_cost": Ref("24.5 H100-GPU-days per model", "papers/2026/02_Qu_TabICLv2 §4.1", "paper-evaluated", 24.5),
    "datasets_seen": Ref("~35M datasets seen (each once)", "papers/2026/02_Qu_TabICLv2 §4.1", "paper-evaluated", 35e6),

    # -- prior-design findings, for 0.2 / 0.3 / 1.x --------------------------
    "oprior_headline": Ref("structural diversity drives transfer", "papers/2026/05_Bouadi_ShapingThePrior Table 2",
                           "paper-evaluated", None, "Hybrid-SCM is the single most impactful component"),
    "oprior_ticlv2": Ref("O'Prior: TabICLv2 baseline 0.791 AUC", "papers/2026/05_Bouadi_ShapingThePrior Table 2",
                         "paper-evaluated", 0.791, "their default reference on 52 classification tasks"),
    "oprior_best": Ref("O'Prior: best variant 0.834 AUC", "papers/2026/05_Bouadi_ShapingThePrior Table 2",
                       "paper-evaluated", 0.834, "Hybrid SCM (SH)"),
    "oprior_creditg": Ref("Credit-g probe 0.60–0.66 (non-diagnostic)", "papers/2026/05_Bouadi_ShapingThePrior §probing",
                          "paper-evaluated", None, "a NULL result — signal too weak to separate priors; do not present as discriminating"),
    "oprior_scope": Ref("O'Prior evaluates classification only", "papers/2026/05_Bouadi_ShapingThePrior §benchmarks",
                        "paper-evaluated", None, "generator supports regression; LGD is their untested gap"),
    "tabforest": Ref("complexity can beat realism", "papers/2024/10_Breejen_TabForestPFN",
                     "paper-evaluated", None, "unrealistic forest-generated data fine-tunes better than realistic"),
    "mitra": Ref("static prior mixtures beat single priors", "papers/2025/10_Zhang_Mitra",
                 "paper-evaluated", None, "selection by real-data performance, diversity, distinctiveness"),

    # -- credit-domain mechanisms (0.2 / 0.3) --------------------------------
    "purucker_highcard": Ref("GBDT edge grows with high-cardinality (ρ=+0.47)", "papers/2026/06_Purucker_BeyondIID Table E.3",
                             "paper-evaluated", 0.466, "num_high_cardinality_cats vs the TFM-minus-GBDT gap"),
    "purucker_rows": Ref("GBDT edge grows with rows (ρ=+0.60)", "papers/2026/06_Purucker_BeyondIID Table E.3",
                         "paper-evaluated", 0.603),
    "purucker_missing": Ref("GBDT edge grows with missingness (ρ=+0.29)", "papers/2026/06_Purucker_BeyondIID Table E.3",
                            "paper-evaluated", 0.286),
    "purucker_icl": Ref("Beyond-IID tests ICL only, not fine-tuning", "papers/2026/06_Purucker_BeyondIID Limitation 2",
                        "paper-evaluated", None, "the non-IID TFM losses are for the frozen regime — Exp2 asks the open question"),
    "klein_rule": Ref("statistical prior blurs a hard rule (~$4,800 vs $5,000)", "papers/2026/01_Klein_Hoffart §intro",
                      "paper (position, no experiments)", None, "the most speculative CreditICL mechanism — labelled so"),
    "tabicl_impute": Ref("TabICLv2 mean-imputes at inference", "TabICL.txt TransformToNumerical",
                         "code-supported", None, "discards informative missingness — a thin file is itself a risk signal"),

    # -- downstream credit evaluation (1.3/1.4, 2.3/2.4) ---------------------
    "tanna_resampling": Ref("context construction beats architecture (+3–4 AUC)", "papers/2026/05_Tanna_DataPresentation Table 2",
                            "paper-evaluated", None, "balanced sampling adds more AUC than the choice of TFM"),
    "tanna_zerorecall": Ref("default-threshold GBDTs collapse to 0% recall", "papers/2026/05_Tanna_DataPresentation Table 1",
                            "paper-evaluated", 0.0, "the accuracy paradox — why we never report accuracy alone"),
    "tanna_paradox": Ref("majority collapse below ~10% base rate", "papers/2026/05_Tanna_DataPresentation §5.1",
                         "paper-evaluated", 0.10, "report MCC / recall / balanced-acc, not accuracy"),
    "calibration_gap": Ref("no ECE/Brier protocol across adaptation exists", "SYNTHESIS.md (editorial)",
                           "editorial", None, "calibration is a first-class selling point yet under-measured — a contribution to fill"),
    "crps": Ref("CRPS scores the whole predictive, not its mean", "SYNTHESIS.md; TabICLv2 §I.7",
                "editorial+paper", None, "why LGD's primary metrics are distributional"),

    # -- credit-domain BENCHMARKS: the only credit-dataset numbers anywhere in the library, all
    #    ROC-AUC so they share one axis. From Tanna 2026 (Home Credit / Lending Club) and Hollmann
    #    2023 (Credit-g). A DIFFERENT protocol from ours — full-dataset context of thousands of rows,
    #    not our 1024-row ICL query — so they are a landscape to set our numbers beside, never a
    #    like-for-like target. The distinction is exactly what §3 of the project rules insists on. --
    "tanna_rf_hc": Ref("Home Credit: RandomForest 0.739", "papers/2026/05_Tanna_DataPresentation §5.3",
                       "paper-evaluated", 0.739, "the classical baseline the TFMs chase; full-dataset protocol, not ICL"),
    "tanna_tabicl_hc": Ref("Home Credit: TabICL 0.771", "papers/2026/05_Tanna_DataPresentation Table 3",
                           "paper-evaluated", 0.771, "50K balanced context on the full dataset — not our 1024-row ICL setting"),
    "tanna_tabpfn_hc": Ref("Home Credit: TabPFN 0.786", "papers/2026/05_Tanna_DataPresentation Table 3",
                           "paper-evaluated", 0.786, "strongest TFM on HC at 50K balanced context; a different protocol from ours"),
    "tanna_xgb_lc": Ref("Lending Club: XGBoost 0.718", "papers/2026/05_Tanna_DataPresentation §5.3",
                        "paper-evaluated", 0.718, "the LC crossover most TFMs do not reach within 50K context"),
    "tanna_hybrid_lc": Ref("Lending Club: best TFM 0.686", "papers/2026/05_Tanna_DataPresentation Table 2",
                           "paper-evaluated", 0.686, "hybrid-context mean; LC is the harder of the two books"),
    "tanna_hc_ceiling": Ref("Home Credit Kaggle ceiling ~0.80", "papers/2026/05_Tanna_DataPresentation §6.3",
                            "paper-described", 0.80, "top competition ensembles, cited as an external ceiling — NOT a TFM result"),
    "creditg_tabpfn": Ref("Credit-g (German): TabPFN 0.789", "papers/2023/09_Hollmann_TabPFN Table 2",
                          "paper-evaluated", 0.789, "TabPFN v1 on OpenML-CC18-small, 60-min budget — not TabICLv2"),

    # -- credit dataset properties the library states, for 0.1 (Tanna 2026 §4.1) ---------------
    "hc_base_rate": Ref("Home Credit default rate ~8%", "papers/2026/05_Tanna_DataPresentation §4.1",
                        "paper-described", 0.08, "a real base rate our controlled-imbalance arm targets"),
    "lc_base_rate": Ref("Lending Club default rate 12-22%", "papers/2026/05_Tanna_DataPresentation §4.1",
                        "paper-described", 0.15, "0.12–0.22 depending on filtering; drawn at the midpoint"),
    "hc_rows": Ref("Home Credit ~307K rows, 120+ features", "papers/2026/05_Tanna_DataPresentation §4.1",
                   "paper-described", 307000, "orders of magnitude past the 1024-row ICL context the model sees"),
    "lc_rows": Ref("Lending Club ~533K rows, 70+ features", "papers/2026/05_Tanna_DataPresentation §4.1",
                   "paper-described", 533000),

    # -- the predictability filter's internals in code, behind filter_rate_clf/reg -------------
    "filter_extratrees_n": Ref("filter fits ExtraTrees (25 trees)", "NanoTabICL.txt rand_dataset_filtered",
                               "code-supported", 25, "the shallow model whose skill decides whether a task is kept"),
    "filter_pval": Ref("filter rejects at bootstrap p >= 0.05", "TabICL.txt should_filter",
                       "code-supported", 0.05, "a significance test, not an R² floor — a weak-but-real signal still passes"),

    # -- EXTERNAL: credit-domain models the library does NOT contain ---------
    "merton_vasicek": Ref("Merton/Vasicek one-factor default model", "EXTERNAL — not in tfm-library (Merton 1974; Vasicek 2002)",
                          "external", None, "our PD mechanism; a credit-domain model outside the TFM library's scope"),
    "basel_retail": Ref("Basel IRB retail correlation 0.03–0.16", "EXTERNAL — Basel Committee IRB formula",
                        "external", None, "domain knowledge, not library-grounded"),
    "basel_corp": Ref("Basel IRB corporate correlation 0.12–0.24", "EXTERNAL — Basel Committee IRB formula",
                      "external", 0.24, "the 0.30 aggressive arm = 0.24 × 1.25 large-financial multiplier"),
}


#: The credit-domain benchmark AUCs in reading order, all ROC-AUC on real credit data so they share
#: one axis. `literature_plots.credit_benchmark_landscape` draws them as the field's landscape — the
#: caption states the protocol differs from ours, so they are context, not a scoreboard we sit on.
CREDIT_BENCHMARKS: tuple[str, ...] = (
    "tanna_rf_hc", "tanna_tabicl_hc", "tanna_tabpfn_hc", "tanna_hc_ceiling",
    "creditg_tabpfn", "tanna_xgb_lc", "tanna_hybrid_lc",
)

#: Which citations name a specific real dataset, keyed by the dataset slug our own results use. Lets a
#: figure or a summary point at the literature FOR THE SAME DATASET — the honest comparison, as long
#: as the protocol caveat in each ref's note travels with it.
DATASET_REFS: dict[str, tuple[str, ...]] = {
    "home_credit": ("tanna_rf_hc", "tanna_tabicl_hc", "tanna_tabpfn_hc", "hc_base_rate"),
    "lendingclub": ("tanna_xgb_lc", "tanna_hybrid_lc", "lc_base_rate"),
    "german": ("creditg_tabpfn",),
}


def get(key: str) -> Ref:
    return REFS[key]


def _draw_colour(ref: Ref) -> str:
    """Teal for a library-grounded value, amber for one cited from outside the library. The colour
    IS the provenance claim, so it must track the tag and never be chosen by the caller."""
    from src.visualize import style

    return style.EXTERNAL if ref.tag == "external" else style.REFERENCE


def line(ax: Any, key: str, *, orient: str = "v", label: str | None = None, **kwargs: Any) -> None:
    """Overlay a citable value on `ax` as a labelled reference line, coloured by provenance.

    A library-grounded value draws in REFERENCE teal; one tagged `external` draws in EXTERNAL amber
    and always carries an "(external)" mark, so a reader can never mistake domain knowledge for a
    library result whatever the caption says. `label` overrides the drawn text (use a short number,
    not the full citation — the interpretation belongs in the caption); the colour is never
    overridable, because it is the honesty signal. Raises if the ref has no numeric value.
    """
    from src.visualize import style

    ref = REFS[key]
    if ref.value is None:
        raise ValueError(f"literature ref {key!r} has no numeric value to draw")
    text = label or ref.label
    if ref.tag == "external":
        text = f"{text} (external)"
    kwargs["color"] = _draw_colour(ref)  # provenance, not the caller's to choose
    style.reference_line(ax, ref.value, text, orient=orient, **kwargs)


def cite(key: str) -> str:
    """`"label (source, tag)"` — a compact inline citation for a printed summary."""
    r = REFS[key]
    return f"{r.label} ({r.source}; {r.tag})"


def references_md(keys: list[str] | None = None) -> str:
    """A markdown reference list for the notebook's printed summary, deduplicated by source."""
    keys = keys or list(REFS)
    seen: dict[str, Ref] = {}
    for k in keys:
        r = REFS[k]
        seen.setdefault(r.source, r)
    lines = [f"References (tfm-library pin {PIN}):"]
    for r in seen.values():
        lines.append(f"  - {r.source}  [{r.tag}]")
    return "\n".join(lines)
