# Experiment 1 view of manifests 1.3 and 1.3a: the WHAS500 rows of the Cox survival case (e1_3.py) and of its
# reference explanation (e1_3a.py). Derived by filtering those cards, so every number is the one in them.
# The full cards (with GBSG2) stay in the Miscellany.
import importlib.util as _u
from pathlib import Path as _P


def _load(name):
    s = _u.spec_from_file_location("_" + name, _P(__file__).with_name(name + ".py"))
    m = _u.module_from_spec(s)
    s.loader.exec_module(m)
    return m.CARD


_C = _load("e1_3")
_R = _load("e1_3a")


def _t(card, i, keep, **kw):
    t = dict(card["results"][i])
    t["rows"] = [r for r in t["rows"] if keep(r)]
    t.update(kw)
    return t


CARD = {
    "id": "1.3",
    "anchor": "e1-3w",
    "group": "e1",
    "name": "Cox survival: WHAS500, the VARIANZ reference point",
    "status": _C["status"],
    "status_note": _C["status_note"] + "; WHAS500 rows of manifests 1.3 and 1.3a (the full cards, with GBSG2, are in the Miscellany)",
    "prereg": _C["prereg"],
    "oneline_question": "Can a PropNEAT network trained on the Cox loss match standard survival models on WHAS500 and be explained in hazard-ratio terms?",
    "headline": "Harrell C 0.830 (ridge Cox 0.819, Cox PH 0.818, forest 0.808); 0 hidden units, 9 of 14 inputs; verifier weakened, 0 refuted",
    "question": "Does PropNEAT with a Cox proportional-hazards loss reach the accuracy of standard survival references on WHAS500 (public survival data, the reference point for VARIANZ in Chapter 8), can the selected network be explained to the three levels and survive a blind check, and does it agree with a classical Cox model input by input?",
    "design": [
        ["Data and splits", "scikit-survival bundled WHAS500: n = 500, 215 events, 14 columns. Event-stratified 80/20 train/test (seed 42), then 20% of train held out for validation (seed 42): 320 / 80 / 100 rows. Same index arrays for every seed and reference."],
        ["Models and config", _C["design"][1][1]],
        ["Selection rule", "Within a run: the generation champion with the lowest validation Cox loss. Across seeds: the seed with the lowest validation Cox loss. Test metrics read only after the choice was written to `selected.json`. Selected: WHAS500 seed 0."],
        ["Explanation and verifier", _C["design"][3][1]],
        ["Metrics", "Test fold: Harrell's C, Uno's C, cumulative/dynamic AUC at 1 / 3 / 5 years, IPCW Brier at the horizon and IBS, decile calibration (E/O) at 3 years, test Cox loss. References: ridge Cox (α on validation), unpenalised Cox PH (with Schoenfeld test), random survival forest (500 trees). Reference reading (1.3a): Cox PH hazard ratios per clinical unit beside the network's."],
        ["Registration", _C["design"][5][1]],
        ["Departures from plan", _C["design"][6][1]],
    ],
    "results": [
        _t(_C, 0, lambda r: r[0] == "WHAS500", note="The selected network has 0 hidden units and keeps 9 of 14 inputs."),
        _t(_C, 1, lambda r: r[0] == "cox-whas500-s0", note=""),
        _t(_R, 1, lambda r: True),
    ],
    "results_text": [
        _C["results_text"][0],
        "Explaining time 21 min (227 numbers); recheck 0 / 0 (source: experiments/survival/cox_case/RESULTS.md §5).",
        "Schoenfeld PH test on the reference: sho=1 p = 0.002, mitype=1 0.015, cvd=1 0.017, los 0.043 (source: RESULTS.md §4).",
        "Interactions are small: Friedman's H² ≤ 0.008 for the forest's top inputs (source: REFERENCE.md §1, §4.1).",
    ],
    "shows": "On WHAS500 the evolved network was as accurate as standard Cox models (slightly ahead of them on this split), and its explanation reads directly as hazard ratios a clinician would recognise: older age, congestive heart complications and cardiogenic shock raise risk. It kept no hidden unit, so it is a sparse linear Cox model, and input by input it says the same thing as a classical Cox fit; the one notable difference is an age effect about half the Cox size. The case shows the survival path works end to end on public data, the reference point for VARIANZ; it does not show non-linear survival structure.",
    "caveats": [c.replace("every seed of both datasets is linear", "every WHAS500 seed is linear")
                for c in _C["caveats"] if "GBSG2" not in c or "every seed" in c][:1]
               + ["One split; a test fold of 100 rows; differences against references are within single-split noise.",
                  "Proportional hazards is inherited; the Schoenfeld tests flag four WHAS500 inputs.",
                  "Granularity is undefined with no hidden units.",
                  "One test-split query sits in the genome's log (the loading script's round-trip check); not quoted by the explainer.",
                  _C["caveats"][6],
                  "Per-unit HRs of continuous inputs are close to 1 by construction (REFERENCE.md has the per-SD table).",
                  "Not signed off; no plain-language draft."],
    "figures": [],
    "sources": [
        _C["sources"][0],
        "DB explaneat_dev: experiment 8d424f4c (whas500, genome f96ebc8d)",
        _R["sources"][0],
        "Branches exp/cox-case and exp/cox-reference-explanation (merged)",
    ],
    "thesis_note": "Experiment 1, the WHAS500 row of `ch7_experiment1_cases.tex` and `thesis/ch7/exports/ch7_cox_case_table.tex`; the reference reading is `ch7_cox_reference_table.tex`. GBSG2 was also run and, like WHAS500, gave a linear model (no new point), so it leaves Experiment 1; its rows are in the full cards in the Miscellany. Also the template for Chapter 8: the network against the published PREDICT equation on VARIANZ.",
}
