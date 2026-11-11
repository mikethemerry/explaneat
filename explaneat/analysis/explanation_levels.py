"""The three levels of an explanation, reported side by side.

| Level      | Claim                                                   | Test                                   |
|------------|---------------------------------------------------------|----------------------------------------|
| faithful   | the hierarchy computes the model's function             | max abs error (function preservation,  |
|            |                                                         | closed form)                           |
| decomposed | each leaf is a small, separately readable part          | leaf count; width-based granularity;   |
|            |                                                         | the swallowed-layer warning            |
| understood | each leaf maps to a concept a domain reader recognises  | concept probe grades (``meaning``      |
|            |                                                         | blocks); ground-truth recovery         |

The pipeline guarantees the first two mechanically; the third is measured by the concept probe
(:mod:`.concept_probe`) and stored on each annotation as a ``meaning`` block:
``{grade, concept, probe: {r2, auc, mi, n, query_id}}``. Depth is not penalised: a 3-4 deep
subnetwork is a good leaf; one leaf holding many parallel units (a wide layer) is not
(``docs/annotation_coverage_model.md``, "Three levels of explanation").
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .concept_probe import GRADES, grade_from_meaning

TINY = 1e-9


def leaf_meanings_from_operations(operations: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The ``meaning`` block of each leaf annotation stored in the operations stream, by
    annotation name (``annotate`` params or a later ``set_meaning``)."""
    from .composition import annotation_dicts_from_operations
    operations = list(operations)
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for ann in annotation_dicts_from_operations(operations):
        if not ann.get("child_annotation_ids"):
            out[ann["name"]] = None
    for op in operations:
        params = (op.get("params") or {}) if op.get("type") == "annotate" else {}
        if params.get("name") in out and params.get("meaning"):
            out[params["name"]] = params["meaning"]
    return out


def understood_block(leaf_names: List[str], meanings: Optional[Dict[str, Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
    """Grade counts and fractions over the leaves. ``understood_fraction`` is the share of
    leaves graded ``meaningful``; ``partial_fraction``, ``closed_form_only_fraction`` and
    ``unnamed_fraction`` complete it (they sum to 1)."""
    meanings = meanings or {}
    grades = {}
    for name in leaf_names:
        m = meanings.get(name)
        grades[name] = grade_from_meaning(m)
    n = len(leaf_names)
    counts = {g: sum(1 for v in grades.values() if v == g) for g in GRADES}
    frac = (lambda k: round(counts[k] / n, 4) if n else None)
    return {"leaves": n, "graded": n - counts["unnamed"], "counts": counts, "grades": grades,
            "understood_fraction": frac("meaningful"), "partial_fraction": frac("partial"),
            "closed_form_only_fraction": frac("closed_form_only"), "unnamed_fraction": frac("unnamed"),
            "concepts": {name: (m or {}).get("concept") for name, m in meanings.items() if name in grades and m}}


def explanation_levels(coverage: Dict[str, Any], checks: Optional[Dict[str, Any]] = None,
                       meanings: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
                       leaf_names: Optional[List[str]] = None,
                       ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The three levels from a coverage report (with its ``decomposition`` block), the
    automatic checks (``check_explanation``: function preservation and closed-form error) and the
    leaves' ``meaning`` blocks. Anything not available is reported as ``None``, never guessed."""
    dec = coverage.get("decomposition") or {}
    fp = (checks or {}).get("function_preservation_max_abs_err")
    cf = (checks or {}).get("closed_form_max_abs_err")
    faithful = {"function_preservation_max_abs_err": fp, "closed_form_max_abs_err": cf,
                "passed": (fp is not None and fp < TINY) if fp is not None else None,
                "test": "max |output(explained model) - output(phenotype)| over the training rows < 1e-9"}
    coarse = [w for w in dec.get("warnings") or [] if w.startswith("coarse decomposition")]
    decomposed = {"n_leaves": coverage.get("leaf_count"), "n_compositions": coverage.get("composition_count"),
                  "hidden_units": dec.get("hidden_units"), "granularity": dec.get("granularity"),
                  "largest_part": dec.get("largest_part"), "max_width": dec.get("max_width"),
                  "max_depth": dec.get("max_depth"), "warnings": dec.get("warnings") or [],
                  "passed": (not coarse and dec.get("units_only_as_entries", 0) == 0
                             and dec.get("units_only_in_root", 0) == 0) if dec.get("granularity") is not None else None,
                  "test": f"share of hidden units whose part is at most {dec.get('small_width_max_units', 6)} "
                          "units wide (depth not penalised); no part wider than that holds half the units"}
    if leaf_names is None:
        leaf_names = [n for n, p in (dec.get("parts") or {}).items() if p.get("leaf")]
    understood = understood_block(leaf_names, meanings)
    understood["test"] = ("concept probe per leaf: meaningful r2 >= 0.8 or AUC >= 0.9; partial r2 >= 0.5 or "
                          "AUC >= 0.75; else closed_form_only; unnamed when no concept was probed")
    if ground_truth is not None:
        understood["ground_truth"] = ground_truth
    return {"faithful": faithful, "decomposed": decomposed, "understood": understood}


def levels_summary_row(levels: Dict[str, Any]) -> Dict[str, Any]:
    """The flat columns a table or CSV carries."""
    f, d, u = levels["faithful"], levels["decomposed"], levels["understood"]
    return {"faithful_max_err": f.get("function_preservation_max_abs_err"),
            "n_leaves": d.get("n_leaves"), "granularity": d.get("granularity"),
            "understood_fraction": u.get("understood_fraction"),
            "closed_form_only_fraction": u.get("closed_form_only_fraction")}
