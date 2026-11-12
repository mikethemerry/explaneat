"""The three levels of an explanation, reported side by side.

| Level      | Claim                                                   | Test                                   |
|------------|---------------------------------------------------------|----------------------------------------|
| faithful   | the hierarchy computes the model's function             | max abs error (function preservation,  |
|            |                                                         | closed form)                           |
| decomposed | each leaf is a small, separately readable part          | leaf count; width-based granularity;   |
|            |                                                         | the swallowed-layer warning            |
| understood | each leaf maps to a concept a domain reader recognises  | concept probe grades (``meaning``      |
|            |                                                         | blocks); ground-truth recovery; the    |
|            |                                                         | blind verifier's leaf grades beside    |

The pipeline guarantees the first two mechanically; the third is measured by the concept probe
(:mod:`.concept_probe`) and stored on each annotation as a ``meaning`` block:
``{grade, concept, concept_complexity, simpler_than_leaf, probe: {r2, auc, mi, n, score, support,
query_id}}``. Where a blind verifier has graded the leaves (``verdict.json``, ``leaf_meaning``),
the understood block carries ``understood_verifier`` beside ``understood_probe`` and the count
of leaves on which the two disagree. Depth is not penalised: a 3-4 deep subnetwork is a good
leaf; one leaf holding many parallel units (a wide layer) is not
(``docs/annotation_coverage_model.md``, "Three levels of explanation").
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .concept_probe import GRADES, grade_from_meaning

TINY = 1e-9
#: how verifiers spell the grades
GRADE_ALIASES = {"closed-form-only": "closed_form_only", "closed form only": "closed_form_only",
                 "closed_form_only": "closed_form_only", "cfo": "closed_form_only", "meaningful": "meaningful",
                 "partial": "partial", "restatement": "restatement", "restated": "restatement",
                 "unnamed": "unnamed"}


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


def normalise_grade(grade: Any) -> Optional[str]:
    """A verifier's grade word in the probe's vocabulary (None when it is not one)."""
    if grade is None:
        return None
    g = str(grade).strip().lower().replace("_", "-")
    return GRADE_ALIASES.get(g) or GRADE_ALIASES.get(g.replace("-", "_"))


def verifier_leaf_grades(verdict: Optional[Dict[str, Any]], leaf_names: Iterable[str],
                         leaf_ids: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Leaf name -> grade from a verifier's ``verdict.json`` ``leaf_meaning`` block, whose keys
    are leaf names or annotation ids (``leaf_ids``: name -> id maps the latter). Entries that
    name no known leaf, or carry no grade word, are left out."""
    names = [str(n) for n in leaf_names]
    id_to_name = {str(v): k for k, v in (leaf_ids or {}).items()}
    out: Dict[str, str] = {}
    for key, v in ((verdict or {}).get("leaf_meaning") or {}).items():
        g = normalise_grade(v.get("grade") if isinstance(v, dict) else v)
        name = key if key in names else id_to_name.get(str(key))
        if name is None and isinstance(v, dict) and v.get("name") in names:
            name = v["name"]
        if name is not None and g is not None:
            out[name] = g
    return out


def understood_block(leaf_names: List[str], meanings: Optional[Dict[str, Optional[Dict[str, Any]]]],
                     verifier_grades: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Grade counts and fractions over the leaves. ``understood_fraction`` (= ``understood_probe``)
    is the share of leaves graded ``meaningful`` by the probe; ``partial_fraction``,
    ``closed_form_only_fraction``, ``restatement_fraction`` and ``unnamed_fraction`` complete it
    (they sum to 1). With ``verifier_grades`` (leaf name -> grade) the block also carries
    ``understood_verifier`` (= ``understood_fraction_verifier``, the share of verifier-graded
    leaves graded meaningful), ``verifier_graded`` and ``verifier_disagreements``: the leaves
    graded by both where the two grades differ (``verifier_compared`` of them)."""
    meanings = meanings or {}
    grades = {}
    for name in leaf_names:
        m = meanings.get(name)
        grades[name] = grade_from_meaning(m)
    n = len(leaf_names)
    counts = {g: sum(1 for v in grades.values() if v == g) for g in GRADES}
    frac = (lambda k: round(counts[k] / n, 4) if n else None)
    out = {"leaves": n, "graded": n - counts["unnamed"], "counts": counts, "grades": grades,
           "understood_fraction": frac("meaningful"), "understood_probe": frac("meaningful"),
           "partial_fraction": frac("partial"),
           "closed_form_only_fraction": frac("closed_form_only"), "restatement_fraction": frac("restatement"),
           "unnamed_fraction": frac("unnamed"),
           "concepts": {name: (m or {}).get("concept") for name, m in meanings.items() if name in grades and m},
           "understood_verifier": None, "understood_fraction_verifier": None, "verifier_graded": 0,
           "verifier_grades": None, "verifier_compared": 0, "verifier_disagreements": None}
    if verifier_grades is not None:
        vg = {name: verifier_grades.get(name) for name in leaf_names}
        graded = [name for name in leaf_names if vg[name]]
        both = [name for name in graded if grades[name] != "unnamed"]
        nv = len(graded)
        uv = round(sum(1 for name in graded if vg[name] == "meaningful") / nv, 4) if nv else None
        out.update({"understood_verifier": uv, "understood_fraction_verifier": uv, "verifier_graded": nv,
                    "verifier_grades": vg, "verifier_compared": len(both),
                    "verifier_disagreements": sum(1 for name in both if grades[name] != vg[name]),
                    "verifier_counts": {g: sum(1 for name in graded if vg[name] == g) for g in GRADES}})
    return out


def explanation_levels(coverage: Dict[str, Any], checks: Optional[Dict[str, Any]] = None,
                       meanings: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
                       leaf_names: Optional[List[str]] = None,
                       ground_truth: Optional[Dict[str, Any]] = None,
                       verifier_grades: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The three levels from a coverage report (with its ``decomposition`` block), the
    automatic checks (``check_explanation``: function preservation and closed-form error), the
    leaves' ``meaning`` blocks and, when there is one, the blind verifier's leaf grades. Anything
    not available is reported as ``None``, never guessed."""
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
    understood = understood_block(leaf_names, meanings, verifier_grades)
    understood["test"] = ("concept probe per leaf (v2): meaningful score >= 0.8; partial >= 0.5; else "
                          "closed_form_only; restatement when the concept is not simpler than the leaf (more than "
                          "3 terms, or a reweighting of the leaf's entries); a binary concept's AUC counts only "
                          "against a gate of comparable support (>= 1% of rows), else r2; unnamed when no concept "
                          "was probed. understood_verifier: the blind verifier's leaf grades where a verdict exists")
    if ground_truth is not None:
        understood["ground_truth"] = ground_truth
    return {"faithful": faithful, "decomposed": decomposed, "understood": understood}


def levels_summary_row(levels: Dict[str, Any]) -> Dict[str, Any]:
    """The flat columns a table or CSV carries (``understood_fraction_verifier`` last; it is
    None without a verdict)."""
    f, d, u = levels["faithful"], levels["decomposed"], levels["understood"]
    return {"faithful_max_err": f.get("function_preservation_max_abs_err"),
            "n_leaves": d.get("n_leaves"), "granularity": d.get("granularity"),
            "understood_fraction": u.get("understood_fraction"),
            "closed_form_only_fraction": u.get("closed_form_only_fraction"),
            "understood_fraction_verifier": u.get("understood_fraction_verifier")}
