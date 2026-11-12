"""The concept probe: does a named concept predict a leaf's exit activation?

The third level of explanation ("understood": each leaf maps to a concept a domain reader
recognises) is measured, not judged. A concept is a derived feature of the raw inputs
(:mod:`.concepts`); the probe scores it against the leaf's exit activation over the data:

* ``r2``: the R-squared of the exit on the concept (least squares with an intercept; for a binary
  concept this is the between-groups share of variance);
* ``auc``: when the concept is binary, the AUC of the exit activation as a score for the concept
  (sign-free: ``max(auc, 1 - auc)``, with ``direction``); when the concept is continuous and the
  leaf is a gate (a ReLU exit that is 0 on some rows), the AUC of the concept as a score for the
  gate being on. Both binary: the exit's on/off against the concept;
* ``mi``: mutual information in bits between the concept and the exit (each discretised: a
  binary or small-count concept by its values, otherwise into ten quantile bins; the exit into
  on/off plus quantile bins), with ``nmi`` = mi / H(exit);
* ``n``: the rows used.

The grade (``meaning_grade``) follows the score with the stated thresholds and nothing else:
``meaningful`` when score >= 0.8; ``partial`` when score >= 0.5; ``closed_form_only``
otherwise; ``unnamed`` when no concept was given. A leaf with several exits is scored per exit
and takes the mean over exits.

Probe v2 (2026-11-13) adds two conditions a concept must meet before its score counts:

* **Simplicity.** A concept counts only if it is simpler than the leaf: a library concept, or an
  ad-hoc expression over at most ``MAX_CONCEPT_TERMS`` (3) variables that is not a reweighting
  of the leaf's own entry inputs (an expression whose variables all lie in the leaf's entries
  and number more than the limit is the leaf's formula with rounded weights). A concept that
  fails gets the grade ``restatement`` whatever its score, and the ``meaning`` block records
  ``concept_complexity`` and ``simpler_than_leaf``.
* **Support.** A binary concept's AUC enters the score only when the leaf is itself a gate with
  comparable support (both minority shares at least ``MIN_SUPPORT``, 1% of the rows, and within
  a factor of two of each other); otherwise the score is R-squared, the share of the exit's
  variance the concept explains (for a binary concept, the between-groups share). A binary
  concept below the support floor is scored by R-squared alone, with ``low_support`` set, in a
  library sweep as when named: a rare flag can no longer reach AUC 1 on a leaf while
  explaining none of its variance, while a rare flag that *is* the leaf (R-squared 1) still
  counts.

Exit activations come from ``ActivationExtractor.from_structure`` on the explained model state
(the same forward pass as the closed-form checks), so the probe works on split and identity
nodes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .concepts import (Concept, concept_frame, evaluate_expression, expression_variables, library_concepts,
                       library_for)

#: grade thresholds (section "Three levels of explanation" of docs/annotation_coverage_model.md)
MEANINGFUL_R2 = 0.8
MEANINGFUL_AUC = 0.9
PARTIAL_R2 = 0.5
PARTIAL_AUC = 0.75
#: probe v2 conditions
MAX_CONCEPT_TERMS = 3          # an ad-hoc concept may use at most this many variables
MIN_SUPPORT = 0.01             # a binary concept (or a gate) needs this minority share of the rows
COMPARABLE_RATIO = 0.5         # gate and concept minority shares within a factor of two
PROBE_VERSION = 2
GRADES = ("meaningful", "partial", "closed_form_only", "restatement", "unnamed")
ON_TOL = 1e-9


# ============================================================================ scores
def r_squared(concept: np.ndarray, exit_act: np.ndarray) -> Optional[float]:
    c = np.asarray(concept, float)
    y = np.asarray(exit_act, float)
    ok = np.isfinite(c) & np.isfinite(y)
    c, y = c[ok], y[ok]
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(c) == 0:
        return None
    A = np.column_stack([c, np.ones_like(c)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    tot = float(np.sum((y - y.mean()) ** 2))
    return float(max(0.0, 1.0 - float(np.sum(res ** 2)) / tot)) if tot > 0 else None


def _auc(labels: np.ndarray, score: np.ndarray) -> Optional[float]:
    from .explanation_tools import default_auc
    a = default_auc(labels, score)
    return None if a is None or (isinstance(a, float) and math.isnan(a)) else float(a)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _discretise(v: np.ndarray, bins: int = 10, binary_zero: bool = False) -> np.ndarray:
    v = np.asarray(v, float)
    u = np.unique(v)
    if len(u) <= bins:
        return np.searchsorted(u, v)
    out = np.zeros(len(v), dtype=int)
    if binary_zero:                       # a gated exit: 0 is its own bin, the rest quantiles
        on = v > ON_TOL
        if on.sum() >= 2 and np.ptp(v[on]) > 0:
            q = np.quantile(v[on], np.linspace(0, 1, bins + 1)[1:-1])
            out[on] = 1 + np.searchsorted(q, v[on])
        else:
            out[on] = 1
        return out
    q = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(q, v)


def mutual_information(concept: np.ndarray, exit_act: np.ndarray, bins: int = 10) -> Dict[str, Optional[float]]:
    """MI in bits of the discretised concept and exit, and MI normalised by the exit's entropy."""
    c = np.asarray(concept, float)
    y = np.asarray(exit_act, float)
    ok = np.isfinite(c) & np.isfinite(y)
    c, y = c[ok], y[ok]
    if len(y) < 3:
        return {"mi": None, "nmi": None}
    ci = _discretise(c, bins)
    yi = _discretise(y, bins, binary_zero=bool(np.any(np.abs(y) <= ON_TOL)))
    joint = np.zeros((ci.max() + 1, yi.max() + 1))
    np.add.at(joint, (ci, yi), 1.0)
    joint /= joint.sum()
    pc, py = joint.sum(1), joint.sum(0)
    hy = _entropy(py)
    mi = _entropy(pc) + hy - _entropy(joint.ravel())
    mi = float(max(0.0, mi))
    return {"mi": round(mi, 6), "nmi": (round(mi / hy, 6) if hy > 0 else None)}


# ============================================================================ the support condition
def minority_share(flag: np.ndarray) -> float:
    """The smaller of the two class shares of a 0/1 vector."""
    p = float(np.mean(np.asarray(flag, bool)))
    return min(p, 1.0 - p)


def gate_share(exit_act: np.ndarray) -> Optional[float]:
    """The minority share of a gated exit's on/off (None when the exit is not a gate)."""
    y = np.asarray(exit_act, float).ravel()
    on = y > ON_TOL
    if not (np.any(on) and np.any(np.abs(y) <= ON_TOL)):
        return None
    return minority_share(on)


def comparable_support(concept_share: Optional[float], gate: Optional[float],
                       min_support: float = MIN_SUPPORT, ratio: float = COMPARABLE_RATIO) -> bool:
    """A binary concept and a gate have comparable support when both minority shares are at least
    ``min_support`` and within a factor of ``1/ratio`` of each other."""
    if concept_share is None or gate is None:
        return False
    lo, hi = min(concept_share, gate), max(concept_share, gate)
    return lo >= min_support and hi > 0 and lo / hi >= ratio


def score_concept(concept: Concept, exit_act: np.ndarray, min_support: float = MIN_SUPPORT) -> Dict[str, Any]:
    """Every score of one concept against one exit activation vector. ``score`` is
    max(r2, 2(AUC - 0.5)) when the AUC may count (``auc_in_score``), else r2 (``score_basis``)."""
    y = np.asarray(exit_act, float).ravel()
    c = np.asarray(concept.values, float).ravel()
    if len(c) != len(y):
        raise ValueError(f"concept {concept.name!r} has {len(c)} rows, the exit {len(y)}")
    gated = bool(np.any(np.abs(y) <= ON_TOL)) and bool(np.any(y > ON_TOL))
    gshare = gate_share(y) if gated else None
    out: Dict[str, Any] = {"concept": concept.name, "kind": concept.kind, "source": concept.source,
                           "n": int(len(y)), "r2": None, "auc": None, "auc_of": None, "direction": None,
                           "mi": None, "nmi": None, "leaf_gated": gated,
                           "gate_share": None if gshare is None else round(gshare, 6),
                           "support": None, "low_support": False, "auc_in_score": False, "score_basis": "r2"}
    r2 = r_squared(c, y)
    out["r2"] = None if r2 is None else round(r2, 6)
    if concept.is_binary and np.ptp(c) > 0:
        s = minority_share(c > 0.5)
        out["support"] = round(s, 6)
        comparable = comparable_support(s, gshare, min_support)
        out["low_support"] = bool(s < min_support and not comparable)
        a = _auc(c > 0.5, y)
        if a is not None:
            out["auc"], out["auc_of"] = round(max(a, 1 - a), 6), "exit activation as a score for the concept"
            out["direction"] = "exit higher when the concept holds" if a >= 0.5 else "exit lower when the concept holds"
            out["auc_in_score"] = bool(comparable)
    elif gated:
        on = y > ON_TOL
        a = _auc(on, c)
        if a is not None:
            out["auc"], out["auc_of"] = round(max(a, 1 - a), 6), "concept as a score for the gate being on"
            out["direction"] = "gate on when the concept is high" if a >= 0.5 else "gate on when the concept is low"
            out["auc_in_score"] = True
    out.update(mutual_information(c, y))
    out["score_basis"] = "max(r2, 2(auc-0.5))" if out["auc_in_score"] else "r2"
    out["score"] = probe_score(out, use_auc=out["auc_in_score"])
    return out


def probe_score(s: Dict[str, Any], use_auc: Optional[bool] = None) -> Optional[float]:
    """The single number the grade is read from: max(r2, 2*(auc - 0.5)), so an AUC of 0.9 counts
    as 0.8 and 0.75 as 0.5 (the thresholds coincide). With ``use_auc`` False (a binary concept
    without a comparable gate, probe v2) the AUC is left out and the score is r2. A score
    dict's own ``auc_in_score`` is honoured when ``use_auc`` is None."""
    if use_auc is None:
        use_auc = bool(s.get("auc_in_score", True))
    vals = []
    if s.get("r2") is not None:
        vals.append(float(s["r2"]))
    if use_auc and s.get("auc") is not None:
        vals.append(2.0 * (float(s["auc"]) - 0.5))
    return round(max(vals), 6) if vals else None


def meaning_grade(score: Optional[float], concept_given: bool = True, simpler: bool = True) -> str:
    """The grade from the score and the stated thresholds (no judgement). A concept that is not
    simpler than its leaf grades ``restatement`` whatever its score."""
    if not concept_given:
        return "unnamed"
    if not simpler:
        return "restatement"
    if score is None:
        return "closed_form_only"
    if score >= MEANINGFUL_R2:
        return "meaningful"
    if score >= PARTIAL_R2:
        return "partial"
    return "closed_form_only"


# ============================================================================ the simplicity condition
def _matches_entry(var: str, entries: Sequence[str]) -> bool:
    base = var.split(":", 1)[0]
    return any(var == e or base == e for e in entries)


def concept_simplicity(concept: Concept, entry_inputs: Iterable[str] = (), columns: Optional[Iterable[str]] = None,
                       max_terms: int = MAX_CONCEPT_TERMS) -> Dict[str, Any]:
    """Is the concept simpler than the leaf? A library concept is one named term (a pairwise
    product or conjunction, two). An ad-hoc expression has one term per distinct variable it
    refers to; it is a *reweighting of the leaf's entries* when every variable is one of the
    leaf's entry inputs and there are more than ``max_terms`` of them (the leaf's own weighted
    sum with rounded weights). Simpler = at most ``max_terms`` terms and not a reweighting."""
    entries = [str(e) for e in entry_inputs]
    if concept.source == "expression":
        variables = expression_variables(concept.name, columns)
        complexity = len(variables)
    else:
        variables = [concept.name]
        complexity = 2 if concept.source == "pair" else 1
    in_entries = [v for v in variables if _matches_entry(v, entries)]
    reweighting = bool(variables) and len(in_entries) == len(variables) and complexity > max_terms
    return {"concept_complexity": int(complexity), "concept_variables": variables, "max_terms": int(max_terms),
            "simpler_than_leaf": bool(complexity <= max_terms and not reweighting),
            "reweighting_of_entries": bool(reweighting)}


# ============================================================================ the probe on a model
def leaf_exit_activations(structure, X: np.ndarray, annotation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Exit activations of an annotation on the explained model state, by exit node id."""
    from .activation_extractor import ActivationExtractor
    ex = ActivationExtractor.from_structure(structure)
    _, acts = ex.extract(np.asarray(X, float), annotation)
    acts = np.asarray(acts, float)
    if acts.ndim == 1:
        acts = acts[:, None]
    return {str(e): acts[:, i] for i, e in enumerate(annotation["exit_nodes"])}


def _annotation_dict(ann) -> Dict[str, Any]:
    if isinstance(ann, dict):
        return {"id": ann.get("id"), "name": ann.get("name"), "entry_nodes": [str(n) for n in ann["entry_nodes"]],
                "exit_nodes": [str(n) for n in ann["exit_nodes"]],
                "subgraph_nodes": [str(n) for n in ann.get("subgraph_nodes", [])],
                "subgraph_connections": [[str(a), str(b)] for a, b in ann.get("subgraph_connections", [])],
                "child_annotation_ids": list(ann.get("child_annotation_ids") or [])}
    return {"id": getattr(ann, "annotation_id", None), "name": ann.name,
            "entry_nodes": [str(n) for n in ann.entry_nodes], "exit_nodes": [str(n) for n in ann.exit_nodes],
            "subgraph_nodes": [str(n) for n in ann.subgraph_nodes],
            "subgraph_connections": [[str(a), str(b)] for a, b in ann.subgraph_connections],
            "child_annotation_ids": []}


def probe_leaf(structure, X: np.ndarray, X_raw: np.ndarray, feature_names: Sequence[str], dataset_name: str,
               annotation, concept: Optional[str] = None, entry_inputs: Iterable[str] = (),
               top: int = 5, include_inputs: bool = True, max_terms: int = MAX_CONCEPT_TERMS,
               min_support: float = MIN_SUPPORT) -> Dict[str, Any]:
    """Score one concept (a library name or an expression) or, with ``concept`` None, the whole
    library against every exit of the annotation.

    Returns the per-exit scores, the leaf score (mean over exits of the concept's score, or of
    each exit's best library concept), the grade and the ``meaning`` block. ``entry_inputs`` are
    the leaf's entry inputs (base variable names) for the simplicity condition; ``max_terms``
    and ``min_support`` are the probe v2 conditions.
    """
    ann = _annotation_dict(annotation)
    exits = leaf_exit_activations(structure, X, ann)
    entry_inputs = [str(e) for e in entry_inputs]
    lib = library_concepts(dataset_name, X_raw, feature_names, extra_pairs=entry_inputs,
                           include_inputs=include_inputs)
    by_name = {c.name: c for c in lib}
    res: Dict[str, Any] = {"annotation": ann["id"], "name": ann["name"], "exit_nodes": list(exits),
                           "library": library_for(dataset_name), "n": int(np.asarray(X).shape[0]),
                           "n_library_concepts": len(lib), "thresholds": thresholds(),
                           "probe_version": PROBE_VERSION, "entry_inputs": entry_inputs,
                           "conditions": {"max_terms": int(max_terms), "min_support": float(min_support),
                                          "comparable_ratio": COMPARABLE_RATIO}}
    if concept is not None:
        if concept in by_name:
            c = by_name[concept]
            simp = concept_simplicity(c, entry_inputs, None, max_terms)
        else:
            frame = concept_frame(dataset_name, X_raw, feature_names, lib)
            c = evaluate_expression(concept, frame)
            simp = concept_simplicity(c, entry_inputs, list(frame.columns), max_terms)
        per_exit = {e: score_concept(c, a, min_support) for e, a in exits.items()}
        scores = [s["score"] for s in per_exit.values() if s["score"] is not None]
        leaf_score = round(float(np.mean(scores)), 6) if scores else None
        first = next(iter(per_exit.values()), {})
        res.update({"concept": c.name, "concept_description": c.description, "concept_kind": c.kind,
                    "concept_source": c.source, "per_exit": per_exit, "score": leaf_score,
                    "r2": _mean_of(per_exit, "r2"), "auc": _mean_of(per_exit, "auc"), "mi": _mean_of(per_exit, "mi"),
                    "support": first.get("support"),
                    "low_support": any(s.get("low_support") for s in per_exit.values()),
                    "auc_in_score": any(s.get("auc_in_score") for s in per_exit.values()),
                    **simp, "grade": meaning_grade(leaf_score, True, simp["simpler_than_leaf"])})
    else:
        per_exit = {}
        low: set = set()
        for e, a in exits.items():
            ranked = sorted((score_concept(c, a, min_support) for c in lib), key=lambda s: -(s["score"] or -1))
            low |= {s["concept"] for s in ranked if s["low_support"]}
            per_exit[e] = {"best": ranked[0] if ranked else None, "top": ranked[:top],
                           "best_domain": next((s for s in ranked if s["source"] == "domain"), None)}
        bests = [p["best"]["score"] for p in per_exit.values() if p["best"] and p["best"]["score"] is not None]
        leaf_score = round(float(np.mean(bests)), 6) if bests else None
        best_names = sorted({p["best"]["concept"] for p in per_exit.values() if p["best"]})
        dom = [p["best_domain"]["score"] for p in per_exit.values() if p["best_domain"] and p["best_domain"]["score"] is not None]
        best_cx = [concept_simplicity(by_name[n], entry_inputs, None, max_terms)["concept_complexity"]
                   for n in best_names if n in by_name]
        res.update({"concept": None, "per_exit": per_exit, "score": leaf_score,
                    "best_concepts": best_names, "grade": meaning_grade(leaf_score, True),
                    "r2": _mean_of({e: p["best"] for e, p in per_exit.items() if p["best"]}, "r2"),
                    "auc": _mean_of({e: p["best"] for e, p in per_exit.items() if p["best"]}, "auc"),
                    "mi": _mean_of({e: p["best"] for e, p in per_exit.items() if p["best"]}, "mi"),
                    "support": next((p["best"].get("support") for p in per_exit.values() if p["best"]), None),
                    "low_support": any(p["best"].get("low_support") for p in per_exit.values() if p["best"]),
                    "auc_in_score": any(p["best"].get("auc_in_score") for p in per_exit.values() if p["best"]),
                    "concept_complexity": (max(best_cx) if best_cx else None), "simpler_than_leaf": True,
                    "reweighting_of_entries": False, "max_terms": int(max_terms),
                    "low_support_concepts": sorted(low),
                    "domain_score": round(float(np.mean(dom)), 6) if dom else None,
                    "domain_grade": meaning_grade(round(float(np.mean(dom)), 6) if dom else None, True),
                    "best_domain_concepts": sorted({p["best_domain"]["concept"] for p in per_exit.values()
                                                    if p["best_domain"]})})
    res["meaning"] = meaning_block(res)
    return res


def _mean_of(per_exit: Dict[str, Dict[str, Any]], key: str) -> Optional[float]:
    vals = [s[key] for s in per_exit.values() if s and s.get(key) is not None]
    return round(float(np.mean(vals)), 6) if vals else None


def thresholds() -> Dict[str, float]:
    return {"meaningful_r2": MEANINGFUL_R2, "meaningful_auc": MEANINGFUL_AUC,
            "partial_r2": PARTIAL_R2, "partial_auc": PARTIAL_AUC}


MEANING_PROBE_KEYS = ("r2", "auc", "mi", "n", "score", "support", "query_id")


def meaning_block(res: Dict[str, Any], query_id: Optional[str] = None) -> Dict[str, Any]:
    """The ``meaning`` block an annotation carries: ``{grade, concept, concept_complexity,
    simpler_than_leaf, probe: {r2, auc, mi, n, score, support, query_id}}``. The grade is derived
    from the probe with the stated thresholds and the simplicity condition."""
    concept = res.get("concept") or (", ".join(res.get("best_concepts") or []) or None)
    return {"grade": res["grade"] if concept else "unnamed", "concept": concept,
            "concept_complexity": res.get("concept_complexity"),
            "simpler_than_leaf": res.get("simpler_than_leaf"),
            "probe": {"r2": res.get("r2"), "auc": res.get("auc"), "mi": res.get("mi"), "n": res.get("n"),
                      "score": res.get("score"), "support": res.get("support"), "query_id": query_id}}


def grade_from_meaning(meaning: Optional[Dict[str, Any]]) -> str:
    """The grade a stored meaning block gives (re-derived from its probe when present, so a
    typed grade cannot disagree with its score; ``restatement`` when the block records that the
    concept is not simpler than the leaf)."""
    if not meaning:
        return "unnamed"
    concept = meaning.get("concept")
    probe = meaning.get("probe") or {}
    if not concept:
        return "unnamed"
    if meaning.get("simpler_than_leaf") is False:
        return "restatement"
    score = probe.get("score")
    if score is None:
        score = probe_score({"r2": probe.get("r2"), "auc": probe.get("auc")})
    return meaning_grade(score, True)
