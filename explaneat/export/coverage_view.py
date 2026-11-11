"""The coverage block of an export, computed with the current code.

``coverage_report`` (which backs the MCP ``get_coverage`` tool) is called unchanged and its
output stored verbatim under ``report``; the export adds only *derived* views of it:

* the Null parts with a category (``disconnected_input``, ``dead_end``; edges ``zero_weight``,
  ``touches_null_node``), see ``docs/annotation_coverage_model.md`` § "Null explanations";
* the uncovered nodes and edges, and the missing compositions;
* the output-node convention, stated explicitly.

Nothing here recomputes coverage differently from ``get_coverage``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..analysis.coverage_report import coverage_report

CONVENTIONS = {
    "candidate_nodes": "all nodes of the explained model state except the output nodes "
                       "(output nodes are excluded from the candidate set)",
    "node_covered": "a candidate node v is covered when some annotation A in the hierarchy "
                    "(a leaf, or a composition expanded over its descendants, i.e. including "
                    "its junction nodes) has v in V_A and every enabled outgoing edge of v in E_A",
    "exact_null": "nodes with no directed path to an output over enabled, non-zero-weight "
                  "edges (disconnected inputs, dead ends), and edges that are zero-weight or "
                  "touch such a node, count as covered and are listed under null; a parentless, "
                  "childless annotation made only of Null nodes is exempt from the single-root "
                  "rule (hierarchy.exempt_disconnected)",
    "support_level_null": "not automatic: a part with zero effect only on the training data "
                          "needs an explicit, evidenced Null annotation",
    "compositional": "composed / required, where every annotation except the single global "
                     "root (whose expanded subgraph contains every output) must have a parent; "
                     "1.0 iff one root, containing every output, with every annotation under it",
    "source": "explaneat.analysis.coverage_report.coverage_report (backs MCP get_coverage)",
}


def _null_categories(state, report) -> Dict[str, Any]:
    inputs = {str(n) for n in state.input_node_ids}
    null_nodes = set(report["null"]["nodes"])
    enabled = [c for c in state.connections if c.enabled]
    out_by_node: Dict[str, List[Any]] = {}
    for c in enabled:
        out_by_node.setdefault(str(c.from_node), []).append(c)

    nodes = []
    for n in sorted(null_nodes):
        outs = out_by_node.get(n, [])
        if not outs:
            reason = "no enabled outgoing edges"
        elif all(c.weight == 0 for c in outs):
            reason = "only zero-weight outgoing edges"
        else:
            reason = "every outgoing path ends without reaching an output"
        nodes.append({
            "id": n,
            "category": "disconnected_input" if n in inputs else "dead_end",
            "reason": reason,
        })

    weight = {(str(c.from_node), str(c.to_node)): c.weight for c in enabled}
    edges = []
    for u, v in report["null"]["edges"]:
        w = weight.get((u, v))
        edges.append({
            "from": u, "to": v,
            "category": "zero_weight" if w == 0 else "touches_null_node",
        })
    counts: Dict[str, int] = {}
    for x in nodes + edges:
        counts[x["category"]] = counts.get(x["category"], 0) + 1
    return {"basis": report["null"]["basis"], "nodes": nodes, "edges": edges,
            "counts": dict(sorted(counts.items())),
            "exempt_annotations": list((report.get("hierarchy") or {}).get("exempt_disconnected") or [])}


def _missing_compositions(annotations, report) -> List[Dict[str, str]]:
    h = report.get("hierarchy")
    if not h:
        return []
    exempt = set(h.get("exempt_disconnected") or [])
    global_root = h.get("global_root")
    missing = []
    for a in annotations:
        if a.name in exempt or a.name == global_root:
            continue
        if a.parent_annotation_id is None:
            missing.append({"annotation": a.name,
                            "missing": "no parent composition (not under the global root)"})
    if global_root is None:
        missing.append({"annotation": None,
                        "missing": "no global root: no root annotation's expanded subgraph "
                                   "contains every output node"})
    return missing


def coverage_block(state, annotations) -> Dict[str, Any]:
    """The export's ``coverage`` object for a model state and its ``AnnotationData`` list."""
    report = coverage_report(state, annotations)
    uncovered_nodes = list(report["node_coverage"]["uncovered"])
    uncovered_edges = [list(e) for e in report["edge_coverage"]["uncovered"]]
    missing = _missing_compositions(annotations, report)
    structural = report["structural_coverage"]
    compositional = report["compositional_coverage"]
    full = (structural == 1.0 and compositional == 1.0 and not uncovered_nodes
            and not missing)
    return {
        "conventions": CONVENTIONS,
        "report": report,
        "summary": {
            "structural_coverage": structural,
            "compositional_coverage": compositional,
            "full_coverage": full,
            "candidate_nodes": report["node_coverage"]["total_candidate"],
            "covered_nodes": len(report["node_coverage"]["covered"]),
            "uncovered_nodes": len(uncovered_nodes),
            "edges": report["edge_coverage"]["total"],
            "uncovered_edges": len(uncovered_edges),
            "null_nodes": len(report["null"]["nodes"]),
            "null_edges": len(report["null"]["edges"]),
            "missing_compositions": len(missing),
            "annotations": report["annotations_count"],
            "leaves": report["leaf_count"],
            "compositions": report["composition_count"],
        },
        "null": _null_categories(state, report),
        "uncovered_nodes": uncovered_nodes,
        "uncovered_edges": uncovered_edges,
        "missing_compositions": missing,
    }


def levels_rows(doc: Dict[str, Any], esc=lambda s: s, num=None) -> List[tuple]:
    """The three levels of explanation as (label, text) rows for the certificate summary table.
    ``esc`` escapes text for the target format; ``num`` formats a number (default ``repr``)."""
    lv = (doc.get("coverage") or {}).get("levels")
    if not lv:
        return []
    num = num or (lambda v, d=4: ("n/a" if v is None else (f"{v:.{d}g}" if isinstance(v, float) else str(v))))
    f, d, u = lv["faithful"], lv["decomposed"], lv["understood"]
    fp = f.get("function_preservation_max_abs_err")
    faithful = (f"{'yes' if f.get('passed') else 'no'}: max abs error {num(fp)} (function preservation)"
                + (f", {num(f['closed_form_max_abs_err'])} (closed forms)" if f.get("closed_form_max_abs_err") is not None else "")
                if fp is not None else "not checked")
    lp = d.get("largest_part") or {}
    decomposed = (f"{d.get('n_leaves')} leaves; granularity {num(d.get('granularity'))}; largest part "
                  f"{esc(str(lp.get('name')))} ({lp.get('hidden_units')} units, width {lp.get('width')}, depth {lp.get('depth')})"
                  + ("; " + esc("; ".join(w.split(":")[0] for w in d["warnings"])) if d.get("warnings") else "")
                  if d.get("granularity") is not None else f"{d.get('n_leaves')} leaves; no hidden units")
    c = u.get("counts") or {}
    understood = (f"{num(u.get('understood_fraction'))} of {u.get('leaves')} leaves meaningful "
                  f"({c.get('meaningful', 0)} meaningful, {c.get('partial', 0)} partial, "
                  f"{c.get('closed_form_only', 0)} closed-form only, {c.get('unnamed', 0)} unnamed)"
                  if u.get("leaves") else "no leaves")
    return [("Level 1, faithful", faithful), ("Level 2, decomposed", decomposed),
            ("Level 3, understood", understood)]
