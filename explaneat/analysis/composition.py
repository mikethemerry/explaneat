"""Composition (hierarchical) annotations as functions of their whole subtree.

A composition annotation's own ``subgraph_nodes`` contain only its junction
nodes: the nodes of its children (and their descendants) are owned by those
annotations, because the annotate validator forbids covering a node twice.
To *evaluate* or *render* a composition we therefore need the union of the
subtree.  ``expand_annotation`` builds that view:

* the subgraph is the backward closure from the composition's exits over
  the union of its own and all descendants' nodes, stopping at the declared
  entry nodes — so a composition declared over its children's raw entries
  includes the children's internals, while one declared over its children's
  exits (the "collapsed" style) stays a function of those exits;
* source nodes of that closure which are not declared entries (typically
  grandchild entries that the parent could not list before the validator
  counted descendants) are appended to ``entry_nodes`` and reported in
  ``added_entry_nodes``;
* ``subgraph_connections`` are the enabled model connections inside the
  closure, i.e. exactly what the forward pass computes.

Leaf annotations are returned unchanged.
"""

from typing import Any, Dict, Iterable, List, Optional, Set

from ..core.genome_network import NetworkStructure


def annotation_dicts_from_operations(operations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotation dicts (as returned by the MCP/API lookups) for every annotate op."""
    out = []
    for op in operations or []:
        if op.get("type") != "annotate":
            continue
        params = op.get("params", {}) or {}
        result = op.get("result") or {}
        ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"
        out.append({
            "id": ann_id,
            "name": params.get("name"),
            "hypothesis": params.get("hypothesis"),
            "entry_nodes": [str(n) for n in params.get("entry_nodes", []) or []],
            "exit_nodes": [str(n) for n in params.get("exit_nodes", []) or []],
            "subgraph_nodes": [str(n) for n in params.get("subgraph_nodes", []) or []],
            "subgraph_connections": [
                [str(c[0]), str(c[1])] for c in params.get("subgraph_connections", []) or []
            ],
            "evidence": params.get("evidence") or {},
            "child_annotation_ids": list(params.get("child_annotation_ids", []) or []),
            "parent_annotation_id": params.get("parent_annotation_id"),
        })
    return out


def _find(ref: Any, annotations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find an annotation by name (preferred) or synthetic id."""
    ref = str(ref)
    for ann in annotations:
        if ann.get("name") == ref:
            return ann
    for ann in annotations:
        if ann.get("id") == ref:
            return ann
    return None


def child_annotations(annotation: Dict[str, Any], annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Direct children of ``annotation`` (refs may be names or ann_<seq> ids)."""
    out = []
    for ref in annotation.get("child_annotation_ids", []) or []:
        child = _find(ref, annotations)
        if child is not None and child is not annotation:
            out.append(child)
    return out


def descendant_annotations(annotation: Dict[str, Any], annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All descendants (children, grandchildren, ...) in breadth-first order."""
    out: List[Dict[str, Any]] = []
    seen = {id(annotation)}
    frontier = [annotation]
    while frontier:
        nxt = []
        for ann in frontier:
            for child in child_annotations(ann, annotations):
                if id(child) in seen:
                    continue
                seen.add(id(child))
                out.append(child)
                nxt.append(child)
        frontier = nxt
    return out


def expand_annotation(
    annotation: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    structure: NetworkStructure,
) -> Dict[str, Any]:
    """Return ``annotation`` expanded over its descendants (see module docs)."""
    descendants = descendant_annotations(annotation, annotations)
    if not descendants:
        return annotation

    declared_entries = [str(n) for n in annotation.get("entry_nodes", [])]
    exits = [str(n) for n in annotation.get("exit_nodes", [])]
    entry_set = set(declared_entries)

    union: Set[str] = set(str(n) for n in annotation.get("subgraph_nodes", []))
    union.update(exits)
    union.update(declared_entries)
    for desc in descendants:
        union.update(str(n) for n in desc.get("subgraph_nodes", []))
        union.update(str(n) for n in desc.get("entry_nodes", []))
        union.update(str(n) for n in desc.get("exit_nodes", []))

    model_nodes = set(structure.get_node_ids())
    union &= model_nodes

    preds: Dict[str, List[str]] = {}
    for c in structure.connections:
        if c.enabled and c.from_node in union and c.to_node in union:
            preds.setdefault(c.to_node, []).append(c.from_node)

    # Backward closure from the exits, stopping at declared entries.
    needed: Set[str] = set()
    stack = [n for n in exits if n in union]
    while stack:
        n = stack.pop()
        if n in needed:
            continue
        needed.add(n)
        if n in entry_set:
            continue
        stack.extend(p for p in preds.get(n, []) if p not in needed)

    # Sources of the closure that the annotation did not declare as entries.
    added: List[str] = []
    ordered_candidates = []
    for desc in descendants:
        ordered_candidates.extend(str(n) for n in desc.get("entry_nodes", []))
    ordered_candidates.extend(sorted(needed))
    for n in ordered_candidates:
        if n in needed and n not in entry_set and n not in added and not preds.get(n):
            added.append(n)

    entries = declared_entries + added
    nodes = list(declared_entries)
    for n in sorted(needed):
        if n not in nodes:
            nodes.append(n)
    node_set = set(nodes)
    internal = node_set - set(entries)
    connections = [
        [c.from_node, c.to_node]
        for c in structure.connections
        if c.enabled and c.to_node in internal and c.from_node in node_set
    ]

    expanded = dict(annotation)
    expanded.update({
        "entry_nodes": entries,
        "exit_nodes": exits,
        "subgraph_nodes": nodes,
        "subgraph_connections": connections,
        "added_entry_nodes": added,
        "descendant_annotation_ids": [d.get("id") for d in descendants],
    })
    return expanded
