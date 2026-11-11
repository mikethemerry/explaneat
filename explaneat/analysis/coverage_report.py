"""Coverage report for an explained model state (paper Def 10-11).

Shared by the MCP ``get_coverage`` tool and the API
``GET /genomes/{id}/analyze/coverage`` route so that both report the same
numbers.
"""

from typing import Any, Dict, List

from .coverage import (
    CoverageComputer,
    compute_compositional_coverage_from_hierarchy,
    nodes_without_path_to_outputs,
)


def coverage_report(model_state, annotations_data) -> Dict[str, Any]:
    """Structural coverage, compositional coverage and the hierarchy block.

    Args:
        model_state: the current ``NetworkStructure`` (``engine.current_state``).
        annotations_data: ``engine.annotations`` (list of ``AnnotationData``).
    """
    all_node_ids = {str(nid) for nid in model_state.get_node_ids()}
    input_ids = {str(nid) for nid in model_state.input_node_ids}
    output_ids = {str(nid) for nid in model_state.output_node_ids}
    all_edges = {
        (str(c.from_node), str(c.to_node))
        for c in model_state.connections
        if c.enabled
    }
    # Null (exact, structural; decided 7 Nov 2026): anything that provably has
    # zero effect on every output, for every input, has the trivial explanation
    # "Null". That is every node with no path to an output over enabled,
    # non-zero-weight edges (disconnected inputs, dead ends), and every edge
    # that is zero-weight or touches such a node. Null parts count as covered
    # and are exempt from the single-root rule; they are reported separately.
    # Support-level nulls (e.g. a unit never active on the training data) are
    # NOT included here: they need an explicit, evidenced Null annotation.
    live_edges = {
        (str(c.from_node), str(c.to_node))
        for c in model_state.connections
        if c.enabled and c.weight != 0
    }
    null_nodes = nodes_without_path_to_outputs(all_node_ids, live_edges, output_ids)
    null_edges = {
        e for e in all_edges
        if e not in live_edges or e[0] in null_nodes or e[1] in null_nodes
    }

    # No annotations case
    if not annotations_data:
        candidate_nodes = all_node_ids - output_ids
        null_cand = null_nodes & candidate_nodes
        result = {
            "structural_coverage": (
                round(len(null_cand) / len(candidate_nodes), 4) if candidate_nodes else 0.0
            ),
            "compositional_coverage": 0.0,
            "node_coverage": {
                "covered": sorted(null_cand),
                "uncovered": sorted(candidate_nodes - null_cand),
                "total_candidate": len(candidate_nodes),
                "by_annotation": {},
            },
            "edge_coverage": {
                "covered": sorted([list(e) for e in null_edges]),
                "uncovered": sorted([list(e) for e in all_edges - null_edges]),
                "total": len(all_edges),
            },
            "null": {
                "basis": "exact: no path to an output over enabled, non-zero-weight edges",
                "nodes": sorted(null_cand),
                "edges": sorted([list(e) for e in null_edges]),
            },
            "annotations_count": 0,
            "leaf_count": 0,
            "composition_count": 0,
            "decomposition": decomposition_report(model_state, annotations_data),
        }
        return result

    # Convert AnnotationData to dicts expected by CoverageComputer
    ann_dicts = []
    for ann in annotations_data:
        ann_dicts.append({
            "id": ann.name,
            "entry_nodes": [str(n) for n in ann.entry_nodes],
            "exit_nodes": [str(n) for n in ann.exit_nodes],
            "subgraph_nodes": [str(n) for n in ann.subgraph_nodes],
            "subgraph_connections": [
                [str(e[0]), str(e[1])] for e in ann.subgraph_connections
            ],
        })

    # Classify leaf vs composition annotations
    # A leaf is any annotation with no children (regardless of whether it has a parent).
    # A composition is any annotation that has children.
    parent_names = {
        other.parent_annotation_id
        for other in annotations_data
        if other.parent_annotation_id is not None
    }
    leaf_anns = [
        a for a, ad in zip(ann_dicts, annotations_data)
        if ad.name not in parent_names
    ]
    composition_anns = [
        a for a, ad in zip(ann_dicts, annotations_data)
        if ad.name in parent_names
    ]
    leaf_count = len(leaf_anns)
    composition_count = len(composition_anns)

    # Build CoverageComputer with the model state graph
    computer = CoverageComputer(
        all_nodes=all_node_ids,
        all_edges=all_edges,
        input_nodes=input_ids,
        output_nodes=output_ids,
    )

    # Expand composition annotations to include all descendant subgraphs.
    # For covered_A(v) = (v ∈ V_A) ∧ (E_out(v) ⊆ E_A), a composition's
    # effective V_A/E_A is the union of its own subgraph and all
    # descendants' subgraphs recursively.
    ann_by_name: Dict[str, Dict] = {
        ad.name: a for a, ad in zip(ann_dicts, annotations_data)
    }

    def _descendant_names(name: str) -> set:
        children = [
            ad.name for ad in annotations_data
            if ad.parent_annotation_id == name
        ]
        result = set(children)
        for c in children:
            result.update(_descendant_names(c))
        return result

    expanded_ann_dicts = []
    for a, ad in zip(ann_dicts, annotations_data):
        if ad.name in parent_names:
            # Composition — merge descendant subgraphs
            desc_names = _descendant_names(ad.name)
            expanded_nodes = set(a["subgraph_nodes"])
            expanded_conns = {
                tuple(e) for e in a["subgraph_connections"]
            }
            for dn in desc_names:
                if dn in ann_by_name:
                    da = ann_by_name[dn]
                    expanded_nodes.update(da["subgraph_nodes"])
                    expanded_conns.update(
                        tuple(e) for e in da["subgraph_connections"]
                    )
            expanded_a = dict(a)
            expanded_a["subgraph_nodes"] = list(expanded_nodes)
            expanded_a["subgraph_connections"] = [
                list(e) for e in expanded_conns
            ]
            expanded_ann_dicts.append(expanded_a)
        else:
            expanded_ann_dicts.append(a)

    # Compute coverage using ALL annotations (with expanded compositions).
    # Structural coverage considers every annotation, not just leaves —
    # compositions own nodes (e.g. direct inputs) and edges that leaves
    # don't, and child exit nodes become covered only when a parent
    # composition absorbs both the child and the downstream target.
    node_cov, edge_cov = computer.compute_coverage(expanded_ann_dicts)

    # Candidate nodes: all non-output nodes (per paper)
    candidate_nodes = all_node_ids - output_ids
    covered_nodes = {n for n in candidate_nodes if n in node_cov or n in null_nodes}
    uncovered_nodes = candidate_nodes - covered_nodes

    structural_coverage = (
        len(covered_nodes) / len(candidate_nodes)
        if candidate_nodes
        else 1.0
    )

    # Compositional coverage (Def 11): fraction of the composition links
    # needed for a single global root (containing every output) that exist.
    total_annotations = len(ann_dicts)
    hierarchy = compute_compositional_coverage_from_hierarchy(
        [
            {
                "name": ad.name,
                "subgraph_nodes": [str(n) for n in ad.subgraph_nodes],
                "exit_nodes": [str(n) for n in ad.exit_nodes],
                "parent_annotation_id": ad.parent_annotation_id,
            }
            for ad in annotations_data
        ],
        output_ids,
        disconnected_nodes=null_nodes,
    )
    compositional_coverage = hierarchy["compositional_coverage"]

    # Build per-annotation node mapping
    by_annotation: Dict[str, List[str]] = {}
    for node_id, ann_ids in node_cov.items():
        if node_id in candidate_nodes:
            for aid in ann_ids:
                by_annotation.setdefault(aid, []).append(node_id)
    for aid in by_annotation:
        by_annotation[aid].sort()

    # Edge coverage
    covered_edges = sorted([list(e) for e in set(edge_cov.keys()) | (null_edges & all_edges)])
    all_edges_set = all_edges
    uncovered_edges = sorted(
        [list(e) for e in all_edges_set if e not in edge_cov and e not in null_edges]
    )

    result = {
        "structural_coverage": round(structural_coverage, 4),
        "compositional_coverage": round(compositional_coverage, 4),
        "node_coverage": {
            "covered": sorted(covered_nodes),
            "uncovered": sorted(uncovered_nodes),
            "total_candidate": len(candidate_nodes),
            "by_annotation": by_annotation,
        },
        "edge_coverage": {
            "covered": covered_edges,
            "uncovered": uncovered_edges,
            "total": len(all_edges_set),
        },
        "null": {
            "basis": "exact: no path to an output over enabled, non-zero-weight edges",
            "nodes": sorted(null_nodes & candidate_nodes),
            "edges": sorted([list(e) for e in null_edges]),
        },
        "annotations_count": total_annotations,
        "leaf_count": leaf_count,
        "composition_count": composition_count,
        "hierarchy": {
            "roots": hierarchy["roots"],
            "global_root": hierarchy["global_root"],
            "root_contains_outputs": hierarchy["root_contains_outputs"],
            "complete": hierarchy["complete"],
            "composed": hierarchy["composed"],
            "required": hierarchy["required"],
            "exempt_disconnected": hierarchy.get("exempt", []),
        },
        "decomposition": decomposition_report(model_state, annotations_data),
    }
    return result


# ============================================================================ decomposition
#: a part whose widest internal layer has more hidden units than this is "wide" (it holds
#: parallel units that are not read separately); depth is never penalised
SMALL_WIDTH = 6


def _base_id(node_id) -> str:
    """The original hidden unit a node is a copy of: ``5_a`` and ``5_ac`` -> ``5``."""
    s = str(node_id)
    return s.split("_", 1)[0] if not s.startswith("identity_") else s


def part_shape(ann, hidden_base: set) -> Dict[str, Any]:
    """Width and depth of one annotation's internal hidden units (split copies counted once).

    Layer of a non-entry node = longest path from an entry node over the annotation's own
    connections. ``width`` = most distinct hidden units in one layer; ``depth`` = number of
    layers. A chain of four units has width 1 and depth 4; a layer of 32 units in one part has
    width 32 and depth 1.
    """
    entries = {str(n) for n in ann.entry_nodes}
    nodes = {str(n) for n in ann.subgraph_nodes}
    succ: Dict[str, List[str]] = {}
    pred_count: Dict[str, int] = {n: 0 for n in nodes}
    for a, b in ann.subgraph_connections:
        a, b = str(a), str(b)
        if a in nodes and b in nodes:
            succ.setdefault(a, []).append(b)
            pred_count[b] = pred_count.get(b, 0) + 1
    layer: Dict[str, int] = {n: 0 for n in nodes}
    order = [n for n in nodes if pred_count.get(n, 0) == 0]
    seen = set(order)
    i = 0
    while i < len(order):
        n = order[i]
        i += 1
        for m in succ.get(n, []):
            layer[m] = max(layer[m], layer[n] + 1)
            pred_count[m] -= 1
            if pred_count[m] == 0 and m not in seen:
                seen.add(m)
                order.append(m)
    units_by_layer: Dict[int, set] = {}
    for n in nodes:
        if n in entries:
            continue
        b = _base_id(n)
        if b in hidden_base:
            units_by_layer.setdefault(max(layer.get(n, 1), 1), set()).add(b)
    units = set().union(*units_by_layer.values()) if units_by_layer else set()
    return {"units": len(units), "width": max((len(s) for s in units_by_layer.values()), default=0),
            "depth": len(units_by_layer), "unit_ids": sorted(units)}


def decomposition_report(model_state, annotations_data, small_width: int = SMALL_WIDTH) -> Dict[str, Any]:
    """How finely the explanation divides the hidden units: the second level (decomposed).

    Coverage can be 1.0 while one part swallows a whole wide layer (the explanation ladder: a
    32-unit layer as a single leaf). Each original hidden unit (split copies counted once) is
    assigned to the smallest annotation that owns a copy of it as a non-entry node. A part's
    **width** is its widest internal layer (:func:`part_shape`); depth is not penalised: a 3-4
    deep chain is a good leaf. ``granularity`` is the share of hidden units whose part has width
    at most ``small_width``. Warnings: one part wider than ``small_width`` holds at least half of
    the hidden units (a swallowed layer); hidden units owned only by the root composition;
    hidden units that appear in parts only as entry nodes.
    """
    from ..core.genome_network import NodeType

    if hasattr(model_state, "nodes"):
        hidden = {_base_id(n.id) for n in model_state.nodes
                  if n.type == NodeType.HIDDEN and not str(n.id).startswith("identity_")}
    else:   # a minimal graph object: everything that is neither input nor output
        io = {str(n) for n in model_state.input_node_ids} | {str(n) for n in model_state.output_node_ids}
        hidden = {_base_id(n) for n in model_state.get_node_ids()
                  if str(n) not in io and not str(n).startswith("identity_")}
    n_params = sum(1 for c in model_state.connections if c.enabled) + len(hidden)
    if not hidden or not annotations_data:
        return {"hidden_units": len(hidden), "granularity": None, "largest_part": None,
                "units_only_in_root": 0, "units_only_as_entries": len(hidden) if annotations_data else 0,
                "parts": {}, "warnings": [], "small_width_max_units": small_width}
    has_child = {ad.parent_annotation_id for ad in annotations_data if ad.parent_annotation_id}
    roots = {ad.name for ad in annotations_data if ad.parent_annotation_id is None}
    shapes = {ad.name: part_shape(ad, hidden) for ad in annotations_data}
    owned = {name: set(s["unit_ids"]) for name, s in shapes.items()}
    part_of: Dict[str, str] = {}
    for u in sorted(hidden):
        cands = [a for a, us in owned.items() if u in us]
        if cands:
            part_of[u] = min(cands, key=lambda a: (len(owned[a]), a))
    params_of = {}
    for ad in annotations_data:
        nodes = {str(n) for n in ad.subgraph_nodes}
        params_of[ad.name] = sum(1 for a, b in ad.subgraph_connections if str(a) in nodes and str(b) in nodes) \
            + shapes[ad.name]["units"]
    small = [u for u, a in part_of.items() if shapes[a]["width"] <= small_width]
    largest = max(shapes, key=lambda a: (shapes[a]["units"], a))
    root_only = sorted(u for u, a in part_of.items() if a in roots and a in has_child)
    entry_only = sorted(hidden - set(part_of))
    warnings = []
    if shapes[largest]["width"] > small_width and shapes[largest]["units"] >= 0.5 * len(hidden):
        warnings.append(f"coarse decomposition: part '{largest}' holds {shapes[largest]['units']} of the "
                        f"{len(hidden)} hidden units in a layer {shapes[largest]['width']} wide; coverage can "
                        "be complete while these units are explained only as a whole")
    if entry_only:
        warnings.append(f"{len(entry_only)} of the {len(hidden)} hidden units are inside no part: they "
                        "appear only as entry nodes, so no part explains them")
    if root_only:
        warnings.append(f"{len(root_only)} of the {len(hidden)} hidden units sit only in the root "
                        "composition, not in any leaf or sub-composition")
    parts = {a: {"units": s["units"], "width": s["width"], "depth": s["depth"], "leaf": a not in has_child,
                 "param_share": round(params_of[a] / n_params, 4) if n_params else None}
             for a, s in sorted(shapes.items(), key=lambda kv: (-kv[1]["units"], kv[0])) if s["units"]}
    return {
        "hidden_units": len(hidden),
        "small_width_max_units": small_width,
        "granularity": round(len(small) / len(hidden), 4),
        "largest_part": {"name": largest, "hidden_units": shapes[largest]["units"],
                         "width": shapes[largest]["width"], "depth": shapes[largest]["depth"],
                         "leaf": largest not in has_child,
                         "param_share": round(params_of[largest] / n_params, 4) if n_params else None},
        "max_width": max((s["width"] for s in shapes.values()), default=0),
        "max_depth": max((s["depth"] for s in shapes.values()), default=0),
        "units_only_in_root": len(root_only),
        "units_only_as_entries": len(entry_only),
        "parts": parts,
        "warnings": warnings,
    }
