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

    # No annotations case
    if not annotations_data:
        candidate_nodes = all_node_ids - output_ids
        result = {
            "structural_coverage": 0.0,
            "compositional_coverage": 0.0,
            "node_coverage": {
                "covered": [],
                "uncovered": sorted(candidate_nodes),
                "total_candidate": len(candidate_nodes),
                "by_annotation": {},
            },
            "edge_coverage": {
                "covered": [],
                "uncovered": sorted([list(e) for e in all_edges]),
                "total": len(all_edges),
            },
            "annotations_count": 0,
            "leaf_count": 0,
            "composition_count": 0,
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
    covered_nodes = {n for n in candidate_nodes if n in node_cov}
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
        disconnected_nodes=nodes_without_path_to_outputs(all_node_ids, all_edges, output_ids),
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
    covered_edges = sorted([list(e) for e in edge_cov.keys()])
    all_edges_set = all_edges
    uncovered_edges = sorted(
        [list(e) for e in all_edges_set if e not in edge_cov]
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
    }
    return result
