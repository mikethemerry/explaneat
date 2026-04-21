"""MCP tools for coverage analysis and node classification."""

import json
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from ..server import get_db
from ..helpers import _to_uuid, build_model_state, build_engine


def classify_nodes(
    genome_id: str,
    node_ids: str,
) -> str:
    """Classify nodes within a proposed coverage as entry, intermediate, or exit.

    Entry nodes receive inputs from outside the coverage.
    Exit nodes send outputs to outside the coverage.
    Intermediate nodes only connect within the coverage.

    Args:
        genome_id: UUID of the genome.
        node_ids: JSON array of node IDs forming the proposed coverage.
    """
    from explaneat.analysis.node_classification import classify_coverage

    coverage = json.loads(node_ids)

    db = get_db()
    with db.session_scope() as session:
        model_state = build_model_state(session, genome_id)
        result = classify_coverage(model_state, coverage)

        violations = []
        for v in result.violations:
            violations.append({
                "node_id": v["node_id"],
                "reason": v["reason"],
                "external_inputs": v["external_inputs"],
                "external_outputs": v["external_outputs"],
            })

        output = {
            "coverage": result.coverage,
            "classification": {
                "entry": result.entry_nodes,
                "intermediate": result.intermediate_nodes,
                "exit": result.exit_nodes,
            },
            "valid": result.valid,
            "violations": violations,
        }
        return json.dumps(output, indent=2, default=str)


def detect_splits(
    genome_id: str,
    node_ids: str,
) -> str:
    """Detect nodes that must be split before creating an annotation.

    Given a proposed coverage, identifies nodes that violate entry/exit
    constraints and would need to be split first. Returns violations with
    suggested split_node operations to fix them.

    Args:
        genome_id: UUID of the genome.
        node_ids: JSON array of node IDs forming the proposed coverage.
    """
    from explaneat.analysis.split_detection import analyze_coverage_for_splits

    coverage = json.loads(node_ids)

    db = get_db()
    with db.session_scope() as session:
        model_state = build_model_state(session, genome_id)
        result = analyze_coverage_for_splits(model_state, coverage)

        violations = []
        for v in result.violations:
            violations.append({
                "node_id": v.node_id,
                "reason": v.reason,
                "external_inputs": list(v.external_inputs),
                "external_outputs": list(v.external_outputs),
                "internal_outputs": list(v.internal_outputs) if v.internal_outputs else None,
            })

        suggested_operations = []
        for op in result.suggested_operations:
            suggested_operations.append({
                "type": op["type"],
                "params": op["params"],
            })

        output = {
            "proposed_coverage": result.proposed_coverage,
            "violations": violations,
            "suggested_operations": suggested_operations,
            "adjusted_coverage": result.adjusted_coverage,
        }
        return json.dumps(output, indent=2, default=str)


def get_coverage(genome_id: str) -> str:
    """Get coverage analysis for the current explanation using paper definitions (Def 10-11).

    Uses the CoverageComputer to compute formal coverage metrics:

    - **Structural coverage** (Def 10): fraction of candidate nodes covered by
      leaf annotations. A node v is covered by annotation A when v is in A's
      subgraph AND all of v's outgoing edges are contained within A's subgraph
      (covered_A(v) = (v in V_A) and (E_out(v) subset E_A)). Candidate nodes
      are all non-output nodes.
    - **Compositional coverage** (Def 11): fraction of required composition
      annotations that exist. For a hierarchy with leaf and composition
      annotations, measures how completely the hierarchy is specified.

    Args:
        genome_id: UUID of the genome.
    """
    from explaneat.analysis.coverage import CoverageComputer

    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        model_state = engine.current_state
        annotations_data = engine.annotations  # List[AnnotationData]

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
                "compositional_coverage": 1.0,
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
            return json.dumps(result, indent=2, default=str)

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
        leaf_anns = [
            a for a, ad in zip(ann_dicts, annotations_data)
            if ad.parent_annotation_id is None
            and not any(
                other.parent_annotation_id == ad.name
                for other in annotations_data
            )
        ]
        composition_anns = [
            a for a, ad in zip(ann_dicts, annotations_data)
            if any(
                other.parent_annotation_id == ad.name
                for other in annotations_data
            )
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

        # Compute coverage using leaf annotations (structural coverage = C_V(A_leaf))
        # Use all annotations for the full picture, leaf for structural
        leaf_node_cov, leaf_edge_cov = computer.compute_coverage(
            leaf_anns if leaf_anns else ann_dicts
        )

        # Candidate nodes: all non-output nodes (per paper)
        candidate_nodes = all_node_ids - output_ids
        covered_nodes = {n for n in candidate_nodes if n in leaf_node_cov}
        uncovered_nodes = candidate_nodes - covered_nodes

        structural_coverage = (
            len(covered_nodes) / len(candidate_nodes)
            if candidate_nodes
            else 1.0
        )

        # Compositional coverage: composition_count / required_compositions
        # required = total_annotations - leaf_count (number of internal hierarchy nodes)
        total_annotations = len(ann_dicts)
        required_compositions = max(0, total_annotations - leaf_count)
        if required_compositions == 0:
            compositional_coverage = 1.0 if composition_count == 0 else 0.0
        else:
            compositional_coverage = min(1.0, composition_count / required_compositions)

        # Build per-annotation node mapping
        by_annotation: Dict[str, List[str]] = {}
        for node_id, ann_ids in leaf_node_cov.items():
            if node_id in candidate_nodes:
                for aid in ann_ids:
                    by_annotation.setdefault(aid, []).append(node_id)
        for aid in by_annotation:
            by_annotation[aid].sort()

        # Edge coverage
        covered_edges = sorted([list(e) for e in leaf_edge_cov.keys()])
        all_edges_set = all_edges
        uncovered_edges = sorted(
            [list(e) for e in all_edges_set if e not in leaf_edge_cov]
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
        }
        return json.dumps(result, indent=2, default=str)


def register(mcp: FastMCP) -> None:
    """Register coverage tools with the MCP server."""
    mcp.tool()(classify_nodes)
    mcp.tool()(detect_splits)
    mcp.tool()(get_coverage)
