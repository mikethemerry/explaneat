"""MCP tools for coverage analysis and node classification."""

import json
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from ..server import get_db
from ..helpers import _to_uuid, build_model_state


def classify_nodes(
    genome_id: str,
    node_ids: str,
    entry_node_ids: str,
    exit_node_ids: str,
) -> str:
    """Classify nodes within a proposed coverage as entry, intermediate, or exit.

    Entry nodes have external inputs but no external outputs.
    Intermediate nodes have no external inputs or outputs.
    Exit nodes have external outputs.

    Args:
        genome_id: UUID of the genome.
        node_ids: JSON array of node IDs forming the proposed coverage.
        entry_node_ids: JSON array of proposed entry node IDs.
        exit_node_ids: JSON array of proposed exit node IDs.
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
    entry_node_ids: str,
    exit_node_ids: str,
) -> str:
    """Detect nodes that must be split before creating an annotation.

    Given a proposed coverage, identifies nodes that violate entry/exit
    constraints and would need to be split first.

    Args:
        genome_id: UUID of the genome.
        node_ids: JSON array of node IDs forming the proposed coverage.
        entry_node_ids: JSON array of proposed entry node IDs.
        exit_node_ids: JSON array of proposed exit node IDs.
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
    """Get coverage analysis for the current explanation.

    Returns structural coverage metrics: fraction of hidden nodes covered
    by annotations, lists of covered and uncovered nodes.

    Args:
        genome_id: UUID of the genome.
    """
    from explaneat.db.models import Explanation

    db = get_db()
    with db.session_scope() as session:
        model_state = build_model_state(session, genome_id)

        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == _to_uuid(genome_id))
            .first()
        )

        all_node_ids = set(model_state.get_node_ids())
        input_ids = set(model_state.input_node_ids)
        output_ids = set(model_state.output_node_ids)
        hidden_nodes = all_node_ids - input_ids - output_ids

        if not explanation or not explanation.operations:
            output = {
                "structural_coverage": 0.0,
                "covered_nodes": [],
                "uncovered_nodes": sorted(hidden_nodes),
                "total_hidden_nodes": len(hidden_nodes),
                "annotations_count": 0,
            }
            return json.dumps(output, indent=2, default=str)

        covered_nodes = set()
        annotations_count = 0

        for op in explanation.operations:
            if op.get("type") == "annotate":
                params = op.get("params", {})
                subgraph_nodes = params.get("subgraph_nodes", [])
                covered_nodes.update(subgraph_nodes)
                annotations_count += 1

        # Only count hidden nodes for coverage
        covered_hidden = covered_nodes & hidden_nodes
        uncovered_hidden = hidden_nodes - covered_nodes

        if len(hidden_nodes) > 0:
            structural_coverage = len(covered_hidden) / len(hidden_nodes)
        else:
            structural_coverage = 0.0

        output = {
            "structural_coverage": structural_coverage,
            "covered_nodes": sorted(covered_hidden),
            "uncovered_nodes": sorted(uncovered_hidden),
            "total_hidden_nodes": len(hidden_nodes),
            "annotations_count": annotations_count,
        }
        return json.dumps(output, indent=2, default=str)


def register(mcp: FastMCP) -> None:
    """Register coverage tools with the MCP server."""
    mcp.tool()(classify_nodes)
    mcp.tool()(detect_splits)
    mcp.tool()(get_coverage)
