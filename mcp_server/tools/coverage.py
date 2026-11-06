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
        engine = build_engine(session, genome_id)
        model_state = engine.current_state
        result = classify_coverage(model_state, coverage, annotations=engine.annotations)

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
            "already_covered": result.already_covered,
        }
        return json.dumps(output, indent=2, default=str)


def detect_splits(
    genome_id: str,
    node_ids: str,
) -> str:
    """Detect nodes that must be split before creating an annotation.

    Given a proposed coverage, identifies nodes that violate entry/exit
    constraints and would need to be split first. Returns violations with
    suggested split_node operations to fix them. Input nodes that also feed
    nodes outside the coverage (fan-out across annotations) are flagged with
    reason "input_feeds_outside_coverage". Nodes already owned by an existing
    annotation are listed in "already_covered" (annotate would reject them).

    Args:
        genome_id: UUID of the genome.
        node_ids: JSON array of node IDs forming the proposed coverage.
    """
    from explaneat.analysis.split_detection import analyze_coverage_for_splits

    coverage = json.loads(node_ids)

    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        model_state = engine.current_state
        result = analyze_coverage_for_splits(
            model_state, coverage, annotations=engine.annotations,
        )

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
            "already_covered": result.already_covered,
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
    - **Compositional coverage** (Def 11): fraction of the composition links
      required for a single global root that exist. Every annotation except
      the global root must have a parent; if no root's subgraph (including
      descendants) contains every output node, the global composition is
      missing too. 1.0 iff there is exactly one root, it contains every
      output, and every annotation descends from it. The "hierarchy" field
      reports roots / global_root / composed / required.

    Args:
        genome_id: UUID of the genome.
    """
    from explaneat.analysis.coverage_report import coverage_report

    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        result = coverage_report(engine.current_state, engine.annotations)
        return json.dumps(result, indent=2, default=str)


def register(mcp: FastMCP) -> None:
    """Register coverage tools with the MCP server."""
    mcp.tool()(classify_nodes)
    mcp.tool()(detect_splits)
    mcp.tool()(get_coverage)
