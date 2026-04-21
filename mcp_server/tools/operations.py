"""MCP tools for managing the operation event stream on genome explanations."""

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from sqlalchemy.orm.attributes import flag_modified

from ..server import get_db
from ..helpers import (
    _to_uuid,
    build_engine,
    serialize_network,
)
from explaneat.db.models import Explanation, Genome


def list_operations(genome_id: str) -> str:
    """List all operations in the event stream for a genome.

    Returns the ordered list of operations that have been applied to the
    genome's model state. Each operation includes seq, type, params, result,
    created_at, and optional notes.

    Args:
        genome_id: UUID of the genome.
    """
    db = get_db()
    with db.session_scope() as session:
        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == _to_uuid(genome_id))
            .first()
        )
        if not explanation:
            return json.dumps({"operations": [], "total": 0}, indent=2)

        ops = explanation.operations or []
        return json.dumps({"operations": ops, "total": len(ops)}, indent=2, default=str)


def apply_operation(genome_id: str, operation_type: str, params: str, notes: Optional[str] = None) -> str:
    """Apply an operation to the genome's model state.

    Validates and applies the operation, persisting the updated event stream.
    Returns the new operation details and updated model state summary.

    Operation types and example params:
    - split_node: {"node_id": "5", "suffixes": ["a", "b"]}
    - consolidate_node: {"node_id": "5_a", "target_suffix": "a"}
    - add_identity_node: {"target_node_id": "5", "connection_sources": ["-1", "-2"]}
    - add_node: {"connection_from": "-1", "connection_to": "5"}
    - remove_node: {"node_id": "identity_5"}
    - annotate: {"name": "my_ann", "entry_nodes": ["-1", "-2"], "exit_nodes": ["5"], "subgraph_nodes": ["-1", "-2", "5"], "hypothesis": "Computes sum"}
    - rename_node: {"node_id": "5", "display_name": "Sum Node"}
    - rename_annotation: {"annotation_id": "ann_1", "display_name": "Better Name"}
    - disable_connection: {"from_node": "-1", "to_node": "5"}
    - enable_connection: {"from_node": "-1", "to_node": "5"}

    Args:
        genome_id: UUID of the genome.
        operation_type: One of the operation types listed above.
        params: JSON string of operation parameters.
        notes: Optional human-readable note about the operation.
    """
    db = get_db()
    with db.session_scope() as session:
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON params: {e}"}, indent=2)

        try:
            # Get or create explanation
            genome_uuid = _to_uuid(genome_id)
            genome_db = session.query(Genome).filter_by(id=genome_uuid).first()
            if not genome_db:
                return json.dumps({"error": f"Genome not found: {genome_id}"}, indent=2)

            explanation = (
                session.query(Explanation)
                .filter(Explanation.genome_id == genome_uuid)
                .first()
            )
            if not explanation:
                explanation = Explanation(
                    genome_id=genome_uuid,
                    is_well_formed=False,
                    operations=[],
                )
                session.add(explanation)
                session.flush()

            # Build engine and apply operation
            engine = build_engine(session, genome_id)
            new_op = engine.add_operation(operation_type, params_dict, validate=True, notes=notes)

            # Save updated operations
            explanation.operations = engine.to_dict()["operations"]
            flag_modified(explanation, "operations")
            session.flush()

            result = {
                "status": "applied",
                "operation": new_op.to_dict(),
                "total_operations": len(engine.operations),
            }
            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)


def validate_operation(genome_id: str, operation_type: str, params: str) -> str:
    """Validate an operation without applying it (dry run).

    Checks whether the operation would succeed against the current model state.
    Returns validation result with any errors.

    Args:
        genome_id: UUID of the genome.
        operation_type: Operation type to validate.
        params: JSON string of operation parameters.
    """
    db = get_db()
    with db.session_scope() as session:
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            return json.dumps({"valid": False, "errors": [f"Invalid JSON params: {e}"]}, indent=2)

        try:
            engine = build_engine(session, genome_id)
            errors = engine.validate_operation(operation_type, params_dict)
            return json.dumps({"valid": len(errors) == 0, "errors": errors}, indent=2)
        except Exception as e:
            return json.dumps({"valid": False, "errors": [str(e)]}, indent=2)


def undo_operation(genome_id: str, seq: int) -> str:
    """Remove an operation and all subsequent operations (undo).

    Removes the operation at the given sequence number and everything after it.
    The model state is recomputed from remaining operations.

    Args:
        genome_id: UUID of the genome.
        seq: Sequence number of the operation to remove (and all after it).
    """
    db = get_db()
    with db.session_scope() as session:
        try:
            genome_uuid = _to_uuid(genome_id)
            explanation = (
                session.query(Explanation)
                .filter(Explanation.genome_id == genome_uuid)
                .first()
            )
            if not explanation or not explanation.operations:
                return json.dumps({"error": "No operations to remove"}, indent=2)

            engine = build_engine(session, genome_id)
            removed = engine.remove_operation(seq)

            # Save updated operations
            explanation.operations = engine.to_dict()["operations"]
            flag_modified(explanation, "operations")
            session.flush()

            result = {
                "status": "removed",
                "removed_count": len(removed),
                "remaining_operations": len(engine.operations),
                "model_state": serialize_network(engine.current_state),
            }
            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)


def get_annotations(genome_id: str) -> str:
    """Get all annotations for a genome with hierarchy information.

    Extracts annotations from 'annotate' operations and resolves parent-child
    relationships. Also applies any rename_annotation operations for display names.

    Args:
        genome_id: UUID of the genome.
    """
    db = get_db()
    with db.session_scope() as session:
        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == _to_uuid(genome_id))
            .first()
        )

        if not explanation or not explanation.operations:
            return json.dumps({"annotations": [], "total": 0}, indent=2)

        # First pass: collect all annotations
        raw_annotations = []
        for op in explanation.operations:
            if op.get("type") != "annotate":
                continue

            params = op.get("params", {})
            result = op.get("result", {})

            ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"

            raw_annotations.append({
                "id": ann_id,
                "name": params.get("name"),
                "entry_nodes": [str(n) for n in (params.get("entry_nodes") or [])],
                "exit_nodes": [str(n) for n in (params.get("exit_nodes") or [])],
                "subgraph_nodes": [str(n) for n in (params.get("subgraph_nodes") or [])],
                "child_annotation_ids": params.get("child_annotation_ids") or [],
                "hypothesis": params.get("hypothesis"),
                "evidence": params.get("evidence"),
            })

        # Build name -> [ids] mapping
        name_to_ids: dict[str, list[str]] = {}
        for ann in raw_annotations:
            if ann["name"]:
                name_to_ids.setdefault(ann["name"], []).append(ann["id"])

        # Resolve children_ids for each annotation
        for ann in raw_annotations:
            children_ids = []
            for child_name in ann["child_annotation_ids"]:
                children_ids.extend(name_to_ids.get(child_name, []))
            ann["_children_ids"] = children_ids

        # Derive parent_annotation_id by inverting children
        child_to_parent_id: dict[str, str] = {}
        for ann in raw_annotations:
            for child_id in ann["_children_ids"]:
                if child_id not in child_to_parent_id:
                    child_to_parent_id[child_id] = ann["id"]

        # Scan for rename_annotation operations
        annotation_display_names: dict[str, Optional[str]] = {}
        for op in explanation.operations:
            if op.get("type") == "rename_annotation":
                params = op.get("params", {})
                ann_id = params.get("annotation_id")
                if ann_id:
                    annotation_display_names[ann_id] = params.get("display_name")

        # Build final annotation list
        annotations = []
        for ann in raw_annotations:
            parent_id = child_to_parent_id.get(ann["id"])
            children_ids = ann["_children_ids"]

            # Compute evidence metadata
            evidence = ann.get("evidence") or {}
            evidence_count = 0
            evidence_types = []
            if "records" in evidence:
                records = evidence["records"]
                evidence_count = len(records)
                evidence_types = sorted({r.get("type", "") for r in records if r.get("type")})
            else:
                # Legacy category-based evidence entries
                evidence_count = sum(
                    1 for k, v in evidence.items()
                    if isinstance(v, dict) or isinstance(v, list)
                )

            annotations.append({
                "id": ann["id"],
                "name": ann["name"],
                "display_name": annotation_display_names.get(ann["name"]),
                "hypothesis": ann.get("hypothesis"),
                "entry_nodes": ann["entry_nodes"],
                "exit_nodes": ann["exit_nodes"],
                "subgraph_nodes": ann["subgraph_nodes"],
                "parent_annotation_id": parent_id,
                "children_ids": children_ids,
                "is_leaf": len(children_ids) == 0,
                "is_composition": len(children_ids) > 0,
                "evidence_count": evidence_count,
                "evidence_types": evidence_types,
            })

        return json.dumps({"annotations": annotations, "total": len(annotations)}, indent=2)


def register(mcp: FastMCP) -> None:
    """Register operation tools with the MCP server."""
    mcp.tool()(list_operations)
    mcp.tool()(apply_operation)
    mcp.tool()(validate_operation)
    mcp.tool()(undo_operation)
    mcp.tool()(get_annotations)
