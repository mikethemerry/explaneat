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
from explaneat.db.explanation_lock import lock_explanation
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

    Operation types and example params. Node ids are strings; connections are
    [from, to] pairs. The examples apply, in this order, to a network with
    inputs -1,-2,-3, hidden 5,6, output 0 and edges -1->5, -2->5, 5->0, 5->6,
    -3->6, 6->0, -3->0 (checked by tests/test_mcp/test_apply_operation_docstring.py):
    - split_node: {"node_id": "-3"}
    - consolidate_node: {"node_ids": ["-3_a", "-3_b"]}
    - add_identity_node: {"target_node": "0", "connections": [["6", "0"]], "new_node_id": "identity_0"}
    - add_node: {"connection": ["-1", "5"], "new_node_id": "n1"}
    - remove_node: {"node_id": "n1"}
    - disable_connection: {"from_node": "-2", "to_node": "5"}
    - enable_connection: {"from_node": "-2", "to_node": "5"}
    - rename_node: {"node_id": "5", "display_name": "sumNode"}
    - annotate: {"name": "A", "hypothesis": "Computes sum", "entry_nodes": ["-1", "-2"], "exit_nodes": ["5"], "subgraph_nodes": ["-1", "-2", "5"], "subgraph_connections": [["-1", "5"], ["-2", "5"]]}
    - rename_annotation: {"annotation_id": "A", "display_name": "sumOfInputs"}
    - prune_connection: {"from_node": "-3_ab", "to_node": "6"}
    - prune_node: {"node_id": "6"}

    Notes:
    - split_node: node needs >= 2 enabled outputs (not an output node); it is
      replaced by one copy per output, named {id}_a, {id}_b, ... in ascending
      target-id order (-3 -> -3_a feeding 0, -3_b feeding 6). consolidate_node
      takes >= 2 split ids of one original and merges them into a NEW node
      named by concatenating suffixes (-3_a + -3_b -> -3_ab), not the original id.
    - add_identity_node: every connection must end at target_node, and at
      least one of target_node's inputs must stay un-redirected.
    - add_node inserts new_node_id into an existing connection (optional
      "bias", default 0.0, and "activation", default "identity").
      remove_node / prune_node require exactly 1 enabled input and 1 output.
    - annotate: all six keys are required; optional "child_annotation_ids"
      (names or ann_<seq> ids) makes a composition annotation. The hypothesis is
      checked at write time (see update_narrative): numbers are referenced as
      {{q:<id>.<path>|<fmt>}} placeholders (rendered from the evidence log; the raw
      template is kept as "hypothesis_template"), exact formulas must reproduce, and in
      strict mode (new explanations) bare typed numbers are refused. Optional
      "allow_numbers": literals that are not computed numbers (e.g. a count in a name).
    - rename_node / rename_annotation: display_name must have no spaces
      (camelCase), or be null to clear; annotation_id may be the name or ann_<seq>.

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
            # Lock (or race-safely create) the explanation row before reading
            # the operations stream; held until the session commits.
            genome_uuid = _to_uuid(genome_id)
            try:
                explanation = lock_explanation(session, genome_uuid, create=True)
            except ValueError:
                return json.dumps({"error": f"Genome not found: {genome_id}"}, indent=2)

            # Build engine and apply operation
            engine = build_engine(session, genome_id)
            allow = params_dict.pop("allow_numbers", None) or []
            new_op = engine.add_operation(operation_type, params_dict, validate=True, notes=notes)

            text_check = None
            if operation_type == "annotate" and new_op.params.get("hypothesis"):
                from explaneat.analysis.narrative_check import process_text
                ops_now = [o.to_dict() for o in engine.operations]
                text_check = process_text(db, genome_id, new_op.params["hypothesis"], ops_now,
                                          part_name=new_op.params.get("name"),
                                          location=f"operations.{new_op.params.get('name')}.hypothesis",
                                          allow=allow, session=session)
                if text_check["errors"]:
                    return json.dumps({"error": "hypothesis refused (strict mode): " + "; ".join(text_check["errors"]),
                                       "text_check": text_check}, indent=2, default=str)
                new_op.params["hypothesis"] = text_check["text"]
                if text_check["template"] is not None:
                    new_op.params["hypothesis_template"] = text_check["template"]
                    new_op.params["provenance"] = text_check["provenance"]

            # Save updated operations
            explanation.operations = engine.to_dict()["operations"]
            flag_modified(explanation, "operations")
            session.flush()

            result = {
                "status": "applied",
                "operation": new_op.to_dict(),
                "total_operations": len(engine.operations),
            }
            if text_check is not None:
                result["text_check"] = {k: text_check[k] for k in ("strict", "warnings", "findings", "provenance")}
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
            explanation = lock_explanation(session, genome_uuid)
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
