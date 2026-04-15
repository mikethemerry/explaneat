"""
Operation handlers for model manipulations.

Each operation modifies the NetworkStructure in place and returns
a result dictionary with information about what was changed.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union, TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from .model_state import AnnotationData

from .genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)


class OperationError(Exception):
    """Raised when an operation fails."""

    pass


# =============================================================================
# Operation classification
# =============================================================================

IDENTITY_OPS = frozenset({
    "split_node",
    "consolidate_node",
    "remove_node",
    "add_node",
    "add_identity_node",
    "annotate",
    "rename_node",
    "rename_annotation",
})
"""Operations that restructure the network without changing its computed function."""

NON_IDENTITY_OPS = frozenset({
    "disable_connection",
    "enable_connection",
    "prune_node",
    "prune_connection",
    "retrain",
})
"""Operations that change the network's computed function."""


def is_identity_op(op_type: str) -> bool:
    """Return True if the operation preserves the network's function."""
    return op_type in IDENTITY_OPS


# =============================================================================
# Validation
# =============================================================================


def validate_operation(
    model: NetworkStructure,
    op_type: str,
    params: Dict[str, Any],
    covered_nodes: Set[str],
    covered_connections: Set[Tuple[str, str]],
    existing_annotations: Optional[List["AnnotationData"]] = None,
) -> List[str]:
    """
    Validate an operation against the current model state.

    Args:
        model: Current network structure
        op_type: Operation type
        params: Operation parameters
        covered_nodes: Nodes covered by annotations (immutable)
        covered_connections: Connections covered by annotations (immutable)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    node_ids = model.get_node_ids()

    if op_type == "split_node":
        node_id = params.get("node_id")
        if not node_id:
            errors.append("node_id is required")
        elif node_id not in node_ids:
            errors.append(f"Node {node_id} does not exist")
        elif node_id in covered_nodes:
            errors.append(f"Node {node_id} is covered by an annotation and cannot be modified")
        else:
            # Check node has multiple outputs
            outputs = model.get_connections_from(node_id)
            enabled_outputs = [c for c in outputs if c.enabled]
            if len(enabled_outputs) < 2:
                errors.append(f"Node {node_id} has fewer than 2 outputs, cannot split")

            # Check not an output node (inputs and hidden nodes can be split)
            node = model.get_node_by_id(node_id)
            if node and node.type == NodeType.OUTPUT:
                errors.append(f"Cannot split output node {node_id}")

    elif op_type == "consolidate_node":
        node_ids_to_consolidate = params.get("node_ids", [])
        if len(node_ids_to_consolidate) < 2:
            errors.append("At least 2 node_ids required for consolidation")
        else:
            # Check all nodes exist
            for nid in node_ids_to_consolidate:
                if nid not in node_ids:
                    errors.append(f"Node {nid} does not exist")
                if nid in covered_nodes:
                    errors.append(f"Node {nid} is covered by an annotation")

            # Check all are from the same original split
            base_ids = set()
            for nid in node_ids_to_consolidate:
                base = _get_base_node_id(nid)
                if base:
                    base_ids.add(base)
                else:
                    errors.append(f"Node {nid} is not a split node")

            if len(base_ids) > 1:
                errors.append("Cannot consolidate nodes from different original splits")

    elif op_type == "remove_node":
        node_id = params.get("node_id")
        if not node_id:
            errors.append("node_id is required")
        elif node_id not in node_ids:
            errors.append(f"Node {node_id} does not exist")
        elif node_id in covered_nodes:
            errors.append(f"Node {node_id} is covered by an annotation")
        else:
            node = model.get_node_by_id(node_id)
            if node and node.type in (NodeType.INPUT, NodeType.OUTPUT):
                errors.append(f"Cannot remove {node.type.value} node {node_id}")

            # Check exactly 1 input and 1 output
            inputs = [c for c in model.get_connections_to(node_id) if c.enabled]
            outputs = [c for c in model.get_connections_from(node_id) if c.enabled]
            if len(inputs) != 1:
                errors.append(f"Node {node_id} must have exactly 1 input (has {len(inputs)})")
            if len(outputs) != 1:
                errors.append(f"Node {node_id} must have exactly 1 output (has {len(outputs)})")

    elif op_type == "add_node":
        connection = params.get("connection")
        new_node_id = params.get("new_node_id")

        if not connection or len(connection) != 2:
            errors.append("connection must be [from_node, to_node]")
        else:
            from_node, to_node = connection
            # Check connection exists
            conn_exists = any(
                c.from_node == from_node and c.to_node == to_node and c.enabled
                for c in model.connections
            )
            if not conn_exists:
                errors.append(f"Connection [{from_node}, {to_node}] does not exist")
            if (from_node, to_node) in covered_connections:
                errors.append(f"Connection [{from_node}, {to_node}] is covered by an annotation")

        if not new_node_id:
            errors.append("new_node_id is required")
        elif new_node_id in node_ids:
            errors.append(f"Node {new_node_id} already exists")

    elif op_type == "add_identity_node":
        target_node = params.get("target_node")
        connections = params.get("connections", [])
        new_node_id = params.get("new_node_id")

        if not target_node:
            errors.append("target_node is required")
        elif target_node not in node_ids:
            errors.append(f"Target node {target_node} does not exist")

        if not connections:
            errors.append("At least one connection required")
        else:
            # Check all connections exist and end at target
            all_inputs = model.get_connections_to(target_node) if target_node else []
            all_input_pairs = {(c.from_node, c.to_node) for c in all_inputs if c.enabled}

            for conn in connections:
                if len(conn) != 2:
                    errors.append(f"Invalid connection format: {conn}")
                    continue
                from_node, to_node = conn
                if to_node != target_node:
                    errors.append(f"Connection {conn} does not end at target node {target_node}")
                if (from_node, to_node) not in all_input_pairs:
                    errors.append(f"Connection {conn} does not exist")
                if (from_node, to_node) in covered_connections:
                    errors.append(f"Connection {conn} is covered by an annotation")

            # Check not ALL connections are specified
            if len(connections) >= len(all_input_pairs) and len(all_input_pairs) > 0:
                errors.append("Cannot redirect ALL connections to target (must leave at least one)")

        if not new_node_id:
            errors.append("new_node_id is required")
        elif new_node_id in node_ids:
            errors.append(f"Node {new_node_id} already exists")

    elif op_type == "annotate":
        # Validate annotation parameters
        required = ["name", "hypothesis", "entry_nodes", "exit_nodes", "subgraph_nodes", "subgraph_connections"]
        for field in required:
            if field not in params:
                errors.append(f"{field} is required")

        if not errors:
            subgraph_nodes = set(params["subgraph_nodes"])
            entry_nodes = set(params["entry_nodes"])
            exit_nodes = set(params["exit_nodes"])

            # Check all nodes exist
            for nid in subgraph_nodes:
                if nid not in node_ids:
                    errors.append(f"Node {nid} does not exist")
                if nid in covered_nodes:
                    errors.append(f"Node {nid} is already covered by another annotation")

            # Check connections exist
            for conn in params["subgraph_connections"]:
                if len(conn) != 2:
                    errors.append(f"Invalid connection format: {conn}")
                    continue
                from_node, to_node = conn
                conn_exists = any(
                    c.from_node == from_node and c.to_node == to_node
                    for c in model.connections
                )
                if not conn_exists:
                    errors.append(f"Connection {conn} does not exist")

            # --- Composition-specific checks ---
            child_annotation_ids = set(params.get("child_annotation_ids", []))
            existing = existing_annotations or []

            # Exit nodes must not be empty
            if not exit_nodes:
                errors.append("Annotation must have at least one exit node")

            if child_annotation_ids:
                # Children must exist
                for child_name in child_annotation_ids:
                    found = any(ann.name == child_name for ann in existing)
                    if not found:
                        errors.append(f"Child annotation '{child_name}' not found")

                # No dual parents — child must not already have a parent
                for child_name in child_annotation_ids:
                    for ann in existing:
                        if ann.name == child_name and ann.parent_annotation_id is not None:
                            errors.append(
                                f"Child '{child_name}' already has parent '{ann.parent_annotation_id}'"
                            )

                # Children must be leaf annotations (no grandchildren skipping)
                for child_name in child_annotation_ids:
                    for ann in existing:
                        if ann.name == child_name:
                            child_has_children = any(
                                a.parent_annotation_id == ann.name for a in existing
                            )
                            if child_has_children:
                                errors.append(
                                    f"Child '{child_name}' already has children — cannot be claimed as leaf"
                                )

            # Build set of nodes inside child annotations (for compositional annotations)
            # Connections to these nodes are considered "internal" even though they're
            # not in this annotation's subgraph (they belong to child annotations)
            nodes_inside_children = set()
            if child_annotation_ids:
                for ann in existing:
                    if ann.name in child_annotation_ids:
                        nodes_inside_children.update(ann.subgraph_nodes)

            # Combined set of "internal" nodes for validation
            internal_nodes = subgraph_nodes | nodes_inside_children

            # Check entry/exit are subsets of internal nodes
            # (entry/exit can be in this annotation's subgraph or in child subgraphs)
            if not entry_nodes.issubset(internal_nodes):
                errors.append("entry_nodes must be subset of subgraph_nodes (including children)")
            if not exit_nodes.issubset(internal_nodes):
                errors.append("exit_nodes must be subset of subgraph_nodes (including children)")

            # --- Three-precondition boundary validation ---
            # Check the annotation's own subgraph_nodes against the model.
            # For compositions, child annotation internals are NOT re-checked here
            # (they were validated when created). We use internal_nodes only to
            # determine what counts as "inside" the annotation boundary.

            # Precondition 1: Entry-only ingress
            # All edges entering the subgraph from outside must target entry nodes
            for nid in subgraph_nodes:
                if nid in entry_nodes:
                    continue
                inputs = model.get_connections_to(nid)
                for conn in inputs:
                    if conn.enabled and conn.from_node not in internal_nodes:
                        errors.append(
                            f"P1 violation: non-entry node {nid} has external input from {conn.from_node}"
                        )

            # Precondition 2: Exit-only egress
            # All edges leaving the subgraph must originate from exit nodes
            for nid in subgraph_nodes:
                if nid in exit_nodes:
                    continue
                outputs = model.get_connections_from(nid)
                for conn in outputs:
                    if conn.enabled and conn.to_node not in internal_nodes:
                        errors.append(
                            f"P2 violation: non-exit node {nid} has external output to {conn.to_node}"
                        )

            # Precondition 3: Pure exits
            # Exit nodes must only receive inputs from within the annotation
            for exit_id in exit_nodes:
                inputs = model.get_connections_to(exit_id)
                for conn in inputs:
                    if conn.enabled and conn.from_node not in internal_nodes:
                        errors.append(
                            f"P3 violation: exit node {exit_id} has external input from {conn.from_node}"
                        )

    elif op_type == "disable_connection":
        from_node = params.get("from_node")
        to_node = params.get("to_node")
        if not from_node:
            errors.append("from_node is required")
        if not to_node:
            errors.append("to_node is required")
        if from_node and to_node:
            if (from_node, to_node) in covered_connections:
                errors.append(f"Connection [{from_node}, {to_node}] is covered by an annotation")
            else:
                conn = next(
                    (c for c in model.connections if c.from_node == from_node and c.to_node == to_node),
                    None,
                )
                if conn is None:
                    errors.append(f"Connection [{from_node}, {to_node}] not found")
                elif not conn.enabled:
                    errors.append(f"Connection [{from_node}, {to_node}] is already disabled")

    elif op_type == "enable_connection":
        from_node = params.get("from_node")
        to_node = params.get("to_node")
        if not from_node:
            errors.append("from_node is required")
        if not to_node:
            errors.append("to_node is required")
        if from_node and to_node:
            if (from_node, to_node) in covered_connections:
                errors.append(f"Connection [{from_node}, {to_node}] is covered by an annotation")
            else:
                conn = next(
                    (c for c in model.connections if c.from_node == from_node and c.to_node == to_node),
                    None,
                )
                if conn is None:
                    errors.append(f"Connection [{from_node}, {to_node}] not found")
                elif conn.enabled:
                    errors.append(f"Connection [{from_node}, {to_node}] is already enabled")

    elif op_type == "rename_node":
        node_id = params.get("node_id")
        display_name = params.get("display_name")
        if not node_id:
            errors.append("node_id is required")
        elif node_id not in node_ids:
            errors.append(f"Node '{node_id}' not found")
        if display_name is not None:
            if not display_name:
                errors.append("display_name cannot be empty")
            if isinstance(display_name, str) and " " in display_name:
                errors.append("display_name cannot contain spaces (use camelCase)")

    elif op_type == "rename_annotation":
        annotation_id = params.get("annotation_id")
        display_name = params.get("display_name")
        if not annotation_id:
            errors.append("annotation_id is required")
        else:
            known_ids = {a.name for a in annotations} if annotations else set()
            if annotation_id not in known_ids:
                errors.append(f"Annotation '{annotation_id}' not found")
        if display_name is not None:
            if not display_name:
                errors.append("display_name cannot be empty")
            if isinstance(display_name, str) and " " in display_name:
                errors.append("display_name cannot contain spaces (use camelCase)")

    elif op_type == "prune_node":
        node_id = params.get("node_id")
        if not node_id:
            errors.append("node_id is required")
        elif node_id not in node_ids:
            errors.append(f"Node {node_id} does not exist")
        elif node_id in covered_nodes:
            errors.append(f"Node {node_id} is covered by an annotation and cannot be pruned")
        else:
            node = model.get_node_by_id(node_id)
            if node and node.type in (NodeType.INPUT, NodeType.OUTPUT):
                errors.append(f"Cannot prune {node.type.value} node {node_id}")

            # Check connectivity: must have exactly 1 enabled in, 1 enabled out
            inputs = [c for c in model.get_connections_to(node_id) if c.enabled]
            outputs = [c for c in model.get_connections_from(node_id) if c.enabled]
            if len(inputs) != 1 or len(outputs) != 1:
                errors.append(
                    f"Prune requires exactly 1 input and 1 output "
                    f"(node {node_id} has {len(inputs)} inputs, {len(outputs)} outputs)"
                )

            # Warn if node is entry/exit of any annotation
            if existing_annotations:
                for ann in existing_annotations:
                    if node_id in ann.entry_nodes:
                        errors.append(
                            f"Node {node_id} is an entry node of annotation '{ann.name}'"
                        )
                    if node_id in ann.exit_nodes:
                        errors.append(
                            f"Node {node_id} is an exit node of annotation '{ann.name}'"
                        )

    elif op_type == "prune_connection":
        from_node = params.get("from_node")
        to_node = params.get("to_node")
        if not from_node:
            errors.append("from_node is required")
        if not to_node:
            errors.append("to_node is required")
        if from_node and to_node:
            if (from_node, to_node) in covered_connections:
                errors.append(
                    f"Connection [{from_node}, {to_node}] is covered by an annotation and cannot be pruned"
                )
            else:
                conn = next(
                    (c for c in model.connections
                     if c.from_node == from_node and c.to_node == to_node),
                    None,
                )
                if conn is None:
                    errors.append(f"Connection [{from_node}, {to_node}] not found")

    elif op_type == "retrain":
        # Retrain operation — validated at a higher level (training pipeline)
        pass

    else:
        errors.append(f"Unknown operation type: {op_type}")

    return errors


# =============================================================================
# Operation Handlers
# =============================================================================


def _get_base_node_id(node_id: str) -> Optional[str]:
    """
    Extract base node ID from a potentially split node ID.

    "5" -> "5"
    "5_a" -> "5"
    "5_abc" -> "5"

    Returns None if not a valid split node format.
    """
    if "_" not in node_id:
        return node_id  # Already a base node

    parts = node_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isalpha():
        return parts[0]
    return None


def _get_split_suffix(node_id: str) -> Optional[str]:
    """
    Extract suffix from a split node ID.

    "5_a" -> "a"
    "5_abc" -> "abc"
    "5" -> None
    """
    if "_" not in node_id:
        return None

    parts = node_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isalpha():
        return parts[1]
    return None


def _replace_node_ids_in_list(
    id_list: List[str], old_ids: Set[str], new_id: str
) -> None:
    """
    Replace occurrences of old_ids in id_list with new_id (in place).

    The first occurrence of any old ID is replaced with new_id;
    all other occurrences are removed.
    """
    replaced = False
    i = 0
    while i < len(id_list):
        if id_list[i] in old_ids:
            if not replaced:
                id_list[i] = new_id
                replaced = True
                i += 1
            else:
                id_list.pop(i)
        else:
            i += 1


def apply_split_node(
    model: NetworkStructure,
    node_id: str,
    covered_nodes: Set[str],
) -> Dict[str, Any]:
    """
    Split a node into multiple nodes, one per output connection.

    Each split node gets:
    - All incoming connections of the original (copied)
    - Exactly one outgoing connection

    Split nodes are named {original_id}_{letter} where letters are
    assigned alphabetically based on ascending target node ID.

    Args:
        model: Network structure (modified in place)
        node_id: ID of node to split
        covered_nodes: Nodes that cannot be modified

    Returns:
        Result dict with created_nodes, removed_nodes
    """
    # Get the original node
    original_node = model.get_node_by_id(node_id)
    if not original_node:
        raise OperationError(f"Node {node_id} not found")

    if node_id in covered_nodes:
        raise OperationError(f"Node {node_id} is covered by annotation")

    # Get outgoing connections, sorted by target node ID
    outgoing = [c for c in model.get_connections_from(node_id) if c.enabled]
    if len(outgoing) < 2:
        raise OperationError(f"Node {node_id} has fewer than 2 outputs")

    # Sort by target node ID for deterministic naming
    outgoing.sort(key=lambda c: (c.to_node.lstrip("-").zfill(10), c.to_node))

    # Get incoming connections
    incoming = model.get_connections_to(node_id)

    # Check if this is a re-split of a consolidated node
    suffix = _get_split_suffix(node_id)
    if suffix and len(suffix) > 1:
        # Re-splitting a consolidated node like "5_ac" -> restore "5_a", "5_c"
        letters = list(suffix)
        if len(letters) != len(outgoing):
            raise OperationError(
                f"Cannot re-split {node_id}: expected {len(letters)} outputs, got {len(outgoing)}"
            )
        base_id = _get_base_node_id(node_id)
        split_ids = [f"{base_id}_{letter}" for letter in letters]
    else:
        # Normal split: assign new letters
        base_id = _get_base_node_id(node_id) or node_id
        existing_suffixes = set()

        # Find existing split nodes from same base to avoid collisions
        for n in model.nodes:
            if _get_base_node_id(n.id) == base_id:
                s = _get_split_suffix(n.id)
                if s:
                    existing_suffixes.update(s)

        # Assign new letters
        available_letters = [chr(ord('a') + i) for i in range(26)]
        available_letters = [l for l in available_letters if l not in existing_suffixes]

        if len(available_letters) < len(outgoing):
            raise OperationError("Too many splits, ran out of letters")

        split_ids = [f"{base_id}_{available_letters[i]}" for i in range(len(outgoing))]

    # Create split nodes
    created_nodes = []
    for i, (split_id, out_conn) in enumerate(zip(split_ids, outgoing)):
        # Derive display_name suffix from split_id
        suffix = split_id.rsplit("_", 1)[-1] if "_" in split_id else None
        split_display_name = (
            f"{original_node.display_name}_{suffix}"
            if original_node.display_name and suffix
            else None
        )
        # Create new node with same properties
        new_node = NetworkNode(
            id=split_id,
            type=original_node.type,
            bias=original_node.bias,
            activation=original_node.activation,
            response=original_node.response,
            aggregation=original_node.aggregation,
            display_name=split_display_name,
        )
        model.nodes.append(new_node)
        created_nodes.append(split_id)

        # Copy all incoming connections to the new node
        for in_conn in incoming:
            new_in_conn = NetworkConnection(
                from_node=in_conn.from_node,
                to_node=split_id,
                weight=in_conn.weight,
                enabled=in_conn.enabled,
                innovation=in_conn.innovation,
            )
            model.connections.append(new_in_conn)

        # Create the single outgoing connection
        new_out_conn = NetworkConnection(
            from_node=split_id,
            to_node=out_conn.to_node,
            weight=out_conn.weight,
            enabled=out_conn.enabled,
            innovation=out_conn.innovation,
        )
        model.connections.append(new_out_conn)

    # Remove original node and its connections
    model.nodes = [n for n in model.nodes if n.id != node_id]
    model.connections = [
        c for c in model.connections
        if c.from_node != node_id and c.to_node != node_id
    ]

    # Update input_node_ids if we split an input node
    if node_id in model.input_node_ids:
        idx = model.input_node_ids.index(node_id)
        model.input_node_ids[idx:idx+1] = created_nodes

    # Update output_node_ids if we split an output node
    if node_id in model.output_node_ids:
        idx = model.output_node_ids.index(node_id)
        model.output_node_ids[idx:idx+1] = created_nodes

    # Verify each split node has exactly 1 output (invariant check)
    for split_id in created_nodes:
        outputs = [c for c in model.connections if c.from_node == split_id and c.enabled]
        if len(outputs) != 1:
            import logging
            logging.warning(
                f"Split node {split_id} has {len(outputs)} outputs (expected 1). "
                f"Outputs: {[(c.from_node, c.to_node) for c in outputs]}"
            )

    return {
        "created_nodes": created_nodes,
        "removed_nodes": [node_id],
    }


def apply_consolidate_node(
    model: NetworkStructure,
    node_ids: List[str],
    covered_nodes: Set[str],
) -> Dict[str, Any]:
    """
    Consolidate previously split nodes back together.

    The consolidated node combines outputs from all specified split nodes.
    Its name is the base ID plus sorted suffix letters (e.g., "5_a" + "5_c" -> "5_ac").

    Args:
        model: Network structure (modified in place)
        node_ids: List of split node IDs to consolidate
        covered_nodes: Nodes that cannot be modified

    Returns:
        Result dict with created_nodes, removed_nodes
    """
    if len(node_ids) < 2:
        raise OperationError("Need at least 2 nodes to consolidate")

    # Verify all nodes exist and are from the same base
    base_ids = set()
    suffixes = []

    for nid in node_ids:
        if nid in covered_nodes:
            raise OperationError(f"Node {nid} is covered by annotation")

        node = model.get_node_by_id(nid)
        if not node:
            raise OperationError(f"Node {nid} not found")

        base = _get_base_node_id(nid)
        suffix = _get_split_suffix(nid)

        if not base or not suffix:
            raise OperationError(f"Node {nid} is not a split node")

        base_ids.add(base)
        suffixes.append(suffix)

    if len(base_ids) != 1:
        raise OperationError("Cannot consolidate nodes from different bases")

    base_id = base_ids.pop()

    # Create consolidated node ID with sorted suffixes
    combined_suffix = "".join(sorted("".join(suffixes)))
    consolidated_id = f"{base_id}_{combined_suffix}"

    # Get reference node for properties
    ref_node = model.get_node_by_id(node_ids[0])

    # Create consolidated node
    new_node = NetworkNode(
        id=consolidated_id,
        type=ref_node.type,
        bias=ref_node.bias,
        activation=ref_node.activation,
        response=ref_node.response,
        aggregation=ref_node.aggregation,
    )
    model.nodes.append(new_node)

    # Collect all outgoing connections (deduplicated)
    outgoing_targets = {}
    for nid in node_ids:
        for conn in model.get_connections_from(nid):
            if conn.enabled:
                # Keep first weight encountered
                if conn.to_node not in outgoing_targets:
                    outgoing_targets[conn.to_node] = conn

    # Create outgoing connections for consolidated node
    for target, ref_conn in outgoing_targets.items():
        new_conn = NetworkConnection(
            from_node=consolidated_id,
            to_node=target,
            weight=ref_conn.weight,
            enabled=True,
            innovation=ref_conn.innovation,
        )
        model.connections.append(new_conn)

    # Collect incoming connections (deduplicated by source)
    incoming_sources = {}
    for nid in node_ids:
        for conn in model.get_connections_to(nid):
            if conn.enabled:
                if conn.from_node not in incoming_sources:
                    incoming_sources[conn.from_node] = conn

    # Create incoming connections for consolidated node
    for source, ref_conn in incoming_sources.items():
        new_conn = NetworkConnection(
            from_node=source,
            to_node=consolidated_id,
            weight=ref_conn.weight,
            enabled=True,
            innovation=ref_conn.innovation,
        )
        model.connections.append(new_conn)

    # Remove original split nodes and their connections
    node_ids_set = set(node_ids)
    model.nodes = [n for n in model.nodes if n.id not in node_ids_set]
    model.connections = [
        c for c in model.connections
        if c.from_node not in node_ids_set and c.to_node not in node_ids_set
    ]

    # Update input_node_ids: replace first occurrence of any consolidated node
    # with the consolidated ID, remove the rest
    _replace_node_ids_in_list(model.input_node_ids, node_ids_set, consolidated_id)
    _replace_node_ids_in_list(model.output_node_ids, node_ids_set, consolidated_id)

    return {
        "created_nodes": [consolidated_id],
        "removed_nodes": list(node_ids),
    }


def apply_remove_node(
    model: NetworkStructure,
    node_id: str,
    covered_nodes: Set[str],
) -> Dict[str, Any]:
    """
    Remove a pass-through node, combining its input and output connections.

    The node must have exactly one input and one output connection.
    The new connection weight is the product of the two original weights.

    Args:
        model: Network structure (modified in place)
        node_id: ID of node to remove
        covered_nodes: Nodes that cannot be modified

    Returns:
        Result dict with removed_nodes, created_connections, removed_connections
    """
    if node_id in covered_nodes:
        raise OperationError(f"Node {node_id} is covered by annotation")

    node = model.get_node_by_id(node_id)
    if not node:
        raise OperationError(f"Node {node_id} not found")

    if node.type in (NodeType.INPUT, NodeType.OUTPUT):
        raise OperationError(f"Cannot remove {node.type.value} node")

    # Get connections
    inputs = [c for c in model.get_connections_to(node_id) if c.enabled]
    outputs = [c for c in model.get_connections_from(node_id) if c.enabled]

    if len(inputs) != 1 or len(outputs) != 1:
        raise OperationError(
            f"Node must have exactly 1 input and 1 output (has {len(inputs)} inputs, {len(outputs)} outputs)"
        )

    in_conn = inputs[0]
    out_conn = outputs[0]

    # Create combined connection
    new_weight = in_conn.weight * out_conn.weight
    new_conn = NetworkConnection(
        from_node=in_conn.from_node,
        to_node=out_conn.to_node,
        weight=new_weight,
        enabled=True,
        innovation=None,  # New connection, no innovation number
    )
    model.connections.append(new_conn)

    # Remove node and its connections
    model.nodes = [n for n in model.nodes if n.id != node_id]
    model.connections = [
        c for c in model.connections
        if c.from_node != node_id and c.to_node != node_id
    ]

    return {
        "removed_nodes": [node_id],
        "created_connections": [(in_conn.from_node, out_conn.to_node)],
        "removed_connections": [
            (in_conn.from_node, node_id),
            (node_id, out_conn.to_node),
        ],
    }


def apply_add_node(
    model: NetworkStructure,
    connection: Tuple[str, str],
    new_node_id: str,
    covered_connections: Set[Tuple[str, str]],
    bias: float = 0.0,
    activation: str = "identity",
) -> Dict[str, Any]:
    """
    Insert a node into an existing connection.

    The connection is split with the new node in between.
    Weight assignment: [from -> new] = 1.0, [new -> to] = original_weight

    Args:
        model: Network structure (modified in place)
        connection: (from_node, to_node) tuple
        new_node_id: ID for the new node
        covered_connections: Connections that cannot be modified
        bias: Bias for new node (default 0.0)
        activation: Activation function (default "identity")

    Returns:
        Result dict with created_nodes, created_connections, removed_connections
    """
    from_node, to_node = connection

    if connection in covered_connections:
        raise OperationError(f"Connection {connection} is covered by annotation")

    # Find the connection
    conn = None
    for c in model.connections:
        if c.from_node == from_node and c.to_node == to_node and c.enabled:
            conn = c
            break

    if not conn:
        raise OperationError(f"Connection {connection} not found")

    if model.get_node_by_id(new_node_id):
        raise OperationError(f"Node {new_node_id} already exists")

    # Create new node
    new_node = NetworkNode(
        id=new_node_id,
        type=NodeType.HIDDEN,
        bias=bias,
        activation=activation,
        response=1.0,
        aggregation="sum",
    )
    model.nodes.append(new_node)

    # Create new connections
    conn_in = NetworkConnection(
        from_node=from_node,
        to_node=new_node_id,
        weight=1.0,
        enabled=True,
        innovation=None,
    )
    conn_out = NetworkConnection(
        from_node=new_node_id,
        to_node=to_node,
        weight=conn.weight,
        enabled=True,
        innovation=None,
    )
    model.connections.append(conn_in)
    model.connections.append(conn_out)

    # Remove original connection
    model.connections = [
        c for c in model.connections
        if not (c.from_node == from_node and c.to_node == to_node and c is not conn_in and c is not conn_out)
    ]

    return {
        "created_nodes": [new_node_id],
        "created_connections": [(from_node, new_node_id), (new_node_id, to_node)],
        "removed_connections": [connection],
    }


def apply_add_identity_node(
    model: NetworkStructure,
    target_node: str,
    connections: List[Tuple[str, str]],
    new_node_id: str,
    covered_connections: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Intercept a subset of connections to a target node through an identity node.

    The specified connections are redirected through the new identity node.
    Original weights are preserved on the redirected connections.
    The new connection from identity to target has weight 1.0.

    Args:
        model: Network structure (modified in place)
        target_node: Node that connections currently go to
        connections: List of (from, to) connections to redirect
        new_node_id: ID for the new identity node
        covered_connections: Connections that cannot be modified

    Returns:
        Result dict with created_nodes, created_connections, removed_connections
    """
    target = model.get_node_by_id(target_node)
    if not target:
        raise OperationError(f"Target node {target_node} not found")

    if model.get_node_by_id(new_node_id):
        raise OperationError(f"Node {new_node_id} already exists")

    # Validate connections
    for conn in connections:
        if conn in covered_connections:
            raise OperationError(f"Connection {conn} is covered by annotation")

    # Find matching connections
    conns_to_redirect = []
    for from_node, to_node in connections:
        if to_node != target_node:
            raise OperationError(f"Connection {(from_node, to_node)} does not end at {target_node}")

        found = None
        for c in model.connections:
            if c.from_node == from_node and c.to_node == to_node and c.enabled:
                found = c
                break

        if not found:
            raise OperationError(f"Connection {(from_node, to_node)} not found")

        conns_to_redirect.append(found)

    # Check not all connections
    all_inputs = [c for c in model.get_connections_to(target_node) if c.enabled]
    if len(conns_to_redirect) >= len(all_inputs):
        raise OperationError("Cannot redirect ALL connections to target")

    # Create identity node
    new_node = NetworkNode(
        id=new_node_id,
        type=NodeType.HIDDEN,  # Could be a new "identity" type
        bias=0.0,
        activation="identity",
        response=1.0,
        aggregation=target.aggregation or "sum",
    )
    model.nodes.append(new_node)

    # Redirect connections
    created_connections = []
    removed_connections = []

    for conn in conns_to_redirect:
        # Create new connection to identity node (preserve weight)
        new_conn = NetworkConnection(
            from_node=conn.from_node,
            to_node=new_node_id,
            weight=conn.weight,
            enabled=True,
            innovation=conn.innovation,
        )
        model.connections.append(new_conn)
        created_connections.append((conn.from_node, new_node_id))
        removed_connections.append((conn.from_node, conn.to_node))

    # Remove original redirected connections
    redirect_set = {(c.from_node, c.to_node) for c in conns_to_redirect}
    model.connections = [
        c for c in model.connections
        if (c.from_node, c.to_node) not in redirect_set or c.to_node == new_node_id
    ]

    # Create connection from identity to target
    identity_to_target = NetworkConnection(
        from_node=new_node_id,
        to_node=target_node,
        weight=1.0,
        enabled=True,
        innovation=None,
    )
    model.connections.append(identity_to_target)
    created_connections.append((new_node_id, target_node))

    return {
        "created_nodes": [new_node_id],
        "created_connections": created_connections,
        "removed_connections": removed_connections,
    }


def apply_disable_connection(
    model: NetworkStructure,
    from_node: str,
    to_node: str,
    covered_connections: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """Disable a connection (set enabled=False).

    Args:
        model: Network structure (modified in place)
        from_node: Source node ID
        to_node: Target node ID
        covered_connections: Connections protected by annotations (cannot be modified)

    Returns:
        Result dict with from_node, to_node, previous_weight

    Raises:
        OperationError: If connection not found, already disabled, or covered
    """
    if (from_node, to_node) in covered_connections:
        raise OperationError(
            f"Connection [{from_node}, {to_node}] is covered by an annotation and cannot be modified"
        )

    conn = next(
        (c for c in model.connections if c.from_node == from_node and c.to_node == to_node),
        None,
    )
    if conn is None:
        raise OperationError(f"Connection [{from_node}, {to_node}] not found")

    if not conn.enabled:
        raise OperationError(f"Connection [{from_node}, {to_node}] is already disabled")

    previous_weight = conn.weight
    conn.enabled = False

    return {
        "from_node": from_node,
        "to_node": to_node,
        "previous_weight": previous_weight,
    }


def apply_enable_connection(
    model: NetworkStructure,
    from_node: str,
    to_node: str,
    covered_connections: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """Re-enable a previously disabled connection.

    Args:
        model: Network structure (modified in place)
        from_node: Source node ID
        to_node: Target node ID
        covered_connections: Connections protected by annotations (cannot be modified)

    Returns:
        Result dict with from_node, to_node

    Raises:
        OperationError: If connection not found, already enabled, or covered
    """
    if (from_node, to_node) in covered_connections:
        raise OperationError(
            f"Connection [{from_node}, {to_node}] is covered by an annotation and cannot be modified"
        )

    conn = next(
        (c for c in model.connections if c.from_node == from_node and c.to_node == to_node),
        None,
    )
    if conn is None:
        raise OperationError(f"Connection [{from_node}, {to_node}] not found")

    if conn.enabled:
        raise OperationError(f"Connection [{from_node}, {to_node}] is already enabled")

    conn.enabled = True

    return {
        "from_node": from_node,
        "to_node": to_node,
    }


def apply_rename_node(
    model: NetworkStructure,
    node_id: str,
    display_name: Union[str, None],
    covered_nodes: Set[str],
) -> Dict[str, Any]:
    """Set or clear the display_name on a node.

    Args:
        model: The network structure (mutated in place).
        node_id: ID of the node to rename.
        display_name: New display name, or None to clear.
        covered_nodes: Nodes covered by annotations (cannot be renamed).

    Returns:
        Dict with node_id and display_name.
    """
    node = model.get_node_by_id(node_id)
    if node is None:
        raise OperationError(f"Node '{node_id}' not found in structure")

    if display_name is not None:
        if not display_name:
            raise OperationError("display_name cannot be empty")
        if " " in display_name:
            raise OperationError("display_name cannot contain spaces (use camelCase)")

    node.display_name = display_name
    return {"node_id": node_id, "display_name": display_name}


def apply_prune_node(
    model: NetworkStructure,
    node_id: str,
    covered_nodes: Set[str],
) -> Dict[str, Any]:
    """
    Bypass a node by connecting its single input directly to its single output.

    The node must have exactly one enabled input and one enabled output
    connection. The bypass connection weight is the product of the two
    original weights. This is a non-identity operation that changes the
    network's computed function (the node's activation/bias are removed).

    Args:
        model: Network structure (modified in place)
        node_id: ID of node to prune
        covered_nodes: Nodes that cannot be modified

    Returns:
        Result dict with removed_nodes, created_connections, removed_connections
    """
    if node_id in covered_nodes:
        raise OperationError(f"Node {node_id} is covered by annotation")

    node = model.get_node_by_id(node_id)
    if not node:
        raise OperationError(f"Node {node_id} not found")

    if node.type in (NodeType.INPUT, NodeType.OUTPUT):
        raise OperationError(f"Cannot prune {node.type.value} node {node_id}")

    # Get enabled connections
    inputs = [c for c in model.get_connections_to(node_id) if c.enabled]
    outputs = [c for c in model.get_connections_from(node_id) if c.enabled]

    if len(inputs) != 1 or len(outputs) != 1:
        raise OperationError(
            f"Prune requires exactly 1 input and 1 output "
            f"(node {node_id} has {len(inputs)} inputs, {len(outputs)} outputs)"
        )

    in_conn = inputs[0]
    out_conn = outputs[0]

    # Create bypass connection: weight = product of the two
    new_weight = in_conn.weight * out_conn.weight
    new_conn = NetworkConnection(
        from_node=in_conn.from_node,
        to_node=out_conn.to_node,
        weight=new_weight,
        enabled=True,
        innovation=None,
    )
    model.connections.append(new_conn)

    # Remove node and its connections
    model.nodes = [n for n in model.nodes if n.id != node_id]
    model.connections = [
        c for c in model.connections
        if c.from_node != node_id and c.to_node != node_id
    ]

    return {
        "removed_nodes": [node_id],
        "created_connections": [(in_conn.from_node, out_conn.to_node)],
        "removed_connections": [
            (in_conn.from_node, node_id),
            (node_id, out_conn.to_node),
        ],
    }


def apply_prune_connection(
    model: NetworkStructure,
    from_node: str,
    to_node: str,
    covered_connections: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Permanently remove a connection from the network.

    Unlike disable_connection (which toggles the enabled flag and can be
    reversed with enable_connection), prune_connection permanently removes
    the connection from the structure. This is a non-identity operation.

    Args:
        model: Network structure (modified in place)
        from_node: Source node ID
        to_node: Target node ID
        covered_connections: Connections protected by annotations

    Returns:
        Result dict with removed_connections

    Raises:
        OperationError: If connection not found or covered
    """
    if (from_node, to_node) in covered_connections:
        raise OperationError(
            f"Connection [{from_node}, {to_node}] is covered by an annotation and cannot be pruned"
        )

    conn = next(
        (c for c in model.connections if c.from_node == from_node and c.to_node == to_node),
        None,
    )
    if conn is None:
        raise OperationError(f"Connection [{from_node}, {to_node}] not found")

    # Remove the connection
    model.connections = [
        c for c in model.connections
        if not (c.from_node == from_node and c.to_node == to_node)
    ]

    return {
        "removed_connections": [(from_node, to_node)],
    }


def apply_retrain(
    model: NetworkStructure,
    weight_updates: Dict[Tuple[str, str], float],
    bias_updates: Dict[str, float],
) -> Dict[str, Any]:
    """
    Apply weight and bias updates from retraining to the network.

    Args:
        model: Network structure (modified in place)
        weight_updates: Map of (from_node, to_node) -> new_weight
        bias_updates: Map of node_id -> new_bias

    Returns:
        Result dict with counts of updated weights and biases
    """
    weights_updated = 0
    for (from_node, to_node), new_weight in weight_updates.items():
        for conn in model.connections:
            if conn.from_node == from_node and conn.to_node == to_node:
                conn.weight = new_weight
                weights_updated += 1
                break

    biases_updated = 0
    for node_id, new_bias in bias_updates.items():
        node = model.get_node_by_id(node_id)
        if node is not None:
            node.bias = new_bias
            biases_updated += 1

    return {
        "weights_updated": weights_updated,
        "biases_updated": biases_updated,
    }
