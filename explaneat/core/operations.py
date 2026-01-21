"""
Operation handlers for model manipulations.

Each operation modifies the NetworkStructure in place and returns
a result dictionary with information about what was changed.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from copy import deepcopy

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
# Validation
# =============================================================================


def validate_operation(
    model: NetworkStructure,
    op_type: str,
    params: Dict[str, Any],
    covered_nodes: Set[str],
    covered_connections: Set[Tuple[str, str]],
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

            # Check not an input node
            node = model.get_node_by_id(node_id)
            if node and node.type == NodeType.INPUT:
                errors.append(f"Cannot split input node {node_id}")

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

            # Check entry/exit are subsets
            if not entry_nodes.issubset(subgraph_nodes):
                errors.append("entry_nodes must be subset of subgraph_nodes")
            if not exit_nodes.issubset(subgraph_nodes):
                errors.append("exit_nodes must be subset of subgraph_nodes")

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

            # Validate entry nodes have no external outputs
            for entry_id in entry_nodes:
                outputs = model.get_connections_from(entry_id)
                for conn in outputs:
                    if conn.enabled and conn.to_node not in subgraph_nodes:
                        errors.append(
                            f"Entry node {entry_id} has external output to {conn.to_node} (side effect)"
                        )

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
        # Create new node with same properties
        new_node = NetworkNode(
            id=split_id,
            type=original_node.type,
            bias=original_node.bias,
            activation=original_node.activation,
            response=original_node.response,
            aggregation=original_node.aggregation,
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
    model.nodes = [n for n in model.nodes if n.id not in node_ids]
    model.connections = [
        c for c in model.connections
        if c.from_node not in node_ids and c.to_node not in node_ids
    ]

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
