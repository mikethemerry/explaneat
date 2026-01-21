"""
Node classification for annotation coverage.

Classifies nodes within a proposed coverage as entry, intermediate, or exit
based on their external connections.
"""

from typing import List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..core.genome_network import NetworkStructure, NodeType


class NodeRole(str, Enum):
    """Classification of a node's role within an annotation."""

    ENTRY = "entry"
    INTERMEDIATE = "intermediate"
    EXIT = "exit"


@dataclass
class NodeClassificationDetail:
    """Detailed classification information for a single node."""

    node_id: str
    role: NodeRole
    external_inputs: List[Tuple[str, str]]  # connections from outside
    external_outputs: List[Tuple[str, str]]  # connections to outside
    internal_inputs: List[Tuple[str, str]]  # connections from inside
    internal_outputs: List[Tuple[str, str]]  # connections to inside
    is_input_node: bool = False  # True if this is a network input node
    is_output_node: bool = False  # True if this is a network output node


@dataclass
class ClassificationResult:
    """Result of node classification analysis."""

    coverage: List[str]
    entry_nodes: List[str]
    intermediate_nodes: List[str]
    exit_nodes: List[str]
    details: List[NodeClassificationDetail]
    valid: bool
    violations: List[dict] = field(default_factory=list)


def classify_node(
    model: NetworkStructure,
    node_id: str,
    coverage: Set[str],
) -> NodeClassificationDetail:
    """
    Classify a single node based on its connections relative to coverage.

    Classification rules:
    - Entry: Has external inputs, no external outputs
    - Intermediate: No external inputs or outputs
    - Exit: Has external outputs (may have internal inputs)

    Note: A node with BOTH external input AND external output is a violation
    and will be classified as "entry" but flagged.

    Args:
        model: Network structure
        node_id: ID of node to classify
        coverage: Set of node IDs in the coverage

    Returns:
        NodeClassificationDetail with role and connection info
    """
    node = model.get_node_by_id(node_id)
    is_input = node and node.type == NodeType.INPUT
    is_output = node and node.type == NodeType.OUTPUT

    # Get all connections
    inputs = model.get_connections_to(node_id)
    outputs = model.get_connections_from(node_id)

    # Classify connections
    external_inputs = [
        (c.from_node, c.to_node) for c in inputs
        if c.enabled and c.from_node not in coverage
    ]
    internal_inputs = [
        (c.from_node, c.to_node) for c in inputs
        if c.enabled and c.from_node in coverage
    ]
    external_outputs = [
        (c.from_node, c.to_node) for c in outputs
        if c.enabled and c.to_node not in coverage
    ]
    internal_outputs = [
        (c.from_node, c.to_node) for c in outputs
        if c.enabled and c.to_node in coverage
    ]

    # Input nodes are special - they have "external input" from the environment
    has_external_input = bool(external_inputs) or is_input
    has_external_output = bool(external_outputs)

    # Determine role
    if has_external_input and not has_external_output:
        role = NodeRole.ENTRY
    elif not has_external_input and not has_external_output:
        role = NodeRole.INTERMEDIATE
    elif has_external_output and not has_external_input:
        role = NodeRole.EXIT
    else:
        # Has both external input and output - violation!
        # Classify as entry but this should be flagged
        role = NodeRole.ENTRY

    return NodeClassificationDetail(
        node_id=node_id,
        role=role,
        external_inputs=external_inputs,
        external_outputs=external_outputs,
        internal_inputs=internal_inputs,
        internal_outputs=internal_outputs,
        is_input_node=is_input,
        is_output_node=is_output,
    )


def classify_coverage(
    model: NetworkStructure,
    coverage: List[str],
) -> ClassificationResult:
    """
    Classify all nodes in a proposed coverage.

    Validates that the classification is valid:
    - Entry nodes must have no external outputs (no side effects)
    - Intermediate nodes must have no external I/O

    Args:
        model: Network structure
        coverage: List of node IDs in the proposed coverage

    Returns:
        ClassificationResult with classifications and validation
    """
    coverage_set = set(coverage)
    details = []
    entry_nodes = []
    intermediate_nodes = []
    exit_nodes = []
    violations = []

    for node_id in coverage:
        detail = classify_node(model, node_id, coverage_set)
        details.append(detail)

        # Group by role
        if detail.role == NodeRole.ENTRY:
            entry_nodes.append(node_id)
        elif detail.role == NodeRole.INTERMEDIATE:
            intermediate_nodes.append(node_id)
        else:
            exit_nodes.append(node_id)

        # Check for violations
        if detail.role == NodeRole.ENTRY and detail.external_outputs:
            violations.append({
                "node_id": node_id,
                "reason": "has_external_input_and_output",
                "external_inputs": detail.external_inputs,
                "external_outputs": detail.external_outputs,
            })

    return ClassificationResult(
        coverage=coverage,
        entry_nodes=entry_nodes,
        intermediate_nodes=intermediate_nodes,
        exit_nodes=exit_nodes,
        details=details,
        valid=len(violations) == 0,
        violations=violations,
    )


def validate_annotation_params(
    model: NetworkStructure,
    entry_nodes: List[str],
    exit_nodes: List[str],
    subgraph_nodes: List[str],
    subgraph_connections: List[Tuple[str, str]],
) -> List[str]:
    """
    Validate annotation parameters against the model.

    Checks:
    1. All nodes exist in model
    2. Entry/exit are subsets of subgraph_nodes
    3. All connections exist
    4. Entry nodes have no external outputs
    5. Intermediate nodes have no external I/O
    6. Classification matches declared entry/exit

    Args:
        model: Network structure
        entry_nodes: Declared entry nodes
        exit_nodes: Declared exit nodes
        subgraph_nodes: All nodes in subgraph
        subgraph_connections: All connections in subgraph

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    node_ids = model.get_node_ids()
    coverage_set = set(subgraph_nodes)
    entry_set = set(entry_nodes)
    exit_set = set(exit_nodes)

    # Check all nodes exist
    for nid in subgraph_nodes:
        if nid not in node_ids:
            errors.append(f"Node {nid} does not exist in model")

    # Check entry/exit are subsets
    if not entry_set.issubset(coverage_set):
        errors.append("entry_nodes must be a subset of subgraph_nodes")
    if not exit_set.issubset(coverage_set):
        errors.append("exit_nodes must be a subset of subgraph_nodes")

    # Check connections exist
    for from_node, to_node in subgraph_connections:
        conn_exists = any(
            c.from_node == from_node and c.to_node == to_node
            for c in model.connections
        )
        if not conn_exists:
            errors.append(f"Connection [{from_node}, {to_node}] does not exist")

    # Classify and validate
    classification = classify_coverage(model, subgraph_nodes)

    # Check declared entry nodes match classification
    computed_entries = set(classification.entry_nodes)
    if entry_set != computed_entries:
        missing = computed_entries - entry_set
        extra = entry_set - computed_entries
        if missing:
            errors.append(f"Missing entry nodes: {list(missing)}")
        if extra:
            errors.append(f"Declared entry nodes that are not entries: {list(extra)}")

    # Check declared exit nodes match classification
    computed_exits = set(classification.exit_nodes)
    if exit_set != computed_exits:
        missing = computed_exits - exit_set
        extra = exit_set - computed_exits
        if missing:
            errors.append(f"Missing exit nodes: {list(missing)}")
        if extra:
            errors.append(f"Declared exit nodes that are not exits: {list(extra)}")

    # Add any classification violations
    for violation in classification.violations:
        errors.append(
            f"Node {violation['node_id']} has both external input and output (side effect)"
        )

    return errors


def get_subgraph_connections(
    model: NetworkStructure,
    coverage: Set[str],
) -> List[Tuple[str, str]]:
    """
    Get all connections that are internal to a coverage.

    A connection is internal if BOTH endpoints are in the coverage.

    Args:
        model: Network structure
        coverage: Set of node IDs

    Returns:
        List of (from_node, to_node) tuples for internal connections
    """
    internal_connections = []
    for conn in model.connections:
        if conn.enabled and conn.from_node in coverage and conn.to_node in coverage:
            internal_connections.append((conn.from_node, conn.to_node))
    return internal_connections


def auto_detect_entry_exit(
    model: NetworkStructure,
    coverage: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Automatically detect entry and exit nodes for a coverage.

    Useful when the user provides only the coverage and wants the
    system to figure out entry/exit classification.

    Args:
        model: Network structure
        coverage: List of node IDs

    Returns:
        Tuple of (entry_nodes, exit_nodes)
    """
    classification = classify_coverage(model, coverage)
    return classification.entry_nodes, classification.exit_nodes
