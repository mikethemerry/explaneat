"""
Split detection algorithm for pre-annotation validation.

Identifies nodes that must be split before an annotation can be created,
based on the entry/intermediate/exit constraints defined in the design spec.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field

from ..core.genome_network import NetworkStructure, NodeType


@dataclass
class ViolationDetail:
    """Details about a node that violates annotation constraints."""

    node_id: str
    reason: str  # "has_external_input_and_output"
    external_inputs: List[Tuple[str, str]]  # (from_node, to_node) connections
    external_outputs: List[Tuple[str, str]]  # (from_node, to_node) connections
    internal_outputs: List[Tuple[str, str]]  # internal connections (for context)


@dataclass
class SplitDetectionResult:
    """Result of split detection analysis."""

    proposed_coverage: List[str]
    violations: List[ViolationDetail]
    suggested_operations: List[Dict[str, Any]]
    adjusted_coverage: Optional[List[str]]
    # Nodes of the proposed coverage already owned by an existing annotation
    # ({"node_id", "annotation"}).  annotate would reject them as "already
    # covered"; they cannot be split (covered nodes are immutable).
    already_covered: List[Dict[str, str]] = field(default_factory=list)


def covered_node_owners(annotations) -> Dict[str, str]:
    """Map node id -> name of the existing annotation whose subgraph owns it."""
    owners: Dict[str, str] = {}
    for ann in annotations or []:
        name = ann.get("name") if isinstance(ann, dict) else ann.name
        nodes = ann.get("subgraph_nodes", []) if isinstance(ann, dict) else ann.subgraph_nodes
        for nid in nodes:
            owners.setdefault(str(nid), name)
    return owners


def _is_input_node(model: NetworkStructure, node_id: str) -> bool:
    node = model.get_node_by_id(node_id)
    if node is not None and node.type == NodeType.INPUT:
        return True
    return node_id in set(model.input_node_ids)


def detect_required_splits(
    model: NetworkStructure,
    proposed_coverage: Set[str],
) -> List[ViolationDetail]:
    """
    Identify nodes that must be split before annotation can be created.

    A node requires splitting if and only if it has:
    - At least one external input (from outside proposed coverage), AND
    - At least one external output (to outside proposed coverage)

    Network input nodes receive their value from the environment, i.e. they
    always have an external input (as in ``classify_node``).  An input node
    in the coverage that also feeds a node outside it therefore fans out
    across the annotation boundary (e.g. into the output's direct-edge group
    and into a hidden gate) and must be split: otherwise whichever
    annotation claims it first leaves the other unable to cover it.

    Such a node cannot be cleanly classified as entry, intermediate, or exit:
    - It has external input → would be an "entry"
    - But entries cannot have external output → violation

    Args:
        model: Current network structure
        proposed_coverage: Set of node IDs the user wants to annotate

    Returns:
        List of ViolationDetail for each violating node
    """
    violations = []

    for node_id in proposed_coverage:
        # Get connections TO this node (inputs)
        inputs = model.get_connections_to(node_id)
        external_inputs = [
            (conn.from_node, conn.to_node)
            for conn in inputs
            if conn.enabled and conn.from_node not in proposed_coverage
        ]

        # Get connections FROM this node (outputs)
        outputs = model.get_connections_from(node_id)
        external_outputs = [
            (conn.from_node, conn.to_node)
            for conn in outputs
            if conn.enabled and conn.to_node not in proposed_coverage
        ]
        internal_outputs = [
            (conn.from_node, conn.to_node)
            for conn in outputs
            if conn.enabled and conn.to_node in proposed_coverage
        ]

        is_input = _is_input_node(model, node_id)

        # Violation: has BOTH external input AND external output
        if (external_inputs or is_input) and external_outputs:
            violations.append(
                ViolationDetail(
                    node_id=node_id,
                    reason=(
                        "input_feeds_outside_coverage"
                        if is_input and not external_inputs
                        else "has_external_input_and_output"
                    ),
                    external_inputs=external_inputs,
                    external_outputs=external_outputs,
                    internal_outputs=internal_outputs,
                )
            )

    return violations


def suggest_split_operations(
    violations: List[ViolationDetail],
) -> List[Dict[str, Any]]:
    """
    Generate suggested split_node operations to resolve violations.

    For each violating node, suggests a split_node operation.

    Args:
        violations: List of violation details

    Returns:
        List of operation dictionaries
    """
    operations = []
    for violation in violations:
        operations.append({
            "type": "split_node",
            "params": {"node_id": violation.node_id},
        })
    return operations


def compute_adjusted_coverage(
    model: NetworkStructure,
    original_coverage: Set[str],
    split_results: Dict[str, List[str]],  # original_node_id -> [split_node_ids]
) -> Set[str]:
    """
    Compute adjusted coverage after splitting violating nodes.

    After splitting, only include split nodes whose outputs are ALL internal
    to the coverage. Split nodes with external outputs are excluded.

    Args:
        model: Network structure AFTER splits have been applied
        original_coverage: The original proposed coverage
        split_results: Mapping of original node ID to its split node IDs

    Returns:
        Adjusted coverage set
    """
    # Start with nodes that weren't split
    split_originals = set(split_results.keys())
    new_coverage = original_coverage - split_originals

    # For each split, include only nodes with internal outputs
    for original_id, split_ids in split_results.items():
        for split_id in split_ids:
            outputs = model.get_connections_from(split_id)

            # Check if ALL outputs go to nodes in the adjusted coverage
            # Note: other split nodes from same original are considered "internal"
            internal_targets = (original_coverage - {original_id}) | set(split_ids)

            all_outputs_internal = all(
                conn.to_node in internal_targets
                for conn in outputs
                if conn.enabled
            )

            if all_outputs_internal:
                new_coverage.add(split_id)
            # else: split_node stays outside coverage (becomes external source)

    return new_coverage


def analyze_coverage_for_splits(
    model: NetworkStructure,
    proposed_coverage: List[str],
    annotations=None,
) -> SplitDetectionResult:
    """
    Complete analysis of proposed coverage for required splits.

    This is the main entry point for split detection. It:
    1. Detects violations
    2. Suggests split operations
    3. (Optionally) computes what the adjusted coverage would be

    Note: This does NOT apply the splits - it only analyzes and suggests.
    The actual splits must be applied via the operations API.

    Args:
        model: Current network structure
        proposed_coverage: List of node IDs to annotate
        annotations: Optional existing annotations (AnnotationData or dicts).
            Nodes of the coverage they already own are reported in
            ``already_covered`` and get no split suggestion.

    Returns:
        SplitDetectionResult with violations, suggestions, and adjusted coverage
    """
    coverage_set = set(proposed_coverage)

    owners = covered_node_owners(annotations)
    already_covered = [
        {"node_id": nid, "annotation": owners[nid]}
        for nid in proposed_coverage
        if nid in owners
    ]
    covered_ids = {c["node_id"] for c in already_covered}

    # Detect violations
    violations = detect_required_splits(model, coverage_set)

    # Generate suggestions (covered nodes are immutable: cannot be split)
    suggested_ops = suggest_split_operations(
        [v for v in violations if v.node_id not in covered_ids]
    )

    # For now, we don't compute adjusted_coverage since splits aren't applied
    # The frontend can compute this after applying the suggested operations
    adjusted_coverage = None

    return SplitDetectionResult(
        proposed_coverage=proposed_coverage,
        violations=violations,
        suggested_operations=suggested_ops,
        adjusted_coverage=adjusted_coverage,
        already_covered=already_covered,
    )


def iterative_split_resolution(
    model: NetworkStructure,
    proposed_coverage: Set[str],
    max_iterations: int = 10,
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """
    Iteratively detect and suggest splits until no violations remain.

    This is a planning function - it simulates what splits would be needed
    but does NOT actually apply them.

    In complex cases, splitting one node may reveal new violations
    (if the split changes what's "internal" vs "external").

    Args:
        model: Current network structure
        proposed_coverage: Initial proposed coverage
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        Tuple of (final_coverage_suggestion, list_of_split_operations)
    """
    coverage = set(proposed_coverage)
    all_operations = []

    for _ in range(max_iterations):
        violations = detect_required_splits(model, coverage)
        if not violations:
            break

        # For each violation, we would split the node
        # Since we can't actually modify the model here, we simulate
        # by assuming the split creates nodes that separate internal vs external

        for violation in violations:
            node_id = violation.node_id
            all_operations.append({
                "type": "split_node",
                "params": {"node_id": node_id},
            })

            # Simulate coverage adjustment:
            # Remove original, add placeholder for internal-output splits
            coverage.discard(node_id)

            # If node has internal outputs, at least one split will be internal
            if violation.internal_outputs:
                # Add a placeholder (actual split ID determined at apply time)
                coverage.add(f"{node_id}_internal")

    return coverage, all_operations
