"""
Collapse validation for annotations.

Implements the collapse operation and its preconditions as defined in
docs/annotation_collapsing_model.md. This is a pure graph-logic module
with no database access, mirroring the pattern of CoverageComputer.

The collapse operation replaces an annotation's internal computation
(intermediate + exit nodes) with a single annotation node, preserving
entry nodes as the annotation's interface.

Three preconditions must hold for a valid collapse:
1. Entry-only ingress: external edges into the annotation target only entry nodes
2. Exit-only egress: edges leaving the annotation originate only from exit nodes
3. Pure exits: exit nodes receive inputs only from within the annotation
"""

import warnings
from dataclasses import dataclass, field
from typing import Set, Tuple, List, Dict, Optional, FrozenSet
from enum import Enum


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CollapseGraph:
    """
    Immutable graph representation for collapse operations.

    All node IDs are strings to support both regular nodes ("5", "-1")
    and split nodes ("5_a", "5_b").
    """

    nodes: FrozenSet[str]
    edges: FrozenSet[Tuple[str, str]]
    input_nodes: FrozenSet[str]
    output_nodes: FrozenSet[str]

    @staticmethod
    def from_sets(
        nodes: Set[str],
        edges: Set[Tuple[str, str]],
        input_nodes: Set[str],
        output_nodes: Set[str],
    ) -> "CollapseGraph":
        """Create a CollapseGraph from mutable sets."""
        return CollapseGraph(
            nodes=frozenset(nodes),
            edges=frozenset(edges),
            input_nodes=frozenset(input_nodes),
            output_nodes=frozenset(output_nodes),
        )


@dataclass(frozen=True)
class CollapseAnnotation:
    """
    Immutable annotation representation for collapse operations.

    Represents the subgraph S = (V_entry, V_exit, V_A, E_A) from the paper.
    """

    id: str
    entry_nodes: FrozenSet[str]
    exit_nodes: FrozenSet[str]
    subgraph_nodes: FrozenSet[str]
    subgraph_edges: FrozenSet[Tuple[str, str]]
    parent_id: Optional[str] = None

    @staticmethod
    def from_sets(
        id: str,
        entry_nodes: Set[str],
        exit_nodes: Set[str],
        subgraph_nodes: Set[str],
        subgraph_edges: Set[Tuple[str, str]],
        parent_id: Optional[str] = None,
    ) -> "CollapseAnnotation":
        """Create a CollapseAnnotation from mutable sets."""
        return CollapseAnnotation(
            id=id,
            entry_nodes=frozenset(entry_nodes),
            exit_nodes=frozenset(exit_nodes),
            subgraph_nodes=frozenset(subgraph_nodes),
            subgraph_edges=frozenset(subgraph_edges),
            parent_id=parent_id,
        )

    @property
    def internal_nodes(self) -> FrozenSet[str]:
        """V_internal = V_A \\ V_entry (intermediate + exit nodes)."""
        return self.subgraph_nodes - self.entry_nodes


class PreconditionType(str, Enum):
    """Types of collapse precondition violations."""

    ENTRY_ONLY_INGRESS = "entry_only_ingress"
    EXIT_ONLY_EGRESS = "exit_only_egress"
    PURE_EXITS = "pure_exits"


class FixType(str, Enum):
    """Types of fixes for precondition violations."""

    IDENTITY_NODE = "identity_node"
    SPLIT_NODE = "split_node"
    EXPAND_SELECTION = "expand_selection"


@dataclass
class PreconditionViolation:
    """A single precondition violation with details."""

    precondition: PreconditionType
    node: str
    edge: Tuple[str, str]
    message: str


@dataclass
class CollapsePreconditionResult:
    """Result of checking collapse preconditions."""

    is_valid: bool
    violations: List[PreconditionViolation] = field(default_factory=list)

    @property
    def entry_ingress_violations(self) -> List[PreconditionViolation]:
        return [
            v
            for v in self.violations
            if v.precondition == PreconditionType.ENTRY_ONLY_INGRESS
        ]

    @property
    def exit_egress_violations(self) -> List[PreconditionViolation]:
        return [
            v
            for v in self.violations
            if v.precondition == PreconditionType.EXIT_ONLY_EGRESS
        ]

    @property
    def pure_exit_violations(self) -> List[PreconditionViolation]:
        return [
            v
            for v in self.violations
            if v.precondition == PreconditionType.PURE_EXITS
        ]


@dataclass
class CollapseResult:
    """Result of a collapse operation."""

    graph: CollapseGraph
    annotation_node_id: str
    hidden_nodes: FrozenSet[str]
    rerouted_edges: List[Tuple[str, str]]


@dataclass
class CollapseValidationResult:
    """Result of validating a collapsed graph or round-trip."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class FixSuggestion:
    """A suggested fix for a precondition violation."""

    fix_type: FixType
    node: str
    description: str
    details: Dict


# =============================================================================
# Validator
# =============================================================================


class CollapseValidator:
    """
    Validates and performs collapse operations on annotation graphs.

    Pure graph-logic module with no database access.
    See docs/annotation_collapsing_model.md for mathematical definitions.
    """

    @staticmethod
    def get_annotation_node_id(annotation: CollapseAnnotation) -> str:
        """Get the synthetic node ID for a collapsed annotation."""
        return f"a_{annotation.id}"

    # -------------------------------------------------------------------------
    # Precondition Checking
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_collapsible(
        graph: CollapseGraph, annotation: CollapseAnnotation
    ) -> CollapsePreconditionResult:
        """
        Check the three collapse preconditions.

        1. Entry-only ingress: external->annotation edges target only entries
        2. Exit-only egress: annotation->external edges originate only from exits
        3. Pure exits: exit nodes have no external inputs

        Args:
            graph: The full graph
            annotation: The annotation to check

        Returns:
            CollapsePreconditionResult with is_valid and any violations
        """
        violations = []

        for u, v in graph.edges:
            u_in = u in annotation.subgraph_nodes
            v_in = v in annotation.subgraph_nodes

            # Precondition 1: Entry-only ingress
            # External -> annotation must target entry nodes
            if not u_in and v_in and v not in annotation.entry_nodes:
                violations.append(
                    PreconditionViolation(
                        precondition=PreconditionType.ENTRY_ONLY_INGRESS,
                        node=v,
                        edge=(u, v),
                        message=f"External node {u} connects to non-entry node {v} inside annotation",
                    )
                )

            # Precondition 2: Exit-only egress
            # Annotation -> external must originate from exit nodes
            if u_in and not v_in and u not in annotation.exit_nodes:
                violations.append(
                    PreconditionViolation(
                        precondition=PreconditionType.EXIT_ONLY_EGRESS,
                        node=u,
                        edge=(u, v),
                        message=f"Non-exit node {u} inside annotation connects to external node {v}",
                    )
                )

            # Precondition 3: Pure exits
            # Exit nodes must not have external inputs
            if not u_in and v_in and v in annotation.exit_nodes:
                violations.append(
                    PreconditionViolation(
                        precondition=PreconditionType.PURE_EXITS,
                        node=v,
                        edge=(u, v),
                        message=f"Exit node {v} has external input from {u}",
                    )
                )

        return CollapsePreconditionResult(
            is_valid=len(violations) == 0,
            violations=violations,
        )

    # -------------------------------------------------------------------------
    # Collapse Operation
    # -------------------------------------------------------------------------

    @staticmethod
    def collapse(
        graph: CollapseGraph, annotation: CollapseAnnotation
    ) -> CollapseResult:
        """
        Perform the collapse operation.

        Collapse(G, A) = G' = (V', E') where:
        - V' = (V \\ V_internal) u {a_A}
        - E' = edges between non-internal nodes
              u entry->a_A edges
              u a_A->external edges (from exits)

        Raises ValueError if preconditions are not met.

        .. deprecated::
            Use :func:`collapse_transform.collapse_structure` instead.

        Args:
            graph: The full graph
            annotation: The annotation to collapse

        Returns:
            CollapseResult with the collapsed graph
        """
        warnings.warn(
            "Use collapse_transform.collapse_structure instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # Check preconditions
        precondition_result = CollapseValidator.validate_collapsible(
            graph, annotation
        )
        if not precondition_result.is_valid:
            violation_msgs = [v.message for v in precondition_result.violations]
            raise ValueError(
                f"Cannot collapse: preconditions violated.\n"
                + "\n".join(f"  - {msg}" for msg in violation_msgs)
            )

        internal_nodes = annotation.internal_nodes
        annotation_node_id = CollapseValidator.get_annotation_node_id(annotation)

        # V' = (V \ V_internal) u {a_A}
        new_nodes = (graph.nodes - internal_nodes) | {annotation_node_id}

        # Build E'
        new_edges: Set[Tuple[str, str]] = set()
        rerouted: List[Tuple[str, str]] = []

        # Keep edges between non-internal nodes
        for u, v in graph.edges:
            if u not in internal_nodes and v not in internal_nodes:
                new_edges.add((u, v))

        # Entry -> a_A edges (if entry connected to any internal node)
        for entry in annotation.entry_nodes:
            has_internal_target = any(
                (entry, w) in graph.edges for w in internal_nodes
            )
            if has_internal_target:
                edge = (entry, annotation_node_id)
                new_edges.add(edge)
                rerouted.append(edge)

        # a_A -> external edges (from exit nodes to outside)
        for u, v in graph.edges:
            if u in annotation.exit_nodes and v not in annotation.subgraph_nodes:
                edge = (annotation_node_id, v)
                new_edges.add(edge)
                rerouted.append(edge)

        # Update input/output nodes
        new_input_nodes = graph.input_nodes - internal_nodes
        new_output_nodes = graph.output_nodes - internal_nodes

        collapsed_graph = CollapseGraph(
            nodes=frozenset(new_nodes),
            edges=frozenset(new_edges),
            input_nodes=frozenset(new_input_nodes),
            output_nodes=frozenset(new_output_nodes),
        )

        return CollapseResult(
            graph=collapsed_graph,
            annotation_node_id=annotation_node_id,
            hidden_nodes=internal_nodes,
            rerouted_edges=rerouted,
        )

    # -------------------------------------------------------------------------
    # Expand Operation (Inverse)
    # -------------------------------------------------------------------------

    @staticmethod
    def expand(
        collapsed_graph: CollapseGraph,
        original_graph: CollapseGraph,
        annotation: CollapseAnnotation,
    ) -> CollapseGraph:
        """
        Expand a collapsed annotation back to the original graph.

        Expand(G', G, A) = G

        .. deprecated::
            Use :func:`collapse_transform.collapse_structure` instead.

        Args:
            collapsed_graph: The collapsed graph
            original_graph: The original graph before collapse
            annotation: The annotation that was collapsed

        Returns:
            The original graph (restored)
        """
        warnings.warn(
            "Use collapse_transform.collapse_structure instead",
            DeprecationWarning,
            stacklevel=2,
        )
        annotation_node_id = CollapseValidator.get_annotation_node_id(annotation)

        # Remove annotation node and its edges
        new_nodes = (collapsed_graph.nodes - {annotation_node_id}) | annotation.internal_nodes

        # Remove edges involving annotation node, restore original edges
        new_edges: Set[Tuple[str, str]] = set()

        # Keep non-annotation-node edges from collapsed graph
        for u, v in collapsed_graph.edges:
            if u != annotation_node_id and v != annotation_node_id:
                new_edges.add((u, v))

        # Restore all original edges involving internal nodes
        for u, v in original_graph.edges:
            if u in annotation.subgraph_nodes or v in annotation.subgraph_nodes:
                new_edges.add((u, v))

        # Restore input/output nodes
        new_input_nodes = original_graph.input_nodes
        new_output_nodes = original_graph.output_nodes

        return CollapseGraph(
            nodes=frozenset(new_nodes),
            edges=frozenset(new_edges),
            input_nodes=frozenset(new_input_nodes),
            output_nodes=frozenset(new_output_nodes),
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_collapsed_graph(
        original: CollapseGraph,
        collapsed: CollapseGraph,
        annotation: CollapseAnnotation,
    ) -> CollapseValidationResult:
        """
        Validate that a collapsed graph is correct.

        Checks:
        - Annotation node exists
        - Internal nodes are removed
        - Entry nodes are preserved
        - Edge count is consistent
        - No cycles introduced (for DAGs)

        Args:
            original: The original graph
            collapsed: The collapsed graph
            annotation: The annotation that was collapsed

        Returns:
            CollapseValidationResult with is_valid and any errors
        """
        errors = []
        annotation_node_id = CollapseValidator.get_annotation_node_id(annotation)
        internal_nodes = annotation.internal_nodes

        # Check annotation node exists
        if annotation_node_id not in collapsed.nodes:
            errors.append(f"Annotation node {annotation_node_id} not in collapsed graph")

        # Check internal nodes are removed
        for node in internal_nodes:
            if node in collapsed.nodes:
                errors.append(f"Internal node {node} still in collapsed graph")

        # Check entry nodes are preserved
        for node in annotation.entry_nodes:
            if node in original.nodes and node not in collapsed.nodes:
                errors.append(f"Entry node {node} was removed from collapsed graph")

        # Check non-annotation nodes are preserved
        for node in original.nodes:
            if node not in internal_nodes and node not in collapsed.nodes:
                errors.append(f"Non-internal node {node} was removed from collapsed graph")

        # Check no edges reference removed nodes (except via annotation node)
        valid_nodes = collapsed.nodes
        for u, v in collapsed.edges:
            if u not in valid_nodes:
                errors.append(f"Edge ({u}, {v}) references non-existent source node")
            if v not in valid_nodes:
                errors.append(f"Edge ({u}, {v}) references non-existent target node")

        # Check for cycles (the original should be a DAG, collapse should preserve this)
        if CollapseValidator._has_cycle(collapsed):
            errors.append("Collapsed graph contains a cycle (original DAG property violated)")

        return CollapseValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    @staticmethod
    def validate_round_trip(
        graph: CollapseGraph, annotation: CollapseAnnotation
    ) -> CollapseValidationResult:
        """
        Validate that collapse + expand = identity.

        Args:
            graph: The original graph
            annotation: The annotation to test

        Returns:
            CollapseValidationResult
        """
        errors = []

        try:
            collapse_result = CollapseValidator.collapse(graph, annotation)
            restored = CollapseValidator.expand(
                collapse_result.graph, graph, annotation
            )

            # Check nodes match
            if restored.nodes != graph.nodes:
                missing = graph.nodes - restored.nodes
                extra = restored.nodes - graph.nodes
                if missing:
                    errors.append(f"Missing nodes after round-trip: {missing}")
                if extra:
                    errors.append(f"Extra nodes after round-trip: {extra}")

            # Check edges match
            if restored.edges != graph.edges:
                missing = graph.edges - restored.edges
                extra = restored.edges - graph.edges
                if missing:
                    errors.append(f"Missing edges after round-trip: {missing}")
                if extra:
                    errors.append(f"Extra edges after round-trip: {extra}")

        except ValueError as e:
            errors.append(f"Collapse failed: {e}")

        return CollapseValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    @staticmethod
    def validate_composition(
        graph: CollapseGraph,
        parent: CollapseAnnotation,
        children: List[CollapseAnnotation],
    ) -> CollapseValidationResult:
        """
        Validate the composition property:
        Collapse(Collapse(G, child), parent) = Collapse(G, parent)

        Args:
            graph: The original graph
            parent: The parent annotation
            children: The child annotations

        Returns:
            CollapseValidationResult
        """
        errors = []

        try:
            # Collapse parent directly
            direct_result = CollapseValidator.collapse(graph, parent)

            # Collapse children first, then parent
            current_graph = graph
            for child in children:
                child_result = CollapseValidator.collapse(current_graph, child)
                current_graph = child_result.graph

            # Now collapse parent on the graph with collapsed children
            # The parent's annotation needs to be updated to reference
            # annotation nodes instead of the children's internal nodes
            parent_with_collapsed_children = _adapt_parent_for_collapsed_children(
                parent, children
            )
            composed_result = CollapseValidator.collapse(
                current_graph, parent_with_collapsed_children
            )

            # Compare the two results
            # The node sets should match
            if direct_result.graph.nodes != composed_result.graph.nodes:
                missing = direct_result.graph.nodes - composed_result.graph.nodes
                extra = composed_result.graph.nodes - direct_result.graph.nodes
                if missing:
                    errors.append(f"Composition missing nodes: {missing}")
                if extra:
                    errors.append(f"Composition extra nodes: {extra}")

            # The edge sets should match
            if direct_result.graph.edges != composed_result.graph.edges:
                missing = direct_result.graph.edges - composed_result.graph.edges
                extra = composed_result.graph.edges - direct_result.graph.edges
                if missing:
                    errors.append(f"Composition missing edges: {missing}")
                if extra:
                    errors.append(f"Composition extra edges: {extra}")

        except ValueError as e:
            errors.append(f"Composition validation failed: {e}")

        return CollapseValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    # -------------------------------------------------------------------------
    # Fix Suggestions
    # -------------------------------------------------------------------------

    @staticmethod
    def suggest_fixes(
        graph: CollapseGraph, annotation: CollapseAnnotation
    ) -> List[FixSuggestion]:
        """
        Suggest fixes for precondition violations.

        Maps violations to fix types:
        - Precondition 1 (entry-only ingress) -> expand selection
        - Precondition 2 (exit-only egress) -> split node
        - Precondition 3 (pure exits) -> identity node

        Args:
            graph: The full graph
            annotation: The annotation to check

        Returns:
            List of FixSuggestion objects
        """
        result = CollapseValidator.validate_collapsible(graph, annotation)
        if result.is_valid:
            return []

        suggestions: List[FixSuggestion] = []
        seen_fixes: Set[Tuple[str, str]] = set()  # (fix_type, node) dedup

        for violation in result.violations:
            key = (violation.precondition.value, violation.node)
            if key in seen_fixes:
                continue
            seen_fixes.add(key)

            if violation.precondition == PreconditionType.ENTRY_ONLY_INGRESS:
                # External input to non-entry node -> expand selection
                external_node = violation.edge[0]
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.EXPAND_SELECTION,
                        node=violation.node,
                        description=(
                            f"Expand selection to include {external_node}, "
                            f"which connects to non-entry node {violation.node}"
                        ),
                        details={
                            "external_source": external_node,
                            "target_node": violation.node,
                            "edge": violation.edge,
                        },
                    )
                )

            elif violation.precondition == PreconditionType.EXIT_ONLY_EGRESS:
                # Non-exit node has external output -> split node
                external_target = violation.edge[1]
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.SPLIT_NODE,
                        node=violation.node,
                        description=(
                            f"Split node {violation.node} to separate internal "
                            f"and external outputs (external: {external_target})"
                        ),
                        details={
                            "node_to_split": violation.node,
                            "external_target": external_target,
                            "edge": violation.edge,
                        },
                    )
                )

            elif violation.precondition == PreconditionType.PURE_EXITS:
                # Exit node has external input -> identity node
                external_source = violation.edge[0]
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.IDENTITY_NODE,
                        node=violation.node,
                        description=(
                            f"Add identity node before exit {violation.node} to "
                            f"intercept annotation's connections (external input "
                            f"from {external_source})"
                        ),
                        details={
                            "exit_node": violation.node,
                            "external_source": external_source,
                            "edge": violation.edge,
                        },
                    )
                )

        return suggestions

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _has_cycle(graph: CollapseGraph) -> bool:
        """Check if a graph has any cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {node: WHITE for node in graph.nodes}

        # Build adjacency list
        adj: Dict[str, List[str]] = {node: [] for node in graph.nodes}
        for u, v in graph.edges:
            if u in adj:
                adj[u].append(v)

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adj[node]:
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    return True  # Back edge = cycle
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in graph.nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False


# =============================================================================
# Internal Helpers
# =============================================================================


def _adapt_parent_for_collapsed_children(
    parent: CollapseAnnotation,
    children: List[CollapseAnnotation],
) -> CollapseAnnotation:
    """
    Adapt a parent annotation for a graph where children are already collapsed.

    When children are collapsed, their internal nodes are replaced by annotation
    nodes. The parent's subgraph needs to reference these annotation nodes instead.

    Args:
        parent: The parent annotation
        children: The child annotations (already collapsed)

    Returns:
        Adapted parent annotation for the collapsed graph
    """
    new_subgraph_nodes = set(parent.subgraph_nodes)
    new_subgraph_edges = set(parent.subgraph_edges)
    new_entry_nodes = set(parent.entry_nodes)
    new_exit_nodes = set(parent.exit_nodes)

    for child in children:
        child_internal = child.internal_nodes
        child_ann_id = CollapseValidator.get_annotation_node_id(child)

        # Remove child's internal nodes from parent's subgraph
        new_subgraph_nodes -= child_internal
        # Add annotation node
        new_subgraph_nodes.add(child_ann_id)

        # Update edges: replace references to child's internal nodes
        edges_to_remove = set()
        edges_to_add = set()

        for u, v in new_subgraph_edges:
            u_is_internal = u in child_internal
            v_is_internal = v in child_internal

            if u_is_internal and v_is_internal:
                # Internal edge of child — remove
                edges_to_remove.add((u, v))
            elif u_is_internal and not v_is_internal:
                # Child internal -> parent node: replace with a_child -> parent node
                edges_to_remove.add((u, v))
                # Only if u was an exit of the child
                if u in child.exit_nodes:
                    edges_to_add.add((child_ann_id, v))
            elif not u_is_internal and v_is_internal:
                # Parent node -> child internal: replace with parent node -> a_child
                edges_to_remove.add((u, v))
                # Only if v would route to the annotation node (entry->internal)
                if u in child.entry_nodes:
                    edges_to_add.add((u, child_ann_id))

        new_subgraph_edges -= edges_to_remove
        new_subgraph_edges |= edges_to_add

        # Update entry/exit: child's entries might be parent's entries
        # but child's internal nodes in parent's entry/exit sets need updating
        if child_internal & new_entry_nodes:
            new_entry_nodes -= child_internal
            new_entry_nodes.add(child_ann_id)
        if child_internal & new_exit_nodes:
            new_exit_nodes -= child_internal
            new_exit_nodes.add(child_ann_id)

    return CollapseAnnotation.from_sets(
        id=parent.id,
        entry_nodes=new_entry_nodes,
        exit_nodes=new_exit_nodes,
        subgraph_nodes=new_subgraph_nodes,
        subgraph_edges=new_subgraph_edges,
        parent_id=parent.parent_id,
    )
