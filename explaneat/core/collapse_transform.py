"""
Collapse transform for annotations.

Produces a derived NetworkStructure where collapsed annotations are replaced
by function nodes. This is a pure function -- no mutations, no side effects.

The transform replaces the cycle-prone graph-surgery approach with a
term-rewriting approach that is provably cycle-free.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from .genome_network import (
    FunctionNodeMetadata,
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from .model_state import AnnotationData


def _find_descendant_names(
    parent_name: str,
    children_map: Dict[str, List[AnnotationData]],
) -> Set[str]:
    """Recursively find all descendant annotation names."""
    result: Set[str] = set()
    for child in children_map.get(parent_name, []):
        result.add(child.name)
        result.update(_find_descendant_names(child.name, children_map))
    return result


def _compute_effective_subgraph_nodes(
    annotation: AnnotationData,
    annotation_by_name: Dict[str, AnnotationData],
    children_map: Dict[str, List[AnnotationData]],
) -> Set[str]:
    """Compute effective subgraph nodes by unioning all descendant subgraphs.

    For compositional annotations with empty subgraph_nodes, the effective
    subgraph is the union of all descendants' subgraph_nodes plus the
    annotation's own entry and exit nodes.
    """
    effective = set(annotation.subgraph_nodes)
    # Ensure entry/exit nodes are included (compositional annotations may
    # have empty subgraph_nodes but populated entry/exit lists)
    effective.update(annotation.entry_nodes)
    effective.update(annotation.exit_nodes)
    for desc_name in _find_descendant_names(annotation.name, children_map):
        desc = annotation_by_name.get(desc_name)
        if desc:
            effective.update(desc.subgraph_nodes)
    return effective


def compute_effective_entries_exits(
    effective_subgraph: Set[str],
    structure: NetworkStructure,
) -> Tuple[Set[str], Set[str]]:
    """Derive true entry and exit nodes from the effective subgraph boundary.

    Entry nodes: subgraph nodes that receive at least one connection from
    outside the subgraph (or are input nodes with no predecessors within
    the subgraph).

    Exit nodes: subgraph nodes that send at least one connection to a node
    outside the subgraph.

    This replaces manually-declared entry/exit for compositional annotations,
    which can be wrong when internal wiring between children is mislabeled.
    """
    entries: Set[str] = set()
    exits: Set[str] = set()

    for conn in structure.connections:
        if not conn.enabled:
            continue
        from_in = conn.from_node in effective_subgraph
        to_in = conn.to_node in effective_subgraph

        if not from_in and to_in:
            entries.add(conn.to_node)
        elif from_in and not to_in:
            exits.add(conn.from_node)

    # Input nodes in the subgraph with no internal predecessors are also entries
    input_ids = set(structure.input_node_ids)
    for node_id in effective_subgraph:
        if node_id in input_ids:
            has_internal_pred = any(
                c.from_node in effective_subgraph
                for c in structure.connections
                if c.enabled and c.to_node == node_id
            )
            if not has_internal_pred:
                entries.add(node_id)

    return entries, exits


def collapse_structure(
    structure: NetworkStructure,
    annotations: List[AnnotationData],
    collapsed_ids: Set[str],
) -> NetworkStructure:
    """
    Produce a derived NetworkStructure with collapsed annotations replaced
    by function nodes.

    For each annotation name in collapsed_ids:
    1. Remove internal nodes (subgraph - entry)
    2. Add a function node ``fn_{annotation_name}``
    3. Rewire connections: entry->fn_node, fn_node->external targets

    Annotations are processed children-before-parents (sorted by effective
    subgraph size, smallest first). For compositional annotations, the
    effective subgraph is the union of all descendants' subgraphs.

    Args:
        structure: The input NetworkStructure (not mutated).
        annotations: List of all annotations.
        collapsed_ids: Set of annotation names to collapse.

    Returns:
        A new NetworkStructure with collapsed annotations replaced by
        function nodes.
    """
    if not collapsed_ids:
        return deepcopy(structure)

    # Build lookup of annotations by name
    annotation_by_name: Dict[str, AnnotationData] = {
        ann.name: ann for ann in annotations
    }

    # Filter to only annotations that are actually requested
    to_collapse: List[AnnotationData] = [
        annotation_by_name[name]
        for name in collapsed_ids
        if name in annotation_by_name
    ]

    if not to_collapse:
        return deepcopy(structure)

    # Build children map for compositional annotations
    children_map: Dict[str, List[AnnotationData]] = {}
    for ann in annotations:
        if ann.parent_annotation_id and ann.parent_annotation_id in annotation_by_name:
            children_map.setdefault(ann.parent_annotation_id, []).append(ann)

    # Compute effective subgraph sizes for correct sorting (children before parents)
    effective_sizes: Dict[str, int] = {}
    for ann in to_collapse:
        effective = _compute_effective_subgraph_nodes(ann, annotation_by_name, children_map)
        effective_sizes[ann.name] = len(effective)

    # Sort children-before-parents: smallest effective subgraph first
    to_collapse.sort(key=lambda ann: effective_sizes.get(ann.name, 0))

    # Work on a deep copy so we never mutate the input
    result = deepcopy(structure)

    for annotation in to_collapse:
        descendant_names = _find_descendant_names(annotation.name, children_map)

        if descendant_names:
            # Compositional annotation: compute effective subgraph and derive
            # true entry/exit nodes from boundary connections.
            original_effective = _compute_effective_subgraph_nodes(
                annotation, annotation_by_name, children_map
            )
            current_node_ids = {n.id for n in result.nodes}
            current_subgraph: Set[str] = set()
            for nid in original_effective:
                if nid in current_node_ids:
                    current_subgraph.add(nid)
            for desc_name in descendant_names:
                fn_id = f"fn_{desc_name}"
                if fn_id in current_node_ids:
                    current_subgraph.add(fn_id)

            # Auto-derive correct entries/exits from the CURRENT structure
            # (after children have been collapsed). Must use current_subgraph,
            # not original_effective, because original node IDs may no longer
            # exist in the structure (replaced by fn_ nodes).
            effective_entries, effective_exits = compute_effective_entries_exits(
                current_subgraph, result
            )

            # Create a corrected annotation with derived entry/exit.
            # Sort for deterministic output_index assignment.
            corrected = AnnotationData(
                name=annotation.name,
                hypothesis=annotation.hypothesis,
                entry_nodes=sorted(effective_entries),
                exit_nodes=sorted(effective_exits),
                subgraph_nodes=list(current_subgraph),
                subgraph_connections=annotation.subgraph_connections,
                evidence=annotation.evidence,
                parent_annotation_id=annotation.parent_annotation_id,
            )

            result = _collapse_one(
                result, corrected, effective_nodes_override=current_subgraph
            )
        else:
            result = _collapse_one(result, annotation)

    return result


def _collapse_one(
    structure: NetworkStructure,
    annotation: AnnotationData,
    effective_nodes_override: Optional[Set[str]] = None,
) -> NetworkStructure:
    """Collapse a single annotation into a function node.

    Returns a new NetworkStructure (does not mutate the input).

    If ``effective_nodes_override`` is provided it is used as the subgraph
    node set instead of ``annotation.subgraph_nodes``.  This handles
    compositional annotations whose effective subgraph includes descendants'
    nodes and ``fn_`` replacement nodes from already-collapsed children.
    """
    entry_set = set(annotation.entry_nodes)
    exit_set = set(annotation.exit_nodes)
    subgraph_set = (
        effective_nodes_override
        if effective_nodes_override is not None
        else set(annotation.subgraph_nodes)
    )
    internal_nodes = subgraph_set - entry_set  # intermediate + exit nodes

    fn_node_id = f"fn_{annotation.name}"

    # --- Compute formula (best-effort) ---
    formula_latex = _try_compute_formula(structure, annotation)

    # --- Capture node properties and connection weights ---
    # Preserve bias/activation for all subgraph nodes so the function
    # can be reconstructed for forward passes after collapse.
    nodes_by_id = {n.id: n for n in structure.nodes}
    node_properties: Dict[str, dict] = {}
    for nid in subgraph_set:
        node_obj = nodes_by_id.get(nid)
        if node_obj is not None:
            node_properties[nid] = {
                "bias": node_obj.bias if node_obj.bias is not None else 0.0,
                "activation": node_obj.activation or "relu",
            }

    # Derive subgraph connections from structure (handles compositional case
    # where annotation.subgraph_connections may be empty).
    enabled_conns = {
        (c.from_node, c.to_node): c.weight
        for c in structure.connections
        if c.enabled
    }
    effective_connections = [
        (c.from_node, c.to_node)
        for c in structure.connections
        if c.enabled and c.from_node in subgraph_set and c.to_node in subgraph_set
    ]
    connection_weights: Dict[Tuple[str, str], float] = {}
    for from_n, to_n in effective_connections:
        connection_weights[(from_n, to_n)] = enabled_conns.get((from_n, to_n), 0.0)

    # --- Collect child function metadata for fn_* nodes in this subgraph ---
    # Needed so StructureNetwork can recursively evaluate nested function nodes
    # when reconstructing the forward pass from this metadata.
    child_function_metadata = {}
    for nid in subgraph_set:
        node_obj = nodes_by_id.get(nid)
        if (
            node_obj is not None
            and node_obj.type == NodeType.FUNCTION
            and node_obj.function_metadata is not None
        ):
            child_function_metadata[nid] = node_obj.function_metadata

    # --- Create function node metadata ---
    display_map = {n.id: n.display_label for n in structure.nodes}
    metadata = FunctionNodeMetadata(
        annotation_name=annotation.name,
        annotation_id=annotation.name,
        hypothesis=annotation.hypothesis,
        n_inputs=len(annotation.entry_nodes),
        n_outputs=len(annotation.exit_nodes),
        input_names=[display_map.get(n, n) for n in annotation.entry_nodes],
        output_names=[display_map.get(n, n) for n in annotation.exit_nodes],
        formula_latex=formula_latex,
        subgraph_nodes=list(subgraph_set),
        subgraph_connections=effective_connections,
        node_properties=node_properties,
        connection_weights=connection_weights,
        child_function_metadata=child_function_metadata,
    )

    fn_node = NetworkNode(
        id=fn_node_id,
        type=NodeType.FUNCTION,
        function_metadata=metadata,
    )

    # --- Build new node list: remove internal nodes, add function node ---
    new_nodes = [n for n in structure.nodes if n.id not in internal_nodes]
    new_nodes.append(fn_node)

    # --- Build exit_node -> output_index mapping ---
    exit_index: Dict[str, int] = {
        node_id: idx for idx, node_id in enumerate(annotation.exit_nodes)
    }

    # --- Rewire connections ---
    new_connections: List[NetworkConnection] = []

    # Track seen connections for deduplication
    seen_entry_to_fn: Set[str] = set()  # entry node ids that already have entry->fn
    seen_exit_to_ext: Set[Tuple[str, str, Optional[int]]] = set()  # (fn_node, target, output_index)

    for conn in structure.connections:
        from_in_subgraph = conn.from_node in subgraph_set
        to_in_subgraph = conn.to_node in subgraph_set
        from_is_internal = conn.from_node in internal_nodes
        to_is_internal = conn.to_node in internal_nodes

        if from_is_internal:
            if to_is_internal:
                # Internal -> internal: DROP
                continue
            elif not to_in_subgraph:
                # Internal -> external (exit -> outside): Reroute fn_node -> external
                # Only exit nodes should have external outputs if preconditions hold
                if conn.from_node in exit_set:
                    out_idx = exit_index.get(conn.from_node)
                    dedup_key = (fn_node_id, conn.to_node, out_idx)
                    if dedup_key not in seen_exit_to_ext:
                        seen_exit_to_ext.add(dedup_key)
                        new_connections.append(
                            NetworkConnection(
                                from_node=fn_node_id,
                                to_node=conn.to_node,
                                weight=conn.weight,
                                enabled=conn.enabled,
                                output_index=out_idx,
                            )
                        )
                # else: skip/drop (shouldn't happen if preconditions hold)
            else:
                # Internal -> entry (within subgraph): DROP
                # Arises in compositional annotations where fn_child -> parent_entry
                continue

        elif not from_in_subgraph and to_is_internal:
            # External -> internal: Should not happen if preconditions hold.
            # Skip/drop.
            continue

        elif from_in_subgraph and not from_is_internal and to_is_internal:
            # Entry -> internal: Reroute entry -> fn_node
            if conn.from_node not in seen_entry_to_fn:
                seen_entry_to_fn.add(conn.from_node)
                new_connections.append(
                    NetworkConnection(
                        from_node=conn.from_node,
                        to_node=fn_node_id,
                        weight=conn.weight,
                        enabled=conn.enabled,
                    )
                )

        else:
            # Everything else: preserve as-is
            # This includes:
            #   - external -> external
            #   - external -> entry (entry is not internal)
            #   - entry -> external (entry has external outputs)
            #   - entry -> entry (within subgraph, both survive)
            new_connections.append(conn)

    # --- Filter out any dangling connections ---
    valid_node_ids = {n.id for n in new_nodes}
    clean_connections = [
        c for c in new_connections
        if c.from_node in valid_node_ids and c.to_node in valid_node_ids
    ]

    # --- Build new structure ---
    return NetworkStructure(
        nodes=new_nodes,
        connections=clean_connections,
        input_node_ids=list(structure.input_node_ids),
        output_node_ids=list(structure.output_node_ids),
        metadata=dict(structure.metadata),
    )


def _try_compute_formula(
    structure: NetworkStructure,
    annotation: AnnotationData,
) -> Optional[str]:
    """Best-effort LaTeX formula extraction. Returns None on any failure."""
    try:
        from ..analysis.annotation_function import AnnotationFunction

        af = AnnotationFunction.from_structure(annotation, structure)
        return af.to_latex()
    except Exception:
        return None
