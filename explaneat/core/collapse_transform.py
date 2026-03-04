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

    Annotations are processed children-before-parents (sorted by subgraph
    size, smallest first).

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

    # Sort children-before-parents: smallest subgraph first
    to_collapse.sort(key=lambda ann: len(ann.subgraph_nodes))

    # Work on a deep copy so we never mutate the input
    result = deepcopy(structure)

    for annotation in to_collapse:
        result = _collapse_one(result, annotation)

    return result


def _collapse_one(
    structure: NetworkStructure,
    annotation: AnnotationData,
) -> NetworkStructure:
    """Collapse a single annotation into a function node.

    Returns a new NetworkStructure (does not mutate the input).
    """
    entry_set = set(annotation.entry_nodes)
    exit_set = set(annotation.exit_nodes)
    subgraph_set = set(annotation.subgraph_nodes)
    internal_nodes = subgraph_set - entry_set  # intermediate + exit nodes

    fn_node_id = f"fn_{annotation.name}"

    # --- Compute formula (best-effort) ---
    formula_latex = _try_compute_formula(structure, annotation)

    # --- Capture node properties and connection weights ---
    # Preserve bias/activation for all subgraph nodes so the function
    # can be reconstructed for forward passes after collapse.
    nodes_by_id = {n.id: n for n in structure.nodes}
    node_properties: Dict[str, dict] = {}
    for nid in annotation.subgraph_nodes:
        node_obj = nodes_by_id.get(nid)
        if node_obj is not None:
            node_properties[nid] = {
                "bias": node_obj.bias if node_obj.bias is not None else 0.0,
                "activation": node_obj.activation or "relu",
            }

    # Preserve connection weights for the subgraph connections.
    enabled_conns = {
        (c.from_node, c.to_node): c.weight
        for c in structure.connections
        if c.enabled
    }
    connection_weights: Dict[Tuple[str, str], float] = {}
    for from_n, to_n in annotation.subgraph_connections:
        w = enabled_conns.get((from_n, to_n), 0.0)
        connection_weights[(from_n, to_n)] = w

    # --- Create function node metadata ---
    metadata = FunctionNodeMetadata(
        annotation_name=annotation.name,
        annotation_id=annotation.name,
        hypothesis=annotation.hypothesis,
        n_inputs=len(annotation.entry_nodes),
        n_outputs=len(annotation.exit_nodes),
        input_names=list(annotation.entry_nodes),
        output_names=list(annotation.exit_nodes),
        formula_latex=formula_latex,
        subgraph_nodes=list(annotation.subgraph_nodes),
        subgraph_connections=list(annotation.subgraph_connections),
        node_properties=node_properties,
        connection_weights=connection_weights,
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

        if from_is_internal and to_is_internal:
            # Internal -> internal: DROP
            continue

        elif from_is_internal and not to_in_subgraph:
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
            new_connections.append(conn)

    # --- Build new structure ---
    return NetworkStructure(
        nodes=new_nodes,
        connections=new_connections,
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
