"""MCP tools for model structure inspection (phenotype, model state, node info)."""

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ..server import get_db
from ..helpers import (
    build_engine,
    load_genome_and_config,
    serialize_network,
)
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.collapse_transform import collapse_structure


def get_phenotype(genome_id: str) -> str:
    """Get the phenotype (active subgraph) of a genome.

    Returns the pruned network structure containing only nodes and connections
    reachable from inputs to outputs.

    Args:
        genome_id: UUID of the genome to inspect.
    """
    db = get_db()
    with db.session_scope() as session:
        neat_genome, config, genome_db = load_genome_and_config(session, genome_id)
        explainer = ExplaNEAT(neat_genome, config)
        phenotype = explainer.get_phenotype_network()
        return json.dumps(serialize_network(phenotype), indent=2)


def get_model_state(genome_id: str, collapsed: Optional[str] = None) -> str:
    """Get the current model state with all operations applied.

    Returns the network structure after replaying all operations (splits,
    identity nodes, annotations) on the phenotype. Optionally collapses
    specified annotations into function nodes.

    Args:
        genome_id: UUID of the genome to inspect.
        collapsed: Comma-separated annotation names to collapse into function nodes.
    """
    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        model_state = engine.current_state

        collapsed_names = []
        if collapsed:
            collapsed_names = [name.strip() for name in collapsed.split(",") if name.strip()]

        if collapsed_names:
            collapsed_ids = set(collapsed_names)
            model_state = collapse_structure(
                model_state, engine.annotations, collapsed_ids
            )

        # Serialize annotations
        annotations_list = []
        for ann in engine.annotations:
            annotations_list.append({
                "name": ann.name,
                "hypothesis": ann.hypothesis,
                "entry_nodes": ann.entry_nodes,
                "exit_nodes": ann.exit_nodes,
                "subgraph_nodes": ann.subgraph_nodes,
                "subgraph_connections": [list(c) for c in ann.subgraph_connections],
                "parent_annotation_id": ann.parent_annotation_id,
                "display_name": ann.display_name,
            })

        result = serialize_network(model_state)
        result["annotations"] = annotations_list
        if collapsed_names:
            result["collapsed_annotations"] = collapsed_names
        return json.dumps(result, indent=2)


def get_node_info(genome_id: str, node_id: str) -> str:
    """Get detailed information about a specific node in the model state.

    Returns node properties (type, bias, activation, etc.) plus incoming
    and outgoing connections.

    Args:
        genome_id: UUID of the genome containing the node.
        node_id: ID of the node to inspect.
    """
    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        model_state = engine.current_state

        # Find the node
        target_node = None
        for node in model_state.nodes:
            if node.id == node_id:
                target_node = node
                break

        if target_node is None:
            return json.dumps({"error": f"Node '{node_id}' not found in model state"}, indent=2)

        # Build node info
        node_info = {
            "id": target_node.id,
            "type": target_node.type.value,
            "bias": target_node.bias,
            "activation": target_node.activation,
            "response": target_node.response,
            "aggregation": target_node.aggregation,
        }

        if target_node.display_name:
            node_info["display_name"] = target_node.display_name

        if target_node.function_metadata:
            fm = target_node.function_metadata
            node_info["function_metadata"] = {
                "annotation_name": fm.annotation_name,
                "annotation_id": fm.annotation_id,
                "hypothesis": fm.hypothesis,
                "n_inputs": fm.n_inputs,
                "n_outputs": fm.n_outputs,
                "input_names": fm.input_names,
                "output_names": fm.output_names,
            }

        # Gather connections
        incoming = []
        outgoing = []
        for conn in model_state.connections:
            if not conn.enabled:
                continue
            if conn.to_node == node_id:
                incoming.append({
                    "from_node": conn.from_node,
                    "weight": conn.weight,
                })
            if conn.from_node == node_id:
                outgoing.append({
                    "to_node": conn.to_node,
                    "weight": conn.weight,
                })

        node_info["incoming_connections"] = incoming
        node_info["outgoing_connections"] = outgoing

        return json.dumps(node_info, indent=2)


def register(mcp: FastMCP) -> None:
    """Register model structure tools with the MCP server."""
    mcp.tool()(get_phenotype)
    mcp.tool()(get_model_state)
    mcp.tool()(get_node_info)
