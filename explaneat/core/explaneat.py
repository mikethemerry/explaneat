import pandas as pd
import numpy as np
import random
import logging
from collections import deque
from typing import List, Set, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

import pprint

from explaneat.core.neuralneat import NeuralNeat
from explaneat.core.genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)


class ExplaNEAT:
    def __init__(self, genome, config, neat_class=NeuralNeat):
        self.genome = genome
        self.config = config
        self.net = neat_class(genome, config)
        self.phenotype = self.net

    def shapes(self):
        return self.net.shapes()

    def n_genome_params(self):
        nNodes = len(self.genome.nodes)
        nConnections = len(self.genome.connections)

        return nNodes + nConnections

    def density(self):
        nParams = self.n_genome_params()
        denseSize = 0
        for ix, shape in self.shapes().items():
            denseSize += shape[0] * shape[1]
        return nParams / denseSize

    def depth(self):
        return self.net.n_layers

    def node_depth(self, nodeId):
        return self.net.node_mapping.node_mapping[nodeId]["depth"]

    def skippines(self):
        skippy_sum = 0
        for connection in self.genome.connections:
            skippy = self.node_depth(connection[1]) - self.node_depth(connection[0]) - 1
            skippy_sum += skippy
        return skippy_sum / len(self.genome.connections)

    def _get_input_nodes(self) -> List[int]:
        """Get input node IDs from config."""
        if hasattr(self.config, "genome_config"):
            return list(self.config.genome_config.input_keys)
        # Fallback: negative node IDs are inputs
        return [n for n in self.genome.nodes.keys() if n < 0]

    def _get_output_nodes(self) -> List[int]:
        """Get output node IDs from config."""
        if hasattr(self.config, "genome_config"):
            return list(self.config.genome_config.output_keys)
        # Fallback: node 0 is the output
        return [n for n in self.genome.nodes.keys() if n == 0]

    def _get_node_type(self, node_id: int) -> NodeType:
        """Determine node type from ID and config."""
        input_nodes = set(self._get_input_nodes())
        output_nodes = set(self._get_output_nodes())
        
        if node_id in input_nodes:
            return NodeType.INPUT
        elif node_id in output_nodes:
            return NodeType.OUTPUT
        else:
            return NodeType.HIDDEN

    def get_genotype_network(self) -> NetworkStructure:
        """
        Get the full genotype network structure (all nodes and connections).
        
        Returns:
            NetworkStructure containing all nodes and connections from the genome
        """
        # Get input/output node IDs from config
        input_node_ids = self._get_input_nodes()
        output_node_ids = self._get_output_nodes()
        
        # Build set of all node IDs that should exist
        all_node_ids = set(self.genome.nodes.keys())
        all_node_ids.update(input_node_ids)
        all_node_ids.update(output_node_ids)
        
        nodes = []
        for node_id in sorted(all_node_ids):
            # If node exists in genome, use its properties
            if node_id in self.genome.nodes:
                node = self.genome.nodes[node_id]
                node_type = self._get_node_type(node_id)
                nodes.append(
                    NetworkNode(
                        id=node_id,
                        type=node_type,
                        bias=getattr(node, "bias", None),
                        activation=getattr(node, "activation", None),
                        response=getattr(node, "response", None),
                        aggregation=getattr(node, "aggregation", None),
                    )
                )
            else:
                # Node doesn't exist in genome (e.g., input with no connections)
                # Create a placeholder node
                node_type = self._get_node_type(node_id)
                nodes.append(
                    NetworkNode(
                        id=node_id,
                        type=node_type,
                        bias=None,
                        activation=None,
                        response=None,
                        aggregation=None,
                    )
                )

        connections = []
        for conn_key, conn in self.genome.connections.items():
            from_node, to_node = conn_key
            connections.append(
                NetworkConnection(
                    from_node=from_node,
                    to_node=to_node,
                    weight=conn.weight,
                    enabled=conn.enabled,
                    innovation=getattr(conn, "key", None),
                )
            )

        return NetworkStructure(
            nodes=nodes,
            connections=connections,
            input_node_ids=input_node_ids,
            output_node_ids=output_node_ids,
            metadata={"representation": "genotype"},
        )

    def _traverse_nodes(
        self, network: NetworkStructure, start_nodes: List[int], direction: str
    ) -> Set[int]:
        """
        Traverse network from start nodes in given direction.
        
        Args:
            network: NetworkStructure to traverse
            start_nodes: Starting node IDs
            direction: "out" for forward, "in" for backward
        
        Returns:
            Set of reachable node IDs
        """
        reachable: Set[int] = set()
        queue = deque(start_nodes)
        node_ids = network.get_node_ids()
        enabled_conns = network.get_enabled_connections()

        # Build adjacency lists for enabled connections only
        if direction == "out":
            # Forward: from -> to
            adj = {node_id: [] for node_id in node_ids}
            for conn in enabled_conns:
                if conn.from_node in adj:
                    adj[conn.from_node].append(conn.to_node)
        else:
            # Backward: to -> from
            adj = {node_id: [] for node_id in node_ids}
            for conn in enabled_conns:
                if conn.to_node in adj:
                    adj[conn.to_node].append(conn.from_node)

        while queue:
            node = queue.popleft()
            if node in reachable or node not in node_ids:
                continue
            reachable.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in reachable:
                    queue.append(neighbor)

        return reachable

    def get_phenotype_network(self) -> NetworkStructure:
        """
        Get the phenotype network structure (pruned to active, reachable subgraph).
        
        Only includes nodes and connections that are on at least one path
        from an input to an output.
        
        Returns:
            NetworkStructure containing only active, reachable nodes/connections
        """
        genotype = self.get_genotype_network()
        
        # Get input and output nodes that exist in the genotype
        # These should always be included in phenotype
        input_nodes = [
            nid for nid in genotype.input_node_ids if nid in genotype.get_node_ids()
        ]
        output_nodes = [
            nid for nid in genotype.output_node_ids if nid in genotype.get_node_ids()
        ]

        # Always include all input and output nodes in phenotype, even if they have no connections
        # If no inputs/outputs found, return genotype as-is
        if not input_nodes and not output_nodes:
            logging.warning(
                "No input or output nodes found in genotype, returning genotype as phenotype"
            )
            return genotype

        # Find nodes reachable forward from inputs (if we have inputs)
        if input_nodes:
            forward_reachable = self._traverse_nodes(genotype, input_nodes, direction="out")
        else:
            forward_reachable = set()
        
        # Find nodes reachable backward from outputs (if we have outputs)
        if output_nodes:
            backward_reachable = self._traverse_nodes(
                genotype, output_nodes, direction="in"
            )
        else:
            backward_reachable = set()

        # Active nodes are those reachable in both directions
        # BUT: always include all input and output nodes regardless of reachability
        active_node_ids = (forward_reachable & backward_reachable) | set(input_nodes) | set(output_nodes)

        if not active_node_ids:
            logging.warning("No active nodes found in phenotype")
            return genotype  # Return genotype if no active nodes

        # Filter nodes to only active ones
        active_nodes = [
            node for node in genotype.nodes if node.id in active_node_ids
        ]

        # Filter connections to only those between active nodes
        active_connections = [
            conn
            for conn in genotype.connections
            if conn.from_node in active_node_ids
            and conn.to_node in active_node_ids
            and conn.enabled
        ]

        # Filter input/output lists to only active nodes
        active_input_ids = [nid for nid in input_nodes if nid in active_node_ids]
        active_output_ids = [nid for nid in output_nodes if nid in active_node_ids]

        return NetworkStructure(
            nodes=active_nodes,
            connections=active_connections,
            input_node_ids=active_input_ids,
            output_node_ids=active_output_ids,
            metadata={
                "representation": "phenotype",
                "pruned_nodes": len(genotype.nodes) - len(active_nodes),
                "pruned_connections": len(genotype.connections) - len(active_connections),
            },
        )
