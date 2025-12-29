"""
Standard network structure representation for NEAT genomes.

Provides a unified format for genotype (all nodes/connections) and phenotype
(active, reachable subgraph) representations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class NodeType(str, Enum):
    """Node type classification."""
    INPUT = "input"
    OUTPUT = "output"
    HIDDEN = "hidden"


@dataclass
class NetworkNode:
    """Represents a single node in the network."""
    id: int
    type: NodeType
    bias: Optional[float] = None
    activation: Optional[str] = None
    response: Optional[float] = None
    aggregation: Optional[str] = None


@dataclass
class NetworkConnection:
    """Represents a single connection in the network."""
    from_node: int
    to_node: int
    weight: float
    enabled: bool
    innovation: Optional[int] = None


@dataclass
class NetworkStructure:
    """
    Standard representation of a neural network structure.
    
    Can represent either genotype (all nodes/connections) or phenotype
    (pruned to active, reachable subgraph).
    """
    nodes: List[NetworkNode]
    connections: List[NetworkConnection]
    input_node_ids: List[int] = field(default_factory=list)
    output_node_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_node_by_id(self, node_id: int) -> Optional[NetworkNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_connections_from(self, node_id: int) -> List[NetworkConnection]:
        """Get all connections originating from a node."""
        return [conn for conn in self.connections if conn.from_node == node_id]
    
    def get_connections_to(self, node_id: int) -> List[NetworkConnection]:
        """Get all connections terminating at a node."""
        return [conn for conn in self.connections if conn.to_node == node_id]
    
    def get_node_ids(self) -> Set[int]:
        """Get set of all node IDs."""
        return {node.id for node in self.nodes}
    
    def get_enabled_connections(self) -> List[NetworkConnection]:
        """Get only enabled connections."""
        return [conn for conn in self.connections if conn.enabled]
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the network structure.
        
        Returns:
            Dict with 'is_valid' bool and optional 'errors' list
        """
        errors = []
        
        # Check that all connection endpoints reference existing nodes
        node_ids = self.get_node_ids()
        for conn in self.connections:
            if conn.from_node not in node_ids:
                errors.append(f"Connection from non-existent node {conn.from_node}")
            if conn.to_node not in node_ids:
                errors.append(f"Connection to non-existent node {conn.to_node}")
        
        # Check that input/output node IDs are valid
        for input_id in self.input_node_ids:
            if input_id not in node_ids:
                errors.append(f"Input node {input_id} not in nodes list")
        
        for output_id in self.output_node_ids:
            if output_id not in node_ids:
                errors.append(f"Output node {output_id} not in nodes list")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }


def get_phenotype_with_splits(explanation_id: str) -> NetworkStructure:
    """
    Get phenotype graph with splits applied for an explanation.
    
    Original nodes are replaced by split nodes where splits exist.
    Split nodes use their specific outgoing_connections.
    All split nodes share the original node's incoming connections.
    
    Args:
        explanation_id: UUID of the explanation
        
    Returns:
        NetworkStructure with split nodes integrated
    """
    from ..db import db, Explanation, Genome, NodeSplit
    from ..db.serialization import deserialize_genome
    import neat
    
    with db.session_scope() as session:
        explanation = session.get(Explanation, explanation_id)
        if not explanation:
            raise ValueError(f"Explanation {explanation_id} not found")
        
        # Get genome
        genome_record = session.get(Genome, explanation.genome_id)
        if not genome_record:
            raise ValueError(f"Genome {explanation.genome_id} not found")
        
        # Get all splits for this explanation
        splits = explanation.node_splits
        
        # Build mapping: original_node_id -> list of (split_node_id, outgoing_connections)
        splits_by_original: Dict[int, List[Tuple[int, List[Tuple[int, int]]]]] = {}
        for split in splits:
            orig_id = split.original_node_id
            split_id = split.split_node_id
            outgoing = split.get_outgoing_connections()
            if orig_id not in splits_by_original:
                splits_by_original[orig_id] = []
            splits_by_original[orig_id].append((split_id, outgoing))
        
        # Load genome to get phenotype
        # We need the config to deserialize
        population = genome_record.population
        experiment = population.experiment
        
        neat_config_text = experiment.neat_config_text or ""
        if not neat_config_text or not neat_config_text.strip():
            raise ValueError("Experiment has no stored NEAT configuration")
        
        # Create minimal config for deserialization
        config_path = "config-file.cfg"  # Default, should ideally come from experiment
        try:
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
        except:
            # If config file doesn't exist, we can't proceed
            raise ValueError("Cannot load NEAT config. Cannot generate phenotype with splits.")
        
        neat_genome = deserialize_genome(genome_record.genome_data, config)
        
        # Get phenotype using ExplaNEAT
        from ..core.explaneat import ExplaNEAT
        explainer = ExplaNEAT(neat_genome, config)
        phenotype = explainer.get_phenotype_network()
        
        # Apply splits: replace original nodes with split nodes
        # Build mapping of original nodes to their incoming connections
        incoming_by_node: Dict[int, List[NetworkConnection]] = {}
        for conn in phenotype.connections:
            if conn.to_node not in incoming_by_node:
                incoming_by_node[conn.to_node] = []
            incoming_by_node[conn.to_node].append(conn)
        
        # Create new nodes and connections with splits applied
        new_nodes: List[NetworkNode] = []
        new_connections: List[NetworkConnection] = []
        original_nodes_to_remove: Set[int] = set()
        
        # Process each node
        for node in phenotype.nodes:
            node_id = node.id
            
            # If this node has splits, replace it with split nodes
            if node_id in splits_by_original:
                original_nodes_to_remove.add(node_id)
                
                # Get incoming connections for original node (shared by all splits)
                incoming_conns = incoming_by_node.get(node_id, [])
                
                # Create a split node for each split
                for split_id, split_outgoing in splits_by_original[node_id]:
                    # Create new node with split_node_id
                    split_node = NetworkNode(
                        id=split_id,
                        type=node.type,
                        bias=node.bias,
                        activation=node.activation,
                        response=node.response,
                        aggregation=node.aggregation,
                    )
                    new_nodes.append(split_node)
                    
                    # Add incoming connections (shared by all splits)
                    for incoming_conn in incoming_conns:
                        new_conn = NetworkConnection(
                            from_node=incoming_conn.from_node,
                            to_node=split_id,  # Point to split node
                            weight=incoming_conn.weight,
                            enabled=incoming_conn.enabled,
                            innovation=incoming_conn.innovation,
                        )
                        new_connections.append(new_conn)
                    
                    # Add outgoing connections (specific to this split)
                    for from_node, to_node in split_outgoing:
                        # Find matching connection in phenotype to get weight/enabled
                        matching_conn = None
                        for conn in phenotype.connections:
                            if conn.from_node == node_id and conn.to_node == to_node:
                                matching_conn = conn
                                break
                        
                        if matching_conn:
                            new_conn = NetworkConnection(
                                from_node=split_id,  # From split node
                                to_node=to_node,
                                weight=matching_conn.weight,
                                enabled=matching_conn.enabled,
                                innovation=matching_conn.innovation,
                            )
                            new_connections.append(new_conn)
            else:
                # No splits, keep original node
                new_nodes.append(node)
        
        # Add connections that don't involve split nodes
        for conn in phenotype.connections:
            # Skip connections involving nodes that were split
            if conn.from_node not in original_nodes_to_remove and conn.to_node not in original_nodes_to_remove:
                # Check if this connection is already added (might be added as part of split processing)
                if not any(
                    c.from_node == conn.from_node and c.to_node == conn.to_node
                    for c in new_connections
                ):
                    new_connections.append(conn)
        
        # Update input/output node IDs if they were split
        new_input_node_ids = phenotype.input_node_ids.copy()
        new_output_node_ids = phenotype.output_node_ids.copy()
        
        # If input/output nodes were split, we need to handle that
        # For now, keep original IDs (splits typically don't apply to input/output nodes)
        
        return NetworkStructure(
            nodes=new_nodes,
            connections=new_connections,
            input_node_ids=new_input_node_ids,
            output_node_ids=new_output_node_ids,
            metadata={
                **phenotype.metadata,
                "has_splits": True,
                "explanation_id": explanation_id,
            },
        )


