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

