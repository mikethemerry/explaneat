"""
Node Splitting Manager for creating and managing node splits.

Node splits allow dual-function nodes (nodes with edges both inside and outside
an annotation) to be conceptually split into multiple nodes. When a node is
split, it is fully split - meaning a dedicated split node is created for each
outgoing connection. Each split node carries exactly one outgoing connection.

The split is stored as a single row per original_node_id, with split_mappings
containing all split nodes: {"5_a": [[5, 10]], "5_b": [[5, 20]], "5_c": [[5, 30]]}
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import uuid

from ..db import db, NodeSplit, Genome, Annotation, Explanation
from ..core.genome_network import NetworkStructure
from .explanation_manager import ExplanationManager


class NodeSplitManager:
    """
    Manager class for creating and managing node splits.
    
    Handles split creation, validation, and queries.
    Each split is stored as a single row with all split mappings.
    """
    
    @staticmethod
    def format_split_node_display(split_node_id: str) -> str:
        """
        Format a split node ID for display.
        
        Args:
            split_node_id: The split node ID (string like "5_a", "5_b")
            
        Returns:
            Display string like "5_a", "5_b", etc.
        """
        return split_node_id
    
    @staticmethod
    def get_original_node_id_from_split(split_node_id: str) -> Optional[str]:
        """
        Extract original node ID from split node ID string.
        
        Args:
            split_node_id: The split node ID (string like "5_a", "5_b")
            
        Returns:
            Original node ID as string if format matches (e.g., "5_a" -> "5"), None otherwise
        """
        try:
            # Format is "node_id_letter" (e.g., "5_a", "10_b")
            parts = split_node_id.split("_", 1)
            if len(parts) == 2:
                return parts[0]  # Return as string
        except (ValueError, AttributeError):
            pass
        return None

    @staticmethod
    def create_complete_split(
        genome_id: str,
        original_node_id: str,
        split_mappings: Dict[str, List[Tuple[str, str]]],
        explanation_id: Optional[str] = None,
        annotation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete node split with all split mappings.
        
        This creates or updates a single row containing all splits for the original node.
        split_mappings format: {"5_a": [("5", "10")], "5_b": [("5", "20")], "5_c": [("5", "30")]}
        
        Args:
            genome_id: UUID of the genome
            original_node_id: ID of the node being split (string, e.g., "5")
            split_mappings: Dict mapping split_node_id (str) -> list of outgoing connections (str tuples)
                Each split node should have exactly one outgoing connection (full splitting)
            explanation_id: Optional UUID of explanation this split belongs to (auto-created if not provided)
            annotation_id: Optional UUID of annotation using this split
            
        Returns:
            Dictionary representation of the created/updated split
            
        Raises:
            ValueError: If validation fails
        """
        # Validate: Don't allow splitting input nodes (negative string IDs like "-1")
        try:
            orig_id_int = int(original_node_id)
            if orig_id_int < 0:
                raise ValueError(
                    f"Cannot split input node {original_node_id}. Input nodes (negative IDs) should not be split."
                )
        except ValueError:
            # If original_node_id can't be converted to int, it's likely already a string split node
            # Allow it (split nodes can connect to other split nodes)
            pass
        
        if not split_mappings:
            raise ValueError("split_mappings cannot be empty")
        
        # Validate each split has exactly one connection (full splitting)
        for split_node_id, conns in split_mappings.items():
            if not conns:
                raise ValueError(f"Split node {split_node_id} must have at least one outgoing connection")
            # With full splitting, each split should have exactly one connection
            if len(conns) > 1:
                raise ValueError(
                    f"Split node {split_node_id} has {len(conns)} connections. "
                    "Full splitting requires exactly one connection per split node."
                )
            # Validate connection format
            for conn in conns:
                if not isinstance(conn, (list, tuple)) or len(conn) != 2:
                    raise ValueError(f"Invalid connection format: {conn}. Expected [from_node, to_node]")
                # Convert to strings for comparison
                from_node_str = str(conn[0])
                if from_node_str != original_node_id:
                    raise ValueError(
                        f"Connection {conn} has incorrect from_node. Expected {original_node_id}"
                    )
        
        # Get or create explanation for this genome (single explanation per genome)
        if not explanation_id:
            explanation = ExplanationManager.get_or_create_explanation(genome_id)
            explanation_id = explanation["id"]
        
        # Validate explanation exists
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if not explanation:
                raise ValueError(f"Explanation {explanation_id} not found")
            if str(explanation.genome_id) != genome_id:
                raise ValueError("Explanation must belong to the same genome")
            
            # Get genome to validate node exists
            genome = session.get(Genome, genome_id)
            if not genome:
                raise ValueError(f"Genome {genome_id} not found")
            
            # Check if split already exists for this node
            existing_split = (
                session.query(NodeSplit)
                .filter_by(
                    genome_id=genome_id,
                    original_node_id=original_node_id,
                    explanation_id=explanation_id,
                )
                .first()
            )
            
            if existing_split:
                # Update existing split
                existing_split.split_mappings = split_mappings
                if annotation_id:
                    existing_split.annotation_id = annotation_id
                session.commit()
                return existing_split.to_dict()
            else:
                # Create new split
                node_split = NodeSplit(
                    genome_id=genome_id,
                    original_node_id=str(original_node_id),  # Ensure it's a string
                    split_mappings=split_mappings,
                    annotation_id=annotation_id,
                    explanation_id=explanation_id,
                )
                session.add(node_split)
                session.commit()
                split_id = node_split.id
                
                # Return fresh query
                node_split = session.get(NodeSplit, split_id)
                if node_split:
                    return node_split.to_dict()
                return None

    @staticmethod
    def get_splits_for_node(
        genome_id: str,
        original_node_id: str,
        explanation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the split for a node (single row per node).
        
        Args:
            genome_id: UUID of the genome
            original_node_id: ID of the original node
            explanation_id: Optional UUID of explanation to filter by
            
        Returns:
            Node split dictionary or None if not found
        """
        with db.session_scope() as session:
            query = session.query(NodeSplit).filter_by(
                genome_id=genome_id, original_node_id=str(original_node_id)
            )
            if explanation_id:
                query = query.filter_by(explanation_id=explanation_id)
            split = query.first()
            if split:
                return split.to_dict()
            return None

    @staticmethod
    def get_splits_for_explanation(explanation_id: str = None, genome_id: str = None) -> List[Dict[str, Any]]:
        """
        Get all splits for an explanation.
        Can be called with either explanation_id or genome_id.
        
        Args:
            explanation_id: UUID of the explanation (optional)
            genome_id: UUID of the genome (optional, used if explanation_id not provided)
            
        Returns:
            List of node split dictionaries
        """
        with db.session_scope() as session:
            if explanation_id:
                splits = (
                    session.query(NodeSplit)
                    .filter_by(explanation_id=explanation_id)
                    .all()
                )
                return [split.to_dict() for split in splits]
            elif genome_id:
                explanation = (
                    session.query(Explanation)
                    .filter_by(genome_id=genome_id)
                    .first()
                )
                if not explanation:
                    return []  # No explanation yet, so no splits
                return [split.to_dict() for split in explanation.node_splits]
            else:
                raise ValueError("Must provide either explanation_id or genome_id")

    @staticmethod
    def get_splits_for_genome(genome_id: str) -> List[Dict[str, Any]]:
        """
        Get all splits for a genome.
        
        Args:
            genome_id: UUID of the genome
            
        Returns:
            List of node split dictionaries
        """
        with db.session_scope() as session:
            splits = (
                session.query(NodeSplit)
                .filter_by(genome_id=genome_id)
                .all()
            )
            return [split.to_dict() for split in splits]

    @staticmethod
    def get_split_by_split_node_id(genome_id: str, split_node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the split containing a specific split_node_id.
        
        Args:
            genome_id: UUID of the genome
            split_node_id: ID of the split node (e.g., "5_a")
            
        Returns:
            Node split dictionary or None
        """
        original_node_id = NodeSplitManager.get_original_node_id_from_split(split_node_id)
        if not original_node_id:
            return None
        
        split_dict = NodeSplitManager.get_splits_for_node(genome_id, original_node_id)
        if split_dict and split_node_id in split_dict.get("split_mappings", {}):
            return split_dict
        return None

    @staticmethod
    def get_split_node_connections(
        genome_id: str, split_node_id: str
    ) -> List[Tuple[str, str]]:
        """
        Get outgoing connections for a specific split node.
        
        Args:
            genome_id: UUID of the genome
            split_node_id: ID of the split node (e.g., "5_a")
            
        Returns:
            List of (from_node, to_node) tuples for outgoing connections (both strings)
        """
        split_dict = NodeSplitManager.get_split_by_split_node_id(genome_id, split_node_id)
        if not split_dict:
            raise ValueError(f"Split node {split_node_id} not found")
        
        split_mappings = split_dict.get("split_mappings", {})
        if split_node_id not in split_mappings:
            raise ValueError(f"Split node {split_node_id} not found in split mappings")
        
        conns = split_mappings[split_node_id]
        return [
            (str(conn[0]), str(conn[1])) if isinstance(conn, (list, tuple)) and len(conn) == 2 else conn
            for conn in conns
        ]

    @staticmethod
    def get_original_node_id(genome_id: str, split_node_id: str) -> str:
        """
        Get original node ID from split node ID.
        
        Args:
            genome_id: UUID of the genome
            split_node_id: ID of the split node (e.g., "5_a")
            
        Returns:
            Original node ID as string
        """
        original_node_id = NodeSplitManager.get_original_node_id_from_split(split_node_id)
        if original_node_id is None:
            raise ValueError(f"Invalid split_node_id format: {split_node_id}")
        
        # Verify the split exists
        split_dict = NodeSplitManager.get_split_by_split_node_id(genome_id, split_node_id)
        if not split_dict:
            raise ValueError(f"Split node {split_node_id} not found")
        
        return original_node_id

    @staticmethod
    def validate_splits_complete(
        genome_id: str, original_node_id: str, explanation_id: str,
        expected_outgoing_connections: Set[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Validate that splits for a node cover all expected outgoing connections.
        
        Args:
            genome_id: UUID of the genome
            original_node_id: ID of the original node
            explanation_id: UUID of the explanation
            expected_outgoing_connections: Set of all outgoing connections the node should have
            
        Returns:
            Dictionary with validation results
        """
        split_dict = NodeSplitManager.get_splits_for_node(
            genome_id, original_node_id, explanation_id
        )
        
        if not split_dict:
            return {
                "is_complete": False,
                "message": "No splits found for this node",
            }
        
        # Collect all outgoing connections from splits
        split_connections: Set[Tuple[str, str]] = set()
        split_mappings = split_dict.get("split_mappings", {})
        for split_node_id, conns in split_mappings.items():
            for conn in conns:
                # Convert to tuple of strings
                if isinstance(conn, (list, tuple)) and len(conn) == 2:
                    split_connections.add((str(conn[0]), str(conn[1])))
                else:
                    split_connections.add(conn)
        
        missing = expected_outgoing_connections - split_connections
        extra = split_connections - expected_outgoing_connections
        
        is_complete = len(missing) == 0 and len(extra) == 0
        
        return {
            "is_complete": is_complete,
            "split_count": len(split_mappings),
            "total_connections_covered": len(split_connections),
            "expected_connections": len(expected_outgoing_connections),
            "missing_connections": list(missing) if missing else None,
            "extra_connections": list(extra) if extra else None,
        }
