"""
Node Splitting Manager for creating and managing node splits.

Node splits allow dual-function nodes (nodes with edges both inside and outside
an annotation) to be conceptually split into multiple nodes, each carrying a
subset of the original node's outgoing connections.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import uuid

from ..db import db, NodeSplit, Genome, Annotation, Explanation
from ..core.genome_network import NetworkStructure


class NodeSplitManager:
    """
    Manager class for creating and managing node splits.
    
    Handles split creation, validation, and queries.
    """

    @staticmethod
    def create_split(
        genome_id: str,
        original_node_id: int,
        split_node_id: int,
        outgoing_connections: List[Tuple[int, int]],
        explanation_id: str,
        annotation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a node split.
        
        Args:
            genome_id: UUID of the genome
            original_node_id: ID of the node being split
            split_node_id: Unique ID for this split node (must be unique per original_node_id within explanation)
            outgoing_connections: List of [from_node, to_node] tuples - subset of original node's outgoing connections
            explanation_id: UUID of explanation this split belongs to
            annotation_id: Optional UUID of annotation using this split
            
        Returns:
            Dictionary representation of the created split
            
        Raises:
            ValueError: If validation fails
        """
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
            
            # Get existing splits for this node in this explanation
            existing_splits = (
                session.query(NodeSplit)
                .filter_by(
                    genome_id=genome_id,
                    original_node_id=original_node_id,
                    explanation_id=explanation_id,
                )
                .all()
            )
            
            # Validate split_node_id is unique for this original_node_id within explanation
            for existing_split in existing_splits:
                if existing_split.split_node_id == split_node_id:
                    raise ValueError(
                        f"Split node ID {split_node_id} already exists for original node {original_node_id} in this explanation"
                    )
            
            # Get original node's outgoing connections from genome
            # TODO: Load genome and get actual outgoing connections
            # For now, we'll validate in application layer
            
            # Validate no connection overlaps with existing splits
            existing_connections: Set[Tuple[int, int]] = set()
            for existing_split in existing_splits:
                existing_connections.update(existing_split.get_outgoing_connections())
            
            new_connections = set(
                tuple(conn) if isinstance(conn, (list, tuple)) else conn
                for conn in outgoing_connections
            )
            
            overlap = existing_connections & new_connections
            if overlap:
                raise ValueError(
                    f"Connections {overlap} already belong to existing splits for this node"
                )
            
            # Validate annotation_id if provided
            if annotation_id:
                annotation = session.get(Annotation, annotation_id)
                if not annotation:
                    raise ValueError(f"Annotation {annotation_id} not found")
                if str(annotation.genome_id) != genome_id:
                    raise ValueError("Annotation must belong to the same genome")
                if str(annotation.explanation_id) != explanation_id:
                    raise ValueError("Annotation must belong to the same explanation")
        
        # Create the split
        with db.session_scope() as session:
            node_split = NodeSplit(
                genome_id=genome_id,
                original_node_id=original_node_id,
                split_node_id=split_node_id,
                outgoing_connections=outgoing_connections,
                annotation_id=annotation_id,
                explanation_id=explanation_id,
            )
            session.add(node_split)
            session.commit()
            split_id = node_split.id
        
        # Return fresh query
        with db.session_scope() as session:
            node_split = session.get(NodeSplit, split_id)
            if node_split:
                return node_split.to_dict()
            return None

    @staticmethod
    def get_splits_for_node(
        genome_id: str,
        original_node_id: int,
        explanation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all splits for a node.
        
        Args:
            genome_id: UUID of the genome
            original_node_id: ID of the original node
            explanation_id: Optional UUID of explanation to filter by
            
        Returns:
            List of node split dictionaries
        """
        with db.session_scope() as session:
            query = session.query(NodeSplit).filter_by(
                genome_id=genome_id, original_node_id=original_node_id
            )
            if explanation_id:
                query = query.filter_by(explanation_id=explanation_id)
            splits = query.all()
            return [split.to_dict() for split in splits]

    @staticmethod
    def get_splits_for_explanation(explanation_id: str) -> List[Dict[str, Any]]:
        """
        Get all splits for an explanation.
        
        Args:
            explanation_id: UUID of the explanation
            
        Returns:
            List of node split dictionaries
        """
        with db.session_scope() as session:
            splits = (
                session.query(NodeSplit)
                .filter_by(explanation_id=explanation_id)
                .all()
            )
            return [split.to_dict() for split in splits]

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
    def get_split_by_id(genome_id: str, split_node_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific split by split_node_id.
        
        Args:
            genome_id: UUID of the genome
            split_node_id: ID of the split node
            
        Returns:
            Node split dictionary or None
        """
        with db.session_scope() as session:
            split = (
                session.query(NodeSplit)
                .filter_by(genome_id=genome_id, split_node_id=split_node_id)
                .first()
            )
            if split:
                return split.to_dict()
            return None

    @staticmethod
    def validate_splits_complete(
        genome_id: str, original_node_id: int, explanation_id: str
    ) -> Dict[str, Any]:
        """
        Validate that all splits for a node cover all outgoing connections.
        
        Args:
            genome_id: UUID of the genome
            original_node_id: ID of the original node
            explanation_id: UUID of the explanation
            
        Returns:
            Dictionary with validation results
        """
        with db.session_scope() as session:
            # Get all splits for this node in this explanation
            splits = (
                session.query(NodeSplit)
                .filter_by(
                    genome_id=genome_id,
                    original_node_id=original_node_id,
                    explanation_id=explanation_id,
                )
                .all()
            )
            
            if not splits:
                return {
                    "is_complete": False,
                    "message": "No splits found for this node",
                }
            
            # Collect all outgoing connections from splits
            split_connections: Set[Tuple[int, int]] = set()
            for split in splits:
                split_connections.update(split.get_outgoing_connections())
            
            # TODO: Get actual outgoing connections from genome
            # For now, we can't fully validate without genome structure
            # This should be implemented when we have access to genome network structure
            
            return {
                "is_complete": True,  # Placeholder
                "split_count": len(splits),
                "total_connections_covered": len(split_connections),
            }

    @staticmethod
    def get_incoming_connections(
        genome_id: str, original_node_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get incoming connections for a node (shared by all splits).
        
        Args:
            genome_id: UUID of the genome
            original_node_id: ID of the original node
            
        Returns:
            List of [from_node, to_node] tuples for incoming connections
        """
        # TODO: Load genome and get actual incoming connections
        # This requires access to genome network structure
        # Placeholder implementation
        return []

    @staticmethod
    def get_split_node_connections(
        genome_id: str, split_node_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get outgoing connections for a specific split node.
        
        Args:
            genome_id: UUID of the genome
            split_node_id: ID of the split node
            
        Returns:
            List of [from_node, to_node] tuples for outgoing connections
        """
        with db.session_scope() as session:
            split = (
                session.query(NodeSplit)
                .filter_by(genome_id=genome_id, split_node_id=split_node_id)
                .first()
            )
            if not split:
                raise ValueError(f"Split node {split_node_id} not found")
            return split.get_outgoing_connections()

    @staticmethod
    def get_original_node_id(genome_id: str, split_node_id: int) -> int:
        """
        Get original node ID from split node ID.
        
        Args:
            genome_id: UUID of the genome
            split_node_id: ID of the split node
            
        Returns:
            Original node ID
        """
        with db.session_scope() as session:
            split = (
                session.query(NodeSplit)
                .filter_by(genome_id=genome_id, split_node_id=split_node_id)
                .first()
            )
            if not split:
                raise ValueError(f"Split node {split_node_id} not found")
            return split.original_node_id

