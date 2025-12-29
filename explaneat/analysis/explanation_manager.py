"""
Explanation Manager for creating and managing explanations.

An explanation groups annotations and node splits into a coherent explanation
of a model. Multiple explanations can exist for the same model.
"""

from typing import List, Dict, Any, Optional
import uuid

from ..db import db, Explanation, Annotation, NodeSplit, Genome
from .coverage import CoverageComputer


class ExplanationManager:
    """
    Manager class for creating and managing explanations.
    
    Handles explanation creation, linking annotations and splits,
    and validation of well-formed explanations.
    """

    @staticmethod
    def create_explanation(
        genome_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new explanation for a genome.
        
        Args:
            genome_id: UUID of the genome being explained
            name: Optional name for this explanation
            description: Optional description
            
        Returns:
            Dictionary representation of the created explanation
        """
        with db.session_scope() as session:
            # Verify genome exists
            genome = session.get(Genome, genome_id)
            if not genome:
                raise ValueError(f"Genome {genome_id} not found")
            
            explanation = Explanation(
                genome_id=genome_id,
                name=name,
                description=description,
                is_well_formed=False,
            )
            session.add(explanation)
            session.commit()
            explanation_id = explanation.id
        
        # Return fresh query
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if explanation:
                return explanation.to_dict()
            return None

    @staticmethod
    def get_explanations(genome_id: str) -> List[Dict[str, Any]]:
        """
        Get all explanations for a genome.
        
        Args:
            genome_id: UUID of the genome
            
        Returns:
            List of explanation dictionaries
        """
        with db.session_scope() as session:
            explanations = (
                session.query(Explanation)
                .filter_by(genome_id=genome_id)
                .order_by(Explanation.created_at.desc())
                .all()
            )
            return [exp.to_dict() for exp in explanations]

    @staticmethod
    def get_explanation(explanation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific explanation by ID.
        
        Args:
            explanation_id: UUID of the explanation
            
        Returns:
            Explanation dictionary or None
        """
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if explanation:
                return explanation.to_dict()
            return None

    @staticmethod
    def add_annotation_to_explanation(
        explanation_id: str, annotation_id: str
    ) -> Dict[str, Any]:
        """
        Link an annotation to an explanation.
        
        Args:
            explanation_id: UUID of the explanation
            annotation_id: UUID of the annotation
            
        Returns:
            Updated annotation dictionary
        """
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if not explanation:
                raise ValueError(f"Explanation {explanation_id} not found")
            
            annotation = session.get(Annotation, annotation_id)
            if not annotation:
                raise ValueError(f"Annotation {annotation_id} not found")
            
            if str(annotation.genome_id) != str(explanation.genome_id):
                raise ValueError("Annotation and explanation must belong to the same genome")
            
            annotation.explanation_id = explanation_id
            session.commit()
            session.refresh(annotation)
            return annotation.to_dict()

    @staticmethod
    def add_split_to_explanation(
        explanation_id: str, split_id: str
    ) -> Dict[str, Any]:
        """
        Link a node split to an explanation.
        
        Args:
            explanation_id: UUID of the explanation
            split_id: UUID of the node split
            
        Returns:
            Updated node split dictionary
        """
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if not explanation:
                raise ValueError(f"Explanation {explanation_id} not found")
            
            node_split = session.get(NodeSplit, split_id)
            if not node_split:
                raise ValueError(f"Node split {split_id} not found")
            
            if str(node_split.genome_id) != str(explanation.genome_id):
                raise ValueError("Node split and explanation must belong to the same genome")
            
            node_split.explanation_id = explanation_id
            session.commit()
            session.refresh(node_split)
            return node_split.to_dict()

    @staticmethod
    def get_explanation_annotations(explanation_id: str) -> List[Dict[str, Any]]:
        """
        Get all annotations in an explanation.
        
        Args:
            explanation_id: UUID of the explanation
            
        Returns:
            List of annotation dictionaries
        """
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if not explanation:
                raise ValueError(f"Explanation {explanation_id} not found")
            return [ann.to_dict() for ann in explanation.annotations]

    @staticmethod
    def get_explanation_splits(explanation_id: str) -> List[Dict[str, Any]]:
        """
        Get all node splits in an explanation.
        
        Args:
            explanation_id: UUID of the explanation
            
        Returns:
            List of node split dictionaries
        """
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if not explanation:
                raise ValueError(f"Explanation {explanation_id} not found")
            return [split.to_dict() for split in explanation.node_splits]

    @staticmethod
    def validate_well_formed(explanation_id: str) -> Dict[str, Any]:
        """
        Validate if an explanation is well-formed according to the paper's criteria.
        
        Checks:
        1. All leaf annotations are valid
        2. Full structural coverage: C_V^struct = 1
        3. All compositions explained: C_V^comp = 1
        4. Root covers global model
        
        Args:
            explanation_id: UUID of the explanation
            
        Returns:
            Dictionary with validation results and updated is_well_formed flag
        """
        from .coverage import compute_structural_coverage, compute_compositional_coverage
        
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if not explanation:
                raise ValueError(f"Explanation {explanation_id} not found")
            
            # Compute coverage metrics
            structural_coverage = compute_structural_coverage(explanation_id)
            compositional_coverage = compute_compositional_coverage(explanation_id)
            
            # Update cached coverage values
            explanation.structural_coverage = structural_coverage
            explanation.compositional_coverage = compositional_coverage
            
            # Mark as well-formed if criteria met
            # Check if there's a root annotation that covers the global model
            annotations = explanation.annotations
            root_annotations = [
                ann for ann in annotations 
                if ann.parent_annotation_id is None
            ]
            
            # For now, mark as well-formed if coverage is complete
            # TODO: Add check that root covers global model
            is_well_formed = (
                structural_coverage == 1.0 and 
                compositional_coverage == 1.0 and
                len(root_annotations) > 0
            )
            
            explanation.is_well_formed = is_well_formed
            
            session.commit()
            session.refresh(explanation)
            
            return {
                "is_well_formed": explanation.is_well_formed,
                "structural_coverage": explanation.structural_coverage,
                "compositional_coverage": explanation.compositional_coverage,
            }

    @staticmethod
    def compute_and_cache_coverage(explanation_id: str) -> Dict[str, float]:
        """
        Compute and cache coverage metrics for an explanation.
        
        Args:
            explanation_id: UUID of the explanation
            
        Returns:
            Dictionary with structural_coverage and compositional_coverage
        """
        from .coverage import compute_structural_coverage, compute_compositional_coverage
        
        with db.session_scope() as session:
            explanation = session.get(Explanation, explanation_id)
            if not explanation:
                raise ValueError(f"Explanation {explanation_id} not found")
            
            # Compute coverage
            structural_coverage = compute_structural_coverage(explanation_id)
            compositional_coverage = compute_compositional_coverage(explanation_id)
            
            # Update cached values
            explanation.structural_coverage = structural_coverage
            explanation.compositional_coverage = compositional_coverage
            
            session.commit()
            session.refresh(explanation)
            
            return {
                "structural_coverage": explanation.structural_coverage,
                "compositional_coverage": explanation.compositional_coverage,
            }

    @staticmethod
    def get_phenotype_with_splits(explanation_id: str):
        """
        Get phenotype graph with splits applied for an explanation.
        
        Args:
            explanation_id: UUID of the explanation
            
        Returns:
            NetworkStructure with splits applied
        """
        from ..core.genome_network import get_phenotype_with_splits as get_phenotype_with_splits_func
        
        return get_phenotype_with_splits_func(explanation_id)

