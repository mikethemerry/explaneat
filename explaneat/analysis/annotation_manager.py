"""
Annotation Manager for creating and managing genome annotations

Provides CRUD operations for annotations with validation.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import uuid

from ..db import db, Annotation, Genome, Explanation
from .subgraph_validator import SubgraphValidator
from .evidence_schema import EvidenceBuilder, create_empty_evidence


class AnnotationManager:
    """
    Manager class for creating and managing genome annotations.

    Handles validation of subgraphs, evidence structure, and provides
    CRUD operations for annotations.
    """

    @staticmethod
    def create_annotation(
        genome_id: str,
        nodes: List[int],
        connections: List[Tuple[int, int]],
        hypothesis: str,
        entry_nodes: Optional[List[int]] = None,
        exit_nodes: Optional[List[int]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        parent_annotation_id: Optional[str] = None,
        explanation_id: Optional[str] = None,
        validate_against_genome: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a new annotation for a genome.

        Args:
            genome_id: UUID of the genome to annotate
            nodes: List of node IDs in the subgraph
            connections: List of connection tuples (from_node, to_node)
            hypothesis: Description of what the subgraph does
            entry_nodes: List of entry node IDs (if None, will infer from nodes)
            exit_nodes: List of exit node IDs (if None, will infer from nodes)
            evidence: Optional evidence dictionary (will use empty if None)
            name: Optional name/title for the annotation
            parent_annotation_id: Optional UUID of parent annotation (for composition annotations)
            explanation_id: Optional UUID of explanation this annotation belongs to
            validate_against_genome: If True, validate nodes/connections exist in genome

        Returns:
            Dictionary representation of the created annotation (to avoid detached instance issues)

        Raises:
            ValueError: If validation fails
        """
        # Validate connectivity
        connectivity_result = SubgraphValidator.validate_connectivity(
            nodes, connections
        )
        if not connectivity_result["is_connected"]:
            raise ValueError(
                f"Subgraph is not connected: {connectivity_result.get('error_message', 'Unknown error')}"
            )

        # Validate evidence structure if provided
        if evidence is not None:
            is_valid, error_msg = EvidenceBuilder.validate_evidence(evidence)
            if not is_valid:
                raise ValueError(f"Invalid evidence structure: {error_msg}")
        else:
            evidence = create_empty_evidence()

        # Validate against genome if requested
        if validate_against_genome:
            with db.session_scope() as session:
                genome_record = session.get(Genome, genome_id)
                if not genome_record:
                    raise ValueError(f"Genome {genome_id} not found")

                # Load NEAT config to deserialize genome
                import neat
                from ..db.serialization import deserialize_genome

                # Try to get config from experiment (for future use if needed)
                # population = genome_record.population
                # experiment = population.experiment

                # Create a minimal config for validation
                # In practice, you'd load the actual config
                config_path = (
                    "config-file.cfg"  # Default, should be stored in experiment
                )
                try:
                    config = neat.Config(
                        neat.DefaultGenome,
                        neat.DefaultReproduction,
                        neat.DefaultSpeciesSet,
                        neat.DefaultStagnation,
                        config_path,
                    )
                    neat_genome = deserialize_genome(genome_record.genome_data, config)

                    genome_validation = SubgraphValidator.validate_against_genome(
                        neat_genome, nodes, connections, config=config
                    )
                    if not genome_validation["is_valid"]:
                        raise ValueError(
                            f"Subgraph validation failed: {genome_validation.get('error_message', 'Unknown error')}"
                        )
                except Exception as e:
                    # If we can't load the genome, skip genome validation but warn
                    import warnings

                    warnings.warn(
                        f"Could not validate against genome: {e}. Skipping genome validation."
                    )

        # Infer entry/exit nodes if not provided
        if entry_nodes is None or exit_nodes is None:
            # For now, we'll set them to empty and require explicit specification
            # In the future, we could infer from the graph structure
            if entry_nodes is None:
                entry_nodes = []
            if exit_nodes is None:
                exit_nodes = []
        
        # Validate that entry and exit nodes are in the subgraph
        nodes_set = set(nodes)
        if entry_nodes and not all(n in nodes_set for n in entry_nodes):
            raise ValueError("All entry nodes must be in the subgraph nodes")
        if exit_nodes and not all(n in nodes_set for n in exit_nodes):
            raise ValueError("All exit nodes must be in the subgraph nodes")
        
        # Validate parent_annotation_id if provided
        if parent_annotation_id:
            with db.session_scope() as session:
                parent = session.get(Annotation, parent_annotation_id)
                if not parent:
                    raise ValueError(f"Parent annotation {parent_annotation_id} not found")
                if str(parent.genome_id) != genome_id:
                    raise ValueError("Parent annotation must belong to the same genome")
        
        # Validate explanation_id if provided
        if explanation_id:
            with db.session_scope() as session:
                explanation = session.get(Explanation, explanation_id)
                if not explanation:
                    raise ValueError(f"Explanation {explanation_id} not found")
                if str(explanation.genome_id) != genome_id:
                    raise ValueError("Explanation must belong to the same genome")
        
        # Create annotation
        with db.session_scope() as session:
            annotation = Annotation(
                genome_id=genome_id,
                name=name,
                hypothesis=hypothesis,
                evidence=evidence,
                entry_nodes=entry_nodes,
                exit_nodes=exit_nodes,
                subgraph_nodes=nodes,
                subgraph_connections=connections,
                is_connected=connectivity_result["is_connected"],
                parent_annotation_id=parent_annotation_id,
                explanation_id=explanation_id,
            )
            session.add(annotation)
            session.commit()
            # Extract ID before session closes
            annotation_id = annotation.id

        # Return a fresh query to avoid detached instance issues
        # Convert to dict while still in session to avoid detached instance errors
        with db.session_scope() as session:
            annotation = session.get(Annotation, annotation_id)
            if annotation:
                return annotation.to_dict()
            return None

    @staticmethod
    def get_annotations(genome_id: str) -> List[Dict[str, Any]]:
        """
        Get all annotations for a genome.

        Args:
            genome_id: UUID of the genome

        Returns:
            List of annotation dictionaries (to avoid detached instance issues)
        """
        with db.session_scope() as session:
            annotations = (
                session.query(Annotation)
                .filter_by(genome_id=genome_id)
                .order_by(Annotation.created_at.desc())
                .all()
            )
            # Convert to dictionaries before session closes to avoid detached instance issues
            return [ann.to_dict() for ann in annotations]

    @staticmethod
    def get_annotation(annotation_id: str) -> Optional[Annotation]:
        """
        Get a specific annotation by ID.

        Args:
            annotation_id: UUID of the annotation

        Returns:
            Annotation object or None if not found
        """
        with db.session_scope() as session:
            return session.get(Annotation, annotation_id)

    @staticmethod
    def update_annotation(
        annotation_id: str,
        hypothesis: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        nodes: Optional[List[int]] = None,
        connections: Optional[List[Tuple[int, int]]] = None,
        entry_nodes: Optional[List[int]] = None,
        exit_nodes: Optional[List[int]] = None,
    ) -> Annotation:
        """
        Update an existing annotation.

        Args:
            annotation_id: UUID of the annotation to update
            hypothesis: Optional new hypothesis
            evidence: Optional new evidence (will be merged if partial)
            name: Optional new name
            nodes: Optional new node list (requires connections if provided)
            connections: Optional new connection list (requires nodes if provided)
            entry_nodes: Optional new entry nodes list
            exit_nodes: Optional new exit nodes list

        Returns:
            Updated Annotation object

        Raises:
            ValueError: If validation fails or annotation not found
        """
        with db.session_scope() as session:
            annotation = session.get(Annotation, annotation_id)
            if not annotation:
                raise ValueError(f"Annotation {annotation_id} not found")

            # Update fields
            if name is not None:
                annotation.name = name
            if hypothesis is not None:
                annotation.hypothesis = hypothesis

            # Update subgraph if provided
            if nodes is not None or connections is not None:
                if nodes is None or connections is None:
                    raise ValueError(
                        "Both nodes and connections must be provided together"
                    )

                # Validate connectivity
                connectivity_result = SubgraphValidator.validate_connectivity(
                    nodes, connections
                )
                if not connectivity_result["is_connected"]:
                    raise ValueError(
                        f"Subgraph is not connected: {connectivity_result.get('error_message', 'Unknown error')}"
                    )

                annotation.subgraph_nodes = nodes
                annotation.subgraph_connections = connections
                annotation.is_connected = connectivity_result["is_connected"]
                
                # Update entry/exit nodes if provided, otherwise keep existing
                if entry_nodes is not None:
                    nodes_set = set(nodes)
                    if not all(n in nodes_set for n in entry_nodes):
                        raise ValueError("All entry nodes must be in the subgraph nodes")
                    annotation.entry_nodes = entry_nodes
                if exit_nodes is not None:
                    nodes_set = set(nodes)
                    if not all(n in nodes_set for n in exit_nodes):
                        raise ValueError("All exit nodes must be in the subgraph nodes")
                    annotation.exit_nodes = exit_nodes

            # Update evidence
            if evidence is not None:
                # Validate evidence structure
                is_valid, error_msg = EvidenceBuilder.validate_evidence(evidence)
                if not is_valid:
                    raise ValueError(f"Invalid evidence structure: {error_msg}")
                annotation.evidence = evidence

            session.commit()
            session.refresh(annotation)
            return annotation

    @staticmethod
    def add_evidence(
        annotation_id: str, evidence_type: str, evidence_data: Dict[str, Any]
    ) -> Annotation:
        """
        Add evidence to an existing annotation.

        Args:
            annotation_id: UUID of the annotation
            evidence_type: Type of evidence ('analytical_method', 'visualization', 'counterfactual', 'other')
            evidence_data: Evidence data dictionary

        Returns:
            Updated Annotation object

        Raises:
            ValueError: If annotation not found or invalid evidence type
        """
        with db.session_scope() as session:
            annotation = session.get(Annotation, annotation_id)
            if not annotation:
                raise ValueError(f"Annotation {annotation_id} not found")

            # Get or create evidence structure
            if annotation.evidence is None:
                evidence = create_empty_evidence()
            else:
                evidence = annotation.evidence.copy()

            # Build evidence using EvidenceBuilder
            builder = EvidenceBuilder.from_dict(evidence)

            # Add evidence based on type
            if evidence_type == "analytical_method":
                builder.add_analytical_method(
                    method=evidence_data.get("method", "unknown"),
                    description=evidence_data.get("description", ""),
                    result=evidence_data.get("result"),
                    metadata=evidence_data.get("metadata"),
                )
            elif evidence_type == "visualization":
                builder.add_visualization(
                    viz_type=evidence_data.get("type", "unknown"),
                    description=evidence_data.get("description", ""),
                    data=evidence_data.get("data"),
                    source_genome_id=evidence_data.get("source_genome_id"),
                    generation=evidence_data.get("generation"),
                    metadata=evidence_data.get("metadata"),
                )
            elif evidence_type == "counterfactual":
                builder.add_counterfactual(
                    description=evidence_data.get("description", ""),
                    scenario=evidence_data.get("scenario", ""),
                    analysis=evidence_data.get("analysis"),
                    metadata=evidence_data.get("metadata"),
                )
            elif evidence_type == "other":
                builder.add_other_evidence(
                    evidence_type=evidence_data.get("type", "unknown"),
                    description=evidence_data.get("description", ""),
                    data=evidence_data.get("data"),
                    metadata=evidence_data.get("metadata"),
                )
            else:
                raise ValueError(f"Invalid evidence type: {evidence_type}")

            annotation.evidence = builder.build()
            session.commit()
            session.refresh(annotation)
            return annotation

    @staticmethod
    def delete_annotation(annotation_id: str) -> bool:
        """
        Delete an annotation.

        Args:
            annotation_id: UUID of the annotation to delete

        Returns:
            True if deleted, False if not found
        """
        with db.session_scope() as session:
            annotation = session.get(Annotation, annotation_id)
            if not annotation:
                return False
            session.delete(annotation)
            session.commit()
            return True

    @staticmethod
    def validate_subgraph(
        genome, nodes: List[int], connections: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Validate a subgraph against a genome.

        Args:
            genome: NEAT genome object
            nodes: List of node IDs
            connections: List of connection tuples

        Returns:
            Validation result dictionary
        """
        # Validate connectivity
        connectivity_result = SubgraphValidator.validate_connectivity(
            nodes, connections
        )

        # Validate against genome
        genome_result = SubgraphValidator.validate_against_genome(
            genome, nodes, connections
        )

        return {
            "connectivity": connectivity_result,
            "genome_validation": genome_result,
            "is_valid": connectivity_result["is_valid"] and genome_result["is_valid"],
        }

    @staticmethod
    def create_composition_annotation(
        genome_id: str,
        child_annotation_ids: List[str],
        hypothesis: str,
        entry_nodes: List[int],
        exit_nodes: List[int],
        nodes: List[int],
        connections: List[Tuple[int, int]],
        evidence: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        explanation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a composition annotation that combines child annotations.
        
        Args:
            genome_id: UUID of the genome
            child_annotation_ids: List of child annotation UUIDs to compose
            hypothesis: Description of how the children combine
            entry_nodes: Entry nodes for the composition (typically exits of children)
            exit_nodes: Exit nodes for the composition
            nodes: Nodes in the junction subgraph
            connections: Connections in the junction subgraph
            evidence: Optional evidence for the composition
            name: Optional name for the composition annotation
            explanation_id: Optional UUID of explanation this belongs to
            
        Returns:
            Dictionary representation of the created composition annotation
        """
        if not child_annotation_ids:
            raise ValueError("At least one child annotation is required")
        
        # Validate all children exist and belong to same genome/explanation
        with db.session_scope() as session:
            children = []
            for child_id in child_annotation_ids:
                child = session.get(Annotation, child_id)
                if not child:
                    raise ValueError(f"Child annotation {child_id} not found")
                if str(child.genome_id) != genome_id:
                    raise ValueError(f"Child annotation {child_id} belongs to different genome")
                if explanation_id and str(child.explanation_id) != explanation_id:
                    raise ValueError(f"Child annotation {child_id} belongs to different explanation")
                children.append(child)
        
        # Create the composition annotation
        composition_ann = AnnotationManager.create_annotation(
            genome_id=genome_id,
            nodes=nodes,
            connections=connections,
            hypothesis=hypothesis,
            entry_nodes=entry_nodes,
            exit_nodes=exit_nodes,
            evidence=evidence,
            name=name,
            explanation_id=explanation_id,
            validate_against_genome=True,
        )
        
        # Set parent_annotation_id on all children
        composition_id = composition_ann["id"]
        with db.session_scope() as session:
            for child_id in child_annotation_ids:
                child = session.get(Annotation, child_id)
                if child:
                    child.parent_annotation_id = composition_id
            session.commit()
        
        return composition_ann
    
    @staticmethod
    def get_annotation_children(annotation_id: str) -> List[Dict[str, Any]]:
        """
        Get direct children of an annotation.
        
        Args:
            annotation_id: UUID of the annotation
            
        Returns:
            List of child annotation dictionaries
        """
        with db.session_scope() as session:
            annotation = session.get(Annotation, annotation_id)
            if not annotation:
                raise ValueError(f"Annotation {annotation_id} not found")
            return [child.to_dict() for child in annotation.children]
    
    @staticmethod
    def get_annotation_parent(annotation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get parent annotation.
        
        Args:
            annotation_id: UUID of the annotation
            
        Returns:
            Parent annotation dictionary or None
        """
        with db.session_scope() as session:
            annotation = session.get(Annotation, annotation_id)
            if not annotation:
                raise ValueError(f"Annotation {annotation_id} not found")
            return annotation.parent.to_dict() if annotation.parent else None
    
    @staticmethod
    def get_leaf_annotations(explanation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all leaf annotations (annotations with no children).
        
        Args:
            explanation_id: Optional UUID of explanation to filter by
            
        Returns:
            List of leaf annotation dictionaries
        """
        with db.session_scope() as session:
            query = session.query(Annotation).filter(Annotation.parent_annotation_id.is_(None))
            if explanation_id:
                query = query.filter(Annotation.explanation_id == explanation_id)
            annotations = query.all()
            # Filter to those with no children
            leaf_annotations = [ann for ann in annotations if len(ann.children) == 0]
            return [ann.to_dict() for ann in leaf_annotations]
    
    @staticmethod
    def get_composition_annotations(explanation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all composition annotations (annotations with children).
        
        Args:
            explanation_id: Optional UUID of explanation to filter by
            
        Returns:
            List of composition annotation dictionaries
        """
        with db.session_scope() as session:
            query = session.query(Annotation)
            if explanation_id:
                query = query.filter(Annotation.explanation_id == explanation_id)
            annotations = query.all()
            # Filter to those with children
            composition_annotations = [ann for ann in annotations if len(ann.children) > 0]
            return [ann.to_dict() for ann in composition_annotations]
