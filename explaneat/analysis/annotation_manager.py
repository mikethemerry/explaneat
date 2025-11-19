"""
Annotation Manager for creating and managing genome annotations

Provides CRUD operations for annotations with validation.
"""

from typing import List, Tuple, Dict, Any, Optional

from ..db import db, Annotation, Genome
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
        evidence: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        validate_against_genome: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a new annotation for a genome.

        Args:
            genome_id: UUID of the genome to annotate
            nodes: List of node IDs in the subgraph
            connections: List of connection tuples (from_node, to_node)
            hypothesis: Description of what the subgraph does
            evidence: Optional evidence dictionary (will use empty if None)
            name: Optional name/title for the annotation
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

        # Create annotation
        with db.session_scope() as session:
            annotation = Annotation(
                genome_id=genome_id,
                name=name,
                hypothesis=hypothesis,
                evidence=evidence,
                subgraph_nodes=nodes,
                subgraph_connections=connections,
                is_connected=connectivity_result["is_connected"],
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
