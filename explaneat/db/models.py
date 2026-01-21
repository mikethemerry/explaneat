import uuid
import json
from datetime import datetime
from typing import List, Tuple
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Index,
    LargeBinary,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Dataset(Base, TimestampMixin):
    """Stores dataset metadata and information"""

    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50))
    source = Column(String(255))  # e.g., 'PMLB', 'sklearn', 'custom'
    source_url = Column(String(500))
    description = Column(Text)

    # Dataset statistics
    num_samples = Column(Integer)
    num_features = Column(Integer)
    num_classes = Column(Integer)
    feature_names = Column(JSONB)  # List of feature names
    feature_descriptions = Column(JSONB)  # Dict mapping feature names to descriptions
    feature_types = Column(
        JSONB
    )  # Dict mapping feature names to types (e.g., 'numeric', 'categorical')
    target_name = Column(String(255))
    target_description = Column(Text)
    class_names = Column(JSONB)  # For classification tasks

    # Dataset metadata
    additional_metadata = Column(
        JSONB
    )  # Additional metadata (e.g., license, citation, etc.)

    # Relationships
    splits = relationship(
        "DatasetSplit", back_populates="dataset", cascade="all, delete-orphan"
    )
    experiments = relationship("Experiment", back_populates="dataset")

    __table_args__ = (Index("idx_datasets_name_version", "name", "version"),)

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
        }


class DatasetSplit(Base, TimestampMixin):
    """Stores train/test split information for reproducibility"""

    __tablename__ = "dataset_splits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    experiment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Split parameters for reproducibility
    split_type = Column(
        String(50), nullable=False
    )  # e.g., 'train_test', 'k_fold', 'custom'
    test_size = Column(Float)  # Proportion of test set (for train_test_split)
    random_state = Column(Integer)  # Random seed for reproducibility
    shuffle = Column(Boolean, default=True)
    stratify = Column(Boolean, default=False)  # Whether stratification was used

    # Split indices stored as arrays for exact reproducibility
    train_indices = Column(JSONB, nullable=False)  # List of indices for training set
    test_indices = Column(JSONB, nullable=False)  # List of indices for test set
    validation_indices = Column(JSONB)  # Optional validation set indices

    # Preprocessing information
    scaler_type = Column(String(50))  # e.g., 'StandardScaler', 'MinMaxScaler', None
    scaler_params = Column(
        JSONB
    )  # Scaler parameters (mean, std, etc.) for reproducibility
    preprocessing_steps = Column(JSONB)  # List of preprocessing steps applied

    # Split statistics
    train_size = Column(Integer)
    test_size_actual = Column(Integer)
    validation_size = Column(Integer)

    # Relationships
    dataset = relationship("Dataset", back_populates="splits")
    experiment = relationship("Experiment", back_populates="dataset_split")

    __table_args__ = (
        Index("idx_splits_experiment", "experiment_id"),
        Index("idx_splits_dataset", "dataset_id"),
    )


class Experiment(Base, TimestampMixin):
    """Stores metadata about each NEAT experiment run"""

    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_sha = Column(String(40), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    dataset_name = Column(String(255), index=True)  # Deprecated: use dataset_id instead
    dataset_version = Column(String(50))  # Deprecated: use dataset_id instead
    dataset_id = Column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    config_json = Column(JSONB, nullable=False)
    neat_config_text = Column(Text, nullable=False)
    start_time = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    end_time = Column(DateTime(timezone=True))
    status = Column(String(50), default="running")
    git_commit_sha = Column(String(40))
    git_branch = Column(String(255))
    hardware_info = Column(JSONB)

    # Random seed for experiment reproducibility
    random_seed = Column(Integer)

    # Relationships
    dataset = relationship("Dataset", back_populates="experiments")
    dataset_split = relationship(
        "DatasetSplit", back_populates="experiment", uselist=False
    )
    populations = relationship(
        "Population", back_populates="experiment", cascade="all, delete-orphan"
    )
    checkpoints = relationship(
        "Checkpoint", back_populates="experiment", cascade="all, delete-orphan"
    )
    results = relationship(
        "Result", back_populates="experiment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'paused')",
            name="check_status",
        ),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "experiment_sha": self.experiment_sha,
            "name": self.name,
            "description": self.description,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_id": str(self.dataset_id) if self.dataset_id else None,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "random_seed": self.random_seed,
        }


class Population(Base, TimestampMixin):
    """Stores population state at each generation"""

    __tablename__ = "populations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    generation = Column(Integer, nullable=False)
    population_size = Column(Integer, nullable=False)
    num_species = Column(Integer, nullable=False)
    best_fitness = Column(Float)
    mean_fitness = Column(Float)
    stdev_fitness = Column(Float)
    config_json = Column(JSONB, nullable=False)
    generation_time_seconds = Column(Float)
    backprop_time_seconds = Column(Float)
    improvement_count = Column(Integer)

    # Relationships
    experiment = relationship("Experiment", back_populates="populations")
    species = relationship(
        "Species", back_populates="population", cascade="all, delete-orphan"
    )
    genomes = relationship(
        "Genome", back_populates="population", cascade="all, delete-orphan"
    )
    checkpoints = relationship(
        "Checkpoint", back_populates="population", cascade="all, delete-orphan"
    )
    training_metrics = relationship(
        "TrainingMetric", back_populates="population", cascade="all, delete-orphan"
    )
    results = relationship("Result", back_populates="population")

    __table_args__ = (
        UniqueConstraint(
            "experiment_id", "generation", name="uq_experiment_generation"
        ),
        Index("idx_populations_fitness", "best_fitness"),
    )

    @hybrid_property
    def best_genome(self):
        """Get the best genome in this population"""
        return max(self.genomes, key=lambda g: g.fitness or float("-inf"), default=None)


class Species(Base, TimestampMixin):
    """Tracks species within populations"""

    __tablename__ = "species"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    population_id = Column(
        UUID(as_uuid=True),
        ForeignKey("populations.id", ondelete="CASCADE"),
        nullable=False,
    )
    species_id = Column(Integer, nullable=False)  # NEAT's internal species ID
    size = Column(Integer, nullable=False)
    fitness_mean = Column(Float)
    fitness_max = Column(Float)
    fitness_min = Column(Float)
    age = Column(Integer, nullable=False)
    last_improved = Column(Integer, nullable=False)
    representative_genome_id = Column(
        UUID(as_uuid=True), ForeignKey("genomes.id", ondelete="SET NULL")
    )

    # Relationships
    population = relationship("Population", back_populates="species")
    genomes = relationship(
        "Genome", back_populates="species", foreign_keys="Genome.species_id"
    )
    representative_genome = relationship(
        "Genome", foreign_keys=[representative_genome_id], post_update=True
    )

    __table_args__ = (Index("idx_species_internal_id", "population_id", "species_id"),)


class Genome(Base, TimestampMixin):
    """Stores individual genomes"""

    __tablename__ = "genomes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    population_id = Column(
        UUID(as_uuid=True),
        ForeignKey("populations.id", ondelete="CASCADE"),
        nullable=False,
    )
    species_id = Column(
        UUID(as_uuid=True), ForeignKey("species.id", ondelete="SET NULL")
    )
    genome_id = Column(Integer, nullable=False)  # NEAT's internal genome ID
    fitness = Column(Float, index=True)
    adjusted_fitness = Column(Float)
    genome_data = Column(JSONB, nullable=False)  # Full genome serialization
    parent1_id = Column(UUID(as_uuid=True), ForeignKey("genomes.id"))
    parent2_id = Column(UUID(as_uuid=True), ForeignKey("genomes.id"))
    mutation_history = Column(JSONB)
    network_depth = Column(Integer)
    network_width = Column(Integer)
    num_nodes = Column(Integer)
    num_connections = Column(Integer)
    num_enabled_connections = Column(Integer)

    # Relationships
    population = relationship("Population", back_populates="genomes")
    species = relationship(
        "Species", back_populates="genomes", foreign_keys=[species_id]
    )
    parent1 = relationship("Genome", foreign_keys=[parent1_id], remote_side=[id])
    parent2 = relationship("Genome", foreign_keys=[parent2_id], remote_side=[id])
    children_as_parent1 = relationship(
        "Genome", foreign_keys=[parent1_id], back_populates="parent1"
    )
    children_as_parent2 = relationship(
        "Genome", foreign_keys=[parent2_id], back_populates="parent2"
    )
    training_metrics = relationship(
        "TrainingMetric", back_populates="genome", cascade="all, delete-orphan"
    )
    results = relationship("Result", back_populates="genome")

    __table_args__ = (Index("idx_genomes_parents", "parent1_id", "parent2_id"),)

    def to_neat_genome(self, config):
        """Convert back to a NEAT-Python genome object"""
        from .serialization import deserialize_genome

        return deserialize_genome(self.genome_data, config)

    @classmethod
    def from_neat_genome(
        cls,
        neat_genome,
        population_id,
        species_id=None,
        parent1_id=None,
        parent2_id=None,
    ):
        """Create from a NEAT-Python genome object"""
        from .serialization import serialize_genome, calculate_genome_stats

        genome_data = serialize_genome(neat_genome)
        stats = calculate_genome_stats(neat_genome)

        # Handle infinite fitness values
        fitness = neat_genome.fitness
        if (
            fitness == float("inf") or fitness == float("-inf") or (fitness != fitness)
        ):  # NaN check
            fitness = None

        return cls(
            population_id=population_id,
            species_id=species_id,
            genome_id=neat_genome.key,
            fitness=fitness,
            genome_data=genome_data,
            parent1_id=parent1_id,
            parent2_id=parent2_id,
            **stats,
        )


class TrainingMetric(Base, TimestampMixin):
    """Stores detailed training metrics"""

    __tablename__ = "training_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    genome_id = Column(
        UUID(as_uuid=True), ForeignKey("genomes.id", ondelete="CASCADE"), nullable=False
    )
    population_id = Column(
        UUID(as_uuid=True),
        ForeignKey("populations.id", ondelete="CASCADE"),
        nullable=False,
    )
    epoch = Column(Integer, nullable=False)
    loss = Column(Float)
    accuracy = Column(Float)
    validation_loss = Column(Float)
    validation_accuracy = Column(Float)
    backprop_time_seconds = Column(Float)
    additional_metrics = Column(JSONB)

    # Relationships
    genome = relationship("Genome", back_populates="training_metrics")
    population = relationship("Population", back_populates="training_metrics")

    __table_args__ = (Index("idx_metrics_epoch", "genome_id", "epoch"),)


class Checkpoint(Base, TimestampMixin):
    """Stores full population checkpoints for resumption"""

    __tablename__ = "checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    population_id = Column(
        UUID(as_uuid=True),
        ForeignKey("populations.id", ondelete="CASCADE"),
        nullable=False,
    )
    generation = Column(Integer, nullable=False)
    checkpoint_data = Column(LargeBinary, nullable=False)  # Pickled population state
    file_path = Column(String(500))

    # Relationships
    experiment = relationship("Experiment", back_populates="checkpoints")
    population = relationship("Population", back_populates="checkpoints")

    __table_args__ = (
        Index("idx_checkpoints_generation", "experiment_id", "generation"),
    )


class Result(Base, TimestampMixin):
    """Stores experiment results and measurements"""

    __tablename__ = "results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    population_id = Column(
        UUID(as_uuid=True), ForeignKey("populations.id", ondelete="CASCADE")
    )
    genome_id = Column(UUID(as_uuid=True), ForeignKey("genomes.id", ondelete="CASCADE"))
    measurement_type = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    iteration = Column(Integer)
    params = Column(JSONB)

    # Relationships
    experiment = relationship("Experiment", back_populates="results")
    population = relationship("Population", back_populates="results")
    genome = relationship("Genome", back_populates="results")

    __table_args__ = (Index("idx_results_created", "created_at"),)


class GeneOrigin(Base, TimestampMixin):
    """Tracks when each gene (innovation number) first appeared in evolution"""

    __tablename__ = "gene_origins"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    innovation_number = Column(Integer, nullable=False, index=True)
    gene_type = Column(String(20), nullable=False)  # 'node' or 'connection'
    origin_genome_id = Column(
        UUID(as_uuid=True), ForeignKey("genomes.id", ondelete="CASCADE"), nullable=False
    )
    origin_generation = Column(Integer, nullable=False, index=True)

    # For connections, store the connection details
    connection_from = Column(Integer)  # input node ID
    connection_to = Column(Integer)  # output node ID

    # For nodes, store node details
    node_id = Column(Integer)

    # Initial gene parameters when first introduced
    initial_params = Column(JSONB)  # Store initial weight/bias/activation etc.

    # Relationships
    experiment = relationship("Experiment", backref="gene_origins")
    origin_genome = relationship("Genome", backref="originated_genes")

    __table_args__ = (
        UniqueConstraint(
            "innovation_number",
            "gene_type",
            "experiment_id",
            name="uq_innovation_type_experiment",
        ),
        Index("idx_gene_origin_type", "gene_type", "innovation_number"),
        Index("idx_gene_origin_generation", "origin_generation"),
    )


class Explanation(Base, TimestampMixin):
    """Groups annotations and splits into a coherent explanation of a model"""

    __tablename__ = "explanations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    genome_id = Column(
        UUID(as_uuid=True),
        ForeignKey("genomes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(255))  # Optional name for this explanation
    description = Column(Text)  # Optional description
    is_well_formed = Column(
        Boolean, nullable=False, default=False
    )  # Whether this explanation meets well-formed criteria
    structural_coverage = Column(Float)  # Cached C_V^struct value
    compositional_coverage = Column(Float)  # Cached C_V^comp value

    # Operations event stream stored as JSON array
    # Format: [{"seq": 0, "type": "split_node", "params": {...}, "result": {...}, "created_at": "..."}]
    operations = Column(JSONB, nullable=False, default=list)

    # Relationships
    genome = relationship("Genome", backref="explanations")
    annotations = relationship(
        "Annotation", back_populates="explanation", cascade="all, delete-orphan"
    )
    node_splits = relationship(
        "NodeSplit", back_populates="explanation", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_explanations_genome", "genome_id"),)

    def to_dict(self):
        """Convert explanation to dictionary"""
        return {
            "id": str(self.id),
            "genome_id": str(self.genome_id),
            "name": self.name,
            "description": self.description,
            "is_well_formed": self.is_well_formed,
            "structural_coverage": self.structural_coverage,
            "compositional_coverage": self.compositional_coverage,
            "operations": self.operations or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class NodeSplit(Base, TimestampMixin):
    """Stores node splits for dual-function nodes.
    
    Each row represents a complete split of one original node.
    The split_mappings JSONB contains all split nodes for this original node.
    
    split_mappings format: {"5_a": [[5, 10]], "5_b": [[5, 20]], "5_c": [[5, 30]]}
    Maps split_node_id (string like "5_a") -> list of outgoing connections.
    With full splitting, each split node has exactly one outgoing connection.
    """

    __tablename__ = "node_splits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    genome_id = Column(
        UUID(as_uuid=True),
        ForeignKey("genomes.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_node_id = Column(String(50), nullable=False)  # The node being split (string to support split nodes connecting to split nodes)
    split_mappings = Column(
        JSONB, nullable=False
    )  # Maps split_node_id (str) -> list of outgoing connections: {"5_a": [[5, 10]], "5_b": [[5, 20]]}
    annotation_id = Column(
        UUID(as_uuid=True), ForeignKey("annotations.id", ondelete="SET NULL"), nullable=True
    )  # Which annotation uses this split (for tracking)
    explanation_id = Column(
        UUID(as_uuid=True), ForeignKey("explanations.id", ondelete="SET NULL"), nullable=True
    )  # Which explanation this split belongs to

    # Relationships
    genome = relationship("Genome", backref="node_splits")
    annotation = relationship("Annotation", backref="node_splits")
    explanation = relationship("Explanation", back_populates="node_splits")

    __table_args__ = (
        Index("idx_node_splits_genome_original", "genome_id", "original_node_id"),
        Index("idx_node_splits_explanation", "explanation_id"),
        Index("idx_node_splits_unique", "genome_id", "original_node_id", "explanation_id", unique=True),
    )

    def to_dict(self):
        """Convert node split to dictionary"""
        return {
            "id": str(self.id),
            "genome_id": str(self.genome_id),
            "original_node_id": self.original_node_id,
            "split_mappings": self.split_mappings,
            "annotation_id": str(self.annotation_id) if self.annotation_id else None,
            "explanation_id": str(self.explanation_id) if self.explanation_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def get_split_node_ids(self) -> List[str]:
        """Get list of all split_node_ids for this split"""
        if not self.split_mappings:
            return []
        return list(self.split_mappings.keys())

    def get_outgoing_connections_for_split(self, split_node_id: str) -> List[Tuple[int, int]]:
        """Get outgoing connections for a specific split node"""
        if not self.split_mappings or split_node_id not in self.split_mappings:
            return []
        conns = self.split_mappings[split_node_id]
        return [
            tuple(conn) if isinstance(conn, (list, tuple)) else conn
            for conn in conns
        ]

    def get_all_outgoing_connections(self) -> List[Tuple[int, int]]:
        """Get all outgoing connections across all split nodes"""
        all_conns = []
        if not self.split_mappings:
            return all_conns
        for split_node_id, conns in self.split_mappings.items():
            all_conns.extend([
                tuple(conn) if isinstance(conn, (list, tuple)) else conn
                for conn in conns
            ])
        return all_conns


class Annotation(Base, TimestampMixin):
    """Stores annotations for connected subgraphs of genomes"""

    __tablename__ = "annotations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    genome_id = Column(
        UUID(as_uuid=True),
        ForeignKey("genomes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(255))  # Optional name/title for the annotation
    hypothesis = Column(Text, nullable=False)  # Description of what the subgraph does
    evidence = Column(
        JSONB
    )  # Structured evidence: analytical_methods, visualizations, counterfactuals, other_evidence
    entry_nodes = Column(JSONB, nullable=False)  # Array of entry node IDs (integers or string split_node_ids like "5_a")
    exit_nodes = Column(JSONB, nullable=False)  # Array of exit node IDs (integers or string split_node_ids like "5_a")
    subgraph_nodes = Column(JSONB, nullable=False)  # Array of all node IDs in the subgraph (integers or string split_node_ids like "5_a")
    subgraph_connections = Column(
        JSONB, nullable=False
    )  # Array of connection tuples [from_node, to_node] (nodes can be integers or string split_node_ids)
    is_connected = Column(
        Boolean, nullable=False, default=False
    )  # Validated connectivity flag
    parent_annotation_id = Column(
        UUID(as_uuid=True), ForeignKey("annotations.id", ondelete="SET NULL"), nullable=True
    )  # Parent annotation in hierarchy (for composition annotations)
    explanation_id = Column(
        UUID(as_uuid=True), ForeignKey("explanations.id", ondelete="SET NULL"), nullable=True
    )  # Which explanation this annotation belongs to

    # Relationships
    genome = relationship("Genome", backref="annotations")
    parent = relationship("Annotation", remote_side=[id], backref="children")
    explanation = relationship("Explanation", back_populates="annotations")

    __table_args__ = (
        Index("idx_annotations_genome", "genome_id"),
        Index("idx_annotations_parent", "parent_annotation_id"),
        Index("idx_annotations_explanation", "explanation_id"),
    )

    def is_leaf(self) -> bool:
        """Check if this is a leaf annotation (no children)"""
        return self.parent_annotation_id is None and len(self.children) == 0

    def is_composition(self) -> bool:
        """Check if this is a composition annotation (has children)"""
        return len(self.children) > 0

    def get_children(self):
        """Get direct children annotations"""
        return self.children

    def get_descendants(self):
        """Get all descendant annotations (children, grandchildren, etc.)"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def get_parent(self):
        """Get parent annotation"""
        return self.parent

    def get_explanation(self):
        """Get the explanation this annotation belongs to"""
        return self.explanation

    def to_dict(self):
        """Convert annotation to dictionary"""
        return {
            "id": str(self.id),
            "genome_id": str(self.genome_id),
            "name": self.name,
            "hypothesis": self.hypothesis,
            "evidence": self.evidence,
            "entry_nodes": self.entry_nodes,
            "exit_nodes": self.exit_nodes,
            "subgraph_nodes": self.subgraph_nodes,
            "subgraph_connections": self.subgraph_connections,
            "is_connected": self.is_connected,
            "parent_annotation_id": str(self.parent_annotation_id) if self.parent_annotation_id else None,
            "explanation_id": str(self.explanation_id) if self.explanation_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
