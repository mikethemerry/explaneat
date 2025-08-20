import uuid
import json
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean,
    ForeignKey, UniqueConstraint, Index, LargeBinary, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class Experiment(Base, TimestampMixin):
    """Stores metadata about each NEAT experiment run"""
    __tablename__ = 'experiments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_sha = Column(String(40), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    dataset_name = Column(String(255), index=True)
    dataset_version = Column(String(50))
    config_json = Column(JSONB, nullable=False)
    neat_config_text = Column(Text, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime(timezone=True))
    status = Column(String(50), default='running')
    git_commit_sha = Column(String(40))
    git_branch = Column(String(255))
    hardware_info = Column(JSONB)
    
    # Relationships
    populations = relationship('Population', back_populates='experiment', cascade='all, delete-orphan')
    checkpoints = relationship('Checkpoint', back_populates='experiment', cascade='all, delete-orphan')
    results = relationship('Result', back_populates='experiment', cascade='all, delete-orphan')
    
    __table_args__ = (
        CheckConstraint("status IN ('running', 'completed', 'failed', 'paused')", name='check_status'),
    )
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'experiment_sha': self.experiment_sha,
            'name': self.name,
            'description': self.description,
            'dataset_name': self.dataset_name,
            'dataset_version': self.dataset_version,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }


class Population(Base, TimestampMixin):
    """Stores population state at each generation"""
    __tablename__ = 'populations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
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
    experiment = relationship('Experiment', back_populates='populations')
    species = relationship('Species', back_populates='population', cascade='all, delete-orphan')
    genomes = relationship('Genome', back_populates='population', cascade='all, delete-orphan')
    checkpoints = relationship('Checkpoint', back_populates='population', cascade='all, delete-orphan')
    training_metrics = relationship('TrainingMetric', back_populates='population', cascade='all, delete-orphan')
    results = relationship('Result', back_populates='population')
    
    __table_args__ = (
        UniqueConstraint('experiment_id', 'generation', name='uq_experiment_generation'),
        Index('idx_populations_fitness', 'best_fitness'),
    )
    
    @hybrid_property
    def best_genome(self):
        """Get the best genome in this population"""
        return max(self.genomes, key=lambda g: g.fitness or float('-inf'), default=None)


class Species(Base, TimestampMixin):
    """Tracks species within populations"""
    __tablename__ = 'species'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    population_id = Column(UUID(as_uuid=True), ForeignKey('populations.id', ondelete='CASCADE'), nullable=False)
    species_id = Column(Integer, nullable=False)  # NEAT's internal species ID
    size = Column(Integer, nullable=False)
    fitness_mean = Column(Float)
    fitness_max = Column(Float)
    fitness_min = Column(Float)
    age = Column(Integer, nullable=False)
    last_improved = Column(Integer, nullable=False)
    representative_genome_id = Column(UUID(as_uuid=True), ForeignKey('genomes.id', ondelete='SET NULL'))
    
    # Relationships
    population = relationship('Population', back_populates='species')
    genomes = relationship('Genome', back_populates='species', foreign_keys='Genome.species_id')
    representative_genome = relationship('Genome', foreign_keys=[representative_genome_id], post_update=True)
    
    __table_args__ = (
        Index('idx_species_internal_id', 'population_id', 'species_id'),
    )


class Genome(Base, TimestampMixin):
    """Stores individual genomes"""
    __tablename__ = 'genomes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    population_id = Column(UUID(as_uuid=True), ForeignKey('populations.id', ondelete='CASCADE'), nullable=False)
    species_id = Column(UUID(as_uuid=True), ForeignKey('species.id', ondelete='SET NULL'))
    genome_id = Column(Integer, nullable=False)  # NEAT's internal genome ID
    fitness = Column(Float, index=True)
    adjusted_fitness = Column(Float)
    genome_data = Column(JSONB, nullable=False)  # Full genome serialization
    parent1_id = Column(UUID(as_uuid=True), ForeignKey('genomes.id'))
    parent2_id = Column(UUID(as_uuid=True), ForeignKey('genomes.id'))
    mutation_history = Column(JSONB)
    network_depth = Column(Integer)
    network_width = Column(Integer)
    num_nodes = Column(Integer)
    num_connections = Column(Integer)
    num_enabled_connections = Column(Integer)
    
    # Relationships
    population = relationship('Population', back_populates='genomes')
    species = relationship('Species', back_populates='genomes', foreign_keys=[species_id])
    parent1 = relationship('Genome', foreign_keys=[parent1_id], remote_side=[id])
    parent2 = relationship('Genome', foreign_keys=[parent2_id], remote_side=[id])
    children_as_parent1 = relationship('Genome', foreign_keys=[parent1_id], back_populates='parent1')
    children_as_parent2 = relationship('Genome', foreign_keys=[parent2_id], back_populates='parent2')
    training_metrics = relationship('TrainingMetric', back_populates='genome', cascade='all, delete-orphan')
    results = relationship('Result', back_populates='genome')
    
    __table_args__ = (
        Index('idx_genomes_parents', 'parent1_id', 'parent2_id'),
    )
    
    def to_neat_genome(self, config):
        """Convert back to a NEAT-Python genome object"""
        from .serialization import deserialize_genome
        return deserialize_genome(self.genome_data, config)
    
    @classmethod
    def from_neat_genome(cls, neat_genome, population_id, species_id=None, parent1_id=None, parent2_id=None):
        """Create from a NEAT-Python genome object"""
        from .serialization import serialize_genome, calculate_genome_stats
        
        genome_data = serialize_genome(neat_genome)
        stats = calculate_genome_stats(neat_genome)
        
        # Handle infinite fitness values
        fitness = neat_genome.fitness
        if fitness == float('inf') or fitness == float('-inf') or (fitness != fitness):  # NaN check
            fitness = None
        
        return cls(
            population_id=population_id,
            species_id=species_id,
            genome_id=neat_genome.key,
            fitness=fitness,
            genome_data=genome_data,
            parent1_id=parent1_id,
            parent2_id=parent2_id,
            **stats
        )


class TrainingMetric(Base, TimestampMixin):
    """Stores detailed training metrics"""
    __tablename__ = 'training_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    genome_id = Column(UUID(as_uuid=True), ForeignKey('genomes.id', ondelete='CASCADE'), nullable=False)
    population_id = Column(UUID(as_uuid=True), ForeignKey('populations.id', ondelete='CASCADE'), nullable=False)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float)
    accuracy = Column(Float)
    validation_loss = Column(Float)
    validation_accuracy = Column(Float)
    backprop_time_seconds = Column(Float)
    additional_metrics = Column(JSONB)
    
    # Relationships
    genome = relationship('Genome', back_populates='training_metrics')
    population = relationship('Population', back_populates='training_metrics')
    
    __table_args__ = (
        Index('idx_metrics_epoch', 'genome_id', 'epoch'),
    )


class Checkpoint(Base, TimestampMixin):
    """Stores full population checkpoints for resumption"""
    __tablename__ = 'checkpoints'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    population_id = Column(UUID(as_uuid=True), ForeignKey('populations.id', ondelete='CASCADE'), nullable=False)
    generation = Column(Integer, nullable=False)
    checkpoint_data = Column(LargeBinary, nullable=False)  # Pickled population state
    file_path = Column(String(500))
    
    # Relationships
    experiment = relationship('Experiment', back_populates='checkpoints')
    population = relationship('Population', back_populates='checkpoints')
    
    __table_args__ = (
        Index('idx_checkpoints_generation', 'experiment_id', 'generation'),
    )


class Result(Base, TimestampMixin):
    """Stores experiment results and measurements"""
    __tablename__ = 'results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    population_id = Column(UUID(as_uuid=True), ForeignKey('populations.id', ondelete='CASCADE'))
    genome_id = Column(UUID(as_uuid=True), ForeignKey('genomes.id', ondelete='CASCADE'))
    measurement_type = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    iteration = Column(Integer)
    params = Column(JSONB)
    
    # Relationships
    experiment = relationship('Experiment', back_populates='results')
    population = relationship('Population', back_populates='results')
    genome = relationship('Genome', back_populates='results')
    
    __table_args__ = (
        Index('idx_results_created', 'created_at'),
    )