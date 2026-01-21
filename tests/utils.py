"""
Test utility functions for creating test data.

These functions provide convenient ways to create test data structures
for use in tests, reducing boilerplate code.
"""

import uuid
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene

from explaneat.db import (
    db, Experiment, Population, Genome, Dataset, Annotation, 
    Explanation, NodeSplit, Species, TrainingMetric
)
from explaneat.db.serialization import serialize_population_config


def create_test_experiment(
    session,
    name: str = "Test Experiment",
    status: str = "completed",
    neat_config: Optional[neat.Config] = None,
) -> Experiment:
    """
    Create a test experiment with minimal required fields.
    
    Args:
        session: Database session
        name: Experiment name
        status: Experiment status
        neat_config: Optional NEAT config (will create minimal if None)
    
    Returns:
        Created Experiment instance
    """
    if neat_config is None:
        # Create minimal config
        import tempfile
        import os
        
        config_text = """
[NEAT]
fitness_criterion = max
pop_size = 50

[DefaultGenome]
num_inputs = 10
num_outputs = 1
"""
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
        config_file.write(config_text)
        config_file.close()
        
        try:
            neat_config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_file.name
            )
        finally:
            try:
                os.unlink(config_file.name)
            except:
                pass
    
    experiment = Experiment(
        experiment_sha=f"test_sha_{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Test experiment: {name}",
        dataset_name="test_dataset",
        config_json=serialize_population_config(neat_config),
        neat_config_text="# Test config",
        status=status,
        start_time=datetime.utcnow(),
        random_seed=42
    )
    session.add(experiment)
    session.flush()
    return experiment


def create_test_genome(
    session,
    population_id: uuid.UUID,
    genome_id: int = 1,
    fitness: float = 1.5,
    num_inputs: int = 10,
    num_outputs: int = 1,
    add_connections: bool = True,
    neat_config: Optional[neat.Config] = None,
) -> Genome:
    """
    Create a test genome with specified structure.
    
    Args:
        session: Database session
        population_id: Population ID to attach genome to
        genome_id: NEAT genome ID
        fitness: Genome fitness value
        num_inputs: Number of input nodes
        num_outputs: Number of output nodes
        add_connections: Whether to add connections from inputs to outputs
        neat_config: Optional NEAT config (needed for deserialization)
    
    Returns:
        Created Genome instance
    """
    # Create a simple NEAT genome
    neat_genome = neat.DefaultGenome(genome_id)
    neat_genome.fitness = fitness
    
    # Add input nodes (negative IDs)
    for i in range(-num_inputs, 0):
        neat_genome.nodes[i] = DefaultNodeGene(i)
        neat_genome.nodes[i].bias = 0.0
        neat_genome.nodes[i].activation = 'relu'
        neat_genome.nodes[i].aggregation = 'sum'
        neat_genome.nodes[i].response = 1.0
    
    # Add output nodes (0 to num_outputs-1)
    for i in range(num_outputs):
        neat_genome.nodes[i] = DefaultNodeGene(i)
        neat_genome.nodes[i].bias = 0.5
        neat_genome.nodes[i].activation = 'relu'
        neat_genome.nodes[i].aggregation = 'sum'
        neat_genome.nodes[i].response = 1.0
    
    # Add connections from inputs to outputs if requested
    if add_connections:
        for input_id in range(-num_inputs, 0):
            for output_id in range(num_outputs):
                conn_key = (input_id, output_id)
                neat_genome.connections[conn_key] = DefaultConnectionGene(conn_key)
                neat_genome.connections[conn_key].weight = 0.5
                neat_genome.connections[conn_key].enabled = True
    
    # Convert to database genome
    db_genome = Genome.from_neat_genome(neat_genome, population_id)
    session.add(db_genome)
    session.flush()
    return db_genome


def create_test_population(
    session,
    experiment_id: uuid.UUID,
    generation: int = 0,
    population_size: int = 10,
    neat_config: Optional[neat.Config] = None,
) -> Population:
    """
    Create a test population for an experiment.
    
    Args:
        session: Database session
        experiment_id: Experiment ID to attach population to
        generation: Generation number
        population_size: Size of population
        neat_config: Optional NEAT config
    
    Returns:
        Created Population instance
    """
    if neat_config is None:
        # Create minimal config
        import tempfile
        import os
        
        config_text = """
[NEAT]
pop_size = 50
"""
        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
        config_file.write(config_text)
        config_file.close()
        
        try:
            neat_config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_file.name
            )
        finally:
            try:
                os.unlink(config_file.name)
            except:
                pass
    
    population = Population(
        experiment_id=experiment_id,
        generation=generation,
        population_size=population_size,
        num_species=1,
        best_fitness=1.5,
        mean_fitness=1.0,
        stdev_fitness=0.2,
        config_json=serialize_population_config(neat_config)
    )
    session.add(population)
    session.flush()
    return population


def create_test_annotation(
    session,
    genome_id: uuid.UUID,
    nodes: List[int],
    connections: List[Tuple[int, int]],
    hypothesis: str = "Test annotation",
    name: Optional[str] = None,
    entry_nodes: Optional[List[Any]] = None,  # Can contain int node IDs or str split_node_ids like "5_a"
    exit_nodes: Optional[List[Any]] = None,  # Can contain int node IDs or str split_node_ids like "5_a"
    explanation_id: Optional[uuid.UUID] = None,
) -> Annotation:
    """
    Create a test annotation for a genome.
    
    Args:
        session: Database session
        genome_id: Genome ID to annotate
        nodes: List of node IDs in subgraph
        connections: List of connection tuples (from_node, to_node)
        hypothesis: Annotation hypothesis
        name: Optional annotation name
        entry_nodes: Optional entry nodes (defaults to nodes if None)
        exit_nodes: Optional exit nodes (defaults to nodes if None)
        explanation_id: Optional explanation ID
    
    Returns:
        Created Annotation instance
    """
    if entry_nodes is None:
        entry_nodes = nodes
    if exit_nodes is None:
        exit_nodes = nodes
    
    annotation = Annotation(
        genome_id=genome_id,
        name=name or f"Test Annotation {uuid.uuid4().hex[:8]}",
        hypothesis=hypothesis,
        entry_nodes=entry_nodes,
        exit_nodes=exit_nodes,
        subgraph_nodes=nodes,
        subgraph_connections=connections,
        is_connected=True,
        explanation_id=explanation_id
    )
    session.add(annotation)
    session.flush()
    return annotation


def create_test_explanation(
    session,
    genome_id: uuid.UUID,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Explanation:
    """
    Create a test explanation for a genome.
    
    Args:
        session: Database session
        genome_id: Genome ID to explain
        name: Optional explanation name
        description: Optional description
    
    Returns:
        Created Explanation instance
    """
    explanation = Explanation(
        genome_id=genome_id,
        name=name or f"Test Explanation {uuid.uuid4().hex[:8]}",
        description=description or "Test explanation",
        is_well_formed=False
    )
    session.add(explanation)
    session.flush()
    return explanation


def create_test_node_split(
    session,
    genome_id: uuid.UUID,
    original_node_id: int,
    split_node_id: str,
    outgoing_connections: List[Tuple[int, int]],
    explanation_id: Optional[uuid.UUID] = None,
    annotation_id: Optional[uuid.UUID] = None,
) -> NodeSplit:
    """
    Create a test node split.
    
    Args:
        session: Database session
        genome_id: Genome ID
        original_node_id: Original node ID being split
        split_node_id: Split node ID (e.g., "5_a")
        outgoing_connections: List of outgoing connections for this split
        explanation_id: Optional explanation ID
        annotation_id: Optional annotation ID
    
    Returns:
        Created NodeSplit instance
    """
    node_split = NodeSplit(
        genome_id=genome_id,
        original_node_id=original_node_id,
        split_node_id=split_node_id,
        outgoing_connections=outgoing_connections,
        explanation_id=explanation_id,
        annotation_id=annotation_id
    )
    session.add(node_split)
    session.flush()
    return node_split


def create_test_dataset(
    session,
    name: str = "test_dataset",
    num_samples: int = 100,
    num_features: int = 10,
) -> Dataset:
    """
    Create a test dataset.
    
    Args:
        session: Database session
        name: Dataset name
        num_samples: Number of samples
        num_features: Number of features
    
    Returns:
        Created Dataset instance
    """
    dataset = Dataset(
        name=name,
        version="1.0",
        source="test",
        num_samples=num_samples,
        num_features=num_features,
        num_classes=2,
        feature_names=[f"feature_{i}" for i in range(num_features)],
        target_name="target"
    )
    session.add(dataset)
    session.flush()
    return dataset
