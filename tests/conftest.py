"""
Pytest configuration and shared fixtures for ExplaNEAT tests.

This module provides fixtures for database setup, test data creation,
and other common test utilities.
"""

import os
import tempfile
import pytest
import neat
from typing import Generator, Dict, Any
from datetime import datetime

from explaneat.db import db, Base, Experiment, Population, Genome, Dataset, Annotation, Explanation, NodeSplit
from explaneat.db.serialization import serialize_population_config


@pytest.fixture(scope="function")
def test_db() -> Generator:
    """
    Create an in-memory SQLite database for testing.
    
    This fixture creates a fresh database for each test, ensuring
    complete isolation between tests.
    """
    # Use in-memory SQLite for fast tests
    test_db_url = "sqlite:///:memory:"
    
    # Initialize database with test URL
    db.init_db(test_db_url)
    
    # Create all tables
    db.create_all()
    
    yield db
    
    # Cleanup: drop all tables and close connections
    db.drop_all()
    db.close()


@pytest.fixture(scope="function")
def db_session(test_db):
    """
    Provide a database session scoped to each test.
    
    The session is automatically rolled back after each test
    to ensure no data persists between tests.
    """
    with db.session_scope() as session:
        yield session
        # Session is automatically rolled back by session_scope context manager


@pytest.fixture(scope="function")
def neat_config(tmp_path) -> neat.Config:
    """
    Create a NEAT configuration for testing.
    
    Uses a temporary config file that is cleaned up after the test.
    """
    config_text = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 40.0
pop_size              = 50
reset_on_extinction   = False

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2

[DefaultGenome]
activation_default      = relu
activation_mutate_rate  = 0.0
activation_options      = relu

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.0
bias_replace_rate       = 0.0

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 1.5

conn_add_prob           = 0.7
conn_delete_prob        = 0.65

enabled_default         = True
enabled_mutate_rate     = 0.00

feed_forward            = True
initial_connection      = full

node_add_prob           = 0.6
node_delete_prob        = 0.55

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.00
weight_replace_rate     = 0.005

num_inputs              = 10
num_hidden              = 0
num_outputs             = 1
"""
    
    # Create temporary config file
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    config_file.write(config_text)
    config_file.close()
    
    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file.name
        )
        yield config
    finally:
        # Clean up temp file
        try:
            os.unlink(config_file.name)
        except:
            pass


@pytest.fixture(scope="function")
def test_experiment(db_session, neat_config) -> Experiment:
    """
    Create a test experiment with minimal required fields.
    """
    experiment = Experiment(
        experiment_sha="test_sha_12345",
        name="Test Experiment",
        description="Test experiment for unit tests",
        dataset_name="test_dataset",
        config_json=serialize_population_config(neat_config),
        neat_config_text="# Test config",
        status="completed",
        start_time=datetime.utcnow(),
        random_seed=42
    )
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)
    return experiment


@pytest.fixture(scope="function")
def test_population(db_session, test_experiment, neat_config) -> Population:
    """
    Create a test population for an experiment.
    """
    population = Population(
        experiment_id=test_experiment.id,
        generation=0,
        population_size=10,
        num_species=1,
        best_fitness=1.5,
        mean_fitness=1.0,
        stdev_fitness=0.2,
        config_json=serialize_population_config(neat_config)
    )
    db_session.add(population)
    db_session.commit()
    db_session.refresh(population)
    return population


@pytest.fixture(scope="function")
def test_genome(db_session, test_population, neat_config) -> Genome:
    """
    Create a simple test genome.
    """
    from neat.genes import DefaultNodeGene, DefaultConnectionGene
    
    # Create a simple NEAT genome
    neat_genome = neat.DefaultGenome(1)
    neat_genome.fitness = 1.5
    
    # Add input nodes
    for i in range(-10, 0):
        neat_genome.nodes[i] = DefaultNodeGene(i)
        neat_genome.nodes[i].bias = 0.0
        neat_genome.nodes[i].activation = 'relu'
        neat_genome.nodes[i].aggregation = 'sum'
        neat_genome.nodes[i].response = 1.0
    
    # Add output node
    neat_genome.nodes[0] = DefaultNodeGene(0)
    neat_genome.nodes[0].bias = 0.5
    neat_genome.nodes[0].activation = 'relu'
    neat_genome.nodes[0].aggregation = 'sum'
    neat_genome.nodes[0].response = 1.0
    
    # Add connections from inputs to output
    for i in range(-10, 0):
        conn_key = (i, 0)
        neat_genome.connections[conn_key] = DefaultConnectionGene(conn_key)
        neat_genome.connections[conn_key].weight = 0.5
        neat_genome.connections[conn_key].enabled = True
    
    # Convert to database genome
    db_genome = Genome.from_neat_genome(neat_genome, test_population.id)
    db_session.add(db_genome)
    db_session.commit()
    db_session.refresh(db_genome)
    return db_genome


@pytest.fixture(scope="function")
def test_dataset(db_session) -> Dataset:
    """
    Create a test dataset.
    """
    dataset = Dataset(
        name="test_dataset",
        version="1.0",
        source="test",
        num_samples=100,
        num_features=10,
        num_classes=2,
        feature_names=["feature_" + str(i) for i in range(10)],
        target_name="target"
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


@pytest.fixture(scope="function")
def test_explanation(db_session, test_genome) -> Explanation:
    """
    Create a test explanation for a genome.
    """
    explanation = Explanation(
        genome_id=test_genome.id,
        name="Test Explanation",
        description="Test explanation for unit tests",
        is_well_formed=False
    )
    db_session.add(explanation)
    db_session.commit()
    db_session.refresh(explanation)
    return explanation


@pytest.fixture(scope="function")
def test_annotation(db_session, test_genome, test_explanation) -> Annotation:
    """
    Create a test annotation for a genome.
    """
    annotation = Annotation(
        genome_id=test_genome.id,
        name="Test Annotation",
        hypothesis="This is a test annotation",
        entry_nodes=[-1, -2],
        exit_nodes=[0],
        subgraph_nodes=[-1, -2, 0],
        subgraph_connections=[[-1, 0], [-2, 0]],
        is_connected=True,
        explanation_id=test_explanation.id
    )
    db_session.add(annotation)
    db_session.commit()
    db_session.refresh(annotation)
    return annotation


@pytest.fixture(scope="function")
def test_node_split(db_session, test_genome, test_explanation) -> NodeSplit:
    """
    Create a test node split.
    """
    node_split = NodeSplit(
        genome_id=test_genome.id,
        original_node_id=5,
        split_node_id="5_a",
        outgoing_connections=[[5, 10], [5, 11]],
        explanation_id=test_explanation.id
    )
    db_session.add(node_split)
    db_session.commit()
    db_session.refresh(node_split)
    return node_split


# Pytest markers for test organization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "db: marks tests as database tests")
    config.addinivalue_line("markers", "cli: marks tests as CLI tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
