"""
Comprehensive tests for all database models.

Tests CRUD operations, relationships, and model-specific methods
for all database models in the ExplaNEAT system.
"""

import pytest
import uuid
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from explaneat.db import (
    db, Dataset, DatasetSplit, Experiment, Population, Species,
    Genome, TrainingMetric, Checkpoint, Result, GeneOrigin,
    Annotation, Explanation, NodeSplit
)
from tests.utils import (
    create_test_experiment, create_test_genome, create_test_population,
    create_test_annotation, create_test_explanation, create_test_node_split,
    create_test_dataset
)


@pytest.mark.db
@pytest.mark.unit
class TestDatasetModel:
    """Test Dataset model CRUD operations."""

    def test_create_dataset(self, db_session):
        """Test creating a dataset."""
        dataset = Dataset(
            name="test_dataset",
            version="1.0",
            source="test",
            num_samples=100,
            num_features=10,
            num_classes=2,
            feature_names=["feature_1", "feature_2"],
            target_name="target"
        )
        db_session.add(dataset)
        db_session.commit()
        
        assert dataset.id is not None
        assert dataset.name == "test_dataset"
        assert dataset.created_at is not None

    def test_query_dataset_by_name(self, db_session):
        """Test querying dataset by name."""
        dataset = create_test_dataset(db_session, name="query_test")
        
        found = db_session.query(Dataset).filter_by(name="query_test").first()
        assert found is not None
        assert found.id == dataset.id

    def test_update_dataset(self, db_session):
        """Test updating dataset metadata."""
        dataset = create_test_dataset(db_session)
        original_name = dataset.name
        
        dataset.name = "updated_name"
        dataset.num_samples = 200
        db_session.commit()
        
        db_session.refresh(dataset)
        assert dataset.name == "updated_name"
        assert dataset.num_samples == 200
        assert dataset.name != original_name

    def test_delete_dataset(self, db_session):
        """Test deleting a dataset."""
        dataset = create_test_dataset(db_session)
        dataset_id = dataset.id
        
        db_session.delete(dataset)
        db_session.commit()
        
        found = db_session.get(Dataset, dataset_id)
        assert found is None

    def test_dataset_to_dict(self, db_session):
        """Test dataset to_dict method."""
        dataset = create_test_dataset(db_session)
        data = dataset.to_dict()
        
        assert data['id'] == str(dataset.id)
        assert data['name'] == dataset.name
        assert 'num_samples' in data


@pytest.mark.db
@pytest.mark.unit
class TestExperimentModel:
    """Test Experiment model CRUD operations."""

    def test_create_experiment(self, db_session, neat_config):
        """Test creating an experiment."""
        experiment = create_test_experiment(db_session, neat_config=neat_config)
        
        assert experiment.id is not None
        assert experiment.name == "Test Experiment"
        assert experiment.status == "completed"

    def test_query_experiment_by_sha(self, db_session, neat_config):
        """Test querying experiment by SHA."""
        experiment = create_test_experiment(db_session, neat_config=neat_config)
        
        found = db_session.query(Experiment).filter_by(
            experiment_sha=experiment.experiment_sha
        ).first()
        assert found is not None
        assert found.id == experiment.id

    def test_update_experiment_status(self, db_session, neat_config):
        """Test updating experiment status."""
        experiment = create_test_experiment(
            db_session, status="running", neat_config=neat_config
        )
        
        experiment.status = "completed"
        experiment.end_time = datetime.utcnow()
        db_session.commit()
        
        db_session.refresh(experiment)
        assert experiment.status == "completed"
        assert experiment.end_time is not None

    def test_experiment_to_dict(self, db_session, neat_config):
        """Test experiment to_dict method."""
        experiment = create_test_experiment(db_session, neat_config=neat_config)
        data = experiment.to_dict()
        
        assert data['id'] == str(experiment.id)
        assert data['name'] == experiment.name
        assert data['status'] == experiment.status


@pytest.mark.db
@pytest.mark.unit
class TestPopulationModel:
    """Test Population model CRUD operations."""

    def test_create_population(self, db_session, test_experiment, neat_config):
        """Test creating a population."""
        population = create_test_population(
            db_session, test_experiment.id, neat_config=neat_config
        )
        
        assert population.id is not None
        assert population.generation == 0
        assert population.experiment_id == test_experiment.id

    def test_query_population_by_generation(self, db_session, test_experiment, neat_config):
        """Test querying population by generation."""
        population = create_test_population(
            db_session, test_experiment.id, generation=5, neat_config=neat_config
        )
        
        found = db_session.query(Population).filter_by(
            experiment_id=test_experiment.id,
            generation=5
        ).first()
        assert found is not None
        assert found.id == population.id

    def test_population_best_genome_property(self, db_session, test_population, neat_config):
        """Test population best_genome hybrid property."""
        # Create genomes with different fitnesses
        genome1 = create_test_genome(
            db_session, test_population.id, fitness=1.0, neat_config=neat_config
        )
        genome2 = create_test_genome(
            db_session, test_population.id, fitness=2.0, neat_config=neat_config
        )
        genome3 = create_test_genome(
            db_session, test_population.id, fitness=1.5, neat_config=neat_config
        )
        
        db_session.refresh(test_population)
        best = test_population.best_genome
        
        assert best is not None
        assert best.fitness == 2.0
        assert best.id == genome2.id

    def test_population_unique_constraint(self, db_session, test_experiment, neat_config):
        """Test population unique constraint on experiment_id + generation."""
        create_test_population(
            db_session, test_experiment.id, generation=0, neat_config=neat_config
        )
        
        # Try to create duplicate
        with pytest.raises(IntegrityError):
            create_test_population(
                db_session, test_experiment.id, generation=0, neat_config=neat_config
            )
            db_session.commit()


@pytest.mark.db
@pytest.mark.unit
class TestGenomeModel:
    """Test Genome model CRUD operations."""

    def test_create_genome_from_neat(self, db_session, test_population, neat_config):
        """Test creating genome from NEAT genome."""
        genome = create_test_genome(
            db_session, test_population.id, neat_config=neat_config
        )
        
        assert genome.id is not None
        assert genome.genome_id == 1
        assert genome.fitness == 1.5
        assert genome.population_id == test_population.id

    def test_genome_to_neat_genome(self, db_session, test_genome, neat_config):
        """Test converting genome back to NEAT genome."""
        neat_genome = test_genome.to_neat_genome(neat_config)
        
        assert neat_genome is not None
        assert neat_genome.key == test_genome.genome_id
        assert neat_genome.fitness == test_genome.fitness

    def test_genome_parent_relationships(self, db_session, test_population, neat_config):
        """Test genome parent/child relationships."""
        parent1 = create_test_genome(
            db_session, test_population.id, genome_id=1, neat_config=neat_config
        )
        parent2 = create_test_genome(
            db_session, test_population.id, genome_id=2, neat_config=neat_config
        )
        
        child = create_test_genome(
            db_session, test_population.id, genome_id=3, neat_config=neat_config
        )
        child.parent1_id = parent1.id
        child.parent2_id = parent2.id
        db_session.commit()
        
        db_session.refresh(child)
        assert child.parent1.id == parent1.id
        assert child.parent2.id == parent2.id
        assert parent1 in child.parent1.children_as_parent1 or parent1 in child.parent2.children_as_parent2

    def test_query_genome_by_fitness(self, db_session, test_population, neat_config):
        """Test querying genomes by fitness."""
        genome1 = create_test_genome(
            db_session, test_population.id, fitness=1.0, neat_config=neat_config
        )
        genome2 = create_test_genome(
            db_session, test_population.id, fitness=2.0, neat_config=neat_config
        )
        
        high_fitness = db_session.query(Genome).filter(
            Genome.fitness >= 1.5
        ).all()
        
        assert len(high_fitness) == 1
        assert high_fitness[0].id == genome2.id


@pytest.mark.db
@pytest.mark.unit
class TestAnnotationModel:
    """Test Annotation model CRUD operations."""

    def test_create_annotation(self, db_session, test_genome, test_explanation):
        """Test creating an annotation."""
        annotation = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, -2, 0],
            connections=[(-1, 0), (-2, 0)],
            explanation_id=test_explanation.id
        )
        
        assert annotation.id is not None
        assert annotation.genome_id == test_genome.id
        assert len(annotation.subgraph_nodes) == 3

    def test_annotation_hierarchy_methods(self, db_session, test_genome, test_explanation):
        """Test annotation hierarchy methods."""
        parent = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        child = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1],
            connections=[],
            parent_annotation_id=parent.id,
            explanation_id=test_explanation.id
        )
        
        db_session.refresh(parent)
        db_session.refresh(child)
        
        assert child.is_leaf() is True
        assert parent.is_composition() is True
        assert len(parent.get_children()) == 1
        assert child in parent.get_children()

    def test_annotation_to_dict(self, db_session, test_genome, test_explanation):
        """Test annotation to_dict method."""
        annotation = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        data = annotation.to_dict()
        assert data['id'] == str(annotation.id)
        assert data['genome_id'] == str(annotation.genome_id)
        assert 'subgraph_nodes' in data


@pytest.mark.db
@pytest.mark.unit
class TestExplanationModel:
    """Test Explanation model CRUD operations."""

    def test_create_explanation(self, db_session, test_genome):
        """Test creating an explanation."""
        explanation = create_test_explanation(db_session, test_genome.id)
        
        assert explanation.id is not None
        assert explanation.genome_id == test_genome.id
        assert explanation.is_well_formed is False

    def test_explanation_to_dict(self, db_session, test_genome):
        """Test explanation to_dict method."""
        explanation = create_test_explanation(db_session, test_genome.id)
        data = explanation.to_dict()
        
        assert data['id'] == str(explanation.id)
        assert data['genome_id'] == str(explanation.genome_id)
        assert 'is_well_formed' in data

    def test_explanation_relationships(self, db_session, test_genome):
        """Test explanation relationships with annotations and splits."""
        explanation = create_test_explanation(db_session, test_genome.id)
        
        annotation = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=explanation.id
        )
        
        node_split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=explanation.id
        )
        
        db_session.refresh(explanation)
        assert len(explanation.annotations) == 1
        assert len(explanation.node_splits) == 1


@pytest.mark.db
@pytest.mark.unit
class TestNodeSplitModel:
    """Test NodeSplit model CRUD operations."""

    def test_create_node_split(self, db_session, test_genome, test_explanation):
        """Test creating a node split."""
        node_split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10), (5, 11)],
            explanation_id=test_explanation.id
        )
        
        assert node_split.id is not None
        assert node_split.original_node_id == 5
        assert node_split.split_node_id == "5_a"

    def test_node_split_get_outgoing_connections(self, db_session, test_genome, test_explanation):
        """Test get_outgoing_connections method."""
        node_split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10), (5, 11)],
            explanation_id=test_explanation.id
        )
        
        connections = node_split.get_outgoing_connections()
        assert len(connections) == 2
        assert (5, 10) in connections
        assert (5, 11) in connections

    def test_node_split_to_dict(self, db_session, test_genome, test_explanation):
        """Test node_split to_dict method."""
        node_split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        data = node_split.to_dict()
        assert data['id'] == str(node_split.id)
        assert data['original_node_id'] == 5
        assert data['split_node_id'] == "5_a"


@pytest.mark.db
@pytest.mark.unit
class TestOtherModels:
    """Test other database models."""

    def test_species_model(self, db_session, test_population):
        """Test Species model."""
        species = Species(
            population_id=test_population.id,
            species_id=1,
            size=5,
            fitness_mean=1.5,
            fitness_max=2.0,
            fitness_min=1.0,
            age=1,
            last_improved=0
        )
        db_session.add(species)
        db_session.commit()
        
        assert species.id is not None
        assert species.population_id == test_population.id

    def test_training_metric_model(self, db_session, test_genome, test_population):
        """Test TrainingMetric model."""
        metric = TrainingMetric(
            genome_id=test_genome.id,
            population_id=test_population.id,
            epoch=1,
            loss=0.5,
            accuracy=0.9
        )
        db_session.add(metric)
        db_session.commit()
        
        assert metric.id is not None
        assert metric.genome_id == test_genome.id

    def test_result_model(self, db_session, test_experiment, test_population, test_genome):
        """Test Result model."""
        result = Result(
            experiment_id=test_experiment.id,
            population_id=test_population.id,
            genome_id=test_genome.id,
            measurement_type="accuracy",
            value=0.95
        )
        db_session.add(result)
        db_session.commit()
        
        assert result.id is not None
        assert result.value == 0.95

    def test_gene_origin_model(self, db_session, test_experiment, test_genome):
        """Test GeneOrigin model."""
        gene_origin = GeneOrigin(
            experiment_id=test_experiment.id,
            innovation_number=1,
            gene_type="connection",
            origin_genome_id=test_genome.id,
            origin_generation=0,
            connection_from=-1,
            connection_to=0
        )
        db_session.add(gene_origin)
        db_session.commit()
        
        assert gene_origin.id is not None
        assert gene_origin.innovation_number == 1
