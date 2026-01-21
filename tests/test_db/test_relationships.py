"""
Tests for database model relationships.

Tests SQLAlchemy relationships, cascade deletes, and relationship navigation.
"""

import pytest

from explaneat.db import (
    db, Experiment, Population, Genome, Species, Annotation,
    Explanation, NodeSplit, Dataset, DatasetSplit
)
from tests.utils import (
    create_test_experiment, create_test_genome, create_test_population,
    create_test_annotation, create_test_explanation, create_test_node_split,
    create_test_dataset
)


@pytest.mark.db
@pytest.mark.unit
class TestExperimentRelationships:
    """Test Experiment model relationships."""

    def test_experiment_populations_relationship(self, db_session, neat_config):
        """Test experiment -> populations relationship."""
        experiment = create_test_experiment(db_session, neat_config=neat_config)
        
        pop1 = create_test_population(
            db_session, experiment.id, generation=0, neat_config=neat_config
        )
        pop2 = create_test_population(
            db_session, experiment.id, generation=1, neat_config=neat_config
        )
        
        db_session.refresh(experiment)
        assert len(experiment.populations) == 2
        assert pop1 in experiment.populations
        assert pop2 in experiment.populations

    def test_experiment_cascade_delete_populations(self, db_session, neat_config):
        """Test that deleting experiment cascades to populations."""
        experiment = create_test_experiment(db_session, neat_config=neat_config)
        pop = create_test_population(
            db_session, experiment.id, neat_config=neat_config
        )
        pop_id = pop.id
        
        db_session.delete(experiment)
        db_session.commit()
        
        # Population should be deleted
        found_pop = db_session.get(Population, pop_id)
        assert found_pop is None

    def test_experiment_dataset_relationship(self, db_session, neat_config):
        """Test experiment -> dataset relationship."""
        dataset = create_test_dataset(db_session)
        experiment = create_test_experiment(db_session, neat_config=neat_config)
        experiment.dataset_id = dataset.id
        db_session.commit()
        
        db_session.refresh(experiment)
        assert experiment.dataset is not None
        assert experiment.dataset.id == dataset.id


@pytest.mark.db
@pytest.mark.unit
class TestPopulationRelationships:
    """Test Population model relationships."""

    def test_population_genomes_relationship(self, db_session, test_population, neat_config):
        """Test population -> genomes relationship."""
        genome1 = create_test_genome(
            db_session, test_population.id, genome_id=1, neat_config=neat_config
        )
        genome2 = create_test_genome(
            db_session, test_population.id, genome_id=2, neat_config=neat_config
        )
        
        db_session.refresh(test_population)
        assert len(test_population.genomes) == 2
        assert genome1 in test_population.genomes
        assert genome2 in test_population.genomes

    def test_population_cascade_delete_genomes(self, db_session, test_population, neat_config):
        """Test that deleting population cascades to genomes."""
        genome = create_test_genome(
            db_session, test_population.id, neat_config=neat_config
        )
        genome_id = genome.id
        
        db_session.delete(test_population)
        db_session.commit()
        
        # Genome should be deleted
        found_genome = db_session.get(Genome, genome_id)
        assert found_genome is None

    def test_population_species_relationship(self, db_session, test_population):
        """Test population -> species relationship."""
        species1 = Species(
            population_id=test_population.id,
            species_id=1,
            size=5,
            age=1,
            last_improved=0
        )
        species2 = Species(
            population_id=test_population.id,
            species_id=2,
            size=3,
            age=1,
            last_improved=0
        )
        db_session.add(species1)
        db_session.add(species2)
        db_session.commit()
        
        db_session.refresh(test_population)
        assert len(test_population.species) == 2


@pytest.mark.db
@pytest.mark.unit
class TestGenomeRelationships:
    """Test Genome model relationships."""

    def test_genome_parent_relationships(self, db_session, test_population, neat_config):
        """Test genome parent1/parent2 relationships."""
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

    def test_genome_children_relationships(self, db_session, test_population, neat_config):
        """Test genome children relationships."""
        parent = create_test_genome(
            db_session, test_population.id, genome_id=1, neat_config=neat_config
        )
        
        child1 = create_test_genome(
            db_session, test_population.id, genome_id=2, neat_config=neat_config
        )
        child1.parent1_id = parent.id
        
        child2 = create_test_genome(
            db_session, test_population.id, genome_id=3, neat_config=neat_config
        )
        child2.parent1_id = parent.id
        
        db_session.commit()
        db_session.refresh(parent)
        
        # Check children via parent1 relationship
        children = [g for g in parent.children_as_parent1]
        assert len(children) == 2
        assert child1 in children
        assert child2 in children

    def test_genome_annotations_relationship(self, db_session, test_genome, test_explanation):
        """Test genome -> annotations relationship."""
        ann1 = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        ann2 = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-2, 0],
            connections=[(-2, 0)],
            explanation_id=test_explanation.id
        )
        
        db_session.refresh(test_genome)
        # Access via backref
        assert len(test_genome.annotations) == 2


@pytest.mark.db
@pytest.mark.unit
class TestAnnotationRelationships:
    """Test Annotation model relationships."""

    def test_annotation_parent_child_relationship(self, db_session, test_genome, test_explanation):
        """Test annotation parent/child hierarchy."""
        parent = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, -2, 0],
            connections=[(-1, 0), (-2, 0)],
            explanation_id=test_explanation.id
        )
        
        child = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            parent_annotation_id=parent.id,
            explanation_id=test_explanation.id
        )
        
        db_session.refresh(parent)
        db_session.refresh(child)
        
        assert child.parent is not None
        assert child.parent.id == parent.id
        assert child in parent.children

    def test_annotation_explanation_relationship(self, db_session, test_genome):
        """Test annotation -> explanation relationship."""
        explanation = create_test_explanation(db_session, test_genome.id)
        annotation = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=explanation.id
        )
        
        db_session.refresh(annotation)
        assert annotation.explanation is not None
        assert annotation.explanation.id == explanation.id
        
        db_session.refresh(explanation)
        assert annotation in explanation.annotations


@pytest.mark.db
@pytest.mark.unit
class TestExplanationRelationships:
    """Test Explanation model relationships."""

    def test_explanation_annotations_relationship(self, db_session, test_genome):
        """Test explanation -> annotations relationship."""
        explanation = create_test_explanation(db_session, test_genome.id)
        
        ann1 = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=explanation.id
        )
        ann2 = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-2, 0],
            connections=[(-2, 0)],
            explanation_id=explanation.id
        )
        
        db_session.refresh(explanation)
        assert len(explanation.annotations) == 2
        assert ann1 in explanation.annotations
        assert ann2 in explanation.annotations

    def test_explanation_node_splits_relationship(self, db_session, test_genome):
        """Test explanation -> node_splits relationship."""
        explanation = create_test_explanation(db_session, test_genome.id)
        
        split1 = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=explanation.id
        )
        split2 = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_b",
            outgoing_connections=[(5, 11)],
            explanation_id=explanation.id
        )
        
        db_session.refresh(explanation)
        assert len(explanation.node_splits) == 2
        assert split1 in explanation.node_splits
        assert split2 in explanation.node_splits

    def test_explanation_cascade_delete_annotations(self, db_session, test_genome):
        """Test that deleting explanation cascades to annotations."""
        explanation = create_test_explanation(db_session, test_genome.id)
        annotation = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=explanation.id
        )
        ann_id = annotation.id
        
        db_session.delete(explanation)
        db_session.commit()
        
        # Annotation should be deleted
        found_ann = db_session.get(Annotation, ann_id)
        assert found_ann is None


@pytest.mark.db
@pytest.mark.unit
class TestNodeSplitRelationships:
    """Test NodeSplit model relationships."""

    def test_node_split_genome_relationship(self, db_session, test_genome, test_explanation):
        """Test node_split -> genome relationship."""
        node_split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        db_session.refresh(node_split)
        assert node_split.genome is not None
        assert node_split.genome.id == test_genome.id

    def test_node_split_explanation_relationship(self, db_session, test_genome, test_explanation):
        """Test node_split -> explanation relationship."""
        node_split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        db_session.refresh(node_split)
        assert node_split.explanation is not None
        assert node_split.explanation.id == test_explanation.id

    def test_node_split_annotation_relationship(self, db_session, test_genome, test_explanation):
        """Test node_split -> annotation relationship."""
        annotation = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        node_split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id,
            annotation_id=annotation.id
        )
        
        db_session.refresh(node_split)
        assert node_split.annotation is not None
        assert node_split.annotation.id == annotation.id
