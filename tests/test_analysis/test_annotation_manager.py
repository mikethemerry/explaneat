"""
Tests for AnnotationManager class.

Tests CRUD operations, validation, and hierarchy management for annotations.
"""

import pytest

from explaneat.analysis.annotation_manager import AnnotationManager
from explaneat.db import db, Annotation, Genome
from tests.utils import (
    create_test_experiment, create_test_genome, create_test_population,
    create_test_explanation, create_test_annotation
)


@pytest.mark.unit
class TestAnnotationManagerCreate:
    """Test AnnotationManager.create_annotation method."""

    def test_create_annotation_basic(self, db_session, test_genome, neat_config):
        """Test creating a basic annotation."""
        annotation_dict = AnnotationManager.create_annotation(
            genome_id=str(test_genome.id),
            nodes=[-1, -2, 0],
            connections=[(-1, 0), (-2, 0)],
            hypothesis="Test annotation",
            entry_nodes=[-1, -2],
            exit_nodes=[0],
            validate_against_genome=False  # Skip genome validation for test
        )
        
        assert annotation_dict is not None
        assert annotation_dict['genome_id'] == str(test_genome.id)
        assert annotation_dict['hypothesis'] == "Test annotation"
        assert len(annotation_dict['subgraph_nodes']) == 3

    def test_create_annotation_with_name(self, db_session, test_genome):
        """Test creating annotation with name."""
        annotation_dict = AnnotationManager.create_annotation(
            genome_id=str(test_genome.id),
            nodes=[-1, 0],
            connections=[(-1, 0)],
            hypothesis="Named annotation",
            name="Test Annotation",
            validate_against_genome=False
        )
        
        assert annotation_dict['name'] == "Test Annotation"

    def test_create_annotation_auto_explanation(self, db_session, test_genome):
        """Test that annotation auto-creates explanation if needed."""
        annotation_dict = AnnotationManager.create_annotation(
            genome_id=str(test_genome.id),
            nodes=[-1, 0],
            connections=[(-1, 0)],
            hypothesis="Auto explanation test",
            validate_against_genome=False
        )
        
        assert annotation_dict['explanation_id'] is not None

    def test_create_annotation_with_explanation(self, db_session, test_genome, test_explanation):
        """Test creating annotation with existing explanation."""
        annotation_dict = AnnotationManager.create_annotation(
            genome_id=str(test_genome.id),
            nodes=[-1, 0],
            connections=[(-1, 0)],
            hypothesis="With explanation",
            explanation_id=str(test_explanation.id),
            validate_against_genome=False
        )
        
        assert annotation_dict['explanation_id'] == str(test_explanation.id)

    def test_create_annotation_validation_fails_disconnected(self, db_session, test_genome):
        """Test that disconnected subgraph raises error."""
        with pytest.raises(ValueError, match="not connected"):
            AnnotationManager.create_annotation(
                genome_id=str(test_genome.id),
                nodes=[-1, -2, 0],
                connections=[(-1, 0)],  # Missing connection to -2
                hypothesis="Disconnected",
                validate_against_genome=False
            )

    def test_create_annotation_parent_validation(self, db_session, test_genome, test_explanation):
        """Test parent annotation validation."""
        parent = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, -2, 0],
            connections=[(-1, 0), (-2, 0)],
            explanation_id=test_explanation.id
        )
        
        child = AnnotationManager.create_annotation(
            genome_id=str(test_genome.id),
            nodes=[-1, 0],
            connections=[(-1, 0)],
            hypothesis="Child annotation",
            parent_annotation_id=str(parent.id),
            explanation_id=str(test_explanation.id),
            validate_against_genome=False
        )
        
        assert child['parent_annotation_id'] == str(parent.id)

    def test_create_annotation_wrong_parent_genome(self, db_session, test_genome, neat_config):
        """Test that parent from different genome raises error."""
        # Create another experiment/genome
        exp2 = create_test_experiment(db_session, name="Exp 2", neat_config=neat_config)
        pop2 = create_test_population(db_session, exp2.id, neat_config=neat_config)
        genome2 = create_test_genome(db_session, pop2.id, neat_config=neat_config)
        expl2 = create_test_explanation(db_session, genome2.id)
        
        parent = create_test_annotation(
            db_session,
            genome2.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=expl2.id
        )
        
        with pytest.raises(ValueError, match="same genome"):
            AnnotationManager.create_annotation(
                genome_id=str(test_genome.id),
                nodes=[-1, 0],
                connections=[(-1, 0)],
                hypothesis="Wrong parent",
                parent_annotation_id=str(parent.id),
                validate_against_genome=False
            )


@pytest.mark.unit
class TestAnnotationManagerGet:
    """Test AnnotationManager get methods."""

    def test_get_annotations(self, db_session, test_genome, test_explanation):
        """Test getting all annotations for a genome."""
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
        
        annotations = AnnotationManager.get_annotations(str(test_genome.id))
        
        assert len(annotations) >= 2
        ann_ids = [ann['id'] for ann in annotations]
        assert str(ann1.id) in ann_ids
        assert str(ann2.id) in ann_ids

    def test_get_annotation_by_id(self, db_session, test_genome, test_explanation):
        """Test getting annotation by ID."""
        ann = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        retrieved = AnnotationManager.get_annotation(str(ann.id))
        
        assert retrieved is not None
        assert retrieved.id == ann.id
        assert retrieved.hypothesis == ann.hypothesis

    def test_get_annotation_not_found(self, db_session):
        """Test getting non-existent annotation."""
        import uuid
        fake_id = str(uuid.uuid4())
        retrieved = AnnotationManager.get_annotation(fake_id)
        assert retrieved is None


@pytest.mark.unit
class TestAnnotationManagerUpdate:
    """Test AnnotationManager.update_annotation method."""

    def test_update_annotation_hypothesis(self, db_session, test_genome, test_explanation):
        """Test updating annotation hypothesis."""
        ann = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        updated = AnnotationManager.update_annotation(
            str(ann.id),
            hypothesis="Updated hypothesis"
        )
        
        assert updated.hypothesis == "Updated hypothesis"

    def test_update_annotation_name(self, db_session, test_genome, test_explanation):
        """Test updating annotation name."""
        ann = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        updated = AnnotationManager.update_annotation(
            str(ann.id),
            name="Updated Name"
        )
        
        assert updated.name == "Updated Name"

    def test_update_annotation_subgraph(self, db_session, test_genome, test_explanation):
        """Test updating annotation subgraph."""
        ann = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        updated = AnnotationManager.update_annotation(
            str(ann.id),
            nodes=[-1, -2, 0],
            connections=[(-1, 0), (-2, 0)],
            entry_nodes=[-1, -2],
            exit_nodes=[0],
            validate_against_genome=False
        )
        
        assert len(updated.subgraph_nodes) == 3
        assert len(updated.subgraph_connections) == 2

    def test_update_annotation_not_found(self, db_session):
        """Test updating non-existent annotation."""
        import uuid
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(ValueError, match="not found"):
            AnnotationManager.update_annotation(fake_id, hypothesis="Test")


@pytest.mark.unit
class TestAnnotationManagerDelete:
    """Test AnnotationManager.delete_annotation method."""

    def test_delete_annotation(self, db_session, test_genome, test_explanation):
        """Test deleting an annotation."""
        ann = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        ann_id = ann.id
        
        result = AnnotationManager.delete_annotation(str(ann_id))
        
        assert result is True
        
        # Verify deleted
        with db.session_scope() as session:
            found = session.get(Annotation, ann_id)
            assert found is None

    def test_delete_annotation_not_found(self, db_session):
        """Test deleting non-existent annotation."""
        import uuid
        fake_id = str(uuid.uuid4())
        
        result = AnnotationManager.delete_annotation(fake_id)
        assert result is False


@pytest.mark.unit
class TestAnnotationManagerHierarchy:
    """Test annotation hierarchy methods."""

    def test_get_annotation_children(self, db_session, test_genome, test_explanation):
        """Test getting annotation children."""
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
        
        children = AnnotationManager.get_annotation_children(str(parent.id))
        
        assert len(children) == 1
        assert children[0]['id'] == str(child.id)

    def test_get_annotation_parent(self, db_session, test_genome, test_explanation):
        """Test getting annotation parent."""
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
        
        parent_dict = AnnotationManager.get_annotation_parent(str(child.id))
        
        assert parent_dict is not None
        assert parent_dict['id'] == str(parent.id)

    def test_get_leaf_annotations(self, db_session, test_genome, test_explanation):
        """Test getting leaf annotations."""
        parent = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, -2, 0],
            connections=[(-1, 0), (-2, 0)],
            explanation_id=test_explanation.id
        )
        
        leaf = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            parent_annotation_id=parent.id,
            explanation_id=test_explanation.id
        )
        
        leaves = AnnotationManager.get_leaf_annotations(str(test_explanation.id))
        
        # Should include the leaf but not the parent
        leaf_ids = [ann['id'] for ann in leaves]
        assert str(leaf.id) in leaf_ids
        assert str(parent.id) not in leaf_ids

    def test_get_composition_annotations(self, db_session, test_genome, test_explanation):
        """Test getting composition annotations."""
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
        
        compositions = AnnotationManager.get_composition_annotations(str(test_explanation.id))
        
        # Should include parent but not child
        comp_ids = [ann['id'] for ann in compositions]
        assert str(parent.id) in comp_ids
        assert str(child.id) not in comp_ids
