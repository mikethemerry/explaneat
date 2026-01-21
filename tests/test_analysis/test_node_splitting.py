"""
Tests for NodeSplitManager class.

Tests node split creation, validation, and querying.
"""

import pytest

from explaneat.analysis.node_splitting import NodeSplitManager
from explaneat.db import db, NodeSplit
from tests.utils import (
    create_test_experiment, create_test_genome, create_test_population,
    create_test_explanation, create_test_annotation, create_test_node_split
)


@pytest.mark.unit
class TestNodeSplitManagerCreate:
    """Test NodeSplitManager.create_split method."""

    def test_create_split(self, db_session, test_genome, test_explanation):
        """Test creating a node split."""
        split_dict = NodeSplitManager.create_split(
            genome_id=str(test_genome.id),
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=str(test_explanation.id)
        )
        
        assert split_dict is not None
        assert split_dict['original_node_id'] == 5
        assert split_dict['split_node_id'] == "5_a"
        assert len(split_dict['outgoing_connections']) == 1

    def test_create_split_auto_explanation(self, db_session, test_genome):
        """Test that split auto-creates explanation if needed."""
        split_dict = NodeSplitManager.create_split(
            genome_id=str(test_genome.id),
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)]
        )
        
        assert split_dict is not None
        assert split_dict['explanation_id'] is not None

    def test_create_split_input_node_fails(self, db_session, test_genome, test_explanation):
        """Test that splitting input nodes raises error."""
        with pytest.raises(ValueError, match="Cannot split input node"):
            NodeSplitManager.create_split(
                genome_id=str(test_genome.id),
                original_node_id=-1,  # Input node
                split_node_id="-1_a",
                outgoing_connections=[(-1, 0)],
                explanation_id=str(test_explanation.id)
            )

    def test_create_split_duplicate_id_fails(self, db_session, test_genome, test_explanation):
        """Test that duplicate split_node_id raises error."""
        create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        with pytest.raises(ValueError, match="already exists"):
            NodeSplitManager.create_split(
                genome_id=str(test_genome.id),
                original_node_id=5,
                split_node_id="5_a",  # Duplicate
                outgoing_connections=[(5, 11)],
                explanation_id=str(test_explanation.id)
            )

    def test_create_split_connection_overlap_fails(self, db_session, test_genome, test_explanation):
        """Test that overlapping connections raise error."""
        create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        with pytest.raises(ValueError, match="already belong"):
            NodeSplitManager.create_split(
                genome_id=str(test_genome.id),
                original_node_id=5,
                split_node_id="5_b",
                outgoing_connections=[(5, 10)],  # Overlapping connection
                explanation_id=str(test_explanation.id)
            )


@pytest.mark.unit
class TestNodeSplitManagerGet:
    """Test NodeSplitManager get methods."""

    def test_get_splits_for_node(self, db_session, test_genome, test_explanation):
        """Test getting splits for a node."""
        split1 = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        split2 = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_b",
            outgoing_connections=[(5, 11)],
            explanation_id=test_explanation.id
        )
        
        splits = NodeSplitManager.get_splits_for_node(
            str(test_genome.id),
            5,
            str(test_explanation.id)
        )
        
        assert len(splits) == 2
        split_ids = [s['id'] for s in splits]
        assert str(split1.id) in split_ids
        assert str(split2.id) in split_ids

    def test_get_splits_for_explanation(self, db_session, test_genome, test_explanation):
        """Test getting splits for an explanation."""
        split1 = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        split2 = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=6,
            split_node_id="6_a",
            outgoing_connections=[(6, 12)],
            explanation_id=test_explanation.id
        )
        
        splits = NodeSplitManager.get_splits_for_explanation(
            explanation_id=str(test_explanation.id)
        )
        
        assert len(splits) == 2
        split_ids = [s['id'] for s in splits]
        assert str(split1.id) in split_ids
        assert str(split2.id) in split_ids

    def test_get_splits_for_genome(self, db_session, test_genome, test_explanation):
        """Test getting all splits for a genome."""
        split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        splits = NodeSplitManager.get_splits_for_genome(str(test_genome.id))
        
        assert len(splits) >= 1
        split_ids = [s['id'] for s in splits]
        assert str(split.id) in split_ids

    def test_get_split_by_id(self, db_session, test_genome, test_explanation):
        """Test getting split by split_node_id."""
        split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        found = NodeSplitManager.get_split_by_id(
            str(test_genome.id),
            "5_a"
        )
        
        assert found is not None
        assert found['id'] == str(split.id)

    def test_get_split_node_connections(self, db_session, test_genome, test_explanation):
        """Test getting connections for a split node."""
        split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10), (5, 11)],
            explanation_id=test_explanation.id
        )
        
        connections = NodeSplitManager.get_split_node_connections(
            str(test_genome.id),
            "5_a"
        )
        
        assert len(connections) == 2
        assert (5, 10) in connections
        assert (5, 11) in connections

    def test_get_original_node_id(self, db_session, test_genome, test_explanation):
        """Test getting original node ID from split node ID."""
        split = create_test_node_split(
            db_session,
            test_genome.id,
            original_node_id=5,
            split_node_id="5_a",
            outgoing_connections=[(5, 10)],
            explanation_id=test_explanation.id
        )
        
        original_id = NodeSplitManager.get_original_node_id(
            str(test_genome.id),
            "5_a"
        )
        
        assert original_id == 5


@pytest.mark.unit
class TestNodeSplitManagerHelpers:
    """Test NodeSplitManager helper methods."""

    def test_format_split_node_display(self):
        """Test formatting split node ID for display."""
        result = NodeSplitManager.format_split_node_display("5_a")
        assert result == "5_a"

    def test_get_original_node_id_from_split(self):
        """Test extracting original node ID from split node ID."""
        original = NodeSplitManager.get_original_node_id_from_split("5_a")
        assert original == 5
        
        original = NodeSplitManager.get_original_node_id_from_split("10_b")
        assert original == 10
        
        # Invalid format
        original = NodeSplitManager.get_original_node_id_from_split("invalid")
        assert original is None
