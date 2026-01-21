"""
Tests for GenomeExplorerCLI class.

Tests CLI methods for experiment management, genome display, and annotation operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from genome_explorer_cli import GenomeExplorerCLI
from explaneat.analysis.genome_explorer import GenomeExplorer
from tests.utils import (
    create_test_experiment, create_test_genome, create_test_population,
    create_test_explanation, create_test_annotation
)


@pytest.mark.cli
@pytest.mark.unit
class TestGenomeExplorerCLIExperimentManagement:
    """Test CLI experiment management methods."""

    def test_list_experiments_empty(self, db_session, neat_config):
        """Test listing experiments when none exist."""
        cli = GenomeExplorerCLI()
        
        df, page = cli.list_experiments(show_numbers=True, page=1, page_size=20, interactive=False)
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert page == 1

    def test_list_experiments_with_data(self, db_session, neat_config):
        """Test listing experiments with data."""
        exp1 = create_test_experiment(db_session, name="Exp 1", neat_config=neat_config)
        exp2 = create_test_experiment(db_session, name="Exp 2", neat_config=neat_config)
        
        cli = GenomeExplorerCLI()
        df, page = cli.list_experiments(show_numbers=True, page=1, page_size=20, interactive=False)
        
        assert not df.empty
        assert len(df) >= 2
        assert page == 1

    def test_list_experiments_pagination(self, db_session, neat_config):
        """Test experiment listing pagination."""
        # Create multiple experiments
        for i in range(25):
            create_test_experiment(db_session, name=f"Exp {i}", neat_config=neat_config)
        
        cli = GenomeExplorerCLI()
        df, page = cli.list_experiments(page=1, page_size=10, interactive=False)
        
        assert len(df) == 10
        assert page == 1

    def test_select_experiment_by_uuid(self, db_session, neat_config):
        """Test selecting experiment by UUID."""
        exp = create_test_experiment(db_session, neat_config=neat_config)
        pop = create_test_population(db_session, exp.id, neat_config=neat_config)
        create_test_genome(db_session, pop.id, fitness=2.0, neat_config=neat_config)
        
        cli = GenomeExplorerCLI()
        result = cli.select_experiment(str(exp.id))
        
        assert result is True
        assert cli.current_experiment_id == str(exp.id)
        assert cli.current_explorer is not None

    def test_select_experiment_by_index(self, db_session, neat_config):
        """Test selecting experiment by index."""
        exp = create_test_experiment(db_session, neat_config=neat_config)
        pop = create_test_population(db_session, exp.id, neat_config=neat_config)
        create_test_genome(db_session, pop.id, fitness=2.0, neat_config=neat_config)
        
        cli = GenomeExplorerCLI()
        result = cli.select_experiment("0")  # First experiment
        
        assert result is True

    def test_select_experiment_latest(self, db_session, neat_config):
        """Test selecting latest experiment."""
        exp1 = create_test_experiment(db_session, name="Old", neat_config=neat_config)
        exp2 = create_test_experiment(db_session, name="New", neat_config=neat_config)
        pop2 = create_test_population(db_session, exp2.id, neat_config=neat_config)
        create_test_genome(db_session, pop2.id, fitness=2.0, neat_config=neat_config)
        
        cli = GenomeExplorerCLI()
        result = cli.select_experiment("latest")
        
        assert result is True
        # Should select the most recent one
        assert cli.current_experiment_id == str(exp2.id)

    def test_select_experiment_not_found(self, db_session):
        """Test selecting non-existent experiment."""
        cli = GenomeExplorerCLI()
        result = cli.select_experiment("nonexistent-uuid")
        
        assert result is False


@pytest.mark.cli
@pytest.mark.unit
class TestGenomeExplorerCLIHelpers:
    """Test CLI helper methods."""

    def test_parse_node_list_comma_separated(self):
        """Test parsing comma-separated node list."""
        nodes = GenomeExplorerCLI.parse_node_list("-1,-2,0")
        assert nodes == [-1, -2, 0]

    def test_parse_node_list_space_separated(self):
        """Test parsing space-separated node list."""
        nodes = GenomeExplorerCLI.parse_node_list("-1 -2 0")
        assert nodes == [-1, -2, 0]

    def test_parse_node_list_mixed(self):
        """Test parsing mixed format node list."""
        nodes = GenomeExplorerCLI.parse_node_list("-1, -2, 0")
        assert nodes == [-1, -2, 0]

    def test_parse_node_list_invalid(self):
        """Test parsing invalid node list."""
        with pytest.raises(ValueError):
            GenomeExplorerCLI.parse_node_list("invalid")

    def test_parse_annotate_command(self):
        """Test parsing annotate command."""
        start, end, hypothesis, name = GenomeExplorerCLI.parse_annotate_command(
            ["annotate", "-1,-2", "0", "Test hypothesis", "Test Name"]
        )
        
        assert start == "-1,-2"
        assert end == "0"
        assert hypothesis == "Test hypothesis"
        assert name == "Test Name"

    def test_parse_annotate_command_no_name(self):
        """Test parsing annotate command without name."""
        start, end, hypothesis, name = GenomeExplorerCLI.parse_annotate_command(
            ["annotate", "-1", "0", "Test hypothesis"]
        )
        
        assert start == "-1"
        assert end == "0"
        assert hypothesis == "Test hypothesis"
        assert name is None


@pytest.mark.cli
@pytest.mark.unit
class TestGenomeExplorerCLIAnnotation:
    """Test CLI annotation methods."""

    @patch('genome_explorer_cli.SubgraphValidator')
    @patch('genome_explorer_cli.AnnotationManager')
    def test_create_annotation(self, mock_ann_mgr, mock_validator, db_session, test_genome, neat_config):
        """Test creating annotation via CLI."""
        # Setup mocks
        mock_validator.find_reachable_subgraph.return_value = {
            "is_valid": True,
            "nodes": [-1, -2, 0],
            "connections": [(-1, 0), (-2, 0)]
        }
        mock_ann_mgr.create_annotation.return_value = {
            "id": "test-ann-id",
            "name": "Test Annotation"
        }
        
        # Create explorer
        pop = create_test_population(db_session, test_genome.population_id, neat_config=neat_config)
        exp = create_test_experiment(db_session, neat_config=neat_config)
        
        cli = GenomeExplorerCLI()
        cli.current_explorer = Mock()
        cli.current_explorer.genome_info.genome_id = test_genome.id
        cli.current_explorer.neat_genome = Mock()
        cli.current_explorer.config = neat_config
        
        result = cli.create_annotation("-1,-2", "0", "Test hypothesis", "Test Name")
        
        assert result is not None
        assert result["id"] == "test-ann-id"

    def test_list_annotations(self, db_session, test_genome, test_explanation):
        """Test listing annotations."""
        ann1 = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        cli = GenomeExplorerCLI()
        cli.current_explorer = Mock()
        cli.current_explorer.genome_info.genome_id = test_genome.id
        
        # Mock AnnotationManager.get_annotations
        with patch('genome_explorer_cli.AnnotationManager.get_annotations') as mock_get:
            mock_get.return_value = [ann1.to_dict()]
            cli.list_annotations()
            mock_get.assert_called_once_with(str(test_genome.id))

    def test_show_annotation_by_index(self, db_session, test_genome, test_explanation):
        """Test showing annotation by index."""
        ann = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            name="Test Annotation",
            explanation_id=test_explanation.id
        )
        
        cli = GenomeExplorerCLI()
        cli.current_explorer = Mock()
        cli.current_explorer.genome_info.genome_id = test_genome.id
        
        # Mock AnnotationManager methods
        with patch('genome_explorer_cli.AnnotationManager.get_annotations') as mock_get_anns:
            mock_get_anns.return_value = [ann.to_dict()]
            cli.show_annotation("0")  # First annotation
            mock_get_anns.assert_called_once()

    def test_delete_annotation(self, db_session, test_genome, test_explanation):
        """Test deleting annotation."""
        ann = create_test_annotation(
            db_session,
            test_genome.id,
            nodes=[-1, 0],
            connections=[(-1, 0)],
            explanation_id=test_explanation.id
        )
        
        cli = GenomeExplorerCLI()
        cli.current_explorer = Mock()
        cli.current_explorer.genome_info.genome_id = test_genome.id
        
        with patch('genome_explorer_cli.AnnotationManager.delete_annotation') as mock_delete:
            mock_delete.return_value = True
            cli.delete_annotation(str(ann.id))
            mock_delete.assert_called_once_with(str(ann.id))


@pytest.mark.cli
@pytest.mark.unit
class TestGenomeExplorerCLIExplanation:
    """Test CLI explanation methods."""

    def test_create_explanation(self, db_session, test_genome):
        """Test creating explanation via CLI."""
        cli = GenomeExplorerCLI()
        cli.current_explorer = Mock()
        cli.current_explorer.genome_info.genome_id = test_genome.id
        
        with patch('genome_explorer_cli.ExplanationManager.create_explanation') as mock_create:
            mock_create.return_value = {"id": "test-exp-id"}
            cli.create_explanation("Test Explanation", "Test description")
            mock_create.assert_called_once()

    def test_list_explanations(self, db_session, test_genome, test_explanation):
        """Test listing explanations."""
        cli = GenomeExplorerCLI()
        cli.current_explorer = Mock()
        cli.current_explorer.genome_info.genome_id = test_genome.id
        
        with patch('genome_explorer_cli.ExplanationManager.get_explanations') as mock_get:
            mock_get.return_value = [test_explanation.to_dict()]
            cli.list_explanations()
            mock_get.assert_called_once()

    def test_show_coverage_metrics(self, db_session, test_genome, test_explanation):
        """Test showing coverage metrics."""
        cli = GenomeExplorerCLI()
        cli.current_explorer = Mock()
        cli.current_explorer.genome_info.genome_id = test_genome.id
        
        with patch('genome_explorer_cli.ExplanationManager.get_or_create_explanation') as mock_get:
            mock_get.return_value = {
                "id": str(test_explanation.id),
                "structural_coverage": 0.8,
                "compositional_coverage": 0.9
            }
            cli.show_coverage_metrics()
            mock_get.assert_called_once()
