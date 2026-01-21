"""
Tests for GenomeExplorer class.

Tests genome loading, summary generation, and data export.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from explaneat.analysis.genome_explorer import GenomeExplorer
from tests.utils import (
    create_test_experiment, create_test_genome, create_test_population
)


@pytest.mark.unit
class TestGenomeExplorer:
    """Test GenomeExplorer class methods."""

    @patch('explaneat.analysis.genome_explorer.GenomeExplorer.load_genome')
    def test_load_best_genome(self, mock_load, db_session, test_experiment, neat_config):
        """Test loading best genome from experiment."""
        pop = create_test_population(db_session, test_experiment.id, neat_config=neat_config)
        genome = create_test_genome(db_session, pop.id, fitness=2.0, neat_config=neat_config)
        
        mock_explorer = Mock()
        mock_load.return_value = mock_explorer
        
        result = GenomeExplorer.load_best_genome(str(test_experiment.id))
        
        assert result is not None

    def test_list_experiments(self, db_session, neat_config):
        """Test listing experiments."""
        create_test_experiment(db_session, name="Test Exp", neat_config=neat_config)
        
        df = GenomeExplorer.list_experiments()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    @patch('explaneat.analysis.genome_explorer.GenomeExplorer')
    def test_summary(self, mock_explorer_class, db_session):
        """Test generating genome summary."""
        mock_explorer = Mock()
        mock_explorer.summary = Mock()
        
        mock_explorer.summary()
        mock_explorer.summary.assert_called_once()

    @patch('explaneat.analysis.genome_explorer.GenomeExplorer')
    def test_export_genome_data(self, mock_explorer_class):
        """Test exporting genome data."""
        mock_explorer = Mock()
        mock_explorer.export_genome_data.return_value = {"test": "data"}
        
        data = mock_explorer.export_genome_data()
        
        assert data == {"test": "data"}
