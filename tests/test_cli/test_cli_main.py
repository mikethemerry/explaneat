"""
Tests for CLI main entry point.

Tests command-line argument parsing and main function execution.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

from genome_explorer_cli import main


@pytest.mark.cli
@pytest.mark.unit
class TestCLIMain:
    """Test CLI main function."""

    @patch('genome_explorer_cli.db.init_db')
    @patch('genome_explorer_cli.GenomeExplorerCLI')
    def test_main_list_command(self, mock_cli_class, mock_init_db):
        """Test main with --list argument."""
        mock_cli = MagicMock()
        mock_cli_class.return_value = mock_cli
        mock_cli.list_experiments.return_value = (MagicMock(), 1)
        
        with patch.object(sys, 'argv', ['genome_explorer_cli.py', '--list']):
            main()
        
        mock_init_db.assert_called_once()
        mock_cli.list_experiments.assert_called_once()

    @patch('genome_explorer_cli.db.init_db')
    @patch('genome_explorer_cli.GenomeExplorerCLI')
    def test_main_interactive(self, mock_cli_class, mock_init_db):
        """Test main with --interactive argument."""
        mock_cli = MagicMock()
        mock_cli_class.return_value = mock_cli
        
        with patch.object(sys, 'argv', ['genome_explorer_cli.py', '--interactive']):
            main()
        
        mock_init_db.assert_called_once()
        mock_cli.interactive_mode.assert_called_once()

    @patch('genome_explorer_cli.db.init_db')
    def test_main_db_init_failure(self, mock_init_db):
        """Test main when database initialization fails."""
        mock_init_db.side_effect = Exception("DB error")
        
        with patch.object(sys, 'argv', ['genome_explorer_cli.py', '--list']):
            with pytest.raises(SystemExit):
                main()
