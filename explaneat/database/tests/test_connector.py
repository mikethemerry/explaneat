"""
Tests for NEATDatabaseConnector.

These tests focus on the interface and behavior without requiring a live database connection.
"""

import unittest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from explaneat.database.connector import NEATDatabaseConnector


class MockGenome:
    """Mock genome object for testing."""
    def __init__(self, key, fitness=1.0):
        self.key = key
        self.fitness = fitness
        self.adjusted_fitness = fitness * 0.9
        
        # Create mock nodes with proper structure
        mock_node1 = Mock()
        mock_node1.type = 'input'
        mock_node1.bias = 0.0
        mock_node1.response = 1.0
        mock_node1.activation = 'sigmoid'
        mock_node1.aggregation = 'sum'
        
        mock_node2 = Mock()
        mock_node2.type = 'output'
        mock_node2.bias = 0.5
        mock_node2.response = 1.0
        mock_node2.activation = 'sigmoid'
        mock_node2.aggregation = 'sum'
        
        self.nodes = {1: mock_node1, 2: mock_node2}
        
        # Create mock connections with proper structure  
        mock_connection = Mock()
        mock_connection.weight = 0.5
        mock_connection.enabled = True
        mock_connection.in_node_id = 1
        mock_connection.out_node_id = 2
        mock_connection.innovation = 1
        
        self.connections = {(1, 2): mock_connection}


class TestNEATDatabaseConnector(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Test with disabled database to avoid connection issues
        self.connector = NEATDatabaseConnector(enabled=False)
    
    def test_init_disabled(self):
        """Test initialization with disabled database."""
        connector = NEATDatabaseConnector(enabled=False)
        self.assertFalse(connector.enabled)
        self.assertIsNone(connector.pool)
    
    def test_init_missing_psycopg2(self):
        """Test initialization when psycopg2 is not available."""
        with patch('explaneat.database.connector.PSYCOPG2_AVAILABLE', False):
            connector = NEATDatabaseConnector(enabled=True)
            self.assertFalse(connector.enabled)
    
    def test_connection_string_from_env(self):
        """Test connection string construction from environment variables."""
        env_vars = {
            'POSTGRES_HOST': 'testhost',
            'POSTGRES_PORT': '5433',
            'POSTGRES_DB': 'testdb',
            'POSTGRES_USER': 'testuser',
            'POSTGRES_PASSWORD': 'testpass'
        }
        
        with patch.dict(os.environ, env_vars):
            connector = NEATDatabaseConnector(enabled=False)
            expected = "postgresql://testuser:testpass@testhost:5433/testdb"
            self.assertEqual(connector._get_connection_string(), expected)
    
    def test_connection_string_from_database_url(self):
        """Test connection string from DATABASE_URL environment variable."""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@host:5432/db'}):
            connector = NEATDatabaseConnector(enabled=False)
            self.assertEqual(connector._get_connection_string(), 'postgresql://user:pass@host:5432/db')
    
    def test_disabled_operations_return_none_or_false(self):
        """Test that operations return None/False when database is disabled."""
        connector = NEATDatabaseConnector(enabled=False)
        
        self.assertIsNone(connector.create_experiment("test", {}))
        self.assertFalse(connector.save_generation(1, 1, []))
        self.assertIsNone(connector.load_generation(1, 1))
        self.assertIsNone(connector.get_best_genomes(1))
        self.assertIsNone(connector.get_experiment_info(1))
        self.assertFalse(connector.initialize_schema())
    
    @patch('explaneat.database.connector.psycopg2.pool.ThreadedConnectionPool')
    def test_init_with_connection_pool_success(self, mock_pool_class):
        """Test successful initialization with connection pool."""
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        connector = NEATDatabaseConnector(
            connection_string="postgresql://test:test@localhost/test",
            pool_size=3,
            enabled=True
        )
        
        self.assertTrue(connector.enabled)
        self.assertEqual(connector.pool, mock_pool)
        mock_pool_class.assert_called_once_with(
            minconn=1,
            maxconn=3,
            dsn="postgresql://test:test@localhost/test"
        )
    
    @patch('explaneat.database.connector.psycopg2.pool.ThreadedConnectionPool')
    def test_init_with_connection_pool_failure(self, mock_pool_class):
        """Test initialization failure with connection pool."""
        mock_pool_class.side_effect = Exception("Connection failed")
        
        connector = NEATDatabaseConnector(
            connection_string="postgresql://test:test@localhost/test",
            enabled=True
        )
        
        self.assertFalse(connector.enabled)
        self.assertIsNone(connector.pool)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with patch('explaneat.database.connector.psycopg2.pool.ThreadedConnectionPool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool_class.return_value = mock_pool
            
            with NEATDatabaseConnector(connection_string="postgresql://test:test@localhost/test") as connector:
                self.assertTrue(connector.enabled)
            
            mock_pool.closeall.assert_called_once()
    
    def test_retry_logic_configuration(self):
        """Test retry logic configuration."""
        connector = NEATDatabaseConnector(
            enabled=False,
            retry_attempts=5,
            retry_delay=2.0
        )
        
        self.assertEqual(connector.retry_attempts, 5)
        self.assertEqual(connector.retry_delay, 2.0)


class TestNEATDatabaseConnectorMockOperations(unittest.TestCase):
    """Test database operations with mocked connections."""
    
    def setUp(self):
        """Set up test with mocked pool and connections."""
        self.mock_pool = Mock()
        self.mock_conn = Mock()
        self.mock_cursor = Mock()
        
        # Set up proper context manager mocking for cursor
        self.mock_conn.cursor.return_value = Mock()
        self.mock_conn.cursor.return_value.__enter__ = Mock(return_value=self.mock_cursor)
        self.mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        
        self.mock_pool.getconn.return_value = self.mock_conn
        
        with patch('explaneat.database.connector.psycopg2.pool.ThreadedConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = self.mock_pool
            self.connector = NEATDatabaseConnector(
                connection_string="postgresql://test:test@localhost/test"
            )
    
    def test_create_experiment_success(self):
        """Test successful experiment creation."""
        self.mock_cursor.fetchone.return_value = [123]
        
        result = self.connector.create_experiment("test_exp", {"param": "value"}, "Test description")
        
        self.assertEqual(result, 123)
        self.mock_cursor.execute.assert_called_once()
        self.mock_conn.commit.assert_called_once()
    
    def test_create_experiment_failure(self):
        """Test experiment creation failure with rollback."""
        self.mock_cursor.execute.side_effect = Exception("DB Error")
        
        # Since retry logic will attempt 3 times, disable it for this test
        self.connector.retry_attempts = 1
        
        result = self.connector.create_experiment("test_exp", {"param": "value"})
        
        self.assertIsNone(result)
        self.mock_conn.rollback.assert_called_once()
    
    def test_save_generation_success(self):
        """Test successful generation saving."""
        # Mock the generation ID return
        self.mock_cursor.fetchone.return_value = [456]
        # Mock genome ID mapping query
        self.mock_cursor.fetchall.return_value = [(1, 101), (2, 102)]
        
        # Create test genomes
        genomes = [
            (1, MockGenome(1, fitness=0.8)),
            (2, MockGenome(2, fitness=0.9))
        ]
        
        result = self.connector.save_generation(123, 1, genomes)
        
        self.assertTrue(result)
        # Should have multiple execute calls for different tables
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 1)
        self.mock_conn.commit.assert_called_once()
    
    def test_save_generation_empty_population(self):
        """Test saving generation with empty population."""
        # Mock the generation insertion
        self.mock_cursor.fetchone.return_value = [456]
        # Mock the genome ID mapping query to return empty list
        self.mock_cursor.fetchall.return_value = []
        
        # Set retry attempts to 1 to avoid retry loops
        self.connector.retry_attempts = 1
        
        result = self.connector.save_generation(123, 1, [])
        
        # Empty population should still succeed  
        self.assertTrue(result)
        self.mock_cursor.execute.assert_called()
        self.mock_conn.commit.assert_called()
    
    def test_load_generation_success(self):
        """Test successful generation loading."""
        # Mock generation data
        gen_data = {
            'id': 456,
            'generation_number': 1,
            'population_size': 2,
            'best_fitness': 0.9
        }
        
        # Mock genome data with nodes and connections
        genome_data = [
            {
                'id': 101,
                'genome_id': 1,
                'fitness': 0.8,
                'nodes': [{'node_id': 1, 'type': 'input'}],
                'connections': [{'innovation': 1, 'weight': 0.5}]
            }
        ]
        
        # Configure mock cursor for different queries
        self.mock_cursor.fetchone.return_value = gen_data
        self.mock_cursor.fetchall.return_value = genome_data
        
        result = self.connector.load_generation(123, 1)
        
        self.assertIsNotNone(result)
        self.assertIn('generation_info', result)
        self.assertIn('genomes', result)
        self.assertEqual(len(result['genomes']), 1)
    
    def test_load_generation_not_found(self):
        """Test loading non-existent generation."""
        self.mock_cursor.fetchone.return_value = None
        
        result = self.connector.load_generation(123, 999)
        
        self.assertIsNone(result)
    
    def test_get_best_genomes_success(self):
        """Test getting best genomes."""
        genome_data = [
            {'genome_id': 1, 'fitness': 0.9, 'generation_number': 1},
            {'genome_id': 2, 'fitness': 0.8, 'generation_number': 1}
        ]
        self.mock_cursor.fetchall.return_value = genome_data
        
        result = self.connector.get_best_genomes(123, 10)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['fitness'], 0.9)
    
    def test_get_experiment_info_success(self):
        """Test getting experiment information."""
        exp_data = {
            'id': 123,
            'name': 'test_exp',
            'config_json': '{"param": "value"}',
            'description': 'Test experiment'
        }
        self.mock_cursor.fetchone.return_value = exp_data
        
        result = self.connector.get_experiment_info(123)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'test_exp')


if __name__ == '__main__':
    unittest.main()