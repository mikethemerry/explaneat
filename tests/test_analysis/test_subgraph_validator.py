"""
Tests for SubgraphValidator class.

Tests subgraph connectivity validation and genome validation.
"""

import pytest
import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene

from explaneat.analysis.subgraph_validator import SubgraphValidator


@pytest.mark.unit
class TestSubgraphValidatorConnectivity:
    """Test SubgraphValidator.validate_connectivity method."""

    def test_validate_connected_subgraph(self):
        """Test validating a connected subgraph."""
        nodes = [-1, -2, 0]
        connections = [(-1, 0), (-2, 0)]
        
        result = SubgraphValidator.validate_connectivity(nodes, connections)
        
        assert result["is_connected"] is True
        assert result["is_valid"] is True
        assert result["error_message"] is None

    def test_validate_disconnected_subgraph(self):
        """Test validating a disconnected subgraph."""
        nodes = [-1, -2, 0]
        connections = [(-1, 0)]  # Missing connection to -2
        
        result = SubgraphValidator.validate_connectivity(nodes, connections)
        
        assert result["is_connected"] is False
        assert len(result["connected_components"]) > 1

    def test_validate_single_node(self):
        """Test validating single node subgraph."""
        nodes = [0]
        connections = []
        
        result = SubgraphValidator.validate_connectivity(nodes, connections)
        
        assert result["is_connected"] is True
        assert result["is_valid"] is True

    def test_validate_empty_subgraph(self):
        """Test validating empty subgraph."""
        nodes = []
        connections = []
        
        result = SubgraphValidator.validate_connectivity(nodes, connections)
        
        assert result["is_connected"] is False
        assert result["is_valid"] is False

    def test_validate_invalid_connection(self):
        """Test validating subgraph with invalid connection."""
        nodes = [-1, 0]
        connections = [(-1, 0), (-2, 0)]  # -2 not in nodes
        
        result = SubgraphValidator.validate_connectivity(nodes, connections)
        
        assert result["is_valid"] is False
        assert "not in subgraph" in result["error_message"]

    def test_validate_directed_graph(self):
        """Test validating directed graph."""
        nodes = [1, 2, 3]
        connections = [(1, 2), (2, 3)]
        
        result = SubgraphValidator.validate_connectivity(nodes, connections, directed=True)
        
        assert result["is_connected"] is True


@pytest.mark.unit
class TestSubgraphValidatorGenome:
    """Test SubgraphValidator.validate_against_genome method."""

    def test_validate_against_genome_valid(self, neat_config):
        """Test validating valid subgraph against genome."""
        # Create simple genome
        genome = neat.DefaultGenome(1)
        genome.nodes[-1] = DefaultNodeGene(-1)
        genome.nodes[0] = DefaultNodeGene(0)
        genome.connections[(-1, 0)] = DefaultConnectionGene((-1, 0))
        genome.connections[(-1, 0)].enabled = True
        
        nodes = [-1, 0]
        connections = [(-1, 0)]
        
        result = SubgraphValidator.validate_against_genome(genome, nodes, connections, neat_config)
        
        assert result["is_valid"] is True

    def test_validate_against_genome_invalid_node(self, neat_config):
        """Test validating subgraph with node not in genome."""
        genome = neat.DefaultGenome(1)
        genome.nodes[-1] = DefaultNodeGene(-1)
        genome.nodes[0] = DefaultNodeGene(0)
        
        nodes = [-1, 0, 99]  # 99 not in genome
        connections = [(-1, 0)]
        
        result = SubgraphValidator.validate_against_genome(genome, nodes, connections, neat_config)
        
        assert result["is_valid"] is False
        assert "not found in genome" in result["error_message"]

    def test_validate_against_genome_invalid_connection(self, neat_config):
        """Test validating subgraph with connection not in genome."""
        genome = neat.DefaultGenome(1)
        genome.nodes[-1] = DefaultNodeGene(-1)
        genome.nodes[0] = DefaultNodeGene(0)
        
        nodes = [-1, 0]
        connections = [(-1, 0), (-1, 99)]  # (-1, 99) not in genome
        
        result = SubgraphValidator.validate_against_genome(genome, nodes, connections, neat_config)
        
        assert result["is_valid"] is False
