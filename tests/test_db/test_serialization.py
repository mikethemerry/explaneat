"""
Tests for genome serialization and deserialization.

Tests conversion between NEAT-Python genomes and JSON storage format,
including edge cases like infinite fitness and NaN values.
"""

import pytest
import math
import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene

from explaneat.db.serialization import (
    serialize_genome,
    deserialize_genome,
    calculate_genome_stats,
    serialize_population_config,
)


@pytest.mark.db
@pytest.mark.unit
class TestGenomeSerialization:
    """Test genome serialization functions."""

    def test_serialize_simple_genome(self, neat_config):
        """Test serializing a simple genome."""
        # Create a simple genome
        genome = neat.DefaultGenome(1)
        genome.fitness = 1.5
        
        # Add a node
        genome.nodes[0] = DefaultNodeGene(0)
        genome.nodes[0].bias = 0.5
        genome.nodes[0].activation = 'relu'
        
        # Add a connection
        genome.connections[(0, 1)] = DefaultConnectionGene((0, 1))
        genome.connections[(0, 1)].weight = 0.7
        genome.connections[(0, 1)].enabled = True
        
        # Serialize
        serialized = serialize_genome(genome)
        
        # Verify structure
        assert 'nodes' in serialized
        assert 'connections' in serialized
        assert 'fitness' in serialized
        assert 'key' in serialized
        
        # Verify values
        assert serialized['fitness'] == 1.5
        assert serialized['key'] == 1
        assert '0' in serialized['nodes']
        assert serialized['nodes']['0']['bias'] == 0.5
        assert serialized['nodes']['0']['activation'] == 'relu'

    def test_deserialize_simple_genome(self, neat_config):
        """Test deserializing a simple genome."""
        # Create and serialize a genome
        genome = neat.DefaultGenome(1)
        genome.fitness = 1.5
        genome.nodes[0] = DefaultNodeGene(0)
        genome.nodes[0].bias = 0.5
        genome.nodes[0].activation = 'relu'
        genome.connections[(0, 1)] = DefaultConnectionGene((0, 1))
        genome.connections[(0, 1)].weight = 0.7
        genome.connections[(0, 1)].enabled = True
        
        serialized = serialize_genome(genome)
        
        # Deserialize
        deserialized = deserialize_genome(serialized, neat_config)
        
        # Verify values
        assert deserialized.key == 1
        assert deserialized.fitness == 1.5
        assert 0 in deserialized.nodes
        assert deserialized.nodes[0].bias == 0.5
        assert deserialized.nodes[0].activation == 'relu'
        assert (0, 1) in deserialized.connections
        assert deserialized.connections[(0, 1)].weight == 0.7
        assert deserialized.connections[(0, 1)].enabled is True

    def test_round_trip_serialization(self, neat_config):
        """Test round-trip serialization (genome -> JSON -> genome)."""
        # Create a complex genome
        genome = neat.DefaultGenome(42)
        genome.fitness = 2.5
        
        # Add multiple nodes
        for i in range(-5, 3):
            genome.nodes[i] = DefaultNodeGene(i)
            genome.nodes[i].bias = float(i) * 0.1
            genome.nodes[i].activation = 'relu' if i < 0 else 'sigmoid'
        
        # Add multiple connections
        for i in range(-5, 0):
            for j in range(0, 3):
                conn_key = (i, j)
                genome.connections[conn_key] = DefaultConnectionGene(conn_key)
                genome.connections[conn_key].weight = float(i + j) * 0.1
                genome.connections[conn_key].enabled = (i + j) % 2 == 0
        
        # Round trip
        serialized = serialize_genome(genome)
        deserialized = deserialize_genome(serialized, neat_config)
        
        # Verify all nodes
        assert len(deserialized.nodes) == len(genome.nodes)
        for node_id in genome.nodes:
            assert node_id in deserialized.nodes
            assert deserialized.nodes[node_id].bias == genome.nodes[node_id].bias
            assert deserialized.nodes[node_id].activation == genome.nodes[node_id].activation
        
        # Verify all connections
        assert len(deserialized.connections) == len(genome.connections)
        for conn_key in genome.connections:
            assert conn_key in deserialized.connections
            assert deserialized.connections[conn_key].weight == genome.connections[conn_key].weight
            assert deserialized.connections[conn_key].enabled == genome.connections[conn_key].enabled

    def test_serialize_infinite_fitness_positive(self, neat_config):
        """Test serializing positive infinite fitness."""
        genome = neat.DefaultGenome(1)
        genome.fitness = float('inf')
        
        serialized = serialize_genome(genome)
        
        # Should convert to string
        assert serialized['fitness'] == "Infinity"
        
        # Deserialize should restore
        deserialized = deserialize_genome(serialized, neat_config)
        assert deserialized.fitness == float('inf')
        assert math.isinf(deserialized.fitness)
        assert deserialized.fitness > 0

    def test_serialize_infinite_fitness_negative(self, neat_config):
        """Test serializing negative infinite fitness."""
        genome = neat.DefaultGenome(1)
        genome.fitness = float('-inf')
        
        serialized = serialize_genome(genome)
        
        # Should convert to string
        assert serialized['fitness'] == "-Infinity"
        
        # Deserialize should restore
        deserialized = deserialize_genome(serialized, neat_config)
        assert deserialized.fitness == float('-inf')
        assert math.isinf(deserialized.fitness)
        assert deserialized.fitness < 0

    def test_serialize_nan_fitness(self, neat_config):
        """Test serializing NaN fitness."""
        genome = neat.DefaultGenome(1)
        genome.fitness = float('nan')
        
        serialized = serialize_genome(genome)
        
        # Should convert to None
        assert serialized['fitness'] is None
        
        # Deserialize should restore as None (not NaN)
        deserialized = deserialize_genome(serialized, neat_config)
        assert deserialized.fitness is None

    def test_serialize_infinite_weight(self, neat_config):
        """Test serializing infinite connection weights."""
        genome = neat.DefaultGenome(1)
        genome.connections[(0, 1)] = DefaultConnectionGene((0, 1))
        genome.connections[(0, 1)].weight = float('inf')
        
        serialized = serialize_genome(genome)
        
        # Should convert to string
        assert serialized['connections']['0_1']['weight'] == "Infinity"
        
        # Deserialize should restore
        deserialized = deserialize_genome(serialized, neat_config)
        assert math.isinf(deserialized.connections[(0, 1)].weight)

    def test_serialize_nan_bias(self, neat_config):
        """Test serializing NaN node bias."""
        genome = neat.DefaultGenome(1)
        genome.nodes[0] = DefaultNodeGene(0)
        genome.nodes[0].bias = float('nan')
        
        serialized = serialize_genome(genome)
        
        # Should convert to None
        assert serialized['nodes']['0']['bias'] is None
        
        # Deserialize should restore as None
        deserialized = deserialize_genome(serialized, neat_config)
        assert deserialized.nodes[0].bias is None

    def test_calculate_genome_stats(self, neat_config):
        """Test calculating genome statistics."""
        genome = neat.DefaultGenome(1)
        
        # Add input nodes (negative IDs)
        for i in range(-5, 0):
            genome.nodes[i] = DefaultNodeGene(i)
        
        # Add output node
        genome.nodes[0] = DefaultNodeGene(0)
        
        # Add hidden node
        genome.nodes[5] = DefaultNodeGene(5)
        
        # Add connections
        for i in range(-5, 0):
            genome.connections[(i, 0)] = DefaultConnectionGene((i, 0))
            genome.connections[(i, 0)].enabled = True
        
        genome.connections[(0, 5)] = DefaultConnectionGene((0, 5))
        genome.connections[(0, 5)].enabled = False  # Disabled
        
        stats = calculate_genome_stats(genome)
        
        assert stats['num_nodes'] == 7  # 5 inputs + 1 output + 1 hidden
        assert stats['num_connections'] == 6  # 5 enabled + 1 disabled
        assert stats['num_enabled_connections'] == 5
        assert stats['network_depth'] >= 1
        assert stats['network_width'] >= 1

    def test_serialize_population_config(self, neat_config):
        """Test serializing NEAT population configuration."""
        config_dict = serialize_population_config(neat_config)
        
        # Verify structure
        assert 'pop_size' in config_dict
        assert 'genome' in config_dict
        assert 'species' in config_dict
        assert 'stagnation' in config_dict
        assert 'reproduction' in config_dict
        
        # Verify values match config
        assert config_dict['pop_size'] == neat_config.pop_size
        assert config_dict['genome']['num_inputs'] == neat_config.genome_config.num_inputs
        assert config_dict['genome']['num_outputs'] == neat_config.genome_config.num_outputs

    def test_serialize_empty_genome(self, neat_config):
        """Test serializing an empty genome (no nodes or connections)."""
        genome = neat.DefaultGenome(1)
        genome.fitness = 0.0
        
        serialized = serialize_genome(genome)
        
        assert serialized['nodes'] == {}
        assert serialized['connections'] == {}
        assert serialized['fitness'] == 0.0
        
        # Should deserialize correctly
        deserialized = deserialize_genome(serialized, neat_config)
        assert len(deserialized.nodes) == 0
        assert len(deserialized.connections) == 0

    def test_serialize_genome_with_none_fitness(self, neat_config):
        """Test serializing genome with None fitness."""
        genome = neat.DefaultGenome(1)
        genome.fitness = None
        
        serialized = serialize_genome(genome)
        
        assert serialized['fitness'] is None
        
        # Deserialize should preserve None
        deserialized = deserialize_genome(serialized, neat_config)
        assert deserialized.fitness is None
