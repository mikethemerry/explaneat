"""Genome serialization utilities for NEAT-Python genomes"""
import json
import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from typing import Dict, Any, Optional


def serialize_genome(genome: neat.DefaultGenome) -> Dict[str, Any]:
    """Convert a NEAT-Python genome to a JSON-serializable dictionary"""
    
    # Helper function to handle infinite values
    def handle_infinite(value):
        if value == float('inf'):
            return "Infinity"
        elif value == float('-inf'):
            return "-Infinity"
        elif value != value:  # NaN check
            return None
        return value
    
    # Serialize nodes
    nodes = {}
    for node_id, node in genome.nodes.items():
        nodes[str(node_id)] = {
            'bias': handle_infinite(node.bias),
            'response': handle_infinite(node.response),
            'activation': node.activation,
            'aggregation': node.aggregation
        }
    
    # Serialize connections
    connections = {}
    for conn_key, conn in genome.connections.items():
        key_str = f"{conn_key[0]}_{conn_key[1]}"
        connections[key_str] = {
            'weight': handle_infinite(conn.weight),
            'enabled': conn.enabled,
            'in_node': conn_key[0],
            'out_node': conn_key[1]
        }
    
    return {
        'nodes': nodes,
        'connections': connections,
        'fitness': handle_infinite(genome.fitness),
        'key': genome.key
    }


def deserialize_genome(genome_data: Dict[str, Any], config: neat.Config) -> neat.DefaultGenome:
    """Convert a JSON dictionary back to a NEAT-Python genome"""
    
    # Helper function to restore infinite values
    def restore_infinite(value):
        if value == "Infinity":
            return float('inf')
        elif value == "-Infinity":
            return float('-inf')
        return value
    
    # Create new genome instance
    genome = neat.DefaultGenome(genome_data['key'])
    genome.fitness = restore_infinite(genome_data.get('fitness'))
    
    # Restore nodes
    for node_id_str, node_data in genome_data['nodes'].items():
        node_id = int(node_id_str)
        node = DefaultNodeGene(node_id)
        node.bias = restore_infinite(node_data['bias'])
        node.response = restore_infinite(node_data['response'])
        node.activation = node_data['activation']
        node.aggregation = node_data['aggregation']
        genome.nodes[node_id] = node
    
    # Restore connections
    for conn_key_str, conn_data in genome_data['connections'].items():
        in_node = conn_data['in_node']
        out_node = conn_data['out_node']
        conn_key = (in_node, out_node)
        
        conn = DefaultConnectionGene(conn_key)
        conn.weight = restore_infinite(conn_data['weight'])
        conn.enabled = conn_data['enabled']
        genome.connections[conn_key] = conn
    
    return genome


def calculate_genome_stats(genome: neat.DefaultGenome) -> Dict[str, int]:
    """Calculate network statistics for a genome"""
    
    # Count enabled connections
    enabled_connections = sum(1 for conn in genome.connections.values() if conn.enabled)
    
    # Calculate network depth and width (simplified)
    input_nodes = set()
    output_nodes = set()
    hidden_nodes = set()
    
    for node_id in genome.nodes:
        if node_id < 0:  # Input nodes typically have negative IDs
            input_nodes.add(node_id)
        elif node_id == 0:  # Output node typically has ID 0
            output_nodes.add(node_id)
        else:
            hidden_nodes.add(node_id)
    
    # Simple depth calculation: assume feedforward network
    # In a more complex implementation, you'd do graph traversal
    depth = len(hidden_nodes) + 1 if hidden_nodes else 1
    width = max(len(input_nodes), len(hidden_nodes), len(output_nodes))
    
    return {
        'num_nodes': len(genome.nodes),
        'num_connections': len(genome.connections),
        'num_enabled_connections': enabled_connections,
        'network_depth': depth,
        'network_width': width
    }


def serialize_population_config(pop_config: neat.Config) -> Dict[str, Any]:
    """Serialize NEAT configuration for storage"""
    
    # Extract key configuration parameters
    config_dict = {
        'pop_size': pop_config.pop_size,
        'fitness_criterion': pop_config.fitness_criterion,
        'fitness_threshold': pop_config.fitness_threshold,
        'no_fitness_termination': pop_config.no_fitness_termination,
        'reset_on_extinction': pop_config.reset_on_extinction,
        
        # Genome config
        'genome': {
            'activation_default': pop_config.genome_config.activation_default,
            'activation_mutate_rate': pop_config.genome_config.activation_mutate_rate,
            'activation_options': pop_config.genome_config.activation_options,
            'aggregation_default': pop_config.genome_config.aggregation_default,
            'aggregation_mutate_rate': pop_config.genome_config.aggregation_mutate_rate,
            'aggregation_options': pop_config.genome_config.aggregation_options,
            'bias_init_mean': pop_config.genome_config.bias_init_mean,
            'bias_init_stdev': pop_config.genome_config.bias_init_stdev,
            'bias_max_value': pop_config.genome_config.bias_max_value,
            'bias_min_value': pop_config.genome_config.bias_min_value,
            'bias_mutate_power': pop_config.genome_config.bias_mutate_power,
            'bias_mutate_rate': pop_config.genome_config.bias_mutate_rate,
            'bias_replace_rate': pop_config.genome_config.bias_replace_rate,
            'compatibility_disjoint_coefficient': pop_config.genome_config.compatibility_disjoint_coefficient,
            'compatibility_weight_coefficient': pop_config.genome_config.compatibility_weight_coefficient,
            'conn_add_prob': pop_config.genome_config.conn_add_prob,
            'conn_delete_prob': pop_config.genome_config.conn_delete_prob,
            'enabled_default': pop_config.genome_config.enabled_default,
            'enabled_mutate_rate': pop_config.genome_config.enabled_mutate_rate,
            'feed_forward': pop_config.genome_config.feed_forward,
            'initial_connection': pop_config.genome_config.initial_connection,
            'node_add_prob': pop_config.genome_config.node_add_prob,
            'node_delete_prob': pop_config.genome_config.node_delete_prob,
            'num_hidden': pop_config.genome_config.num_hidden,
            'num_inputs': pop_config.genome_config.num_inputs,
            'num_outputs': pop_config.genome_config.num_outputs,
            'response_init_mean': pop_config.genome_config.response_init_mean,
            'response_init_stdev': pop_config.genome_config.response_init_stdev,
            'response_max_value': pop_config.genome_config.response_max_value,
            'response_min_value': pop_config.genome_config.response_min_value,
            'response_mutate_power': pop_config.genome_config.response_mutate_power,
            'response_mutate_rate': pop_config.genome_config.response_mutate_rate,
            'response_replace_rate': pop_config.genome_config.response_replace_rate,
            'weight_init_mean': pop_config.genome_config.weight_init_mean,
            'weight_init_stdev': pop_config.genome_config.weight_init_stdev,
            'weight_max_value': pop_config.genome_config.weight_max_value,
            'weight_min_value': pop_config.genome_config.weight_min_value,
            'weight_mutate_power': pop_config.genome_config.weight_mutate_power,
            'weight_mutate_rate': pop_config.genome_config.weight_mutate_rate,
            'weight_replace_rate': pop_config.genome_config.weight_replace_rate
        },
        
        # Species config
        'species': {
            'compatibility_threshold': pop_config.species_set_config.compatibility_threshold
        },
        
        # Stagnation config
        'stagnation': {
            'species_fitness_func': pop_config.stagnation_config.species_fitness_func,
            'max_stagnation': pop_config.stagnation_config.max_stagnation,
            'species_elitism': pop_config.stagnation_config.species_elitism
        },
        
        # Reproduction config
        'reproduction': {
            'elitism': pop_config.reproduction_config.elitism,
            'survival_threshold': pop_config.reproduction_config.survival_threshold,
            'min_species_size': pop_config.reproduction_config.min_species_size
        }
    }
    
    return config_dict