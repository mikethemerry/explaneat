"""
AncestryAnalyzer: Traces genome ancestry and gene origins through generations

This class provides functionality to:
- Build ancestry trees from parent relationships in the database
- Trace when genes (nodes/connections) were first introduced
- Compare genomes across generations
- Analyze evolutionary patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import networkx as nx

from ..db import db, Genome, Population


@dataclass
class AncestorInfo:
    """Information about a genome's ancestor"""
    genome_id: str
    neat_genome_id: int
    generation: int
    fitness: float
    num_nodes: int
    num_connections: int
    num_enabled_connections: int
    network_depth: int
    network_width: int
    parent1_id: Optional[str]
    parent2_id: Optional[str]


class AncestryAnalyzer:
    """
    Analyzes the evolutionary ancestry of a genome.
    
    This class traces backwards through parent relationships to understand
    how a genome evolved and when specific genes were introduced.
    """
    
    def __init__(self, genome_id: str):
        self.genome_id = genome_id
        self._ancestry_cache = None
        self._gene_origins_cache = None
        
    def get_ancestry_tree(self, max_generations: int = 10) -> pd.DataFrame:
        """
        Build the ancestry tree for this genome.
        
        Args:
            max_generations: Maximum number of generations to trace back
            
        Returns:
            DataFrame with ancestor information sorted by generation (oldest first)
        """
        if self._ancestry_cache is not None:
            return self._ancestry_cache.head(max_generations)
            
        ancestors = []
        visited = set()
        to_process = [(self.genome_id, 0)]  # (genome_id, generation_offset)
        
        with db.session_scope() as session:
            while to_process and len(ancestors) < max_generations:
                current_id, gen_offset = to_process.pop(0)
                
                if current_id in visited or current_id is None:
                    continue
                    
                visited.add(current_id)
                
                # Get genome record
                genome_record = session.get(Genome, current_id)
                if not genome_record:
                    continue
                    
                # Get population info for generation
                population = session.get(Population, genome_record.population_id)
                if not population:
                    continue
                    
                # Add to ancestors list
                ancestor_info = AncestorInfo(
                    genome_id=str(genome_record.id),
                    neat_genome_id=genome_record.genome_id,
                    generation=population.generation,
                    fitness=genome_record.fitness or 0.0,
                    num_nodes=genome_record.num_nodes,
                    num_connections=genome_record.num_connections,
                    num_enabled_connections=genome_record.num_enabled_connections,
                    network_depth=genome_record.network_depth,
                    network_width=genome_record.network_width,
                    parent1_id=genome_record.parent1_id,
                    parent2_id=genome_record.parent2_id
                )
                ancestors.append(ancestor_info)
                
                # Add parents to processing queue
                if genome_record.parent1_id:
                    to_process.append((genome_record.parent1_id, gen_offset + 1))
                if genome_record.parent2_id:
                    to_process.append((genome_record.parent2_id, gen_offset + 1))
        
        # Convert to DataFrame and sort by generation (oldest first)
        if ancestors:
            df = pd.DataFrame([
                {
                    'genome_id': a.genome_id,
                    'neat_genome_id': a.neat_genome_id,
                    'generation': a.generation,
                    'fitness': a.fitness,
                    'num_nodes': a.num_nodes,
                    'num_connections': a.num_connections,
                    'num_enabled_connections': a.num_enabled_connections,
                    'network_depth': a.network_depth,
                    'network_width': a.network_width,
                    'parent1_id': a.parent1_id,
                    'parent2_id': a.parent2_id
                } for a in ancestors
            ])
            df = df.sort_values('generation').reset_index(drop=True)
        else:
            df = pd.DataFrame()
            
        self._ancestry_cache = df
        return df.head(max_generations)
    
    def trace_gene_origins(self, current_genome) -> pd.DataFrame:
        """
        Trace when each gene (node/connection) was first introduced in the ancestry.
        
        Args:
            current_genome: The NEAT genome to analyze
            
        Returns:
            DataFrame with gene origin information
        """
        if self._gene_origins_cache is not None:
            return self._gene_origins_cache
            
        ancestry_df = self.get_ancestry_tree()
        if ancestry_df.empty:
            return pd.DataFrame()
            
        gene_origins = []
        
        # Load NEAT config (needed for deserialization)
        import neat
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "config-file.cfg"
        )
        
        # Get genome data for each ancestor
        ancestor_genomes = {}
        with db.session_scope() as session:
            for _, ancestor in ancestry_df.iterrows():
                genome_record = session.get(Genome, ancestor['genome_id'])
                if genome_record:
                    try:
                        neat_genome = genome_record.to_neat_genome(config)
                        ancestor_genomes[ancestor['generation']] = neat_genome
                    except Exception:
                        continue
        
        # Trace node origins
        current_nodes = set(current_genome.nodes.keys())
        for node_id in current_nodes:
            origin_generation = None
            
            # Check each ancestor generation (from oldest to newest)
            for generation in sorted(ancestor_genomes.keys()):
                ancestor_genome = ancestor_genomes[generation]
                if node_id in ancestor_genome.nodes:
                    origin_generation = generation
                    break
            
            gene_origins.append({
                'gene_type': 'node',
                'gene_id': str(node_id),
                'origin_generation': origin_generation,
                'current_generation': ancestry_df['generation'].max(),
                'age': ancestry_df['generation'].max() - origin_generation if origin_generation is not None else None,
                'gene_info': {
                    'bias': current_genome.nodes[node_id].bias,
                    'activation': current_genome.nodes[node_id].activation,
                    'aggregation': current_genome.nodes[node_id].aggregation
                }
            })
        
        # Trace connection origins
        current_connections = set(current_genome.connections.keys())
        for conn_key in current_connections:
            origin_generation = None
            
            # Check each ancestor generation (from oldest to newest)
            for generation in sorted(ancestor_genomes.keys()):
                ancestor_genome = ancestor_genomes[generation]
                if conn_key in ancestor_genome.connections:
                    origin_generation = generation
                    break
            
            gene_origins.append({
                'gene_type': 'connection',
                'gene_id': f"{conn_key[0]}→{conn_key[1]}",
                'origin_generation': origin_generation,
                'current_generation': ancestry_df['generation'].max(),
                'age': ancestry_df['generation'].max() - origin_generation if origin_generation is not None else None,
                'gene_info': {
                    'weight': current_genome.connections[conn_key].weight,
                    'enabled': current_genome.connections[conn_key].enabled,
                    'from_node': conn_key[0],
                    'to_node': conn_key[1]
                }
            })
        
        df = pd.DataFrame(gene_origins)
        self._gene_origins_cache = df
        return df
    
    def compare_with_ancestor(self, current_genome, ancestor_generation: int) -> Dict[str, Any]:
        """
        Compare current genome with a specific ancestor generation.
        
        Args:
            current_genome: The current NEAT genome
            ancestor_generation: Generation number of ancestor to compare with
            
        Returns:
            Dictionary with comparison results
        """
        ancestry_df = self.get_ancestry_tree()
        ancestor_info = ancestry_df[ancestry_df['generation'] == ancestor_generation]
        
        if ancestor_info.empty:
            return {'error': f'No ancestor found at generation {ancestor_generation}'}
        
        ancestor_row = ancestor_info.iloc[0]
        
        # Load ancestor genome
        import neat
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "config-file.cfg"
        )
        
        with db.session_scope() as session:
            ancestor_record = session.get(Genome, ancestor_row['genome_id'])
            if not ancestor_record:
                return {'error': 'Could not load ancestor genome'}
                
            try:
                ancestor_genome = ancestor_record.to_neat_genome(config)
            except Exception as e:
                return {'error': f'Could not deserialize ancestor genome: {e}'}
        
        # Compare structures
        current_nodes = set(current_genome.nodes.keys())
        ancestor_nodes = set(ancestor_genome.nodes.keys())
        current_connections = set(current_genome.connections.keys())
        ancestor_connections = set(ancestor_genome.connections.keys())
        
        # Node changes
        added_nodes = current_nodes - ancestor_nodes
        removed_nodes = ancestor_nodes - current_nodes
        shared_nodes = current_nodes & ancestor_nodes
        
        # Connection changes
        added_connections = current_connections - ancestor_connections
        removed_connections = ancestor_connections - current_connections
        shared_connections = current_connections & ancestor_connections
        
        # Calculate weight changes for shared connections
        weight_changes = []
        for conn_key in shared_connections:
            current_weight = current_genome.connections[conn_key].weight
            ancestor_weight = ancestor_genome.connections[conn_key].weight
            weight_changes.append({
                'connection': f"{conn_key[0]}→{conn_key[1]}",
                'ancestor_weight': ancestor_weight,
                'current_weight': current_weight,
                'change': current_weight - ancestor_weight
            })
        
        # Calculate bias changes for shared nodes
        bias_changes = []
        for node_id in shared_nodes:
            current_bias = current_genome.nodes[node_id].bias
            ancestor_bias = ancestor_genome.nodes[node_id].bias
            bias_changes.append({
                'node_id': node_id,
                'ancestor_bias': ancestor_bias,
                'current_bias': current_bias,
                'change': current_bias - ancestor_bias
            })
        
        return {
            'ancestor_generation': ancestor_generation,
            'current_generation': ancestry_df['generation'].max(),
            'fitness_change': current_genome.fitness - ancestor_row['fitness'],
            'structure_changes': {
                'nodes_added': len(added_nodes),
                'nodes_removed': len(removed_nodes),
                'connections_added': len(added_connections),
                'connections_removed': len(removed_connections),
                'added_nodes': list(added_nodes),
                'removed_nodes': list(removed_nodes),
                'added_connections': [f"{k[0]}→{k[1]}" for k in added_connections],
                'removed_connections': [f"{k[0]}→{k[1]}" for k in removed_connections]
            },
            'parameter_changes': {
                'weight_changes': weight_changes,
                'bias_changes': bias_changes,
                'avg_weight_change': np.mean([w['change'] for w in weight_changes]) if weight_changes else 0,
                'avg_bias_change': np.mean([b['change'] for b in bias_changes]) if bias_changes else 0
            },
            'complexity_changes': {
                'node_count_change': len(current_nodes) - len(ancestor_nodes),
                'connection_count_change': len(current_connections) - len(ancestor_connections),
                'depth_change': ancestor_row['network_depth'],  # Would need current depth calculation
                'width_change': ancestor_row['network_width']   # Would need current width calculation
            }
        }
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """
        Get statistical analysis of the lineage.
        
        Returns:
            Dictionary with lineage statistics
        """
        ancestry_df = self.get_ancestry_tree()
        if ancestry_df.empty:
            return {}
        
        return {
            'lineage_length': len(ancestry_df),
            'fitness_progression': {
                'initial_fitness': ancestry_df['fitness'].iloc[0],
                'final_fitness': ancestry_df['fitness'].iloc[-1],
                'best_fitness': ancestry_df['fitness'].max(),
                'worst_fitness': ancestry_df['fitness'].min(),
                'average_fitness': ancestry_df['fitness'].mean(),
                'fitness_trend': 'improving' if ancestry_df['fitness'].iloc[-1] > ancestry_df['fitness'].iloc[0] else 'declining'
            },
            'complexity_progression': {
                'initial_nodes': ancestry_df['num_nodes'].iloc[0],
                'final_nodes': ancestry_df['num_nodes'].iloc[-1],
                'max_nodes': ancestry_df['num_nodes'].max(),
                'initial_connections': ancestry_df['num_connections'].iloc[0],
                'final_connections': ancestry_df['num_connections'].iloc[-1],
                'max_connections': ancestry_df['num_connections'].max(),
                'complexity_trend': 'growing' if ancestry_df['num_connections'].iloc[-1] > ancestry_df['num_connections'].iloc[0] else 'shrinking'
            },
            'generation_span': {
                'earliest_generation': ancestry_df['generation'].min(),
                'latest_generation': ancestry_df['generation'].max(),
                'span': ancestry_df['generation'].max() - ancestry_df['generation'].min()
            }
        }
    
    def find_common_ancestor(self, other_genome_id: str) -> Optional[Dict[str, Any]]:
        """
        Find the most recent common ancestor with another genome.
        
        Args:
            other_genome_id: Database ID of the other genome
            
        Returns:
            Information about the common ancestor, or None if not found
        """
        # Get ancestry for current genome
        self_ancestry = self.get_ancestry_tree()
        
        # Get ancestry for other genome
        other_analyzer = AncestryAnalyzer(other_genome_id)
        other_ancestry = other_analyzer.get_ancestry_tree()
        
        if self_ancestry.empty or other_ancestry.empty:
            return None
        
        # Find common ancestors (genomes that appear in both lineages)
        self_ancestors = set(self_ancestry['genome_id'])
        other_ancestors = set(other_ancestry['genome_id'])
        common_ancestors = self_ancestors & other_ancestors
        
        if not common_ancestors:
            return None
        
        # Find the most recent common ancestor (highest generation number)
        common_df = self_ancestry[self_ancestry['genome_id'].isin(common_ancestors)]
        most_recent = common_df.loc[common_df['generation'].idxmax()]
        
        return {
            'common_ancestor_id': most_recent['genome_id'],
            'generation': most_recent['generation'],
            'fitness': most_recent['fitness'],
            'distance_from_self': self_ancestry['generation'].max() - most_recent['generation'],
            'distance_from_other': other_ancestry['generation'].max() - most_recent['generation'],
            'total_common_ancestors': len(common_ancestors)
        }