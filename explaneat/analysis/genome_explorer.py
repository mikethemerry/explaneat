"""
GenomeExplorer: Main class for analyzing individual genomes and their ancestry

This class provides a comprehensive API for loading genomes from the database
and exploring their evolutionary history, performance, and network structure.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..db import db, Experiment, Population, Genome, TrainingMetric, Result
from ..core.explaneat import ExplaNEAT
from .ancestry_analyzer import AncestryAnalyzer
from .visualization import GenomeVisualizer

@dataclass
class GenomeInfo:
    """Information about a genome and its context"""
    genome_id: str
    neat_genome_id: int
    fitness: float
    generation: int
    experiment_id: str
    experiment_name: str
    population_id: str
    network_stats: Dict[str, Any]
    training_metrics: List[Dict[str, Any]]
    genome_data: Dict[str, Any]

class GenomeExplorer:
    """
    Main class for exploring individual genomes and their evolutionary history.
    
    Usage:
        explorer = GenomeExplorer.load_genome(genome_id)
        explorer.show_network()
        explorer.plot_ancestry_fitness()
        explorer.trace_gene_origins()
    """
    
    def __init__(self, genome_info: GenomeInfo, neat_genome, config):
        self.genome_info = genome_info
        self.neat_genome = neat_genome
        self.config = config
        self._ancestry_analyzer = None
        self._visualizer = None
        self._explainer = None
        
    @classmethod
    def load_genome(cls, genome_id: str):
        """
        Load a genome from the database and create an explorer instance.
        
        Args:
            genome_id: Database UUID of the genome to load
            
        Returns:
            GenomeExplorer instance
        """
        with db.session_scope() as session:
            # Get genome record
            genome_record = session.get(Genome, genome_id)
            if not genome_record:
                raise ValueError(f"Genome {genome_id} not found in database")
            
            # Get population and experiment info
            population = session.get(Population, genome_record.population_id)
            experiment = session.get(Experiment, population.experiment_id)
            
            # Get training metrics
            training_metrics = session.query(TrainingMetric).filter_by(
                genome_id=genome_id
            ).order_by(TrainingMetric.epoch).all()
            
            # Load NEAT config (simplified - you might want to store this properly)
            import neat
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                "config-file.cfg"
            )
            
            # Deserialize the NEAT genome
            neat_genome = genome_record.to_neat_genome(config)
            
            # Create genome info
            genome_info = GenomeInfo(
                genome_id=str(genome_record.id),
                neat_genome_id=genome_record.genome_id,
                fitness=genome_record.fitness or 0.0,
                generation=population.generation,
                experiment_id=str(experiment.id),
                experiment_name=experiment.name,
                population_id=str(population.id),
                network_stats={
                    'num_nodes': genome_record.num_nodes,
                    'num_connections': genome_record.num_connections,
                    'num_enabled_connections': genome_record.num_enabled_connections,
                    'network_depth': genome_record.network_depth,
                    'network_width': genome_record.network_width
                },
                training_metrics=[
                    {
                        'epoch': tm.epoch,
                        'loss': tm.loss,
                        'accuracy': tm.accuracy,
                        'additional_metrics': tm.additional_metrics
                    } for tm in training_metrics
                ],
                genome_data=genome_record.genome_data
            )
            
        return cls(genome_info, neat_genome, config)
    
    @classmethod
    def load_best_genome(cls, experiment_id: str):
        """Load the best genome from an experiment"""
        with db.session_scope() as session:
            best_genome = session.query(Genome).join(Population).filter(
                Population.experiment_id == experiment_id,
                Genome.fitness.isnot(None)
            ).order_by(Genome.fitness.desc()).first()
            
            if not best_genome:
                raise ValueError(f"No genomes found in experiment {experiment_id}")
            
            # Get the ID while still in session
            best_genome_id = str(best_genome.id)
                
        return cls.load_genome(best_genome_id)
    
    @classmethod
    def list_experiments(cls) -> pd.DataFrame:
        """Get a list of all experiments in the database"""
        with db.session_scope() as session:
            experiments = session.query(Experiment).order_by(Experiment.created_at.desc()).all()
            
            data = []
            for exp in experiments:
                # Get some basic stats
                populations = session.query(Population).filter_by(experiment_id=exp.id).count()
                genomes = session.query(Genome).join(Population).filter(
                    Population.experiment_id == exp.id
                ).count()
                
                # Get best fitness
                best_genome = session.query(Genome).join(Population).filter(
                    Population.experiment_id == exp.id,
                    Genome.fitness.isnot(None)
                ).order_by(Genome.fitness.desc()).first()
                
                data.append({
                    'experiment_id': str(exp.id),
                    'name': exp.name,
                    'status': exp.status,
                    'dataset': exp.dataset_name,
                    'generations': populations,
                    'total_genomes': genomes,
                    'best_fitness': best_genome.fitness if best_genome else None,
                    'created_at': exp.created_at,
                    'duration': exp.end_time - exp.start_time if exp.end_time else None
                })
                
        return pd.DataFrame(data)
    
    # Properties for easy access
    @property
    def ancestry_analyzer(self) -> AncestryAnalyzer:
        """Lazy-loaded ancestry analyzer"""
        if self._ancestry_analyzer is None:
            self._ancestry_analyzer = AncestryAnalyzer(self.genome_info.genome_id)
        return self._ancestry_analyzer
    
    @property
    def visualizer(self) -> GenomeVisualizer:
        """Lazy-loaded genome visualizer"""
        if self._visualizer is None:
            self._visualizer = GenomeVisualizer(self.neat_genome, self.config)
        return self._visualizer
    
    @property
    def explainer(self) -> ExplaNEAT:
        """Lazy-loaded ExplaNEAT explainer"""
        if self._explainer is None:
            self._explainer = ExplaNEAT(self.neat_genome, self.config)
        return self._explainer
    
    # Core analysis methods
    def summary(self) -> None:
        """Print a comprehensive summary of this genome"""
        print("🧬 GENOME SUMMARY")
        print("=" * 50)
        print(f"Genome ID: {self.genome_info.neat_genome_id}")
        print(f"Database ID: {self.genome_info.genome_id}")
        print(f"Fitness: {self.genome_info.fitness:.4f}")
        print(f"Generation: {self.genome_info.generation}")
        print(f"Experiment: {self.genome_info.experiment_name}")
        print()
        
        print("📊 NETWORK STRUCTURE")
        print("-" * 30)
        stats = self.genome_info.network_stats
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Connections: {stats['num_connections']} ({stats['num_enabled_connections']} enabled)")
        print(f"Depth: {stats['network_depth']}")
        print(f"Width: {stats['network_width']}")
        print()
        
        # ExplaNEAT analysis
        try:
            print("🔍 EXPLANEAT ANALYSIS")
            print("-" * 30)
            print(f"Density: {self.explainer.density():.4f}")
            print(f"Skippiness: {self.explainer.skippines():.4f}")
            print(f"Parameters: {self.explainer.n_genome_params()}")
        except Exception as e:
            print(f"ExplaNEAT analysis failed: {e}")
        print()
        
        # Training metrics
        if self.genome_info.training_metrics:
            print("📈 TRAINING METRICS")
            print("-" * 30)
            metrics_df = pd.DataFrame(self.genome_info.training_metrics)
            if 'loss' in metrics_df.columns:
                print(f"Loss range: {metrics_df['loss'].min():.4f} - {metrics_df['loss'].max():.4f}")
            if 'accuracy' in metrics_df.columns:
                print(f"Accuracy range: {metrics_df['accuracy'].min():.4f} - {metrics_df['accuracy'].max():.4f}")
            print(f"Training epochs: {len(metrics_df)}")
    
    def get_ancestry_tree(self, max_generations: int = 10) -> pd.DataFrame:
        """Get the ancestry tree as a DataFrame"""
        return self.ancestry_analyzer.get_ancestry_tree(max_generations)
    
    def plot_training_metrics(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot training metrics over epochs"""
        if not self.genome_info.training_metrics:
            print("No training metrics available for this genome")
            return
            
        df = pd.DataFrame(self.genome_info.training_metrics)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        if 'loss' in df.columns and df['loss'].notna().any():
            axes[0].plot(df['epoch'], df['loss'], 'b-o', linewidth=2, markersize=4)
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'accuracy' in df.columns and df['accuracy'].notna().any():
            axes[1].plot(df['epoch'], df['accuracy'], 'g-o', linewidth=2, markersize=4)
            axes[1].set_title('Training Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Metrics - Genome {self.genome_info.neat_genome_id}')
        plt.tight_layout()
        plt.show()
    
    def plot_ancestry_fitness(self, max_generations: int = 10, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot fitness progression through ancestry"""
        ancestry_df = self.get_ancestry_tree(max_generations)
        
        if ancestry_df.empty:
            print("No ancestry data available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Fitness over generations
        ax1.plot(ancestry_df['generation'], ancestry_df['fitness'], 'o-', linewidth=2, markersize=6)
        ax1.set_title('Fitness Through Ancestry')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.grid(True, alpha=0.3)
        
        # Mark current genome
        current_gen = self.genome_info.generation
        current_fitness = self.genome_info.fitness
        ax1.scatter([current_gen], [current_fitness], color='red', s=100, zorder=5, label='Current Genome')
        ax1.legend()
        
        # Network complexity over generations
        ax2.plot(ancestry_df['generation'], ancestry_df['num_connections'], 'g-o', label='Connections', linewidth=2)
        ax2.plot(ancestry_df['generation'], ancestry_df['num_nodes'], 'b-s', label='Nodes', linewidth=2)
        ax2.set_title('Network Complexity Through Ancestry')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_network(self, **kwargs) -> None:
        """Display the network structure"""
        self.visualizer.plot_network(**kwargs)
    
    def trace_gene_origins(self) -> pd.DataFrame:
        """Trace when each gene (node/connection) was first introduced"""
        return self.ancestry_analyzer.trace_gene_origins(self.neat_genome)
    
    def compare_with_ancestor(self, ancestor_generation: int) -> Dict[str, Any]:
        """Compare this genome with a specific ancestor"""
        return self.ancestry_analyzer.compare_with_ancestor(
            self.neat_genome, ancestor_generation
        )
    
    def get_performance_context(self) -> Dict[str, Any]:
        """Get performance context within the population and experiment"""
        with db.session_scope() as session:
            # Get population stats
            population = session.get(Population, self.genome_info.population_id)
            
            # Get all genomes in the same generation
            generation_genomes = session.query(Genome).filter_by(
                population_id=self.genome_info.population_id
            ).all()
            
            # Get experiment progression
            experiment_populations = session.query(Population).filter_by(
                experiment_id=self.genome_info.experiment_id
            ).order_by(Population.generation).all()
            
            fitness_values = [g.fitness for g in generation_genomes if g.fitness is not None]
            
            return {
                'generation_rank': sorted(fitness_values, reverse=True).index(self.genome_info.fitness) + 1,
                'generation_size': len(generation_genomes),
                'generation_best': max(fitness_values) if fitness_values else None,
                'generation_mean': np.mean(fitness_values) if fitness_values else None,
                'generation_std': np.std(fitness_values) if fitness_values else None,
                'experiment_generations': len(experiment_populations),
                'experiment_best_fitness': population.best_fitness,
                'is_generation_best': self.genome_info.fitness == max(fitness_values) if fitness_values else False
            }
    
    def export_genome_data(self) -> Dict[str, Any]:
        """Export all genome data for external analysis"""
        return {
            'genome_info': self.genome_info.__dict__,
            'neat_genome_nodes': {
                str(node_id): {
                    'bias': node.bias,
                    'response': node.response,
                    'activation': node.activation,
                    'aggregation': node.aggregation
                } for node_id, node in self.neat_genome.nodes.items()
            },
            'neat_genome_connections': {
                f"{conn_key[0]}_{conn_key[1]}": {
                    'weight': conn.weight,
                    'enabled': conn.enabled
                } for conn_key, conn in self.neat_genome.connections.items()
            },
            'ancestry_tree': self.get_ancestry_tree().to_dict('records'),
            'performance_context': self.get_performance_context(),
            'gene_origins': self.trace_gene_origins().to_dict('records')
        }