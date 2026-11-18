"""
Simple experiment script that works with the existing DatabaseBackpropPopulation

This creates a basic experiment to test the database integration and 
GenomeExplorer functionality without getting into the complexities of
the BackpropPopulation's training logic.
"""
import numpy as np
import neat
import logging
from datetime import datetime
import sys

from explaneat.db import db
from explaneat.db.population import DatabaseBackpropPopulation
from explaneat.analysis import GenomeExplorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_simple_config():
    """Create a basic NEAT configuration"""
    config_text = """
#--- Simple NEAT configuration ---

[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.9
pop_size              = 20
reset_on_extinction   = False

[DefaultGenome]
activation_default      = sigmoid
activation_mutate_rate  = 0.1
activation_options      = sigmoid

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 10.0
bias_min_value          = -10.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.3
conn_delete_prob        = 0.3

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

node_add_prob           = 0.2
node_delete_prob        = 0.2

num_hidden              = 0
num_inputs              = 5
num_outputs             = 1

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 10.0
response_min_value      = -10.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 10
weight_min_value        = -10
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 10
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    
    # Write config
    config_filename = 'simple_config.cfg'
    with open(config_filename, 'w') as f:
        f.write(config_text)
    
    # Load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_filename
    )
    
    return config


def run_simple_experiment(num_generations=10):
    """Run a simple experiment to test the database integration"""
    
    logger.info("🧪 Starting Simple Database Test Experiment")
    logger.info("=" * 50)
    
    # Initialize database
    db.init_db()
    
    # Create simple dummy data (binary format for BackpropPopulation)
    X_dummy = np.random.randn(50, 5)  # 50 samples, 5 features
    y_dummy = np.random.randint(0, 2, (50, 1)).astype(float)  # Binary targets
    
    logger.info(f"Created dummy data: X={X_dummy.shape}, y={y_dummy.shape}")
    
    # Create config
    config = create_simple_config()
    
    # Create experiment
    experiment_name = f"Simple Test {num_generations}gen"
    
    # Create population
    population = DatabaseBackpropPopulation(
        config=config,
        x_train=X_dummy,
        y_train=y_dummy,
        experiment_name=experiment_name,
        dataset_name="Random Binary Data",
        description=f"Simple test for {num_generations} generations"
    )
    
    logger.info(f"Population size: {config.pop_size}")
    
    # Define simple fitness function that works
    def simple_fitness_function(population_dict, config, xs, ys, device):
        """Simple fitness function that assigns reasonable fitness values"""
        fitnesses = []
        
        for genome_id, genome in population_dict.items():
            # Simple fitness based on network structure with some randomness
            num_connections = len([c for c in genome.connections.values() if c.enabled])
            num_nodes = len(genome.nodes)
            
            # Reward moderate complexity
            ideal_connections = 8
            complexity_penalty = abs(num_connections - ideal_connections) / ideal_connections
            
            # Base fitness
            base_fitness = 1.0 / (1.0 + complexity_penalty)
            
            # Add randomness for evolution
            noise = np.random.uniform(0, 0.3)
            fitness = max(0.1, base_fitness + noise)
            
            genome.fitness = fitness
            fitnesses.append(fitness)
            
        logger.info(f"Assigned fitness to {len(fitnesses)} genomes, range: {min(fitnesses):.3f}-{max(fitnesses):.3f}")
    
    # Add simple reporter
    population.add_reporter(neat.StdOutReporter(True))
    
    # Run evolution
    logger.info(f"🔄 Starting evolution for {num_generations} generations...")
    
    try:
        winner = population.run(
            simple_fitness_function,
            n=num_generations,
            nEpochs=5  # Keep backprop epochs low
        )
        
        logger.info("✅ Evolution completed!")
        logger.info(f"🏆 Best fitness: {winner.fitness:.4f}")
        
        # Get experiment ID
        experiment_id = population.experiment_id
        
        # Quick analysis
        logger.info("\n📊 Quick Analysis:")
        explorer = GenomeExplorer.load_best_genome(experiment_id)
        
        # Show summary
        explorer.summary()
        
        # Show visualizations
        logger.info("\n🎨 Generating visualizations...")
        
        # Network structure
        explorer.show_network(figsize=(10, 6))
        
        # Training metrics
        if explorer.genome_info.training_metrics:
            explorer.plot_training_metrics()
        
        # Ancestry (if we have multiple generations)
        ancestry_df = explorer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            explorer.plot_ancestry_fitness()
            
            # Get lineage statistics
            lineage_stats = explorer.ancestry_analyzer.get_lineage_statistics()
            if lineage_stats:
                fp = lineage_stats['fitness_progression']
                logger.info(f"\nFitness progression: {fp['initial_fitness']:.3f} → {fp['final_fitness']:.3f}")
                logger.info(f"Trend: {fp['fitness_trend']}")
        
        logger.info(f"\n✅ Experiment ID: {experiment_id}")
        
        return winner, experiment_id
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simple database test experiment')
    parser.add_argument('--generations', type=int, default=10,
                      help='Number of generations (default: 10)')
    
    args = parser.parse_args()
    
    winner, experiment_id = run_simple_experiment(args.generations)
    
    print("\n" + "=" * 50)
    print("🧪 SIMPLE EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Experiment ID: {experiment_id}")
    print(f"Best fitness: {winner.fitness:.4f}")
    print(f"\nUse GenomeExplorer to analyze:")
    print(f"  explorer = GenomeExplorer.load_best_genome('{experiment_id}')")


if __name__ == "__main__":
    main()