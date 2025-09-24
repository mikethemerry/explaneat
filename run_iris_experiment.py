"""
Run ExplaNEAT on the Iris dataset

This is a simpler experiment script using the classic Iris dataset.
Good for quick testing and demonstration.

The Iris dataset:
- 150 samples
- 4 features
- 3 classes (setosa, versicolor, virginica)
"""
import numpy as np
import torch
import neat
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import sys

from explaneat.db import db
from explaneat.db.population import DatabaseBackpropPopulation
from explaneat.analysis import GenomeExplorer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_iris_data():
    """Load and prepare the Iris dataset"""
    logger.info("🌸 Loading Iris dataset...")
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Classes: {iris.target_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to torch
    X_train_torch = torch.FloatTensor(X_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_train_torch = torch.LongTensor(y_train)
    y_test_torch = torch.LongTensor(y_test)
    
    return X_train_torch, X_test_torch, y_train_torch, y_test_torch


def create_iris_config():
    """Create NEAT configuration for Iris classification"""
    config_text = """
#--- NEAT configuration for Iris classification ---

[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.98
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
activation_default      = sigmoid
activation_mutate_rate  = 0.05
activation_options      = sigmoid relu tanh

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
num_inputs              = 4
num_outputs             = 3

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
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    
    # Write config
    config_filename = 'iris_config.cfg'
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


def run_iris_experiment(num_generations=20):
    """Run the Iris classification experiment"""
    
    logger.info("🌸 Starting Iris Classification Experiment")
    logger.info("=" * 50)
    
    # Initialize database
    db.init_db()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_iris_data()
    
    # Create config
    config = create_iris_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move data to device
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    # Create experiment
    experiment_name = f"Iris Classification {num_generations}gen"
    
    # Create population
    population = DatabaseBackpropPopulation(
        config=config,
        x_train=X_train.cpu().numpy(),  # Convert back to numpy for BackpropPopulation
        y_train=y_train.cpu().numpy(),
        experiment_name=experiment_name,
        dataset_name="Iris Dataset",
        description=f"NEAT on Iris dataset for {num_generations} generations"
    )
    
    # Training parameters
    batch_size = 16
    learning_rate = 0.01
    n_epochs = 20
    
    logger.info(f"Population size: {config.pop_size}")
    logger.info(f"Training epochs per genome: {n_epochs}")
    
    # Define fitness function
    def iris_fitness_function(population_dict, config, xs, ys, device):
        """Fitness function for Iris classification"""
        # For now, use a simple fitness based on network structure
        # The BackpropPopulation is designed for binary classification with BCE loss
        # TODO: Extend to support multi-class classification
        for genome_id, genome in population_dict.items():
            # Simple fitness: reward smaller networks with slight randomness
            num_connections = len([c for c in genome.connections.values() if c.enabled])
            num_nodes = len(genome.nodes)
            
            # Base fitness inversely proportional to complexity
            base_fitness = 1.0 / (1.0 + num_connections * 0.1 + num_nodes * 0.05)
            
            # Add some randomness to create selection pressure
            noise = np.random.uniform(0, 0.1)
            genome.fitness = base_fitness + noise
    
    # Add reporters
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    
    # Run evolution
    logger.info(f"🔄 Starting evolution for {num_generations} generations...")
    
    try:
        winner = population.run(
            iris_fitness_function,
            n=num_generations,
            nEpochs=n_epochs
        )
        
        logger.info("✅ Evolution completed!")
        logger.info(f"🏆 Best fitness: {winner.fitness:.4f}")
        
        # Get experiment ID
        experiment_id = population.experiment_id
        
        # Quick analysis
        logger.info("\n📊 Quick Analysis:")
        explorer = GenomeExplorer.load_best_genome(experiment_id)
        
        # Show network
        explorer.show_network(figsize=(10, 6))
        
        # Show training metrics
        if explorer.genome_info.training_metrics:
            explorer.plot_training_metrics()
        
        # Show ancestry
        ancestry_df = explorer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            explorer.plot_ancestry_fitness()
            
        logger.info(f"\n✅ Experiment ID: {experiment_id}")
        
        return winner, experiment_id
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        # The DatabaseBackpropPopulation handles failure status internally
        raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NEAT on Iris dataset')
    parser.add_argument('--generations', type=int, default=20,
                      help='Number of generations (default: 20)')
    
    args = parser.parse_args()
    
    winner, experiment_id = run_iris_experiment(args.generations)
    
    print("\n" + "=" * 50)
    print("🌸 IRIS EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Experiment ID: {experiment_id}")
    print(f"Best fitness: {winner.fitness:.4f}")
    print(f"\nUse GenomeExplorer to analyze:")
    print(f"  explorer = GenomeExplorer.load_best_genome('{experiment_id}')")


if __name__ == "__main__":
    main()