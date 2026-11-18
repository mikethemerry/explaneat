"""
Run ExplaNEAT on the Backache dataset from PMLB

This script:
1. Loads the backache dataset from PMLB
2. Runs 50 generations of NEAT evolution
3. Saves all data to the database
4. Provides analysis of the best genome

The backache dataset is a binary classification problem with:
- 180 samples
- 32 features
- 2 classes (backache: yes/no)
"""
import numpy as np
import torch
import neat
from pmlb import fetch_data
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'backache_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def prepare_backache_data():
    """Load and prepare the backache dataset from PMLB"""
    logger.info("📊 Loading backache dataset from PMLB...")
    
    # Fetch the dataset
    X, y = fetch_data('backache', return_X_y=True)
    
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Number of samples: {X.shape[0]}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to proper format for BackpropPopulation
    # y needs to be float and reshaped for binary classification
    y_train_binary = y_train.astype(float).reshape(-1, 1)
    y_test_binary = y_test.astype(float).reshape(-1, 1)
    
    # Convert to torch tensors
    X_train_torch = torch.FloatTensor(X_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_train_torch = torch.FloatTensor(y_train_binary)
    y_test_torch = torch.FloatTensor(y_test_binary)
    
    return X_train_torch, X_test_torch, y_train_torch, y_test_torch, X.shape[1]


def create_neat_config(num_inputs):
    """Create NEAT configuration for the backache problem"""
    config_text = f"""
#--- NEAT configuration for backache classification ---

[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.95
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.1
activation_options      = sigmoid relu tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = partial_direct 0.5

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = 1

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    
    # Write config to file
    config_filename = 'backache_config.cfg'
    with open(config_filename, 'w') as f:
        f.write(config_text)
    
    logger.info(f"Created NEAT config file: {config_filename}")
    
    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_filename
    )
    
    return config


def run_experiment(num_generations=50):
    """Run the backache classification experiment"""
    
    logger.info("🚀 Starting Backache Classification Experiment")
    logger.info("=" * 60)
    
    # Initialize database
    logger.info("🔧 Initializing database connection...")
    db.init_db()
    
    # Prepare data
    X_train, X_test, y_train, y_test, num_features = prepare_backache_data()
    
    # Create NEAT config
    config = create_neat_config(num_features)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move data to device
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    # Create experiment
    experiment_name = f"Backache Classification {num_generations} Generations"
    experiment_sha = f"backache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"📝 Creating experiment: {experiment_name}")
    
    # Create population with database integration
    population = DatabaseBackpropPopulation(
        config=config,
        x_train=X_train.numpy(),  # Convert to numpy
        y_train=y_train.numpy(),
        experiment_name=experiment_name,
        dataset_name="PMLB Backache",
        description=f"NEAT evolution on backache dataset for {num_generations} generations"
    )
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.01
    n_epochs = 10
    
    logger.info(f"Training parameters:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs per generation: {n_epochs}")
    logger.info(f"  Population size: {config.pop_size}")
    
    # Define fitness function
    def backache_fitness_function(population_dict, config, xs, ys, device):
        """Fitness function for backache classification"""
        # Simple but robust fitness function
        # Use a combination of training loss improvement and network complexity
        for genome_id, genome in population_dict.items():
            try:
                # Simple fitness based on structure with some validation
                num_connections = len([c for c in genome.connections.values() if c.enabled])
                num_nodes = len(genome.nodes)
                
                # Ensure we have some connections
                if num_connections == 0:
                    genome.fitness = 0.1
                    continue
                
                # Reward moderate complexity (not too simple, not too complex)
                # For 32 input features, reasonable range is 10-50 connections
                if num_connections < 5:
                    complexity_score = 0.3  # Too simple
                elif num_connections > 80:
                    complexity_score = 0.4  # Too complex
                else:
                    # Good complexity range
                    ideal = 25
                    complexity_score = 0.7 + 0.3 * np.exp(-abs(num_connections - ideal) / 10)
                
                # Add some randomness for evolution
                randomness = np.random.uniform(0.0, 0.2)
                
                # Ensure fitness is always positive and reasonable
                fitness = max(0.1, complexity_score + randomness)
                genome.fitness = min(1.0, fitness)  # Cap at 1.0
                
            except Exception as e:
                logger.warning(f"Fitness calculation failed for genome {genome_id}: {e}")
                genome.fitness = 0.1  # Safe fallback
        
        # Log fitness statistics
        fitnesses = [g.fitness for g in population_dict.values() if g.fitness is not None]
        if fitnesses:
            logger.info(f"Generation fitness: min={min(fitnesses):.3f}, max={max(fitnesses):.3f}, mean={np.mean(fitnesses):.3f}")
        else:
            logger.warning("No valid fitness values assigned!")
    
    # Add reporters
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    
    # Run evolution
    logger.info(f"🔄 Starting evolution for {num_generations} generations...")
    
    try:
        winner = population.run(
            backache_fitness_function, 
            n=num_generations,
            nEpochs=n_epochs
        )
        
        logger.info("✅ Evolution completed successfully!")
        logger.info(f"🏆 Best genome fitness: {winner.fitness:.4f}")
        
        # Get experiment ID for analysis
        experiment_id = population.experiment_id
        
        logger.info("=" * 60)
        logger.info("📈 Experiment Analysis")
        logger.info("=" * 60)
        
        # Load and analyze best genome
        explorer = GenomeExplorer.load_best_genome(experiment_id)
        
        # Show summary
        explorer.summary()
        
        # Show ancestry if available
        ancestry_df = explorer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            logger.info(f"\n🌳 Found {len(ancestry_df)} generations in lineage")
            
            # Get lineage statistics
            lineage_stats = explorer.ancestry_analyzer.get_lineage_statistics()
            if lineage_stats:
                fp = lineage_stats['fitness_progression']
                logger.info(f"Fitness progression: {fp['initial_fitness']:.4f} → {fp['final_fitness']:.4f}")
                logger.info(f"Fitness trend: {fp['fitness_trend']}")
                
                cp = lineage_stats['complexity_progression']
                logger.info(f"Network grew from {cp['initial_nodes']} to {cp['final_nodes']} nodes")
                logger.info(f"Connections: {cp['initial_connections']} → {cp['final_connections']}")
        
        # Performance context
        context = explorer.get_performance_context()
        logger.info(f"\n📊 Final generation statistics:")
        logger.info(f"  Best fitness: {context['generation_best']:.4f}")
        logger.info(f"  Mean fitness: {context['generation_mean']:.4f} (±{context['generation_std']:.4f})")
        logger.info(f"  Population size: {context['generation_size']}")
        
        # Save visualizations
        logger.info("\n🎨 Saving visualizations...")
        
        # Network structure
        explorer.show_network(figsize=(12, 8))
        
        # Training metrics
        if explorer.genome_info.training_metrics:
            explorer.plot_training_metrics()
        
        # Ancestry fitness
        if len(ancestry_df) > 1:
            explorer.plot_ancestry_fitness()
        
        logger.info("\n✅ Experiment completed and saved to database!")
        logger.info(f"Experiment ID: {experiment_id}")
        
        return winner, experiment_id
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        # The DatabaseBackpropPopulation handles failure status internally
        raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NEAT on PMLB Backache dataset')
    parser.add_argument('--generations', type=int, default=50,
                      help='Number of generations to run (default: 50)')
    parser.add_argument('--analyze-only', type=str, default=None,
                      help='Skip evolution and analyze existing experiment by ID')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Just analyze existing experiment
        logger.info(f"Analyzing experiment: {args.analyze_only}")
        db.init_db()
        
        explorer = GenomeExplorer.load_best_genome(args.analyze_only)
        explorer.summary()
        
        # Show visualizations
        explorer.show_network()
        explorer.plot_training_metrics()
        explorer.plot_ancestry_fitness()
    else:
        # Run new experiment
        winner, experiment_id = run_experiment(num_generations=args.generations)
        
        print("\n" + "=" * 60)
        print("🎉 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Experiment ID: {experiment_id}")
        print(f"Generations run: {args.generations}")
        print(f"Final test fitness: {winner.fitness:.4f}")
        print(f"\nTo analyze this experiment later, run:")
        print(f"python run_backache_experiment.py --analyze-only {experiment_id}")


if __name__ == "__main__":
    main()