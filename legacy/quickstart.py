import numpy as np
import torch
import neat
import logging
import sys
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.backproppop import BackpropPopulation
from explaneat.evaluators.evaluators import binary_cross_entropy

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting ExplaNEAT quickstart script")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load your data
logger.info("Generating example data...")
X_train = np.random.randn(100, 10)  # Example data
y_train = np.random.randint(0, 2, (100, 1))
logger.info(f"Data shape - X: {X_train.shape}, y: {y_train.shape}")
logger.info(f"Data types - X: {X_train.dtype}, y: {y_train.dtype}")

# Configure NEAT
logger.info("Loading NEAT configuration from config-file.cfg...")
try:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-file.cfg",  # Your NEAT config file
    )
    logger.info("NEAT configuration loaded successfully")
    logger.info(f"Population size: {config.pop_size}")
    logger.info(f"Fitness threshold: {config.fitness_threshold}")
except Exception as e:
    logger.error(f"Failed to load NEAT configuration: {e}")
    raise

# Create population
logger.info("Creating BackpropPopulation...")
try:
    population = BackpropPopulation(config, X_train, y_train)
    logger.info("BackpropPopulation created successfully")
    
    # Add reporters to track progress
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    logger.info("Added NEAT reporters for progress tracking")
except Exception as e:
    logger.error(f"Failed to create BackpropPopulation: {e}")
    raise

# Evolve the network
logger.info("Starting evolution process...")
logger.info(f"Running for maximum {10} generations with {5} epochs per generation")
try:
    winner = population.run(binary_cross_entropy, n=10, nEpochs=5)
    logger.info("Evolution completed successfully")
    logger.info(f"Winner fitness: {winner.fitness if hasattr(winner, 'fitness') else 'N/A'}")
except Exception as e:
    logger.error(f"Evolution failed: {e}")
    raise

# Create explainable model
logger.info("Creating ExplaNEAT explainer...")
try:
    explainer = ExplaNEAT(winner, config)
    logger.info("ExplaNEAT explainer created successfully")
except Exception as e:
    logger.error(f"Failed to create ExplaNEAT explainer: {e}")
    raise

# Analyze the network
logger.info("Analyzing the evolved network...")
try:
    depth = explainer.depth()
    logger.info(f"Network depth: {depth}")
    print(f"Network depth: {depth}")
    
    density = explainer.density()
    logger.info(f"Network density: {density}")
    print(f"Network density: {density}")
    
    skippiness = explainer.skippines()
    logger.info(f"Network skippiness: {skippiness}")
    print(f"Network skippiness: {skippiness}")
    
    n_params = explainer.n_genome_params()
    logger.info(f"Number of parameters: {n_params}")
    print(f"Number of parameters: {n_params}")
    
    logger.info("Quickstart completed successfully!")
except Exception as e:
    logger.error(f"Failed to analyze network: {e}")
    raise
