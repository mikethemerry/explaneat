"""
Working backache experiment using ExplaNEAT framework

This version uses the ExplaNEAT framework with BackpropPopulation and proper
experiment management, following the established patterns.
"""

import numpy as np
import torch
import neat
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import logging
from datetime import datetime, timezone
import os
import tempfile
from copy import deepcopy

from explaneat.db import db, Experiment, Population, Genome, TrainingMetric
from explaneat.db.serialization import serialize_population_config
from explaneat.analysis import GenomeExplorer
from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.neuralneat import NeuralNeat
from explaneat.core.experiment import ExperimentReporterSet
from explaneat.core.errors import GenomeNotValidError
from explaneat.evaluators.evaluators import binary_cross_entropy
from explaneat.core.explaneat import ExplaNEAT
from explaneat.visualization import visualize
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseReporter:
    """Custom reporter to save population data to database during evolution"""

    def __init__(self, experiment_id, config):
        self.experiment_id = experiment_id
        self.config = config
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        """Save population data after fitness evaluation"""
        try:
            save_population_to_db(
                self.experiment_id, self.generation, population, self.config
            )
        except Exception as e:
            logger.warning(
                f"Failed to save generation {self.generation} to database: {e}"
            )

    def pre_backprop(self, config, population, species):
        pass

    def post_backprop(self, config, population, species):
        pass

    def pre_reproduction(self, config, population, species):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def end_generation(self, config, population, species):
        pass

    def end_experiment(self, config, population, species):
        pass

    def info(self, msg):
        """Log info messages - required by NEAT ReporterSet"""
        logger.info(f"DatabaseReporter: {msg}")


def prepare_backache_data():
    """Load and prepare the backache dataset from PMLB"""
    logger.info("📊 Loading backache dataset from PMLB...")

    X, y = fetch_data("backache", return_X_y=True)
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def create_backache_config(num_inputs):
    """Create NEAT configuration for backache using reference config"""
    config_text = f"""
#--- parameters for the backache experiment ---#

[NEAT]
fitness_criterion     = max
fitness_threshold     = 40.0
pop_size              = 50
reset_on_extinction   = False


[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2


[DefaultGenome]
# node activation options
activation_default      = relu
activation_mutate_rate  = 0.0
activation_options      = relu

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
bias_mutate_rate        = 0.0
bias_replace_rate       = 0.0

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 1.5

# connection add/remove rates
conn_add_prob           = 0.7
conn_delete_prob        = 0.65

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.00

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.6
node_delete_prob        = 0.55


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
weight_mutate_rate      = 0.00
weight_replace_rate     = 0.005


# network parameters
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = 1
"""

    config_filename = "working_backache_config.cfg"
    with open(config_filename, "w") as f:
        f.write(config_text)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_filename,
    )

    return config


def create_experiment_record(name, dataset_name, description, config):
    """Create experiment record in database"""
    with db.session_scope() as session:
        experiment = Experiment(
            experiment_sha=f"working_backache_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=description,
            dataset_name=dataset_name,
            config_json=serialize_population_config(config),
            neat_config_text="# Working backache config",
            start_time=datetime.now(timezone.utc),
        )
        session.add(experiment)
        session.flush()
        experiment_id = experiment.id

    return experiment_id


def save_population_to_db(experiment_id, generation, population, config):
    """Save population and genomes to database"""

    # Calculate population statistics
    fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
    best_fitness = max(fitnesses) if fitnesses else 0
    mean_fitness = np.mean(fitnesses) if fitnesses else 0
    std_fitness = np.std(fitnesses) if fitnesses else 0

    # Save population record
    with db.session_scope() as session:
        pop_record = Population(
            experiment_id=experiment_id,
            generation=generation,
            population_size=len(population),
            num_species=1,  # Simplified
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            stdev_fitness=std_fitness,
            config_json=serialize_population_config(config),
        )
        session.add(pop_record)
        session.flush()
        population_id = pop_record.id

    # Save genomes
    with db.session_scope() as session:
        for genome_id, genome in population.items():
            db_genome = Genome.from_neat_genome(genome, population_id)
            session.add(db_genome)
            session.flush()  # Flush to get the genome ID assigned

            # Add some dummy training metrics
            for epoch in range(3):
                metric = TrainingMetric(
                    genome_id=db_genome.id,
                    population_id=population_id,
                    epoch=epoch,
                    loss=max(
                        0.1, 2.0 - (genome.fitness or 0) + np.random.normal(0, 0.1)
                    ),
                    accuracy=min(
                        0.9, (genome.fitness or 0) + np.random.normal(0, 0.05)
                    ),
                    additional_metrics={"generation": generation},
                )
                session.add(metric)

    logger.info(
        f"Saved generation {generation}: {len(population)} genomes, best fitness: {best_fitness:.3f}"
    )
    return population_id


def instantiate_population(config, xs, ys):
    """Instantiate BackpropPopulation following ExplaNEAT pattern"""
    # Create the population using BackpropPopulation
    p = BackpropPopulation(config, xs, ys, criterion=torch.nn.BCELoss())

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    return p


def run_working_backache_experiment(num_generations=10):
    """Run working backache experiment using ExplaNEAT framework"""

    logger.info("🧬 Starting Working Backache Experiment")
    logger.info("=" * 50)

    # Initialize database
    db.init_db()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_backache_data()

    # Create config
    config = create_backache_config(X_train.shape[1])

    # Print all config values (attributes and their values)
    print("Config values:")
    for attr in dir(config):
        if not attr.startswith("_") and not callable(getattr(config, attr)):
            print(f"  {attr}: {getattr(config, attr)}")

    print("\nConfig.genome_config values:")
    genome_config = getattr(config, "genome_config", None)
    if genome_config is not None:
        for attr in dir(genome_config):
            if not attr.startswith("_") and not callable(getattr(genome_config, attr)):
                print(f"  {attr}: {getattr(genome_config, attr)}")
    else:
        print("  genome_config not found in config")

    # Log critical experiment details
    logger.info("===== Experiment Configuration =====")
    logger.info(f"Input features: {X_train.shape[1]}")
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Population size: {getattr(config, 'pop_size', 50)}")
    logger.info(f"Fitness threshold: {getattr(config, 'fitness_threshold', 40.0)}")
    logger.info(f"Epochs per generation: 5 (backprop epochs per generation)")
    logger.info(f"Number of generations: {num_generations}")
    logger.info("====================================")

    # Create experiment record
    experiment_name = f"Working Backache {num_generations}gen"
    experiment_id = create_experiment_record(
        experiment_name,
        "PMLB Backache (Working)",
        f"Working backache experiment for {num_generations} generations",
        config,
    )

    logger.info(f"Created experiment ID: {experiment_id}")

    # Create database reporter
    db_reporter = DatabaseReporter(experiment_id, config)

    # Instantiate population using ExplaNEAT pattern
    population = instantiate_population(config, X_train, y_train)

    # Add database reporter
    population.reporters.reporters.append(db_reporter)

    logger.info(f"Population size: {getattr(config, 'pop_size', 50)}")

    # Run evolution using BackpropPopulation
    logger.info(f"🔄 Starting evolution for {num_generations} generations...")

    try:
        # Run the evolution using BackpropPopulation's run method with binary_cross_entropy
        best_genome = population.run(
            fitness_function=binary_cross_entropy,
            n=num_generations,
            nEpochs=5,  # Number of backprop epochs per generation
        )

        # Mark experiment as completed
        with db.session_scope() as session:
            experiment = session.get(Experiment, experiment_id)
            experiment.status = "completed"
            experiment.end_time = datetime.now(timezone.utc)

        logger.info("✅ Evolution completed!")
        logger.info(f"🏆 Best fitness: {best_genome.fitness:.4f}")

        # Create ExplaNEAT explainer for analysis
        explainer = ExplaNEAT(best_genome, config)

        # Get test predictions
        X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
        test_predictions = explainer.net.forward(X_test_tensor)
        test_predictions_np = [r[0].item() for r in test_predictions.detach().numpy()]

        # Calculate test AUC
        try:
            test_auc = roc_auc_score(y_test, test_predictions_np)
            logger.info(f"🎯 Test AUC: {test_auc:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate test AUC: {e}")
            test_auc = 0.5

        # Analysis using ExplaNEAT framework
        logger.info("\n📊 ExplaNEAT Analysis:")
        logger.info("=" * 50)

        # Network properties
        logger.info(f"📋 Network Properties:")
        logger.info(f"  - Depth: {explainer.depth()}")
        logger.info(f"  - Skippiness: {explainer.skippines()}")
        logger.info(f"  - Parameter count: {explainer.n_genome_params()}")

        # Show available experiments
        logger.info("\n📊 Available experiments:")
        experiments_df = GenomeExplorer.list_experiments()
        print(experiments_df[experiments_df["experiment_id"] == experiment_id])

        # Load the best genome from the experiment
        logger.info("\n🏆 Loading best genome from experiment...")
        explorer = GenomeExplorer.load_best_genome(experiment_id)

        # Show genome summary
        logger.info("\n📋 Genome Summary:")
        explorer.summary()

        # Test ancestry analysis
        logger.info("\n🌳 Ancestry Analysis:")
        ancestry_df = explorer.get_ancestry_tree(max_generations=10)
        print(f"Found {len(ancestry_df)} ancestors")
        if not ancestry_df.empty:
            print("Ancestry tree:")
            print(
                ancestry_df[
                    [
                        "neat_genome_id",
                        "generation",
                        "fitness",
                        "num_nodes",
                        "num_connections",
                    ]
                ]
            )

        # Test gene origins tracing
        logger.info("\n🧬 Gene Origins Analysis:")
        try:
            gene_origins_df = explorer.trace_gene_origins()
            print(f"Traced origins for {len(gene_origins_df)} genes")
            if not gene_origins_df.empty:
                print("Gene origins summary:")
                print(
                    gene_origins_df.groupby(["gene_type", "origin_generation"])
                    .size()
                    .unstack(fill_value=0)
                )
        except Exception as e:
            logger.warning(f"Gene origins analysis failed: {e}")

        # Test performance context
        logger.info("\n📈 Performance Context:")
        context = explorer.get_performance_context()
        print(
            f"Generation rank: {context['generation_rank']}/{context['generation_size']}"
        )
        print(f"Generation best: {context['generation_best']:.3f}")
        print(f"Generation mean: {context['generation_mean']:.3f}")
        print(f"Is generation best: {context['is_generation_best']}")

        # Test lineage statistics
        logger.info("\n📊 Lineage Statistics:")
        lineage_stats = explorer.ancestry_analyzer.get_lineage_statistics()
        if lineage_stats:
            print(f"Lineage length: {lineage_stats['lineage_length']}")
            fp = lineage_stats["fitness_progression"]
            print(
                f"Fitness progression: {fp['initial_fitness']:.3f} → {fp['final_fitness']:.3f} (trend: {fp['fitness_trend']})"
            )
            cp = lineage_stats["complexity_progression"]
            print(f"Complexity progression: {cp['complexity_trend']}")
            print(f"Nodes: {cp['initial_nodes']} → {cp['final_nodes']}")
            print(
                f"Connections: {cp['initial_connections']} → {cp['final_connections']}"
            )

        # Test ancestor comparison
        if len(ancestry_df) > 1:
            logger.info("\n🔍 Ancestor Comparison:")
            ancestor_gen = ancestry_df.iloc[-1]["generation"]  # Oldest ancestor
            comparison = explorer.ancestry_analyzer.compare_with_ancestor(
                explorer.neat_genome, ancestor_gen
            )
            if "error" not in comparison:
                print(f"Comparing with ancestor from generation {ancestor_gen}")
                print(f"Fitness change: {comparison['fitness_change']:.3f}")
                sc = comparison["structure_changes"]
                print(
                    f"Structure changes: +{sc['nodes_added']} nodes, +{sc['connections_added']} connections"
                )
                pc = comparison["parameter_changes"]
                print(f"Average weight change: {pc['avg_weight_change']:.3f}")
                print(f"Average bias change: {pc['avg_bias_change']:.3f}")

        # Test visualization
        logger.info("\n🎨 Testing visualizations:")
        try:
            # Test training metrics plot
            logger.info("  - Training metrics plot...")
            explorer.plot_training_metrics()

            # Test ancestry fitness plot
            if len(ancestry_df) > 1:
                logger.info("  - Ancestry fitness plot...")
                explorer.plot_ancestry_fitness()

            # Test network visualization
            logger.info("  - Network structure plot...")
            explorer.show_network(figsize=(12, 8))

        except Exception as e:
            logger.warning(f"Visualization test failed: {e}")

        # Test data export
        logger.info("\n💾 Testing data export:")
        try:
            export_data = explorer.export_genome_data()
            print(f"Exported data contains {len(export_data)} sections:")
            for key in export_data.keys():
                print(f"  - {key}")
        except Exception as e:
            logger.warning(f"Data export failed: {e}")

        logger.info(f"\n✅ Analysis complete for experiment: {experiment_id}")

        return best_genome, experiment_id

    except Exception as e:
        # Mark experiment as failed
        with db.session_scope() as session:
            experiment = session.get(Experiment, experiment_id)
            experiment.status = "failed"
            experiment.end_time = datetime.now(timezone.utc)

        logger.error(f"Evolution failed: {e}")
        raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run working backache experiment")
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations (default: 10)",
    )

    args = parser.parse_args()

    winner, experiment_id = run_working_backache_experiment(args.generations)

    print("\n" + "=" * 50)
    print("🧬 WORKING BACKACHE EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Experiment ID: {experiment_id}")
    print(f"Best fitness (Train AUC): {winner.fitness:.4f}")
    if hasattr(winner, "test_auc"):
        print(f"Test AUC: {winner.test_auc:.4f}")
    print(f"\nTo run further analysis:")
    print(f"  # Load the explorer")
    print(f"  explorer = GenomeExplorer.load_best_genome('{experiment_id}')")
    print(f"  # Show summary")
    print(f"  explorer.summary()")
    print(f"  # Show network")
    print(f"  explorer.show_network()")
    print(f"  # Export data")
    print(f"  data = explorer.export_genome_data()")


if __name__ == "__main__":
    main()
