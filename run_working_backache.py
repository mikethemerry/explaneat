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

from explaneat.db import db, Experiment, Population, Genome, TrainingMetric, Dataset, DatasetSplit
from explaneat.db.serialization import serialize_population_config
from explaneat.db.dataset_utils import (
    save_dataset_to_db,
    save_dataset_split_with_indices,
)
from explaneat.analysis import GenomeExplorer
from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.neuralneat import NeuralNeat
from explaneat.core.experiment import ExperimentReporterSet
from explaneat.core.errors import GenomeNotValidError
from explaneat.core.ancestry_reporter import AncestryReporter
from explaneat.core.live_reporter import LiveReporter, QuietLogger
from explaneat.core.gene_origin_tracker import GeneOriginTracker
from explaneat.evaluators.evaluators import binary_cross_entropy, auc_fitness
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

    def __init__(self, experiment_id, config, ancestry_reporter=None, gene_tracker=None):
        self.experiment_id = experiment_id
        self.config = config
        self.generation = 0
        self.ancestry_reporter = ancestry_reporter
        self.gene_tracker = gene_tracker

    def start_generation(self, generation):
        self.generation = generation
        if self.ancestry_reporter:
            self.ancestry_reporter.start_generation(generation)

    def post_evaluate(self, config, population, species, best_genome):
        """Save population data after fitness evaluation"""
        try:
            save_population_to_db(
                self.experiment_id, self.generation, population, self.config,
                ancestry_reporter=self.ancestry_reporter,
                gene_tracker=self.gene_tracker
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

    def species_stagnant(self, sid, species):
        """Handle stagnant species - required by NEAT ReporterSet"""
        logger.info(f"Species {sid} is stagnant with {len(species.members)} members")

    def found_solution(self, config, generation, best):
        """Handle when a solution is found - required by NEAT ReporterSet"""
        logger.info(
            f"Solution found at generation {generation} with fitness {best.fitness}"
        )

    def complete_extinction(self):
        """Handle complete extinction - required by NEAT ReporterSet"""
        logger.warning("All species have gone extinct!")

    def start_experiment(self, config):
        """Start of experiment - required by NEAT ReporterSet"""
        logger.info("Starting NEAT experiment")

    def end_experiment(self, config, population, species):
        """End of experiment - required by NEAT ReporterSet"""
        logger.info("NEAT experiment completed")


def prepare_backache_data(random_state=42):
    """Load and prepare the backache dataset from PMLB
    
    Args:
        random_state: Random seed for train/test split
        
    Returns:
        Tuple of (X, y, X_train, X_test, y_train, y_test, train_indices, test_indices, scaler)
        where X, y are the original full dataset
    """
    logger.info("📊 Loading backache dataset from PMLB...")

    X, y = fetch_data("backache", return_X_y=True)
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Get indices before split for exact reproducibility
    # Create array of indices
    indices = np.arange(len(X))
    
    # Split with indices for reproducibility
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Convert to lists for JSON serialization
    train_indices = train_indices.tolist()
    test_indices = test_indices.tolist()
    
    # Now split the actual data using the indices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X, y, X_train, X_test, y_train, y_test, train_indices, test_indices, scaler


def create_backache_config(num_inputs):
    """Create NEAT configuration for backache using reference config"""
    config_text = f"""
#--- parameters for the backache experiment ---#

[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.95
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
node_delete_prob        = 0.30


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


def create_experiment_record(name, dataset_name, description, config, dataset_id=None, random_seed=None):
    """Create experiment record in database
    
    Args:
        name: Experiment name
        dataset_name: Dataset name (deprecated, use dataset_id)
        description: Experiment description
        config: NEAT config
        dataset_id: UUID of the dataset (if already created)
        random_seed: Random seed used for the experiment
        
    Returns:
        Experiment ID
    """
    with db.session_scope() as session:
        experiment = Experiment(
            experiment_sha=f"working_backache_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=description,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            random_seed=random_seed,
            config_json=serialize_population_config(config),
            neat_config_text="# Working backache config",
            start_time=datetime.now(timezone.utc),
        )
        session.add(experiment)
        session.flush()
        experiment_id = experiment.id

    return experiment_id


def save_population_to_db(experiment_id, generation, population, config, ancestry_reporter=None, gene_tracker=None):
    """Save population and genomes to database with ancestry and gene origin tracking

    Args:
        experiment_id: Database ID of the experiment
        generation: Current generation number
        population: Dictionary of NEAT genomes
        config: NEAT configuration
        ancestry_reporter: Optional AncestryReporter for parent tracking
        gene_tracker: Optional GeneOriginTracker for gene origin tracking
    """

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

    # Track genome ID mappings for gene origin tracker
    genome_id_mapping = {}

    # Save genomes with parent relationships
    with db.session_scope() as session:
        for genome_id, genome in population.items():
            # Get parent database IDs if ancestry reporter is available
            parent1_db_id = None
            parent2_db_id = None
            if ancestry_reporter is not None:
                parent1_db_id, parent2_db_id = ancestry_reporter.get_parent_ids(genome_id)

            # Create genome record with parent IDs
            db_genome = Genome.from_neat_genome(
                genome,
                population_id,
                parent1_id=parent1_db_id,
                parent2_id=parent2_db_id
            )
            session.add(db_genome)
            session.flush()  # Flush to get the genome ID assigned

            # Register this genome in the ancestry tracker for next generation
            if ancestry_reporter is not None:
                ancestry_reporter.register_genome(genome_id, db_genome.id)

            # Track mapping for gene origin tracker
            genome_id_mapping[genome_id] = db_genome.id

    # Process gene origins after all genomes are saved
    if gene_tracker is not None:
        gene_tracker.process_population(population, generation, genome_id_mapping)

    # Only log if not in quiet mode (check root logger level)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            f"Saved generation {generation}: {len(population)} genomes, best fitness: {best_fitness:.3f}"
        )
    return population_id


def print_experiment_summary(experiment_id, best_genome):
    """Print comprehensive experiment summary"""
    from explaneat.analysis.genome_explorer import GenomeExplorer

    print("\n" + "=" * 80)
    print("🧬 EXPERIMENT SUMMARY")
    print("=" * 80)

    try:
        # Load the genome explorer for detailed analysis
        explorer = GenomeExplorer.load_best_genome(experiment_id)

        # Basic experiment info
        print(f"📊 Experiment ID: {experiment_id}")
        print(f"🏆 Best Genome ID: {explorer.genome_info.neat_genome_id}")
        print(f"🎯 Best Fitness: {explorer.genome_info.fitness:.4f}")
        print(f"🧬 Generation: {explorer.genome_info.generation}")

        # Network structure
        stats = explorer.genome_info.network_stats
        print(f"\n🕸️  Network Structure:")
        print(f"   Nodes: {stats['num_nodes']}")
        print(
            f"   Connections: {stats['num_connections']} ({stats['num_enabled_connections']} enabled)"
        )
        print(f"   Depth: {stats['network_depth']}")
        print(f"   Width: {stats['network_width']}")

        # Ancestry analysis
        ancestry_df = explorer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            print(f"\n🌳 Ancestry Analysis ({len(ancestry_df)} generations):")
            print(
                f"   Fitness range: {ancestry_df['fitness'].min():.3f} → {ancestry_df['fitness'].max():.3f}"
            )
            print(
                f"   Fitness improvement: {ancestry_df['fitness'].max() - ancestry_df['fitness'].min():.3f}"
            )
            print(
                f"   Network growth: {ancestry_df['num_nodes'].iloc[0]} → {ancestry_df['num_nodes'].iloc[-1]} nodes"
            )
            print(
                f"   Connection growth: {ancestry_df['num_connections'].iloc[0]} → {ancestry_df['num_connections'].iloc[-1]} connections"
            )

        # Performance context
        context = explorer.get_performance_context()
        print(f"\n📈 Performance Context:")
        print(
            f"   Rank in generation: {context['generation_rank']}/{context['generation_size']}"
        )
        print(f"   Generation best: {context['generation_best']:.3f}")
        print(
            f"   Generation mean: {context['generation_mean']:.3f} (±{context['generation_std']:.3f})"
        )
        print(f"   Is best in generation: {context['is_generation_best']}")
        print(f"   Total experiment generations: {context['experiment_generations']}")

        print(f"\n🔍 Next Steps:")
        print(
            f"   # Load explorer: explorer = GenomeExplorer.load_best_genome('{experiment_id}')"
        )
        print(f"   # Show network: explorer.show_network()")
        print(f"   # Plot ancestry: explorer.plot_ancestry_fitness()")
        print(f"   # Plot evolution: explorer.plot_evolution_progression()")
        print(f"   # Export data: explorer.export_genome_data()")

    except Exception as e:
        logger.warning(f"Could not generate detailed summary: {e}")
        print(f"📊 Basic Summary:")
        print(f"   Experiment ID: {experiment_id}")
        print(f"   Best Fitness: {best_genome.fitness:.4f}")


def instantiate_population(config, xs, ys):
    """Instantiate BackpropPopulation following ExplaNEAT pattern"""
    # Create the population using BackpropPopulation
    p = BackpropPopulation(config, xs, ys, criterion=torch.nn.BCELoss())

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    return p


def run_working_backache_experiment(num_generations=10, compact=True):
    """Run working backache experiment using ExplaNEAT framework

    Args:
        num_generations: Number of generations to evolve
        compact: If True, use compact single-line status. If False, use multi-line dashboard.
    """

    logger.info("🧬 Starting Working Backache Experiment")
    logger.info("=" * 50)

    # Initialize database
    db.init_db()

    # Set random seed for reproducibility
    random_seed = 42

    # Prepare data (now returns full dataset and indices)
    X, y, X_train, X_test, y_train, y_test, train_indices, test_indices, scaler = prepare_backache_data(random_state=random_seed)

    # Save dataset to database
    logger.info("💾 Saving dataset to database...")
    dataset = save_dataset_to_db(
        name="backache",
        X=X,
        y=y,
        source="PMLB",
        description="Backache dataset from PMLB - binary classification problem",
        target_name="backache",
        target_description="Binary classification target (0=no backache, 1=backache)",
        class_names=["no_backache", "backache"],
    )
    dataset_id = str(dataset.id)
    logger.info(f"Dataset saved with ID: {dataset_id}")

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
    logger.info(f"Fitness threshold: {getattr(config, 'fitness_threshold', 0.95)}")
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
        dataset_id=dataset_id,
        random_seed=random_seed,
    )

    logger.info(f"Created experiment ID: {experiment_id}")

    # Save dataset split to database for reproducibility
    logger.info("💾 Saving dataset split to database...")
    save_dataset_split_with_indices(
        dataset_id=dataset_id,
        experiment_id=str(experiment_id),
        train_indices=train_indices,
        test_indices=test_indices,
        split_type="train_test",
        test_size=0.2,
        random_state=random_seed,
        shuffle=True,
        stratify=True,
        scaler=scaler,
        preprocessing_steps=[{"step": "StandardScaler", "fit_on": "train"}],
    )
    logger.info("Dataset split saved successfully")

    # Instantiate population using ExplaNEAT pattern
    population = instantiate_population(config, X_train, y_train)

    # Create ancestry reporter for parent tracking
    ancestry_reporter = AncestryReporter()
    # Link ancestry reporter to reproduction object
    ancestry_reporter.reproduction = population.reproduction

    # Create gene origin tracker for innovation tracking
    gene_tracker = GeneOriginTracker(experiment_id)

    # Create database reporter with ancestry and gene tracking
    db_reporter = DatabaseReporter(experiment_id, config,
                                   ancestry_reporter=ancestry_reporter,
                                   gene_tracker=gene_tracker)

    # Create live reporter for real-time status display
    live_reporter = LiveReporter(max_generations=num_generations, compact=compact)

    # Add reporters
    population.reporters.reporters.append(db_reporter)
    population.reporters.reporters.append(ancestry_reporter)
    population.reporters.reporters.append(live_reporter)

    logger.info(f"Population size: {getattr(config, 'pop_size', 50)}")

    # Run evolution using BackpropPopulation
    logger.info(f"🔄 Starting evolution for {num_generations} generations...")
    print()  # Add blank line before live status starts

    try:
        # Run the evolution using BackpropPopulation's run method with AUC fitness
        best_genome = population.run(
            fitness_function=auc_fitness,
            n=100,
            nEpochs=5,  # Number of backprop epochs per generation
        )

        # Mark experiment as completed
        with db.session_scope() as session:
            experiment = session.get(Experiment, experiment_id)
            experiment.status = "completed"
            experiment.end_time = datetime.now(timezone.utc)

        logger.info("✅ Evolution completed!")
        logger.info(f"🏆 Best fitness (Train AUC): {best_genome.fitness:.4f}")

        # Print comprehensive experiment summary
        print_experiment_summary(experiment_id, best_genome)

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
            # Test training metrics plot (shows epochs within a single genome)
            logger.info("  - Training metrics plot (epochs within genome)...")
            explorer.plot_training_metrics()

            # Test ancestry fitness plot (shows evolution across generations)
            if len(ancestry_df) > 1:
                logger.info(
                    "  - Ancestry fitness plot (evolution across generations)..."
                )
                explorer.plot_ancestry_fitness()

            # Test full evolution progression (shows population-level evolution)
            logger.info("  - Full evolution progression plot...")
            explorer.plot_evolution_progression()

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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity during evolution (only show live status)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Use multi-line dashboard instead of compact status line",
    )

    args = parser.parse_args()

    # Adjust logging level if quiet mode
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    winner, experiment_id = run_working_backache_experiment(
        args.generations,
        compact=not args.dashboard
    )

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
