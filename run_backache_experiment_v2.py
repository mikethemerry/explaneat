"""
Advanced backache experiment with database integration

Based on the standard experiment structure but enhanced with:
- Database storage for genomes and populations
- GenomeExplorer integration for analysis
- AUC-based fitness evaluation
- Proper configuration management
"""

import argparse
import os
import datetime
import random
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from pmlb import fetch_data
import logging
from datetime import datetime, timezone
import neat

from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.experiment import ExperimentReporter
from explaneat.core.explaneat import ExplaNEAT
from explaneat.visualization import visualize
from explaneat.db import db, Experiment, Population, Genome, TrainingMetric, Result
from explaneat.db.serialization import serialize_population_config
from explaneat.analysis import GenomeExplorer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BackacheExperiment:
    """Advanced backache experiment with database integration"""

    def __init__(self, config_path, experiment_name=None, random_seed=42):
        self.config_path = config_path
        self.experiment_name = (
            experiment_name
            or f"Backache_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.random_seed = random_seed

        # Initialize database
        db.init_db()

        # Set device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load and prepare data
        self.prepare_data()

        # Create NEAT config
        self.create_config()

        # Create experiment record
        self.create_experiment_record()

    def prepare_data(self):
        """Load and prepare the backache dataset"""
        logger.info("📊 Loading backache dataset from PMLB...")

        X, y = fetch_data("backache", return_X_y=True)
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        # Initial train/test split
        self.X_train_base, self.X_test, self.y_train_base, self.y_test = (
            train_test_split(
                X, y, test_size=0.2, random_state=self.random_seed, stratify=y
            )
        )

        # Scale data
        self.scaler = StandardScaler()
        self.X_train_base = self.scaler.fit_transform(self.X_train_base)
        self.X_test = self.scaler.transform(self.X_test)

        # Convert test data to tensors
        self.X_test_tt = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        self.y_test_tt = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)

        logger.info(f"Train base: {self.X_train_base.shape}, Test: {self.X_test.shape}")

    def create_config(self):
        """Create NEAT configuration"""
        config_text = f"""
#--- NEAT configuration for backache classification ---

[NEAT]
fitness_criterion     = max
fitness_threshold     = 0.95
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
activation_default      = sigmoid
activation_mutate_rate  = 0.1
activation_options      = sigmoid tanh relu

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

conn_add_prob           = 0.5
conn_delete_prob        = 0.3

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = partial_direct 0.5

node_add_prob           = 0.3
node_delete_prob        = 0.2

num_hidden              = 0
num_inputs              = {self.X_train_base.shape[1]}
num_outputs             = 1

response_init_mean      = 1.0
response_init_stdev     = 0.1
response_max_value      = 10.0
response_min_value      = -10.0
response_mutate_power   = 0.1
response_mutate_rate    = 0.1
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

        # Write config to temp file
        self.config_filename = "backache_experiment_config.cfg"
        with open(self.config_filename, "w") as f:
            f.write(config_text)

        self.base_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_filename,
        )

    def create_experiment_record(self):
        """Create experiment record in database"""
        with db.session_scope() as session:
            self.experiment = Experiment(
                experiment_sha=f"backache_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=self.experiment_name,
                description="Advanced backache experiment with BackpropPopulation and database integration",
                dataset_name="PMLB Backache",
                config_json=serialize_population_config(self.base_config),
                neat_config_text=open(self.config_filename, "r").read(),
                start_time=datetime.now(timezone.utc),
            )
            session.add(self.experiment)
            session.flush()
            self.experiment_id = self.experiment.id

        logger.info(f"Created experiment ID: {self.experiment_id}")

    def fitness_function(self, genomes, config):
        """AUC-based fitness function with database storage"""
        X_train, y_train = self.current_train_data
        X_validate, y_validate = self.current_validate_data

        for genome_id, genome in genomes:
            try:
                # Create neural network
                net = neat.nn.FeedForwardNetwork.create(genome, config)

                # Get predictions on training set
                train_predictions = []
                for i in range(len(X_train)):
                    output = net.activate(X_train[i])
                    train_predictions.append(output[0])

                # Calculate metrics
                train_predictions = np.array(train_predictions)
                train_auc = roc_auc_score(y_train, train_predictions)
                train_accuracy = accuracy_score(
                    y_train, (train_predictions > 0.5).astype(int)
                )
                train_loss = log_loss(y_train, train_predictions)

                # Validation metrics
                val_predictions = []
                for i in range(len(X_validate)):
                    output = net.activate(X_validate[i])
                    val_predictions.append(output[0])

                val_predictions = np.array(val_predictions)
                val_auc = roc_auc_score(y_validate, val_predictions)
                val_accuracy = accuracy_score(
                    y_validate, (val_predictions > 0.5).astype(int)
                )
                val_loss = log_loss(y_validate, val_predictions)

                # Use validation AUC as fitness
                genome.fitness = val_auc

                # Store additional metrics
                genome.train_auc = train_auc
                genome.train_accuracy = train_accuracy
                genome.train_loss = train_loss
                genome.val_auc = val_auc
                genome.val_accuracy = val_accuracy
                genome.val_loss = val_loss

            except Exception as e:
                logger.warning(f"Fitness evaluation failed for genome {genome_id}: {e}")
                genome.fitness = 0.5  # Default to random performance

    def save_population_to_db(self, generation, population, species, config):
        """Save population and genomes to database"""
        # Calculate population statistics
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        best_fitness = max(fitnesses) if fitnesses else 0
        mean_fitness = np.mean(fitnesses) if fitnesses else 0
        std_fitness = np.std(fitnesses) if fitnesses else 0

        # Save population record
        with db.session_scope() as session:
            pop_record = Population(
                experiment_id=self.experiment_id,
                generation=generation,
                population_size=len(population),
                num_species=len(species.species) if species else 1,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                stdev_fitness=std_fitness,
                config_json=serialize_population_config(config),
            )
            session.add(pop_record)
            session.flush()
            population_id = pop_record.id

        # Save genomes with training metrics
        with db.session_scope() as session:
            for genome_id, genome in population.items():
                db_genome = Genome.from_neat_genome(genome, population_id)
                session.add(db_genome)
                session.flush()

                # Add training metrics if available
                if hasattr(genome, "train_loss"):
                    metric = TrainingMetric(
                        genome_id=db_genome.id,
                        population_id=population_id,
                        epoch=0,  # Since we're not doing backprop epochs here
                        loss=genome.train_loss,
                        accuracy=genome.train_accuracy,
                        validation_loss=genome.val_loss,
                        validation_accuracy=genome.val_accuracy,
                        additional_metrics={
                            "train_auc": genome.train_auc,
                            "val_auc": genome.val_auc,
                            "generation": generation,
                        },
                    )
                    session.add(metric)

        logger.info(
            f"Saved generation {generation}: {len(population)} genomes, "
            f"best fitness: {best_fitness:.3f}"
        )

    def run_iteration(self, iteration_no, max_generations=50, epochs_per_generation=10):
        """Run a single experiment iteration"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting iteration {iteration_no}")
        logger.info(f"{'='*50}")

        start_time = datetime.now()

        # Set random seed for this iteration
        iteration_seed = self.random_seed + iteration_no
        random.seed(iteration_seed)
        np.random.seed(iteration_seed)
        torch.manual_seed(iteration_seed)

        # Split data for this iteration
        X_train, X_validate, y_train, y_validate = train_test_split(
            self.X_train_base,
            self.y_train_base,
            test_size=0.3,
            random_state=iteration_seed,
        )

        self.current_train_data = (X_train, y_train)
        self.current_validate_data = (X_validate, y_validate)

        # Create population with BackpropPopulation
        config = self.base_config
        population = BackpropPopulation(
            config, X_train, y_train, criterion=nn.BCELoss()
        )

        # Add reporters
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))

        # Run evolution
        best_genome = None
        for generation in range(max_generations):
            logger.info(f"\n--- Generation {generation} ---")

            # Run backprop training for this generation
            population.train_genomes(epochs_per_generation)

            # Evaluate fitness
            self.fitness_function(list(population.population.items()), config)

            # Save to database
            self.save_population_to_db(
                generation, population.population, population.species, config
            )

            # Track best genome
            gen_best = max(population.population.values(), key=lambda g: g.fitness)
            if best_genome is None or gen_best.fitness > best_genome.fitness:
                best_genome = gen_best

            logger.info(f"Best fitness: {gen_best.fitness:.4f}")

            # Check for solution
            if gen_best.fitness >= config.fitness_threshold:
                logger.info(f"🎯 Fitness threshold reached: {gen_best.fitness:.4f}")
                break

            # Create next generation
            if generation < max_generations - 1:
                population.run_one_generation(self.fitness_function)

        # Save results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Store iteration results in database
        with db.session_scope() as session:
            result = Result(
                experiment_id=self.experiment_id,
                iteration=iteration_no,
                metric_name="training_duration",
                metric_value=duration,
                metadata={
                    "best_fitness": best_genome.fitness,
                    "generations": generation + 1,
                    "final_population_size": len(population.population),
                },
            )
            session.add(result)

        # Analyze with ExplaNEAT
        explainer = ExplaNEAT(best_genome, config)

        # Get final predictions on test set
        propneat_results_tt = explainer.net.forward(self.X_test_tt)
        propneat_results = propneat_results_tt.detach().cpu().numpy().flatten()

        # Calculate test metrics
        test_auc = roc_auc_score(self.y_test, propneat_results)
        test_accuracy = accuracy_score(
            self.y_test, (propneat_results > 0.5).astype(int)
        )
        test_loss = log_loss(self.y_test, propneat_results)

        logger.info(f"\nTest Results:")
        logger.info(f"  AUC: {test_auc:.4f}")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Loss: {test_loss:.4f}")

        # Store test results
        with db.session_scope() as session:
            test_result = Result(
                experiment_id=self.experiment_id,
                iteration=iteration_no,
                metric_name="test_metrics",
                metric_value=test_auc,
                metadata={
                    "test_auc": test_auc,
                    "test_accuracy": test_accuracy,
                    "test_loss": test_loss,
                    "predictions": propneat_results.tolist(),
                },
            )
            session.add(test_result)

            # Store network structure metrics
            structure_result = Result(
                experiment_id=self.experiment_id,
                iteration=iteration_no,
                metric_name="network_structure",
                metric_value=explainer.n_genome_params(),
                metadata={
                    "skippiness": explainer.skippines(),
                    "depth": explainer.depth(),
                    "num_nodes": len(best_genome.nodes),
                    "num_connections": len(
                        [c for c in best_genome.connections.values() if c.enabled]
                    ),
                },
            )
            session.add(structure_result)

        return best_genome, test_auc

    def analyze_experiment(self):
        """Analyze the experiment using GenomeExplorer"""
        logger.info("\n📊 Experiment Analysis:")
        logger.info("=" * 50)

        # Load best genome from experiment
        explorer = GenomeExplorer.load_best_genome(self.experiment_id)

        # Show summary
        logger.info("\n📋 Genome Summary:")
        explorer.summary()

        # Ancestry analysis
        logger.info("\n🌳 Ancestry Analysis:")
        ancestry_df = explorer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            logger.info(f"Found {len(ancestry_df)} ancestors")

            # Show lineage statistics
            lineage_stats = explorer.ancestry_analyzer.get_lineage_statistics()
            if lineage_stats:
                fp = lineage_stats["fitness_progression"]
                logger.info(
                    f"Fitness progression: {fp['initial_fitness']:.3f} → {fp['final_fitness']:.3f}"
                )

        # Generate visualizations
        logger.info("\n🎨 Generating visualizations...")
        explorer.show_network(figsize=(12, 8))

        if len(ancestry_df) > 1:
            explorer.plot_ancestry_fitness()

        # Export data
        export_data = explorer.export_genome_data()
        logger.info(f"\n💾 Exported data contains {len(export_data)} sections")

        return explorer

    def run(self, n_iterations=3, max_generations=50, epochs_per_generation=10):
        """Run the complete experiment"""
        logger.info(f"\n🧬 Starting {self.experiment_name}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Running {n_iterations} iterations")

        best_results = []

        try:
            for i in range(n_iterations):
                best_genome, test_auc = self.run_iteration(
                    i, max_generations, epochs_per_generation
                )
                best_results.append(
                    {"iteration": i, "test_auc": test_auc, "genome": best_genome}
                )

            # Mark experiment as completed
            with db.session_scope() as session:
                experiment = session.get(Experiment, self.experiment_id)
                experiment.status = "completed"
                experiment.end_time = datetime.now(timezone.utc)

            # Analyze results
            explorer = self.analyze_experiment()

            # Summary
            logger.info(f"\n✅ Experiment completed successfully!")
            logger.info(
                f"Best test AUC across iterations: {max(r['test_auc'] for r in best_results):.4f}"
            )

            return best_results, explorer

        except Exception as e:
            # Mark experiment as failed
            with db.session_scope() as session:
                experiment = session.get(Experiment, self.experiment_id)
                experiment.status = "failed"
                experiment.end_time = datetime.now(timezone.utc)

            logger.error(f"Experiment failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run advanced backache experiment")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to experiment config file"
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations (default: 3)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Max generations per iteration (default: 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Backprop epochs per generation (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Create and run experiment
    experiment = BackacheExperiment(
        config_path=args.config, experiment_name=args.name, random_seed=args.seed
    )

    results, explorer = experiment.run(
        n_iterations=args.iterations,
        max_generations=args.generations,
        epochs_per_generation=args.epochs,
    )

    print("\n" + "=" * 50)
    print("🧬 BACKACHE EXPERIMENT V2 COMPLETE")
    print("=" * 50)
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Best results by iteration:")
    for r in results:
        print(f"  Iteration {r['iteration']}: Test AUC = {r['test_auc']:.4f}")
    print(f"\nAnalyze with GenomeExplorer:")
    print(f"  explorer = GenomeExplorer.load_best_genome('{experiment.experiment_id}')")


if __name__ == "__main__":
    main()
