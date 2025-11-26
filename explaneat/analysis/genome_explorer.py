"""
GenomeExplorer: Main class for analyzing individual genomes and their ancestry

This class provides a comprehensive API for loading genomes from the database
and exploring their evolutionary history, performance, and network structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import re
import os
import tempfile

from ..db import db, Experiment, Population, Genome, TrainingMetric, GeneOrigin
from ..core.explaneat import ExplaNEAT
from .ancestry_analyzer import AncestryAnalyzer
from .visualization import GenomeVisualizer, InteractiveNetworkViewer
from .annotation_manager import AnnotationManager


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

    This class provides three different types of plotting methods:

    1. plot_training_metrics(): Shows training progress (epochs) within a single genome
    2. plot_ancestry_fitness(): Shows evolution across generations for a specific lineage
    3. plot_evolution_progression(): Shows population-level evolution across all generations

    Usage:
        explorer = GenomeExplorer.load_genome(genome_id)
        explorer.show_network()
        explorer.plot_training_metrics()      # Epochs within genome
        explorer.plot_ancestry_fitness()      # Generations in lineage
        explorer.plot_evolution_progression() # Full population evolution
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
            training_metrics = (
                session.query(TrainingMetric)
                .filter_by(genome_id=genome_id)
                .order_by(TrainingMetric.epoch)
                .all()
            )

            # Load NEAT config from experiment's stored config text
            import neat

            # Get the config text from the experiment
            neat_config_text = experiment.neat_config_text or ""

            # If config text is empty, raise an error
            if not neat_config_text or not neat_config_text.strip():
                raise ValueError(
                    f"Experiment {experiment.id} has no stored NEAT configuration. "
                    "Cannot load genome without configuration."
                )

            # Get missing parameters from config_json as fallback
            config_json = experiment.config_json or {}
            genome_config = config_json.get("genome", {})
            species_config = config_json.get("species", {})
            stagnation_config = config_json.get("stagnation", {})
            reproduction_config = config_json.get("reproduction", {})

            # List of required NEAT section parameters
            required_neat_params = {
                "no_fitness_termination": config_json.get(
                    "no_fitness_termination", False
                ),
                "pop_size": config_json.get("pop_size", 50),  # Default fallback
                "fitness_criterion": config_json.get("fitness_criterion", "max"),
                "fitness_threshold": config_json.get("fitness_threshold", 3.9),
                "reset_on_extinction": config_json.get("reset_on_extinction", False),
            }

            # Check and add missing parameters to the config text
            if "[NEAT]" in neat_config_text:
                # First, collect all missing parameters
                missing_params = []
                for param_name, param_value in required_neat_params.items():
                    # Check if parameter is explicitly set (not just in comments)
                    param_pattern = re.compile(
                        r"^\s*" + re.escape(param_name) + r"\s*=", re.MULTILINE
                    )

                    if not param_pattern.search(neat_config_text):
                        missing_params.append((param_name, param_value))

                # Add all missing parameters at once after [NEAT] section header
                if missing_params:
                    params_str = "\n".join(f"{k} = {v}" for k, v in missing_params)
                    neat_config_text = re.sub(
                        r"(\[NEAT\])", rf"\1\n{params_str}", neat_config_text, count=1
                    )
            else:
                # If no [NEAT] section, create one with all required parameters
                neat_params_str = "\n".join(
                    f"{k} = {v}" for k, v in required_neat_params.items()
                )
                neat_config_text = f"[NEAT]\n{neat_params_str}\n\n" + neat_config_text

            # Check for required sections and add if missing
            # Generate section content from config_json with sensible defaults
            def generate_default_genome_section(genome_cfg):
                """Generate DefaultGenome section from config_json"""
                return f"""activation_default      = {genome_cfg.get('activation_default', 'sigmoid')}
activation_mutate_rate  = {genome_cfg.get('activation_mutate_rate', 0.1)}
activation_options      = {genome_cfg.get('activation_options', 'sigmoid')}
aggregation_default     = {genome_cfg.get('aggregation_default', 'sum')}
aggregation_mutate_rate = {genome_cfg.get('aggregation_mutate_rate', 0.0)}
aggregation_options     = {genome_cfg.get('aggregation_options', 'sum')}
bias_init_mean          = {genome_cfg.get('bias_init_mean', 0.0)}
bias_init_stdev         = {genome_cfg.get('bias_init_stdev', 1.0)}
bias_max_value          = {genome_cfg.get('bias_max_value', 30.0)}
bias_min_value          = {genome_cfg.get('bias_min_value', -30.0)}
bias_mutate_power       = {genome_cfg.get('bias_mutate_power', 0.5)}
bias_mutate_rate        = {genome_cfg.get('bias_mutate_rate', 0.7)}
bias_replace_rate       = {genome_cfg.get('bias_replace_rate', 0.1)}
compatibility_disjoint_coefficient = {genome_cfg.get('compatibility_disjoint_coefficient', 1.0)}
compatibility_weight_coefficient   = {genome_cfg.get('compatibility_weight_coefficient', 0.5)}
conn_add_prob           = {genome_cfg.get('conn_add_prob', 0.5)}
conn_delete_prob        = {genome_cfg.get('conn_delete_prob', 0.5)}
enabled_default         = {genome_cfg.get('enabled_default', True)}
enabled_mutate_rate     = {genome_cfg.get('enabled_mutate_rate', 0.01)}
feed_forward            = {genome_cfg.get('feed_forward', True)}
initial_connection      = {genome_cfg.get('initial_connection', 'full_direct')}
node_add_prob           = {genome_cfg.get('node_add_prob', 0.2)}
node_delete_prob        = {genome_cfg.get('node_delete_prob', 0.2)}
num_hidden              = {genome_cfg.get('num_hidden', 0)}
num_inputs              = {genome_cfg.get('num_inputs', 10)}
num_outputs             = {genome_cfg.get('num_outputs', 1)}
response_init_mean      = {genome_cfg.get('response_init_mean', 1.0)}
response_init_stdev     = {genome_cfg.get('response_init_stdev', 0.0)}
response_max_value      = {genome_cfg.get('response_max_value', 30.0)}
response_min_value      = {genome_cfg.get('response_min_value', -30.0)}
response_mutate_power   = {genome_cfg.get('response_mutate_power', 0.0)}
response_mutate_rate    = {genome_cfg.get('response_mutate_rate', 0.0)}
response_replace_rate   = {genome_cfg.get('response_replace_rate', 0.0)}
single_structural_mutation = {genome_cfg.get('single_structural_mutation', False)}
structural_mutation_surer = {genome_cfg.get('structural_mutation_surer', 'default')}
weight_init_mean        = {genome_cfg.get('weight_init_mean', 0.0)}
weight_init_stdev       = {genome_cfg.get('weight_init_stdev', 1.0)}
weight_max_value        = {genome_cfg.get('weight_max_value', 30)}
weight_min_value        = {genome_cfg.get('weight_min_value', -30)}
weight_mutate_power     = {genome_cfg.get('weight_mutate_power', 0.5)}
weight_mutate_rate      = {genome_cfg.get('weight_mutate_rate', 0.8)}
weight_replace_rate     = {genome_cfg.get('weight_replace_rate', 0.1)}"""

            def generate_species_section(species_cfg):
                """Generate DefaultSpeciesSet section from config_json"""
                return f"compatibility_threshold = {species_cfg.get('compatibility_threshold', 3.0)}"

            def generate_stagnation_section(stagnation_cfg):
                """Generate DefaultStagnation section from config_json"""
                return f"""species_fitness_func = {stagnation_cfg.get('species_fitness_func', 'max')}
max_stagnation       = {stagnation_cfg.get('max_stagnation', 20)}
species_elitism      = {stagnation_cfg.get('species_elitism', 2)}"""

            def generate_reproduction_section(reproduction_cfg):
                """Generate DefaultReproduction section from config_json"""
                return f"""elitism            = {reproduction_cfg.get('elitism', 2)}
survival_threshold = {reproduction_cfg.get('survival_threshold', 0.2)}"""

            required_sections = {
                "[DefaultGenome]": generate_default_genome_section(genome_config),
                "[DefaultSpeciesSet]": generate_species_section(species_config),
                "[DefaultStagnation]": generate_stagnation_section(stagnation_config),
                "[DefaultReproduction]": generate_reproduction_section(
                    reproduction_config
                ),
            }

            # Check for required sections and add missing parameters
            for section_name, section_content in required_sections.items():
                if section_name not in neat_config_text:
                    # Section doesn't exist, add it at the end
                    neat_config_text += f"\n{section_name}\n{section_content}\n"
                else:
                    # Section exists, but check for missing required parameters
                    # Extract parameters from the generated section
                    section_lines = section_content.split("\n")
                    for line in section_lines:
                        if "=" in line and line.strip():
                            param_name = line.split("=")[0].strip()
                            param_value = line.split("=")[1].strip()

                            # Check if this parameter exists in the section
                            # Look for the parameter after the section header
                            section_start = neat_config_text.find(section_name)
                            if section_start != -1:
                                # Find the next section or end of file
                                next_section = len(neat_config_text)
                                for other_section in [
                                    "[NEAT]",
                                    "[DefaultGenome]",
                                    "[DefaultSpeciesSet]",
                                    "[DefaultStagnation]",
                                    "[DefaultReproduction]",
                                ]:
                                    if other_section != section_name:
                                        pos = neat_config_text.find(
                                            other_section,
                                            section_start + len(section_name),
                                        )
                                        if pos != -1 and pos < next_section:
                                            next_section = pos

                                section_text = neat_config_text[
                                    section_start:next_section
                                ]

                                # Check if parameter exists
                                param_pattern = re.compile(
                                    r"^\s*" + re.escape(param_name) + r"\s*=",
                                    re.MULTILINE,
                                )

                                if not param_pattern.search(section_text):
                                    # Add missing parameter to the section
                                    # Insert after section header
                                    insert_pos = section_start + len(section_name)
                                    neat_config_text = (
                                        neat_config_text[:insert_pos]
                                        + f"\n{param_name} = {param_value}"
                                        + neat_config_text[insert_pos:]
                                    )

            # Create a temporary config file
            temp_config_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".cfg", delete=False
            )
            temp_config_path = temp_config_file.name
            try:
                temp_config_file.write(neat_config_text)
                temp_config_file.close()

                config = neat.Config(
                    neat.DefaultGenome,
                    neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,
                    neat.DefaultStagnation,
                    temp_config_path,
                )

                # Deserialize the NEAT genome (Config reads file during init, so this is safe)
                neat_genome = genome_record.to_neat_genome(config)
            finally:
                # Clean up the temporary file after config is loaded
                try:
                    os.unlink(temp_config_path)
                except:
                    pass

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
                    "num_nodes": genome_record.num_nodes,
                    "num_connections": genome_record.num_connections,
                    "num_enabled_connections": genome_record.num_enabled_connections,
                    "network_depth": genome_record.network_depth,
                    "network_width": genome_record.network_width,
                },
                training_metrics=[
                    {
                        "epoch": tm.epoch,
                        "loss": tm.loss,
                        "accuracy": tm.accuracy,
                        "additional_metrics": tm.additional_metrics,
                    }
                    for tm in training_metrics
                ],
                genome_data=genome_record.genome_data,
            )

        return cls(genome_info, neat_genome, config)

    @classmethod
    def load_best_genome(cls, experiment_id: str):
        """Load the best genome from an experiment"""
        with db.session_scope() as session:
            best_genome = (
                session.query(Genome)
                .join(Population)
                .filter(
                    Population.experiment_id == experiment_id,
                    Genome.fitness.isnot(None),
                )
                .order_by(Genome.fitness.desc())
                .first()
            )

            if not best_genome:
                raise ValueError(f"No genomes found in experiment {experiment_id}")

            # Get the ID while still in session
            best_genome_id = str(best_genome.id)

        return cls.load_genome(best_genome_id)

    @classmethod
    def list_experiments(cls) -> pd.DataFrame:
        """Get a list of all experiments in the database"""
        with db.session_scope() as session:
            experiments = (
                session.query(Experiment).order_by(Experiment.created_at.desc()).all()
            )

            data = []
            for exp in experiments:
                # Get some basic stats
                populations = (
                    session.query(Population).filter_by(experiment_id=exp.id).count()
                )
                genomes = (
                    session.query(Genome)
                    .join(Population)
                    .filter(Population.experiment_id == exp.id)
                    .count()
                )

                # Get best fitness
                best_genome = (
                    session.query(Genome)
                    .join(Population)
                    .filter(
                        Population.experiment_id == exp.id, Genome.fitness.isnot(None)
                    )
                    .order_by(Genome.fitness.desc())
                    .first()
                )

                data.append(
                    {
                        "experiment_id": str(exp.id),
                        "name": exp.name,
                        "status": exp.status,
                        "dataset": exp.dataset_name,
                        "generations": populations,
                        "total_genomes": genomes,
                        "best_fitness": best_genome.fitness if best_genome else None,
                        "created_at": exp.created_at,
                        "duration": (
                            exp.end_time - exp.start_time if exp.end_time else None
                        ),
                    }
                )

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
        print(
            f"Connections: {stats['num_connections']} ({stats['num_enabled_connections']} enabled)"
        )
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
            if "loss" in metrics_df.columns:
                print(
                    f"Loss range: {metrics_df['loss'].min():.4f} - {metrics_df['loss'].max():.4f}"
                )
            if "accuracy" in metrics_df.columns:
                print(
                    f"Accuracy range: {metrics_df['accuracy'].min():.4f} - {metrics_df['accuracy'].max():.4f}"
                )
            print(f"Training epochs: {len(metrics_df)}")

    def get_ancestry_tree(self, max_generations: Optional[int] = None) -> pd.DataFrame:
        """Get the ancestry tree as a DataFrame

        Args:
            max_generations: Maximum generations to trace. None = unlimited (full history)
        """
        return self.ancestry_analyzer.get_ancestry_tree(max_generations)

    def plot_training_metrics(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot training metrics over epochs for this specific genome"""
        if not self.genome_info.training_metrics:
            print("No training metrics available for this genome")
            return

        df = pd.DataFrame(self.genome_info.training_metrics)

        _, axes = plt.subplots(1, 2, figsize=figsize)

        # Loss plot
        if "loss" in df.columns and df["loss"].notna().any():
            axes[0].plot(df["epoch"], df["loss"], "b-o", linewidth=2, markersize=4)
            axes[0].set_title("Training Loss (Epochs)")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        if "accuracy" in df.columns and df["accuracy"].notna().any():
            axes[1].plot(df["epoch"], df["accuracy"], "g-o", linewidth=2, markersize=4)
            axes[1].set_title("Training Accuracy (Epochs)")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"Training Metrics - Genome {self.genome_info.neat_genome_id} (Generation {self.genome_info.generation})"
        )
        plt.tight_layout()
        plt.show()

    def plot_ancestry_fitness(
        self, max_generations: Optional[int] = None, figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Plot fitness progression through ancestry across generations

        Args:
            max_generations: Maximum generations to trace. None = unlimited (full history)
            figsize: Figure size for plots
        """
        ancestry_df = self.get_ancestry_tree(max_generations)

        if ancestry_df.empty:
            print("No ancestry data available")
            return

        # Remove duplicates and sort by generation
        ancestry_df = ancestry_df.drop_duplicates(subset=["generation"]).sort_values(
            "generation"
        )

        if len(ancestry_df) < 2:
            print(
                "Insufficient ancestry data for plotting (need at least 2 generations)"
            )
            return

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Fitness over generations
        ax1.plot(
            ancestry_df["generation"],
            ancestry_df["fitness"],
            "o-",
            linewidth=2,
            markersize=6,
            color="blue",
        )
        ax1.set_title("Fitness Evolution Through Generations")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.grid(True, alpha=0.3)

        # Mark current genome
        current_gen = self.genome_info.generation
        current_fitness = self.genome_info.fitness
        ax1.scatter(
            [current_gen],
            [current_fitness],
            color="red",
            s=100,
            zorder=5,
            label="Current Genome",
        )
        ax1.legend()

        # Add generation range info
        gen_range = f"Generations {ancestry_df['generation'].min()}-{ancestry_df['generation'].max()}"
        ax1.text(
            0.02,
            0.98,
            gen_range,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Network complexity over generations
        ax2.plot(
            ancestry_df["generation"],
            ancestry_df["num_connections"],
            "g-o",
            label="Connections",
            linewidth=2,
        )
        ax2.plot(
            ancestry_df["generation"],
            ancestry_df["num_nodes"],
            "b-s",
            label="Nodes",
            linewidth=2,
        )
        ax2.set_title("Network Complexity Evolution")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Count")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Evolutionary Progression - Genome {self.genome_info.neat_genome_id}"
        )
        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"\n📊 Ancestry Summary ({len(ancestry_df)} generations):")
        print(
            f"  Fitness range: {ancestry_df['fitness'].min():.3f} → {ancestry_df['fitness'].max():.3f}"
        )
        print(
            f"  Fitness improvement: {ancestry_df['fitness'].max() - ancestry_df['fitness'].min():.3f}"
        )
        print(
            f"  Network growth: {ancestry_df['num_nodes'].iloc[0]} → {ancestry_df['num_nodes'].iloc[-1]} nodes"
        )
        print(
            f"  Connection growth: {ancestry_df['num_connections'].iloc[0]} → {ancestry_df['num_connections'].iloc[-1]} connections"
        )

    def plot_evolution_progression(
        self, max_generations: int = 50, figsize: Tuple[int, int] = (15, 8)
    ) -> None:
        """Plot the full evolutionary progression across all generations in the experiment"""
        from ..db import Population, Experiment, Genome

        with db.session_scope() as session:
            # Get the experiment
            experiment = session.get(Experiment, self.genome_info.experiment_id)
            if not experiment:
                print("No experiment data found")
                return

            # Get all populations for this experiment
            populations = (
                session.query(Population)
                .filter_by(experiment_id=self.genome_info.experiment_id)
                .order_by(Population.generation)
                .limit(max_generations)
                .all()
            )

            if len(populations) < 2:
                print("Insufficient population data for evolution plotting")
                return

            # Extract data
            generations = [p.generation for p in populations]
            best_fitness = [p.best_fitness for p in populations]
            mean_fitness = [p.mean_fitness for p in populations]
            std_fitness = [p.stdev_fitness for p in populations]
            pop_sizes = [p.population_size for p in populations]

            # Get ancestor fitness data
            ancestry_df = self.get_ancestry_tree(max_generations)
            ancestor_fitness = []
            ancestor_generations = []

            if not ancestry_df.empty:
                # Create a mapping of generation to ancestor fitness
                ancestry_by_gen = ancestry_df.groupby("generation")["fitness"].max()
                for gen in generations:
                    if gen in ancestry_by_gen.index:
                        ancestor_fitness.append(ancestry_by_gen[gen])
                        ancestor_generations.append(gen)

            _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # Fitness evolution with ancestor tracking
            ax1.plot(
                generations,
                best_fitness,
                "r-o",
                label="Best Fitness",
                linewidth=2,
                markersize=4,
            )
            ax1.plot(
                generations,
                mean_fitness,
                "b-s",
                label="Mean Fitness",
                linewidth=2,
                markersize=4,
            )

            # Plot ancestor fitness if available
            if ancestor_fitness and ancestor_generations:
                ax1.plot(
                    ancestor_generations,
                    ancestor_fitness,
                    "g-^",
                    label="Best Ancestor Fitness",
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                )

            ax1.fill_between(
                generations,
                [m - s for m, s in zip(mean_fitness, std_fitness)],
                [m + s for m, s in zip(mean_fitness, std_fitness)],
                alpha=0.3,
                color="blue",
                label="±1 Std Dev",
            )
            ax1.set_title("Population Fitness Evolution")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Population size evolution
            ax2.plot(generations, pop_sizes, "g-o", linewidth=2, markersize=4)
            ax2.set_title("Population Size Evolution")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Population Size")
            ax2.grid(True, alpha=0.3)

            # Fitness improvement over time
            fitness_improvement = [
                best_fitness[i] - best_fitness[0] for i in range(len(best_fitness))
            ]
            ax3.plot(
                generations,
                fitness_improvement,
                "purple",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax3.set_title("Cumulative Fitness Improvement")
            ax3.set_xlabel("Generation")
            ax3.set_ylabel("Fitness Improvement")
            ax3.grid(True, alpha=0.3)

            # Generation-to-generation improvement
            gen_improvements = [
                best_fitness[i] - best_fitness[i - 1] if i > 0 else 0
                for i in range(len(best_fitness))
            ]
            colors = ["green" if x > 0 else "red" for x in gen_improvements]
            ax4.bar(generations, gen_improvements, color=colors, alpha=0.7)
            ax4.set_title("Generation-to-Generation Improvement")
            ax4.set_xlabel("Generation")
            ax4.set_ylabel("Fitness Change")
            ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f"Full Evolution Progression - Experiment {experiment.name}")
            plt.tight_layout()
            plt.show()

            # Print evolution summary
            print(f"\n🧬 Evolution Summary ({len(generations)} generations):")
            print(f"  Initial fitness: {best_fitness[0]:.3f}")
            print(f"  Final fitness: {best_fitness[-1]:.3f}")
            print(f"  Total improvement: {best_fitness[-1] - best_fitness[0]:.3f}")
            print(
                f"  Average improvement per generation: {(best_fitness[-1] - best_fitness[0]) / len(generations):.3f}"
            )
            print(
                f"  Best generation: {generations[best_fitness.index(max(best_fitness))]} (fitness: {max(best_fitness):.3f})"
            )

            # Print ancestor analysis if available
            if ancestor_fitness and ancestor_generations:
                print(f"\n🌳 Ancestor Analysis:")
                print(
                    f"  Ancestor fitness range: {min(ancestor_fitness):.3f} → {max(ancestor_fitness):.3f}"
                )
                print(
                    f"  Ancestor fitness improvement: {max(ancestor_fitness) - min(ancestor_fitness):.3f}"
                )
                print(f"  Ancestor generations tracked: {len(ancestor_generations)}")

    def show_network(
        self, interactive: bool = False, renderer: str = "react", **kwargs
    ) -> Optional[str]:
        """
        Display the network structure.

        Args:
            interactive: If True, use interactive web-based visualization (Pyvis)
            **kwargs: Additional arguments passed to visualization method

        Returns:
            If interactive=True, returns path to HTML file. Otherwise None.
        """
        if interactive:
            return self.show_interactive_network(renderer=renderer, **kwargs)
        else:
            self.visualizer.plot_network(**kwargs)
            return None

    def show_interactive_network(self, renderer: str = "react", **kwargs) -> str:
        """
        Display interactive web-based network visualization with filtering controls.

        Loads annotations from database and integrates them into the visualization.

        Args:
            renderer: "react" (default) or "pyvis" for visualization backend
            **kwargs: Additional arguments passed to InteractiveNetworkViewer.show()

        Returns:
            Path to generated HTML file
        """
        # Load annotations for this genome
        annotations = AnnotationManager.get_annotations(self.genome_info.genome_id)

        # Get phenotype network structure from ExplaNEAT
        from explaneat.core.explaneat import ExplaNEAT
        
        explainer = ExplaNEAT(self.neat_genome, self.config)
        phenotype_network = explainer.get_phenotype_network()

        # Create interactive viewer with phenotype network
        viewer = InteractiveNetworkViewer(
            phenotype_network, self.config, annotations=annotations
        )
        # Store genome_info for annotation export
        viewer.genome_info = self.genome_info

        # Show the visualization
        if renderer == "react":
            return viewer.show_react(**kwargs)
        elif renderer == "pyvis":
            return viewer.show(**kwargs)
        else:
            raise ValueError(
                f"Unknown renderer '{renderer}'. Expected 'react' or 'pyvis'."
            )

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
            generation_genomes = (
                session.query(Genome)
                .filter_by(population_id=self.genome_info.population_id)
                .all()
            )

            # Get experiment progression
            experiment_populations = (
                session.query(Population)
                .filter_by(experiment_id=self.genome_info.experiment_id)
                .order_by(Population.generation)
                .all()
            )

            fitness_values = [
                g.fitness for g in generation_genomes if g.fitness is not None
            ]

            return {
                "generation_rank": sorted(fitness_values, reverse=True).index(
                    self.genome_info.fitness
                )
                + 1,
                "generation_size": len(generation_genomes),
                "generation_best": max(fitness_values) if fitness_values else None,
                "generation_mean": np.mean(fitness_values) if fitness_values else None,
                "generation_std": np.std(fitness_values) if fitness_values else None,
                "experiment_generations": len(experiment_populations),
                "experiment_best_fitness": population.best_fitness,
                "is_generation_best": (
                    self.genome_info.fitness == max(fitness_values)
                    if fitness_values
                    else False
                ),
            }

    def export_genome_data(self) -> Dict[str, Any]:
        """Export all genome data for external analysis"""
        return {
            "genome_info": self.genome_info.__dict__,
            "neat_genome_nodes": {
                str(node_id): {
                    "bias": node.bias,
                    "response": node.response,
                    "activation": node.activation,
                    "aggregation": node.aggregation,
                }
                for node_id, node in self.neat_genome.nodes.items()
            },
            "neat_genome_connections": {
                f"{conn_key[0]}_{conn_key[1]}": {
                    "weight": conn.weight,
                    "enabled": conn.enabled,
                }
                for conn_key, conn in self.neat_genome.connections.items()
            },
            "ancestry_tree": self.get_ancestry_tree().to_dict("records"),
            "performance_context": self.get_performance_context(),
            "gene_origins": self.trace_gene_origins().to_dict("records"),
        }

    def get_gene_origins(self, gene_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get all gene origins for the current genome's experiment.

        Args:
            gene_type: Filter by 'node' or 'connection'. If None, returns all genes.

        Returns:
            DataFrame with gene origin information
        """
        with db.session_scope() as session:
            query = session.query(GeneOrigin).filter_by(
                experiment_id=self.genome_info.experiment_id
            )

            if gene_type:
                query = query.filter_by(gene_type=gene_type)

            gene_origins = query.order_by(
                GeneOrigin.origin_generation, GeneOrigin.innovation_number
            ).all()

            if not gene_origins:
                return pd.DataFrame()

            data = []
            for gene in gene_origins:
                data.append(
                    {
                        "innovation_number": gene.innovation_number,
                        "gene_type": gene.gene_type,
                        "origin_generation": gene.origin_generation,
                        "origin_genome_id": str(gene.origin_genome_id),
                        "connection_from": gene.connection_from,
                        "connection_to": gene.connection_to,
                        "node_id": gene.node_id,
                        "initial_params": gene.initial_params,
                    }
                )

            return pd.DataFrame(data)

    def get_gene_origin_by_innovation(
        self, innovation_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the origin information for a specific innovation number.

        Args:
            innovation_number: The innovation number to look up

        Returns:
            Dictionary with gene origin info, or None if not found
        """
        with db.session_scope() as session:
            gene_origin = (
                session.query(GeneOrigin)
                .filter_by(
                    experiment_id=self.genome_info.experiment_id,
                    innovation_number=innovation_number,
                )
                .first()
            )

            if not gene_origin:
                return None

            return {
                "innovation_number": gene_origin.innovation_number,
                "gene_type": gene_origin.gene_type,
                "origin_generation": gene_origin.origin_generation,
                "origin_genome_id": str(gene_origin.origin_genome_id),
                "connection_from": gene_origin.connection_from,
                "connection_to": gene_origin.connection_to,
                "node_id": gene_origin.node_id,
                "initial_params": gene_origin.initial_params,
                "created_at": gene_origin.created_at,
            }

    def get_genes_by_generation(self, generation: int) -> pd.DataFrame:
        """
        Get all genes that first appeared in a specific generation.

        Args:
            generation: Generation number

        Returns:
            DataFrame with genes that originated in that generation
        """
        with db.session_scope() as session:
            gene_origins = (
                session.query(GeneOrigin)
                .filter_by(
                    experiment_id=self.genome_info.experiment_id,
                    origin_generation=generation,
                )
                .order_by(GeneOrigin.gene_type, GeneOrigin.innovation_number)
                .all()
            )

            if not gene_origins:
                return pd.DataFrame()

            data = []
            for gene in gene_origins:
                data.append(
                    {
                        "innovation_number": gene.innovation_number,
                        "gene_type": gene.gene_type,
                        "origin_genome_id": str(gene.origin_genome_id),
                        "connection_from": gene.connection_from,
                        "connection_to": gene.connection_to,
                        "node_id": gene.node_id,
                        "initial_params": gene.initial_params,
                    }
                )

            return pd.DataFrame(data)

    def get_innovation_timeline(self) -> pd.DataFrame:
        """
        Get a timeline of innovation introductions across all generations.

        Returns:
            DataFrame with innovation counts per generation
        """
        with db.session_scope() as session:
            gene_origins = (
                session.query(GeneOrigin)
                .filter_by(experiment_id=self.genome_info.experiment_id)
                .order_by(GeneOrigin.origin_generation)
                .all()
            )

            if not gene_origins:
                return pd.DataFrame()

            # Group by generation and gene type
            timeline_data = {}
            for gene in gene_origins:
                gen = gene.origin_generation
                if gen not in timeline_data:
                    timeline_data[gen] = {
                        "generation": gen,
                        "nodes": 0,
                        "connections": 0,
                        "total": 0,
                    }

                if gene.gene_type == "node":
                    timeline_data[gen]["nodes"] += 1
                elif gene.gene_type == "connection":
                    timeline_data[gen]["connections"] += 1
                timeline_data[gen]["total"] += 1

            return pd.DataFrame(list(timeline_data.values())).sort_values("generation")

    def plot_innovation_timeline(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the timeline of innovation introductions.

        Shows how many new nodes and connections were introduced in each generation.
        """
        timeline_df = self.get_innovation_timeline()

        if timeline_df.empty:
            print("No innovation data available")
            return

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Stacked bar chart
        ax1.bar(
            timeline_df["generation"], timeline_df["nodes"], label="Nodes", alpha=0.7
        )
        ax1.bar(
            timeline_df["generation"],
            timeline_df["connections"],
            bottom=timeline_df["nodes"],
            label="Connections",
            alpha=0.7,
        )
        ax1.set_title("New Innovations per Generation")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Number of New Innovations")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative innovation count
        timeline_df["cumulative_nodes"] = timeline_df["nodes"].cumsum()
        timeline_df["cumulative_connections"] = timeline_df["connections"].cumsum()
        timeline_df["cumulative_total"] = timeline_df["total"].cumsum()

        ax2.plot(
            timeline_df["generation"],
            timeline_df["cumulative_nodes"],
            "o-",
            label="Cumulative Nodes",
            linewidth=2,
        )
        ax2.plot(
            timeline_df["generation"],
            timeline_df["cumulative_connections"],
            "s-",
            label="Cumulative Connections",
            linewidth=2,
        )
        ax2.plot(
            timeline_df["generation"],
            timeline_df["cumulative_total"],
            "^-",
            label="Cumulative Total",
            linewidth=2,
            alpha=0.6,
        )
        ax2.set_title("Cumulative Innovation Growth")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Total Innovations")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Innovation Timeline - Experiment {self.genome_info.experiment_name}"
        )
        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"\n🧬 Innovation Summary:")
        print(f"  Total innovations: {timeline_df['total'].sum()}")
        print(f"  Total nodes: {timeline_df['nodes'].sum()}")
        print(f"  Total connections: {timeline_df['connections'].sum()}")
        print(f"  Generations with innovations: {len(timeline_df)}")
        print(
            f"  Average innovations per generation: {timeline_df['total'].mean():.2f}"
        )
