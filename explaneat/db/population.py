"""Database-integrated population management for ExplaNEAT"""
import hashlib
import json
import logging
import os
import time as _time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import neat
import numpy as np

from .base import db
from .models import Experiment, Population, Species, Genome, TrainingMetric, Result
from .serialization import serialize_population_config
from ..core.backproppop import BackpropPopulation

logger = logging.getLogger(__name__)


class DatabaseBackpropPopulation(BackpropPopulation):
    """BackpropPopulation with database integration for experiment tracking"""
    
    def __init__(self, config, x_train, y_train, xs_val=None, ys_val=None,
                 experiment_name: str = None,
                 dataset_name: str = None, description: str = None,
                 database_url: str = None, ancestry_reporter=None,
                 dataset_id: str = None, config_template_id: str = None,
                 initial_state=None,
                 _existing_experiment_id: Optional[str] = None):
        """
        Initialize population with database tracking

        Args:
            config: NEAT configuration
            x_train: Training data features
            y_train: Training data labels
            xs_val: Optional validation data features
            ys_val: Optional validation data labels
            experiment_name: Name for this experiment
            dataset_name: Name of the dataset being used
            description: Description of the experiment
            database_url: Database connection URL (optional)
            ancestry_reporter: Optional AncestryReporter for parent tracking
            dataset_id: Optional database ID of the dataset
            config_template_id: Optional database ID of the ConfigTemplate used
            initial_state: Optional (population_dict, species, generation) tuple
                for resuming from a prior state. Passed through to the parent
                BackpropPopulation constructor.
            _existing_experiment_id: Optional experiment UUID to reuse instead
                of creating a new Experiment row. Used by ``resume_from_db``.
        """
        super().__init__(config, x_train, y_train, xs_val=xs_val, ys_val=ys_val,
                         initial_state=initial_state)

        # Initialize database connection
        if database_url:
            db.init_db(database_url)
        else:
            db.init_db()

        # Create or reuse experiment record
        if _existing_experiment_id:
            self.experiment_id = _existing_experiment_id
        else:
            self.experiment_id = self._create_experiment(
                experiment_name or f"NEAT_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dataset_name,
                description,
                config,
                dataset_id=dataset_id,
                config_template_id=config_template_id,
            )

        # Track current population and generation
        self.current_population_id = None
        self.current_generation = initial_state[2] if initial_state else 0

        # Store ancestry reporter for parent tracking
        self.ancestry_reporter = ancestry_reporter
        if self.ancestry_reporter:
            self.ancestry_reporter.reproduction = self.reproduction
        
    def _create_experiment(self, name: str, dataset_name: str, description: str,
                          config: neat.Config, dataset_id: str = None,
                          config_template_id: str = None) -> str:
        """Create and save experiment record to database"""
        
        # Generate experiment SHA from config and parameters
        config_str = str(serialize_population_config(config))
        experiment_sha = hashlib.sha256(config_str.encode()).hexdigest()[:40]
        
        # Get git info if available
        git_commit_sha = None
        git_branch = None
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            git_commit_sha = repo.head.object.hexsha[:40]
            git_branch = repo.active_branch.name
        except:
            pass
        
        # Get hardware info
        hardware_info = {
            'cpu_count': os.cpu_count(),
            'platform': os.name
        }
        
        # Read NEAT config file content if available
        neat_config_text = ""
        if hasattr(config, 'filename') and config.filename:
            try:
                with open(config.filename, 'r') as f:
                    neat_config_text = f.read()
            except:
                neat_config_text = "Config file not found"
        
        with db.session_scope() as session:
            experiment = Experiment(
                experiment_sha=experiment_sha,
                name=name,
                description=description,
                dataset_name=dataset_name,
                config_json=serialize_population_config(config),
                neat_config_text=neat_config_text,
                start_time=datetime.utcnow(),
                git_commit_sha=git_commit_sha,
                git_branch=git_branch,
                hardware_info=hardware_info
            )
            if dataset_id:
                experiment.dataset_id = uuid.UUID(dataset_id) if isinstance(dataset_id, str) else dataset_id
            if config_template_id:
                experiment.config_template_id = (
                    uuid.UUID(config_template_id)
                    if isinstance(config_template_id, str)
                    else config_template_id
                )
            session.add(experiment)
            session.flush()  # Get the ID
            experiment_id = experiment.id

        return experiment_id
    
    @classmethod
    def resume_from_db(cls, experiment_id: str, config, x_train, y_train,
                       xs_val=None, ys_val=None, **kwargs) -> "DatabaseBackpropPopulation":
        """Reconstruct a population from the latest saved generation.

        Loads all genomes from the highest-generation Population row for the
        experiment, deserializes them into a NEAT population dict, re-speciates,
        and returns an instance ready to continue evolving.

        The existing experiment_id is reused — no new Experiment row is created.
        """
        from .serialization import deserialize_genome
        from .models import Population, Genome

        last_gen = cls._get_latest_generation(experiment_id)
        if last_gen is None:
            raise ValueError(
                f"No saved populations found for experiment {experiment_id}"
            )

        with db.session_scope() as session:
            pop_row = (
                session.query(Population)
                .filter_by(experiment_id=uuid.UUID(experiment_id), generation=last_gen)
                .first()
            )
            if not pop_row:
                raise ValueError(
                    f"Population row not found for experiment {experiment_id} gen {last_gen}"
                )

            genome_rows = session.query(Genome).filter_by(population_id=pop_row.id).all()
            population_dict = {}
            for g in genome_rows:
                neat_genome = deserialize_genome(g.genome_data, config)
                population_dict[neat_genome.key] = neat_genome

        # Build species set fresh — accepts loss of species continuity per design
        species = config.species_set_type(config.species_set_config, None)
        species.speciate(config, population_dict, last_gen + 1)

        initial_state = (population_dict, species, last_gen + 1)
        return cls(
            config, x_train, y_train, xs_val=xs_val, ys_val=ys_val,
            initial_state=initial_state,
            _existing_experiment_id=experiment_id,
            **kwargs,
        )

    @staticmethod
    def _get_latest_generation(experiment_id: str) -> Optional[int]:
        """Return the highest generation number saved for an experiment.

        Returns None if no Population rows exist for this experiment.
        """
        from .models import Population
        with db.session_scope() as session:
            result = (
                session.query(Population.generation)
                .filter_by(experiment_id=uuid.UUID(experiment_id))
                .order_by(Population.generation.desc())
                .first()
            )
            return result[0] if result else None

    def _save_population_state(self, generation: int) -> str:
        """Save current population state to database"""
        
        # Calculate population statistics
        fitnesses = [genome.fitness for genome in self.population.values() if genome.fitness is not None]
        best_fitness = max(fitnesses) if fitnesses else None
        mean_fitness = np.mean(fitnesses) if fitnesses else None
        stdev_fitness = np.std(fitnesses) if len(fitnesses) > 1 else None
        
        # Count species
        species_counts = {}
        for genome_id, genome in self.population.items():
            species_id = getattr(genome, 'species_id', None)
            if species_id is not None:
                species_counts[species_id] = species_counts.get(species_id, 0) + 1
        
        with db.session_scope() as session:
            population_record = Population(
                experiment_id=self.experiment_id,
                generation=generation,
                population_size=len(self.population),
                num_species=len(species_counts),
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                stdev_fitness=stdev_fitness,
                config_json=serialize_population_config(self.config)
            )
            session.add(population_record)
            session.flush()
            population_id = population_record.id
            
        return population_id
    
    def _save_genomes(self, population_id: str, ancestry_reporter=None):
        """Save all genomes in current population to database with ancestry tracking

        Args:
            population_id: Database ID of the population
            ancestry_reporter: Optional AncestryReporter for parent tracking
        """

        with db.session_scope() as session:
            for genome_id, neat_genome in self.population.items():
                # Determine species assignment (simplified)
                species_id = getattr(neat_genome, 'species_id', None)

                # Get parent database IDs if ancestry reporter is available
                parent1_db_id = None
                parent2_db_id = None
                if ancestry_reporter is not None:
                    parent1_db_id, parent2_db_id = ancestry_reporter.get_parent_ids(genome_id)

                genome_record = Genome.from_neat_genome(
                    neat_genome,
                    population_id,
                    species_id=species_id,
                    parent1_id=parent1_db_id,
                    parent2_id=parent2_db_id
                )
                session.add(genome_record)
                session.flush()  # Flush to get the genome ID assigned

                # Register this genome in the ancestry tracker for next generation
                if ancestry_reporter is not None:
                    ancestry_reporter.register_genome(genome_id, genome_record.id)
    
    def _save_training_metrics(self, genome_id: int, population_id: str, 
                              epoch: int, loss: float, additional_metrics: Dict[str, Any] = None):
        """Save training metrics for a genome"""
        
        with db.session_scope() as session:
            # Find the genome record
            genome_record = session.query(Genome).filter_by(
                population_id=population_id, 
                genome_id=genome_id
            ).first()
            
            if genome_record:
                metric = TrainingMetric(
                    genome_id=genome_record.id,
                    population_id=population_id,
                    epoch=epoch,
                    loss=loss,
                    additional_metrics=additional_metrics or {}
                )
                session.add(metric)
    
    def _save_result(self, measurement_type: str, value: float, iteration: int = None,
                    params: Dict[str, Any] = None, genome_id: int = None):
        """Save a general result measurement"""
        
        with db.session_scope() as session:
            genome_record_id = None
            if genome_id and self.current_population_id:
                genome_record = session.query(Genome).filter_by(
                    population_id=self.current_population_id,
                    genome_id=genome_id
                ).first()
                if genome_record:
                    genome_record_id = genome_record.id
            
            result = Result(
                experiment_id=self.experiment_id,
                population_id=self.current_population_id,
                genome_id=genome_record_id,
                measurement_type=measurement_type,
                value=value,
                iteration=iteration,
                params=params or {}
            )
            session.add(result)
    
    def run(self, fitness_function, n=None, nEpochs=5, patience=None):
        """Run evolution with database tracking.

        Args:
            fitness_function: Fitness evaluation function
            n: Number of generations to run
            nEpochs: Backpropagation epochs per generation
            patience: Early stopping patience (generations without improvement).
                      None disables early stopping.
        """
        # Mark experiment as running
        with db.session_scope() as session:
            experiment = session.get(Experiment, self.experiment_id)
            experiment.status = 'running'

        generations_without_improvement = 0
        best_fitness_seen = None

        logger.info("Starting evolution: %d generations, %d epochs BP, pop_size=%d",
                     n or 0, nEpochs, len(self.population))

        generation = 0
        while n is None or generation < n:
            try:
                gen_start = _time.time()
                logger.info("=== Generation %d/%d ===", generation, n or 0)
                self.reporters.start_generation(generation)

                # Save population state before evaluation
                self.current_population_id = self._save_population_state(generation)
                self.current_generation = generation

                # Run one generation of evolution
                if generation == 0:
                    # Create initial population
                    self._create_initial_population()

                # Evaluate population
                logger.info("  Evaluating population (%d genomes, %d BP epochs)...",
                            len(self.population), nEpochs)
                eval_start = _time.time()
                self._evaluate_population(fitness_function, nEpochs)
                logger.info("  Evaluation done in %.1fs", _time.time() - eval_start)

                # Ensure all genomes have valid fitness values for NEAT
                for genome_id, genome in self.population.items():
                    if genome.fitness is None or genome.fitness != genome.fitness:  # NaN check
                        genome.fitness = -1000.0  # Set a default poor fitness

                # Find best genome and report
                valid_genomes = [g for g in self.population.values() if g.fitness is not None]
                best_genome = max(valid_genomes, key=lambda g: g.fitness) if valid_genomes else None

                fitnesses = [g.fitness for g in valid_genomes] if valid_genomes else []
                logger.info("  Best fitness: %.4f | Mean: %.4f | Species: %d",
                            best_genome.fitness if best_genome else 0,
                            float(np.mean(fitnesses)) if fitnesses else 0,
                            len(self.species.species) if self.species else 0)

                # Track the best genome ever seen
                if best_genome and (self.best_genome is None or best_genome.fitness > self.best_genome.fitness):
                    self.best_genome = best_genome

                # Early stopping via patience
                if patience is not None and best_genome:
                    if best_fitness_seen is None or best_genome.fitness > best_fitness_seen:
                        best_fitness_seen = best_genome.fitness
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                    if generations_without_improvement >= patience:
                        logger.info("  Early stopping: no improvement for %d generations", patience)
                        self.reporters.post_evaluate(self.config, self.population, self.species, best_genome)
                        break

                self.reporters.post_evaluate(self.config, self.population, self.species, best_genome)

                # Save genomes after evaluation with ancestry tracking
                self._save_genomes(self.current_population_id, self.ancestry_reporter)

                # Save generation results
                if best_genome:
                    self._save_result('best_fitness', best_genome.fitness, generation)

                    # Check for completion
                    if self.config.fitness_threshold is not None:
                        if best_genome.fitness >= self.config.fitness_threshold:
                            logger.info("  Fitness threshold reached!")
                            break
                else:
                    # No valid fitness values, save 0 as best fitness
                    self._save_result('best_fitness', 0.0, generation)

                self._save_result('population_size', len(self.population), generation)

                # Create next generation
                self.population = self.reproduction.reproduce(self.config, self.species,
                                                            self.config.pop_size, generation)

                # Check for extinction
                if not self.species.species:
                    logger.warning("  Complete extinction at generation %d", generation)
                    break

                # Re-speciate the new population
                self.species.speciate(self.config, self.population, generation)

                logger.info("  Generation %d complete in %.1fs", generation, _time.time() - gen_start)

                generation += 1
                
            except Exception as e:
                # Mark experiment as failed
                with db.session_scope() as session:
                    experiment = session.get(Experiment, self.experiment_id)
                    experiment.status = 'failed'
                    experiment.end_time = datetime.utcnow()
                raise e
        
        # Mark experiment as completed
        logger.info("Evolution complete: %d generations, best fitness %.4f",
                     generation + 1,
                     self.best_genome.fitness if self.best_genome else 0)
        with db.session_scope() as session:
            experiment = session.get(Experiment, self.experiment_id)
            experiment.status = 'completed'
            experiment.end_time = datetime.utcnow()

        # Return best genome
        return max(self.population.values(), key=lambda g: g.fitness or float('-inf'))
    
    def _create_initial_population(self):
        """Create initial population (wrapper for parent method)"""
        # This uses the parent class method
        pass
    
    def _evaluate_population(self, fitness_function, nEpochs):
        """Evaluate population using backpropagation and fitness function"""

        try:
            # First run backpropagation on the population
            bp_start = _time.time()
            logger.info("    Backpropagating %d genomes × %d epochs...",
                        len(self.population), nEpochs)
            self.backpropagate(self.xs, self.ys, nEpochs=nEpochs)
            logger.info("    Backprop done in %.1fs", _time.time() - bp_start)

            # Then run the fitness function
            fit_start = _time.time()
            fitness_function(self.population, self.config, self.xs, self.ys, self.device)
            logger.info("    Fitness eval done in %.1fs", _time.time() - fit_start)

            # Save training metrics for each genome (simplified)
            for genome_id, genome in self.population.items():
                for epoch in range(nEpochs):
                    self._save_training_metrics(
                        genome_id,
                        self.current_population_id,
                        epoch,
                        genome.fitness or 0,  # Use actual fitness after evaluation
                        {'epoch_fitness': genome.fitness or 0}
                    )

        except Exception as e:
            logger.exception("Error evaluating population")
            # Set all genome fitness to a small negative value on failure (not infinity)
            for genome_id, genome in self.population.items():
                genome.fitness = -1000.0
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the current experiment"""
        
        with db.session_scope() as session:
            experiment = session.get(Experiment, self.experiment_id)
            populations = session.query(Population).filter_by(experiment_id=experiment.id).all()
            
            return {
                'experiment_id': str(experiment.id),
                'experiment_name': experiment.name,
                'status': experiment.status,
                'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
                'end_time': experiment.end_time.isoformat() if experiment.end_time else None,
                'generations_completed': len(populations),
                'dataset_name': experiment.dataset_name,
                'best_fitness': max([p.best_fitness for p in populations if p.best_fitness]) if populations else None
            }


def compute_remaining_generations(last_gen: int, target: int) -> int:
    """How many more generations to run after the last completed one.

    Returns 0 if we've already reached or exceeded the target.
    """
    remaining = target - (last_gen + 1)
    return max(0, remaining)