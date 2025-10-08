"""Database-integrated population management for ExplaNEAT"""
import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import neat
import numpy as np

from .base import db
from .models import Experiment, Population, Species, Genome, TrainingMetric, Result
from .serialization import serialize_population_config
from ..core.backproppop import BackpropPopulation


class DatabaseBackpropPopulation(BackpropPopulation):
    """BackpropPopulation with database integration for experiment tracking"""
    
    def __init__(self, config, x_train, y_train, experiment_name: str = None,
                 dataset_name: str = None, description: str = None,
                 database_url: str = None, ancestry_reporter=None):
        """
        Initialize population with database tracking

        Args:
            config: NEAT configuration
            x_train: Training data features
            y_train: Training data labels
            experiment_name: Name for this experiment
            dataset_name: Name of the dataset being used
            description: Description of the experiment
            database_url: Database connection URL (optional)
            ancestry_reporter: Optional AncestryReporter for parent tracking
        """
        super().__init__(config, x_train, y_train)

        # Initialize database connection
        if database_url:
            db.init_db(database_url)
        else:
            db.init_db()

        # Create experiment record and store ID
        self.experiment_id = self._create_experiment(
            experiment_name or f"NEAT_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_name,
            description,
            config
        )

        # Track current population and generation
        self.current_population_id = None
        self.current_generation = 0

        # Store ancestry reporter for parent tracking
        self.ancestry_reporter = ancestry_reporter
        if self.ancestry_reporter:
            self.ancestry_reporter.reproduction = self.reproduction
        
    def _create_experiment(self, name: str, dataset_name: str, description: str, 
                          config: neat.Config) -> str:
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
            session.add(experiment)
            session.flush()  # Get the ID
            experiment_id = experiment.id
            
        return experiment_id
    
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
    
    def run(self, fitness_function, n=None, nEpochs=5):
        """Run evolution with database tracking"""
        
        # Mark experiment as running
        with db.session_scope() as session:
            experiment = session.get(Experiment, self.experiment_id)
            experiment.status = 'running'
        
        generation = 0
        while n is None or generation < n:
            try:
                # Save population state before evaluation
                self.current_population_id = self._save_population_state(generation)
                self.current_generation = generation
                
                # Run one generation of evolution
                if generation == 0:
                    # Create initial population
                    self._create_initial_population()
                
                # Evaluate population
                self._evaluate_population(fitness_function, nEpochs)
                
                # Ensure all genomes have valid fitness values for NEAT
                for genome_id, genome in self.population.items():
                    if genome.fitness is None or genome.fitness != genome.fitness:  # NaN check
                        genome.fitness = -1000.0  # Set a default poor fitness
                
                # Save genomes after evaluation with ancestry tracking
                self._save_genomes(self.current_population_id, self.ancestry_reporter)
                
                # Save generation results
                valid_genomes = [g for g in self.population.values() if g.fitness is not None]
                if valid_genomes:
                    best_genome = max(valid_genomes, key=lambda g: g.fitness)
                    self._save_result('best_fitness', best_genome.fitness, generation)
                    
                    # Check for completion
                    if self.config.fitness_threshold is not None:
                        if best_genome.fitness >= self.config.fitness_threshold:
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
                    break
                
                generation += 1
                
            except Exception as e:
                # Mark experiment as failed
                with db.session_scope() as session:
                    experiment = session.get(Experiment, self.experiment_id)
                    experiment.status = 'failed'
                    experiment.end_time = datetime.utcnow()
                raise e
        
        # Mark experiment as completed
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
            self.backpropagate(self.xs, self.ys, nEpochs=nEpochs)
            
            # Then run the fitness function
            fitness_function(self.population, self.config, self.xs, self.ys, self.device)
            
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
            print(f"Error evaluating population: {e}")
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