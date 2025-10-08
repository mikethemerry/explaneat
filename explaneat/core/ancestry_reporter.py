"""
AncestryReporter: Captures parent-child relationships during evolution

This reporter maintains a mapping between NEAT genome IDs and database UUIDs
to enable proper ancestry tracking when saving genomes to the database.
"""
from typing import Dict, Optional, Tuple
import uuid


class AncestryReporter:
    """
    Reporter that tracks genome ancestry through evolution.

    Maintains mappings between:
    - NEAT genome IDs (integers) and database genome UUIDs
    - Parent relationships from NEAT's reproduction.ancestors

    This enables proper parent tracking when persisting genomes to database.
    """

    def __init__(self):
        # Map from NEAT genome ID to database UUID
        self.neat_to_db_id: Dict[int, uuid.UUID] = {}

        # Map from database UUID back to NEAT genome ID
        self.db_to_neat_id: Dict[uuid.UUID, int] = {}

        # Current generation number
        self.current_generation = 0

        # Reference to reproduction object (set externally)
        self.reproduction = None

    def register_genome(self, neat_genome_id: int, db_genome_id: uuid.UUID):
        """
        Register a genome's NEAT ID to database UUID mapping.

        Args:
            neat_genome_id: The NEAT-assigned integer genome ID
            db_genome_id: The database-assigned UUID for this genome
        """
        self.neat_to_db_id[neat_genome_id] = db_genome_id
        self.db_to_neat_id[db_genome_id] = neat_genome_id

    def get_db_id(self, neat_genome_id: int) -> Optional[uuid.UUID]:
        """
        Get the database UUID for a NEAT genome ID.

        Args:
            neat_genome_id: The NEAT-assigned integer genome ID

        Returns:
            Database UUID if found, None otherwise
        """
        return self.neat_to_db_id.get(neat_genome_id)

    def get_neat_id(self, db_genome_id: uuid.UUID) -> Optional[int]:
        """
        Get the NEAT genome ID for a database UUID.

        Args:
            db_genome_id: The database-assigned UUID

        Returns:
            NEAT genome ID if found, None otherwise
        """
        return self.db_to_neat_id.get(db_genome_id)

    def get_parent_ids(self, neat_genome_id: int) -> Tuple[Optional[uuid.UUID], Optional[uuid.UUID]]:
        """
        Get the database UUIDs of a genome's parents.

        Args:
            neat_genome_id: The NEAT-assigned integer genome ID

        Returns:
            Tuple of (parent1_db_id, parent2_db_id). Either may be None if:
            - This is a generation 0 genome (no parents)
            - Parent was not found in the mapping (shouldn't happen)
        """
        if self.reproduction is None:
            return None, None

        # Get parent NEAT IDs from reproduction.ancestors
        parent_neat_ids = self.reproduction.ancestors.get(neat_genome_id)

        if not parent_neat_ids:
            # Generation 0 genome or not found
            return None, None

        # Unpack parent IDs (could be empty tuple, single ID, or pair)
        if len(parent_neat_ids) == 0:
            return None, None
        elif len(parent_neat_ids) == 1:
            # Asexual reproduction (shouldn't happen in standard NEAT, but handle it)
            parent1_neat_id = parent_neat_ids[0]
            return self.get_db_id(parent1_neat_id), None
        else:
            # Standard crossover with two parents
            parent1_neat_id, parent2_neat_id = parent_neat_ids[0], parent_neat_ids[1]
            parent1_db_id = self.get_db_id(parent1_neat_id)
            parent2_db_id = self.get_db_id(parent2_neat_id)
            return parent1_db_id, parent2_db_id

    def start_generation(self, generation: int):
        """Called at the start of a new generation."""
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        """Called after fitness evaluation."""
        pass

    def pre_backprop(self, config, population, species):
        """Called before backpropagation - required by ExperimentReporterSet."""
        pass

    def post_backprop(self, config, population, species):
        """Called after backpropagation - required by ExperimentReporterSet."""
        pass

    def post_reproduction(self, config, population, species):
        """Called after reproduction - required by NEAT ReporterSet."""
        pass

    def pre_reproduction(self, config, population, species):
        """Called before reproduction - required by NEAT ReporterSet."""
        pass

    def end_generation(self, config, population, species_set):
        """Called at the end of a generation."""
        pass

    def info(self, msg: str):
        """Log info messages - required by NEAT ReporterSet."""
        pass

    def species_stagnant(self, sid, species):
        """Handle stagnant species - required by NEAT ReporterSet."""
        pass

    def found_solution(self, config, generation, best):
        """Handle when solution is found - required by NEAT ReporterSet."""
        pass

    def complete_extinction(self):
        """Handle complete extinction - required by NEAT ReporterSet."""
        pass

    def start_experiment(self, config):
        """Called at start of experiment - required by NEAT ReporterSet."""
        pass

    def end_experiment(self, config, population, species):
        """Called at end of experiment - required by NEAT ReporterSet."""
        pass

    def clear(self):
        """Clear all mappings. Useful when starting a new experiment."""
        self.neat_to_db_id.clear()
        self.db_to_neat_id.clear()
        self.current_generation = 0
