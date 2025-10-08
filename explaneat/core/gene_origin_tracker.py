"""
GeneOriginTracker: Tracks when each gene (innovation number) first appears

This reporter monitors the evolution and records the first appearance of each
gene (node or connection) in the population, storing:
- Which genome first introduced the gene
- What generation it appeared in
- Initial parameters (weight, bias, activation, etc.)
"""
from typing import Dict, Set, Tuple
import uuid

from ..db import db, GeneOrigin


class GeneOriginTracker:
    """
    Reporter that tracks gene origins during evolution.

    Monitors when each innovation number (node or connection) first appears
    and records the originating genome and generation.
    """

    def __init__(self, experiment_id: uuid.UUID):
        """
        Initialize gene origin tracker.

        Args:
            experiment_id: Database ID of the experiment
        """
        self.experiment_id = experiment_id

        # Cache of known innovation numbers to avoid database lookups
        # Key: (innovation_number, gene_type)
        self.known_innovations: Set[Tuple[int, str]] = set()

        # Load existing innovations from database for this experiment
        self._load_existing_innovations()

    def _load_existing_innovations(self):
        """Load already-recorded innovations from database to avoid duplicates."""
        with db.session_scope() as session:
            existing = session.query(
                GeneOrigin.innovation_number,
                GeneOrigin.gene_type
            ).filter_by(experiment_id=self.experiment_id).all()

            self.known_innovations = set(existing)

    def _record_gene_origin(self, innovation_number: int, gene_type: str,
                           genome_id: uuid.UUID, generation: int,
                           initial_params: Dict):
        """
        Record a gene origin in the database.

        Args:
            innovation_number: The NEAT innovation number
            gene_type: 'node' or 'connection'
            genome_id: Database UUID of the genome that introduced this gene
            generation: Generation number when gene first appeared
            initial_params: Initial parameters of the gene
        """
        with db.session_scope() as session:
            gene_origin = GeneOrigin(
                experiment_id=self.experiment_id,
                innovation_number=innovation_number,
                gene_type=gene_type,
                origin_genome_id=genome_id,
                origin_generation=generation,
                initial_params=initial_params
            )

            # Set type-specific fields
            if gene_type == 'connection':
                gene_origin.connection_from = initial_params.get('from_node')
                gene_origin.connection_to = initial_params.get('to_node')
            elif gene_type == 'node':
                gene_origin.node_id = innovation_number

            session.add(gene_origin)

    def _check_genome_for_new_genes(self, genome_id: uuid.UUID,
                                    neat_genome, generation: int):
        """
        Check a genome for new genes and record their origins.

        Args:
            genome_id: Database UUID of the genome
            neat_genome: The NEAT genome object
            generation: Current generation number
        """
        # Check nodes
        for node_id, node_gene in neat_genome.nodes.items():
            innovation_key = (node_id, 'node')

            if innovation_key not in self.known_innovations:
                # New node discovered!
                initial_params = {
                    'node_id': node_id,
                    'bias': node_gene.bias,
                    'response': node_gene.response,
                    'activation': node_gene.activation,
                    'aggregation': node_gene.aggregation
                }

                self._record_gene_origin(
                    innovation_number=node_id,
                    gene_type='node',
                    genome_id=genome_id,
                    generation=generation,
                    initial_params=initial_params
                )

                self.known_innovations.add(innovation_key)

        # Check connections
        for conn_key, conn_gene in neat_genome.connections.items():
            from_node, to_node = conn_key
            # Use a hash of the connection key as the innovation number
            # NEAT's connection key IS the innovation identifier
            innovation_number = hash(conn_key)

            innovation_key = (innovation_number, 'connection')

            if innovation_key not in self.known_innovations:
                # New connection discovered!
                initial_params = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'weight': conn_gene.weight,
                    'enabled': conn_gene.enabled
                }

                self._record_gene_origin(
                    innovation_number=innovation_number,
                    gene_type='connection',
                    genome_id=genome_id,
                    generation=generation,
                    initial_params=initial_params
                )

                self.known_innovations.add(innovation_key)

    def post_evaluate(self, config, population, species, best_genome):
        """
        Called after fitness evaluation - check all genomes for new genes.

        Args:
            config: NEAT config
            population: Dict of genome_id -> genome
            species: SpeciesSet object
            best_genome: Best genome in population
        """
        # We need to get the generation number and genome database IDs
        # This requires the ancestry_reporter to have already saved genomes
        # So we'll check genomes that were recently saved

        # For now, we'll scan all genomes in the population
        # In production, we'd want to only check new genomes

        # Note: This is called AFTER DatabaseReporter has saved genomes,
        # so the genomes should exist in the database with their UUIDs

        # We need a way to map NEAT genome IDs to database UUIDs
        # This will be passed from the ancestry_reporter

        # For now, we'll implement the core logic and wire it up later
        pass

    def process_population(self, population: Dict, generation: int,
                          genome_id_mapping: Dict[int, uuid.UUID]):
        """
        Process a population to find and record new genes.

        Args:
            population: Dict of NEAT genome_id -> genome object
            generation: Current generation number
            genome_id_mapping: Dict mapping NEAT genome IDs to database UUIDs
        """
        for neat_genome_id, neat_genome in population.items():
            # Get database UUID for this genome
            db_genome_id = genome_id_mapping.get(neat_genome_id)

            if db_genome_id is None:
                # Genome not yet in database, skip
                continue

            self._check_genome_for_new_genes(db_genome_id, neat_genome, generation)

    # NEAT Reporter Interface Methods

    def start_generation(self, generation: int):
        """Called at start of generation."""
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        """Called at end of generation."""
        pass

    def pre_backprop(self, config, population, species):
        """Called before backpropagation."""
        pass

    def post_backprop(self, config, population, species):
        """Called after backpropagation."""
        pass

    def post_reproduction(self, config, population, species):
        """Called after reproduction."""
        pass

    def pre_reproduction(self, config, population, species):
        """Called before reproduction."""
        pass

    def info(self, msg: str):
        """Log info messages."""
        pass

    def species_stagnant(self, sid, species):
        """Handle stagnant species."""
        pass

    def found_solution(self, config, generation, best):
        """Handle when solution is found."""
        pass

    def complete_extinction(self):
        """Handle complete extinction."""
        pass

    def start_experiment(self, config):
        """Called at start of experiment."""
        pass

    def end_experiment(self, config, population, species):
        """Called at end of experiment."""
        pass
