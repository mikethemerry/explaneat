#!/usr/bin/env python3
"""
Example usage of the NEAT Database Connector.

This example demonstrates how to:
1. Initialize the database connector
2. Create an experiment
3. Save genome generations 
4. Query experiment data

Note: This example requires a running PostgreSQL database.
Set the following environment variables:
- POSTGRES_HOST (default: localhost)
- POSTGRES_PORT (default: 5432)
- POSTGRES_DB (default: explaneat)
- POSTGRES_USER (default: postgres)
- POSTGRES_PASSWORD (required)

Or set DATABASE_URL with a complete connection string.
"""

import os
import sys
import logging
from explaneat.database import NEATDatabaseConnector

# Enable logging to see what the connector is doing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MockGenome:
    """Mock genome for demonstration purposes."""
    def __init__(self, genome_id, fitness=0.5):
        self.key = genome_id
        self.fitness = fitness
        self.adjusted_fitness = fitness * 0.9
        
        # Simple node structure
        self.nodes = {
            1: type('Node', (), {'type': 'input', 'bias': 0.0, 'response': 1.0, 
                               'activation': 'sigmoid', 'aggregation': 'sum'}),
            2: type('Node', (), {'type': 'hidden', 'bias': 0.5, 'response': 1.0,
                               'activation': 'sigmoid', 'aggregation': 'sum'}), 
            3: type('Node', (), {'type': 'output', 'bias': -0.2, 'response': 1.0,
                               'activation': 'sigmoid', 'aggregation': 'sum'})
        }
        
        # Simple connection structure  
        self.connections = {
            (1, 2): type('Connection', (), {'weight': 0.7, 'enabled': True, 
                                          'in_node_id': 1, 'out_node_id': 2, 'innovation': 1}),
            (2, 3): type('Connection', (), {'weight': -0.3, 'enabled': True,
                                          'in_node_id': 2, 'out_node_id': 3, 'innovation': 2}),
            (1, 3): type('Connection', (), {'weight': 0.1, 'enabled': False,
                                          'in_node_id': 1, 'out_node_id': 3, 'innovation': 3})
        }


def main():
    """Run the database connector example."""
    print("NEAT Database Connector Example")
    print("=" * 40)
    
    # Check if database is configured
    if not any(var in os.environ for var in ['DATABASE_URL', 'POSTGRES_PASSWORD']):
        print("WARNING: No database configuration found.")
        print("This example will run with database disabled.")
        print("\nTo enable database persistence, set environment variables:")
        print("  DATABASE_URL=postgresql://user:pass@host:port/database")
        print("  OR")
        print("  POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB")
        print("\nRunning with database disabled...\n")
    
    # Initialize connector (will be disabled if no database configured)
    with NEATDatabaseConnector() as db:
        print(f"Database enabled: {db.enabled}")
        
        if not db.enabled:
            print("Database operations will be no-ops.")
            print("All methods will return None/False gracefully.")
        
        # Try to initialize schema (safe even if disabled)
        print("\nInitializing database schema...")
        schema_success = db.initialize_schema(drop_existing=False)
        print(f"Schema initialization: {'Success' if schema_success else 'Skipped (disabled)'}")
        
        # Create an experiment
        print("\nCreating experiment...")
        config = {
            'pop_size': 50,
            'fitness_threshold': 0.95,
            'no_fitness_termination': False,
            'random_seed': 42
        }
        
        experiment_id = db.create_experiment(
            name="example_experiment",
            config=config,
            description="Example NEAT experiment for database connector demo"
        )
        print(f"Experiment created with ID: {experiment_id}")
        
        # Create mock population
        print("\nCreating mock population...")
        population = []
        for i in range(10):
            fitness = 0.1 + (i * 0.08)  # Increasing fitness from 0.1 to 0.82
            genome = MockGenome(i + 1, fitness)
            population.append((i + 1, genome))
        
        print(f"Created {len(population)} genomes with fitness range {population[0][1].fitness:.2f} - {population[-1][1].fitness:.2f}")
        
        if experiment_id:
            # Save multiple generations
            for generation in range(3):
                print(f"\nSaving generation {generation}...")
                
                # Slightly improve fitness each generation
                for i, (genome_id, genome) in enumerate(population):
                    genome.fitness += 0.05 * generation  # Slight improvement
                
                success = db.save_generation(experiment_id, generation, population)
                print(f"Generation {generation} saved: {'Success' if success else 'Failed/Skipped'}")
            
            # Query best genomes
            print("\nQuerying best genomes...")
            best_genomes = db.get_best_genomes(experiment_id, n=5)
            if best_genomes:
                print(f"Found {len(best_genomes)} best genomes:")
                for genome in best_genomes:
                    print(f"  Genome {genome['genome_id']}: fitness={genome['fitness']:.3f}, generation={genome['generation_number']}")
            else:
                print("No best genomes found (database disabled or empty)")
            
            # Load a specific generation
            print("\nLoading generation 1...")
            gen_data = db.load_generation(experiment_id, 1)
            if gen_data:
                print(f"Loaded generation with {len(gen_data['genomes'])} genomes")
                print(f"Best fitness in generation: {gen_data['generation_info']['best_fitness']:.3f}")
                print(f"Average fitness: {gen_data['generation_info']['avg_fitness']:.3f}")
            else:
                print("Generation not found (database disabled or empty)")
            
            # Get experiment info
            print("\nGetting experiment info...")
            exp_info = db.get_experiment_info(experiment_id)
            if exp_info:
                print(f"Experiment: {exp_info['name']}")
                print(f"Created: {exp_info['created_at']}")
                print(f"Status: {exp_info['status']}")
            else:
                print("Experiment info not found (database disabled)")
        
        print("\nExample completed successfully!")
        print("Database connection will be closed automatically.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        print(f"\nError running example: {e}")
        sys.exit(1)