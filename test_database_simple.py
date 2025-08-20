"""
Simple database test that bypasses NEAT reproduction issues

This script tests the database integration by creating a minimal 
experiment and manually storing genome data.
"""
import numpy as np
import torch
import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene
import logging
import sys
from explaneat.db import db, Experiment, Population, Genome
from explaneat.db.serialization import serialize_genome, serialize_population_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_integration():
    """Test database integration with manual data creation"""
    
    logger.info("Testing database integration...")
    
    # Initialize database
    db.init_db()
    
    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-file.cfg"
    )
    
    # Create a sample experiment
    with db.session_scope() as session:
        experiment = Experiment(
            experiment_sha="test_experiment_123",
            name="Database Integration Test",
            description="Testing manual genome storage",
            dataset_name="Test Dataset",
            config_json=serialize_population_config(config),
            neat_config_text="# Test config"
        )
        session.add(experiment)
        session.flush()
        experiment_id = experiment.id
        
    logger.info(f"Created experiment: {experiment_id}")
    
    # Create a sample population
    with db.session_scope() as session:
        population = Population(
            experiment_id=experiment_id,
            generation=0,
            population_size=5,
            num_species=1,
            best_fitness=1.5,
            mean_fitness=1.0,
            config_json=serialize_population_config(config)
        )
        session.add(population)
        session.flush()
        population_id = population.id
        
    logger.info(f"Created population: {population_id}")
    
    # Create sample genomes
    with db.session_scope() as session:
        for i in range(5):
            # Create a simple NEAT genome
            neat_genome = neat.DefaultGenome(i + 1)
            neat_genome.fitness = 1.0 + (i * 0.1)  # Give it a fitness
            
            # Add some nodes and connections
            neat_genome.nodes[0] = DefaultNodeGene(0)  # Output node
            neat_genome.nodes[0].bias = 0.5
            neat_genome.nodes[0].activation = 'relu'
            neat_genome.nodes[0].aggregation = 'sum'
            neat_genome.nodes[0].response = 1.0
            
            # Add input connections (assuming 10 inputs like the quickstart)
            for input_id in range(-10, 0):
                neat_genome.nodes[input_id] = DefaultNodeGene(input_id)
                neat_genome.nodes[input_id].bias = 0.0
                neat_genome.nodes[input_id].activation = 'relu'
                neat_genome.nodes[input_id].aggregation = 'sum'
                neat_genome.nodes[input_id].response = 1.0
                
                # Add connection from input to output
                conn_key = (input_id, 0)
                neat_genome.connections[conn_key] = DefaultConnectionGene(conn_key)
                neat_genome.connections[conn_key].weight = np.random.randn()
                neat_genome.connections[conn_key].enabled = True
            
            # Convert to database genome
            db_genome = Genome.from_neat_genome(neat_genome, population_id)
            session.add(db_genome)
            
        logger.info("Created 5 sample genomes")
    
    # Query the data back
    with db.session_scope() as session:
        # Get experiment
        exp = session.get(Experiment, experiment_id)
        logger.info(f"Retrieved experiment: {exp.name}")
        
        # Get populations
        pops = session.query(Population).filter_by(experiment_id=experiment_id).all()
        logger.info(f"Retrieved {len(pops)} populations")
        
        # Get genomes
        genomes = session.query(Genome).filter_by(population_id=population_id).all()
        logger.info(f"Retrieved {len(genomes)} genomes")
        
        # Show genome details
        for genome in genomes:
            logger.info(f"Genome {genome.genome_id}: fitness={genome.fitness}, "
                       f"nodes={genome.num_nodes}, connections={genome.num_connections}")
            
            # Test deserialization
            neat_genome = genome.to_neat_genome(config)
            logger.info(f"  Deserialized genome has {len(neat_genome.nodes)} nodes, "
                       f"{len(neat_genome.connections)} connections")
    
    logger.info("Database integration test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_database_integration()
        if success:
            print("\n" + "="*50)
            print("✅ DATABASE INTEGRATION TEST PASSED!")
            print("="*50)
            print("The database can successfully:")
            print("- Store experiments, populations, and genomes")
            print("- Serialize/deserialize NEAT genomes")
            print("- Handle JSON storage of genome data")
            print("- Query data back from the database")
        else:
            print("❌ Database integration test failed")
    except Exception as e:
        print(f"❌ Database integration test failed with error: {e}")
        import traceback
        traceback.print_exc()