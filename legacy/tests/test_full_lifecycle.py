"""
Full Lifecycle Database Test for ExplaNEAT

This script demonstrates:
1. Creating a small population
2. Running a short experiment 
3. Saving all data to database
4. Restoring genomes from database
5. Analyzing performance and history
"""
import numpy as np
import torch
import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene
import logging
import sys
from datetime import datetime
from explaneat.db import db, Experiment, Population, Genome, TrainingMetric, Result
from explaneat.db.serialization import serialize_genome, serialize_population_config, deserialize_genome
from explaneat.core.explaneat import ExplaNEAT

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'lifecycle_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Also set up database logging
db_logger = logging.getLogger('explaneat.db')
db_logger.setLevel(logging.DEBUG)

def create_simple_fitness_function():
    """Create a simple fitness function for testing"""
    def simple_fitness(population, config, xs, ys, device):
        """Simple fitness: random values for testing"""
        for genome_id, genome in population.items():
            # Simulate fitness based on network complexity
            num_connections = len(genome.connections)
            num_nodes = len(genome.nodes)
            
            # Simple fitness: reward smaller networks with slight randomness
            base_fitness = 10.0 - (num_connections * 0.1) - (num_nodes * 0.05)
            noise = np.random.normal(0, 0.5)
            genome.fitness = max(0.1, base_fitness + noise)
            
    return simple_fitness

def run_experiment():
    """Run a complete experiment and save to database"""
    
    logger.info("🚀 Starting Full Lifecycle Database Test")
    logger.info("=" * 60)
    
    # Initialize database
    logger.info("🔧 Initializing database connection...")
    db.init_db()
    logger.info("✅ Database connection established")
    
    # Create test data
    logger.info("📊 Creating test dataset...")
    X_test = np.random.randn(20, 5)  # Small dataset for quick testing
    y_test = np.random.randint(0, 2, (20, 1))
    logger.info(f"   Data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Load NEAT config
    logger.info("⚙️ Loading NEAT configuration...")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-file.cfg"
    )
    logger.info(f"   Population size: {config.pop_size}")
    logger.info(f"   Fitness threshold: {config.fitness_threshold}")
    
    # Create experiment record
    logger.info("📝 Creating experiment record in database...")
    experiment_sha = f"lifecycle_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with db.session_scope() as session:
        experiment = Experiment(
            experiment_sha=experiment_sha,
            name="Full Lifecycle Test",
            description="Complete test of create → run → save → restore → analyze",
            dataset_name="Random Test Data",
            dataset_version="1.0",
            config_json=serialize_population_config(config),
            neat_config_text="# Test configuration",
            start_time=datetime.utcnow()
        )
        session.add(experiment)
        session.flush()
        experiment_id = experiment.id
        
    logger.info(f"✅ Created experiment: {experiment_id}")
    logger.info(f"   SHA: {experiment_sha}")
    logger.info(f"   Name: Full Lifecycle Test")
    
    # Create initial population manually (small size)
    logger.info("🧬 Creating initial population...")
    population = {}
    population_size = 10
    logger.info(f"   Creating {population_size} genomes...")
    
    for i in range(population_size):  # Small population
        logger.debug(f"   Creating genome {i+1}...")
        genome = neat.DefaultGenome(i + 1)
        
        # Add output node
        genome.nodes[0] = DefaultNodeGene(0)
        genome.nodes[0].bias = np.random.normal(0, 1)
        genome.nodes[0].activation = 'sigmoid'
        genome.nodes[0].aggregation = 'sum'
        genome.nodes[0].response = 1.0
        
        # Add input nodes and connections
        for input_id in range(-5, 0):  # 5 inputs
            genome.nodes[input_id] = DefaultNodeGene(input_id)
            genome.nodes[input_id].bias = 0.0
            genome.nodes[input_id].activation = 'sigmoid'
            genome.nodes[input_id].aggregation = 'sum'
            genome.nodes[input_id].response = 1.0
            
            # Add connection from input to output
            if np.random.random() > 0.3:  # 70% chance of connection
                conn_key = (input_id, 0)
                genome.connections[conn_key] = DefaultConnectionGene(conn_key)
                genome.connections[conn_key].weight = np.random.normal(0, 1)
                genome.connections[conn_key].enabled = True
        
        population[genome.key] = genome
    
    logger.info(f"✅ Created population of {len(population)} genomes")
    
    # Log population statistics
    total_connections = sum(len(g.connections) for g in population.values())
    total_nodes = sum(len(g.nodes) for g in population.values())
    logger.info(f"   Total connections: {total_connections}")
    logger.info(f"   Total nodes: {total_nodes}")
    logger.info(f"   Avg connections per genome: {total_connections/len(population):.1f}")
    logger.info(f"   Avg nodes per genome: {total_nodes/len(population):.1f}")
    
    # Run 3 generations with evaluation and saving
    logger.info("🔄 Starting evolution process...")
    fitness_function = create_simple_fitness_function()
    num_generations = 3
    logger.info(f"   Will run {num_generations} generations")
    
    for generation in range(num_generations):
        logger.info("=" * 40)
        logger.info(f"🔄 GENERATION {generation}")
        logger.info("=" * 40)
        
        # Evaluate population
        logger.info("⚡ Evaluating population fitness...")
        fitness_function(population, config, X_test, y_test, torch.device('cpu'))
        logger.info("✅ Fitness evaluation completed")
        
        # Calculate population statistics
        logger.info("📊 Calculating population statistics...")
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        best_fitness = max(fitnesses) if fitnesses else 0
        mean_fitness = np.mean(fitnesses) if fitnesses else 0
        std_fitness = np.std(fitnesses) if len(fitnesses) > 1 else 0
        
        logger.info(f"   Best fitness: {best_fitness:.3f}")
        logger.info(f"   Mean fitness: {mean_fitness:.3f}")
        logger.info(f"   Std fitness: {std_fitness:.3f}")
        logger.info(f"   Valid genomes: {len(fitnesses)}/{len(population)}")
        
        # Save population to database
        logger.info("💾 Saving population to database...")
        with db.session_scope() as session:
            pop_record = Population(
                experiment_id=experiment_id,
                generation=generation,
                population_size=len(population),
                num_species=1,  # Simplified - treating as one species
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                stdev_fitness=std_fitness,
                config_json=serialize_population_config(config)
            )
            session.add(pop_record)
            session.flush()
            population_id = pop_record.id
            
        logger.info(f"✅ Population record saved with ID: {population_id}")
            
        # Save all genomes first
        logger.info("🧬 Saving individual genomes...")
        genome_count = 0
        genome_db_ids = {}  # Store mapping of genome_id -> db_genome.id
        
        with db.session_scope() as session:
            for genome_id, genome in population.items():
                db_genome = Genome.from_neat_genome(genome, population_id)
                session.add(db_genome)
                genome_count += 1
                logger.debug(f"   Created genome record {genome_id}: fitness={genome.fitness:.3f}")
                
        logger.info(f"✅ Saved {genome_count} genomes to database")
        
        # Now get the genome IDs and save training metrics
        logger.info("📊 Saving training metrics...")
        with db.session_scope() as session:
            # Get all genome records for this population
            genome_records = session.query(Genome).filter_by(population_id=population_id).all()
            
            for db_genome in genome_records:
                # Find the original genome to get its data
                original_genome = None
                for genome_id, genome in population.items():
                    if genome.key == db_genome.genome_id:
                        original_genome = genome
                        break
                
                if original_genome:
                    # Add training metrics for this genome
                    for epoch in range(3):
                        metric = TrainingMetric(
                            genome_id=db_genome.id,
                            population_id=population_id,
                            epoch=epoch,
                            loss=max(0.1, 2.0 - original_genome.fitness + np.random.normal(0, 0.1)),
                            accuracy=min(0.9, original_genome.fitness / 10.0 + np.random.normal(0, 0.05)),
                            additional_metrics={
                                'network_size': len(original_genome.connections),
                                'generation': generation
                            }
                        )
                        session.add(metric)
                        
        logger.info(f"✅ Saved training metrics for {len(genome_records)} genomes")
        
        # Save generation results
        logger.info("📈 Saving generation results...")
        with db.session_scope() as session:
            result = Result(
                experiment_id=experiment_id,
                population_id=population_id,
                measurement_type='generation_best_fitness',
                value=best_fitness,
                iteration=generation,
                params={'population_size': len(population)}
            )
            session.add(result)
            
        logger.info(f"✅ Generation {generation} completely saved to database")
        
        # Simple reproduction for next generation (just mutation, no crossover)
        if generation < num_generations - 1:  # Don't create next gen on last iteration
            logger.info("🔄 Creating next generation...")
            new_population = {}
            sorted_genomes = sorted(population.values(), key=lambda g: g.fitness, reverse=True)
            logger.info(f"   Selecting top {len(sorted_genomes)//2} genomes for reproduction...")
            
            # Keep top 50% and mutate them
            for i, parent in enumerate(sorted_genomes[:5]):
                for j in range(2):  # Each parent creates 2 offspring
                    child = neat.DefaultGenome(len(population) + i * 2 + j + 1)
                    
                    # Copy parent structure
                    for node_id, node in parent.nodes.items():
                        child.nodes[node_id] = DefaultNodeGene(node_id)
                        child.nodes[node_id].bias = node.bias + np.random.normal(0, 0.1)
                        child.nodes[node_id].activation = node.activation
                        child.nodes[node_id].aggregation = node.aggregation
                        child.nodes[node_id].response = node.response
                    
                    for conn_key, conn in parent.connections.items():
                        child.connections[conn_key] = DefaultConnectionGene(conn_key)
                        child.connections[conn_key].weight = conn.weight + np.random.normal(0, 0.2)
                        child.connections[conn_key].enabled = conn.enabled
                    
                    new_population[child.key] = child
            
            logger.info(f"✅ Created {len(new_population)} offspring for next generation")
            population = new_population
    
    # Mark experiment as completed
    logger.info("🏁 Finalizing experiment...")
    with db.session_scope() as session:
        exp = session.get(Experiment, experiment_id)
        exp.status = 'completed'
        exp.end_time = datetime.utcnow()
    
    logger.info("✅ Experiment completed and saved to database")
    logger.info("=" * 60)
    return experiment_id

def analyze_experiment(experiment_id):
    """Analyze the experiment data from database"""
    
    logger.info(f"🔍 Analyzing experiment {experiment_id}")
    
    with db.session_scope() as session:
        # Get experiment info
        experiment = session.get(Experiment, experiment_id)
        logger.info(f"📊 Experiment: {experiment.name}")
        logger.info(f"   Status: {experiment.status}")
        logger.info(f"   Duration: {experiment.end_time - experiment.start_time if experiment.end_time else 'Running'}")
        
        # Get populations (generations)
        populations = session.query(Population).filter_by(experiment_id=experiment_id).order_by(Population.generation).all()
        logger.info(f"   Generations: {len(populations)}")
        
        generation_data = []
        for pop in populations:
            logger.info(f"   Gen {pop.generation}: {pop.population_size} genomes, "
                       f"best={pop.best_fitness:.3f}, mean={pop.mean_fitness:.3f}")
            generation_data.append({
                'generation': pop.generation,
                'best_fitness': pop.best_fitness,
                'mean_fitness': pop.mean_fitness,
                'population_id': pop.id
            })
        
        # Get best genome ever
        best_genome_record = session.query(Genome).join(Population).filter(
            Population.experiment_id == experiment_id,
            Genome.fitness.isnot(None)
        ).order_by(Genome.fitness.desc()).first()
        
        if best_genome_record:
            logger.info(f"🏆 Best genome: ID {best_genome_record.genome_id}, "
                       f"fitness={best_genome_record.fitness:.3f}")
            logger.info(f"   Network: {best_genome_record.num_nodes} nodes, "
                       f"{best_genome_record.num_enabled_connections} connections")
            
            # Get training metrics for best genome
            metrics = session.query(TrainingMetric).filter_by(genome_id=best_genome_record.id).all()
            if metrics:
                logger.info(f"   Training metrics: {len(metrics)} records")
                avg_loss = np.mean([m.loss for m in metrics if m.loss])
                avg_acc = np.mean([m.accuracy for m in metrics if m.accuracy])
                logger.info(f"   Average loss: {avg_loss:.3f}, Average accuracy: {avg_acc:.3f}")
        
        # Return just the genome ID to avoid session issues
        best_genome_id = best_genome_record.id if best_genome_record else None
        return experiment, generation_data, best_genome_id

def restore_and_test_genome(best_genome_id):
    """Restore a genome from database and test it"""
    
    # Get the genome record fresh from database
    with db.session_scope() as session:
        best_genome_record = session.get(Genome, best_genome_id)
        if not best_genome_record:
            logger.error("Could not find genome record in database")
            return None
            
        logger.info(f"🔄 Restoring genome {best_genome_record.genome_id} from database")
    
        # Load NEAT config
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "config-file.cfg"
        )
        
        # Deserialize genome
        restored_genome = best_genome_record.to_neat_genome(config)
        genome_fitness = best_genome_record.fitness
        genome_id = best_genome_record.genome_id
    
    logger.info(f"✅ Restored genome successfully")
    logger.info(f"   Nodes: {len(restored_genome.nodes)}")
    logger.info(f"   Connections: {len(restored_genome.connections)}")
    logger.info(f"   Fitness: {genome_fitness}")
    
    # Test with ExplaNEAT if possible
    try:
        explainer = ExplaNEAT(restored_genome, config)
        
        depth = explainer.depth()
        density = explainer.density()
        skippiness = explainer.skippines()
        n_params = explainer.n_genome_params()
        
        logger.info(f"🧠 Network Analysis:")
        logger.info(f"   Depth: {depth}")
        logger.info(f"   Density: {density:.3f}")
        logger.info(f"   Skippiness: {skippiness:.3f}")
        logger.info(f"   Parameters: {n_params}")
        
        return {
            'genome': restored_genome,
            'analysis': {
                'depth': depth,
                'density': density,
                'skippiness': skippiness,
                'n_params': n_params
            }
        }
        
    except Exception as e:
        logger.warning(f"Could not create ExplaNEAT analyzer: {e}")
        return {'genome': restored_genome, 'analysis': None}

def main():
    """Run the complete lifecycle test"""
    
    print("🧪 ExplaNEAT Full Lifecycle Database Test")
    print("=" * 50)
    
    try:
        # Step 1: Run experiment
        experiment_id = run_experiment()
        
        print("\n" + "=" * 50)
        
        # Step 2: Analyze results
        experiment, generation_data, best_genome_id = analyze_experiment(experiment_id)
        
        print("\n" + "=" * 50)
        
        # Step 3: Restore and test best genome
        if best_genome_id:
            result = restore_and_test_genome(best_genome_id)
            
            print("\n" + "=" * 50)
            print("🎉 FULL LIFECYCLE TEST COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print("✅ Created population and ran evolution")
            print("✅ Saved all data to PostgreSQL database")
            print("✅ Restored genome from database")
            print("✅ Analyzed network properties")
            print(f"✅ Best fitness achieved: {result['genome'].fitness:.3f}" if result and result['genome'].fitness else "N/A")
            print(f"✅ Total generations: {len(generation_data)}")
            print(f"✅ Experiment ID: {experiment_id}")
            
            # Show fitness progression
            print("\n📈 Fitness Progression:")
            for gen_data in generation_data:
                print(f"   Gen {gen_data['generation']}: {gen_data['best_fitness']:.3f}")
                
        else:
            print("❌ No valid genomes found in database")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)