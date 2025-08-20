"""
Database verification script for ExplaNEAT

This script connects to the database and shows what data has been stored
from your NEAT experiments.
"""
from explaneat.db import db, Experiment, Population, Genome, Species, Result, TrainingMetric

def check_database():
    """Check what data is in the database"""
    
    # Initialize database connection
    db.init_db()
    
    print("=" * 60)
    print("EXPLANEAT DATABASE CONTENTS")
    print("=" * 60)
    
    with db.session_scope() as session:
        # Check experiments
        experiments = session.query(Experiment).all()
        print(f"\n📊 EXPERIMENTS ({len(experiments)} total):")
        print("-" * 40)
        
        for exp in experiments:
            print(f"  ID: {exp.id}")
            print(f"  Name: {exp.name}")
            print(f"  Status: {exp.status}")
            print(f"  Dataset: {exp.dataset_name}")
            print(f"  Start: {exp.start_time}")
            print(f"  End: {exp.end_time}")
            print(f"  SHA: {exp.experiment_sha[:10]}...")
            print()
            
            # Check populations for this experiment
            populations = session.query(Population).filter_by(experiment_id=exp.id).all()
            print(f"  🧬 POPULATIONS ({len(populations)} generations):")
            
            for pop in populations:
                print(f"    Gen {pop.generation}: {pop.population_size} genomes, "
                      f"best fitness = {pop.best_fitness:.4f if pop.best_fitness else 'None'}")
                
                # Check genomes for this population
                genome_count = session.query(Genome).filter_by(population_id=pop.id).count()
                print(f"      └─ {genome_count} genomes stored")
                
                # Check species for this population
                species_count = session.query(Species).filter_by(population_id=pop.id).count()
                if species_count > 0:
                    print(f"      └─ {species_count} species")
            
            print()
            
            # Check results for this experiment
            results = session.query(Result).filter_by(experiment_id=exp.id).all()
            if results:
                print(f"  📈 RESULTS ({len(results)} measurements):")
                result_types = {}
                for result in results:
                    result_types[result.measurement_type] = result_types.get(result.measurement_type, 0) + 1
                
                for measurement_type, count in result_types.items():
                    print(f"    {measurement_type}: {count} measurements")
                print()
            
            # Check training metrics for this experiment
            training_metrics = session.query(TrainingMetric).join(Population).filter_by(experiment_id=exp.id).all()
            if training_metrics:
                print(f"  🏋️ TRAINING METRICS ({len(training_metrics)} records):")
                epochs = set(tm.epoch for tm in training_metrics)
                print(f"    Epochs tracked: {sorted(epochs)}")
                print()
            
            print("-" * 40)
        
        if not experiments:
            print("  No experiments found in database.")
            print("  Run 'python quickstart_with_db.py' to create some data!")
    
    print("\n" + "=" * 60)
    print("DATABASE CHECK COMPLETE")
    print("=" * 60)

def query_best_genomes():
    """Show the best genomes across all experiments"""
    
    print("\n🏆 TOP 5 BEST GENOMES:")
    print("-" * 30)
    
    with db.session_scope() as session:
        best_genomes = session.query(Genome).join(Population).join(Experiment).filter(
            Genome.fitness.isnot(None)
        ).order_by(Genome.fitness.desc()).limit(5).all()
        
        for i, genome in enumerate(best_genomes, 1):
            print(f"{i}. Fitness: {genome.fitness:.4f}")
            print(f"   Experiment: {genome.population.experiment.name}")
            print(f"   Generation: {genome.population.generation}")
            print(f"   Nodes: {genome.num_nodes}, Connections: {genome.num_enabled_connections}")
            print()

if __name__ == "__main__":
    try:
        check_database()
        query_best_genomes()
    except Exception as e:
        print(f"Error checking database: {e}")
        print("Make sure PostgreSQL is running and the database exists.")
        print("Run 'python -m explaneat db init' if needed.")