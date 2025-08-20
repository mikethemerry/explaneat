"""Quick database check"""
from explaneat.db import db, Experiment, Population, Genome

try:
    db.init_db()
    
    with db.session_scope() as session:
        exp_count = session.query(Experiment).count()
        pop_count = session.query(Population).count()  
        genome_count = session.query(Genome).count()
        
        print(f"Experiments: {exp_count}")
        print(f"Populations: {pop_count}")
        print(f"Genomes: {genome_count}")
        
        if exp_count > 0:
            exp = session.query(Experiment).first()
            print(f"Latest experiment: {exp.name} ({exp.status})")
            
except Exception as e:
    print(f"Database error: {e}")