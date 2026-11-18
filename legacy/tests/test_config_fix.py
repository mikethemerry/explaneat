#!/usr/bin/env python3
"""Quick test to verify config reconstruction works"""
import sys
from explaneat.db import db
from explaneat.analysis.genome_explorer import GenomeExplorer

# Initialize database
db.init_db()

# List experiments
experiments = GenomeExplorer.list_experiments()
if experiments.empty:
    print("No experiments found")
    sys.exit(1)

print(f"Found {len(experiments)} experiments")
print("\nTesting experiment selection...")

# Try to select the first experiment
exp_id = experiments.iloc[0]['experiment_id']
print(f"Attempting to load experiment: {exp_id}")

try:
    explorer = GenomeExplorer.load_best_genome(exp_id)
    print("✅ Successfully loaded experiment!")
    print(f"   Genome ID: {explorer.genome_info.genome_id}")
    print(f"   Fitness: {explorer.genome_info.fitness}")
except Exception as e:
    print(f"❌ Failed to load experiment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)




