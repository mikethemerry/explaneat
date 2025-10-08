#!/usr/bin/env python3
"""
Quick test script for genome visualization improvements
"""
import sys
from explaneat.analysis.genome_explorer import GenomeExplorer
from explaneat.db import db

# Initialize database
db.init_db()

# List experiments
print("Available experiments:")
experiments_df = GenomeExplorer.list_experiments()
if experiments_df.empty:
    print("No experiments found")
    sys.exit(1)

experiments_df = experiments_df.sort_values('created_at', ascending=False).reset_index(drop=True)
print(experiments_df[['name', 'generations', 'best_fitness']].head())

# Use the most recent experiment
experiment_id = experiments_df.iloc[0]['experiment_id']
print(f"\nLoading best genome from: {experiments_df.iloc[0]['name']}")

# Load the best genome
explorer = GenomeExplorer.load_best_genome(experiment_id)

# Test the new visualization
print("\nDisplaying network with new layered layout...")
print("Close the plot window to continue\n")

# Show with new layered layout (default)
explorer.show_network(layout='layered', show_layers=True, show_weights=True)

print("\nVisualization test complete!")
