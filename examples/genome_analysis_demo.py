"""
Genome Analysis Demo

This script demonstrates how to use the GenomeExplorer API to:
1. Load and analyze genomes from experiments
2. Trace ancestry and evolutionary history
3. Visualize network structures and performance
4. Export data for further analysis

Run this after running test_full_lifecycle.py to have data in the database.
"""
import matplotlib.pyplot as plt
from explaneat.db import db
from explaneat.analysis import GenomeExplorer

def main():
    # Initialize database
    db.init_db()
    
    print("🧬 ExplaNEAT Genome Analysis Demo")
    print("=" * 50)
    
    # List available experiments
    print("\n📊 Available Experiments:")
    experiments_df = GenomeExplorer.list_experiments()
    print(experiments_df[['name', 'status', 'generations', 'best_fitness', 'created_at']].to_string())
    
    if experiments_df.empty:
        print("\nNo experiments found. Please run test_full_lifecycle.py first.")
        return
    
    # Load the best genome from the most recent experiment
    latest_experiment_id = experiments_df.iloc[0]['experiment_id']
    print(f"\n🔍 Analyzing experiment: {experiments_df.iloc[0]['name']}")
    
    # Create explorer instance
    explorer = GenomeExplorer.load_best_genome(latest_experiment_id)
    
    # Show genome summary
    print("\n" + "=" * 50)
    explorer.summary()
    
    # Analyze ancestry
    print("\n🌳 Ancestry Analysis:")
    ancestry_df = explorer.get_ancestry_tree()
    print(f"Found {len(ancestry_df)} generations in lineage")
    
    # Get lineage statistics
    lineage_stats = explorer.ancestry_analyzer.get_lineage_statistics()
    if lineage_stats:
        print(f"\nLineage Statistics:")
        fp = lineage_stats['fitness_progression']
        print(f"  Fitness trend: {fp['fitness_trend']}")
        print(f"  Initial fitness: {fp['initial_fitness']:.3f}")
        print(f"  Final fitness: {fp['final_fitness']:.3f}")
        print(f"  Best fitness: {fp['best_fitness']:.3f}")
        
        cp = lineage_stats['complexity_progression']
        print(f"\n  Complexity trend: {cp['complexity_trend']}")
        print(f"  Initial network: {cp['initial_nodes']} nodes, {cp['initial_connections']} connections")
        print(f"  Final network: {cp['final_nodes']} nodes, {cp['final_connections']} connections")
    
    # Trace gene origins
    print("\n🧬 Gene Origins:")
    gene_origins_df = explorer.trace_gene_origins()
    if not gene_origins_df.empty:
        origin_summary = gene_origins_df.groupby(['gene_type', 'origin_generation']).size()
        print("Genes introduced by generation:")
        print(origin_summary.to_string())
    
    # Performance context
    print("\n📈 Performance Context:")
    context = explorer.get_performance_context()
    print(f"  Rank in generation: {context['generation_rank']}/{context['generation_size']}")
    print(f"  Generation best fitness: {context['generation_best']:.3f}")
    print(f"  Is best in generation: {context['is_generation_best']}")
    
    # Visualizations
    print("\n🎨 Generating visualizations...")
    
    # 1. Network structure
    print("  - Network structure")
    explorer.show_network(figsize=(10, 8), layout='hierarchical')
    
    # 2. Training metrics
    if explorer.genome_info.training_metrics:
        print("  - Training metrics")
        explorer.plot_training_metrics()
    
    # 3. Ancestry fitness progression
    if len(ancestry_df) > 1:
        print("  - Ancestry fitness progression")
        explorer.plot_ancestry_fitness()
    
    # 4. Visualizer demonstrations
    print("  - Node properties")
    explorer.visualizer.plot_node_properties()
    
    print("  - Connection properties")
    explorer.visualizer.plot_connection_properties()
    
    # 5. Ancestry visualizations (if we have ancestry)
    if len(ancestry_df) > 1:
        print("  - Ancestry tree")
        explorer.ancestry_analyzer.visualizer.plot_ancestry_tree()
        
        print("  - Lineage progression")
        explorer.ancestry_analyzer.visualizer.plot_lineage_progression()
        
        print("  - Gene origins timeline")
        explorer.ancestry_analyzer.visualizer.plot_gene_origins_timeline(explorer.neat_genome)
    
    # Export data
    print("\n💾 Exporting genome data...")
    export_data = explorer.export_genome_data()
    print(f"Exported {len(export_data)} data sections")
    
    # Example: Compare with ancestor (if available)
    if len(ancestry_df) > 1:
        print("\n🔍 Comparing with oldest ancestor...")
        oldest_generation = ancestry_df['generation'].min()
        comparison = explorer.compare_with_ancestor(oldest_generation)
        
        if 'error' not in comparison:
            print(f"Changes from generation {oldest_generation} to {comparison['current_generation']}:")
            print(f"  Fitness change: {comparison['fitness_change']:.3f}")
            sc = comparison['structure_changes']
            print(f"  Nodes: +{sc['nodes_added']} / -{sc['nodes_removed']}")
            print(f"  Connections: +{sc['connections_added']} / -{sc['connections_removed']}")
    
    print("\n✅ Demo completed!")
    print("\nTips for using GenomeExplorer:")
    print("1. Use explorer.summary() for a quick overview")
    print("2. Use explorer.get_ancestry_tree() to trace lineage")
    print("3. Use explorer.trace_gene_origins() to see when genes appeared")
    print("4. Use explorer.show_network() to visualize the network")
    print("5. Use explorer.export_genome_data() to get all data for external analysis")

if __name__ == "__main__":
    main()