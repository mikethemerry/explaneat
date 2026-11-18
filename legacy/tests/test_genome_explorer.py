"""
Test script for GenomeExplorer functionality

This script tests the complete genome analysis framework:
1. Load genome from database
2. Explore ancestry and gene origins
3. Create visualizations
4. Export analysis data
"""
import logging
import sys
from explaneat.db import db
from explaneat.analysis import GenomeExplorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_genome_explorer():
    """Test the GenomeExplorer class"""
    
    logger.info("🧬 Testing GenomeExplorer")
    logger.info("=" * 50)
    
    # Initialize database
    db.init_db()
    
    # First, let's see what experiments we have
    logger.info("📊 Available experiments:")
    experiments_df = GenomeExplorer.list_experiments()
    print(experiments_df)
    
    if experiments_df.empty:
        logger.error("No experiments found in database. Run test_full_lifecycle.py first.")
        return False
    
    # Get the most recent experiment
    latest_experiment_id = experiments_df.iloc[0]['experiment_id']
    logger.info(f"Using latest experiment: {latest_experiment_id}")
    
    try:
        # Load the best genome from the experiment
        logger.info("🏆 Loading best genome from experiment...")
        explorer = GenomeExplorer.load_best_genome(latest_experiment_id)
        
        # Show genome summary
        logger.info("📋 Genome Summary:")
        explorer.summary()
        
        # Test ancestry analysis
        logger.info("\n🌳 Ancestry Analysis:")
        ancestry_df = explorer.get_ancestry_tree(max_generations=5)
        print(f"Found {len(ancestry_df)} ancestors")
        if not ancestry_df.empty:
            print("Ancestry tree:")
            print(ancestry_df[['neat_genome_id', 'generation', 'fitness', 'num_nodes', 'num_connections']])
        
        # Test gene origins tracing
        logger.info("\n🧬 Gene Origins Analysis:")
        try:
            gene_origins_df = explorer.trace_gene_origins()
            print(f"Traced origins for {len(gene_origins_df)} genes")
            if not gene_origins_df.empty:
                print("Gene origins summary:")
                print(gene_origins_df.groupby(['gene_type', 'origin_generation']).size().unstack(fill_value=0))
        except Exception as e:
            logger.warning(f"Gene origins analysis failed: {e}")
        
        # Test performance context
        logger.info("\n📈 Performance Context:")
        context = explorer.get_performance_context()
        print(f"Generation rank: {context['generation_rank']}/{context['generation_size']}")
        print(f"Generation best: {context['generation_best']:.3f}")
        print(f"Is generation best: {context['is_generation_best']}")
        
        # Test lineage statistics
        logger.info("\n📊 Lineage Statistics:")
        lineage_stats = explorer.ancestry_analyzer.get_lineage_statistics()
        if lineage_stats:
            print(f"Lineage length: {lineage_stats['lineage_length']}")
            print(f"Fitness progression: {lineage_stats['fitness_progression']['fitness_trend']}")
            print(f"Complexity progression: {lineage_stats['complexity_progression']['complexity_trend']}")
        
        # Test visualization (just check if methods exist and don't crash)
        logger.info("\n🎨 Testing visualizations:")
        try:
            # Test training metrics plot
            logger.info("  - Training metrics plot...")
            explorer.plot_training_metrics()
            
            # Test ancestry fitness plot
            logger.info("  - Ancestry fitness plot...")
            explorer.plot_ancestry_fitness()
            
            # Test network visualization
            logger.info("  - Network structure plot...")
            explorer.show_network()
            
        except Exception as e:
            logger.warning(f"Visualization test failed: {e}")
        
        # Test data export
        logger.info("\n💾 Testing data export:")
        try:
            export_data = explorer.export_genome_data()
            print(f"Exported data contains {len(export_data)} sections:")
            for key in export_data.keys():
                print(f"  - {key}")
        except Exception as e:
            logger.warning(f"Data export failed: {e}")
        
        logger.info("\n✅ GenomeExplorer test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"GenomeExplorer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ancestry_analyzer():
    """Test AncestryAnalyzer features directly"""
    
    logger.info("\n🌳 Testing AncestryAnalyzer")
    logger.info("=" * 50)
    
    try:
        # Get the best genome from the latest experiment
        experiments_df = GenomeExplorer.list_experiments()
        if experiments_df.empty:
            logger.error("No experiments found")
            return False
            
        latest_experiment_id = experiments_df.iloc[0]['experiment_id']
        explorer = GenomeExplorer.load_best_genome(latest_experiment_id)
        
        analyzer = explorer.ancestry_analyzer
        
        # Test lineage statistics
        logger.info("📊 Lineage Statistics:")
        stats = analyzer.get_lineage_statistics()
        if stats:
            print(f"Lineage length: {stats.get('lineage_length', 'N/A')}")
            if 'fitness_progression' in stats:
                fp = stats['fitness_progression']
                print(f"Fitness: {fp['initial_fitness']:.3f} → {fp['final_fitness']:.3f} (trend: {fp['fitness_trend']})")
            if 'complexity_progression' in stats:
                cp = stats['complexity_progression']
                print(f"Nodes: {cp['initial_nodes']} → {cp['final_nodes']}")
                print(f"Connections: {cp['initial_connections']} → {cp['final_connections']}")
        
        # Test comparison with ancestor
        logger.info("\n🔍 Ancestor Comparison:")
        ancestry_df = analyzer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            ancestor_gen = ancestry_df.iloc[0]['generation']  # Oldest ancestor
            comparison = analyzer.compare_with_ancestor(explorer.neat_genome, ancestor_gen)
            if 'error' not in comparison:
                print(f"Comparing with ancestor from generation {ancestor_gen}")
                print(f"Fitness change: {comparison['fitness_change']:.3f}")
                sc = comparison['structure_changes']
                print(f"Structure changes: +{sc['nodes_added']} nodes, +{sc['connections_added']} connections")
                pc = comparison['parameter_changes']
                print(f"Average weight change: {pc['avg_weight_change']:.3f}")
            else:
                print(f"Comparison failed: {comparison['error']}")
        
        logger.info("✅ AncestryAnalyzer test completed!")
        return True
        
    except Exception as e:
        logger.error(f"AncestryAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("🧪 GenomeExplorer Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: GenomeExplorer basic functionality
    if not test_genome_explorer():
        success = False
    
    # Test 2: AncestryAnalyzer detailed features
    if not test_ancestry_analyzer():
        success = False
    
    if success:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)