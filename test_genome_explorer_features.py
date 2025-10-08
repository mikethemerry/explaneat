#!/usr/bin/env python3
"""
Test script to demonstrate the new genome explorer features:
1. Enhanced experiment summary
2. CLI explorer functionality
3. Enhanced population fitness evolution with ancestor tracking
"""

import logging
from explaneat.analysis.genome_explorer import GenomeExplorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_features():
    """Test the enhanced genome explorer features"""
    
    logger.info("🧬 Testing Enhanced Genome Explorer Features")
    logger.info("=" * 60)
    
    try:
        # Load the best genome from the most recent experiment
        explorer = GenomeExplorer.load_best_genome()
        
        logger.info("✅ Successfully loaded genome explorer")
        logger.info(f"   Genome ID: {explorer.genome_info.neat_genome_id}")
        logger.info(f"   Generation: {explorer.genome_info.generation}")
        logger.info(f"   Fitness: {explorer.genome_info.fitness:.3f}")
        
        # Test 1: Enhanced summary
        logger.info("\n📋 Test 1: Enhanced Summary")
        logger.info("   This shows comprehensive genome information")
        explorer.summary()
        
        # Test 2: Enhanced ancestry fitness plotting
        logger.info("\n🌳 Test 2: Enhanced Ancestry Fitness Plotting")
        logger.info("   This shows evolution across generations with better data processing")
        ancestry_df = explorer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            explorer.plot_ancestry_fitness()
        else:
            logger.warning("   Insufficient ancestry data for plotting")
        
        # Test 3: Enhanced evolution progression with ancestor tracking
        logger.info("\n🧬 Test 3: Enhanced Evolution Progression with Ancestor Tracking")
        logger.info("   This shows population evolution with best ancestor fitness overlay")
        explorer.plot_evolution_progression()
        
        # Test 4: Network visualization
        logger.info("\n🕸️  Test 4: Network Structure")
        logger.info("   This shows the neural network structure")
        explorer.show_network(figsize=(12, 8))
        
        # Test 5: Data export
        logger.info("\n💾 Test 5: Data Export")
        logger.info("   This exports all genome data for external analysis")
        export_data = explorer.export_genome_data()
        logger.info(f"   Exported data contains {len(export_data)} sections")
        
        logger.info("\n✅ All enhanced features tested successfully!")
        
        # Show CLI usage
        logger.info("\n🔧 CLI Explorer Usage:")
        logger.info("   # List experiments: python genome_explorer_cli.py --list")
        logger.info("   # Interactive mode: python genome_explorer_cli.py --interactive")
        logger.info("   # Direct analysis: python genome_explorer_cli.py --experiment-id <ID> --summary --network")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_enhanced_features()
