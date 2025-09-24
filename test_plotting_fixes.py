#!/usr/bin/env python3
"""
Test script to demonstrate the fixed plotting functionality.

This script shows the difference between:
1. plot_training_metrics() - Shows epochs within a single genome
2. plot_ancestry_fitness() - Shows evolution across generations for a lineage
3. plot_evolution_progression() - Shows population-level evolution
"""

import logging
from explaneat.analysis.genome_explorer import GenomeExplorer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_plotting_methods():
    """Test the different plotting methods to show the fixes"""

    logger.info("🔍 Testing Genome Explorer Plotting Methods")
    logger.info("=" * 60)

    try:
        # Load the best genome from the most recent experiment
        explorer = GenomeExplorer.load_best_genome()

        logger.info("✅ Successfully loaded genome explorer")
        logger.info(f"   Genome ID: {explorer.genome_info.neat_genome_id}")
        logger.info(f"   Generation: {explorer.genome_info.generation}")
        logger.info(f"   Fitness: {explorer.genome_info.fitness:.3f}")

        # Test 1: Training metrics (epochs within genome)
        logger.info("\n📈 Test 1: Training Metrics (Epochs within genome)")
        logger.info("   This shows how a single genome improved during training")
        explorer.plot_training_metrics()

        # Test 2: Ancestry fitness (generations in lineage)
        logger.info("\n🌳 Test 2: Ancestry Fitness (Evolution across generations)")
        logger.info("   This shows how the lineage evolved across generations")
        ancestry_df = explorer.get_ancestry_tree()
        if len(ancestry_df) > 1:
            explorer.plot_ancestry_fitness()
        else:
            logger.warning("   Insufficient ancestry data for plotting")

        # Test 3: Evolution progression (population-level)
        logger.info("\n🧬 Test 3: Evolution Progression (Population-level evolution)")
        logger.info(
            "   This shows how the entire population evolved across generations"
        )
        explorer.plot_evolution_progression()

        # Test 4: Network visualization
        logger.info("\n🕸️  Test 4: Network Structure")
        logger.info("   This shows the neural network structure")
        explorer.show_network(figsize=(12, 8))

        logger.info("\n✅ All plotting tests completed successfully!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_plotting_methods()
