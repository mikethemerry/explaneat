"""
Test suite for the annotation system

Tests subgraph validation, evidence schema, annotation manager, and integration.
"""

import logging
import sys
import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene
import numpy as np

from explaneat.db import db, Experiment, Population, Genome, Annotation
from explaneat.db.serialization import serialize_genome, serialize_population_config
from explaneat.analysis.subgraph_validator import SubgraphValidator
from explaneat.analysis.evidence_schema import EvidenceBuilder, create_empty_evidence
from explaneat.analysis.annotation_manager import AnnotationManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_subgraph_validator():
    """Test SubgraphValidator connectivity validation"""
    logger.info("Testing SubgraphValidator...")

    # Test 1: Single node (always connected)
    result = SubgraphValidator.validate_connectivity([1], [])
    assert result["is_connected"], "Single node should be connected"
    assert result["is_valid"], "Single node should be valid"
    logger.info("✓ Single node test passed")

    # Test 2: Two connected nodes
    result = SubgraphValidator.validate_connectivity([1, 2], [(1, 2)])
    assert result["is_connected"], "Two connected nodes should be connected"
    assert result["is_valid"], "Two connected nodes should be valid"
    logger.info("✓ Two connected nodes test passed")

    # Test 3: Disconnected nodes
    result = SubgraphValidator.validate_connectivity([1, 2, 3], [(1, 2)])
    assert not result["is_connected"], "Disconnected nodes should not be connected"
    assert result["is_valid"], "Disconnected nodes should still be valid structure"
    assert len(result["connected_components"]) == 2, "Should have 2 components"
    logger.info("✓ Disconnected nodes test passed")

    # Test 4: Invalid connection (node not in subgraph)
    result = SubgraphValidator.validate_connectivity([1, 2], [(1, 3)])
    assert not result["is_valid"], "Invalid connection should make subgraph invalid"
    logger.info("✓ Invalid connection test passed")

    # Test 5: Complex connected graph
    nodes = [1, 2, 3, 4, 5]
    connections = [(1, 2), (2, 3), (3, 4), (4, 5)]
    result = SubgraphValidator.validate_connectivity(nodes, connections)
    assert result["is_connected"], "Complex graph should be connected"
    logger.info("✓ Complex connected graph test passed")

    # Test 6: Undirected vs directed
    nodes = [1, 2, 3]
    connections = [(1, 2), (3, 2)]  # 1->2<-3, connected in undirected
    result_undirected = SubgraphValidator.validate_connectivity(
        nodes, connections, directed=False
    )
    result_directed = SubgraphValidator.validate_connectivity(
        nodes, connections, directed=True
    )
    assert result_undirected["is_connected"], "Should be connected in undirected graph"
    assert not result_directed[
        "is_connected"
    ], "Should not be connected in directed graph"
    logger.info("✓ Directed vs undirected test passed")

    logger.info("✅ SubgraphValidator tests passed!")
    return True


def test_evidence_schema():
    """Test EvidenceBuilder and evidence schema validation"""
    logger.info("Testing EvidenceBuilder...")

    # Test 1: Create empty evidence
    evidence = create_empty_evidence()
    assert "analytical_methods" in evidence
    assert "visualizations" in evidence
    assert "counterfactuals" in evidence
    assert "other_evidence" in evidence
    logger.info("✓ Empty evidence creation test passed")

    # Test 2: Build evidence with EvidenceBuilder
    builder = EvidenceBuilder()
    builder.add_analytical_method(
        method="closed_form_computation",
        description="Computed closed form",
        result="f(x) = x^2 + 2x + 1",
    )
    builder.add_visualization(
        viz_type="subgraph_visualization",
        description="Visualization of subgraph",
        data={"image_path": "/path/to/image.png"},
    )
    builder.add_counterfactual(
        description="What if weights were doubled?",
        scenario="Double all weights",
        analysis="Performance would increase by 15%",
    )

    evidence = builder.build()
    assert len(evidence["analytical_methods"]) == 1
    assert len(evidence["visualizations"]) == 1
    assert len(evidence["counterfactuals"]) == 1
    logger.info("✓ Evidence building test passed")

    # Test 3: Validate evidence
    is_valid, error = EvidenceBuilder.validate_evidence(evidence)
    assert is_valid, f"Valid evidence should pass validation: {error}"
    logger.info("✓ Evidence validation test passed")

    # Test 4: Invalid evidence (missing required field)
    invalid_evidence = {
        "analytical_methods": [{"method": "test"}],  # Missing required fields
        "visualizations": [],
        "counterfactuals": [],
        "other_evidence": [],
    }
    is_valid, error = EvidenceBuilder.validate_evidence(invalid_evidence)
    assert not is_valid, "Invalid evidence should fail validation"
    logger.info("✓ Invalid evidence detection test passed")

    # Test 5: Add evidence with metadata
    builder = EvidenceBuilder()
    builder.add_visualization(
        viz_type="variation_comparison",
        description="Comparison with earlier generation",
        data={"comparison_data": "..."},
        source_genome_id="test-genome-id",
        generation=5,
        metadata={"author": "test", "method": "manual"},
    )
    evidence = builder.build()
    viz = evidence["visualizations"][0]
    assert viz["source_genome_id"] == "test-genome-id"
    assert viz["generation"] == 5
    assert viz["metadata"]["author"] == "test"
    logger.info("✓ Evidence with metadata test passed")

    logger.info("✅ EvidenceBuilder tests passed!")
    return True


def test_annotation_manager():
    """Test AnnotationManager CRUD operations"""
    logger.info("Testing AnnotationManager...")

    # Initialize database
    db.init_db()

    # Create test experiment and genome
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-file.cfg",
    )

    with db.session_scope() as session:
        experiment = Experiment(
            experiment_sha="test_annotation_123",
            name="Annotation Test",
            description="Testing annotation system",
            dataset_name="Test Dataset",
            config_json=serialize_population_config(config),
            neat_config_text="# Test config",
        )
        session.add(experiment)
        session.flush()
        experiment_id = experiment.id

        population = Population(
            experiment_id=experiment_id,
            generation=0,
            population_size=1,
            num_species=1,
            config_json=serialize_population_config(config),
        )
        session.add(population)
        session.flush()
        population_id = population.id

        # Create a test genome with nodes and connections
        neat_genome = neat.DefaultGenome(1)
        neat_genome.fitness = 1.0

        # Add nodes: -1, -2 (inputs), 0 (output), 1, 2 (hidden)
        for node_id in [-2, -1, 0, 1, 2]:
            neat_genome.nodes[node_id] = DefaultNodeGene(node_id)
            neat_genome.nodes[node_id].bias = 0.5
            neat_genome.nodes[node_id].activation = "relu"

        # Add connections: -2->1, -1->1, 1->2, 2->0
        connections = [(-2, 1), (-1, 1), (1, 2), (2, 0)]
        for from_node, to_node in connections:
            conn_key = (from_node, to_node)
            neat_genome.connections[conn_key] = DefaultConnectionGene(conn_key)
            neat_genome.connections[conn_key].weight = 1.0
            neat_genome.connections[conn_key].enabled = True

        db_genome = Genome.from_neat_genome(neat_genome, population_id)
        session.add(db_genome)
        session.flush()
        genome_id = db_genome.id

    logger.info(f"Created test genome: {genome_id}")

    # Test 1: Create annotation
    nodes = [1, 2, 0]  # Hidden nodes and output
    connections = [(1, 2), (2, 0)]
    entry_nodes = [1]  # Entry node
    exit_nodes = [0]  # Exit node
    hypothesis = "This subgraph performs feature transformation"

    annotation = AnnotationManager.create_annotation(
        genome_id=str(genome_id),
        nodes=nodes,
        connections=connections,
        hypothesis=hypothesis,
        entry_nodes=entry_nodes,
        exit_nodes=exit_nodes,
        name="Test Annotation",
        validate_against_genome=False,  # Skip genome validation for simplicity
    )

    assert annotation is not None
    assert annotation["hypothesis"] == hypothesis
    assert annotation["is_connected"]
    assert len(annotation["subgraph_nodes"]) == 3
    assert annotation["entry_nodes"] == entry_nodes
    assert annotation["exit_nodes"] == exit_nodes
    logger.info("✓ Create annotation test passed")

    # Test 2: Get annotations
    annotations = AnnotationManager.get_annotations(str(genome_id))
    assert len(annotations) == 1
    assert annotations[0]["id"] == annotation["id"]
    logger.info("✓ Get annotations test passed")

    # Test 3: Get specific annotation
    retrieved = AnnotationManager.get_annotation(str(annotation["id"]))
    assert retrieved is not None
    assert retrieved.hypothesis == hypothesis
    logger.info("✓ Get specific annotation test passed")

    # Test 4: Update annotation
    updated = AnnotationManager.update_annotation(
        str(annotation["id"]), hypothesis="Updated hypothesis", name="Updated Name"
    )
    assert updated.hypothesis == "Updated hypothesis"
    assert updated.name == "Updated Name"
    logger.info("✓ Update annotation test passed")

    # Test 5: Add evidence
    evidence_data = {
        "method": "closed_form_computation",
        "description": "Computed closed form",
        "result": "f(x) = x^2",
        "metadata": {},
    }
    updated = AnnotationManager.add_evidence(
        str(annotation["id"]), "analytical_method", evidence_data
    )
    assert updated.evidence is not None
    assert len(updated.evidence["analytical_methods"]) == 1
    logger.info("✓ Add evidence test passed")

    # Test 6: Delete annotation
    deleted = AnnotationManager.delete_annotation(str(annotation["id"]))
    assert deleted
    retrieved = AnnotationManager.get_annotation(str(annotation["id"]))
    assert retrieved is None
    logger.info("✓ Delete annotation test passed")

    # Test 7: Invalid subgraph (disconnected)
    try:
        AnnotationManager.create_annotation(
            genome_id=str(genome_id),
            nodes=[1, 2, 3],
            connections=[(1, 2)],  # 3 is disconnected
            hypothesis="Should fail",
            validate_against_genome=False,
        )
        assert False, "Should have raised ValueError for disconnected subgraph"
    except ValueError as e:
        assert "not connected" in str(e).lower()
        logger.info("✓ Invalid subgraph rejection test passed")

    logger.info("✅ AnnotationManager tests passed!")
    return True


def test_integration():
    """Test integration with GenomeExplorer"""
    logger.info("Testing GenomeExplorer integration...")

    # This would require a full GenomeExplorer setup, which is complex
    # For now, we'll just verify the imports work
    from explaneat.analysis.genome_explorer import GenomeExplorer

    # Check that annotation methods exist
    assert hasattr(GenomeExplorer, "get_annotations")
    assert hasattr(GenomeExplorer, "add_annotation")
    assert hasattr(GenomeExplorer, "get_annotation")
    assert hasattr(GenomeExplorer, "update_annotation")
    assert hasattr(GenomeExplorer, "add_evidence_to_annotation")
    assert hasattr(GenomeExplorer, "delete_annotation")
    logger.info("✓ GenomeExplorer integration methods exist")

    logger.info("✅ Integration tests passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 ANNOTATION SYSTEM TEST SUITE")
    print("=" * 60)

    success = True

    try:
        # Test 1: SubgraphValidator
        if not test_subgraph_validator():
            success = False
        print()

        # Test 2: EvidenceBuilder
        if not test_evidence_schema():
            success = False
        print()

        # Test 3: AnnotationManager
        if not test_annotation_manager():
            success = False
        print()

        # Test 4: Integration
        if not test_integration():
            success = False
        print()

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        success = False

    if success:
        print("=" * 60)
        print("✅ ALL ANNOTATION SYSTEM TESTS PASSED!")
        print("=" * 60)
        print("\nThe annotation system can successfully:")
        print("- Validate subgraph connectivity")
        print("- Build and validate evidence structures")
        print("- Create, read, update, and delete annotations")
        print("- Integrate with GenomeExplorer")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

    return success


if __name__ == "__main__":
    main()



