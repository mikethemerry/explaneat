"""
Tests for direct connections annotation creation and coverage.

Tests that the CLI command correctly creates annotations for direct input-output
connections and that coverage computation works correctly for these annotations.
"""

import pytest
import uuid
from typing import List, Tuple

from explaneat.analysis.annotation_manager import AnnotationManager
from explaneat.analysis.coverage import CoverageComputer
from explaneat.analysis.genome_explorer import GenomeExplorer
from explaneat.db import db, Genome, Experiment, Population
from explaneat.core.explaneat import ExplaNEAT
import neat


class TestDirectConnectionsAnnotation:
    """Test direct connections annotation creation and coverage"""

    @pytest.fixture
    def simple_genome_with_direct_connections(self, tmp_path):
        """Create a simple genome with some direct input-output connections"""
        # This is a placeholder - in real tests, you'd create a genome with direct connections
        # For now, we'll test the logic conceptually
        pass

    def test_create_direct_connections_annotation_basic(self):
        """Test that direct connections annotation includes only inputs with no other connections"""
        # This test would:
        # 1. Create a genome with:
        #    - Input A -> Output 1 (only connection from A)
        #    - Input B -> Output 1 (only connection from B)
        #    - Input C -> Hidden 1 -> Output 1 (C has other connections)
        # 2. Call create_direct_connections_annotation()
        # 3. Verify annotation includes A and B, but not C
        # 4. Verify annotation includes edges (A, Output1) and (B, Output1)
        pass

    def test_direct_connections_annotation_coverage(self):
        """Test that coverage works correctly for direct connections annotation"""
        # This test would:
        # 1. Create a direct connections annotation
        # 2. Compute coverage
        # 3. Verify inputs with only direct connections are covered
        # 4. Verify inputs with other connections are NOT covered
        # 5. Verify output nodes are NOT covered (per paper spec)
        # 6. Verify edges are covered when both endpoints are covered
        pass

    def test_direct_connections_filtering(self):
        """Test that filtering works correctly when direct connections annotation is hidden"""
        # This test would:
        # 1. Create a direct connections annotation
        # 2. Hide the annotation
        # 3. Verify inputs with only direct connections are hidden
        # 4. Verify output nodes remain visible (always visible per paper spec)
        # 5. Verify edges are hidden when annotation is hidden
        pass

    def test_direct_connections_annotation_entry_exit_nodes(self):
        """Test that direct connections annotation has correct entry and exit nodes"""
        # This test would:
        # 1. Create a direct connections annotation
        # 2. Verify entry_nodes contains all qualifying input nodes
        # 3. Verify exit_nodes contains all output nodes receiving direct connections
        pass

    def test_direct_connections_annotation_with_explanation(self):
        """Test that direct connections annotation can be assigned to an explanation"""
        # This test would:
        # 1. Create an explanation
        # 2. Create direct connections annotation with explanation_id
        # 3. Verify annotation belongs to explanation
        pass

    def test_direct_connections_annotation_no_qualifying_inputs(self):
        """Test behavior when no inputs have only direct connections"""
        # This test would:
        # 1. Create a genome where all inputs have other connections
        # 2. Call create_direct_connections_annotation()
        # 3. Verify appropriate message is shown (no annotation created)
        pass

    def test_coverage_algorithm_direct_connections(self):
        """Test that coverage algorithm correctly handles direct connections"""
        # Create a simple test case:
        # Input nodes: [-1, -2]
        # Output nodes: [1]
        # Edges: [(-1, 1), (-2, 1)]
        # Direct connections annotation: entry=[-1, -2], exit=[1], nodes=[-1, -2, 1], edges=[(-1, 1), (-2, 1)]
        
        all_nodes = {-1, -2, 1}
        all_edges = {(-1, 1), (-2, 1)}
        input_nodes = {-1, -2}
        output_nodes = {1}
        
        computer = CoverageComputer(all_nodes, all_edges, input_nodes, output_nodes)
        
        # Create direct connections annotation
        annotation = {
            "id": "direct-ann-1",
            "entry_nodes": [-1, -2],
            "exit_nodes": [1],
            "subgraph_nodes": [-1, -2, 1],
            "subgraph_connections": [(-1, 1), (-2, 1)],
        }
        
        # Compute coverage
        node_coverage, edge_coverage = computer.compute_coverage([annotation])
        
        # Input -1 should be covered (all outgoing edges in annotation)
        assert -1 in node_coverage
        assert "direct-ann-1" in node_coverage[-1]
        
        # Input -2 should be covered (all outgoing edges in annotation)
        assert -2 in node_coverage
        assert "direct-ann-1" in node_coverage[-2]
        
        # Output 1 should NOT be covered (output nodes never covered)
        assert 1 not in node_coverage
        
        # Edges should be covered
        assert (-1, 1) in edge_coverage
        assert "direct-ann-1" in edge_coverage[(-1, 1)]
        assert (-2, 1) in edge_coverage
        assert "direct-ann-1" in edge_coverage[(-2, 1)]

    def test_coverage_algorithm_input_with_other_connections(self):
        """Test that inputs with other connections are NOT covered by direct connections annotation"""
        # Create test case:
        # Input nodes: [-1, -2]
        # Hidden nodes: [5]
        # Output nodes: [1]
        # Edges: [(-1, 1), (-2, 1), (-2, 5)]
        # Direct connections annotation: entry=[-1], exit=[1], nodes=[-1, 1], edges=[(-1, 1)]
        # Note: -2 has other connection to 5, so not included in annotation
        
        all_nodes = {-1, -2, 5, 1}
        all_edges = {(-1, 1), (-2, 1), (-2, 5)}
        input_nodes = {-1, -2}
        output_nodes = {1}
        
        computer = CoverageComputer(all_nodes, all_edges, input_nodes, output_nodes)
        
        # Direct connections annotation only includes -1 (which has only direct connection)
        annotation = {
            "id": "direct-ann-1",
            "entry_nodes": [-1],
            "exit_nodes": [1],
            "subgraph_nodes": [-1, 1],
            "subgraph_connections": [(-1, 1)],
        }
        
        # Compute coverage
        node_coverage, edge_coverage = computer.compute_coverage([annotation])
        
        # Input -1 should be covered (all outgoing edges in annotation)
        assert -1 in node_coverage
        
        # Input -2 should NOT be covered (has other outgoing edge to 5)
        assert -2 not in node_coverage
        
        # Edge (-1, 1) should be covered
        assert (-1, 1) in edge_coverage
        
        # Edge (-2, 1) should NOT be covered (endpoint -2 not covered)
        assert (-2, 1) not in edge_coverage

