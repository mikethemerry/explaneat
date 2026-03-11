"""Tests for composed annotation entry/exit auto-derivation."""

import numpy as np
import pytest
import torch
from explaneat.core.collapse_transform import (
    compute_effective_entries_exits,
    _compute_effective_subgraph_nodes,
    collapse_structure,
)
from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import AnnotationData
from explaneat.core.structure_network import StructureNetwork


def _make_node(id, type=NodeType.HIDDEN, **kwargs):
    return NetworkNode(id=id, type=type, **kwargs)


def _make_conn(from_node, to_node, weight=1.0, enabled=True, **kwargs):
    return NetworkConnection(
        from_node=from_node, to_node=to_node, weight=weight, enabled=enabled, **kwargs
    )


def _make_annotation(name, entry_nodes, exit_nodes, subgraph_nodes,
                     subgraph_connections=None, parent_annotation_id=None):
    return AnnotationData(
        name=name,
        hypothesis="test",
        entry_nodes=entry_nodes,
        exit_nodes=exit_nodes,
        subgraph_nodes=subgraph_nodes,
        subgraph_connections=subgraph_connections or [],
        parent_annotation_id=parent_annotation_id,
    )


def _megaann_structure():
    """
    Models the real MegaAnn1 topology (simplified):

        -24 --> A1678 --> 1676 ---> identity_7 --> A20608(ann_26) --> 608_a --> out
                                |-> identity_6 --> A20608(ann_24) --> 608_b --> out
        -20_a ----------------------------------------^(ann_26)
        -20_b ----------------------------------------^(ann_24)

    Child annotations:
      A1678: entries=[-24, -4_c], exits=[1676], subgraph=[-24, 1676, 1559_b, -4_c]
      A20608_b: entries=[-20_b, identity_6], exits=[608_b], subgraph=[608_b, -20_b, identity_6]
      A20608_a: entries=[-20_a, identity_7], exits=[608_a], subgraph=[608_a, -20_a, identity_7]

    Parent MegaAnn1 (compositional): children=[A1678, A20608_b, A20608_a]
    """
    nodes = [
        _make_node("-24", NodeType.INPUT),
        _make_node("-20_a", NodeType.INPUT),
        _make_node("-20_b", NodeType.INPUT),
        _make_node("1559_b", bias=0.5),
        _make_node("-4_c", NodeType.INPUT),
        _make_node("1676", bias=1.2),
        _make_node("identity_6", bias=0.0, activation="identity"),
        _make_node("identity_7", bias=0.0, activation="identity"),
        _make_node("608_a", bias=-1.0),
        _make_node("608_b", bias=-1.0),
        _make_node("0", NodeType.OUTPUT, bias=-0.5),
    ]
    connections = [
        # A1678 internals
        _make_conn("-24", "1676", 1.0),
        _make_conn("-4_c", "1559_b", 1.0),
        _make_conn("1559_b", "1676", -1.7),
        # 1676 -> identity nodes (internal wiring between children)
        _make_conn("1676", "identity_6", 0.35),
        _make_conn("1676", "identity_7", 0.35),
        # A20608_a internals
        _make_conn("-20_a", "608_a", -0.58),
        _make_conn("identity_7", "608_a", 1.0),
        # A20608_b internals
        _make_conn("-20_b", "608_b", -0.58),
        _make_conn("identity_6", "608_b", 1.0),
        # Outputs from composed region to rest of network
        _make_conn("608_a", "0", 0.63),
        _make_conn("608_b", "0", 2.19),
    ]
    structure = NetworkStructure(
        nodes=nodes,
        connections=connections,
        input_node_ids=["-24", "-20_a", "-20_b", "-4_c"],
        output_node_ids=["0"],
    )

    child1 = _make_annotation(
        "A1678",
        entry_nodes=["-24", "-4_c"],
        exit_nodes=["1676"],
        subgraph_nodes=["-24", "1676", "1559_b", "-4_c"],
        parent_annotation_id="MegaAnn1",
    )
    child2 = _make_annotation(
        "A20608_b",
        entry_nodes=["-20_b", "identity_6"],
        exit_nodes=["608_b"],
        subgraph_nodes=["608_b", "-20_b", "identity_6"],
        parent_annotation_id="MegaAnn1",
    )
    child3 = _make_annotation(
        "A20608_a",
        entry_nodes=["-20_a", "identity_7"],
        exit_nodes=["608_a"],
        subgraph_nodes=["608_a", "-20_a", "identity_7"],
        parent_annotation_id="MegaAnn1",
    )
    parent = _make_annotation(
        "MegaAnn1",
        entry_nodes=["-24", "-20_a", "-20_b", "identity_6"],
        exit_nodes=["-20_a", "identity_7"],
        subgraph_nodes=[],
    )

    all_annotations = [child1, child2, child3, parent]
    return structure, all_annotations


class TestComputeEffectiveEntriesExits:
    """Test auto-derivation of entry/exit nodes for compositional annotations.

    NOTE: These tests call compute_effective_entries_exits on the ORIGINAL
    (uncollapsed) structure to validate the boundary detection algorithm.
    In collapse_structure(), it is called on the CURRENT structure (after
    children are collapsed), so the exits will be fn_ nodes instead.
    """

    def test_megaann_effective_entries(self):
        structure, annotations = _megaann_structure()
        ann_by_name = {a.name: a for a in annotations}
        children_map = {}
        for a in annotations:
            if a.parent_annotation_id and a.parent_annotation_id in ann_by_name:
                children_map.setdefault(a.parent_annotation_id, []).append(a)

        effective_subgraph = _compute_effective_subgraph_nodes(
            ann_by_name["MegaAnn1"], ann_by_name, children_map
        )
        entries, exits = compute_effective_entries_exits(
            effective_subgraph, structure
        )
        assert "-24" in entries
        assert "-20_a" in entries
        assert "-20_b" in entries
        assert "-4_c" in entries
        assert "identity_6" not in entries
        assert "identity_7" not in entries

    def test_megaann_effective_exits(self):
        structure, annotations = _megaann_structure()
        ann_by_name = {a.name: a for a in annotations}
        children_map = {}
        for a in annotations:
            if a.parent_annotation_id and a.parent_annotation_id in ann_by_name:
                children_map.setdefault(a.parent_annotation_id, []).append(a)

        effective_subgraph = _compute_effective_subgraph_nodes(
            ann_by_name["MegaAnn1"], ann_by_name, children_map
        )
        entries, exits = compute_effective_entries_exits(
            effective_subgraph, structure
        )
        assert "608_a" in exits
        assert "608_b" in exits
        assert "-20_a" not in exits
        assert "identity_7" not in exits


class TestComposedAnnotationCollapseConnectivity:
    """Test that collapsing composed annotations produces connected function nodes."""

    def test_fn_megaann_has_outgoing_connections(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_conns_out = [
            c for c in collapsed.connections if c.from_node == "fn_MegaAnn1"
        ]
        assert len(fn_conns_out) > 0, "fn_MegaAnn1 must have outgoing connections"

    def test_fn_megaann_has_incoming_connections(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_conns_in = [
            c for c in collapsed.connections if c.to_node == "fn_MegaAnn1"
        ]
        assert len(fn_conns_in) > 0, "fn_MegaAnn1 must have incoming connections"

    def test_fn_megaann_connects_to_output(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_to_out = [
            c for c in collapsed.connections
            if c.from_node == "fn_MegaAnn1" and c.to_node == "0"
        ]
        assert len(fn_to_out) == 2
        indices = {c.output_index for c in fn_to_out}
        assert indices == {0, 1}

    def test_fn_megaann_metadata_has_correct_entries_exits(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        fn_node = next(n for n in collapsed.nodes if n.id == "fn_MegaAnn1")
        meta = fn_node.function_metadata
        assert set(meta.input_names) == {"-24", "-20_a", "-20_b", "-4_c"}
        assert set(meta.output_names) == {"fn_A20608_a", "fn_A20608_b"}

    def test_collapsed_structure_validates(self):
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        result = collapsed.validate()
        assert result["is_valid"], f"Validation errors: {result['errors']}"

    def test_no_cycle_after_composed_collapse(self):
        """Verify cycle freedom (using DFS)."""
        structure, annotations = _megaann_structure()
        collapsed = collapse_structure(
            structure, annotations, {"A1678", "A20608_b", "A20608_a", "MegaAnn1"}
        )
        adj = {}
        for c in collapsed.connections:
            adj.setdefault(c.from_node, []).append(c.to_node)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n.id: WHITE for n in collapsed.nodes}
        def dfs(u):
            color[u] = GRAY
            for v in adj.get(u, []):
                if color.get(v) == GRAY:
                    return True
                if color.get(v) == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False
        has_cycle = any(dfs(n.id) for n in collapsed.nodes if color[n.id] == WHITE)
        assert not has_cycle


class TestComposedCollapseForwardPass:
    """Verify that collapsing a composed annotation preserves forward-pass output."""

    def test_megaann_forward_pass_equivalence(self):
        structure, annotations = _megaann_structure()

        expanded_net = StructureNetwork(structure)
        x = torch.tensor([[0.5, 0.3, -0.2, 0.8]], dtype=torch.float64)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(
            structure, annotations,
            {"A1678", "A20608_b", "A20608_a", "MegaAnn1"},
        )
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)

    def test_megaann_batch_forward_pass(self):
        structure, annotations = _megaann_structure()

        x = torch.tensor([
            [0.5, 0.3, -0.2, 0.8],
            [1.0, -1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float64)

        expanded_net = StructureNetwork(structure)
        expanded_out = expanded_net.forward(x).detach().numpy()

        collapsed = collapse_structure(
            structure, annotations,
            {"A1678", "A20608_b", "A20608_a", "MegaAnn1"},
        )
        collapsed_net = StructureNetwork(collapsed)
        collapsed_out = collapsed_net.forward(x).detach().numpy()

        np.testing.assert_allclose(expanded_out, collapsed_out, atol=1e-10)
