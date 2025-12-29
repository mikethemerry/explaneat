from types import SimpleNamespace

import pytest

from explaneat.analysis import visualization
from explaneat.analysis.visualization import InteractiveNetworkViewer
from explaneat.core.genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)


@pytest.fixture(autouse=True)
def enable_pyvis(monkeypatch):
    monkeypatch.setattr(visualization, "PYVIS_AVAILABLE", True)


class DummyConnection(SimpleNamespace):
    def __init__(self, weight=1.0, enabled=True):
        super().__init__(weight=weight, enabled=enabled)


class DummyGenome:
    def __init__(self, nodes, connections):
        self.nodes = {node_id: object() for node_id in nodes}
        self.connections = connections


class DummyConfig:
    def __init__(self, input_keys, output_keys):
        self.genome_config = SimpleNamespace(
            input_keys=input_keys,
            output_keys=output_keys,
        )


def _build_viewer(nodes, connections, annotations=None):
    # Convert test data to NetworkStructure
    network_nodes = []
    for node_id in nodes:
        if node_id < 0:
            node_type = NodeType.INPUT
        elif node_id == 0:
            node_type = NodeType.OUTPUT
        else:
            node_type = NodeType.HIDDEN
        network_nodes.append(NetworkNode(id=node_id, type=node_type))
    
    network_connections = []
    for (from_node, to_node), conn in connections.items():
        network_connections.append(
            NetworkConnection(
                from_node=from_node,
                to_node=to_node,
                weight=conn.weight,
                enabled=conn.enabled,
            )
        )
    
    input_keys = [node for node in nodes if node < 0]
    output_keys = [node for node in nodes if node == 0]
    
    network = NetworkStructure(
        nodes=network_nodes,
        connections=network_connections,
        input_node_ids=input_keys,
        output_node_ids=output_keys,
    )
    
    config = DummyConfig(input_keys, output_keys)
    return InteractiveNetworkViewer(network, config, annotations=annotations or [])


def test_annotation_nodes_stay_contiguous():
    nodes = [-1, 0, 1, 2, 3]
    connections = {
        (-1, 1): DummyConnection(),
        (-1, 2): DummyConnection(),
        (-1, 3): DummyConnection(),
        (1, 0): DummyConnection(),
        (2, 0): DummyConnection(),
        (3, 0): DummyConnection(),
    }
    annotations = [
        {
            "id": "A",
            "name": "annotated hidden nodes",
            "subgraph_nodes": [1, 2],
            "subgraph_connections": [(1, 0), (2, 0)],
        }
    ]
    viewer = _build_viewer(nodes, connections, annotations)

    positions = viewer._calculate_layered_positions()
    hidden_depth = viewer.node_depths[1]

    layer_nodes = sorted(
        [
            (node_id, positions[node_id][1])
            for node_id, depth in viewer.node_depths.items()
            if depth == hidden_depth
        ],
        key=lambda item: item[1],
    )
    annotated_nodes = {1, 2}
    annotated_indices = [
        idx
        for idx, (node_id, _) in enumerate(layer_nodes)
        if node_id in annotated_nodes
    ]

    assert annotated_indices, "Annotated nodes missing from expected layer ordering"
    assert max(annotated_indices) - min(annotated_indices) + 1 == len(
        annotated_indices
    ), "Annotated nodes should occupy contiguous slots within their layer"


def test_direct_connections_use_bottom_lane():
    nodes = [-2, -1, 0, 1]
    connections = {
        (-1, 0): DummyConnection(),  # Direct input -> output path
        (-2, 1): DummyConnection(),
        (1, 0): DummyConnection(),
    }
    viewer = _build_viewer(nodes, connections)

    positions = viewer._calculate_layered_positions()
    direct_nodes = viewer.direct_connection_nodes
    assert direct_nodes, "Expected direct connection nodes to be detected"

    lane_y_values = {node_id: positions[node_id][1] for node_id in direct_nodes}
    non_lane_ys = [
        positions[node_id][1] for node_id in positions if node_id not in direct_nodes
    ]

    assert len(set(lane_y_values.values())) == 1, "Lane nodes should share the same y"
    assert all(
        lane_y_values[next(iter(lane_y_values))] < y for y in non_lane_ys
    ), "Lane nodes should appear below all other nodes"
