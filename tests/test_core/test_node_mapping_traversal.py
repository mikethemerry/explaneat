"""Layer discovery must follow the DAG, without enumerating every input-output path."""
from types import SimpleNamespace

from explaneat.core.neuralneat import NodeMapping


def network(inputs, hidden, edges):
    genome = SimpleNamespace(
        nodes={n: SimpleNamespace(bias=0.0) for n in [*hidden, 0]},
        connections={(a, b): SimpleNamespace(enabled=enabled, weight=0.25)
                     for a, b, enabled in edges},
    )
    config = SimpleNamespace(genome_config=SimpleNamespace(input_keys=inputs, output_keys=[0]))
    return genome, config


def test_longest_path_depths_preserve_skips_and_reachability():
    genome, config = network(
        [-1, -2, -3], [1, 2, 3, 4, 5, 6],
        [(-1, 1, True), (1, 2, True), (2, 3, True), (3, 0, True),
         (-1, 0, True), (-2, 3, True), (-2, 4, True), (5, 0, True), (-1, 6, False)],
    )
    mapping = NodeMapping(genome, config)
    assert {n: info["depth"] for n, info in mapping.node_mapping.items()} == {
        -1: 0, -2: 0, -3: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 0, 6: 0, 0: 4,
    }
    assert set(mapping.unreachable_nodes) == {4, 5, 6}
    assert mapping.node_mapping[0]["skips_in"]
    assert mapping.node_mapping[-1]["skips_out"]
    assert mapping.connection_map[(-1, 0)] == (4, 0, 0)


def test_dense_dag_mapping_work_is_not_proportional_to_number_of_paths():
    class TrackedNode(int):
        lookups = 0

        def __hash__(self):
            type(self).lookups += 1
            return int.__hash__(self)

    inputs = [TrackedNode(-1), TrackedNode(-2)]
    layers = [[TrackedNode(1 + layer * 6 + i) for i in range(6)] for layer in range(4)]
    output = TrackedNode(0)
    edges = [(a, b, True) for before, after in zip([inputs, *layers], [*layers, [output]])
             for a in before for b in after]
    genome, config = network(inputs, [n for layer in layers for n in layer], edges)
    config.genome_config.output_keys = [output]
    TrackedNode.lookups = 0
    mapping = NodeMapping(genome, config)
    work = TrackedNode.lookups
    # A deterministic work bound avoids a machine-dependent wall-clock timeout.
    # This tiny DAG has 2,592 paths but only 27 nodes and 126 edges.
    assert mapping.depth == 5
    assert work < 25_000, f"mapping enumerated paths: {work} node-key lookups"
