"""Serial unit leaves can share upstream computations through private identity entries."""

import numpy as np
import torch

from explaneat.analysis.coverage_report import coverage_report
from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType
from explaneat.core.model_state import ModelStateEngine
from explaneat.core.structure_network import StructureNetwork


def test_serial_unit_leaves_need_no_duplicated_upstream_hidden_units():
    # Unequal signed weights and biases exercise weighted identity bridges,
    # rather than relying on identical units or cancellation.
    nodes = [NetworkNode(id=i, type=NodeType.INPUT) for i in ['-1', '-2']]
    nodes += [NetworkNode(id=str(i), type=NodeType.HIDDEN, bias=b, activation='relu',
                          response=1.0, aggregation='sum')
              for i, b in enumerate([0.2, -0.4, 0.3, -0.1], start=1)]
    nodes.append(NetworkNode(id='0', type=NodeType.OUTPUT, bias=0.1,
                             activation='sigmoid', response=1.0, aggregation='sum'))
    pairs = [('-1', '1', 0.3), ('-2', '1', -0.7), ('-1', '2', -1.2), ('-2', '2', 0.4),
             ('1', '3', 0.9), ('2', '3', -0.6), ('1', '4', -0.5), ('2', '4', 1.1),
             ('3', '0', 0.8), ('4', '0', -1.3)]
    original = NetworkStructure(nodes=nodes,
                                connections=[NetworkConnection(a, b, w, True) for a, b, w in pairs],
                                input_node_ids=['-1', '-2'], output_node_ids=['0'])
    engine = ModelStateEngine(original)
    for v in ['3', '4']:
        for u in ['1', '2']:
            engine.add_operation('add_identity_node', dict(
                target_node=v, connections=[[u, v]], new_node_id=f'identity_{u}_{v}'))
    for x in ['-1', '-2']:
        engine.add_operation('split_node', {'node_id': x})

    def annotate(name, subgraph, entries, exits, children=None):
        internal = set(subgraph)
        conns = [[c.from_node, c.to_node] for c in engine.current_state.connections
                 if c.enabled and (children or (c.from_node in internal and c.to_node in internal))]
        params = dict(name=name, hypothesis='structural regression', subgraph_nodes=subgraph,
                      entry_nodes=entries, exit_nodes=exits, subgraph_connections=conns)
        if children:
            params['child_annotation_ids'] = children
        engine.add_operation('annotate', params)

    for u, suffix in [('1', 'a'), ('2', 'b')]:
        entries = [f'-1_{suffix}', f'-2_{suffix}']
        annotate(f'G_{u}', entries + [u], entries, [u])
    for v in ['3', '4']:
        entries = [f'identity_{u}_{v}' for u in ['1', '2']]
        annotate(f'H_{v}', entries + [v], entries, [v])
    annotate('root', ['0'], ['-1_a', '-1_b', '-2_a', '-2_b'], ['0'],
             children=['G_1', 'G_2', 'H_3', 'H_4'])

    x = torch.as_tensor(np.random.default_rng(7).normal(size=(100, 2)))
    before = StructureNetwork(original).forward(x).detach().numpy()
    after = StructureNetwork(engine.current_state).forward(x).detach().numpy()
    np.testing.assert_allclose(after, before, atol=1e-12, rtol=0)
    assert {n.id for n in engine.current_state.nodes if n.type == NodeType.HIDDEN
            and not n.id.startswith('identity_')} == {'1', '2', '3', '4'}
    coverage = coverage_report(engine.current_state, engine.annotations)
    assert coverage['structural_coverage'] == coverage['compositional_coverage'] == 1.0
    assert coverage['hierarchy']['complete']
    assert coverage['decomposition']['granularity'] == 1.0
    assert coverage['decomposition']['units_only_in_root'] == 0
