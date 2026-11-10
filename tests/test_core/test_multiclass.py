"""Multi-output (softmax) PropNEAT: NeuralNeat layering, StructureNetwork parity,
criterion / fitness, and that single-output behaviour is unchanged.

See ``explaneat/core/multiclass.py``.  With several outputs NeuralNeat puts
every output in one final layer (``NodeMapping._align_outputs``) in
``output_keys`` order; StructureNetwork returns ``[N, k]`` in
``output_node_ids`` order.  Single-output genomes take the original code path.
"""

import copy
import os
import random
import tempfile

import neat
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.errors import GenomeNotValidError
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.multiclass import (
    OutputSinkGenome,
    SoftmaxCrossEntropyCriterion,
    make_multiclass_fitness,
    use_output_sink_genome,
)
from explaneat.core.neuralneat import NeuralNeat, set_output_activation
from explaneat.core.structure_network import StructureNetwork

CFG = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 999.0
pop_size              = 12
reset_on_extinction   = False

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2

[DefaultGenome]
activation_default      = relu
activation_mutate_rate  = 0.0
activation_options      = relu
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.0
bias_mutate_rate        = 0.0
bias_replace_rate       = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.3
enabled_default         = True
enabled_mutate_rate     = 0.1
feed_forward            = True
initial_connection      = {init}
node_add_prob           = 0.4
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = {n_in}
num_outputs             = {n_out}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.0
weight_mutate_rate      = 0.0
weight_replace_rate     = 0.0
"""


def make_config(n_in=8, n_out=4, init="partial_direct 0.5", genome_type=OutputSinkGenome,
                identity=True):
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        f.write(CFG.format(n_in=n_in, n_out=n_out, init=init))
        path = f.name
    try:
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                          neat.DefaultStagnation, path)
    finally:
        os.unlink(path)
    if genome_type is OutputSinkGenome:
        use_output_sink_genome(cfg)
    if identity:
        set_output_activation(cfg, "identity")
    return cfg


def evolved_genome(cfg, seed, n_node=6, n_conn=25):
    random.seed(seed)
    g = cfg.genome_type(1)
    g.configure_new(cfg.genome_config)
    for _ in range(n_node):
        g.mutate_add_node(cfg.genome_config)
    for _ in range(n_conn):
        g.mutate_add_connection(cfg.genome_config)
    rng = np.random.default_rng(seed)
    for c in g.connections.values():
        c.weight = float(rng.normal(0, 1.0))
    for n in g.nodes.values():
        n.bias = float(rng.normal(0, 0.5))
    return g


def test_sink_genome_never_adds_connection_from_output():
    cfg = make_config()
    for seed in range(5):
        g = evolved_genome(cfg, seed, n_node=10, n_conn=200)
        assert not any(i in cfg.genome_config.output_keys for i, _ in g.connections)


@pytest.mark.parametrize("seed", range(6))
def test_multi_output_columns_and_structure_parity(seed):
    cfg = make_config()
    g = evolved_genome(cfg, seed)
    net = NeuralNeat(g, cfg)
    outs = cfg.genome_config.output_keys
    out_layer = net.layers[max(net.layers)]
    assert list(out_layer["nodes"]) == list(outs)  # exactly the outputs, in order
    X = np.random.default_rng(seed).normal(size=(200, 8))
    nn_out = net.forward(torch.tensor(X)).detach().numpy()
    assert nn_out.shape == (200, 4)

    pheno = ExplaNEAT(g, cfg).get_phenotype_network()
    assert pheno.metadata.get("output_activation") == "identity"
    sn = StructureNetwork(pheno)
    sn_out = sn.forward(torch.tensor(X)).detach().numpy()
    np.testing.assert_allclose(sn_out, nn_out, rtol=1e-12, atol=1e-12)


def test_output_columns_follow_output_keys_not_dict_order():
    """Each column k equals a hand computation of output k (no column mix-up)."""
    cfg = make_config(n_in=3, n_out=3, init="full_direct")
    random.seed(0)
    g = cfg.genome_type(1)
    g.configure_new(cfg.genome_config)
    # reorder the node dict so outputs are not first / in order
    nodes = dict(reversed(list(g.nodes.items())))
    g.nodes = nodes
    W = np.arange(9, dtype=float).reshape(3, 3) / 10.0  # W[in, out]
    for (i, o), c in g.connections.items():
        c.weight = W[-i - 1, o]
    for k, n in g.nodes.items():
        n.bias = float(k)
    X = np.random.default_rng(1).normal(size=(5, 3))
    out = NeuralNeat(g, cfg).forward(torch.tensor(X)).detach().numpy()
    np.testing.assert_allclose(out, X @ W + np.array([0.0, 1.0, 2.0]), atol=1e-12)


def test_unconnected_output_is_constant_in_both_engines():
    cfg = make_config(n_in=4, n_out=3, init="full_direct")
    g = evolved_genome(cfg, 3, n_node=2, n_conn=5)
    for key in [k for k in g.connections if k[1] == 2]:
        del g.connections[key]
    g.nodes[2].bias = 0.7
    X = np.random.default_rng(0).normal(size=(10, 4))
    nn_out = NeuralNeat(g, cfg).forward(torch.tensor(X)).detach().numpy()
    np.testing.assert_allclose(nn_out[:, 2], 0.7)
    sn_out = StructureNetwork(ExplaNEAT(g, cfg).get_phenotype_network()).forward(
        torch.tensor(X)).detach().numpy()
    np.testing.assert_allclose(sn_out, nn_out, atol=1e-12)


def test_output_feeding_a_node_is_rejected_for_multi_output():
    cfg = make_config(n_in=3, n_out=2, init="full_direct", genome_type=neat.DefaultGenome)
    random.seed(0)
    g = cfg.genome_type(1)
    g.configure_new(cfg.genome_config)
    g.mutate_add_node(cfg.genome_config)
    hidden = max(g.nodes)
    # output 0 -> hidden -> (already) output: makes output 0 a hidden unit too
    cg = g.create_connection(cfg.genome_config, 0, hidden)
    g.connections[cg.key] = cg
    with pytest.raises(GenomeNotValidError):
        NeuralNeat(g, cfg)


def test_criterion_and_fitness():
    cfg = make_config()
    g = evolved_genome(cfg, 1)
    X = torch.tensor(np.random.default_rng(0).normal(size=(50, 8)))
    y = torch.tensor(np.random.default_rng(1).integers(0, 4, 50), dtype=torch.float64)
    logits = NeuralNeat(g, cfg).forward(X)
    ce = F.cross_entropy(logits, y.long())
    crit = SoftmaxCrossEntropyCriterion()
    assert float(crit(logits, y.unsqueeze(1))) == pytest.approx(float(ce), abs=1e-15)
    assert SoftmaxCrossEntropyCriterion.describe() == {"name": "softmax_cross_entropy"}
    genomes = {1: g}
    make_multiclass_fitness("neg_ce")(genomes, cfg, X, y.unsqueeze(1), "cpu")
    assert g.fitness == pytest.approx(-float(ce), abs=1e-12)
    make_multiclass_fitness("accuracy")(genomes, cfg, X, y.unsqueeze(1), "cpu")
    assert g.fitness == pytest.approx(float((logits.argmax(1) == y.long()).double().mean()))


def test_population_trains_with_softmax_ce():
    cfg = make_config()
    random.seed(4)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 8))
    y = (X[:, :4].argmax(axis=1)).astype(float)
    pop = BackpropPopulation(cfg, X, y, criterion=SoftmaxCrossEntropyCriterion())
    assert pop.ys.shape == (120, 1)
    before = copy.deepcopy(pop.population)
    pop.backpropagate(pop.xs, pop.ys, nEpochs=10)
    crit = SoftmaxCrossEntropyCriterion()
    improved = 0
    for k, g in before.items():
        with torch.no_grad():
            l0 = float(crit(NeuralNeat(g, cfg).forward(pop.xs), pop.ys))
            l1 = float(crit(NeuralNeat(pop.population[k], cfg).forward(pop.xs), pop.ys))
        improved += l1 < l0
    assert improved >= len(before) - 1
    best = pop.run(make_multiclass_fitness(), n=2, nEpochs=3)
    assert best.fitness < 0


def test_single_output_mapping_unchanged():
    """num_outputs = 1 never enters the multi-output code (layering identical)."""
    from explaneat.core import neuralneat as nnmod

    cfg = make_config(n_in=6, n_out=1, init="full_direct", genome_type=neat.DefaultGenome,
                      identity=False)
    called = []
    orig = nnmod.NodeMapping._align_outputs
    nnmod.NodeMapping._align_outputs = staticmethod(lambda *a: called.append(1) or orig(*a))
    try:
        for seed in range(4):
            g = evolved_genome(cfg, seed, n_node=4, n_conn=10)
            net = NeuralNeat(g, cfg)
            # original ordering: output layer in genome.nodes order, node 0 present
            assert 0 in net.layers[max(net.layers)]["nodes"]
            assert net.forward(torch.zeros(3, 6, dtype=torch.float64)).shape == (3, 1)
    finally:
        nnmod.NodeMapping._align_outputs = orig
    assert called == []
