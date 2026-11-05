"""Opt-in connectivity mask for PropNEAT backprop (``mask_phantom``).

NeuralNeat builds a dense weight matrix per layer. Entries without an enabled
connection gene ("phantom" weights) start at 0 but are ordinary parameters, so
backprop trains them, and ``update_genome_weights`` then drops them: the stored
genome is not the trained network. ``mask_phantom=True`` multiplies each layer
matrix by a constant 0/1 mask in ``forward`` so phantom entries stay exactly 0
with zero gradient. ``mask_phantom=False`` (the default) must be bit-identical
to the original code.
"""

import copy
import os
import tempfile

import neat
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from neat.genes import DefaultConnectionGene, DefaultNodeGene

import explaneat.core.backproppop as backproppop_mod
from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.neuralneat import (
    LAYER_TYPE_CONNECTED,
    LAYER_TYPE_INPUT,
    LAYER_TYPE_OUTPUT,
    NeuralNeat,
)

N_INPUTS = 4

CONFIG_TEXT = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 999.0
pop_size              = 8
reset_on_extinction   = True

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 1

[DefaultReproduction]
elitism            = 1
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
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = partial_direct 0.5
node_add_prob           = 0.3
node_delete_prob        = 0.05
num_hidden              = 2
num_inputs              = {N_INPUTS}
num_outputs             = 1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
"""


@pytest.fixture(scope="module")
def config():
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
        f.write(CONFIG_TEXT)
        path = f.name
    try:
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
    finally:
        os.unlink(path)


def _genome():
    """Inputs -1..-4, hidden 1 (depth 1), 2 (depth 2), output 0 (depth 3).

    Includes skip connections (-1->0, 1->0) and a *disabled* gene (-2->2) so
    the dense matrices have phantom entries, including one that sits where a
    disabled gene would go.
    """
    nodes = {0: 0.1, 1: -0.2, 2: 0.3}
    conns = {(-1, 1): 0.7, (-2, 1): -0.4, (1, 2): 0.9, (-3, 2): 0.5,
             (2, 0): 1.1, (-1, 0): -0.6, (1, 0): 0.8, (-4, 1): 0.2}
    g = neat.DefaultGenome(1)
    for nid, bias in nodes.items():
        n = DefaultNodeGene(nid)
        n.bias, n.response, n.activation, n.aggregation = bias, 1.0, "relu", "sum"
        g.nodes[nid] = n
    for k, w in conns.items():
        c = DefaultConnectionGene(k)
        c.weight, c.enabled = w, True
        g.connections[k] = c
    c = DefaultConnectionGene((-2, 2))
    c.weight, c.enabled = 0.33, False
    g.connections[(-2, 2)] = c
    return g


def _data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_INPUTS))
    y = (X[:, 0] - 0.5 * X[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(np.float64)
    return (torch.tensor(X, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64).unsqueeze(1))


# --- frozen copy of the original (pre-mask) computation ----------------------

def _reference_forward(net, x):
    """Verbatim arithmetic of the original NeuralNeat.forward (no mask)."""
    outputs = {}
    for layer_id in range(net.n_layers):
        layer_type = net.layer_types[layer_id]
        if layer_type == LAYER_TYPE_INPUT:
            outputs[layer_id] = x
            continue
        layer_input = torch.cat([outputs[ii] for ii in net.layer_inputs[layer_id]], dim=1)
        z = torch.matmul(layer_input, net.weights[layer_id]) + net.biases[layer_id]
        if layer_type == LAYER_TYPE_CONNECTED:
            outputs[layer_id] = torch.relu(z)
        else:
            outputs[layer_id] = torch.sigmoid(z)
        if layer_type == LAYER_TYPE_OUTPUT:
            return outputs[layer_id]


def _reference_backprop_one(genome, config, xs, ys, n_epochs):
    """Original BackpropPopulation.backpropagate body for one genome, using
    the reference forward. Returns (trained genome, post-backprop loss)."""
    net = NeuralNeat(genome, config, criterion=nn.BCELoss())
    optimizer = optim.Adadelta(net.parameters(), lr=1.5)
    optimizer.zero_grad()
    bce = nn.BCELoss()
    bce(_reference_forward(net, xs), ys)
    for _ in range(n_epochs):
        loss = bce(_reference_forward(net, xs), ys)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    post = bce(_reference_forward(net, xs), ys)
    net.update_genome_weights()
    return net.genome, post.item()


def _genome_params(g):
    return ({k: c.weight for k, c in g.connections.items()},
            {k: n.bias for k, n in g.nodes.items()})


def _phantom_entries(net):
    """(layer_id, row, col) of every non-input dense entry without an enabled gene."""
    out = []
    for L, layer in net.layers.items():
        if layer["layer_type"] == LAYER_TYPE_INPUT:
            continue
        real = set(layer["input_map"].values())
        rows, cols = net.weights[L].shape
        out += [(L, r, c) for r in range(rows) for c in range(cols) if (r, c) not in real]
    return out


def _train(net, xs, ys, n_epochs):
    optimizer = optim.Adadelta(net.parameters(), lr=1.5)
    optimizer.zero_grad()
    bce = nn.BCELoss()
    for _ in range(n_epochs):
        loss = bce(net.forward(xs), ys)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        return bce(net.forward(xs), ys).item()


# --- (a) default is bit-identical ---------------------------------------------

class TestDefaultUnchanged:
    def test_default_registers_no_mask_and_same_params(self, config):
        net = NeuralNeat(_genome(), config)
        assert net.mask_phantom is False
        assert list(net.buffers()) == []
        names = [n for n, _ in net.named_parameters()]
        assert names == ([f"weight_{L}" for L in net.weights]
                         + [f"bias_{L}" for L in net.biases])

    def test_forward_bit_identical(self, config):
        xs, _ = _data()
        net = NeuralNeat(_genome(), config)
        assert torch.equal(net.forward(xs), _reference_forward(net, xs))
        net2 = NeuralNeat(_genome(), config, mask_phantom=False)
        assert torch.equal(net2.forward(xs), _reference_forward(net, xs))

    def test_population_backprop_bit_identical(self, config):
        torch.manual_seed(0)
        xs, ys = _data()
        import random
        random.seed(3)
        pop = BackpropPopulation(config, xs, ys, mask_phantom=False)
        assert pop.mask_phantom is False
        before = copy.deepcopy(pop.population)
        pop.backpropagate(pop.xs, pop.ys, nEpochs=5)
        best_ref = []
        for k, g in before.items():
            ref_g, ref_loss = _reference_backprop_one(g, config, pop.xs, pop.ys, 5)
            assert _genome_params(pop.population[k]) == _genome_params(ref_g)
            best_ref.append(ref_loss)
        assert pop.generation_details[-1]["best_loss"] == min(best_ref)

    def test_default_trains_phantoms(self, config):
        """Documents the bug the mask fixes: phantoms move without the mask."""
        xs, ys = _data()
        net = NeuralNeat(_genome(), config)
        _train(net, xs, ys, 5)
        assert max(abs(float(net.weights[L][r, c]))
                   for L, r, c in _phantom_entries(net)) > 0


# --- (b) mask holds phantoms at exactly 0 --------------------------------------

class TestMask:
    def test_mask_is_buffer_marking_enabled_genes(self, config):
        net = NeuralNeat(_genome(), config, mask_phantom=True)
        buffers = dict(net.named_buffers())
        params = dict(net.named_parameters())
        assert not any(n.startswith("weight_mask") for n in params)
        for L, layer in net.layers.items():
            if layer["layer_type"] == LAYER_TYPE_INPUT:
                continue
            M = buffers[f"weight_mask_{L}"]
            assert M.shape == net.weights[L].shape
            assert not M.requires_grad
            expected = torch.zeros_like(M)
            for (r, c) in layer["input_map"].values():
                expected[r, c] = 1.0
            assert torch.equal(M, expected)
        # The disabled gene (-2, 2) has no mask entry.
        assert all((-2, 2) not in layer["input_map"] for layer in net.layers.values())

    def test_phantoms_stay_exactly_zero(self, config):
        xs, ys = _data()
        net = NeuralNeat(_genome(), config, mask_phantom=True)
        phantoms = _phantom_entries(net)
        assert len(phantoms) > 0
        _train(net, xs, ys, 25)
        for L, r, c in phantoms:
            assert float(net.weights[L][r, c]) == 0.0
        # real weights did train
        g0 = _genome()
        moved = [abs(float(net.weights[L][r, c]) - g0.connections[k].weight)
                 for L, layer in net.layers.items()
                 for k, (r, c) in layer["input_map"].items()]
        assert max(moved) > 1e-3

    def test_phantom_gradient_is_zero(self, config):
        xs, ys = _data()
        net = NeuralNeat(_genome(), config, mask_phantom=True)
        nn.BCELoss()(net.forward(xs), ys).backward()
        for L, r, c in _phantom_entries(net):
            assert float(net.weights[L].grad[r, c]) == 0.0

    def test_initial_forward_matches_unmasked(self, config):
        """Phantoms start at 0, so the untrained masked net is the same function."""
        xs, _ = _data()
        a = NeuralNeat(_genome(), config).forward(xs)
        b = NeuralNeat(_genome(), config, mask_phantom=True).forward(xs)
        assert torch.equal(a, b)

    def test_no_padding_biases(self, config):
        """Every bias entry of a non-input layer belongs to a real genome node,
        so biases need no mask (input-layer biases are never used in forward)."""
        net = NeuralNeat(_genome(), config, mask_phantom=True)
        for L, layer in net.layers.items():
            if layer["layer_type"] == LAYER_TYPE_INPUT:
                continue
            assert len(layer["nodes"]) == net.biases[L].numel()
            assert all(nid in net.genome.nodes for nid in layer["nodes"])


# --- (c) written-back genome is the trained network -----------------------------

class TestWriteBack:
    def test_writeback_genome_equals_trained_net(self, config):
        xs, ys = _data()
        bce = nn.BCELoss()
        net = NeuralNeat(copy.deepcopy(_genome()), config, mask_phantom=True)
        trained_loss = _train(net, xs, ys, 10)
        net.update_genome_weights()
        fresh = NeuralNeat(net.genome, config)
        with torch.no_grad():
            fresh_loss = bce(fresh.forward(xs), ys).item()
        assert abs(fresh_loss - trained_loss) <= 1e-12

    def test_population_writeback_matches_best_loss(self, config):
        import random
        torch.manual_seed(0)
        random.seed(3)
        xs, ys = _data()
        pop = BackpropPopulation(config, xs, ys, mask_phantom=True)
        pop.backpropagate(pop.xs, pop.ys, nEpochs=5)
        bce = nn.BCELoss()
        with torch.no_grad():
            fresh = [bce(NeuralNeat(g, config).forward(pop.xs), pop.ys).item()
                     for g in pop.population.values()]
        assert abs(min(fresh) - pop.generation_details[-1]["best_loss"]) <= 1e-12


# --- (d) seeded determinism ------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("mask", [False, True])
    def test_seeded_population_backprop_repeats(self, config, mask):
        import random
        xs, ys = _data()

        def run():
            config.genome_config.node_indexer = None  # neat's global node counter
            random.seed(11)
            np.random.seed(11)
            torch.manual_seed(11)
            pop = BackpropPopulation(config, xs, ys, mask_phantom=mask)
            pop.backpropagate(pop.xs, pop.ys, nEpochs=5)
            return {k: _genome_params(g) for k, g in pop.population.items()}

        assert run() == run()


# --- plumbing --------------------------------------------------------------------

class TestPlumbing:
    @pytest.mark.parametrize("mask", [False, True])
    def test_backpropagate_passes_flag(self, config, mask, monkeypatch):
        seen = []
        real = backproppop_mod.nneat

        def spy(*args, **kwargs):
            seen.append(kwargs.get("mask_phantom"))
            return real(*args, **kwargs)

        monkeypatch.setattr(backproppop_mod, "nneat", spy)
        import random
        random.seed(3)
        xs, ys = _data()
        pop = BackpropPopulation(config, xs, ys, mask_phantom=mask)
        pop.backpropagate(pop.xs, pop.ys, nEpochs=1)
        assert seen and all(s is mask for s in seen)
