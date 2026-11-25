import copy

import neat
import numpy as np
import pytest
import torch

import netfn
from explaneat.core.errors import GenomeNotValidError
from explaneat.core.neuralneat import NeuralNeat


@pytest.fixture(scope="module")
def cfg():
    return netfn.load_config()


@pytest.fixture
def genome(cfg):
    g = neat.DefaultGenome(1)
    g.configure_new(cfg.genome_config)
    return g


def _X(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 0.0, 1.0], size=(n, 42))


def test_config_shape(cfg):
    assert cfg.pop_size == 150
    assert cfg.genome_config.num_inputs == 42
    assert cfg.genome_config.num_outputs == 1


def test_valuefn_shape_range_deterministic(cfg, genome):
    f = netfn.ValueFn(genome, cfg)
    X = _X(5)
    v = f(X)
    assert v.shape == (5,)
    assert v.dtype == np.float64
    assert np.all((v > 0) & (v < 1))
    np.testing.assert_array_equal(v, f(X))


def test_valuefn_single_row(cfg, genome):
    f = netfn.ValueFn(genome, cfg)
    X = _X(3)
    v1 = f(X[1])
    assert v1.shape == (1,)
    assert v1[0] == f(X)[1]


def test_valuefn_matches_neuralneat(cfg, genome):
    X = _X(7, seed=3)
    expected = (
        NeuralNeat(genome, cfg, mask_phantom=True)
        .forward(torch.tensor(X))
        .detach()
        .numpy()
        .reshape(-1)
    )
    np.testing.assert_array_equal(netfn.ValueFn(genome, cfg)(X), expected)


def test_valuefn_does_not_mutate_genome(cfg, genome):
    before = copy.deepcopy(genome)
    netfn.ValueFn(genome, cfg)(_X(4))
    assert set(before.connections) == set(genome.connections)
    for k, c in before.connections.items():
        assert genome.connections[k].weight == c.weight
        assert genome.connections[k].enabled == c.enabled
    for k, n in before.nodes.items():
        assert genome.nodes[k].bias == n.bias


def test_valuefn_invalid_genome_raises(cfg, genome):
    for c in genome.connections.values():
        c.enabled = False
    with pytest.raises(GenomeNotValidError):
        netfn.ValueFn(genome, cfg)


def test_constvalue():
    c = netfn.ConstValue()
    v = c(_X(4))
    assert v.shape == (4,)
    assert np.all(v == 0.5)
    assert netfn.ConstValue(0.3)(_X(1)[0]).tolist() == [0.3]
