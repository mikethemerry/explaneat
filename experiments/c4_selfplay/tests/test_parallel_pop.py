"""ParallelBackpropPopulation: bitwise identical to the serial BackpropPopulation.backpropagate."""
import copy
import random
import struct

import numpy as np
import pytest
import torch

import netfn
from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.errors import GenomeNotValidError
from fitness import neg_bce_fitness
from parallel_pop import ParallelBackpropPopulation

N_ROWS = 1500
EPOCHS = 4


@pytest.fixture(autouse=True)
def _one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)  # the serial reference (run.py uses 1 thread)
    yield
    torch.set_num_threads(old)


def bits(x):
    return struct.pack("<d", x)


def data(seed, n=N_ROWS):
    rng = np.random.default_rng(seed)
    X = rng.choice([-1.0, 0.0, 1.0], size=(n, 42))
    y = rng.choice([0.0, 0.5, 1.0], size=n)
    return (torch.tensor(X, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64).reshape(-1, 1))


def config_hidden(tmp_path, num_hidden):
    text = netfn.DEFAULT_CONFIG.read_text().replace("num_hidden = 0", f"num_hidden = {num_hidden}")
    assert f"num_hidden = {num_hidden}" in text
    p = tmp_path / f"h{num_hidden}.ini"
    p.write_text(text)
    return netfn.load_config(p)


def make_state(config, seed, pop_size=12, mutate=False, invalid=False):
    """(population, species, generation) of a seeded population, optionally with added hidden
    nodes/connections and one genome with every connection disabled (NeuralNeat rejects it)."""
    config.pop_size = pop_size
    random.seed(seed)
    xs, ys = data(seed)
    pop = BackpropPopulation(config, xs, ys)
    if mutate:
        for i, g in enumerate(pop.population.values()):
            for _ in range(i % 4):
                g.mutate_add_node(config.genome_config)
            for _ in range(i % 3):
                g.mutate_add_connection(config.genome_config)
    if invalid:
        g = next(iter(pop.population.values()))
        for c in g.connections.values():
            c.enabled = False
    return pop.population, pop.species, pop.generation


def genome_sig(g):
    conns = sorted((k, bits(c.weight), c.enabled) for k, c in g.connections.items())
    nodes = sorted((k, bits(n.bias), bits(n.response), n.activation, n.aggregation)
                   for k, n in g.nodes.items())
    fit = None if g.fitness is None else bits(g.fitness)
    return g.key, conns, nodes, fit, sorted(vars(g))


def pop_sig(pop):
    return [genome_sig(g) for g in pop.population.values()]


def species_sig(pop):
    out = []
    for sid, s in sorted(pop.species.species.items()):
        out.append((sid, sorted(s.members), s.representative.key, s.created, s.last_improved,
                    None if s.fitness is None else bits(s.fitness),
                    None if s.adjusted_fitness is None else bits(s.adjusted_fitness),
                    [bits(f) for f in s.fitness_history]))
    return out


def details_sig(pop):
    skip = {"mean_time_per_epoch", "total_time_per_backprop"}
    return [{k: bits(float(v)) for k, v in d.items() if k not in skip}
            for d in pop.generation_details]


def build(cls, config, state, xs, ys, **kw):
    # own config copy: genome_config.node_indexer is mutable state that new nodes advance
    return cls(copy.deepcopy(config), xs, ys, initial_state=copy.deepcopy(state),
               mask_phantom=True, **kw)


def rng_states():
    return (random.getstate(), np.random.get_state()[1].tobytes(),
            torch.get_rng_state().numpy().tobytes())


def compare_backprop(config, state, workers, calls=1, skip_invalid=False):
    xs, ys = data(100)
    serial = build(BackpropPopulation, config, state, xs, ys, skip_invalid_genomes=skip_invalid)
    par = build(ParallelBackpropPopulation, config, state, xs, ys,
                skip_invalid_genomes=skip_invalid, workers=workers)
    assert pop_sig(serial) == pop_sig(par)  # identical start
    try:
        for _ in range(calls):
            before = rng_states()
            serial.backpropagate(xs, ys, nEpochs=EPOCHS)
            assert rng_states() == before  # the serial loop draws no random numbers
            par.backpropagate(xs, ys, nEpochs=EPOCHS)
            assert rng_states() == before
        assert pop_sig(serial) == pop_sig(par)
        assert details_sig(serial) == details_sig(par)
        assert len(serial.backprop_times) == len(par.backprop_times)
        # written into the same objects the species reference
        members = {id(g) for s in par.species.species.values() for g in s.members.values()}
        assert members <= {id(g) for g in par.population.values()}
        # training changed something
        start = build(BackpropPopulation, config, state, xs, ys)
        assert pop_sig(start) != pop_sig(par)
    finally:
        par.close()


def test_base_config_workers2():
    config = netfn.load_config()
    compare_backprop(config, make_state(config, 0), workers=2, calls=2)


def test_workers1_is_serial(monkeypatch):
    config = netfn.load_config()
    state = make_state(config, 1, pop_size=4)
    xs, ys = data(1)
    par = build(ParallelBackpropPopulation, config, state, xs, ys, workers=1)
    called = []
    orig = BackpropPopulation.backpropagate
    monkeypatch.setattr(BackpropPopulation, "backpropagate",
                        lambda self, *a, **k: (called.append(1), orig(self, *a, **k))[1])
    par.backpropagate(xs, ys, nEpochs=2)
    assert called and par._pool is None


@pytest.mark.slow
def test_mutated_hidden_workers4():
    config = netfn.load_config()
    compare_backprop(config, make_state(config, 2, mutate=True), workers=4)


@pytest.mark.slow
def test_num_hidden_10(tmp_path):
    config = config_hidden(tmp_path, 10)
    state = make_state(config, 3, mutate=True)
    assert all(len(g.nodes) >= 11 for g in state[0].values())
    for w in (2, 4):
        compare_backprop(config, state, workers=w)


def test_invalid_genome_skipped_and_raised():
    config = netfn.load_config()
    state = make_state(config, 4, pop_size=6, invalid=True)
    compare_backprop(config, state, workers=2, skip_invalid=True)
    xs, ys = data(4)
    for cls, kw in ((BackpropPopulation, {}), (ParallelBackpropPopulation, {"workers": 2})):
        p = build(cls, config, state, xs, ys, **kw)
        with pytest.raises(GenomeNotValidError):
            p.backpropagate(xs, ys, nEpochs=2)
        if kw:
            p.close()


def test_data_swap_picked_up():
    """A new xs/ys tensor (as run.py sets each round) and an in-place edit both reach the
    workers; the pool persists."""
    config = netfn.load_config()
    state = make_state(config, 5, pop_size=8)
    x1, y1 = data(11)
    serial = build(BackpropPopulation, config, state, x1, y1)
    par = build(ParallelBackpropPopulation, config, state, x1, y1, workers=2)
    try:
        par.backpropagate(par.xs, par.ys, nEpochs=EPOCHS)
        serial.backpropagate(serial.xs, serial.ys, nEpochs=EPOCHS)
        pool, dir1 = par._pool, par._state_dir
        x2, y2 = data(12, n=900)
        for p in (serial, par):
            p.xs, p.ys = x2.clone(), y2.clone()
            p.backpropagate(p.xs, p.ys, nEpochs=EPOCHS)
        assert par._pool is pool and par._state_dir != dir1
        assert pop_sig(serial) == pop_sig(par)
        dir2 = par._state_dir
        for p in (serial, par):
            p.ys.mul_(0.5)  # in place: same tensor object, new version
            p.backpropagate(p.xs, p.ys, nEpochs=EPOCHS)
        assert par._state_dir != dir2
        assert pop_sig(serial) == pop_sig(par)
        assert details_sig(serial) == details_sig(par)
        # the swapped data mattered: a run that never swapped differs
        stay = build(BackpropPopulation, config, state, x1, y1)
        for _ in range(3):
            stay.backpropagate(x1, y1, nEpochs=EPOCHS)
        assert pop_sig(stay) != pop_sig(par)
    finally:
        par.close()


@pytest.mark.slow
def test_run_three_generations():
    config = netfn.load_config()
    state = make_state(config, 6, pop_size=20)
    xs, ys = data(66)
    out = []
    for cls, kw in ((BackpropPopulation, {}), (ParallelBackpropPopulation, {"workers": 3})):
        random.seed(123)
        torch.manual_seed(0)
        p = build(cls, config, state, xs, ys, skip_invalid_genomes=True, **kw)
        best = p.run(neg_bce_fitness, n=3, nEpochs=EPOCHS)
        out.append((pop_sig(p), species_sig(p), details_sig(p), genome_sig(best),
                    p.best_generation, p.generation, p.ancestry_rows(), rng_states()))
        if kw:
            p.close()
    assert out[0] == out[1]
