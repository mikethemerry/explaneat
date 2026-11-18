"""ConvNEAT prototype: totality, function preservation, crossover/disable logging."""

import numpy as np
import pytest
import torch

from explaneat.conv.crossover import crossover, distance, kernel_distance
from explaneat.conv.genome import ConvNeatConfig, InnovationTracker, is_valid, minimal_genome
from explaneat.conv.mutation import (mutate, mutate_add_conn, mutate_add_node, mutate_kernel_size,
                                     mutate_toggle_enable, mutate_weights, random_genome_by_mutation)
from explaneat.conv.network import NaiveConvNet
from explaneat.conv.propneat import PropNEATConvNet

torch.set_num_threads(1)


def _out(g, cfg, x, engine="naive"):
    with torch.no_grad():
        cls = NaiveConvNet if engine == "naive" else PropNEATConvNet
        return cls(g, cfg, dtype=torch.float64)(x)


def _grown(cfg, seed, n_nodes=4, n_conns=4):
    """A genome with hidden nodes and non-trivial weights (same-level splits only)."""
    rng = np.random.default_rng(seed)
    tr = InnovationTracker(cfg.n_out_maps)
    g = minimal_genome(0, cfg, tr, rng)
    for _ in range(n_nodes):
        mutate_add_node(g, cfg, tr, rng, coarse=False)
    for _ in range(n_conns):
        mutate_add_conn(g, cfg, tr, rng)
    for n in g.nodes.values():
        if n.kind == "hidden":
            n.level = int(rng.integers(0, cfg.max_level + 1))
            n.bias = float(rng.normal() * 0.1)
    for c in g.conns.values():
        c.kernel = (rng.standard_normal(c.kernel.shape) * 0.5).astype(np.float32)
    return g, tr, rng


# ---------------------------------------------------------------- totality
def test_totality_random_genomes():
    """1,000+ random genomes from random mutations all build and forward-pass."""
    configs = [
        ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=1, max_level=2),
        ConvNeatConfig(in_channels=1, image_hw=(8, 8), n_classes=10, max_level=2),
        ConvNeatConfig(in_channels=3, image_hw=(7, 9), n_classes=3, max_level=3),
    ]
    count = 0
    for ci, cfg in enumerate(configs):
        tr = InnovationTracker(cfg.n_out_maps)
        rng = np.random.default_rng(100 + ci)
        x = torch.rand(2, cfg.in_channels, *cfg.image_hw, dtype=torch.float64) - 0.3
        for i in range(350):
            g = random_genome_by_mutation(i, cfg, tr, rng, n_steps=int(rng.integers(0, 40)))
            ok, why = is_valid(g, cfg)
            assert ok, why
            for engine in ("naive", "propneat"):
                out = _out(g, cfg, x, engine)
                assert out.shape == (2, cfg.n_classes)
                assert torch.isfinite(out).all()
            count += 1
    assert count >= 1000


def test_totality_under_crossover():
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    tr = InnovationTracker(1)
    rng = np.random.default_rng(7)
    pool = [random_genome_by_mutation(i, cfg, tr, rng, n_steps=20) for i in range(40)]
    x = torch.rand(2, 1, 16, 16, dtype=torch.float64)
    for i in range(300):
        a, b = pool[rng.integers(40)], pool[rng.integers(40)]
        a.fitness, b.fitness = float(rng.integers(0, 2)), float(rng.integers(0, 2))  # many ties
        child = crossover(a, b, 1000 + i, cfg, rng)
        mutate(child, cfg, tr, rng)
        ok, why = is_valid(child, cfg)
        assert ok, why
        assert torch.isfinite(_out(child, cfg, x)).all()


# ---------------------------------------------------------------- preservation
@pytest.mark.parametrize("seed", range(10))
def test_add_node_preserves_function(seed):
    """Default split rule (N at A's or B's level), mixed resolutions, both engines."""
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    g, tr, rng = _grown(cfg, seed)
    x = torch.rand(8, 1, 16, 16, dtype=torch.float64)  # inputs >= 0, as in all datasets here
    before = _out(g, cfg, x)
    cross = 0
    for _ in range(6):
        mutate_add_node(g, cfg, tr, rng, coarse=False)
        for engine in ("naive", "propneat"):
            after = _out(g, cfg, x, engine)
            assert (after - before).abs().max().item() < 1e-5
    cross = sum(1 for c in g.conns.values() if g.nodes[c.src].level != g.nodes[c.dst].level)
    assert cross > 0  # the cross-resolution case is exercised


def test_coarse_split_is_approximate():
    """Documented limit: N coarser than both A and B is a resolution bottleneck."""
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    rng = np.random.default_rng(0)
    tr = InnovationTracker(1)
    g = minimal_genome(0, cfg, tr, rng)
    x = torch.rand(4, 1, 16, 16, dtype=torch.float64)
    before = _out(g, cfg, x)
    mutate_add_node(g, cfg, tr, rng, coarse=True)
    assert g.nodes[g.hidden_keys()[0]].level == 1
    assert (_out(g, cfg, x) - before).abs().max().item() > 1e-4


def test_add_node_is_approximate_for_negative_inputs():
    """Documented limit: A = raw input with negative values is clipped by relu(N)."""
    cfg = ConvNeatConfig(image_hw=(8, 8), max_level=1)
    rng = np.random.default_rng(0)
    tr = InnovationTracker(1)
    g = minimal_genome(0, cfg, tr, rng)
    x = torch.rand(4, 1, 8, 8, dtype=torch.float64) - 0.5
    before = _out(g, cfg, x)
    mutate_add_node(g, cfg, tr, rng, coarse=False)
    assert (_out(g, cfg, x) - before).abs().max().item() > 1e-3


@pytest.mark.parametrize("seed", range(10))
def test_add_conn_zero_kernel_preserves_function(seed):
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    g, tr, rng = _grown(cfg, seed, n_conns=0)
    x = torch.rand(8, 1, 16, 16, dtype=torch.float64) - 0.2
    before = _out(g, cfg, x)
    added = 0
    for _ in range(5):
        if mutate_add_conn(g, cfg, tr, rng):
            added += 1
        for engine in ("naive", "propneat"):
            assert (_out(g, cfg, x, engine) - before).abs().max().item() < 1e-10
    assert added > 0


@pytest.mark.parametrize("old_k,new_k", [(1, 3), (1, 5), (3, 5)])
def test_kernel_growth_preserves_function(old_k, new_k):
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    for seed in range(5):
        g, tr, rng = _grown(cfg, seed)
        for c in g.conns.values():
            c.kernel = (rng.standard_normal((old_k, old_k)) * 0.5).astype(np.float32)
        x = torch.rand(8, 1, 16, 16, dtype=torch.float64) - 0.2
        before = _out(g, cfg, x)
        for inn in list(g.conns):
            mutate_kernel_size(g, cfg, rng, new_k=new_k, innov=inn)
            assert g.conns[inn].k == new_k
        for engine in ("naive", "propneat"):
            assert (_out(g, cfg, x, engine) - before).abs().max().item() < 1e-6


# ---------------------------------------------------------------- crossover
def _related_pair(seed):
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    rng = np.random.default_rng(seed)
    tr = InnovationTracker(1)
    anc = minimal_genome(0, cfg, tr, rng)
    for _ in range(3):
        mutate_add_node(anc, cfg, tr, rng, coarse=False)
    a, b = anc.copy(1), anc.copy(2)
    for g in (a, b):
        for _ in range(6):
            mutate(g, cfg, tr, rng)
        mutate_weights(g, cfg, rng)
    return cfg, tr, rng, a, b


def _all_disables_logged(child):
    logged = {e["innov"] for e in child.disable_log}
    causes = {e["cause"] for e in child.disable_log}
    assert causes <= {"mutation", "recombination", "repair"}
    for c in child.conns.values():
        if not c.enabled:
            assert c.innov in logged, f"disabled gene {c.innov} has no logged cause"


@pytest.mark.parametrize("seed", range(10))
def test_crossover_related_genomes_valid_and_logged(seed):
    cfg, tr, rng, a, b = _related_pair(seed)
    a.fitness, b.fitness = 0.9, 0.8
    child = crossover(a, b, 3, cfg, rng)
    ok, why = is_valid(child, cfg)
    assert ok, why
    assert set(child.conns) == set(a.conns)  # disjoint/excess from the fitter parent
    assert any(h.startswith("recombination:") for h in child.history)
    _all_disables_logged(child)
    x = torch.rand(4, 1, 16, 16, dtype=torch.float64)
    assert torch.isfinite(_out(child, cfg, x)).all()
    assert distance(a, b, cfg) >= 0 and distance(a, a, cfg) == 0


def test_crossover_tie_cycle_is_repaired_and_logged():
    """On a tie, disjoint genes from both parents can close a cycle -> junk DNA."""
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    rng = np.random.default_rng(0)
    tr = InnovationTracker(1)
    anc = minimal_genome(0, cfg, tr, rng)
    mutate_add_node(anc, cfg, tr, rng, coarse=False)   # in -> h1 -> out
    mutate_add_node(anc, cfg, tr, rng, coarse=False)   # split again
    h = anc.hidden_keys()
    assert len(h) == 2
    a, b = anc.copy(1), anc.copy(2)
    from explaneat.conv.genome import ConnGene, creates_cycle
    h1, h2 = h
    # parent a: h1 -> h2 if acyclic in a, parent b: the reverse
    for g, (s, d) in ((a, (h1, h2)), (b, (h2, h1))):
        g.conns = {i: c for i, c in g.conns.items() if not (c.src in h and c.dst in h)}
        assert not creates_cycle([(c.src, c.dst) for c in g.conns.values()], (s, d))
        inn = tr.get_conn(s, d)
        g.conns[inn] = ConnGene(inn, s, d, np.ones((3, 3), np.float32))
    a.fitness = b.fitness = 0.5
    child = crossover(a, b, 3, cfg, rng)
    ok, why = is_valid(child, cfg)
    assert ok, why
    assert any(e["cause"] == "repair" for e in child.disable_log)
    _all_disables_logged(child)


def test_reenable_is_validated():
    cfg = ConvNeatConfig(image_hw=(8, 8), max_level=1)
    rng = np.random.default_rng(1)
    tr = InnovationTracker(1)
    g = minimal_genome(0, cfg, tr, rng)
    mutate_add_node(g, cfg, tr, rng, coarse=False)
    mutate_add_node(g, cfg, tr, rng, coarse=False)
    h1, h2 = g.hidden_keys()
    from explaneat.conv.genome import ConnGene
    # a disabled h2->h1 gene plus an enabled h1->h2 path: re-enabling must be refused
    for s, d, en in ((h1, h2, True), (h2, h1, False)):
        inn = tr.get_conn(s, d)
        g.conns[inn] = ConnGene(inn, s, d, np.ones((1, 1), np.float32), en)
    bad = tr.get_conn(h2, h1)
    g.conns = {i: c for i, c in g.conns.items() if i == bad or c.enabled}
    for _ in range(50):
        mutate_toggle_enable(g, cfg, rng)
        ok, why = is_valid(g, cfg)
        assert ok, why
    assert not g.conns[bad].enabled
    assert any(e["cause"] == "repair" and e["innov"] == bad for e in g.disable_log)


def test_kernel_distance_pads():
    a = np.ones((1, 1), np.float32)
    b = np.zeros((3, 3), np.float32)
    b[1, 1] = 1.0
    assert kernel_distance(a, b) == 0.0
