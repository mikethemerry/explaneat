"""Search schedules (explaneat/conv/schedule.py): default path unchanged, deletion mutations valid,
phase switching at the configured generations, self-adaptive inheritance."""

import copy
import hashlib

import numpy as np
import pytest
import torch

from explaneat.conv.crossover import crossover
from explaneat.conv.data import digits_splits
from explaneat.conv.genome import ConvNeatConfig, InnovationTracker, is_valid
from explaneat.conv.mutation import mutate, random_genome_by_mutation
from explaneat.conv.network import NaiveConvNet
from explaneat.conv.population import ConvNeatPopulation
from explaneat.conv.propneat import PropNEATConvNet
from explaneat.conv.schedule import (FIXED_4X, PHASE_CYCLE, Phase, PhaseSchedule, ScheduledPopulation,
                                     SelfAdaptivePopulation, SuccessiveHalvingPopulation, mutate_delete_conn,
                                     mutate_delete_node, mutate_deletions, prune_dead_ends)

torch.set_num_threads(1)
CPU = torch.device("cpu")


def _small_digits():
    d = digits_splits(0)
    return {k: (v[:300] if k.startswith(("X_", "y_")) else v) for k, v in d.items()}


def _cfg(**kw):
    c = dict(in_channels=1, image_hw=(8, 8), n_classes=10, pop_size=8, epochs_per_gen=1,
             complexity_penalty=0.001)
    c.update(kw)
    return ConvNeatConfig(**c)


def _fingerprint(pop, gens=4):
    fits = []
    for gen in range(gens):
        pop.step(gen, verbose=False)
        fits.append([g.fitness for g in pop.population])
        if gen < gens - 1:
            pop.reproduce()
    h = hashlib.sha256()
    for g in pop.population:
        for k in sorted(g.conns):
            c = g.conns[k]
            h.update(np.ascontiguousarray(c.kernel).tobytes())
            h.update(str((k, c.src, c.dst, c.enabled)).encode())
    return fits, h.hexdigest()


# Recorded with the code *before* the schedule hooks were added to population.py (commit f9b3a3c,
# CPU, 1 thread): 4 generations, population 8, digits10 (300 rows), structural rates x4.
GOLDEN_HASH = "1cef2a498c58871faf16b7b1dd556227fc737030d74ffd4993ac98044d4bba2a"


def test_default_path_bit_for_bit_against_pre_hook_code():
    cfg = _cfg(p_add_node=1.0, p_add_conn=1.4, p_kernel_size=0.4)
    pop = ConvNeatPopulation(cfg, _small_digits(), 0, CPU, fitness_split="train")
    _, h = _fingerprint(pop)
    assert h == GOLDEN_HASH


def test_fixed_schedule_equals_plain_population():
    """Arm 1 written as a one-phase schedule reproduces the plain loop exactly."""
    data = _small_digits()
    plain_cfg = _cfg()
    for f in ("p_add_node", "p_add_conn", "p_kernel_size"):
        setattr(plain_cfg, f, getattr(plain_cfg, f) * 4)
    a = _fingerprint(ConvNeatPopulation(plain_cfg, data, 0, CPU, fitness_split="train"))
    b = _fingerprint(ScheduledPopulation(_cfg(), data, 0, CPU, FIXED_4X, fitness_split="train"))
    assert a == b


# ---------------------------------------------------------------- phase switching
def test_phase_cycle_switches_at_configured_generations():
    sw = PHASE_CYCLE.switches(25)
    assert sw == [(0, "explore"), (5, "exploit"), (8, "scarcity"), (10, "explore"), (15, "exploit"),
                  (18, "scarcity"), (20, "explore")]
    assert PHASE_CYCLE.cycle == 10


def test_scheduled_population_applies_phase_settings():
    sched = PhaseSchedule([Phase("a", 2, struct_scale=3.0, epochs=1),
                           Phase("b", 1, struct_scale=0.0, other_struct=False, weight_scale=0.0, epochs=2,
                                 penalty_scale=5.0, p_delete_conn=1.0)])
    base = _cfg(pop_size=6)
    pop = ScheduledPopulation(copy.deepcopy(base), _small_digits(), 1, CPU, sched, fitness_split="train")
    seen = []
    for gen in range(6):
        row = pop.step(gen, verbose=False)
        seen.append((pop.schedule_log(gen)["phase"], pop.cfg.epochs_per_gen, pop.cfg.complexity_penalty))
        assert row["generation"] == gen
        if gen < 5:
            pop.reproduce()
            ph = sched.phase_at(gen + 1)
            assert pop.cfg.p_add_node == pytest.approx(base.p_add_node * ph.struct_scale)
    assert [s[0] for s in seen] == ["a", "a", "b", "a", "a", "b"]
    assert [s[1] for s in seen] == [1, 1, 2, 1, 1, 2]
    assert seen[2][2] == pytest.approx(5 * base.complexity_penalty)
    assert seen[0][2] == pytest.approx(base.complexity_penalty)
    # generation 2 was created in phase b: deletions only, so no add-node / add-conn events
    assert pop.event_counts.get("mutation:delete_conn", 0) > 0


# ---------------------------------------------------------------- deletion mutations
def _forward_ok(g, cfg, x):
    for eng in (NaiveConvNet, PropNEATConvNet):
        with torch.no_grad():
            out = eng(g, cfg, dtype=torch.float64)(x)
        assert out.shape == (x.shape[0], cfg.n_classes)
        assert torch.isfinite(out).all()


def test_totality_with_deletion_mutations():
    configs = [ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=1, max_level=2),
               ConvNeatConfig(in_channels=1, image_hw=(8, 8), n_classes=10, max_level=2),
               ConvNeatConfig(in_channels=3, image_hw=(7, 9), n_classes=3, max_level=3)]
    count = 0
    for ci, cfg in enumerate(configs):
        tr = InnovationTracker(cfg.n_out_maps)
        rng = np.random.default_rng(200 + ci)
        x = torch.rand(2, cfg.in_channels, *cfg.image_hw, dtype=torch.float64) - 0.3
        for i in range(120):
            g = random_genome_by_mutation(i, cfg, tr, rng, n_steps=int(rng.integers(0, 40)))
            for _ in range(int(rng.integers(1, 12))):
                op = rng.integers(3)
                if op == 0:
                    mutate_delete_conn(g, cfg, rng)
                elif op == 1:
                    mutate_delete_node(g, cfg, rng)
                else:
                    mutate(g, cfg, tr, rng)  # interleave growth and deletion
            ok, why = is_valid(g, cfg)
            assert ok, why
            _forward_ok(g, cfg, x)
            count += 1
    assert count >= 350


def test_totality_crossover_after_deletions():
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    tr = InnovationTracker(1)
    rng = np.random.default_rng(9)
    pool = []
    for i in range(30):
        g = random_genome_by_mutation(i, cfg, tr, rng, n_steps=25)
        mutate_deletions(g, cfg, rng, 2.0, 1.0)
        pool.append(g)
    x = torch.rand(2, 1, 16, 16, dtype=torch.float64)
    for i in range(200):
        a, b = pool[rng.integers(30)], pool[rng.integers(30)]
        a.fitness, b.fitness = float(rng.integers(0, 2)), float(rng.integers(0, 2))
        child = crossover(a, b, 1000 + i, cfg, rng)
        mutate_deletions(child, cfg, rng, 1.0, 0.5)
        mutate(child, cfg, tr, rng)
        ok, why = is_valid(child, cfg)
        assert ok, why
        _forward_ok(child, cfg, x)


def test_delete_conn_prefers_small_kernels_and_keeps_inputs():
    cfg = ConvNeatConfig(image_hw=(8, 8), n_classes=10, max_level=1)
    tr = InnovationTracker(10)
    rng = np.random.default_rng(3)
    g = random_genome_by_mutation(0, cfg, tr, rng, n_steps=40)
    for c in g.conns.values():
        c.kernel = (np.ones_like(c.kernel) * (1 + c.innov)).astype(np.float32)
    n_in = {k: 0 for k in g.nodes}
    for c in g.enabled_conns():
        n_in[c.dst] += 1
    eligible = sorted((c for c in g.enabled_conns() if n_in[c.dst] > 1),
                      key=lambda c: (float(np.linalg.norm(c.kernel)), c.innov))
    smallest = {c.innov for c in eligible[:max(1, int(np.ceil(0.25 * len(eligible))))]}
    before = set(g.conns)
    assert mutate_delete_conn(g, cfg, rng) == "delete_conn"
    assert (before - set(g.conns)) & smallest
    # nothing that had an input lost its last one
    n_in2 = {k: 0 for k in g.nodes}
    for c in g.enabled_conns():
        n_in2[c.dst] += 1
    for k in g.nodes:
        if n_in[k] > 0 and k in n_in2:
            assert n_in2[k] > 0


def test_prune_dead_ends_is_exact():
    cfg = ConvNeatConfig(image_hw=(8, 8), n_classes=3, max_level=1)
    tr = InnovationTracker(3)
    rng = np.random.default_rng(5)
    x = torch.rand(4, 1, 8, 8, dtype=torch.float64)
    n_pruned = 0
    for i in range(40):
        g = random_genome_by_mutation(i, cfg, tr, rng, n_steps=30)
        with torch.no_grad():
            before = NaiveConvNet(g, cfg, dtype=torch.float64)(x)
        n_pruned += prune_dead_ends(g)
        assert is_valid(g, cfg)[0]
        with torch.no_grad():
            after = NaiveConvNet(g, cfg, dtype=torch.float64)(x)
        assert torch.equal(before, after)
    assert n_pruned > 0


# ---------------------------------------------------------------- self-adaptive and halving
def test_self_adaptive_strategy_inherited_and_logged():
    pop = SelfAdaptivePopulation(_cfg(pop_size=10), _small_digits(), 0, CPU, fitness_split="train",
                                 epochs_range=(1.0, 3.0), init_epochs=2.0)
    for gen in range(3):
        pop.step(gen, verbose=False)
        log = pop.schedule_log(gen)
        assert 1.0 <= log["epochs_mean"] <= 3.0 and log["mut_scale_mean"] > 0
        if gen < 2:
            parents = {g.key: g.strategy for g in pop.population}
            pop.reproduce()
            vals = {(round(s["mut_scale"], 9), round(s["epochs"], 9)) for s in parents.values()}
            for g in pop.population:
                s = g.strategy
                # each value is a parent's, possibly x0.8 or x1.25 (clipped)
                assert any(np.isclose(s["mut_scale"], m * f) or np.isclose(s["mut_scale"], 16.0)
                           or np.isclose(s["mut_scale"], 0.25)
                           for m, _ in vals for f in (1.0, 0.8, 1.25))


def test_successive_halving_budget():
    pop = SuccessiveHalvingPopulation(_cfg(pop_size=10), _small_digits(), 0, CPU, fitness_split="train",
                                      base=1, top_frac=0.2, top_extra=2, leader_total=4)
    pop.step(0, verbose=False)
    log = pop.schedule_log(0)
    # 2 top genomes get +2; the only species leader (generation 0: one group) is brought to 4
    assert log["extra_epochs"] == 2 * 2 + (4 - 3)
    assert pop.n_trainings == 10 + 2 + 1
