"""DatabaseBackpropPopulation records parents by default, and doing so is RNG-neutral.

A seeded run with ancestry tracking (the default) and one with
``track_ancestry=False`` must produce bit-identical genomes and fitness in
every generation. With tracking on, every genome after generation 0 has a
parent row in the previous generation; elite copies (same NEAT key) point at
their own previous row (see explaneat/core/lineage_capture.py).
"""
import random

import numpy as np
import pytest
import torch

from tests.test_core.test_neuralneat_mask_phantom import CONFIG_TEXT, _data


N_GEN = 6


def _fresh_config(tmp_path):
    """A new Config per run: neat-python keeps the hidden-node id counter on the
    genome config, so reusing one Config across runs shifts node ids."""
    import neat
    path = tmp_path / "cfg.ini"
    path.write_text(CONFIG_TEXT)
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


def _seed(s=7):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _run(tmp_path, name, **kwargs):
    from explaneat.db import db
    from explaneat.db.models import Genome, Population
    from explaneat.db.population import DatabaseBackpropPopulation
    from explaneat.evaluators.evaluators import auc_fitness

    config = _fresh_config(tmp_path)
    url = f"sqlite:///{tmp_path / (name + '.db')}"
    db.init_db(url)
    db.create_all()
    try:
        xs, ys = _data()
        _seed()
        pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name=name,
                                         database_url=url, mask_phantom=False, **kwargs)
        pop.run(auc_fitness, n=N_GEN, nEpochs=2)
        with db.session_scope() as s:
            rows = (s.query(Population.generation, Genome.id, Genome.genome_id,
                            Genome.fitness, Genome.genome_data,
                            Genome.parent1_id, Genome.parent2_id)
                    .join(Genome, Genome.population_id == Population.id)
                    .filter(Population.experiment_id == pop.experiment_id).all())
        return pop, [tuple(r) for r in rows]
    finally:
        db.close()


def test_ancestry_tracking_is_rng_neutral_and_fills_parents(tmp_path):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        pop_on, on = _run(tmp_path, "on")
        pop_off, off = _run(tmp_path, "off", track_ancestry=False)
    finally:
        torch.set_num_threads(n_threads)
    assert pop_on.ancestry_reporter is not None
    assert pop_off.ancestry_reporter is None

    def content(rows):
        return sorted((g, k, f, repr(sorted(d["connections"].items())),
                       repr(sorted(d["nodes"].items())))
                      for g, _, k, f, d, _, _ in rows)

    assert len(on) == len(off) > 0
    assert content(on) == content(off)  # bit-identical genomes and fitness
    assert {g for g, *_ in on} == set(range(N_GEN))

    # Opt-out leaves the columns NULL.
    assert all(p1 is None and p2 is None for *_, p1, p2 in off)

    by_id = {rid: (g, k) for g, rid, k, *_ in on}
    n_elite = n_child = 0
    for g, rid, k, f, d, p1, p2 in on:
        if g == 0:
            assert p1 is None and p2 is None
            continue
        assert p1 is not None
        pg, pk = by_id[p1]
        assert pg == g - 1
        log = pop_on.parent_log[g][k]
        if log[2]:  # elite copy: own key one generation earlier
            n_elite += 1
            assert pk == k and p2 is None
        else:
            n_child += 1
            assert (pk, by_id[p2][1]) == (log[0], log[1])
            assert by_id[p2][0] == g - 1
    assert n_elite > 0 and n_child > 0


def test_in_memory_parent_log(tmp_path):
    from explaneat.core.backproppop import BackpropPopulation
    from explaneat.evaluators.evaluators import auc_fitness

    config = _fresh_config(tmp_path)
    xs, ys = _data()
    _seed()
    pop = BackpropPopulation(config, xs, ys, mask_phantom=False)
    pop.run(auc_fitness, n=4, nEpochs=1)
    assert sorted(pop.parent_log) == list(range(5))
    assert all(v == (None, None, False) for v in pop.parent_log[0].values())
    for gen in range(1, 5):
        prev = set(pop.parent_log[gen - 1])
        for k, (p1, p2, elite) in pop.parent_log[gen].items():
            if elite:
                assert p1 == k and k in prev and p2 is None
            else:
                assert p1 in prev and p2 in prev and k not in prev
    rows = pop.ancestry_rows()
    assert len(rows) == sum(len(v) for v in pop.parent_log.values())


def test_generation_parents_unit():
    from explaneat.core.lineage_capture import generation_parents
    anc = {1: (), 2: (), 3: (1, 2), 4: (2, 2), 1_000: (9, 9)}
    out = generation_parents([1, 3, 4], previous_keys=[1, 2], ancestors=anc)
    assert out == {1: (1, None, True), 3: (1, 2, False), 4: (2, 2, False)}
    assert generation_parents([1, 2], (), anc) == {1: (None, None, False),
                                                   2: (None, None, False)}
