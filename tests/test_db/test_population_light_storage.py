"""Light genome storage (``genome_storage="best"``) and the ``on_generation`` stop hook.

The default (``"all"``) is unchanged. In ``"best"`` mode each generation stores only its best
genome, except the ``full_save_generations`` and the generation at which ``on_generation`` stops
the run; the evolution itself is bit-identical to a run that stores everything.
"""
import random

import numpy as np
import torch

from tests.test_core.test_neuralneat_mask_phantom import _data
from tests.test_db.test_population_ancestry import _fresh_config, _seed

N_GEN = 6


def _run(tmp_path, name, n=N_GEN, on_generation=None, **kwargs):
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
        pop.run(auc_fitness, n=n, nEpochs=2, on_generation=on_generation)
        with db.session_scope() as s:
            rows = (s.query(Population.generation, Genome.genome_id, Genome.fitness,
                            Genome.genome_data, Genome.parent1_id)
                    .join(Genome, Genome.population_id == Population.id)
                    .filter(Population.experiment_id == pop.experiment_id).all())
            n_pop = s.query(Population).filter(Population.experiment_id == pop.experiment_id).count()
        return pop, [tuple(r) for r in rows], n_pop
    finally:
        db.close()


def _key(r):
    g, k, f, d, _ = r
    return (g, k, f, repr(sorted(d["connections"].items())), repr(sorted(d["nodes"].items())))


def test_best_storage_keeps_evolution_identical_and_stores_less(tmp_path):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, full, n_full = _run(tmp_path, "all")
        pop, light, n_light = _run(tmp_path, "best", genome_storage="best", full_save_generations=[2])
    finally:
        torch.set_num_threads(n_threads)
    assert n_full == n_light == N_GEN
    by_gen = {}
    for r in light:
        by_gen.setdefault(r[0], []).append(r)
    full_by_gen = {}
    for r in full:
        full_by_gen.setdefault(r[0], []).append(r)
    # generation 2 in full, every other generation one row: the best of the full run's generation
    assert len(by_gen[2]) == len(full_by_gen[2])
    for g in range(N_GEN):
        if g == 2:
            assert sorted(map(_key, by_gen[g])) == sorted(map(_key, full_by_gen[g]))
            continue
        assert len(by_gen[g]) == 1
        best = max(full_by_gen[g], key=lambda r: (r[2], -r[1]))
        assert _key(by_gen[g][0]) == _key(best)
    # no DB parent ids in light mode; the in-memory log has them
    assert all(r[4] is None for r in light)
    assert pop.ancestry_reporter is None
    assert {row[0] for row in pop.ancestry_rows()} >= set(range(1, N_GEN))


def test_on_generation_stops_and_stores_the_stop_generation_in_full(tmp_path):
    seen = []

    def stop_at_3(gen, population, best):
        seen.append((gen, best.fitness))
        return "monitor plateau" if gen == 3 else None

    torch.set_num_threads(1)
    pop, rows, n_pop = _run(tmp_path, "stop", n=10, on_generation=stop_at_3, genome_storage="best")
    assert pop.stop_reason == "monitor plateau"
    assert [g for g, _ in seen] == [0, 1, 2, 3]
    assert n_pop == 4
    gens = [r[0] for r in rows]
    assert gens.count(3) == len(pop.population)   # stop generation stored in full
    assert all(gens.count(g) == 1 for g in (0, 1, 2))


def test_default_is_all(tmp_path):
    from explaneat.db.population import DatabaseBackpropPopulation
    import inspect
    sig = inspect.signature(DatabaseBackpropPopulation.__init__)
    assert sig.parameters["genome_storage"].default == "all"
    assert inspect.signature(DatabaseBackpropPopulation.run).parameters["on_generation"].default is None
