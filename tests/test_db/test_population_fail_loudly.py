"""DatabaseBackpropPopulation fails loudly when evaluation raises (2026-11-06).

``_evaluate_population`` used to catch every exception and set every genome's
fitness to -1000, so a Postgres outage mid-run silently turned evolution into a
search on a flat landscape (churn100mask incident, RUN_INCIDENTS.md). It now logs
and re-raises; ``run`` marks the experiment failed and the run can be continued
with ``resume_from_db``. The per-genome None/NaN -> -1000 guard stays, with a
logged warning.
"""

import logging

import pytest

from tests.test_core.test_neuralneat_mask_phantom import _data, config  # noqa: F401


class _Boom(RuntimeError):
    pass


def _raising_fitness(population, config, xs, ys, device):
    raise _Boom("database went away")


def test_fitness_exception_propagates_and_marks_failed(config, tmp_path):
    from explaneat.db import db
    from explaneat.db.models import Experiment
    from explaneat.db.population import DatabaseBackpropPopulation

    url = f"sqlite:///{tmp_path / 'boom.db'}"
    db.init_db(url)
    db.create_all()
    try:
        xs, ys = _data()
        pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name="boom",
                                         database_url=url)
        with pytest.raises(_Boom):
            pop.run(_raising_fitness, n=2, nEpochs=1)
        # No genome was given the old silent fallback fitness.
        assert all(g.fitness != -1000.0 for g in pop.population.values())
        with db.session_scope() as s:
            assert s.get(Experiment, pop.experiment_id).status == "failed"
    finally:
        db.close()


def test_evaluate_population_reraises(config, tmp_path):
    from explaneat.db import db
    from explaneat.db.population import DatabaseBackpropPopulation

    url = f"sqlite:///{tmp_path / 'eval.db'}"
    db.init_db(url)
    db.create_all()
    try:
        xs, ys = _data()
        pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name="eval",
                                         database_url=url)
        with pytest.raises(_Boom):
            pop._evaluate_population(_raising_fitness, nEpochs=1)
    finally:
        db.close()


def test_nan_fitness_guard_still_applies_and_warns(config, tmp_path, caplog):
    from explaneat.db import db
    from explaneat.db.population import DatabaseBackpropPopulation

    def nan_for_some(population, config, xs, ys, device):
        for i, g in enumerate(population.values()):
            g.fitness = float("nan") if i % 2 == 0 else 0.5

    url = f"sqlite:///{tmp_path / 'nan.db'}"
    db.init_db(url)
    db.create_all()
    try:
        xs, ys = _data()
        pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name="nan",
                                         database_url=url)
        n_nan = (len(pop.population) + 1) // 2
        with caplog.at_level(logging.WARNING, logger="explaneat.db.population"):
            pop.run(nan_for_some, n=1, nEpochs=1)
        msgs = [r.getMessage() for r in caplog.records
                if "None/NaN fitness" in r.getMessage()]
        assert msgs and msgs[0].strip().startswith(f"{n_nan}/")
    finally:
        db.close()
