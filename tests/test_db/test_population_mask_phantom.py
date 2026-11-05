"""DatabaseBackpropPopulation threads and records ``mask_phantom`` (default True since 2026-11-05)."""

import json

import pytest

from tests.test_core.test_neuralneat_mask_phantom import _data, config  # noqa: F401


class TestDatabasePopulationMaskPhantom:
    @pytest.mark.parametrize("mask", [False, True])
    def test_database_population_records_flag(self, config, mask, tmp_path):
        from explaneat.db import db
        from explaneat.db.models import Experiment
        from explaneat.db.population import DatabaseBackpropPopulation
        from explaneat.db.serialization import serialize_population_config

        url = f"sqlite:///{tmp_path / 'mask.db'}"
        db.init_db(url)
        db.create_all()
        try:
            xs, ys = _data()
            kwargs = {"mask_phantom": mask}
            pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name="t",
                                             database_url=url, **kwargs)
            assert pop.mask_phantom is mask
            with db.session_scope() as s:
                exp = s.get(Experiment, pop.experiment_id)
                cj = dict(exp.config_json)
            if mask:
                assert cj["training"] == {"mask_phantom": True}
                cj.pop("training")
            else:
                assert "training" not in cj
            # Everything else is the unchanged serialized NEAT config.
            assert cj == serialize_population_config(config)
        finally:
            db.close()


def test_default_is_masked(config, tmp_path):
    from explaneat.db import db
    from explaneat.db.population import DatabaseBackpropPopulation
    from explaneat.core.backproppop import BackpropPopulation
    import inspect
    assert inspect.signature(BackpropPopulation.__init__).parameters["mask_phantom"].default is True
    url = f"sqlite:///{tmp_path / 'default.db'}"
    db.init_db(url); db.create_all()
    try:
        xs, ys = _data()
        pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name="d", database_url=url)
        assert pop.mask_phantom is True
    finally:
        db.close()


@pytest.mark.parametrize("stored", [None, False, True])
def test_resume_reads_stored_flag(config, tmp_path, stored):
    """Resume keeps the setting the experiment was created with; rows without
    a training.mask_phantom key predate the mask and resume unmasked."""
    from explaneat.db import db
    from explaneat.db.models import Experiment
    from explaneat.db.population import DatabaseBackpropPopulation
    from sqlalchemy.orm.attributes import flag_modified
    url = f"sqlite:///{tmp_path / 'resume.db'}"
    db.init_db(url); db.create_all()
    try:
        xs, ys = _data()
        pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name="r", database_url=url,
                                         mask_phantom=bool(stored))
        pop._save_genomes(pop._save_population_state(0))
        with db.session_scope() as s:
            exp = s.get(Experiment, pop.experiment_id)
            cj = dict(exp.config_json)
            cj.pop("training", None)
            if stored is not None:
                cj["training"] = {"mask_phantom": stored}
            exp.config_json = cj
            flag_modified(exp, "config_json")
        resumed = DatabaseBackpropPopulation.resume_from_db(
            str(pop.experiment_id), config, xs, ys, database_url=url)
        assert resumed.mask_phantom is bool(stored)
    finally:
        db.close()
