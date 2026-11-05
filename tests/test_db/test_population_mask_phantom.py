"""DatabaseBackpropPopulation threads and records the opt-in ``mask_phantom``."""

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
            kwargs = {"mask_phantom": True} if mask else {}
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
