"""Survival experiments and datasets in the DB (no schema change).

* ``DatabaseBackpropPopulation`` records a non-default output activation and a
  describable criterion in ``config_json["training"]``; classification rows are
  unchanged.
* ``load_neat_config`` + ``resume_from_db`` rebuild the identity output and the
  Cox criterion from that record.
* A survival dataset stores ``y_data`` as [N, 2] in the existing column, tagged
  ``task_type: "survival"``; splits are stratified by event.
"""

import copy

import numpy as np
import pytest

from tests.test_core.test_neuralneat_mask_phantom import _data, config  # noqa: F401


def _surv_y(xs, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(xs)
    T = rng.exponential(1.0 / np.exp(x[:, 0]))
    C = rng.exponential(2.0, size=len(T))
    return np.column_stack([np.minimum(T, C) + 1e-3, (T <= C).astype(float)])


def test_records_output_activation_and_criterion(config, tmp_path):
    from explaneat.core.neuralneat import get_output_activation, set_output_activation
    from explaneat.db import db
    from explaneat.db.models import Experiment
    from explaneat.db.population import DatabaseBackpropPopulation
    from explaneat.db.serialization import serialize_population_config
    from explaneat.survival import CoxCriterion

    url = f"sqlite:///{tmp_path / 'surv.db'}"
    db.init_db(url)
    db.create_all()
    try:
        xs, _ = _data()
        cfg = copy.deepcopy(config)
        set_output_activation(cfg, "identity")
        pop = DatabaseBackpropPopulation(cfg, xs, _surv_y(xs), experiment_name="s",
                                         database_url=url, criterion=CoxCriterion("efron"))
        assert isinstance(pop.criterion, CoxCriterion)
        with db.session_scope() as s:
            exp = s.get(Experiment, pop.experiment_id)
            cj = dict(exp.config_json)
        assert cj["training"] == {"mask_phantom": True, "output_activation": "identity",
                                  "criterion": {"name": "cox_ph", "ties": "efron"}}
        # (load_neat_config itself is covered in test_engine_parity; this
        # fixture's partial_direct config does not round-trip through it)
        from explaneat.core.config_utils import apply_training_settings
        loaded = apply_training_settings(copy.deepcopy(config), cj)
        assert get_output_activation(loaded) == "identity"
        cj.pop("training")
        assert cj == serialize_population_config(cfg)

        # resume rebuilds the activation and the criterion from the record
        pop._save_genomes(pop._save_population_state(0))
        fresh = copy.deepcopy(config)  # caller passes a plain config
        resumed = DatabaseBackpropPopulation.resume_from_db(
            str(pop.experiment_id), fresh, xs, _surv_y(xs), database_url=url)
        assert get_output_activation(fresh) == "identity"
        assert isinstance(resumed.criterion, CoxCriterion)
    finally:
        db.close()


def test_classification_row_unchanged(config, tmp_path):
    from explaneat.db import db
    from explaneat.db.models import Experiment
    from explaneat.db.population import DatabaseBackpropPopulation

    url = f"sqlite:///{tmp_path / 'cls.db'}"
    db.init_db(url)
    db.create_all()
    try:
        xs, ys = _data()
        pop = DatabaseBackpropPopulation(config, xs, ys, experiment_name="c", database_url=url)
        with db.session_scope() as s:
            cj = dict(s.get(Experiment, pop.experiment_id).config_json)
        assert cj["training"] == {"mask_phantom": True}
        assert not hasattr(config, "output_activation")
    finally:
        db.close()


def test_survival_dataset_roundtrip_and_stratified_split(tmp_path):
    from explaneat.db import db
    from explaneat.db.models import Dataset, DatasetSplit
    from explaneat.survival.data import (
        is_survival_dataset,
        save_survival_dataset_to_db,
        save_survival_split_to_db,
        survival_split_indices,
    )

    url = f"sqlite:///{tmp_path / 'ds.db'}"
    db.init_db(url)
    db.create_all()
    try:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 3))
        y = _surv_y(X)
        ds = save_survival_dataset_to_db("sim", X, y, feature_names=["a", "b", "c"],
                                         time_unit="years")
        splits = survival_split_indices(y[:, 1], test_size=0.25, val_size=0.2, seed=1)
        sp = save_survival_split_to_db(ds.id, splits, seed=1, test_size=0.25, X=X)
        with db.session_scope() as s:
            row = s.get(Dataset, ds.id)
            X2, y2 = row.get_data()
            assert is_survival_dataset(row)
            assert row.num_classes is None
            assert row.additional_metadata["time_unit"] == "years"
            split = s.get(DatasetSplit, sp.id)
            tr, te, va = split.train_indices, split.test_indices, split.validation_indices
            assert split.stratify is True
        np.testing.assert_array_equal(X2, X)
        np.testing.assert_array_equal(y2, y)
        assert y2.shape == (400, 2)
        assert not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(te) & set(va))
        assert len(tr) + len(te) + len(va) == 400
        rate = y[:, 1].mean()
        for idx in (tr, te, va):
            assert abs(y[idx, 1].mean() - rate) < 0.03
    finally:
        db.close()


def test_survival_target_validation():
    from explaneat.survival.data import make_survival_targets

    with pytest.raises(ValueError):
        make_survival_targets([1.0, -1.0], [1, 0])
    with pytest.raises(ValueError):
        make_survival_targets([1.0, 2.0], [1, 3])
    assert make_survival_targets([1.0, 2.0], [True, False]).shape == (2, 2)
