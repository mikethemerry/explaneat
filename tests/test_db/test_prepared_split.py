"""Tests for prepared-dataset split resolution.

Regression tests for the bug where auto-preparing (one-hot encoding) a dataset
during experiment creation stamped the encoded-data scaler onto the *raw*
dataset's split. That produced a split whose ``dataset_id`` pointed at the raw
N-feature dataset but whose scaler had the encoded M-feature dimension, which
crashed the evidence pipeline with a broadcast error (M != N).

The invariant: a split's scaler dimension must match its dataset's feature
count. When training auto-prepares, the split (and its scaler) must belong to
the prepared dataset, not the raw source dataset.
"""
import numpy as np
import pytest

from explaneat.db import Dataset, DatasetSplit
from explaneat.db.dataset_utils import get_or_create_prepared_split


def _make_dataset(session, name, n_features, n_rows=50, source_id=None):
    ds = Dataset(
        name=name,
        version="1.0",
        source="test",
        num_samples=n_rows,
        num_features=n_features,
        num_classes=2,
        feature_names=[f"f{i}" for i in range(n_features)],
        target_name="target",
        source_dataset_id=source_id,
    )
    rng = np.random.default_rng(0)
    ds.set_data(rng.uniform(size=(n_rows, n_features)), rng.integers(0, 2, n_rows))
    session.add(ds)
    session.flush()
    return ds


def _make_split(session, dataset_id, train_idx, test_idx):
    sp = DatasetSplit(
        dataset_id=dataset_id,
        split_type="train_test",
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=False,
        train_indices=train_idx,
        test_indices=test_idx,
        train_size=len(train_idx),
        test_size_actual=len(test_idx),
    )
    session.add(sp)
    session.flush()
    return sp


@pytest.mark.db
@pytest.mark.unit
class TestGetOrCreatePreparedSplit:

    def test_creates_split_on_prepared_dataset(self, db_session):
        """Returned split belongs to the prepared dataset, mirroring indices."""
        raw = _make_dataset(db_session, "raw", n_features=14)
        prepared = _make_dataset(
            db_session, "raw (prepared)", n_features=107, source_id=raw.id
        )
        raw_split = _make_split(db_session, raw.id, [0, 1, 2, 3], [4, 5])

        new_split = get_or_create_prepared_split(db_session, raw_split, prepared.id)

        assert new_split.dataset_id == prepared.id
        assert new_split.id != raw_split.id
        assert new_split.train_indices == [0, 1, 2, 3]
        assert new_split.test_indices == [4, 5]

    def test_idempotent(self, db_session):
        """Calling twice reuses the same prepared split (no duplicates)."""
        raw = _make_dataset(db_session, "raw", n_features=14)
        prepared = _make_dataset(
            db_session, "raw (prepared)", n_features=107, source_id=raw.id
        )
        raw_split = _make_split(db_session, raw.id, [0, 1, 2, 3], [4, 5])

        first = get_or_create_prepared_split(db_session, raw_split, prepared.id)
        first_id = first.id
        second = get_or_create_prepared_split(db_session, raw_split, prepared.id)

        assert second.id == first_id
        count = (
            db_session.query(DatasetSplit)
            .filter_by(dataset_id=prepared.id)
            .count()
        )
        assert count == 1

    def test_returns_original_when_already_on_prepared(self, db_session):
        """If the split already belongs to the target dataset, return it."""
        prepared = _make_dataset(db_session, "prepared", n_features=107)
        split = _make_split(db_session, prepared.id, [0, 1], [2, 3])

        result = get_or_create_prepared_split(db_session, split, prepared.id)

        assert result.id == split.id


@pytest.mark.db
@pytest.mark.unit
class TestLoadSplitDataScalerGuard:
    """The evidence loader must reject a scaler/dataset dimension mismatch
    with a clear error instead of a cryptic numpy broadcast crash."""

    def test_dimension_mismatch_raises_clear_error(self, db_session):
        from fastapi import HTTPException
        from explaneat.api.routes.evidence import _load_split_data

        raw = _make_dataset(db_session, "raw", n_features=14, n_rows=20)
        split = _make_split(db_session, raw.id, list(range(16)), list(range(16, 20)))
        # Corrupt: 107-dim scaler on a 14-feature dataset.
        split.scaler_type = "StandardScaler"
        split.scaler_params = {"mean": [0.0] * 107, "scale": [1.0] * 107}
        db_session.flush()

        with pytest.raises(HTTPException) as exc:
            _load_split_data(db_session, str(split.id), "both", 1.0, 10)
        assert exc.value.status_code == 400
        assert "107 features" in exc.value.detail
        assert "14" in exc.value.detail

    def test_matching_dimensions_ok(self, db_session):
        from explaneat.api.routes.evidence import _load_split_data

        raw = _make_dataset(db_session, "raw", n_features=14, n_rows=20)
        split = _make_split(db_session, raw.id, list(range(16)), list(range(16, 20)))
        split.scaler_type = "StandardScaler"
        split.scaler_params = {"mean": [0.0] * 14, "scale": [1.0] * 14}
        db_session.flush()

        X, y, feats, classes, ncls = _load_split_data(
            db_session, str(split.id), "both", 1.0, 10
        )
        assert X.shape[1] == 14
