import pytest
import numpy as np
from backend.models.dataset import Dataset, DatasetColumn


def test_dataset_creation(db_session):
    dataset = Dataset(
        name="Test Dataset",
        source="UCI",
        version="1.0",
        n_samples=100,
        n_features=4,
        feature_types=["numeric", "numeric", "categorical", "numeric"],
        task_type="classification",
        data_url="http://example.com/dataset",
        data_blob=np.random.rand(100, 4).tobytes(),
        data_format="numpy",
        n_classes=2,
    )
    db_session.add(dataset)
    db_session.commit()

    assert dataset.id is not None
    assert dataset.name == "Test Dataset"
    assert len(dataset.feature_types) == 4


def test_dataset_to_dict(sample_dataset):
    dataset_dict = sample_dataset.to_dict()
    assert dataset_dict["name"] == "Test Dataset"
    assert dataset_dict["source"] == "UCI"
    assert "created_at" in dataset_dict
    assert "updated_at" in dataset_dict


def test_dataset_relationships(db_session, sample_dataset):
    column = DatasetColumn(
        dataset_id=sample_dataset.id,
        name="feature1",
        data_type="numeric",
        ordinal_position=0,
    )
    db_session.add(column)
    db_session.commit()

    assert len(sample_dataset.columns) == 1
    assert sample_dataset.columns[0].name == "feature1"
