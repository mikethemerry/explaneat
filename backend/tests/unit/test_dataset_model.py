import pytest
import numpy as np
from backend.models.dataset import Dataset, DatasetColumn, DatasetSplit


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
        features_blob=np.random.rand(100, 4).tobytes(),
        targets_blob=np.random.rand(100, 1).tobytes(),
        data_format="numpy",
        n_classes=2,
    )
    db_session.add(dataset)
    db_session.commit()

    assert dataset.id is not None
    assert dataset.name == "Test Dataset"
    assert len(dataset.feature_types) == 4


def test_dataset_relationships(db_session):
    dataset = Dataset(
        name="Test Dataset",
        source="UCI",
        version="1.0",
        n_samples=100,
        n_features=4,
        feature_types=["numeric", "numeric", "categorical", "numeric"],
        task_type="classification",
        data_url="http://example.com/dataset",
        features_blob=np.random.rand(100, 4).tobytes(),
        targets_blob=np.random.rand(100, 1).tobytes(),
        data_format="numpy",
        n_classes=2,
    )

    # Add columns
    for i in range(4):
        column = DatasetColumn(
            name=f"feature_{i}",
            data_type="numeric",
            ordinal_position=i,
            is_target=False,
        )
        dataset.columns.append(column)

    # Add target column
    target_column = DatasetColumn(
        name="target", data_type="numeric", ordinal_position=4, is_target=True
    )
    dataset.columns.append(target_column)

    # Add splits
    train_split = DatasetSplit(
        split_type="train",
        seed=42,
        features_blob=np.random.rand(80, 4).tobytes(),
        targets_blob=np.random.rand(80, 1).tobytes(),
    )
    test_split = DatasetSplit(
        split_type="test",
        seed=42,
        features_blob=np.random.rand(20, 4).tobytes(),
        targets_blob=np.random.rand(20, 1).tobytes(),
    )
    dataset.splits.extend([train_split, test_split])

    db_session.add(dataset)
    db_session.commit()

    # Test relationships
    assert len(dataset.columns) == 5  # 4 features + 1 target
    assert len(dataset.splits) == 2
    assert dataset.columns[-1].is_target == True
    assert dataset.splits[0].split_type == "train"
    assert dataset.splits[1].split_type == "test"


def test_dataset_to_dict(db_session):
    dataset = Dataset(
        name="Test Dataset",
        source="UCI",
        version="1.0",
        n_samples=100,
        n_features=4,
        feature_types=["numeric", "numeric", "categorical", "numeric"],
        task_type="classification",
        data_url="http://example.com/dataset",
        features_blob=np.random.rand(100, 4).tobytes(),
        targets_blob=np.random.rand(100, 1).tobytes(),
        data_format="numpy",
        n_classes=2,
    )

    db_session.add(dataset)
    db_session.commit()

    dataset_dict = dataset.to_dict()
    assert dataset_dict["name"] == "Test Dataset"
    assert dataset_dict["source"] == "UCI"
    assert dataset_dict["n_features"] == 4
    assert "created_at" in dataset_dict
    assert "updated_at" in dataset_dict
    assert "columns" in dataset_dict
    assert "splits" in dataset_dict
