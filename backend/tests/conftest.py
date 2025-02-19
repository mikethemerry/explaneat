import pytest
from backend import create_app, db
from backend.models.dataset import Dataset, DatasetColumn, DatasetSplit
import numpy as np
from backend.models.neat_model import NEATModel


@pytest.fixture
def app():
    app = create_app()
    app.config.update(
        {"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"}
    )
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db_session(app):
    with app.app_context():
        db.create_all()
        yield db.session
        db.session.remove()
        db.drop_all()


@pytest.fixture
def sample_dataset(db_session):
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
    return dataset


@pytest.fixture
def sample_model(db_session):
    model = NEATModel(
        model_name="Test Model",
        dataset="Test Dataset",
        version="1.0",
        raw_data=b"Sample raw data",
        parsed_model={"sample": "data"},
    )
    db_session.add(model)
    db_session.commit()
    return model
