from backend import db
from datetime import datetime, UTC
from sqlalchemy.dialects.postgresql import JSON


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    source = db.Column(db.String(50), nullable=False)  # UCI or PMLB
    version = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Dataset characteristics
    n_samples = db.Column(db.Integer, nullable=False)
    n_features = db.Column(db.Integer, nullable=False)
    n_classes = db.Column(db.Integer)  # Nullable for regression tasks
    feature_types = db.Column(JSON, nullable=False)  # List of feature data types
    task_type = db.Column(db.String(50), nullable=False)  # Classification/Regression

    # Metadata
    description = db.Column(db.Text)
    paper_url = db.Column(db.String(500))
    data_url = db.Column(db.String(500), nullable=False)
    missing_values = db.Column(db.Boolean, default=False)

    # Data storage
    data_blob = db.Column(db.LargeBinary, nullable=False)  # Compressed data array
    data_format = db.Column(
        db.String(50), nullable=False
    )  # Format of stored data (e.g. 'numpy', 'pandas')

    # Relationships
    columns = db.relationship("DatasetColumn", backref="dataset", lazy=True)
    splits = db.relationship("DatasetSplit", backref="dataset", lazy=True)

    def __init__(
        self,
        name,
        source,
        version,
        n_samples,
        n_features,
        feature_types,
        task_type,
        data_url,
        data_blob,
        data_format,
        n_classes=None,
        description=None,
        paper_url=None,
        missing_values=False,
    ):
        self.name = name
        self.source = source
        self.version = version
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.feature_types = feature_types
        self.task_type = task_type
        self.description = description
        self.paper_url = paper_url
        self.data_url = data_url
        self.missing_values = missing_values
        self.data_blob = data_blob
        self.data_format = data_format

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_types": self.feature_types,
            "task_type": self.task_type,
            "description": self.description,
            "paper_url": self.paper_url,
            "data_url": self.data_url,
            "missing_values": self.missing_values,
            "data_format": self.data_format,
            "columns": [column.to_dict() for column in self.columns],
            "splits": [split.to_dict() for split in self.splits],
        }


class DatasetColumn(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    data_type = db.Column(
        db.String(50), nullable=False
    )  # numeric, categorical, datetime, etc
    description = db.Column(db.Text)
    units = db.Column(db.String(50))  # Units of measurement if applicable
    min_value = db.Column(db.Float)  # For numeric columns
    max_value = db.Column(db.Float)  # For numeric columns
    categories = db.Column(JSON)  # List of possible values for categorical
    missing_count = db.Column(db.Integer)
    unique_count = db.Column(db.Integer)
    mean = db.Column(db.Float)  # For numeric columns
    std = db.Column(db.Float)  # For numeric columns
    median = db.Column(db.Float)  # For numeric columns
    is_target = db.Column(db.Boolean, default=False)
    ordinal_position = db.Column(
        db.Integer, nullable=False
    )  # Column position in dataset

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "description": self.description,
            "units": self.units,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "categories": self.categories,
            "missing_count": self.missing_count,
            "unique_count": self.unique_count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "is_target": self.is_target,
            "ordinal_position": self.ordinal_position,
        }


class DatasetSplit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id"), nullable=False)
    split_type = db.Column(db.String(50), nullable=False)  # 'train' or 'test'
    seed = db.Column(db.Integer, nullable=False)
    data_blob = db.Column(db.LargeBinary, nullable=False)  # Compressed data array
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "split_type": self.split_type,
            "seed": self.seed,
            "created_at": self.created_at.isoformat(),
        }
