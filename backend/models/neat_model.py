from backend import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON


class NEATModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    dataset = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    parsed_model = db.Column(JSON, nullable=False)
    raw_data = db.Column(db.Text, nullable=False)

    def __init__(self, model_name, dataset, version, raw_data, parsed_model):
        self.model_name = model_name
        self.dataset = dataset
        self.version = version
        self.raw_data = raw_data
        self.parsed_model = parsed_model

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parsed_model": self.parsed_model,
            "raw_data": self.raw_data,
        }
