from flask import Blueprint, jsonify
from backend.models.dataset import Dataset

bp = Blueprint("dataset", __name__)


@bp.route("/api/datasets", methods=["GET"])
def get_datasets():
    """Get list of all datasets"""
    datasets = Dataset.query.all()
    return jsonify(
        [
            {
                "id": d.id,
                "name": d.name,
                "source": d.source,
                "n_samples": d.n_samples,
                "n_features": d.n_features,
                "task_type": d.task_type,
                "n_classes": d.n_classes,
            }
            for d in datasets
        ]
    )


@bp.route("/api/datasets/<int:dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    """Get detailed information about a specific dataset"""
    dataset = Dataset.query.get_or_404(dataset_id)
    return jsonify(dataset.to_dict())
