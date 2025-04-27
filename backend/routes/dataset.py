from flask import Blueprint, jsonify, request
from backend.models.dataset import Dataset
import numpy as np

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


@bp.route("/api/datasets/<int:dataset_id>/data", methods=["GET"])
def get_dataset_data(dataset_id):
    """Get dataset data with optional split"""
    split_type = request.args.get('split_type', None)
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if split_type:
        split = next((s for s in dataset.splits if s.split_type == split_type), None)
        if not split:
            return jsonify({"error": f"Split {split_type} not found"}), 404
        X = np.frombuffer(split.features_blob).reshape(-1, dataset.n_features)
        y = np.frombuffer(split.targets_blob).reshape(-1, 1)
    else:
        X = np.frombuffer(dataset.features_blob).reshape(-1, dataset.n_features)
        y = np.frombuffer(dataset.targets_blob).reshape(-1, 1)
    
    feature_columns = [col.name for col in dataset.columns if not col.is_target]
    target_column = next(col.name for col in dataset.columns if col.is_target)
    
    # Convert to list for JSON serialization
    data = {
        "features": X.tolist(),
        "targets": y.tolist(),
        "feature_columns": feature_columns,
        "target_column": target_column
    }
    return jsonify(data)
