"""MCP tools for dataset discovery and metadata."""

import json
import uuid

from mcp.server.fastmcp import FastMCP

from ..server import get_db


def list_datasets() -> str:
    """List all datasets available in the database.

    Returns a list of datasets with their metadata: id, name, source,
    num_samples, num_features, num_classes, feature_names, target_name,
    class_names, task_type, and description.
    """
    from explaneat.db.models import Dataset

    db = get_db()
    with db.session_scope() as session:
        datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).all()
        results = []
        for d in datasets:
            meta = d.additional_metadata or {}
            results.append({
                "id": str(d.id),
                "name": d.name,
                "source": d.source,
                "num_samples": d.num_samples,
                "num_features": d.num_features,
                "num_classes": d.num_classes,
                "feature_names": d.feature_names,
                "target_name": d.target_name,
                "class_names": d.class_names,
                "task_type": meta.get("task_type"),
                "description": d.description,
            })
        return json.dumps({"datasets": results, "total": len(results)}, indent=2, default=str)


def get_dataset(dataset_id: str) -> str:
    """Get full metadata for a specific dataset.

    Returns all dataset fields including version, source_url, feature_types,
    encoding_config, and additional_metadata.

    Args:
        dataset_id: UUID of the dataset.
    """
    from explaneat.db.models import Dataset

    db = get_db()
    with db.session_scope() as session:
        dataset = session.query(Dataset).filter_by(id=uuid.UUID(dataset_id)).first()
        if not dataset:
            return json.dumps({"error": f"Dataset not found: {dataset_id}"})

        d = dataset.to_dict()
        meta = d.pop("additional_metadata", None) or {}
        d["task_type"] = meta.get("task_type")
        d["additional_metadata"] = meta
        return json.dumps(d, indent=2, default=str)


def get_dataset_splits(dataset_id: str) -> str:
    """List all splits for a dataset.

    Returns split metadata including id, name, split_type, train/test sizes,
    and whether a scaler is attached.

    Args:
        dataset_id: UUID of the dataset.
    """
    from explaneat.db.models import DatasetSplit

    db = get_db()
    with db.session_scope() as session:
        splits = (
            session.query(DatasetSplit)
            .filter_by(dataset_id=uuid.UUID(dataset_id))
            .all()
        )
        results = []
        for s in splits:
            results.append({
                "id": str(s.id),
                "name": s.name,
                "split_type": s.split_type,
                "train_size": s.train_size,
                "test_size": s.test_size,
                "test_size_actual": s.test_size_actual,
                "random_state": s.random_state,
                "has_scaler": bool(s.scaler_type),
            })
        return json.dumps({"splits": results, "total": len(results)}, indent=2, default=str)


def register(mcp: FastMCP) -> None:
    """Register dataset tools with the MCP server."""
    mcp.tool()(list_datasets)
    mcp.tool()(get_dataset)
    mcp.tool()(get_dataset_splits)
