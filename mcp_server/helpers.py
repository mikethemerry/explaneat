"""Shared helper functions for MCP server tools.

These mirror the helper functions from FastAPI routes but are decoupled from
request/response types, raising plain exceptions instead of HTTPException.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from explaneat.core.config_utils import load_neat_config
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import ModelStateEngine
from explaneat.db.dataset_utils import sample_dataset
from explaneat.db.models import Dataset, DatasetSplit, Explanation, Genome


def _to_uuid(value: str) -> uuid.UUID:
    """Convert a string to a UUID, raising ValueError on failure."""
    return uuid.UUID(value)


def load_genome_and_config(session, genome_id: str):
    """Load genome DB record, NEAT genome, and config.

    Returns:
        Tuple of (neat_genome, config, genome_db)

    Raises:
        ValueError: If genome not found or experiment has no config.
    """
    genome_db = session.query(Genome).filter_by(id=_to_uuid(genome_id)).first()
    if not genome_db:
        raise ValueError(f"Genome not found: {genome_id}")

    experiment = genome_db.population.experiment
    config = load_neat_config(
        experiment.neat_config_text or "",
        experiment.config_json,
    )
    neat_genome = genome_db.to_neat_genome(config)
    return neat_genome, config, genome_db


def build_engine(session, genome_id: str) -> ModelStateEngine:
    """Build a ModelStateEngine for a genome with all operations replayed.

    Returns the engine (with .current_state and .annotations available).
    """
    neat_genome, config, genome_db = load_genome_and_config(session, genome_id)
    explainer = ExplaNEAT(neat_genome, config)
    phenotype = explainer.get_phenotype_network()

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == _to_uuid(genome_id))
        .first()
    )

    engine = ModelStateEngine(phenotype)
    if explanation and explanation.operations:
        engine.load_operations({"operations": explanation.operations})
    return engine


def build_model_state(session, genome_id: str) -> NetworkStructure:
    """Build the current annotated model by replaying operations on the phenotype."""
    return build_engine(session, genome_id).current_state


def find_annotation_in_operations(
    session, genome_id: str, annotation_id: str
) -> Dict[str, Any]:
    """Find an annotation from the explanation's operations array.

    Annotations are derived from 'annotate' operations and have synthetic IDs
    like 'ann_37' (from the operation sequence number).

    Raises:
        ValueError: If genome, explanation, or annotation not found.
    """
    genome_db = session.query(Genome).filter_by(id=_to_uuid(genome_id)).first()
    if not genome_db:
        raise ValueError(f"Genome not found: {genome_id}")

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == _to_uuid(genome_id))
        .first()
    )
    if not explanation or not explanation.operations:
        raise ValueError("No explanation/operations found")

    for op in explanation.operations:
        if op.get("type") != "annotate":
            continue
        params = op.get("params", {})
        result = op.get("result", {})
        ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"

        if ann_id == annotation_id or params.get("name") == annotation_id:
            return {
                "id": ann_id,
                "entry_nodes": params.get("entry_nodes", []),
                "exit_nodes": params.get("exit_nodes", []),
                "subgraph_nodes": params.get("subgraph_nodes", []),
                "subgraph_connections": params.get("subgraph_connections", []),
                "name": params.get("name"),
                "hypothesis": params.get("hypothesis"),
                "evidence": params.get("evidence") or {},
                "child_annotation_ids": params.get("child_annotation_ids", []),
                "parent_annotation_id": params.get("parent_annotation_id"),
            }

    raise ValueError(f"Annotation '{annotation_id}' not found")


def load_split_data(
    session,
    split_id: str,
    split_choice: str = "both",
    sample_fraction: float = 0.1,
    max_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Optional[List[str]], Optional[int]]:
    """Load X, y data and dataset metadata from a dataset split.

    Applies the scaler stored on the split (if any) so that evaluation
    uses the same preprocessing as training.

    Returns:
        Tuple of (X, y, feature_names, class_names, num_classes)

    Raises:
        ValueError: If split or dataset not found, or data is unavailable.
    """
    split = session.query(DatasetSplit).filter_by(id=_to_uuid(split_id)).first()
    if not split:
        raise ValueError(f"Dataset split not found: {split_id}")

    dataset = session.query(Dataset).filter_by(id=split.dataset_id).first()
    if not dataset:
        raise ValueError("Dataset not found")

    data = dataset.get_data()
    if data is None:
        raise ValueError("Dataset has no stored data")

    X_full, y_full = data

    if split_choice == "train":
        indices = split.train_indices
    elif split_choice == "test":
        indices = split.test_indices
    elif split_choice == "val":
        indices = split.validation_indices
        if not indices:
            raise ValueError("No validation split available")
    else:  # both
        indices = (split.train_indices or []) + (split.test_indices or [])
        if split.validation_indices:
            indices += split.validation_indices

    if not indices:
        raise ValueError("No indices for requested split")

    X = X_full[indices]
    y = y_full[indices]

    # Apply scaler if the split was trained with one
    if split.scaler_type and split.scaler_params:
        if split.scaler_type == "StandardScaler":
            mean = np.array(split.scaler_params["mean"])
            scale = np.array(split.scaler_params["scale"])
            X = (X - mean) / scale
        elif split.scaler_type == "MinMaxScaler":
            data_min = np.array(split.scaler_params["data_min_"])
            scale = np.array(split.scaler_params["scale_"])
            X = X * scale + np.array(split.scaler_params["min_"])

    # Sample for visualization
    X, y = sample_dataset(X, y, fraction=sample_fraction, max_samples=max_samples)

    return X, y, dataset.feature_names, dataset.class_names, dataset.num_classes


def serialize_network(ns: NetworkStructure) -> Dict[str, Any]:
    """Convert a NetworkStructure to a JSON-serializable dict.

    Returns a dict with keys: nodes, connections, input_node_ids, output_node_ids.
    """
    nodes = []
    for node in ns.nodes:
        node_dict: Dict[str, Any] = {
            "id": node.id,
            "type": node.type.value,
            "bias": node.bias,
            "activation": node.activation,
            "response": node.response,
            "aggregation": node.aggregation,
        }
        if node.display_name:
            node_dict["display_name"] = node.display_name
        nodes.append(node_dict)

    connections = []
    for conn in ns.connections:
        conn_dict: Dict[str, Any] = {
            "from_node": conn.from_node,
            "to_node": conn.to_node,
            "weight": conn.weight,
            "enabled": conn.enabled,
        }
        if conn.innovation is not None:
            conn_dict["innovation"] = conn.innovation
        connections.append(conn_dict)

    return {
        "nodes": nodes,
        "connections": connections,
        "input_node_ids": ns.input_node_ids,
        "output_node_ids": ns.output_node_ids,
    }
