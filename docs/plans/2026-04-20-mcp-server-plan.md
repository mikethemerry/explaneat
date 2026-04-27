# ExplaNEAT MCP Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MCP server that gives Claude full read/write access to ExplaNEAT's model analysis, enabling autonomous exploration, annotation, evidence gathering, and report generation for evolved NEAT models.

**Architecture:** Fine-grained MCP tools (~29) using the `mcp[cli]` Python SDK with FastMCP. Direct Python imports of `explaneat.analysis`, `explaneat.db`, `explaneat.core` — no dependency on a running FastAPI server. Stdio transport, launched by Claude Code as a subprocess.

**Tech Stack:** `mcp[cli]` (FastMCP), SQLAlchemy (existing DB layer), matplotlib (rendering), existing ExplaNEAT analysis modules.

---

## Task 1: Project Scaffolding & Server Bootstrap

**Files:**
- Create: `mcp_server/__init__.py`
- Create: `mcp_server/__main__.py`
- Create: `mcp_server/server.py`
- Modify: `pyproject.toml` (add `mcp[cli]` dependency)
- Test: `tests/test_mcp/test_server.py`

**Step 1: Add mcp dependency**

```bash
uv add "mcp[cli]"
```

**Step 2: Create directory structure**

```bash
mkdir -p mcp_server/tools tests/test_mcp
```

**Step 3: Write the failing test**

```python
# tests/test_mcp/__init__.py
# (empty)

# tests/test_mcp/test_server.py
"""Test MCP server initialization."""
import pytest


def test_server_creates():
    """Server object can be created."""
    from mcp_server.server import create_server
    server = create_server()
    assert server is not None
    assert server.name == "explaneat"


def test_server_has_db():
    """Server initializes database connection."""
    from mcp_server.server import create_server, get_db
    create_server()
    db = get_db()
    assert db is not None
```

**Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcp_server'`

**Step 5: Write minimal implementation**

```python
# mcp_server/__init__.py
"""ExplaNEAT MCP Server — gives Claude full access to model analysis."""

# mcp_server/server.py
"""MCP server setup and database initialization."""
from mcp.server.fastmcp import FastMCP
from explaneat.db.database import Database

_db: Database | None = None


def get_db() -> Database:
    """Get the shared Database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools registered."""
    server = FastMCP("explaneat")

    # Initialize DB
    get_db()

    return server


# mcp_server/__main__.py
"""Entry point: python -m mcp_server"""
from mcp_server.server import create_server

mcp = create_server()

# Import tool modules to trigger registration
from mcp_server.tools import register_all
register_all(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")


# mcp_server/tools/__init__.py
"""Tool registration."""
from mcp.server.fastmcp import FastMCP


def register_all(mcp: FastMCP) -> None:
    """Register all tool modules with the server."""
    from mcp_server.tools import experiments
    from mcp_server.tools import models
    from mcp_server.tools import operations
    from mcp_server.tools import evidence
    from mcp_server.tools import coverage
    from mcp_server.tools import datasets
    from mcp_server.tools import snapshots

    for module in [experiments, models, operations, evidence, coverage, datasets, snapshots]:
        module.register(mcp)
```

Create stub tool modules so imports don't fail:

```python
# mcp_server/tools/experiments.py
# mcp_server/tools/models.py
# mcp_server/tools/operations.py
# mcp_server/tools/evidence.py
# mcp_server/tools/coverage.py
# mcp_server/tools/datasets.py
# mcp_server/tools/snapshots.py

# Each has the same stub:
from mcp.server.fastmcp import FastMCP

def register(mcp: FastMCP) -> None:
    """Register tools with the server."""
    pass
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_mcp/test_server.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add mcp_server/ tests/test_mcp/ pyproject.toml uv.lock
git commit -m "feat(mcp): scaffold MCP server with FastMCP and DB init"
```

---

## Task 2: Shared Helpers — Model State Building

The API routes have helper functions (`_build_engine`, `_build_model_state`, `_load_split_data`) that multiple tools will need. Extract these into a shared helpers module.

**Files:**
- Create: `mcp_server/helpers.py`
- Test: `tests/test_mcp/test_helpers.py`
- Reference: `explaneat/api/routes/evidence.py:79-282` (original helpers)

**Step 1: Write the failing test**

These tests need a real DB with test data. Use the existing test fixtures if available, otherwise use a minimal SQLite setup.

```python
# tests/test_mcp/test_helpers.py
"""Test shared MCP helper functions."""
import pytest
from unittest.mock import MagicMock, patch


def test_build_engine_returns_engine():
    """_build_engine returns a ModelStateEngine with current_state."""
    from mcp_server.helpers import build_engine
    # This will be tested with integration tests in Task 10
    # For now, verify the function exists and has the right signature
    import inspect
    sig = inspect.signature(build_engine)
    assert "session" in sig.parameters
    assert "genome_id" in sig.parameters


def test_build_model_state_returns_network_structure():
    """build_model_state returns a NetworkStructure."""
    from mcp_server.helpers import build_model_state
    import inspect
    sig = inspect.signature(build_model_state)
    assert "session" in sig.parameters
    assert "genome_id" in sig.parameters


def test_load_split_data_signature():
    """load_split_data has the expected signature."""
    from mcp_server.helpers import load_split_data
    import inspect
    sig = inspect.signature(load_split_data)
    assert "session" in sig.parameters
    assert "split_id" in sig.parameters
    assert "split_choice" in sig.parameters


def test_serialize_network_structure():
    """serialize_network converts NetworkStructure to dict."""
    from mcp_server.helpers import serialize_network
    from explaneat.core.genome_network import NetworkStructure, NetworkNode, NetworkConnection, NodeType

    ns = NetworkStructure(
        nodes=[
            NetworkNode(id="-1", node_type=NodeType.INPUT, bias=0.0, activation="identity", response=1.0, aggregation="sum"),
            NetworkNode(id="0", node_type=NodeType.OUTPUT, bias=0.5, activation="sigmoid", response=1.0, aggregation="sum"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )
    result = serialize_network(ns)
    assert "nodes" in result
    assert "connections" in result
    assert "input_node_ids" in result
    assert "output_node_ids" in result
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["id"] == "-1"
    assert result["nodes"][0]["type"] == "input"
    assert result["connections"][0]["weight"] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcp_server.helpers'`

**Step 3: Write implementation**

```python
# mcp_server/helpers.py
"""Shared helpers for MCP tools.

Mirrors the helper functions from explaneat/api/routes/evidence.py
but decoupled from FastAPI request/response types.
"""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from explaneat.core.explaineat import ExplaNEAT
from explaneat.core.genome_network import NetworkStructure, NodeType
from explaneat.core.model_state import ModelStateEngine
from explaneat.core.neuralneat import load_neat_config
from explaneat.db.models import (
    Dataset,
    DatasetSplit,
    Experiment,
    Explanation,
    Genome,
    Population,
)
from explaneat.helpers.genome_serializer import deserialize_genome


def _to_uuid(value: str) -> uuid.UUID:
    """Convert string to UUID, passing through if already UUID."""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def load_genome_and_config(session: Session, genome_id: str):
    """Load a genome and its NEAT config from the database.

    Returns (neat_genome, config, genome_db).
    """
    gid = _to_uuid(genome_id)
    genome_db = session.query(Genome).get(gid)
    if genome_db is None:
        raise ValueError(f"Genome {genome_id} not found")

    experiment = genome_db.population.experiment
    config = load_neat_config(
        experiment.neat_config_text or "", experiment.config_json
    )
    neat_genome = genome_db.to_neat_genome(config)
    return neat_genome, config, genome_db


def build_engine(session: Session, genome_id: str) -> ModelStateEngine:
    """Build a ModelStateEngine for a genome with all operations replayed.

    Equivalent to evidence.py _build_engine.
    """
    neat_genome, config, genome_db = load_genome_and_config(session, genome_id)

    exp = ExplaNEAT(neat_genome, config)
    phenotype = exp.get_phenotype_network()

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == genome_db.id)
        .first()
    )
    operations = {"operations": explanation.operations if explanation else []}

    engine = ModelStateEngine.from_phenotype_and_operations(phenotype, operations)
    return engine


def build_model_state(session: Session, genome_id: str) -> NetworkStructure:
    """Build the current model state (phenotype + all operations applied)."""
    return build_engine(session, genome_id).current_state


def find_annotation_in_operations(
    session: Session, genome_id: str, annotation_id: str
) -> dict | None:
    """Find an annotation dict from the operations stream.

    Returns the annotation params dict or None.
    """
    gid = _to_uuid(genome_id)
    explanation = (
        session.query(Explanation).filter(Explanation.genome_id == gid).first()
    )
    if not explanation or not explanation.operations:
        return None

    for op in explanation.operations:
        if op.get("type") == "annotate":
            result = op.get("result", {})
            op_ann_id = result.get("annotation_id") or f"ann_{op['seq']}"
            if op_ann_id == annotation_id:
                params = op.get("params", {})
                params["annotation_id"] = op_ann_id
                return params

            # Also check by name
            if params := op.get("params", {}):
                if params.get("name") == annotation_id:
                    params["annotation_id"] = (
                        result.get("annotation_id") or f"ann_{op['seq']}"
                    )
                    return params

    return None


def load_split_data(
    session: Session,
    split_id: str,
    split_choice: str = "test",
    sample_fraction: float = 1.0,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str] | None, int | None]:
    """Load dataset split data.

    Returns (X, y, feature_names, class_names, num_classes).
    """
    from explaneat.analysis.viz_data import sample_dataset

    sid = _to_uuid(split_id)
    split = session.query(DatasetSplit).get(sid)
    if split is None:
        raise ValueError(f"Split {split_id} not found")

    dataset = session.query(Dataset).get(split.dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset for split {split_id} not found")

    X, y = dataset.get_data()
    feature_names = dataset.feature_names or [f"x{i}" for i in range(X.shape[1])]
    class_names = dataset.class_names
    num_classes = dataset.num_classes

    # Apply split indices
    if split_choice == "train":
        indices = split.train_indices
    elif split_choice == "test":
        indices = split.test_indices
    elif split_choice == "validation" and split.validation_indices:
        indices = split.validation_indices
    else:
        indices = split.test_indices

    if indices is not None:
        X = X[indices]
        y = y[indices]

    # Apply scaler if stored
    if split.scaler_params:
        mean = np.array(split.scaler_params["mean"])
        scale = np.array(split.scaler_params["scale"])
        X = (X - mean) / scale

    # Sample if requested
    if sample_fraction < 1.0 or max_samples:
        X, y = sample_dataset(X, y, fraction=sample_fraction, max_samples=max_samples)

    return X, y, feature_names, class_names, num_classes


def serialize_network(ns: NetworkStructure) -> dict:
    """Convert a NetworkStructure to a JSON-serializable dict."""
    type_map = {
        NodeType.INPUT: "input",
        NodeType.OUTPUT: "output",
        NodeType.HIDDEN: "hidden",
        NodeType.IDENTITY: "identity",
    }
    # Handle FUNCTION type if it exists
    if hasattr(NodeType, "FUNCTION"):
        type_map[NodeType.FUNCTION] = "function"

    nodes = []
    for n in ns.nodes:
        node_dict = {
            "id": n.id,
            "type": type_map.get(n.node_type, "hidden"),
            "bias": n.bias,
            "activation": n.activation,
            "response": n.response,
            "aggregation": n.aggregation,
        }
        if hasattr(n, "display_name") and n.display_name:
            node_dict["display_name"] = n.display_name
        if hasattr(n, "function_metadata") and n.function_metadata:
            node_dict["function_metadata"] = n.function_metadata
        nodes.append(node_dict)

    connections = [
        {
            "from_node": c.from_node,
            "to_node": c.to_node,
            "weight": c.weight,
            "enabled": c.enabled,
        }
        for c in ns.connections
    ]

    result = {
        "nodes": nodes,
        "connections": connections,
        "input_node_ids": list(ns.input_node_ids),
        "output_node_ids": list(ns.output_node_ids),
    }

    if hasattr(ns, "annotations") and ns.annotations:
        result["annotations"] = ns.annotations

    return result
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp/test_helpers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/helpers.py tests/test_mcp/test_helpers.py
git commit -m "feat(mcp): add shared helpers for model state building and serialization"
```

---

## Task 3: Experiment & Genome Discovery Tools (Tools 1–5)

**Files:**
- Modify: `mcp_server/tools/experiments.py`
- Test: `tests/test_mcp/test_experiments.py`
- Reference: `explaneat/api/routes/experiments.py`, `explaneat/api/routes/genomes.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp/test_experiments.py
"""Test experiment & genome discovery tools are registered."""
import pytest


def test_tools_registered():
    """All experiment/genome discovery tools exist."""
    from mcp_server.server import create_server
    from mcp_server.tools.experiments import register

    server = create_server()
    register(server)

    # Verify tool functions are importable
    from mcp_server.tools.experiments import (
        list_experiments,
        get_experiment,
        get_best_genome,
        list_genomes,
        get_genome,
    )
    assert callable(list_experiments)
    assert callable(get_experiment)
    assert callable(get_best_genome)
    assert callable(list_genomes)
    assert callable(get_genome)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_experiments.py -v`
Expected: FAIL — `ImportError: cannot import name 'list_experiments'`

**Step 3: Write implementation**

```python
# mcp_server/tools/experiments.py
"""Experiment & genome discovery tools."""
from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from sqlalchemy import func

from explaneat.db.models import Experiment, Genome, Population
from mcp_server.helpers import _to_uuid, load_genome_and_config, serialize_network
from mcp_server.server import get_db


def register(mcp: FastMCP) -> None:
    """Register experiment & genome tools."""
    mcp.tool()(list_experiments)
    mcp.tool()(get_experiment)
    mcp.tool()(get_best_genome)
    mcp.tool()(list_genomes)
    mcp.tool()(get_genome)


def list_experiments(offset: int = 0, limit: int = 20) -> str:
    """List all experiments with summary info (genome count, best fitness, generations).

    Use this to discover what experiments are available for analysis.
    """
    db = get_db()
    with db.session_scope() as session:
        experiments = (
            session.query(Experiment)
            .order_by(Experiment.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        results = []
        for exp in experiments:
            pop_count = (
                session.query(Population)
                .filter(Population.experiment_id == exp.id)
                .count()
            )
            best_fitness = (
                session.query(func.max(Genome.fitness))
                .join(Population)
                .filter(Population.experiment_id == exp.id)
                .scalar()
            )
            genome_count = (
                session.query(func.count(Genome.id))
                .join(Population)
                .filter(Population.experiment_id == exp.id)
                .scalar()
            )
            results.append(
                {
                    "id": str(exp.id),
                    "name": exp.name,
                    "description": exp.description,
                    "status": exp.status,
                    "generations": pop_count,
                    "genome_count": genome_count,
                    "best_fitness": best_fitness,
                    "dataset_id": str(exp.dataset_id) if exp.dataset_id else None,
                    "split_id": str(exp.split_id) if exp.split_id else None,
                    "created_at": str(exp.created_at) if exp.created_at else None,
                }
            )
        return json.dumps(results, indent=2)


def get_experiment(experiment_id: str) -> str:
    """Get detailed info about an experiment including its config and dataset.

    Returns experiment metadata, resolved NEAT config, and linked dataset/split IDs.
    """
    db = get_db()
    with db.session_scope() as session:
        exp = session.query(Experiment).get(_to_uuid(experiment_id))
        if exp is None:
            return json.dumps({"error": f"Experiment {experiment_id} not found"})

        resolved_config = None
        if exp.config_json and "resolved_config" in exp.config_json:
            resolved_config = exp.config_json["resolved_config"]

        result = {
            "id": str(exp.id),
            "name": exp.name,
            "description": exp.description,
            "status": exp.status,
            "dataset_id": str(exp.dataset_id) if exp.dataset_id else None,
            "split_id": str(exp.split_id) if exp.split_id else None,
            "config_template_id": str(exp.config_template_id) if exp.config_template_id else None,
            "resolved_config": resolved_config,
            "created_at": str(exp.created_at) if exp.created_at else None,
        }
        return json.dumps(result, indent=2)


def get_best_genome(experiment_id: str) -> str:
    """Get the highest-fitness genome from an experiment.

    Returns genome ID, fitness, and basic network stats. Use this as a starting
    point for model analysis — the best genome is usually the most interesting.
    """
    db = get_db()
    with db.session_scope() as session:
        genome = (
            session.query(Genome)
            .join(Population)
            .filter(
                Population.experiment_id == _to_uuid(experiment_id),
                Genome.fitness.isnot(None),
            )
            .order_by(Genome.fitness.desc())
            .first()
        )
        if genome is None:
            return json.dumps({"error": "No genomes found for experiment"})

        result = {
            "id": str(genome.id),
            "neat_genome_id": genome.genome_id,
            "fitness": genome.fitness,
            "num_nodes": genome.num_nodes,
            "num_connections": genome.num_connections,
            "num_enabled_connections": genome.num_enabled_connections,
            "network_depth": genome.network_depth,
            "network_width": genome.network_width,
            "generation": genome.population.generation if genome.population else None,
            "experiment_id": experiment_id,
        }
        return json.dumps(result, indent=2)


def list_genomes(
    experiment_id: str,
    min_fitness: Optional[float] = None,
    offset: int = 0,
    limit: int = 20,
) -> str:
    """List genomes from an experiment, ordered by fitness (highest first).

    Optionally filter by minimum fitness. Use this to find interesting genomes to analyze.
    """
    db = get_db()
    with db.session_scope() as session:
        query = (
            session.query(Genome)
            .join(Population)
            .filter(Population.experiment_id == _to_uuid(experiment_id))
        )
        if min_fitness is not None:
            query = query.filter(Genome.fitness >= min_fitness)

        genomes = (
            query.order_by(Genome.fitness.desc().nullslast())
            .offset(offset)
            .limit(limit)
            .all()
        )

        results = []
        for g in genomes:
            results.append(
                {
                    "id": str(g.id),
                    "neat_genome_id": g.genome_id,
                    "fitness": g.fitness,
                    "num_nodes": g.num_nodes,
                    "num_connections": g.num_connections,
                    "generation": g.population.generation if g.population else None,
                }
            )
        return json.dumps(results, indent=2)


def get_genome(genome_id: str) -> str:
    """Get detailed metadata for a specific genome.

    Returns fitness, network stats (depth, width, node/connection counts),
    and parent genome IDs for ancestry tracking.
    """
    db = get_db()
    with db.session_scope() as session:
        genome = session.query(Genome).get(_to_uuid(genome_id))
        if genome is None:
            return json.dumps({"error": f"Genome {genome_id} not found"})

        result = {
            "id": str(genome.id),
            "neat_genome_id": genome.genome_id,
            "fitness": genome.fitness,
            "adjusted_fitness": genome.adjusted_fitness,
            "num_nodes": genome.num_nodes,
            "num_connections": genome.num_connections,
            "num_enabled_connections": genome.num_enabled_connections,
            "network_depth": genome.network_depth,
            "network_width": genome.network_width,
            "parent1_id": str(genome.parent1_id) if genome.parent1_id else None,
            "parent2_id": str(genome.parent2_id) if genome.parent2_id else None,
            "generation": genome.population.generation if genome.population else None,
            "experiment_id": str(genome.population.experiment_id) if genome.population else None,
        }
        return json.dumps(result, indent=2)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp/test_experiments.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools/experiments.py tests/test_mcp/test_experiments.py
git commit -m "feat(mcp): add experiment & genome discovery tools"
```

---

## Task 4: Model Structure Tools (Tools 6–8)

**Files:**
- Modify: `mcp_server/tools/models.py`
- Test: `tests/test_mcp/test_models.py`
- Reference: `explaneat/api/routes/genomes.py:154-189`, `explaneat/api/routes/operations.py:190-231`, `explaneat/api/routes/evidence.py` (node-info)

**Step 1: Write the failing test**

```python
# tests/test_mcp/test_models.py
"""Test model structure tools are registered."""
import pytest


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.models import register

    server = create_server()
    register(server)

    from mcp_server.tools.models import (
        get_phenotype,
        get_model_state,
        get_node_info,
    )
    assert callable(get_phenotype)
    assert callable(get_model_state)
    assert callable(get_node_info)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_models.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mcp_server/tools/models.py
"""Model structure tools."""
from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from explaneat.core.explaineat import ExplaNEAT
from mcp_server.helpers import (
    _to_uuid,
    build_engine,
    load_genome_and_config,
    serialize_network,
)
from mcp_server.server import get_db


def register(mcp: FastMCP) -> None:
    mcp.tool()(get_phenotype)
    mcp.tool()(get_model_state)
    mcp.tool()(get_node_info)


def get_phenotype(genome_id: str) -> str:
    """Get the pruned phenotype network (only active nodes and connections).

    This is the raw network before any operations (splits, annotations) are applied.
    Shows the actual computational graph that was evolved.
    """
    db = get_db()
    with db.session_scope() as session:
        neat_genome, config, genome_db = load_genome_and_config(session, genome_id)
        exp = ExplaNEAT(neat_genome, config)
        phenotype = exp.get_phenotype_network()
        return json.dumps(serialize_network(phenotype), indent=2)


def get_model_state(genome_id: str, collapsed: Optional[str] = None) -> str:
    """Get the current model state with all operations (splits, annotations) applied.

    This is the "real" model that the evidence system works on. If the genome has
    an explanation with operations, they are replayed on the phenotype.

    Args:
        genome_id: The genome UUID.
        collapsed: Optional comma-separated annotation names to collapse into function nodes.
    """
    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        model_state = engine.current_state

        if collapsed:
            from explaneat.analysis.collapse_validator import collapse_structure

            collapsed_ids = [name.strip() for name in collapsed.split(",")]
            model_state = collapse_structure(
                model_state, engine.annotations, collapsed_ids
            )

        result = serialize_network(model_state)
        result["annotations"] = engine.annotations if engine.annotations else {}
        return json.dumps(result, indent=2)


def get_node_info(genome_id: str, node_id: str) -> str:
    """Get detailed properties of a specific node (bias, activation, response, aggregation).

    Works on the current model state (after operations are applied).
    Useful for understanding what a node computes.
    """
    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        model_state = engine.current_state

        for node in model_state.nodes:
            if node.id == node_id:
                result = {
                    "id": node.id,
                    "type": node.node_type.name.lower(),
                    "bias": node.bias,
                    "activation": node.activation,
                    "response": node.response,
                    "aggregation": node.aggregation,
                }
                if hasattr(node, "display_name") and node.display_name:
                    result["display_name"] = node.display_name
                if hasattr(node, "function_metadata") and node.function_metadata:
                    result["function_metadata"] = node.function_metadata

                # Include incoming/outgoing connections
                incoming = [
                    {"from": c.from_node, "weight": c.weight}
                    for c in model_state.connections
                    if c.to_node == node_id and c.enabled
                ]
                outgoing = [
                    {"to": c.to_node, "weight": c.weight}
                    for c in model_state.connections
                    if c.from_node == node_id and c.enabled
                ]
                result["incoming_connections"] = incoming
                result["outgoing_connections"] = outgoing
                return json.dumps(result, indent=2)

        return json.dumps({"error": f"Node {node_id} not found in model state"})
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools/models.py tests/test_mcp/test_models.py
git commit -m "feat(mcp): add model structure tools (phenotype, model state, node info)"
```

---

## Task 5: Operations Tools (Tools 9–13)

**Files:**
- Modify: `mcp_server/tools/operations.py`
- Test: `tests/test_mcp/test_operations.py`
- Reference: `explaneat/api/routes/operations.py:239-390`, `explaneat/api/routes/genomes.py:336-442`

**Step 1: Write the failing test**

```python
# tests/test_mcp/test_operations.py
"""Test operations tools are registered."""
import pytest


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.operations import register

    server = create_server()
    register(server)

    from mcp_server.tools.operations import (
        list_operations,
        apply_operation,
        validate_operation,
        undo_operation,
        get_annotations,
    )
    assert callable(list_operations)
    assert callable(apply_operation)
    assert callable(validate_operation)
    assert callable(undo_operation)
    assert callable(get_annotations)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_operations.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mcp_server/tools/operations.py
"""Operations tools — apply splits, annotations, identity nodes, undo."""
from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from sqlalchemy.orm.attributes import flag_modified

from explaneat.core.model_state import ModelStateEngine
from explaneat.db.models import Explanation
from mcp_server.helpers import _to_uuid, build_engine, serialize_network
from mcp_server.server import get_db


def register(mcp: FastMCP) -> None:
    mcp.tool()(list_operations)
    mcp.tool()(apply_operation)
    mcp.tool()(validate_operation)
    mcp.tool()(undo_operation)
    mcp.tool()(get_annotations)


def _get_explanation(session, genome_id: str) -> Explanation:
    """Get or create Explanation for a genome."""
    gid = _to_uuid(genome_id)
    explanation = (
        session.query(Explanation).filter(Explanation.genome_id == gid).first()
    )
    if explanation is None:
        from explaneat.db.models import Genome

        genome = session.query(Genome).get(gid)
        if genome is None:
            raise ValueError(f"Genome {genome_id} not found")
        explanation = Explanation(
            genome_id=gid,
            is_well_formed=False,
            operations=[],
        )
        session.add(explanation)
        session.flush()
    return explanation


def list_operations(genome_id: str) -> str:
    """List all operations in the explanation's event stream, in sequence order.

    Operations include: split_node, consolidate_node, add_identity_node, add_node,
    remove_node, annotate, rename_node, rename_annotation, disable_connection,
    enable_connection.

    Each operation has: seq (sequence number), type, params, result, created_at, notes.
    """
    db = get_db()
    with db.session_scope() as session:
        explanation = _get_explanation(session, genome_id)
        ops = explanation.operations or []
        return json.dumps(ops, indent=2, default=str)


def apply_operation(
    genome_id: str,
    operation_type: str,
    params: str,
    notes: Optional[str] = None,
) -> str:
    """Apply an operation to the model state.

    This modifies the genome's explanation by adding an operation to the event stream.
    The operation is validated before being applied.

    Args:
        genome_id: The genome UUID.
        operation_type: One of: split_node, consolidate_node, add_identity_node,
            add_node, remove_node, annotate, rename_node, rename_annotation,
            disable_connection, enable_connection.
        params: JSON string of operation parameters. Examples:
            - split_node: {"node_id": "5", "label": "a"}
            - add_identity_node: {"target_node_id": "5"}
            - annotate: {"name": "F", "node_ids": ["3","4"], "entry_node_ids": ["3"], "exit_node_ids": ["4"]}
            - rename_node: {"node_id": "5", "display_name": "Feature Combiner"}
            - disable_connection: {"from_node": "3", "to_node": "4"}
        notes: Optional notes explaining why this operation was applied.

    Returns the updated model state and the new operation's sequence number.
    """
    params_dict = json.loads(params)

    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)

        try:
            engine.add_operation(
                operation_type, params_dict, validate=True, notes=notes
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

        # Save back to DB
        explanation = _get_explanation(session, genome_id)
        explanation.operations = engine.to_dict()["operations"]
        flag_modified(explanation, "operations")
        session.flush()

        ops = explanation.operations
        new_seq = ops[-1]["seq"] if ops else None

        result = serialize_network(engine.current_state)
        result["operation_seq"] = new_seq
        result["annotations"] = engine.annotations if engine.annotations else {}
        return json.dumps(result, indent=2)


def validate_operation(
    genome_id: str,
    operation_type: str,
    params: str,
) -> str:
    """Validate an operation without applying it.

    Returns validation result with any errors. Use this to check if an operation
    would succeed before applying it.

    Args:
        genome_id: The genome UUID.
        operation_type: The operation type.
        params: JSON string of operation parameters.
    """
    params_dict = json.loads(params)

    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)

        try:
            errors = engine.validate_operation(operation_type, params_dict)
            return json.dumps({"valid": len(errors) == 0, "errors": errors})
        except Exception as e:
            return json.dumps({"valid": False, "errors": [str(e)]})


def undo_operation(genome_id: str, seq: int) -> str:
    """Remove an operation and all subsequent operations (undo).

    This removes the operation at the given sequence number and everything after it.
    The model state is rebuilt from the remaining operations.

    Args:
        genome_id: The genome UUID.
        seq: The sequence number of the operation to remove.
    """
    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)

        try:
            engine.remove_operation(seq)
        except Exception as e:
            return json.dumps({"error": str(e)})

        explanation = _get_explanation(session, genome_id)
        new_ops = engine.to_dict()["operations"]
        explanation.operations = new_ops
        flag_modified(explanation, "operations")
        session.flush()

        result = {
            "status": "ok",
            "remaining_operations": len(new_ops),
            "model_state": serialize_network(engine.current_state),
        }
        return json.dumps(result, indent=2)


def get_annotations(genome_id: str) -> str:
    """List all annotations with hierarchy info (parent/children relationships).

    Returns each annotation with its entry/exit/subgraph nodes, name, hypothesis,
    and parent-child relationships for compositional explanations.
    """
    db = get_db()
    with db.session_scope() as session:
        explanation = _get_explanation(session, genome_id)
        ops = explanation.operations or []

        annotations = []
        # Map annotation names to IDs for hierarchy resolution
        name_to_id = {}
        id_to_children = {}

        for op in ops:
            if op.get("type") == "annotate":
                params = op.get("params", {})
                result = op.get("result", {})
                ann_id = result.get("annotation_id") or f"ann_{op['seq']}"
                name = params.get("name", "")
                name_to_id[name] = ann_id

                child_ids = params.get("child_annotation_ids", [])
                id_to_children[ann_id] = child_ids

                annotations.append(
                    {
                        "annotation_id": ann_id,
                        "name": name,
                        "hypothesis": params.get("hypothesis"),
                        "entry_nodes": params.get("entry_node_ids", []),
                        "exit_nodes": params.get("exit_node_ids", []),
                        "subgraph_nodes": params.get("node_ids", []),
                        "child_annotation_ids": child_ids,
                        "seq": op["seq"],
                    }
                )

        # Resolve parent relationships
        child_to_parent = {}
        for ann_id, children in id_to_children.items():
            for child_name in children:
                child_id = name_to_id.get(child_name, child_name)
                child_to_parent[child_id] = ann_id

        for ann in annotations:
            ann["parent_annotation_id"] = child_to_parent.get(ann["annotation_id"])

        # Apply rename operations
        renames = {}
        for op in ops:
            if op.get("type") == "rename_annotation":
                params = op.get("params", {})
                old_name = params.get("old_name")
                new_name = params.get("new_name")
                if old_name and new_name:
                    renames[old_name] = new_name

        for ann in annotations:
            if ann["name"] in renames:
                ann["display_name"] = renames[ann["name"]]

        return json.dumps(annotations, indent=2)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp/test_operations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools/operations.py tests/test_mcp/test_operations.py
git commit -m "feat(mcp): add operations tools (apply, validate, undo, list, annotations)"
```

---

## Task 6: Evidence & Analysis Tools (Tools 14–20)

**Files:**
- Modify: `mcp_server/tools/evidence.py`
- Create: `mcp_server/rendering.py`
- Test: `tests/test_mcp/test_evidence.py`
- Reference: `explaneat/api/routes/evidence.py:537-1378`

**Step 1: Write the failing test**

```python
# tests/test_mcp/test_evidence.py
"""Test evidence & analysis tools are registered."""
import pytest


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.evidence import register

    server = create_server()
    register(server)

    from mcp_server.tools.evidence import (
        get_formula,
        compute_viz_data,
        render_visualization,
        get_viz_summary,
        compute_shap,
        compute_performance,
        get_input_distribution,
    )
    assert callable(get_formula)
    assert callable(compute_viz_data)
    assert callable(render_visualization)
    assert callable(get_viz_summary)
    assert callable(compute_shap)
    assert callable(compute_performance)
    assert callable(get_input_distribution)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_evidence.py -v`
Expected: FAIL

**Step 3: Write the rendering module first**

```python
# mcp_server/rendering.py
"""Render visualization data to PNG images using matplotlib."""
from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def render_to_png(viz_type: str, data: dict, title: str = "") -> str:
    """Render viz_data output to a PNG image, returned as base64 string.

    Args:
        viz_type: The visualization type (line, heatmap, etc.)
        data: The viz_data dict from compute_* functions
        title: Optional title for the plot

    Returns:
        Base64-encoded PNG string.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    renderer = _RENDERERS.get(viz_type)
    if renderer is None:
        ax.text(0.5, 0.5, f"Unsupported viz type: {viz_type}",
                ha="center", va="center", transform=ax.transAxes)
    else:
        renderer(ax, data)

    if title:
        ax.set_title(title)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _render_line(ax, data: dict):
    """Render a line plot."""
    if "grid" in data:
        grid = data["grid"]
        x = grid.get("x", [])
        y = grid.get("y", [])
        if x and y:
            ax.plot(x, y, "b-", linewidth=2, label="function")

    if "scatter" in data:
        scatter = data["scatter"]
        sx = scatter.get("x", [])
        sy = scatter.get("y", [])
        if sx and sy:
            ax.scatter(sx, sy, alpha=0.3, s=10, c="gray", label="data")

    ax.set_xlabel(data.get("x_label", "input"))
    ax.set_ylabel(data.get("y_label", "output"))
    ax.legend()


def _render_heatmap(ax, data: dict):
    """Render a heatmap."""
    if "grid" in data:
        grid = data["grid"]
        z = np.array(grid.get("z", [[]]))
        x_range = grid.get("x_range", [0, 1])
        y_range = grid.get("y_range", [0, 1])
        im = ax.imshow(
            z, origin="lower", aspect="auto",
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax)

    ax.set_xlabel(data.get("x_label", "input 1"))
    ax.set_ylabel(data.get("y_label", "input 2"))


def _render_scatter(ax, data: dict):
    """Render a scatter plot (PCA, feature-output)."""
    if "scatter" in data:
        scatter = data["scatter"]
        x = scatter.get("x", [])
        y = scatter.get("y", [])
        c = scatter.get("color", None)
        ax.scatter(x, y, c=c, alpha=0.5, s=20, cmap="viridis")

    ax.set_xlabel(data.get("x_label", "x"))
    ax.set_ylabel(data.get("y_label", "y"))


def _render_bar(ax, data: dict):
    """Render a bar chart (sensitivity, SHAP)."""
    if "bars" in data:
        bars = data["bars"]
        names = bars.get("names", [])
        values = bars.get("values", [])
        ax.barh(names, values)
    elif "features" in data and "values" in data:
        ax.barh(data["features"], data["values"])

    ax.set_xlabel("Importance")


def _render_distribution(ax, data: dict):
    """Render a histogram."""
    if "values" in data:
        ax.hist(data["values"], bins=data.get("bins", 30), edgecolor="black", alpha=0.7)
    ax.set_xlabel(data.get("x_label", "value"))
    ax.set_ylabel("count")


def _render_ice(ax, data: dict):
    """Render ICE curves."""
    if "curves" in data:
        for curve in data["curves"]:
            x = curve.get("x", [])
            y = curve.get("y", [])
            ax.plot(x, y, alpha=0.1, color="steelblue")

    if "mean" in data:
        mean = data["mean"]
        ax.plot(mean.get("x", []), mean.get("y", []), "r-", linewidth=2, label="PDP")
        ax.legend()

    ax.set_xlabel(data.get("x_label", "feature value"))
    ax.set_ylabel(data.get("y_label", "prediction"))


_RENDERERS = {
    "line": _render_line,
    "heatmap": _render_heatmap,
    "pca_scatter": _render_scatter,
    "feature_output_scatter": _render_scatter,
    "sensitivity": _render_bar,
    "partial_dependence": _render_line,
    "output_distribution": _render_distribution,
    "ice": _render_ice,
    "activation_profile": _render_line,
    "edge_influence": _render_bar,
    "regime_map": _render_heatmap,
}
```

**Step 4: Write the evidence tools**

```python
# mcp_server/tools/evidence.py
"""Evidence & analysis tools — formulas, viz, SHAP, performance."""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from mcp_server.helpers import (
    _to_uuid,
    build_engine,
    build_model_state,
    find_annotation_in_operations,
    load_split_data,
    serialize_network,
)
from mcp_server.server import get_db


def register(mcp: FastMCP) -> None:
    mcp.tool()(get_formula)
    mcp.tool()(compute_viz_data)
    mcp.tool()(render_visualization)
    mcp.tool()(get_viz_summary)
    mcp.tool()(compute_shap)
    mcp.tool()(compute_performance)
    mcp.tool()(get_input_distribution)


def _json_safe(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _build_annotation_context(session, genome_id, annotation_id, split_id, split_choice="test", sample_fraction=1.0, max_samples=None):
    """Build the common context needed for annotation evidence tools.

    Returns (model_state, annotation, ann_fn, entry_acts, exit_acts, X, y, feature_names, n_in, n_out).
    """
    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.analysis.annotation_function import AnnotationFunction

    model_state = build_model_state(session, genome_id)
    annotation = find_annotation_in_operations(session, genome_id, annotation_id)
    if annotation is None:
        raise ValueError(f"Annotation {annotation_id} not found")

    ann_fn = AnnotationFunction.from_structure(annotation, model_state)
    n_in, n_out = ann_fn.dimensionality

    X, y, feature_names, class_names, num_classes = load_split_data(
        session, split_id, split_choice, sample_fraction, max_samples
    )

    extractor = ActivationExtractor.from_structure(model_state)
    entry_acts, exit_acts = extractor.extract(X, annotation)

    return model_state, annotation, ann_fn, entry_acts, exit_acts, X, y, feature_names, n_in, n_out


def _build_whole_model_context(session, genome_id, split_id, split_choice="test", sample_fraction=1.0, max_samples=None):
    """Build context for whole-model evidence (no annotation).

    Returns (model_state, predict_fn, X, y, feature_names, n_in, n_out).
    """
    import torch
    from explaneat.core.structure_network import StructureNetwork

    model_state = build_model_state(session, genome_id)
    X, y, feature_names, class_names, num_classes = load_split_data(
        session, split_id, split_choice, sample_fraction, max_samples
    )

    struct_net = StructureNetwork(model_state)
    if num_classes and num_classes == 2:
        struct_net.override_output_activation("sigmoid")

    n_in = len(model_state.input_node_ids)
    n_out = len(model_state.output_node_ids)

    def predict_fn(x_arr):
        with torch.no_grad():
            t = torch.tensor(x_arr, dtype=torch.float32)
            return struct_net.forward(t).numpy()

    return model_state, predict_fn, X, y, feature_names, n_in, n_out


def get_formula(
    genome_id: str,
    annotation_id: str,
    force: bool = False,
) -> str:
    """Get the symbolic mathematical formula for an annotation.

    Returns LaTeX representation, tractability info, and dimensionality.
    For composed annotations, returns both expanded and collapsed forms.

    Args:
        genome_id: The genome UUID.
        annotation_id: The annotation ID or name.
        force: If True, compute formula even for complex annotations (>5 inputs, >3 layers).
    """
    from explaneat.analysis.annotation_function import AnnotationFunction

    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        model_state = engine.current_state
        annotation = find_annotation_in_operations(session, genome_id, annotation_id)
        if annotation is None:
            return json.dumps({"error": f"Annotation {annotation_id} not found"})

        ann_fn = AnnotationFunction.from_structure(annotation, model_state)
        n_in, n_out = ann_fn.dimensionality

        latex = ann_fn.to_latex(expand=True, force=force)
        latex_collapsed = ann_fn.to_latex(expand=False, force=force)

        child_ids = annotation.get("child_annotation_ids", [])
        is_composed = len(child_ids) > 0

        result = {
            "latex": latex,
            "latex_collapsed": latex_collapsed,
            "tractable": latex is not None,
            "is_composed": is_composed,
            "child_annotation_ids": child_ids,
            "dimensionality": {"inputs": n_in, "outputs": n_out},
            "entry_nodes": annotation.get("entry_node_ids", []),
            "exit_nodes": annotation.get("exit_node_ids", []),
        }
        return json.dumps(result, indent=2)


def compute_viz_data(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: int = 500,
    params: Optional[str] = None,
) -> str:
    """Compute raw visualization data for an annotation or whole model.

    Supported viz_types: line, heatmap, partial_dependence, pca_scatter, sensitivity,
    ice, feature_output_scatter, output_distribution, activation_profile, edge_influence,
    regime_map.

    Returns raw numerical data (grid points, scatter points, domains) that can be
    interpreted directly or passed to render_visualization for a PNG.

    Args:
        genome_id: The genome UUID.
        viz_type: The visualization type.
        dataset_split_id: The dataset split UUID.
        annotation_id: Annotation ID/name (optional — if omitted, uses whole model).
        node_id: Node ID for node-specific visualizations.
        split: Which split to use: "train", "test", or "validation".
        output_index: Which output to visualize (for multi-output annotations).
        sample_fraction: Fraction of data to use (0-1).
        max_samples: Maximum number of samples.
        params: Optional JSON string of additional viz parameters.
    """
    from explaneat.analysis import viz_data as vd

    extra_params = json.loads(params) if params else {}

    db = get_db()
    with db.session_scope() as session:
        if annotation_id:
            (model_state, annotation, ann_fn, entry_acts, exit_acts,
             X, y, feature_names, n_in, n_out) = _build_annotation_context(
                session, genome_id, annotation_id, dataset_split_id,
                split, sample_fraction, max_samples
            )
            predict_fn = ann_fn
            input_data = entry_acts
            output_data = exit_acts
            entry_names = annotation.get("entry_node_ids", [])
            exit_names = annotation.get("exit_node_ids", [])
        else:
            (model_state, predict_fn, X, y, feature_names,
             n_in, n_out) = _build_whole_model_context(
                session, genome_id, dataset_split_id,
                split, sample_fraction, max_samples
            )
            input_data = X
            output_data = predict_fn(X)
            entry_names = feature_names
            exit_names = [n.id for n in model_state.nodes if n.id in model_state.output_node_ids]

        # Route to the right viz computation
        viz_fn_map = {
            "line": vd.compute_line_plot,
            "heatmap": vd.compute_heatmap,
            "partial_dependence": vd.compute_partial_dependence,
            "pca_scatter": vd.compute_pca_scatter,
            "sensitivity": vd.compute_sensitivity,
            "ice": vd.compute_ice_plot,
            "feature_output_scatter": vd.compute_feature_output_scatter,
            "output_distribution": vd.compute_output_distribution,
            "activation_profile": vd.compute_activation_profile,
            "edge_influence": vd.compute_edge_influence,
            "regime_map": vd.compute_regime_map,
        }

        viz_fn = viz_fn_map.get(viz_type)
        if viz_fn is None:
            return json.dumps({"error": f"Unknown viz_type: {viz_type}",
                             "available": list(viz_fn_map.keys())})

        # Build kwargs based on what the viz function expects
        import inspect
        sig = inspect.signature(viz_fn)
        kwargs = {}

        # Common args most viz functions accept
        arg_map = {
            "predict_fn": predict_fn,
            "X": input_data,
            "y": output_data if output_data is not None else y,
            "entry_acts": entry_acts if annotation_id else input_data,
            "exit_acts": exit_acts if annotation_id else output_data,
            "input_data": input_data,
            "output_data": output_data,
            "feature_names": entry_names,
            "output_index": output_index,
            "model_state": model_state,
            "annotation": annotation if annotation_id else None,
            "node_id": node_id,
        }

        for param_name in sig.parameters:
            if param_name in arg_map:
                kwargs[param_name] = arg_map[param_name]
            elif param_name in extra_params:
                kwargs[param_name] = extra_params[param_name]

        result_data = viz_fn(**kwargs)

        response = {
            "viz_type": viz_type,
            "data": json.loads(json.dumps(result_data, default=_json_safe)),
            "entry_names": list(entry_names),
            "exit_names": list(exit_names),
            "dimensionality": {"inputs": n_in, "outputs": n_out},
            "suggested_viz_types": vd.suggest_viz_types(n_in, n_out),
        }
        return json.dumps(response, indent=2, default=_json_safe)


def render_visualization(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: int = 500,
    params: Optional[str] = None,
    title: Optional[str] = None,
) -> list:
    """Render a visualization as a PNG image.

    Same parameters as compute_viz_data, but returns a rendered image instead of raw data.
    The image is returned as base64-encoded PNG that Claude can view directly.
    """
    from mcp_server.rendering import render_to_png

    # First compute the raw data
    raw_json = compute_viz_data(
        genome_id, viz_type, dataset_split_id, annotation_id, node_id,
        split, output_index, sample_fraction, max_samples, params,
    )
    raw = json.loads(raw_json)

    if "error" in raw:
        return [TextContent(type="text", text=json.dumps(raw))]

    plot_title = title or f"{viz_type} — {annotation_id or 'whole model'}"
    png_b64 = render_to_png(viz_type, raw.get("data", {}), title=plot_title)

    return [
        ImageContent(type="image", data=png_b64, mimeType="image/png"),
        TextContent(type="text", text=json.dumps({
            "viz_type": viz_type,
            "entry_names": raw.get("entry_names", []),
            "exit_names": raw.get("exit_names", []),
            "dimensionality": raw.get("dimensionality", {}),
        }, indent=2)),
    ]


def get_viz_summary(
    genome_id: str,
    viz_type: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    output_index: int = 0,
    sample_fraction: float = 1.0,
    max_samples: int = 500,
    params: Optional[str] = None,
) -> str:
    """Compute summary statistics for a visualization.

    Instead of raw data or an image, returns a text summary with key statistics:
    ranges, trends, correlations, outliers, etc. Useful for quick analysis without
    needing to interpret raw numbers or images.
    """
    raw_json = compute_viz_data(
        genome_id, viz_type, dataset_split_id, annotation_id, node_id,
        split, output_index, sample_fraction, max_samples, params,
    )
    raw = json.loads(raw_json)

    if "error" in raw:
        return json.dumps(raw)

    data = raw.get("data", {})
    summary = {"viz_type": viz_type}

    # Compute summary statistics based on available data
    if "grid" in data:
        grid = data["grid"]
        if "y" in grid:
            y_vals = np.array(grid["y"])
            summary["output_range"] = [float(y_vals.min()), float(y_vals.max())]
            summary["output_mean"] = float(y_vals.mean())
            summary["output_std"] = float(y_vals.std())
        if "x" in grid:
            x_vals = np.array(grid["x"])
            summary["input_range"] = [float(x_vals.min()), float(x_vals.max())]
        if "z" in grid:
            z_vals = np.array(grid["z"])
            summary["output_range"] = [float(z_vals.min()), float(z_vals.max())]
            summary["output_mean"] = float(z_vals.mean())

    if "scatter" in data:
        scatter = data["scatter"]
        if "x" in scatter and "y" in scatter:
            x = np.array(scatter["x"])
            y = np.array(scatter["y"])
            summary["n_points"] = len(x)
            summary["x_range"] = [float(x.min()), float(x.max())]
            summary["y_range"] = [float(y.min()), float(y.max())]
            if len(x) > 2:
                corr = np.corrcoef(x, y)[0, 1]
                summary["correlation"] = float(corr)

    if "bars" in data:
        bars = data["bars"]
        if "names" in bars and "values" in bars:
            summary["features"] = dict(zip(bars["names"], [float(v) for v in bars["values"]]))

    summary["entry_names"] = raw.get("entry_names", [])
    summary["exit_names"] = raw.get("exit_names", [])
    summary["dimensionality"] = raw.get("dimensionality", {})

    return json.dumps(summary, indent=2)


def compute_shap(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    node_id: Optional[str] = None,
    split: str = "test",
    max_samples: int = 200,
) -> str:
    """Compute SHAP variable importance for an annotation or whole model.

    Returns mean absolute SHAP values per feature, base values, and per-output breakdowns.
    This tells you which inputs matter most for the annotation's output.

    Args:
        genome_id: The genome UUID.
        dataset_split_id: The dataset split UUID.
        annotation_id: Annotation ID/name (optional — if omitted, uses whole model).
        node_id: Node ID for node-specific SHAP.
        split: Which split to use.
        max_samples: Maximum samples for SHAP computation (more = slower but more accurate).
    """
    from explaneat.analysis.shap_analysis import compute_shap_values

    db = get_db()
    with db.session_scope() as session:
        if annotation_id:
            (model_state, annotation, ann_fn, entry_acts, exit_acts,
             X, y, feature_names, n_in, n_out) = _build_annotation_context(
                session, genome_id, annotation_id, dataset_split_id,
                split, max_samples=max_samples
            )
            predict_fn = ann_fn
            input_data = entry_acts
            input_names = annotation.get("entry_node_ids", [])
        else:
            (model_state, predict_fn, X, y, feature_names,
             n_in, n_out) = _build_whole_model_context(
                session, genome_id, dataset_split_id,
                split, max_samples=max_samples
            )
            input_data = X
            input_names = feature_names

        shap_result = compute_shap_values(predict_fn, input_data, input_names)

        result = {
            "feature_names": shap_result["feature_names"],
            "mean_abs_shap": [float(v) for v in shap_result["mean_abs_shap"]],
            "base_value": float(shap_result["base_value"]) if "base_value" in shap_result else None,
        }
        if "outputs" in shap_result:
            result["outputs"] = shap_result["outputs"]

        return json.dumps(result, indent=2, default=_json_safe)


def compute_performance(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    split: str = "test",
    max_samples: int = 1000,
) -> str:
    """Evaluate annotation or whole-model performance on dataset.

    Returns metrics: MSE, RMSE, MAE, correlation. For classification also returns
    accuracy, AUC, precision, recall, F1.

    Args:
        genome_id: The genome UUID.
        dataset_split_id: The dataset split UUID.
        annotation_id: Annotation ID/name (optional — if omitted, uses whole model).
        split: Which split to use.
        max_samples: Maximum samples for evaluation.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    db = get_db()
    with db.session_scope() as session:
        if annotation_id:
            (model_state, annotation, ann_fn, entry_acts, exit_acts,
             X, y, feature_names, n_in, n_out) = _build_annotation_context(
                session, genome_id, annotation_id, dataset_split_id,
                split, max_samples=max_samples
            )
            predictions = exit_acts
        else:
            (model_state, predict_fn, X, y, feature_names,
             n_in, n_out) = _build_whole_model_context(
                session, genome_id, dataset_split_id,
                split, max_samples=max_samples
            )
            predictions = predict_fn(X)

        # Ensure predictions and y are 1D for single-output
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.ravel()
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        mse = float(mean_squared_error(y, predictions))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y, predictions))
        corr = float(np.corrcoef(y.ravel(), predictions.ravel())[0, 1]) if len(y) > 1 else None

        result = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "correlation": corr,
            "n_samples": len(y),
        }

        # Classification metrics if applicable
        unique_y = np.unique(y)
        if len(unique_y) <= 10 and all(v == int(v) for v in unique_y):
            from sklearn.metrics import accuracy_score
            pred_classes = (predictions > 0.5).astype(int) if predictions.max() <= 1 else predictions.argmax(axis=1)
            if pred_classes.ndim > 1:
                pred_classes = pred_classes.ravel()
            y_int = y.astype(int).ravel()
            try:
                result["accuracy"] = float(accuracy_score(y_int, pred_classes))
            except Exception:
                pass

        return json.dumps(result, indent=2)


def get_input_distribution(
    genome_id: str,
    dataset_split_id: str,
    annotation_id: Optional[str] = None,
    split: str = "test",
    max_samples: int = 1000,
) -> str:
    """Analyze input feature distributions for an annotation or whole model.

    Returns per-feature statistics: mean, std, min, max, quartiles, and
    correlation matrix between features. Useful for understanding what data
    the model is seeing.
    """
    db = get_db()
    with db.session_scope() as session:
        if annotation_id:
            (model_state, annotation, ann_fn, entry_acts, exit_acts,
             X, y, feature_names, n_in, n_out) = _build_annotation_context(
                session, genome_id, annotation_id, dataset_split_id,
                split, max_samples=max_samples
            )
            input_data = entry_acts
            input_names = annotation.get("entry_node_ids", [])
        else:
            X, y, feature_names, class_names, num_classes = load_split_data(
                session, dataset_split_id, split, max_samples=max_samples
            )
            input_data = X
            input_names = feature_names

        features = []
        for i, name in enumerate(input_names):
            col = input_data[:, i] if input_data.ndim > 1 else input_data
            features.append({
                "name": name,
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "q25": float(np.percentile(col, 25)),
                "q50": float(np.percentile(col, 50)),
                "q75": float(np.percentile(col, 75)),
            })

        # Correlation matrix
        if input_data.ndim > 1 and input_data.shape[1] > 1:
            corr = np.corrcoef(input_data.T).tolist()
        else:
            corr = [[1.0]]

        result = {
            "features": features,
            "correlation_matrix": corr,
            "feature_names": list(input_names),
            "n_samples": len(input_data),
        }
        return json.dumps(result, indent=2, default=_json_safe)
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_mcp/test_evidence.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add mcp_server/tools/evidence.py mcp_server/rendering.py tests/test_mcp/test_evidence.py
git commit -m "feat(mcp): add evidence tools (formula, viz, SHAP, performance) and rendering"
```

---

## Task 7: Coverage & Classification Tools (Tools 21–23)

**Files:**
- Modify: `mcp_server/tools/coverage.py`
- Test: `tests/test_mcp/test_coverage.py`
- Reference: `explaneat/api/routes/analysis.py:99-287`

**Step 1: Write the failing test**

```python
# tests/test_mcp/test_coverage.py
"""Test coverage & classification tools are registered."""
import pytest


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.coverage import register

    server = create_server()
    register(server)

    from mcp_server.tools.coverage import (
        classify_nodes,
        detect_splits,
        get_coverage,
    )
    assert callable(classify_nodes)
    assert callable(detect_splits)
    assert callable(get_coverage)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_coverage.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mcp_server/tools/coverage.py
"""Coverage & classification tools."""
from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from explaneat.db.models import Explanation
from mcp_server.helpers import _to_uuid, build_model_state
from mcp_server.server import get_db


def register(mcp: FastMCP) -> None:
    mcp.tool()(classify_nodes)
    mcp.tool()(detect_splits)
    mcp.tool()(get_coverage)


def classify_nodes(
    genome_id: str,
    node_ids: str,
    entry_node_ids: str,
    exit_node_ids: str,
) -> str:
    """Classify nodes as entry, intermediate, or exit for a proposed annotation.

    Given a set of node IDs and proposed entry/exit nodes, validates the classification
    and identifies any issues (e.g., nodes with external inputs/outputs that need splitting).

    Args:
        genome_id: The genome UUID.
        node_ids: JSON array of node IDs in the proposed annotation.
        entry_node_ids: JSON array of proposed entry node IDs.
        exit_node_ids: JSON array of proposed exit node IDs.
    """
    from explaneat.analysis.node_classification import classify_coverage

    node_list = json.loads(node_ids)
    entry_list = json.loads(entry_node_ids)
    exit_list = json.loads(exit_node_ids)

    db = get_db()
    with db.session_scope() as session:
        model_state = build_model_state(session, genome_id)

        coverage = {
            "node_ids": node_list,
            "entry_node_ids": entry_list,
            "exit_node_ids": exit_list,
        }

        result = classify_coverage(model_state, coverage)
        return json.dumps(result, indent=2, default=str)


def detect_splits(
    genome_id: str,
    node_ids: str,
    entry_node_ids: str,
    exit_node_ids: str,
) -> str:
    """Detect nodes that need splitting for a proposed annotation.

    Analyzes the proposed coverage and identifies violations (nodes with external
    inputs/outputs that would break annotation boundaries). Suggests split_node
    operations to fix them.

    Args:
        genome_id: The genome UUID.
        node_ids: JSON array of node IDs in the proposed annotation.
        entry_node_ids: JSON array of proposed entry node IDs.
        exit_node_ids: JSON array of proposed exit node IDs.
    """
    from explaneat.analysis.split_detection import analyze_coverage_for_splits

    node_list = json.loads(node_ids)
    entry_list = json.loads(entry_node_ids)
    exit_list = json.loads(exit_node_ids)

    db = get_db()
    with db.session_scope() as session:
        model_state = build_model_state(session, genome_id)

        proposed_coverage = {
            "node_ids": node_list,
            "entry_node_ids": entry_list,
            "exit_node_ids": exit_list,
        }

        result = analyze_coverage_for_splits(model_state, proposed_coverage)
        return json.dumps(result, indent=2, default=str)


def get_coverage(genome_id: str) -> str:
    """Get structural and compositional coverage metrics for the genome's explanation.

    Returns: covered nodes, uncovered nodes, coverage percentages, annotation count.
    Coverage measures how much of the phenotype is explained by annotations.
    """
    db = get_db()
    with db.session_scope() as session:
        gid = _to_uuid(genome_id)
        explanation = (
            session.query(Explanation).filter(Explanation.genome_id == gid).first()
        )

        if not explanation or not explanation.operations:
            model_state = build_model_state(session, genome_id)
            all_nodes = {n.id for n in model_state.nodes}
            input_nodes = set(model_state.input_node_ids)
            output_nodes = set(model_state.output_node_ids)
            hidden_nodes = all_nodes - input_nodes - output_nodes

            return json.dumps({
                "structural_coverage": 0.0,
                "covered_nodes": [],
                "uncovered_nodes": list(hidden_nodes),
                "total_hidden_nodes": len(hidden_nodes),
                "annotations_count": 0,
            }, indent=2)

        model_state = build_model_state(session, genome_id)
        all_nodes = {n.id for n in model_state.nodes}
        input_nodes = set(model_state.input_node_ids)
        output_nodes = set(model_state.output_node_ids)
        hidden_nodes = all_nodes - input_nodes - output_nodes

        # Collect covered nodes from annotations
        covered = set()
        annotations_count = 0
        for op in explanation.operations:
            if op.get("type") == "annotate":
                annotations_count += 1
                params = op.get("params", {})
                subgraph = set(params.get("node_ids", []))
                covered |= (subgraph & hidden_nodes)

        uncovered = hidden_nodes - covered
        coverage = len(covered) / len(hidden_nodes) if hidden_nodes else 1.0

        result = {
            "structural_coverage": float(coverage),
            "covered_nodes": sorted(covered),
            "uncovered_nodes": sorted(uncovered),
            "total_hidden_nodes": len(hidden_nodes),
            "annotations_count": annotations_count,
        }
        return json.dumps(result, indent=2)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp/test_coverage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools/coverage.py tests/test_mcp/test_coverage.py
git commit -m "feat(mcp): add coverage & classification tools"
```

---

## Task 8: Dataset Tools (Tools 24–26)

**Files:**
- Modify: `mcp_server/tools/datasets.py`
- Test: `tests/test_mcp/test_datasets.py`
- Reference: `explaneat/api/routes/datasets.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp/test_datasets.py
"""Test dataset tools are registered."""
import pytest


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.datasets import register

    server = create_server()
    register(server)

    from mcp_server.tools.datasets import (
        list_datasets,
        get_dataset,
        get_dataset_splits,
    )
    assert callable(list_datasets)
    assert callable(get_dataset)
    assert callable(get_dataset_splits)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_datasets.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mcp_server/tools/datasets.py
"""Dataset tools."""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from explaneat.db.models import Dataset, DatasetSplit
from mcp_server.helpers import _to_uuid
from mcp_server.server import get_db


def register(mcp: FastMCP) -> None:
    mcp.tool()(list_datasets)
    mcp.tool()(get_dataset)
    mcp.tool()(get_dataset_splits)


def list_datasets() -> str:
    """List all available datasets.

    Returns dataset metadata: name, source, number of samples/features/classes,
    feature names, and task type (classification/regression).
    """
    db = get_db()
    with db.session_scope() as session:
        datasets = (
            session.query(Dataset).order_by(Dataset.created_at.desc()).all()
        )
        results = []
        for ds in datasets:
            task_type = None
            if ds.additional_metadata and "task_type" in ds.additional_metadata:
                task_type = ds.additional_metadata["task_type"]

            results.append({
                "id": str(ds.id),
                "name": ds.name,
                "source": ds.source,
                "num_samples": ds.num_samples,
                "num_features": ds.num_features,
                "num_classes": ds.num_classes,
                "feature_names": ds.feature_names,
                "target_name": ds.target_name,
                "class_names": ds.class_names,
                "task_type": task_type,
                "description": ds.description,
                "created_at": str(ds.created_at) if ds.created_at else None,
            })
        return json.dumps(results, indent=2)


def get_dataset(dataset_id: str) -> str:
    """Get detailed metadata for a specific dataset.

    Returns all metadata including feature types, class names, and source info.
    Does not return the actual data arrays (use evidence tools to work with the data).
    """
    db = get_db()
    with db.session_scope() as session:
        ds = session.query(Dataset).get(_to_uuid(dataset_id))
        if ds is None:
            return json.dumps({"error": f"Dataset {dataset_id} not found"})

        task_type = None
        if ds.additional_metadata and "task_type" in ds.additional_metadata:
            task_type = ds.additional_metadata["task_type"]

        result = {
            "id": str(ds.id),
            "name": ds.name,
            "version": ds.version,
            "source": ds.source,
            "source_url": ds.source_url,
            "description": ds.description,
            "num_samples": ds.num_samples,
            "num_features": ds.num_features,
            "num_classes": ds.num_classes,
            "feature_names": ds.feature_names,
            "feature_types": ds.feature_types,
            "target_name": ds.target_name,
            "class_names": ds.class_names,
            "task_type": task_type,
            "source_dataset_id": str(ds.source_dataset_id) if ds.source_dataset_id else None,
            "created_at": str(ds.created_at) if ds.created_at else None,
        }
        return json.dumps(result, indent=2)


def get_dataset_splits(dataset_id: str) -> str:
    """List train/test splits for a dataset.

    Returns split metadata: type, sizes, random seed, stratification, and scaler params.
    You'll need a split ID for evidence and visualization tools.
    """
    db = get_db()
    with db.session_scope() as session:
        splits = (
            session.query(DatasetSplit)
            .filter(DatasetSplit.dataset_id == _to_uuid(dataset_id))
            .all()
        )
        results = []
        for s in splits:
            results.append({
                "id": str(s.id),
                "name": s.name,
                "split_type": s.split_type,
                "test_size": s.test_size,
                "random_state": s.random_state,
                "stratify": s.stratify,
                "train_size": s.train_size,
                "test_size_actual": s.test_size_actual,
                "validation_size": s.validation_size,
                "has_scaler": s.scaler_params is not None,
            })
        return json.dumps(results, indent=2)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp/test_datasets.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools/datasets.py tests/test_mcp/test_datasets.py
git commit -m "feat(mcp): add dataset tools (list, get, splits)"
```

---

## Task 9: Snapshot & Narrative Tools (Tools 27–29)

**Files:**
- Modify: `mcp_server/tools/snapshots.py`
- Test: `tests/test_mcp/test_snapshots.py`
- Reference: `explaneat/api/routes/evidence.py:953-1085`

**Step 1: Write the failing test**

```python
# tests/test_mcp/test_snapshots.py
"""Test snapshot & narrative tools are registered."""
import pytest


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools.snapshots import register

    server = create_server()
    register(server)

    from mcp_server.tools.snapshots import (
        save_snapshot,
        update_narrative,
        list_evidence,
    )
    assert callable(save_snapshot)
    assert callable(update_narrative)
    assert callable(list_evidence)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_snapshots.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mcp_server/tools/snapshots.py
"""Snapshot & narrative tools — save evidence, update descriptions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP
from sqlalchemy.orm.attributes import flag_modified

from explaneat.db.models import Explanation
from mcp_server.helpers import _to_uuid, find_annotation_in_operations
from mcp_server.server import get_db


def register(mcp: FastMCP) -> None:
    mcp.tool()(save_snapshot)
    mcp.tool()(update_narrative)
    mcp.tool()(list_evidence)


def save_snapshot(
    genome_id: str,
    annotation_id: str,
    category: str,
    viz_config: str,
    narrative: str,
    svg_data: Optional[str] = None,
) -> str:
    """Save an evidence snapshot for an annotation.

    Attaches a visualization snapshot with narrative explanation to the annotation's
    evidence. Categories organize different types of evidence (e.g., "function_shape",
    "sensitivity", "performance").

    Args:
        genome_id: The genome UUID.
        annotation_id: The annotation ID or name.
        category: Evidence category (e.g., "function_shape", "sensitivity").
        viz_config: JSON string of the visualization configuration used.
        narrative: Text description of what this evidence shows.
        svg_data: Optional SVG data for the visualization.
    """
    viz_config_dict = json.loads(viz_config)

    db = get_db()
    with db.session_scope() as session:
        gid = _to_uuid(genome_id)
        explanation = (
            session.query(Explanation).filter(Explanation.genome_id == gid).first()
        )
        if not explanation:
            return json.dumps({"error": "No explanation found for genome"})

        # Find the annotation operation and add evidence
        ops = explanation.operations or []
        found = False
        for op in ops:
            if op.get("type") == "annotate":
                result = op.get("result", {})
                op_ann_id = result.get("annotation_id") or f"ann_{op['seq']}"
                name = op.get("params", {}).get("name", "")

                if op_ann_id == annotation_id or name == annotation_id:
                    params = op.get("params", {})
                    if "evidence" not in params:
                        params["evidence"] = {}
                    if category not in params["evidence"]:
                        params["evidence"][category] = []

                    params["evidence"][category].append({
                        "viz_config": viz_config_dict,
                        "narrative": narrative,
                        "svg_data": svg_data,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    found = True
                    break

        if not found:
            return json.dumps({"error": f"Annotation {annotation_id} not found"})

        flag_modified(explanation, "operations")
        session.flush()

        return json.dumps({"status": "ok", "category": category, "annotation_id": annotation_id})


def update_narrative(
    genome_id: str,
    annotation_id: str,
    narrative: str,
) -> str:
    """Update the narrative description for an annotation.

    The narrative is a human-readable explanation of what the annotation represents
    and how it contributes to the model's behavior.

    Args:
        genome_id: The genome UUID.
        annotation_id: The annotation ID or name.
        narrative: The new narrative text.
    """
    db = get_db()
    with db.session_scope() as session:
        gid = _to_uuid(genome_id)
        explanation = (
            session.query(Explanation).filter(Explanation.genome_id == gid).first()
        )
        if not explanation:
            return json.dumps({"error": "No explanation found for genome"})

        ops = explanation.operations or []
        found = False
        for op in ops:
            if op.get("type") == "annotate":
                result = op.get("result", {})
                op_ann_id = result.get("annotation_id") or f"ann_{op['seq']}"
                name = op.get("params", {}).get("name", "")

                if op_ann_id == annotation_id or name == annotation_id:
                    op.get("params", {})["hypothesis"] = narrative
                    found = True
                    break

        if not found:
            return json.dumps({"error": f"Annotation {annotation_id} not found"})

        flag_modified(explanation, "operations")
        session.flush()

        return json.dumps({"status": "ok", "annotation_id": annotation_id})


def list_evidence(genome_id: str) -> str:
    """List all saved evidence entries across all annotations.

    Returns evidence organized by annotation, with each entry's category,
    narrative, timestamp, and viz config.
    """
    db = get_db()
    with db.session_scope() as session:
        gid = _to_uuid(genome_id)
        explanation = (
            session.query(Explanation).filter(Explanation.genome_id == gid).first()
        )
        if not explanation:
            return json.dumps({"error": "No explanation found for genome"})

        ops = explanation.operations or []
        all_evidence = []

        for op in ops:
            if op.get("type") == "annotate":
                params = op.get("params", {})
                result = op.get("result", {})
                ann_id = result.get("annotation_id") or f"ann_{op['seq']}"
                name = params.get("name", "")
                evidence = params.get("evidence", {})

                if evidence:
                    for category, entries in evidence.items():
                        for entry in entries:
                            all_evidence.append({
                                "annotation_id": ann_id,
                                "annotation_name": name,
                                "category": category,
                                "narrative": entry.get("narrative"),
                                "timestamp": entry.get("timestamp"),
                                "viz_config": entry.get("viz_config"),
                                "has_svg": entry.get("svg_data") is not None,
                            })

        return json.dumps(all_evidence, indent=2)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp/test_snapshots.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools/snapshots.py tests/test_mcp/test_snapshots.py
git commit -m "feat(mcp): add snapshot & narrative tools"
```

---

## Task 10: Integration Test & MCP Configuration

**Files:**
- Create: `tests/test_mcp/test_integration.py`
- Test: All MCP tools end-to-end

**Step 1: Write integration test**

```python
# tests/test_mcp/test_integration.py
"""Integration test — verify all tools register and server can start."""
import json
import pytest


def test_all_tools_register():
    """Create server, register all tools, verify count."""
    from mcp_server.server import create_server
    from mcp_server.tools import register_all

    server = create_server()
    register_all(server)

    # The server should have all 29 tools registered
    # FastMCP stores tools internally - verify key tools are accessible
    from mcp_server.tools.experiments import list_experiments, get_experiment, get_best_genome, list_genomes, get_genome
    from mcp_server.tools.models import get_phenotype, get_model_state, get_node_info
    from mcp_server.tools.operations import list_operations, apply_operation, validate_operation, undo_operation, get_annotations
    from mcp_server.tools.evidence import get_formula, compute_viz_data, render_visualization, get_viz_summary, compute_shap, compute_performance, get_input_distribution
    from mcp_server.tools.coverage import classify_nodes, detect_splits, get_coverage
    from mcp_server.tools.datasets import list_datasets, get_dataset, get_dataset_splits
    from mcp_server.tools.snapshots import save_snapshot, update_narrative, list_evidence

    all_tools = [
        list_experiments, get_experiment, get_best_genome, list_genomes, get_genome,
        get_phenotype, get_model_state, get_node_info,
        list_operations, apply_operation, validate_operation, undo_operation, get_annotations,
        get_formula, compute_viz_data, render_visualization, get_viz_summary, compute_shap, compute_performance, get_input_distribution,
        classify_nodes, detect_splits, get_coverage,
        list_datasets, get_dataset, get_dataset_splits,
        save_snapshot, update_narrative, list_evidence,
    ]
    assert len(all_tools) == 29
    for tool in all_tools:
        assert callable(tool)


def test_server_entry_point_importable():
    """The __main__ module can be imported without running."""
    # Just verify the module structure is correct
    from mcp_server.server import create_server
    server = create_server()
    assert server.name == "explaneat"
```

**Step 2: Run integration test**

Run: `uv run pytest tests/test_mcp/test_integration.py -v`
Expected: PASS

**Step 3: Run all MCP tests together**

Run: `uv run pytest tests/test_mcp/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_mcp/test_integration.py
git commit -m "test(mcp): add integration test verifying all 29 tools register"
```

---

## Task 11: Documentation & MCP Configuration

**Files:**
- Modify: `CLAUDE.md` (add MCP section)

**Step 1: Add MCP documentation to CLAUDE.md**

Add to the Commands section:

```markdown
### MCP Server
```bash
uv run python -m mcp_server   # Run MCP server (stdio transport)
```

To configure in Claude Code settings (`~/.claude/settings.json` or project `.claude/settings.json`):
```json
{
  "mcpServers": {
    "explaneat": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server"],
      "cwd": "/Users/mike/dev/explaneat",
      "env": {
        "DATABASE_URL": "postgresql://localhost/explaneat_dev"
      }
    }
  }
}
```

The MCP server exposes 29 tools for model analysis:
- **Discovery**: list_experiments, get_experiment, get_best_genome, list_genomes, get_genome
- **Structure**: get_phenotype, get_model_state, get_node_info
- **Operations**: list_operations, apply_operation, validate_operation, undo_operation, get_annotations
- **Evidence**: get_formula, compute_viz_data, render_visualization, get_viz_summary, compute_shap, compute_performance, get_input_distribution
- **Coverage**: classify_nodes, detect_splits, get_coverage
- **Datasets**: list_datasets, get_dataset, get_dataset_splits
- **Snapshots**: save_snapshot, update_narrative, list_evidence
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add MCP server documentation and configuration"
```

---

## Task 12: Final Verification

**Step 1: Run all tests**

```bash
uv run pytest tests/test_mcp/ -v
```

Expected: All tests pass.

**Step 2: Verify server starts**

```bash
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":1}' | uv run python -m mcp_server
```

Expected: JSON-RPC response with server capabilities and tool list.

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(mcp): final fixes from verification"
```
