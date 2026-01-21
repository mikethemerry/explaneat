# API Implementation Plan

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Node.js Frontend  │  REST   │   Python API        │
│   (React + D3/Cyto) │ ◀─────▶ │   (FastAPI)         │
└─────────────────────┘         └──────────┬──────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │   PostgreSQL        │
                                │   (existing DB)     │
                                └─────────────────────┘
```

## Data Model Changes

### Explanation Table (updated)

```python
class Explanation(Base):
    id: UUID
    genome_id: UUID
    name: Optional[str]
    description: Optional[str]

    # NEW: Operations stored as JSON array
    operations: List[dict]  # JSON column

    # Cached state (recomputed on change)
    is_well_formed: bool
    structural_coverage: float
    compositional_coverage: float

    created_at: datetime
    updated_at: datetime
```

### Operation Schema (within operations JSON)

```python
{
    "seq": 0,
    "type": "split_node",  # | "consolidate_node" | "remove_node" | "add_node" | "add_identity_node" | "annotate"
    "params": {
        # Type-specific parameters
    },
    "created_at": "ISO8601 timestamp"
}
```

## API Endpoints

### Genomes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/genomes` | List all genomes |
| GET | `/api/genomes/{id}` | Get genome metadata |
| GET | `/api/genomes/{id}/phenotype` | Get original phenotype (pruned network) |

### Explanations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/genomes/{id}/explanation` | Get explanation document |
| PUT | `/api/genomes/{id}/explanation` | Update explanation (name, description) |
| DELETE | `/api/genomes/{id}/explanation` | Delete explanation (reset to empty) |

### Model State

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/genomes/{id}/model` | Get current model state (phenotype + operations applied) |
| GET | `/api/genomes/{id}/model/coverage` | Get coverage analysis for current state |

### Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/genomes/{id}/operations` | Get all operations |
| POST | `/api/genomes/{id}/operations` | Append new operation |
| DELETE | `/api/genomes/{id}/operations/{seq}` | Remove operation at seq (and all after) |
| POST | `/api/genomes/{id}/operations/validate` | Validate proposed operation without applying |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/genomes/{id}/analyze/split-detection` | Detect required splits for proposed coverage |
| POST | `/api/genomes/{id}/analyze/classify-nodes` | Classify nodes as entry/intermediate/exit for coverage |

## API Response Schemas

### Phenotype / Model Response

```json
{
    "nodes": [
        {
            "id": "5",
            "type": "hidden",  // "input" | "hidden" | "output" | "identity"
            "bias": 0.5,
            "activation": "sigmoid",
            "position": {"x": 100, "y": 200}  // for visualization
        }
    ],
    "connections": [
        {
            "from": "-1",
            "to": "5",
            "weight": 0.8,
            "enabled": true
        }
    ],
    "metadata": {
        "input_nodes": ["-1", "-2"],
        "output_nodes": ["0"],
        "is_original": false  // true if no operations applied
    }
}
```

### Operation Request/Response

```json
{
    "type": "split_node",
    "params": {
        "node_id": "5"
    }
}
```

Response includes the result:

```json
{
    "seq": 3,
    "type": "split_node",
    "params": {"node_id": "5"},
    "result": {
        "created_nodes": ["5_a", "5_b"],
        "removed_nodes": ["5"]
    },
    "created_at": "2024-01-22T10:00:00Z"
}
```

### Split Detection Response

```json
{
    "proposed_coverage": ["5", "6", "7"],
    "violations": [
        {
            "node_id": "5",
            "reason": "has_external_input_and_output",
            "external_inputs": [["-1", "5"]],
            "external_outputs": [["5", "10"]]
        }
    ],
    "suggested_resolution": [
        {
            "type": "split_node",
            "params": {"node_id": "5"}
        }
    ],
    "adjusted_coverage": ["5_a", "6", "7"]
}
```

### Node Classification Response

```json
{
    "coverage": ["5", "6", "7"],
    "classification": {
        "entry": ["5"],
        "intermediate": ["6"],
        "exit": ["7"]
    },
    "valid": true,
    "violations": []
}
```

## Implementation Phases

### Phase 1: Core API Infrastructure

1. Set up FastAPI application structure
2. Create API routes skeleton
3. Add database session management
4. Implement basic error handling

Files:
- `explaneat/api/__init__.py`
- `explaneat/api/app.py`
- `explaneat/api/routes/genomes.py`
- `explaneat/api/routes/operations.py`
- `explaneat/api/schemas.py`

### Phase 2: Model State Engine

1. Implement `ModelStateEngine` class
   - Load phenotype from genome
   - Apply operations sequentially
   - Return current model state

2. Implement operation handlers:
   - `apply_split_node()`
   - `apply_consolidate_node()`
   - `apply_remove_node()`
   - `apply_add_node()`
   - `apply_add_identity_node()`
   - `apply_annotate()`

Files:
- `explaneat/core/model_state.py`
- `explaneat/core/operations.py`

### Phase 3: Validation & Analysis

1. Implement split detection algorithm
2. Implement node classification (entry/intermediate/exit)
3. Implement annotation validation
4. Implement coverage computation on new model

Files:
- `explaneat/analysis/split_detection.py`
- `explaneat/analysis/node_classification.py`
- Updates to existing `coverage.py`

### Phase 4: Database Updates

1. Add `operations` JSON column to Explanation table
2. Create migration
3. Update ExplanationManager for new model

Files:
- `alembic/versions/xxx_add_operations_column.py`
- Updates to `explaneat/db/models.py`
- Updates to `explaneat/analysis/explanation_manager.py`

### Phase 5: Node.js Frontend

1. Set up React + TypeScript project
2. Implement graph visualization (Cytoscape.js or D3)
3. Create operation controls UI
4. Implement operation history panel

Directory: `frontend/`

## File Structure

```
explaneat/
├── api/
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   ├── dependencies.py     # Dependency injection
│   ├── schemas.py          # Pydantic models
│   └── routes/
│       ├── __init__.py
│       ├── genomes.py
│       ├── operations.py
│       └── analysis.py
├── core/
│   ├── model_state.py      # ModelStateEngine
│   └── operations.py       # Operation handlers
├── analysis/
│   ├── split_detection.py  # NEW
│   ├── node_classification.py  # NEW
│   └── ... (existing)
└── ...

frontend/                   # NEW
├── package.json
├── src/
│   ├── App.tsx
│   ├── api/
│   │   └── client.ts       # API client
│   ├── components/
│   │   ├── GraphViewer.tsx
│   │   ├── OperationPanel.tsx
│   │   └── OperationHistory.tsx
│   └── types/
│       └── index.ts        # TypeScript types matching API schemas
└── ...
```

## Running Locally

```bash
# Terminal 1: Python API
uv run uvicorn explaneat.api.app:app --reload --port 8000

# Terminal 2: Node.js frontend
cd frontend && npm run dev
```

## Migration Strategy

The existing Flask web viewer and CLI remain functional. The new API is additive:

1. Existing code continues to work
2. New API provides enhanced functionality
3. Frontend is optional - API can be used with CLI tools too
4. Gradual migration of features as needed
