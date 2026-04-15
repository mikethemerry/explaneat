# Next Phases: Feature Names, Connection Ops, SHAP, Operation Notes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the explanation toolkit so that variable names flow through formulas/charts, connections can be added/removed, SHAP values provide variable-importance evidence, and operations carry human-readable notes — enabling a full explained model workflow on the backache dataset before moving to the adult dataset.

**Architecture:** Four sequential phases, each independently shippable. Phase 1 (variable names in viz) unblocks meaningful chart interpretation. Phase 2 (connection operations) enables model manipulation (removing inputs like ID). Phase 3 (operation notes) captures the researcher's rationale. Phase 4 (SHAP integration) provides variable-importance evidence to justify manipulations.

**Tech Stack:** Python (FastAPI, NumPy, SymPy, SHAP), React + TypeScript (Observable Plot), SQLAlchemy (JSONB operations)

---

## Phase 1: Feature Names in Viz Data

The viz functions (`viz_data.py`) currently emit labels like `x_0`, `x_1`. The annotation function's `to_sympy` already uses `get_display_map()` for formula rendering, but the viz pipeline does not pass entry node names through to chart labels. Fix: thread entry/exit node display names from the evidence route into viz functions.

### Task 1.1: Pass entry/exit names through viz_data functions

**Files:**
- Modify: `explaneat/analysis/viz_data.py` — all compute_* functions
- Test: `tests/test_analysis/test_viz_data_labels.py` (create)

- [ ] **Step 1: Write failing test for compute_line_plot with entry_names**

```python
"""Tests that viz_data functions use entry_names when provided."""
import numpy as np
from explaneat.analysis.viz_data import compute_line_plot, compute_heatmap, compute_sensitivity


def _dummy_fn(x):
    return x @ np.array([[1.0], [0.5]])


class TestVizDataLabels:
    def test_line_plot_uses_entry_names(self):
        entry = np.random.randn(50, 2)
        exit_ = _dummy_fn(entry)
        result = compute_line_plot(
            _dummy_fn, entry, exit_,
            input_dim=0, output_dim=0,
            entry_names=["pregWeight", "gestMonth"],
            exit_names=["backache"],
        )
        assert result["x_label"] == "pregWeight"
        assert result["y_label"] == "backache"

    def test_line_plot_falls_back_to_x_i(self):
        entry = np.random.randn(50, 2)
        exit_ = _dummy_fn(entry)
        result = compute_line_plot(_dummy_fn, entry, exit_, input_dim=1, output_dim=0)
        assert result["x_label"] == "x_1"
        assert result["y_label"] == "y_0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analysis/test_viz_data_labels.py -v`
Expected: FAIL — `compute_line_plot() got an unexpected keyword argument 'entry_names'`

- [ ] **Step 3: Add entry_names/exit_names params to compute_line_plot**

In `explaneat/analysis/viz_data.py`, update `compute_line_plot`:

```python
def compute_line_plot(
    fn: Callable,
    entry_acts: np.ndarray,
    exit_acts: np.ndarray,
    input_dim: int = 0,
    output_dim: int = 0,
    grid_points: int = 200,
    entry_names: Optional[List[str]] = None,
    exit_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # ... existing logic unchanged ...
    x_label = entry_names[input_dim] if entry_names and input_dim < len(entry_names) else f"x_{input_dim}"
    y_label = exit_names[output_dim] if exit_names and output_dim < len(exit_names) else f"y_{output_dim}"
    return {
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "scatter_x": entry_acts[:, input_dim].tolist(),
        "scatter_y": exit_acts[:, output_dim].tolist(),
        "x_label": x_label,
        "y_label": y_label,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_analysis/test_viz_data_labels.py::TestVizDataLabels::test_line_plot_uses_entry_names -v`
Expected: PASS

- [ ] **Step 5: Add entry_names/exit_names to remaining viz functions**

Apply the same pattern to `compute_heatmap`, `compute_partial_dependence`, `compute_pca_scatter`, and `compute_sensitivity`. Each function gets optional `entry_names: Optional[List[str]] = None` and `exit_names: Optional[List[str]] = None` params. In label generation, use the name list when available, falling back to `x_i` / `y_i`.

For `compute_sensitivity`, change:
```python
"input_labels": [entry_names[i] if entry_names and i < len(entry_names) else f"x_{i}" for i in range(n_in)],
```

- [ ] **Step 6: Write and run tests for heatmap and sensitivity with names**

```python
    def test_heatmap_uses_entry_names(self):
        entry = np.random.randn(50, 3)
        exit_ = entry[:, :1]
        result = compute_heatmap(
            lambda x: x[:, :1], entry, exit_,
            input_dims=(0, 2),
            entry_names=["age", "height", "weight"],
            exit_names=["risk"],
        )
        assert result["x_label"] == "age"
        assert result["y_label"] == "weight"
        assert result["z_label"] == "risk"

    def test_sensitivity_uses_entry_names(self):
        entry = np.random.randn(50, 2)
        result = compute_sensitivity(
            lambda x: x @ np.array([[1.0], [0.5]]),
            entry,
            entry_names=["pregWeight", "gestMonth"],
        )
        assert result["input_labels"] == ["pregWeight", "gestMonth"]
```

Run: `uv run pytest tests/test_analysis/test_viz_data_labels.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_analysis/test_viz_data_labels.py explaneat/analysis/viz_data.py
git commit -m "feat: add entry_names/exit_names to viz_data functions for labeled charts"
```

### Task 1.2: Thread display names from evidence route into viz functions

**Files:**
- Modify: `explaneat/api/routes/evidence.py` — `compute_viz_data` endpoint

- [ ] **Step 1: Read the compute_viz_data endpoint to understand current flow**

The endpoint already has `model_state` and `annotation`. We need to resolve entry/exit node IDs to display names using `model_state.get_display_map()`.

- [ ] **Step 2: Build entry/exit name lists and pass to viz functions**

In `explaneat/api/routes/evidence.py`, in `compute_viz_data`, after building `ann_fn` and `annotation`:

```python
        # Resolve entry/exit display names for chart labels
        display_map = model_state.get_display_map()
        entry_names = [display_map.get(n, n) for n in annotation["entry_nodes"]]
        exit_names = [display_map.get(n, n) for n in annotation["exit_nodes"]]
```

Then pass `entry_names=entry_names, exit_names=exit_names` to every `vd.compute_*` call. For `compute_sensitivity`, pass `entry_names=entry_names`.

- [ ] **Step 3: Test end-to-end by verifying viz responses include display names**

This is an integration test — verify manually via the API or write an API test if there's existing infrastructure.

- [ ] **Step 4: Commit**

```bash
git add explaneat/api/routes/evidence.py
git commit -m "feat: thread display names into viz data for labeled chart axes"
```

---

## Phase 2: Connection Operations (Disable/Remove)

Currently no operation exists to disable or remove individual connections. This is the critical gap for removing inputs like ID — you need to sever the connection(s) from an input node without removing the node itself (since INPUT nodes can't be removed).

### Task 2.1: Add disable_connection operation handler

**Files:**
- Modify: `explaneat/core/operations.py`
- Test: `tests/test_core/test_connection_ops.py` (create)

- [ ] **Step 1: Write failing test for disable_connection**

```python
"""Tests for connection-level operations."""
import pytest
from explaneat.core.genome_network import (
    NetworkStructure, NetworkNode, NetworkConnection, NodeType,
)
from explaneat.core.operations import apply_disable_connection, OperationError


def _simple_network():
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="sigmoid"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.1, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=0.5, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=0.8, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes, connections=connections,
        input_node_ids=["-1", "-2"], output_node_ids=["0"],
    )


class TestDisableConnection:
    def test_disable_existing_connection(self):
        net = _simple_network()
        result = apply_disable_connection(net, "-1", "5", set())
        assert result["from_node"] == "-1"
        assert result["to_node"] == "5"
        conn = [c for c in net.connections if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is False

    def test_disable_nonexistent_connection_raises(self):
        net = _simple_network()
        with pytest.raises(OperationError, match="not found"):
            apply_disable_connection(net, "-1", "0", set())

    def test_disable_covered_connection_raises(self):
        net = _simple_network()
        covered = {("-1", "5")}
        with pytest.raises(OperationError, match="covered"):
            apply_disable_connection(net, "-1", "5", covered)

    def test_disable_already_disabled_raises(self):
        net = _simple_network()
        apply_disable_connection(net, "-1", "5", set())
        with pytest.raises(OperationError, match="already disabled"):
            apply_disable_connection(net, "-1", "5", set())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_connection_ops.py -v`
Expected: FAIL — `cannot import name 'apply_disable_connection'`

- [ ] **Step 3: Implement apply_disable_connection**

In `explaneat/core/operations.py`, add:

```python
def apply_disable_connection(
    model: NetworkStructure,
    from_node: str,
    to_node: str,
    covered_connections: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """Disable a connection (set enabled=False).

    Raises OperationError if connection not found, already disabled, or covered.
    """
    if (from_node, to_node) in covered_connections:
        raise OperationError(
            f"Connection ({from_node} -> {to_node}) is covered by an annotation"
        )

    for conn in model.connections:
        if conn.from_node == from_node and conn.to_node == to_node:
            if not conn.enabled:
                raise OperationError(
                    f"Connection ({from_node} -> {to_node}) is already disabled"
                )
            conn.enabled = False
            return {"from_node": from_node, "to_node": to_node, "previous_weight": conn.weight}

    raise OperationError(f"Connection ({from_node} -> {to_node}) not found")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_connection_ops.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add explaneat/core/operations.py tests/test_core/test_connection_ops.py
git commit -m "feat: add apply_disable_connection operation handler"
```

### Task 2.2: Add enable_connection operation handler

**Files:**
- Modify: `explaneat/core/operations.py`
- Modify: `tests/test_core/test_connection_ops.py`

- [ ] **Step 1: Write failing test for enable_connection**

```python
class TestEnableConnection:
    def test_enable_disabled_connection(self):
        net = _simple_network()
        apply_disable_connection(net, "-1", "5", set())
        result = apply_enable_connection(net, "-1", "5", set())
        assert result["from_node"] == "-1"
        conn = [c for c in net.connections if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is True

    def test_enable_already_enabled_raises(self):
        net = _simple_network()
        with pytest.raises(OperationError, match="already enabled"):
            apply_enable_connection(net, "-1", "5", set())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_connection_ops.py::TestEnableConnection -v`
Expected: FAIL

- [ ] **Step 3: Implement apply_enable_connection**

```python
def apply_enable_connection(
    model: NetworkStructure,
    from_node: str,
    to_node: str,
    covered_connections: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """Re-enable a previously disabled connection."""
    if (from_node, to_node) in covered_connections:
        raise OperationError(
            f"Connection ({from_node} -> {to_node}) is covered by an annotation"
        )

    for conn in model.connections:
        if conn.from_node == from_node and conn.to_node == to_node:
            if conn.enabled:
                raise OperationError(
                    f"Connection ({from_node} -> {to_node}) is already enabled"
                )
            conn.enabled = True
            return {"from_node": from_node, "to_node": to_node}

    raise OperationError(f"Connection ({from_node} -> {to_node}) not found")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_connection_ops.py::TestEnableConnection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add explaneat/core/operations.py tests/test_core/test_connection_ops.py
git commit -m "feat: add apply_enable_connection operation handler"
```

### Task 2.3: Wire connection operations into ModelStateEngine

**Files:**
- Modify: `explaneat/core/model_state.py`
- Modify: `tests/test_core/test_connection_ops.py`

- [ ] **Step 1: Write failing test for engine integration**

```python
from explaneat.core.model_state import ModelStateEngine


class TestConnectionOpsInEngine:
    def test_disable_via_engine(self):
        net = _simple_network()
        engine = ModelStateEngine(net)
        op = engine.add_operation("disable_connection", {"from_node": "-1", "to_node": "5"})
        assert op.type == "disable_connection"
        conn = [c for c in engine.current_state.connections
                if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is False

    def test_undo_disable_re_enables(self):
        net = _simple_network()
        engine = ModelStateEngine(net)
        op = engine.add_operation("disable_connection", {"from_node": "-1", "to_node": "5"})
        engine.remove_operation(op.seq)
        conn = [c for c in engine.current_state.connections
                if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is True

    def test_enable_via_engine(self):
        net = _simple_network()
        engine = ModelStateEngine(net)
        engine.add_operation("disable_connection", {"from_node": "-1", "to_node": "5"})
        engine.add_operation("enable_connection", {"from_node": "-1", "to_node": "5"})
        conn = [c for c in engine.current_state.connections
                if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_connection_ops.py::TestConnectionOpsInEngine -v`
Expected: FAIL — engine doesn't know about `disable_connection`

- [ ] **Step 3: Add connection ops to ModelStateEngine._apply_operation**

In `explaneat/core/model_state.py`, in the `_apply_operation` method (the dispatch switch), add:

```python
elif op_type == "disable_connection":
    from .operations import apply_disable_connection
    result = apply_disable_connection(
        self._current_state, params["from_node"], params["to_node"],
        self._covered_connections,
    )
elif op_type == "enable_connection":
    from .operations import apply_enable_connection
    result = apply_enable_connection(
        self._current_state, params["from_node"], params["to_node"],
        self._covered_connections,
    )
```

Also add these types to `validate_operation` in `operations.py`:

```python
elif op_type == "disable_connection":
    # Validate connection exists and is enabled
    ...
elif op_type == "enable_connection":
    # Validate connection exists and is disabled
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_connection_ops.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add explaneat/core/model_state.py explaneat/core/operations.py tests/test_core/test_connection_ops.py
git commit -m "feat: wire disable/enable_connection into ModelStateEngine"
```

### Task 2.4: Add API schemas and route for connection operations

**Files:**
- Modify: `explaneat/api/schemas.py`
- Modify: `explaneat/api/routes/operations.py`

- [ ] **Step 1: Add DisableConnectionParams and EnableConnectionParams schemas**

In `explaneat/api/schemas.py`:

```python
class DisableConnectionParams(BaseModel):
    """Parameters for disable_connection operation."""
    from_node: str
    to_node: str


class EnableConnectionParams(BaseModel):
    """Parameters for enable_connection operation."""
    from_node: str
    to_node: str
```

Add `"disable_connection"` and `"enable_connection"` to `OperationRequest.type` Literal. Add the params classes to the `params` Union.

- [ ] **Step 2: Update operations route docstring**

In `explaneat/api/routes/operations.py`, update the `add_operation` docstring to list the new operation types.

- [ ] **Step 3: Test via curl or API test**

```bash
# Manually test with a real genome_id
curl -X POST http://localhost:8000/api/genomes/{genome_id}/operations \
  -H "Content-Type: application/json" \
  -d '{"type": "disable_connection", "params": {"from_node": "-1", "to_node": "5"}}'
```

- [ ] **Step 4: Commit**

```bash
git add explaneat/api/schemas.py explaneat/api/routes/operations.py
git commit -m "feat: add disable/enable_connection API schemas and route support"
```

### Task 2.5: Frontend UI for connection operations

**Files:**
- Modify: `web/react-explorer/src/api/client.ts` — no changes needed if OperationRequest already accepts arbitrary params
- Modify: `web/react-explorer/src/components/OperationsPanel.tsx` or `NetworkViewer.tsx`

- [ ] **Step 1: Add connection click handler to NetworkViewer**

When a user clicks a connection (edge) in the React Flow graph, show a context menu with "Disable connection" / "Enable connection" options.

- [ ] **Step 2: Wire click handler to addOperation API call**

```typescript
const handleDisableConnection = async (fromNode: string, toNode: string) => {
  await addOperation(genomeId, {
    type: "disable_connection",
    params: { from_node: fromNode, to_node: toNode },
  });
  onOperationChange();
};
```

- [ ] **Step 3: Visual feedback for disabled connections**

Render disabled connections with dashed stroke and reduced opacity in the network graph.

- [ ] **Step 4: Commit**

```bash
git add web/react-explorer/src/components/NetworkViewer.tsx web/react-explorer/src/components/OperationsPanel.tsx
git commit -m "feat: frontend UI for disable/enable connection operations"
```

---

## Phase 3: Operation Notes

Add a `notes` field to operations so the researcher can record why they performed each manipulation (e.g., "Removed ID input — not a plausible predictor variable").

### Task 3.1: Add notes field to Operation dataclass and serialization

**Files:**
- Modify: `explaneat/core/model_state.py`
- Test: `tests/test_core/test_operation_notes.py` (create)

- [ ] **Step 1: Write failing test**

```python
"""Tests for operation notes field."""
from explaneat.core.model_state import ModelStateEngine, Operation
from explaneat.core.genome_network import (
    NetworkStructure, NetworkNode, NetworkConnection, NodeType,
)


def _simple_structure():
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="0", weight=1.0, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes, connections=connections,
        input_node_ids=["-1"], output_node_ids=["0"],
    )


class TestOperationNotes:
    def test_add_operation_with_notes(self):
        engine = ModelStateEngine(_simple_structure())
        op = engine.add_operation(
            "rename_node",
            {"node_id": "-1", "display_name": "age"},
            notes="Mapped from dataset feature column 0",
        )
        assert op.notes == "Mapped from dataset feature column 0"

    def test_notes_round_trip_serialization(self):
        engine = ModelStateEngine(_simple_structure())
        engine.add_operation(
            "rename_node",
            {"node_id": "-1", "display_name": "age"},
            notes="Dataset column mapping",
        )
        data = engine.to_dict()
        assert data["operations"][0]["notes"] == "Dataset column mapping"

        engine2 = ModelStateEngine(_simple_structure())
        engine2.load_operations(data)
        assert engine2.operations[0].notes == "Dataset column mapping"

    def test_notes_default_none(self):
        engine = ModelStateEngine(_simple_structure())
        op = engine.add_operation("rename_node", {"node_id": "-1", "display_name": "age"})
        assert op.notes is None

    def test_notes_omitted_from_serialization_when_none(self):
        engine = ModelStateEngine(_simple_structure())
        engine.add_operation("rename_node", {"node_id": "-1", "display_name": "age"})
        data = engine.to_dict()
        assert "notes" not in data["operations"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_operation_notes.py -v`
Expected: FAIL — `add_operation() got an unexpected keyword argument 'notes'`

- [ ] **Step 3: Add notes field to Operation dataclass**

In `explaneat/core/model_state.py`, update `Operation`:

```python
@dataclass
class Operation:
    seq: int
    type: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
```

Update `to_dict()` to include notes when not None:
```python
def to_dict(self) -> Dict[str, Any]:
    d = {
        "seq": self.seq,
        "type": self.type,
        "params": self.params,
        "result": self.result,
        "created_at": self.created_at.isoformat() if self.created_at else None,
    }
    if self.notes is not None:
        d["notes"] = self.notes
    return d
```

Update `from_dict()` to read notes:
```python
notes=data.get("notes"),
```

Update `add_operation()` signature to accept `notes: Optional[str] = None` and pass it through.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_operation_notes.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add explaneat/core/model_state.py tests/test_core/test_operation_notes.py
git commit -m "feat: add notes field to Operation dataclass for researcher rationale"
```

### Task 3.2: Add notes to API schemas and operation route

**Files:**
- Modify: `explaneat/api/schemas.py`
- Modify: `explaneat/api/routes/operations.py`

- [ ] **Step 1: Add notes to OperationRequest and OperationResponse**

In `explaneat/api/schemas.py`:

```python
class OperationRequest(BaseModel):
    type: Literal[...]
    params: Union[...]
    notes: Optional[str] = None  # Human-readable justification
```

```python
class OperationResponse(BaseModel):
    seq: int
    type: str
    params: Dict[str, Any]
    result: Optional[OperationResult] = None
    created_at: datetime
    notes: Optional[str] = None
```

- [ ] **Step 2: Pass notes through in the operations route**

In `explaneat/api/routes/operations.py`, in `add_operation`:

```python
new_op = engine.add_operation(operation.type, params, validate=True, notes=operation.notes)
```

In `_operation_to_response`:
```python
return OperationResponse(
    seq=op.seq,
    type=op.type,
    params=op.params,
    result=OperationResult(**op.result) if op.result else None,
    created_at=op.created_at,
    notes=op.notes,
)
```

In `list_operations`, read notes from stored ops:
```python
operations.append(OperationResponse(
    ...
    notes=op_data.get("notes"),
))
```

- [ ] **Step 3: Commit**

```bash
git add explaneat/api/schemas.py explaneat/api/routes/operations.py
git commit -m "feat: add notes field to operation API request/response"
```

### Task 3.3: Frontend operation history with notes

**Files:**
- Modify: `web/react-explorer/src/components/OperationsPanel.tsx` or create `OperationHistory.tsx`

- [ ] **Step 1: Add operation history list showing notes**

Display a collapsible list of all operations with their type, seq, params summary, timestamp, and notes (if present).

- [ ] **Step 2: Add notes input when performing operations**

When the user performs an operation (split, disable connection, etc.), show an optional text field for notes.

- [ ] **Step 3: Commit**

```bash
git add web/react-explorer/src/components/
git commit -m "feat: frontend operation history panel with notes display and input"
```

---

## Phase 4: SHAP Integration

Add SHAP (SHapley Additive exPlanations) as an evidence type. SHAP values show variable importance — critical for justifying whether to keep or remove inputs like ID. The SHAP library wraps our model as a black-box predictor.

### Task 4.1: Add shap dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add shap to dependencies**

```toml
    # XAI evidence
    "shap>=0.43.0",
```

- [ ] **Step 2: Install**

Run: `uv pip install -e .`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add shap for variable importance evidence"
```

### Task 4.2: Create SHAP analysis module

**Files:**
- Create: `explaneat/analysis/shap_analysis.py`
- Test: `tests/test_analysis/test_shap_analysis.py` (create)

- [ ] **Step 1: Write failing test**

```python
"""Tests for SHAP analysis integration."""
import numpy as np
import pytest
from explaneat.analysis.shap_analysis import compute_shap_values


class TestShapAnalysis:
    def test_compute_shap_returns_values_and_labels(self):
        """SHAP values should be computed for a simple linear model."""
        X = np.random.randn(100, 3)
        # Simple linear model: y = 2*x0 + 0*x1 + 1*x2
        def predict(x):
            return x @ np.array([2.0, 0.0, 1.0])

        result = compute_shap_values(
            predict_fn=predict,
            X=X,
            feature_names=["pregWeight", "ID", "gestMonth"],
        )
        assert "shap_values" in result
        assert "feature_names" in result
        assert "mean_abs_shap" in result
        assert len(result["feature_names"]) == 3
        assert len(result["mean_abs_shap"]) == 3
        # pregWeight (coeff=2) should have highest importance
        assert result["mean_abs_shap"][0] > result["mean_abs_shap"][1]

    def test_feature_names_default_to_indices(self):
        X = np.random.randn(50, 2)
        result = compute_shap_values(lambda x: x[:, 0], X)
        assert result["feature_names"] == ["x_0", "x_1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analysis/test_shap_analysis.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement compute_shap_values**

```python
"""SHAP analysis for ExplaNEAT models.

Computes Shapley values to assess variable importance — complementary
to the structural explanation. Helps identify variables that can be
safely removed vs. those that are critical to the model.
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def compute_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP values for a prediction function.

    Args:
        predict_fn: (n_samples, n_features) -> (n_samples,) or (n_samples, n_out)
        X: Background/reference data for SHAP
        feature_names: Names for each feature column
        max_samples: Max background samples for KernelExplainer

    Returns:
        Dict with shap_values (n_samples x n_features), feature_names,
        mean_abs_shap (per-feature importance), and base_value.
    """
    import shap

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    # Subsample background if needed
    bg = X if len(X) <= max_samples else shap.sample(X, max_samples)

    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(X)

    # Handle multi-output: take first output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.mean(np.abs(shap_values), axis=0).tolist()

    return {
        "shap_values": shap_values.tolist(),
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs,
        "base_value": float(explainer.expected_value)
        if not isinstance(explainer.expected_value, np.ndarray)
        else float(explainer.expected_value[0]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_analysis/test_shap_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add explaneat/analysis/shap_analysis.py tests/test_analysis/test_shap_analysis.py
git commit -m "feat: add SHAP analysis module for variable importance evidence"
```

### Task 4.3: Add SHAP API endpoint

**Files:**
- Modify: `explaneat/api/schemas.py`
- Modify: `explaneat/api/routes/evidence.py`

- [ ] **Step 1: Add ShapRequest and ShapResponse schemas**

```python
class ShapRequest(BaseModel):
    """Request for SHAP value computation."""
    dataset_split_id: str
    annotation_id: Optional[str] = None  # None = whole model
    split: Literal["train", "test", "both"] = "both"
    max_samples: int = 100


class ShapResponse(BaseModel):
    """Response with SHAP values."""
    feature_names: List[str]
    mean_abs_shap: List[float]
    base_value: float
    # Full shap_values omitted for size — provide summary only
```

- [ ] **Step 2: Add endpoint**

In `explaneat/api/routes/evidence.py`:

```python
@router.post("/shap", response_model=ShapResponse)
async def compute_shap(
    request: ShapRequest,
    genome_id: str = Path(...),
):
    """Compute SHAP values for the model or a specific annotation subgraph."""
    with db.session_scope() as session:
        engine = _build_engine(session, genome_id)
        model_state = engine.current_state
        X, y = _load_split_data(
            session, request.dataset_split_id, request.split,
            sample_fraction=1.0, max_samples=request.max_samples,
        )

        display_map = model_state.get_display_map()

        if request.annotation_id:
            annotation = _find_annotation_in_operations(session, genome_id, request.annotation_id)
            ann_fn = AnnotationFunction.from_structure(annotation, model_state)
            extractor = ActivationExtractor.from_structure(model_state)
            entry_acts, _ = extractor.extract(X, annotation)
            feature_names = [display_map.get(n, n) for n in annotation["entry_nodes"]]

            from ..analysis.shap_analysis import compute_shap_values
            result = compute_shap_values(ann_fn, entry_acts, feature_names, request.max_samples)
        else:
            # Whole model SHAP
            from ..core.structure_network import StructureNetwork
            struct_net = StructureNetwork(model_state)
            feature_names = [display_map.get(n, n) for n in model_state.input_node_ids]

            def model_predict(x):
                return struct_net.forward(x)

            from ..analysis.shap_analysis import compute_shap_values
            result = compute_shap_values(model_predict, X, feature_names, request.max_samples)

        return ShapResponse(
            feature_names=result["feature_names"],
            mean_abs_shap=result["mean_abs_shap"],
            base_value=result["base_value"],
        )
```

- [ ] **Step 3: Commit**

```bash
git add explaneat/api/schemas.py explaneat/api/routes/evidence.py
git commit -m "feat: add SHAP values API endpoint for variable importance"
```

### Task 4.4: Frontend SHAP visualization

**Files:**
- Modify: `web/react-explorer/src/components/VizCanvas.tsx`
- Modify: `web/react-explorer/src/components/EvidencePanel.tsx`
- Modify: `web/react-explorer/src/api/client.ts`

- [ ] **Step 1: Add SHAP API client function**

```typescript
export async function computeShap(
  genomeId: string,
  params: { dataset_split_id: string; annotation_id?: string; split?: string; max_samples?: number }
): Promise<{ feature_names: string[]; mean_abs_shap: number[]; base_value: number }> {
  const resp = await fetch(`${API_BASE}/genomes/${genomeId}/evidence/shap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}
```

- [ ] **Step 2: Add SHAP bar chart renderer to VizCanvas**

Render a horizontal bar chart with feature names on y-axis and mean |SHAP value| on x-axis, sorted by importance.

- [ ] **Step 3: Add "SHAP Values" button to EvidencePanel**

Add a button or tab in the evidence panel that triggers SHAP computation and displays the result.

- [ ] **Step 4: Commit**

```bash
git add web/react-explorer/src/
git commit -m "feat: frontend SHAP visualization with importance bar chart"
```

---

## Implementation Order & Dependencies

```
Phase 1 (viz labels)  ─┐
                        ├─> Phase 3 (notes)  ─> Phase 5 (adult dataset - future)
Phase 2 (connections) ─┘
                        └─> Phase 4 (SHAP)
```

Phases 1 and 2 are independent and can be done in parallel. Phase 3 (notes) builds on the operation infrastructure from Phase 2. Phase 4 (SHAP) is independent but most valuable after Phase 2 exists (so you can compute SHAP, then disable connections, with notes explaining why).

## Timeline Context

From the meeting: roughly 4 weeks before the Europe trip. Target is to have all four phases done before presenting to the cardiovascular risk team. Phase 1 is quickest (~1 day). Phase 2 is the most impactful (~2 days). Phase 3 is straightforward (~half day). Phase 4 requires the SHAP library integration (~1-2 days).
