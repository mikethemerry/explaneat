# One-Hot Encoding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto one-hot encode categorical/ordinal features, creating prepared datasets with proper naming, and add a source/network view toggle in the evidence panel.

**Architecture:** A `prepare_dataset()` engine reads `feature_types` and `encoding_config` from a source Dataset, expands categoricals into `feature:value` columns, and saves a new Dataset linked via `source_dataset_id`. The evidence panel adds a view toggle that reverses encoding/scaling using the stored config.

**Tech Stack:** Python/SQLAlchemy/NumPy for encoding, Alembic for migration, React/TypeScript for UI.

---

### Task 1: Database Migration — Add source_dataset_id and encoding_config to Dataset

**Files:**
- Create: `alembic/versions/i2j3k4l5m6n7_add_dataset_encoding_fields.py`
- Modify: `explaneat/db/models.py:43-75`

**Step 1: Add columns to Dataset model**

In `explaneat/db/models.py`, add after the existing `dataset_id` / `version` columns (around line 55):

```python
source_dataset_id = Column(
    UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE")
)
encoding_config = Column(JSONB)  # Records how this prepared dataset was created
```

Add relationship:

```python
source_dataset = relationship("Dataset", remote_side="Dataset.id", foreign_keys=[source_dataset_id])
```

**Step 2: Create migration**

Create `alembic/versions/i2j3k4l5m6n7_add_dataset_encoding_fields.py`:

```python
"""add encoding fields to datasets

Revision ID: i2j3k4l5m6n7
Revises: h1i2j3k4l5m6
Create Date: 2026-04-15
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'i2j3k4l5m6n7'
down_revision: Union[str, None] = 'h1i2j3k4l5m6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column('datasets', sa.Column('source_dataset_id', sa.UUID(), nullable=True))
    op.add_column('datasets', sa.Column('encoding_config', postgresql.JSONB(), nullable=True))
    op.create_foreign_key(
        'fk_datasets_source_dataset_id', 'datasets', 'datasets',
        ['source_dataset_id'], ['id'], ondelete='CASCADE'
    )

def downgrade() -> None:
    op.drop_constraint('fk_datasets_source_dataset_id', 'datasets', type_='foreignkey')
    op.drop_column('datasets', 'encoding_config')
    op.drop_column('datasets', 'source_dataset_id')
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ -x -q`
Expected: All pass (model changes are additive)

**Step 4: Commit**

```bash
git add explaneat/db/models.py alembic/versions/i2j3k4l5m6n7_add_dataset_encoding_fields.py
git commit -m "feat: add source_dataset_id and encoding_config to Dataset model"
```

---

### Task 2: Encoding Engine — prepare_dataset()

**Files:**
- Create: `explaneat/db/encoding.py`
- Create: `tests/test_db/test_encoding.py`

**Step 1: Write failing tests**

Create `tests/test_db/test_encoding.py`:

```python
"""Tests for dataset one-hot encoding."""
import numpy as np
import pytest
from explaneat.db.encoding import prepare_dataset_arrays


class TestPrepareDatasetArrays:
    """Test the pure encoding function (no DB)."""

    def test_passthrough_numeric(self):
        """Numeric features pass through unchanged."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        feature_names = ["age", "income"]
        feature_types = {"age": "numeric", "income": "numeric"}
        encoding_config = {"passthrough": ["age", "income"]}

        X_enc, names_enc, types_enc = prepare_dataset_arrays(
            X, feature_names, feature_types, encoding_config
        )

        np.testing.assert_array_equal(X_enc, X)
        assert names_enc == ["age", "income"]
        assert all(t == "numeric" for t in types_enc.values())

    def test_categorical_one_hot(self):
        """Categorical features are one-hot encoded with feature:value naming."""
        X = np.array([[0, 10.0], [1, 20.0], [2, 30.0], [0, 40.0]])
        feature_names = ["color", "size"]
        feature_types = {"color": "categorical", "size": "numeric"}
        encoding_config = {
            "categorical": {"color": ["red", "green", "blue"]},
            "passthrough": ["size"],
        }

        X_enc, names_enc, types_enc = prepare_dataset_arrays(
            X, feature_names, feature_types, encoding_config
        )

        assert X_enc.shape == (4, 4)  # 3 one-hot + 1 numeric
        assert names_enc == ["color:red", "color:green", "color:blue", "size"]
        # Row 0: color=0 (red) → [1,0,0]
        np.testing.assert_array_equal(X_enc[0], [1, 0, 0, 10.0])
        # Row 2: color=2 (blue) → [0,0,1]
        np.testing.assert_array_equal(X_enc[2], [0, 0, 1, 30.0])
        assert types_enc["color:red"] == "binary"

    def test_ordinal_as_ranked(self):
        """Ordinal features are mapped to sequential integers by default."""
        X = np.array([[2], [0], [1], [2]])
        feature_names = ["edu"]
        feature_types = {"edu": "ordinal"}
        encoding_config = {
            "ordinal_as_ranked": {"edu": ["low", "mid", "high"]},
        }

        X_enc, names_enc, types_enc = prepare_dataset_arrays(
            X, feature_names, feature_types, encoding_config
        )

        assert X_enc.shape == (4, 1)
        assert names_enc == ["edu"]
        # Values should be 0, 1, 2 mapping to the rank order
        np.testing.assert_array_equal(X_enc[:, 0], [2, 0, 1, 2])

    def test_ordinal_as_onehot(self):
        """Ordinal features can optionally be one-hot encoded."""
        X = np.array([[0], [1], [2]])
        feature_names = ["edu"]
        feature_types = {"edu": "ordinal"}
        encoding_config = {
            "ordinal_as_onehot": {"edu": ["low", "mid", "high"]},
        }

        X_enc, names_enc, types_enc = prepare_dataset_arrays(
            X, feature_names, feature_types, encoding_config
        )

        assert X_enc.shape == (3, 3)
        assert names_enc == ["edu:low", "edu:mid", "edu:high"]
        np.testing.assert_array_equal(X_enc[0], [1, 0, 0])

    def test_mixed_features(self):
        """Mixed feature types are handled in correct column order."""
        X = np.array([
            [25, 0, 1, 50000],
            [30, 1, 0, 60000],
        ])
        feature_names = ["age", "workclass", "edu", "income"]
        feature_types = {
            "age": "numeric",
            "workclass": "categorical",
            "edu": "ordinal",
            "income": "numeric",
        }
        encoding_config = {
            "categorical": {"workclass": ["private", "gov"]},
            "ordinal_as_ranked": {"edu": ["low", "high"]},
            "passthrough": ["age", "income"],
        }

        X_enc, names_enc, types_enc = prepare_dataset_arrays(
            X, feature_names, feature_types, encoding_config
        )

        # age + workclass:private + workclass:gov + edu + income = 5
        assert X_enc.shape == (2, 5)
        assert names_enc == ["age", "workclass:private", "workclass:gov", "edu", "income"]

    def test_unknown_category_gets_all_zeros(self):
        """Values not in the category list produce all-zero one-hot rows."""
        X = np.array([[0], [1], [99]])  # 99 is unknown
        feature_names = ["color"]
        feature_types = {"color": "categorical"}
        encoding_config = {
            "categorical": {"color": ["red", "green"]},
        }

        X_enc, names_enc, types_enc = prepare_dataset_arrays(
            X, feature_names, feature_types, encoding_config
        )

        assert X_enc.shape == (3, 2)
        np.testing.assert_array_equal(X_enc[2], [0, 0])


class TestBuildEncodingConfig:
    """Test automatic encoding config generation from feature_types."""

    def test_auto_config_from_types(self):
        from explaneat.db.encoding import build_encoding_config

        X = np.array([[0, 1.5, 0], [1, 2.5, 1], [2, 3.5, 2]])
        feature_names = ["color", "height", "edu"]
        feature_types = {"color": "categorical", "height": "numeric", "edu": "ordinal"}

        config = build_encoding_config(X, feature_names, feature_types)

        assert "height" in config["passthrough"]
        assert "color" in config["categorical"]
        assert len(config["categorical"]["color"]) == 3  # 3 unique values
        assert "edu" in config["ordinal_as_ranked"]
        assert len(config["ordinal_as_ranked"]["edu"]) == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db/test_encoding.py -v`
Expected: FAIL — module not found

**Step 3: Implement encoding engine**

Create `explaneat/db/encoding.py`:

```python
"""Dataset encoding engine — one-hot encode categorical/ordinal features."""

from typing import Dict, List, Optional, Tuple

import numpy as np


def build_encoding_config(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[str, str],
    ordinal_onehot: Optional[List[str]] = None,
    ordinal_orders: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Build an encoding config from feature metadata.

    Auto-detects unique values for categorical/ordinal columns.
    Uses integer-label ordering by default unless ordinal_orders is provided.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Column names
        feature_types: Mapping of feature name → type
        ordinal_onehot: List of ordinal features to one-hot encode (instead of rank)
        ordinal_orders: Explicit orderings for ordinal features
    """
    ordinal_onehot = set(ordinal_onehot or [])
    ordinal_orders = ordinal_orders or {}
    config: Dict = {
        "categorical": {},
        "ordinal_as_ranked": {},
        "ordinal_as_onehot": {},
        "passthrough": [],
    }

    for i, name in enumerate(feature_names):
        ftype = feature_types.get(name, "numeric")

        if ftype == "categorical":
            unique_vals = sorted(np.unique(X[:, i]).astype(int))
            config["categorical"][name] = [str(v) for v in unique_vals]

        elif ftype == "ordinal":
            if name in ordinal_orders:
                order = ordinal_orders[name]
            else:
                order = [str(v) for v in sorted(np.unique(X[:, i]).astype(int))]

            if name in ordinal_onehot:
                config["ordinal_as_onehot"][name] = order
            else:
                config["ordinal_as_ranked"][name] = order

        else:
            config["passthrough"].append(name)

    return config


def prepare_dataset_arrays(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[str, str],
    encoding_config: Dict,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Apply one-hot encoding to a feature matrix.

    Args:
        X: Raw feature matrix (n_samples, n_features)
        feature_names: Original column names
        feature_types: Original type metadata
        encoding_config: Encoding decisions (from build_encoding_config)

    Returns:
        (X_encoded, new_feature_names, new_feature_types)
    """
    categorical = encoding_config.get("categorical", {})
    ordinal_ranked = encoding_config.get("ordinal_as_ranked", {})
    ordinal_onehot = encoding_config.get("ordinal_as_onehot", {})
    passthrough = set(encoding_config.get("passthrough", []))

    columns = []
    new_names = []
    new_types = {}

    for i, name in enumerate(feature_names):
        col = X[:, i]

        if name in categorical or name in ordinal_onehot:
            categories = categorical.get(name) or ordinal_onehot.get(name)
            # One-hot encode
            for cat in categories:
                ohe_name = f"{name}:{cat}"
                ohe_col = (col == int(cat) if cat.lstrip("-").isdigit()
                           else col == float(cat)).astype(np.float64)
                columns.append(ohe_col)
                new_names.append(ohe_name)
                new_types[ohe_name] = "binary"

        elif name in ordinal_ranked:
            order = ordinal_ranked[name]
            # Map to rank indices
            val_to_rank = {int(v) if v.lstrip("-").isdigit() else v: rank
                           for rank, v in enumerate(order)}
            ranked = np.array([val_to_rank.get(int(v), v) for v in col],
                              dtype=np.float64)
            columns.append(ranked)
            new_names.append(name)
            new_types[name] = "ordinal"

        else:
            # Passthrough
            columns.append(col.astype(np.float64))
            new_names.append(name)
            new_types[name] = feature_types.get(name, "numeric")

    X_encoded = np.column_stack(columns) if columns else np.empty((X.shape[0], 0))
    return X_encoded, new_names, new_types
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_db/test_encoding.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add explaneat/db/encoding.py tests/test_db/test_encoding.py
git commit -m "feat: encoding engine for one-hot and ordinal feature preparation"
```

---

### Task 3: API — Prepare Dataset Endpoint

**Files:**
- Modify: `explaneat/api/routes/datasets.py:95-143`
- Modify: `explaneat/api/schemas.py:439-480`
- Modify: `web/react-explorer/src/api/client.ts:553-589`

**Step 1: Add schemas**

In `explaneat/api/schemas.py`, add after `DatasetUpdateRequest`:

```python
class PrepareDatasetRequest(BaseModel):
    """Request to create a prepared (one-hot encoded) dataset."""
    name: Optional[str] = None  # Override name, default: "{source_name} (prepared)"
    encoding_config: Optional[Dict[str, Any]] = None  # Auto-build if not provided
    ordinal_onehot: Optional[List[str]] = None  # Ordinal features to one-hot encode
    ordinal_orders: Optional[Dict[str, List[str]]] = None  # Explicit ordinal orderings
```

Update `DatasetResponse` to include new fields:

```python
# Add to DatasetResponse class:
source_dataset_id: Optional[str] = None
encoding_config: Optional[Dict[str, Any]] = None
```

**Step 2: Add API endpoint**

In `explaneat/api/routes/datasets.py`, add after `list_splits`:

```python
@router.post("/{dataset_id}/prepare", response_model=DatasetResponse)
async def prepare_dataset(dataset_id: str, request: PrepareDatasetRequest):
    """Create a prepared dataset with one-hot encoded categorical features."""
    from ...db.encoding import build_encoding_config, prepare_dataset_arrays

    with db.session_scope() as session:
        dataset = session.query(Dataset).filter_by(id=uuid.UUID(dataset_id)).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        data = dataset.get_data()
        if data is None:
            raise HTTPException(status_code=400, detail="Dataset has no stored data")
        X, y = data

        feature_names = dataset.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        feature_types = dataset.feature_types or {}

        # Build encoding config if not provided
        encoding_config = request.encoding_config
        if encoding_config is None:
            encoding_config = build_encoding_config(
                X, feature_names, feature_types,
                ordinal_onehot=request.ordinal_onehot,
                ordinal_orders=request.ordinal_orders,
            )

        # Check if matching prepared dataset already exists
        existing = (
            session.query(Dataset)
            .filter_by(source_dataset_id=dataset.id)
            .all()
        )
        for ex in existing:
            if ex.encoding_config == encoding_config:
                session.expunge(ex)
                return _dataset_to_response(ex)

        # Apply encoding
        X_enc, new_names, new_types = prepare_dataset_arrays(
            X, feature_names, feature_types, encoding_config
        )

        # Create prepared dataset
        prep_name = request.name or f"{dataset.name} (prepared)"
        prepared = Dataset(
            name=prep_name,
            source=dataset.source,
            version=dataset.version,
            description=f"Prepared from {dataset.name} with one-hot encoding",
            num_samples=X_enc.shape[0],
            num_features=X_enc.shape[1],
            feature_names=new_names,
            feature_types=new_types,
            feature_descriptions=dataset.feature_descriptions,
            target_name=dataset.target_name,
            target_description=dataset.target_description,
            task_type=dataset.task_type,
            num_classes=dataset.num_classes,
            class_names=dataset.class_names,
            source_dataset_id=dataset.id,
            encoding_config=encoding_config,
        )
        prepared.set_data(X_enc, y)
        session.add(prepared)
        session.flush()
        session.expunge(prepared)
        return _dataset_to_response(prepared)
```

Update `_dataset_to_response` to include new fields:

```python
# Add to the DatasetResponse construction:
source_dataset_id=str(dataset.source_dataset_id) if dataset.source_dataset_id else None,
encoding_config=dataset.encoding_config,
```

**Step 3: Add frontend API client function**

In `web/react-explorer/src/api/client.ts`, add `DatasetResponse` fields and the new function:

```typescript
// Add to DatasetResponse type:
source_dataset_id: string | null;
encoding_config: Record<string, any> | null;

// Add function:
export async function prepareDataset(
  datasetId: string,
  name?: string,
  encodingConfig?: Record<string, any>,
  ordinalOnehot?: string[],
  ordinalOrders?: Record<string, string[]>,
): Promise<DatasetResponse> {
  return fetchJson<DatasetResponse>(
    `${API_BASE}/datasets/${datasetId}/prepare`,
    {
      method: "POST",
      body: JSON.stringify({
        name: name || undefined,
        encoding_config: encodingConfig || undefined,
        ordinal_onehot: ordinalOnehot || undefined,
        ordinal_orders: ordinalOrders || undefined,
      }),
    },
  );
}
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ -x -q`
Expected: All pass

**Step 5: Commit**

```bash
git add explaneat/api/routes/datasets.py explaneat/api/schemas.py web/react-explorer/src/api/client.ts
git commit -m "feat: prepare dataset API endpoint for one-hot encoding"
```

---

### Task 4: Auto-Prepare at Experiment Creation

**Files:**
- Modify: `explaneat/api/routes/experiments.py:372-450`

**Step 1: Update create_and_run_experiment**

In `explaneat/api/routes/experiments.py`, in `create_and_run_experiment()`, after loading the dataset and split but before the scaler, add auto-preparation logic:

```python
    # Auto-prepare dataset if it has categorical/ordinal features
    feature_types = dataset.feature_types or {}
    needs_encoding = any(
        ft in ("categorical", "ordinal")
        for ft in feature_types.values()
    )

    if needs_encoding and not dataset.source_dataset_id:
        # This is an original dataset with categoricals — prepare it
        from ...db.encoding import build_encoding_config, prepare_dataset_arrays

        encoding_config = build_encoding_config(
            X_full, dataset.feature_names or [], feature_types,
        )
        # Check for existing prepared version
        prepared = (
            db_session.query(Dataset)
            .filter_by(source_dataset_id=dataset.id)
            .filter(Dataset.encoding_config == encoding_config)
            .first()
        )
        if not prepared:
            X_enc, new_names, new_types = prepare_dataset_arrays(
                X_full, dataset.feature_names or [], feature_types, encoding_config
            )
            prepared = Dataset(
                name=f"{dataset.name} (prepared)",
                source=dataset.source,
                num_samples=X_enc.shape[0],
                num_features=X_enc.shape[1],
                feature_names=new_names,
                feature_types=new_types,
                feature_descriptions=dataset.feature_descriptions,
                target_name=dataset.target_name,
                target_description=dataset.target_description,
                task_type=dataset.task_type,
                num_classes=dataset.num_classes,
                class_names=dataset.class_names,
                source_dataset_id=dataset.id,
                encoding_config=encoding_config,
            )
            prepared.set_data(X_enc, y_full)
            db_session.add(prepared)
            db_session.flush()

        # Switch to prepared dataset for training
        dataset = prepared
        X_full, y_full = dataset.get_data()

    # ... existing split indexing, scaler, and training code follows
```

Also update `num_inputs` to use the (potentially expanded) shape:

```python
    X_train = X_full[train_indices]
    # ... scaler code ...
    num_inputs = X_train.shape[1]  # Already correct — uses post-encoding shape
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ -x -q`
Expected: All pass

**Step 3: Commit**

```bash
git add explaneat/api/routes/experiments.py
git commit -m "feat: auto-prepare dataset with one-hot encoding at experiment creation"
```

---

### Task 5: Dataset Detail UI — Prepare Button and Ordinal Config

**Files:**
- Modify: `web/react-explorer/src/components/DatasetDetail.tsx`

**Step 1: Add "Prepare Dataset" section**

Add a new `Section` in DatasetDetail after the Feature Metadata section. This provides:
- A summary of which features will be encoded (based on current `feature_types`)
- Ordinal ordering UI (drag-to-reorder or manual list) for ordinal features
- Checkbox per ordinal feature: "One-hot encode" (default unchecked = ranked integer)
- Name override text input (default: `{name} (prepared)`)
- "Prepare Dataset" button that calls `prepareDataset()`
- Shows list of existing prepared versions with their encoding configs

This is a frontend-only task. The component should:
1. Scan `feature_types` for categorical/ordinal features
2. For each ordinal feature, load unique values from the dataset to suggest ordering
3. Call `POST /datasets/{id}/prepare` with the configured options
4. Show success with link to the new prepared dataset

**Step 2: Run type check**

Run: `cd web/react-explorer && npx tsc --noEmit`
Expected: No new errors

**Step 3: Commit**

```bash
git add web/react-explorer/src/components/DatasetDetail.tsx
git commit -m "feat: prepare dataset UI with ordinal ordering and one-hot config"
```

---

### Task 6: Evidence Panel — Source/Network View Toggle

**Files:**
- Modify: `explaneat/api/routes/evidence.py:165-220, 350-408`
- Modify: `explaneat/api/schemas.py` (VizDataRequest/Response)
- Modify: `web/react-explorer/src/components/EvidencePanel.tsx`

**Step 1: Add view parameter to evidence endpoint**

In the viz-data endpoint in `evidence.py`, accept an optional `view` query param (`"network"` or `"source"`). When `view=source`:

1. After computing viz data normally, reverse the transformations:
   - **Unscale**: If split has `scaler_params`, reverse StandardScaler on axis values: `x_original = x_scaled * scale + mean`
   - **Regroup one-hot**: If dataset has `encoding_config` and `source_dataset_id`, group `feature:value` columns back to original feature names

2. For SHAP values in source view: sum absolute SHAP values for columns sharing the same base feature name (everything before `:`)

3. For feature name lists: replace `["workclass:Private", "workclass:Gov", ...]` with `["workclass"]`

Add helper functions:

```python
def _unscale_value(value: float, feature_idx: int, scaler_params: dict) -> float:
    """Reverse StandardScaler: original = scaled * scale + mean."""
    mean = scaler_params["mean"][feature_idx]
    scale = scaler_params["scale"][feature_idx]
    return value * scale + mean


def _group_onehot_features(
    feature_names: List[str],
    encoding_config: dict,
) -> Dict[str, List[int]]:
    """Map original feature name → list of column indices in encoded array."""
    groups = {}
    for i, name in enumerate(feature_names):
        if ":" in name:
            base = name.split(":")[0]
        else:
            base = name
        groups.setdefault(base, []).append(i)
    return groups
```

**Step 2: Add frontend toggle**

In `EvidencePanel.tsx`, add a toggle next to the dataset selector:

```typescript
const [viewMode, setViewMode] = useState<"network" | "source">("network");
```

Render as a simple button group: **Network** | **Source**

Pass `viewMode` as a query param when fetching viz data.

**Step 3: Run tests and type check**

Run: `uv run pytest tests/test_core/ tests/test_api/ -x -q`
Run: `cd web/react-explorer && npx tsc --noEmit`
Expected: All pass / no new errors

**Step 4: Commit**

```bash
git add explaneat/api/routes/evidence.py explaneat/api/schemas.py web/react-explorer/src/components/EvidencePanel.tsx
git commit -m "feat: source/network view toggle in evidence panel"
```

---

### Task 7: Integration Test — End-to-End Encoding Pipeline

**Files:**
- Create: `tests/test_db/test_encoding_integration.py`

**Step 1: Write integration test**

```python
"""Integration test: encoding → training → evidence round-trip."""
import numpy as np
import pytest
from explaneat.db.encoding import build_encoding_config, prepare_dataset_arrays


class TestEncodingRoundTrip:
    """Test that encoding and reverse-lookup are consistent."""

    def test_encoding_config_round_trip(self):
        """build_encoding_config → prepare_dataset_arrays produces correct shape."""
        # Simulate adult-like dataset
        n = 100
        X = np.column_stack([
            np.random.randint(18, 65, n),      # age (numeric)
            np.random.randint(0, 5, n),         # workclass (categorical, 5 values)
            np.random.randint(0, 3, n),         # education (ordinal, 3 values)
            np.random.uniform(0, 100000, n),    # income (numeric)
        ])
        feature_names = ["age", "workclass", "education", "income"]
        feature_types = {
            "age": "numeric",
            "workclass": "categorical",
            "education": "ordinal",
            "income": "numeric",
        }

        config = build_encoding_config(X, feature_names, feature_types)
        X_enc, names, types = prepare_dataset_arrays(
            X, feature_names, feature_types, config
        )

        # age (1) + workclass one-hot (5) + education ranked (1) + income (1) = 8
        assert X_enc.shape == (n, 8)
        assert len(names) == 8
        assert names[0] == "age"
        assert all(n.startswith("workclass:") for n in names[1:6])
        assert names[6] == "education"
        assert names[7] == "income"

    def test_onehot_columns_are_binary(self):
        """One-hot encoded columns contain only 0 and 1."""
        X = np.array([[0], [1], [2], [0], [1]])
        config = {"categorical": {"feat": ["0", "1", "2"]}}
        X_enc, _, _ = prepare_dataset_arrays(
            X, ["feat"], {"feat": "categorical"}, config
        )
        assert set(np.unique(X_enc)) == {0.0, 1.0}

    def test_each_row_has_exactly_one_hot(self):
        """Each row of a one-hot group sums to 1 (or 0 for unknown)."""
        X = np.array([[0], [1], [2]])
        config = {"categorical": {"feat": ["0", "1", "2"]}}
        X_enc, _, _ = prepare_dataset_arrays(
            X, ["feat"], {"feat": "categorical"}, config
        )
        assert np.all(X_enc.sum(axis=1) == 1.0)

    def test_feature_grouping_by_colon(self):
        """Features with ':' can be grouped back to original name."""
        names = ["age", "workclass:Private", "workclass:Gov", "income"]
        groups = {}
        for i, name in enumerate(names):
            base = name.split(":")[0]
            groups.setdefault(base, []).append(i)

        assert groups == {"age": [0], "workclass": [1, 2], "income": [3]}
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_db/test_encoding.py tests/test_db/test_encoding_integration.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_db/test_encoding_integration.py
git commit -m "test: integration tests for encoding pipeline round-trip"
```

---

### Task Summary

| Task | What | Files |
|------|------|-------|
| 1 | DB migration | models.py, migration |
| 2 | Encoding engine + tests | encoding.py, test_encoding.py |
| 3 | Prepare dataset API + client | datasets.py, schemas.py, client.ts |
| 4 | Auto-prepare at experiment creation | experiments.py |
| 5 | Dataset Detail UI | DatasetDetail.tsx |
| 6 | Source/Network view toggle | evidence.py, EvidencePanel.tsx |
| 7 | Integration tests | test_encoding_integration.py |
