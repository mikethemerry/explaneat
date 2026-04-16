# Config Templates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow configuring all NEAT mutation rates, backprop settings, and training params via the experiment creation modal, with reusable templates that experiments can be based on.

**Architecture:** New `ConfigTemplate` table holds named JSONB configs. Experiments get a `config_template_id` FK and store the final resolved config (defaults + template + overrides) in the existing `config_json` column. `_default_neat_config_text()` is refactored to accept a full config dict so templates can override any value.

**Tech Stack:** SQLAlchemy/Alembic for DB, FastAPI for endpoints, React/TypeScript for UI.

---

### Task 1: Database — ConfigTemplate Model + Migration

**Files:**
- Modify: `explaneat/db/models.py` (add ConfigTemplate model)
- Modify: `explaneat/db/models.py` (add `config_template_id` to Experiment)
- Create: `alembic/versions/j3k4l5m6n7o8_add_config_templates.py`

**Step 1: Add ConfigTemplate model**

In `explaneat/db/models.py`, add after the Dataset-related models (after DatasetSplit, before Experiment, around line 180):

```python
class ConfigTemplate(Base, TimestampMixin):
    """Reusable training configuration templates."""

    __tablename__ = "config_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    config = Column(JSONB, nullable=False)

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
```

**Step 2: Add config_template_id to Experiment**

In `explaneat/db/models.py`, in the `Experiment` class, after `split_id` column add:

```python
config_template_id = Column(
    UUID(as_uuid=True), ForeignKey("config_templates.id", ondelete="SET NULL")
)
```

**Step 3: Create migration**

Create `alembic/versions/j3k4l5m6n7o8_add_config_templates.py`:

```python
"""add config templates

Revision ID: j3k4l5m6n7o8
Revises: i2j3k4l5m6n7
Create Date: 2026-04-16
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'j3k4l5m6n7o8'
down_revision: Union[str, None] = 'i2j3k4l5m6n7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEFAULT_CONFIG = {
    "training": {
        "population_size": 150,
        "n_generations": 10,
        "n_epochs_backprop": 5,
        "fitness_function": "bce",
    },
    "neat": {
        "bias_mutate_rate": 0.7,
        "bias_mutate_power": 0.5,
        "bias_replace_rate": 0.1,
        "weight_mutate_rate": 0.8,
        "weight_mutate_power": 0.5,
        "weight_replace_rate": 0.1,
        "enabled_mutate_rate": 0.01,
        "node_add_prob": 0.15,
        "node_delete_prob": 0.05,
        "conn_add_prob": 0.3,
        "conn_delete_prob": 0.1,
        "compatibility_threshold": 3.0,
        "compatibility_disjoint_coefficient": 1.0,
        "compatibility_weight_coefficient": 0.5,
        "max_stagnation": 15,
        "species_elitism": 2,
        "elitism": 2,
        "survival_threshold": 0.2,
    },
    "backprop": {
        "learning_rate": 1.5,
        "optimizer": "adadelta",
    },
}


def upgrade() -> None:
    op.create_table(
        'config_templates',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('config', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    op.add_column('experiments', sa.Column('config_template_id', sa.UUID(), nullable=True))
    op.create_foreign_key(
        'fk_experiments_config_template_id', 'experiments', 'config_templates',
        ['config_template_id'], ['id'], ondelete='SET NULL'
    )

    # Seed Default template
    import json
    op.execute(
        f"INSERT INTO config_templates (id, name, description, config) "
        f"VALUES (gen_random_uuid(), 'Default', 'Default NEAT training configuration', "
        f"'{json.dumps(DEFAULT_CONFIG)}'::jsonb)"
    )


def downgrade() -> None:
    op.drop_constraint('fk_experiments_config_template_id', 'experiments', type_='foreignkey')
    op.drop_column('experiments', 'config_template_id')
    op.drop_table('config_templates')
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ tests/test_db/ -x -q`
Expected: All pass (additive changes)

**Step 5: Commit**

```bash
git add explaneat/db/models.py alembic/versions/j3k4l5m6n7o8_add_config_templates.py
git commit -m "feat: add ConfigTemplate model and config_template_id to Experiment"
```

---

### Task 2: Config Resolution Utility + Tests

**Files:**
- Create: `explaneat/core/config_resolution.py`
- Create: `tests/test_core/test_config_resolution.py`

**Step 1: Write failing tests**

Create `tests/test_core/test_config_resolution.py`:

```python
"""Tests for config resolution (defaults + template + overrides)."""
from explaneat.core.config_resolution import (
    DEFAULT_CONFIG,
    resolve_config,
    config_to_neat_text,
)


class TestResolveConfig:

    def test_defaults_only(self):
        """With no template or overrides, returns defaults."""
        result = resolve_config()
        assert result == DEFAULT_CONFIG

    def test_template_overrides_defaults(self):
        """Template values override defaults."""
        template = {"training": {"population_size": 200}}
        result = resolve_config(template=template)
        assert result["training"]["population_size"] == 200
        # Unspecified values still come from defaults
        assert result["training"]["n_generations"] == DEFAULT_CONFIG["training"]["n_generations"]

    def test_overrides_beat_template(self):
        """Per-experiment overrides beat template values."""
        template = {"training": {"population_size": 200}}
        overrides = {"training": {"population_size": 300}}
        result = resolve_config(template=template, overrides=overrides)
        assert result["training"]["population_size"] == 300

    def test_partial_group_override(self):
        """Overriding one key in a group preserves other keys."""
        overrides = {"neat": {"bias_mutate_rate": 0.5}}
        result = resolve_config(overrides=overrides)
        assert result["neat"]["bias_mutate_rate"] == 0.5
        # Other neat values unchanged
        assert result["neat"]["weight_mutate_rate"] == DEFAULT_CONFIG["neat"]["weight_mutate_rate"]

    def test_unknown_groups_ignored(self):
        """Unknown top-level keys in overrides are ignored (safe)."""
        overrides = {"bogus_group": {"x": 1}}
        result = resolve_config(overrides=overrides)
        assert "bogus_group" not in result


class TestConfigToNeatText:

    def test_produces_valid_neat_config_text(self):
        """Resolved config converts to a NEAT config text with required sections."""
        cfg = resolve_config()
        text = config_to_neat_text(cfg, num_inputs=10, num_outputs=1)
        assert "[NEAT]" in text
        assert "[DefaultGenome]" in text
        assert "[DefaultSpeciesSet]" in text
        assert "[DefaultStagnation]" in text
        assert "[DefaultReproduction]" in text
        assert "num_inputs              = 10" in text

    def test_neat_text_uses_config_values(self):
        """Mutation rates from config appear in the generated text."""
        cfg = resolve_config(overrides={"neat": {"bias_mutate_rate": 0.42}})
        text = config_to_neat_text(cfg, num_inputs=5, num_outputs=1)
        assert "bias_mutate_rate        = 0.42" in text

    def test_population_size_from_config(self):
        """pop_size is drawn from training.population_size, not a separate arg."""
        cfg = resolve_config(overrides={"training": {"population_size": 42}})
        text = config_to_neat_text(cfg, num_inputs=5, num_outputs=1)
        assert "pop_size              = 42" in text
```

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_core/test_config_resolution.py -v`
Expected: FAIL — module not found

**Step 3: Implement module**

Create `explaneat/core/config_resolution.py`:

```python
"""Config resolution: merge defaults + template + overrides.

Templates and overrides use a grouped JSON structure:
    {
        "training": {...},
        "neat": {...},
        "backprop": {...}
    }
"""
from typing import Any, Dict, Optional
from copy import deepcopy


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "training": {
        "population_size": 150,
        "n_generations": 10,
        "n_epochs_backprop": 5,
        "fitness_function": "bce",
    },
    "neat": {
        "bias_mutate_rate": 0.7,
        "bias_mutate_power": 0.5,
        "bias_replace_rate": 0.1,
        "weight_mutate_rate": 0.8,
        "weight_mutate_power": 0.5,
        "weight_replace_rate": 0.1,
        "enabled_mutate_rate": 0.01,
        "node_add_prob": 0.15,
        "node_delete_prob": 0.05,
        "conn_add_prob": 0.3,
        "conn_delete_prob": 0.1,
        "compatibility_threshold": 3.0,
        "compatibility_disjoint_coefficient": 1.0,
        "compatibility_weight_coefficient": 0.5,
        "max_stagnation": 15,
        "species_elitism": 2,
        "elitism": 2,
        "survival_threshold": 0.2,
    },
    "backprop": {
        "learning_rate": 1.5,
        "optimizer": "adadelta",
    },
}


def resolve_config(
    template: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Merge defaults, template, and overrides into a final config.

    Merge order (last wins): defaults -> template -> overrides
    Unknown groups in template/overrides are ignored.
    """
    result = deepcopy(DEFAULT_CONFIG)

    for layer in (template, overrides):
        if not layer:
            continue
        for group, values in layer.items():
            if group not in result:
                continue
            if not isinstance(values, dict):
                continue
            result[group].update(values)

    return result


def config_to_neat_text(
    config: Dict[str, Dict[str, Any]],
    num_inputs: int,
    num_outputs: int,
) -> str:
    """Convert resolved config to NEAT config text."""
    training = config["training"]
    neat = config["neat"]

    return f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 999.0
pop_size              = {training["population_size"]}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = relu
activation_mutate_rate  = 0.0
activation_options      = relu
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = {neat["bias_mutate_power"]}
bias_mutate_rate        = {neat["bias_mutate_rate"]}
bias_replace_rate       = {neat["bias_replace_rate"]}
compatibility_disjoint_coefficient = {neat["compatibility_disjoint_coefficient"]}
compatibility_weight_coefficient   = {neat["compatibility_weight_coefficient"]}
conn_add_prob           = {neat["conn_add_prob"]}
conn_delete_prob        = {neat["conn_delete_prob"]}
enabled_default         = True
enabled_mutate_rate     = {neat["enabled_mutate_rate"]}
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = {neat["node_add_prob"]}
node_delete_prob        = {neat["node_delete_prob"]}
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = {neat["weight_mutate_power"]}
weight_mutate_rate      = {neat["weight_mutate_rate"]}
weight_replace_rate     = {neat["weight_replace_rate"]}

[DefaultSpeciesSet]
compatibility_threshold = {neat["compatibility_threshold"]}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {neat["max_stagnation"]}
species_elitism      = {neat["species_elitism"]}

[DefaultReproduction]
elitism            = {neat["elitism"]}
survival_threshold = {neat["survival_threshold"]}
"""
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_config_resolution.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add explaneat/core/config_resolution.py tests/test_core/test_config_resolution.py
git commit -m "feat: config resolution utility for defaults + template + overrides"
```

---

### Task 3: ConfigTemplate API Endpoints

**Files:**
- Create: `explaneat/api/routes/config_templates.py`
- Modify: `explaneat/api/schemas.py` (add ConfigTemplate schemas)
- Modify: `explaneat/api/app.py` (register router)

**Step 1: Add schemas**

Append to `explaneat/api/schemas.py`:

```python
class ConfigTemplateResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConfigTemplateListResponse(BaseModel):
    templates: List[ConfigTemplateResponse]
    total: int


class ConfigTemplateCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]


class ConfigTemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
```

**Step 2: Create router**

Create `explaneat/api/routes/config_templates.py`:

```python
"""Config template CRUD endpoints."""
import uuid
from fastapi import APIRouter, HTTPException

from ...db.base import db
from ...db.models import ConfigTemplate
from ..schemas import (
    ConfigTemplateResponse,
    ConfigTemplateListResponse,
    ConfigTemplateCreateRequest,
    ConfigTemplateUpdateRequest,
)


router = APIRouter()


@router.get("", response_model=ConfigTemplateListResponse)
async def list_templates():
    with db.session_scope() as session:
        templates = session.query(ConfigTemplate).order_by(ConfigTemplate.created_at.desc()).all()
        return ConfigTemplateListResponse(
            templates=[ConfigTemplateResponse(**t.to_dict()) for t in templates],
            total=len(templates),
        )


@router.get("/{template_id}", response_model=ConfigTemplateResponse)
async def get_template(template_id: str):
    with db.session_scope() as session:
        template = session.query(ConfigTemplate).filter_by(id=uuid.UUID(template_id)).first()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return ConfigTemplateResponse(**template.to_dict())


@router.post("", response_model=ConfigTemplateResponse)
async def create_template(request: ConfigTemplateCreateRequest):
    with db.session_scope() as session:
        template = ConfigTemplate(
            name=request.name,
            description=request.description,
            config=request.config,
        )
        session.add(template)
        session.flush()
        return ConfigTemplateResponse(**template.to_dict())


@router.patch("/{template_id}", response_model=ConfigTemplateResponse)
async def update_template(template_id: str, request: ConfigTemplateUpdateRequest):
    with db.session_scope() as session:
        template = session.query(ConfigTemplate).filter_by(id=uuid.UUID(template_id)).first()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        if request.name is not None:
            template.name = request.name
        if request.description is not None:
            template.description = request.description
        if request.config is not None:
            template.config = request.config
        session.flush()
        return ConfigTemplateResponse(**template.to_dict())


@router.delete("/{template_id}")
async def delete_template(template_id: str):
    with db.session_scope() as session:
        template = session.query(ConfigTemplate).filter_by(id=uuid.UUID(template_id)).first()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        session.delete(template)
        return {"status": "deleted"}
```

**Step 3: Register router in app.py**

In `explaneat/api/app.py`, in `create_app()`, add import and router registration next to other routes:

```python
from .routes import config_templates

# ...

app.include_router(
    config_templates.router,
    prefix="/api/config-templates",
    tags=["config-templates"],
)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ tests/test_db/ -x -q`
Expected: All pass

**Step 5: Commit**

```bash
git add explaneat/api/routes/config_templates.py explaneat/api/schemas.py explaneat/api/app.py
git commit -m "feat: ConfigTemplate CRUD API endpoints"
```

---

### Task 4: Wire Templates into Experiment Creation

**Files:**
- Modify: `explaneat/api/schemas.py` (add fields to ExperimentCreateRequest)
- Modify: `explaneat/api/routes/experiments.py` (use resolve_config in create_and_run_experiment)

**Step 1: Update ExperimentCreateRequest**

In `explaneat/api/schemas.py`, find `ExperimentCreateRequest` and add:

```python
config_template_id: Optional[str] = None
config_overrides: Optional[Dict[str, Any]] = None
```

**Step 2: Update experiment creation endpoint**

In `explaneat/api/routes/experiments.py`, `create_and_run_experiment()`:

1. Add imports at top:
```python
from ...core.config_resolution import resolve_config, config_to_neat_text
from ...db.models import ConfigTemplate
```

2. After loading dataset/split but before building the config text (around line 470), resolve the config:

```python
    # Resolve config from template + overrides
    template = None
    if request.config_template_id:
        template_obj = db_session.get(ConfigTemplate, UUID(request.config_template_id))
        if not template_obj:
            raise HTTPException(status_code=404, detail="Config template not found")
        template = template_obj.config

    resolved_config = resolve_config(
        template=template,
        overrides=request.config_overrides,
    )
```

3. Replace the hardcoded config_text / config_json build (lines 478-487):

```python
    num_inputs = X_train.shape[1]
    num_outputs = 1
    if dataset.num_classes and dataset.num_classes > 2:
        num_outputs = dataset.num_classes

    config_text = config_to_neat_text(resolved_config, num_inputs, num_outputs)
    config_json = {
        "pop_size": resolved_config["training"]["population_size"],
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "fitness_criterion": "max",
        "fitness_threshold": 999.0,
        "resolved_config": resolved_config,
        "config_template_id": request.config_template_id,
    }
```

4. Update the `experiment_runner.start()` call to pull training values from `resolved_config`:

```python
    job_id = await experiment_runner.start(
        config_text=config_text,
        config_json=config_json,
        X_train=X_train,
        y_train=y_train,
        experiment_name=request.name,
        dataset_name=dataset.name,
        n_generations=resolved_config["training"]["n_generations"],
        n_epochs_backprop=resolved_config["training"]["n_epochs_backprop"],
        fitness_function=resolved_config["training"]["fitness_function"],
        description=request.description,
        dataset_id=str(dataset.id),
        split_id=str(split.id),
        config_template_id=request.config_template_id,
    )
```

5. Update `experiment_runner.start()` signature in `explaneat/api/experiment_runner.py` to accept `config_template_id` and set it on the created Experiment record. Find where Experiment is created (search for `Experiment(`) and add `config_template_id=UUID(config_template_id) if config_template_id else None,` to the constructor.

Note: the top-level `request.population_size`, `request.n_generations`, `request.n_epochs_backprop`, `request.fitness_function` fields stay in the schema for backwards compatibility, but the resolved config takes precedence. If the UI still sends them at the top level, they're ignored — the UI must send them inside `config_overrides.training`.

**Step 3: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ tests/test_db/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add explaneat/api/schemas.py explaneat/api/routes/experiments.py explaneat/api/experiment_runner.py
git commit -m "feat: use resolved config with template support in experiment creation"
```

---

### Task 5: Frontend — API Client Types and Functions

**Files:**
- Modify: `web/react-explorer/src/api/client.ts`

**Step 1: Add types**

Append new types to `web/react-explorer/src/api/client.ts`:

```typescript
// Config templates
export type TrainingConfig = {
  population_size: number;
  n_generations: number;
  n_epochs_backprop: number;
  fitness_function: "bce" | "auc";
};

export type NeatConfig = {
  bias_mutate_rate: number;
  bias_mutate_power: number;
  bias_replace_rate: number;
  weight_mutate_rate: number;
  weight_mutate_power: number;
  weight_replace_rate: number;
  enabled_mutate_rate: number;
  node_add_prob: number;
  node_delete_prob: number;
  conn_add_prob: number;
  conn_delete_prob: number;
  compatibility_threshold: number;
  compatibility_disjoint_coefficient: number;
  compatibility_weight_coefficient: number;
  max_stagnation: number;
  species_elitism: number;
  elitism: number;
  survival_threshold: number;
};

export type BackpropConfig = {
  learning_rate: number;
  optimizer: string;
};

export type ResolvedConfig = {
  training: TrainingConfig;
  neat: NeatConfig;
  backprop: BackpropConfig;
};

export type ConfigTemplateResponse = {
  id: string;
  name: string;
  description: string | null;
  config: ResolvedConfig;
  created_at: string | null;
  updated_at: string | null;
};

export type ConfigTemplateListResponse = {
  templates: ConfigTemplateResponse[];
  total: number;
};
```

**Step 2: Add client functions**

Append to the same file:

```typescript
export async function listConfigTemplates(): Promise<ConfigTemplateListResponse> {
  return fetchJson(`${API_BASE}/config-templates`);
}

export async function getConfigTemplate(id: string): Promise<ConfigTemplateResponse> {
  return fetchJson(`${API_BASE}/config-templates/${id}`);
}

export async function createConfigTemplate(
  name: string,
  config: ResolvedConfig,
  description?: string,
): Promise<ConfigTemplateResponse> {
  return fetchJson(`${API_BASE}/config-templates`, {
    method: "POST",
    body: JSON.stringify({ name, description, config }),
  });
}

export async function updateConfigTemplate(
  id: string,
  updates: { name?: string; description?: string; config?: ResolvedConfig },
): Promise<ConfigTemplateResponse> {
  return fetchJson(`${API_BASE}/config-templates/${id}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export async function deleteConfigTemplate(id: string): Promise<void> {
  await fetchJson(`${API_BASE}/config-templates/${id}`, { method: "DELETE" });
}
```

**Step 3: Type-check**

Run: `cd web/react-explorer && npx tsc --noEmit`
Expected: No new errors in client.ts

**Step 4: Commit**

```bash
git add web/react-explorer/src/api/client.ts
git commit -m "feat: API client types and functions for config templates"
```

---

### Task 6: Experiment Create Modal — Template Picker + Advanced Config

**Files:**
- Modify: `web/react-explorer/src/components/ExperimentCreateModal.tsx`

**Step 1: Add template picker and Advanced Config section**

In `ExperimentCreateModal.tsx`:

1. Add state for selected template and config editing:
```typescript
const [templates, setTemplates] = useState<ConfigTemplateResponse[]>([]);
const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
const [config, setConfig] = useState<ResolvedConfig | null>(null);
const [advancedOpen, setAdvancedOpen] = useState(false);
```

2. Load templates on mount and pick the "Default" one:
```typescript
useEffect(() => {
  listConfigTemplates().then((res) => {
    setTemplates(res.templates);
    const defaultT = res.templates.find(t => t.name === "Default") || res.templates[0];
    if (defaultT) {
      setSelectedTemplateId(defaultT.id);
      setConfig(defaultT.config);
    }
  });
}, []);
```

3. When template changes, reset config (with confirm if user has edited):
```typescript
const onTemplateChange = (id: string) => {
  const t = templates.find(t => t.id === id);
  if (!t) return;
  setSelectedTemplateId(id);
  setConfig(t.config);
};
```

4. Add template dropdown near the top of the form (after name/description):
```tsx
<label>Template</label>
<select value={selectedTemplateId || ""} onChange={e => onTemplateChange(e.target.value)}>
  {templates.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
</select>
```

5. Replace the existing n_generations / n_epochs_backprop / population_size / fitness_function fields with an Advanced Config section (collapsible). Each field is bound to `config.training.*`, `config.neat.*`, or `config.backprop.*`:

```tsx
<button type="button" onClick={() => setAdvancedOpen(!advancedOpen)}>
  {advancedOpen ? "▼" : "▶"} Advanced Config
</button>

{advancedOpen && config && (
  <div className="advanced-config">
    <h4>Training</h4>
    <label>Population size</label>
    <input type="number" value={config.training.population_size}
      onChange={e => setConfig({...config, training: {...config.training, population_size: parseInt(e.target.value) || 0}})} />
    {/* ... other training fields ... */}

    <h4>NEAT Mutation & Topology</h4>
    <label>Bias mutate rate</label>
    <input type="number" step="0.01" value={config.neat.bias_mutate_rate}
      onChange={e => setConfig({...config, neat: {...config.neat, bias_mutate_rate: parseFloat(e.target.value) || 0}})} />
    {/* ... other neat fields ... */}

    <h4>Backprop</h4>
    <label>Learning rate</label>
    <input type="number" step="0.01" value={config.backprop.learning_rate}
      onChange={e => setConfig({...config, backprop: {...config.backprop, learning_rate: parseFloat(e.target.value) || 0}})} />
  </div>
)}
```

6. Build a helper `renderNumberField(label, group, key)` to avoid 20+ copies of the same pattern.

7. On submit, send `config_template_id` and `config_overrides` (the edited config) instead of the old top-level fields:

```typescript
const request: ExperimentCreateRequest = {
  name,
  description,
  dataset_id: selectedDatasetId,
  dataset_split_id: selectedSplitId,
  config_template_id: selectedTemplateId || undefined,
  config_overrides: config || undefined,
};
```

8. Add "Save as new template" button that prompts for a name and calls `createConfigTemplate()`:

```tsx
<button type="button" onClick={async () => {
  const name = prompt("Template name?");
  if (!name || !config) return;
  const t = await createConfigTemplate(name, config);
  setTemplates([...templates, t]);
  setSelectedTemplateId(t.id);
}}>Save as new template</button>
```

**Step 2: Update ExperimentCreateRequest TypeScript type** in `client.ts`:

```typescript
export type ExperimentCreateRequest = {
  name: string;
  description?: string;
  dataset_id: string;
  dataset_split_id: string;
  config_template_id?: string;
  config_overrides?: ResolvedConfig;
};
```

**Step 3: Type-check**

Run: `cd web/react-explorer && npx tsc --noEmit`
Expected: No new errors

**Step 4: Commit**

```bash
git add web/react-explorer/src/components/ExperimentCreateModal.tsx web/react-explorer/src/api/client.ts
git commit -m "feat: template picker and Advanced Config in experiment create modal"
```

---

### Task 7: Templates Management Page

**Files:**
- Create: `web/react-explorer/src/components/TemplatesPage.tsx`
- Modify: `web/react-explorer/src/App.tsx` (add route/view)
- Modify: `web/react-explorer/src/components/NavBar.tsx` (add nav link)

**Step 1: Create TemplatesPage component**

Create `web/react-explorer/src/components/TemplatesPage.tsx`:

```tsx
import { useEffect, useState } from "react";
import {
  listConfigTemplates,
  createConfigTemplate,
  updateConfigTemplate,
  deleteConfigTemplate,
  type ConfigTemplateResponse,
  type ResolvedConfig,
} from "../api/client";

// Default empty config for new templates
const EMPTY_CONFIG: ResolvedConfig = {
  training: { population_size: 150, n_generations: 10, n_epochs_backprop: 5, fitness_function: "bce" },
  neat: { /* ... populate with default values ... */ },
  backprop: { learning_rate: 1.5, optimizer: "adadelta" },
};

export function TemplatesPage() {
  const [templates, setTemplates] = useState<ConfigTemplateResponse[]>([]);
  const [editing, setEditing] = useState<ConfigTemplateResponse | null>(null);

  const load = () => listConfigTemplates().then(r => setTemplates(r.templates));
  useEffect(() => { load(); }, []);

  // List view + edit form
  // Reuse the Advanced Config field layout from ExperimentCreateModal
  // (consider extracting a shared <ConfigEditor config={...} onChange={...}/> component)
}
```

**Step 2: Extract a shared ConfigEditor component**

Create `web/react-explorer/src/components/ConfigEditor.tsx`:

```tsx
import { type ResolvedConfig } from "../api/client";

type Props = {
  config: ResolvedConfig;
  onChange: (config: ResolvedConfig) => void;
};

export function ConfigEditor({ config, onChange }: Props) {
  // Grouped fields: Training / NEAT Mutation & Topology / Backprop
  // Use a helper to render number fields from (group, key, label)
}
```

Then use `<ConfigEditor>` in both `ExperimentCreateModal` (inside the Advanced Config expandable) and `TemplatesPage`. Refactor the modal from Task 6 to use this shared component — delete any duplicated field code there.

**Step 3: Wire into App.tsx / NavBar**

Add a "Templates" link in NavBar that navigates to a new route/view rendering `<TemplatesPage />`. Existing routing pattern: follow how other pages are wired (check `App.tsx` for the current view-switching approach).

**Step 4: Type-check**

Run: `cd web/react-explorer && npx tsc --noEmit`
Expected: No new errors

**Step 5: Commit**

```bash
git add web/react-explorer/src/components/TemplatesPage.tsx web/react-explorer/src/components/ConfigEditor.tsx web/react-explorer/src/components/ExperimentCreateModal.tsx web/react-explorer/src/App.tsx web/react-explorer/src/components/NavBar.tsx
git commit -m "feat: templates management page and shared ConfigEditor component"
```

---

### Task 8: Experiment Detail — Training Config Section

**Files:**
- Modify: `web/react-explorer/src/api/client.ts` (ExperimentResponse type additions)
- Modify: `explaneat/api/schemas.py` (add resolved_config to experiment response)
- Modify: `explaneat/api/routes/experiments.py` (expose resolved_config via GET)
- Modify: `web/react-explorer/src/components/GenomeExplorer.tsx` (or the experiment detail wrapper — add TrainingConfigSection)

**Step 1: Extend experiment response schema**

Find the schema used for experiment detail (likely `ExperimentDetailResponse` or similar in `schemas.py`). Add:

```python
config_template_id: Optional[str] = None
config_template_name: Optional[str] = None
resolved_config: Optional[Dict[str, Any]] = None
```

Update the route that serves experiment detail to populate `resolved_config` from `experiment.config_json.get("resolved_config")` and look up the template name if `config_template_id` is set.

**Step 2: Display config section in the UI**

In `GenomeExplorer.tsx` (or wherever the experiment header/info lives), add a collapsible "Training Config" section near the top. Reuse `<ConfigEditor>` with `readOnly` support (add a `readOnly?: boolean` prop that disables inputs).

If `config_template_id` is set and we have a `config_template_name`, show at the top: "Based on template: [Name]".

**Step 3: Type-check and tests**

Run: `cd web/react-explorer && npx tsc --noEmit`
Run: `uv run pytest tests/test_core/ tests/test_api/ tests/test_db/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add explaneat/api/schemas.py explaneat/api/routes/experiments.py web/react-explorer/src/api/client.ts web/react-explorer/src/components/GenomeExplorer.tsx web/react-explorer/src/components/ConfigEditor.tsx
git commit -m "feat: training config section on experiment detail view"
```

---

### Task Summary

| Task | What | Files |
|------|------|-------|
| 1 | DB model + migration + seed Default | models.py, migration |
| 2 | Config resolution utility + tests | config_resolution.py, tests |
| 3 | ConfigTemplate CRUD endpoints | config_templates.py, schemas.py, app.py |
| 4 | Wire templates into experiment creation | experiments.py, experiment_runner.py, schemas.py |
| 5 | Frontend API client types + functions | client.ts |
| 6 | Experiment Create Modal: picker + advanced | ExperimentCreateModal.tsx |
| 7 | Templates Management Page + shared ConfigEditor | TemplatesPage.tsx, ConfigEditor.tsx, App.tsx, NavBar.tsx |
| 8 | Experiment detail Training Config section | GenomeExplorer.tsx, schemas.py, experiments.py |
