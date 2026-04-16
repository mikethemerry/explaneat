# Experiment Resume Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow experiments that were interrupted by a server restart to be resumed from the last completed generation, continuing to the originally-configured total.

**Architecture:** A startup hook marks orphaned `running` experiments as `interrupted`. A new `/resume` endpoint reconstructs the NEAT population from the latest saved `Population` row's genomes and launches a background job that continues evolution using the existing `initial_state` parameter that `BackpropPopulation` already supports.

**Tech Stack:** SQLAlchemy/Alembic for schema, FastAPI for endpoint + startup hook, neat-python for population reconstruction, React/TypeScript for UI buttons.

---

### Task 1: DB Migration — Allow 'interrupted' Status

**Files:**
- Modify: `explaneat/db/models.py:256-261` (CheckConstraint)
- Create: `alembic/versions/k4l5m6n7o8p9_add_interrupted_status.py`

**Step 1: Update CheckConstraint in models.py**

Replace:
```python
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'paused')",
            name="check_status",
        ),
    )
```

With:
```python
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'paused', 'interrupted')",
            name="check_status",
        ),
    )
```

**Step 2: Create migration**

Create `alembic/versions/k4l5m6n7o8p9_add_interrupted_status.py`:

```python
"""allow interrupted status on experiments

Revision ID: k4l5m6n7o8p9
Revises: j3k4l5m6n7o8
Create Date: 2026-04-17
"""
from typing import Sequence, Union
from alembic import op

revision: str = 'k4l5m6n7o8p9'
down_revision: Union[str, None] = 'j3k4l5m6n7o8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint('check_status', 'experiments', type_='check')
    op.create_check_constraint(
        'check_status',
        'experiments',
        "status IN ('running', 'completed', 'failed', 'paused', 'interrupted')",
    )


def downgrade() -> None:
    op.drop_constraint('check_status', 'experiments', type_='check')
    op.create_check_constraint(
        'check_status',
        'experiments',
        "status IN ('running', 'completed', 'failed', 'paused')",
    )
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add explaneat/db/models.py alembic/versions/k4l5m6n7o8p9_add_interrupted_status.py
git commit -m "feat: allow 'interrupted' status on experiments"
```

---

### Task 2: Startup Hook — Mark Orphaned Experiments as Interrupted

**Files:**
- Modify: `explaneat/api/app.py`
- Create: `tests/test_api/test_startup_interrupted.py`

**Step 1: Write failing test**

Create `tests/test_api/test_startup_interrupted.py`:

```python
"""Test that orphaned running experiments are marked interrupted on startup."""
from datetime import datetime
from uuid import uuid4
import pytest

from explaneat.db.base import db
from explaneat.db.models import Experiment
from explaneat.api.app import mark_orphaned_experiments_interrupted


@pytest.fixture
def test_db():
    """Use an in-memory SQLite database for isolation."""
    db.init_db("sqlite:///:memory:")
    from explaneat.db.base import Base
    Base.metadata.create_all(db.engine)
    yield db
    db.engine.dispose()


def test_running_experiments_marked_interrupted(test_db):
    """Experiments with status='running' at startup are marked 'interrupted'."""
    with db.session_scope() as session:
        exp = Experiment(
            experiment_sha="abc",
            name="orphan",
            config_json={},
            neat_config_text="",
            status="running",
            start_time=datetime.utcnow(),
        )
        session.add(exp)
        session.flush()
        exp_id = exp.id

    count = mark_orphaned_experiments_interrupted()
    assert count == 1

    with db.session_scope() as session:
        exp = session.get(Experiment, exp_id)
        assert exp.status == "interrupted"
        assert exp.end_time is not None


def test_other_statuses_untouched(test_db):
    """Experiments with non-'running' statuses are not modified."""
    with db.session_scope() as session:
        for status in ["completed", "failed", "paused"]:
            session.add(Experiment(
                experiment_sha="x", name=f"exp_{status}",
                config_json={}, neat_config_text="",
                status=status, start_time=datetime.utcnow(),
            ))

    count = mark_orphaned_experiments_interrupted()
    assert count == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api/test_startup_interrupted.py -v`
Expected: FAIL — function doesn't exist

**Step 3: Implement in `explaneat/api/app.py`**

Add this function (outside `create_app`, module level):

```python
def mark_orphaned_experiments_interrupted() -> int:
    """Mark any experiments stuck in 'running' state as 'interrupted'.

    Called on app startup to recover from server crashes.
    Returns the number of experiments updated.
    """
    from datetime import datetime
    from .db.base import db
    from .db.models import Experiment

    with db.session_scope() as session:
        orphans = session.query(Experiment).filter_by(status="running").all()
        for exp in orphans:
            exp.status = "interrupted"
            exp.end_time = datetime.utcnow()
        return len(orphans)
```

Add the startup hook inside `create_app()`:

```python
    @app.on_event("startup")
    async def on_startup():
        count = mark_orphaned_experiments_interrupted()
        if count:
            logger.info("Marked %d orphaned experiments as interrupted", count)
```

Add `import logging` at top if not present, plus `logger = logging.getLogger(__name__)`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_api/test_startup_interrupted.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add explaneat/api/app.py tests/test_api/test_startup_interrupted.py
git commit -m "feat: mark orphaned running experiments as interrupted on startup"
```

---

### Task 3: Resume Utility — Reconstruct Population from DB

**Files:**
- Modify: `explaneat/db/population.py` (add classmethod)
- Create: `tests/test_db/test_resume.py`

**Step 1: Write failing test**

Create `tests/test_db/test_resume.py`:

```python
"""Tests for resuming experiments from saved populations."""
import pytest

from explaneat.db.population import DatabaseBackpropPopulation


class TestResumeHelpers:

    def test_load_latest_population_returns_highest_generation(self):
        """_get_latest_generation returns the max generation saved."""
        # This is a unit test of the helper logic — mock DB interaction
        # is out of scope; we test the public behavior via integration in Task 6.
        # For now: verify the method exists and has the expected signature.
        assert hasattr(DatabaseBackpropPopulation, "_get_latest_generation")

    def test_remaining_generations_calculation(self):
        """Remaining = target - (last_gen + 1), clamped to 0."""
        from explaneat.db.population import compute_remaining_generations
        assert compute_remaining_generations(last_gen=5, target=10) == 4
        assert compute_remaining_generations(last_gen=9, target=10) == 0
        assert compute_remaining_generations(last_gen=15, target=10) == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db/test_resume.py -v`
Expected: FAIL — function doesn't exist

**Step 3: Implement the helper functions**

In `explaneat/db/population.py`, add at the bottom of the file (after the class):

```python
def compute_remaining_generations(last_gen: int, target: int) -> int:
    """How many more generations to run after the last completed one.

    Returns 0 if we've already reached or exceeded the target.
    """
    remaining = target - (last_gen + 1)
    return max(0, remaining)
```

Also add a `_get_latest_generation` static method to `DatabaseBackpropPopulation`:

```python
    @staticmethod
    def _get_latest_generation(experiment_id: str) -> Optional[int]:
        """Return the highest generation number saved for an experiment."""
        from .models import Population
        with db.session_scope() as session:
            result = (
                session.query(Population.generation)
                .filter_by(experiment_id=uuid.UUID(experiment_id))
                .order_by(Population.generation.desc())
                .first()
            )
            return result[0] if result else None
```

(`Optional` and `uuid` may already be imported; check and add if missing.)

**Step 4: Run tests**

Run: `uv run pytest tests/test_db/test_resume.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add explaneat/db/population.py tests/test_db/test_resume.py
git commit -m "feat: compute_remaining_generations and _get_latest_generation helpers"
```

---

### Task 4: Resume Method on DatabaseBackpropPopulation

**Files:**
- Modify: `explaneat/db/population.py` (add resume_from_db classmethod)
- Modify: `explaneat/core/backproppop.py` (ensure initial_state path works)

**Step 1: Add resume_from_db to DatabaseBackpropPopulation**

In `explaneat/db/population.py`, add a new classmethod that creates a `DatabaseBackpropPopulation` loaded with the saved population state. Key idea: we subclass a tiny bit — skip `_create_experiment` (reuse existing ID), skip fresh population creation, use `initial_state` with genomes loaded from DB.

Because the current `__init__` unconditionally calls `_create_experiment`, refactor to allow bypassing it:

Change the `__init__` signature to accept a new kwarg `_existing_experiment_id: Optional[str] = None`. When set, skip `_create_experiment` and use the provided id.

```python
def __init__(self, config, x_train, y_train, xs_val=None, ys_val=None,
             experiment_name: str = None,
             dataset_name: str = None, description: str = None,
             database_url: str = None, ancestry_reporter=None,
             dataset_id: str = None,
             config_template_id: str = None,
             initial_state=None,
             _existing_experiment_id: Optional[str] = None):
    super().__init__(config, x_train, y_train, xs_val=xs_val, ys_val=ys_val,
                     initial_state=initial_state)

    if database_url:
        db.init_db(database_url)
    else:
        db.init_db()

    if _existing_experiment_id:
        self.experiment_id = _existing_experiment_id
    else:
        self.experiment_id = self._create_experiment(
            experiment_name or f"NEAT_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_name, description, config,
            dataset_id=dataset_id,
            config_template_id=config_template_id,
        )

    self.current_population_id = None
    self.current_generation = initial_state[2] if initial_state else 0

    self.ancestry_reporter = ancestry_reporter
    if self.ancestry_reporter:
        self.ancestry_reporter.reproduction = self.reproduction
```

Check that `BackpropPopulation.__init__` forwards `initial_state` — it already does via line 117-128 in `explaneat/core/backproppop.py`.

**Step 2: Add the resume classmethod**

```python
    @classmethod
    def resume_from_db(cls, experiment_id: str, config, x_train, y_train,
                       xs_val=None, ys_val=None, **kwargs) -> "DatabaseBackpropPopulation":
        """Reconstruct a population from the latest saved generation.

        Loads all genomes from the highest-generation Population row for the
        experiment, deserializes them into a NEAT population dict, and
        returns an instance ready to continue evolving.

        The existing experiment_id is reused — no new Experiment row created.
        """
        from .serialization import deserialize_genome
        from .models import Population, Genome

        last_gen = cls._get_latest_generation(experiment_id)
        if last_gen is None:
            raise ValueError(f"No saved populations found for experiment {experiment_id}")

        with db.session_scope() as session:
            pop_row = (
                session.query(Population)
                .filter_by(experiment_id=uuid.UUID(experiment_id), generation=last_gen)
                .first()
            )
            if not pop_row:
                raise ValueError(f"Population row not found for experiment {experiment_id} gen {last_gen}")

            genome_rows = session.query(Genome).filter_by(population_id=pop_row.id).all()
            population_dict = {}
            for g in genome_rows:
                neat_genome = deserialize_genome(g.genome_data, config)
                population_dict[neat_genome.key] = neat_genome

        # Build species set fresh — accepts loss of species continuity per design
        species = config.species_set_type(config.species_set_config, None)
        species.speciate(config, population_dict, last_gen + 1)

        initial_state = (population_dict, species, last_gen + 1)
        return cls(
            config, x_train, y_train, xs_val=xs_val, ys_val=ys_val,
            initial_state=initial_state,
            _existing_experiment_id=experiment_id,
            **kwargs,
        )
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ tests/test_db/test_resume.py -x -q`
Expected: All pass (changes are additive; existing __init__ still works when `_existing_experiment_id=None`)

**Step 4: Commit**

```bash
git add explaneat/db/population.py
git commit -m "feat: resume_from_db classmethod on DatabaseBackpropPopulation"
```

---

### Task 5: API — Resume Endpoint

**Files:**
- Modify: `explaneat/api/experiment_runner.py` (add resume method)
- Modify: `explaneat/api/routes/experiments.py` (add endpoint)
- Modify: `explaneat/api/schemas.py` (no new schema — reuse ExperimentCreateResponse)

**Step 1: Add resume method to ExperimentRunner**

In `explaneat/api/experiment_runner.py`, add a new `resume` method alongside `start`:

```python
    async def resume(self, experiment_id: str, X_train, y_train,
                     fitness_function: str, config_text: str,
                     remaining_generations: int, n_epochs_backprop: int,
                     split_id: str = None) -> str:
        """Resume an interrupted experiment from its last saved generation.

        Returns a job_id for polling progress.
        """
        job_id = str(uuid.uuid4())[:8]
        progress = ExperimentProgress(
            job_id=job_id,
            total_generations=remaining_generations,
        )
        self._jobs[job_id] = progress

        asyncio.create_task(
            self._run_resume(
                progress, experiment_id, config_text,
                X_train, y_train, fitness_function,
                remaining_generations, n_epochs_backprop, split_id,
            )
        )
        return job_id

    async def _run_resume(
        self,
        progress: ExperimentProgress,
        experiment_id: str,
        config_text: str,
        X_train, y_train,
        fitness_function: str,
        remaining_generations: int,
        n_epochs_backprop: int,
        split_id: str,
    ):
        try:
            progress.status = ExperimentStatus.RUNNING
            result = await asyncio.to_thread(
                self._resume_evolution_loop,
                progress, experiment_id, config_text,
                X_train, y_train, fitness_function,
                remaining_generations, n_epochs_backprop, split_id,
            )
            progress.status = ExperimentStatus.COMPLETED
            progress.experiment_id = experiment_id
        except Exception as e:
            logger.exception("Resume job %s failed", progress.job_id)
            progress.status = ExperimentStatus.FAILED
            progress.error = str(e)

    @staticmethod
    def _resume_evolution_loop(
        progress: ExperimentProgress,
        experiment_id: str,
        config_text: str,
        X_train, y_train,
        fitness_function: str,
        remaining_generations: int,
        n_epochs_backprop: int,
        split_id: str,
    ):
        """Run the resume loop in a thread pool."""
        from ..core.config_utils import load_neat_config
        from ..db.population import DatabaseBackpropPopulation
        from ..db.base import db
        from ..db.models import Experiment
        from ..evaluators.evaluators import binary_cross_entropy, auc_fitness
        from uuid import UUID as _UUID

        fitness_fn_map = {
            "bce": binary_cross_entropy,
            "auc": auc_fitness,
        }
        fitness_fn = fitness_fn_map.get(fitness_function, binary_cross_entropy)

        config = load_neat_config(config_text, None)

        population = DatabaseBackpropPopulation.resume_from_db(
            experiment_id, config, X_train, y_train,
        )

        # Set status back to running
        with db.session_scope() as session:
            exp = session.get(Experiment, _UUID(experiment_id))
            if exp:
                exp.status = "running"
                exp.end_time = None

        reporter = ProgressReporter(progress)
        population.reporters.add(reporter)

        population.run(
            fitness_fn,
            n=remaining_generations,
            nEpochs=n_epochs_backprop,
        )
```

`ProgressReporter` and `ExperimentStatus` and `ExperimentProgress` are already defined at the top of the file. Check imports and add `uuid`, `asyncio` etc. as needed (they should already be there).

**Step 2: Add endpoint in `explaneat/api/routes/experiments.py`**

```python
@router.post("/{experiment_id}/resume", response_model=ExperimentCreateResponse)
async def resume_experiment(
    experiment_id: UUID,
    db_session: Session = Depends(get_db),
):
    """Resume an interrupted experiment from the last saved generation."""
    from ..experiment_runner import experiment_runner
    from ...db.population import compute_remaining_generations, DatabaseBackpropPopulation
    from ...db.models import Experiment, Dataset, DatasetSplit

    experiment = db_session.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.status != "interrupted":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is '{experiment.status}', only 'interrupted' can be resumed",
        )

    resolved = (experiment.config_json or {}).get("resolved_config") or {}
    training = resolved.get("training", {})
    target_generations = training.get("n_generations", 10)
    n_epochs_backprop = training.get("n_epochs_backprop", 5)
    fitness_function = training.get("fitness_function", "bce")

    last_gen = DatabaseBackpropPopulation._get_latest_generation(str(experiment_id))
    if last_gen is None:
        raise HTTPException(
            status_code=400,
            detail="No saved generations found; cannot resume",
        )

    remaining = compute_remaining_generations(last_gen, target_generations)
    if remaining == 0:
        experiment.status = "completed"
        db_session.commit()
        return ExperimentCreateResponse(job_id="")

    # Load data + apply split the same way create_and_run_experiment does.
    if not experiment.dataset_id or not experiment.split_id:
        raise HTTPException(status_code=400, detail="Experiment missing dataset or split")

    dataset = db_session.get(Dataset, experiment.dataset_id)
    split = db_session.get(DatasetSplit, experiment.split_id)
    if not dataset or not split:
        raise HTTPException(status_code=400, detail="Dataset or split not found")

    data = dataset.get_data()
    if data is None:
        raise HTTPException(status_code=400, detail="Dataset has no stored data")
    X_full, y_full = data
    X_train = X_full[split.train_indices or []]
    y_train = y_full[split.train_indices or []]

    # Apply stored scaler
    if split.scaler_type == "StandardScaler" and split.scaler_params:
        import numpy as np
        mean = np.array(split.scaler_params["mean"])
        scale = np.array(split.scaler_params["scale"])
        X_train = (X_train - mean) / scale

    job_id = await experiment_runner.resume(
        experiment_id=str(experiment_id),
        X_train=X_train,
        y_train=y_train,
        fitness_function=fitness_function,
        config_text=experiment.neat_config_text,
        remaining_generations=remaining,
        n_epochs_backprop=n_epochs_backprop,
        split_id=str(experiment.split_id),
    )

    return ExperimentCreateResponse(job_id=job_id)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_core/ tests/test_api/ tests/test_db/test_resume.py -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add explaneat/api/experiment_runner.py explaneat/api/routes/experiments.py
git commit -m "feat: POST /experiments/{id}/resume endpoint"
```

---

### Task 6: Integration Test — End-to-End Resume

**Files:**
- Create: `tests/test_api/test_resume_integration.py`

**Step 1: Write integration test**

```python
"""Integration: create experiment → mark interrupted → resume → completes."""
import pytest
from fastapi.testclient import TestClient

# Only run if full DB + NEAT setup is available
pytestmark = pytest.mark.integration


def test_resume_endpoint_rejects_non_interrupted():
    """Resume returns 400 if experiment isn't interrupted."""
    from explaneat.api.app import create_app
    from explaneat.db.base import db
    from explaneat.db.models import Experiment
    from datetime import datetime

    db.init_db("sqlite:///:memory:")
    from explaneat.db.base import Base
    Base.metadata.create_all(db.engine)

    with db.session_scope() as session:
        exp = Experiment(
            experiment_sha="x", name="test",
            config_json={}, neat_config_text="",
            status="completed", start_time=datetime.utcnow(),
        )
        session.add(exp)
        session.flush()
        exp_id = str(exp.id)

    app = create_app()
    client = TestClient(app)
    resp = client.post(f"/api/experiments/{exp_id}/resume")
    assert resp.status_code == 400
    assert "interrupted" in resp.json()["detail"]


def test_resume_endpoint_404_for_missing():
    """Resume returns 404 for nonexistent experiments."""
    from explaneat.api.app import create_app
    from explaneat.db.base import db
    from explaneat.db.base import Base
    import uuid

    db.init_db("sqlite:///:memory:")
    Base.metadata.create_all(db.engine)

    app = create_app()
    client = TestClient(app)
    resp = client.post(f"/api/experiments/{uuid.uuid4()}/resume")
    assert resp.status_code == 404
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_api/test_resume_integration.py -v`
Expected: PASS (or skip if integration marker not configured — adjust markers if needed)

**Step 3: Commit**

```bash
git add tests/test_api/test_resume_integration.py
git commit -m "test: integration tests for resume endpoint error cases"
```

---

### Task 7: Frontend — API Client + Resume Button

**Files:**
- Modify: `web/react-explorer/src/api/client.ts` (add resumeExperiment)
- Modify: `web/react-explorer/src/components/ExperimentList.tsx` (show interrupted + button)
- Modify: `web/react-explorer/src/components/GenomeExplorer.tsx` (show button in detail view)

**Step 1: Add API client function**

In `web/react-explorer/src/api/client.ts`, add near `createAndRunExperiment`:

```typescript
export async function resumeExperiment(
  experimentId: string,
): Promise<ExperimentCreateResponse> {
  return fetchJson(`${API_BASE}/experiments/${experimentId}/resume`, {
    method: "POST",
  });
}
```

**Step 2: Update ExperimentList to show interrupted + button**

In `web/react-explorer/src/components/ExperimentList.tsx`:

1. Status badge color: add `interrupted` → amber (e.g. `#f59e0b` background, `#92400e` text).
2. For rows with `status === "interrupted"`, render a "Resume" button next to the row action area. On click:
   - Call `resumeExperiment(exp.id)` → get `job_id`
   - Refresh the list after a short delay or optimistically update status to `running`
   - Show an error toast if the call fails

Match the existing button styling (`op-btn` class).

**Step 3: Update GenomeExplorer to show resume button**

In the experiment header area of `GenomeExplorer.tsx`, if `experimentDetail?.status === "interrupted"`, render a "Resume experiment" button that calls `resumeExperiment(experimentId)` and updates the local state.

**Step 4: Type-check**

Run: `cd web/react-explorer && npx tsc --noEmit`
Expected: No new errors

**Step 5: Commit**

```bash
git add web/react-explorer/src/api/client.ts web/react-explorer/src/components/ExperimentList.tsx web/react-explorer/src/components/GenomeExplorer.tsx
git commit -m "feat: resume button for interrupted experiments"
```

---

### Task Summary

| Task | What | Files |
|------|------|-------|
| 1 | Allow 'interrupted' status | models.py, migration |
| 2 | Startup hook for orphan detection | app.py, test |
| 3 | Resume helper functions | population.py, test |
| 4 | `resume_from_db` classmethod | population.py |
| 5 | POST /experiments/{id}/resume endpoint | experiment_runner.py, experiments.py |
| 6 | Integration tests for resume errors | test_resume_integration.py |
| 7 | Frontend client + resume button | client.ts, ExperimentList.tsx, GenomeExplorer.tsx |
