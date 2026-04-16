# Experiment Checkpointing and Resume

**Date:** 2026-04-17
**Status:** Approved

## Problem

When the API server restarts during local development (e.g., from uvicorn hot-reload triggered by a code change), background threads running NEAT experiments die silently. The experiment's status column stays `running` forever, and hours of evolved population work become orphaned with no way to continue. The user must start from scratch.

## Design

### Core Insight

Most resume infrastructure already exists. Each generation's genomes are persisted via `_save_genomes()`, and a `Population` record captures the generation number. Resume does not need to reconstruct the exact in-memory NEAT state — it just needs to load the latest population's genomes and continue evolving from that point with the same config.

**Explicit tradeoff:** Resume is best-effort, not deterministic. Species state, innovation counter, RNG state, and optimizer state all reset. Evolution picks up from the saved population and continues productively, but the exact sequence won't match a non-crashed run. For reproducible runs the user will not use dev-mode hot-reload.

### Data Model

**Experiment status** gains a new allowed value:

| Status | Meaning |
|---|---|
| `running` | Currently executing |
| `completed` | Finished all generations successfully |
| `failed` | Errored out during run |
| `paused` | User-triggered pause (existing, unchanged) |
| **`interrupted`** | **Was running when server died; resumable** |

No new tables. The existing `Population` table, with its `(experiment_id, generation)` constraint and FK-attached genomes, serves as the per-generation checkpoint. The unused `Checkpoint` model stays in place for a future deterministic-resume feature if ever needed.

### Startup Detection

On FastAPI app startup:

1. Query `Experiment` where `status == "running"`
2. For each, set `status = "interrupted"` and `end_time = now()`

This is a single SQL UPDATE, idempotent, safe to run every time the server starts.

Implementation: `@app.on_event("startup")` hook in `explaneat/api/app.py` that executes the UPDATE.

### Resume Flow

New endpoint: `POST /api/experiments/{id}/resume`

Steps:

1. Load experiment; verify `status == "interrupted"` (404 otherwise)
2. Find highest generation in `Population` for this experiment → `last_gen`
3. Load all genomes from that `Population` record
4. Read original generation target from `experiment.config_json["resolved_config"]["training"]["n_generations"]`
5. Compute `remaining = n_generations - (last_gen + 1)`. If ≤ 0, mark `completed` and return
6. Reconstruct NEAT config from `experiment.neat_config_text`
7. Deserialize genomes into a population dict keyed by `genome.key`
8. Launch a new background job with:
   - The same config
   - `initial_state = (population_dict, fresh_species_set, last_gen + 1)`
   - `remaining` generations to run
   - The SAME `experiment_id` (reusing the record, not creating a new one)
9. Set `status = "running"`
10. Return the new `job_id`

The heavy lift happens in `DatabaseBackpropPopulation` — it needs to support being constructed with an existing `experiment_id` (instead of creating a new Experiment row) and with `initial_state` from saved genomes. A new classmethod `resume_from_db(experiment_id, ...)` encapsulates the loading.

### UI Changes

**ExperimentList:**
- New status badge for `interrupted` in distinct color (amber/orange)
- **Resume** button appears on rows with `status == "interrupted"`

**GenomeExplorer (experiment detail):**
- If `status == "interrupted"`, show a **Resume** button near the header

Both buttons call `resumeExperiment(id)` and then poll progress on the returned `job_id` using existing infrastructure.

### API Client

New function in `web/react-explorer/src/api/client.ts`:

```typescript
export async function resumeExperiment(
  experimentId: string,
): Promise<ExperimentCreateResponse>
```

Returns a `job_id` matching the shape of `POST /experiments/run`, so the existing polling machinery handles progress.

### Migration

Update the `CheckConstraint` on `experiments.status` to add `'interrupted'` to the allowed values.

### What We Accept Losing on Resume

- **Species state** — fresh speciation; generations immediately after resume may briefly show odd species churn, stabilizing within a few generations
- **Innovation counter** — NEAT-Python recomputes from existing genome innovations, so this is functionally fine
- **RNG state** — future mutations won't match what a non-crashed run would have produced
- **Optimizer state (Adadelta)** — already resets per-generation in the existing code (each backprop creates a fresh optimizer), so no additional loss

### Testing

- Unit test: startup hook transitions `running` → `interrupted`
- Unit test: `resume_from_db` computes `remaining_generations` correctly
- Unit test: resume endpoint returns 404 if experiment is not interrupted
- Unit test: resume endpoint returns completed status if `last_gen + 1 >= n_generations`
- Integration test: create experiment → directly set status to interrupted → call resume → experiment runs to originally-configured total
