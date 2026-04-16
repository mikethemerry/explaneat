# Configurable NEAT Training + Config Templates

**Date:** 2026-04-16
**Status:** Approved

## Problem

The experiment creation form only exposes four parameters (name, dataset, population size, generations, epochs, fitness function). All other NEAT mutation rates, topology probabilities, backprop settings, and species configuration are hardcoded in `_default_neat_config_text()`. Researchers cannot tune the evolution/training strategy without editing code, and there is no way to save and reuse a good configuration across experiments.

## Design

### 1. Data Model

**New `ConfigTemplate` table:**

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | primary key |
| `name` | str(255) | display name |
| `description` | Text | optional |
| `config` | JSONB | structured config (see below) |
| `created_at` | timestamp | |
| `updated_at` | timestamp | |

**`Experiment` additions:**

- `config_template_id` (optional FK → `config_templates.id`, `ondelete="SET NULL"`) — records which template the experiment was based on

The existing `config_json` column on `Experiment` continues to hold the **final resolved config** (after merging template + overrides). No change needed.

### 2. Config Structure

Templates and resolved configs use a flat JSON structure organized by conceptual groups:

```json
{
  "training": {
    "population_size": 150,
    "n_generations": 10,
    "n_epochs_backprop": 5,
    "fitness_function": "bce"
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
    "survival_threshold": 0.2
  },
  "backprop": {
    "learning_rate": 1.5,
    "optimizer": "adadelta"
  }
}
```

Deliberately excluded (stay hardcoded for now): `activation_default`, `aggregation_default`, `feed_forward`, initial weight stats, value bounds. These can be added later if needed.

### 3. Config Resolution

On experiment creation, the final config is resolved in three layers:

1. **Defaults** — the hardcoded values currently in `_default_neat_config_text()`
2. **Template** — values from the selected template (if any), overriding defaults
3. **Overrides** — per-experiment overrides from the creation request, overriding template values

The resolved config is stored in `config_json` on the experiment and also fed into `build_neat_config_text()` to produce the actual NEAT config text used for the run.

### 4. API

**New endpoints:**

- `GET /api/config-templates` — list all
- `GET /api/config-templates/{id}` — get one
- `POST /api/config-templates` — create
- `PATCH /api/config-templates/{id}` — update
- `DELETE /api/config-templates/{id}` — delete

**Modified `POST /api/experiments/run`:**

Add to `ExperimentCreateRequest`:
- `config_template_id: Optional[str]` — template to use as base
- `config_overrides: Optional[Dict]` — same flat structure, merged over the template

The existing top-level fields (`population_size`, `n_generations`, `n_epochs_backprop`, `fitness_function`) continue to work as shortcuts for `training.*` overrides — backwards compatible.

### 5. UI

**Experiment Create Modal:**

- **Template dropdown** at the top of the form — lists all templates plus "Custom (no template)". Default selection: the "Default" seeded template.
- **Expandable "Advanced Config"** section (collapsed by default) with three sub-groups:
  - **Training** — population_size, n_generations, n_epochs_backprop, fitness_function
  - **NEAT Mutation & Topology** — all mutation rates and add/delete probs, species settings
  - **Backprop** — learning_rate, optimizer
- Each field is pre-filled from the selected template. Editing creates an override.
- Template switching re-populates fields (with an "unsaved changes will be lost" confirm if the user has edited).
- **"Save as new template"** button captures current values into a new template.

**Templates Management Page** (new, accessed via nav):

- List all templates with name, description, created date
- Row actions: Edit / Duplicate / Delete
- Click row to edit (same grouped field layout as Advanced Config section in the modal)
- "Create new" button

**Genome Explorer (Experiment Detail):**

- New **"Training Config"** section near the top, collapsible
- Shows the resolved config grouped by Training / NEAT / Backprop
- If `config_template_id` is set, shows "Based on template: [Template Name](link)" at the top
- Read-only — this is history

### 6. Migration

- Create `config_templates` table
- Add `config_template_id` column to `experiments` with FK constraint
- Seed a single "Default" template with the values currently hardcoded in `_default_neat_config_text()`

### 7. What Stays the Same

- `_default_neat_config_text()` continues to be the source of hardcoded defaults, but now acts as the base layer that templates/overrides merge on top of
- `neat_config_text` on experiments is still the generated text used for the actual run
- `DatabaseBackpropPopulation`, training code paths — no changes needed
- Existing experiments (no template_id) continue to display their config from the existing `config_json`
