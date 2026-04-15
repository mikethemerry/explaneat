# One-Hot Encoding and Source/Network View Toggle

**Date:** 2026-04-15
**Status:** Approved

## Problem

The system stores categorical feature type metadata but never acts on it. Networks train on raw integer-encoded categoricals, which treats `workclass=3` as "3x the effect of workclass=1" — meaningless for unordered categories. Additionally, the evidence panel only shows data in the network's transformed space (scaled, encoded), with no way to view original human-readable values.

## Design

### 1. Data Model Changes

`Dataset` gets two new columns:

- **`source_dataset_id`** (optional FK → `datasets.id`): Links a prepared dataset back to its original. NULL for original datasets.
- **`encoding_config`** (JSONB): Records how this prepared dataset was created.

```json
{
  "categorical": {
    "workclass": ["Private", "Self-emp-not-inc", "Federal-gov", "..."]
  },
  "ordinal_as_onehot": {
    "education": ["Preschool", "1st-4th", "5th-6th", "...", "Doctorate"]
  },
  "ordinal_as_ranked": {
    "education_num": ["low", "medium", "high"]
  },
  "passthrough": ["age", "hours_per_week", "capital_gain"]
}
```

### 2. Column Naming Convention

| Original Type | Treatment | Resulting Columns |
|---|---|---|
| numeric / integer | Passthrough | `age` (unchanged) |
| categorical | One-hot | `workclass:Private`, `workclass:Self-emp`, ... |
| ordinal (default) | Ranked integer | `education` (values remapped to 0, 1, 2...) |
| ordinal (opted in) | One-hot | `education:Bachelors`, `education:Masters`, ... |

Delimiter is `:` — clean in display, easy to parse, won't collide with existing feature names.

### 3. Preparation Flow

**Manual (Dataset Detail page):**
1. User sets `feature_types` to categorical/ordinal on the dataset
2. For ordinal features, user specifies rank ordering of values
3. For ordinal features, user optionally checks "one-hot encode" (default: ranked integer)
4. User clicks "Prepare Dataset", optionally overrides the default name (`{name} (prepared)`)
5. System creates a new `Dataset` record with:
   - Expanded X data (one-hot columns replacing categoricals)
   - Updated `feature_names` reflecting new columns
   - Updated `feature_types` (one-hot columns become `binary`)
   - `source_dataset_id` pointing to the original
   - `encoding_config` recording all decisions

**Automatic (Experiment creation):**
1. When creating an experiment, if the selected dataset has categorical/ordinal features and no prepared version exists with matching encoding config, create one automatically using default name
2. If a matching prepared version already exists, reuse it
3. The experiment trains on the prepared dataset

### 4. Encoding Engine

A function `prepare_dataset(dataset, encoding_config) -> Dataset` that:

1. Loads the source dataset's X data and feature_names
2. For each feature, based on encoding_config:
   - **passthrough**: Copy column as-is
   - **categorical**: One-hot encode using the provided category list. Unknown values get all-zeros row.
   - **ordinal_as_ranked**: Map values to integer ranks per the provided ordering
   - **ordinal_as_onehot**: One-hot encode like categorical, but store ordering in config for potential future use
3. Builds new feature_names with `:` delimiter for expanded columns
4. Creates and saves a new Dataset record

### 5. Evidence Panel: Source vs Network View Toggle

A toggle in the evidence panel's dataset section: **Network** (default) | **Source**.

| Aspect | Network View | Source View |
|---|---|---|
| Feature names | `workclass:Private` | `workclass` |
| Feature values on axes | Scaled (-2.1, 0.3, ...) | Original (39, 50000, ...) |
| One-hot features | Individual binary columns | Grouped back to single original feature |
| Line plot X axis | Scaled range | Original value range |
| SHAP | Per one-hot column importance | Summed importance per original feature |

**Implementation:**

The evidence endpoint gets an optional `view` parameter: `"network"` (default) or `"source"`. When `source`:

1. **Unscale values**: Use `scaler_params` from the split to reverse StandardScaler
2. **Regroup one-hot columns**: Use `encoding_config` from the prepared dataset + `source_dataset_id` to map expanded columns back to original features
3. **SHAP grouping**: Sum absolute SHAP values for columns belonging to the same original feature
4. **Categorical plots**: For one-hot features in source view, show grouped/bar representations instead of continuous axes

**Data needed** (all available):
- `source_dataset_id` on prepared dataset → load original feature names
- `encoding_config` on prepared dataset → reverse the one-hot mapping
- `scaler_params` on split → reverse scaling

### 6. Migration

Add to `datasets` table:
- `source_dataset_id` UUID FK (nullable, ondelete CASCADE)
- `encoding_config` JSONB (nullable)

### 7. What Stays the Same

- `StructureNetwork`, `NeuralNeat`, forward pass code — unchanged. They work with whatever shape the dataset has.
- `StandardScaler` in experiment creation — still applied after one-hot encoding
- Split creation — works on the prepared dataset
- All existing visualization code — unchanged, just receives different feature names
