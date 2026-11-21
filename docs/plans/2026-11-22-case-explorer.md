# Case Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the researcher see, edit and choose the reference case that "others held fixed" plots use, and see how typical that case is.

**Architecture:** A reference case is one full input row x in the dataset's raw columns (before the split's scaler).
`explaneat/analysis/reference_case.py` resolves a spec (median case / mean / quantile / dataset row, plus
per-column and per-one-hot-group overrides) into x and describes it (labels, prediction, nearest training rows,
per-column percentiles, one-hot pattern support). The client sends the resolved vector as
`params.reference = {x: [...], spec: {...}}`; the viz route scales x, forward-passes it to the scope's entry
activations, and passes them as `fixed=` to the fix-at-median functions in `viz_data`. No `reference` means the
old per-column median (default; static appendix keys unchanged). Design: `2026-11-22-case-explorer-design.md`.

**Tech Stack:** Python (numpy, FastAPI, pytest), React + TypeScript (Vite).

---

### Task 1: `viz_data` accepts a fixed background
**Files:** Modify `explaneat/analysis/viz_data.py` (`compute_line_plot`, `compute_heatmap`,
`compute_partial_dependence`, `compute_feature_output_scatter`, `compute_pairs`); Test `tests/test_analysis/test_viz_reference.py`.
- Each takes `fixed: Optional[np.ndarray] = None` (length n_in). None -> `np.median(entry_acts, axis=0)` as now.
- Each result gains `held: list` (the background used) and `held_rule: "median" | "reference"`.
- `compute_pairs`: with `fix="median"` and `fixed` given, B = fixed; `reference` = fn(fixed); ignored for `fix="average"`.
- Tests: `fixed=None` output equals the old output byte-for-byte (compare to medians); with `fixed`, grid rows equal
  `fixed` except the varied dim(s); pairs reference equals fn(fixed).

### Task 2: `reference_case.py`
**Files:** Create `explaneat/analysis/reference_case.py`; Test `tests/test_analysis/test_reference_case.py`.
- `column_kinds(X, names)` -> per column `continuous | binary | onehot` and `onehot_groups` (reuse
  `explanation_tools.onehot_groups`).
- `resolve(spec, X_raw, names, X_rows_index=None)` -> x (raw units). Spec keys: `base` in
  `median | mean | quantile | row | per_column_median`, `q`, `row` (dataset row index), `overrides`
  `{column name: value}`, `groups` `{base name: level name | null}` (null = all zeros).
  median case: continuous median, binary mode, one-hot group = most common pattern among rows.
  `per_column_median`: the literal median of each column (what plots use by default).
- `describe(x, X_raw, y, names, predict, labels, task)`:
  per column `{column, raw, label, value, kind, percentile, base_value}`; groups `{base, levels, active, support}`
  (training rows with the same pattern); prediction `{outputs, percentile, class?}`; `nearest`: k=5 rows by
  standardised Euclidean distance `{row, distance, outcome, prediction, differs: [col...]}`; `typicality`:
  percentile of x's NN distance among the NN distances of up to 500 seeded training rows; `warnings` (e.g.
  "group marital: no training row has this pattern").
- Tests on a tiny synthetic dataset with a one-hot group: median case has exactly one level on; per-column median
  has none and is flagged; row spec returns that row; nearest[0] of a real row is itself at distance 0; deterministic.

### Task 3: API
**Files:** Modify `explaneat/api/routes/evidence.py`, `explaneat/api/schemas.py`; Test `tests/test_api/test_reference_case_api.py`.
- `POST /api/genomes/{id}/evidence/reference-case` body `{dataset_split_id, annotation_id?, node_id?, spec}` ->
  `{x, columns, groups, prediction, nearest, typicality, warnings, entry: {names, values}}` (`entry` = the scope's
  entry activations at x, what the plot holds).
- `POST .../evidence/rows` body `{dataset_split_id, split, offset, limit, outcome?}` -> rows with index, outcome, prediction.
- `viz-data`: `params.reference.x` -> scale with the split's scaler -> entry row (whole model: the scaled row;
  annotation/node: `ActivationExtractor.extract`) -> `fixed=`; 400 when its length is wrong.
- Source view for line/feature_output_scatter (`_apply_source_view`) untouched; `held` stays in network units.

### Task 4: MCP
**Files:** Modify `mcp_server/tools/evidence.py`; doc string of `compute_viz_data` mentions `reference`.
Share the helper `reference_entry_values` from `reference_case.py`.

### Task 5: Frontend
**Files:** Create `web/react-explorer/src/components/ReferenceCasePanel.tsx`, `src/api/referenceCase.ts`,
`src/components/referenceCase.ts` (pure helpers: URL encode/decode, spec edits); Modify `EvidencePanel.tsx`
(context + strip + params), `VizCanvas.tsx` (marker at `held`), `pairsPlot.ts` (reference crosshair, text),
`api/staticData.ts` (no change in keys; case calls return null -> strip read-only), `GenomeExplorer.tsx` (provider + `?case=`).
- One case per genome view (context in GenomeExplorer), opt-in toggle "Hold others at a reference case".
- Strip: summary line; expand -> table (value input, median/mean/q25/q75, binary toggle, group select), row picker
  (index box + "nearest rows" list + "pick a random row with outcome …"), reset.
- Plots not using a background say "reference not used". Descriptions say "others at the reference case".
- Fix the partial dependence text (it holds others at the median; it does not average).
- Pins and snapshots carry `params.reference` automatically.

### Task 6: Docs, bundle, tests
- `docs/react_explorer_requirements.md` section "Reference case"; CLAUDE.md line under EvidencePanel.
- `cd web/react-explorer && npm ci && npx tsc --noEmit -p . && npm run build`, `uv run python -m explaneat setup --skip-db`, commit bundle.
- `uv run pytest tests/test_analysis tests/test_api tests/test_install/test_ui_bundle.py -q`.

### Task 7: Verify live
Run the API against the local SQLite (`~/.explaneat`) or a read-only Postgres, open a genome, toggle the case,
edit a one-hot group, pick a row, check the plot moves and the marker sits at the case.
