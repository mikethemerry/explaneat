# Case explorer: design (approved by Mike, 22 Nov 2026)

## How it works today
- `viz_data.compute_line_plot`, `compute_heatmap`, `compute_partial_dependence`, `compute_feature_output_scatter`
  and `compute_pairs(fix="median")` all build the background row as `np.median(entry_acts, axis=0)`.
  That is a per-column median in the annotation's ENTRY space (hidden activations for a non-leaf part),
  over the rows sent (train split, possibly sampled).
- Consequences: (a) one-hot groups: each level column's median is usually 0, so the "median case"
  often has NO category set (or several), not a real person; (b) for an annotation with hidden entries,
  the per-entry medians need not come from any input at all; (c) nothing shows the case or its prediction.
- `partial_dependence` with `fix_values="median"` is the same sweep as `line` (the help text in
  EvidencePanel says it averages over the data; the code does not). Fix this in passing.
- Static appendix keys viz files on `input_dim`/`dims`/`fix` (`staticData.postKey`).
- Snapshots store `viz_config`, pins store `viz.params`: anything put in `params` is stored for free.

## Proposed design (approach A: one case in input space)
**Reference case** = one full input vector x* in the dataset's columns, chosen as
`params.reference = {base: "median"|"mean"|"row"|"quantile", q?, row?: {split, index},
overrides: {<column raw name>: value}}` (one-hot groups overridden by level: `{"marital": "2"}`).
Defaults: continuous = median, binary = mode, one-hot group = modal level (exactly one 1).
The server turns it into x* (scaled), runs the forward pass, and uses the case's ENTRY activations
as the fixed background for every "others held" viz, for the whole model and for annotations alike.
So every part is read at the same, realistic case, and a hidden entry is held at what x* gives it.

- `viz_data`: each fix-at-median function takes `fixed: Optional[np.ndarray]` (None = today's
  per-column median, kept for old snapshots/pins and static files). Output carries `reference_point`
  (entry values) so plots draw a marker at the case (line: vertical rule; heatmap/pairs: crosshair).
- New library `explaneat/analysis/reference_case.py`: resolve spec -> x*; describe the case:
  per column raw name, Feature key label, value in source units (unscaled), base used;
  prediction (prob / class; survival: log-risk + risk at horizon); typicality: k=5 nearest train rows
  (standardised Euclidean over used columns) with their outcome and model prediction, distance
  percentile of x* vs the train rows' own nearest-neighbour distances, per-column percentile,
  one-hot validity flag.
- New route `POST /api/genomes/{id}/evidence/reference-case` (spec in, case out) and
  `GET .../evidence/rows?split=&q=` (find a row by index / outcome / feature filter, small pages).
- UI: a "Reference case" strip under the viz controls in EvidencePanel (one line: "median case ·
  predicted 0.31 · nearest row #412, outcome 1 · typical (38th pct)"), expanding to an editable table:
  per input a value box (source units) with median/mean/q25/q75 buttons, toggles for binary, a select
  per one-hot group, "use row #…", "reset". State lives in GenomeExplorer (one case for all plots and
  parts), mirrored in the URL (`?case=` compact JSON) so a view is shareable. Plot recomputes on change
  (debounced). Plots that do not use a background (ICE, sensitivity, pairs "averaged", histograms)
  show "reference not used".
- Reproducibility: snapshots and pins store `params.reference` plus the resolved x* (source units),
  so a reopened plot uses the same case even if the split's median moves.
- Static appendix: no `reference` in params -> same keys as now. The case strip is shown read-only
  as "median (per column)" with a note that editing needs the live server.
- Restricted data (`EXPLANEAT_RESTRICTED=1`): nearest real rows are individual records; show them in
  the local UI only (never via MCP), or hide them -- Mike to decide.

## Alternatives considered
- B: per-plot overrides in entry space only (edit the held column values directly). Smaller, but a
  hidden entry can be set to impossible values and there is no "case" or prediction to look at.
- C: entry-space overrides as an "advanced" layer on top of A. Possible later; not proposed now.

## Decisions (Mike, 22 Nov 2026)
1. Default background stays the per-column median (existing figures unchanged); the reference case is
   opt-in. The case strip flags when the median case is unrealistic (e.g. an empty one-hot group).
2. Restricted datasets (`EXPLANEAT_RESTRICTED=1`): nearest real rows are hidden (counts and distances only).
3. MCP `compute_viz_data` accepts the same `params.reference`.
