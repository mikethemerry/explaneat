# ExConNEAT: graph decomposition and management (ExplaNEAT's method on conv graphs)

Branch `conv-neat-decomposition` (local, off `conv-neat-prototype`). No database, CPU only,
validation images only (the test split is never read).

## What was built

The previous `explaneat/conv/explain/` pipeline grouped feature maps into structures and split
shared maps. It had no operation stream, no collapse checks and no clean entries and exits, so
edges crossed into and out of the structures anywhere. This branch puts ExplaNEAT's own
decomposition machinery under the conv graph instead.

| File | What it does |
|---|---|
| `explaneat/conv/model_state.py` | **`ConvModelState`**: the phenotype as an ExplaNEAT `NetworkStructure` (string ids, node types, bias, activation; one `NetworkConnection` per kernel edge) plus two conv side tables, `levels` and `kernels` (keyed by `NetworkConnection.innovation`). Includes a float64 forward pass (naive-engine semantics, string ids, optional ancestor-only evaluation). **`ConvModelStateEngine`**: the operation stream (the core `Operation` records), replay, `add_operation` with validation, and `remove_operation(seq)` for undo. Also the JSON record (`make_record` / `save_record` / `load_record`), which carries the genome so a record replays on its own. |
| `explaneat/conv/operations.py` | **Handlers.** `split_node {node_id, partition}`, `consolidate_node`, `add_identity_node` (side `target`, ExplaNEAT's form, or side `source`, a "port"), `annotate`, `rename_annotation`. **`validate_conv_operation`.** **`annotation_strategy`** (UI rules, two modes), `execute_strategy`, `apply_splits`, `final_annotation`, and the graph helpers `reentry_nodes`, `collapse_cycle` and `classify`. |
| `explaneat/conv/decomposition.py` | **`suggest_candidates`** (suggestion only), **`repair_candidates`** (blocking, then merge or expand), the **`decompose`** driver, and the summary. The summary covers boundaries before and after, coverage, function preservation and blocking issues left. |
| `explaneat/conv/hypotheses.py` | A mechanism hypothesis for every annotation, tested on its exit and stored in its `annotate` op. |
| `explaneat/conv/decomposition_plot.py` | The static check graphs: the collapsed view and one expanded structure. |
| `experiments/conv_neat/decomposition/run_decomposition.py` | Runs the driver on the MNIST genome, writes the record, re-reads it from disk with every operation validated again, and draws the figures. |
| `tests/test_conv/test_decomposition.py` | 9 tests (below). |

### The operations (each one exact)

- **`split_node`.** Applies ExplaNEAT's `apply_split_node`: one copy per outgoing edge, ids
  `<base>_<letter>`, and every copy keeps all incoming edges. It then applies
  `apply_consolidate_node` once per group of `partition`, so each node becomes one copy per role
  (e.g. `19_ade`, `19_bc`). Kernels follow automatically, because the core handlers copy
  `innovation` onto every copied connection.
- **`add_identity_node`, side `target`** (ExplaNEAT's `apply_add_identity_node`). The intercepted
  edges go into a new identity map at the target's level, each keeping its kernel and resampling.
  The new map feeds the target through a 1×1 unit-delta kernel.
- **`add_identity_node`, side `source`** (a **port**). The edges out of one source are re-rooted
  at a new identity map at the source's level: `source → port` is the unit delta, and each
  `port → dst` keeps its kernel and resampling. The port is an exact copy of the source map. It
  serves as the **entry** of the structure it feeds, so every kernel into a structure is inside
  that structure.
- **`annotate`.** Records the name, hypothesis, display name, entries, exits, subgraph nodes and
  connections, evidence, and `child_annotation_ids`. Its nodes and connections become covered
  (immutable), as in ExplaNEAT.

### Validation on every `annotate`

Validation runs in three steps:

1. ExplaNEAT's `validate_operation`, unchanged: existence checks, the already-covered check,
   hierarchy checks, and **P1/P2/P3 on the annotation's own nodes**.
2. The three preconditions again on the **expanded** subgraph (own nodes plus all descendants),
   using `CollapseValidator.validate_collapsible`. Without this step, a composition with no own
   nodes would be checked against nothing.
3. Two conv additions:
   - an entry must be fed from outside only;
   - the collapse must stay a DAG, i.e. there is no path from an exit back to an entry through
     outside nodes. P1–P3 alone do not prevent that cycle.

Messages read, for example, `P1 violation: non-entry node 4 has external input from 3`,
`P3 violation (expanded subgraph): Exit node … has external input from …`, or
`collapse would create a cycle: exit X reaches entry Y through outside node W`.

### The annotation strategy (ported from `OperationsPanel.tsx`)

There are two modes.

- **`mode="ui"`: the rules as written.**
  - Entry = all inputs from outside; exit = all outputs outside; anything else is intermediate.
  - One target-side identity node when exactly one exit has external inputs. It replaces that exit.
  - Splits for entry and intermediate nodes that have external outputs.
  - A blocking issue for any intermediate or exit node with external inputs.
- **`mode="ports"`: the conv default.**
  - Every external input into the candidate is intercepted by a source-side identity node, one
    port per external source.
  - An input **cannot** be intercepted when its source lies on a path that leaves the candidate
    and re-enters it, because intercepting it would make the collapse cyclic. These are the
    blocking issues, reported as "expand the selection to include W".
  - Splits as in the UI, but per role (`role_of`). In the driver the roles are: inside the leaf,
    a sibling sub-part of the same structure, outside the structure.

Order: identity → splits → annotate. Splits are applied one at a time with the strategy
recomputed after each, because every split renames a node.

**Why ports rather than the UI's literal rules.** On a dense conv DAG most structure maps
receive skip edges from the image or from other structures. Under the literal rules every one of
those is a blocking issue, and "expand the selection" would merge everything into one structure.
A port is the same identity operation applied on the source side. It keeps each map's full
computation, with all its kernels, inside one structure, and the structure's exits are its real
feature maps, which is what the hypotheses are tested on.

**A finding about the UI rule.** The UI makes the split copy of an entry an exit. If the entry is
a hidden map, that copy is an exit fed from outside, which is a P3 violation. The validator
rejected exactly this case in the `ui`-mode test. `execute_strategy` therefore leaves such a
copy outside the annotation: it is fed from outside and read from outside. For input nodes the
UI is right, because they have no inputs. This probably applies to the main UI too
(`OperationsPanel.tsx`); it is worth checking there.

### Hierarchy

The hierarchy is: root → structures (5–8) → sub-parts.

- **Leaves.** Every sub-part, or a whole structure when it has a single sub-part. Entries are its
  ports; exits are its maps with an edge leaving it.
- **Compositions.** One per multi-leaf structure. It has no own nodes; its connections are the
  junction edges between its children; its entries are the children's ports whose source lies
  outside the structure.
- **Root.** Own nodes are the image and the score maps. Its connections are every junction edge
  between structures, from the image into ports, and into the score maps.

ExplaNEAT's composition checks (child exists, no second parent, no cycle in the hierarchy) apply
unchanged.

## Reused vs adapted vs new

**Reused directly, unchanged:**

- From `explaneat/core/operations.py`: `apply_split_node`, `apply_consolidate_node`,
  `apply_add_identity_node`, `validate_operation` (including P1–P3), `resolve_annotation_ref`,
  `canonicalize_annotation_refs` and `annotation_descendants`.
- From `explaneat/core/model_state.py`: `Operation` and `AnnotationData`.
- From `explaneat/core/genome_network.py`: `NetworkStructure` and its node and connection classes.
- From `explaneat/analysis/collapse_validator.py`: `CollapseGraph`, `CollapseAnnotation` and
  `CollapseValidator.validate_collapsible`.
- From `explaneat/analysis/coverage.py`: `CoverageComputer` and
  `compute_compositional_coverage_from_hierarchy`.
- From the conv explain code:
  - `structure.attribution_flow` and `group_structures`, now the suggester;
  - `split_mixed_roles`, used only to measure the previous pipeline;
  - `parts._quotient_acyclic`;
  - `structured.analyse_exit`, `name_and_text` and `library_mech`, i.e. the mechanism tests
    (within-digit, ink control, image edits).

**Adapted:**

- **The engine.** `ConvModelStateEngine` mirrors `ModelStateEngine` (replay, validate-then-apply,
  undo by truncation). It is a mirror rather than a subclass because conv states carry levels and
  kernels.
- **Identity kernels.** The core identity handler leaves the `identity → target` connection with
  no innovation and weight 1.0. The conv wrapper assigns it a 1×1 unit-delta kernel and the
  target's level.

**New, because the scalar code cannot do it:**

- the conv forward pass;
- the source-side identity (port);
- the expanded-subgraph, entry and acyclic-collapse checks;
- per-role split partitions;
- the strategy's `ports` mode;
- the sub-part suggester;
- the hierarchy driver, the hypotheses for compositions and the root, and the plots.

## Tests

`tests/test_conv/test_decomposition.py`: 9 tests, all passing. The whole `tests/test_conv` suite
passes: **111 tests** (the previous 102 plus these 9).

- **The state matches the genome.** Logits agree with `NaiveConvNet` to < 1e-12.
- **Every operation preserves the function.** A split by partition, a target-side identity, a
  source-side identity (port) and a consolidation each leave the logits unchanged to < 1e-12.
- **The validator rejects each precondition with a clear message.** It rejects P1 (an
  intermediate fed from outside), P2 (a non-exit with an outside edge) and P3 (an exit fed from
  outside), naming the node and the edge. A clean selection passes after the strategy has run.
- **Re-entry is a blocking issue.** A non-convex selection is blocked by re-entry ("…re-enters
  it…"). `repair_candidates` merges the blocking node in, and the annotation then succeeds.
- **The strategy makes shared nodes collapsible.** On a synthetic graph with a shared map and a
  mixed-input map, the strategy adds ports for the image and for b, and splits the shared map.
  Both annotations pass `CollapseValidator`, and the function is unchanged.
- **UI mode works as specified.** It adds one target-side identity node for an exit with an
  external input, and reports a blocking issue for an intermediate with an external input.
- **Undo works.** Removing from seq 0 restores the phenotype exactly; removing the last op undoes
  just the annotation.
- **The full driver on a synthetic genome is valid and replays exactly.**
  - The run: hand-made candidates, two leaves, a composition and the root.
  - The hierarchy: every annotation is collapsible, there are 0 unclean crossings, and
    compositional coverage is 1.0.
  - The replay: the JSON record is reloaded from disk with every op validated again. It gives the
    same logits (difference exactly 0) and the same annotations.
- **Coverage matches hand computation.**
  - Leaves only: 2/5.
  - Two parentless leaves and no global root: compositional coverage 0.
  - After the root is added: structural coverage 1.0 and compositional coverage 1.0.

## Applied to the MNIST 10-class genome

- **Genome.** `monitor_best` of a read-only copy of `results/sweep_p50_m4_e5/checkpoint.pkl`
  (`checkpoint_copy.pkl` here). It is the best-by-validation genome: key 2553, generation 51,
  validation accuracy 0.9511, 34 hidden maps, 98 enabled edges.
- **Images.** The suggester and the hypotheses use the previous pipeline's explain subset of the
  validation split (2,000 images, stratified, seed 0). The function check uses another 1,000
  validation images.
- **Record.** `experiments/conv_neat/explanations/mnist10_sweep_p50_m4_e5_gen51.json`: the genome,
  69 operations, the summary, and per-node display data.
- **Summary.** `summary.json`.

The suggester proposes the **same 7 structures as the previous pipeline**, because it is the same
finder. None were blocked, so no repair was needed. Inside S2, S3 and S4 it proposes 3 sub-parts
each, giving 13 leaves.

### Before vs after (same 7 structures)

"Before" is the previous pipeline: `group_structures` plus `split_mixed_roles`, with entries and
exits classified by the UI rule. "After" is this decomposition. The preconditions are counted
with `CollapseValidator.validate_collapsible` in both cases.

| | Previous pipeline | This decomposition |
|---|---|---|
| Edges crossing a structure boundary | 88 | 65 (a port carries a source into a structure once) |
| … not into an entry, or not out of an exit | **28** | **0** |
| P1 violations (non-entry fed from outside) | 28 | 0 |
| P2 violations (non-exit sending outside) | 0 | 0 |
| P3 violations (exit fed from outside) | 34 | 0 |
| Collapsible structures | **0 of 7** | **7 of 7** (and 13 of 13 leaves, 0 unclean crossings among them) |
| Node splits | 13 copies (not recorded as operations) | 13 split operations (16 copies added), role-based |
| Identity nodes | 0 | 39 ports, each an exact copy of a source map |
| Recorded operation stream | none | 69 ops (39 identity, 13 split, 17 annotate) |
| Function preserved | logits checked once | max \|Δ logit\| = 2.1e-14 over 1,000 validation images, both live and replayed from the JSON record with every op re-validated |
| Blocking issues left | n/a | none |

**Coverage.**

- Compositional coverage is **1.0**: a single global root, with 16 of 16 compositions present.
- Structural coverage over all annotations is **1.0**.
- Over the leaves alone it is 0.61. That is expected under Def 10: leaf exits feed junction edges,
  and only the compositions and the root hold those edges.
- Every hidden map is in exactly one leaf.

The 39 ports are many. They are the price of a clean boundary on a graph where the image and most
maps feed several structures. For example, S3 has 9 entries: copies of the image, 26, 140, 148
and 178. Where a source feeds two sub-parts of one structure it gets two ports, one per sub-part.
That happens for 140_bc and 178 in S3.

### Structures and annotations (hypotheses stored in the `annotate` ops)

Each structure is listed with its maps, entries and exits; leaves are 1–4 maps.

| Ann. | Kind | Status | Name, and what it fires on |
|---|---|---|---|
| S1 | leaf (maps 11, 20, 282; 1 entry ← x; exits 11_a, 20) | correlate | "1/4/6/7/9 structure (mechanism unknown)": closest concept is a tall stem (s = +0.49). The add-stem edit goes as predicted, but the ink control fails. |
| S2 | composition of S2.1–S2.3 (11 maps; entries ← x, x, h20; 11 exits; splits 19, 62, 134, 140, 296) | proven | "No-stem detector": S2.2 feeds S2.3. **Together they separate 5s from 8s (94% balanced accuracy vs 77% for the best part alone).** |
| S2.1 | leaf | correlate | "0/2/3/5/6/8 structure (mechanism unknown)" |
| S2.2 | leaf | proven | No-stem detector. Fires on 3, 2, 5, 0, 8, 6 (57–92%), and on 4s and 7s without a stem. Add-stem edit −1.47 SD. |
| S2.3 | leaf | proven | No-stem detector. Add-stem edit −1.65 SD. |
| S3 | composition of S3.1–S3.3 (7 maps; 9 entries; 5 exits; splits 15, 18, 212) | supported | "Closed-left-side detector". **Together they separate 2s from 3s (96% vs 81%).** |
| S3.1 | leaf | correlate | "0/1/2/4/7 structure" |
| S3.2, S3.3 | leaves | supported | Closed-left-side detectors. Within-digit and ink tests hold; no image edit exists for this concept. |
| S4 | composition of S4.1–S4.3 (8 maps; 6 entries; 6 exits; splits 14, 25, 160) | proven | "No-stem detector". **Together they separate 7s from 8s (96% vs 83%).** |
| S4.1, S4.2 | leaves | proven | No-stem detectors. Add-stem edits −0.50 and −1.23 SD. |
| S4.3 | leaf | correlate | "0/2/5/6 structure" |
| S5 | leaf (maps 213, 262; 3 entries; split 213) | correlate | "4/6 structure" |
| S6 | leaf (map 12; 4 entries) | **proven** | **Lower-loop detector.** Fires on 8, 0, 9, 3, 6, 2. Within-digit s = +0.57; ink-controlled r = +0.50; the fill edit gives −1.76 SD as predicted. |
| S7 | leaf (maps 34, 471; 5 entries) | proven | No-stem detector |
| root | root | — | Reads the structures through the score maps and the head (exact linear terms; reconstruction error recorded in the evidence). Largest push to each image's own digit: '0' S4, '1' S3, '2' S4, '3' S4, '4' S3, '5' S3, '6' S3, '7' the image directly, '8' S6, '9' S6. |

Each hypothesis follows the format Mike asked for. It gives the name, then one sentence on what
the structure computes, then "So it fires on … (rates), and on … (exceptions)". It closes with
the evidence: the within-digit s, the ink-controlled r and the edit effect in SD. Where nothing
holds within digits, the name and sentence read "correlates with X; mechanism unknown" instead.

### Figures

- `collapsed.png`: the 7 structures as function nodes, plus the image and the score maps. Edge
  labels are the number of kernel edges between them.
- `expanded_S2.png`: S2 opened.
  - Diamonds are entries (ports, labelled with the map they copy). Red-outlined nodes are exits.
    The coloured boxes are the three sub-parts.
  - Green edges go only into entries, orange edges leave only from exits, and blue edges are the
    two junction edges between sub-parts.
  - There are 0 unclean (red) edges.

## What the report and the explaining-agent step need next

Not built here, by request.

1. **Report rendering.** Read the record:
   - `operations` (annotate params carry `display_name`, `hypothesis` and `evidence`);
   - `display.top_level` and `display.nodes` (level, size, bias, activation, base id, and port
     provenance `{identity: "source", of: <map>}`).

   Draw the collapsed view by default and expand on click, as in `NetworkViewer.tsx`. Hide ports
   by default, or draw them as small entry ticks on the box edge; merge split copies visually by
   `base`. `load_record` gives the live engine for maps and evidence.
2. **Hypotheses.**
   - **Compositions.** They currently take their name and mechanism test from their main exit,
     which is a child's exit, so they repeat a child's title. Their distinctive content is the
     combination sentence. The explaining agent should write each composition's own name from
     its combination: e.g. "5-vs-8 separator built from two no-stem detectors".
   - **Repeated titles.** "No-stem detector" appears six times. The concept library is dominated
     by the stem concept. Adding concepts (stroke direction, curvature, position-specific bars)
     or within-structure contrasts would give the agent more to name.
   - **Correlates.** The "correlate" leaves (S1, S2.1, S3.1, S4.3, S5) are where the agent should
     probe.
3. **Granularity.** Five structures are 1–3 maps; S2 has 11 maps and 11 exits, because almost
   every map feeds the score maps directly. If the report wants fewer exits per structure, a
   second pass could consolidate exits that only feed the score maps (an `output` role in the
   split partition). The operations already support that.
4. **The main UI.** Port the source-side identity (port) and the entry-split fix to
   `OperationsPanel.tsx` / `docs/annotation_coverage_model.md` if they prove useful beyond conv.

## Phase 2: the report renders the record; open-curve concepts; 3 vs 8 vs noise

### The report (`experiments/conv_neat/report/`)

- The record is now the source of truth for structure. `record_view.py` turns the decomposed state
  into an equivalent integer-id genome (`state_to_genome`; a port becomes a ReLU copy, which is exact
  because everything it copies is non-negative; checked to 1e-14). The existing per-map evidence code
  runs on that graph unchanged, with the record's leaves passed in as the groups, so the old
  clustering-based parts are gone (the finder only suggests).
- Graph (`template.html`): opens collapsed to the top-level structures. Clicking one opens one level;
  bands are nested. Layout is a nested block layout, so a band encloses only its own members.
  Entries are diamonds labelled with the map they copy, split copies are dashed, and exits are
  outlined in orange. A dot on each collapsed structure shows its status. Drag, pan and zoom are kept.
- Each annotation card leads with its stored hypothesis: name, what it computes, the behaviour with a
  per-class bar strip, then the evidence. The evidence covers the within-class table, a tested second
  concept, and the image edits as before/after pictures with their SD effect. A composition card
  shows its children and how they combine: the wiring and the class pair they separate together.
- The class nodes, scores, path highlighting, contrast, confusion picker and walk-through run on the
  decomposed graph. Highlighted paths and structure brightness aggregate over the record's
  annotations through `unitOf`.
- A record panel shows edges crossing boundaries before vs after, the operation counts and the exact
  per-structure direct push into each class score.
- Check: `check_record.py <report> <network index>` (Playwright). It covers collapse/expand, ports,
  splits, exits, hypothesis cards, edit images, class nodes, paths, contrast, confusions,
  walk-through, drag and phone width. All checks pass for both networks.

### Open-curve concepts and edits (`explaneat/conv/explain/mechanism.py`)

**Concepts.** These are "pockets": background pixels inside the ink box, not in a hole, with ink on
three sides. Each concept fires when such pixels cover at least 3% of the box. Rates per digit on
validation images:

| Concept | Fires mostly on |
|---|---|
| `open_upper_left` | 2 (90%), 3 (87%), 7 (76%) |
| `open_upper_right` | 5 (80%), 6 (88%) |
| `open_lower_left` | 3 (88%), 5 (78%), 7 (72%), 2 and 9 (50%) |
| `two_left_openings` | 3 (79%), 7 (61%), 2 (47%) |
| `open_top` | 4 (90%) |

**Edits.** Each is checked to remove or add its concept against its control:

- `close_upper` closes the mouth of the upper curve. The concept goes from 100% to 0%; control 92%.
- `open_upper` erases the ink left of an upper loop. `hole_upper` goes from 100% to 0%;
  `open_upper_left` goes from 6% to 96%.
- `bridge_left` draws a stroke down the left side. `open_left` goes from 42% to 0%; control 42%.

`edit_sign(concept, kind)` gives the direction each edit moves each concept.

**Shared names.** Exits that share a main concept are now told apart by their strongest other
concept that passes the same within-class and ink tests and is not contradicted by its edit.
Compositions are named by the class pair their parts separate together.

**Before vs after on the generation-51 genome** (same structures, same images):

| | v1 concepts | open-curve concepts |
|---|---|---|
| Proven annotations | 8 of 16 | 9 of 16 |
| Proven leaves | 6 of 13 | 7 of 13 |
| Supported | 3 | 4 |
| Titles containing "No-stem detector" | 7 | 5 |

- **Qualified by a tested second concept:** S2.3 and S7 (with a top bar), S4.2 (with an opening at
  the top).
- **Still plain "No-stem":** S2.2 and S4.1. No second concept passes the tests on them.
- **S3 now reads as open-curve detectors:**
  - S3.3 "No-left-open-upper-curve detector", proven: `close_upper` +0.77 SD and `open_upper`
    −0.67 SD, both as predicted.
  - S3.1 "No-right-open-upper-curve", supported.
  - S5 "Open-top (4-shape) detector", supported.
- **Compositions:** S2 is the 5-vs-8 separator, S3 the 2-vs-3 separator, S4 the 7-vs-8 separator.

### 3 vs 8 vs noise (`run_three8noise.py`, `check_three8noise_reading.py`)

- **Setup.**
  - The library adds the shape concepts (`one_piece`, `coherent`, `filled`) and three edits:
    - `shuffle_ink`: the pixels in the ink box are permuted; control = the digit mirrored.
    - `patch_box`: 3×3 tiles of the ink box are permuted, keeping ink position; control = the
      digit mirrored.
    - `bridge_loop`: two stroke ends are joined where that closes a loop; control = the same line
      pointing away.
  - Classes are named 3, 8 and noise; structures 3–4, sub-parts of at most 2 maps.
  - The record replays exactly: 3.6e-15.
- **Annotations.** S1 (h5, h3, h31, h14) is the 3-vs-noise separator, proven. Its sub-parts:
  - S1.1 = h5, "Connected-stroke detector · with one coherent shape", proven: `shuffle_ink`
    −1.24 SD.
  - S1.2 = h3 and h31.
  - S1.3 = h14.
  - S2 (h9, h120, h33) is the 8-vs-noise separator, proven.
  - S3 (h4) is a coherent-shape detector, proven: `patch_box` −0.33 SD and `shuffle_ink` −1.37 SD.
- **The preliminary reading, checked structurally:**
  - **Confirmed.** h3 (28×28) is low on coherent shapes (s = −0.77 within classes) and pushes the
    noise score up (+11.6). h31 (7×7) is loop-averse (s = −0.70) and also pushes noise up (+7.2).
    h5 (14×14), the coherent "digit strokes" map, pushes noise down (−9.5) and 3 and 8 up. These
    are exact direct terms on validation images. h31 reads h5, and h3 reads h31. The decomposition
    puts the two noise votes in one sub-part (S1.2), next to the shared strokes map (S1.1), inside
    one structure.
  - **Confirmed.** The closed-loop map h14 has `hole_any` s = +0.87 (+0.41 within classes) and
    pushes 8 (+4.6), not 3 (−1.0).
  - **Added.** h4 (S3, 14×14) is a third noise vote (+6.4). h9 → h120 act only through h33 and h4,
    with no direct term.
- The report shows this network second, with a data panel: examples of each noise type and the
  ink-matching table from the run's data report (training split).

### Headline networks (pruned and unpruned), as built into the report

The test numbers come from the runs' results files (read once by their finalize). Nothing here
reads test.

| | Pruned (`headline_genome.json`) | Unpruned (`sweep_p50_m4_e5/selected_genome.json`) |
|---|---|---|
| Test / fine-tuned / validation | 0.9511 / 0.9553 / 0.9473 | 0.9582 / 0.9597 / 0.9537 |
| Maps, kernels, parameters | 36, 88, 1,268 | 40, 112, 1,320 |
| Structures, leaves | 8, 12 | 7, 13 |
| Operations (ports, splits) | 65 (34, 14) | 77 (42, 18) |
| Unclean boundary edges, before → after | 19 of 78 → 0 of 54 | 33 of 94 → 0 of 68 |
| Collapsible structures, before → after | 0 → 8 of 8 | 0 → 7 of 7 |
| Function preserved (max \|Δ logit\|, 1,000 validation images) | 1.1e-14 | 1.4e-14 |
| Proven annotations (leaves) | 9 of 16 (7 of 12) | 10 of 16 (8 of 13) |
| Supported | 1 | 4 |

- **A new repair: candidates that feed each other.** On the pruned genome the finder's structures
  were each convex, but three of them fed each other (A → B → C → A through different maps).
  Collapsing them would make a cycle that no per-candidate check sees. The repair now finds cycles
  in the structure quotient. It moves one map across a crossing edge when that alone breaks the
  cycle and keeps both candidates convex, and otherwise merges the candidates. Test:
  `test_groups_that_feed_each_other_are_repaired`.
- **Report build:** `experiments/conv_neat/report/build_draft.sh`. The order is pruned, unpruned,
  3 vs 8 vs noise, then the earlier networks from the cache. Entries and split copies keep one
  picture per map, which brings the page to 13.1 MB, under the 16 MB limit.

### Third MNIST network: the 50k warm start (`mnist10_ws50k`)

`results/mnist10_ws50k_halving/selected_genome.json`, generation 97: a warm-start continuation of the
pruned population, trained on all 50,000 training rows with successive halving. Test 0.9761 (read
once by the run's finalize), validation 0.9765, 38 maps, 93 kernels, 1,283 parameters. Same driver
(`run_decomposition.py --genome-json ... --label mnist10_ws50k`), same 2,000 + 1,000 validation
images; nothing here reads test.

- **Decomposition.** 73 operations (41 ports, 15 splits). Unclean boundary edges 24 of 73 → 0 of 53;
  precondition violations (entry-only ingress, exit-only egress, pure exits) 24/0/24 → 0/0/0;
  collapsible structures 1 → 8 of 8. Function preserved and replayed from disk: max |Δ logit|
  1.4e-14. Hierarchy: root → 8 structures (2 compositions, 6 single leaves) → 14 leaves.
- **Annotations.** 9 of 16 proven (9 of 14 leaves), 3 supported, 1 contested, 2 with no mechanism
  found (correlates), 1 with no link (unknown).
  - S1 "0-vs-6 separator" (12 maps; mechanism unknown at the composition level): S1.1 closed-loop
    detector (proven: `fill` −3.6 SD), S1.2 no-stem detector without an open left side (proven),
    S1.3 correlates with 0/2/3/5/6/8 (mechanism unknown), S1.4 weak closed-left-side detector
    (contested: `bridge_left` went against the prediction).
  - S2, S6 no-stem detectors (proven: `add_stem` −1.3 and −1.2 SD); S7 weak no-stem (proven).
  - S4 no-left-open-upper-curve detector (proven) and S5 its mirror, a left-open upper-curve
    detector (proven: `close_upper` −1.0 SD, `open_upper` +0.6 SD).
  - S8 "3-vs-8 separator" (10 maps, supported): two no-stem detectors with a closed lower loop
    (S8.1 proven, S8.2 supported), a no-stem detector (S8.3, supported), a lower-loop detector
    (S8.4, proven).
  - S3 (one map, h320) has no tested concept: 7/8/9 structure, mechanism unknown.
- **Compared with the 10k networks.** The vocabulary is the same: no-stem detectors are again the
  most common part, and the no-left-open-upper-curve and lower-loop detectors recur. New here: a
  proven closed-loop detector (S1.1) and the positive left-open upper-curve detector (S5); the
  top-bar and open-top (4-shape) detectors of the 10k networks do not appear. The warm start keeps
  the pruned network's genes: 34 of the 38 maps are maps of the pruned headline genome, and most
  structures re-form around the same maps (S4 and S7 equal pruned S4 and S7; S2 and S6 are the
  cores of pruned S2 and S6; S8 gathers pruned S8 with the rest of pruned S6, two maps of S2 and one of S1; S1 merges
  pruned S1 and S3). Their weights were retrained on
  50k rows, so several names change (pruned S5 "top bar" → S5 "left-open upper curve"). Only 4 maps
  are shared with the unpruned network. Proven share: 9 of 16, as for the pruned (9 of 16) and
  unpruned (10 of 16) networks; more leaves are proven (9 of 14 vs 7 of 12 and 8 of 13).
- **Figures:** `collapsed_mnist10_ws50k.png`, `expanded_mnist10_ws50k_S1.png`. In the report it is
  the third network, "MNIST 10-class (ExConNEAT, 50k warm start)", after pruned and unpruned. With
  three MNIST networks the page reached 18.4 MB; every embedded picture except the walk-through
  inputs is now stored at its native resolution (the page draws them pixelated at a fixed size, so
  they look the same) and as a palette PNG when that is smaller (`shrink_maps` in
  `build_report.py`). The page is 15.25 MB; `check_record.py` passes 24 of 24 on each MNIST network.

### Behaviour-first hypotheses (pruned network, 18 Nov 2026)

The cards' hypotheses had been picked by testing a fixed concept list on each exit and keeping the
best fit, so most parts came out as "no-stem detector" or "mechanism unknown (closest: a tall stem)".
The per-map table also used a different list from the cards (it showed loops; the cards never did).
Now every annotation of `mnist10_pruned` is read from its own behaviour first, following the
explanation-authoring skill's order. Tools: `explaneat/conv/explain/investigate.py`,
`concept_library.py` (one library of 24 concepts, loops included); driver and how-to:
`experiments/conv_neat/explanations/behaviour_first.py`, `BEHAVIOUR_FIRST.md`; this network's
hypotheses are in `behaviour_first/mnist10_pruned_spec.py`.

The steps for each part:
1. Look at the part's behaviour: where it fires, strong vs weak responders within one digit (also
   at equal ink), the mean gradient, and gradient × input.
2. Explore with image edits on the 2,000 discovery validation images.
3. Write the mechanism and a prediction.
4. Self-probe the prediction against a matched control on 1,000 disjoint validation images.
5. Check all 24 concepts within digits, with the ink count held fixed.

The decomposition and the operation stream are unchanged; only the annotate operations' hypothesis,
display name and evidence were rewritten. The old automated reading is kept as
`evidence.superseded`. Function is preserved after the write (max |Δ logit| 1.1e-14), and the record
replays with every operation validated.

| | old reading (status) | behaviour-first reading (status) |
|---|---|---|
| S1.1 | 0/2/3/5/6/8 structure, mechanism unknown (correlate) | Ink-amount meter (proven) |
| S1.2 | 0/8 structure, mechanism unknown (correlate) | Ink-amount meter, second copy (proven) |
| S2.1 | No-stem detector (proven) | Horizontal-stroke detector (proven) |
| S2.2 | No-stem detector (proven) | Lowest-stroke detector (contested) |
| S3.1 | 0/2/3/6/8 structure, mechanism unknown (correlate) | Central-band ink meter (proven) |
| S3.2 | 0/3/5 structure, mechanism unknown (correlate) | Ink-amount meter, coarse, from S3.1 (proven) |
| S4 | No-left-open-upper-curve detector (proven) | Left-of-centre stroke detector (proven) |
| S5 | Top-bar detector (proven) | Top-bar detector (proven) |
| S6.1 | No-stem detector with an upper curve open right (proven) | Falling-diagonal stroke detector (proven at 0.2 SD; supported under the shared 0.3 SD bar, see below) |
| S6.2 | No-stem detector (proven) | Horizontal-stroke detector, second (proven) |
| S7 | No-stem detector (proven) | Low-crossbar detector (proven) |
| S8 | Weak open-top (4-shape) detector (supported) | Upright-stroke detector (proven) |
| S1 | 4-vs-7 separator (correlate) | Shared ink measure (proven) |
| S2 | 3-vs-5 separator (proven) | Horizontal strokes, then the lowest one (proven) |
| S3 | 5-vs-9 separator (correlate) | Central ink, re-read coarsely (proven) |
| S6 | 0-vs-8 separator (proven) | Flat and falling strokes of wide digits (proven) |
| root | direct-push table only (no status) | Digit scores from the structures (proven) |

- **How many changed.** All 17 hypothesis texts changed, and 15 of 16 parts got a new name (S5 is
  still the top-bar detector, now with the edit evidence of its own). Statuses before: 9 proven,
  1 supported, 6 correlates. After: 16 proven and 1 contested (S2.2).
- **Where the readings differ.**
  - *"No-stem" was a by-product.* Five parts read as "no-stem" are horizontal-stroke detectors: S2.1
    and S6.2 respond to horizontal strokes (a horizontal 10-pixel stroke vs the same stroke upright:
    +0.59 and +0.78 SD). S2.2 keeps the lowest of those strokes. S7 weighs them by height (a stroke
    at row 20 vs row 5: +2.02 SD). S6.1 responds to falling diagonals ('\' vs '/': +1.13 SD).
    "Tall stem: less with it" still shows on their concept tables. It is what these mechanisms do
    to 1s, not what they compute.
  - *Correlates became ink meters.* S1.1, S1.2 and S3.2 track the amount of ink: within-digit r
    0.99, 0.98 and 0.74. Thickening every stroke raises them by +1.45, +1.54 and +1.11 SD against a
    one-pixel shift. For S1.1 a stroke adds the same wherever and whichever way it is drawn (+0.01 SD,
    predicted no change). S3.1 counts only ink in a central upright band: making the digit wider
    instead of taller lowers it by 0.45 SD.
  - *Location, not a curve.* S4 was the "no-left-open-upper-curve detector". Behaviour says it
    counts ink in a band just left of centre and counts ink further left against it (an upright
    stroke on the left edge vs the right edge: +0.72 SD; the digit moved 4 px left vs right:
    −0.85 SD). The old concept still matches on its card ("upper curve open left: less with it",
    19% vs 36% within digits).
  - *S8* was a weak "open-top (4-shape)" reading. It responds to tall upright and '/' strokes
    (30% taller vs 30% wider: +1.14 SD).
  - *Compositions* now say what combining adds. S1's two children are one ink measure handed to
    six structures: main exits r = 0.995, 10-class accuracy 44% together vs 36% alone. S2 adds
    location: a bottom vs top stroke moves S2.2 by +0.81 SD but S2.1 by only +0.10 SD.
  - *Loops* now appear on the cards as concepts, not names. S7 fires less on images with a loop or
    a lower loop, within the same digit and with ink held fixed. No part is a loop detector by its
    behaviour.
  - *Root.* The old root only listed direct pushes. Behaviour first shows that many digits are
    recognised by what stays silent. Holding one structure's term at its average costs 92% of 1s
    for S2, and 70% of 4s and 86% of 6s for S5, 66% of 3s for S4. The edits that wake that
    structure confirm it: a horizontal stroke across a 1 lowers the '1' score margin by 3.03 SD;
    a top bar on a 6 lowers the '6' margin by 5.50 SD. The dependence replicates on the probe images
    for 8 of 10 digits.
- **What failed.** S2.2's first prediction failed: erasing the bottom quarter vs as many pixels from
  the top changed it by only +0.05 SD. On a 4 or 9, erasing the bottom leaves the crossbar or loop as
  the new lowest stroke. I rewrote it as "the lowest horizontal stroke". Then a stroke drawn below
  a 0/2/3/5 lowered it as predicted (−0.88 SD), but the same stroke below a 1/4/7/9 did not raise
  it (−0.24 SD, predicted up). It stays **contested**; I did not refit a second time.
- **Caveats.**
  - Mean pooling is the response throughout. That is what flows to the scores through the global
    average, but a part's local detection is read from its maps.
  - The probes were designed after exploring on the discovery images. Every one replicated on the
    disjoint probe images.
  - S3.2 is "proven" as an ink meter, but ink explains about half of its within-digit variation.
    The rest has no reading.
- **Report.** Each annotation card now leads with the mechanism and its prediction, then shows:
  - the self-probes, with as-written, edited and control pictures;
  - "where it fires · what drives it" (mean map, mean gradient, gradient × input, and the
    strong/weak contrast within a digit, also at equal ink);
  - the behaviour bars as the conclusion;
  - the full 24-concept table (fires WITH vs WITHOUT, same digit, ink-controlled verdict).

  The per-map inspector uses the same library on all three MNIST networks. The superseded reading
  sits under "show the maths". Unpruned and 50k warm start keep their automated hypotheses until
  they are redone with `BEHAVIOUR_FIRST.md`.

### Behaviour-first across all four record networks (shared bar, 18 Nov 2026)

`mnist10_unpruned`, `mnist10_ws50k` and `three8noise_run_s0` were then re-read the same way
(specs in `experiments/conv_neat/explanations/behaviour_first/<label>_spec.py`). Those three runs
patched a stricter bar into their specs; that bar now lives in the shared code and every record was
re-written with it (`investigate.probe_verdict`, `root_status`, `children_correlation`; see
`BEHAVIOUR_FIRST.md`):

- a directional self-probe agrees only at **0.3 SD** or more (0.2 to 0.3 SD is "too small", so a part
  relying on it is supported, not proven); "no change" probes still need the whole 95 % interval
  within ±0.2 SD;
- the root's replication bar is a share of the classes (80 %: 8 of 10, 3 of 3), and a root with no
  image-edit probes is never proven;
- generated texts say "class" for a non-digit class set and count the scores from k;
- a composition's `r_children` is the mean pairwise correlation of its children (any number).

The pruned network's readings and probes were not changed, only re-judged. One status moved:
**S6.1** (falling-diagonal stroke detector) proven → **supported**, because its lean probe (sheared to
lean left vs right) is +0.28 SD, now below the bar. Every other status on all four networks is as its
own run reported. Function is preserved on every record after the write (max |Δ logit| 1.1e-14
pruned, 1.4e-14 unpruned, 1.4e-14 ws50k, 3.6e-15 three8noise).

| network | automated concept pick (before) | behaviour-first (after) |
|---|---|---|
| MNIST pruned (17) | 9 proven, 1 supported, 6 correlate, root unrated | 15 proven, 1 supported (S6.1), 1 contested (S2.2) |
| MNIST unpruned (17) | 10 proven, 4 supported, 2 correlate, root unrated | 16 proven, 1 supported (S5.2) |
| MNIST 50k warm start (17) | 9 proven, 3 supported, 1 contested, 2 correlate, 1 unknown, root unrated | 13 proven, 3 supported (S3, S6, S8.2), root contested |
| 3 vs 8 vs noise (9) | 5 proven, 2 supported, 1 correlate, root unrated | 9 proven |

No part is "described, not explained" on any network.

**Recurring mechanisms.** The same few readings come back on every network, though each was read
afresh:

- *Ink meters*: a part that tracks how much ink the image has (thickening every stroke moves it by
  more than 1 SD, and a stroke adds the same wherever it is drawn): pruned S1.1, S1.2, S3.2 (and S3.1
  for a central band), unpruned S3 (and S1 inverted), ws50k S1.3 (S1.2 tilted to rising strokes),
  three8noise S3 (upright strokes counting more). The first structure of each MNIST network is one.
- *Horizontal strokes*, often split by height: a plain horizontal-stroke detector (pruned S2.1, S6.2;
  unpruned S2.1; ws50k S2, S6), the top bar or upper strokes (pruned S5; unpruned S2.2, S2.3, S5.2;
  ws50k S5) and the lowest or base strokes (pruned S2.2, S7; unpruned S5.1, S7; ws50k S7, S8.2).
  The old "no-stem detector" label on most of these was what such a part does to 1s.
- *Diagonals and lean*: falling ('\') or rising ('/') strokes and the digit's slant (pruned S6.1, S8;
  unpruned S4.4; ws50k S8.1, S8.3, S8.4).
- *Left side of the digit*: upright ink on the left (pruned S4; unpruned S4.2, S4.3; ws50k S4;
  three8noise S1.3) or the digit's left boundary (ws50k S1.4), with right-side counterparts
  (unpruned S4.1, a right-edge upright stroke; ws50k S1.1, the right-hand edge of strokes).
- *Enclosed space* is read only once by behaviour (unpruned S6: filling the holes lowers it by
  1.8 SD); elsewhere loops show up as concepts on the cards, not as the mechanism.
- On 3 vs 8 vs noise the reading is about *connected strokes against scattered ink* (shuffling a
  digit's own ink moves S1.1, S1.2, S2.1 by 1.5 to 2.6 SD), which MNIST does not need.
- At the root, many digits are recognised by a structure that stays *silent* for them; image edits
  that wake it lower that digit's margin on every network.

**What stays unexplained.**
- pruned S2.2: the lowest-stroke reading fails for strokes drawn under a stem (contested); S6.1's
  lean claim is below the bar; S3.2's ink reading covers about half of its within-digit variation.
- ws50k S3 is a near-silent vestige whose within-digit measure is too weak; S6's top-bar discount and
  S8.2's "a stroke below a stem" probe are too small; the root is contested (a base stroke under a 1
  raises the '1' margin, against the prediction) and its seven-base edit is too small.
- unpruned S5.2: the claim that a high stroke counts more when horizontal than upright is too small
  (+0.29 SD).
- Every reading uses mean pooling; local detection inside the maps is not separately tested.

Report: rebuilt from all four records (15.97 MB; the cards' concept tables are now stored as compact
rows and each probe picture once, which kept the draft under 16 MB). `check_record.py` passes 24 of 24
checks with no failure on each of the four record networks. The 3 vs 8 vs noise cards now use the
record's concept library for short names, and a composition with more than two children shows the
mean pairwise correlation of its children.
