# Coverage of the paper explanations under both options for D4 (task 5)

Written 7 Nov 2026 on branch `thesis/numbers`. This is a report for decision D4; no reported number is changed.

- **Script:** `scripts/thesis/paper_coverage_options.py`
- **Raw output:** `paper_coverage_options.json`
- **Read-only:** every edit was made on an in-memory copy of the operations. No explanation in the database was changed.

**Definitions** are the current, corrected ones (`explaneat/analysis/coverage_report.py`, the code behind MCP `get_coverage`):
- **Structural coverage:** the share of candidate nodes covered, where candidate nodes are all non-output nodes.
- **Compositional coverage:** composed ÷ required. It is 1.0 iff there is exactly one root, the root contains the output, and every annotation descends from it.

**The two options:**
- **(a) Exempt.** An annotation is exempt from the single-root rule when every node in it is an input with no enabled outgoing connection. It still counts for structural coverage, but needs no parent.
- **(b) Edited under the root.** On a copy of the stream, the disconnected-input annotation's name is appended to the root composition's `child_annotation_ids`. Where it came after the root in the stream (MONK-2), it is moved to just before the root. The edited stream is replayed through `ModelStateEngine.add_operation(validate=True)`.
  - The edit **passes the current annotate validator** for both explanations.
  - The output is unchanged (max |Δ| = 0), as expected, since annotations are metadata.

## Table

| Paper explanation | Option | Structural | Compositional | composed / required | Roots | Single root holding the output (`hierarchy.complete`) |
|---|---|---|---|---|---|---|
| **Adult** `4bc8fa07` / expl. `7e3761fe` | Current (as stored) | 1.0 | **0.9** | 9 / 10 | `full_model`, `unused_inputs` | no |
| | (a) Exempt | 1.0 | **1.0** | 9 / 9 | `full_model` | yes |
| | (b) Edited under the root | 1.0 | **1.0** | 10 / 10 | `full_model` | yes |
| **MONK-2** `23dde0fa` / expl. `d6ff7ca0` | Current (as stored) | 1.0 | **0.75** | 3 / 4 | `exactly_two_of_six`, `pruned_inputs` | no |
| | (a) Exempt | 1.0 | **1.0** | 3 / 3 | `exactly_two_of_six` | yes |
| | (b) Edited under the root | 1.0 | **1.0** | 4 / 4 | `exactly_two_of_six` | yes |
| **Heart** `9022a9eb` / expl. `a03ebc86` | Current (as stored) | 1.0 | **1.0** | 5 / 5 | `heart_disease_risk_score` | yes |
| | (a) and (b) | 1.0 | 1.0 | 5 / 5 | (nothing to exempt or move) | yes |

**The expected 0.9 and 0.75 are confirmed.**

Candidate nodes and enabled edges, which are the same under every option:

| | Candidate nodes | Enabled edges | Uncovered nodes / edges |
|---|---|---|---|
| Adult | 130 | 92 | 0 / 0 |
| MONK-2 | 26 | 24 | 0 / 0 |
| Heart | 41 | 34 | 0 / 0 |

These match the paper's figures: 130 / 92, 26 / 24 and 41 / 34.

## Annotations involved

| Explanation | Annotation | Nodes | Where it sits in the stream |
|---|---|---|---|
| Adult | `unused_inputs` (op seq 18) | The 53 unconnected one-hot inputs: −3, −4, −7, −9, −11 to −16, −18 to −20, −22, −23, −26, −27, −29, −30, −34, −35, −37, −42, −44, −47, −48, −50, −53, −54, −57, −58, −60, −61, −66, −70 to −72, −74 to −76, −79, −86, −87, −90, −95, −96, −98, −99, −101, −102, −104, −106, −107 | Before the root `full_model` (seq 19). Option (b) only adds it to `full_model`'s children. |
| MONK-2 | `pruned_inputs` (op seq 11) | a2:2 (−5) and a4:3 (−11) | After the root `exactly_two_of_six` (seq 10). Option (b) also moves it before the root. |
| Heart | — | The 7 unconnected inputs (−3, −6, −12, −14, −17, −21, −23) are entries of the root `heart_disease_risk_score`, as in the 10-seed explanations. | — |

## Effect on the paper's stated counts

Neither option changes the paper's annotation counts:
- Adult: 11 annotations, 10 leaf and 1 composition;
- MONK-2: 5 annotations, 4 leaf and 1 composition.

Under (b), each disconnected-input annotation becomes a leaf child of the root. The root's own entry list is not changed.

A third route would change those counts: the one the 10-seed and Heart explanations take, listing the unconnected inputs as entries of the root and having no separate annotation. It would give Adult 10 annotations and MONK-2 4.

**For D4:** (a) is a change to the definition. It would be stated once and would apply to every explanation. (b) changes two stored explanations; the changed explanations are re-validated, and the paper's prose and counts stay as they are. Either option gives 1.0 / 1.0 for all three paper networks.
