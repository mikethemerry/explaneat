---
name: explanation-authoring
description: Use when explaining one ExplaNEAT genome end to end (refactoring, leaf and composition annotations, hypotheses, evidence, final report) with the provenance-stamped query tools, so every stated number is a reference to a stored query with the right quantity and scope, every scope word is proved, formulas are "=" only when exact, each part's hypothesis survives a self-probe, and the report meets the accuracy gates.
---

# Explanation authoring (v4)

You explain one evolved network: split nodes, group the network into annotated parts (leaves,
compositions, one root), write a hypothesis and evidence for each, and report. The tools make
numbers traceable; they cannot choose **which** number answers a sentence, or the words around
it. Of the 16 errors found in the earlier explanations, most were right numbers for the wrong
population or scope, simplifications written as exact, and numbers typed from scratch work. This
skill is about those choices. Describe what the **model** does ("the model gives smokers a
higher predicted risk"), not the world. Never name the assistant in anything you write.

## 1. Ground rules

1. **Only your genome.** Never write to another genome (`apply_operation`, `update_narrative`,
   `add_evidence`, `amend_evidence`, `save_snapshot`).
2. **Function-preserving operations only:** `split_node`, `consolidate_node`,
   `add_identity_node`, `rename_node`, `rename_annotation`, `annotate`. Never `add_node`,
   `remove_node`, `disable_connection`, `enable_connection`, `prune_*`.
3. **Training split, full sample.** Leave `split` at "train" and do not sample (SHAP on large
   data is the one exception; quote the n its `split` block gives). Never look at test numbers.
4. **Time box 3 hours** from your first call (`date` at the start; check it). Stop when it ends.
5. **Honesty.** Partial coverage, "described, not explained" and `resisted` are valid results.
   Never force a hypothesis the evidence does not support.
6. **Read-only repository.** Do not edit code. If a tool is broken, work around it with other
   tools and note it in `tool_bugs`.
7. **No number from a scratch script or from memory.** Scratch work is for exploring only.
8. **Correct, don't append.** A wrong evidence record is fixed with `amend_evidence` (with a
   reason), not left in place with a later disclaimer.

## 2. Tools

```bash
export DATABASE_URL=<given in your task> && uv run python experiments/10seed/xn.py --list
export DATABASE_URL=... && uv run python experiments/10seed/xn.py query_subgroup '{"genome_id":"<G>","where":"sex == \"1\""}' 2>/dev/null
export DATABASE_URL=... && uv run python experiments/10seed/xn.py --calls <scratch>/calls.json 2>/dev/null   # [[tool, kwargs], ...]
export DATABASE_URL=... && uv run python experiments/10seed/rename_inputs.py <genome_id> <dataset_id>     # once, first
```
Write each command literally: the sandbox refuses shell variables, loops and `cd` in compound
commands, so batch calls with `--calls`. Arguments `params`, `payload`, `sets`, `sweeps`,
`parts`, `nodes`, `edges` are JSON **strings** inside the JSON kwargs. Every evidence and query
tool returns `provenance.id` (`q_...`). `begin_explanation` describes an interactive workflow
and single-brace placeholders: ignore that; you work unattended and write `{{q:...}}`.

| Question | Tool | Paths you will quote |
|---|---|---|
| n, outcome rate, mean risk of a subgroup; where each part acts | `query_subgroup(where=...)` | `subgroup.n`, `subgroup.outcome_rate`, `subgroup.mean_predicted_risk`, `subgroup.parts.<ann>...` |
| prediction for constructed cases; sweeps; node values | `query_probe(sets, sweeps, nodes)` | `rows[k].risk`, `rows[k].score`, `rows[k].parts...` |
| what a part / node / edge / the direct edges is worth | `query_ablate(parts / nodes / edges / direct)` | `data.auc_with`, `data.auc_without`, `data.mean_risk_change`, `typical.risk_without`, `switched_off[i].weight` |
| every cell of a small discrete space; a rule | `query_enumerate(rule, group_by, where)` | see `get_query` |
| restore gains of nonlinear units | `query_linearise` | see `get_query` |
| coverage, function preservation, closed-form error | `check_explanation` | `function_preservation_max_abs_err`, `closed_form_max_abs_err`, `leaves[i].closed_form_max_abs_err` |

`where` is a pandas query over real-unit columns: quote codes (`thal == '7'`), backticks for
names with spaces or hyphens, `node_<id>` for a node's value (`node_17 > 0`), `risk`, `outcome`.
**Trap:** `{{q:<id>.n}}` is the size of the *split*, not the subgroup; the subgroup's is
`subgroup.n`. Run `get_query` and read the path before you quote it.

## 3. Numbers are references

Write every number as `{{q:<id>.<path>|<fmt>}}`; `<fmt>` is a Python format (`.2f`, `.1%`),
`pct` (a risk as a rounded percentage) or `d`. The text is stored rendered, with its template.
The explanation is **strict**: a write with a bare typed number, an unresolved placeholder or an
exact formula that does not reproduce is refused. Every write returns `text_check`: **read it and
fix its findings now**, while the computation is fresh.

- **Query first, then write.** Never write a number you have not yet computed (E2: a count typed
  three minutes before it was counted).
- **Rounding is the renderer's job.** Do not round by hand, and do not pre-format.
- **Literals that are not computed numbers** (a code in a name, "the 1-16 scale") go in
  `allow_numbers`. Never use `allow_numbers` for a computed value.
- **No tool answers it?** Then don't state the number. Say it in words without a figure, or
  find the query that does answer it. "Practically none" without a number is worse than
  `{{q:..subgroup.n}} of {{q:..rest.n}}`.
- **Evidence:** `add_evidence(..., query_id="q_...")` attaches the stored result; never type a
  payload table by hand (E11: 108 for 107). Formulas come from `get_formula` (keep its LaTeX).
  Evidence types: `analytical_formula`, `empirical_viz`, `subgroup`, `probe`, `ablation`,
  `enumeration`, `shap_importance`.

## 4. Choose the quantity before the tool

Pick the row first; the row fixes the query and the scope words in the sentence.

| The sentence says | Quantity | Query | Scope words |
|---|---|---|---|
| how much an input moves the score along its own edge | the edge weight (per sd) | `query_ablate(edges=["-9->0"])` (after `add_identity_node` on the output, the edge ends at the identity node: `"-9->identity_0"`), or `direct=true` for all of them | "directly", "through its own edge" |
| what a part adds for the rows it acts on | the part's contribution, mean over **those rows** | `query_subgroup` with the acting condition (`node_<exit> > 0`) | "through this part", "for the N it acts on" |
| what changing an input does to a prediction | whole-model change, population stated | `query_probe` sweep, or `query_subgroup` | "in the whole model, on average over X" |
| what a part is worth | ablation | `query_ablate(parts=[...])` | "switching this part off changes ..." |
| a rule on a discrete space | agreement and counterexamples | `query_enumerate(rule=...)` | "for every combination", "except N cases" |
| a count, share or rate in a subgroup | n, share, rate | **one** `query_subgroup` | "N of M training patients with ..." |

- **One clause, one query.** An n and a mean in the same clause come from the same query. E1
  added two cells for n and copied one cell's mean.
- **Give both when they differ.** If an input feeds a part as well as its own edge, say both:
  "a direct term of {{q:a.switched_off[0].weight|.2f}} per sd; in the whole model
  {{q:b.subgroup...|.2f}} on average for holders, because the gate adds the rest" (E8, E9 were
  whole-model averages labelled "linear term").
- **Units are part of the quantity.** Inputs are z-scored, so a one-hot "off" level is not 0
  (E6). Ask `query_probe` or `query_enumerate` in real units; never evaluate a z-scored formula
  by hand.

## 5. Scope and number words

"Only", "all", "every", "none", "no", "zero", "never", "always", "exactly", "solely" and "the
same" are claims about **every** row of a population. They need proof in the same sentence:

| Word | Allowed only with |
|---|---|
| zero, none, never, no X | a `query_subgroup` / `query_enumerate` count of 0 over **every cell** of the stated population (E5 skipped one of four cells: "zero in area 415" was wrong for the 30 rows of one state) |
| only, solely | a `query_ablate` of every other path; for "only through these gates" also `query_ablate(direct=true)` or the edge list, showing there is no direct edge (an input said to act "only through" two gates also had a small direct edge) |
| all, every, always | a `query_subgroup` with share 1.0 |
| exactly, = | the write-time exact check (formulas, section 6) |
| anything else | rewrite with the number: "for {{q:x.subgroup.n}} of the {{q:y.subgroup.n}} customers in the area" |

The lint warns on these words in a sentence with no reference. **Do not reword to escape it** ("virtually
all", "essentially only"): prove the word, or state the count.

## 6. Formulas

- **"="** only when exact: the write-time check reproduces it on every training row at the
  precision written. `get_formula` rounds coefficients to three significant figures, so in
  practice a typed formula is "≈"; "=" is for exact identities (a copy, a sum of parts). Put a
  typed formula **before any placeholder** in the text (the checker reads formula spans from the
  start). `get_formula` needs `force: true` for large leaves and the root.
- **Long linear sums** (a leaf of many direct edges) are refused as bare numbers: attach the
  formula as `analytical_formula` evidence (its LaTeX is exempt) and state the terms that
  matter with `query_ablate` edge weights.
- **"≈"** carries its tolerance and the rows it is off on, from a query:
  "node ≈ relu(a·age − b) (off by up to {{q:c.leaves[2].closed_form_max_abs_err|.2g}}; exact except on the
  {{q:d.subgroup.n}} rows of one country, where a dropped term is non-zero)".
- **Keep a dropped term when it matters**: when it is non-zero on a group the hypothesis talks
  about, or its error is over 5% of the node's range. E3 dropped a country's term from a
  hypothesis that talked about that country.
- **Raw units must say so** ("in mm Hg, not z").

## 7. When a part is explained

A part is **explained** when all four hold:
1. **Prediction.** The hypothesis states at least one testable behaviour: a threshold, a
   direction, a partner, a subgroup or a rule.
2. **Self-probe.** You construct that case (a `query_probe` sweep across the threshold, the
   partner at two levels, a `query_enumerate` rule) and the model does it. Attach it with
   `add_evidence(query_id=...)`. If the model does not do it, rewrite the hypothesis or move the
   part to `resisted`.
3. **Not a restatement.** It says what the part does to the prediction, and for whom, in domain
   terms. "A weighted sum of X and Y, kept where positive" is the formula in words: that part is
   "described, not explained". Say so; it is an honest result.
4. **Small regimes named.** Any regime on fewer than 30 rows (or under 1% of training rows) is
   stated with its n; if it looks fitted, say "fitted to {{q:..subgroup.n}} rows".

Self-probe every leaf, the root, and the three largest parts by ablation AUC loss.

## 8. Workflow

1. **Analyse:** `rename_inputs.py`, `begin_explanation`, `get_model_state`,
   `get_input_distribution`, `get_node_info` as needed. Map inputs to features.
2. **Candidates:** hidden-node gates, serial chains, the direct input→output edges. Plan the
   whole hierarchy first: a node sits in only one annotation's subgraph.
   **2b. Plan the queries** for each candidate: which rows of section 4 its hypothesis needs.
3. **Refactor:** `classify_nodes` / `detect_splits`; `split_node` an entry or intermediate node
   with external outputs; `add_identity_node` an exit with external inputs; `validate_operation`
   first. Finish all structural operations before computing quoted numbers: a query computed
   before a later split is flagged as stale.
4. **Leaves:** `annotate` (name, entry/exit/subgraph nodes and connections, hypothesis), then
   `get_formula`, the queries, `add_evidence(query_id=...)`, and `update_narrative` (it
   **overwrites** the hypothesis: write the whole text). **4b.** The self-probe (section 7).
5. **Compositions:** `annotate` with `child_annotation_ids`, saying how the children combine
   (added, gated, serial), up to one root containing the output. Unconnected inputs are
   zero-effect entry nodes of the root, not a separate annotation.
6. **Coverage:** `get_coverage` and `check_explanation`; iterate until structural and
   compositional coverage are 1.0, `hierarchy.complete` is true and function preservation is
   below 1e-9, or the time box ends. `undo_operation(seq)` removes that operation and every
   later one.
   **6b. Number audit:** keep a list of your writes (the template and its placeholders) as you
   go; at the end, `get_query` each id and check its population, quantity and scope (section 4)
   and that the path is the one you meant.
   **6c. Recheck (last):**
   ```bash
   export DATABASE_URL=... && uv run python experiments/10seed/xn.py --recheck <genome_id> \
       <scratch>/recheck_summary.json --record <scratch>/record.json 2>/dev/null
   ```
   It checks only this explanation (do not run `explaneat verify recheck <genome_id>`: it would
   compare with an earlier record of the genome).
   **6d.** Fix every `mismatch` (`update_narrative`, `amend_evidence`) and rerun. Answer every
   `needs_review`: fix it, or say in `recheck_notes` why it is not an error. Never mark the
   explanation complete with a mismatch open.

## 9. Final report

Your last message is this JSON (the v3 fields plus `accuracy`). `summary` and `resisted` follow
the same rules as hypotheses: give each as a template (`summary_template`, `resisted_template`)
with placeholders, and the rendered text (`summary`, `resisted`) from
`xn.py --render <genome_id> <file with the template>` (renders and lints; stores nothing).

```json
{"label": "...", "genome_id": "...", "start": "<ISO>", "end": "<ISO>",
 "structural_coverage": 0.0, "compositional_coverage": 0.0, "full_coverage": false,
 "within_budget": true, "n_leaf_annotations": 0, "n_composition_annotations": 0,
 "n_splits": 0, "n_identity_nodes": 0, "hidden_nodes": 0,
 "hidden_nodes_single_leaf_no_split": 0, "hidden_nodes_in_any_annotation": 0,
 "hierarchy_complete": false, "function_preservation_max_abs_err": 0.0,
 "closed_form_max_abs_err": 0.0,
 "summary": "...", "summary_template": "...", "resisted": "...", "resisted_template": "...",
 "recheck_summary": {"mismatch": 0, "needs_review": 0, "needs_review_items": [],
                     "unchecked_freetext_sentences": 0, "complete_ok": true},
 "recheck_notes": "...", "tool_bugs": "...",
 "accuracy": {"open_findings": 0, "numbers": 0, "provenance_backed": 0, "allow_listed": 0,
              "scope_mismatch": 0, "formulas": {"exact": 0, "approx_with_tolerance": 0, "other": 0},
              "audit": {"references": 0, "checked": 0},
              "self_probe": {"parts": ["..."], "refuted_and_fixed": ["..."]},
              "restatements": 0}}
```
Fill `accuracy` from the tools (`list_queries`, the last `text_check`s, `recheck`), not from
memory. `numbers` counts the numbers your text states; `provenance_backed` how many are
placeholders.

## 10. Self-check before the report

- [ ] Every number in hypotheses, narratives, payloads, summary and resisted is a placeholder,
      and the audit reran all of them.
- [ ] Every effect says its scope, matching its query; direct and whole-model both given where
      they differ.
- [ ] Each n and statistic in one clause come from one query.
- [ ] Every zero / only / all / never / exactly / = has its proof (section 5).
- [ ] Every formula is "=" (exact check passed) or "≈" with its tolerance and rows.
- [ ] Categorical levels were queried in real units, every cell.
- [ ] Every leaf, the root and the three largest parts have a stored self-probe.
- [ ] No hypothesis restates its closed form; regimes under 30 rows carry their n.
- [ ] Zero open `text_check` errors; `recheck` mismatches 0 and every `needs_review` answered.
