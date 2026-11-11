---
name: plain-language-explanations
description: Use when drafting plain-language explanations of an explained ExplaNEAT genome (summaries, per-annotation paragraphs, "worth checking" flags, proposed hypothesis revisions) from its verified claims in experiments/10seed/plain_language/<label>.json, so every sentence is plain, real-world, describes the shape of each effect, is led by the change in predicted risk for a typical patient, worded with evidence-based confidence, traceable to its claims and marked as a draft for Mike's review.
---

# Plain-language explanations (layer v4)

You turn **verified claims** about an explained network into short, plain sentences that a
clinician, a domain expert or a thesis examiner can check against what is known about the
world. Mike's examples: "smoking increases your risk, and even more so if you are diabetic";
"women and children were more likely to survive"; "for a typical patient, going from 0 to 2
narrowed vessels raises the predicted risk from about 10% to about 75%".

The claims are computed and checked by code (`explaneat/analysis/plain_language.py`). Your job
is the writing. You never add a fact that is not in a verified claim.

## v5g: linked inputs and universal words (2026-11-11; applies to every version below)

The authoring-v4 evaluation found the remaining draft errors were of two kinds. Both are now
checked by code, and the lint refuses the sentences.

**1. The typical case and linked inputs.** Some inputs are tied together in the data: an
education category and the years-of-education number, "wife" and sex, day minutes and day
charge, voice-mail messages and the plan. The generator finds these links
(`layer.linked_inputs`). When a claim changes one of them for the typical case, it moves the
linked inputs to their typical value among the real rows with the new value. The claim then
records what it moved in `linked.moved` (per end: `from`, `to`, and for interactions
`linked.groups[].moved.modifier`).
- **Name what moved.** For example: "for a typical person with that education level (and the
  education-num such people have in the data), from about {{C..|pct}} to about {{C..|pct}}"; "a typical wife (so a woman) rather than a typical
  husband". Do not write "people whose highest education is ..." for a typical-case number:
  it is one constructed person, not the group. Say "for a typical person whose ...".
- **`off_data`.** When the moves are impossible, the claim carries `off_data.reasons`. This
  happens, for example, when the claim itself sets two linked inputs separately. The case is a
  combination no training row has.
  - Never state such a claim about real people.
  - Either leave it out, or say "in a constructed case for the typical ... (no training person
    has ...)".
  - The templates leave these claims out, and the lint refuses real-people wording and
    sentences without "constructed".

**2. The proof rule for "only", "all", "never".**
- **Which words.** "only", "only if", "all", "never", "nothing", "none of", "always", "no
  one", "everyone", "regardless" and superlatives ("more than anything else", "the biggest
  factor") are universal statements. One real counterexample makes the sentence false.
- **When one may stay.** Use such a word only when a cited claim checked it over the whole
  population. The check is the claim's `statement.support_check`, with `population`, `n`,
  `exceptions` and `check`, and it must show `exceptions: 0`. Then state the check in the
  sentence: "for all 242 training patients". Claims with a check:
  - thresholds: no patient's curve moves on the wrong side of the threshold;
  - `only_if` interactions: nobody in the other group is affected;
  - subgroups: exact counts;
  - rules.
- **Otherwise drop the word or give the exceptions.** For example: "adds to the risk once X rises
  above about T; for {{C..statement.support_check.exceptions}} of the {{C..statement.support_check.n}}
  patients it already acts below that".
- **Superlatives.** No claim checks one, because that would need every input over its full
  range. Never use them. Order the effects by size instead (biggest first) without saying
  "the biggest".
- **The lint.** `lint_universal` refuses a universal word without a cited claim whose check
  shows no exception. "On the training data only", "only slightly" and "nothing until" are
  exempt.

## v5: built up, read down (heart-paper only for now; adds to v4 below)

After the v4 summary the document walks the annotation hierarchy from the root down
(`readdown_order(layer)`: root, then each child, depth first; copies are described by their
parent). The v5 draft adds `readdown` and a title for every part including the root:
```json
"readdown": {"ann_39": {"hypothesis": [...], "evidence": [...]}, "ann_34": {...}}
```
- **Hypothesis** (1-3 sentences): what the part does for the prediction, in the v4 style (numbers,
  shape words, no qualifiers, no maths, ids or "ReLU"). Start from the proposed revision.
- **Evidence** (1-5 sentences): what was checked and what it showed, with numbers: who it acts on
  (subgroup claims: "it acts on 118 of 242 patients"; "the risk is about 2% with it and about 3%
  without it"), its effects, and "its formula reproduces the part exactly" (`structure` claim,
  kind `formula`). A composition's evidence starts from what it is made of: cite its
  `structure` claim (kind `composition`: `fed_by_parts`, `direct_inputs`, `combination` =
  `added` or `combined non-additively`, `feeds`) and name any copy through the copy's own
  `structure` claim (kind `copy`). Sections cite only their own and their descendants' claims
  (the root may cite any).
- **Additive paragraph:** protective items, risk-raising items and rare results, each a sentence
  with a `label` ("Lowering the risk", "Raising the risk", "Rare results"; the review starts a
  paragraph at each). An input that also feeds a part is quoted with its **direct** contribution
  only (claims with `scope: "direct"`, "through this sum alone"), plus "also acts through …
  (below)"; the lint refuses its whole-model claim there.
- The review puts the original hypothesis, the closed form and the full evidence table in
  "Technical detail" at the end; the plain text reads first.

## v4: structure follows the hierarchy (Heart pilot; overrides sections 4, 5 and 8 where they differ)

A v4 draft (layer schema 4.0.0) has this shape; the lint switches to the v4 rules when `blocks`
is present:
```json
{"lead": [{"text": "One line: the typical patient's risk and how the model is built.", "claims": ["C30"]}],
 "blocks": [{"block": "additive", "sentences": [...]}, {"block": "ann_34", "sentences": [...]}],
 "titles": {"additive": "Inputs that add up", "ann_34": "Low-risk profile", "ann_36": "..."},
 "annotations": {"ann_39": [...], "ann_34": [...], "ann_36": [...]},
 "worth_checking": [...], "hypothesis_revisions": {...}}
```
- **Blocks = the root's children.** `summary_blocks(layer)` (in `plain_language_store.py`) gives
  the order: first `additive` (the inputs that feed the model score directly, `parts[root].
  direct_variable_keys`, plus root children that are a weighted sum, `nonlinear: false`), then
  one block per non-additive child of the root. One short paragraph per block (at most 4
  sentences), citing only that block's claims (`block_claims`): the additive paragraph cites
  whole-model claims; a composition's paragraph cites its own and its descendants' claims and
  the whole-model claims about its inputs.
  - **Additive paragraph:** "These add up independently: …", each item with what it is and its
    risk numbers. Inputs that also enter a non-additive block: say so ("also act through the
    parts below"), because their whole-model numbers include those parts.
  - **Composition paragraph:** what it combines, how, and for whom ("acts on 165 of 242
    patients, nearly all without chest pain on exercise"), then its main effects.
- **Detail sections** (`annotations`): one per part, following the hierarchy (the review prints
  them under "Parts in detail" with a sub-heading per part and a diagram one level down). The
  root's paragraph is the additive block's detail (at most 8 sentences); other parts at most 6.
- **Titles:** a plain name for the additive block and for every part (no underscores, no ids);
  they label the diagrams and headings.
- **No confidence words.** Never "clearly", "slightly", "consistently" or "on the training data
  only" (the lint rejects them): the numbers carry it. Mike's style: "…it falls from about 29% to
  about 3% without chest pain on exercise, but stays at about 32% to 33% with it." Shape words
  stay (steadily, nothing until, in a jump, in an uneven way): they say how the change happens.
- **Small groups only:** a claim with `confidence.n_ok: false` gets "(the model has little data
  on this, n = 15)" in the same sentence. The test check, ranges and n otherwise stay in the
  evidence table.

## 0. Workflow

1. Extract (DB read-only) and read the layer:
   ```bash
   uv run python -m explaneat plain-language extract --labels <label>
   uv run python -m explaneat plain-language show <label>      # template text + every claim
   ```
   Read the JSON too: `experiments/10seed/plain_language/<label>.json` (`claims`, `parts`,
   `reference_patient`, `population`, `outcome`, `codebook.gaps`, `codebook.checks`,
   `template`). For each claim look at `risk`, `confidence`, `test_check`, `groups`, `small_n`.
2. Write a draft file outside the repo (e.g. your scratchpad):
   ```json
   {"summary": [{"text": "...", "claims": ["C02", "C03"]}],
    "annotations": {"ann_39": [{"text": "...", "claims": ["C10"]}]},
    "worth_checking": [{"text": "Worth checking: ...", "claims": ["C09"], "flag": "high fasting blood sugar lowers risk"}],
    "hypothesis_revisions": {"ann_36": {"proposed": "...", "claims": ["C44"]}}}
   ```
   One object per sentence. Keys under `annotations` / `hypothesis_revisions` are part ids.
3. Lint, then store (the store refuses a draft that breaks the rules):
   ```bash
   uv run python -m explaneat plain-language lint <label> --from draft.json
   uv run python -m explaneat plain-language draft <label> --from draft.json
   ```
   The stored draft gets `"drafted_by": "ai-assisted"`, `"verified_by": null`. Only Mike sets
   the verdict (`explaneat export record-plain-verdict <label> --verified`).

## 1. Evidence first

- **Only verified claims.** Every sentence cites claim ids (`C07`) with `verified: true`. A
  failing claim is never turned into a statement, not even hedged; it is listed as "not
  supported by the model".
- **Nothing without a claim.** No background knowledge, no mechanism stories, no numbers of
  your own. If the claims do not say it, do not write it.
- **Numbers only from claims, as slots (2026-11-11).** Do not type a number. Write a **slot**
  that names the claim field, `{{C086.risk.to|pct}}`, `{{C142.statement.threshold}}`,
  `{{C33.groups[1].risk_to|pct}}`; `set-draft` and `lint` fill it by code, **rounded once**,
  and add the claim to the sentence's citations (a number cannot then lack its claim: E4, E15).
  When the claims are re-extracted the slots are filled again from the new claims. The only
  numbers you may write yourself are the typical patient's values the layer gives you in words.
- **One number, one population.** Put each number next to the group it is for (the slot's
  claim says which: `groups[i].label`, `statement.conditions`). Never attach the typical
  patient's numbers to an exception group (E7: "for patients with a reversible defect it rises
  in a jump: from about 12% to 28%" used the typical patient's curve).
- **What "verified" means:** the claim holds for the model *consistently across patients*
  (at least 90% of training patients go the same way; thresholds have one fold; interactions
  are confirmed by the part's own counterfactual). It is not a significance statement.
- **The range across patients** (10th to 90th percentile of per-patient effects) shows how
  the effect varies between patients. It is not a confidence interval. A linear term has a
  zero-width range; that says nothing about how sure we can be.
- **The test check** (`test_check`) is the inferential part: does the mechanism help the model
  rank *test* patients (test AUC with minus without its information, paired bootstrap over test
  patients, seed 42)? Status `agrees`, `unclear` (interval includes no gain), `disagrees`
  (no gain or the opposite sign), or `not available`.

## 2. Voice

- Everyday words. One idea per sentence. Active voice.
- Talk about the population the dataset is about: `population.plural` / `.singular`
  ("patients", "people", "customers", "cases"). "You" is fine in a summary.
- **Name the real-world variable and unit** (`variables[].label`, `unit`, `statement.change`):
  "resting blood pressure from 110 to 155 mm Hg", "a flat ST segment at peak exercise",
  "people whose highest education is a bachelor's degree", "customers with an international
  plan".
- **Never** in the text: node ids, annotation ids (`ann_39`), column names (`cp:4`, `a1:1`),
  "z-score", "standardised", "ReLU", "sigmoid", "logit", "pre-activation", weights, formulas,
  **odds or odds ratios** (they stay in the evidence table). "This part of the model" names an
  annotation.

## 3. Magnitudes: predicted risk for a typical patient

- **Lead every effect with the change in predicted risk for a typical patient**, from the
  claim's `risk` block: "for a typical patient, going from 0 to 2 vessels raises the predicted
  risk of heart disease from about 29% to about 98%". Use the outcome's own words
  (`outcome.predicted`: "the predicted risk of heart disease", "the predicted chance of an
  income of $50,000 or less", "the predicted chance that a customer leaves", "the predicted
  probability of 'yes'").
- **The typical patient** is `risk.reference`: the median of every number and the most common
  category (`reference_patient`, not a real patient); for a part that does not act on that
  patient, "the typical patient this part acts on" (`risk.reference_values`). Say which, and
  describe the typical patient once per genome (e.g. "a 56-year-old man with no chest pain
  symptoms and median test results").
- **Interactions:** give the change in each group (`risk.groups`): "…from about 29% to about
  3% if exercise does not bring on chest pain, and from about 33% to about 32% if it does".
- **Subgroups:** risk without the part, then with it, for the typical patient it acts on.
- Round to whole percentages; "below 1%" and "above 99%" at the ends. **Round once, from the
  stored value** (0.0445 is "about 4%", not 4.5% then 5%), as `plain_language_templates.pct`
  does. With slots (`|pct`) the code does it; the lint rejects a percentage that looks rounded
  twice, and a "typical patient ... predicts ... about N%" that is not
  `reference_patient.predicted_risk` rounded once.
- **Survival models** (`risk_scale.kind == "survival"`): the same pattern with the predicted
  risk at the stated horizon ("predicted 5-year risk").

## 4. Confidence words (exact rules; the lint enforces them)

Each verified claim carries `confidence` = {`word`, `size`, `test`, `n_ok`, `qualifiers`},
computed from (a) |change in predicted risk for the typical patient| (`risk.delta`), (b) the
test check and (c) n:

| Word | Use when (and only when) |
|---|---|
| **"clearly"** | `confidence.word == "clearly"`: the risk change is meaningful (at least 5 percentage points), the test check `agrees`, and every compared group has at least 20 training patients. Every claim a "clearly" sentence cites must qualify. |
| **"slightly"** (or "a little") | `confidence.size == "slight"`: the risk change for the typical patient is below 5 points. Required for such a claim; not allowed without one. |
| **"on the training data only"** | the test check is `unclear`, `disagrees` or `not available`. Required for such a claim. |
| **"the model has little data on this"** | `small_n` (a compared group has fewer than 20 training patients). Required; say how many. |
| **"consistently across patients"** | optional, for a verified claim; never imply significance with it. |

- Size words: "a large change" only for at least 20 points (`size == "large"`).
- No other hedges or boosters ("strongly", "significantly", "proves", "definitely").
- The lint applies these to summary and paragraph sentences that state an effect; a count
  sentence ("this part acts on 8 of 242 patients") or a data fact ("7 of them had heart
  disease") needs no qualifier. Sentences citing claims of different classes must carry every
  required qualifier, so split them by class: "X and Y clearly raise it. Z also raises it
  (on the training data only)."

## 5. Shapes and order

- **Biggest risk changes first**, in the summary and in each paragraph.
- **Interactions:** "X raises the predicted risk, and even more so if Y" (`amplified`); "…,
  but only if Y" (`only_if`, and only when its `support_check` shows no exception; else "mainly
  if Y" with the exceptions); "… if Y, but the other way if not" (`reversed`). Y is
  `statement.stronger_group.cond`.
- **Thresholds:** "once X rises above about T" (`kind: above`), "while X stays below about T"
  (`below`); counts: "from 1 vessel upwards". Say for whom (`statement.conditions`). Add "only"
  only when `statement.support_check.exceptions` is 0 (v5g); otherwise give the exceptions.
- **Rules** (MONK-2, `type: "rule"`): phrase in rule terms on the attribute values, with no
  real-world meaning: "the model predicts 'yes' when exactly two attributes take their first
  value". Attributes are written `a1 = 1`.
- **Categories a part cannot tell apart** come as one class ("country of birth other than the
  United States"); keep that wording.

### 5a. The nature of each shift (shapes; the lint enforces this)

Every claim about a numeric or ordinal variable carries `shape`: the model's response curve on
its internal score, over the observed range (1st to 99th percentile, or every level of an
ordinal). Describe **the mechanism first, then the risk numbers as its consequence**:
"higher cholesterol raises the predicted risk steadily: for a typical patient, from about 14%
at 193 mg/dL to about 50% at 308 mg/dL". Use these plain words (never "linear", "non-linear",
"convex", "concave", "monotone" or "logit"):

| `shape.class` | Say |
|---|---|
| `steady` | "steadily"; "the same amount for each …" |
| `hinge` | "nothing until about X, then steadily" (`hinge_kind: flat_then_moving`); "steadily up to about X, then no further" (`moving_then_flat`) |
| `step` | "mostly in a jump around X" |
| `levelling_off` | "rises quickly at first, then levels off" |
| `accelerating` | "rises slowly at first, then faster" |
| `peak` / `dip` | "highest at about X, lower either side" / "lowest at about X, higher either side" |
| `multi_segment` | "in stages", then summarise the pieces (`shape.pieces`) if it matters |
| `flat` | "barely" |
| ordinal, `ordinal_pattern: equal_steps` | "each extra vessel adds about the same" |
| ordinal, `ordinal_pattern: first_step` | "mainly the first vessel" |

**Exceptions that hold only for some of a group** (`shape.exceptions[0].class == "mixed within
subgroup"`, with `n_alt` of `n_group`): say "for most patients …, but for **some** patients with
… (n_alt of the n_group) it …"; never "for patients with … it …", which states it for the whole
group (the lint refuses it; heart-s2 E7: 21 of 90). **Folds per group**
(`statement.threshold.by_group`): quote each group's fold ("about 160 beats per minute with a
normal resting ECG and about 176 otherwise"), not the pooled median. **Windows**
(`statement.only_within_window`): "stays at about 38%" holds only between
`statement.change.from` and `.to`; give the range or what happens beyond it (heart100-s7).
| `status: fallback` | direction only, "in an uneven way" |
| `status: mixed` | "for most patients …, but for `exceptions[].who` it …" |

- X is `shape.points_text` (real units). Use a shape only as its `status` allows: `verified`
  (at least 90% of affected patients have the same shape and the class is stable when the
  tolerances move by 25%), `mixed`, or `fallback`.
- **Saturation is appearance, not mechanism.** When `shape.risk_saturation` is set, add after the
  numbers: "…, flattening out near the top because the predicted risk cannot go much higher"
  (or "near the bottom … much lower"). Do not call a steady effect "levelling off" because the
  risk saturates.
- **One point per fold.** When `shape.point_reconciled` is true, the shape's hinge and the
  threshold claim are the same fold: quote only `shape.point.text` (the median fold across the
  part's patients), never the typical patient's own hinge as a second number. When
  `shape.point.wide` is true add the spread in plain words: "from about 1 mm (between 0.8 and
  1.9 mm for different patients)". When it is false, folds of different parts (or a whole-model
  curve) are separate points: name each with its own part. The lint rejects two points for one
  fold.
- **Weave the shape into the sentence.** One idea per sentence, the shape words inside it
  ("Higher cholesterol raises the predicted risk steadily: …"). Never append a list such as
  "; by shape: X steadily; Y in an uneven way" (the lint rejects it).
- **Paragraph length:** at most 6 sentences per part and 8 for the whole model. Merge closely
  related shape statements ("Older age and higher blood pressure both raise it steadily …"), or
  move minor effects (below "slightly") into one closing sentence: "Smaller effects, each
  changing the predicted risk only slightly: …".
- **Tie the shape to the structure** where `shape.matched_fold` or a threshold claim exists:
  "nothing until about 285 mg/dL, where one part of the model switches on, then steadily". Say
  the fold once: cite the direction claim and the threshold claim together, do not repeat it.
- Interactions: give the shape for the group named in `stronger_group` (`shape.groups` has
  each group's).
- Categorical variables have `shape.class: categorical` (levels relative to the reference,
  `ordering_high_to_low`); no shape words are needed.
- Shape-based "worth checking" flags only where real-world plausibility is in question, neutral:
  "Worth checking: the predicted risk is highest at about 55 years and lower for older patients"
  (flag e.g. "risk peaks at middle age then falls").

## 6. What to leave out

- **Null parts:** one line ("It ignores resting blood pressure."; "One small part barely
  changes the predictions.").
- **Duplicates** (`parts[].role == "duplicate"`): not described as structure.
- Refactoring operations, coverage, formulas, odds: in the certificate, not here.
- Long lists of tiny effects (e.g. dozens of states or countries): group them in one sentence
  with "slightly" and the right qualifiers, or leave them to the template and the evidence
  table; the paragraph names the largest few.

## 7. The model's view, not the world's

- Describe what the **model** does: "the model gives women a lower predicted risk". Never
  "X causes", "leads to".
- Do not judge real-world plausibility in the text. That is Mike's verification.
- When a claim looks surprising in real-world terms, put it in `worth_checking`, phrased
  neutrally ("Worth checking: the model treats fasting blood sugar above 120 mg/dL as lowering
  the risk."), with a short neutral **`flag`** for the cross-genome table. Reuse the same flag
  wording across genomes of one dataset (e.g. "high fasting blood sugar lowers risk", "fixed
  thallium defect lowers risk", "typical angina lowest risk", "ignores age"), so recurring flags
  line up. Also flag undecoded categories and very small groups.

## 8. Two versions, plus hypothesis revisions

- **Summary:** 3 to 6 sentences for the whole model: the 2 or 3 biggest effects, the one or
  two most important interactions, thresholds or rules, then what it ignores. Prefer claims
  with adequate data and decoded categories.
- **Per-annotation paragraph:** for each non-duplicate part, 2 to 6 sentences.
- **Hypothesis revisions:** each annotation's hypothesis rewritten in the same plain style, as a
  *proposed* revision citing its claims. Never overwrite the original (no `apply_operation`,
  `update_narrative` or DB writes).

## 9. Mike's thesis rules

- Never print the assistant's name ("Claude") in any output.
- Everything you write is a **draft for Mike's review**: `drafted_by: "ai-assisted"`,
  `verified_by: null`.

## 10. Self-check before storing

- [ ] Every sentence cites verified claims and says nothing beyond them.
- [ ] Every number is a slot `{{C..|pct}}` (or a value of the typical patient), rounded once by code.
- [ ] Every exception, fold and "stays at" says for whom and over which range (section 5a).
- [ ] Every "only / all / never / nothing" cites a claim whose `support_check` found no exception,
      and the sentence states the check; no superlatives (v5g).
- [ ] Every typical-case number names what was moved with it (`linked.moved`); no `off_data`
      claim is stated about real people (v5g).
- [ ] Every effect leads with the typical patient's predicted risk; no odds anywhere.
- [ ] Every numeric or ordinal effect carries its shape words (section 5a), or "in an uneven way",
      woven into the sentence; one point per fold; paragraphs within 6 (part) / 8 (whole model).
- [ ] "clearly" / "slightly" / "on the training data only" / "little data" follow section 4.
- [ ] No ids, column names, z-scores, activation names, formulas, causal words.
- [ ] Biggest changes first; interactions, thresholds and rules in the standard shapes.
- [ ] Surprises in `worth_checking`, neutral, each with a `flag`.
- [ ] `plain-language lint` passes.

The offline template renderer (`explaneat/analysis/plain_language_templates.py`) applies the
same rules mechanically, with no language model, for VARIANZ restricted mode, and passes the
same lint. Your prose should be at least as faithful and easier to read.
