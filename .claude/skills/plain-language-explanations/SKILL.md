---
name: plain-language-explanations
description: Use when drafting plain-language explanations of an explained ExplaNEAT genome (summaries, per-annotation paragraphs, proposed hypothesis revisions) from its verified claims in experiments/10seed/plain_language/<label>.json, so every sentence is plain, real-world, traceable to evidence and marked as a draft for Mike's review.
---

# Plain-language explanations

You turn **verified claims** about an explained network into short, plain sentences that a
clinician or a thesis examiner can read and check against what is known about the world.
Mike's example: "smoking increases your risk, and even more so if you are diabetic". Titanic:
"women and children were more likely to survive".

The claims are computed and checked by code (`explaneat/analysis/plain_language.py`). Your job
is the writing. You never add a fact that is not in a verified claim.

## 0. Workflow

1. Extract (DB read-only) and read the layer:
   ```bash
   uv run python -m explaneat plain-language extract --labels <label>
   uv run python -m explaneat plain-language show <label>      # template text + every claim
   ```
   Read the JSON too: `experiments/10seed/plain_language/<label>.json` (`claims`, `parts`,
   `reference_patient`, `codebook.gaps`, `template`).
2. Write a draft file (anywhere outside the repo, e.g. your scratchpad):
   ```json
   {"summary": [{"text": "...", "claims": ["C02", "C03"]}],
    "annotations": {"ann_39": [{"text": "...", "claims": ["C10"]}]},
    "worth_checking": [{"text": "...", "claims": ["C09"]}],
    "hypothesis_revisions": {"ann_36": {"proposed": "...", "claims": ["C44"]}}}
   ```
   One object per sentence. Keys under `annotations` / `hypothesis_revisions` are the part ids
   in `parts` (annotation ids).
3. Lint, then store (the store refuses a draft that breaks the rules below):
   ```bash
   uv run python -m explaneat plain-language lint <label> --from draft.json
   uv run python -m explaneat plain-language draft <label> --from draft.json
   uv run python -m explaneat plain-language review --labels <a>,<b> --out ~/.explaneat/exports/<name>
   ```
   The stored draft gets `"drafted_by": "ai-assisted"`, `"verified_by": null`. Only Mike sets
   the verdict: `uv run python -m explaneat export record-plain-verdict <label> --verified`.

## 1. Evidence first

- **Only verified claims.** Every sentence cites one or more claim ids (`C07`) whose
  `verified` is `true`. A claim with `verified: false` is never turned into a statement, not
  even hedged. The renderer lists those as "not supported by the model"; leave them there.
- **Nothing without a claim.** No background knowledge, no mechanism stories, no numbers you
  computed in your head. If the claims do not say it, do not write it.
- **Read the evidence, not just the statement:** `estimate` (value, `ci95`, odds ratio),
  `groups` (n per group), `small_n`, `test_check` (does the test split agree?), `evidence.who`
  (who a part acts on), `reasons`.
- What the intervals mean: bootstrap 95% intervals over training patients (seed 42). They show
  how the model's effect varies across the sampled patients, not uncertainty about the
  model's weights or the world. A linear term has a zero-width interval; that is correct.

## 2. Voice

- Everyday words. One idea per sentence. Active voice.
- Talk about people: "patients who ...", "women", "you" is fine in a summary.
- **Name the real-world variable and its unit** from the claim (`variables[].label`, `unit`,
  `statement.change.from/to.text`): "resting blood pressure from 110 to 155 mm Hg", "a flat ST
  segment at peak exercise", "fasting blood sugar above 120 mg/dL".
- **Never** in the text: node ids (`534`, `204_a`, `-12`), annotation ids (`ann_39`), column
  names (`cp:4`), "z-score", "standardised", "ReLU", "sigmoid", "logit", "pre-activation",
  weights or formulas. Use "this part of the model" for an annotation. (Annotation names may
  appear as headings in the review document; not in sentences.)
- Numbers: round. "about 2.3 times the odds", "about 285 mg/dL (7.4 mmol/L)", "8 of 242
  patients", "two in three". Odds ratios above 10 to whole numbers; huge ones ("×1400") are
  better said as "a very large increase in the odds" plus the number once.

## 3. Order and shapes

- **Biggest effects first** (largest |estimate|), in the summary and inside each paragraph.
- **Interactions:** "X raises the risk, and even more so if Y" (`pattern: amplified`);
  "X raises the risk, but only if Y" (`only_if`); "X raises the risk if Y but lowers it if
  not" (`reversed`). Y is `statement.stronger_group.cond`.
- **Thresholds:** "only once X rises above about T" (`kind: above`), "only while X stays below
  about T" (`below`). For counts, name the step: "from 1 vessel upwards". Say for whom when
  `statement.conditions` is set ("for patients with a normal resting ECG").
- **Subgroups:** "This part acts on 8 of 242 patients, all of them women whose chest pain is
  brought on by exercise". Use `evidence.who.conditions` for "nearly all of them are ...".
  The outcome count in the group ("7 of these 8 had heart disease") is a data fact from the
  claim and may be stated.
- **Magnitude:** always "on the model's odds scale", and state the reference patient once
  (`reference_patient`: the median training patient for every number, the most common
  category for the rest; it is not a real patient).

## 4. Confidence words come from the evidence

| Word | Only when |
|---|---|
| "clearly" | the claim's interval excludes 0 **and** `small_n` is false |
| "slightly" | the effect is small: `statement.strength` is `slight` or `negligible` (|log-odds| < 0.25, odds ratio between about 0.78 and 1.28) |
| "the model has little data on this" | `small_n` is true (a group has fewer than 20 training patients); say how many |
| "the test patients agree" / "worth checking on new data" | from `test_check.consistent` (test n is 61, so only as a check) |

No other hedges or boosters ("strongly", "significantly", "proves").

## 5. What to leave out

- **Null parts:** one line. "It ignores resting blood pressure." Near-inert parts: "One small
  part barely changes the predictions." Do not describe them further.
- **Duplicates** (`parts[].role == "duplicate"`): a copy of another part's calculation made
  during refactoring. Do not mention it as structure; its effect is in the part it feeds.
- Refactoring operations, coverage, formulas: they are in the certificate, not here.

## 6. The model's view, not the world's

- Describe what the **model** does: "the model treats higher cholesterol as raising the risk",
  "the model gives women a lower risk". Never "high cholesterol causes heart disease", never
  "leads to". These are associations the model uses, on its own odds scale.
- Do not assert real-world truth and do not judge plausibility in the text. That is Mike's
  verification.
- When a claim looks surprising in real-world terms, flag it **neutrally** in `worth_checking`:
  "Worth checking: the model treats fasting blood sugar above 120 mg/dL as lowering the risk."
  Also flag categories the codebook could not decode (`codebook.gaps`) and very small groups.

## 7. Two versions, plus hypothesis revisions

- **Summary:** 3 to 6 sentences for the whole model: the 2 or 3 biggest effects, the one or two
  most important interactions or thresholds, then what it ignores. Prefer claims with adequate
  data and decoded categories.
- **Per-annotation paragraph:** for each non-duplicate part, 2 to 6 sentences: who it acts on,
  its interaction or threshold, its directions. The root part's paragraph is the whole model.
- **Hypothesis revisions:** rewrite each annotation's hypothesis in the same plain style, as a
  *proposed* revision citing its claims. Never overwrite the original (`apply_operation`,
  `update_narrative` and DB writes are forbidden here); the original stays next to the
  proposal in the review document.

## 8. Mike's thesis rules

- Never print the assistant's name ("Claude") in any output: text, file names, PDF.
- Everything you write is a **draft for Mike's review**: `drafted_by: "ai-assisted"`,
  `verified_by: null`. The review PDF and the appendix mark it as an AI-assisted draft.

## 9. Self-check before storing

- [ ] Every sentence cites verified claims, and says nothing beyond them.
- [ ] No ids, column names, z-scores, activation names, formulas, causal words.
- [ ] Real-world names and units; odds numbers say "on the model's odds scale".
- [ ] "clearly" / "slightly" / "little data" match section 4.
- [ ] Biggest effects first; interactions and thresholds in the standard shapes.
- [ ] Surprising claims are in `worth_checking`, phrased neutrally.
- [ ] `plain-language lint` passes.

The offline template renderer (`explaneat/analysis/plain_language_templates.py`) applies the
same rules mechanically, with no language model, for VARIANZ restricted mode. Your prose should
be at least as faithful to the claims as the template, and easier to read.
