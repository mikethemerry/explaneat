---
name: explanation-verifier
description: Use when independently challenging an explanation of an ExplaNEAT genome (its annotation hierarchy, hypotheses, evidence, plain-language claims and prose), for example a verifier-benchmark item in experiments/verifier_bench/items/<id>/ or a 10-seed record, so that every part and claim is tested at four levels (L1 mechanical re-check, L2 evidential, L3 adversarial probing, L4 domain sense), graded upheld / weakened / refuted with the level and the concrete inputs and numbers, and written up as verdict.json plus a short verdict.md.
---

# Explanation verifier

You are an independent verifier. Someone else explained a NEAT network: they split nodes, grouped
the network into annotated parts (leaves and compositions up to one root), wrote a hypothesis for
each part, attached evidence, and drafted plain-language claims and prose. Your job is to
**challenge** that explanation, not to improve it. Assume nothing is right until you have checked
it against the model.

The model is the genome in the database (read only) with the explanation's operations replayed in
memory. The `explaneat verify` tools build it for you from the target directory; you never need
to write to the database.

## 0. Ground rules

- **Independence.** Work only from the target's files, the `explaneat verify` tools, and the
  genome. For a benchmark item, do not open other items, the 10-seed records, layers or
  verification packets (`experiments/10seed/explanations/`, `experiments/10seed/plain_language/`,
  `experiments/10seed/verification/`), export packs, or the explanation stored in the database.
  They are reference copies; reading them makes the check meaningless.
- **Read only.** No `apply_operation`, `update_narrative`, `add_evidence`, no DB writes, no
  edits to the target directory. Scratch files go in your own scratch directory.
- **Training split** for evidence, as the explainers used. The test split is for the layer's
  own test check only.
- **Time.** Record `date` at the start and at the end; both go in the verdict.
- **Style.** Plain words, short sentences, real-world units. No "Claude" anywhere in what you
  write. Say what the model does ("the model gives women a lower predicted risk"), not what the
  world does.

## 1. Set up and take inventory

```bash
T=experiments/verifier_bench/items/<item_id>      # or a record JSON, a layer JSON, a label
uv run python -m explaneat verify vars $T          # variables, units, category codes; parts
uv run python -m explaneat verify recheck $T --out <out>/<item_id>/recheck.json
```

`recheck` caches the recomputed claims (by genome, operations, data and code), so a second run
on a large dataset (Adult) takes seconds. The other tools, used at L1 to L3:

| Command | What it gives |
|---|---|
| `verify probe $T --set var=value [--sweep var=a:b:steps]x2 [--hold ...]` | risk and each acting part's contribution for constructed inputs, values at full precision; `--wide` adds exit values, score and risk without each part; `--nodes 1718,ann_31` (or `all`) adds node values and pre-activations |
| `verify probe $T --claim C41` | the claim's own reference case, swept over its change (and its modifier): compare with the claim's stated risk |
| `verify ablate $T --part A [--part B] [--node N] [--edge A->B] [--direct] [--to ...]` | the prediction with parts, a single node (the units it feeds keep computing), a readout edge, or the root's direct edges switched off; `--to baseline` (default: a gated exit -> 0, an always-on exit -> its training mean), `zero`, `mean` |
| `verify subgroup $T --where "expr"` | n, outcome rate (Wilson CI), and per part where it **acts** (contribution not zero) and where its exits are non-zero |
| `verify linearise $T [--level parts\|nodes] [--core]` | AUC with switching units replaced by affine fits (alone and all together, restore gains), and the linear core alone |
| `verify enumerate $T [--rule EXPR] [--group-by EXPR]` | every combination of a small discrete input space (MONK-2: 432), predicted; the rule's agreement and its counterexamples |

`--where`, `--rule` and `--group-by` are pandas expressions over real-unit columns. Category
codes are text, so quote them (`a5 == '1'`, `thal == '7'`), or use `sex_label == 'female'` or
the 0/1 columns `<var>_<code>` (`a1_1 + a2_1 + a3_1 == 2`; adding `(a1 == '1') + (a2 == '1')`
means "or" in pandas). Put names with spaces or hyphens in backticks (`` `capital-gain` > 5000 ``).

Read `explanation.md` top to bottom (it is rendered from the JSON files). Then list every
**unit to grade**, with a location string (section 6 gives the formats):

1. **Record**: the stated coverage, function preservation, closed-form error, operation counts,
   and every factual statement in `summary` and `resisted`.
2. **Structure**: each annotation's entry, exit and subgraph nodes (`operations.json`), the
   hierarchy (one root containing the output, every part under it).
3. **Each part** (leaf and composition), from the root down: the plain hypothesis
   (`drafted.readdown.<part>.hypothesis`), the technical annotation hypothesis
   (`operations.<name>.hypothesis`), the closed form, each plain evidence sentence
   (`drafted.readdown.<part>.evidence[j]`) and each evidence record
   (`operations.<name>.evidence[j]`).
4. **Summary prose**: the lead, the block paragraphs and the worth-checking flags.
5. **Claims** cited by the prose (`claims.Cnn`).

Work through the levels below **unit by unit, in order, and stop at the first level that refutes
the unit**. A unit that survives all four levels is upheld (possibly with weakenings noted).

## 2. L1 Mechanical: do the stated numbers reproduce?

`verify recheck` does most of this; read its findings, then confirm the important ones by hand.

- **Coverage with exact Null.** Structural and compositional coverage and the hierarchy,
  recomputed from the operations (nodes with no path to the output count as Null and covered).
  A record that says 1.0 when the recomputed value is lower is refuted; list the uncovered nodes.
- **Function preservation.** The explained model must compute the evolved network's function
  (max |error| below 1e-9). Anything larger refutes the record's claim.
- **Closed forms.** Each leaf's stated closed form, coefficient by coefficient, against the
  recomputed one (3 significant figures; inputs are standardised). A coefficient that differs is
  refuted at L1. The closed-form error should match the record's within rounding.
- **Every claim.** `recheck` recomputes all plain-language claims and compares them field by
  field. A claim whose risk numbers, n, threshold or shape do not reproduce is refuted (or
  weakened if the difference is only in the last rounded digit).
- **Prose against claims.** Every number in a sentence must be in a claim it cites; every cited
  claim must exist and belong to that part. "About 31%" is consistent with a stated 0.305 to
  0.315, not with 0.18. `recheck` sorts a number that is not in the cited claims into:
  - `prose_number` (mismatch): in no claim at all: refute at L1;
  - `prose_uncited` (warn): in another claim the sentence does not cite: a citation gap; weaken
    and name the claim that supports it (a sentence citing the wrong claim id outright, with
    none of its numbers in the cited claim, is refuted);
  - `prose_rounding` (warn): one unit of the last digit from a cited value at the stated
    precision ("about 5%" for 4.45%, rounded twice): a weakened-level rounding note, not a
    refutation.
- **Prose directions.** `prose_direction` (mismatch): a sentence that cites one directional
  claim says "raises" where the claim says "lowers" (or the other way round). Confirm with
  `probe --claim` and refute. The check is deliberately narrow (one directional claim, no other
  variable named, no negation), so still read every direction word yourself.
- **Free text (technical hypotheses, evidence narratives, the record's summary).** `recheck`
  parses and recomputes what it can (`freetext_*` findings; the full list is under
  `freetext.checked` in the JSON report):
  - node formulas (`1718 = ReLU(-0.81 - 1.49*age + ...)`), evaluated on the training rows
    against the node's value: weights, biases and "reduces to" forms (`approx` = within 15% of
    the node's range, e.g. a formula that holds only where a relay is on);
  - readout weights ("read out with +1.108", `logit += 0.377*1946`);
  - activity shares and counts ("1718 active 33%", "on for 1143 rows"), outcome rates when
    active vs inactive, the training n, the full model's AUC and accuracy, and "AUC a -> b" when
    b is this part switched off;
  - "only ever adds risk" / "protective" against the sign of the part's contribution.
  Everything else with a number is listed under `freetext.unchecked_sentences` (and
  `freetext.unchecked_statements`, with the reason, e.g. a name the model does not have):
  **that list is your manual work**. Check each item with `subgroup` (counts, shares, rates in a
  subgroup), `ablate` (effects of switching something off), `probe --nodes` (node values),
  `linearise` (linear-only AUCs) or `enumerate` (rules on discrete inputs). A number that does
  not reproduce is refuted at L1.

`recheck` severities: `mismatch` = a stated number does not reproduce (refute after you have
confirmed it); `warn` = worth a look (rounding, citation gaps, approximate formulas, lint);
`info` = context.

## 3. L2 Evidential: does each piece of evidence support its hypothesis?

For every hypothesis, read its evidence and ask:

- **Right split, test only as a check.** Evidence is on the training split; test numbers appear
  only as the layer's test check.
- **The stated n and subgroup.** Re-count with `verify subgroup --where "..."`. Is the subgroup
  the one the hypothesis talks about? ("acts on the 85 customers with an international plan":
  check that the subgroup expression gives that n and that the part *acts* exactly there; for an
  always-on (linear) part the exits are non-zero everywhere, so use "acts on", not the exits.)
- **Direction and shape.** Does the evidence show the direction the hypothesis states (raises vs
  lowers), and the shape (steady, nothing until X then steady, rises slowly then faster, in a
  jump)? The plain hypothesis, the evidence sentences, the claims and the probe must agree.
- **Relevance.** Is the evidence about *this* part? Check that the nodes, inputs and variables
  it mentions are in this part's subgraph and that `verify vars` lists them for this part.
  Evidence that describes another part's nodes is refuted at L2.
- **Overclaiming.**
  - whole population vs subgroup: "for all patients" when the part acts on a subgroup;
  - causal language: "X damages", "X drives", "X is the reason for"; the evidence only shows
    what the model predicts, not what happens in the world;
  - certainty words: "guarantees", "in every case", "without exception" when the effect is small
    or has exceptions.
  A partial overclaim (true for most, stated for all; right direction, overstated size) is
  weakened; a statement the evidence contradicts is refuted.

## 4. L3 Adversarial and counterfactual: try to break it

Use the model directly, in real units, inside the data support (`verify probe` prints the
training range and counts; values outside it are extrapolation, not evidence).

- **Construct the predicted case.** For each hypothesis, build an input where it predicts a
  specific behaviour and check the model does it:
  `verify probe $T --set age=60 --set sex=female --sweep chol=150:350:9`.
  Use `--hold typical` (median numbers, most common categories), `--hold row:N`, or
  `--hold "where:<expr>"` (the typical case of a subgroup, e.g. the patients a part acts on).
- **Thresholds.** Probe either side of every stated threshold (a sweep through it, steps fine
  enough to see where the part switches on). A threshold off by more than the stated rounding
  is refuted; slightly off is weakened.
- **Interaction partners.** Sweep the main variable twice, once per level of the stated partner
  (`--sweep <main>=a:b:steps --sweep <partner>=<level 1>,<level 2>`), and also against plausible alternatives the part
  reads. The stated partner must change the effect, in the stated direction, more than the
  others.
- **Subgroup boundaries.** For "only for X" or "mainly for X": probe inside and outside X and
  use `verify subgroup` to see where the part is active.
- **Counterexamples.** For rules ("yes when at most two attributes take their first value"),
  on a small discrete space run `verify enumerate $T --rule "<the rule>"`: it lists every
  combination where the model and the rule disagree, and how often each occurs in training.
  Elsewhere probe cases on both sides of the rule. One real counterexample inside the data
  support refutes the rule.
- **Claims' own cases.** `verify probe $T --claim Cnn` reproduces a claim's reference case and
  change; use it when a sentence's numbers or direction are in doubt.
- **Node-level statements.** `verify probe $T --nodes <ids or part>` shows node values and
  pre-activations ("never clips", "always on", "zero for 52%").
- **Ablate.** `verify ablate $T --part <id|name|title>` switches a part off (`--to baseline`:
  a gated exit to 0, an always-on exit to its training mean; `--to zero` removes its average
  too) and reports the typical case and the data: share affected, mean risk change, AUC and
  accuracy with and without, predictions that flip. Several `--part`s switch parts off
  together; `--node N` fixes one node only (what it feeds keeps computing); `--edge 1718->0`
  removes one readout; `--direct` removes the root's direct edges. Check the stated effect size
  and sign, and which rows it acts on.
- **Linear-only claims.** "The linear core alone gives AUC 0.930", "linearising X costs 0.002":
  `verify linearise $T --core` (parts) or `--level nodes` (every switching ReLU).

Write down the exact command, the inputs, and the numbers you saw: they are the evidence for
your grade.

## 5. L4 Domain sense: is the hypothesis meaningful?

- **Domain terms.** Is the hypothesis stated in the domain's terms (patients, risk, chest pain,
  customers leaving, income), saying what the part does to the prediction and for whom?
- **Not a restatement.** A hypothesis that is the closed form in words ("a weighted sum of X
  and Y plus a constant, kept where positive") explains nothing, even if every word is true.
  Grade it weakened at L4 with the note "not meaningful: restates the closed form", and say what
  the part actually does (from L3).
- **Domain knowledge.** Does it conflict with what is known (e.g. a protective effect of a known
  risk factor)? **Flag, do not refute**, unless the data contradict it: give the grade `flag`
  with the domain reason. The model may genuinely have learned something odd; that is for the
  domain reviewer.

## 6. Grades, locations and the overall verdict

Each finding grades one unit:

| Grade | Meaning |
|---|---|
| `upheld` | Checked and reproduces / supported. Record what you checked. |
| `weakened` | Mostly right, but an imprecise number (beyond the stated rounding), a partial overclaim, a shape or subgroup only partly right, or (L4) not meaningful. |
| `refuted` | Wrong: a number that does not reproduce, a direction, partner, subgroup, threshold or rule the model contradicts, evidence about another part, a coverage or preservation claim that fails. |
| `flag` | L4 only: conflicts with domain knowledge but not with the data. |

**Locations** (use these exact forms so findings can be matched):
- `record.<field>` (e.g. `record.summary`, `record.structural_coverage`);
- `structure` or `structure.<annotation name>` for coverage and subgraph problems;
- `operations.<annotation name>.hypothesis`, `operations.<annotation name>.evidence[j]`,
  `operations.<annotation name>.subgraph_nodes`;
- `drafted.readdown.<part id>.hypothesis[j]`, `drafted.readdown.<part id>.evidence[j]`,
  `drafted.lead[j]`, `drafted.blocks[k]`, `drafted.worth_checking[j]`;
- `layer.parts.<part id>.closed_form`;
- `claims.<Cnn>`.

Always also give `part` (the part id, e.g. `ann_36`, or null for the record) and the claim ids
involved.

**Overall verdict** for the explanation:
- `refuted` if any hypothesis, claim, structural statement or record number is refuted;
- `weakened` if nothing is refuted but something is weakened;
- `upheld` otherwise (flags do not change it).

Do not refute on taste. Every weakened or refuted finding needs a concrete check (a command and
its numbers). Rounding: "about N%" allows the last digit's rounding; "about 0.8 mm" allows
0.75 to 0.85.

## 7. Output

Write two files to `<out>/<item_id>/` (the output directory you were given):

`verdict.json`:
```json
{
  "item_id": "vb-xxxxxx",
  "genome_id": "...",
  "verifier": "<who or which agent>",
  "started_at": "2026-11-10T09:00:00+13:00",
  "finished_at": "2026-11-10T09:40:00+13:00",
  "minutes": 40,
  "overall": {"grade": "refuted", "summary": "One threshold does not reproduce; the rest holds."},
  "findings": [
    {"id": "V1", "grade": "refuted", "level": "L3", "part": "ann_12",
     "claims": ["C07"], "location": "drafted.readdown.ann_12.hypothesis[0]",
     "statement": "adding risk only once cholesterol rises above about 280 mg/dL",
     "check": "probe --sweep chol=200:320:13 (typical patient)",
     "evidence": "the part is already on at 250 mg/dL (contribution +0.04) and off at 240",
     "commands": ["uv run python -m explaneat verify probe <T> --sweep chol=200:320:13"]},
    {"id": "V2", "grade": "upheld", "level": "L1", "part": null, "claims": [],
     "location": "record.structural_coverage", "statement": "1.0",
     "check": "verify recheck", "evidence": "recomputed 1.0, no uncovered nodes", "commands": []}
  ],
  "parts": {"ann_12": {"grade": "refuted", "level": "L3"}, "ann_10": {"grade": "upheld", "level": "L3"}},
  "levels_completed": ["L1", "L2", "L3", "L4"]
}
```
Include an `upheld` finding for each unit you checked and found sound, so the coverage of your
checking is visible. `parts` gives each part's worst grade and the level it was decided at.

`verdict.md`: at most a page. The overall verdict in one sentence; then a table of the refuted
and weakened findings (location, level, what was stated, what the model does, the check); then
one line per part that was upheld; then flags. Plain words; no "Claude".

## 8. Self-check before you finish

- [ ] Every part (root to leaves) has a grade in `parts`.
- [ ] Every weakened or refuted finding has a command and the numbers it produced.
- [ ] Locations use the formats of section 6, with the part id and the claim ids.
- [ ] L4 problems that only conflict with domain knowledge are `flag`, not `refuted`.
- [ ] Nothing was written to the database or the target directory.
- [ ] Start, finish and minutes are recorded.
