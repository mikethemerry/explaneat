# Errata to the stored explanations, 2026-11-11

**Scope.** In verifier-benchmark run 1, blind verifiers found genuine errors in several original explanations
(`experiments/verifier_bench/runs/run1/ADJUDICATION.md`). Each of those errors was replicated by more than one
verifier. I then ran a sweep with `explaneat verify recheck` (v2, including the free-text checks) over every
stored explanation: the 60 10-seed records, the 3 AAAI paper genomes, the 13 species-lead records and the
4 complex-real records (80 in all). The sweep produced 53 mismatch findings and 35 rounding, uncited-citation,
lint or formula warnings (`sweep_triage.csv`). Every candidate was checked against the model with
`verify probe`, `subgroup`, `ablate` or `linearise`, or recomputed from node weights and values. A value was
corrected only once it was confirmed. Most sweep mismatches turned out to be the tool misreading the text,
for example:
- a conditional share ("46% of non-voicemail rows") compared with the overall share;
- an AUC read as a readout weight;
- a formula written in raw units.

They are marked "not an error" in the CSV, each with the check that settled it.

Before any change, the records, the plain-language layers and the database `Explanation.operations` of every
affected explanation were snapshotted to `snapshot/`, with sha256 values in `SHA256SUMS` and `MANIFEST.json`.
The git tag is `explanations-pre-errata-2026-11-11`.

## Errors

"Blind" = found by the blind verifiers (ADJUDICATION.md). "Sweep" = flagged by `recheck` v2. "Confirm" = found
while confirming a neighbouring error. "Layer" = `experiments/<study>/plain_language/<label>.json`; "DB" =
`Explanation.operations`; every DB hypothesis edit is mirrored in the layer's copy
(`parts[].original_hypothesis`). All rows were **corrected**, except E13, which was withdrawn in v2. See the v2 section below for E8, E9, E13 and E16 and for the second check of every row.

| # | Explanation | Location | Said | Correct | Confirmed by | Found by |
|---|---|---|---|---|---|---|
| E1 | heart-s5 | DB `ischaemia_pathway` hypothesis and `evidence[1].narrative` | 35 low-BP patients (290 on, 204 off): mean predicted risk 0.13 | 0.18 (0.1813) | node values on train: 135 with 290 on, 35 of them with 204_a/204_b off, mean risk 0.1813 | Blind |
| E2 | adult-s5 | DB `age_profile` hypothesis | Thai respondents "(27 rows)" | 26 rows | train rows with native-country = Thailand: 26 (all 26 have 2134 on) | Blind |
| E3 | adult-s5 | DB `age_profile` hypothesis | `2990=relu(1.446 age-0.48)` | `2990=relu(1.446 age+0.137*2134-0.48)` (the Thailand unit 2134 also feeds 2990) | incoming edges of 2990: age 1.446, 2134 0.1374, bias -0.481; `recheck` mismatch 4.45 on the 26 Thai rows, 0 after | Sweep (mismatch) |
| E4 | adult-s5 | layer `blocks[3][0]`, `readdown.ann_125.evidence[0]`, `summary[11]` (claims) | "about 50 hours a week" cited C143 only | cites C142 (ann_125's 50-hour threshold claim) as well; "(no further effect above that)" added, the shape phrase the lint requires for C142 | `recheck` prose_uncited; C142 threshold 49.9 h | Blind, Sweep (warn) |
| E5 | churn100-s4 | DB `area415_state47` hypothesis | 595 "zero in area 415"; "Zero in area 415 (2,016 rows)" | zero for 1,986 of the 2,016 area-415 rows; 1.03 (logit +0.21) for the 30 Washington (state 47) customers in area 415 | node 595 by area code × state 47: area 415 non-WA 0 (n = 1,986), area 415 WA 1.0334 (n = 30) | Blind |
| E6 | churn100-s4 | same hypothesis | offset "~0.15" / "0.16 for other areas (logit +0.03)" | 0.11 (0.1058), logit +0.02 | same table: areas 408 and 510 without WA, 595 = 0.1058; × 0.203 = 0.021 | Confirm |
| E7 | heart-s2 | layer `blocks[1][2]`, `readdown.ann_31.evidence[1]`, `summary[7]` | "but for patients with a reversible defect on the thallium scan it rises in a jump: for a typical patient from about 12% at 42 to about 28% at 66 years" | the 12% to 28% rise is for the typical patient (normal scan); "but for a typical patient with a reversible defect on the thallium scan this part is off at every age"; now also cites C33 | `probe --sweep age=35:75:9 --sweep thal=3,7`: ann_31 contribution 0 at every age with thal = 7; `subgroup "thal == '7'"`: the part acts on 3 of 90 | Blind |
| E8 | churn100-s1 | DB `churnModelRoot` hypothesis; record `summary` | voice-mail plan "-2.37" logit for holders, labelled as the linear (direct) term | **Mislabelled quantity, not a wrong number** (v2): -2.37 is the average whole-model effect for holders; the direct term is -1.96. Both are now stated with their labels | direct weight -0.8641 / sd 0.4415 = -1.957; whole-model mean (plan off → on) over the 1,061 holders -2.368 | Blind |
| E9 | churn100-s1 | same hypothesis; record `summary` | international plan "+2.67" logit for holders, labelled as the linear (direct) term | **Mislabelled quantity, not a wrong number** (v2): +2.67 is the average whole-model effect for holders; the direct term is +2.62. Both are now stated with their labels | 0.769 / sd 0.2932 = 2.623; whole-model mean over the 380 holders +2.666 | Confirm |
| E10 | churn100-s1 | layer `blocks[2][2]`, `readdown.ann_79.evidence[2]`, `readdown.ann_83.evidence[2]`, `summary[11]` | "about 5%" | about 4% (cited value 4.45%, rounded twice) | `recheck` prose_rounding | Blind, Sweep (warn) |
| E11 | monk2-s0 | DB root `upper_edge_count_classifier` `evidence[2].payload` (a1=3); record `resisted` | "108/108 pos" | 107/107 | `subgroup "a1 == '3'"`: n = 107, all predicted yes | Blind |
| E12 | churn100-s7 | layer `blocks[0][2]`, `readdown.ann_80.evidence[2]` (and summary copy) | "about 15%" | about 14% (14.48%) | lint; `recheck` prose_rounding | Lint, Sweep (warn) |
| E13 | heart100-s0 | layer `lead[0]` (and summary copy) | "about 20%" | **Withdrawn: not an error (lint false positive)** (v2). The typical patient's risk is 19.80% (`reference_patient.predicted_risk` 0.197958), so "about 20%" was right; the lint had matched a 19.3% category share (C55 `evidence.who.categories[3].share_out` 0.192982), not a risk. "About 20%" restored; the lint no longer reads `evidence.who` | `probe heart100-s0 --hold typical`: 19.8% | Lint, Sweep (warn) |
| E14 | heart-s5-c1 | layer `blocks[0][0]` (and summary copy), `readdown.ann_35.evidence[3]` | "about 28% at 118"; "about 27% at 66 years" | about 27% (27.50%); about 26% (26.45%) | `recheck` prose_rounding | Sweep (warn) |
| E15 | heart100-s7 | layer `blocks[2][1]`, `readdown.ann_38.evidence[2]` (and summary copy) (claims) | "about 168 beats per minute" cited C82 only | cites C83 (ann_38's 168 bpm threshold) as well, with its spread phrase "(between 160 and 176 beats per minute for different patients)" | `recheck` prose_uncited; C83 threshold 168.2 | Sweep (warn) |
| E16 | heart-s5 | DB `heart_disease_logit` (root) hypothesis (layer copy) | "oldpeak and thal:7 influence the output ONLY through these gates (oldpeak has no direct edge)" | "oldpeak influences the output ONLY through these gates (it has no direct edge); thal:7 acts mainly through them (raising the risk) but also has a small direct edge in the linear core (-0.07), which lowers it and is its only effect where the gates are off" (v2) | edge -26_a → 0 enabled, weight -0.0654; `ablate heart-s5 --edge '-26_a->0'`: typical risk 19.4% → 18.6% | Blind (vb-e37014 V2) |

**Listed, not corrected:**
- **AAAI genomes** (materials not to be changed):
  - test-split AUCs that `recheck` cannot check on train (heart-paper ×3, adult-paper);
  - activity shares written as 70% and 17.5%, recomputed as 68.2% and 19.0% (heart-paper);
  - closed-form display warnings (node ids in the written LaTeX);
  - a monk2-paper direction false positive.
- **Approximate formulas** (`freetext_formula` warn; each is stated as "nearly" or "simplified", or is off on a few rows only): heart-s2, heart-s2-c1, churn200-s0, heart100-s4, monk2-s4.
- **Closed-form display warnings:** heart-s0, heart-s6.
- **heart100-s6.** "Block contributes train AUC 0.905 -> 0.958". Removing both readouts of the block gives 0.9036. The method is not stated, so this is ambiguous.
- **c4-tuned-s0.** "29 squares" is correct (ann_44 reads 29 inputs, with one direction claim each). The number happens to match the uncited C146.
- **heart-s2, generated text.** The source of E7 is claim C30's computed shape, which lists "step" as an exception for reversible-defect patients. That exception holds only for the 29 of 90 such patients whose part is on at a young age. The auto-generated `template` text ("moves mostly in a jump") still carries it. Claims and templates are code output, so they were not hand-edited.
- **Stale rendered copies.** The verification packets in `experiments/10seed/verification/` (`02_heart-s5.md`, `16_monk2-s0.md`, `20_heart-s2.md`, `23_adult-s5.md`, `packets.html`, `live_checks.json`, `supplementary/C100-01_churn100-s4.md`, `C100-02_churn100-s1.md`) quote the old text. They were not regenerated. The verifier-benchmark items are frozen copies and were left as they are.

## Effect on reported numbers

There is none. Every error is in explanation text: hypotheses, evidence narratives or payloads, record
summary/resisted, and the plain-language drafts. The changes are:
- **Structure:** no structural operation changed. Before writing, the correction script checked that the subgraphs, entry and exit nodes and children of every operation were identical.
- **Recheck:** `check_explanation` and `get_coverage` were re-run on all ten corrected genomes (`check_after.json`). Structural and compositional coverage (1.0), hierarchy completeness, function preservation (max error ≤ 1e-15), closed-form error and operation counts all equal the stored `verified_by_orchestrator` values.
- **CSVs:** `explanations.csv` was rebuilt. Only monk2-s0's `resisted` text changed; the heart100 and churn CSVs are byte-identical.
- **Unchanged files:** `summary.md`, `ch7_table.tex`, the runs CSVs and the AAAI materials.
- **`mike_verification`:** untouched. Each corrected record carries an `errata` list (date, location, was, now, ref to this file).

**After the correction** (`after.json`; the edits themselves are in `apply_errata.py`), `recheck` reports no remaining genuine mismatch on the ten corrected explanations. Its
leftover findings are the false positives above (churn100-s7, heart-s5-c1, heart100-s7) and the heart-s2
approximate-formula warning. All corrected drafts pass `plain-language lint`. These outputs were rebuilt:
- the nine per-genome PDFs (`~/.explaneat/exports/plain-language/<label>.pdf`);
- the first-10 verification pack (heart-s5 is in it), whose checklist pages now print "Errata applied 2026-11-11";
- heart-s5-c1's certificate (`experiments/species_leads/verification/heart-s5-c1/heart-s5-c1.pdf`).

The species-leads combined `review.pdf` files were built with earlier export code. Rebuilding them would
change every lead's pages, so they were left alone. They still show heart-s5-c1's two old percentages until
`experiments/species_leads/make_pack.py` is re-run.

**Update (2026-11-11, commit `4d619be`):** the verification packets in `experiments/10seed/verification/`
and the species-leads verification pack, including its combined `review.pdf`, have since been regenerated.
No rendered copy now shows the old text.

## v2 (2026-11-11): follow-up to the root-cause review

The v2 changes follow `docs/reviews/explanation-errors-2026-11.md`. A snapshot of heart-s5, churn100-s1 and
heart100-s0 (record, layer and DB operations, with sha256) is in `experiments/errata/2026-11-11-v2/snapshot/`,
under the tag `explanations-pre-errata-v2-2026-11-11`. The edits were made by
`experiments/errata/2026-11-11-v2/apply_errata_v2.py`: text only, under the row lock, refused unless every
operation's structural fields are unchanged.

- **E13 withdrawn.** "About 20%" is restored in heart100-s0. The double-rounding lint no longer reads
  `evidence.who` (the category shares of a subgroup), with a test.
- **E8 and E9 relabelled.** The churn100-s1 hypothesis and record summary now give both quantities, with
  their labels. Direct term: −1.96 / +2.62. Average whole-model effect for holders: −2.37 / +2.67.
- **E16 added.** heart-s5 root: thal:7 also has a small direct edge.
- **Coverage and function unchanged.** Coverage, hierarchy, function preservation, closed-form error and
  operation counts are the same as before (`experiments/errata/2026-11-11-v2/check_after.json`). No reported
  number changed. All 80 drafts pass the lint.

**Second check.** For every row, an independent sub-agent checked the original text against the proposed text
with the verify tools. It saw only the two texts and the brief
(`experiments/errata/2026-11-11-v2/second_check/BRIEF.md`), not this file and not the reasoning behind the
corrections. Its outputs are in `second_check/*.json`. Results: 15 confirmed and 1 partly confirmed (E15).
None was rejected.

| # | Verdict | Command (abridged) | Result |
|---|---|---|---|
| E1 | confirm | forward pass on train (`vm.m.ev.forward`), 290 on and 204 off | 35 patients, mean predicted risk 0.181 (median 0.057) |
| E2 | confirm | `verify vars adult-s5`; node 2134 on train | 26 Thai training rows; 2134 > 0 on exactly those rows |
| E3 | confirm | `vm.structure` for node 2990; formula on train | relu(1.446·age + 0.137·2134 − 0.481) exact (error 9e-16); the age-only form is wrong on the 26 Thai rows |
| E4 | confirm | claims C142/C143; `probe adult-s5 --sweep hours-per-week=40..99 --nodes ann_125` | C142 threshold 49.93 h (same part); the part's contribution is 0 from 50 to 99 hours |
| E5 | confirm | node 595 by area 415 × state 47 on train | area 415: 1,986 rows at 0, 30 Washington rows at 1.033 (logit +0.21) |
| E6 | confirm | same | outside 415 and outside WA: 0.106 (logit +0.02); 0.15/0.16 fits neither reading |
| E7 | confirm | `probe heart-s2 --sweep age=35..66 --set thal=7` (and thal=3); all 90 reversible-defect patients | thal = 7: 41.9% at every age, ann_31 contributes 0; no jump for any of the 90 |
| E8 | confirm | logit with the plan on vs off for every training row | direct −0.8641/0.4415 = −1.957; whole-model mean over 1,061 holders −2.368 |
| E9 | confirm | same, international plan | direct +2.623; whole-model mean over 380 holders +2.666 |
| E10 | confirm | `probe churn100-s1 --claim C086`, `--claim C012` | 0.0445 at 249 minutes: rounded once, 4% |
| E11 | confirm | `P.frame(vm)`, a1 = 3 grouped by first-value count | 107 cases, all predicted positive |
| E12 | confirm | `probe churn100-s7 --claim C055` | 0.1448 at 4 calls: 14% |
| E13 (withdrawal) | confirm | `probe heart100-s0 --hold typical`; values in C25/C55/C84 near 0.193 | typical risk 19.8%: "about 20%"; 0.193 is a category share (C55) |
| E14 | confirm | `probe heart-s5-c1 --claim C22`, `--claim C23` | 0.27496 → 27%; 0.26451 → 26% |
| E15 | **partly** | claims C82/C83; `probe heart100-s7 --set restecg=0 --sweep thalach=150:180:31 --nodes ann_38`; per-patient fold | the C83 citation is right, but the sentence is still wrong (see below) |
| E16 | confirm | `vm.structure`; thal:7 on vs off for every training row | edge −26_a → 0 enabled, −0.0654; oldpeak has no direct edge; the direct edge works against the gates. "(raising the risk) … which lowers it" was added from this check |

**E15 is still wrong, and is listed, not corrected.** Part ann_38 switches on at about 160 beats per minute
with a normal resting ECG, and at about 176 with any other resting ECG.
- **Check:** `probe heart100-s7 --sweep thalach=150..190 --sweep restecg=0,2`. With restecg = 2 the risk is
  38.0% up to 176, 31.6% at 180 and 18.2% at 190.
- **What is wrong:** the sentence's "only with a normal resting ECG … the risk stays at about 38%" is false
  above 176 bpm, and 10 of 121 training patients are there. "About 168" is the median of the two folds and is
  no patient's fold.
- **Why the prose follows the claims:** the sentence is faithful to the verified claims. C82's window ends at
  175 bpm, and C83's median and spread describe two groups. Correcting the text needs the claims regenerated
  (a claim-generator issue, as for heart-s2's C30), so it is listed for the generator fix.
