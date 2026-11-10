# Run 1: adjudication of findings on controls and paraphrases

*Exploratory, not pre-registered.*

`SCORE.md` reports the following "false positives" on unmodified items: 6 of 24 items refuted, and
17 of 24 weakened or refuted.

Every refuted finding on a control or paraphrase was checked by hand against the verifier's own
evidence, and against the other blind verifiers' findings on other variants of the same base. Each one
turned out to be a **genuine error in the original explanation**, and most were found by more than one
verifier. They are errors the answer key did not list as pre-existing.

| Base | Error in the original explanation | Found independently in |
|---|---|---|
| heart-s5 | Ischaemia gate: mean predicted risk of the 35 low-blood-pressure patients stated 0.13, recomputed 0.18 | vb-acd834, vb-26c88a, vb-e8ef4b, vb-e37014 |
| adult-s5 | "27 people born in Thailand"; there are 26 in train | vb-b9ec8f, vb-eccf26, vb-caa642 |
| churn100-s4 | area415_state47 "zero in area 415"; it is on for the 30 Washington customers in area 415 | vb-2459d3, vb-61bbd6 |
| heart-s2 | Age "jumps" for reversible-defect patients; the part is off at every age there | vb-890449, vb-2103cc |
| churn100-s1 | Voice-mail logit change stated −2.37; probe gives −1.96 | vb-274350 |
| monk2-s0 | "108/108"; there are 107. The record's own `resisted` field discloses this | vb-523773 |

On that basis, the verifier's **refuted** false-positive rate on genuinely correct content is about 0%.

Most "weakened" findings on controls are also genuine. They are mainly:
- 4.45% rounded to "about 5%";
- numbers cited from a claim the sentence does not cite;
- small scope overclaims.

**Follow-ups:**
- **Errata to the original explanations:** record each error against its original record. Mike decides whether to correct the original records or keep errata alongside.
- **Fix the tool gaps** the verifiers reported.
- **Blindness caveat:** batch 4 ran one sub-verifier per item. Two of them (vb-523773, vb-9fa8f0) started after earlier items' results were in their parent's context.
