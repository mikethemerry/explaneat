# Transplant study C2: graft features whose explained signal the recipient lacks

Written 2026-11-05, before any C2 run. Requested by Mike, who said the important part is grafting
where, by explanation, the two models do **not** share a feature, i.e. the two features point to
different signals in the data.

## Why a new rule

Study C's "complementary" rule (amendments 2–3) compared input features: Jaccard < 0.5 between
the sets of entry features. Almost every churn network uses day usage and service calls, so a
donor could pass that rule while computing a signal the recipient already has. Both
complementary donors in C did: their gains were matched by size-matched shams. C2 replaces
input overlap with **signal overlap**.

## Definitions (train split only)

- **Candidate donor features:** churn100 leaf annotations that
  - have all-input entries,
  - pass the graft preconditions,
  - have a readout edge to the output,
  - and have a train restore gain ≥ 0.02 (the same restore-gain definition as studies A and C).
- **Donor signal** `s_d(x)`: the leaf's logit contribution (readout weight × exit activation)
  on the train inputs, computed with the donor genome.
- **Recipient signal basis** `B_r(x)`: the recipient's 71 inputs plus the activations of
  **every** hidden node of the recipient, on the same train inputs.
- **Novelty** `nov(d, r) = 1 − R²` of an OLS regression of `s_d` on `B_r` (with intercept). It is
  the share of the donor's signal that the recipient cannot reproduce linearly from anything it
  already computes.
- **Train incremental value** `inc(d, r)`: train AUC after a readout refit of `s_d` added to the
  recipient's logit (the same refit as study C: the donor readout scale and the output bias, on
  train), minus the recipient's train AUC.

## Selection (fixed before evaluation, written to `transplant2_selection.csv`)

For each of the **10 churn100 recipients**:
1. Restrict to donors from other seeds with `nov ≥ 0.5`.
2. Pick the one with the largest `inc`.
3. If none qualifies, record "none", plus the recipient's maximum novelty.

Each selected pair's annotation names and hypotheses are recorded next to the recipient's
nonlinear leaves. That is the qualitative check that the explanations name different signals.

## Conditions per recipient (masked fine-tune, E = 500; seeds as in study C)

1. **base**
2. **ft-only**
3. **graft + refit**, no fine-tune
4. **graft + refit + ft**
5. **sham + refit + ft:** same topology and entries as the graft, N(0,1) weights, seed =
   recipient seed

## Outcomes

- **Primary:** Δ test AUC of (4) against (2), and of (4) against (5), per recipient with paired
  bootstrap CIs. Across recipients: the median, and a sign test over the recipients with a
  selected donor.
- **Secondary:** the same deltas for (3); novelty against outcome.
- **Hypothesis:** with novel signals, graft + refit + ft beats both ft-only and the sham.
  Either outcome is reported.
- **Follow-up:** re-explain the best recipient, to check whether the grafted feature appears
  as its own annotation.

## Notes

- Study C's two recipients (s9, s4) are included again. Their earlier test numbers are known,
  but C2's selection rule uses train quantities only.
- Selection, grafts and refits are all built before any test evaluation, as in C.
