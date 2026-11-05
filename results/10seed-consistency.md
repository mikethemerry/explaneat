# 10-seed consistency experiment: results (thesis Ch 7)

Status as of 2026-11-05. Claude-explained results are complete for all 30 pre-registered
networks and the 10 additional Heart-100 networks. **None has been verified by Mike yet.**
Mike-verified counts are reported separately below and are currently 0.

- Pre-registration and its dated amendments: [`experiments/10seed/PREREG.md`](../experiments/10seed/PREREG.md)
- Per-run metrics: `experiments/10seed/runs.csv`, `runs_heart100.csv`, `runs_paper_reference.csv`
- Per-explanation records: `experiments/10seed/explanations.csv`, `explanations_heart100.csv`,
  and `experiments/10seed/explanations/<label>.json`. Each record holds the agent's report plus an
  independent `verified_by_orchestrator` block.
- Explanation protocol: [`experiments/10seed/EXPLAIN_PROTOCOL.md`](../experiments/10seed/EXPLAIN_PROTOCOL.md)
- Incidents (crashes, pause, throttling): `experiments/10seed/RUN_INCIDENTS.md`
- Code: branch `exp/10seed-consistency`.

## 1. Pre-registered outcome

> The consistency claim is supported for a dataset if ≥6 of its 10 best genomes reach full
> structural and compositional coverage within the time box.

Full coverage means: tool structural coverage = 1.0, tool compositional coverage = 1.0, the
hierarchy check passes (a single root containing the output, with every annotation beneath it),
and the function is preserved (max |Δ output| < 1e-9 on train). See the PREREG amendment of
2026-11-03.

| Dataset | Claude-explained, full coverage | Mike-verified, full coverage | Supported? |
|---|---|---|---|
| Heart (40 gen) | **10 / 10** | 0 / 10 (not yet verified) | Yes (Claude-explained) |
| MONK-2 | **10 / 10** | 0 / 10 (not yet verified) | Yes (Claude-explained) |
| Adult | **10 / 10** | 0 / 10 (not yet verified) | Yes (Claude-explained) |
| *Heart (100 gen, additional)* | *10 / 10* | — | *not part of the threshold* |

Every explanation finished well inside the 3-hour box. I re-ran every coverage and function
check myself, independently of the agent that wrote the explanation, and all 40 agreed with the
agents' own reports. The fallback wording is not needed.

## 2. Training results

Medians, with the range across seeds in brackets. Paper-network values are in the last block.

| Metric | Heart | Heart-100 | MONK-2 | Adult |
|---|---|---|---|---|
| Train AUC | 0.944 (0.939–0.947) | 0.956 (0.951–0.964) | 0.920 (0.766–1.000) | 0.909 (0.906–0.913) |
| Test AUC | 0.878 (0.865–0.912) | 0.892 (0.863–0.926) | 0.856 (0.639–1.000) | 0.906 (0.903–0.909) |
| Test accuracy | 0.803 (0.787–0.836) | 0.803 (0.787–0.836) | 0.822 (0.425–0.989) | 0.853 (0.851–0.857) |
| Hidden nodes | 5 (3–6) | 8.5 (5–13) | 3 (1–7) | 15 (11–17) |
| Enabled connections | 30 (25–36) | 40 (33–50) | 23 (19–35) | 87 (70–95) |
| Connected inputs | 20 / 26 | 20 / 26 | 13 / 17 | 60 / 107 |
| Depth | 3 (2–6) | 5 (3–7) | 3 (2–5) | 4 (3–7) |
| Skip connections | 4 (0–8) | 12 (6–24) | 4 (0–9) | 14 (9–18) |
| Direct input→output edges | 16 (14–19) | 15 (12–20) | 10 (6–14) | 54 (43–57) |
| Structural proxy: share of hidden nodes with ≤2 upstream features | 0.50 | 0.55 | 0.58 | 0.93 |
| Generation of best genome | 39 (31–39) | 99 (93–99) | 99 (55–99) | 99 (87–99) |

| Paper network (reference) | Train AUC | Test AUC | Hidden | Connections |
|---|---|---|---|---|
| Heart `9022a9eb` | 0.9455 | 0.8966 | 5 | 31 |
| MONK-2 `23dde0fa` | 1.000 | 1.000 | 2 | 23 |
| Adult `4bc8fa07` | 0.9085 | 0.9055 | 14 | 83 |

Context: the paper's L2 logistic-regression test AUCs are Adult 0.905, Heart 0.919 and
MONK-2 0.423. Test CIs are 95% paired bootstrap intervals (10,000 resamples, seed 42) and are
in `runs.csv`. Heart's test set has n = 61, so its CIs are about 0.18 wide.

**Reading:**
- **Heart:** the paper network is typical in size and shape, and on the lucky side on test.
  Its test AUC of 0.897 ranks 3rd among the 11 networks. No seed beats L2 logistic regression
  on Heart test AUC, although all differences are inside the CIs.
- **Adult:** very consistent. The 10 seeds span 0.006 test AUC, they match logistic regression,
  and the paper network sits mid-range. Gradient boosting reaches about 0.930 on a comparable
  encoding, so about 0.024 of headroom is not captured at 100 generations and population 250.
- **MONK-2:** bimodal. 3 of 10 seeds (s1, s5, s7) reach train AUC 1.0 and test AUC about 1.0;
  the other seven stall between 0.77 and 0.95. The paper's network is one of the lucky outcomes.
- **Training continues to pay off:** the best genome usually comes from the final generation, so
  none of these budgets is converged. Heart at 40 generations (the paper's budget, where the
  original run was interrupted) stops especially early.

**Wall clock:** `runs.csv` reports raw end-minus-start times.
- Heart runs (3.5 h each) used the old code. Its slow per-genome database lookups dominated the
  run time; this was fixed in `9259bf1`, with bit-identical results.
- Heart-100 and MONK-2 took 0.2–0.5 h per run on the new code.
- Adult's raw times (about 47 h) include a 42.5 h pause, and for s4, s5, s7, s8 and s9 about
  1.3 h more of throttling (see `RUN_INCIDENTS.md`). Net training was roughly 4–6 h per run,
  with 5–10 runs in parallel on one core each.

**Reproducibility:** seeded runs are bit-identical.
- Two concurrent runs with the same seed matched genome for genome (750/750).
- All 10 Heart-100 runs reproduce the first 40 generations of the Heart runs exactly
  (100,000/100,000 genomes), across a code change in the database layer.

## 3. Explanation results

| | Heart | Heart-100 | MONK-2 | Adult |
|---|---|---|---|---|
| Full coverage | 10/10 | 10/10 | 10/10 | 10/10 |
| Claude time, minutes | 12.5 (9–17) | 9.6 (7–22) | 44 (41–54) | 36 (15–71) |
| Leaf annotations | 3 (2–6) | 4 (4–7) | 3.5 (2–6) | 9.5 (6–14) |
| Composition annotations | 1 (1–5) | 3 (2–5) | 1 (1–2) | 1 (1–5) |
| Node splits | 4 (2–10) | 6.5 (4–11) | 6 (3–8) | 1.5 (0–8) |
| Identity nodes | 0 (0–5) | 1 (0–5) | 1 (1–4) | 6.5 (0–17) |
| Annotation monosemanticity* | 1.00 (0.33–1.00) | 0.67 (0.43–1.00) | 1.00 (0.50–1.00) | 1.00 (0.88–1.00) |
| Function preservation, worst | 3.6e-15 | 5.6e-16 | 2.4e-15 | 7.2e-16 |
| Closed-form error as displayed, median | 0.013 | 0.015 | 0.028 | 0.044 |

\* Annotation monosemanticity is the share of original hidden nodes that sit in exactly one leaf
annotation without being split. Nodes that can only live in a composition count against it.

**On the closed-form error:** leaf formulas evaluated numerically match the subgraph to about
1e-15. The "displayed" error comes from rounding weights to 3 significant figures in the LaTeX
form. It is largest on Adult (up to 0.57 logit, in a saturated region), where standardised rare
one-hots reach values of 20–60.

**Time:** Heart and Heart-100 were explained in minutes. MONK-2 and Adult took about 15–70
minutes, partly because those agents shared the CPU with training runs and SHAP on Adult is slow.

## 4. What the networks compute

### Heart: the same near-logistic-regression, with seed-specific corrections

Cross-seed comparison: `experiments/10seed/compare_heart.py` and `compare_heart.json`.

- **Near-identical function:** pairwise correlation of the training logits has median 0.983
  (minimum 0.952), and decisions agree on 94.6% of patients. Each seed's logit correlates
  0.97–0.99 with the paper network's.
- **Linear core:** 97–99.9% of each network's logit variance is linear in the inputs. `ca`,
  `cp:4` and `chol` appear in all 10 linear cores with the same sign; `slope:2` appears in 9.
- **Nonlinear corrections are mostly specific to the seed:** residual correlation after removing
  the best linear fit has median 0.13. The recurring motifs:
  - a sex × exercise-angina interaction (s0, s6, s7, s9 and the paper network; r = 0.33–0.67);
  - a fixed-thallium-defect × normal-fasting-sugar gate (s4, s9; r = 0.70);
  - an ST-depression hinge conditioned on a normal resting ECG (s1, s5; r = 0.60);
  - oldpeak in some form (6 of 10).
- **Small, fragile corrections:** they are worth 0.005–0.022 train AUC, and some act on only
  8–15 patients.

### Heart-100: longer training grows the nonlinear part

Train AUC rises from 0.943 to 0.956 and test AUC from 0.878 to 0.892 (medians). The price is
about double the hidden nodes and three times the skip connections.

In every seed the 40-generation motifs survive and are **strengthened and connected**, not
replaced:
- **Nonlinear block size:** its logit SD typically rises from about 0.8 to 2–3. In s2 it is now
  larger than the linear part.
- **s5:** the ischaemia correction became the main pathway (+0.027 rather than +0.010 train AUC).
- **s4:** the thal/fasting-sugar gate became a gate making male sex and exercise angina
  sub-additive.
- **s9:** a correction acting on 10 patients became an age hinge acting on 82.

All 10 Heart-100 networks were still explained, in 7–22 minutes each. Annotation
monosemanticity drops from 1.0 to 0.67 (median): more hidden nodes take inputs from several
annotations, so they can only be placed in compositions.

### MONK-2: explainable even when wrong

Only 3 of 10 seeds learned the target rule ("exactly two of six attributes take their first
value"). **All 10 were explained to full coverage, including the 7 that are wrong**, and each
explanation states exactly how its network deviates from the rule (`rule_agreement` in
`explanations.csv`).

- **One shared construction:** every network that represents the window builds it the same way,
  from a linear count ramp and one ReLU gate at the opposite edge. All of them use a descending
  ramp with a low-count veto, which mirrors the paper network's construction.
- **Ranking is not the same as thresholding:** two of the three AUC-1.0 networks (s1, s5) rank
  every input correctly but misclassify 8–11% at 0.5.
- **Recurring failure:** four seeds (s3, s4, s6, s9) treat `a5 = 1` as a veto instead of
  counting it as an indicator.
- **Missing lower edge:** two seeds (s0, s2) learned only the upper edge of the window, so 0 or
  1 first values also come out positive.

### Adult: sparse generalised additive models

All 10 are additive in the logit, with almost no interactions between feature groups.
- **Linear backbone:** a sparse backbone over 41–55 inputs carries most of the signal. Removing
  it drops train AUC to 0.61–0.79.
- **Capital-gain penalty:** the one nonlinearity found in nearly every seed is a concave,
  piecewise-linear penalty with kinks at about 3.8–5.2k USD.
- **Seed-specific nonlinearities:** an age hinge (U- or V-shaped; s3, s4, s5, s6), a convex hours
  term (s9), and an education hinge (s2).
- **Inert hidden units:** most other hidden units are exact single-category offsets, always-on
  relays, always-off units, or structure that can be proven dead. One example is s4's 4-node
  chain, which is identically zero for every valid input.
- **Rare-category switches:** z-scored one-hot inputs turn rare categories into very large logit
  switches fitted to a handful of rows. Examples are Honduras +183 logit (14 rows), Armed-Forces
  −80 (12), Married-AF-spouse −59, Preschool +43 (64) and Without-pay +61 (17). These are
  probably overfitting. They are visible, named and annotated in the explanations, which makes
  them candidates for targeted repair.

### Cross-cutting conclusion for Ch 7

PropNEAT reliably finds sparse networks that can be explained in under an hour each. Mostly,
though, they are explainable *because* they are nearly linear: a linear core plus a few hinges
and gates. On these three datasets the nonlinear parts add 0.002–0.05 AUC, and PropNEAT matches
(Adult) or trails (Heart) L2 logistic regression on test. The method's further value shows
where the network is *wrong or fragile*:
- the MONK-2 networks that fail;
- the rare-category switches in Adult;
- corrections in Heart that rest on very few patients.

A dataset with real nonlinear headroom (see §7) is needed to show a performance benefit.

## 5. Tooling changes made during the experiment

These are all on `exp/10seed-consistency`.

- **Seeded runner:** `experiments/10seed/run_seed.py` and `launch.sh`.
- **DB write speed-up (`9259bf1`):** `_save_training_metrics` did 1,250 unindexed genome lookups
  per generation and wrote nothing. They are now batched, with identical output: about 73 s →
  0.7 s per Heart generation.
- **Engine parity (`88c1b00` merge):**
  - **Output activation:** `StructureNetwork` applied ReLU at the output, but the trained model
    (`NeuralNeat`) applies a sigmoid. On heart-s1 that gave train AUC 0.895 against the true
    0.939. Whole-model MCP evidence (performance, SHAP, output plots) was affected, and probably
    so was the paper's.
  - **Same merge:** compositions are now evaluated over their full subtree, the validator counts
    descendants, `detect_splits` flags input fan-out, identity nodes are no longer clipped,
    `rename_annotation` is fixed, and `line_plot` is accepted as an alias.
- **SHAP fix:** it kept only the top-10 features per row and zeroed the rest (`l1_reg`). That
  erased small but real effects; paper SHAP figures on Adult are suspect.
- **Compositional coverage:** it was always 1.0. It is now defined as progress towards a single
  global root (paper Def 11, well-formedness criterion 4).
- **Concurrency (`c0db7fb` merge):**
  - The operations stream is updated under a row lock, because concurrent `add_evidence` calls
    had lost writes.
  - All `apply_operation` docstring examples were corrected and are now tested.
- **Experiment tooling:** `xn.py` (MCP tools from the shell), `rename_inputs.py`,
  `check_explanation.py`, `verify_record.py`, `compute_runs.py`, `build_explanations_csv.py`.

**Known open issues:**
- The annotate validator accepts compositions with empty `subgraph_connections`.
  `get_coverage` catches the result.
- `get_viz_summary` reports a hidden exit's activation before the output weight.
- `activation_profile` requires `node_id`.
- KernelSHAP on Adult is slow (3–36 min), and it gives unconnected inputs small non-zero values
  (estimator noise).

## 6. Open decisions for Mike

1. **The paper's disconnected-input annotations.** Under the corrected compositional coverage,
   the paper's Adult explanation scores 0.9 and MONK-2 scores 0.75: `unused_inputs` and
   `pruned_inputs` are separate roots. Either exempt annotations of disconnected inputs from the
   single-root rule, or edit the two stored explanations to put them under the root. The 10-seed
   explanations list unconnected inputs as entries of the root, so they pass either way.
2. **Verification.** Verify down the pre-registered order, starting with adult-s0, heart-s5 and
   monk2-s2 (`PREREG.md`). Record `verified_by_mike`, the time taken and any changes in
   `explanations.csv`.
3. **Recompute for the thesis:** whole-model evidence and SHAP from the paper, because of the
   output-activation and SHAP bugs above.

## 7. Next: a dataset with real nonlinear headroom

Logistic regression vs gradient boosting, test AUC, on PMLB with an 80:20 split (seed 42):

| Dataset | n | Features | LR | GBM | Gap |
|---|---|---|---|---|---|
| churn | 5,000 | 20 | 0.862 | 0.934 | +0.072 |
| magic | 19,020 | 10 | 0.847 | 0.942 | +0.095 |
| phoneme | 5,404 | 5 | 0.787 | 0.940 | +0.153 |
| adult | 48,842 | 14 | 0.901 | 0.930 | +0.029 |
| monk1 | 556 | 6 | 0.717 | 1.000 | +0.283 |

Churn is next. It is real data with a +0.07 headroom, it has interpretable threshold and
interaction effects (customer-service calls, international plan × international minutes), and
it has a 51-category `state` feature. That feature is a natural test of
"see the rare-category overfit, explain it, repair it".

## 8. Still to produce (brief §7)

- figures (strip plots per dataset, training and explanation time);
- `ch7_table.tex`;
- a reproduction README;
- the permutation-test design paragraph (deferred to future work).
