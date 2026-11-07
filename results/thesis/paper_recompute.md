# AAAI paper: whole-model evidence and SHAP, recomputed with the current code (task 3)

Written 7 Nov 2026 on branch `thesis/numbers`. **This is a report for decision D5. No paper text, results file or figure has been edited.** The original figures stay where they are, and everything regenerated is in `paper_recompute/`.

**Paper source.** `PhD/drafts/explaneat-aaai27-reframed.tex`, read only.

**Genomes:**
- Heart `9022a9eb`, explanation `a03ebc86`;
- Adult `4bc8fa07`, explanation `7e3761fe`;
- MONK-2 `23dde0fa`, explanation `d6ff7ca0`.

**Two fixes since the paper:**
1. **Output activation.** StructureNetwork used to apply the output gene's ReLU at the output; the trained model, NeuralNeat, applies a sigmoid. The current code calls `override_output_activation(model_output_activation)`.
2. **SHAP `l1_reg`.** KernelExplainer's default `num_features(10)` keeps only the top 10 features per row and gives every other feature exactly 0. The current code passes `l1_reg=False`.

**Method:**
- **Scripts:**
  - `scripts/thesis/paper_recompute.py`: `core`, `shap <label>` and `figures`;
  - `scripts/thesis/paper_shap_summary.py`.
- **Two columns per comparison:**
  - **old:** the pre-fix behaviour, with a ReLU output and shap's default `l1_reg`;
  - **new:** the current code path.
- **Performance and ablations:** full train and test splits. Bootstrap CIs use 10,000 paired resamples with seed 42, as in `scripts/aaai27/common.py`.
- **SHAP:** test split, through the MCP `compute_shap` path:
  - the 200-row seeded sample (`max_samples = 200`, the MCP default; this is the full test set for Heart, n = 61, and MONK-2, n = 87);
  - background = the same rows;
  - `np.random.seed(42)` before each call;
  - shap 0.49.1;
  - one torch thread.
- **Determinism:** Adult KernelSHAP took about 30 s per call, not the 3–36 min expected. A second run gave bit-identical values, so both variants were run for every genome.
- **Raw output:**
  - `paper_recompute/core.json`;
  - `paper_recompute/shap_{heart,adult,monk2}.json`;
  - `paper_recompute/shap_summary.json`.
- **Figures:**
  - `paper_recompute/shap_{heart,adult,monk2}.png`: old (grey) beside new (blue), top 15 by mean |SHAP|;
  - `paper_recompute/capgain-response_recomputed.png`.

## Headline

- **No reported number in the paper changes.** Every performance, ablation and confidence-interval figure was computed with NeuralNeat, the training engine, which was always correct. Each one reproduces to the paper's precision with the current StructureNetwork.
- The pre-fix StructureNetwork would have given different numbers, for example Heart test AUC 0.873 instead of 0.897. None of those numbers reached the paper.
- **Every claim still holds.** Three items need attention:
  1. **Adult SHAP.** With the current code, capital-gain ranks **3rd overall** by mean |SHAP|; it ranked 10th before. The backbone's share of total mean |SHAP| falls from 96% to 87%. The claim, that the backbone terms dominate and are led by marital-status, education-num and relationship, still holds as worded. But "the bulk of the model is linear… this linear backbone carries most of the model's predictive power" is now 87%, not about 96%. And the deep capital-gain pathway is the third-largest single attribution, not a minor one.
  2. **MONK-2 composition formula.** The paper writes the output as $\mathrm{ReLU}(2.72 + \mathrm{Ramp} - 7.34\cdot\mathrm{Gate} - 2.87\cdot\mathrm{Inert})$. The trained model applies a **sigmoid** at the output, so the closed form is $\sigma(\cdot)$ of the same expression. Every constant, and the decision boundary at a pre-activation of 0, is unchanged; only the outer function is wrong.
  3. **Adult "output unit's clipping".** The sentence "remaining knots fall below the data floor or are masked by the output unit's clipping" assumes a clipping output. The output is a sigmoid and does not clip. The stored evidence says "the output hub's clipping", which is hidden node 655, a ReLU. That reading is correct, so the fix is wording only.
- **Native-country SHAP ≈ 0 (Adult) holds, but it is a weak piece of evidence** (see the SHAP table below). It is unaffected by the fixes.

## Table: every whole-model evidence and SHAP item in the paper

"Paper" gives the value as printed. "Old" is the pre-fix StructureNetwork or SHAP. "New" is the current code; NeuralNeat gives the same values to 1e-15 throughout.

### Performance and ablation

| # | Paper item (section) | Paper | Old (pre-fix StructureNetwork) | New (current code) | Change to the paper value | Claim holds? |
|---|---|---|---|---|---|---|
| 1 | Adult train / test AUC (Exp 1) | 0.909 / 0.905 | 0.899 / 0.896 | 0.9085 / 0.9055 | none | yes |
| 2 | Adult native-country ablation: test accuracy (Exp 1) | 0.851 → 0.852 | 0.846 → 0.846 | 0.8514 → 0.8520 | none | yes |
| 3 | Adult native-country ablation: ΔAUC, 95% CI | −0.0001 [−0.0003, +0.0002] | +0.0003 [−0.0001, +0.0006] | −0.00008 [−0.00032, +0.00024] | none (equal at 4 dp) | yes |
| 4 | Heart train / test AUC (Exp 2) | 0.946 / 0.897 | 0.919 / 0.873 | 0.9455 / 0.8966 | none | yes |
| 5 | Heart: linear core only (Table) | 0.860, Δ −0.037 [−0.079, −0.005] | 0.837, −0.036 [−0.080, −0.004] | 0.860, −0.037 [−0.079, −0.005] | none | yes |
| 6 | Heart: − benign-profile gate | 0.869, −0.028 [−0.069, +0.001] | 0.849, −0.024 [−0.063, 0.000] | 0.869, −0.028 [−0.069, +0.001] | none | yes |
| 7 | Heart: − resting-BP interaction | 0.885, −0.012 [−0.031, +0.002] | 0.859, −0.013 [−0.044, +0.010] | 0.885, −0.012 [−0.031, +0.002] | none | yes |
| 8 | Heart: − risk-aggregation gate | 0.899, +0.002 [−0.011, +0.016] | 0.873, 0.000 [−0.009, +0.008] | 0.899, +0.002 [−0.011, +0.016] | none | yes |
| 9 | Heart: train AUC, full → without risk-aggregation gate | 0.946 → 0.942 | 0.919 → 0.929 (the opposite direction) | 0.9455 → 0.9423 | none | yes |
| 10 | MONK-2 train / test AUC (Exp 3) | 1.000 / 1.000 | 1.000 / 1.000 (accuracy 0.988 / 0.908) | 1.000 / 1.000 | none | yes |
| 11 | MONK-2 thresholded predictions equal the rule on all 432 configurations | 432 / 432 | **420 / 432** at 0.5 | 432 / 432 (NeuralNeat 432 / 432) | none | yes |
| 12 | LR baselines (Method): Adult; Heart; MONK-2 | 0.905 [0.899, 0.912]; 0.919 [0.841, 0.978]; 0.423 | — | refit: 0.905 [0.899, 0.912] (C = 0.1); 0.919 [0.841, 0.978] (C = 0.1); 0.423 | none | yes |
| 13 | Densified twin: test AUC 0.905; 38–60 effective inputs per unit | — | — | not recomputed: a separate MLP, not touched by either fix | — | — |

Notes:
- Row 9: the old StructureNetwork would have *reversed* the train-side half of the paper's overfitting argument, since removing the gate raises train AUC there. The paper used NeuralNeat, so the argument stands.
- Row 11: under the old ReLU output a prediction is positive only if the pre-activation exceeds 0.5. The 12 configurations that fail are positives with pre-activations in (0.115, 0.5). The paper's claim is about the trained model (sigmoid, threshold at a pre-activation of 0), and there it holds exactly.

### SHAP

| # | Paper item | Old (ReLU output, `l1_reg` = top 10) | New (sigmoid output, `l1_reg=False`) | Change | Claim holds? |
|---|---|---|---|---|---|
| 14 | Adult: "by mean-absolute SHAP its [linear backbone's] terms dominate, led by marital-status, education-num, and relationship" | Overall top 5: marital-status:2, education-num, relationship:1, age, sex. Capital-gain 10th. Backbone (40 inputs) = **96.4%** of total mean \|SHAP\| | Overall top 5: marital-status:2, education-num, **capital-gain**, relationship:1, age. Backbone = **87.5%**. Capital-gain-core inputs 3.4% → **9.4%**. Top 3 *backbone* terms unchanged | **Flag:** capital-gain rises from 10th to 3rd; backbone share falls 96% → 87%. Spearman ρ (old vs new, 107 features) = 0.80 | yes, as worded; the emphasis changes (see Headline) |
| 15 | Adult: five native-country one-hots in the core have test SHAP ≈ 0 | All five exactly 0 (zeroed by `l1_reg`) | native-country:15, :22, :3, :34 exactly 0; :23 = 0.0005 (rank 31 / 107) | Values effectively unchanged | yes, but see the caveat below |
| 16 | Heart: "by mean-absolute SHAP its terms dominate, led by the number of major vessels, type-4 chest pain, and female sex" | ca, cp:4, slope:2, sex:0, chol | **ca, cp:4, sex:0**, slope:2, chol | Ranks 3 and 4 swap (the stored hypothesis already had sex:0 3rd) | yes, exactly as worded |
| 17 | MONK-2 | — (the paper reports no SHAP) | Top: a6:2, a6:1, a1:3, a1:2, a1:1 | — | — |

**Caveat on item 15.** In the 200-row SHAP sample, four of the five native-country columns are never 1, and the fifth is 1 in a single row. KernelSHAP therefore assigns 0 by construction, not because the edges are inert.
- On the full test split (n = 9,769) they are 1 in 1, 17, 25, 26 and 5 rows (−73, −81, −82, −89, −94).
- In train they are 1 in 0, 88, 81, 96 and 16 rows. native-country:15 (−73, weight +1.10) has **no** positive training row at all.

The paper's actual evidence for "inert" is the ablation (rows 2–3), which holds. The "test SHAP is ~0" half of the "structure vs attribution" tension is weaker than it reads. Rewording it to "SHAP on a sample cannot see them (they are almost never active)" would be accurate.

The magnitudes of mean |SHAP| change about 7–8× between old and new, because the old ReLU output is on an unbounded scale and the new sigmoid output is on the probability scale. The paper prints no SHAP magnitudes, so none needs changing. Exact zeros fall from 77 to 32 (Adult), 9 to 1 (Heart) and 4 to 0 (MONK-2). The remaining Adult zeros are columns that are constant in the 200-row sample. The 53 unconnected Adult inputs get non-zero SHAP of at most 0.0004 under the new code, which is estimator noise; it is a known open issue.

### Figures and closed forms

| # | Paper item | Recomputed | Change | Claim holds? |
|---|---|---|---|---|
| 18 | `fig:capgain` (commented out in the `.tex`; caption from `scripts/aaai27/run_d_capgain_figure.py`): two knots at \$3,225 and \$4,300, 92% at zero capital gain | Regenerated with the current code: identical. Two knots at \$3,225 and \$4,300, floor contribution +0.1417, terminal slope −6.237 per z, and the cross-check against StructureNetwork passes. The figure plots the pathway's contribution to the output *pre-activation*, which neither fix touches | none | yes |
| 19 | Exp 1 text: "masked by the output unit's clipping" | The output unit is a sigmoid and does not clip. The hub (655), a hidden ReLU, does | wording | yes, if it reads "the hub's clipping" |
| 20 | Exp 3: $\mathrm{ReLU}(2.72 + \mathrm{Ramp} - 7.34\cdot\mathrm{Gate} - 2.87\cdot\mathrm{Inert})$ | The trained output is $\sigma(2.716 + \mathrm{Ramp} - 7.339\cdot\mathrm{Gate} - 2.874\cdot\mathrm{Inert})$, with the decision boundary at a pre-activation of 0. Exact on all 432 inputs: count 2 ≥ +0.115, count ≤ 1 ≤ −0.646, count ≥ 3 ≤ −6.42 | outer function ReLU → σ | yes: peak at count 2, the ramp crosses between 1 and 2, and the gate forms the upper edge |
| 21 | Exp 3: gate margins (1.82 off / 1.26 on), coefficients 4.49–7.07, constant −14.86 | Hidden-unit quantities, not affected by either fix | none | yes |
| 22 | `fig:explorer` (screenshot) | Not regenerated. A UI screenshot; any formula or plot in it may show pre-fix output values | — | — |

**Stored-evidence discrepancies (not in the paper).** The MONK-2 explanation's evidence records "min positive-class pre-activation +0.42" and "max count-3 pre-activation −2.03". The record itself notes these were computed with full-dataset scaler statistics. The exact values on the split's scaler are **+0.115** and **−6.42**. The paper does not print them.

Several root hypotheses in the stored explanations describe the output as ReLU:
- Adult `full_model`: "output = ReLU(0.527 + …)";
- Heart `heart_disease_risk_score`: "y0 = ReLU(bias 0.318 + …)";
- MONK-2 `exactly_two_of_six`.

Each should read σ. They are explanation text, not paper numbers, and they were left unchanged (read-only).

## For D5

There are no numeric errata. The candidate text changes are:
- (a) the MONK-2 composition's outer function, ReLU → σ (item 20);
- (b) "output unit's clipping" → "hub's clipping" (item 19);
- (c) optionally, give the Adult backbone's SHAP share (87%) and mention that capital-gain is the third-largest attribution (item 14);
- (d) optionally, soften the native-country SHAP sentence (item 15).
