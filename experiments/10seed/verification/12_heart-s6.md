# Verification packet 12 · heart-s6

Pre-registered verification order: 12 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/9b644a62-ebe7-48cc-82cf-c53f7da36073?experiment=84fd741a-296e-413d-93a6-8874237311e9&name=10seed-heart-s6), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 6 |
| Genome id | `9b644a62-ebe7-48cc-82cf-c53f7da36073` |
| Experiment | 10seed-heart-s6 (84fd741a-296e-413d-93a6-8874237311e9); best at generation 39 of 40, NEAT key 9768 |
| Test AUC (runs csv) | 0.865 \[95% CI 0.765, 0.946\] |
| Train AUC | 0.946 |
| Size | 18 of 26 inputs connected; 3 hidden nodes; 25 enabled connections; depth 3; 3 skip connections + 14 direct input→output edges |
| Annotations | 2 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 3 of 3 |
| Claude's time | 9.3 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:33:15+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Mostly logistic regression: 14 direct edges (logit SD 3.42, train AUC 0.926 alone). One nonlinear block (logit SD 1.04): a two-gate sex x exercise-angina interaction; gate 1213 scores 'female without exercise angina' (read out -0.760), gate 47 fires for no-angina patients with low 1213 score (mostly men) and adds +0.384. Exercise angina has no direct edge, so its whole effect runs through this block. Block adds +0.020 train AUC (0.926 -\> 0.946). slope:1 term (3660) is an inert affine pass-through.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_30` |
| Entry nodes | -1 (age), -3 (sex:1), -5\_a (cp:2\_a), -6 (cp:3), -7 (cp:4), -9 (chol), -11 (fbs:1), -18 (oldpeak), -20 (slope:2), -21 (slope:3), -22 (ca), -24 (thal:4), -25 (thal:6), -26\_a (thal:7\_a), -4 (cp:1), -8 (trestbps), -12 (restecg:0), -13 (restecg:1), -14 (restecg:2), -15 (thalach), -17 (exang:1), -23 (thal:3) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | sex\_exang\_interaction, slope1\_offset |
| Dimensionality (in → out, incl. descendants) | 28 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{0.978 e^{- 0.363 x_{age} - 1.57 x_{ca} - 0.65 x_{chol} - 0.448 x_{cp:2\_a} - 0.315 x_{cp:3} - 1.3 x_{cp:4} - 0.0516 x_{fbs:1} - 0.665 x_{oldpeak} - 0.382 x_{sex:1} - 1.04 x_{slope:2} - 0.0815 x_{slope:3} - 0.313 x_{thal:4} + 0.134 x_{thal:6} - 0.576 x_{thal:7\_a} + 0.76 \operatorname{sex\_exang\_interaction\_0}{\left(x_{sex:0},x_{cp:2\_b},x_{fbs:0},x_{exang:0},x_{thal:7\_b} \right)} - 0.384 \operatorname{sex\_exang\_interaction\_1}{\left(x_{sex:0},x_{cp:2\_b},x_{fbs:0},x_{exang:0},x_{thal:7\_b} \right)} + 0.0648 \operatorname{slope\_1\_offset}{\left(x_{slope:1} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}z = 0.363x_{age}+1.57x_{ca}+0.65x_{chol}+0.448x_{cp:2}+0.315x_{cp:3}+1.3x_{cp:4}+0.0516x_{fbs:1}+0.665x_{oldpeak}+0.382x_{sex:1}+1.04x_{slope:2}+0.0815x_{slope:3}+0.313x_{thal:4}-0.134x_{thal:6}+0.576x_{thal:7}-0.76\,h_{1213}+0.384\,g_{47}-0.0648\,\mathrm{slope1\_offset}+0.0221;\; p=\sigma(\operatorname{relu}(z))\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: the risk logit is additive: a logistic-regression core over 14 direct input edges (dominated by ca +1.57, cp:4 +1.30, slope:2 +1.04, oldpeak +0.66, chol +0.65, thal:7 +0.58; all risk-raising except thal:6 -0.13) plus two child terms: -0.760 h1213 + 0.384 g47 from sex\_exang\_interaction (the only route by which exercise angina affects risk, and only materially for women) and -0.065 x 3660 from slope1\_offset (near-inert). The output node applies ReLU and the model then applies a sigmoid, so p = sigmoid(relu(z)): negative logits (60% of train) all map to p=0.5 and the decision is z\>0. The 8 inputs cp:1, trestbps, restecg:0/1/2, thalach, exang:1 and thal:3 have no edges in the phenotype and are included as entries with zero effect. Evidence (train, n=242): full AUC 0.9457 / acc 0.876; linear core alone (children at their means) AUC 0.9255 / acc 0.855; logit SD 3.58 total, 3.42 linear core, 1.04 interaction block, 0.029 slope:1 term. Node-0 sensitivity ranks ca 0.63, cp:4 0.52, slope:2 0.42, oldpeak 0.27, chol 0.26, exang:0 0.26 (via the block).

**Evidence (2 records, DB):**

- `analytical_formula`: Additive composition: linear core + sex\_exang\_interaction (-0.760 h1213 + 0.384 g47) + slope1\_offset (-0.065).
- `ablation_study`: The linear core carries most of the signal (AUC 0.926 alone); the sex x exercise-angina block adds +0.020 train AUC; the slope:1 term is inert.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "get_formula displays the output as relu(z); the model applies a sigmoid on top for this binary task, so p = sigmoid(relu(z)) and the decision is z>0. Composition is purely additive in the logit.",
 "unused_inputs": [
  "cp:1",
  "trestbps",
  "restecg:0",
  "restecg:1",
  "restecg:2",
  "thalach",
  "exang:1",
  "thal:3"
 ],
 "direct_edge_logit_sd": {
  "ca": 1.572,
  "age": 0.363,
  "chol": 0.65,
  "cp:2": 0.448,
  "cp:3": 0.315,
  "cp:4": 1.301,
  "fbs:1": 0.052,
  "sex:1": 0.382,
  "thal:4": 0.313,
  "thal:6": 0.134,
  "thal:7": 0.576,
  "oldpeak": 0.665,
  "slope:2": 1.043,
  "slope:3": 0.081
 }
}
```

```json
// ablation_study
{
 "n": 242,
 "full": {
  "acc": 0.876,
  "auc": 0.9457
 },
 "split": "train",
 "logit_sd": {
  "total": 3.58,
  "linear_core": 3.42,
  "slope1_offset": 0.029,
  "sex_exang_interaction": 1.04
 },
 "gate47_to_mean": {
  "auc": 0.9414
 },
 "sensitivity_node0": {
  "ca": 0.63,
  "age": 0.146,
  "chol": 0.261,
  "cp:4": 0.522,
  "sex:0": 0.169,
  "sex:1": 0.153,
  "exang:0": 0.26,
  "oldpeak": 0.267,
  "slope:2": 0.418,
  "cp:2(direct)": 0.18,
  "thal:7(direct)": 0.231
 },
 "frac_logit_positive": 0.401,
 "function_preservation": "splits of cp:2 and thal:7 only; see check_explanation",
 "h1213_readout_to_mean": {
  "auc": 0.9347
 },
 "slope1_offset_to_mean": {
  "auc": 0.9454
 },
 "linear_core_only (children replaced by mean)": {
  "acc": 0.855,
  "auc": 0.9255
 }
}
```

</details>

### A.1 Leaf: `sex_exang_interaction`

|  |  |
|---|---|
| Annotation id | `ann_28` |
| Entry nodes | -2 (sex:0), -5\_b (cp:2\_b), -10 (fbs:0), -16 (exang:0), -26\_b (thal:7\_b) |
| Exit nodes | 1213, 47 |
| Internal / exit nodes in its own subgraph | 1213, 47 |
| Dimensionality (in → out, incl. descendants) | 5 → 2 |
| Live closed-form error (3 s.f. display) | 0.00821 |
| Live numeric-function error | 4.44e-16 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( - 0.509 x_{cp:2\_b} \\
&\quad + 1.31 x_{exang:0} \\
&\quad - 0.563 x_{fbs:0} \\
&\quad + 1.23 x_{sex:0} \\
&\quad + 0.476 x_{thal:7\_b} \\
&\quad + 0.091 \bigr)
\end{aligned} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.956 x_{exang:0} \\
&\quad - 1.93 \operatorname{relu}{\left(- 0.509 x_{cp:2\_b} + 1.31 x_{exang:0} - 0.563 x_{fbs:0} + 1.23 x_{sex:0} + 0.476 x_{thal:7\_b} + 0.091 \right)} \\
&\quad + 1.43 \bigr)
\end{aligned}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}h_{1213}=\operatorname{relu}(1.23x_{sex:0}+1.31x_{exang:0}-0.509x_{cp:2}-0.563x_{fbs:0}+0.476x_{thal:7}+0.091),\; g_{47}=\operatorname{relu}(1.43+0.956x_{exang:0}-1.93h_{1213})\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: a sex x exercise-angina interaction built from two chained ReLU gates. Gate 1213 = relu(1.23 sex:0 + 1.31 exang:0 - 0.509 cp:2 - 0.563 fbs:0 + 0.476 thal:7 + 0.091) is a female-without-exercise-angina score; gate 47 = relu(1.43 + 0.956 exang:0 - 1.93 h1213) fires for patients without exercise angina whose 1213 score is low (mostly men) and is shut off by 1213. The output reads -0.760 h1213 + 0.384 g47. Evidence (train, n=242): group mean block logit is +0.04 (male, exang), +0.18 (male, no exang), -0.39 (female, exang), -1.76 (female, no exang). Exercise angina has no direct edge to the output, so its entire effect is this block: flipping exang no-\>yes raises the logit by +1.81 on average for women (range 0.78..2.11) but by -0.17 for men (range -0.78..1.87). g47 is active for 84% of men without angina but 19% of women without angina. cp:2, fbs:0 and thal:7 only shift h1213 (sensitivities 0.22-0.26 vs 0.57-0.60 for sex:0/exang:0). Odd signs (thal:7 and fbs:1 making the block slightly less protective / more protective respectively) are small modifiers we do not interpret clinically. Ablating gate 47 to its mean drops train AUC 0.9457-\>0.9414; ablating 1213-\>output drops it to 0.9347.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form: one ReLU gate on sex:0/exang:0 (plus small cp:2, fbs:0, thal:7 modifiers) and a second gate inhibited by it.
- `empirical_viz`: Train split, full sample (242). The block is a sex x exercise-angina interaction: flipping exang from no to yes raises the logit by +1.81 for women on average but by -0.17 for men.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "readout": "logit += -0.760*h1213 + 0.384*g47",
 "piecewise": "with no exang (x_exang0=0.716): g47=relu(2.115-1.926h), so block = 0.812-1.500h for h<1.098 and -0.760h beyond; with exang (x_exang0=-1.397): g47=relu(0.094-1.926h)~0, block ~ -0.760h",
 "interpretation": "block contribution is monotone decreasing in h1213; g47 adds a +0.81 offset only when there is no exercise angina and h is small (mostly men)"
}
```

```json
// empirical_viz
{
 "n": 242,
 "note": "exang has no direct edge to the output (exang:1 is unused), so its whole effect on risk goes through this block",
 "split": "train",
 "activation": {
  "47": {
   "max": 2.115,
   "mean": 0.6,
   "zero_rate": 0.31
  },
  "1213": {
   "max": 4.02,
   "mean": 0.849,
   "zero_rate": 0.541
  }
 },
 "group_means": [
  {
   "n": 63,
   "g47": 0.1,
   "group": "male, exang",
   "h1213": 0.0,
   "block_logit": 0.04
  },
  {
   "n": 96,
   "g47": 1.42,
   "group": "male, no exang",
   "h1213": 0.48,
   "block_logit": 0.18
  },
  {
   "n": 19,
   "g47": 0.06,
   "group": "female, exang",
   "h1213": 0.54,
   "block_logit": -0.39
  },
  {
   "n": 64,
   "g47": 0.02,
   "group": "female, no exang",
   "h1213": 2.33,
   "block_logit": -1.76
  }
 ],
 "sensitivity_47": {
  "cp:2": 0.146,
  "fbs:0": 0.161,
  "sex:0": 0.353,
  "thal:7": 0.136,
  "exang:0": 0.751
 },
 "g47_active_rate": {
  "male_no_exang": 0.84,
  "female_no_exang": 0.19
 },
 "sensitivity_1213": {
  "cp:2": 0.233,
  "fbs:0": 0.258,
  "sex:0": 0.566,
  "thal:7": 0.218,
  "exang:0": 0.601
 },
 "counterfactual_exang_flip_logit_change": {
  "male": {
   "max": 1.867,
   "min": -0.775,
   "mean": -0.172
  },
  "female": {
   "max": 2.105,
   "min": 0.776,
   "mean": 1.809
  }
 }
}
```

</details>

### A.2 Leaf: `slope1_offset`

|  |  |
|---|---|
| Annotation id | `ann_29` |
| Entry nodes | -19 (slope:1) |
| Exit nodes | 3660 |
| Internal / exit nodes in its own subgraph | 3660 |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 2.51e-04 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.446 x_{slope:1} + 0.797 \right)}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}y=\operatorname{relu}(0.446x_{slope:1}+0.797)\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Hypothesis: a near-inert affine pass-through of slope:1 (upsloping ST segment). Evidence: the ReLU pre-activation 0.446 slope:1 + 0.797 lies in \[0.373, 1.266\] on train, so it never clips; through the -0.0648 readout it adds -0.029 slope:1 - 0.052 to the logit (logit SD 0.029). Ablating it to its mean changes train AUC 0.9457-\>0.9454. It is a vestigial linear term, slightly protective for upsloping ST.

**Evidence (2 records, DB):**

- `analytical_formula`: Affine on the data domain.
- `empirical_viz`: Never clips; contributes a logit SD of only 0.029, so effectively inert.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "readout": "3660 -> 0 weight -0.0648 => logit += -0.0289 slope:1 - 0.0516",
 "simplified_on_data": "slope:1 in [-0.952,1.051] so pre-activation in [0.373,1.266]: the ReLU never clips; 3660 = 0.446 slope:1 + 0.797 exactly"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "ablation": "replacing 3660 by its mean changes train AUC 0.9457 -> 0.9454",
 "activation": {
  "max": 1.266,
  "min": 0.373,
  "zero_rate": 0.0
 },
 "sensitivity": 0.446,
 "logit_contribution_sd": 0.029
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 3 |
| rename\_node | 26 |
| split\_node | 2 |

2 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -5 into -5\_a, -5\_b. Final copies: -5\_a → heart\_disease\_logit (entry) \| -5\_b → sex\_exang\_interaction (entry)
- `#27` split -26 into -26\_a, -26\_b. Final copies: -26\_a → heart\_disease\_logit (entry) \| -26\_b → sex\_exang\_interaction (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-07T14:35:07+13:00 (commit `546261f`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 8.88e-16 | 8.88e-16 | 2.22e-16 | agree |
| Closed-form max abs error (all leaves) | 0.008213 | 0.008213 | 0.008213 | agree |
| Numeric-function max abs error | n/a | 4.44e-16 | 4.44e-16 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 2 / 1 | n/a | 2 / 1 | agree |
| Operation counts | 2 splits, 0 identity | {"annotate": 3, "rename\_node": 26, "split\_node": 2} | {"annotate": 3, "rename\_node": 26, "split\_node": 2} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `sex_exang_interaction` | 0.008213 | 4.44e-16 |
| `slope1_offset` | 2.51e-04 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Two small modifiers inside gate 1213 have clinically odd signs (thal:7 and fbs:1 both make the block more protective); reported, not interpreted. 8 inputs with no phenotype edges (cp:1, trestbps, restecg:0/1/2, thalach, exang:1, thal:3) are root entries with zero effect; without them structural coverage was 0.742.

**Tool notes (record):** Root composition validation requires child\_annotation\_ids as annotation names, not ann\_N ids. get\_formula shows output as ReLU. Heatmap viz summary on multi-exit leaf only gives first-two-entry correlation; sex x exang table computed locally on train.

## What to look at

- Largest closed-form error: `sex_exang_interaction` (0.00821). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- 2 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s6.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s6', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
