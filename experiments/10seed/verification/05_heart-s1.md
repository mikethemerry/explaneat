# Verification packet 05 · heart-s1

Pre-registered verification order: 5 of 30 (`PREREG.md`, `PRE` in `build_explanations_csv.py`).

Open in the UI (with `uv run python -m explaneat api` running): [genome view](http://127.0.0.1:8000/genome/d3b1f42b-8347-408c-951e-db9111eaaf72?experiment=00a47573-5a7f-4ced-842d-de01e848fb74&name=10seed-heart-s1), then Tools → Review & verify. Back to the [README and progress table](README.md).

## Header

| Field | Value |
|---|---|
| Dataset / seed | heart / 1 |
| Genome id | `d3b1f42b-8347-408c-951e-db9111eaaf72` |
| Experiment | 10seed-heart-s1 (00a47573-5a7f-4ced-842d-de01e848fb74); best at generation 31 of 40, NEAT key 7173 |
| Test AUC (runs csv) | 0.88 \[95% CI 0.785, 0.956\] |
| Train AUC | 0.939 |
| Size | 17 of 26 inputs connected; 5 hidden nodes; 28 enabled connections; depth 3; 5 skip connections + 15 direct input→output edges |
| Annotations | 4 leaves, 1 compositions |
| Hidden nodes in exactly one leaf, unsplit | 5 of 5 |
| Claude's time | 9.4 min (2026-11-03T09:23:57+13:00 → 2026-11-03T09:33:20+13:00); within the 3 h box: yes |
| Protocol | v2 (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN\_PROTOCOL.md) |
| Full coverage | agent: yes; orchestrator record: yes; live: yes |

### Summary (from the record)

> Essentially logistic regression: 15 direct edges carry logit SD 5.3 of 7.0. ca enters by three parallel affine routes; cp:3 only via affine relay 1173; 2009 adds a small negative max-heart-rate term. The one consequential nonlinearity is gate 1801 = ReLU(1.48 restecg:0 + 0.805 oldpeak - 1.43): ST depression adds risk mainly when resting ECG is normal. Replacing all ReLUs by identity drops train AUC 0.939 -\> 0.916, almost all from 1801.

## Annotations (root → compositions → leaves)

Closed forms are computed live with the same code as `get_formula(force=True)` on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. Live errors are max \|closed form or numeric function − the subgraph's exit activations\| over all 242 training rows.

### A Composition: `heart_disease_logit`

|  |  |
|---|---|
| Annotation id | `ann_34` |
| Entry nodes | -2 (sex:0), -4 (cp:1), -5 (cp:2), -7 (cp:4), -8 (trestbps), -9 (chol), -11 (fbs:1), -13 (restecg:1), -16 (exang:0), -23 (thal:3), -25 (thal:6), -20\_a (slope:2\_a), -22\_a (ca\_a), -12\_a (restecg:0\_a), -18\_a (oldpeak\_a), -1 (age), -3 (sex:1), -10 (fbs:0), -14 (restecg:2), -17 (exang:1), -19 (slope:1), -21 (slope:3), -24 (thal:4), -26 (thal:7) |
| Exit nodes | 0 |
| Internal / exit nodes in its own subgraph | 0 |
| Children | ca\_slope2\_gate, cp3\_affine\_relay, normal\_ecg\_st\_gate, ca\_thalach\_chain |
| Dimensionality (in → out, incl. descendants) | 31 → 1 |

**Closed form (children as functions):**

$$
\begin{gathered}y_0 = \frac{1}{1.17 e^{- 1.53 x_{ca\_a} - 1.01 x_{chol} + 0.927 x_{cp:1} - 0.442 x_{cp:2} - 1.32 x_{cp:4} + 0.347 x_{exang:0} + 0.286 x_{fbs:1} - 0.869 x_{oldpeak\_a} + 0.0177 x_{restecg:0\_a} - 0.108 x_{restecg:1} + 2.01 x_{sex:0} - 1.85 x_{slope:2\_a} + 1.04 x_{thal:3} + 0.967 x_{thal:6} - 0.718 x_{trestbps} - 0.983 \operatorname{ca\_slope2\_gate}{\left(x_{slope:2\_b},x_{ca\_b} \right)} - 1.67 \operatorname{ca\_thalach\_chain\_0}{\left(x_{ca\_c},x_{thalach} \right)} + 0.165 \operatorname{ca\_thalach\_chain\_1}{\left(x_{ca\_c},x_{thalach} \right)} + 1.31 \operatorname{cp\_3\_affine\_relay}{\left(x_{cp:3} \right)} - 1.33 \operatorname{normal\_ecg\_st\_gate}{\left(x_{restecg:0\_b},x_{oldpeak\_b} \right)}} + 1}\end{gathered}
$$

*Note: the analytical\_formula stored with this annotation's evidence differs from the live text (it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.*

<details><summary>Recorded formula (evidence record)</summary>

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 1.53 x_{ca\_a} \\
&\quad + 1.01 x_{chol} \\
&\quad - 0.927 x_{cp:1} \\
&\quad + 0.442 x_{cp:2} \\
&\quad + 1.32 x_{cp:4} \\
&\quad - 0.347 x_{exang:0} \\
&\quad - 0.286 x_{fbs:1} \\
&\quad + 0.869 x_{oldpeak\_a} \\
&\quad - 0.0177 x_{restecg:0\_a} \\
&\quad + 0.108 x_{restecg:1} \\
&\quad - 2.01 x_{sex:0} \\
&\quad + 1.85 x_{slope:2\_a} \\
&\quad - 1.04 x_{thal:3} \\
&\quad - 0.967 x_{thal:6} \\
&\quad + 0.718 x_{trestbps} \\
&\quad + 0.983 \operatorname{ca\_slope2\_gate}{\left(-20_{b},-22_{b} \right)} \\
&\quad + 1.67 \operatorname{ca\_thalach\_chain\_0}{\left(-22_{c},-15 \right)} \\
&\quad - 0.165 \operatorname{ca\_thalach\_chain\_1}{\left(-22_{c},-15 \right)} \\
&\quad - 1.31 \operatorname{cp\_3\_affine\_relay}{\left(-6 \right)} \\
&\quad + 1.33 \operatorname{normal\_ecg\_st\_gate}{\left(-12_{b},-18_{b} \right)} \\
&\quad -0.16 \bigr)
\end{aligned}\end{gathered}
$$

</details>

**Hypothesis (DB):**

> Root: output node 0 computes the logit z = -0.160 + LINEAR CORE + hidden readouts, and (binary task) the model applies a sigmoid at the output (get\_formula shows node 0 as ReLU; decision is z\>0). LINEAR CORE: 15 direct edges, largest \|w\|: sex:0 -2.01 (female lowers risk), slope:2 +1.85, ca +1.53, cp:4 (asymptomatic) +1.32, thal:3 (normal) -1.04, chol +1.01, thal:6 -0.97, cp:1 -0.93, oldpeak +0.87, trestbps +0.72; SD 5.29 of a 7.04 total logit SD. Children add: +0.983\*ca\_slope2\_gate (near-affine in ca, slope:2), -1.305\*cp3\_affine\_relay (affine, -0.66 cp:3), +1.669\*2245 and -0.165\*2009 from ca\_thalach\_chain (mostly affine: +1.30 ca, small -thalach), and +1.335\*normal\_ecg\_st\_gate (the one real interaction: oldpeak adds extra risk only when restecg:0=+1). Composition is purely additive at node 0. Hypothesis (supported): the network is a logistic regression in which ca appears through three parallel affine routes (total \~+3.4 per SD), plus one gated ST-depression term. Evidence (train, n=242): AUC 0.939, acc 0.888. Replacing every ReLU by identity gives AUC 0.916, almost exactly the loss from removing only the 1801 ReLU (0.916), so 1801 is the only nonlinearity that matters; the other ReLUs change AUC by \<=0.0002. Inputs age, sex:1, fbs:0, restecg:2, exang:1, slope:1, slope:3, thal:4, thal:7 are unconnected.

**Evidence (2 records, DB):**

- `analytical_formula`: Collapsed closed form of the root (get\_formula force=True, latex\_collapsed).
- `ablation_study`: Train split, full sample; ablations on in-memory copies (no graph edits).

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "note": "plain latex of composition drops child terms; latex_collapsed used",
 "plain": "z = -0.160 + [1.53 ca + 1.85 slope:2 - 2.01 sex:0 + 1.32 cp:4 + 1.01 chol - 1.04 thal:3 - 0.967 thal:6 - 0.927 cp:1 + 0.869 oldpeak + 0.718 trestbps + 0.442 cp:2 - 0.347 exang:0 - 0.286 fbs:1 + 0.108 restecg:1 - 0.018 restecg:0] + 0.983*ca_slope2_gate - 1.305*cp3_affine_relay + 1.335*normal_ecg_st_gate + 1.669*ca_thalach_chain[2245] - 0.165*ca_thalach_chain[2009]",
 "output_activation": "get_formula and the StructureNetwork forward pass show node 0 as ReLU (forward output == relu(z) to 3.6e-15); for this binary task the model applies a sigmoid at the output, so the decision is z > 0 (train acc 0.888 = acc of z>0)"
}
```

```json
// ablation_study
{
 "n": 242,
 "split": "train",
 "coverage": {
  "structural": 1.0,
  "compositional": 1.0,
  "hierarchy_complete": true
 },
 "logit_sd": 7.04,
 "train_acc": 0.888,
 "train_auc": 0.9392,
 "auc_ladder": {
  "full": 0.9392,
  "664 ReLU removed": 0.939,
  "2009 ReLU removed": 0.9391,
  "1173/2245 ReLU removed": 0.9392,
  "only 1801 ReLU removed": 0.9162,
  "direct edges only (all hidden mean-ablated)": 0.9265,
  "all ReLUs replaced by identity (pure linear)": 0.916
 },
 "operations": {
  "node_splits": 4,
  "function_changing": 0,
  "identity_node_insertions": 0
 },
 "linear_direct_sd": 5.29,
 "output_sensitivity_top": {
  "ca_a": 0.62,
  "ca_c": 0.46,
  "chol": 0.41,
  "cp:1": 0.38,
  "cp:4": 0.53,
  "sex:0": 0.81,
  "thal:3": 0.42,
  "thal:6": 0.39,
  "oldpeak_a": 0.35,
  "slope:2_a": 0.75
 },
 "hidden_logit_contrib_sd": {
  "664": 0.695,
  "1173": 0.658,
  "1801": 0.546,
  "2009": 0.199,
  "2245": 1.304
 }
}
```

</details>

### A.1 Leaf: `ca_slope2_gate`

|  |  |
|---|---|
| Annotation id | `ann_30` |
| Entry nodes | -20\_b (slope:2\_b), -22\_b (ca\_b) |
| Exit nodes | 664 (ca\_slope2\_term) |
| Internal / exit nodes in its own subgraph | 664 (ca\_slope2\_term) |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.00158 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.601 x_{ca\_b} \\
&\quad + 0.326 x_{slope:2\_b} \\
&\quad + 0.664 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hidden node 664 = ReLU(0.601\*ca + 0.326\*slope:2 + 0.664), read out with +0.983. Hypothesis (supported): a near-affine risk term in number of major vessels coloured (ca) and flat ST slope (slope:2). On train the pre-activation ranges \[-0.071, 2.607\]; the ReLU clips only the 85 rows with ca at its minimum and slope:2=0, and only by \<=0.071 (\<=0.07 logit). Replacing the ReLU by identity leaves train AUC unchanged (0.9392 -\> 0.9390); mean-ablating the node costs 0.002 AUC. Contribution SD 0.70 logit. So this is effectively +0.59\*ca + 0.32\*slope:2 added to the logit, on top of the direct ca and slope:2 edges.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula(force=True); simplification on the train data domain.
- `empirical_viz`: Train split (n=242, full sample): activation profile from forward pass, get\_viz\_summary, and local ablations (ReLU-\>identity, mean-ablation) on in-memory copies.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "664 = relu(0.601 ca + 0.326 slope:2 + 0.664); readout 664->0 +0.983",
 "interpretation": "near-affine: pre-activation min -0.071 on train, clipped for 85/242 rows (all ca=min, slope:2=0) by at most 0.07 logit units"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "auc_full": 0.9392,
 "active_rate": 0.649,
 "clipped_rows": 85,
 "auc_mean_ablated": 0.9369,
 "auc_relu_removed": 0.939,
 "logit_contrib_sd": 0.695,
 "pre_activation_range": [
  -0.071,
  2.607
 ],
 "viz_heatmap_entry_corr": 0.17
}
```

</details>

### A.2 Leaf: `cp3_affine_relay`

|  |  |
|---|---|
| Annotation id | `ann_31` |
| Entry nodes | -6 (cp:3) |
| Exit nodes | 1173 (cp3\_relay) |
| Internal / exit nodes in its own subgraph | 1173 (cp3\_relay) |
| Dimensionality (in → out, incl. descendants) | 1 → 1 |
| Live closed-form error (3 s.f. display) | 0.00487 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.505 x_{cp:3} + 1.92 \right)}\end{gathered}
$$

**Hypothesis (DB):**

> Hidden node 1173 = ReLU(0.505\*cp:3 + 1.92) is active on 100% of train rows (pre-activation in \[1.59, 2.71\]), so it is an affine relay: with readout -1.305 it contributes -0.658\*cp:3 - 2.50 to the logit. Supported: line plot entry/exit correlation 1.0; removing the ReLU changes no prediction (AUC 0.9392 unchanged); mean-ablation costs 0.004 AUC. Meaning: non-anginal chest pain (cp:3) lowers predicted disease risk. cp:3 has no direct edge, so this is its only route to the output.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula(force=True); simplification on the train data domain.
- `empirical_viz`: Train split (n=242, full sample): activation profile from forward pass, get\_viz\_summary, and local ablations (ReLU-\>identity, mean-ablation) on in-memory copies.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "1173 = relu(0.505 cp:3 + 1.92); readout 1173->0 -1.305 => -0.658 cp:3 - 2.50",
 "interpretation": "always active; affine relay"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "active_rate": 1.0,
 "line_plot_corr": 1.0,
 "auc_mean_ablated": 0.9348,
 "auc_relu_removed": 0.9392,
 "logit_contrib_sd": 0.658,
 "pre_activation_range": [
  1.594,
  2.707
 ]
}
```

</details>

### A.3 Leaf: `normal_ecg_st_gate`

|  |  |
|---|---|
| Annotation id | `ann_32` |
| Entry nodes | -12\_b (restecg:0\_b), -18\_b (oldpeak\_b) |
| Exit nodes | 1801 (normal\_ecg\_st\_gate) |
| Internal / exit nodes in its own subgraph | 1801 (normal\_ecg\_st\_gate) |
| Dimensionality (in → out, incl. descendants) | 2 → 1 |
| Live closed-form error (3 s.f. display) | 0.0014 |
| Live numeric-function error | 0 |

**Closed form:**

$$
\begin{gathered}\begin{aligned}
y_0 &= \operatorname{relu}\bigl( 0.805 x_{oldpeak\_b} \\
&\quad + 1.48 x_{restecg:0\_b} \\
&\quad -1.43 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Hidden node 1801 = ReLU(1.48\*restecg:0 + 0.805\*oldpeak - 1.43), readout +1.335. Hypothesis (supported): an interaction gate, the network's only consequential nonlinearity. restecg:0 takes values +/-1 (121 rows each on train). With restecg:0=+1 the gate passes ReLU(0.805\*oldpeak + 0.05), active on 39.7% of those rows (exercise ST depression above about its median); with restecg:0=-1 it is active on 1/121 rows (needs oldpeak \> 3.6 SD). So ST depression adds up to \~+3.4 extra logit, but only when restecg:0=+1. The ReLU matters: replacing it by identity drops train AUC 0.9392 -\> 0.9162 (the linear version would add a large restecg:0 main effect); mean-ablating the whole node costs only 0.0001 AUC, so its value is the shape, not the extra ranking signal over the direct oldpeak edge (+0.869). Overall active rate 20%, contribution SD 0.55 logit.

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula(force=True); simplification on the train data domain.
- `empirical_viz`: Train split (n=242, full sample): activation profile from forward pass, get\_viz\_summary, and local ablations (ReLU-\>identity, mean-ablation) on in-memory copies.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "1801 = relu(1.48 restecg:0 + 0.805 oldpeak - 1.43); readout 1801->0 +1.335",
 "interpretation": "restecg:0 is standardised to +/-1; normal resting ECG => relu(0.805 oldpeak + 0.05); abnormal => relu(0.805 oldpeak - 2.91), closed unless oldpeak > 3.6 SD"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "auc_full": 0.9392,
 "active_rate": 0.202,
 "auc_mean_ablated": 0.9391,
 "auc_relu_removed": 0.9162,
 "logit_contrib_sd": 0.546,
 "pre_activation_range": [
  -3.61,
  3.15
 ],
 "mean_act_restecg0_neg": 0.005,
 "mean_act_restecg0_pos": 0.271,
 "viz_heatmap_entry_corr": -0.135,
 "active_rate_restecg0_neg": 0.008,
 "active_rate_restecg0_pos": 0.397
}
```

</details>

### A.4 Leaf: `ca_thalach_chain`

|  |  |
|---|---|
| Annotation id | `ann_33` |
| Entry nodes | -22\_c (ca\_c), -15 (thalach) |
| Exit nodes | 2245 (ca\_relay), 2009 (thalach\_ca\_term) |
| Internal / exit nodes in its own subgraph | 2245 (ca\_relay), 2009 (thalach\_ca\_term) |
| Dimensionality (in → out, incl. descendants) | 2 → 2 |
| Live closed-form error (3 s.f. display) | 0.0123 |
| Live numeric-function error | 8.88e-16 |

**Closed form:**

$$
\begin{gathered}y_0 = \operatorname{relu}{\left(0.781 x_{ca\_c} + 1.25 \right)} \\ \begin{aligned}
y_1 &= \operatorname{relu}\bigl( 0.949 x_{thalach} \\
&\quad + 1.36 \operatorname{relu}{\left(0.781 x_{ca\_c} + 1.25 \right)} \\
&\quad + 0.179 \bigr)
\end{aligned}\end{gathered}
$$

**Hypothesis (DB):**

> Serial chain ca -\> 2245 -\> 2009 \<- thalach. 2245 = ReLU(0.781\*ca + 1.25) is always active on train (pre in \[0.68, 3.31\]); its readout +1.669 adds +1.30\*ca + 2.08 to the logit (SD 1.30, the largest hidden contribution; mean-ablation costs 0.007 AUC). 2009 = ReLU(0.949\*thalach + 1.36\*2245 + 0.179) = ReLU(0.949\*thalach + 1.06\*ca + 1.87) is active on 93.8% of rows (clipped for 15 rows with very low max heart rate and few vessels); readout -0.165 gives a small -0.157\*thalach - 0.175\*ca term (SD 0.20). Hypothesis (supported): mostly affine; ca raises risk, higher max heart rate slightly lowers it; the 2009 ReLU is near-inert (removing it: AUC 0.9392 -\> 0.9391).

**Evidence (2 records, DB):**

- `analytical_formula`: Closed form from get\_formula(force=True); simplification on the train data domain.
- `empirical_viz`: Train split (n=242, full sample): activation profile from forward pass, get\_viz\_summary, and local ablations (ReLU-\>identity, mean-ablation) on in-memory copies.

<details><summary>Evidence payloads (formula text omitted)</summary>

```json
// analytical_formula
{
 "plain": "2245 = relu(0.781 ca + 1.25) (always active); 2009 = relu(0.949 thalach + 1.36*2245 + 0.179) = relu(0.949 thalach + 1.060 ca + 1.871); readouts 2245->0 +1.669, 2009->0 -0.165"
}
```

```json
// empirical_viz
{
 "n": 242,
 "split": "train",
 "2009_pre_range": [
  -2.248,
  5.172
 ],
 "2245_pre_range": [
  0.679,
  3.307
 ],
 "2009_active_rate": 0.938,
 "2009_sensitivity": {
  "ca": 0.999,
  "thalach": 0.894
 },
 "2245_active_rate": 1.0,
 "2009_clipped_rows": 15,
 "2009_logit_contrib_sd": 0.199,
 "2245_logit_contrib_sd": 1.304,
 "auc_2009_mean_ablated": 0.9382,
 "auc_2009_relu_removed": 0.9391,
 "auc_2245_mean_ablated": 0.9319
}
```

</details>

## Refactoring operations

| Operation | Count |
|---|---|
| annotate | 5 |
| rename\_node | 31 |
| split\_node | 4 |

4 splits, 0 identity nodes, 0 consolidations (0 of the splits are of hidden nodes). Renames are not listed. All are function-preserving; the event stream holds no add\_node, remove\_node or connection toggles.

- `#26` split -20 into -20\_a, -20\_b. Final copies: -20\_a → heart\_disease\_logit (entry) \| -20\_b → ca\_slope2\_gate (entry)
- `#27` split -22 into -22\_a, -22\_b, -22\_c. Final copies: -22\_a → heart\_disease\_logit (entry) \| -22\_b → ca\_slope2\_gate (entry) \| -22\_c → ca\_thalach\_chain (entry)
- `#28` split -12 into -12\_a, -12\_b. Final copies: -12\_a → heart\_disease\_logit (entry) \| -12\_b → normal\_ecg\_st\_gate (entry)
- `#29` split -18 into -18\_a, -18\_b. Final copies: -18\_a → heart\_disease\_logit (entry) \| -18\_b → normal\_ecg\_st\_gate (entry)

## Checks: recorded vs live

Live values were recomputed on 2026-11-11T12:25:06+13:00 (commit `77513b9`) by `verification/collect_live.py`, which calls `check_explanation.check()` and `get_coverage` on the training split through a read-only database session. Agent values are the agent's own report (often rounded to 3 s.f.); orchestrator values are `verified_by_orchestrator` in the record.

| Check | Agent | Orchestrator record | Live | Status |
|---|---|---|---|---|
| Function preservation, max \|Δ output\| | 3.55e-15 | 3.55e-15 | 3.89e-16 | agree |
| Closed-form max abs error (all leaves) | 0.01228 | 0.01228 | 0.01228 | agree |
| Numeric-function max abs error | n/a | 8.88e-16 | 8.88e-16 | agree |
| Structural coverage | 1 | 1 | 1 | agree |
| Compositional coverage | 1 | 1 | 1 | agree |
| Hierarchy complete (one root, contains output, all under it) | yes | yes | yes | agree |
| Root annotation | n/a | heart\_disease\_logit | heart\_disease\_logit | agree |
| Leaf / composition counts | 4 / 1 | n/a | 4 / 1 | agree |
| Operation counts | 4 splits, 0 identity | {"annotate": 5, "rename\_node": 31, "split\_node": 4} | {"annotate": 5, "rename\_node": 31, "split\_node": 4} | agree |
| Full coverage | yes | yes | yes | agree |

**Recorded vs live: all agree.** Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise).

Closed-form error per leaf (live; the record keeps only the maximum):

| Leaf | Closed-form error (3 s.f. display) | Numeric-function error |
|---|---|---|
| `ca_thalach_chain` | 0.01228 | 8.88e-16 |
| `cp3_affine_relay` | 0.004868 | 0 |
| `ca_slope2_gate` | 0.001578 | 0 |
| `normal_ecg_st_gate` | 0.001401 | 0 |

## What resisted (from the record)

> Nothing resisted coverage. Only input fan-out needed splitting (slope:2, ca x3, restecg:0, oldpeak). Four of five hidden ReLUs are affine or near-affine on the train domain (1173, 2245 always active; 664 clips \<=0.07; 2009 clips 15/242), so their annotations describe relays rather than distinct concepts. 9 unconnected inputs (age, sex:1, fbs:0, restecg:2, exang:1, slope:1, slope:3, thal:4, thal:7) listed as root entries.

**Tool notes (record):** Root composition plain latex drops child terms (used latex\_collapsed). Forward output equals relu(z) to 3.6e-15 (output node is ReLU). Tool compositional\_coverage always 1.0.

## What to look at

- Largest closed-form error: `ca_thalach_chain` (0.0123). This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. Check that the rounded formula still supports the hypothesis as worded.
- Strong claim in `ca_slope2_gate`: *“the ReLU clips only the 85 rows with ca at its minimum and slope:2=0, and only by \<=0.071 (\<=0.07 logit).”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- Strong claim in `normal_ecg_st_gate`: *“Hypothesis (supported): an interaction gate, the network's only consequential nonlinearity.”* Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?
- 4 split(s), all of input nodes (an input feeding two annotations). These are routine; check only that each input copy sits in the annotation whose hypothesis uses it.
- Explained with protocol v2 tooling: get\_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.
- Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?

## Verdict

Fill in and add to `experiments/10seed/explanations/heart-s1.json` as a top-level key, then run `uv run python experiments/10seed/build_explanations_csv.py`. Or use the UI's Review & verify form, which writes the same field. verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing).

```json
"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}
```

One-line equivalent (same writer as the UI):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('heart-s1', True, minutes=0, changes='', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```
