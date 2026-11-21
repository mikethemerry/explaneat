# Heart cholesterol figure: proposed patient profiles (for Mike to confirm; figure NOT built)

Every number below is read from `thesis/ch7/exp1/data/heart_profiles.json`, written by
`uv run python thesis/ch7/exp1/heart_profiles.py` (read-only `explaneat_dev`; the original Heart network,
genome `9022a9eb`, explanation `a03ebc86`; 242 training patients; rerun twice, byte-identical). The figure
itself is not drawn until the definition is confirmed.

## What the figure would show

The logit of the original Heart network as serum cholesterol varies over the training range (126 to
417 mg/dL, `grid_mg_dl`, 301 points), every other input fixed, for two patients. The logit splits exactly
into three terms (`direct_chol_terms`, `gate_term`, `rest`; `rest` is constant along the grid):

- **direct linear term**: cholesterol's edge to the output (`weights.chol_to_output`, 0.676 per SD) plus its
  path through the always-on blood-pressure unit 2379 (`weights.chol_via_2379`, -0.193 per SD); together
  about +0.48 logit per SD (SD = 48.2 mg/dL, `weights.chol_scale_mg_dl`), the same for every patient;
- **risk-aggregation gate**: unit 534 read out with weight -0.884 (`weights.gate_to_output`). Its inputs are the
  benign-profile copy (sex, exercise angina, fasting sugar), ca, cholesterol and the cholesterol threshold
  unit 3040. While it is on it lowers the logit; rising cholesterol turns it off.

Proposed drawing: one panel, logit against cholesterol (mg/dL), one line per profile, the gate term dashed
under each line, and a tick rug of each subgroup's own cholesterol values.

## Proposed profiles

**How chosen.** The subgroups are defined by the inputs the explanation names, not fitted: the benign-profile
gate's two main inputs (sex, exercise-induced angina) and the 534 gate's largest other input (number of
vessels coloured by fluoroscopy, `ca`, weight -1.16 into 534).

| | Benign profile | High-risk profile |
|---|---|---|
| Subgroup rule (training rows) | female, no exercise-induced angina, ca = 0 | male, exercise-induced angina, ca 1 to 3 |
| Training patients in the subgroup (`n_train`) | 43 of 242 | 37 of 242 |
| Disease rate in the subgroup (`disease_rate`) | 2.3% | 91.9% |
| Share with gate 534 on, at their own values | 88.4% | 0% |
| Subgroup's own cholesterol, median (10th to 90th percentile) | 248 (186 to 327) mg/dL | 256 (199 to 301) mg/dL |
| Gate 534 along the sweep | on from 126 mg/dL, switches off at 326 mg/dL | on only below 147 mg/dL (below the subgroup's 10th percentile, 199), off above |

The cholesterol threshold unit 3040 turns on from 286 mg/dL for both (`threshold_unit_3040_on_from_mg_dl`);
it feeds the gate, so for the benign profile it steepens the gate's switch-off between 286 and 326 mg/dL.

**The fixed inputs: two variants** (both computed; option A is my proposal).

- **(A) the subgroup's median patient** (`benign/A_subgroup_median`, `high_risk/A_subgroup_median`): every
  numeric input at the subgroup's training median, every category at its most common level in the subgroup.
  - Benign: age 53, resting BP 130, max heart rate 160, ST depression 0.4, non-anginal chest pain, sugar
    120 or less, normal resting ECG, upsloping ST, normal thallium scan.
  - High risk: age 59, ca 1, resting BP 128, max heart rate 131, ST depression 1.4, no chest pain
    (asymptomatic), sugar 120 or less, left ventricular hypertrophy on the ECG, flat ST, reversible defect.
- **(B) the typical patient** of the plain-language layer (training median / most common category over all
  242 patients) with only sex, exercise angina and ca changed to the subgroup's values (ca: the subgroup
  median, 0 and 1).

The gate's own inputs (sex, exercise angina, fasting sugar, ca, cholesterol) are the same in A and B, so the
gate term and the switch-off points are identical; A and B differ only by a constant shift of each line (the
`rest` term). A puts each line where that kind of patient actually sits (logit about -10 to -5 for benign,
+5 to +8 for high risk); B keeps everything but the defining inputs at the typical patient (about -8 to -2
and 0 to 3).

**Count of resembling patients.** "Resemble" here means "in the subgroup" (the three defining inputs match
exactly): 43 and 37 training patients. No other input is matched; tell me if you want a stricter rule (e.g.
also within one SD on age and blood pressure), and I will add the counts.

## To confirm

1. Subgroup rules (sex, exercise angina, ca) as above?
2. Variant A (subgroup median patient) or B (typical patient with three inputs changed)?
3. Logit on the y-axis (as proposed), or predicted risk?
