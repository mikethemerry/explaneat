# Verdict vb-2103cc

**Overall: weakened.** All numbers reproduce and the three parts behave as stated; one sentence attributes an age jump to patients with a reversible thallium defect, for whom the typical patient sees no age effect through the gate.

Genome `0057df7a-62a2-43c3-b107-9afe3735a7c3`. Started 2026-11-10T23:22:09+13:00, finished 2026-11-10T23:25:20+13:00 (3 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V11 | `drafted.readdown.ann_31.evidence[1]` | L3 | weakened | for patients with a reversible defect on the thallium scan, age rises in a jump (typical patient from about 12% at 42 to about 28% at 66) | with a reversible defect the gate is off at every age for the typical patient (risk flat at 41.9%, contribution 0); only 3 of 90 such patients have the gate on (subgroup thal==7). The 12%->28% numbers belong to the normal-scan case; the jump applies to a small minority (14% of moving curves) | probe age sweep with thal=7 (typical patient) |

## Parts upheld

- ann_34: upheld (checked to L4).
- ann_33: upheld (checked to L3).
- ann_32: upheld (checked to L4).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[2]`: fasting blood sugar above 120 mg/dL lowers risk -- raised blood sugar is a known risk factor; the model gives the opposite sign
- `drafted.worth_checking[1]`: typical angina gets the lowest risk, below no-symptom patients -- asymptomatic patients in this dataset are known to carry high disease rates, so this is plausible for the data but counter-intuitive clinically
- `drafted.worth_checking[3]`: fixed thallium defect lowers risk (n=15) -- a fixed defect is usually a sign of past infarction; small group
- `drafted.readdown.ann_32.evidence[2]`: higher cholesterol lowers risk through this part -- counter-directional inside the part; offset by cholesterol's direct positive edge
