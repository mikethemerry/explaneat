# Verdict: lad-b58a2 (genome ad642a9d)

**Overall: weakened.** Everything that could be checked reproduced: the coverage and preservation figures, the closed forms, every count and threshold, the directions and the AUCs. Two sentences are a little imprecise.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.low_income_hinge.hypothesis | L3 | Above the threshold, income "still lowers risk a little" (through the age ramp and the protective unit) | Above the threshold the logit slope is about a third of the slope below it (-0.032 vs -0.10 per thousand at the typical case), an odds ratio of about 4.9 between 50k and 100k. At age 70, risk falls from 81% to 38% between 50k and 100k. The attribution is right; "a little" understates it | `verify probe --sweep income=10:100:10 --sweep age=35,50,70 --nodes 3` |
| operations.age_smoker_ramp.hypothesis | L3 | The smoker's risk "levels off at 94%" from age 65 | 93.7% at 65, 95.0% at 75, 96.4% at 90. It keeps rising slowly through the low-income hinge, as the same text says for the non-smoker | `verify probe --sweep age=20:90:15 --sweep smoker=0,1` |

Upheld:
- **Record:** coverage 1.0/1.0 with a complete hierarchy; function preservation error 0; closed-form error 0.0203. Every number in the summary and in the resisted text reproduces.
- **risk_model_root (ann_24):**
  - There are no direct edges, so the logit simply adds the four units.
  - The smoking-by-age score changes reproduce (2.04, 2.06, 0.27).
  - The minor-input sweeps reproduce.
  - All-affine AUC 0.890, with restore gains 0.017 and 0.006.
- **bmi_obesity_hinge (ann_20):**
  - Off below a bmi of about 29 to 30 and steep above it.
  - All counts, shares and outcome rates reproduce.
  - AUC 0.871 when switched off.
- **nonsmoker_young_affluent_unit (ann_21):**
  - Smoking switches it off for most smokers: 2157 of 7255 keep it.
  - It is larger for younger, richer and leaner people.
  - It carries part of the noise effect.
  - AUC 0.911 when switched off.
- **low_income_hinge (ann_22):** apart from the sentence above, the threshold, the age shift, the counts and the AUC of 0.895 all reproduce.
- **age_smoker_ramp (ann_23):**
  - Unit values, counts and the ramp-off subgroup reproduce.
  - AUC 0.793 when switched off, the largest drop of any part.
  - Its affine fit alone gives AUC 0.9130.

Leaf meaning:
- **meaningful:** bmi_obesity_hinge, low_income_hinge, age_smoker_ramp.
- **partial:** nonsmoker_young_affluent_unit. It reads the noise input about as strongly as income, and its effect on the AUC is minor.

Flags: none. The noise input moves the typical case's risk from 13.7% to 10.2%, more than any of the five minor real inputs. The record already reports this as a fitted weight with no domain reading.
