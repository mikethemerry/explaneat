# Verdict: lad-6f6c3 (genome 69ef07cd)

**Overall: weakened.** Every number, direction, threshold and count we checked reproduces on the training split, and coverage and function preservation hold. One wording overclaim remains: the age-BMI hinge is said to be "off for younger or leaner people", but lean older people have it on.

Findings: 0 refuted, 1 weakened, 28 upheld, 0 flags. About 144 statements were checked. There are no drafts or claims for this network.

## Weakened

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| `operations.age_bmi_hinge.hypothesis` | L3 | "so it is off for younger or leaner people" | The hinge switches on along one line that combines age and BMI. At age 80 and BMI 22 it is on (node 2.08, risk 92%), and that case is in the hypothesis's own self-probe. In the training data, 166 rows with BMI below 22 and 67 rows under age 40 have it on. It is off only for people who are young and lean together. The rest of the hypothesis says this correctly. | `probe --sweep age=40,60,80 --sweep bmi=22,27,32,37 --nodes 1`; subgroups `node_1 > 0 and bmi < 22`, `node_1 > 0 and age < 40` |

## Upheld parts

- **risk_model (ann_24)**: The prediction is a sigmoid of a weighted sum of the four units: 1.49, -0.374, 1.32 and -1.29 on units 1 to 4, plus a constant of -0.412. No gating. The age-70 cases at score 10 reproduce (85% and 4%, terms -6.13 and +3.34), and so do the AUCs with one part off (0.731, 0.748, 0.787, 0.834). Female, urban, insured and noise matter little: switching off all 16 of their edges leaves the AUC at 0.8369, and the typical-case risks are 30/29%.
- **visits_smoker_hinge (ann_22)**: The thresholds reproduce: off at 3 and on at 4 visits for non-smokers, off at 2 and on at 3 for smokers. Once the hinge is on, smoking adds a constant 1.47 to the node. All the subgroup counts reproduce (2388/2388, 2604/5503, 15/2914, 481/4843, 0/11,266, 5473).
- **income_score_hinge (ann_23)**: The income-by-score grid reproduces (2.57, 1.08, 0.00 and 1.72; risks 2%, 10% and 28%). It is on for 11,156 rows, with a mean contribution of -1.92. No row with income above 100 has it off.
- **linear_age_bmi_score (ann_21)**: It is positive on 23,987 rows and zero on 13 (where the age-BMI hinge adds +9.29). The self-probe reproduces (1.59 to 0.47, and 0.79 to 1.50). Replaced by its training mean, the AUC is 0.834. Set to zero, the mean risk changes by -0.081. Its meaning is only partial: a small slope plus the baseline, as the record itself says.
- **Record**: Coverage is 1.0/1.0 and the hierarchy is complete. Function-preservation error is 0. Closed-form error is 0.0344, and the per-leaf errors and unit maxima match. The operation counts match. The AUCs in the summary match.

## Flags

None. The directions are plausible for synthetic data. "Score" has no stated real-world meaning, so the income-score reading can only be relative.

Note: the recheck report gives no `target_hash` for a record target, so it is null in `verdict.json`.
