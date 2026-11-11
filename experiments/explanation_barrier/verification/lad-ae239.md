# Verdict: lad-ae239 (genome 2fa03acd)

**Overall: refuted.** Every stated number reproduces (recheck 0 mismatches; about 170 statements checked by hand with probe and subgroup), coverage is 1.0 and the function is preserved. But the root draws a wrong conclusion from its linearisation check, and two leaf readings are broader in the model than the hypotheses say.

## Refuted and weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.root_risk_score.hypothesis | L3 refuted | Affine fits of both sides give AUC 0.939 (gain 0.0012), "so the interactions shape the predicted risk levels far more than the ranking of rows" | The affine fits are in the 32 first-layer ReLU features, so the test keeps the first layer's non-linearity. In the inputs, an additive spline model (no interactions) gets AUC 0.899 and a main-effects logistic model 0.886, against the model's 0.940. Interactions are worth about 0.04 AUC of ranking. | `verify linearise`; refit on the 24000 training rows |
| operations.high_side_core.hypothesis | L3 weakened | "the main carrier of the rise in risk with age" | Typical man, age 25 to 75: high side +2.66, low side +4.40. From 25 to 60: +0.71 against +3.38. The high side only leads above about 60. | `probe --sweep age=25..75 --sweep female=0,1` |
| operations.unit38_score_urban.hypothesis | L3 weakened | switches on for a high score in rural patients "(and for some very old ... profiles)" | Only 1565 of its 5756 active rows (27%) have a high score and are rural. It acts on 45% of rows aged 60+ with a score under 80. It is age-gated: at score 95 for a rural patient it gives +0.31 at age 30 and +1.24 at age 75, and 0 for a rural woman aged 35. | `subgroup`; `probe --set score=95 --set urban=0 --sweep age` |
| record.resisted | L1 weakened | "the two core leaves ... have partial readings" | record.leaves grades low_side_core "meaningful": the record contradicts itself. | read record |
| structure.root_risk_score | L4 weakened | first layer "described, not explained" | Coverage is 1.0 in form only. Every leaf's input-level reading rests on 32 first-layer units with no meaning given. The record says so itself. | structure dump |

## Upheld parts
- risk_raising_units (ann_112): the age sweep, the 4.02 against 1.47 split, AUC 0.910 and the score change of 2.02 all reproduce.
- risk_lowering_units (ann_113): -5.38, -1.46 and 0.00; -7.23 and -1.18; AUC 0.905. All reproduce.
- low_side_core (ann_107): all numbers reproduce, and the directions hold away from the typical case. Graded partial for meaning.
- unit36_visits_bmi_smoker (ann_108): all numbers reproduce. Meaningful.
- unit42_age_female (ann_110): all numbers reproduce. It fades sooner for men. Meaningful.
- unit47_income (ann_111): all numbers reproduce. Its partial reading is accurate. The evidence record mislabelled "ablation" is disclosed.
- Root: every probe number reproduces (age, sex by age, bmi, income by score, score by urban, visits, insurance by smoking, noise). The noise effect is small on the data: mean change in risk 0.9 points, AUC 0.9403 to 0.9398.

## Flags
- At age 30, raising bmi from 25 to 38 lowers predicted risk (1.7% to 1.2%). A protective high bmi in young people is odd; this is for the domain reviewer.

Note: `recheck` gave no `target_hash` for a record target.
