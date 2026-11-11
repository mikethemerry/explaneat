# Verdict: lad-cb415 (genome face2eeb-c7f0-41fa-907f-698d49c850e4)

**Overall: refuted.** The structure, coverage, closed forms and every quoted number reproduce. One direction is wrong: "higher income lowers the risk" does not hold for high-score people. At high scores, higher income raises the predicted risk, because two other parts read income with positive weights.

## Refuted and weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.income_low_score_lowering.hypothesis | L3 refuted | "in the model higher income lowers risk" | Typical urban man at score 95: income 10 -> 72 raises the risk from 40.9% to 68.0%. Rural man at score 80: income 10 -> 45 raises it from 66.5% to 79.0%. Adding 10k of income raises the risk on 22.6% of training rows. | probe income x score; income +10k on every training row |
| record.summary | L3 refuted | "higher income lowers it, more so when the score is low" | At high scores the effect reverses; it does not just get weaker (same numbers as above). | as above |
| operations.age_bmi_rise.hypothesis | L3 weakened | "bmi raises risk mainly for older people; below the threshold bmi has little effect" | Rural man, score 85, age 40: bmi 18 -> 36 lowers the risk from 96.7% to 36.3%. Adding 3 to bmi lowers the risk on 23% of rows. The statement holds only for the typical case. | probe bmi x age at score 85, rural |
| operations.score_rural_rise.hypothesis | L2 weakened | "larger for rural, lower bmi, uninsured" | It leaves out the income weight (+0.35), which causes the income reversal. The bmi weight is as large as the urban one, yet the summary presents the part as score x rural only. | closed form, probes |
| operations.risk_root.hypothesis | L3 weakened | income acts only through income_low_score_lowering; bmi only through age_bmi_rise | It misses the positive income weights in score_rural_rise and age_bmi_rise, and the protective effect of bmi at high scores. | as above |
| record.summary / record.resisted | L3 weakened | "insurance has a small effect in both directions" | Being insured lowers the risk on 65% of rows and raises it on 11%. The mean change is -7.0 points, larger than smoking's +5.8. | insured 1 vs 0 on every training row |

## Upheld parts
- young_female_lowering (ann_41): the age x sex probe, the contributions, the acting counts and AUC 0.889 all reproduce.
- young_uninsured_lowering (ann_42): unit values, the whole-model insured vs uninsured risks, the counts and AUC 0.9358 reproduce. It is honestly described as having no domain reading.
- visits_smoker_rise (ann_45): the visits x smoker probe, the counts and AUC 0.919 reproduce. One minor point: below the threshold, more visits lower the risk slightly where age_bmi_rise is on.
- Record: coverage 1.0/1.0, function preservation 0, closed-form error 0.0555, operation counts, and all ablation AUCs and noise figures reproduce. Age is the strongest input: fixing it at its mean drops the AUC to 0.769.

## Flags (domain, not data)
- young_uninsured_lowering: the uninsured get a lower predicted risk up to middle age (19.8% vs 28.9% at 45).
- score_rural_rise: thinner and richer people with high scores get a higher predicted risk.
