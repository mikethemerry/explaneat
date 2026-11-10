# Verdict vb-785c68 (genome 7a8237f3, adult_ohe_v2)

**Overall: refuted.** The age part's plain hypothesis swaps Ireland and Thailand, and the self-employment technical hypothesis gives the wrong share of rows. Everything else reproduces: coverage 1.0 / 1.0, function preservation 5.6e-16, all leaf closed forms, all 200 claims and every stated ablation. A few category statements hold only for people who do not exist in the data.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_121.hypothesis[0] | L3 refuted | Ireland: chance rises with age; Thailand lowers it | Ireland keeps the V shape, shifted down 1.57 logit (61% at 22, 20.5% at 46, 35% at 70). Thailand is the group that rises with age (66% to 89.5%), and from about 32 years it raises the chance relative to the US | probe age 22:70 x country 39,21,37 |
| operations.self_emp_not_inc_offset.hypothesis | L1 refuted | "12.4% of rows" | 3034 of 39073 = 7.8% (as its own evidence record says) | subgroup workclass == '6' |
| operations.age_profile.hypothesis | L1 weakened | Thailand 27 rows | 26 rows | subgroup / vars |
| claims.C137 (age prose, worth_checking[6]) | L3 weakened | lowest at about 47 years | minimum at 46.1 years (the technical hypothesis's 46 is right) | probe age 45.8:46.6 |
| drafted.readdown.ann_129.evidence[3] / blocks[1][3] | L3 weakened | 7th-8th grade lowers to 35%, 9th 44%, 10th 49% | only with education level held at 10, which no 7th-8th grader has (0 of 739). With matching levels: 84.6%, 84.4%, 82.0% vs 71.5% for high-school graduate | probe education x education-num; subgroup |
| drafted.readdown.ann_129.evidence[1] / blocks[1][1] | L3 weakened | wife lowers to 41%, woman raises to 74% | 41% is a male wife (3 rows). A woman who is a wife: 53.3% vs 63.1% for a male husband; same direction, smaller | probe sex x relationship |

**Upheld parts:**
- ann_131 income_logit: an additive sum of 8 parts; 41 unconnected inputs; bias 3.385 (SHAP numbers not re-run).
- ann_130 capital_gain_effect: the curve bends more steeply at about $4.8k and about $5.2k; the stated groups switch off the bends (9th grade, Jamaica, Nicaragua, the Philippines and Scotland switch off the $4.8k bend; Ecuador switches off the $5.2k bend); band means reproduce; with this part switched off, AUC is 0.894.
- ann_122 / ann_123: the bend points, slopes, activity (5.1%) and ablation numbers reproduce.
- ann_125 short_hours_hinge: steady up to 50 hours, flat after (79.5% at 24, 50.8% from 50); active for 80%.
- ann_128 country offsets, ann_126 other relatives, ann_127 rare categories: the effect of switching each category on reproduces, and so do the ablation results.
- ann_129 backbone: the technical coefficients and the ablation (AUC 0.790) reproduce; it is weakened only by the off-support prose above.
- ann_124: the plain hypothesis and claims are upheld; only the 12.4% in the technical text is wrong.

**Flags (L4, domain):** capital losses lower the predicted chance of a low income; the census sampling weight affects predictions; the Thailand reversal rests on 26 people; the extreme effects of AF-spouse and never-worked come from 29 and 9 people. The draft's worth-checking list already raises all four.

Not re-run: whole-model SHAP, the combined backbone+country ablation, and the Thai "net +0.10" average.
