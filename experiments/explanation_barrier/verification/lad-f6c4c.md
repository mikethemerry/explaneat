# Verdict: lad-f6c4c (genome 3feef87b)

**Overall: weakened.** Every number, count, probe value, ablation and the parity rule reproduce, and the structure, coverage and function preservation hold. Two leaf headlines are looser than their own evidence.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.smokerOrRuralLowerer.hypothesis (ann_41) | L2 | "lowers the predicted risk of smokers and of rural people ... weak or off for urban non-smokers" | Also weak for urban female smokers, who are smokers: mean -0.29, and 0 from age 70 at the typical case. Being female damps it (closed form -1.17). The later counts show this; the lead sentence does not. | `verify subgroup --where "smoker==1 and female==1 and urban==1"`; `verify probe --set smoker=1 --set female=1 --sweep age=20:80:7 --sweep urban=0,1` |
| operations.fewFlagsRaiser.hypothesis (ann_40) | L4 | "raises risk of people with at most one of the three flags" | Mainly a rural-man detector: 15.4 logits for non-smoking rural men and 12.1 for rural male smokers, but only about 3 once female or urban is present (closed form -1.56 female, -1.52 urban, -0.54 smoker). "Few flags" is an envelope, not the pattern. | per-group `verify subgroup`; closed form |

Upheld parts:
- parityOfSmokerFemaleUrban (ann_46): odd 12008 rows, mean 94%, 0 under one half; even 11992, mean 7%, 12 at or over one half (11 rural non-smoking men aged 80+, and one urban male smoker at 51%). Age raises risk in all eight groups. Age-to-mean AUC 0.9455, six minor inputs 0.9618, all parts affine 0.551. All reproduce.
- urbanNonSmokingWomanLowerer (ann_42): all counts reproduce. Switching it off flips all 4963 non-smoking urban women.
- smokerOrFemaleRaiser (ann_43): all counts and age self-probes reproduce. Off in urban female smokers, 1026 of 2115 flip.
- urbanNonSmokingManRaiser (ann_44): all counts reproduce. The typical urban non-smoking man drops from 96% to 40% without it.
- nonSmokingManLowerer (ann_45): all counts reproduce. AUC 0.707 without it is the largest drop of the six parts.
- fewFlagsRaiser and smokerOrRuralLowerer: every number reproduces (see the table for the wording issues).
- Record and structure: coverage 1.0/1.0, function preservation 0, closed-form error 0.0135, and the operation counts all reproduce.

Flags: none. The data are synthetic.
